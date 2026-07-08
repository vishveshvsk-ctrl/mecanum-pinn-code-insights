#!/usr/bin/env python
# =============================================================================
# roller_audit.py — step-0 headroom diagnostic for the promoted roller term.
#
# Runs against EXISTING v1 3-state checkpoints (v1 imports only).  Evaluates the
# quasi-static roller torque balance on (a) ground-truth states and (b) the
# checkpoint's predictions, binned by slip speed.  Ground truth defines the
# quasi-static floor; the prediction-vs-floor gap in high-slip bins is the
# recoverable error justifying v2's roller-loss campaign.
#
# CLI: python observer_v1_py/roller_audit.py --run-dir observer_v1_py/runs/S1_train_w32
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.feather  # noqa: F401
import torch

from mecanum_observer import config as C
from mecanum_observer import data as D
from mecanum_observer.config import ObserverConfig
from mecanum_observer.models import build_model
from mecanum_observer.physics import roller_residual

SLIP_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf])


def _centers(edges: np.ndarray) -> np.ndarray:
    c = 0.5 * (edges[:-1] + edges[1:])
    c[-1] = edges[-2] * 1.5 if np.isinf(edges[-1]) else c[-1]
    return c


def _load_cfg(run_dir: Path) -> ObserverConfig:
    """Reconstruct v1 ObserverConfig from the run checkpoint / metrics file."""
    ckpt = run_dir / "checkpoint.pt"
    stt = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_dict = stt.get("cfg")
    if cfg_dict is None and (run_dir / "metrics.json").exists():
        with open(run_dir / "metrics.json") as fh:
            metrics = json.load(fh)
        cfg_dict = metrics.get("cfg")
    if cfg_dict is None:
        raise ValueError(f"Cannot reconstruct config from {run_dir}")
    cfg = ObserverConfig(**cfg_dict)
    # Force physics_loss True so make_windows emits the ph_* block.
    cfg.physics_loss = True
    return cfg


@torch.no_grad()
def audit_run(run_dir: Path, regime: Path | None, cache_dir: str, bins: np.ndarray
              ) -> pd.DataFrame:
    cfg = _load_cfg(run_dir)
    if regime is not None:
        cfg.regime_name = D.load_regime(regime).get("regime", {}).get("name", cfg.regime_name)
    if cache_dir:
        cfg.cache_dir = cache_dir
    cfg = cfg.resolved()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    model = build_model(cfg).to(device)
    stt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(stt["model"])
    model.eval()

    splits = D.split_files(D.discover(cfg), cfg)
    rows: List[dict] = []
    for split_key, split_label in (("val", "same_subset"), ("test", "cross_subset")):
        files = splits.get(split_key, [])
        if not files:
            continue
        rows.extend(_audit_split(files, split_label, model, nrm, cfg, device, bins))
    return pd.DataFrame(rows)


@torch.no_grad()
def _audit_split(files: List[Path], split_label: str, model, nrm: D.Normalizer,
                 cfg: ObserverConfig, device: torch.device, bins: np.ndarray) -> List[dict]:
    nb = len(bins) - 1
    centers = _centers(bins)
    # Accumulators: [wheel, bin] for ground-truth and predicted states.
    sse_gt = np.zeros((C.N_WHEELS, nb))
    n_gt = np.zeros((C.N_WHEELS, nb), np.int64)
    sse_pred = np.zeros((C.N_WHEELS, nb))
    n_pred = np.zeros((C.N_WHEELS, nb), np.int64)

    for fi, path in enumerate(files):
        a = D.read_arrays(path, cfg.cache_dir)
        win = D.make_windows(a, nrm, cfg)
        del a
        if win is None:
            continue
        # Ground-truth states in physical units.
        y_mean = nrm.y_mean.reshape(1, 1, -1)
        y_std = nrm.y_std.reshape(1, 1, -1)
        Yphys = win["Yt"] * y_std + y_mean
        gamma_gt = Yphys[:, :, 0]
        zx_gt = Yphys[:, :, 1]
        zy_gt = Yphys[:, :, 2]
        zs_gt = np.zeros_like(gamma_gt)

        # Predicted states in physical units.
        Gw = torch.from_numpy(win["Gw"]).to(device)
        Pw = torch.from_numpy(win["Pw"]).to(device)
        pred_n = model(Gw, Pw).cpu().numpy()                 # [M,4,4]
        pred_p = pred_n * nrm.y_std + nrm.y_mean
        gamma_pred = pred_p[:, :, 0]
        # For zx/zy in the predicted residual, use ground-truth labels (same as v2
        # physics-loss design) so the audit isolates γ-prediction error.
        zx_pred, zy_pred = zx_gt, zy_gt
        zs_pred = np.zeros_like(gamma_pred)

        mu = win["ph_mu"]
        chi = win["ph_chi"]
        r_gt = roller_residual(np, gamma_gt, zx_gt, zy_gt, zs_gt, mu, chi,
                               win["ph_psi_dot"], win["ph_Vpx0"], win["ph_Vpy0"],
                               win["ph_cti"], win["ph_sti"])
        r_pred = roller_residual(np, gamma_pred, zx_pred, zy_pred, zs_pred, mu, chi,
                                 win["ph_psi_dot"], win["ph_Vpx0"], win["ph_Vpy0"],
                                 win["ph_cti"], win["ph_sti"])
        slip_bin = np.clip(np.digitize(win["vpm"], bins) - 1, 0, nb - 1)

        for w in range(C.N_WHEELS):
            sb = slip_bin[:, w]
            np.add.at(sse_gt[w], sb, r_gt[:, w] ** 2)
            np.add.at(n_gt[w], sb, 1)
            np.add.at(sse_pred[w], sb, r_pred[:, w] ** 2)
            np.add.at(n_pred[w], sb, 1)
        del win
        if (fi + 1) % 50 == 0:
            print(f"[roller-audit:{split_label}] {fi + 1}/{len(files)} files")

    rows: List[dict] = []
    for w in range(C.N_WHEELS):
        for b, c in enumerate(centers):
            cnt_gt = int(n_gt[w, b])
            cnt_pred = int(n_pred[w, b])
            if cnt_gt >= 200:
                rows.append(dict(run=str(run_dir), split=split_label, wheel=w + 1,
                                 slip_bin=float(c), n=cnt_gt, source="gt",
                                 rms_residual=float(np.sqrt(sse_gt[w, b] / cnt_gt))))
            if cnt_pred >= 200:
                rows.append(dict(run=str(run_dir), split=split_label, wheel=w + 1,
                                 slip_bin=float(c), n=cnt_pred, source="pred",
                                 rms_residual=float(np.sqrt(sse_pred[w, b] / cnt_pred))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Roller residual audit on a v1 checkpoint.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--regime", type=Path, default=None)
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--bins", type=float, nargs="+", default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("observer_v1_py/report_max_norm/roller_residual_audit.csv"))
    args = ap.parse_args()

    bins = np.array(args.bins) if args.bins else SLIP_EDGES
    df = audit_run(args.run_dir, args.regime, args.cache_dir, bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[roller-audit] wrote {len(df)} rows -> {args.out}")
    # Print a concise per-wheel summary for the highest-slip bin.
    hi = df[df["slip_bin"] == df["slip_bin"].max()]
    if not hi.empty:
        print("\nHigh-slip-bin RMS residuals (N·m):")
        print(hi.pivot_table(index=["split", "wheel"], columns="source",
                              values="rms_residual").to_string())


if __name__ == "__main__":
    main()
