#!/usr/bin/env python
# =============================================================================
# evaluation.py — per-state observability scoring on the held-out test split.
#
# The per-state reconstruction error IS the observability metric (handoff §3):
# low normalised RMSE = observable; a high irreducible floor = non-unique /
# unobservable (expected for the LuGre bristle z, §4.4). We also score the
# DERIVED omega_z from the gamma head (omega_z = psi_dot + gamma sin_tt cos d),
# which bounds Approach-1's chi-channel.
#
# Streaming + memory-safe: accumulate squared-error sufficient stats per
# (wheel, state, bin); never pool raw predictions.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import data as D
from .config import ObserverConfig, N_STATES, N_WHEELS, TARGET_STATES, COS_DELTA, WZ_P95
from .models import build_model

SLIP_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf])
WZ_EDGES = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf])


def _centers(edges: np.ndarray) -> np.ndarray:
    c = 0.5 * (edges[:-1] + edges[1:])
    c[-1] = edges[-2] * 1.5 if np.isinf(edges[-1]) else c[-1]   # label for +inf bin
    return c


@torch.no_grad()
def evaluate_observer(cfg: ObserverConfig, run_dir: Path, device: torch.device
                      ) -> pd.DataFrame:
    """Score the trained model on BOTH the same-subset (val) and cross-subset
    (test) splits, tagging each row with `split`. The same-vs-cross gap is the
    headline generalization signal of the S1/S2/S3 fold designs."""
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    model = build_model(cfg).to(device)
    st = torch.load(run_dir / "checkpoint.pt", map_location=device,
                    weights_only=False)
    model.load_state_dict(st["model"]); model.eval()
    return evaluate_observer_model(model, cfg, nrm, device)


@torch.no_grad()
def evaluate_observer_model(model, cfg: ObserverConfig, nrm, device: torch.device
                            ) -> pd.DataFrame:
    """Score an already-loaded model on the same-subset (val) and cross-subset
    (test) splits. Used by make_observability_report.py and the physics-ablation
    study so the model/data are not reloaded between final and Adam-only runs."""
    model.eval()
    splits = D.split_files(D.discover(cfg), cfg)
    rows: List[dict] = []
    for split_key, label in (("val", "same_subset"), ("test", "cross_subset")):
        files = splits.get(split_key, [])
        if files:
            rows.extend(_score_split(files, label, model, nrm, cfg, device))
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_split(files: List[Path], split_label: str, model, nrm,
                 cfg: ObserverConfig, device: torch.device) -> List[dict]:
    nb_slip, nb_wz = len(SLIP_EDGES) - 1, len(WZ_EDGES) - 1
    # state index N_STATES is the derived omega_z (an extra pseudo-state)
    NS = N_STATES + 1
    shp_o = (N_WHEELS, NS); shp_s = (N_WHEELS, NS, nb_slip); shp_w = (N_WHEELS, NS, nb_wz)
    sse_n_o = np.zeros(shp_o); sse_p_o = np.zeros(shp_o); n_o = np.zeros(shp_o, np.int64)
    sse_n_s = np.zeros(shp_s); sse_p_s = np.zeros(shp_s); n_s = np.zeros(shp_s, np.int64)
    sse_n_w = np.zeros(shp_w); sse_p_w = np.zeros(shp_w); n_w = np.zeros(shp_w, np.int64)

    for fi, path in enumerate(files):
        a = D.read_arrays(path, cfg.cache_dir)
        win = D.make_windows(a, nrm, cfg)
        del a
        if win is None:
            continue
        Gw = torch.from_numpy(win["Gw"]).to(device)
        Pw = torch.from_numpy(win["Pw"]).to(device)
        pred_n = model(Gw, Pw).cpu().numpy()                 # [M,4,4] normalised
        tgt_n = win["Yt"]                                    # [M,4,4] normalised
        # physical
        pred_p = pred_n * nrm.y_std + nrm.y_mean
        tgt_p = tgt_n * nrm.y_std + nrm.y_mean
        # derived omega_z (physical): psi_dot + gamma * sin_tt * cos(delta)
        psid_signed = win["Gw"][:, -1, 2] * nrm.g_std[2] + nrm.g_mean[2]  # de-norm psi_dot@end
        sin_tt = win["sin_tt"]                               # [M,4] raw
        wz_lbl = win["wz"]                                   # [M,4] raw label
        oz_pred = psid_signed[:, None] + pred_p[:, :, 0] * sin_tt * COS_DELTA[None, :]
        # Frozen global p95 (NOT per-file std, which exploded at low spin and made
        # the metric incomparable to the p95-normalised gamma/zx/zy). See config.WZ_P95.
        oz_std = WZ_P95

        slip_bin = np.clip(np.digitize(win["vpm"], SLIP_EDGES) - 1, 0, nb_slip - 1)
        wzb = np.clip(np.digitize(np.abs(wz_lbl), WZ_EDGES) - 1, 0, nb_wz - 1)

        for w in range(N_WHEELS):
            sb, wb = slip_bin[:, w], wzb[:, w]
            for s in range(N_STATES):
                en = pred_n[:, w, s] - tgt_n[:, w, s]
                ep = pred_p[:, w, s] - tgt_p[:, w, s]
                _accumulate(s, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                            sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)
            # derived omega_z as pseudo-state index N_STATES
            ep = oz_pred[:, w] - wz_lbl[:, w]
            en = ep / oz_std
            _accumulate(N_STATES, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                        sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)
        del win
        if (fi + 1) % 50 == 0:
            print(f"[eval:{split_label}] {fi + 1}/{len(files)} files")

    names = list(TARGET_STATES) + ["omega_z_derived"]
    rows: List[dict] = []

    def _emit(kind, centers, sse_n, sse_p, n):
        for w in range(N_WHEELS):
            for s in range(NS):
                if kind == "overall":
                    cnt = int(n[w, s])
                    if cnt == 0:
                        continue
                    rows.append(dict(model=cfg.model, window=cfg.window,
                                     regime=cfg.regime_name, split=split_label,
                                     wheel=w + 1, state=names[s], bin_kind="overall",
                                     bin_center=np.nan, n=cnt,
                                     rmse_norm=float(np.sqrt(sse_n[w, s] / cnt)),
                                     rmse_phys=float(np.sqrt(sse_p[w, s] / cnt))))
                else:
                    for b in range(len(centers)):
                        cnt = int(n[w, s, b])
                        if cnt < 200:
                            continue
                        rows.append(dict(model=cfg.model, window=cfg.window,
                                         regime=cfg.regime_name, split=split_label,
                                         wheel=w + 1, state=names[s], bin_kind=kind,
                                         bin_center=float(centers[b]), n=cnt,
                                         rmse_norm=float(np.sqrt(sse_n[w, s, b] / cnt)),
                                         rmse_phys=float(np.sqrt(sse_p[w, s, b] / cnt))))

    _emit("overall", None, sse_n_o, sse_p_o, n_o)
    _emit("slip", _centers(SLIP_EDGES), sse_n_s, sse_p_s, n_s)
    _emit("wz", _centers(WZ_EDGES), sse_n_w, sse_p_w, n_w)
    return rows


def _accumulate(s, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w):
    sse_n_o[w, s] += float(en @ en); sse_p_o[w, s] += float(ep @ ep)
    n_o[w, s] += en.size
    np.add.at(sse_n_s[w, s], sb, en * en); np.add.at(sse_p_s[w, s], sb, ep * ep)
    np.add.at(n_s[w, s], sb, 1)
    np.add.at(sse_n_w[w, s], wb, en * en); np.add.at(sse_p_w[w, s], wb, ep * ep)
    np.add.at(n_w[w, s], wb, 1)
