#!/usr/bin/env python
# =============================================================================
# evaluation_v2.py — γ-only observability scoring + derived ω_z.
#
# Per-state reconstruction error is now γ-only.  Also scores the derived
# omega_z = psi_dot + gamma * sin_tt * cos(delta) (WZ_P95-normalized) and emits
# the A1-v2 consumer CSV `gamma_error_by_slip.csv`.  Roller-residual-of-
# predictions is reported in binned form for comparison against the audit floor.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import features as F
from .config_v2 import ObserverConfigV2, ROLLER_SCALE
from .models_v2 import WheelObserverV2, build_model_v2
from .physics import roller_residual

SLIP_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf])
WZ_EDGES = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf])


def _centers(edges: np.ndarray) -> np.ndarray:
    c = 0.5 * (edges[:-1] + edges[1:])
    c[-1] = edges[-2] * 1.5 if np.isinf(edges[-1]) else c[-1]
    return c


@torch.no_grad()
def evaluate_observer_v2(cfg: ObserverConfigV2, run_dir: Path, device: torch.device
                         ) -> pd.DataFrame:
    """Score the trained v2 model on val (same-subset) and test (cross-subset)."""
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    model = build_model_v2(cfg).to(device)
    st = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    return evaluate_observer_model_v2(model, cfg, nrm, device)


@torch.no_grad()
def evaluate_observer_model_v2(model: WheelObserverV2, cfg: ObserverConfigV2,
                               nrm: D.Normalizer, device: torch.device) -> pd.DataFrame:
    """Score an already-loaded v2 model on val + test."""
    model.eval()
    splits = D.split_files(D.discover(cfg), cfg)
    rows: List[dict] = []
    for split_key, label in (("val", "same_subset"), ("test", "cross_subset")):
        files = splits.get(split_key, [])
        if files:
            rows.extend(_score_split(files, label, model, nrm, cfg, device))
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_split(files: List[Path], split_label: str, model: WheelObserverV2,
                 nrm: D.Normalizer, cfg: ObserverConfigV2,
                 device: torch.device) -> List[dict]:
    nb_slip, nb_wz = len(SLIP_EDGES) - 1, len(WZ_EDGES) - 1
    # state 0 = gamma, state 1 = derived omega_z
    NS = 2
    shp_o = (C.N_WHEELS, NS)
    shp_s = (C.N_WHEELS, NS, nb_slip)
    shp_w = (C.N_WHEELS, NS, nb_wz)
    sse_n_o = np.zeros(shp_o); sse_p_o = np.zeros(shp_o); n_o = np.zeros(shp_o, np.int64)
    sse_n_s = np.zeros(shp_s); sse_p_s = np.zeros(shp_s); n_s = np.zeros(shp_s, np.int64)
    sse_n_w = np.zeros(shp_w); sse_p_w = np.zeros(shp_w); n_w = np.zeros(shp_w, np.int64)

    # Roller residual accumulators (physical units, for binned report).
    roll_sse = np.zeros((C.N_WHEELS, nb_slip))
    roll_n = np.zeros((C.N_WHEELS, nb_slip), np.int64)

    for fi, path in enumerate(files):
        a = D.read_arrays(path, cfg.cache_dir)
        win = D.make_windows(a, nrm, cfg)
        del a
        if win is None:
            continue
        Gw = torch.from_numpy(win["Gw"]).to(device)
        Pw = torch.from_numpy(win["Pw"]).to(device)
        gamma_hat_n = model(Gw, Pw).cpu().numpy()            # [M,4] normalised
        gamma_hat_p = gamma_hat_n * nrm.y_std[0] + nrm.y_mean[0]
        gamma_lbl_p = win["Yt"][:, :, 0] * nrm.y_std[0] + nrm.y_mean[0]  # [M,4]

        # Derived omega_z (physical).
        psid_signed = win["Gw"][:, -1, 2] * nrm.g_std[2] + nrm.g_mean[2]
        sin_tt = win["sin_tt"]                               # [M,4]
        wz_lbl = win["wz"]
        oz_pred = F.derived_omega_z(gamma_hat_p, sin_tt, psid_signed)

        slip_bin = np.clip(np.digitize(win["vpm"], SLIP_EDGES) - 1, 0, nb_slip - 1)
        wzb = np.clip(np.digitize(np.abs(wz_lbl), WZ_EDGES) - 1, 0, nb_wz - 1)

        # Roller residual on predictions, physical units.
        # Need mu, chi per sample and the physics arrays at window end.
        mu = win.get("ph_mu")
        chi = win.get("ph_chi")
        if mu is not None:
            # Use label gamma for the force recompute (prediction vs label forces).
            # For the binned report we compare the residual under the predicted γ.
            r_pred = _roller_residual_batch(gamma_hat_p, win, nrm, cfg)
            for w in range(C.N_WHEELS):
                sb = slip_bin[:, w]
                np.add.at(roll_sse[w], sb, r_pred[:, w] ** 2)
                np.add.at(roll_n[w], sb, 1)

        for w in range(C.N_WHEELS):
            sb, wb = slip_bin[:, w], wzb[:, w]
            # gamma error
            en = gamma_hat_n[:, w] - win["Yt"][:, w, 0]
            ep = gamma_hat_p[:, w] - gamma_lbl_p[:, w]
            _accumulate(0, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                        sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)
            # derived omega_z error
            ep = oz_pred[:, w] - wz_lbl[:, w]
            en = ep / C.WZ_P95
            _accumulate(1, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                        sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)
        del win
        if (fi + 1) % 50 == 0:
            print(f"[eval-v2:{split_label}] {fi + 1}/{len(files)} files")

    names = ["gamma", "omega_z_derived"]
    rows: List[dict] = []

    def _emit(kind, centers, sse_n, sse_p, n):
        for w in range(C.N_WHEELS):
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

    # Roller residual binned table.
    for w in range(C.N_WHEELS):
        for b, c in enumerate(_centers(SLIP_EDGES)):
            cnt = int(roll_n[w, b])
            if cnt < 200:
                continue
            rows.append(dict(model=cfg.model, window=cfg.window,
                             regime=cfg.regime_name, split=split_label,
                             wheel=w + 1, state="roller_residual_pred", bin_kind="slip",
                             bin_center=float(c), n=cnt,
                             rmse_norm=np.nan,
                             rmse_phys=float(np.sqrt(roll_sse[w, b] / cnt))))

    return rows


def _roller_residual_batch(gamma_hat_p: np.ndarray, win: Dict[str, np.ndarray],
                           nrm: D.Normalizer, cfg: ObserverConfigV2) -> np.ndarray:
    """Compute roller residual [M,4] (physical units) under predicted γ."""
    # Label bristles in physical units.
    y_mean = nrm.y_mean.reshape(1, 1, -1)
    y_std = nrm.y_std.reshape(1, 1, -1)
    Yphys = win["Yt"] * y_std + y_mean
    zx = Yphys[:, :, 1]
    zy = Yphys[:, :, 2]
    zs = np.zeros_like(gamma_hat_p)
    mu = win["ph_mu"]
    chi = win["ph_chi"]
    return roller_residual(
        np, gamma_hat_p, zx, zy, zs, mu, chi,
        win["ph_psi_dot"], win["ph_Vpx0"], win["ph_Vpy0"],
        win["ph_cti"], win["ph_sti"])


def _accumulate(s, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w):
    sse_n_o[w, s] += float(en @ en); sse_p_o[w, s] += float(ep @ ep)
    n_o[w, s] += en.size
    np.add.at(sse_n_s[w, s], sb, en * en); np.add.at(sse_p_s[w, s], sb, ep * ep)
    np.add.at(n_s[w, s], sb, 1)
    np.add.at(sse_n_w[w, s], wb, en * en); np.add.at(sse_p_w[w, s], wb, ep * ep)
    np.add.at(n_w[w, s], wb, 1)
