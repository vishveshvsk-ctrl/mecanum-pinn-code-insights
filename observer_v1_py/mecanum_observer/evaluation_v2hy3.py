#!/usr/bin/env python
# =============================================================================
# evaluation_v2hy3.py — γ + ΔV + derived ω_z observability scoring.
#
# Extends evaluation_v2.py with ΔV̂ diagnostics (V_used RMSE vs V_true binned by
# slip/profile) and emits the A1-v2 consumer CSV `gamma_error_by_slip.csv`.
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
from .config_v2hy3 import ObserverConfigV2Hy3
from .models_v2hy3 import WheelObserverV2Hy3, build_model_v2hy3

SLIP_EDGES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf])
WZ_EDGES = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf])


def _centers(edges: np.ndarray) -> np.ndarray:
    c = 0.5 * (edges[:-1] + edges[1:])
    c[-1] = edges[-2] * 1.5 if np.isinf(edges[-1]) else c[-1]
    return c


@torch.no_grad()
def evaluate_observer_v2hy3(cfg: ObserverConfigV2Hy3, run_dir: Path,
                            device: torch.device) -> pd.DataFrame:
    """Score the trained v2-Hy3 model on val and test."""
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    model = build_model_v2hy3(cfg).to(device)
    st = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    return evaluate_observer_model_v2hy3(model, cfg, nrm, device)


@torch.no_grad()
def evaluate_observer_model_v2hy3(model: WheelObserverV2Hy3, cfg: ObserverConfigV2Hy3,
                                  nrm: D.Normalizer, device: torch.device) -> pd.DataFrame:
    """Score an already-loaded v2-Hy3 model on val + test."""
    model.eval()
    splits = D.split_files(D.discover(cfg), cfg)
    rows: List[dict] = []
    for split_key, label in (("val", "same_subset"), ("test", "cross_subset")):
        files = splits.get(split_key, [])
        if files:
            rows.extend(_score_split(files, label, model, nrm, cfg, device))
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_split(files: List[Path], split_label: str, model: WheelObserverV2Hy3,
                 nrm: D.Normalizer, cfg: ObserverConfigV2Hy3,
                 device: torch.device) -> List[dict]:
    nb_slip, nb_wz = len(SLIP_EDGES) - 1, len(WZ_EDGES) - 1
    # per-wheel states: 0=gamma, 1=omega_z_derived
    NS_W = 2
    shp_o = (C.N_WHEELS, NS_W)
    shp_s = (C.N_WHEELS, NS_W, nb_slip)
    shp_w = (C.N_WHEELS, NS_W, nb_wz)
    sse_n_o = np.zeros(shp_o); sse_p_o = np.zeros(shp_o); n_o = np.zeros(shp_o, np.int64)
    sse_n_s = np.zeros(shp_s); sse_p_s = np.zeros(shp_s); n_s = np.zeros(shp_s, np.int64)
    sse_n_w = np.zeros(shp_w); sse_p_w = np.zeros(shp_w); n_w = np.zeros(shp_w, np.int64)

    # global states: 0=deltaV (normalized), 1=V_used (physical)
    # bin by the max wheel slip per sample.
    sse_g_n_o = np.zeros((2,)); sse_g_p_o = np.zeros((2,)); n_g_o = np.zeros((2,), np.int64)
    sse_g_n_s = np.zeros((2, nb_slip)); sse_g_p_s = np.zeros((2, nb_slip)); n_g_s = np.zeros((2, nb_slip), np.int64)
    sse_g_n_w = np.zeros((2, nb_wz)); sse_g_p_w = np.zeros((2, nb_wz)); n_g_w = np.zeros((2, nb_wz), np.int64)

    for fi, path in enumerate(files):
        # We need the original arrays for truth and the front-end base.
        from .sensor_frontend_v2 import build_sensor_frontend, load_with_accel
        cols = load_with_accel(path)
        fe = build_sensor_frontend(cols, cfg.vel_filter_crossover_hz,
                                   cfg.vel_filter_integrator, C.TRAIN_HZ, None)

        # Build windows using the same logic as training.
        from .data_v2hy3 import _build_arrays_hy3, make_windows_hy3
        a = _build_arrays_hy3(path, cfg)
        if a is None:
            continue
        win = make_windows_hy3(a, nrm, cfg)
        del a
        if win is None:
            continue

        Gw = torch.from_numpy(win["Gw"]).to(device)
        Pw = torch.from_numpy(win["Pw"]).to(device)
        gamma_hat_n, dv_hat_n = model(Gw, Pw)
        gamma_hat_n = gamma_hat_n.cpu().numpy()                   # [M,4]
        dv_hat_n = dv_hat_n.cpu().numpy()                         # [M,2]

        # Physical units.
        gamma_hat_p = gamma_hat_n * nrm.y_std[0] + nrm.y_mean[0]
        dv_hat_p = dv_hat_n * np.array(cfg.dv_scale)[None, :]
        gamma_lbl_p = win["Yt"][:, :, 0] * nrm.y_std[0] + nrm.y_mean[0]

        V_hat_p = win["V_hat"]                                    # [M,2]
        V_true_p = win["V_true"]                                  # [M,2]
        V_used_p = V_hat_p + dv_hat_p

        psid_signed = win["Gw"][:, -1, 2] * nrm.g_std[2] + nrm.g_mean[2]
        sin_tt = win["sin_tt"]
        wz_lbl = win["wz"]
        oz_pred = F.derived_omega_z(gamma_hat_p, sin_tt, psid_signed)

        slip_bin = np.clip(np.digitize(win["vpm"], SLIP_EDGES) - 1, 0, nb_slip - 1)
        wzb = np.clip(np.digitize(np.abs(wz_lbl), WZ_EDGES) - 1, 0, nb_wz - 1)

        for w in range(C.N_WHEELS):
            sb, wb = slip_bin[:, w], wzb[:, w]

            # gamma
            en = gamma_hat_n[:, w] - win["Yt"][:, w, 0]
            ep = gamma_hat_p[:, w] - gamma_lbl_p[:, w]
            _accumulate(0, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                        sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)

            # derived omega_z
            ep = oz_pred[:, w] - wz_lbl[:, w]
            en = ep / C.WZ_P95
            _accumulate(1, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                        sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w)

        # Global ΔV / V_used metrics, binned by the max wheel slip per sample.
        slip_global = win["vpm"].max(axis=-1)
        sb_g = np.clip(np.digitize(slip_global, SLIP_EDGES) - 1, 0, nb_slip - 1)
        # For wz binning use the max |omega_z| over wheels.
        wz_global = np.abs(wz_lbl).max(axis=-1)
        wb_g = np.clip(np.digitize(wz_global, WZ_EDGES) - 1, 0, nb_wz - 1)

        # ΔV error (normalized).
        dv_target = (V_true_p - V_hat_p) / np.array(cfg.dv_scale)[None, :]
        en_dv = (dv_hat_n - dv_target)
        en_dv_mag = np.linalg.norm(en_dv, axis=-1)
        ep_dv = V_used_p - V_true_p
        ep_dv_mag = np.linalg.norm(ep_dv, axis=-1)

        # V_used error (physical).
        en_vu = ep_dv / np.array(cfg.dv_scale)[None, :]
        en_vu_mag = np.linalg.norm(en_vu, axis=-1)
        ep_vu_mag = np.linalg.norm(V_used_p - V_true_p, axis=-1)

        _accumulate_global(0, en_dv_mag, ep_dv_mag, sb_g, wb_g,
                           sse_g_n_o, sse_g_p_o, n_g_o,
                           sse_g_n_s, sse_g_p_s, n_g_s,
                           sse_g_n_w, sse_g_p_w, n_g_w)
        _accumulate_global(1, en_vu_mag, ep_vu_mag, sb_g, wb_g,
                           sse_g_n_o, sse_g_p_o, n_g_o,
                           sse_g_n_s, sse_g_p_s, n_g_s,
                           sse_g_n_w, sse_g_p_w, n_g_w)

        del win
        if (fi + 1) % 50 == 0:
            print(f"[eval-v2hy3:{split_label}] {fi + 1}/{len(files)} files")

    names_w = ["gamma", "omega_z_derived"]
    names_g = ["deltaV", "V_used"]
    rows: List[dict] = []

    def _emit_wheel(kind, centers, sse_n, sse_p, n):
        for w in range(C.N_WHEELS):
            for s in range(NS_W):
                if kind == "overall":
                    cnt = int(n[w, s])
                    if cnt == 0:
                        continue
                    rows.append(dict(model=cfg.model, window=cfg.window,
                                     regime=cfg.regime_name, split=split_label,
                                     wheel=w + 1, state=names_w[s], bin_kind="overall",
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
                                         wheel=w + 1, state=names_w[s], bin_kind=kind,
                                         bin_center=float(centers[b]), n=cnt,
                                     rmse_norm=float(np.sqrt(sse_n[w, s, b] / cnt)),
                                     rmse_phys=float(np.sqrt(sse_p[w, s, b] / cnt))))

    def _emit_global(kind, centers, sse_n, sse_p, n):
        for s in range(2):
            if kind == "overall":
                cnt = int(n[s])
                if cnt == 0:
                    continue
                rows.append(dict(model=cfg.model, window=cfg.window,
                                 regime=cfg.regime_name, split=split_label,
                                 wheel=0, state=names_g[s], bin_kind="overall",
                                 bin_center=np.nan, n=cnt,
                                 rmse_norm=float(np.sqrt(sse_n[s] / cnt)),
                                 rmse_phys=float(np.sqrt(sse_p[s] / cnt))))
            else:
                for b in range(len(centers)):
                    cnt = int(n[s, b])
                    if cnt < 200:
                        continue
                    rows.append(dict(model=cfg.model, window=cfg.window,
                                     regime=cfg.regime_name, split=split_label,
                                     wheel=0, state=names_g[s], bin_kind=kind,
                                     bin_center=float(centers[b]), n=cnt,
                                     rmse_norm=float(np.sqrt(sse_n[s, b] / cnt)),
                                     rmse_phys=float(np.sqrt(sse_p[s, b] / cnt))))

    _emit_wheel("overall", None, sse_n_o, sse_p_o, n_o)
    _emit_wheel("slip", _centers(SLIP_EDGES), sse_n_s, sse_p_s, n_s)
    _emit_wheel("wz", _centers(WZ_EDGES), sse_n_w, sse_p_w, n_w)
    _emit_global("overall", None, sse_g_n_o, sse_g_p_o, n_g_o)
    _emit_global("slip", _centers(SLIP_EDGES), sse_g_n_s, sse_g_p_s, n_g_s)
    _emit_global("wz", _centers(WZ_EDGES), sse_g_n_w, sse_g_p_w, n_g_w)
    return rows


def _accumulate_global(s, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                       sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w):
    sse_n_o[s] += float(en @ en); sse_p_o[s] += float(ep @ ep)
    n_o[s] += en.size
    np.add.at(sse_n_s[s], sb, en * en); np.add.at(sse_p_s[s], sb, ep * ep)
    np.add.at(n_s[s], sb, 1)
    np.add.at(sse_n_w[s], wb, en * en); np.add.at(sse_p_w[s], wb, ep * ep)
    np.add.at(n_w[s], wb, 1)


def _accumulate(s, w, en, ep, sb, wb, sse_n_o, sse_p_o, n_o,
                sse_n_s, sse_p_s, n_s, sse_n_w, sse_p_w, n_w):
    sse_n_o[w, s] += float(en @ en); sse_p_o[w, s] += float(ep @ ep)
    n_o[w, s] += en.size
    np.add.at(sse_n_s[w, s], sb, en * en); np.add.at(sse_p_s[w, s], sb, ep * ep)
    np.add.at(n_s[w, s], sb, 1)
    np.add.at(sse_n_w[w, s], wb, en * en); np.add.at(sse_p_w[w, s], wb, ep * ep)
    np.add.at(n_w[w, s], wb, 1)
