#!/usr/bin/env python
# =============================================================================
# features.py — measurable-only feature + hidden-state target construction.
#
# Inputs are sensor-measurable ONLY (hard rule). Targets are the hidden states
# read straight from the Arrow columns (JLD2 sidecars are not stored).
#
# Geometry note: theta enters exclusively through sin/cos of the *folded* angle
# theta_tilde, computed with the EXACT smooth approximator the simulator used to
# integrate the labels:
#     theta_tilde = atan2(K*sin(12 theta), K*cos(12 theta) + 1) / 12     (K=60)
# This is C-infinity (no roller-handoff discontinuity) and is the identical
# sin(theta_tilde)/cos(theta_tilde) the friction law / omega_z consumed, so the
# input geometry matches the target geometry with zero systematic offset.
# =============================================================================
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C


def sawtooth_tanh(theta: np.ndarray, k: float = C.TANH_K) -> np.ndarray:
    """Smooth (C-inf) folded wheel angle, period pi/6, range +/-pi/12.

    Replicates run_one.jl `sawtooth_tanh` exactly (SAWTOOTH=:tanh, TANH_K=60)."""
    s = np.sin(12.0 * theta)
    c = np.cos(12.0 * theta)
    return np.arctan2(k * s, k * c + 1.0) / 12.0


# Columns we actually pull from each Arrow file (measurables + targets + aux).
def required_columns() -> list[str]:
    cols = list(C.GLOBAL_COLS)
    for i in range(1, C.N_WHEELS + 1):
        for tmpl in C.PERWHEEL_MEAS_COLS:
            cols.append(tmpl.format(i=i))
        for st in C.TARGET_STATES:
            cols.append(C.TARGET_COL[st].format(i=i))
        for tmpl in C.AUX_COLS:
            cols.append(tmpl.format(i=i))
    return cols


def build_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """From a (decimated) trajectory DataFrame build:

      G    [T, 3]            global measurable features (Vx, Vy, psi_dot)
      P    [T, 4, 6]         per-wheel measurable features
      Y    [T, 4, 4]         hidden-state targets (gamma, zx, zy, zs)
      wz   [T, 4]            omega_z label (for derived-omega_z eval)
      vpm  [T, 4]            true slip magnitude |Vp| (for slip binning)
      psid [T]               |psi_dot| (for yaw-rate stratification)
    """
    T = len(df)
    Vx = df["Vx"].to_numpy(np.float64)
    Vy = df["Vy"].to_numpy(np.float64)
    psi_dot = df["psi_dot"].to_numpy(np.float64)
    G = np.stack([Vx, Vy, psi_dot], axis=1)               # [T, 3]

    P = np.empty((T, C.N_WHEELS, C.N_PERWHEEL), np.float64)   # INPUT features
    Y = np.empty((T, C.N_WHEELS, C.N_STATES), np.float64)
    wz = np.empty((T, C.N_WHEELS), np.float64)
    vpm = np.empty((T, C.N_WHEELS), np.float64)
    # gamma=0 slip surrogate — NOT a network input (see config.PERWHEEL_FEATURES),
    # kept only for the physics loss / contact-kinematics residuals.
    Vpx0 = np.empty((T, C.N_WHEELS), np.float64)
    Vpy0 = np.empty((T, C.N_WHEELS), np.float64)

    for k in range(C.N_WHEELS):
        i = k + 1
        msat = df[f"Msat_{i}"].to_numpy(np.float64)
        w = df[f"w{i}"].to_numpy(np.float64)
        theta = df[f"theta{i}"].to_numpy(np.float64)
        tt = sawtooth_tanh(theta)                          # theta_tilde
        sin_tt, cos_tt = np.sin(tt), np.cos(tt)
        P[:, k, 0] = msat
        P[:, k, 1] = w
        P[:, k, 2] = sin_tt
        P[:, k, 3] = cos_tt

        # gamma=0 contact-point slip surrogate (for physics loss only):
        #   DYi   = Rd*tan(delta_i)*tan(theta_tilde_i)
        #   Vpx0  = Vx - psi_dot*(py_i + DYi) - w_i*R ;  Vpy0 = Vy + psi_dot*px_i
        DYi = C.RD * C.TAN_DELTA[k] * (sin_tt / cos_tt)
        Vpx0[:, k] = Vx - psi_dot * (C.PY[k] + DYi) - w * C.R
        Vpy0[:, k] = Vy + psi_dot * C.PX[k]

        Y[:, k, 0] = df[f"gamma{i}"].to_numpy(np.float64)   # roller rate
        Y[:, k, 1] = df[f"zx_{i}"].to_numpy(np.float64)     # linear bristle x
        Y[:, k, 2] = df[f"zy_{i}"].to_numpy(np.float64)     # linear bristle y
        # zs (spin bristle) intentionally not a target — see config.TARGET_STATES

        wz[:, k] = df[f"wz_{i}"].to_numpy(np.float64)
        vpm[:, k] = np.hypot(df[f"Vpx_{i}"].to_numpy(np.float64),
                             df[f"Vpy_{i}"].to_numpy(np.float64))

    return dict(G=G, P=P, Y=Y, wz=wz, vpm=vpm, psid=np.abs(psi_dot),
                Vpx0=Vpx0, Vpy0=Vpy0)


def derived_omega_z(gamma: np.ndarray, sin_tt: np.ndarray,
                    psi_dot: np.ndarray) -> np.ndarray:
    """omega_z = psi_dot + gamma * sin(theta_tilde) * cos(delta).

    `gamma`, `sin_tt` are [..., 4]; `psi_dot` is [...]. cos(delta) is the
    constant 1/sqrt(2) on all four wheels. Used to score the chi-channel bound
    from a gamma prediction (handoff §4.4)."""
    return psi_dot[..., None] + gamma * sin_tt * C.COS_DELTA[None, :]
