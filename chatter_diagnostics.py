#!/usr/bin/env python
# =============================================================================
# chatter_diagnostics.py — profiles-era trajectory chatter screen.
#
# Flags trajectories whose force/torque signals carry UNWANTED chatter
#   (ASMC sliding-mode switching that leaks into the physics, or numerical hash)
# WITHOUT penalising the two physical oscillations the PINN must learn:
#   roller-switching ripple  and  LuGre stick-slip oscillation.
#
# Pure signal-processing on the recorded Arrow columns — no Julia, no ODE
# re-solve, no label re-injection. Every input is already a column written by
# datastore.jl. See Trajectory_Chatter_Diagnostics_PLAN.md for the full spec
# and the physics derivation behind every band/threshold.
#
# CLI (mirrors the Data_Generation_Julia.jl discipline: streaming, resume):
#   python chatter_diagnostics.py --data-dir ../data/Simulation_Data_... \
#       --out chatter_report.csv [--profiles octagon,spin_creep] \
#       [--jobs 8] [--limit 50]
#
# Thresholds in CLASSIFIER_THRESHOLDS are FIRST-PRINCIPLES PLACEHOLDERS, to be
# replaced after the metric-sweep + per-profile histogram pass (plan §8.4).
# =============================================================================
from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.signal import butter, coherence, decimate, hilbert, sosfiltfilt, stft, welch

# =============================================================================
# 1. Cross-language contract — filename scheme + columns (from datastore.jl).
#
# Filename (datastore.jl output_prefix/expected_output):
#   <profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
#   e.g.  octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
# `fm` is anchored to the known set so the underscores in profile/fm can't
# collide with the field delimiters. NOTE: data.py's _FNAME_RE is still on the
# old beta/amp scheme; this regex is the migration target (plan §6).
# =============================================================================
_FNAME_RE = re.compile(
    r'^(?P<profile>.+?)_c(?P<combo>\d{3})_mu_(?P<mu>[0-9.eE+\-]+)'
    r'_case(?P<fc>\d+)_(?P<fm>lugre_adamov|lugre_uncoupled)'
    r'_chi_(?P<chi>[0-9.]+)\.arrow$'
)

# ellipse is the only PosRef profile; the rest are VelRef. Only used for
# labelling — the chatter metrics ignore the reference block entirely.
_POSREF_PROFILES = frozenset({'ellipse'})

# Columns pulled per file. Required-contract + the profiles-era bonus columns
# (Fpar/Fperp/util/Msw/Meq/Vp*/wz) that this diagnostic leans on.
def _wheel_cols(stub: str) -> List[str]:
    return [f'{stub}_{i}' for i in range(1, 5)]

REQUIRED_COLS = (['time', 'Vx', 'Vy', 'psi_dot']
                 + [f'w{i}' for i in range(1, 5)]
                 + [f'theta{i}' for i in range(1, 5)]
                 + _wheel_cols('Fpar') + _wheel_cols('Fperp')
                 + _wheel_cols('util') + _wheel_cols('Msw') + _wheel_cols('Meq')
                 + _wheel_cols('Vpx') + _wheel_cols('Vpy') + _wheel_cols('wz')
                 + _wheel_cols('Mz') + _wheel_cols('Msat'))      # Msat for M7 burst

# =============================================================================
# 2. Plant constants that set the physical frequency bands (from the notebook).
#    Derivations + numeric tables: PLAN §1, §3.
# =============================================================================
N_ROLLERS      = 12        # sawtooth_tanh period 2*pi/12
SIGMA0         = 1.64e3    # LuGre translational bristle stiffness [1/m]
V_STR          = 0.01      # Stribeck velocity [m/s]
STICTION_RATIO = 1.1       # mu_s/mu_c
SAVEAT_RATE    = 2000.0    # Hz dense uniform grid (base.toml)
MAX_TORQUE     = 10.0      # base.toml Max_torque (per-wheel cap, N·m) — M7 normalization
BURST_FLOOR    = 0.10      # N·m = 1% of Max_torque — the "silent HF torque" baseline.
                           # M7 normalizes the localized burst by max/(p10_window + FLOOR);
                           # when a trajectory's quiet baseline goes truly silent the p10
                           # window-RMS -> 0, and this floor keeps the ratio a *ratio*
                           # (burst vs the quietest real control level) instead of
                           # degenerating into max/eps and exploding on clean blips.
# M7 high-pass: ~5× the wheel mechanical corner p1/(2π·Jw)=0.11/(2π·5.87e-3)≈3.0 Hz.
# Above the wheel corner the heavy inertia low-passes the torque, so it does no
# useful tracking work; 15 Hz sits above all functional control (refs ≤1 Hz) and
# below the measured ~29 Hz burst (and the 77 Hz roller-dynamics corner).
WHEEL_CORNER_HZ = 0.11 / (2 * math.pi * 5.87e-3)  # ≈ 2.98 Hz (friction_case 1)

F_ROLL_COEF = N_ROLLERS / (2 * math.pi)          # f_roll = 1.910 * |omega|


@dataclass
class ChatterConfig:
    """Tunable knobs for the metric pass. Defaults are physically grounded
    (PLAN §3); classifier cutoffs live in CLASSIFIER_THRESHOLDS separately so
    they can be re-fit from the sweep without touching the kernels."""
    decimate_factor: int = 8           # 2000 -> 250 Hz for ridge/fold (Nyq 125 > 3rd harmonic)
    # ridge band (M1)
    ridge_tol_rel: float = 0.25        # +/-25% of h*f_roll
    ridge_floor_hz: float = 1.0        # band floor near stall
    ridge_n_harmonics: int = 3
    # LuGre-aware noise floor (M2/M5)
    k_lugre: float = 1.5               # widen the LuGre band before calling it hash
    f_floor_hz: float = 40.0           # absolute floor for f_phys
    fphys_pctile: float = 95.0         # per-trajectory hash cutoff = pctile_t(f_phys)
    # spectral estimators
    welch_nperseg: int = 4096
    coh_nperseg: int = 2048
    stft_nperseg: int = 256            # on the decimated grid
    # high-pass pre-conditioning (remove slow trajectory envelope before STFT)
    hp_safety: float = 0.3             # cutoff = hp_safety * min moving f_roll
    hp_floor_hz: float = 1.0
    # theta-fold (M6)
    fold_nbins: int = 80
    chi_coulomb_tol: float = 1e-6
    # M7 control-torque burst
    burst_hp_hz: float = 15.0          # ≈5× wheel corner; see WHEEL_CORNER_HZ note above
    burst_win_sec: float = 0.25        # sliding-window length for localized-burst RMS
    burst_baseline_pctile: float = 10.0  # denominator = this percentile of window-RMS.
                                       # The MEDIAN failed when the disturbed fraction
                                       # (startup chatter + end burst) exceeds ~50% of the
                                       # run — it gets dragged up into the disturbed
                                       # population and the ratio collapses. p10 estimates
                                       # the quiet baseline robustly to a large disturbed
                                       # fraction (validated on coupled_vomega give-up bursts).


# First-principles PLACEHOLDERS. The sweep pass (plan §9.5-6) replaces these
# with per-profile cutoffs picked at the histogram shoulders. Documented so a
# reader knows none of these are yet data-validated.
CLASSIFIER_THRESHOLDS = dict(
    tau_hash=0.15,        # M2: >15% of off-DC energy above the physical ceiling
    tau_coh_low=0.30,     # M4: coherence below this => energy not explained by Msw => hash
    tau_coh_high=0.50,    # M4: coherence above this => coherent with Msw => control chatter
    tau_slipmod=0.30,     # M5: HF envelope-vs-|Vp| corr above this => LuGre (veto hash)
    tau_ctrl=0.20,        # M3: Msw HF energy / RMS(Meq) above this => switching is chattering
    util_sat_frac=0.50,   # M0: >50% of time above 0.8 budget => saturation context
    tau_burst=10.0,       # M7: burstiness max/(p10 window + FLOOR) above this => localized burst.
                          #     Tuned on real data (p10 denominator, FLOOR=0.10): clean trajectories
                          #     ceil at ~5.5 (multisine 1.4), give-up bursts start ~13 — tau=10 sits
                          #     in the >2x gap. (Old 100 was for the max/median denominator, which
                          #     let strong end-of-run bursts like coupled_vomega c057/c051 ~93 escape.)
)


# =============================================================================
# 3. Filename parsing
# =============================================================================
def parse_arrow_filename(name: str) -> Optional[Dict[str, Any]]:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return {
        'profile':        m['profile'],
        'combo_idx':      int(m['combo']),
        'mu':             float(m['mu']),
        'friction_case':  int(m['fc']),
        'friction_model': m['fm'],
        'chi':            float(m['chi']),
        'is_posref':      m['profile'] in _POSREF_PROFILES,
    }


# =============================================================================
# 4. Small DSP helpers
# =============================================================================
def _highpass(sig: np.ndarray, fs: float, fc_hz: float, order: int = 4) -> np.ndarray:
    fc_hz = min(max(fc_hz, 1e-3), 0.45 * fs)
    sos = butter(order, fc_hz, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, sig)


def _hp_cutoff(omega_abs: np.ndarray, fs: float, cfg: ChatterConfig) -> float:
    """Cutoff just below the slowest moving roller frequency (diag-1 rule)."""
    moving = omega_abs > 0.5
    if moving.any():
        f_roll_min = F_ROLL_COEF * float(omega_abs[moving].min())
        fc = max(cfg.hp_floor_hz, cfg.hp_safety * f_roll_min)
    else:
        fc = cfg.hp_floor_hz
    return min(fc, 0.45 * fs)


def _stribeck_g(s: np.ndarray, mu: float) -> np.ndarray:
    """g_t(s) = mu*(1 + (ratio-1) exp(-(s/v_str)^2)).  g ~ mu in gross slip."""
    return mu * (1.0 + (STICTION_RATIO - 1.0) * np.exp(-(s / V_STR) ** 2))


def _welch_psd(sig: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, len(sig))
    f, P = welch(sig, fs=fs, nperseg=nperseg, detrend='constant')
    return f, P


def _band_power(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return 0.0
    return float(np.trapezoid(P[m], f[m]))


# =============================================================================
# 5. Physical frequency bands (per-sample) and the per-trajectory hash cutoff.
# =============================================================================
def f_roll_series(omega_abs: np.ndarray) -> np.ndarray:
    return F_ROLL_COEF * omega_abs


def f_lugre_series(vp_abs: np.ndarray, mu: float) -> np.ndarray:
    """Bristle relaxation corner  f_lugre = sigma0 * Vp / (g_t * 2pi) ~ 653*Vp."""
    g = _stribeck_g(vp_abs, mu)
    return SIGMA0 * vp_abs / (g * 2 * math.pi)


def hash_cutoff_hz(omega_abs: np.ndarray, vp_abs: np.ndarray, mu: float,
                   cfg: ChatterConfig) -> float:
    """F_hash = pctile_t( max(3*f_roll, k*f_lugre, f_floor) ).  The LuGre-aware
    ceiling above which off-ridge energy is unexplained by physics (PLAN §3 M2)."""
    fr = cfg.ridge_n_harmonics * f_roll_series(omega_abs)
    fl = cfg.k_lugre * f_lugre_series(vp_abs, mu)
    f_phys = np.maximum.reduce([fr, fl, np.full_like(fr, cfg.f_floor_hz)])
    return float(np.percentile(f_phys, cfg.fphys_pctile))


# =============================================================================
# 6. Metric kernels  (each per-wheel; reduced by median over wheels at the end)
# =============================================================================
def m1_ridge_concentration(fpar: np.ndarray, omega_abs: np.ndarray, fs: float,
                           cfg: ChatterConfig) -> float:
    """Fraction of off-DC STFT energy of Fpar within +/-tol of h*f_roll(t).
    Computed on the DECIMATED grid. High => roller-ridge dominated => good."""
    fc = _hp_cutoff(omega_abs, fs, cfg)
    sig = _highpass(fpar, fs, fc)
    nperseg = min(cfg.stft_nperseg, len(sig))
    if nperseg < 8:
        return np.nan
    noverlap = nperseg - max(1, nperseg // 8)
    f, tt, Z = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                    detrend='constant', boundary=None, padded=False)
    P = np.abs(Z) ** 2
    if P.shape[1] == 0:
        return np.nan
    # f_roll on the STFT time grid
    t_idx = np.linspace(0, len(omega_abs) - 1, P.shape[1]).astype(int)
    fr = f_roll_series(omega_abs[t_idx])                       # (T_stft,)
    mask = np.zeros_like(P, dtype=bool)
    for h in range(1, cfg.ridge_n_harmonics + 1):
        center = h * fr
        tol = np.maximum(cfg.ridge_tol_rel * center, cfg.ridge_floor_hz)
        mask |= np.abs(f[:, None] - center[None, :]) <= tol[None, :]
    mask[0, :] = False
    off_dc = P[1:, :].sum()
    if off_dc < 1e-30:
        return np.nan
    return float(P[mask].sum() / off_dc)


def m2_hash_fraction(sig: np.ndarray, fs: float, f_hash: float,
                     cfg: ChatterConfig) -> float:
    """Off-DC PSD energy above the LuGre-aware physical ceiling. FULL rate.
    Low = clean. This is the core unwanted-chatter detector."""
    f, P = _welch_psd(sig, fs, cfg.welch_nperseg)
    off_dc = _band_power(f, P, f[1] if len(f) > 1 else 0.0, fs / 2)
    if off_dc < 1e-30:
        return np.nan
    hf = _band_power(f, P, f_hash, fs / 2)
    return float(hf / off_dc)


def m3_control_chatter_index(msw: np.ndarray, meq: np.ndarray, fs: float,
                             f_hash: float, cfg: ChatterConfig) -> float:
    """sqrt(HF power of Msw above f_hash) / (RMS(Meq)+eps). FULL rate.
    High => the switching wrench itself is chattering (ASMC limit-cycle)."""
    f, P = _welch_psd(msw, fs, cfg.welch_nperseg)
    hf = _band_power(f, P, f_hash, fs / 2)
    rms_meq = float(np.sqrt(np.mean(meq.astype(np.float64) ** 2)))
    return float(math.sqrt(max(hf, 0.0)) / (rms_meq + 1e-9))


def m4_msw_force_coherence_hf(msw: np.ndarray, fpar: np.ndarray, fs: float,
                              f_hash: float, cfg: ChatterConfig) -> float:
    """Msw-power-WEIGHTED magnitude-squared coherence gamma^2(Msw, Fpar) over the
    band > f_hash. FULL rate. Weighting by the Msw PSD asks the right question —
    'where the switching wrench has HF energy, is the force coherent with it?' —
    so a narrow coherent chatter tone gets full weight instead of being averaged
    away against the quiet rest of the band. High => coherent control chatter;
    low + high M2 => incoherent numerical hash."""
    nperseg = min(cfg.coh_nperseg, len(msw))
    if nperseg < 16:
        return np.nan
    f, Cxy = coherence(msw, fpar, fs=fs, nperseg=nperseg)
    _, Pmsw = welch(msw, fs=fs, nperseg=nperseg, detrend='constant')   # same f grid
    m = f >= f_hash
    if not m.any():
        return np.nan
    w = Pmsw[m]
    if w.sum() < 1e-30:
        return float(np.nanmean(Cxy[m]))
    return float(np.sum(Cxy[m] * w) / np.sum(w))


def m5_hf_slip_modulation(fpar: np.ndarray, vp_abs: np.ndarray, omega_abs: np.ndarray,
                          fs: float, f_hash: float, cfg: ChatterConfig) -> float:
    """Corr( HF-force envelope, |Vp| ) in the HASH band [f_hash, 0.45*fs] — the
    SAME band M2 flags. FULL rate. This is the LuGre rescue: if the above-ceiling
    energy is amplitude-modulated by slip it is genuine high-slip stick-slip that
    poked above the percentile cutoff (signal) => veto the hash verdict; if it is
    white (flat envelope, no slip correlation) it is numerical hash. NaN when
    there is no energy above f_hash (then M2 is ~0 and the hash branch can't fire
    anyway)."""
    bp_lo, bp_hi = f_hash, 0.45 * fs
    if bp_hi <= bp_lo:
        return np.nan
    sos = butter(4, [bp_lo, bp_hi], btype='bandpass', fs=fs, output='sos')
    band = sosfiltfilt(sos, fpar)
    env = np.abs(hilbert(band))
    if np.std(env) < 1e-12 or np.std(vp_abs) < 1e-12:
        return np.nan
    return float(np.corrcoef(env, vp_abs)[0, 1])


def m6_theta_fold_tightness(sig: np.ndarray, theta: np.ndarray, fs: float,
                            omega_abs: np.ndarray, is_coulomb: bool,
                            cfg: ChatterConfig) -> float:
    """median_bin IQR / range(bin medians) after folding over theta mod 2pi/12.
    DECIMATED grid. Low = clean periodic-in-theta ripple. (sig = Mz for chi>0,
    high-passed Fperp for chi==0; see diag-1's chi-branch.)"""
    half = (2 * math.pi / N_ROLLERS) / 2.0
    if is_coulomb:
        fc = _hp_cutoff(omega_abs, fs, cfg)
        sig = _highpass(sig, fs, fc)
    elif np.std(sig) < 1e-9:
        return np.nan
    th_mod = np.mod(theta + half, 2 * half) - half
    bins = np.linspace(-half, half, cfg.fold_nbins + 1)
    which = np.clip(np.digitize(th_mod, bins) - 1, 0, cfg.fold_nbins - 1)
    iqr_per_bin, med_per_bin = [], []
    for k in range(cfg.fold_nbins):
        vs = sig[which == k]
        if len(vs) < 3:
            continue
        q25, q50, q75 = np.percentile(vs, [25, 50, 75])
        iqr_per_bin.append(q75 - q25)
        med_per_bin.append(q50)
    if not iqr_per_bin:
        return np.nan
    rng = float(np.max(med_per_bin) - np.min(med_per_bin))
    if rng < 1e-9:
        return np.nan
    return float(np.median(iqr_per_bin) / rng)


def m7_msat_burstiness(msat: np.ndarray, fs: float, cfg: ChatterConfig) -> Tuple[float, float]:
    """Localized control-torque BURST detector (one wheel). FULL rate.
    Returns (burstiness, abs_norm).

    A saturated controller that loses tracking can erupt in a brief, violent
    `Msat` oscillation (an ASMC limit cycle) — typically at the END of a run.
    The spectral metrics MISS this: it is (a) localized, so the whole-trajectory
    PSD dilutes it, and (b) under gross slip `F_hash` inflates above the Nyquist,
    so M2/M3 are exactly 0. M7 sidesteps both by working on the CONTROL torque
    (no roller ripple / LuGre there) and measuring LOCALIZATION, not amplitude:

      * high-pass at `burst_hp_hz` (≈5× the wheel mechanical corner): above this,
        torque can't accelerate the wheel (non-functional), and it sits below the
        measured ~29 Hz burst;
      * sliding-window RMS (`burst_win_sec`) of the high-passed torque;
      * `burstiness = max_window / (p10_window + floor)`: a transient spike
        scores ≫1, while pervasive broadband control (e.g. multisine, which is
        legitimately oscillatory) stays ~1 because its HF is FLAT, not spiky.
        (Absolute HF amplitude FAILS here — multisine ≈ a real burst, ~0.1 — but
        the ratio separates them by orders of magnitude.)
        The denominator is the p10 window-RMS (`burst_baseline_pctile`), NOT the
        median: when the disturbed fraction (startup chatter + an end-of-run burst)
        exceeds ~50% of the run, the median is pulled up into the disturbed
        population and the ratio collapses, letting strong bursts escape. p10
        tracks the quiet baseline robustly to a large disturbed fraction. `FLOOR`
        is a physical silent-torque level so a truly-silent baseline (p10->0)
        doesn't degenerate the ratio into max/eps.
    `abs_norm = max_window_HF_RMS / Max_torque` is reported as secondary context.
    """
    win = max(1, int(cfg.burst_win_sec * fs))
    n = len(msat) // win
    if n < 2:
        return np.nan, np.nan
    fc = min(cfg.burst_hp_hz, 0.45 * fs)
    sos = butter(4, fc, btype='highpass', fs=fs, output='sos')
    hf = np.abs(sosfiltfilt(sos, msat))
    wr = np.sqrt((hf[:n * win].reshape(n, win) ** 2).mean(axis=1))
    mx, base = float(wr.max()), float(np.percentile(wr, cfg.burst_baseline_pctile))
    return mx / (base + BURST_FLOOR), mx / MAX_TORQUE


# =============================================================================
# 7. Per-trajectory driver
# =============================================================================
def _decimate_cols(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor < 2:
        return arr
    return decimate(arr, factor, n=8, ftype='iir', axis=0, zero_phase=True)


def diagnose_columns(t: np.ndarray, W: Dict[str, np.ndarray], omega: np.ndarray,
                     theta: np.ndarray, mu: float, chi: float,
                     cfg: ChatterConfig) -> Dict[str, Any]:
    """Compute the metric row + verdict from already-loaded per-wheel arrays.

    Split out of `diagnose_file` so callers that hold arrays at an arbitrary
    sample rate (e.g. the sampling-rate sensitivity study, which resamples in
    memory) can get a verdict without round-tripping through a temp Arrow file.
    `W` holds (N,4) arrays for Fpar/Fperp/util/Msw/Meq/Vpx/Vpy/wz/Mz; `omega`
    and `theta` are (N,4). The sample rate is inferred from `t`."""
    row: Dict[str, Any] = {}
    if t.size < 64:
        row['error'] = f'too short ({t.size} samples)'
        return row
    fs_full = 1.0 / float(np.median(np.diff(t)))
    is_coulomb = abs(chi) < cfg.chi_coulomb_tol
    vp = np.hypot(W['Vpx'], W['Vpy'])

    # M0 — util saturation (cheap, no spectra)
    util_max = W['util'].max(axis=1)
    row['util_sat_frac_0p8'] = float(np.mean(util_max > 0.8))
    row['util_sat_frac_1p0'] = float(np.mean(util_max > 1.0))

    # Decimated copies for ridge/fold (theta index-sliced, never filtered). The
    # decimate factor is clamped so it never collapses a low-rate input below
    # the STFT window — matters when this runs on already-downsampled arrays.
    fac = max(1, min(cfg.decimate_factor, t.size // (cfg.stft_nperseg * 2) or 1))
    fs_dec = fs_full / fac
    fpar_dec = _decimate_cols(W['Fpar'], fac)
    fperp_dec = _decimate_cols(W['Fperp'], fac)
    mz_dec = _decimate_cols(W['Mz'], fac)
    omega_dec = _decimate_cols(omega, fac)
    theta_dec = theta[::fac]

    m1, m2, m3, m4, m5, m6 = ([] for _ in range(6))
    fhash_list = []
    for i in range(4):
        oa, va = np.abs(omega[:, i]), vp[:, i]
        f_hash = hash_cutoff_hz(oa, va, mu, cfg)
        fhash_list.append(f_hash)
        # full-rate metrics
        m2.append(m2_hash_fraction(W['Fpar'][:, i], fs_full, f_hash, cfg))
        m3.append(m3_control_chatter_index(W['Msw'][:, i], W['Meq'][:, i], fs_full, f_hash, cfg))
        m4.append(m4_msw_force_coherence_hf(W['Msw'][:, i], W['Fpar'][:, i], fs_full, f_hash, cfg))
        m5.append(m5_hf_slip_modulation(W['Fpar'][:, i], va, oa, fs_full, f_hash, cfg))
        # decimated metrics
        m1.append(m1_ridge_concentration(fpar_dec[:, i], np.abs(omega_dec[:, i]), fs_dec, cfg))
        fold_sig = mz_dec[:, i] if not is_coulomb else fperp_dec[:, i]
        m6.append(m6_theta_fold_tightness(fold_sig, theta_dec[:, i], fs_dec,
                                          np.abs(omega_dec[:, i]), is_coulomb, cfg))

    # M7 — localized control-torque burst (on Msat, full rate; max over wheels)
    b7 = [m7_msat_burstiness(W['Msat'][:, i], fs_full, cfg) for i in range(4)]
    burstiness = [b for b, _ in b7]
    burst_abs = [a for _, a in b7]

    nm = lambda v: float(np.nanmedian(v)) if np.any(np.isfinite(v)) else np.nan
    nmax = lambda v: float(np.nanmax(v)) if np.any(np.isfinite(v)) else np.nan
    row.update({
        'n_samples': int(t.size), 'fs': fs_full, 'f_hash_med': float(np.median(fhash_list)),
        'ridge_concentration': nm(m1),
        'hash_fraction': nm(m2),
        'control_chatter_index': nm(m3),
        'msw_force_coherence_hf': nm(m4),
        'hf_slip_modulation': nm(m5),
        'theta_fold_tightness': nm(m6),
        'msat_burstiness': nmax(burstiness),       # worst wheel — a burst on any wheel counts
        'msat_burst_abs': nmax(burst_abs),
        'error': None,
    })
    row['verdict'], row['chatter_flag'] = classify(row, CLASSIFIER_THRESHOLDS)
    # M7 is an independent flag (a localized control burst), reported alongside —
    # not folded into the spectral verdict (which is structurally blind to it).
    row['burst_flag'] = bool(np.isfinite(row['msat_burstiness'])
                             and row['msat_burstiness'] > CLASSIFIER_THRESHOLDS['tau_burst'])
    return row


def load_columns(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read the per-wheel arrays this diagnostic needs from one Arrow file.
    Returns (t, W, omega, theta)."""
    df = feather.read_feather(path, columns=REQUIRED_COLS)
    t = df['time'].to_numpy(np.float64)
    W = {s: df[[f'{s}_{i}' for i in range(1, 5)]].to_numpy(np.float64)
         for s in ('Fpar', 'Fperp', 'util', 'Msw', 'Meq', 'Vpx', 'Vpy', 'wz', 'Mz', 'Msat')}
    omega = df[[f'w{i}' for i in range(1, 5)]].to_numpy(np.float64)
    theta = df[[f'theta{i}' for i in range(1, 5)]].to_numpy(np.float64)
    return t, W, omega, theta


def diagnose_file(path: Path, cfg: ChatterConfig) -> Dict[str, Any]:
    name = Path(path).name
    parsed = parse_arrow_filename(name)
    row: Dict[str, Any] = {'file': name}
    if parsed is None:
        row['error'] = 'filename did not match profile scheme'
        return row
    row.update({k: v for k, v in parsed.items() if k != 'is_posref'})
    try:
        t, W, omega, theta = load_columns(path)
    except Exception as e:                                          # missing cols / IO
        row['error'] = f'read failed: {e}'
        return row
    row.update(diagnose_columns(t, W, omega, theta, parsed['mu'], parsed['chi'], cfg))
    return row


def classify(row: Dict[str, Any], th: Dict[str, float]) -> Tuple[str, bool]:
    """Return (verdict, chatter_flag). hash => hard reject; chatter => kept+flagged."""
    if row.get('error'):
        return 'ERROR', False
    m2 = row['hash_fraction']
    m3 = row['control_chatter_index']
    m4 = row['msw_force_coherence_hf']
    m5 = row['hf_slip_modulation']

    def lt(x, c):  # NaN-safe "x < c"
        return np.isfinite(x) and x < c

    def gt(x, c):
        return np.isfinite(x) and x > c

    # hash: HF energy unexplained by Msw AND not slip-modulated LuGre
    if gt(m2, th['tau_hash']) and lt(m4, th['tau_coh_low']) and lt(m5, th['tau_slipmod']):
        return 'hash', False
    # chatter: switching wrench is chattering AND it's coherent with the force
    if gt(m3, th['tau_ctrl']) and gt(m4, th['tau_coh_high']):
        return 'chatter', True
    # marginal: above-ceiling energy that is NOT explained by slip-modulated
    # LuGre (M5) — a mild, unattributed hash signal worth a human look. If M5
    # says the energy is slip-modulated stick-slip, it is exonerated -> clean.
    slip_explained = gt(m5, th['tau_slipmod'])
    m2_suspicious = gt(m2, 0.5 * th['tau_hash']) and not slip_explained
    if m2_suspicious:
        return 'marginal', False
    return 'clean', False


# =============================================================================
# 8. Batch runner (streaming + resume) and CLI
# =============================================================================
def _worker(path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    return diagnose_file(Path(path_str), ChatterConfig(**cfg_dict))


def run_batch(data_dir: Path, out_csv: Path, cfg: ChatterConfig,
              profiles: Optional[set] = None, limit: Optional[int] = None,
              jobs: int = 1, resume: bool = True, flush_every: int = 200,
              mu: Optional[float] = None) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob('*.arrow'))
    paths = [p for p in paths if parse_arrow_filename(p.name) is not None]
    if profiles:
        paths = [p for p in paths if parse_arrow_filename(p.name)['profile'] in profiles]
    if mu is not None:   # restrict to a single friction coefficient (e.g. a new mu batch)
        paths = [p for p in paths if abs(parse_arrow_filename(p.name)['mu'] - mu) < 1e-9]
    if limit:
        paths = paths[:limit]

    # Read prior rows ONCE (not re-read on each flush, which would double-count).
    base_df = None
    done: set = set()
    if resume and out_csv.exists():
        base_df = pd.read_csv(out_csv)
        done = set(base_df['file'].tolist())
        print(f'[resume] {len(done)} rows already in {out_csv}')
    todo = [p for p in paths if p.name not in done]
    print(f'[batch] {len(todo)} of {len(paths)} files to process  (jobs={jobs})')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    def flush() -> pd.DataFrame:
        df = (pd.concat([base_df, pd.DataFrame(rows)], ignore_index=True)
              if base_df is not None else pd.DataFrame(rows))
        df.to_csv(out_csv, index=False)        # crash-safe checkpoint (Modern Standby)
        return df

    cfg_dict = asdict(cfg)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_worker, str(p), cfg_dict): p for p in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if k % flush_every == 0:
                    flush()
                if k % 50 == 0 or k == len(todo):
                    print(f'  [{k}/{len(todo)}]')
    else:
        for k, p in enumerate(todo, 1):
            rows.append(diagnose_file(p, cfg))
            if k % flush_every == 0:
                flush()
            if k % 50 == 0 or k == len(todo):
                print(f'  [{k}/{len(todo)}] {p.name}')

    df = flush()
    if 'verdict' in df.columns:
        print('\n[verdict counts]\n' + df['verdict'].value_counts().to_string())
    print(f'\n[done] wrote {len(df)} rows -> {out_csv}')
    return df


def write_whitelist(df: pd.DataFrame, path: Path, include_marginal: bool = False) -> int:
    keep = df[df['verdict'].isin(['clean', 'chatter'])]
    if include_marginal:
        keep = pd.concat([keep, df[df['verdict'] == 'marginal']])
    names = sorted(keep['file'].tolist())
    path.write_text('\n'.join(names))
    print(f'[whitelist] {len(names)} files -> {path}  (hash excluded; chatter kept-flagged)')
    return len(names)


def _cli():
    ap = argparse.ArgumentParser(description='Profiles-era trajectory chatter screen.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('chatter_report.csv'))
    ap.add_argument('--whitelist', type=Path, default=None)
    ap.add_argument('--profiles', type=str, default=None, help='comma list to restrict')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--mu', type=float, default=None, help='restrict to a single mu value')
    ap.add_argument('--decimate', type=int, default=8)
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--include-marginal', action='store_true')
    args = ap.parse_args()

    cfg = ChatterConfig(decimate_factor=args.decimate)
    profiles = set(args.profiles.split(',')) if args.profiles else None
    df = run_batch(args.data_dir, args.out, cfg, profiles=profiles,
                   limit=args.limit, jobs=args.jobs, resume=not args.no_resume,
                   mu=args.mu)
    if args.whitelist:
        write_whitelist(df, args.whitelist, include_marginal=args.include_marginal)


if __name__ == '__main__':
    _cli()
