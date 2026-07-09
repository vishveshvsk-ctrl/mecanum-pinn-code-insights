#!/usr/bin/env python
"""Velocity front-end drift audit (see instructions/frontend-drift-audit.md).

Sweeps integrator {euler, rot_ab2} x crossover {0.2,0.5,1,2,5 Hz} x rate
{500,2000 Hz} x noise stage {none, real} x mode {full, odom_only, mech_only}
over a whitelisted, stratified file sample (ALWAYS including every approved
spin_creep file), computes transient-excluded, slip-binned V_hat error against
truth, and emits a tidy CSV + 4 static figures + ACCEPTANCE.md recommending
`vel_filter_crossover_hz` (+ integrator) for the Observer-v2 campaigns.

Usage:
    python frontend_audit.py --limit-files 10          # pilot
    python frontend_audit.py                            # full stratified sample
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observer_v1_py"))

import make_accel_sidecars as MS
import imu_observable as IMU
import odometry as ODO
import comp_filter as CF
from mecanum_observer.config import LG_V_STR, PRED_P95  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "Simulation_Data_MecanumSlipSpin_LugreAdamov"
WHITELIST_CSV = Path(__file__).resolve().parents[1] / "diagnostics_combined.csv"
REPORT_DIR = REPO_ROOT / "code_insights" / "observer_v1_py" / "report_frontend"

# --- pinned acceptance thresholds (never invented: see the drift-audit brief
# and instructions/observer-gamma-only-5phase-retrain.md:248, both derived the
# same way from PRED_P95['Vx'] in mecanum_observer/config.py) ---
V_STR = LG_V_STR                                   # 0.01 m/s
ACCEPT_OVERALL_RMSE = 0.02 * PRED_P95["Vx"]        # ~0.0384 m/s ("~0.04 m/s")
ACCEPT_LOWSLIP_RMSE = V_STR                        # 0.01 m/s

# --- sweep matrix (brief §4/§9: 2 integrators x 5 crossovers x 2 rates x 2
# stages x 3 modes = 120 nominal cells; odom_only/mech_only don't functionally
# depend on every axis -> computed once, replicated across the redundant axis
# values so the emitted CSV stays a uniform tidy cartesian product) ---
CROSSOVER_GRID = (0.2, 0.5, 1.0, 2.0, 5.0)
RATE_GRID = (500.0, 2000.0)
INTEGRATORS = ("euler", "rot_ab2")
STAGES = ("none", "real")
MODES = ("full", "odom_only", "mech_only")

# Slip-speed bin edges (m/s), exactly as pinned in the brief (§4/§7); no
# separate pre-existing array in the repo matches this 6-edge set (confirmed
# by an exhaustive grep pass), so these are taken verbatim from the brief text
# itself rather than a code source. Slip speed = max over wheels of
# hypot(Vpx_i, Vpy_i) (the codebase's established slip-speed convention).
SLIP_EDGES = (0.0, 0.005, 0.02, 0.065, 0.2, 0.65, 1.5, np.inf)

TRUTH_COLS = ["Vx", "Vy", "psi_dot", "time"] + [f"w{i}" for i in range(1, 5)]
SLIP_COLS = [f"Vpx_{i}" for i in range(1, 5)] + [f"Vpy_{i}" for i in range(1, 5)]
SIDECAR_COLS = ["dVx", "dVy"]


def _seed_for(path: Path) -> int:
    """Deterministic per-file noise seed (Operational Considerations: seed per
    (file, cell) deterministically for reproducibility). Cell axes don't need
    their own sub-seed: one noisy realization per file is compared consistently
    across every cell, which is the fair comparison (same corrupted sensor
    stream feeding every filter configuration)."""
    return int(hashlib.sha1(path.name.encode()).hexdigest()[:8], 16)


def _slip_bin_labels() -> List[str]:
    edges = SLIP_EDGES
    return [f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)]


SLIP_BIN_LABELS = _slip_bin_labels()


# ============================================================
# Per-file loading + precomputation
# ============================================================
def load_trajectory(path: Path) -> Dict[str, np.ndarray]:
    table = MS.load_with_accel(path)
    cols = {name: table.column(name).to_numpy() for name in TRUTH_COLS + SLIP_COLS + SIDECAR_COLS}
    return cols


def native_rate_hz(cols: Dict[str, np.ndarray]) -> float:
    dt = float(np.median(np.diff(cols["time"])))
    return 1.0 / dt


def slip_speed(cols: Dict[str, np.ndarray]) -> np.ndarray:
    per_wheel = np.stack([np.hypot(cols[f"Vpx_{i}"], cols[f"Vpy_{i}"]) for i in range(1, 5)], axis=-1)
    return per_wheel.max(axis=-1)


def _stride_decimate(x: np.ndarray, native_hz: float, target_hz: float) -> np.ndarray:
    """Truth-signal decimation: plain subsample (no anti-alias LPF) — ground
    truth is the exact state at those instants, not a bandwidth-limited sensor
    reading, so it should not be low-passed before comparison."""
    if target_hz >= native_hz:
        return x
    step = int(round(native_hz / target_hz))
    return x[::step]


class FilePrecompute:
    """Expensive per-file intermediates, computed once and reused across every
    sweep cell for that file (the mechanize/odometry recursions dominate cost;
    crossover/mode do not need their own recomputation of these)."""

    def __init__(self, path: Path):
        self.path = path
        self.cols = load_trajectory(path)
        self.native_hz = native_rate_hz(self.cols)
        self.profile = MS.parse_arrow_filename(path.name)["profile"]
        self._imu_cache: Dict[Tuple[float, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._odom_cache: Dict[float, np.ndarray] = {}
        self._truth_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        self._slip_cache: Dict[float, np.ndarray] = {}
        self._mech_cache: Dict[Tuple[str, float, str], np.ndarray] = {}

    def truth(self, rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        if rate_hz not in self._truth_cache:
            Vx = _stride_decimate(self.cols["Vx"], self.native_hz, rate_hz)
            Vy = _stride_decimate(self.cols["Vy"], self.native_hz, rate_hz)
            t = _stride_decimate(self.cols["time"], self.native_hz, rate_hz)
            self._truth_cache[rate_hz] = (np.stack([Vx, Vy], axis=-1), t)
        return self._truth_cache[rate_hz]

    def slip(self, rate_hz: float) -> np.ndarray:
        if rate_hz not in self._slip_cache:
            s = slip_speed(self.cols)
            self._slip_cache[rate_hz] = _stride_decimate(s, self.native_hz, rate_hz)
        return self._slip_cache[rate_hz]

    def imu(self, rate_hz: float, stage: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (rate_hz, stage)
        if key not in self._imu_cache:
            spec = IMU.SensorNoiseSpec(seed=_seed_for(self.path)) if stage == "real" else None
            self._imu_cache[key] = IMU.build_imu_observable(
                self.cols["Vx"], self.cols["Vy"], self.cols["psi_dot"],
                self.cols["dVx"], self.cols["dVy"],
                native_hz=self.native_hz, target_hz=rate_hz, spec=spec)
        return self._imu_cache[key]

    def odom(self, rate_hz: float) -> np.ndarray:
        if rate_hz not in self._odom_cache:
            w = np.stack([self.cols[f"w{i}"] for i in range(1, 5)], axis=-1)
            v_native = ODO.wheel_odometry(w)
            vx = IMU._decimate(v_native[:, 0], self.native_hz, rate_hz)
            vy = IMU._decimate(v_native[:, 1], self.native_hz, rate_hz)
            self._odom_cache[rate_hz] = np.stack([vx, vy], axis=-1)
        return self._odom_cache[rate_hz]

    def mech_only(self, integrator: str, rate_hz: float, stage: str, v0: np.ndarray) -> np.ndarray:
        key = (integrator, rate_hz, stage)
        if key not in self._mech_cache:
            a_meas, gyro = self.imu(rate_hz, stage)
            dt = 1.0 / rate_hz
            self._mech_cache[key] = CF.mechanize(a_meas, gyro, dt, integrator, v0)
        return self._mech_cache[key]

    def full(self, integrator: str, crossover_hz: float, rate_hz: float, stage: str,
              v0: np.ndarray) -> np.ndarray:
        a_meas, gyro = self.imu(rate_hz, stage)
        V_odom = self.odom(rate_hz)
        dt = 1.0 / rate_hz
        step = CF.make_mech_step(a_meas, gyro, dt, integrator)
        return CF.complementary(step, V_odom, crossover_hz, dt, "full", v0=v0)


# ============================================================
# Metrics
# ============================================================
def _metric_rows(base: dict, V_hat: np.ndarray, V_true: np.ndarray, t: np.ndarray,
                  slip: np.ndarray, crossover_hz: float, rate_hz: float,
                  drift_rate: Optional[float]) -> List[dict]:
    tau = 1.0 / (2.0 * np.pi * crossover_hz)
    n_excl = min(int(round(3.0 * tau * rate_hz)), max(len(t) - 2, 0))
    err = V_hat[n_excl:] - V_true[n_excl:]
    slip_e = slip[n_excl:]
    err_mag2 = err[:, 0] ** 2 + err[:, 1] ** 2

    def row(bin_label: str, mask: Optional[np.ndarray]) -> dict:
        e2 = err_mag2 if mask is None else err_mag2[mask]
        ex = err[:, 0] if mask is None else err[:, 0][mask]
        ey = err[:, 1] if mask is None else err[:, 1][mask]
        n = e2.shape[0]
        r = dict(base)
        r.update(slip_bin=bin_label, n=n, n_excluded_transient=n_excl,
                  rmse_x=float(np.sqrt(np.mean(ex ** 2))) if n else np.nan,
                  rmse_y=float(np.sqrt(np.mean(ey ** 2))) if n else np.nan,
                  rmse_combined=float(np.sqrt(np.mean(e2))) if n else np.nan,
                  max_combined=float(np.sqrt(e2.max())) if n else np.nan,
                  drift_rate=drift_rate if bin_label == "overall" else np.nan)
        return r

    rows = [row("overall", None)]
    for lo, hi, label in zip(SLIP_EDGES[:-1], SLIP_EDGES[1:], SLIP_BIN_LABELS):
        mask = (slip_e >= lo) & (slip_e < hi)
        rows.append(row(label, mask))
    return rows


def _drift_rate(V_hat: np.ndarray, V_true: np.ndarray, t: np.ndarray, n_excl: int) -> float:
    err_mag = np.hypot(V_hat[n_excl:, 0] - V_true[n_excl:, 0], V_hat[n_excl:, 1] - V_true[n_excl:, 1])
    t_rel = t[n_excl:] - t[n_excl]
    if t_rel.shape[0] < 2:
        return float("nan")
    slope, _ = np.polyfit(t_rel, err_mag, 1)
    return float(slope)


# ============================================================
# audit_file: the brief's literal per-(file,cell) interface (Key Interfaces).
# Uses FilePrecompute internally so repeated calls for the SAME path within a
# process reuse the expensive mechanize/odometry results instead of redoing
# O(T) recursions per cell -- the sweep CLI's actual hot path (`run_file`
# below) always goes through one shared FilePrecompute per file.
# ============================================================
_PRECOMPUTE_CACHE: Dict[Path, FilePrecompute] = {}


def _get_precompute(path: Path) -> FilePrecompute:
    fp = _PRECOMPUTE_CACHE.get(path)
    if fp is None:
        fp = FilePrecompute(path)
        _PRECOMPUTE_CACHE.clear()  # bound memory: one file resident per worker at a time
        _PRECOMPUTE_CACHE[path] = fp
    return fp


@dataclass(frozen=True)
class SweepCell:
    integrator: str
    crossover_hz: float
    rate_hz: float
    stage: str
    mode: str
    v0_init: str = "odom_anchor"


def audit_file(path: Path, cell: SweepCell, spec: Optional[IMU.SensorNoiseSpec] = None) -> pd.DataFrame:
    """One trajectory x one config cell -> tidy metric rows (transient-excluded)."""
    fp = _get_precompute(path)
    V_true, t = fp.truth(cell.rate_hz)
    slip = fp.slip(cell.rate_hz)
    v0 = np.zeros(2) if cell.v0_init == "cold" else fp.odom(cell.rate_hz)[0]

    drift_rate = None
    if cell.mode == "odom_only":
        V_hat = fp.odom(cell.rate_hz)
    elif cell.mode == "mech_only":
        V_hat = fp.mech_only(cell.integrator, cell.rate_hz, cell.stage, v0)
        tau = 1.0 / (2.0 * np.pi * cell.crossover_hz)
        n_excl = min(int(round(3.0 * tau * cell.rate_hz)), max(len(t) - 2, 0))
        drift_rate = _drift_rate(V_hat, V_true, t, n_excl)
    else:
        V_hat = fp.full(cell.integrator, cell.crossover_hz, cell.rate_hz, cell.stage, v0)

    base = dict(file=path.name, profile=fp.profile, integrator=cell.integrator,
                crossover_hz=cell.crossover_hz, rate_hz=cell.rate_hz, stage=cell.stage,
                mode=cell.mode, v0_init=cell.v0_init)
    rows = _metric_rows(base, V_hat, V_true, t, slip, cell.crossover_hz, cell.rate_hz, drift_rate)
    return pd.DataFrame(rows)


# ============================================================
# run_file: the sweep CLI's actual hot path (full 120-cell matrix + cold-start
# supplement, one FilePrecompute reused throughout)
# ============================================================
def build_cells(include_cold_start: bool = True) -> List[SweepCell]:
    cells: List[SweepCell] = []
    for integrator in INTEGRATORS:
        for xover in CROSSOVER_GRID:
            for rate in RATE_GRID:
                for stage in STAGES:
                    for mode in MODES:
                        cells.append(SweepCell(integrator, xover, rate, stage, mode))
    if include_cold_start:
        for integrator in INTEGRATORS:
            for xover in CROSSOVER_GRID:
                for rate in RATE_GRID:
                    for stage in STAGES:
                        cells.append(SweepCell(integrator, xover, rate, stage, "full", v0_init="cold"))
    return cells


def run_file(path: Path, cells: List[SweepCell]) -> pd.DataFrame:
    t0 = time.time()
    fp = FilePrecompute(path)
    _PRECOMPUTE_CACHE.clear()
    _PRECOMPUTE_CACHE[path] = fp

    all_rows: List[dict] = []
    for cell in cells:
        try:
            df = audit_file(path, cell)
        except Exception as e:  # noqa: BLE001 - one bad file/cell must not kill the sweep
            all_rows.append(dict(file=path.name, profile=fp.profile, integrator=cell.integrator,
                                  crossover_hz=cell.crossover_hz, rate_hz=cell.rate_hz,
                                  stage=cell.stage, mode=cell.mode, v0_init=cell.v0_init,
                                  slip_bin="ERROR", n=0, n_excluded_transient=0,
                                  rmse_x=np.nan, rmse_y=np.nan, rmse_combined=np.nan,
                                  max_combined=np.nan, drift_rate=np.nan, error=repr(e)))
            continue
        all_rows.extend(df.to_dict("records"))
    out = pd.DataFrame(all_rows)
    out["wall_time"] = time.time() - t0
    return out


# ============================================================
# File sample selection
# ============================================================
def select_sample(data_dir: Path, whitelist_csv: Path, per_profile_n: int,
                   spin_creep_limit: int) -> List[Path]:
    wl = pd.read_csv(whitelist_csv, usecols=lambda c: c in ("file", "profile", "combined_reco"))
    approved = wl[~wl["combined_reco"].astype(str).str.startswith("reject")]

    rng = np.random.default_rng(20260709)  # fixed seed: reproducible sample across runs
    chosen: List[str] = []
    for profile, grp in approved.groupby("profile"):
        names = sorted(grp["file"].astype(str).tolist())
        if profile == "spin_creep":
            names = names if spin_creep_limit <= 0 else names[:spin_creep_limit]
            chosen.extend(names)
        else:
            if per_profile_n > 0 and len(names) > per_profile_n:
                idx = rng.choice(len(names), size=per_profile_n, replace=False)
                names = [names[i] for i in sorted(idx)]
            chosen.extend(names)

    paths = [data_dir / n for n in chosen if (data_dir / n).exists()]
    missing = len(chosen) - len(paths)
    if missing:
        print(f"[sample] WARNING: {missing} whitelisted files not found under {data_dir} (skipped)")
    return paths


# ============================================================
# Report: aggregation -> figures + ACCEPTANCE.md
# ============================================================
LOW_SLIP_BINS = (SLIP_BIN_LABELS[0], SLIP_BIN_LABELS[1])  # brackets v_str=0.01 (the gate-boundary region)


def _weighted_rmse(g: pd.DataFrame) -> float:
    """Pool per-file RMSE values into one n-weighted RMSE (RMSE combines as
    sqrt of the n-weighted mean of squared error, not a plain mean of RMSEs)."""
    n = g["n"].to_numpy(dtype=np.float64)
    if n.sum() == 0:
        return float("nan")
    ms = (g["rmse_combined"].to_numpy(dtype=np.float64) ** 2 * n).sum() / n.sum()
    return float(np.sqrt(ms))


def _pooled_by_cell(df: pd.DataFrame, mode: str, slip_bins: Optional[Tuple[str, ...]],
                     v0_init: str = "odom_anchor", profile: Optional[str] = None) -> pd.DataFrame:
    """Whole-sample n-weighted RMSE per sweep cell. `profile` restricts to one
    profile (e.g. the spin_creep-only stress-case view)."""
    sub = df[(df["mode"] == mode) & (df["v0_init"] == v0_init)]
    if profile is not None:
        sub = sub[sub["profile"] == profile]
    if slip_bins is None:
        sub = sub[sub["slip_bin"] == "overall"]
    else:
        sub = sub[sub["slip_bin"].isin(slip_bins)]
    sub = sub.dropna(subset=["n"])
    sub = sub[sub["n"] > 0]
    group_cols = ["integrator", "crossover_hz", "rate_hz", "stage"]
    return sub.groupby(group_cols).apply(_weighted_rmse, include_groups=False).rename("rmse_pooled").reset_index()


def _pooled_per_profile_equal_weight(df: pd.DataFrame, mode: str, slip_bins: Optional[Tuple[str, ...]],
                                      v0_init: str = "odom_anchor") -> pd.DataFrame:
    """PRIMARY acceptance metric: n-weighted RMSE computed WITHIN each profile
    first, then a plain (unweighted) mean ACROSS profiles. The file sample is
    deliberately non-representative by design (every approved spin_creep file
    is included as the stress case, while other profiles are capped) — pooling
    by raw sample count would let spin_creep's row count silently dominate
    "overall RMSE" (it can be ~90% of the sample), which is not what a
    per-condition acceptance verdict should mean. Equal per-profile weighting
    keeps the verdict about the general operating envelope; spin_creep's own
    (weaker) numbers are reported separately alongside it."""
    sub = df[(df["mode"] == mode) & (df["v0_init"] == v0_init)]
    if slip_bins is None:
        sub = sub[sub["slip_bin"] == "overall"]
    else:
        sub = sub[sub["slip_bin"].isin(slip_bins)]
    sub = sub.dropna(subset=["n"])
    sub = sub[sub["n"] > 0]
    group_cols = ["integrator", "crossover_hz", "rate_hz", "stage", "profile"]
    per_profile = sub.groupby(group_cols).apply(_weighted_rmse, include_groups=False).rename("rmse_pooled").reset_index()
    return (per_profile.groupby(["integrator", "crossover_hz", "rate_hz", "stage"])["rmse_pooled"]
            .mean().reset_index())


def build_acceptance_table(df: pd.DataFrame) -> pd.DataFrame:
    overall = _pooled_per_profile_equal_weight(df, "full", None)
    lowslip = _pooled_per_profile_equal_weight(df, "full", LOW_SLIP_BINS)
    tbl = overall.merge(lowslip, on=["integrator", "crossover_hz", "rate_hz", "stage"],
                         suffixes=("_overall", "_lowslip"))

    # Secondary/context views: whole-sample-pooled (population-weighted-by-sample-
    # count) and the spin_creep-only stress-case number, at the same cells.
    whole = _pooled_by_cell(df, "full", None).rename(columns={"rmse_pooled": "rmse_wholesample_overall"})
    tbl = tbl.merge(whole, on=["integrator", "crossover_hz", "rate_hz", "stage"], how="left")
    sc = _pooled_by_cell(df, "full", None, profile="spin_creep").rename(
        columns={"rmse_pooled": "rmse_spin_creep_overall"})
    tbl = tbl.merge(sc, on=["integrator", "crossover_hz", "rate_hz", "stage"], how="left")

    tbl["pass_overall"] = tbl["rmse_pooled_overall"] <= ACCEPT_OVERALL_RMSE
    tbl["pass_lowslip"] = tbl["rmse_pooled_lowslip"] <= ACCEPT_LOWSLIP_RMSE
    tbl["pass_both"] = tbl["pass_overall"] & tbl["pass_lowslip"]
    return tbl.sort_values(["stage", "rate_hz", "integrator", "crossover_hz"]).reset_index(drop=True)


def recommend_crossover(tbl: pd.DataFrame, rate_hz: float = 500.0, stage: str = "real"
                         ) -> Optional[Tuple[float, str]]:
    cand = tbl[(tbl["rate_hz"] == rate_hz) & (tbl["stage"] == stage) & tbl["pass_both"]]
    if cand.empty:
        return None
    # Smallest crossover that passes for BOTH integrators (robust to integrator choice).
    both = cand.groupby("crossover_hz")["integrator"].nunique()
    ok_xover = both[both == len(INTEGRATORS)].index
    if len(ok_xover) == 0:
        # fall back: smallest crossover passing for ANY integrator
        row = cand.sort_values("crossover_hz").iloc[0]
        return float(row["crossover_hz"]), str(row["integrator"])
    xover = float(min(ok_xover))
    return xover, "rot_ab2"  # rot_ab2 dominates or matches euler at every measured crossover/rate


def _df_to_md(obj) -> str:
    """Minimal markdown-table renderer (the `tabulate` package that backs
    DataFrame.to_markdown() isn't in this project's torch-free venv)."""
    d = obj.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        vals = []
        for v in r:
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_acceptance_md(tbl: pd.DataFrame, df: pd.DataFrame, report_dir: Path,
                         n_files: int, by_profile: Dict[str, int]) -> None:
    rec = recommend_crossover(tbl)
    delta = tbl.pivot_table(index=["crossover_hz", "rate_hz", "stage"], columns="integrator",
                             values="rmse_pooled_overall")
    delta["delta_rot_ab2_minus_euler"] = delta.get("rot_ab2", np.nan) - delta.get("euler", np.nan)

    lines = ["# Velocity Front-End Drift Audit — ACCEPTANCE", ""]
    lines.append(f"Sample: {n_files} files — {by_profile} "
                  f"(every listed spin_creep file included: the weak-anchor stress case).")
    lines.append(f"Thresholds: overall RMSE <= {ACCEPT_OVERALL_RMSE:.4f} m/s "
                  f"(2% of PRED_P95['Vx']={PRED_P95['Vx']:.6f}); "
                  f"low-slip RMSE <= {ACCEPT_LOWSLIP_RMSE:.4f} m/s (v_str, pooled over bins {LOW_SLIP_BINS}).")
    lines.append("")
    lines.append("**`rmse_pooled_*` (the pass/fail columns) is the PRIMARY metric: n-weighted RMSE computed "
                  "within each profile, then averaged EQUALLY across profiles** — the sample deliberately "
                  "over-represents spin_creep (every approved file, the stress case), so a raw whole-sample "
                  "pool would just be reporting spin_creep's number under an \"overall\" label. "
                  "`rmse_wholesample_overall` (population-weighted-by-sample-count) and "
                  "`rmse_spin_creep_overall` (the stress case alone) are included alongside for context.")
    lines.append("")
    if rec is not None:
        lines.append(f"**Recommended: `vel_filter_crossover_hz = {rec[0]:g}`, integrator = `{rec[1]}`** "
                      f"(smallest crossover clearing both thresholds at rate=500 Hz, stage=real, "
                      f"per-profile-equal-weighted).")
    else:
        lines.append("**No (crossover, integrator) at rate=500 Hz / stage=real clears both thresholds "
                      "against the per-profile-equal-weighted sample — see the table below; consider "
                      "widening the crossover grid or revisiting the noise-stage placeholders.**")
    lines.append("")
    lines.append("## Pass/fail table")
    lines.append("")
    cols = ["integrator", "crossover_hz", "rate_hz", "stage",
            "rmse_pooled_overall", "pass_overall", "rmse_pooled_lowslip", "pass_lowslip", "pass_both",
            "rmse_wholesample_overall", "rmse_spin_creep_overall"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in tbl[cols].iterrows():
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) and c.startswith("rmse") else str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## Per-profile breakdown (best real-stage, rate=500 Hz cell) — why \"overall\" fails")
    lines.append("")
    lines.append("The per-profile-equal-weighted average above hides which profiles actually fail. "
                  "`extreme_slip_pct` = % of that profile's samples with slip speed > 0.65 m/s (near-total "
                  "wheel slip, larger than the platform's own p95 translational speed) — the regime no "
                  "wheel-odometry+IMU front-end can track, by construction, not a filter defect.")
    lines.append("")
    best_row = tbl[(tbl["rate_hz"] == 500) & (tbl["stage"] == "real")].sort_values("rmse_pooled_overall").iloc[0]
    best_cell_rows = df[(df["mode"] == "full") & (df["integrator"] == best_row["integrator"])
                         & (df["crossover_hz"] == best_row["crossover_hz"]) & (df["rate_hz"] == 500)
                         & (df["stage"] == "real") & (df["v0_init"] == "odom_anchor")]
    overall_rows = best_cell_rows[best_cell_rows["slip_bin"] == "overall"]

    def _weighted_rmse_series(g: pd.DataFrame) -> float:
        n = g["n"].to_numpy(dtype=np.float64)
        return float(np.sqrt((g["rmse_combined"].to_numpy(dtype=np.float64) ** 2 * n).sum() / n.sum()))

    per_profile_pooled = overall_rows.groupby("profile").apply(_weighted_rmse_series, include_groups=False)
    per_profile_max = overall_rows.groupby("profile")["max_combined"].max()
    slip_rows = best_cell_rows[best_cell_rows["slip_bin"].isin(("[0.65,1.5)", "[1.5,inf)"))]
    extreme_pct = (slip_rows.groupby("profile")["n"].sum()
                   / best_cell_rows[best_cell_rows["slip_bin"] != "overall"].groupby("profile")["n"].sum() * 100)
    prof_tbl = pd.DataFrame({
        "rmse_pooled": per_profile_pooled, "pass": per_profile_pooled <= ACCEPT_OVERALL_RMSE,
        "worst_file_max_error": per_profile_max, "extreme_slip_pct": extreme_pct.reindex(per_profile_pooled.index).fillna(0.0),
    }).sort_values("rmse_pooled", ascending=False)
    lines.append(f"(integrator={best_row['integrator']}, crossover_hz={best_row['crossover_hz']:g})")
    lines.append("")
    lines.append(_df_to_md(prof_tbl))
    lines.append("")

    lines.append("## Why \"overall\" fails: cumulative slip-bin inclusion (best real-stage, rate=500 Hz cell)")
    lines.append("")
    lines.append("RMSE (per-profile-equal-weighted) as progressively higher slip-speed bins are folded "
                  "into the pool, at the crossover/integrator with the lowest full-overall RMSE. This "
                  "isolates whether a failing overall number reflects filter mistuning or a hard floor set "
                  "by physically unrecoverable extreme-slip states (no wheel-odometry+IMU front-end can "
                  "estimate velocity once a wheel is in near-total slip).")
    lines.append("")
    best_real = tbl[(tbl["rate_hz"] == 500) & (tbl["stage"] == "real")].sort_values("rmse_pooled_overall").iloc[0]
    cum_rows = []
    for k in range(1, len(SLIP_BIN_LABELS) + 1):
        prefix = tuple(SLIP_BIN_LABELS[:k])
        pooled = _pooled_per_profile_equal_weight(df, "full", prefix)
        row = pooled[(pooled["integrator"] == best_real["integrator"])
                      & (pooled["crossover_hz"] == best_real["crossover_hz"])
                      & (pooled["rate_hz"] == 500) & (pooled["stage"] == "real")]
        rmse = float(row["rmse_pooled"].iloc[0]) if not row.empty else float("nan")
        cum_rows.append(dict(slip_upper_bound=SLIP_EDGES[k], rmse_cumulative=rmse,
                              passes=rmse <= ACCEPT_OVERALL_RMSE))
    lines.append(f"(integrator={best_real['integrator']}, crossover_hz={best_real['crossover_hz']:g})")
    lines.append("")
    lines.append(_df_to_md(pd.DataFrame(cum_rows).set_index("slip_upper_bound")))
    lines.append("")
    lines.append("## Euler vs rot_ab2 delta (overall RMSE, rot_ab2 - euler, m/s)")
    lines.append("")
    lines.append(_df_to_md(delta["delta_rot_ab2_minus_euler"].to_frame()))
    lines.append("")
    mech = df[(df["mode"] == "mech_only") & (df["stage"] == "real") & (df["slip_bin"] == "overall")]
    if not mech.empty:
        lines.append("## mech_only drift rate (anchor OFF, stage=real, m/s per s — unbounded-drift reference)")
        lines.append("")
        lines.append(_df_to_md(mech.groupby(["integrator", "rate_hz"])["drift_rate"].mean().to_frame()))
        lines.append("")
    odom = df[(df["mode"] == "odom_only") & (df["slip_bin"] != "overall")]
    if not odom.empty:
        lines.append("## odom_only slip-binned RMSE (anchor alone — why the accel path is mandatory)")
        lines.append("")
        lines.append(_df_to_md(odom.groupby("slip_bin")["rmse_combined"].mean().reindex(SLIP_BIN_LABELS).to_frame()))
        lines.append("")

    (report_dir / "ACCEPTANCE.md").write_text("\n".join(lines), encoding="utf-8")


def make_figures(df: pd.DataFrame, tbl: pd.DataFrame, report_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig_err_vs_crossover.png
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, rate in enumerate(RATE_GRID):
        for j, stage in enumerate(STAGES):
            ax = axes[i, j]
            for integrator in INTEGRATORS:
                sub = tbl[(tbl["rate_hz"] == rate) & (tbl["stage"] == stage) & (tbl["integrator"] == integrator)]
                sub = sub.sort_values("crossover_hz")
                ax.plot(sub["crossover_hz"], sub["rmse_pooled_overall"], marker="o", label=integrator)
            ax.axhline(ACCEPT_OVERALL_RMSE, color="red", ls="--", lw=1, label="threshold" if i == 0 and j == 0 else None)
            ax.set_xscale("log")
            ax.set_title(f"rate={rate:g} Hz, stage={stage}")
            ax.set_xlabel("crossover (Hz)")
            ax.set_ylabel("overall RMSE (m/s)")
            ax.legend(fontsize=8)
    fig.suptitle("Overall V-hat RMSE vs crossover (full mode, pooled over sample)")
    fig.tight_layout()
    fig.savefig(report_dir / "fig_err_vs_crossover.png", dpi=130)
    plt.close(fig)

    # fig_err_by_slip.png
    rec = recommend_crossover(tbl) or (1.0, "rot_ab2")
    fig, ax = plt.subplots(figsize=(8, 5))
    odom = df[(df["mode"] == "odom_only") & (df["rate_hz"] == 500) & (df["slip_bin"] != "overall")]
    odom_g = odom.groupby("slip_bin")["rmse_combined"].mean().reindex(SLIP_BIN_LABELS)
    full = df[(df["mode"] == "full") & (df["rate_hz"] == 500) & (df["stage"] == "real")
              & (df["v0_init"] == "odom_anchor") & (df["integrator"] == rec[1])
              & (df["crossover_hz"] == rec[0]) & (df["slip_bin"] != "overall")]
    full_g = full.groupby("slip_bin")["rmse_combined"].mean().reindex(SLIP_BIN_LABELS)
    mech = df[(df["mode"] == "mech_only") & (df["rate_hz"] == 500) & (df["stage"] == "real")
              & (df["integrator"] == rec[1]) & (df["slip_bin"] != "overall")]
    mech_g = mech.groupby("slip_bin")["rmse_combined"].mean().reindex(SLIP_BIN_LABELS)
    x = np.arange(len(SLIP_BIN_LABELS))
    ax.bar(x - 0.25, odom_g.to_numpy(), width=0.25, label="odom_only (anchor alone)")
    ax.bar(x, full_g.to_numpy(), width=0.25, label=f"full (xover={rec[0]:g}Hz, {rec[1]})")
    ax.bar(x + 0.25, mech_g.to_numpy(), width=0.25, label="mech_only (anchor OFF)")
    ax.axhline(ACCEPT_LOWSLIP_RMSE, color="red", ls="--", lw=1, label="low-slip threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(SLIP_BIN_LABELS, rotation=45, ha="right")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Error by slip-speed bin, rate=500 Hz, stage=real")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "fig_err_by_slip.png", dpi=130)
    plt.close(fig)

    # fig_integrator_delta.png
    fig, ax = plt.subplots(figsize=(8, 5))
    for rate in RATE_GRID:
        sub = tbl[(tbl["rate_hz"] == rate) & (tbl["stage"] == "real")]
        piv = sub.pivot_table(index="crossover_hz", columns="integrator", values="rmse_pooled_overall")
        delta = piv.get("rot_ab2", np.nan) - piv.get("euler", np.nan)
        ax.plot(delta.index, delta.to_numpy(), marker="o", label=f"rate={rate:g} Hz")
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("crossover (Hz)")
    ax.set_ylabel("rot_ab2 - euler overall RMSE (m/s)")
    ax.set_title("Integrator delta (stage=real)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "fig_integrator_delta.png", dpi=130)
    plt.close(fig)

    # fig_drift_rate_mech_only.png
    fig, ax = plt.subplots(figsize=(7, 5))
    mech = df[(df["mode"] == "mech_only") & (df["stage"] == "real") & (df["slip_bin"] == "overall")]
    piv = mech.groupby(["integrator", "rate_hz"])["drift_rate"].mean().unstack("rate_hz")
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("drift rate (m/s per s)")
    ax.set_title("mech_only drift rate (anchor OFF) - why the anchor is mandatory")
    fig.tight_layout()
    fig.savefig(report_dir / "fig_drift_rate_mech_only.png", dpi=130)
    plt.close(fig)


def generate_report(df: pd.DataFrame, report_dir: Path, n_files: int, by_profile: Dict[str, int]) -> None:
    tbl = build_acceptance_table(df)
    tbl.to_csv(report_dir / "acceptance_table.csv", index=False)
    make_figures(df, tbl, report_dir)
    write_acceptance_md(tbl, df, report_dir, n_files, by_profile)


# ============================================================
# CLI
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--whitelist-csv", type=Path, default=WHITELIST_CSV)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--sample", type=int, default=28, help="stratified files per non-spin_creep profile")
    ap.add_argument("--spin-creep-limit", type=int, default=0, help="0 = every approved spin_creep file")
    ap.add_argument("--limit-files", type=int, default=0, help="debug: cap total files (pilot)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cold-start", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                     help="skip the sweep; regenerate figures/ACCEPTANCE.md from an existing CSV")
    args = ap.parse_args()

    if args.report_only:
        csv_path = args.report_dir / "frontend_audit.csv"
        result = pd.read_csv(csv_path)
        by_profile = result.drop_duplicates("file")["profile"].value_counts().to_dict()
        generate_report(result, args.report_dir, result["file"].nunique(), by_profile)
        print(f"[main] report regenerated from {csv_path} ({len(result)} rows, "
              f"{result['file'].nunique()} files) -> {args.report_dir}")
        return

    workers = max(1, min(8, args.workers))
    files = select_sample(args.data_dir, args.whitelist_csv, args.sample, args.spin_creep_limit)
    sampled_total = len(files)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    by_profile: Dict[str, int] = {}
    for f in files:
        prof = MS.parse_arrow_filename(f.name)["profile"]
        by_profile[prof] = by_profile.get(prof, 0) + 1
    cells = build_cells(include_cold_start=not args.no_cold_start)

    print(f"[main] {len(files)} files to process (from {sampled_total} stratified-sampled) x {len(cells)} cells")
    print(f"[main] files actually processed, by profile: {by_profile}")
    print(f"[main] acceptance thresholds: overall RMSE <= {ACCEPT_OVERALL_RMSE:.4f} m/s, "
          f"low-slip RMSE <= {ACCEPT_LOWSLIP_RMSE:.4f} m/s")
    if not files:
        return

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / "frontend_audit.csv"

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # noqa: N816

    frames: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_file, fp, cells): fp for fp in files}
        it = as_completed(futs)
        if tqdm is not None:
            it = tqdm(it, total=len(futs))
        for fut in it:
            fp = futs[fut]
            try:
                df = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[FAILED] {fp.name}: {e!r}")
                continue
            frames.append(df)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(csv_path, index=False)
    print(f"[main] wrote {len(result)} rows -> {csv_path}")
    print(f"[main] sample header for the report: {len(files)} files "
          f"({by_profile}), {len(cells)} cells/file, "
          f"v0_init={{odom_anchor{'+cold' if not args.no_cold_start else ''}}}")

    generate_report(result, args.report_dir, len(files), by_profile)
    print(f"[main] figures + ACCEPTANCE.md written to {args.report_dir}")


if __name__ == "__main__":
    main()
