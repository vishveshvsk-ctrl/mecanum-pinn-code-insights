"""
bound_analysis/run_bound.py — driver: loop trajectories x variants x sensor
grades, run the PCRLB recursion, assemble the summary table, write results.

Inputs:  bound_analysis/traces/*.arrow (export_truth_traces.jl output),
         bound_analysis/reports/ellipse_selection.json
Outputs: bound_analysis/reports/bound_results.npz
           (per-trajectory per-variant per-grade bound sigma series)
         bound_analysis/reports/bound_summary.csv
           (one row per trajectory x variant x sensor grade)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from model import ImuKinematicModel, SENSOR_GRADES, VARIANTS
from pcrlb import pcrlb_recursion, bound_sigmas
from slip_stats import fit_gauss_markov

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SELECTION_PATH = HERE / "reports" / "ellipse_selection.json"
TRACE_DIR = HERE / "traces"
REPORT_DIR = HERE / "reports"
DEFAULT_ESKF_CFG = ROOT / "runs_eskf_noellipse_v2" / "eskf_dxnes" / "best_config.json"


def _wheel_jacobian(l: float, h: float, R: float) -> np.ndarray:
    """Mirrors hybrid_ctrl/estimators.jl EstimatorMod._wheel_jacobian exactly."""
    return np.array([
        [1.0 / R, -1.0 / R, -(l + h) / R],
        [1.0 / R,  1.0 / R,  (l + h) / R],
        [1.0 / R,  1.0 / R, -(l + h) / R],
        [1.0 / R, -1.0 / R,  (l + h) / R],
    ])


def sigma_wheel_vel_for_grade(grade: str, geo: dict) -> float:
    """
    Equivalent body-velocity noise std for the wheel-derived pseudo-velocity
    measurement z_wheel = pinv(Hw) @ omega_noisy, propagated from the per-wheel
    encoder white noise sigma_omega (SENSOR_GRADES[grade]["sigma_omega"],
    read off hybrid_ctrl/sensors.jl SensorMod.SensorModel's sigma_omega).

    Cov(v_est) = sigma_omega^2 * inv(Hw^T Hw) (least-squares noise
    propagation; the Hw^T Hw factors cancel against pinv(Hw)=inv(Hw^T Hw)Hw^T).
    Vx and Vy rows are identical by the O-config's symmetry.

    CAVEAT: this captures only the WHITE encoder-noise component. The
    :realistic grade's scale-factor (sf_omega) and per-wheel constant bias
    terms are NOT folded in here -- the reduced bound model carries a single
    scalar sigma_wheel_vel, not a per-wheel error budget. Flagged as a
    simplification, not an oversight.
    """
    Hw = _wheel_jacobian(geo["l"], geo["h"], geo["R"])
    HtH_inv = np.linalg.inv(Hw.T @ Hw)
    sigma_omega = SENSOR_GRADES[grade]["sigma_omega"]
    return float(np.sqrt(sigma_omega ** 2 * HtH_inv[0, 0]))


def build_J0(model: ImuKinematicModel, P0_scale: float) -> np.ndarray:
    """
    Initial covariance mirroring hybrid_ctrl/estimators.jl init_eskf!'s own
    prior (brief §9: "Set it from the same prior the ESKF uses"): every
    diagonal entry at P0_scale except position (X,Y -> 1.0, a modest, not
    over-confident prior) and, for B3, the slip pair (P0_scale*1e-3, the
    ESKF's own near-frozen-slip weak-observability convention).
    """
    n = model.n
    p0 = np.full(n, P0_scale)
    p0[3] = 1.0   # X
    p0[4] = 1.0   # Y
    if model.variant == "B3":
        p0[8] = P0_scale * 1e-3   # sx
        p0[9] = P0_scale * 1e-3   # sy
    return np.diag(1.0 / p0)


def load_trace(path: Path) -> dict:
    with ipc.open_file(path) as reader:
        df = reader.read_pandas()
    return {c: df[c].to_numpy() for c in df.columns}


def main():
    if not SELECTION_PATH.exists():
        raise SystemExit(f"{SELECTION_PATH} not found — run select_ellipse_combos.jl first.")
    selection = json.loads(SELECTION_PATH.read_text())
    combos = [c for c in selection["combos"] if c["group"] != "unselected"]
    if not combos:
        raise SystemExit("ellipse_selection.json has no selected combos.")

    base_toml_path = ROOT / selection["run_dir"] / "base.toml"
    with open(base_toml_path, "rb") as f:
        base = tomllib.load(f)
    geo = base["platform"]["geometry"]

    with open(DEFAULT_ESKF_CFG) as f:
        eskf_cfg = json.load(f)["config"]
    P0_scale = float(eskf_cfg["P0_scale"])

    sigma_wheel = {g: sigma_wheel_vel_for_grade(g, geo) for g in SENSOR_GRADES}
    print("sigma_wheel_vel by grade:", sigma_wheel)
    print(f"P0_scale (frozen ESKF, {DEFAULT_ESKF_CFG.relative_to(ROOT)}): {P0_scale:.6g}")

    summary_rows = []
    results = {}   # flat dict of arrays for np.savez

    for c in sorted(combos, key=lambda c: int(c["combo_idx"])):
        combo_idx = int(c["combo_idx"])
        trace_path = TRACE_DIR / f"ellipse_c{combo_idx:03d}.arrow"
        if not trace_path.exists():
            print(f"  [skip] combo {combo_idx}: {trace_path.name} not found "
                  f"— run export_truth_traces.jl first")
            continue

        trace = load_trace(trace_path)
        t = trace["t"]
        dt = float(t[1] - t[0])
        T = len(t)

        truth = dict(t=t, psi=trace["psi"], psidot=trace["psidot"],
                     Vx=trace["Vx"], Vy=trace["Vy"], X=trace["X"], Y=trace["Y"],
                     slip=trace["slip"])
        achieved_pos_err = np.hypot(trace["eskf_X"] - trace["X"], trace["eskf_Y"] - trace["Y"])
        achieved_pos_rms_full = float(np.sqrt(np.mean(achieved_pos_err ** 2)))

        # STEADY-STATE window: every variant's bound starts from the SAME
        # artificial J0 prior (sigma_pos(0) = sqrt(2) m, from build_J0's
        # position entries) and converges within ~1 s regardless of variant
        # (confirmed empirically). Squaring that shared transient into a
        # full-trace RMS swamps the real steady-state spread between variants
        # (B0 vs B1 differ by ~6x at steady state, but full-trace RMS reports
        # them as nearly identical) -- so the settle window is excluded from
        # the primary comparison metric. 2 s is a conservative margin over the
        # observed ~1 s settle time; also capped at 10% of T_total so short
        # trajectories still retain a meaningful post-transient window.
        t_settle = min(2.0, 0.10 * float(t[-1]))
        settle_idx = int(np.searchsorted(t, t_settle))
        ss = slice(settle_idx, None)
        achieved_pos_rms_ss = float(np.sqrt(np.mean(achieved_pos_err[ss] ** 2)))

        tau_slip, sigma_slip, slip_fit_r2 = fit_gauss_markov(trace["slip"], dt)
        print(f"combo {combo_idx} ({c['group']}): dt={dt*1000:.1f}ms T={t[-1]:.1f}s "
              f"slip fit tau={tau_slip:.3f}s sigma={sigma_slip:.4f} R2={slip_fit_r2:.2f}")

        for grade, sg in SENSOR_GRADES.items():
            for variant in VARIANTS:
                kw = dict(variant=variant, sigma_acc=sg["sigma_acc"], sigma_gyro=sg["sigma_gyro"],
                          sf_gyro=sg["sf_gyro"], gyro_bias_rw=sg["gyro_bias_rw"])
                if variant in ("B1", "B2", "B3"):
                    kw["sigma_wheel_vel"] = sigma_wheel[grade]
                if variant == "B2":
                    kw["sigma_slip"] = sigma_slip
                if variant == "B3":
                    kw["tau_slip"] = tau_slip
                    kw["sigma_slip"] = sigma_slip

                model = ImuKinematicModel(**kw)
                J0 = build_J0(model, P0_scale)
                _J, P = pcrlb_recursion(model, truth, dt, J0)
                sig = bound_sigmas(P)

                key = f"c{combo_idx:03d}_{variant}_{grade}"
                results[f"{key}_t"] = t
                for name, arr in sig.items():
                    results[f"{key}_{name}"] = arr

                # Full-trace RMS (dominated by the shared ~1 s convergence
                # transient -- kept only as a diagnostic of convergence
                # speed, NOT for cross-variant comparison).
                sigma_pos_rms_full = float(np.sqrt(np.mean(sig["sigma_pos"] ** 2)))
                sigma_vel_rms_full = float(np.sqrt(np.mean(sig["sigma_vel"] ** 2)))
                sigma_psi_rms_full = float(np.sqrt(np.mean(sig["sigma_psi"] ** 2)))
                sigma_bg_rms_full  = float(np.sqrt(np.mean(sig["sigma_bg"] ** 2)))

                # Steady-state RMS (post-transient, t >= t_settle) -- the
                # PRIMARY comparison metric: this is what actually
                # distinguishes the variants and is what "the bound" means in
                # steady operation.
                sigma_pos_rms = float(np.sqrt(np.mean(sig["sigma_pos"][ss] ** 2)))
                sigma_vel_rms = float(np.sqrt(np.mean(sig["sigma_vel"][ss] ** 2)))
                sigma_psi_rms = float(np.sqrt(np.mean(sig["sigma_psi"][ss] ** 2)))
                sigma_bg_rms  = float(np.sqrt(np.mean(sig["sigma_bg"][ss] ** 2)))
                ratio = achieved_pos_rms_ss / sigma_pos_rms if sigma_pos_rms > 0 else np.nan

                summary_rows.append(dict(
                    combo_idx=combo_idx, group=c["group"], role=c["role"],
                    a=c["a"], ratio=c["ratio"], worbit=c["worbit"],
                    u_peak=c["u_peak"], u_rms=c["u_rms"], v_peak=c["v_peak"],
                    T_total=c["T_total"], t_settle=t_settle,
                    variant=variant, sensor_grade=grade,
                    tau_slip=tau_slip, sigma_slip=sigma_slip, slip_fit_r2=slip_fit_r2,
                    sigma_pos_rms=sigma_pos_rms, sigma_vel_rms=sigma_vel_rms,
                    sigma_psi_rms=sigma_psi_rms, sigma_bg_rms=sigma_bg_rms,
                    sigma_pos_rms_full=sigma_pos_rms_full, sigma_vel_rms_full=sigma_vel_rms_full,
                    sigma_psi_rms_full=sigma_psi_rms_full, sigma_bg_rms_full=sigma_bg_rms_full,
                    achieved_pos_rms=achieved_pos_rms_ss, achieved_pos_rms_full=achieved_pos_rms_full,
                    ratio_achieved_to_bound=ratio,
                ))

    if not summary_rows:
        raise SystemExit("No trajectories processed — nothing to write.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "bound_summary.csv", index=False)
    np.savez(REPORT_DIR / "bound_results.npz", **results)

    print(f"\nWrote {REPORT_DIR / 'bound_summary.csv'} ({len(summary)} rows)")
    print(f"Wrote {REPORT_DIR / 'bound_results.npz'} ({len(results)} arrays)")

    # Headline check (brief §7.4 / §10): B0 vs achieved, grouped by u_peak.
    b0 = summary[(summary.variant == "B0") & (summary.sensor_grade == "default")]
    print("\nB0 (default grade) achieved-to-bound ratio by trajectory:")
    print(b0[["combo_idx", "group", "u_peak", "sigma_pos_rms", "achieved_pos_rms",
               "ratio_achieved_to_bound"]].to_string(index=False))


if __name__ == "__main__":
    main()
