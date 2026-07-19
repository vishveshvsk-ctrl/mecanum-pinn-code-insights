"""
energy_audit_dyi.py
====================
Measure the energy-balance residual caused by the roller-contact lateral offset
DYi that the platform torque balance drops (see discussion: Mz_roller_i uses
moment arm `py`, not `py + DYi`).

Model reference:
  - notebook Cell 22 (dynamics_full_mf_asmc!), datastore.jl compute_labels
  - DYi_i = Ra * tan(delta_i) * tan(theta_tilde_i)
  - theta_tilde = sawtooth_approx(theta)   [SAWTOOTH = :tanh in the notebook]

The full energy identity (derived from the 3 KE reservoirs) is:

  dE_KE/dt =  P_friction_contact          (friction power at the true contact)
            + P_motor                      (sum Msat_i * w_i)
            - P_viscous                    (sum p1*w_i^2 + p2*gamma_i^2)

where P_friction_contact = sum_i [ Fx_i*Vpx_i + Fy_i*Vpy_i + Mz_i*wz_i ].

Because the platform torque uses arm `py` (not `py+DYi`), the platform-KE
side credits only psi_dot*(-py*Fx_i), whereas the friction-contact side
contains psi_dot*(-(py+DYi)*Fx_i). The mismatch is exactly

  R_DYi(t) = sum_i  psi_dot(t) * DYi_i(t) * Fx_i(t)

so that  E_KE_actual + integral(R_DYi) == (contact friction + motor - viscous).

This script:
  1. Reconstructs R_DYi(t) per trajectory directly from stored columns
     (Vpx_i, Vpy_i, wz_i, Fx_i, Msat_i, w_i, gamma_i, theta_i are all stored,
      so no ODE re-run is needed; p1/p2 and geometry come from base.toml).
  2. Integrates it over the trajectory -> E_DYi (Joules).
  3. Compares against total motor energy E_motor and total friction energy
     E_fric so you can set a pass threshold as a fraction.
  4. Writes a per-file summary CSV and (optionally) a time-series for one file.

Run:
  python energy_audit_dyi.py --data-dir <dir> --config-dir <dir> [--limit N] [--out csv] [--trace <one_arrow_path>]
"""

import argparse
import glob
import os
import sys
import math
import numpy as np
import pyarrow.feather as fe

# --------------------------------------------------------------------------
# Sawtooth smoother (notebook Cell 11, SAWTOOTH = :tanh, TANH_K = 60.0)
# --------------------------------------------------------------------------
TANH_K = 60.0

def sawtooth_tanh(theta):
    s = np.sin(12.0 * theta)
    c = np.cos(12.0 * theta)
    return np.arctan2(TANH_K * s, TANH_K * c + 1.0) / 12.0

# Structural constants (Cell 7): wheel delta angles, geometry wc_y = py = +/- l
DELTA = np.array([-math.pi/4, math.pi/4, math.pi/4, -math.pi/4])  # sdi sign pattern

# Per-wheel lateral center offset (notebook Cell 7, wc_x/wc_y)
WC_X = np.array([ 0.235,  0.235, -0.235, -0.235])   # h
WC_Y = np.array([ 0.15,  -0.15,   0.15,  -0.15])    # l  (this is `py`)

# Mass / inertia needed for platform KE (Cell 7). These are the SAME for all
# runs (only p1/p2, mu, chi vary per run via base.toml), so hard-code the
# invariant parts and pull p1/p2 + geometry from base.toml.
M = 30.0
M_WHEEL = 1.4
J_WHEEL = 0.00587
J_ROLLER = 1.0e-6
ROLLERS = 12
AX = 0.016
AY = -0.026
IS = 4.42


def load_base_toml(path):
    """Minimal TOML reader for the flat-ish base.toml used here."""
    import re
    cfg = {}
    section = ""
    for raw in open(path, "r", encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\[(.+)\]", line)
        if m:
            section = m.group(1) + "."
            continue
        kv = re.match(r"([A-Za-z0-9_]+)\s*=\s*(.+)", line)
        if kv:
            key = section + kv.group(1)
            val = kv.group(2).strip()
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = val.strip('"')
            cfg[key] = val
    return cfg


def get_geom_p(cfg, run_dir):
    """Read geometry + viscous from base.toml of the run directory."""
    toml = os.path.join(run_dir, "base.toml")
    if not os.path.exists(toml):
        # try to infer from data filename case -> default case1 constants
        return dict(R=0.05, Ra=0.0355, l=0.15, p1=0.11, p2=0.00578)
    c = load_base_toml(toml)
    R = c.get("platform.geometry.R", 0.05)
    Ra = c.get("platform.geometry.Ra", 0.0355)
    l = c.get("platform.geometry.l", 0.15)
    case = int(c.get("physics.friction_case", 1))
    if case == 1:
        p1 = c.get("platform.viscous.p1_case1", 0.11)
        p2 = c.get("platform.viscous.p2_case1", 0.00578)
    else:
        p1 = c.get("platform.viscous.p1_case2", 0.011)
        p2 = c.get("platform.viscous.p2_case2", 0.000578)
    return dict(R=R, Ra=Ra, l=l, p1=p1, p2=p2)


def audit_file(path, geom):
    t = fe.read_table(path)
    cols = t.column_names
    d = {c: t.column(c).to_numpy() for c in cols}
    time = d["time"]
    dt = np.gradient(time)

    psi_dot = d["psi_dot"]
    Vx = d["Vx"]; Vy = d["Vy"]

    # Per-wheel reconstruction
    Fx = np.stack([d[f"Fx_{i}"] for i in range(1, 5)], axis=0)   # (4, N)
    Fy = np.stack([d[f"Fy_{i}"] for i in range(1, 5)], axis=0)
    Mz = np.stack([d[f"Mz_{i}"] for i in range(1, 5)], axis=0)
    Msat = np.stack([d[f"Msat_{i}"] for i in range(1, 5)], axis=0)
    w = np.stack([d[f"w{i}"] for i in range(1, 5)], axis=0)
    gamma = np.stack([d[f"gamma{i}"] for i in range(1, 5)], axis=0)
    theta = np.stack([d[f"theta{i}"] for i in range(1, 5)], axis=0)
    Vpx = np.stack([d[f"Vpx_{i}"] for i in range(1, 5)], axis=0)
    Vpy = np.stack([d[f"Vpy_{i}"] for i in range(1, 5)], axis=0)
    wz = np.stack([d[f"wz_{i}"] for i in range(1, 5)], axis=0)

    # DYi_i = Ra * tan(delta_i) * tan(theta_tilde_i)
    theta_tilde = sawtooth_tanh(theta)
    DYi = geom["Ra"] * np.tan(DELTA)[:, None] * np.tan(theta_tilde)   # (4, N)

    # --- Residual R_DYi(t) = sum_i psi_dot * DYi_i * Fx_i ---
    R_DYi = np.sum(psi_dot[None, :] * DYi * Fx, axis=0)   # (N,)

    # --- Reference energy terms for context / denominator choice ---
    # Friction power at TRUE contact point (what the DYi term is a fraction of)
    P_fric = np.sum(Fx * Vpx + Fy * Vpy + Mz * wz, axis=0)
    # Motor power (saturated torque * wheel speed)
    P_motor = np.sum(Msat * w, axis=0)
    # Viscous dissipation
    P_visc = np.sum(geom["p1"] * w**2 + geom["p2"] * gamma**2, axis=0)

    E_DYi = np.sum(R_DYi * dt)
    E_fric = np.sum(P_fric * dt)
    E_motor = np.sum(P_motor * dt)
    E_visc = np.sum(P_visc * dt)

    # |R_DYi| peak (worst-case instantaneous leakage power)
    R_DYi_peak = float(np.max(np.abs(R_DYi)))

    return dict(
        path=path,
        n=len(time),
        T=time[-1] - time[0],
        E_DYi=E_DYi,
        E_fric=E_fric,
        E_motor=E_motor,
        E_visc=E_visc,
        R_DYi_peak=R_DYi_peak,
        R_DYi=R_DYi, time=time,
        frac_DYi_over_motor=(E_DYi / E_motor) if E_motor != 0 else float("nan"),
        frac_DYi_over_fric=(E_DYi / E_fric) if E_fric != 0 else float("nan"),
        frac_peak_over_motor=(R_DYi_peak / E_motor) if E_motor != 0 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\data\Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--config-dir",
                    default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\trajectory_files_run_0p3_main")
    ap.add_argument("--limit", type=int, default=0, help="audit first N files (0=all)")
    ap.add_argument("--out", default=None, help="summary CSV path")
    ap.add_argument("--trace", default=None, help="one arrow path to dump R_DYi(t) time series")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.arrow")))
    if args.limit:
        files = files[:args.limit]

    geom = get_geom_p(None, args.config_dir)
    print(f"geometry/viscous from {args.config_dir}/base.toml: "
          f"R={geom['R']} Ra={geom['Ra']} l={geom['l']} p1={geom['p1']} p2={geom['p2']}")

    rows = []
    for f in files:
        try:
            r = audit_file(f, geom)
            rows.append(r)
            print(f"{os.path.basename(f):70s} E_motor={r['E_motor']:+9.2f}  "
                  f"E_DYi={r['E_DYi']:+8.3f}J  ({r['frac_DYi_over_motor']*100:+6.3f}% motor)  "
                  f"({r['frac_DYi_over_fric']*100:+6.3f}% fric)  "
                  f"peak|R|={r['R_DYi_peak']:.4f}W")
        except Exception as e:
            print(f"FAILED {os.path.basename(f)}: {e}", file=sys.stderr)

    if args.out and rows:
        import csv
        keys = ["path", "n", "T", "E_DYi", "E_fric", "E_motor", "E_visc",
                "R_DYi_peak", "frac_DYi_over_motor", "frac_DYi_over_fric",
                "frac_peak_over_motor"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote summary -> {args.out}")

    if args.trace:
        r = audit_file(args.trace, geom)
        out = args.trace + ".dyi_trace.npz"
        np.savez(out, time=r["time"], R_DYi=r["R_DYi"])
        print(f"\nwrote trace -> {out}  (R_DYi peak={np.max(np.abs(r['R_DYi'])):.4f} W, "
              f"integral={r['E_DYi']:.4f} J)")


if __name__ == "__main__":
    main()
