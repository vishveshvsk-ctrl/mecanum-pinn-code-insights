"""
check_energy_closure.py
=======================
Test whether the full energy balance closes to machine precision when the
DYi term is INCLUDED in the platform yaw-moment arm.

Identity tested (derived from the 3 KE reservoirs of the 39-D model):

  dE_KE/dt  =  P_fric_contact            (friction power at TRUE contact)
            + P_motor                    (sum Msat_i * w_i)
            - P_visc                     (sum p1*w_i^2 + p2*gamma_i^2)

where P_fric_contact = sum_i [ Fx_i*Vpx_i + Fy_i*Vpy_i + Mz_i*wz_i ]
uses the STORED contact velocities (datastore.jl computes these from the
full Vpi_x = Vx - psi_dot*(py+DYi) - ..., so they already include DYi).

We reconstruct dE_KE/dt TWO independent ways and compare to the RHS side:

  (A) Exact KE from stored velocities, numerically differentiated:
        E_KE = 1/2 v^T M v + 1/2 sum J_w w_i^2 + 1/2 sum (J_r*N) gamma_i^2
        dE_KE/dt = gradient(E_KE) / dt
  (B) Sum of reservoir power rates from the stored forces/torques:
        P_plat   = v * (M_inv * RHS_with_DYi)     [arm py+DYi]
        P_wheel  = sum w_i * (Msat_i - Fx_i*R - p1*w_i) / J_w
        P_roller = sum gamma_i * (-p2*gamma_i - Fx_i*A_i - Fy_i*B_i - Mz_i*C_i)
        dE_KE/dt (B) = P_plat + P_wheel + P_roller

Then form:
  residual = dE_KE/dt (B) - (P_fric_contact + P_motor - P_visc)
and integrate -> E_residual. If the model is self-consistent with DYi
included, this should be ~machine precision (not the -R_DYi we got before).

Run:
  python check_energy_closure.py --data-dir <dir> --config-dir <dir> [--limit N] [--trace path]
"""
import argparse
import glob
import os
import sys
import math
import numpy as np
import pyarrow.feather as fe

TANH_K = 60.0
def sawtooth_tanh(theta):
    s = np.sin(12.0 * theta); c = np.cos(12.0 * theta)
    return np.arctan2(TANH_K * s, TANH_K * c + 1.0) / 12.0

DELTA = np.array([-math.pi/4, math.pi/4, math.pi/4, -math.pi/4])
WC_X = np.array([ 0.235,  0.235, -0.235, -0.235])
WC_Y = np.array([ 0.15,  -0.15,   0.15,  -0.15])
M = 30.0; M_WHEEL = 1.4; J_WHEEL = 0.00587; J_ROLLER = 1.0e-6
ROLLERS = 12; AX = 0.016; AY = -0.026; IS = 4.42


def load_base_toml(path):
    import re
    cfg = {}; section = ""
    for raw in open(path, "r", encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\[(.+)\]", line)
        if m:
            section = m.group(1) + "."; continue
        kv = re.match(r"([A-Za-z0-9_]+)\s*=\s*(.+)", line)
        if kv:
            key = section + kv.group(1); val = kv.group(2).strip()
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try: val = float(val)
                except ValueError: val = val.strip('"')
            cfg[key] = val
    return cfg


def get_geom(run_dir):
    toml = os.path.join(run_dir, "base.toml")
    if not os.path.exists(toml):
        return dict(R=0.05, Ra=0.0355, l=0.15, p1=0.11, p2=0.00578)
    c = load_base_toml(toml)
    R = c.get("platform.geometry.R", 0.05); Ra = c.get("platform.geometry.Ra", 0.0355)
    case = int(c.get("physics.friction_case", 1))
    if case == 1:
        p1 = c.get("platform.viscous.p1_case1", 0.11); p2 = c.get("platform.viscous.p2_case1", 0.00578)
    else:
        p1 = c.get("platform.viscous.p1_case2", 0.011); p2 = c.get("platform.viscous.p2_case2", 0.000578)
    return dict(R=R, Ra=Ra, l=0.15, p1=p1, p2=p2)


def audit(path, geom):
    t = fe.read_table(path); cols = t.column_names
    d = {c: t.column(c).to_numpy() for c in cols}
    time = d["time"]; dt = np.gradient(time)
    psi_dot = d["psi_dot"]; Vx = d["Vx"]; Vy = d["Vy"]
    Fx = np.stack([d[f"Fx_{i}"] for i in range(1,5)], 0)
    Fy = np.stack([d[f"Fy_{i}"] for i in range(1,5)], 0)
    Mz = np.stack([d[f"Mz_{i}"] for i in range(1,5)], 0)
    Msat = np.stack([d[f"Msat_{i}"] for i in range(1,5)], 0)
    w = np.stack([d[f"w{i}"] for i in range(1,5)], 0)
    gamma = np.stack([d[f"gamma{i}"] for i in range(1,5)], 0)
    theta = np.stack([d[f"theta{i}"] for i in range(1,5)], 0)
    Vpx = np.stack([d[f"Vpx_{i}"] for i in range(1,5)], 0)
    Vpy = np.stack([d[f"Vpy_{i}"] for i in range(1,5)], 0)
    wz = np.stack([d[f"wz_{i}"] for i in range(1,5)], 0)

    theta_tilde = sawtooth_tanh(theta)
    DYi = geom["Ra"] * np.tan(DELTA)[:,None] * np.tan(theta_tilde)
    sti_t = np.sin(theta_tilde); cti_t = np.cos(theta_tilde)
    sdi = np.sin(DELTA)[:,None]; cdi = np.cos(DELTA)[:,None]
    A = sdi * (geom["R"] - geom["Ra"]*cti_t) - DYi*sti_t*cdi
    B = cdi * (geom["Ra"] - geom["R"]*cti_t)
    C = sti_t * cdi

    # ---- Power terms ----
    P_fric = np.sum(Fx*Vpx + Fy*Vpy + Mz*wz, 0)      # contact friction power (has DYi)
    P_motor = np.sum(Msat * w, 0)
    P_visc = np.sum(geom["p1"]*w**2 + geom["p2"]*gamma**2, 0)

    # ---- (A) Exact KE from velocities, numerical derivative ----
    ms = M + 4*M_WHEEL
    Mmat = np.array([[ms,0,-M*AY],[0,ms,M*AX],[-M*AY,M*AX,IS]], float)
    Minv = np.linalg.inv(Mmat)
    v = np.stack([Vx, Vy, psi_dot], 0)               # (3,N)
    E_plat = 0.5 * np.einsum("in,ij,jn->n", v, Mmat, v)
    E_wheel = 0.5 * J_WHEEL * np.sum(w**2, 0)
    E_roller = 0.5 * J_ROLLER * ROLLERS * np.sum(gamma**2, 0)
    E_KE = E_plat + E_wheel + E_roller
    dEKE_numeric = np.gradient(E_KE) / dt             # (N,)

    # ---- (B) Reservoir power rates, platform arm = py + DYi ----
    RHS0 = np.sum(Fx,0) + ms*psi_dot*Vy + M*AX*psi_dot**2
    RHS1 = np.sum(Fy,0) - ms*psi_dot*Vx + M*AY*psi_dot**2
    # arm py+DYi (THE FIX):
    RHS2 = np.sum(WC_X[:,None]*Fy - (WC_Y[:,None]+DYi)*Fx, 0) - M*psi_dot*(AX*Vx+AY*Vy)
    dv = Minv @ np.stack([RHS0, RHS1, RHS2], 0)       # (3,N)
    P_plat = np.sum(v * dv, 0)
    P_wheel = np.sum(w * (Msat - Fx*geom["R"] - geom["p1"]*w) / J_WHEEL, 0)
    P_roller = np.sum(gamma * (-geom["p2"]*gamma - Fx*A - Fy*B - Mz*C), 0)
    dEKE_B = P_plat + P_wheel + P_roller

    # ---- Residuals ----
    # (1) numeric-KE vs (B) reconstruction: should be ~machine precision
    resid_AvB = dEKE_numeric - dEKE_B
    E_resid_AvB = np.sum(np.abs(resid_AvB) * dt)
    # (2) (B) vs contact-balance side: should be ~0 once DYi included
    residual = dEKE_B - (P_fric + P_motor - P_visc)
    E_residual = np.sum(np.abs(residual) * dt)
    # (3) numeric-KE vs contact-balance side
    residual_n = dEKE_numeric - (P_fric + P_motor - P_visc)
    E_residual_n = np.sum(np.abs(residual_n) * dt)

    E_motor = np.sum(P_motor * dt)
    return dict(
        path=path, n=len(time), T=time[-1]-time[0], E_motor=E_motor,
        E_KE_total=float(E_KE[-1]-E_KE[0]),
        E_resid_AvB=E_resid_AvB,
        E_residual=E_residual,
        E_residual_n=E_residual_n,
        frac_AvB=(E_resid_AvB/abs(E_motor)) if E_motor else float("nan"),
        frac_resid=(E_residual/abs(E_motor)) if E_motor else float("nan"),
        frac_resid_n=(E_residual_n/abs(E_motor)) if E_motor else float("nan"),
        peak_AvB=float(np.max(np.abs(resid_AvB))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\data\Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--config-dir", default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\trajectory_files_run_0p3_main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.arrow")))
    if args.limit: files = files[:args.limit]
    geom = get_geom(args.config_dir)
    print(f"geom: R={geom['R']} Ra={geom['Ra']} p1={geom['p1']} p2={geom['p2']}")
    print(f"{'file':68s}  E_motor    frac_AvB%   frac_resid%  frac_resid_n%  peak_AvB")
    rows=[]
    for f in files:
        try:
            r = audit(f, geom); rows.append(r)
            print(f"{os.path.basename(f):68s}  {r['E_motor']:+9.1f}  "
                  f"{r['frac_AvB']*100:+10.4f}  {r['frac_resid']*100:+11.4f}  "
                  f"{r['frac_resid_n']*100:+12.4f}  {r['peak_AvB']:.3e}")
        except Exception as e:
            print(f"FAILED {os.path.basename(f)}: {e}", file=sys.stderr)
    if args.out and rows:
        import csv
        keys=["path","n","T","E_motor","E_KE_total","E_resid_AvB","E_residual","E_residual_n",
              "frac_AvB","frac_resid","frac_resid_n","peak_AvB"]
        with open(args.out,"w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=keys,extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
