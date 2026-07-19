"""
power_balance_audit.py
=======================
Instantaneous POWER-balance audit at every timestep (your suggestion):
accumulate the per-step power error over dt, and report its distribution.

Physical identity (per timestep):
    P_KE_rate(t)  =  P_fric_contact(t) + P_motor(t) - P_visc(t)

where
    P_KE_rate = v^T M dv + sum J_w w_i dw_i + sum (J_r N) gamma_i dgamma_i
              (analytic, from the stored acceleration sidecar: dVx,dVy,dpsidot,dw*)
    P_fric_contact = sum_i [ Fx_i*Vpx_i + Fy_i*Vpy_i + Mz_i*wz_i ]   (stored)
    P_motor        = sum_i Msat_i * w_i                               (stored)
    P_visc         = sum_i [ p1*w_i^2 + p2*gamma_i^2 ]                (p1/p2 from base.toml)

We compute the instantaneous relative power error
    eps(t) = |P_KE_rate - (P_fric + P_motor - P_visc)| / P_scale(t)
with P_scale = max(|P_KE_rate|, |P_fric|, |P_motor|, |P_visc|, 1e-6)
and report RMS / median / p95 / p99 over the trajectory, plus the integrated
absolute residual (Joules) and its fraction of total motor energy.

The DYi term is ALREADY inside P_fric_contact (Vpx/Vpy store the full
contact velocity Vx - psi_dot*(py+DYi) - ...), so this tests the balance
WITH the DYi correction included.

Run:
  python power_balance_audit.py --data-dir <dir> --config-dir <dir> [--limit N] [--out csv] [--trace path]
"""
import argparse, glob, os, sys, math
import numpy as np
import pyarrow.feather as fe

M=30.0; M_WHEEL=1.4; J_WHEEL=0.00587; J_ROLLER=1.0e-6; ROLLERS=12
AX=0.016; AY=-0.026; IS=4.42

def load_base_toml(p):
    import re
    cfg={}; sec=""
    for raw in open(p,encoding="utf-8"):
        line=raw.strip()
        if not line or line.startswith("#"): continue
        m=re.match(r"\[(.+)\]",line)
        if m: sec=m.group(1)+"."; continue
        kv=re.match(r"([A-Za-z0-9_]+)\s*=\s*(.+)",line)
        if kv:
            k=sec+kv.group(1); v=kv.group(2).strip()
            if v.lower() in ("true","false"): v=v.lower()=="true"
            else:
                try: v=float(v)
                except ValueError: v=v.strip('"')
            cfg[k]=v
    return cfg

def get_geom(d):
    toml=os.path.join(d,"base.toml")
    if not os.path.exists(toml): return dict(p1=0.11,p2=0.00578)
    c=load_base_toml(toml); case=int(c.get("physics.friction_case",1))
    if case==1:
        p1=c.get("platform.viscous.p1_case1",0.11); p2=c.get("platform.viscous.p2_case1",0.00578)
    else:
        p1=c.get("platform.viscous.p1_case2",0.011); p2=c.get("platform.viscous.p2_case2",0.000578)
    return dict(p1=p1,p2=p2)

def audit(path, geom, accel_path):
    t=fe.read_table(path); cols=t.column_names; d={c:t.column(c).to_numpy() for c in cols}
    time=d["time"]; dt=np.gradient(time)
    Vx=d["Vx"]; Vy=d["Vy"]; psi_dot=d["psi_dot"]
    Fx=np.stack([d[f"Fx_{i}"] for i in range(1,5)],0)
    Fy=np.stack([d[f"Fy_{i}"] for i in range(1,5)],0)
    Mz=np.stack([d[f"Mz_{i}"] for i in range(1,5)],0)
    Msat=np.stack([d[f"Msat_{i}"] for i in range(1,5)],0)
    w=np.stack([d[f"w{i}"] for i in range(1,5)],0)
    gamma=np.stack([d[f"gamma{i}"] for i in range(1,5)],0)
    Vpx=np.stack([d[f"Vpx_{i}"] for i in range(1,5)],0)
    Vpy=np.stack([d[f"Vpy_{i}"] for i in range(1,5)],0)
    wz=np.stack([d[f"wz_{i}"] for i in range(1,5)],0)

    # ---- Analytic KE rate from acceleration sidecar ----
    a=fe.read_table(accel_path); ac={c:a.column(c).to_numpy() for c in a.column_names}
    dVx=ac["dVx"]; dVy=ac["dVy"]; dpsidot=ac["dpsidot"]
    dw=np.stack([ac[f"dw{i}"] for i in range(1,5)],0)

    ms=M+4*M_WHEEL
    # platform: v^T M dv   (M = [[ms,0,-m aY],[0,ms,m aX],[-m aY, m aX, Is]])
    P_plat = (Vx*(ms*dVx - M*AY*dpsidot)
              + Vy*(ms*dVy + M*AX*dpsidot)
              + psi_dot*(-M*AY*dVx + M*AX*dVy + IS*dpsidot))
    P_wheel = np.sum(w * J_WHEEL * dw, 0)
    P_roller= np.sum(gamma * (J_ROLLER*ROLLERS) * 0.0, 0)   # dgamma not stored; roller KE negligible (J_r=1e-6)
    P_KE_rate = P_plat + P_wheel + P_roller

    # ---- Power-balance right-hand side ----
    P_fric = np.sum(Fx*Vpx + Fy*Vpy + Mz*wz, 0)
    P_motor= np.sum(Msat*w, 0)
    P_visc = np.sum(geom["p1"]*w**2 + geom["p2"]*gamma**2, 0)
    P_rhs  = P_fric + P_motor - P_visc

    resid = P_KE_rate - P_rhs                       # (N,)
    P_scale = np.maximum.reduce([np.abs(P_KE_rate), np.abs(P_fric),
                                 np.abs(P_motor), np.abs(P_visc), np.ones_like(P_KE_rate)*1e-6])
    eps = np.abs(resid) / P_scale

    E_motor = np.sum(P_motor*dt)
    E_resid = np.sum(np.abs(resid)*dt)
    return dict(
        path=path, n=len(time), T=time[-1]-time[0], E_motor=E_motor,
        E_resid=E_resid,
        frac_resid=(E_resid/abs(E_motor)) if E_motor else float("nan"),
        eps_rms=float(np.sqrt(np.mean(eps**2))),
        eps_med=float(np.median(eps)),
        eps_p95=float(np.percentile(eps,95)),
        eps_p99=float(np.percentile(eps,99)),
        eps_max=float(np.max(eps)),
        resid=resid, time=time,
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\data\Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--config-dir", default=r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\trajectory_files_run_0p3_main")
    ap.add_argument("--whitelist", default=None, help="diagnostics_combined.csv; only 'keep*' rows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", default=None)
    args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(args.data_dir,"*.arrow")))
    if args.whitelist:
        import pandas as pd
        wl = pd.read_csv(args.whitelist)
        keep = set(wl.loc[wl["combined_reco"].astype(str).str.startswith("keep"), "file"])
        files = [f for f in files if os.path.basename(f) in keep]
        print(f"whitelist: {len(keep)} keep* files -> {len(files)} present in data dir")
    if args.limit: files=files[:args.limit]
    geom=get_geom(args.config_dir)
    print(f"geom: p1={geom['p1']} p2={geom['p2']}  (Ke-rate from accel sidecar)")
    print(f"{'file':62s}  E_mot    frac%   eps_rms%  med%   p95%   p99%   max%")
    rows=[]
    for f in files:
        try:
            accel=f.replace(".arrow","_accel.arrow")
            if not os.path.exists(accel):
                accel=os.path.join(os.path.dirname(f),"accel",os.path.basename(f)[:-6]+"_accel.arrow")
            r=audit(f,geom,accel); rows.append(r)
            print(f"{os.path.basename(f):62s}  {r['E_motor']:+8.0f}  {r['frac_resid']*100:+6.2f}  "
                  f"{r['eps_rms']*100:+8.3f}  {r['eps_med']*100:+6.2f}  {r['eps_p95']*100:+6.2f}  "
                  f"{r['eps_p99']*100:+6.2f}  {r['eps_max']*100:+7.2f}")
        except Exception as e:
            print(f"ERR {os.path.basename(f)}: {e}",file=sys.stderr)
    if args.out and rows:
        import csv
        keys=["path","n","T","E_motor","E_resid","frac_resid","eps_rms","eps_med","eps_p95","eps_p99","eps_max"]
        with open(args.out,"w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=keys,extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {args.out}")
    if args.trace:
        r=audit(args.trace, geom, args.trace.replace(".arrow","_accel.arrow"))
        out=args.trace+".power_trace.npz"
        np.savez(out, time=r["time"], resid=r["resid"])
        print(f"\nwrote trace -> {out}  (eps_max={r['eps_max']*100:.3f}%, E_resid/E_motor={r['frac_resid']*100:.3f}%)")

if __name__=="__main__": main()
