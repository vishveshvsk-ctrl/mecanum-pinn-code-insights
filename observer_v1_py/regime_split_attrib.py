#!/usr/bin/env python
# =============================================================================
# regime_split_attrib.py — per-regime head error + |Vp| head attribution, in
# PHYSICAL units. Writes <run_dir>/regime_attrib.json so the RMSEs survive.
#
# Answers three things the aggregate metrics hide:
#   (1) gamma / dV RMSE binned by TRUE slip regime (stick / slip / high).
#   (2) dV reality check: the target dV_true = V_true - V_hat is the FRONT-END's own
#       error; how much of it does the head actually recover, per axis, and how does
#       the residual compare to the 1 cm/s gate width?
#   (3) |Vp| ATTRIBUTION — swap true<->model per head to isolate who breaks the gate:
#         vp(g_hat , dv_hat )  model/model  = reality
#         vp(g_TRUE, dv_hat )  isolates dV error
#         vp(g_hat , dv_TRUE)  isolates gamma error
#         vp(g_TRUE, dv_TRUE)  identity floor (decimation only)
#       reported with stick_frac each, i.e. "would fixing gamma alone un-saturate
#       the gate?"  (Measured on the grnd80_slip02 pair: yes — gamma is ~82% of it.)
#
# Physical scales: gamma_rmse_phys = gamma_rmse_norm * gamma_p95 (norm.npz);
# dV per-axis via cfg.dv_scale; |Vp| and vpm are already m/s.
#
# Run from code_insights/ with the conda myenv python (pyarrow imports before torch):
#   python observer_v1_py/regime_split_attrib.py --run observer_v1_py/runs/<tag>
# =============================================================================
from __future__ import annotations

import pyarrow.feather  # noqa: F401  (import BEFORE torch on Windows)

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer import config as C
from mecanum_observer import data as D
from mecanum_observer import data_v2hy3 as D2H
from mecanum_observer import physics_v2hy3 as P
from mecanum_observer.models_v2hy3 import build_model_v2hy3
from mecanum_observer.training_v2hy3 import _phys_batch, _resolve_precision

FIELDS = ObserverConfigV2Hy3.__dataclass_fields__
GATE = C.LG_V_STR          # 0.01 m/s stick/slip boundary
DVP_DG = 0.0141            # m/rad*s, mean |dVp/dgamma| over theta_t in +/-15 deg


def vp_of(gamma_phys, dv_phys, phys):
    """|Vp| from a (gamma, dV) pair — the exact quantity the physics gate consumes."""
    Vpx, Vpy, _, _, _, _ = P.contact_from_gamma(
        gamma_phys, phys["psi_dot"],
        phys["Vpx0_hat"] + dv_phys[:, 0:1],
        phys["Vpy0_hat"] + dv_phys[:, 1:2],
        phys["cti"], phys["sti"])
    return torch.sqrt(Vpx * Vpx + Vpy * Vpy + C.LG_EPS_REG ** 2)


def _is_gammakin(m: dict) -> bool:
    return (m.get("gamma_parametrization") == "residual_noslip_model1"
            or bool(m["cfg"].get("gamma_residual", False)))


@torch.no_grad()
def collect(run_dir: Path, ckpt: str, split: str, n_files: int, vy_label: float,
            use_vp_components: Optional[bool] = None,
            regime: str = ""):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = json.load(open(run_dir / "metrics.json"))
    gammakin = _is_gammakin(m)
    if gammakin:
        # Load the FULL variant config (dgamma_scale, dv_scale, ... survive) instead
        # of the base ObserverConfigV2Hy3, which would silently drop them and mis-read
        # the Δγ head as γ.
        from mecanum_observer.config_v2hy3_gammakin import ObserverConfigV2Hy3GammaKin as CFGK
        fk = CFGK.__dataclass_fields__
        cfg = CFGK(**{k: v for k, v in m["cfg"].items() if k in fk})
        dgsc = cfg.dgamma_scale
    else:
        cfg = ObserverConfigV2Hy3(**{k: v for k, v in m["cfg"].items() if k in FIELDS})
        dgsc = None
    cfg.jobs = 0
    # Label-source override. Checkpoints predating the .vpcomp sidecars carry no
    # `use_vp_components` field, so they would silently rebuild cfg with the OLD
    # Jensen-biased vpm while a slipLOG run (which saved True) uses the corrected
    # one -- scoring the two against DIFFERENT true_stick_frac and manufacturing a
    # difference. Forcing it here puts every run on one label footing.
    if use_vp_components is not None:
        cfg.use_vp_components = bool(use_vp_components)
    cfg = cfg.resolved()
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    gamma_p95 = float(nrm.y_std[0])
    gstd = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=dev).view(1, 1)
    gmean = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=dev).view(1, 1)
    dvsc = torch.tensor(cfg.dv_scale, dtype=torch.float32, device=dev).view(1, 2)
    amp = _resolve_precision(cfg, dev)

    model = build_model_v2hy3(cfg).to(dev)
    ck = torch.load(run_dir / ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()

    if regime:
        # OOD/held-out eval: discover the files a regime TOML selects (e.g.
        # eval_multisine) instead of the cross-fold split. No fold split.
        reg = D.regime_to_kwargs(D.load_regime(Path(regime)))
        for k, v in reg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.train_fold = ""
        files = D.discover(cfg)
    else:
        files = D2H.discover_and_split(cfg)[split]
    files = list(files)[:: max(1, len(files) // n_files)][:n_files]
    ds = D2H.WindowStreamV2Hy3(files, nrm, cfg, shuffle=False)
    ld = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, num_workers=0)

    A = {k: [] for k in ("tvp", "gerr", "dvex", "dvey", "dvtx", "dvty",
                         "dvhx", "dvhy", "mm", "tm", "mt", "tt")}
    for b in ld:
        Gw = b["Gw"].to(dev); Pw = b["Pw"].to(dev)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            gh, dh = model(Gw, Pw)
        gh = gh.float(); dh = dh.float()
        phys = _phys_batch(b, dev)
        dv_hat = dh * dvsc
        dv_true = b["y_dv"].to(dev) * dvsc
        if gammakin:
            # Assemble γ from the no-slip base + residual, EXACTLY as training does.
            # vy_label picks the base's V_y: 1 = TRUE label (matches lam=1 training),
            # 0 = deployable (V̂ + model ΔV̂) — the honest inference number.
            dv_used = (vy_label * b["y_dv"].to(dev) + (1.0 - vy_label) * dh) * dvsc
            Vpy0_u = phys["Vpy0_hat"] + dv_used[:, 1:2]
            g_hat = P.gamma_noslip(Vpy0_u, phys["cti"]) + gh * dgsc
        else:
            g_hat = gh * gstd + gmean
        g_true = b["y_gamma"].to(dev) * gstd + gmean
        tvp = b["slip_mag"].to(dev)
        rep = lambda t: t[:, None].expand(-1, C.N_WHEELS).reshape(-1).cpu().numpy()
        A["tvp"].append(tvp.reshape(-1).cpu().numpy())
        A["gerr"].append((g_hat - g_true).reshape(-1).cpu().numpy())
        A["dvex"].append(rep(dv_hat[:, 0] - dv_true[:, 0]))
        A["dvey"].append(rep(dv_hat[:, 1] - dv_true[:, 1]))
        A["dvtx"].append(rep(dv_true[:, 0])); A["dvty"].append(rep(dv_true[:, 1]))
        A["dvhx"].append(rep(dv_hat[:, 0]));  A["dvhy"].append(rep(dv_hat[:, 1]))
        A["mm"].append(vp_of(g_hat, dv_hat, phys).reshape(-1).cpu().numpy())
        A["tm"].append(vp_of(g_true, dv_hat, phys).reshape(-1).cpu().numpy())
        A["mt"].append(vp_of(g_hat, dv_true, phys).reshape(-1).cpu().numpy())
        A["tt"].append(vp_of(g_true, dv_true, phys).reshape(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in A.items()}, gamma_p95, len(files)


def _rmse(x):
    return float(np.sqrt(np.mean(x * x)))


def main():
    ap = argparse.ArgumentParser(description="Per-regime head error + |Vp| attribution.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_best.pt")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--n-files", type=int, default=60)
    ap.add_argument("--vy-label", type=float, default=0.0,
                    help="gammakin only: base V_y source. 0 = deployable (V̂ + model "
                         "ΔV̂), the honest inference number; 1 = TRUE V_y label "
                         "(matches lam=1 training). Ignored for baseline runs.")
    ap.add_argument("--regime", default="",
                    help="regime TOML to discover files from (e.g. regimes/"
                         "eval_multisine.toml) — OOD/held-out eval instead of the "
                         "cross-fold split. Output goes to regime_attrib_<stem>*.json.")
    ap.add_argument("--use-vp-components", dest="use_vp", action="store_true",
                    default=None,
                    help="force the component-consistent vpm label (.vpcomp.npz "
                         "sidecars) regardless of what the checkpoint's cfg says. "
                         "REQUIRED to compare runs trained before and after the "
                         "sidecars exist -- otherwise each is scored against its "
                         "own true_stick_frac. Output is tagged _vpc.")
    ap.add_argument("--no-use-vp-components", dest="use_vp", action="store_false",
                    help="force the OLD decimate(hypot(.)) label.")
    args = ap.parse_args()

    run = Path(args.run)
    a, gamma_p95, nf = collect(run, args.checkpoint, args.split, args.n_files,
                               args.vy_label, args.use_vp, args.regime)
    t = a["tvp"]
    bins = [("all", np.ones_like(t, bool)), ("stick<0.01", t < 0.01),
            ("slip0.01-0.6", (t >= 0.01) & (t < 0.6)), ("high>0.6", t >= 0.6)]

    out = {"run": run.name, "checkpoint": args.checkpoint, "split": args.split,
           "n_files": nf, "n_samples": int(t.size), "gamma_p95": gamma_p95,
           "vy_label": args.vy_label, "gate_m_s": GATE,
           "regime": {}, "dv": {}, "attribution": {}}

    print(f"\n########## {run.name}  [{args.split}, {nf} files, n={t.size:,}] ##########")
    print(f"gate boundary = {GATE*100:.1f} cm/s\n")
    print("=== (1) HEAD ERROR BY TRUE SLIP REGIME (physical) ===")
    print(f"  {'bin':13} {'frac':>6} {'gamma rad/s':>11} {'->|Vp| cm/s':>11} "
          f"{'dVx cm/s':>9} {'dVy cm/s':>9}")
    for nm, msk in bins:
        if not msk.sum():
            continue
        g = _rmse(a["gerr"][msk])
        r = dict(frac=float(msk.mean()), gamma_rmse_rad_s=g,
                 gamma_to_vp_cm_s=g * DVP_DG * 100,
                 dvx_rmse_cm_s=_rmse(a["dvex"][msk]) * 100,
                 dvy_rmse_cm_s=_rmse(a["dvey"][msk]) * 100)
        out["regime"][nm] = r
        print(f"  {nm:13} {r['frac']*100:5.1f}% {g:11.3f} {r['gamma_to_vp_cm_s']:11.2f} "
              f"{r['dvx_rmse_cm_s']:9.2f} {r['dvy_rmse_cm_s']:9.2f}")

    print("\n=== (2) dV REALITY CHECK (target = V_true - V_hat = front-end error) ===")
    for ax, ekey, hkey, tkey in (("x", "dvex", "dvhx", "dvtx"),
                                 ("y", "dvey", "dvhy", "dvty")):
        tgt, err, hat = a[tkey], a[ekey], a[hkey]
        rms_t, rms_e = _rmse(tgt), _rmse(err)
        r = dict(target_rms_cm_s=rms_t * 100, residual_rms_cm_s=rms_e * 100,
                 captured_frac=float(1 - rms_e / rms_t),
                 r2=float(1 - rms_e ** 2 / rms_t ** 2),
                 corr=float(np.corrcoef(hat, tgt)[0, 1]),
                 residual_vs_gate=float(rms_e / GATE))
        out["dv"][ax] = r
        print(f"  dV_{ax}: |target| RMS {r['target_rms_cm_s']:6.2f} -> residual "
              f"{r['residual_rms_cm_s']:5.2f} cm/s | captured {r['captured_frac']*100:5.1f}% "
              f" R2={r['r2']:6.3f} corr={r['corr']:6.4f} | vs gate {r['residual_vs_gate']:5.2f}x")

    print("\n=== (3) |Vp| ATTRIBUTION vs true vpm (who breaks the gate?) ===")
    lab = {"mm": "gamma_hat,dv_hat  (reality)", "tm": "gamma_TRUE,dv_hat (dV only)",
           "mt": "gamma_hat,dv_TRUE (gamma only)", "tt": "gamma_TRUE,dv_TRUE (floor)"}
    true_stick = float((t < GATE).mean())
    out["attribution"]["true_stick_frac"] = true_stick
    print(f"  {'variant':32} {'RMSE cm/s':>10} {'stick_frac':>11}")
    print(f"  {'TRUE label':32} {'-':>10} {true_stick:11.3f}")
    for k in ("mm", "tm", "mt", "tt"):
        e = a[k] - t
        rec = dict(vp_rmse_cm_s=_rmse(e) * 100,
                   stick_frac=float((a[k] < GATE).mean()),
                   by_regime={nm: _rmse((a[k] - t)[msk]) * 100
                              for nm, msk in bins if msk.sum()})
        out["attribution"][k] = rec
        print(f"  {lab[k]:32} {rec['vp_rmse_cm_s']:10.2f} {rec['stick_frac']:11.3f}")

    m = json.load(open(run / "metrics.json"))
    suffix = f"_lam{args.vy_label:g}" if _is_gammakin(m) else ""
    reg_tag = f"_{Path(args.regime).stem}" if args.regime else ""
    # `_vpc` keeps the relabelled scoring in its OWN file so the pre-existing
    # baseline JSONs (old label) are never overwritten.
    vpc_tag = "_vpc" if args.use_vp else ""
    out["use_vp_components"] = bool(args.use_vp)
    dst = run / f"regime_attrib{reg_tag}{suffix}{vpc_tag}.json"
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[regime-attrib] metrics -> {dst}")


if __name__ == "__main__":
    main()
