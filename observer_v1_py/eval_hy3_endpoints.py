#!/usr/bin/env python
# =============================================================================
# eval_hy3_endpoints.py — score a trained Observer v2-Hy3 checkpoint on the TWO
# loss endpoints, with de-normalized (physical) gamma RMSE, grouped as requested.
#
#   endpoint 1  GROUNDING-only (supervised): L_sup = MSE(gamma;sup_w) + w_dv*MSE(dV)
#   endpoint 2  GROUND+PHYSICS (terminal)  : 0.1*L_sup + 1.0*L_phys  (== training's
#                                            terminal_val_loss)
#
# Physical gamma error is reported too: gamma_rmse_phys = gamma_rmse_norm * gamma_p95
# (max-norm p95 scale from the run's norm.npz), i.e. RMSE as a fraction of the 95th
# -percentile |gamma| when normalized.
#
# Modes:
#   --mode by-chi   : re-split each run over the FULL chi grid, take the cross-test
#                     fold, group by chi (chi-wise distribution; Q2).
#   --mode regime   : discover the files a regime TOML selects (e.g. eval_multisine),
#                     evaluate ALL of them, optionally grouped by profile (Q3).
#
# Run from code_insights/ with the conda myenv python. Uses the warm hy3 cache.
#   python observer_v1_py/eval_hy3_endpoints.py --mode by-chi
#   python observer_v1_py/eval_hy3_endpoints.py --mode regime \
#          --regime observer_v1_py/regimes/eval_multisine.toml --group-by profile
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.feather  # noqa: F401  (import before torch on Windows)
import torch

from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer.config_v2 import W_SUP_MIN
from mecanum_observer import config as C
from mecanum_observer import data as D
from mecanum_observer import data_v2hy3 as D2H
from mecanum_observer.models_v2hy3 import build_model_v2hy3
from mecanum_observer.losses_v2hy3 import supervised_loss, physics_loss_hy3
from mecanum_observer.training_v2hy3 import _phys_batch, _resolve_precision

RUNS_DEFAULT = ["observer_v1_py/runs/S1_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm",
                "observer_v1_py/runs/S2_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm"]
FIELDS = ObserverConfigV2Hy3.__dataclass_fields__


def _cfg_from_run(run_dir: Path) -> ObserverConfigV2Hy3:
    m = json.load(open(run_dir / "metrics.json"))
    cfg = ObserverConfigV2Hy3(**{k: v for k, v in m["cfg"].items() if k in FIELDS})
    cfg.jobs = 0
    return cfg.resolved()


@torch.no_grad()
def eval_files(model, files, nrm, cfg, device, amp, gstd, gmean, dvsc, gamma_p95):
    """Single pass -> both endpoints + physical gamma/dV RMSE + gate stick_frac."""
    if not files:
        return None
    ds = D2H.WindowStreamV2Hy3(files, nrm, cfg, shuffle=False)
    ld = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                                     drop_last=False, pin_memory=(device.type == "cuda"))
    sup_sum = phys_sum = stick_sum = 0.0
    nb = 0
    g_sse = dv_sse = 0.0
    g_n = dv_n = 0
    for b in ld:
        Gw = b["Gw"].to(device); Pw = b["Pw"].to(device)
        yg = b["y_gamma"].to(device); yd = b["y_dv"].to(device)
        sw = b["sup_weight"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            gh, dh = model(Gw, Pw)
        gh = gh.float(); dh = dh.float()
        l_sup, _ = supervised_loss(gh, dh, yg, yd, sw, cfg.w_dv)
        Vh = Gw[:, -1, :2] * dvsc
        # NORMALIZED gamma: physics_loss_hy3 de-normalizes in-graph (passing gh*gstd
        # here double-scales gamma by gamma_p95 and pins the gate at g=1).
        l_phys, plog = physics_loss_hy3(gh, dh, Vh, _phys_batch(b, device), cfg, gstd, gmean)
        sup_sum += float(l_sup); phys_sum += float(l_phys)
        stick_sum += plog.get("g_stick_frac", 0.0); nb += 1
        g_sse += float(((gh - yg) ** 2).sum()); g_n += yg.numel()
        dv_sse += float(((dh - yd) ** 2).sum()); dv_n += yd.numel()
    if nb == 0:
        return None
    sup = sup_sum / nb; phys = phys_sum / nb
    g_rmse_n = (g_sse / max(g_n, 1)) ** 0.5
    return dict(
        n_win=g_n // C.N_WHEELS,
        grnd=sup,
        terminal=W_SUP_MIN * sup + phys,
        phys=phys,
        gamma_rmse_norm=g_rmse_n,
        gamma_rmse_phys=g_rmse_n * gamma_p95,
        dv_rmse_norm=(dv_sse / max(dv_n, 1)) ** 0.5,
        stick_frac=stick_sum / nb,
    )


def _strided(files, cap):
    if cap and len(files) > cap:
        return files[:: max(1, len(files) // cap)]
    return files


def _row(tag, r):
    if r is None:
        print(f"  {tag:<22} (no windows)"); return
    print(f"  {tag:<22} n={r['n_win']:>7} | grnd={r['grnd']:.4f}  "
          f"terminal={r['terminal']:.4f} (phys={r['phys']:.4f}) | "
          f"gamma_rmse: norm={r['gamma_rmse_norm']:.4f} phys={r['gamma_rmse_phys']:.3f} "
          f"| dV_rmse_norm={r['dv_rmse_norm']:.4f} | stick_frac={r['stick_frac']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Hy3 two-endpoint evaluator.")
    ap.add_argument("--runs", nargs="+", default=RUNS_DEFAULT)
    ap.add_argument("--checkpoint", default="checkpoint_best.pt",
                    help="ckpt file under run dir, e.g. checkpoint_best.pt or "
                         "phase_ckpts/physics_ep199.pt")
    ap.add_argument("--mode", choices=["by-chi", "regime"], required=True)
    ap.add_argument("--regime", default=None, help="regime TOML (mode=regime)")
    ap.add_argument("--group-by", choices=["none", "profile"], default="none",
                    help="mode=regime grouping")
    ap.add_argument("--per-group-cap", type=int, default=250,
                    help="strided subsample cap per group (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        cfg = _cfg_from_run(run_dir)
        nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
        gamma_p95 = float(nrm.y_std[0])
        gstd = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=device).view(1, 1)
        gmean = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=device).view(1, 1)
        dvsc = torch.tensor(cfg.dv_scale, dtype=torch.float32, device=device).view(1, 2)
        amp = _resolve_precision(cfg, device)
        model = build_model_v2hy3(cfg).to(device)
        ck = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()

        fold = cfg.train_fold or run_dir.name[:2]
        print(f"\n########## {run_dir.name}  ckpt={args.checkpoint}  "
              f"(gamma_p95={gamma_p95:.2f}) ##########")

        if args.mode == "by-chi":
            cfg.chi_values = list(C.CHI_GRID)               # full grid
            splits = D.split_files(D.discover(cfg), cfg)
            test = splits["test"]                           # cross-subset fold, all chi
            by = {}
            for p in test:
                m = D._parse_name(p.name)
                if m:
                    by.setdefault(round(m["chi"], 3), []).append(p)
            cross = "S2" if fold == "S1" else "S1"
            print(f"[by-chi] cross-test fold={cross}; chi bins {sorted(by)}")
            for chi in sorted(by):
                fl = _strided(by[chi], args.per_group_cap)
                r = eval_files(model, fl, nrm, cfg, device, amp, gstd, gmean, dvsc, gamma_p95)
                _row(f"chi={chi:<6} ({len(fl)}/{len(by[chi])}f)", r)

        else:  # regime mode
            reg = D.regime_to_kwargs(D.load_regime(Path(args.regime)))
            for k, v in reg.items():
                if k in FIELDS:
                    setattr(cfg, k, v)
            cfg.train_fold = ""                             # no fold split for eval
            files = D.discover(cfg)
            print(f"[regime] {args.regime} -> {len(files)} files "
                  f"(profiles={cfg.include_profiles} chi={cfg.chi_values})")
            groups = {"all": files}
            if args.group_by == "profile":
                groups = {}
                for p in files:
                    m = D._parse_name(p.name)
                    groups.setdefault(m["profile"] if m else "??", []).append(p)
            for gk in sorted(groups):
                fl = _strided(groups[gk], args.per_group_cap)
                r = eval_files(model, fl, nrm, cfg, device, amp, gstd, gmean, dvsc, gamma_p95)
                _row(f"{gk} ({len(fl)}/{len(groups[gk])}f)", r)

    print("\ndone")


if __name__ == "__main__":
    main()
