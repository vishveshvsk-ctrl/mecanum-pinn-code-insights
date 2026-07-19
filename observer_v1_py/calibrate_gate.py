#!/usr/bin/env python
# =============================================================================
# calibrate_gate.py — calibrate the hy3 slip-gate (gate_center/gate_width) from a
# GROUNDED checkpoint's model-derived contact slip.
#
# The physics-loss gate is g = sigmoid((Vp_mag - center)/width), where Vp_mag is
# the MODEL-derived contact speed  |contact_from_gamma(gamma_hat_phys, Vpx0_hat+dV, ...)|
# (exactly as in losses_v2hy3.physics_loss_hy3). With a well-grounded gamma_hat this
# Vp_mag should track the TRUE contact speed vpm = sqrt(Vpx_i^2+Vpy_i^2). We measure
# both, per (window,wheel), and recommend center/width so the gate's stick call
# (g<0.5  <=>  Vp_mag<center) matches the ground-truth stick label (vpm<v_stribeck).
#
#   python observer_v1_py/calibrate_gate.py --run-dir observer_v1_py/runs/S1_train_hy3_grnd
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather  # noqa: F401
import torch

from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer import config as C
from mecanum_observer import data as D
from mecanum_observer import data_v2hy3 as D2H
from mecanum_observer import physics_v2hy3 as P
from mecanum_observer.models_v2hy3 import build_model_v2hy3
from mecanum_observer.training_v2hy3 import _phys_batch, _resolve_precision

FIELDS = ObserverConfigV2Hy3.__dataclass_fields__
V_STRIBECK = C.LG_V_STR          # true stick/slip boundary (0.01 m/s)


@torch.no_grad()
def collect(run_dir: Path, ckpt_name: str, n_files: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = json.load(open(run_dir / "metrics.json"))
    cfg = ObserverConfigV2Hy3(**{k: v for k, v in m["cfg"].items() if k in FIELDS})
    cfg.jobs = 0
    cfg = cfg.resolved()
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")

    gstd = torch.tensor(nrm.y_std[0], dtype=torch.float32, device=device).view(1, 1)
    gmean = torch.tensor(nrm.y_mean[0], dtype=torch.float32, device=device).view(1, 1)
    dvsc = torch.tensor(cfg.dv_scale, dtype=torch.float32, device=device).view(1, 2)
    vhat = torch.tensor([nrm.g_std[0], nrm.g_std[1]], dtype=torch.float32,
                        device=device).view(1, 2)
    amp = _resolve_precision(cfg, device)

    model = build_model_v2hy3(cfg).to(device)
    ck = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()

    # representative sample: val + strided slice of test.
    files = D2H.discover_and_split(cfg)
    sample = list(files["val"])
    test = files["test"]
    sample += test[:: max(1, len(test) // n_files)]
    ds = D2H.WindowStreamV2Hy3(sample, nrm, cfg, shuffle=False)
    ld = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                                     pin_memory=(device.type == "cuda"))

    mvp_all, tvp_all = [], []
    for b in ld:
        Gw = b["Gw"].to(device); Pw = b["Pw"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            gh, dh = model(Gw, Pw)
        gh = gh.float(); dh = dh.float()
        phys = _phys_batch(b, device)
        gamma_phys = gh * gstd + gmean
        dv_phys = dh * dvsc
        # Vpx0_hat ALREADY carries V̂ -> add ONLY the ΔV̂ correction (mirrors
        # losses_v2hy3; adding V̂+ΔV̂ double-counts V̂).
        Vpx0_u = phys["Vpx0_hat"] + dv_phys[:, 0:1]
        Vpy0_u = phys["Vpy0_hat"] + dv_phys[:, 1:2]
        Vpx, Vpy, _, _, _, _ = P.contact_from_gamma(
            gamma_phys, phys["psi_dot"], Vpx0_u, Vpy0_u, phys["cti"], phys["sti"])
        vp_model = torch.sqrt(Vpx * Vpx + Vpy * Vpy + C.LG_EPS_REG ** 2)   # [B,4]
        mvp_all.append(vp_model.reshape(-1).cpu().numpy())
        tvp_all.append(b["slip_mag"].reshape(-1).numpy())
    return np.concatenate(mvp_all), np.concatenate(tvp_all), cfg


def main():
    ap = argparse.ArgumentParser(description="Calibrate the hy3 slip-gate.")
    ap.add_argument("--run-dir", default="observer_v1_py/runs/S1_train_hy3_grnd")
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--n-files", type=int, default=200)
    args = ap.parse_args()

    mvp, tvp, cfg = collect(Path(args.run_dir), args.checkpoint, args.n_files)
    n = mvp.size
    true_stick = tvp < V_STRIBECK
    f_stick = float(true_stick.mean())
    print(f"[calibrate] {args.run_dir}  ckpt={args.checkpoint}  n_samples={n:,}")
    print(f"[truth ] stick(vpm<{V_STRIBECK})={f_stick*100:.2f}%  "
          f"vpm: p50={np.percentile(tvp,50):.4f} p90={np.percentile(tvp,90):.4f}")

    qs = [50, 90, 99]
    print(f"[model ] Vp_mag overall: " +
          "  ".join(f"p{q}={np.percentile(mvp,q):.4f}" for q in qs) +
          f"  (current gate_center={cfg.gate_center})")
    print(f"[model ] Vp_mag | TRUE-stick: " +
          "  ".join(f"p{q}={np.percentile(mvp[true_stick],q):.4f}" for q in qs))
    print(f"[model ] Vp_mag | TRUE-slip : " +
          "  ".join(f"p{q}={np.percentile(mvp[~true_stick],q):.4f}" for q in qs))

    cur = cfg.gate_center
    print(f"[gate  ] current center={cur}: model calls "
          f"{float((mvp<cur).mean())*100:.2f}% stick  "
          f"(agreement w/ truth {float(((mvp<cur)==true_stick).mean())*100:.1f}%)")

    # (a) center that maximizes agreement with the true stick label.
    grid = np.unique(np.quantile(mvp, np.linspace(0.01, 0.6, 120)))
    agree = [float(((mvp < c) == true_stick).mean()) for c in grid]
    c_best = float(grid[int(np.argmax(agree))])
    # (b) center that reproduces the true stick fraction.
    c_match = float(np.quantile(mvp, f_stick))
    # width ~ half-decade of model Vp spread around the boundary (smooth transition).
    near = mvp[(mvp > c_best * 0.3) & (mvp < c_best * 3)]
    width = float(max(np.std(near) if near.size else c_best, c_best * 0.5))

    print(f"[reco  ] max-agreement center={c_best:.4f} "
          f"(agreement {max(agree)*100:.1f}%, stick_frac {float((mvp<c_best).mean())*100:.1f}%)")
    print(f"[reco  ] match-fraction center={c_match:.4f} "
          f"(reproduces true {f_stick*100:.1f}% stick)")
    print(f"[reco  ] suggested gate_center~{c_best:.3f}  gate_width~{width:.3f}  "
          f"(vs current {cfg.gate_center}/{cfg.gate_width})")
    print("done")


if __name__ == "__main__":
    main()
