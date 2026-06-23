#!/usr/bin/env python
# =============================================================================
# refine_physics_lbfgs.py — post-hoc L-BFGS refinement of a trained observer
# checkpoint using the PHYSICS loss (roller + wheel torque-balance residuals).
#
# NOTE (design caveat): the observer was trained SUPERVISED against ground-truth
# states. Refining on physics-ONLY (w_sup=0) drops that anchor; gamma is well
# constrained by torque balance, but the bristle states zx/zy are partly
# non-unique, so this can lower the residual while DRIFTING reconstruction. This
# script therefore (a) never touches the original checkpoint -- it writes a COPY,
# and (b) reports held-out per-state RMSE BEFORE vs AFTER so the effect is
# measured, not assumed.
#
# L-BFGS is full-batch (deterministic, shuffle off). The whole train fold is huge,
# so --train-files-cap bounds the objective to a representative subset (the physics
# residual is per-sample; a few hundred files is plenty for the ~6k-param shared
# encoder). Run from code_insights/ with the myenv python:
#   python observer_v1_py/refine_physics_lbfgs.py --run-dir observer_v1_py/runs/S1_train_w16
# =============================================================================
from __future__ import annotations

import pyarrow.feather  # noqa: F401  (import before torch — Windows load order)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mecanum_observer.config import ObserverConfig, TARGET_STATES, N_STATES
from mecanum_observer import data as D
from mecanum_observer.losses import observer_loss, physics_loss
from mecanum_observer.models import build_model
from mecanum_observer.training import WindowStream, _phys_batch


def _loader(files, nrm, cfg, device):
    ds = WindowStream(files, nrm, cfg, shuffle=False)        # deterministic for L-BFGS
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.jobs,
              drop_last=False, pin_memory=device.type == "cuda")
    if cfg.jobs > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


@torch.no_grad()
def per_state_rmse(model, loader, device):
    """Per-state normalised RMSE (z-scored units; comparable to the report)."""
    model.eval()
    acc = np.zeros(N_STATES); nb = 0
    for batch in loader:
        pred = model(batch["Gw"].to(device), batch["Pw"].to(device)).float()
        _, log = observer_loss(pred, batch["Yt"].to(device))
        acc += np.array([log[f"mse_{s}"] for s in TARGET_STATES]); nb += 1
    mse = acc / max(nb, 1)
    return {s: float(np.sqrt(mse[i])) for i, s in enumerate(TARGET_STATES)}


@torch.no_grad()
def mean_phys_resid(model, loader, device, y_mean_t, y_std_t):
    model.eval()
    tot = 0.0; nb = 0
    for batch in loader:
        pred = model(batch["Gw"].to(device), batch["Pw"].to(device)).float()
        l, _ = physics_loss(pred * y_std_t + y_mean_t, _phys_batch(batch, device))
        tot += float(l); nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser(description="Physics-only L-BFGS refinement of an observer checkpoint.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=25, help="L-BFGS max_iter")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--w-sup", type=float, default=0.0, help="supervised weight (0 = physics-only)")
    ap.add_argument("--train-files-cap", type=int, default=250, help="files used for the L-BFGS objective")
    ap.add_argument("--eval-cross", type=int, default=150, help="cross-subset (held-out fold) files to eval; 0=skip")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tag", type=str, default="", help="output suffix: checkpoint_physlbfgs<tag>.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    st = torch.load(args.run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = ObserverConfig(**st["cfg"]).resolved()
    cfg.physics_loss = True                       # build ph_* window fields + enable residuals
    cfg.batch_size = args.batch_size; cfg.jobs = args.jobs
    print(f"[refine] {args.run_dir.name}: window={cfg.window} regime={cfg.regime_name} "
          f"iters={args.iters} w_sup={args.w_sup} w_phys=1 cap={args.train_files_cap}")

    nrm = D.Normalizer.from_npz(args.run_dir / "norm.npz")
    y_mean_t = torch.tensor(nrm.y_mean, dtype=torch.float32, device=device).view(1, 1, -1)
    y_std_t = torch.tensor(nrm.y_std, dtype=torch.float32, device=device).view(1, 1, -1)

    files = D.discover(cfg); sp = D.split_files(files, cfg)
    train_fit = sp["train"][: args.train_files_cap] if args.train_files_cap > 0 else sp["train"]
    print(f"[refine] split train {len(sp['train'])} (using {len(train_fit)}) "
          f"val {len(sp['val'])} test {len(sp['test'])}")

    fit_loader = _loader(train_fit, nrm, cfg, device)
    val_loader = _loader(sp["val"], nrm, cfg, device)
    cross_loader = (_loader(sp["test"][: args.eval_cross], nrm, cfg, device)
                    if args.eval_cross > 0 and sp["test"] else None)

    model = build_model(cfg).to(device)
    model.load_state_dict(st["model"])
    print(f"[refine] loaded model params={sum(p.numel() for p in model.parameters()):,}")

    def snapshot(tag):
        r = dict(val=per_state_rmse(model, val_loader, device),
                 phys=mean_phys_resid(model, fit_loader, device, y_mean_t, y_std_t))
        if cross_loader is not None:
            r["cross"] = per_state_rmse(model, cross_loader, device)
        v = " ".join(f"{k}={r['val'][k]:.4f}" for k in TARGET_STATES)
        c = (" | cross " + " ".join(f"{k}={r['cross'][k]:.4f}" for k in TARGET_STATES)) if "cross" in r else ""
        print(f"[{tag}] val {v} | phys-resid {r['phys']:.5f}{c}")
        return r

    before = snapshot("before")

    opt = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=args.iters,
                            history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        model.train()
        tot = 0.0; nb = 0
        for batch in fit_loader:
            pred = model(batch["Gw"].to(device), batch["Pw"].to(device)).float()
            l_phys, _ = physics_loss(pred * y_std_t + y_mean_t, _phys_batch(batch, device))
            loss = l_phys
            if args.w_sup > 0:
                l_sup, _ = observer_loss(pred, batch["Yt"].to(device))
                loss = loss + args.w_sup * l_sup
            loss.backward()                       # accumulate over the full (capped) set
            tot += float(loss.detach()); nb += 1
        for p in model.parameters():              # mean gradient (len unknown for IterableDataset)
            if p.grad is not None:
                p.grad /= max(nb, 1)
        return torch.tensor(tot / max(nb, 1), device=device)

    print("[refine] running L-BFGS (physics-only)...")
    try:
        final = opt.step(closure)
        print(f"[refine] L-BFGS done, final objective {float(final):.6f}")
    except Exception as e:                         # pragma: no cover
        print(f"[refine] L-BFGS aborted: {e!r}")

    after = snapshot("after ")

    out = args.run_dir / f"checkpoint_physlbfgs{args.tag}.pt"
    torch.save(dict(model=model.state_dict(), cfg=st["cfg"], refine="physics_lbfgs",
                    w_sup=args.w_sup, iters=args.iters,
                    before=before, after=after), out)
    summary = dict(run=args.run_dir.name, w_sup=args.w_sup, iters=args.iters,
                   before=before, after=after)
    with open(args.run_dir / f"physlbfgs_summary{args.tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[refine] saved -> {out}  (original checkpoint.pt untouched)")
    print("=== before -> after  (lower = better; + = WORSE after refinement) ===")
    for s in TARGET_STATES:
        d = after["val"][s] - before["val"][s]
        print(f"  val   {s:<6} {before['val'][s]:.4f} -> {after['val'][s]:.4f}  ({'+' if d >= 0 else ''}{d:.4f})")
    if "cross" in before and "cross" in after:
        for s in TARGET_STATES:
            d = after["cross"][s] - before["cross"][s]
            print(f"  cross {s:<6} {before['cross'][s]:.4f} -> {after['cross'][s]:.4f}  ({'+' if d >= 0 else ''}{d:.4f})")
    print(f"  phys-resid {before['phys']:.5f} -> {after['phys']:.5f}")


if __name__ == "__main__":
    main()
