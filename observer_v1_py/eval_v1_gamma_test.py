#!/usr/bin/env python
# eval_v2_gamma.py — physical gamma RMSE for a v2 (gamma-only, TRUE-input) run on its
# cross-fold TEST split, comparable to the v2hy3/gammakin regime_attrib numbers.
# The stored metrics.json val_loss = 0.1*sup + phys (unusable for gamma), so evaluate.
from __future__ import annotations
import pyarrow.feather  # noqa: F401  (before torch on Windows)
import json, sys
from pathlib import Path
import numpy as np, torch

OBS = Path(r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\observer_v1_py")
sys.path.insert(0, str(OBS))
from mecanum_observer.config import ObserverConfig          # v1 model + data config
from mecanum_observer import data as D                       # v1-native read + windows
from mecanum_observer.models import build_model              # v1 3-state model

FK1 = ObserverConfig.__dataclass_fields__


@torch.no_grad()
def evalrun(run_dir: Path, n_files: int = 60):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run_dir / "checkpoint.pt", map_location=dev, weights_only=False)
    cfg = ObserverConfig(**{k: v for k, v in ck["cfg"].items() if k in FK1}).resolved()
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    nrm = D.Normalizer.from_npz(run_dir / "norm.npz")
    gp95 = float(nrm.y_std[0])
    cache = getattr(cfg, "cache_dir", "") or ""

    sp = json.load(open(run_dir / "split.json"))
    ddir = Path(cfg.data_dir)
    files = [ddir / n for n in sp["test"]]
    files = files[:: max(1, len(files) // n_files)][:n_files]

    sse = n = 0.0
    for p in files:
        a = D.read_arrays(p, cache)                 # v1-native per-file arrays
        if a is None:
            continue
        w = D.make_windows(a, nrm, cfg)             # Gw[M,W,3] Pw[M,W,4,6] Yt[M,4,4]
        if w is None:
            continue
        Gw = torch.as_tensor(w["Gw"]); Pw = torch.as_tensor(w["Pw"])
        Yt = torch.as_tensor(w["Yt"])[:, :, 0]      # gamma target (normalized)
        for i in range(0, Gw.shape[0], 4096):
            g = Gw[i:i+4096].to(dev); pw = Pw[i:i+4096].to(dev)
            pred = model(g, pw).float()[:, :, 0]    # gamma head [b,4]
            yy = Yt[i:i+4096].to(dev)
            sse += float(((pred - yy) ** 2).sum()); n += yy.numel()
    return float(np.sqrt(sse / max(n, 1)) * gp95), int(n // 4)


if __name__ == "__main__":
    runs = ["S1_train_w32_non_phys_max_norm", "S2_train_w32_non_phys_max_norm",
            "S1_train_w32_non_phys_max_norm_b1024", "S2_train_w32_non_phys_max_norm_b1024",
            "S1_train_w32_non_phys_max_norm_ep180", "S2_train_w32_non_phys_max_norm_ep180"]
    print(f"{'run':42} {'gamma RMSE rad/s (test)':>24} {'n_win':>9}")
    best = (None, 1e9)
    for r in runs:
        try:
            rmse, nw = evalrun(OBS / "runs" / r)
            print(f"{r:42} {rmse:24.3f} {nw:>9,}")
            if rmse < best[1]:
                best = (r, rmse)
        except Exception as e:
            print(f"{r:42} ERROR {e!r}")
    print(f"\nbest v2 (gamma-only, TRUE-input) test gamma RMSE: {best[1]:.3f} rad/s ({best[0]})")
