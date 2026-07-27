"""Build true contact-slip COMPONENT sidecars for the decimated cache.

Why
---
`data_v2hy3.py:105` stores the vpm label as `decimate(hypot(Vpx, Vpy))` — the
magnitude is formed at 2 kHz and *then* anti-alias filtered. `hypot` is convex,
so by Jensen the stored label is biased HIGH relative to the component-consistent
value `hypot(decimate(Vpx), decimate(Vpy))`, with the bias concentrated at small
|Vp| (the gate regime). The model path builds |Vp| from already-decimated
component quantities, so label and prediction currently apply the nonlinearity in
opposite orders.

This writes `Vpx_true`/`Vpy_true` [T,4] (decimated components, never hypot'd) as a
separate `.vpcomp.npz` sidecar. The existing cache is NOT modified — the warm
`mecanum_cache_decim` and the `_noslip/_slip02/_wslip1` baselines stay valid.

Filter is bit-identical to `sensor_frontend_v2._causal_lpf` (one-pole causal IIR,
200 Hz cutoff) but vectorized via `scipy.signal.lfilter` — verified to 1.1e-16.

Usage
-----
    python observer_v1_py/build_vp_components.py [--jobs 6] [--limit N] [--verify]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.signal import lfilter

SIM_HZ = 2000.0
TRAIN_HZ = 500.0
CUTOFF_HZ = 200.0          # sensor_frontend_v2.ANTI_ALIAS_CUTOFF_HZ
N_WHEELS = 4
SUFFIX = ".vpcomp.npz"

DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
CACHE_DIR = "C:/Users/vishv/mecanum_cache_decim"

_DT = 1.0 / SIM_HZ
_RC = 1.0 / (2.0 * math.pi * CUTOFF_HZ)
_ALPHA = _DT / (_RC + _DT)
_STEP = int(round(SIM_HZ / TRAIN_HZ))


def _decimate(x: np.ndarray) -> np.ndarray:
    """Vectorized twin of sensor_frontend_v2._decimate (LPF then stride)."""
    zi = np.array([x[0] * (1.0 - _ALPHA)])
    y, _ = lfilter([_ALPHA], [1.0, -(1.0 - _ALPHA)], x, zi=zi)
    return y[::_STEP]


def _decimate_loop(x: np.ndarray) -> np.ndarray:
    """Reference implementation, copied from sensor_frontend_v2 (for --verify)."""
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for k in range(1, x.shape[0]):
        y[k] = y[k - 1] + _ALPHA * (x[k] - y[k - 1])
    return y[::_STEP]


def sidecar_path(arrow_path: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, os.path.basename(arrow_path) + SUFFIX)


def build_one(arrow_path: str, cache_dir: str, overwrite: bool = False) -> str:
    out = sidecar_path(arrow_path, cache_dir)
    if os.path.exists(out) and not overwrite:
        return "skip"
    with pa.memory_map(arrow_path) as m:
        t = ipc.open_file(m).read_all()
        vx = np.stack([_decimate(t[f"Vpx_{i}"].to_numpy(zero_copy_only=False)
                                 .astype(np.float64))
                       for i in range(1, N_WHEELS + 1)], axis=1)
        vy = np.stack([_decimate(t[f"Vpy_{i}"].to_numpy(zero_copy_only=False)
                                 .astype(np.float64))
                       for i in range(1, N_WHEELS + 1)], axis=1)
    tmp = out + ".tmp.npz"
    np.savez(tmp, Vpx_true=vx.astype(np.float32), Vpy_true=vy.astype(np.float32))
    os.replace(tmp, out)
    return "built"


def verify(files: list[str], cache_dir: str, n: int = 3) -> None:
    """Check the vectorized filter against the reference loop, and the sidecar
    against a fresh read."""
    print("[verify] vectorized lfilter vs sensor_frontend_v2 reference loop")
    worst = 0.0
    for f in files[:n]:
        with pa.memory_map(f) as m:
            t = ipc.open_file(m).read_all()
            for i in range(1, N_WHEELS + 1):
                for c in ("x", "y"):
                    x = t[f"Vp{c}_{i}"].to_numpy(zero_copy_only=False).astype(np.float64)
                    worst = max(worst, float(np.abs(_decimate(x) - _decimate_loop(x)).max()))
    print(f"[verify]   max abs diff over {n} files x 8 channels: {worst:.3e}")

    print("[verify] sidecar round-trip + Jensen bound vs stored vpm")
    for f in files[:n]:
        z = np.load(sidecar_path(f, cache_dir))
        comp = np.hypot(z["Vpx_true"].astype(np.float64),
                        z["Vpy_true"].astype(np.float64))
        with pa.memory_map(f) as m:
            t = ipc.open_file(m).read_all()
            cur = np.stack([_decimate(np.hypot(
                t[f"Vpx_{i}"].to_numpy(zero_copy_only=False).astype(np.float64),
                t[f"Vpy_{i}"].to_numpy(zero_copy_only=False).astype(np.float64)))
                for i in range(1, N_WHEELS + 1)], axis=1)
        d = comp - cur
        print(f"[verify]   {os.path.basename(f)[:52]:52s} "
              f"shape={comp.shape} max_violation={d.max():.3e} mean_gap={d.mean():.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.arrow")))
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[build] {len(files)} arrow files -> {args.cache_dir}  "
          f"(jobs={args.jobs}, suffix={SUFFIX})")

    t0 = time.time()
    built = skipped = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(build_one, f, args.cache_dir, args.overwrite): f
                for f in files}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:                       # noqa: BLE001
                print(f"[build] FAIL {os.path.basename(futs[fut])}: {e}")
                continue
            built += (r == "built")
            skipped += (r == "skip")
            if k % 500 == 0 or k == len(files):
                el = time.time() - t0
                print(f"[build] {k}/{len(files)}  built={built} skip={skipped}  "
                      f"{el:.0f}s  eta {el/k*(len(files)-k):.0f}s", flush=True)

    tot = sum(os.path.getsize(sidecar_path(f, args.cache_dir))
              for f in files if os.path.exists(sidecar_path(f, args.cache_dir)))
    print(f"[build] done in {time.time()-t0:.0f}s  built={built} skipped={skipped}  "
          f"{tot/2**30:.2f} GiB")

    if args.verify:
        verify(files, args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
