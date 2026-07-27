#!/usr/bin/env python
# =============================================================================
# data_v2hy3.py — γ + ΔV window builder for Observer v2-Hy3.
#
# Reuses v1 Arrow discovery, grouped split, and cache mechanics.  Builds the
# sensor-real 5-channel Gw via sensor_frontend_v2.py and emits both the γ target
# and the ΔV = V_true − V̂ target.  The physics dict is GT-free-capable: it
# carries Vpx0_hat/Vpy0_hat (computed from the front-end V̂) and drops zx/zy.
# =============================================================================
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow.feather  # noqa: F401
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import config as C
from . import data as D
from . import sensor_frontend_v2 as SF
from .config_v2hy3 import ObserverConfigV2Hy3

# Default frozen-p95 scales for the two new Gw channels if the scaler CSV lacks
# them.  These are placeholders; a representative sample of the accel sidecars
# should be used to update variable_scaler_percentiles.csv with `a_x`/`a_y` rows.
_A_X_P95_DEFAULT = 5.0
_A_Y_P95_DEFAULT = 5.0

_SLIP_HIGH_THRESHOLD = 0.1


def _load_max_scaler_hy3(csv_path: str) -> D.Normalizer:
    """Load frozen-p95 scaler and extend it to 5 global channels (Vx,Vy ->
    V̂x,V̂y, plus a_x,a_y).  If the CSV already contains `a_x`/`a_y` rows they
    are used; otherwise the defaults above are used and a warning is printed."""
    nrm = D.load_max_scaler(csv_path)
    g = nrm.g_std  # [Vx, Vy, psi_dot]
    ax_p95 = _A_X_P95_DEFAULT
    ay_p95 = _A_Y_P95_DEFAULT

    try:
        import csv as _csv
        with open(csv_path) as fh:
            for row in _csv.DictReader(fh):
                if row["variable"] == "a_x":
                    ax_p95 = float(row["abs_p95"])
                elif row["variable"] == "a_y":
                    ay_p95 = float(row["abs_p95"])
    except Exception:
        pass
    if ax_p95 == _A_X_P95_DEFAULT or ay_p95 == _A_Y_P95_DEFAULT:
        print(f"[data-v2hy3] WARNING: a_x/a_y not in {csv_path}; using defaults "
              f"ax={ax_p95}, ay={ay_p95}. Update the CSV for production runs.")

    g5 = np.array([g[0], g[1], g[2], ax_p95, ay_p95], dtype=np.float64)
    # Return a new Normalizer with 5 global channels; y_mean/y_std are gamma-only.
    return D.Normalizer(np.zeros_like(g5), g5,
                        nrm.p_mean, nrm.p_std,
                        nrm.y_mean[:1], nrm.y_std[:1])


def _build_arrays_hy3(path: Path, cfg: ObserverConfigV2Hy3) -> Optional[Dict[str, np.ndarray]]:
    """Load original+sidecar, run sensor front-end, decimate per-wheel measurables
    and targets, and return the arrays that make_windows_hy3 consumes."""
    try:
        cols = SF.load_with_accel(path)
    except FileNotFoundError as e:
        print(f"[data-v2hy3] skipping {path.name}: {e}")
        return None

    spec = SF.SensorNoiseSpec(seed=hash(path.name) % (2 ** 31)) if cfg.noise_stage == "real" else None
    fe = SF.build_sensor_frontend(
        cols,
        crossover_hz=cfg.vel_filter_crossover_hz,
        integrator=cfg.vel_filter_integrator,
        target_hz=C.TRAIN_HZ,
        spec=spec,
    )

    T = fe["Gw"].shape[0]
    # Per-wheel measurable features [Msat, w, sin_tt, cos_tt] at 500 Hz.
    P = np.empty((T, C.N_WHEELS, C.N_PERWHEEL), dtype=np.float64)
    # Targets: gamma only [T,4,1]
    Y = np.empty((T, C.N_WHEELS, 1), dtype=np.float64)
    # Slip magnitude from true Vp (for supervised upweight).
    vpm = np.empty((T, C.N_WHEELS), dtype=np.float64)
    # Derived omega_z label for evaluation.
    wz = np.empty((T, C.N_WHEELS), dtype=np.float64)

    for k in range(C.N_WHEELS):
        i = k + 1
        # Decimate original columns to 500 Hz to match the front-end length.
        msat = SF._decimate(cols[f"Msat_{i}"], SF.SIM_HZ, C.TRAIN_HZ)
        w = fe["w"][:, k]
        P[:, k, 0] = msat
        P[:, k, 1] = w
        P[:, k, 2] = fe["sin_tt"][:, k]
        P[:, k, 3] = fe["cos_tt"][:, k]

        Y[:, k, 0] = SF._decimate(cols[f"gamma{i}"], SF.SIM_HZ, C.TRAIN_HZ)
        vpm[:, k] = SF._decimate(
            np.hypot(cols[f"Vpx_{i}"], cols[f"Vpy_{i}"]), SF.SIM_HZ, C.TRAIN_HZ)
        wz[:, k] = SF._decimate(cols[f"wz_{i}"], SF.SIM_HZ, C.TRAIN_HZ)

    return dict(
        G=fe["Gw"],           # [T,5] already at 500 Hz
        P=P,                  # [T,4,4]
        Y=Y,                  # [T,4,1]
        wz=wz,
        vpm=vpm,
        psid=np.abs(fe["psi_dot"]),
        Vpx0=fe["Vpx0_hat"],  # front-end base
        Vpy0=fe["Vpy0_hat"],
        V_true=fe["V_true"],  # [T,2] for ΔV target
        V_hat=fe["V_hat"],    # [T,2] for ΔV target
        mu=float(D._parse_name(path.name)["mu"]) if D._parse_name(path.name) else C.MU,
        chi=float(D._parse_name(path.name)["chi"]) if D._parse_name(path.name) else 0.0,
    )


# ---------------------------------------------------------------------------
# Decimated hy3 cache — memoises the full _build_arrays_hy3 output (which already
# carries the IMU accel inputs a_x/a_y in G[:,3:5] and NO wheel angular accel).
#
# This is a SEPARATE cache from the frozen observer `.arrow.npz` cache (whose
# format the v1/v2 brief pins as "reused as-is"). It coexists in the same
# cache_dir under a distinct `.hy3in.<key>.npz` name and stores the sensor-real
# front-end arrays (V̂, ψ̇, a_x, a_y, per-wheel measurables, γ target, physics
# block). Because the front-end output depends on the velocity-filter config and
# the noise stage, those knobs are folded into the cache KEY so a run can never
# read an entry built under a different front-end. One entry serves every window
# size / stride / regime (decimation + front-end are window-independent).
# ---------------------------------------------------------------------------
def _hy3_cache_key(cfg: ObserverConfigV2Hy3) -> str:
    """Front-end-identifying suffix; entries under different keys never collide."""
    xo = f"{cfg.vel_filter_crossover_hz:g}"
    hz = int(round(C.TRAIN_HZ))
    return f"hy3in.xo{xo}_{cfg.vel_filter_integrator}_noise-{cfg.noise_stage}_hz{hz}"


def _apply_vp_components(a: Optional[Dict[str, np.ndarray]], path: Path,
                         cfg: ObserverConfigV2Hy3) -> Optional[Dict[str, np.ndarray]]:
    """Replace the cached vpm with the component-consistent label, if enabled.

    The cache stores vpm = decimate(hypot(Vpx,Vpy)) -- magnitude formed at 2 kHz,
    THEN anti-alias filtered. hypot is convex, so by Jensen that is biased HIGH
    vs hypot(decimate(Vpx),decimate(Vpy)), which is what the model path builds
    (from already-decimated Vpx0_hat/Vpy0_hat). The bias is concentrated at small
    |Vp| -- i.e. at the gate.

    Applied AFTER the cache read on purpose: the `.hy3in.*.npz` cache is left
    byte-identical, so the same warm cache serves use_vp_components True and
    False and the _noslip/_slip02/_wslip1 baselines stay reproducible.

    Missing sidecar is FATAL rather than a silent fallback -- a half-relabelled
    epoch would be an invisible confound. Build with
    `python observer_v1_py/build_vp_components.py`.
    """
    if a is None or not getattr(cfg, "use_vp_components", False):
        return a
    sc = Path(getattr(cfg, "cache_dir", "") or ".") / f"{Path(path).name}.vpcomp.npz"
    if not sc.exists():
        raise FileNotFoundError(
            f"use_vp_components=True but sidecar missing: {sc}\n"
            f"Build it with: python observer_v1_py/build_vp_components.py")
    with np.load(sc) as z:
        vx = z["Vpx_true"].astype(np.float64)
        vy = z["Vpy_true"].astype(np.float64)
    vpm = np.hypot(vx, vy)
    if vpm.shape != a["vpm"].shape:
        raise ValueError(f"vpcomp shape {vpm.shape} != cached vpm "
                         f"{a['vpm'].shape} for {path.name}")
    a["vpm"] = vpm.astype(a["vpm"].dtype)
    return a


def read_arrays_hy3(path: Path, cfg: ObserverConfigV2Hy3
                    ) -> Optional[Dict[str, np.ndarray]]:
    """Decimated hy3 front-end arrays, memoised as float32 `.hy3in.<key>.npz`
    when cfg.cache_dir is set. Mirrors data.read_arrays: best-effort atomic
    per-PID tmp -> os.replace so concurrent warm/train workers can't read a
    half-written file, and any write failure (AV scan / cloud-sync lock) returns
    the computed arrays uncached rather than crashing training."""
    cache_dir = getattr(cfg, "cache_dir", "") or ""
    if not cache_dir:
        return _apply_vp_components(_build_arrays_hy3(path, cfg), path, cfg)

    cp = Path(cache_dir) / f"{Path(path).name}.{_hy3_cache_key(cfg)}.npz"
    if cp.exists():
        try:
            d = np.load(cp)
            a = {k: d[k] for k in d.files}
            a["mu"] = float(a["mu"])
            a["chi"] = float(a["chi"])
            return _apply_vp_components(a, path, cfg)
        except FileNotFoundError:
            raise                                           # missing sidecar -> fatal
        except Exception:
            pass                                            # corrupt/partial -> rebuild

    a = _build_arrays_hy3(path, cfg)
    if a is None:
        return None
    a = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
         for k, v in a.items()}
    tmp = None
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.parent / f"{cp.name}.{os.getpid()}.tmp.npz"   # unique per proc
        np.savez(tmp, **a)
        os.replace(tmp, cp)                                    # atomic on same FS
    except Exception:
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass
    return _apply_vp_components(a, path, cfg)


def warm_hy3_cache(files: List[Path], cfg: ObserverConfigV2Hy3) -> int:
    """Single-process pre-build of the hy3 decimated cache over `files`. Run ONCE
    before fanning out N parallel jobs so the workers never race to write the same
    `.hy3in.*.npz`. Pass the SUPERSET of every job's files (the launcher discovers
    over the full μ/χ grid); all later runs with the same front-end key hit warm."""
    cache_dir = getattr(cfg, "cache_dir", "") or ""
    if not cache_dir:
        print("[warm-cache-hy3] cache_dir empty -> nothing to do")
        return 0
    n = 0
    for j, p in enumerate(files):
        try:
            if read_arrays_hy3(p, cfg) is not None:
                n += 1
        except Exception as e:                              # keep going; log the bad file
            print(f"[warm-cache-hy3] {Path(p).name}: {e!r}")
        if (j + 1) % 200 == 0:
            print(f"[warm-cache-hy3] {j + 1}/{len(files)} files")
    print(f"[warm-cache-hy3] {n}/{len(files)} files -> {cache_dir} "
          f"(key {_hy3_cache_key(cfg)})")
    return n


def make_windows_hy3(a: Dict[str, np.ndarray], nrm: D.Normalizer,
                     cfg: ObserverConfigV2Hy3) -> Optional[Dict[str, np.ndarray]]:
    """Causal windows for Hy3.  Mirrors v1 make_windows but with 5 global
    channels and the front-end-based Vpx0_hat/Vpy0_hat physics block."""
    W, st = cfg.window, cfg.eff_stride
    T = a["G"].shape[0]
    if T < W:
        return None

    Gn = (a["G"] - nrm.g_mean) / nrm.g_std
    Pn = (a["P"] - nrm.p_mean) / nrm.p_std
    Yn = (a["Y"] - nrm.y_mean) / nrm.y_std

    ends = np.arange(W - 1, T - 1, st)
    starts = ends - (W - 1)
    idx = starts[:, None] + np.arange(W)[None, :]

    Gw = Gn[idx].astype(np.float32)
    Pw = Pn[idx].astype(np.float32)
    Yt = Yn[ends].astype(np.float32)

    sin_tt = a["P"][ends, :, 2]
    out = dict(
        Gw=Gw, Pw=Pw, Yt=Yt,
        wz=a["wz"][ends],
        sin_tt=sin_tt,
        psid=a["psid"][ends],
        vpm=a["vpm"][ends],
        V_true=a["V_true"][ends].astype(np.float32),
        V_hat=a["V_hat"][ends].astype(np.float32),
    )

    dt = C.DECIM / C.SIM_HZ
    Pe = a["P"][ends]
    w_dot = (a["P"][ends, :, 1] - a["P"][ends - 1, :, 1]) / dt
    M = ends.shape[0]

    out.update(
        ph_psi_dot=a["G"][ends, 2].astype(np.float32),
        ph_Vpx0=a["Vpx0"][ends].astype(np.float32),
        ph_Vpy0=a["Vpy0"][ends].astype(np.float32),
        ph_cti=Pe[:, :, 3].astype(np.float32),
        ph_sti=Pe[:, :, 2].astype(np.float32),
        ph_Msat=Pe[:, :, 0].astype(np.float32),
        ph_w=Pe[:, :, 1].astype(np.float32),
        ph_w_dot=w_dot.astype(np.float32),
        ph_mu=np.full(M, a["mu"], np.float32),
        ph_chi=np.full(M, a["chi"], np.float32),
    )
    return out


def _gamma_target(Yt: np.ndarray, nrm: D.Normalizer) -> np.ndarray:
    """Yt [M,4,1] normalized -> y_gamma [M,4]."""
    return Yt[:, :, 0].astype(np.float32)


def _dv_target(V_true: np.ndarray, V_hat: np.ndarray,
               dv_scale: tuple) -> np.ndarray:
    """ΔV* = (V_true − V̂) / dv_scale, normalized."""
    scale = np.array(dv_scale, dtype=np.float64)
    return ((V_true - V_hat) / scale[None, :]).astype(np.float32)


def _sup_weight(slip_mag: np.ndarray, upweight: float) -> np.ndarray:
    high = slip_mag.max(axis=-1) > _SLIP_HIGH_THRESHOLD
    return np.where(high, upweight, 1.0).astype(np.float32)


class WindowStreamV2Hy3(IterableDataset):
    """Streaming causal windows for v2-Hy3."""

    def __init__(self, files: List[Path], nrm: D.Normalizer, cfg: ObserverConfigV2Hy3,
                 shuffle: bool):
        super().__init__()
        self.files = list(files)
        self.nrm = nrm
        self.cfg = cfg
        self.shuffle = shuffle
        self.keys = ["Gw", "Pw", "Yt", "V_true", "V_hat"]

    def _my_files(self) -> List[Path]:
        wi = get_worker_info()
        files = self.files if wi is None else self.files[wi.id:: wi.num_workers]
        if self.shuffle:
            random.Random(self.cfg.seed + (0 if wi is None else wi.id)).shuffle(files)
        return files

    def _emit(self, item: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v) for k, v in item.items()}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf: List[Dict[str, np.ndarray]] = []
        cap = self.cfg.shuffle_buffer if self.shuffle else 1
        rng = random.Random(self.cfg.seed)

        for path in self._my_files():
            a = read_arrays_hy3(path, self.cfg)
            if a is None:
                continue
            w = make_windows_hy3(a, self.nrm, self.cfg)
            del a
            if w is None:
                continue
            M = w["Gw"].shape[0]
            supw = _sup_weight(w["vpm"], self.cfg.gamma_high_slip_upweight)
            y_dv = _dv_target(w["V_true"], w["V_hat"], self.cfg.dv_scale)
            for j in range(M):
                item = {
                    "Gw": w["Gw"][j],
                    "Pw": w["Pw"][j],
                    "y_gamma": _gamma_target(w["Yt"][j:j + 1], self.nrm)[0],
                    "y_dv": y_dv[j],
                    "slip_mag": w["vpm"][j].astype(np.float32),
                    "sup_weight": supw[j],
                    "ph_psi_dot": w["ph_psi_dot"][j],
                    "ph_Vpx0": w["ph_Vpx0"][j],
                    "ph_Vpy0": w["ph_Vpy0"][j],
                    "ph_cti": w["ph_cti"][j],
                    "ph_sti": w["ph_sti"][j],
                    "ph_Msat": w["ph_Msat"][j],
                    "ph_w": w["ph_w"][j],
                    "ph_w_dot": w["ph_w_dot"][j],
                    "ph_mu": w["ph_mu"][j],
                    "ph_chi": w["ph_chi"][j],
                }
                if not self.shuffle:
                    yield self._emit(item)
                    continue
                buf.append(item)
                if len(buf) >= cap:
                    k = rng.randrange(len(buf))
                    out = buf[k]
                    buf[k] = buf[-1]
                    buf.pop()
                    yield self._emit(out)
            del w
        rng.shuffle(buf)
        for out in buf:
            yield self._emit(out)


def make_loaders(files: List[Path], nrm: D.Normalizer, cfg: ObserverConfigV2Hy3):
    """Train/val DataLoaders for v2-Hy3."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    def _loader(files_, shuffle):
        ds = WindowStreamV2Hy3(files_, nrm, cfg, shuffle)
        kw = dict(batch_size=cfg.batch_size, num_workers=cfg.jobs,
                  drop_last=shuffle, pin_memory=pin)
        if cfg.jobs > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        return torch.utils.data.DataLoader(ds, **kw)

    return _loader(files["train"], True), _loader(files["val"], False)


def discover_and_split(cfg: ObserverConfigV2Hy3):
    """Re-use v1 discovery + split."""
    files = D.discover(cfg)
    return D.split_files(files, cfg)


def load_max_scaler(csv_path: str) -> D.Normalizer:
    """v2-Hy3 wrapper around the frozen-p95 scaler (5 global channels)."""
    return _load_max_scaler_hy3(csv_path)
