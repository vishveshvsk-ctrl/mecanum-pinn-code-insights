#!/usr/bin/env python
# =============================================================================
# data_v2.py — γ-only window builder for Observer v2.
#
# Thin wrapper around the v1 data pipeline: Arrow discovery, grouped split,
# decimated cache, and max-normalisation are reused unchanged. v2 only changes
# the emitted target (γ instead of γ/zx/zy) and augments the physics dict with
# label-sourced zx/zy (physical units) plus a high-slip supervised upweight.
# =============================================================================
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow.feather  # noqa: F401  (before torch, per CLAUDE.md)
import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import config as C
from . import data as D
from .config_v2 import ObserverConfigV2

# Physics keys expected from v1 make_windows when physics_loss=True.
_PHYS_KEYS = [
    "ph_psi_dot", "ph_Vpx0", "ph_Vpy0", "ph_cti", "ph_sti",
    "ph_Msat", "ph_w", "ph_w_dot", "ph_mu", "ph_chi",
    "ph_Vx", "ph_Vy", "ph_dVx", "ph_dVy", "ph_dpsi_dot",
    "ph_Vx_next", "ph_Vy_next", "ph_psi_dot_next", "ph_w_next",
]

# Slip threshold (m/s) above which a window is considered high-slip and receives
# the supervised upweight.  Same order of magnitude as the 0.1 m/s slip bins.
_SLIP_HIGH_THRESHOLD = 0.1


def _gamma_only_target(Yt: np.ndarray, nrm: D.Normalizer) -> np.ndarray:
    """Yt [M,4,4] normalised (v1 layout: gamma,zx,zy,zs_dummy) -> y [M,4] gamma."""
    return Yt[:, :, 0].astype(np.float32)


def _label_bristles(Yt: np.ndarray, nrm: D.Normalizer) -> Dict[str, np.ndarray]:
    """Return physical-unit zx_lab, zy_lab from the normalised v1 target block.

    v1 Yt order: [gamma, zx, zy, zs_dummy].  Indices 1,2 are the linear bristles.
    """
    y_mean = nrm.y_mean.reshape(1, 1, -1)
    y_std = nrm.y_std.reshape(1, 1, -1)
    Yphys = Yt * y_std + y_mean
    return {
        "zx_lab": Yphys[:, :, 1].astype(np.float32),
        "zy_lab": Yphys[:, :, 2].astype(np.float32),
    }


def _sup_weight(slip_mag: np.ndarray, upweight: float) -> np.ndarray:
    """Per-window scalar supervised weight.  High-slip windows (max wheel slip
    > _SLIP_HIGH_THRESHOLD) receive `upweight`; otherwise 1.0."""
    high = slip_mag.max(axis=-1) > _SLIP_HIGH_THRESHOLD
    return np.where(high, upweight, 1.0).astype(np.float32)


class WindowStreamV2(IterableDataset):
    """Streaming causal windows for v2.  One Arrow file at a time."""

    def __init__(self, files: List[Path], nrm: D.Normalizer, cfg: ObserverConfigV2,
                 shuffle: bool):
        super().__init__()
        self.files = list(files)
        self.nrm = nrm
        self.cfg = cfg
        self.shuffle = shuffle
        self.keys = ["Gw", "Pw", "Yt"] + _PHYS_KEYS

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
            a = D.read_arrays(path, self.cfg.cache_dir)
            # v1 make_windows requires physics_loss=True so the window-end trim and
            # the ph_* keys are produced.
            w = D.make_windows(a, self.nrm, self.cfg)
            del a
            if w is None:
                continue
            M = w["Gw"].shape[0]
            bristles = _label_bristles(w["Yt"], self.nrm)
            supw = _sup_weight(w["vpm"], self.cfg.gamma_high_slip_upweight)
            for j in range(M):
                item = {
                    "Gw": w["Gw"][j],
                    "Pw": w["Pw"][j],
                    "y": _gamma_only_target(w["Yt"][j:j + 1], self.nrm)[0],
                    "slip_mag": w["vpm"][j].astype(np.float32),
                    "sup_weight": supw[j],
                    **{k: w[k][j].astype(np.float32) for k in _PHYS_KEYS},
                    **{k: bristles[k][j] for k in ("zx_lab", "zy_lab")},
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


def make_loaders(files: List[Path], nrm: D.Normalizer, cfg: ObserverConfigV2):
    """Train/val DataLoaders for v2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    def _loader(files_, shuffle):
        ds = WindowStreamV2(files_, nrm, cfg, shuffle)
        kw = dict(batch_size=cfg.batch_size, num_workers=cfg.jobs,
                  drop_last=shuffle, pin_memory=pin)
        if cfg.jobs > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        return torch.utils.data.DataLoader(ds, **kw)

    return _loader(files["train"], True), _loader(files["val"], False)


def discover_and_split(cfg: ObserverConfigV2) -> Dict[str, List[Path]]:
    """Re-use v1 discovery + split with a v2 config (duck-typed where needed)."""
    files = D.discover(cfg)
    return D.split_files(files, cfg)


def load_max_scaler(csv_path: str) -> D.Normalizer:
    """v2 wrapper around the v1 frozen-p95 scaler."""
    return D.load_max_scaler(csv_path)
