#!/usr/bin/env python
# =============================================================================
# data.py — Arrow discovery, whitelist gating, grouped split, streaming windows.
#
# Memory-safe by construction: never holds the whole dataset. Files are streamed
# one at a time (read -> build windows -> yield -> drop -> advance). Normalisation
# statistics are accumulated with streaming sufficient stats over the training
# files only. Causal windows: a length-W window ending at t predicts the hidden
# state AT t (past-only -> deployable filter, not a smoother).
# =============================================================================
from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# Windows/WSL gotcha: import pyarrow.feather before torch (CLAUDE.md §7).
import pyarrow.feather as feather

try:                                  # tomllib is stdlib on 3.11+; fall back to tomli
    import tomllib as _toml
except ModuleNotFoundError:           # pragma: no cover
    import tomli as _toml

from . import config as C
from . import features as F


# ---------------------------------------------------------------------------
# Regime TOML loader  (mirrors the data-generation base.toml + per-profile TOML
# pattern: regimes/base.toml holds shared defaults, each regime overrides it)
# ---------------------------------------------------------------------------
def load_regime(regime_path, base_path=None) -> Dict[str, Any]:
    """Read base.toml + a regime TOML, regime overriding base (per-table merge)."""
    regime_path = Path(regime_path)
    if base_path is None:
        base_path = regime_path.parent / "base.toml"
    merged: Dict[str, Any] = {}
    for p in (Path(base_path), regime_path):
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            d = _toml.load(fh)
        for table, body in d.items():
            if isinstance(body, dict):
                merged.setdefault(table, {}).update(body)
            else:
                merged[table] = body
    return merged


def regime_to_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Map a merged regime dict to ObserverConfig kwargs (only set keys present)."""
    data = d.get("data", {}); split = d.get("split", {}); train = d.get("train", {})
    kw: Dict[str, Any] = {"regime_name": d.get("regime", {}).get("name", "default")}
    _copy = {
        "include_profiles": ("data", "include_profiles", list),
        "exclude_profiles": ("data", "exclude_profiles", list),
        "mu_values": ("data", "mu", lambda v: [float(x) for x in (v if isinstance(v, list) else [v])]),
        "chi_values": ("data", "chi", lambda v: [float(x) for x in (v if isinstance(v, list) else [v])]),
        "per_profile_cap": ("data", "per_profile_cap", int),
        "subsample_fraction": ("data", "subsample_fraction", float),
        "chi_stratify": ("data", "chi_stratify", bool),
        "matched_chi_quads_only": ("data", "matched_chi_quads_only", bool),
        "min_chi_per_combo": ("data", "min_chi_per_combo", int),
        "val_frac": ("split", "val_frac", float),
        "test_frac": ("split", "test_frac", float),
        "seed": ("split", "seed", int),
        "window": ("train", "window", int),
        "phases": ("train", "phases", str),
        "physics_loss": ("train", "physics_loss", bool),
        "train_fold": ("fold", "train_fold", str),
        "backbone_profiles": ("fold", "backbone", list),
        "redundant_S1": ("fold", "redundant_S1", list),
        "redundant_S2": ("fold", "redundant_S2", list),
        "profiles_toml_dir": ("fold", "profiles_toml_dir", str),
        "redundant_sample_frac": ("fold", "redundant_sample_frac", float),
        "redundant_sample_profiles": ("fold", "redundant_sample_profiles", list),
        "chi_fold_test": ("fold", "chi_fold_test", float),
    }
    src = {"data": data, "split": split, "train": train, "fold": d.get("fold", {})}
    for field_name, (table, key, cast) in _copy.items():
        if key in src[table]:
            kw[field_name] = cast(src[table][key])
    return kw


# ---------------------------------------------------------------------------
# Discovery + whitelist
# ---------------------------------------------------------------------------
def _parse_name(name: str) -> Optional[dict]:
    """Local copy of the filename contract (CLAUDE.md §5) to avoid importing
    chatter_diagnostics (keeps the package self-contained)."""
    import re
    m = re.match(
        r"^(?P<profile>.+?)_c(?P<combo>\d{3})_mu_(?P<mu>[0-9.eE+\-]+)"
        r"_case(?P<fc>\d+)_(?P<fm>lugre_adamov|lugre_uncoupled)"
        r"_chi_(?P<chi>[0-9.]+)\.arrow$", name)
    if m is None:
        return None
    return dict(profile=m["profile"], combo=int(m["combo"]),
                mu=float(m["mu"]), fm=m["fm"], chi=float(m["chi"]))


def load_whitelist(csv: Path) -> Optional[set]:
    """Return the set of accepted Arrow filenames (combined_reco not 'reject*'),
    or None if the file/column is absent (=> accept everything, with a warning)."""
    if not Path(csv).exists():
        print(f"[data] WARNING: whitelist {csv} missing -> accepting all files")
        return None
    wl = pd.read_csv(csv, usecols=lambda c: c in ("file", "combined_reco"))
    if "combined_reco" not in wl.columns or "file" not in wl.columns:
        print("[data] WARNING: whitelist lacks file/combined_reco -> accept all")
        return None
    keep = wl[~wl["combined_reco"].astype(str).str.startswith("reject")]
    return set(keep["file"].astype(str))


def discover(cfg: ObserverConfigT) -> List[Path]:
    """Gate by μ / χ-grid / whitelist / profile include-exclude, then apply the
    regime sampler (per-profile cap, χ-quad filter, stratified subsample)."""
    wl = load_whitelist(cfg.whitelist_csv)
    inc = set(cfg.include_profiles)
    exc = set(cfg.exclude_profiles)
    out = []
    for p in sorted(Path(cfg.data_dir).glob("*.arrow")):
        meta = _parse_name(p.name)
        if meta is None:
            continue
        if not any(abs(meta["mu"] - mv) < 1e-9 for mv in cfg.mu_values):
            continue
        if not any(abs(meta["chi"] - g) < C.CHI_TOL for g in cfg.chi_values):
            continue
        if inc and meta["profile"] not in inc:
            continue
        if meta["profile"] in exc:
            continue
        if wl is not None and p.name not in wl:
            continue
        out.append(p)
    out = select_regime(out, cfg)
    if cfg.limit_files > 0:
        out = out[: cfg.limit_files]
    return out


def select_regime(paths: List[Path], cfg: ObserverConfigT) -> List[Path]:
    """Deterministic regime sampling at the (profile, combo) group level so χ-
    siblings always stay together (no train/test leakage, no split bias).

    Steps per profile:
      1. group files by (profile, combo); each group = that combo's χ-siblings
      2. (matched_chi_quads_only) drop groups spanning < min_chi_per_combo χ
      3. order groups: χ-rich first when chi_stratify, else hash-shuffled
      4. keep per_profile_cap groups (0 = all)
      5. keep ceil(subsample_fraction · n) of what remains
    Pure on filenames — unit-testable without disk access."""
    groups: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    for p in paths:
        m = _parse_name(p.name)
        if m is None:
            continue
        groups[(m["profile"], m["combo"])].append(p)

    by_profile: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for key in groups:
        by_profile[key[0]].append(key)

    kept: List[Path] = []
    for profile, keys in by_profile.items():
        # χ count per group (distinct χ among its files)
        def n_chi(k):
            return len({_parse_name(p.name)["chi"] for p in groups[k]})
        if cfg.matched_chi_quads_only:
            keys = [k for k in keys if n_chi(k) >= cfg.min_chi_per_combo]
        if not keys:
            continue
        # deterministic order: χ-rich first (stratify) then by stable hash
        keys.sort(key=lambda k: (-(n_chi(k) if cfg.chi_stratify else 0),
                                 _bucket(f"{k[0]}_c{k[1]:03d}", cfg.seed)))
        if cfg.per_profile_cap > 0:
            keys = keys[: cfg.per_profile_cap]
        # per-profile fraction: targeted downsample for over-represented redundant
        # profiles (coverage-neutral), else the global learning-curve fraction.
        frac = (cfg.redundant_sample_frac if profile in cfg.redundant_sample_profiles
                else cfg.subsample_fraction)
        if frac < 1.0:
            n = max(1, math.ceil(frac * len(keys)))
            keys = keys[:n]
        for k in keys:
            kept.extend(groups[k])
    return sorted(kept)


# ---------------------------------------------------------------------------
# Grouped split (by profile+combo so a trajectory and all its chi-siblings stay
# together -> no leakage between train/val/test)
# ---------------------------------------------------------------------------
def _group_key(name: str) -> str:
    m = _parse_name(name)
    return f"{m['profile']}_c{m['combo']:03d}" if m else name


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF       # deterministic [0,1)


# ---------------------------------------------------------------------------
# Excitation-coverage 2-fold partition (S1/S2)
# ---------------------------------------------------------------------------
def _combo_mode_map(profile: str, toml_dir: Path) -> Dict[int, str]:
    """combo_idx (1-based, as in the filename) -> excitation-mode key, read from
    the per-profile combo TOML. Backbone profiles only: the key groups combos
    that feed the same excitation cells, so the 50/50 split can be stratified to
    keep both folds excitation-complete."""
    cand = (sorted(Path(toml_dir).glob(f"{profile}_mu_*.toml"))
            or sorted(Path(toml_dir).glob(f"{profile}.toml")))
    if not cand:
        return {}
    with open(cand[0], "rb") as fh:
        combos = _toml.load(fh).get("profile", {}).get("combos", {})
    if not combos:
        return {}
    n = len(next(iter(combos.values())))

    def _dclass(deg):
        d = abs(float(deg)) % 180.0
        return "x" if d < 1e-6 else ("y" if abs(d - 90.0) < 1e-6 else "diag")

    out: Dict[int, str] = {}
    for row in range(n):
        if profile == "octagon":
            key = (f"th{int(round(combos['theta0_deg'][row]))}"
                   f"_lat{int(combos['lat_vamp'][row] > 1e-9)}")
        elif profile == "spin_creep":
            key = (f"d{int(round(combos['delta_creep_deg'][row]))}"
                   f"_v{int(abs(combos['v_creep'][row]) >= 0.05)}")
        elif profile == "coupled_vomega":
            key = (f"{combos['V_mode'][row]}_{combos['Om_mode'][row]}"
                   f"_{_dclass(combos['delta_deg'][row])}")
        else:
            key = "all"
        out[row + 1] = key                       # filename c001 -> TOML row 0
    return out


def assign_folds(files: List[Path], cfg: ObserverConfigT) -> Dict[str, str]:
    """file -> "S1"/"S2". (profile,combo) groups (incl. all μ siblings) move
    together. Backbone: stratified 50/50 per (profile, mode). Redundant: wholesale."""
    backbone = set(cfg.backbone_profiles)
    r1, r2 = set(cfg.redundant_S1), set(cfg.redundant_S2)
    mode_maps = {p: _combo_mode_map(p, Path(cfg.profiles_toml_dir)) for p in backbone}

    combos_seen = {(m["profile"], m["combo"])
                   for p in files for m in [_parse_name(p.name)] if m}
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    fold_of: Dict[Tuple[str, int], str] = {}
    for (profile, combo) in combos_seen:
        if profile in backbone:
            strata[(profile, mode_maps[profile].get(combo, "all"))].append(combo)
        elif profile in r2:
            fold_of[(profile, combo)] = "S2"
        else:                                    # r1 or unlisted -> S1
            fold_of[(profile, combo)] = "S1"
    for (profile, _mode), combos in strata.items():
        combos.sort(key=lambda c: _bucket(f"{profile}_c{c:03d}", cfg.seed))
        for i, c in enumerate(combos):           # alternate -> exact 50/50 per stratum
            fold_of[(profile, c)] = "S1" if i % 2 == 0 else "S2"

    res = {}
    for p in files:
        m = _parse_name(p.name)
        if m:
            res[str(p)] = fold_of[(m["profile"], m["combo"])]
    return res


def split_files(files: List[Path], cfg: ObserverConfigT
                ) -> Dict[str, List[Path]]:
    # S3 χ-fold mode: test = held-out χ; train+val = the other χ values.
    if cfg.chi_fold_test >= 0:
        out = {"train": [], "val": [], "test": []}
        for p in files:
            chi = _parse_name(p.name)["chi"]
            if abs(chi - cfg.chi_fold_test) < C.CHI_TOL:
                out["test"].append(p)             # held-out χ (cross-χ test)
            elif _bucket(_group_key(p.name), cfg.seed + 1) < cfg.val_frac:
                out["val"].append(p)
            else:
                out["train"].append(p)
        return out
    # Fold-cross mode: train+val from cfg.train_fold, the OTHER fold is the test set.
    if cfg.train_fold:
        folds = assign_folds(files, cfg)
        out = {"train": [], "val": [], "test": []}
        for p in files:
            if folds[str(p)] != cfg.train_fold:
                out["test"].append(p)             # cross-subset test
            elif _bucket(_group_key(p.name), cfg.seed + 1) < cfg.val_frac:
                out["val"].append(p)              # same-subset val (grouped)
            else:
                out["train"].append(p)
        return out
    # Default grouped 3-way split.
    out = {"train": [], "val": [], "test": []}
    for p in files:
        r = _bucket(_group_key(p.name), cfg.seed)
        if r < cfg.test_frac:
            out["test"].append(p)
        elif r < cfg.test_frac + cfg.val_frac:
            out["val"].append(p)
        else:
            out["train"].append(p)
    return out


# ---------------------------------------------------------------------------
# Per-file read + decimate -> feature arrays
# ---------------------------------------------------------------------------
def _compute_arrays(path: Path) -> Dict[str, Any]:
    df = feather.read_feather(path, columns=F.required_columns())
    df = df.iloc[:: C.DECIM].reset_index(drop=True)        # 2000 -> 500 Hz
    a = F.build_arrays(df)
    m = _parse_name(Path(path).name)                       # per-file mu, chi
    a["mu"] = float(m["mu"]) if m else C.MU
    a["chi"] = float(m["chi"]) if m else 0.0
    return a


def read_arrays(path: Path, cache_dir: str = "") -> Dict[str, Any]:
    """Decimated 500 Hz feature/target arrays. With cache_dir set, results are
    memoised as float32 .npz (decimation is W- and regime-independent, so one
    cache serves every window size and every regime — kills the per-epoch
    re-read + re-decimate of the 2000 Hz Arrow, the dominant I/O cost)."""
    if not cache_dir:
        return _compute_arrays(path)
    cp = Path(cache_dir) / (Path(path).name + ".npz")
    if cp.exists():
        try:
            d = np.load(cp)
            a = {k: d[k] for k in d.files}
            a["mu"] = float(a["mu"]); a["chi"] = float(a["chi"])
            return a
        except Exception:
            pass                                            # corrupt/partial -> rebuild
    a = _compute_arrays(path)
    a = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
         for k, v in a.items()}
    # Caching is BEST-EFFORT: a concurrent writer, AV scan, or cloud-sync lock on
    # the freshly-written .npz must never crash training (seen as WinError 5 on
    # os.replace under OneDrive). On any failure, drop the temp and return the
    # computed arrays uncached -- a later epoch / the warm pass will fill it.
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
    return a


def warm_cache(files: List[Path], cache_dir: str) -> int:
    """Single-process pre-build of the decimated cache over `files`. Run ONCE
    before fanning out N parallel jobs (see parallel_sweep.py --warm-cache) so the
    jobs never race to write the same .npz. Returns the count successfully built.
    Decimation is W-/regime-independent, so pass the SUPERSET of every job's files
    (the launcher discovers with the full μ/χ grid) and all later runs hit warm."""
    if not cache_dir:
        print("[warm-cache] cache_dir empty -> nothing to do")
        return 0
    n = 0
    for j, p in enumerate(files):
        try:
            read_arrays(p, cache_dir)
            n += 1
        except Exception as e:                              # keep going; log the bad file
            print(f"[warm-cache] {Path(p).name}: {e!r}")
        if (j + 1) % 200 == 0:
            print(f"[warm-cache] {j + 1}/{len(files)} files")
    print(f"[warm-cache] {n}/{len(files)} files decimated -> {cache_dir}")
    return n


# ---------------------------------------------------------------------------
# Normalisation (streaming sufficient stats over TRAIN files only)
# ---------------------------------------------------------------------------
@dataclass
class Normalizer:
    g_mean: np.ndarray; g_std: np.ndarray          # [3]
    p_mean: np.ndarray; p_std: np.ndarray          # [6]  (shared across wheels)
    y_mean: np.ndarray; y_std: np.ndarray          # [4]  (per state, shared)

    def to_npz(self, path: Path) -> None:
        np.savez(path, g_mean=self.g_mean, g_std=self.g_std,
                 p_mean=self.p_mean, p_std=self.p_std,
                 y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def from_npz(cls, path: Path) -> "Normalizer":
        d = np.load(path)
        return cls(d["g_mean"], d["g_std"], d["p_mean"], d["p_std"],
                   d["y_mean"], d["y_std"])


def fit_normalizer(train_files: List[Path], cache_dir: str = "") -> Normalizer:
    """Per-feature mean/std. Per-wheel features and per-state targets are pooled
    across wheels (the encoder is wheel-shared, so one scale per channel)."""
    ng, npw, ns = C.N_GLOBAL, C.N_PERWHEEL, C.N_STATES
    gs = np.zeros(ng); gss = np.zeros(ng); gn = 0
    ps = np.zeros(npw); pss = np.zeros(npw); pn = 0
    ys = np.zeros(ns); yss = np.zeros(ns); yn = 0
    for j, p in enumerate(train_files):
        a = read_arrays(p, cache_dir)
        G, P, Y = a["G"], a["P"], a["Y"]
        gs += G.sum(0); gss += (G * G).sum(0); gn += G.shape[0]
        Pf = P.reshape(-1, npw); ps += Pf.sum(0); pss += (Pf * Pf).sum(0); pn += Pf.shape[0]
        Yf = Y.reshape(-1, ns); ys += Yf.sum(0); yss += (Yf * Yf).sum(0); yn += Yf.shape[0]
        del a
        if (j + 1) % 100 == 0:
            print(f"[norm] {j + 1}/{len(train_files)} files")

    def _ms(s, ss, n):
        m = s / n
        v = np.maximum(ss / n - m * m, 1e-12)
        return m, np.sqrt(v)

    gm, gsd = _ms(gs, gss, gn)
    pm, psd = _ms(ps, pss, pn)
    ym, ysd = _ms(ys, yss, yn)
    return Normalizer(gm, gsd, pm, psd, ym, ysd)


def load_max_scaler(csv_path: str) -> "Normalizer":
    """Frozen MAX-normalization scaler: offset 0, scale = p95(|x|) per channel,
    read from variable_scaler_percentiles.csv (build_variable_percentiles.py).
    Returned as a Normalizer with mean=0, std=p95 so make_windows / training /
    evaluation are UNCHANGED (x_norm = (x-0)/p95 = x/p95). sin_tt/cos_tt are left
    UNSCALED (scale 1) to preserve the sin/cos pair. Channel order follows
    config.GLOBAL_COLS / PERWHEEL_FEATURES / TARGET_STATES."""
    import csv as _csv
    p95: Dict[str, float] = {}
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            p95[row["variable"]] = float(row["abs_p95"])
    g = np.array([p95[c] for c in C.GLOBAL_COLS], dtype=np.float64)        # Vx, Vy, psi_dot
    pw = np.array([p95["Msat"], p95["w"], 1.0, 1.0], dtype=np.float64)     # Msat, w, sin_tt(1), cos_tt(1)
    y = np.array([p95[s] for s in C.TARGET_STATES], dtype=np.float64)      # gamma, zx, zy
    return Normalizer(np.zeros_like(g), g, np.zeros_like(pw), pw, np.zeros_like(y), y)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(a: Dict[str, np.ndarray], nrm: Normalizer, cfg: ObserverConfigT
                 ) -> Optional[Dict[str, np.ndarray]]:
    """Build causal windows from one trajectory's arrays.

    Returns batched tensors-to-be (numpy):
      Gw  [M, W, 3]      Pw [M, W, 4, 6]   (normalised inputs)
      Yt  [M, 4, 4]      (normalised targets at window end)
      aux: wz [M,4], sin_tt [M,4], psid [M], vpm [M,4]  (raw, for eval binning)
    where M = number of windows, W = cfg.window.
    """
    W, st = cfg.window, cfg.eff_stride
    T = a["G"].shape[0]
    if T < W:
        return None
    Gn = (a["G"] - nrm.g_mean) / nrm.g_std
    Pn = (a["P"] - nrm.p_mean) / nrm.p_std
    Yn = (a["Y"] - nrm.y_mean) / nrm.y_std
    ends = np.arange(W - 1, T, st)                          # window-end indices
    starts = ends - (W - 1)
    # gather windows via stride tricks would alias memory; explicit index is safe
    idx = starts[:, None] + np.arange(W)[None, :]           # [M, W]
    Gw = Gn[idx]                                            # [M, W, 3]
    Pw = Pn[idx]                                            # [M, W, 4, 6]
    Yt = Yn[ends]                                           # [M, 4, 4]
    # sin_tt at the window end (feature index 2 of P, in RAW units = sin so
    # already physical) -> recompute from raw P (unnormalised)
    sin_tt = a["P"][ends, :, 2]                             # [M, 4]
    out = dict(Gw=Gw.astype(np.float32), Pw=Pw.astype(np.float32),
               Yt=Yt.astype(np.float32),
               wz=a["wz"][ends], sin_tt=sin_tt, psid=a["psid"][ends],
               vpm=a["vpm"][ends])

    # Physical window-end block for the physics loss (raw units). P feature order
    # is [Msat, w, sin_tt, cos_tt]; Vpx0/Vpy0 come from the separate slip arrays
    # (they are NOT inputs); G is [Vx, Vy, psi_dot].
    if cfg.physics_loss:
        dt = C.DECIM / C.SIM_HZ                              # 1/500 s
        Pe = a["P"][ends]                                    # [M,4,4] raw
        w_dot = (a["P"][ends, :, 1] - a["P"][ends - 1, :, 1]) / dt
        M = ends.shape[0]
        out.update(
            ph_psi_dot=a["G"][ends, 2].astype(np.float32),  # [M] (physical psi_dot)
            ph_Vpx0=a["Vpx0"][ends].astype(np.float32),
            ph_Vpy0=a["Vpy0"][ends].astype(np.float32),
            ph_cti=Pe[:, :, 3].astype(np.float32), ph_sti=Pe[:, :, 2].astype(np.float32),
            ph_Msat=Pe[:, :, 0].astype(np.float32), ph_w=Pe[:, :, 1].astype(np.float32),
            ph_w_dot=w_dot.astype(np.float32),
            ph_mu=np.full(M, a["mu"], np.float32), ph_chi=np.full(M, a["chi"], np.float32),
        )
    return out


# Type alias placeholder so annotations above resolve without circular import.
from .config import ObserverConfig as ObserverConfigT  # noqa: E402
