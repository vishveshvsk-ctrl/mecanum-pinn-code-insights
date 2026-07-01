"""Regime-driven train/val/test selection — a FAITHFUL port of
`observer_v1_py/mecanum_observer/data.py` (select_regime / assign_folds /
split_files / _combo_mode_map), so Approach 1 selects the EXACT same trajectories
as Approach 2 from the SAME regime TOMLs (`observer_v1_py/regimes/*.toml`).

It is deterministic on filenames only (seeded SHA1 buckets), reads the same
profile combo TOMLs for the excitation-mode map, and keeps (profile, combo) groups
(all mu/chi siblings) together -> no leakage. See `training_data_split_design.md`.

Public entry point: `compute_regime_split(...) -> {'train':[names],'val':[...],'test':[...]}`.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:                                  # stdlib on 3.11+
    import tomllib as _toml
except ModuleNotFoundError:           # pragma: no cover
    import tomli as _toml

_CHI_TOL = 1e-4
_FNAME_RE = re.compile(
    r"^(?P<profile>.+?)_c(?P<combo>\d{3})_mu_(?P<mu>[0-9.eE+\-]+)"
    r"_case(?P<fc>\d+)_(?P<fm>lugre_adamov|lugre_uncoupled)"
    r"_chi_(?P<chi>[0-9.]+)\.arrow$")


def _parse_name(name: str) -> Optional[dict]:
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    return dict(profile=m["profile"], combo=int(m["combo"]),
                mu=float(m["mu"]), fm=m["fm"], chi=float(m["chi"]))


def _group_key(name: str) -> str:
    m = _parse_name(name)
    return f"{m['profile']}_c{m['combo']:03d}" if m else name


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF       # deterministic [0,1)


# ---------------------------------------------------------------------------
# Regime config (subset of A2's ObserverConfig — only the selection fields)
# ---------------------------------------------------------------------------
@dataclass
class RegimeConfig:
    data_dir: Path
    whitelist_csv: Path
    project_root: Path
    regime_name: str = "default"
    # data gate
    mu_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.8])
    chi_values: List[float] = field(default_factory=lambda: [0.0, 0.002, 0.005, 0.008])
    include_profiles: List[str] = field(default_factory=list)
    exclude_profiles: List[str] = field(default_factory=list)
    per_profile_cap: int = 0
    subsample_fraction: float = 1.0
    chi_stratify: bool = False
    matched_chi_quads_only: bool = False
    min_chi_per_combo: int = 3
    # split
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1234
    # fold
    train_fold: str = ""
    backbone_profiles: List[str] = field(default_factory=list)
    redundant_S1: List[str] = field(default_factory=list)
    redundant_S2: List[str] = field(default_factory=list)
    profiles_toml_dir: str = ""
    redundant_sample_frac: float = 1.0
    redundant_sample_profiles: List[str] = field(default_factory=list)
    chi_fold_test: float = -1.0


def _load_merged_toml(regime_path: Path) -> Dict[str, Any]:
    """base.toml (sibling) + regime TOML, regime overriding per-table."""
    base_path = regime_path.parent / "base.toml"
    merged: Dict[str, Any] = {}
    for p in (base_path, regime_path):
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


def build_regime_config(regime_toml: Path, data_dir, whitelist_csv,
                        project_root, override_chi_fold_test: Optional[float] = None
                        ) -> RegimeConfig:
    d = _load_merged_toml(Path(regime_toml))
    data, split, fold = d.get("data", {}), d.get("split", {}), d.get("fold", {})

    def _flist(v):
        return [float(x) for x in (v if isinstance(v, list) else [v])]

    rc = RegimeConfig(data_dir=Path(data_dir), whitelist_csv=Path(whitelist_csv),
                      project_root=Path(project_root),
                      regime_name=d.get("regime", {}).get("name", "default"))
    if "mu" in data:  rc.mu_values = _flist(data["mu"])
    if "chi" in data: rc.chi_values = _flist(data["chi"])
    for k in ("include_profiles", "exclude_profiles"):
        if k in data: setattr(rc, k, list(data[k]))
    for k, cast in (("per_profile_cap", int), ("subsample_fraction", float),
                    ("chi_stratify", bool), ("matched_chi_quads_only", bool),
                    ("min_chi_per_combo", int)):
        if k in data: setattr(rc, k, cast(data[k]))
    for k, cast in (("val_frac", float), ("test_frac", float), ("seed", int)):
        if k in split: setattr(rc, k, cast(split[k]))
    _fold = {"train_fold": ("train_fold", str), "backbone_profiles": ("backbone", list),
             "redundant_S1": ("redundant_S1", list), "redundant_S2": ("redundant_S2", list),
             "profiles_toml_dir": ("profiles_toml_dir", str),
             "redundant_sample_frac": ("redundant_sample_frac", float),
             "redundant_sample_profiles": ("redundant_sample_profiles", list),
             "chi_fold_test": ("chi_fold_test", float)}
    for attr, (key, cast) in _fold.items():
        if key in fold: setattr(rc, attr, cast(fold[key]))
    if override_chi_fold_test is not None:        # --test-chi equivalent
        rc.chi_fold_test = float(override_chi_fold_test)
    return rc


# ---------------------------------------------------------------------------
# Whitelist + discovery + regime sampler  (verbatim logic from A2)
# ---------------------------------------------------------------------------
def load_whitelist(csv: Path) -> Optional[set]:
    """Read the whitelist CSV without pandas to avoid a heavy/fragile C-extension
    import on Windows (pandas + pyarrow together have been the crash point here)."""
    import csv as _csv
    if not Path(csv).exists():
        print(f"[regime] WARNING: whitelist {csv} missing -> accepting all files")
        return None
    keep: List[str] = []
    with open(csv, newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        if reader.fieldnames is None:
            print("[regime] WARNING: whitelist empty -> accept all")
            return None
        if "file" not in reader.fieldnames:
            print("[regime] WARNING: whitelist lacks 'file' column -> accept all")
            return None
        has_reco = "combined_reco" in reader.fieldnames
        for row in reader:
            if has_reco and str(row.get("combined_reco", "")).startswith("reject"):
                continue
            keep.append(row["file"])
    print(f"[whitelist] {len(keep)} approved trajectories from {csv}")
    return set(keep)


def _discover(cfg: RegimeConfig) -> List[Path]:
    print("[regime-debug] loading whitelist ...")
    wl = load_whitelist(cfg.whitelist_csv)
    print(f"[regime-debug] whitelist size={len(wl) if wl is not None else 'all'}")
    inc, exc = set(cfg.include_profiles), set(cfg.exclude_profiles)
    print(f"[regime-debug] globbing {Path(cfg.data_dir).resolve()} ...")
    arrow_paths = sorted(Path(cfg.data_dir).glob("*.arrow"))
    print(f"[regime-debug] glob returned {len(arrow_paths)} files")
    out = []
    for p in arrow_paths:
        m = _parse_name(p.name)
        if m is None:
            continue
        if not any(abs(m["mu"] - mv) < 1e-9 for mv in cfg.mu_values):
            continue
        if not any(abs(m["chi"] - g) < _CHI_TOL for g in cfg.chi_values):
            continue
        if inc and m["profile"] not in inc:
            continue
        if m["profile"] in exc:
            continue
        if wl is not None and p.name not in wl:
            continue
        out.append(p)
    return _select_regime(out, cfg)


def _select_regime(paths: List[Path], cfg: RegimeConfig) -> List[Path]:
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
        def n_chi(k):
            return len({_parse_name(p.name)["chi"] for p in groups[k]})
        if cfg.matched_chi_quads_only:
            keys = [k for k in keys if n_chi(k) >= cfg.min_chi_per_combo]
        if not keys:
            continue
        keys.sort(key=lambda k: (-(n_chi(k) if cfg.chi_stratify else 0),
                                 _bucket(f"{k[0]}_c{k[1]:03d}", cfg.seed)))
        if cfg.per_profile_cap > 0:
            keys = keys[: cfg.per_profile_cap]
        frac = (cfg.redundant_sample_frac if profile in cfg.redundant_sample_profiles
                else cfg.subsample_fraction)
        if frac < 1.0:
            keys = keys[: max(1, math.ceil(frac * len(keys)))]
        for k in keys:
            kept.extend(groups[k])
    return sorted(kept)


# ---------------------------------------------------------------------------
# Excitation 2-fold (S1/S2) + chi k-fold (S3)  (verbatim logic from A2)
# ---------------------------------------------------------------------------
def _combo_mode_map(profile: str, toml_dir: Path) -> Dict[int, str]:
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
        dd = abs(float(deg)) % 180.0
        return "x" if dd < 1e-6 else ("y" if abs(dd - 90.0) < 1e-6 else "diag")

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


def _assign_folds(files: List[Path], cfg: RegimeConfig) -> Dict[str, str]:
    backbone = set(cfg.backbone_profiles)
    r2 = set(cfg.redundant_S2)
    toml_dir = cfg.project_root / cfg.profiles_toml_dir
    mode_maps = {p: _combo_mode_map(p, toml_dir) for p in backbone}

    combos_seen = {(m["profile"], m["combo"])
                   for p in files for m in [_parse_name(p.name)] if m}
    strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    fold_of: Dict[Tuple[str, int], str] = {}
    for (profile, combo) in combos_seen:
        if profile in backbone:
            strata[(profile, mode_maps[profile].get(combo, "all"))].append(combo)
        elif profile in r2:
            fold_of[(profile, combo)] = "S2"
        else:
            fold_of[(profile, combo)] = "S1"
    for (profile, _mode), combos in strata.items():
        combos.sort(key=lambda c: _bucket(f"{profile}_c{c:03d}", cfg.seed))
        for i, c in enumerate(combos):
            fold_of[(profile, c)] = "S1" if i % 2 == 0 else "S2"

    res = {}
    for p in files:
        m = _parse_name(p.name)
        if m:
            res[p.name] = fold_of[(m["profile"], m["combo"])]
    return res


def _split_files(files: List[Path], cfg: RegimeConfig) -> Dict[str, List[str]]:
    out = {"train": [], "val": [], "test": []}
    if cfg.chi_fold_test >= 0:                    # S3: held-out chi = test
        for p in files:
            chi = _parse_name(p.name)["chi"]
            if abs(chi - cfg.chi_fold_test) < _CHI_TOL:
                out["test"].append(p.name)
            elif _bucket(_group_key(p.name), cfg.seed + 1) < cfg.val_frac:
                out["val"].append(p.name)
            else:
                out["train"].append(p.name)
        return out
    if cfg.train_fold:                            # S1/S2: other fold = test
        folds = _assign_folds(files, cfg)
        for p in files:
            if folds[p.name] != cfg.train_fold:
                out["test"].append(p.name)
            elif _bucket(_group_key(p.name), cfg.seed + 1) < cfg.val_frac:
                out["val"].append(p.name)
            else:
                out["train"].append(p.name)
        return out
    for p in files:                               # default grouped 3-way
        r = _bucket(_group_key(p.name), cfg.seed)
        if r < cfg.test_frac:
            out["test"].append(p.name)
        elif r < cfg.test_frac + cfg.val_frac:
            out["val"].append(p.name)
        else:
            out["train"].append(p.name)
    return out


def compute_regime_split(regime_toml, data_dir, whitelist_csv, project_root,
                         test_chi: Optional[float] = None) -> Dict[str, List[str]]:
    """Top-level: regime TOML -> {'train','val','test'} Arrow FILENAME lists.
    `test_chi` overrides the S3 held-out chi (the --test-chi equivalent)."""
    print(f"[regime-debug] inputs: data_dir={Path(data_dir).resolve()} "
          f"whitelist_csv={Path(whitelist_csv).resolve()} project_root={Path(project_root).resolve()}")
    print(f"[regime-debug] building regime config from {regime_toml} ...")
    cfg = build_regime_config(regime_toml, data_dir, whitelist_csv, project_root,
                              override_chi_fold_test=test_chi)
    print(f"[regime-debug] discovering files ...")
    files = _discover(cfg)
    print(f"[regime-debug] discovered {len(files)} files; splitting ...")
    split = _split_files(files, cfg)
    print(f"[regime] {cfg.regime_name}: train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])} "
          f"(from {len(files)} selected files)")
    return split
