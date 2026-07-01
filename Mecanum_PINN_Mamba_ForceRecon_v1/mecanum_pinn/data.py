"""Data loading + windowed dataset + DataLoader factories (Mamba ForceRecon v1).

Adapted from train_GPU_PINN_v14_py/mecanum_pinn/data.py. Changes:
  - NEW Arrow filename scheme (profile/combo/mu/case/friction_model/chi) — the
    old beta/amp regex is gone.
  - Force targets are roller-frame **Fpar_1..4, Fperp_1..4** (8); Mz dropped.
  - Reads the new lugre_adamov sweep columns (forces are first-class columns).
  - Downsamples every trajectory to `target_hz` (sim grid is 2000 Hz -> 500 Hz).
  - Stratifies by (profile, mu, chi)  [profile replaces the old 'motion'].
  - Optional probe channels (wz_i contact spin, util_i) kept on the traj dict
    for diagnostics / confidence-gating — NEVER fed as model inputs.

Trajectory dict layout:
    { 'path','name','profile','mu','chi','fc','fm',
      'states':(T,11), 'controls':(T,4), 'forces':(T,8), 'times':(T,1),
      'probes':(T,8)|None }   # probes = [wz_1..4, util_1..4]
"""
from __future__ import annotations

import os
import platform
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from torch.utils.data import DataLoader, Dataset

from .physics import F_MAX

# ============================================================
# Scaling constants (module-level np arrays for spawn workers)
# ============================================================
# Max-normalization scales = p95(|x|) over all 5345 trajectories, from
# ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv
# (build_variable_percentiles.py, 2026-06-21). These REPLACE the earlier hand-set
# caps, which under-sized the real data (Vx p95 1.92 vs old 1.0; w p95 39 vs old 25;
# Msat p95 5.28 vs old 10). Per-wheel channels (w, Msat) use the wheel-pooled p95.
# theta is the FOLDED angle (+-pi/12) -> a bounded angle feature, left UNSCALED (=1),
# matching how A2 leaves sin/cos unscaled. Forces keep the physical friction-circle
# bound F_MAX (Fpar/Fperp were not in that CSV; F_MAX is a principled scale, not a
# mis-calibrated cap). To switch forces to p95 too, recompute the CSV with the force
# columns and update force_max here.
state_max   = np.array([1.92, 0.6734, 2.408,                    # Vx, Vy, psi_dot  (p95)
                        39.0888, 39.0888, 39.0888, 39.0888,     # w1..4  (p95, wheel-pooled)
                        1.0, 1.0, 1.0, 1.0],                    # theta1..4 (folded +-pi/12; unscaled)
                       dtype=np.float32)
control_max = np.array([5.2794, 5.2794, 5.2794, 5.2794], dtype=np.float32)  # Msat (p95)
# Fpar/Fperp normalized by the physical friction-circle bound F_MAX (kept).
force_max   = np.array([F_MAX] * 8, dtype=np.float32)   # [Fpar_1..4, Fperp_1..4]

state_max_torch:   Optional[torch.Tensor] = None
control_max_torch: Optional[torch.Tensor] = None
force_max_torch:   Optional[torch.Tensor] = None


def init_torch_globals(device: torch.device) -> None:
    """Populate torch scaling tensors on `device`. Idempotent."""
    global state_max_torch, control_max_torch, force_max_torch
    state_max_torch   = torch.tensor(state_max,   dtype=torch.float32, device=device)
    control_max_torch = torch.tensor(control_max, dtype=torch.float32, device=device).unsqueeze(0)
    force_max_torch   = torch.tensor(force_max,   dtype=torch.float32, device=device).unsqueeze(0)


# ============================================================
# Arrow filename parsing + whitelist
# ============================================================
# e.g. coupled_vomega_c001_mu_0.3_case1_lugre_adamov_chi_0.000.arrow
#  profile may contain underscores/digits (multisine_50percent_cap); combo is
#  zero-padded; friction_model is lugre_adamov | lugre_uncoupled.
_FNAME_RE = re.compile(
    r'^(?P<profile>.+?)_c(?P<combo>\d+)_mu_(?P<mu>[0-9.]+)'
    r'_case(?P<fc>\d+)_(?P<fm>lugre_[a-z]+)_chi_(?P<chi>[0-9.]+)\.arrow$'
)

# Measurable-only model inputs + roller-frame force targets + probes.
_ARROW_STATE_COLS   = ['Vx', 'Vy', 'psi_dot',
                       'w1', 'w2', 'w3', 'w4',
                       'theta1', 'theta2', 'theta3', 'theta4']
_ARROW_CONTROL_COLS = ['Msat_1', 'Msat_2', 'Msat_3', 'Msat_4']
_ARROW_FORCE_COLS   = ['Fpar_1', 'Fpar_2', 'Fpar_3', 'Fpar_4',
                       'Fperp_1', 'Fperp_2', 'Fperp_3', 'Fperp_4']
_ARROW_PROBE_COLS   = ['wz_1', 'wz_2', 'wz_3', 'wz_4',          # contact spin (HIDDEN)
                       'util_1', 'util_2', 'util_3', 'util_4']  # friction-circle util
_ARROW_TIME_COL     = 'time'


def parse_arrow_filename(name: str) -> Optional[Dict[str, Any]]:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return {
        'profile': m['profile'],
        'combo':   int(m['combo']),
        'mu':      float(m['mu']),
        'fc':      int(m['fc']),
        'fm':      m['fm'],
        'chi':     float(m['chi']),
    }


def parse_whitelist(path: Path,
                    subsample_n: Optional[int] = None,
                    subsample_seed: int = 42) -> Optional[set]:
    """Read a whitelist of approved .arrow filenames (one per line, '#' comments).

    Source: the diagnostics pipeline exports the names whose `combined_reco`
    in diagnostics_combined.csv does not start with 'reject' (1568/1776).
    Optional stratified subsample by profile for quick smoke runs.
    """
    path = Path(path)
    if not path.exists():
        print(f"[whitelist] {path} not found - loading all Arrow files")
        return None
    keep: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                keep.append(line)
    if subsample_n is None or subsample_n >= len(keep):
        print(f"[whitelist] {len(keep)} approved trajectories from {path}")
        return set(keep)

    by_profile: Dict[str, List[str]] = defaultdict(list)
    for name in keep:
        parsed = parse_arrow_filename(name)
        by_profile[parsed['profile'] if parsed else 'unknown'].append(name)
    rng = np.random.default_rng(subsample_seed)
    sampled: List[str] = []
    profiles = sorted(by_profile.keys())
    per = max(1, subsample_n // len(profiles))
    for p in profiles:
        names = by_profile[p]; rng.shuffle(names); sampled.extend(names[:per])
    if len(sampled) < subsample_n:
        rest = [n for n in keep if n not in sampled]
        rng.shuffle(rest); sampled.extend(rest[:subsample_n - len(sampled)])
    sampled = sampled[:subsample_n]
    print(f"[whitelist] {len(sampled)} sampled from {len(keep)} "
          f"(stratified across {len(profiles)} profiles)")
    return set(sampled)


# ============================================================
# Trajectory loading
# ============================================================
def _file_passes_filters(parsed: Dict[str, Any],
                         mu_values: Optional[List[float]],
                         chi_values: Optional[List[float]],
                         profiles: Optional[List[str]],
                         friction_models: Optional[List[str]]) -> bool:
    if mu_values is not None and not any(abs(parsed['mu'] - v) < 1e-6 for v in mu_values):
        return False
    if chi_values is not None and not any(abs(parsed['chi'] - c) < 1e-4 for c in chi_values):
        return False
    if profiles is not None and parsed['profile'] not in profiles:
        return False
    if friction_models is not None and parsed['fm'] not in friction_models:
        return False
    return True


def _downsample_factor(times: np.ndarray, target_hz: Optional[float]) -> int:
    """Integer stride to bring a uniform `times` grid down to ~target_hz."""
    if target_hz is None or times.shape[0] < 3:
        return 1
    dt = float(np.median(np.diff(times[:, 0])))
    if dt <= 0:
        return 1
    native_hz = 1.0 / dt
    return max(1, int(round(native_hz / target_hz)))


# ============================================================
# Decimated-trajectory cache  (mirrors observer_v1_py read_arrays)
# ============================================================
# The 2000 Hz Arrow read + downsample is the dominant per-epoch / per-run I/O
# cost and is INDEPENDENT of seq_len, stride, regime, and architecture — so one
# float32 .npz cache (keyed by filename + target_hz + probe flag) serves every
# run and every concurrent worker. This is the prerequisite that makes
# experiment-level parallelism (N independent runs) pay off instead of N workers
# thrashing CPU/disk re-decimating the same Arrows. See parallel_sweep.py.
def _read_decimated(fp: Path, target_hz: Optional[float],
                    load_probes: bool) -> Optional[Dict[str, np.ndarray]]:
    """Read one Arrow, slice to the model columns, downsample to ~target_hz.
    Returns None (so the caller counts skipped_missing) if a column is absent."""
    # Windows: memory-mapping + multi-threading inside PyArrow can trigger native
    # crashes with some Arrow files / drivers. Use the safest options available.
    kw = {}
    if platform.system() == "Windows":
        kw["memory_map"] = False
        kw["use_threads"] = False
    try:
        df = feather.read_feather(fp, **kw)
    except Exception as e:
        try:
            with pa.ipc.open_file(fp, memory_map=False) as reader:
                table = reader.read_all()
            df = table.to_pandas()
        except Exception:
            print(f"[load] {Path(fp).name}: Arrow read failed ({e!r}); skipped")
            return None
    try:
        S = df[_ARROW_STATE_COLS].to_numpy(dtype=np.float32)
        U = df[_ARROW_CONTROL_COLS].to_numpy(dtype=np.float32)
        F = df[_ARROW_FORCE_COLS].to_numpy(dtype=np.float32)
        T = df[_ARROW_TIME_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
        P = df[_ARROW_PROBE_COLS].to_numpy(dtype=np.float32) if load_probes else None
    except KeyError as e:
        print(f"[load] {Path(fp).name}: missing column {e}; skipped")
        return None
    step = _downsample_factor(T, target_hz)
    if step > 1:
        S, U, F, T = S[::step], U[::step], F[::step], T[::step]
        if P is not None:
            P = P[::step]
    out = dict(states=S, controls=U, forces=F, times=T)
    if P is not None:
        out["probes"] = P
    return out


def read_trajectory(fp: Path, target_hz: Optional[float] = 500.0,
                    load_probes: bool = False,
                    cache_dir: str = "") -> Optional[Dict[str, np.ndarray]]:
    """Decimated trajectory arrays, memoised as float32 .npz when cache_dir set.

    Cache key folds in target_hz and the probe flag so a 500 Hz no-probe entry
    can never be mistaken for a different request. Atomic per-PID tmp -> rename so
    concurrent warm/train workers can't read a half-written file."""
    if not cache_dir:
        return _read_decimated(fp, target_hz, load_probes)
    hz = int(round(target_hz)) if target_hz else 0
    cp = Path(cache_dir) / f"{Path(fp).name}.hz{hz}{'_p' if load_probes else ''}.npz"
    if cp.exists():
        try:
            d = np.load(cp)
            return {k: d[k] for k in d.files}
        except Exception:
            pass                                          # corrupt/partial -> rebuild
    out = _read_decimated(fp, target_hz, load_probes)
    if out is None:
        return None
    # Best-effort cache write: a concurrent writer / AV / cloud-sync lock on the
    # fresh .npz must never crash training (WinError 5 on os.replace under
    # OneDrive). On failure, drop the temp and return the arrays uncached.
    tmp = None
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.parent / f"{cp.name}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, cp)                               # atomic on same FS
    except Exception:
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass
    return out


def warm_cache(data_dir,
               whitelist: Optional[set] = None,
               mu_values: Optional[List[float]] = None,
               chi_values: Optional[List[float]] = None,
               profiles: Optional[List[str]] = None,
               friction_models: Optional[List[str]] = ('lugre_adamov',),
               target_hz: Optional[float] = 500.0,
               load_probes: bool = False,
               cache_dir: str = "",
               verbose: bool = True) -> int:
    """Single-process pre-build of the decimated cache over the selected files.

    Run ONCE before fanning out N parallel training jobs so they never race to
    write the same .npz (see parallel_sweep.py --warm-cache). Returns the number
    of files decimated (cache misses filled; hits are skipped cheaply)."""
    if not cache_dir:
        print("[warm-cache] cache_dir empty -> nothing to do")
        return 0
    dirs = ([Path(data_dir)] if isinstance(data_dir, (str, Path))
            else [Path(d) for d in data_dir])
    files: List[Path] = []
    for d in dirs:
        files.extend(sorted(d.glob('*.arrow')))
    sel = []
    for fp in files:
        parsed = parse_arrow_filename(fp.name)
        if parsed is None:
            continue
        if whitelist is not None and fp.name not in whitelist:
            continue
        if not _file_passes_filters(parsed, mu_values, chi_values, profiles, friction_models):
            continue
        sel.append(fp)
    built = 0
    for j, fp in enumerate(sel):
        if read_trajectory(fp, target_hz=target_hz, load_probes=load_probes,
                           cache_dir=cache_dir) is not None:
            built += 1
        if verbose and (j + 1) % 200 == 0:
            print(f"[warm-cache] {j + 1}/{len(sel)} files")
    if verbose:
        print(f"[warm-cache] {built}/{len(sel)} selected files decimated -> {cache_dir}")
    return built


def load_all_arrow_trajectories(data_dir,
                                whitelist: Optional[set] = None,
                                mu_values: Optional[List[float]] = None,
                                chi_values: Optional[List[float]] = None,
                                profiles: Optional[List[str]] = None,
                                friction_models: Optional[List[str]] = ('lugre_adamov',),
                                target_hz: Optional[float] = 500.0,
                                load_probes: bool = False,
                                cache_dir: str = "",
                                verbose: bool = True) -> List[Dict[str, Any]]:
    """Stream the lugre_adamov sweep into trajectory dicts.

    `data_dir` may be a single path or a list of paths (e.g. one dir per mu).
    Forces are roller-frame Fpar/Fperp (8). Trajectories are downsampled to
    `target_hz` (set None to keep native 2000 Hz). With `cache_dir` set, each
    file's decimated arrays are memoised as float32 .npz (the per-run repeat-I/O
    killer; reused across runs and concurrent workers — see warm_cache).
    """
    dirs = [Path(data_dir)] if isinstance(data_dir, (str, Path)) else [Path(d) for d in data_dir]
    arrow_files: List[Path] = []
    for d in dirs:
        arrow_files.extend(sorted(d.glob('*.arrow')))

    trajectories: List[Dict[str, Any]] = []
    skipped_whitelist = skipped_filter = skipped_unparsed = skipped_missing = 0

    for fp in arrow_files:
        parsed = parse_arrow_filename(fp.name)
        if parsed is None:
            skipped_unparsed += 1; continue
        if whitelist is not None and fp.name not in whitelist:
            skipped_whitelist += 1; continue
        if not _file_passes_filters(parsed, mu_values, chi_values, profiles, friction_models):
            skipped_filter += 1; continue

        rec = read_trajectory(fp, target_hz=target_hz, load_probes=load_probes,
                              cache_dir=cache_dir)
        if rec is None:                              # missing column -> _read_decimated logged it
            skipped_missing += 1; continue
        S, U, F, T = rec['states'], rec['controls'], rec['forces'], rec['times']
        P = rec.get('probes')                        # None when load_probes=False

        trajectories.append({
            'path': str(fp), 'name': fp.name,
            'profile': parsed['profile'], 'combo': parsed['combo'],
            'mu': parsed['mu'], 'chi': parsed['chi'],
            'fc': parsed['fc'], 'fm': parsed['fm'],
            'states': S, 'controls': U, 'forces': F, 'times': T, 'probes': P,
        })

    if verbose:
        print(f"[load] kept={len(trajectories)} skipped_whitelist={skipped_whitelist} "
              f"skipped_filter={skipped_filter} skipped_unparsed={skipped_unparsed} "
              f"skipped_missing={skipped_missing}")
        if trajectories:
            t0 = trajectories[0]
            print(f"[load] sample shapes: states={t0['states'].shape} forces={t0['forces'].shape} "
                  f"(downsampled ~{int(round(1.0/np.median(np.diff(t0['times'][:,0]))))} Hz)")
    return trajectories


# ============================================================
# Train/val/test stratified split  (by profile, mu, chi)
# ============================================================
def stratified_split(trajectories: List[Dict[str, Any]],
                     train_ratio: float, val_ratio: float,
                     seed: int = 42) -> Tuple[List, List, List]:
    """Split by (profile, combo) GROUPS so ALL mu/chi variants of a trajectory land in
    the SAME split (no mu-axis leakage -> honest mu-extrapolation). Groups are stratified
    by profile."""
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio >= 0, f"train+val={train_ratio+val_ratio} > 1.0"
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple, List[int]] = defaultdict(list)
    for i, t in enumerate(trajectories):
        groups[(t['profile'], t['combo'])].append(i)        # all mu of a trajectory together
    keys_by_profile: Dict[str, List[Tuple]] = defaultdict(list)
    for key in groups:
        keys_by_profile[key[0]].append(key)

    tr_idx, va_idx, te_idx = [], [], []
    for _prof, keys in keys_by_profile.items():
        rng.shuffle(keys)
        n = len(keys)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(0, int(round(n * val_ratio)))
        if n - n_tr - n_va < 0:
            n_va = max(0, n - n_tr)
        for k in keys[:n_tr]:            tr_idx.extend(groups[k])
        for k in keys[n_tr:n_tr + n_va]: va_idx.extend(groups[k])
        for k in keys[n_tr + n_va:]:     te_idx.extend(groups[k])

    tr = [trajectories[i] for i in tr_idx]
    va = [trajectories[i] for i in va_idx]
    te = [trajectories[i] for i in te_idx]
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
          f"groups={len(groups)} (by profile,combo; all mu together)")
    return tr, va, te


# ============================================================
# Sliding-window dataset
# ============================================================
class MecanumTrajectoryDataset(Dataset):
    """Sliding-window samples; returns (S, U, T, S_next, F_sim, mu, chi).

    F_sim is the 8-channel roller-frame [Fpar_1..4, Fperp_1..4], normalized.
    theta channels (7:11) are folded to +-pi/12 (12-roller periodicity) on both
    S and S_next so state-loss isn't dominated by cumulative wheel rotation.
    """
    def __init__(self, trajectories: List[Dict[str, Any]], seq_len: int, stride: int = 1):
        self.seq_len = seq_len
        self.stride  = max(1, int(stride))
        self.windows = []
        for ti, traj in enumerate(trajectories):
            n = traj['states'].shape[0]
            if n < seq_len + 2:
                continue
            for start in range(0, n - seq_len - 1, self.stride):
                self.windows.append((ti, start))
        self.trajectories = trajectories

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ti, start = self.windows[idx]
        traj = self.trajectories[ti]
        end = start + self.seq_len
        S      = traj['states'][start:end]         / state_max
        S_next = traj['states'][start + 1:end + 1] / state_max
        U      = traj['controls'][start:end]       / control_max
        F_sim  = traj['forces'][start:end]         / force_max
        T      = traj['times'][start:end]

        S[..., 7:11]      = np.arctan2(np.sin(12.0 * S[..., 7:11]),
                                       np.cos(12.0 * S[..., 7:11])) / 12.0
        S_next[..., 7:11] = np.arctan2(np.sin(12.0 * S_next[..., 7:11]),
                                       np.cos(12.0 * S_next[..., 7:11])) / 12.0

        return (
            torch.from_numpy(S).float(),
            torch.from_numpy(U).float(),
            torch.from_numpy(T).float(),
            torch.from_numpy(S_next).float(),
            torch.from_numpy(F_sim).float(),
            torch.tensor(traj['mu'],  dtype=torch.float32),
            torch.tensor(traj['chi'], dtype=torch.float32),
        )


def make_loader(ds: Dataset, config: Dict[str, Any], shuffle: bool, drop_last: bool) -> DataLoader:
    kw = dict(batch_size=config['batch_size'], shuffle=shuffle, drop_last=drop_last,
              num_workers=config['num_workers'], pin_memory=config.get('pin_memory', True))
    if config['num_workers'] > 0:
        kw['persistent_workers'] = config.get('persistent_workers', True)
        kw['prefetch_factor']    = config.get('prefetch_factor', 2)
    return DataLoader(ds, **kw)


def build_loaders_from_lists(train_trajs, val_trajs, test_trajs, config):
    stride = config.get('stride', 1)
    tr_ds = MecanumTrajectoryDataset(train_trajs, seq_len=config['seq_len'], stride=stride)
    va_ds = MecanumTrajectoryDataset(val_trajs,   seq_len=config['seq_len'], stride=stride)
    te_ds = MecanumTrajectoryDataset(test_trajs,  seq_len=config['seq_len'], stride=stride)
    tr = make_loader(tr_ds, config, shuffle=True,  drop_last=True)
    va = make_loader(va_ds, config, shuffle=False, drop_last=False)
    te = make_loader(te_ds, config, shuffle=False, drop_last=False)
    print(f"[data] windows: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} (stride={stride})")
    return tr, va, te


def build_loaders_with_split(trajectories, config):
    tr, va, te = stratified_split(trajectories, config['train_ratio'], config['val_ratio'],
                                  seed=config['seed'])
    tl, vl, sl = build_loaders_from_lists(tr, va, te, config)
    return tl, vl, sl, tr, va, te


def filter_trajectories_by_name(trajectories, names: Sequence[str]):
    by_name = {t['name']: t for t in trajectories}
    out, missing = [], []
    for n in names:
        t = by_name.get(n)
        (out if t is not None else missing).append(t if t is not None else n)
    if missing:
        print(f"[filter] {len(missing)} of {len(names)} names not found "
              f"(first: {missing[0]})")
    return out


def load_regime_split(regime_toml, config: Dict[str, Any]
                      ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Train/val/test trajectory lists from the SHARED regime TOML selection
    (`observer_v1_py/regimes/*.toml`) -> identical trajectory selection to Approach 2.
    The regime TOML supplies its own mu/chi/profile gates + whitelist; we load only
    the selected files and partition them by the regime's split."""
    from .regime_split import compute_regime_split
    split = compute_regime_split(
        regime_toml=regime_toml,
        data_dir=config['data_dir'],
        whitelist_csv=config['whitelist_csv'],
        project_root=config.get('project_root', '.'),
        test_chi=config.get('test_chi'))
    all_names = set(split['train']) | set(split['val']) | set(split['test'])
    trajs = load_all_arrow_trajectories(
        config['data_dir'], whitelist=all_names,
        mu_values=None, chi_values=None, profiles=None,
        friction_models=None, target_hz=config['target_hz'],
        cache_dir=config.get('cache_dir', ''))
    by_name = {t['name']: t for t in trajs}
    pick = lambda names: [by_name[n] for n in names if n in by_name]
    tr, va, te = pick(split['train']), pick(split['val']), pick(split['test'])
    print(f"[regime] loaded train={len(tr)} val={len(va)} test={len(te)} trajectories")
    return tr, va, te
