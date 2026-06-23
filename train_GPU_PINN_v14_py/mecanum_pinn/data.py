"""Data loading + windowed dataset + DataLoader factories.

Layout of the trajectory dicts produced by `load_all_arrow_trajectories`:

    {
        'path':   str,                 # source .arrow path
        'name':   str,                 # filename only
        'mu':     float, 'chi': float, # parsed from filename
        'motion': str,   'fc':  int,
        'states':   np.ndarray (T, 11),
        'controls': np.ndarray (T,  4),
        'forces':   np.ndarray (T, 12),
        'times':    np.ndarray (T,  1),
    }

Scaling constants live here too. The numpy arrays (`state_max`, `control_max`,
`force_max`) are referenced by `MecanumTrajectoryDataset.__getitem__`, which
runs in DataLoader worker processes — by being module-level numpy arrays they
serialize cleanly across `spawn` workers without any per-call setup. The torch
versions are populated by `init_torch_globals(device)` from the main process
only and accessed via attribute lookup on this module (so other modules see
the latest value after init).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import DataLoader, Dataset

from .physics import F_MAX, MZ_MAX

# ============================================================
# Scaling constants
# ============================================================
# Numpy versions are accessed inside DataLoader workers — keep at module
# level so spawn workers re-import this module and see them without any
# per-process initialization.
state_max   = np.array([1.0, 1.0, 1.0,                # Vx, Vy, psi_dot
                        25., 25., 25., 25.,           # w1..4
                        1.0, 1.0, 1.0, 1.0],          # theta1..4
                       dtype=np.float32)
control_max = np.array([10., 10., 10., 10.], dtype=np.float32)

# Forces in N (peak ~30 N at case 1, chi=8mm); moments in N*m (peak ~0.2 N*m).
# Separate scales prevent Mz signal from being washed out by F in MSE.
# Imported from .physics so models.py and data.py share one source of truth
# (see physics.py top-of-file for the derivation).
force_max = np.array([F_MAX]*8 + [MZ_MAX]*4, dtype=np.float32)

# Torch versions are populated lazily by init_torch_globals(). They start
# as None so that worker processes (which never call init_torch_globals)
# don't try to claim a CUDA context at import time.
state_max_torch:   Optional[torch.Tensor] = None
control_max_torch: Optional[torch.Tensor] = None
force_max_torch:   Optional[torch.Tensor] = None


def init_torch_globals(device: torch.device) -> None:
    """Populate the torch scaling tensors on `device`. Idempotent.

    Other modules access these via attribute lookup on this module
    (e.g. `from mecanum_pinn import data; data.state_max_torch`) so they
    always see the value bound by the most recent call.
    """
    global state_max_torch, control_max_torch, force_max_torch
    state_max_torch   = torch.tensor(state_max,   dtype=torch.float32, device=device)
    control_max_torch = torch.tensor(control_max, dtype=torch.float32, device=device).unsqueeze(0)
    force_max_torch   = torch.tensor(force_max,   dtype=torch.float32, device=device).unsqueeze(0)


# ============================================================
# Arrow filename parsing + whitelist
# ============================================================
_FNAME_RE = re.compile(
    r'^test_mu_(?P<mu>[0-9.]+)_(?P<motion>[a-z]+)_asmc_case(?P<fc>\d+)'
    r'_psi_var_beta(?P<beta>[-0-9.]+)_amp(?P<amp>[0-9.]+)_chi_(?P<chi>[0-9.]+)\.arrow$'
)

_ALL_MOTION_CASES = ('straightline', 'sineline', 'infinity', 'circle')

_ARROW_STATE_COLS   = ['Vx', 'Vy', 'psi_dot',
                       'w1', 'w2', 'w3', 'w4',
                       'theta1', 'theta2', 'theta3', 'theta4']
_ARROW_CONTROL_COLS = ['Msat_1', 'Msat_2', 'Msat_3', 'Msat_4']
_ARROW_FORCE_COLS   = ['Fx_1', 'Fx_2', 'Fx_3', 'Fx_4',
                       'Fy_1', 'Fy_2', 'Fy_3', 'Fy_4',
                       'Mz_1', 'Mz_2', 'Mz_3', 'Mz_4']
_ARROW_TIME_COL     = 'time'


def parse_arrow_filename(name: str) -> Optional[Dict[str, Any]]:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return {
        'mu':     float(m['mu']),
        'motion': m['motion'],
        'fc':     int(m['fc']),
        'beta':   float(m['beta']),
        'amp':    float(m['amp']),
        'chi':    float(m['chi']),
    }


def parse_whitelist(path: Path,
                    subsample_n: Optional[int] = None,
                    subsample_seed: int = 42) -> Optional[set]:
    """Read whitelist.txt; optionally take a deterministic random subset.

    subsample_n=None  -> no subsampling (return full whitelist).
    subsample_n=K     -> return min(K, len) names, drawn with seed.
                         Stratified by motion case so a small K still spans
                         all motion types when possible.
    """
    path = Path(path)
    if not path.exists():
        print(f"[whitelist] {path} not found — loading all Arrow files")
        return None

    keep: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            keep.append(line)

    if subsample_n is None or subsample_n >= len(keep):
        print(f"[whitelist] {len(keep)} approved trajectories from {path}")
        return set(keep)

    by_motion: Dict[str, List[str]] = defaultdict(list)
    for name in keep:
        parsed = parse_arrow_filename(name)
        motion = parsed['motion'] if parsed else 'unknown'
        by_motion[motion].append(name)

    rng = np.random.default_rng(subsample_seed)
    sampled: List[str] = []
    motions = sorted(by_motion.keys())
    per_motion = max(1, subsample_n // len(motions))
    for motion in motions:
        names = by_motion[motion]
        rng.shuffle(names)
        sampled.extend(names[:per_motion])

    if len(sampled) < subsample_n:
        remaining = [n for n in keep if n not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[:subsample_n - len(sampled)])
    sampled = sampled[:subsample_n]

    print(f"[whitelist] {len(sampled)} trajectories sampled from {len(keep)} "
          f"(stratified across {len(motions)} motion cases)")
    return set(sampled)


# ============================================================
# Trajectory loading
# ============================================================
def _file_passes_filters(parsed: Dict[str, Any],
                         mu_values: Optional[List[float]],
                         chi_values: Optional[List[float]],
                         motion_cases: Optional[List[str]]) -> bool:
    if mu_values is not None and parsed['mu'] not in mu_values:
        return False
    if chi_values is not None:
        if not any(abs(parsed['chi'] - cv) < 1e-4 for cv in chi_values):
            return False
    if motion_cases is not None and parsed['motion'] not in motion_cases:
        return False
    return True


def load_all_arrow_trajectories(data_dir: Path,
                                whitelist: Optional[set] = None,
                                mu_values: Optional[List[float]] = None,
                                chi_values: Optional[List[float]] = None,
                                motion_cases: Optional[List[str]] = None,
                                verbose: bool = True
                                ) -> List[Dict[str, Any]]:
    data_dir = Path(data_dir)
    arrow_files = sorted(data_dir.glob('*.arrow'))
    trajectories: List[Dict[str, Any]] = []

    skipped_whitelist = 0
    skipped_filter    = 0
    skipped_unparsed  = 0

    for fp in arrow_files:
        parsed = parse_arrow_filename(fp.name)
        if parsed is None:
            skipped_unparsed += 1
            continue
        if whitelist is not None and fp.name not in whitelist:
            skipped_whitelist += 1
            continue
        if not _file_passes_filters(parsed, mu_values, chi_values, motion_cases):
            skipped_filter += 1
            continue

        df = feather.read_feather(fp)
        try:
            S = df[_ARROW_STATE_COLS].to_numpy(dtype=np.float32)
            U = df[_ARROW_CONTROL_COLS].to_numpy(dtype=np.float32)
            Fblock = df[_ARROW_FORCE_COLS].to_numpy(dtype=np.float32)
            T = df[_ARROW_TIME_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
        except KeyError as e:
            print(f"[load] {fp.name}: missing column {e}; skipped")
            continue

        trajectories.append({
            'path':   str(fp),
            'name':   fp.name,
            'mu':     parsed['mu'],
            'chi':    parsed['chi'],
            'motion': parsed['motion'],
            'fc':     parsed['fc'],
            'states':   S,
            'controls': U,
            'forces':   Fblock,
            'times':    T,
        })

    if verbose:
        print(f"[load] kept={len(trajectories)} "
              f"skipped_whitelist={skipped_whitelist} "
              f"skipped_filter={skipped_filter} "
              f"skipped_unparsed={skipped_unparsed}")
        if trajectories:
            t0 = trajectories[0]
            print(f"[load] sample shape: states={t0['states'].shape}, "
                  f"forces={t0['forces'].shape}")
    return trajectories


def load_ood_test_trajectories(data_dir: Path,
                               ood_motion_cases: Optional[List[str]] = None,
                               ood_mu_values: Optional[List[float]] = None,
                               ood_chi_values: Optional[List[float]] = None,
                               whitelist: Optional[set] = None,
                               fraction: float = 1.0,
                               seed: Optional[int] = None,
                               verbose: bool = True
                               ) -> List[Dict[str, Any]]:
    """Load held-out trajectories along any of three axes (motion, mu, chi).

    Pass values you want INCLUDED in the OOD set (i.e. that were absent from
    training). Each filter is independent — None = no constraint. Whitelist
    still applies on top, in case you want OOD restricted to vetted files.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    trajs = load_all_arrow_trajectories(
        data_dir,
        whitelist=whitelist,
        mu_values=ood_mu_values,
        chi_values=ood_chi_values,
        motion_cases=ood_motion_cases,
        verbose=verbose,
    )

    if len(trajs) == 0:
        if verbose:
            print("[ood-loader] WARNING: empty OOD pool — check held-out filters")
        return trajs

    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(round(len(trajs) * fraction)))
        idx = rng.choice(len(trajs), size=n_keep, replace=False)
        trajs = [trajs[i] for i in idx]
        if verbose:
            print(f"[ood-loader] subsampled {n_keep} (fraction={fraction})")

    if verbose:
        from collections import Counter
        m_ct = Counter(t['motion']            for t in trajs)
        u_ct = Counter(round(t['mu'],  3)     for t in trajs)
        c_ct = Counter(round(t['chi'], 4)     for t in trajs)
        print(f"[ood-loader] OOD set: {len(trajs)} trajectories")
        print(f"  motion cases : {dict(m_ct)}")
        print(f"  mu values    : {dict(u_ct)}")
        print(f"  chi values   : {dict(c_ct)}")

    return trajs


# ============================================================
# Train/val/test stratified split
# ============================================================
def stratified_split(trajectories: List[Dict[str, Any]],
                     train_ratio: float, val_ratio: float,
                     seed: int = 42) -> Tuple[List, List, List]:
    """Stratify by (motion, mu, chi); shuffle within stratum; chunk by ratios."""
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio >= 0, f"train+val={train_ratio+val_ratio} > 1.0"

    rng = np.random.default_rng(seed)
    strata: Dict[Tuple, List[int]] = defaultdict(list)
    for i, t in enumerate(trajectories):
        strata[(t['motion'], t['mu'], t['chi'])].append(i)

    tr_idx, va_idx, te_idx = [], [], []
    for key, idxs in strata.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            tr_idx.extend(idxs)
            continue
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(0, int(round(n * val_ratio)))
        n_te = max(0, n - n_tr - n_va)
        if n_te == 0 and n - n_tr > 1:
            n_va = n - n_tr - 1
            n_te = 1
        tr_idx.extend(idxs[:n_tr])
        va_idx.extend(idxs[n_tr:n_tr + n_va])
        te_idx.extend(idxs[n_tr + n_va:])

    tr = [trajectories[i] for i in tr_idx]
    va = [trajectories[i] for i in va_idx]
    te = [trajectories[i] for i in te_idx]
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} "
          f"strata={len(strata)}")
    return tr, va, te


# ============================================================
# Sliding-window dataset
# ============================================================
class MecanumTrajectoryDataset(Dataset):
    """Sliding-window samples over a list of trajectories.

    Parameters
    ----------
    trajectories : list of trajectory dicts (from load_all_arrow_trajectories)
    seq_len      : window length in timesteps
    stride       : timesteps between window starts (1 = every step starts a window)
    """
    def __init__(self, trajectories: List[Dict[str, Any]],
                 seq_len: int, stride: int = 1):
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

        # Scale with module-level numpy arrays. These are visible to spawned
        # workers because they re-import this module on startup.
        S      = traj['states'][start:end]            / state_max
        S_next = traj['states'][start + 1:end + 1]    / state_max
        U      = traj['controls'][start:end]          / control_max
        F_sim  = traj['forces'][start:end]            / force_max
        T      = traj['times'][start:end]

        T      = traj['times'][start:end]

        # Fold wheel angles θ_1..4 to ±π/12 to match the model's
        # atan2-wrapped prediction. The Mecanum friction model is π/6-
        # periodic in θ (12 rollers per wheel), so cumulative rotation is
        # physically irrelevant. Without folding the target here, state-loss
        # MSE on θ is dominated by tens of radians of cumulative rotation
        # the model can't (and shouldn't) match.
        S[..., 7:11]      = np.arctan2(
            np.sin(12.0 * S[..., 7:11]),
            np.cos(12.0 * S[..., 7:11]),
        ) / 12.0
        S_next[..., 7:11] = np.arctan2(
            np.sin(12.0 * S_next[..., 7:11]),
            np.cos(12.0 * S_next[..., 7:11]),
        ) / 12.0

        return (
            torch.from_numpy(S).float(),
            torch.from_numpy(U).float(),
            torch.from_numpy(T).float(),
            torch.from_numpy(S_next).float(),
            torch.from_numpy(F_sim).float(),
            torch.tensor(traj['mu'], dtype=torch.float32),
            torch.tensor(traj['chi'], dtype=torch.float32),
        )


def make_loader(ds: Dataset, config: Dict[str, Any], shuffle: bool,
                drop_last: bool) -> DataLoader:
    """Common DataLoader factory: applies pin_memory, persistent_workers,
    and prefetch_factor knobs from config."""
    kw = dict(
        batch_size=config['batch_size'],
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', True),
    )
    if config['num_workers'] > 0:
        kw['persistent_workers'] = config.get('persistent_workers', True)
        kw['prefetch_factor']    = config.get('prefetch_factor', 2)
    return DataLoader(ds, **kw)


def build_loaders_from_lists(train_trajs: List[Dict[str, Any]],
                             val_trajs:   List[Dict[str, Any]],
                             test_trajs:  List[Dict[str, Any]],
                             config: Dict[str, Any]
                             ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build three DataLoaders from already-split trajectory lists.

    Used both by `build_loaders` (which does the split itself) and by the
    figures / OOD code paths that recover the split from a manifest's
    recorded trajectory name lists. Either-or pattern means there is one
    place that owns the dataset+loader construction.
    """
    stride = config.get('stride', 1)
    tr_ds = MecanumTrajectoryDataset(train_trajs, seq_len=config['seq_len'], stride=stride)
    va_ds = MecanumTrajectoryDataset(val_trajs,   seq_len=config['seq_len'], stride=stride)
    te_ds = MecanumTrajectoryDataset(test_trajs,  seq_len=config['seq_len'], stride=stride)

    tr_loader = make_loader(tr_ds, config, shuffle=True,  drop_last=True)
    va_loader = make_loader(va_ds, config, shuffle=False, drop_last=False)
    te_loader = make_loader(te_ds, config, shuffle=False, drop_last=False)
    print(f"[data] windows: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} "
          f"(stride={stride})")
    return tr_loader, va_loader, te_loader


def build_loaders(trajectories: List[Dict[str, Any]],
                  config: Dict[str, Any]
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified-split the trajectories and build three DataLoaders.

    Convenience wrapper that drops the trajectory lists. If you need the
    split lists too (e.g. to write a manifest), call build_loaders_with_split.
    """
    tr, va, te = stratified_split(trajectories,
                                  config['train_ratio'], config['val_ratio'],
                                  seed=config['seed'])
    return build_loaders_from_lists(tr, va, te, config)


def build_loaders_with_split(trajectories: List[Dict[str, Any]],
                             config: Dict[str, Any]
                             ) -> Tuple[DataLoader, DataLoader, DataLoader,
                                        List[Dict], List[Dict], List[Dict]]:
    """Same as build_loaders but also returns the split trajectory lists.

    The trajectory lists are needed by run_main to write a manifest
    capturing exactly which files went into train/val/test.
    """
    tr, va, te = stratified_split(trajectories,
                                  config['train_ratio'], config['val_ratio'],
                                  seed=config['seed'])
    tr_loader, va_loader, te_loader = build_loaders_from_lists(tr, va, te, config)
    return tr_loader, va_loader, te_loader, tr, va, te


def filter_trajectories_by_name(trajectories: List[Dict[str, Any]],
                                names: Sequence[str]
                                ) -> List[Dict[str, Any]]:
    """Return the subset of `trajectories` whose 'name' is in `names`.

    Order follows the order of `names` so the figures stage gets a
    deterministic ordering when iterating manifest test_names.
    """
    by_name = {t['name']: t for t in trajectories}
    out = []
    missing = []
    for n in names:
        t = by_name.get(n)
        if t is None:
            missing.append(n)
        else:
            out.append(t)
    if missing:
        print(f"[filter] {len(missing)} of {len(names)} requested names not found "
              f"in trajectory list (first missing: {missing[0]})")
    return out
