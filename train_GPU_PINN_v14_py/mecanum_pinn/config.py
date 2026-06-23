"""Config builder + VRAM tier selector + dummy overrides + run_tag.

v12.5 — defaults for the factored four-head forward model plus the
per-wheel factored inverse model. Both models carry zero-initialized
per-wheel learnable embeddings (`embed_dim` controls dimensionality)
which let the network specialize per wheel at sim-to-real fine-tuning
time without architectural surgery.

Key knobs added in v13:
  hidden_dim_wheel : per-wheel GRU hidden dim (compound H = h_wheel · n_wheels)
  dec1_hidden      : friction-head width in the forward model
  dec3_hidden      : ω-head width in the forward model
  dec_theta_hidden : θ-head width in the forward model
  embed_dim        : per-wheel embedding dim, shared by forward and inverse
  inv_hidden       : per-wheel inverse network width
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


_DEFAULT_SEED = 42


# ============================================================
# Run-tag builder
# ============================================================
def _format_decimal_list(values, max_decimals: int = 4) -> str:
    """Decimal-preserving list-to-string for run_tags."""
    needed = 0
    for v in values:
        s = f"{v:.{max_decimals}f}".rstrip("0")
        if "." in s:
            d = len(s.split(".")[1])
            needed = max(needed, d)
    needed = max(needed, 1)
    parts = [f"{v:.{needed}f}".replace(".", "p")
             for v in sorted(float(x) for x in values)]
    return "_".join(parts)


def build_run_tag(config: Dict[str, Any],
                  prefix: str = "",
                  suffix: str = "",
                  *,
                  max_motions_listed: int = 4,
                  include_mu:  bool = True,
                  include_chi: bool = True) -> str:
    motions = config.get('motion_cases') or []
    if not motions:
        motion_part = 'motion_all'
    elif len(motions) > max_motions_listed:
        motion_part = f'motion_{len(motions)}cases'
    else:
        motion_part = 'motion_' + '_'.join(sorted(str(m) for m in motions))

    mu_part  = ''
    chi_part = ''
    if include_mu:
        mu_vals = config.get('mu_values') or []
        if mu_vals:
            mu_part = 'mu_' + _format_decimal_list(mu_vals)
    if include_chi:
        chi_vals = config.get('chi_values') or []
        if chi_vals:
            chi_part = 'chi_' + _format_decimal_list(chi_vals)

    parts = []
    if prefix:
        parts.append(str(prefix).strip('_'))
    parts.append(motion_part)
    if mu_part:
        parts.append(mu_part)
    if chi_part:
        parts.append(chi_part)
    if suffix:
        parts.append(str(suffix).strip('_'))
    tag = '_'.join(parts)
    for bad in (' ', '/', '\\', ':'):
        tag = tag.replace(bad, '_')
    return tag


# ============================================================
# Dummy / smoke-test overrides
# ============================================================
DUMMY_FORWARD_EPOCHS = {
    'grounding_epochs': 2,
    'rampup_epochs':    1,
    'overlap_epochs':   2,
    'rampdown_epochs':  1,
    'physics_epochs':   2,
    'lbfgs_max_iter':   3,
    'use_lbfgs':        False,
}
DUMMY_INVERSE_EPOCHS = {
    'grounding_epochs': 2,
    'rampup_epochs':    1,
    'overlap_epochs':   2,
    'rampdown_epochs':  1,
    'physics_epochs':   2,
    'lbfgs_max_iter':   3,
    'use_lbfgs':        False,
}


def apply_dummy_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates config in place when dummy=True. Idempotent."""
    if not config.get('dummy', False):
        return config
    config['forward'].update(DUMMY_FORWARD_EPOCHS)
    config['inverse'].update(DUMMY_INVERSE_EPOCHS)
    config['forward']['min_epochs'] = 1
    config['inverse']['min_epochs'] = 1
    config['forward']['patience']   = 1
    config['inverse']['patience']   = 1
    config['stride']     = 5
    config['batch_size'] = 32
    config['compile_enabled'] = config.get('compile_enabled_dummy_override', False)
    config['amp_enabled']     = config.get('amp_enabled', True)
    print(f"[dummy] overrides applied (seq_len={config['seq_len']} "
          f"stride={config['stride']} batch={config['batch_size']} "
          f"amp={config['amp_enabled']} compile={config['compile_enabled']})")
    return config


# ============================================================
# Main config builder
# ============================================================
def build_config(*,
                 vram_gb: int = 6,
                 data_dir: str = r"G:\My Drive\SimulationDataSlipSpin_Julia_2",
                 whitelist_path: str = r"G:\My Drive\pinn_training_whitelist.txt",
                 motion_cases=('infinity', 'circle'),
                 mu_values=(0.5, 0.6),
                 chi_values=(0.000, 0.002, 0.005),
                 ckpt_dir: str = 'checkpoints_v14',
                 figure_dir: str = 'figures_v14',
                 run_tag: str = 'run01',
                 dummy: bool = False,
                 seed: int = _DEFAULT_SEED) -> Dict[str, Any]:
    """Build the config dict and apply VRAM tier sizing."""
    SEQ_LEN = 5
    STRIDE  = 1

    if vram_gb == 6:
        BATCH_SIZE = 1024
        NUM_WORKERS, PREFETCH = 6, 4
        AMP_ENABLED = True
    elif vram_gb == 12:
        BATCH_SIZE = 512
        NUM_WORKERS, PREFETCH = 4, 4
        AMP_ENABLED = True
    elif vram_gb == 24:
        BATCH_SIZE = 4096
        NUM_WORKERS, PREFETCH = 12, 8
        AMP_ENABLED = False
    else:
        raise ValueError(
            f"vram_gb={vram_gb} not in {{6, 12, 24}}."
        )
    print(f"[vram-cfg] VRAM={vram_gb} GB -> batch={BATCH_SIZE} seq_len={SEQ_LEN} "
          f"stride={STRIDE} workers={NUM_WORKERS} prefetch={PREFETCH} "
          f"amp={AMP_ENABLED}")

    cfg: Dict[str, Any] = {
        # Environment
        'device':          None,
        'seed':            seed,

        # AMP / mixed precision
        'amp_enabled':     AMP_ENABLED,

        # torch.compile
        'compile_enabled':      True,
        'compile_mode_forward': 'default',
        'compile_mode_inverse': 'reduce-overhead',

        # Data loader
        'data_dir':           Path(data_dir),
        'whitelist_path':     Path(whitelist_path),
        'mu_values':          list(mu_values),
        'chi_values':         list(chi_values),
        'motion_cases':       list(motion_cases),
        'seq_len':            SEQ_LEN,
        'stride':             STRIDE,
        'batch_size':         BATCH_SIZE,
        'num_workers':        NUM_WORKERS,
        'pin_memory':         True,
        'persistent_workers': True,
        'prefetch_factor':    PREFETCH,
        'train_ratio':        0.85,
        'val_ratio':          0.10,

        # Physics loss
        'tau':                0.01,
        'k_steps':            4,
        'adaptive_scaling':   True,
        'ema_alpha':          0.05,

        # Loss weights
        'w_phys_min':         1e-4,
        'w_phys_max':         1e-1,
        'w_cons':             0.0,

        # Checkpoints + figures
        'ckpt_dir':           ckpt_dir,
        'figure_dir':         figure_dir,
        'run_tag':            run_tag,

        # Base LR
        'lr':                 4e-3,

        # ----- Architecture --------------------------------------
        'raw_state_dim':      11,
        'net_state_dim':      15,        # kept for backward compat; unused by new inverse
        'ctrl_dim':           4,
        'force_dim':          12,
        'hidden_dim':         16,        # = n_wheels × hidden_dim_wheel
        'n_wheels':           4,
        # New v12.5 architecture knobs
        'hidden_dim_wheel':   4,
        'dec1_hidden':        32,
        'dec3_hidden':        8,
        'dec_theta_hidden':   8,
        'inv_hidden':         32,        # per-wheel inverse network width
        'embed_dim':          4,         # per-wheel embedding dim (zero-init).
                                          # Set to 0 to disable embeddings entirely.

        # Forward stage — full 5-phase schedule.
        'forward': {
            'grounding_epochs': 100, 'rampup_epochs':   30, 'overlap_epochs': 50,
            'rampdown_epochs':  30, 'physics_epochs':  40,
            'lr_scale':       {'grounding':     1.0, 'phys_rampup': 0.25,
                               'overlap':       0.5, 'grnd_rampdown': 0.25,
                               'physics':       0.1},
            'patience':         8,
            'min_epochs':       20,
            'rel_delta':        1e-3,
            'use_lbfgs':        True,
            'lbfgs_max_iter':   60,
            'lbfgs_lr':         0.5,
            'grad_clip':        1.0,
        },

        # Inverse stage
        'inverse': {
            'grounding_epochs': 60, 'rampup_epochs':   20, 'overlap_epochs': 40,
            'rampdown_epochs':  20, 'physics_epochs':  50,
            'lr_scale':       {'grounding':     1.0, 'phys_rampup': 0.25,
                               'overlap':       0.5, 'grnd_rampdown': 0.25,
                               'physics':       0.1},
            'patience':         6,
            'min_epochs':       15,
            'rel_delta':        1e-3,
            'use_lbfgs':        True,
            'lbfgs_max_iter':   40,
            'lbfgs_lr':         0.5,
            'grad_clip':        1.0,
        },

        # Dummy / smoke-test
        'dummy':                dummy,
        'dummy_n_trajectories': 8,
    }
    return cfg
