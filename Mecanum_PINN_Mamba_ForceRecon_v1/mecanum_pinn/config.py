"""Config builder + VRAM tier selector + run_tag (Mamba ForceRecon v1).

Adapted from train_GPU_PINN_v14_py/mecanum_pinn/config.py. Key changes:
  - Data: new lugre_adamov sweep dir; mu-grid {0.3,0.5,0.8}; single chi per run
    (0.005); profile-based selection (replaces motion_cases); target_hz=500.
  - Architecture: SSM ("Mamba-lite") encoder knobs replace the GRU/dec_theta
    knobs; force_dim=8 (roller-frame Fpar/Fperp, Mz dropped); structured force
    law (multiplicative mu + spin-gated chi, dormant affine hook); inverse uses
    a short causal Delta-state window + a (mu_hat, chi_hat) linear readout.
  - Physics loss = discrete NE-integration residual (reverse-mode), so no JVP
    machinery. Final phase is physics-only (force labels are warm-up only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_SEED = 42


def _format_decimal_list(values, max_decimals: int = 4) -> str:
    needed = 0
    for v in values:
        s = f"{v:.{max_decimals}f}".rstrip("0")
        if "." in s:
            needed = max(needed, len(s.split(".")[1]))
    needed = max(needed, 1)
    return "_".join(f"{v:.{needed}f}".replace(".", "p")
                    for v in sorted(float(x) for x in values))


def build_run_tag(config: Dict[str, Any], prefix: str = "", suffix: str = "") -> str:
    profiles = config.get('profiles') or []
    prof_part = ('profiles_all' if not profiles
                 else f'profiles_{len(profiles)}' if len(profiles) > 4
                 else 'prof_' + '_'.join(sorted(map(str, profiles))))
    parts = []
    if prefix:
        parts.append(str(prefix).strip('_'))
    parts.append(prof_part)
    if config.get('mu_values'):
        parts.append('mu_' + _format_decimal_list(config['mu_values']))
    if config.get('chi_values'):
        parts.append('chi_' + _format_decimal_list(config['chi_values']))
    if suffix:
        parts.append(str(suffix).strip('_'))
    tag = '_'.join(parts)
    for bad in (' ', '/', '\\', ':'):
        tag = tag.replace(bad, '_')
    return tag


DUMMY_EPOCHS = {'grounding_epochs': 2, 'rampup_epochs': 1, 'overlap_epochs': 2,
                'rampdown_epochs': 1, 'physics_epochs': 2,
                'lbfgs_max_iter': 3, 'use_lbfgs': False}


def apply_dummy_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    if not config.get('dummy', False):
        return config
    config['forward'].update(DUMMY_EPOCHS)
    config['inverse'].update(DUMMY_EPOCHS)
    for stage in ('forward', 'inverse'):
        config[stage]['min_epochs'] = 1
        config[stage]['patience']   = 1
    config['stride'] = 5
    config['batch_size'] = 32
    config['compile_enabled'] = config.get('compile_enabled_dummy_override', False)
    print(f"[dummy] overrides applied (seq_len={config['seq_len']} "
          f"stride={config['stride']} batch={config['batch_size']})")
    return config


# Default active sweep dir (relative to code_insights/ project root). Forward
# slashes so Path resolves on BOTH Windows and Linux/WSL (a backslash literal is
# a normal filename char on POSIX -> glob would find nothing on the lab box).
_DEFAULT_DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
_DEFAULT_WHITELIST = "pinn_training_whitelist.txt"
_ALL_PROFILES = ('octagon', 'long_circle', 'coupled_vomega', 'ellipse',
                 'spiral_orbit', 'spin_creep',
                 'multisine_50percent_cap', 'multisine_75percent_cap')


def build_config(*,
                 vram_gb: int = 6,
                 data_dir: str = _DEFAULT_DATA_DIR,
                 whitelist_path: str = _DEFAULT_WHITELIST,
                 profiles=_ALL_PROFILES,
                 mu_values=(0.3, 0.5, 0.8),
                 chi_values=(0.005,),                 # single chi per run (enforced)
                 friction_models=('lugre_adamov',),
                 target_hz: float = 500.0,
                 regime_toml: Optional[str] = None,        # observer_v1_py/regimes/*.toml
                 whitelist_csv: str = 'diagnostics_combined.csv',
                 test_chi: Optional[float] = None,         # S3 held-out chi override
                 cache_dir: Optional[str] = None,          # decimated-500Hz .npz cache (''/None=off)
                 ckpt_dir: str = 'checkpoints_mamba_v1',
                 figure_dir: str = 'figures_mamba_v1',
                 run_tag: str = 'run01',
                 dummy: bool = False,
                 seed: int = _DEFAULT_SEED) -> Dict[str, Any]:
    SEQ_LEN = 5
    STRIDE  = 1

    if vram_gb == 6:
        BATCH, NW, PF, AMP = 1024, 6, 4, True
    elif vram_gb == 12:
        BATCH, NW, PF, AMP = 512, 4, 4, True
    elif vram_gb == 24:
        BATCH, NW, PF, AMP = 4096, 12, 8, False
    else:
        raise ValueError(f"vram_gb={vram_gb} not in {{6,12,24}}")
    print(f"[vram-cfg] {vram_gb}GB -> batch={BATCH} seq_len={SEQ_LEN} workers={NW} amp={AMP}")

    cfg: Dict[str, Any] = {
        'device': None, 'seed': seed, 'amp_enabled': AMP,
        'compile_enabled': True,
        'compile_mode_forward': 'default', 'compile_mode_inverse': 'reduce-overhead',

        # ---- Data ------------------------------------------------
        'data_dir':        Path(data_dir),
        'whitelist_path':  Path(whitelist_path),
        # regime-driven split (shared with Approach 2). If regime_toml is set, the
        # loader uses observer_v1_py/regimes selection instead of the internal split.
        'regime_toml':     regime_toml,
        'whitelist_csv':   Path(whitelist_csv),
        'project_root':    '.',                           # resolves profiles_toml_dir
        'test_chi':        test_chi,
        # decimated-trajectory cache (shared idiom with observer_v1_py); '' = off.
        # Reused across runs + concurrent workers — the per-run repeat-I/O killer.
        'cache_dir':       cache_dir or '',
        'profiles':        list(profiles),
        'mu_values':       list(mu_values),
        'chi_values':      list(chi_values),
        'friction_models': list(friction_models),
        'target_hz':       target_hz,            # downsample 2000 -> 500 Hz
        'seq_len':         SEQ_LEN, 'stride': STRIDE,
        'batch_size':      BATCH, 'num_workers': NW, 'pin_memory': True,
        'persistent_workers': True, 'prefetch_factor': PF,
        'train_ratio':     0.85, 'val_ratio': 0.10,

        # ---- Physics loss (discrete NE-integration residual) -----
        'k_steps':          4,        # rollout lookahead for the physics residual
        'dt':               1.0 / target_hz,
        'adaptive_scaling': True, 'ema_alpha': 0.05,
        'w_phys_min': 1e-4, 'w_phys_max': 1e-1,
        'w_cons':     0.0,            # MONITOR-ONLY: F_fwd<->F_inv consistency is
                                      # computed + logged (plotted) but NOT trained.
                                      # Training it pulls the inverse into a forward-clone,
                                      # destroying mu-agnostic force reading and the
                                      # F_fwd<->F_inv divergence used for (mu,chi)-change
                                      # detection. Do not raise without re-reading that rationale.
        # mu-ID is NOT trained -- it is a test-time residual readout (val + test only).
        # chi-ID is deferred.

        # ---- Checkpoints / LR ------------------------------------
        'ckpt_dir': ckpt_dir, 'figure_dir': figure_dir, 'run_tag': run_tag,
        'lr': 4e-3,

        # ---- Architecture ---------------------------------------
        'raw_state_dim': 11,          # Vx,Vy,psi_dot, w1..4, theta1..4 (measurable)
        'ctrl_dim':      4,           # Msat_1..4
        'force_dim':     8,           # Fpar_1..4, Fperp_1..4  (Mz dropped)
        'n_wheels':      4,
        'embed_dim':     4,           # per-wheel embedding (zero-init)

        # SSM (Mamba-S6 lean core) encoder — lift to D channels, N-dim state each,
        # diagonal selective scan (selective B, C, dt; fixed physical A), plain-PyTorch.
        'encoder':            'ssm',  # 'ssm' only (no GRU fallback this build)
        'ssm_d_model':        32,     # lift / model dim D (features -> D latent channels)
        'ssm_d_state':        16,     # per-channel SSM state size N (Mamba default ~16)
        'ssm_selective_dt':   True,   # input-dependent Delta (Mamba: selectivity is in dt)
        # Delta ANCHORED to the sampling step T_s = 1/target_hz: dt(x) swings log-uniformly
        # in [T_s/4, 4*T_s] (nominal ~ T_s), so the FIXED A becomes a physical rate (1/s).
        'ssm_dt_min':         (1.0 / target_hz) / 4.0,
        'ssm_dt_max':         (1.0 / target_hz) * 4.0,

        # Structured force head:  F = mu*N*[Phi_base + chi*|wz|*Phi_spin]
        'shape_hidden':       32,
        'force_four_term':    True,   # 4-term law mu(A+C*chi) + B(slip) + D(bristle);
                                      # False = drop B,D (pure mu(A+C*chi)) for ablation.

        # Inverse: short causal Delta-state window + (mu_hat,chi_hat) readout
        'inv_window':         3,      # k-2..k (denoised Delta-state); >=2
        'inv_hidden':         32,
        'param_readout':      'linear',  # linear LS in (a=mu, b=mu*chi)

        # ---- Stage schedules (force warm-up -> physics-only) -----
        'forward': {
            'grounding_epochs': 100, 'rampup_epochs': 30, 'overlap_epochs': 50,
            'rampdown_epochs': 30, 'physics_epochs': 40,
            'lr_scale': {'grounding': 1.0, 'phys_rampup': 0.25, 'overlap': 0.5,
                         'grnd_rampdown': 0.25, 'physics': 0.1},
            'patience': 8, 'min_epochs': 20, 'rel_delta': 1e-3,
            'use_lbfgs': True, 'lbfgs_max_iter': 60, 'lbfgs_lr': 0.5, 'grad_clip': 1.0,
        },
        'inverse': {
            'grounding_epochs': 60, 'rampup_epochs': 20, 'overlap_epochs': 40,
            'rampdown_epochs': 20, 'physics_epochs': 50,
            'lr_scale': {'grounding': 1.0, 'phys_rampup': 0.25, 'overlap': 0.5,
                         'grnd_rampdown': 0.25, 'physics': 0.1},
            'patience': 6, 'min_epochs': 15, 'rel_delta': 1e-3,
            'use_lbfgs': True, 'lbfgs_max_iter': 40, 'lbfgs_lr': 0.5, 'grad_clip': 1.0,
        },

        'dummy': dummy, 'dummy_n_trajectories': 8,
    }
    return cfg
