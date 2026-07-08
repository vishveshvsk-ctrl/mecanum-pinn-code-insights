#!/usr/bin/env python
# =============================================================================
# config_v2.py — Observer v2 (γ-only) run configuration.
#
# Imports physical/scale constants from v1 config.py so there is one source of
# truth for geometry, masses, LuGre values, and normalization column order.
# v2 changes vs v1:
#   * single γ target (no zx/zy heads)
#   * 200-epoch 5-phase schedule (0.8x of the v1 250-epoch schedule)
#   * supervised floor pinned at W_SUP_MIN = 0.1 (never zero)
#   * ReduceLROnPlateau scheduler replaces per-phase lr_scale
#   * promoted roller-residual physics term
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Re-use all v1 physical / column-order / scaling constants (single source).
from .config import (
    H, L, R, RD, ROLLERS, TANH_K, DELTA, SIN_DELTA, COS_DELTA, TAN_DELTA, PX, PY,
    M_PLATFORM, M_WHEEL, J_WHEEL, IS, AX, AY, P1, P2,
    N_TOTAL, N_PER_ROLLER,
    LG_SIGMA0, LG_SIGMA1, LG_SIGMA2, LG_SIGMA0_S, LG_SIGMA1_S, LG_SIGMA2_S,
    LG_STICTION_RATIO, LG_V_STR, LG_W_STR, LG_EPS_REG,
    MS, MAX_TORQUE, WHEEL_SCALE, BODY_F_SCALE, BODY_M_SCALE, ROLLER_SCALE,
    M_BODY, M_BODY_INV, PRED_P95,
    SIM_HZ, TRAIN_HZ, DECIM, T_S, SSM_DT_MIN, SSM_DT_MAX, SSM_TAU_MIN, SSM_TAU_MAX,
    GLOBAL_COLS, PERWHEEL_MEAS_COLS, PERWHEEL_FEATURES,
    N_GLOBAL, N_PERWHEEL, N_WHEELS,
    AUX_COLS, WZ_P95,
    MU, CHI_GRID, CHI_TOL,
)

# v2 target is gamma only.
TARGET_STATES_V2 = ["gamma"]
TARGET_COL_V2 = {"gamma": "gamma{i}"}
N_STATES_V2 = len(TARGET_STATES_V2)

# Frozen gamma p95 scale (physical units) used for de-normalisation in the
# physics recompute.  This is overwritten when a max-norm scaler CSV is supplied.
GAMMA_P95_DEFAULT = 82.81

# v2 5-phase curriculum: 200 epochs total (80/24/40/24/32).
# Tuple: (phase_name, epochs).  LR is owned by ReduceLROnPlateau — no lr_scale here.
PHASE_SCHEDULE_V2: List[Tuple[str, int]] = [
    ("grounding",     80),
    ("phys_rampup",   24),
    ("overlap",       40),
    ("grnd_rampdown", 24),
    ("physics",       32),
]
W_SUP_MIN = 0.1   # supervised weight floor; asserted never zero


@dataclass
class ObserverConfigV2:
    # paths
    data_dir: Path = Path("../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    whitelist_csv: Path = Path("diagnostics_combined.csv")
    out_dir: Path = Path("observer_v1_py/runs")

    # model
    model: str = "ssm"               # {ssm, gru}
    window: int = 32
    d_model: int = 32
    state_dim: int = 6
    ssm_dt_min: float = SSM_DT_MIN
    ssm_dt_max: float = SSM_DT_MAX
    ssm_tau_min: float = SSM_TAU_MIN
    ssm_tau_max: float = SSM_TAU_MAX
    emb_dim: int = 4
    freeze_wheel_emb: bool = True
    head_hidden: int = 32

    # optimisation
    epochs: int = 0                  # 0 = derive from phase schedule (200)
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    precision: str = "auto"          # {auto, fp32, fp16, bf16}
    require_gpu: bool = False

    # data plumbing
    jobs: int = 4                    # dataloader workers, <=8
    shuffle_buffer: int = 20000
    window_stride: int = 1
    stride_frac: float = 0.0         # >0 overrides window_stride
    limit_files: int = 0
    cache_dir: str = ""              # decimated-500Hz .npz cache

    # split
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1234

    run_tag_override: str = ""
    norm_method: str = "max"         # v2 pins max-norm (frozen p95)
    scaler_csv: str = ""

    # regime / subset
    regime_name: str = "default"
    mu_values: List[float] = field(default_factory=lambda: [MU])
    chi_values: List[float] = field(default_factory=lambda: list(CHI_GRID))
    include_profiles: List[str] = field(default_factory=list)
    exclude_profiles: List[str] = field(default_factory=list)
    per_profile_cap: int = 0
    subsample_fraction: float = 1.0
    chi_stratify: bool = False
    matched_chi_quads_only: bool = False
    min_chi_per_combo: int = 3

    # S1/S2 fold
    train_fold: str = ""
    backbone_profiles: List[str] = field(default_factory=lambda:
        ["octagon", "spin_creep", "coupled_vomega"])
    redundant_S1: List[str] = field(default_factory=lambda: ["long_circle", "spiral_orbit"])
    redundant_S2: List[str] = field(default_factory=lambda: ["ellipse"])
    profiles_toml_dir: str = "trajectory_files_run_0p5_main/profiles"
    redundant_sample_frac: float = 1.0
    redundant_sample_profiles: List[str] = field(default_factory=list)

    # S3 χ-fold (disabled by default)
    chi_fold_test: float = -1.0

    # training curriculum / loss
    phases: str = "a1_5phase"        # v2 always uses the 5-phase ramp
    physics_loss: bool = True        # v2 enables physics by default
    physics_variant: str = "residual"  # {residual, integrated}
    warm_from: str = ""              # v1 checkpoint path for encoder warm-start

    # promoted roller term
    w_roller: float = 1.0
    roller_slip_weighting: bool = True
    gamma_high_slip_upweight: float = 3.0   # per-window supervised upweight

    # ReduceLROnPlateau (single scheduler across all phases)
    sched_factor: float = 0.5
    sched_patience: int = 10
    sched_min_lr: float = 1e-6
    sched_rel_threshold: float = 1e-4

    # device-target presets (mirrors v1 but batch defaults differ)
    vram_preset: int = 0             # 6 or 24; 0 = use explicit fields

    def resolved(self) -> "ObserverConfigV2":
        self.data_dir = Path(self.data_dir)
        self.whitelist_csv = Path(self.whitelist_csv)
        self.out_dir = Path(self.out_dir)
        if self.jobs > 8:
            raise ValueError("jobs > 8 violates the machine OOM cap")
        if self.norm_method != "max":
            raise ValueError("v2 pins norm_method='max' (frozen p95 scaler)")
        if not self.scaler_csv:
            raise ValueError("v2 requires scaler_csv (variable_scaler_percentiles.csv)")
        return self

    @property
    def eff_stride(self) -> int:
        if self.stride_frac > 0:
            return max(1, round(self.stride_frac * self.window))
        return self.window_stride

    @property
    def phase_plan(self) -> List[Tuple[str, int]]:
        """Return the effective (phase, epochs) list."""
        return list(PHASE_SCHEDULE_V2)

    @property
    def total_epochs(self) -> int:
        if self.epochs > 0:
            return self.epochs
        return sum(n for _, n in self.phase_plan)

    @property
    def in_dim(self) -> int:
        # per-wheel encoder input: globals + per-wheel feats + wheel embedding
        return N_GLOBAL + N_PERWHEEL + self.emb_dim

    @property
    def run_tag(self) -> str:
        if self.run_tag_override:
            return self.run_tag_override
        fold = self.train_fold or "S0"
        return f"{fold}_train_w{self.window}_gamma_v2_phys_max_norm"
