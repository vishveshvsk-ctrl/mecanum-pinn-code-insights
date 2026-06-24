#!/usr/bin/env python
# =============================================================================
# config.py — Approach 2 (state observer) constants + run configuration.
#
# Single source of truth for: the physical geometry needed to build measurable
# features (the gamma=0 slip surrogate + the theta-fold), the column contract,
# the feature/target layout, decimation, and the train/val/test split.
#
# Physical constants are mirrored from trajectory_files/base.toml and the
# notebook's structural SVectors (delta, wc_x, wc_y). They are NOT configurable
# here — editing physics happens in base.toml + a regenerated dataset, never in
# the Python observer. Listed explicitly so feature construction is consistent
# with how the labels were integrated (CLAUDE.md authority rule #1).
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (base.toml [platform.*] + notebook structural fields)
# ---------------------------------------------------------------------------
H = 0.235          # half-length (m)   geo.h
L = 0.15           # half-width  (m)   geo.l
R = 0.05           # wheel outer radius (m)
RD = 0.0355        # roller axle distance Ra/Rd (m)
ROLLERS = 12       # rollers per wheel
TANH_K = 60.0      # sawtooth_tanh steepness (notebook: const TANH_K = 60.0)

# delta = SVector(-pi/4, pi/4, pi/4, -pi/4)  (O-configuration, structural)
DELTA = np.array([-np.pi / 4, np.pi / 4, np.pi / 4, -np.pi / 4])
SIN_DELTA = np.sin(DELTA)            # (-, +, +, -)·(1/sqrt2)
COS_DELTA = np.cos(DELTA)            # all +1/sqrt2 (cos even)
TAN_DELTA = np.tan(DELTA)            # (-1, 1, 1, -1)
# wheel centres: wc_x = (h,h,-h,-h), wc_y = (l,-l,l,-l)
PX = np.array([H, H, -H, -H])        # px_i  (wc_x)
PY = np.array([L, -L, L, -L])        # py_i  (wc_y)

# ---------------------------------------------------------------------------
# Plant constants for the physics loss (base.toml + notebook LuGreParams).
# Only needed when physics_loss=True; the forces are recomputed in torch from
# the predicted states. LuGre values are the notebook struct defaults
# (:lugre_adamov, use_mindlin=True) — NOT in base.toml.
# ---------------------------------------------------------------------------
M_PLATFORM = 30.0
M_WHEEL = 1.4
J_WHEEL = 5.87e-3
AX = 1.6e-2          # COM offset X
AY = -2.6e-2         # COM offset Y
P1 = 0.11            # drivetrain viscous (friction_case 1)  -> wheel balance
P2 = 5.78e-3         # roller-bearing viscous (friction_case 1) -> roller balance
N_TOTAL = M_PLATFORM * 9.81
# per-roller normal load (run_one.jl N_per_roller); wheel order (h,l),(h,-l),(-h,l),(-h,-l)
N_PER_ROLLER = np.array([
    N_TOTAL / 4 * (1 + AX / H + AY / L) + M_WHEEL * 9.81,
    N_TOTAL / 4 * (1 + AX / H - AY / L) + M_WHEEL * 9.81,
    N_TOTAL / 4 * (1 - AX / H + AY / L) + M_WHEEL * 9.81,
    N_TOTAL / 4 * (1 - AX / H - AY / L) + M_WHEEL * 9.81,
])
# LuGre / Stribeck (:lugre_adamov, use_mindlin=True) — notebook struct defaults
LG_SIGMA0 = 1.64e3;  LG_SIGMA1 = 1.6;  LG_SIGMA2 = 0.0
LG_SIGMA0_S = 1.09e3; LG_SIGMA1_S = 1.1; LG_SIGMA2_S = 0.0
LG_STICTION_RATIO = 1.1
LG_V_STR = 0.01;  LG_W_STR = 0.01
LG_EPS_REG = 1e-4

# Physics-residual normalisation (residuals divided by these characteristic
# torques so roller vs wheel terms are comparable; overall scale set by w_phys).
MAX_TORQUE = 10.0      # wheel actuator cap (base.toml) -> wheel-balance scale
ROLLER_TAU = 0.5       # characteristic roller torque (~mu*N*(R-Rd)) -> roller scale

# 5-phase training curriculum (train_GPU_PINN_v14 forward schedule):
#   (name, epochs, lr_scale). Physics weight ramps 0->1 over phys_rampup, stays
#   1 thereafter; supervised weight ramps 1->W_SUP_MIN over grnd_rampdown.
#   With physics_loss=False the physics weight is forced 0 and supervised stays 1.
PHASE_SCHEDULE = [
    ("grounding",     100, 1.00),
    ("phys_rampup",    30, 0.25),
    ("overlap",        50, 0.50),
    ("grnd_rampdown",  30, 0.25),
    ("physics",        40, 0.10),
]
W_SUP_MIN = 0.1        # supervised weight floor during the physics-dominant phases

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SIM_HZ = 2000      # native sim grid (base.toml saveat_rate)
TRAIN_HZ = 500     # observer training rate (handoff §2; Fpar/Fperp recon 1-3%)
DECIM = SIM_HZ // TRAIN_HZ          # stride-4 decimation

# SSM selective-Δ scale wired to the sampling period T_s = 1/TRAIN_HZ, so
# re-discretising at another deployment rate is a one-line change (mirrors A1).
# Δ(x) is log-uniform in [T_s/4, 4·T_s] (nominal = geomean = T_s, selectivity
# kept in Δ). A's modes init to PHYSICAL relaxation times τ ∈ [T_s/5, 20·T_s];
# A = -1/τ, so per-step decay exp(Δ·A) spans sub-ms (quasi-static bristle) to
# ~tens-of-ms (body dynamics). Physical Δ REQUIRES physical A or decay≈1.
T_S = 1.0 / TRAIN_HZ               # 2.0 ms @ 500 Hz
SSM_DT_MIN = T_S / 4.0            # 0.5 ms
SSM_DT_MAX = T_S * 4.0            # 8.0 ms
SSM_TAU_MIN = T_S / 5.0          # 0.4 ms  (fastest A mode)
SSM_TAU_MAX = T_S * 20.0         # 40 ms   (slowest A mode)

# ---------------------------------------------------------------------------
# Column contract (datastore.jl assemble_dataframe — see CLAUDE.md §5)
# ---------------------------------------------------------------------------
# Measurable-only inputs (HARD RULE). Raw Arrow columns we read:
GLOBAL_COLS = ["Vx", "Vy", "psi_dot"]
PERWHEEL_MEAS_COLS = ["Msat_{i}", "w{i}", "theta{i}"]   # i in 1..4
# Hidden-state supervision targets (read from Arrow cols — JLD2 is not stored).
# zs (spin bristle) is DROPPED: its only footprint is Mz, which is low-SNR and
# unused, so zs is unobservable-by-design and not a target. 3 states -> 12 heads.
TARGET_STATES = ["gamma", "zx", "zy"]
TARGET_COL = {"gamma": "gamma{i}", "zx": "zx_{i}", "zy": "zy_{i}"}
# Auxiliary label columns used ONLY for evaluation/binning (never inputs):
AUX_COLS = ["wz_{i}", "Vpx_{i}", "Vpy_{i}"]

# Frozen global |p95| of per-wheel omega_z (rad/s), from the deployment scaler
# variable_scaler_percentiles.csv (row `wz`). Used to normalise the derived-omega_z
# eval metric (evaluation.py). REPLACES a per-file std(wz) normalisation, which was
# pathological at low spin (tiny denom -> exploded the low-|wz| bins) and made the
# omega_z metric incomparable to the p95-normalised gamma/zx/zy. Mirrors design
# decision #1 (frozen p95, deployment-norm).
WZ_P95 = 9.00671100616455

# Derived per-wheel measurable INPUT features (built in features.py).
# The gamma=0 slip surrogate (Vpx0/Vpy0) is NOT an input: Vpy0 is a pure linear
# combo of {Vy, psi_dot} and Vpx0 is linear in {Vx, psi_dot, w} except a single
# psi_dot*tan(theta_tilde) product the nonlinearity learns — feeding them just
# duplicates inputs. (They are still computed for the physics loss.)
PERWHEEL_FEATURES = ["Msat", "w", "sin_tt", "cos_tt"]
N_GLOBAL = len(GLOBAL_COLS)          # 3
N_PERWHEEL = len(PERWHEEL_FEATURES)  # 4
N_STATES = len(TARGET_STATES)        # 3 (gamma, zx, zy)
N_WHEELS = 4

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
MU = 0.5
CHI_GRID = (0.0, 0.002, 0.005, 0.008)
CHI_TOL = 1e-9

# ---------------------------------------------------------------------------
# VRAM / machine presets (applied by train_observer --vram; explicit CLI wins).
#   6  = RTX 3060 Mobile, 6 GB, 16 GB RAM, Ampere (bf16). Single run.
#   24 = Quadro RTX 6000, 24 GB, Turing (NO bf16 -> fp32). Per-run preset; run
#        several concurrently via launch_parallel.py to use the headroom.
# Both enable the decimated cache (the repeat-I/O killer). The model is tiny, so
# batch is a throughput knob, not a memory one (see README).
# ---------------------------------------------------------------------------
VRAM_PRESETS = {
    6:  dict(batch_size=2048, jobs=4, precision="auto", cache_dir="observer_v1_py/cache_decim"),
    24: dict(batch_size=4096, jobs=4, precision="auto", cache_dir="observer_v1_py/cache_decim"),
}


@dataclass
class ObserverConfig:
    # paths
    data_dir: Path = Path("../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    whitelist_csv: Path = Path("diagnostics_combined.csv")
    out_dir: Path = Path("observer_v1_py/runs")
    # model
    model: str = "ssm"               # {ssm, gru}
    window: int = 32                 # causal lookback (samples @ 500 Hz)
    d_model: int = 32                # encoder hidden width (channels); ~16 = handoff
                                     # validated size, 32 leaves headroom for first run
    state_dim: int = 6               # SSM state dim (~4-8; handoff §4.2)
    ssm_dt_min: float = SSM_DT_MIN   # selective-Δ bounds (wired to T_s); change these
    ssm_dt_max: float = SSM_DT_MAX   # (and tau) to re-discretise at another rate
    ssm_tau_min: float = SSM_TAU_MIN # A relaxation-time init range (physical)
    ssm_tau_max: float = SSM_TAU_MAX
    emb_dim: int = 4                 # wheel-embedding width
    freeze_wheel_emb: bool = True    # zero-init + frozen now (identical sim wheels);
                                     # unfreeze later for wheel asymmetry / sim-to-real
    head_hidden: int = 32
    # optimisation
    epochs: int = 40
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    precision: str = "auto"          # {auto, fp32, fp16, bf16}
    require_gpu: bool = False         # hard-fail if CUDA is unavailable
    # data plumbing
    jobs: int = 4                    # <= 8 hard cap (CLAUDE.md §8)
    shuffle_buffer: int = 20000
    window_stride: int = 1           # absolute window step (used if stride_frac<=0)
    stride_frac: float = 0.0         # >0 -> stride = round(frac*window) (e.g. 0.5)
    limit_files: int = 0             # >0 = debug subset
    cache_dir: str = ""              # "" = off; else decimated-500Hz .npz cache (reused
                                     # across W and regimes; the key repeat-I/O killer)
    # split (grouped by (profile, combo_idx) — no trajectory leakage)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1234

    run_tag_override: str = ""       # if set, IS the run_tag (the parallel launcher
                                     # pins it to the job label -> deterministic run dir)
    norm_method: str = "var"         # "var" = z-score (mean/std, fit on train); "max" = frozen p95 scaler
    scaler_csv: str = ""             # max-norm only: path to variable_scaler_percentiles.csv
    # --- regime / data-subset selection (TOML-driven; see observer_v1_py/regimes/) ---
    regime_name: str = "default"
    mu_values: List[float] = field(default_factory=lambda: [MU])   # μ levels to include
    chi_values: List[float] = field(default_factory=lambda: list(CHI_GRID))
    include_profiles: List[str] = field(default_factory=list)   # empty = all
    exclude_profiles: List[str] = field(default_factory=list)
    per_profile_cap: int = 0                # cap on (profile,combo) groups; 0 = none
    subsample_fraction: float = 1.0         # fraction of groups kept (learning curve)
    chi_stratify: bool = False              # prioritise multi-χ combos when capping
    matched_chi_quads_only: bool = False    # keep only combos spanning ≥ min_chi_per_combo χ
    min_chi_per_combo: int = 3
    # --- excitation-coverage 2-fold partition (S1/S2; see regimes/S?_train.toml) ---
    # When train_fold is set, train+val come from that fold (0.9/0.1) and the OTHER
    # fold is the cross-subset test. Backbone profiles are stratified-split 50/50 by
    # excitation mode; redundant profiles are assigned wholesale.
    train_fold: str = ""                    # "" disables; "S1" | "S2"
    backbone_profiles: List[str] = field(default_factory=lambda:
        ["octagon", "spin_creep", "coupled_vomega"])
    redundant_S1: List[str] = field(default_factory=lambda: ["long_circle", "spiral_orbit"])
    redundant_S2: List[str] = field(default_factory=lambda: ["ellipse"])
    profiles_toml_dir: str = "trajectory_files_run_0p5_main/profiles"  # combo→mode source
    # downsample the over-represented redundant profiles (coverage-neutral, so a
    # random skew barely matters) to balance fold sizes toward the smallest one.
    redundant_sample_frac: float = 1.0
    redundant_sample_profiles: List[str] = field(default_factory=list)
    # --- S3: χ k-fold cross-validation (each fold = one χ value) ---
    # When chi_fold_test >= 0, test = that χ; train+val = the other χ values.
    # Independent of the S1/S2 excitation folds; runs on the χ-swept profiles.
    chi_fold_test: float = -1.0
    # --- training curriculum / loss (wired in training.py) ---
    phases: str = "supervised"             # "supervised" | "a1_5phase"
    phase_total_epochs: int = 0            # 0 = full PHASE_SCHEDULE (250); else scale to this
    physics_loss: bool = False             # γ + wheel-balance residuals (χ per-sample)

    target_states: List[str] = field(default_factory=lambda: list(TARGET_STATES))

    def resolved(self) -> "ObserverConfig":
        self.data_dir = Path(self.data_dir)
        self.whitelist_csv = Path(self.whitelist_csv)
        self.out_dir = Path(self.out_dir)
        if self.jobs > 8:
            raise ValueError("jobs > 8 violates the machine OOM cap (CLAUDE.md §8)")
        return self

    @property
    def in_dim(self) -> int:
        # per-wheel encoder input: globals + per-wheel feats + wheel embedding
        return N_GLOBAL + N_PERWHEEL + self.emb_dim

    @property
    def eff_stride(self) -> int:
        if self.stride_frac > 0:
            return max(1, round(self.stride_frac * self.window))
        return self.window_stride

    @property
    def run_tag(self) -> str:
        if self.run_tag_override:                # launcher-pinned -> deterministic run dir
            return self.run_tag_override
        base = f"{self.model}_w{self.window}"
        if self.chi_fold_test >= 0:              # S3: per-held-out-χ run dir
            base = f"holdchi{self.chi_fold_test:.3f}_{base}"
        return base if self.regime_name == "default" else f"{self.regime_name}_{base}"
