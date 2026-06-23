# PROJECT_LAYOUT.md — Mecanum PINN Digital Twin File Structure

**Root: `C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\`**  
**Last updated: 2026-06-23** (automated hierarchical layout sync)

---

## Top-Level Structure (275.6 GB total)

```
mecanum_pinn_head/
├── .claude/                          Claude Code settings (1.02 KB)
├── CLAUDE.md                         Root orientation (6.96 KB)
├── PROJECT_LAYOUT.md                 This file (17.96 KB)
├── code_insights/                    Source code & analysis (1.27 GB) ← MAIN WORKING DIR
└── data/                             Simulation data storage (274.35 GB)
```

---

## code_insights/ — Main Working Directory (1.27 GB)

All source code, notebooks, training packages, and diagnostics live here.

### Configuration & Metadata

| Path | Size | Purpose |
|------|------|---------|
| `.claude/` | 4.49 KB | Claude Code project settings |
| `.headroom/` | 184.0 KB | Headroom memory database (auto-generated) |
| `.ipynb_checkpoints/` | 1.77 MB | Jupyter checkpoint backups (auto-generated) |
| `__pycache__/` | 274.89 KB | Python bytecode cache (auto-generated) |
| `Project.toml` | 1.39 KB | Julia project manifest |
| `Manifest.toml` | 123.13 KB | Julia locked dependencies |

### Documentation & Technical References

| Path | Size | Purpose |
|------|------|---------|
| `CLAUDE.md` | 21.6 KB | **Detailed technical orientation** (physics, simulator, architecture) |
| `TRAJ_DIAGNOSTICRESULTS.md` | 46.2 KB | Summary of trajectory diagnostic results |
| `Trajectory_Chatter_Diagnostics_PLAN.md` | 21.9 KB | Analysis plan for oscillations/chatter in trajectories |
| `training_data_split_design.md` | 8.0 KB | Train/validation/test data split strategy |
| `mu_generation_note.md` | 5.6 KB | Notes on friction coefficient (μ) sweep generation |
| `docs/` | 9.0 MB | Reference papers, LaTeX docs, technical references |

### Julia Core Modules

| File | Size | Purpose |
|------|------|---------|
| `profiles.jl` | 41.6 KB | Trajectory profile library (velocity/position reference builders) |
| `datastore.jl` | 15.5 KB | Arrow I/O, schema definition, label extraction |
| `Data_Generation_Julia.jl` | 21.5 KB | Parallel sweep orchestration (profile × combo enumeration) |
| `run_one.jl` | 42.9 KB | Single-run ODE simulator (extracted from main notebook) |
| `run_one_nbinclude.jl` | 1.8 KB | Notebook-compatible wrapper for run_one.jl |

### Main Simulator Notebook (229.7 MB)

```
Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb (229.71 MB)
```

**The authoritative 39-dimensional ODE simulator:**
- LuGre + Adamov friction model
- ASMC (Adaptive Sliding Mode Control) + DOB (Disturbance Observer)
- Full supertwist dynamics
- Generates Arrow files with ground-truth labels (wheel forces, slip, torques)

**Critical:** Cell 2 is the ONLY parametrization entry (tagged `parameters`)

### Auxiliary Analysis Notebooks

| File | Size | Purpose |
|------|------|---------|
| `Mecanum_PINN_TrajectoryDiagnostics_1.ipynb` | 4.1 MB | Chatter, sampling, slip diagnostics |
| `Mecanum_Trajectory_Physics_Diagnostic_2_Markdown.ipynb` | 2.1 MB | Physics validation checks |
| `Trajectory_Chatter_Diagnostics_Profiles.ipynb` | 30.4 KB | Per-profile chatter analysis |
| `Solver_Ablation_Multisine.ipynb` | 1.6 MB | Solver performance study with multisine excitation |
| `Sampling_Rate_Sensitivity.ipynb` | 16.9 KB | Sampling rate sensitivity study |
| `scan_scaling_factors_windowed_v3.ipynb` | 54.8 KB | Scaling factor exploration |

### Python Diagnostics & Analysis Scripts

| File | Size | Purpose |
|------|------|---------|
| `chatter_diagnostics.py` | 30.6 KB | Detect and characterize trajectory oscillations |
| `chi_identifiability.py` | 13.6 KB | Analyze friction parameter χ (chi) identifiability |
| `mu_identifiability.py` | 18.4 KB | Analyze friction parameter μ (mu) identifiability |
| `roller_slip_fraction.py` | 14.7 KB | Compute per-wheel slip fractions |
| `force_mu_chi_gated.py` | 14.5 KB | Gated force regression (μ, χ recovery) |
| `force_mu_chi_regression.py` | 10.2 KB | Direct force regression (μ, χ recovery) |
| `tracking_gate.py` | 11.3 KB | Tracking error gating for diagnostics |
| `parallel_sweep.py` | 14.1 KB | Parallel sweep driver for diagnostics |
| `sampling_sensitivity.py` | 10.6 KB | Sampling rate sensitivity analysis |
| `dataset_chunker.py` | 5.1 KB | Arrow dataset chunking utility |
| `blend_reports.py` | 4.4 KB | Report blending/merging |
| `extract_run_one.py` | 7.2 KB | Extract notebook code → run_one.jl |
| `make_forcerecon_flowchart.py` | 11.2 KB | Generate force-reconstruction flow diagrams |
| `render_deck.py` | 4.0 KB | Render presentation deck to HTML |
| `keep_awake.py` | 1.7 KB | Background wake-lock (prevent Modern Standby during long sweeps) |
| `_read_nb_helper.py` | 2.0 KB | Notebook reading helper |

### Test Scripts

| File | Size | Purpose |
|------|------|---------|
| `test_chatter_diagnostics.py` | 5.8 KB | Unit tests for chatter diagnostics |
| `test_sampling_sensitivity.py` | 3.6 KB | Unit tests for sampling analysis |

### Analysis & Diagnostic Outputs

| File | Size | Purpose |
|------|------|---------|
| `chatter_report.csv` | 1.8 MB | Chatter metrics per profile/combo |
| `diagnostics_combined.csv` | 4.6 MB | Combined diagnostics across all runs |
| `chi_identifiability.csv` | 18.6 KB | χ identifiability results (all μ) |
| `chi_identifiability_mu03.csv` | 8.4 KB | χ identifiability @ μ=0.3 |
| `chi_identifiability_mu05.csv` | 12.1 KB | χ identifiability @ μ=0.5 |
| `chi_identifiability_mu08.csv` | 13.8 KB | χ identifiability @ μ=0.8 |
| `mu_identifiability.csv` | 21.4 KB | μ identifiability results |
| `force_mu_chi_gated.csv` | 202.2 KB | Gated force regression results |
| `force_mu_chi_regression.csv` | 1010 B | Direct force regression results |
| `roller_slip_fraction.csv` | 32.3 KB | Per-wheel slip fraction data |
| `sampling_sensitivity.csv` | 2.8 MB | Sampling rate sensitivity data |
| `tracking_report.csv` | 1.1 MB | Tracking error metrics |

### Trajectory Configuration Sets

Each directory has `base.toml` (physics/solver settings) + `profiles/` subdirectory (trajectory .toml files).

#### Primary Simulation Runs

| Directory | Size | Profiles | μ Values | Purpose |
|-----------|------|----------|----------|---------|
| `trajectory_files_run_0p3_main/` | 136.8 KB | 9 profiles | μ=0.3 | Main run @ low friction |
| `trajectory_files_run_0p8_main/` | 143.7 KB | 9 profiles | μ=0.8 | Main run @ high friction |
| `trajectory_files_run_0p5_main/` | 395.3 KB | 24 profiles | μ=0.3,0.5,0.8 | Comprehensive grid |

#### Quad (Reduced) Sample Sets

| Directory | Size | Profiles | μ Values | Purpose |
|-----------|------|----------|----------|---------|
| `trajectory_files_run_0p3_quad/` | 50.3 KB | 3 profiles* | μ=0.3 | Subset: coupled_vomega, octagon, spin_creep |
| `trajectory_files_run_0p8_quad/` | 57.1 KB | 3 profiles* | μ=0.8 | Subset: coupled_vomega, octagon, spin_creep |
| `trajectory_files_run_0p5_quad/` | 8.7 KB | 1 profile | μ=0.5 | Single octagon sample |

#### μ Sweep Pilots & Variants

| Directory | Size | Profiles | Purpose |
|-----------|------|----------|---------|
| `trajectory_files_mupilot_0p3/` | 19.4 KB | 2 profiles | Early μ sweep exploration @ μ=0.3 |
| `trajectory_files_mupilot_0p8/` | 29.9 KB | 2 profiles | Early μ sweep exploration @ μ=0.8 |
| `trajectory_files_mu0p3/` | 112.5 KB | 8 profiles | μ-specific run @ μ=0.3 |
| `trajectory_files_mu0p8/` | 123.2 KB | 8 profiles | μ-specific run @ μ=0.8 |

#### Specialized Studies

| Directory | Size | Profiles | Purpose |
|-----------|------|----------|---------|
| `trajectory_files_chinc/` | 37.5 KB | 3 profiles | χ (chi) parameter sensitivity study |
| `trajectory_files_scpilot_0p3/` | 91.1 KB | 3 profiles* | Spin-creep study @ μ=0.3 |
| `trajectory_files_scpilot_0p8/` | 85.7 KB | 3 profiles* | Spin-creep study @ μ=0.8 |

*multisine50%, multisine75%, spin_creep

### Training & Model Packages

#### Train GPU PINN v14 (1.2 MB)

```
train_GPU_PINN_v14_py/
├── mecanum_pinn/                           (446.8 KB) Core training package
│   ├── config.py         (9.3 KB)   Config management
│   ├── data.py           (19.7 KB)  Arrow dataset loading
│   ├── evaluation.py     (15.9 KB)  Evaluation metrics
│   ├── losses.py         (16.9 KB)  Physics-informed loss functions
│   ├── manifest.py       (7.8 KB)   Experiment manifest tracking
│   ├── models.py         (35.5 KB)  Neural network architectures
│   ├── physics.py        (7.0 KB)   Physics constraint implementation
│   ├── plotting.py       (6.6 KB)   Visualization utilities
│   ├── stages.py         (24.8 KB)  Training stages (grounding, physics, etc.)
│   ├── training.py       (25.6 KB)  Main training loop
│   ├── trajectory_eval.py (31.6 KB) Trajectory-level evaluation
│   ├── CHANGES_v12_5.md  (16.7 KB)  Version history
│   └── __pycache__/      (225.9 KB)
├── train.py              (4.6 KB)   Entry point
├── make_manifest.py      (6.9 KB)   Manifest generation utility
├── plot_ood.py           (10.4 KB)  Out-of-distribution plot generation
├── README.md             (9.8 KB)
├── CLI_QUICKREF.md       (6.0 KB)
├── CHANGES_v13.md        (9.4 KB)
├── pinn_training_whitelist.txt (90.2 KB) Allowed data files
└── figures_v13/          (608.6 KB)  Generated OOD figures
```

#### Mecanum PINN Mamba ForceRecon v1 (449.1 KB)

```
Mecanum_PINN_Mamba_ForceRecon_v1/
├── mecanum_pinn/                           (415.2 KB) Core force recon package
│   ├── 12 modules: config, data, models, losses, training, regime_split, etc.
│   ├── models.py         (16.8 KB)  Mamba SSM architecture
│   ├── regime_split.py   (12.9 KB)  Slip/stick regime classification
│   ├── stages.py         (14.1 KB)  Training stages
│   ├── training.py       (12.2 KB)  Main training loop
│   └── __pycache__/      (304.6 KB)
├── train.py              (1.6 KB)   Entry point
├── launch_parallel.py    (7.4 KB)   Parallel training launcher
├── smoke_test.py         (2.3 KB)   Smoke test suite
└── README.md             (6.2 KB)
```

#### Observer v1 (2.8 MB)

```
observer_v1_py/
├── mecanum_observer/                       (270.6 KB) Core observer package
│   ├── config.py         (13.5 KB)  Configuration
│   ├── data.py           (22.1 KB)  Data loading
│   ├── models.py         (7.1 KB)   Observer architectures
│   ├── training.py       (12.0 KB)  Training loop
│   ├── evaluation.py     (6.9 KB)   Evaluation metrics
│   ├── features.py       (4.8 KB)   Feature engineering
│   └── __pycache__/      (195.6 KB)
├── train_observer.py     (6.0 KB)   Entry point
├── launch_parallel.py    (5.2 KB)   Parallel launcher
├── make_observability_report.py (7.2 KB) Report generation
├── refine_physics_lbfgs.py (8.5 KB) Physics refinement
├── build_variable_percentiles.py (5.3 KB) Variable statistics
├── make_pipeline_flowchart_state_recon.py (6.5 KB) Diagram generation
├── README.md             (6.7 KB)
├── regimes/              (9.3 KB)   Configuration sets (7 TOML files)
├── runs/                 (1.8 MB)   Trained checkpoints (6 window-size variants)
│   └── S1_train_w16_non_phys_var_norm/, S1_train_w32_*, S2_train_*, etc.
└── report/               (579.9 KB) Observability analysis figures & data
```

### Trained Model Checkpoints

| Directory | Size | Contents |
|-----------|------|----------|
| `checkpoints_mamba_v1/run01/` | 515.9 KB | Mamba force-recon trained weights (5 forward + 5 inverse stages) |
| `observer_v1_py/runs/` | 1.8 MB | Observer trained checkpoints (6 window-size variants + metrics) |

### Visualization & Presentation

| Directory | Size | Contents |
|-----------|------|----------|
| `rendered_traj_diagnostics/` | 18.7 MB | Per-profile diagnostic HTML reports (14 files) |
| `images_and_plots/` | 6.2 MB | PNG/SVG figures for papers & presentation deck |
| `presentation/` | 7.4 MB | IMECE deck (HTML + PDF 4.9 MB) + assets |

### Solver Benchmarks

```
julia_solver_benchmark_asmc4/                    (988.4 MB)
├── Reference JLD2 files (13 files, ~73 MB each)
│   ├── ref_ms50_fhi1.0_mu0.5_chi0.005_lugre_adamov_r4000_tol1e-10.jld2
│   ├── ref_ms50_fhi1.0_mu0.5_chi0.005_lugre_adamov_r4000_tol1e-12.jld2
│   ├── ... (multisine 50%, multisine 75%, spiral_iso variants)
│
├── Executed Benchmark Notebooks
│   ├── _full_run_executed.ipynb       (1.7 MB)
│   ├── _ref_radau9_executed.ipynb     (560.0 KB)
│   ├── _stageA_run_executed.ipynb     (545.6 KB)
│   ├── _stageB_extend_executed.ipynb  (1.2 MB)
│
├── Analysis & Plots
│   ├── stageA_dtmax_subsidy.png       (33.4 KB)
│   ├── stageA_workprecision_nf.png    (51.0 KB)
│   ├── stageB_margins.png, .svg       (48.9 KB + 126.0 KB)
│   ├── stageA_results.arrow           (11.7 KB)
│   ├── stageB_results.arrow           (10.2 KB)
│   ├── stageB_radau9ref_results.arrow (6.7 KB)
│
├── Documentation
│   ├── SOLVER_SELECTION.md            (12.8 KB)
│   ├── results_note.md                (1.3 KB)
│   ├── solver_block_suggestion.toml   (394 B)
```

**Purpose:** Reference implementations of multisine and spiral trajectories with Radau solvers (tol=1e-10 and 1e-12) for benchmarking PINN prediction error.

### Cross-Session Handoff Briefs (79.8 KB)

Preserved task handoffs for continuity across chat sessions:

```
chat-handoff/
├── A2_observer_parallelism_handoff.md (7.1 KB)
├── approach2_state_observer_handoff.md (7.5 KB)
├── chi_regeneration_handoff.md (6.0 KB)
├── diagnostics_tuning_handoff.md (6.8 KB)
├── forcerecon_v1_parallelism_handoff.md (10.3 KB)
├── mamba_forcerecon_v1_training_handoff.md (6.2 KB)
├── mu_identifiability_handoff.md (5.0 KB)
├── mu_identifiability_handoff_v2.md (6.6 KB)
├── mu_scaling_pipeline_handoff.md (7.7 KB)
├── mu_trajectory_generation_handoff.md (4.8 KB)
├── octagon_chi_relocation_handoff.md (5.3 KB)
└── roller_slip_fraction_handoff.md (6.5 KB)
```

### Temporary & Development (_tmp/ — 7.7 MB)

Exploration, debugging, and intermediate artifacts (keep clean):

```
_tmp/
├── nb_figs/                          (4.7 MB) Generated diagnostic figures
├── sweep_logs/                       (898.4 KB) Orchestration logs (40+ files)
├── obs_runs_smoke/                   (190.3 KB) Observer smoke test results
├── bench_backup_20260614/            (18.3 KB) Old benchmark backup
├── mu_goldfull/                      (109.8 KB) μ full sweep config
├── mu_goldcheck/                     (18.2 KB) μ check sweep config
├── sc_gold/                          (80.7 KB) Spin-creep gold config
├── [various .py, .jl, .sh, .log, .csv] — Exploratory scripts
├── _read_nb_helper_eval/             (27.3 KB) Notebook skill evaluation
└── [empty temp files]
```

### Skill & Helper Utilities

| Directory | Size | Purpose |
|-----------|------|---------|
| `nb-read-skill/` | 7.7 KB | Notebook reading skill (strips images, truncates logs) |
| `_nb_read_eval/` | 27.3 KB | Skill evaluation results |

---

## data/ — Simulation Data Storage (274.3 GB)

**Data files are READ-ONLY.** New sweep outputs require new config or combo IDs; do not overwrite existing .arrow files.

### Active Simulation Sweep

```
Simulation_Data_MecanumSlipSpin_LugreAdamov/    (221.8 GB)
└── ~5670 Arrow (.arrow) files
    ├── Filename contract: <profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
    ├── Example: octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
    │
    └── Encoding:
        ├── <profile>       = trajectory name (octagon, coupled_vomega, ellipse, etc.)
        ├── c<combo>        = 0-padded combo ID (combo counter)
        ├── mu_<mu>         = friction coefficient (0.3, 0.5, 0.8, etc.)
        ├── case<fc>_<fm>   = case ID (friction model indicators)
        └── chi_<chi>       = parameter χ (chi), 3 decimal places
```

### Pilot & Legacy Runs

| Directory | Size | Status | Note |
|-----------|------|--------|------|
| `_mu_pilot2/` | 6.1 GB | Reference only | Pilot run 2 (historical) |
| `SimulationDataSlipSpin_Julia/` | 3.4 GB | **DEPRECATED** | Legacy beta/amplitude grid scheme |
| `SimulationDataSlipSpin_Julia_3/` | 43.0 GB | **DEPRECATED** | Legacy beta/amplitude grid scheme |

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Total Size** | 275.6 GB |
| **code_insights/** | 1.27 GB |
| **data/** | 274.35 GB |
| **Main Simulator Notebook** | 229.71 MB |
| **Julia Solver Benchmark** | 988.36 MB |
| **Rendered Diagnostics** | 18.69 MB |
| **Presentation Deck** | 7.40 MB |
| **Training Packages** | ~4.5 MB (v14, Mamba, Observer) |
| **Active Arrow Files** | ~5,670 files in main sweep |
| **Handoff Briefs** | 13 documents |

---

## Authority Rules (Reference)

1. **base.toml** — Single source of truth for physics & solver settings (per run)
2. **Module Copies** — If profiles.jl or datastore.jl are updated, replace all copies
3. **Execution Context** — All runs execute from code_insights/; CONFIG_DIR resolves to trajectory_files_run_* dirs
4. **Arrow Files** — Never hand-edit output files; existing .arrow = simulation complete
5. **Notebook Parametrization** — **Only Cell 2** in main simulator has the `parameters` tag
6. **Data Immutability** — Arrow files in data/ are read-only; new sweeps require new IDs
7. **Profile-Based Enumeration** — Beta/amplitude grid is DEPRECATED; use profile names

---

## For Reference

- **Root orientation:** [CLAUDE.md](CLAUDE.md) (root)
- **Technical details:** [code_insights/CLAUDE.md](code_insights/CLAUDE.md)
- **Physics & simulator:** code_insights/CLAUDE.md §§1–4
- **Data contract:** code_insights/CLAUDE.md §5
- **PINN intent:** code_insights/CLAUDE.md §7

---

**Last updated: 2026-06-23** (automated hierarchical layout sync)
