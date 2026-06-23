# CLAUDE.md — Mecanum PINN Digital Twin (Root-level orientation)

Orientation file for Claude Code. Quick reference for the project structure and key
conventions. For detailed technical documentation, see `code_insights/CLAUDE.md`.

---

## Project overview

A **Physics-Informed Neural Network (PINN) digital twin** for a KUKA youBot
four-Mecanum-wheel omnidirectional platform, targeting an **IMECE 2026** paper.

**Data pipeline:**
```
   JULIA                          PYTHON
   ODE simulator  ──► Arrow files ──► PyTorch PINN
   (39-D plant)     (+ JLD2)      (forward dynamics + friction ID)
```

Simulator generates richly-exciting trajectories with ground-truth labels (per-wheel friction 
forces, slips, controller torques); PINN learns forward dynamics and recovers friction parameters 
(μ, χ) by inverse identification.

---

## Root directory structure

```
mecanum_pinn_head/
├── .claude/                      Claude Code settings (1.02 KB)
├── CLAUDE.md                     ← This file (root orientation, 6.96 KB)
├── PROJECT_LAYOUT.md             ← Detailed hierarchical breakdown (17.96 KB)
├── code_insights/                ← MAIN WORKING DIR (1.27 GB)
└── data/                         ← ALL simulation data (274.35 GB)
```

**Directory responsibilities:**
- **code_insights/** — Simulator, notebooks, training packages, diagnostics, configs
- **data/** — Arrow files (274.3 GB), never written from code_insights/; pilot runs archived
- **code_insights/_tmp/** — Temporary exploration/scratch; clean up when done
- **code_insights/chat-handoff/** — Cross-session task handoff briefs (12 handoff documents)

---

## Project layout (full hierarchy)

See **[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)** for the complete current file hierarchy
(detailed breakdown with file sizes, descriptions, and 3-level nesting).

**Quick summary of code_insights/ (1.27 GB total):**

| Category | Size | Contents |
|----------|------|----------|
| **Main Simulator Notebook** | 229.71 MB | `Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb` (39-D ODE, LuGre friction, ASMC+DOB) |
| **Solver Benchmark** | 988.36 MB | `julia_solver_benchmark_asmc4/` (13 reference JLD2 files, 3 executed notebooks) |
| **Julia modules** | 121.9 KB | `profiles.jl` (41.56 KB), `datastore.jl` (15.46 KB), `Data_Generation_Julia.jl` (21.52 KB), `run_one.jl` (42.93 KB) |
| **Trajectory configs** | ~1.9 MB | `trajectory_files_run_0p*_main/`, `_quad/`, `_pilot/`, `_chinc/`, `_scpilot/` (13 config sets) |
| **PINN packages** | ~4.5 MB | `train_GPU_PINN_v14_py/` (1.16 MB), `Mecanum_PINN_Mamba_ForceRecon_v1/` (450.29 KB), `observer_v1_py/` (6.39 MB) |
| **Python diagnostics** | ~270 KB | `chatter_diagnostics.py`, `chi_identifiability.py`, `mu_identifiability.py`, `roller_slip_fraction.py`, etc. (14 scripts) |
| **Analysis outputs** | ~10.5 MB | CSVs (identifiability, chatter, tracking, sampling), `diagnostics_combined.csv` (4.57 MB) |
| **Trained checkpoints** | 2.3 MB | `checkpoints_mamba_v1/` (515.87 KB), `observer_v1_py/runs/` (1.8 MB) |
| **Visualization** | 32.3 MB | `rendered_traj_diagnostics/` (18.69 MB), `images_and_plots/` (6.24 MB), `presentation/` (7.40 MB) |
| **Documentation** | ~30.5 MB | `docs/` (8.81 MB), `CLAUDE.md` (21.63 KB), technical notes, strategy docs |
| **Handoff briefs** | 87.28 KB | `chat-handoff/` (13 cross-session task briefs) |
| **Temporary & cache** | ~10.5 MB | `_tmp/` (7.72 MB exploration), `_nb_read_eval/` (27.29 KB), `__pycache__/`, `.ipynb_checkpoints/` |

**data/ contains (274.35 GB total):**
- `Simulation_Data_MecanumSlipSpin_LugreAdamov/` — Main active sweep (~5,670 Arrow files, 221.84 GB)
- `_mu_pilot2/` — Pilot run 2 (6.09 GB); reference only
- `SimulationDataSlipSpin_Julia/` — Legacy (3.42 GB); **DEPRECATED** (do not use)
- `SimulationDataSlipSpin_Julia_3/` — Legacy (42.99 GB); **DEPRECATED** (do not use)

---

## Authority rules (respect these)

1. **base.toml in each trajectory_files_run_*** — single source of truth for physics/solver settings
2. **Module copies** — if profiles.jl or datastore.jl are updated, they replace all copies
3. **Everything runs from code_insights/** — CONFIG_DIR resolves to trajectory_files_run_* directories
4. **Output files are never hand-edited** — an existing .arrow file = completed simulation
5. **Only Cell 2** in simulator notebook carries the `parameters` tag (the only parametrization entry)
6. **Data immutability** — arrow files in data/ are read-only; new outputs require new combo IDs
7. **Profile-based enumeration** — old beta/amplitude grid is DEPRECATED; all configs use profile names

---

## Quick conventions

- **Temp files go in `code_insights/_tmp/`** — never scatter in project root
- **Never write data to code_insights/**; it all goes in `../data/`
- **Long sweeps need keep_awake.py** running in background (Modern Standby kills idle compute)
- **Thread limit: ≤8** unless explicitly requested (higher parallelism OOMs this machine)
- **Arrow filename contract** (read by Python loader):
  ```
  <profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
  e.g. octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
  ```

---

## For detailed info

**Physics/simulator details:** `code_insights/CLAUDE.md` §§1–4  
**Julia data-generation subsystem:** `code_insights/CLAUDE.md` §4  
**Cross-language data contract:** `code_insights/CLAUDE.md` §5  
**PINN architecture intent:** `code_insights/CLAUDE.md` §7  
**Working conventions:** `code_insights/CLAUDE.md` §8  

---

## Key files (by purpose)

| File | Purpose |
|------|---------|
| **This file (root)** | Root-level orientation and quick reference |
| **PROJECT_LAYOUT.md** | Detailed hierarchical layout (2+ levels, sizes, descriptions) |
| **code_insights/CLAUDE.md** | Full technical orientation: physics, simulator, data contract, PINN intent |
| `code_insights/trajectory_files_run_*/base.toml` | Physics constants + solver settings (authoritative per run) |
| `code_insights/profiles.jl` | Trajectory profile library (VelRef, PosRef builders) |
| `code_insights/datastore.jl` | Data I/O, Arrow schema, label extraction |
| `code_insights/Data_Generation_Julia.jl` | Parallel sweep driver (profile × combo enumeration) |
| `code_insights/Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb` | Main simulator (229.7 MB; source of run_one.jl) |
| `code_insights/train_GPU_PINN_v14_py/train.py` | Main PINN training entry point |
| `code_insights/extract_run_one.py` | Notebook → run_one.jl extractor |
| `code_insights/keep_awake.py` | Background wake-lock (run during long sweeps) |

---

## Root working directory

**Windows path:**
```
C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\
```

**Julia environment:** `code_insights/Project.toml` + `Manifest.toml` (version-locked deps)  
**Python side:** Local or WSL2 Ubuntu (separate environment; `mecanum_pinn_main/` repo may have parallel dev)  
**Data storage:** Local `data/` directory (274.3 GB), never written from code_insights/

---

## Last updated

2026-06-23 (automated hierarchical layout sync)
