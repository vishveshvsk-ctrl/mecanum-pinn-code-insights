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

See **[PROJECT_LAYOUT.md](../PROJECT_LAYOUT.md)** for the complete current file hierarchy
(detailed breakdown with file counts, descriptions, and organization by category).

**Quick summary of code_insights/ (6,331 files total):**

| Category | Files | Key Directories |
|----------|-------|-----------------|
| **Infrastructure & Caches** | 2,588 | `.git/` (2,552), Python/Jupyter caches, Claude settings |
| **Hybrid Control** | 955 | `hybrid_ctrl_v2/` (853), `hybrid_ctrl/` (102) |
| **Observer System** | 649 | `observer_v1_py/` (642), `observer_validation_v1_envelope_tuned/` (7) |
| **Utilities & Temp** | 788 | `_tmp/` (740), `chat-handoff/` (28), tools/tuning/batch scripts (20) |
| **Estimators & Kalman** | 358 | `runs_estimator_posfix_velref*` (238), base/IMM/ABS variants (120) |
| **PINN & Training** | 201 | `Mecanum_PINN_Mamba_ForceRecon_v1/` (160), `train_GPU_PINN_v14_py/` (41) |
| **Controllers** | 219 | `runs_controller*` (17 variants: ASMC, PID, ESKF comparisons) |
| **Visualization & Docs** | 140 | `presentation/` (46), `images_and_plots/` (32), `docs/` (21), `instructions/` (29), widgets (12) |
| **Trajectories & Physics** | 101 | 13 config sets (mu/chi/profile sweeps with base.toml) |
| **ESKF Studies** | 75 | `runs_eskf_smoke/` (21), `runs_eskf_full_v2/` (19), `runs_eskf_noellipse_v2/` (16), others (19) |
| **Specialized Studies** | 82 | Bound analysis (28), diagnostics (16), solver benchmarks (17), stick-slip (8), energy/summaries (13) |
| **Comparisons** | 3 | Cross-run comparison studies |

**data/ contains (23,311 files total):**
- `Simulation_Data_MecanumSlipSpin_LugreAdamov/` — Main active sweep (13,151 Arrow files)
- `SimulationDataSlipSpin_Julia_3/` — Legacy (5,186 files); **DEPRECATED** (do not use)
- `SimulationDataSlipSpin_Julia/` — Legacy (4,810 files); **DEPRECATED** (do not use)
- `_mu_pilot2/` — Pilot run 2 (142 files); reference only
- `solver_ablation_studies/` — Solver study data (15 files); reference only
- `IMU_frontend_audit/` — IMU audit (1 file); reference only

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

- **NEVER search files in the parent `mecanum_pinn_head/` folder** — it contains `data/` (~275 GB, ~22k files); recursive `find`/`grep`/`Glob` there hangs. **Scope ALL file searches to the `code_insights/` working directory** (e.g. `Glob(path="…/code_insights", …)`, `grep`/`rg` with an explicit `code_insights` path). Never let a search walk up into `../data/`.
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

2026-08-16 (automated hierarchical layout sync)
- Updated file counts: code_insights 6,331 (↑2,667 from 3,664 post-cleanup), data 23,303 (→unchanged)
- Major categories: Infrastructure (.git + caches: 2,588), Hybrid Ctrl (955), Observer (649), Utilities (788), Estimators (358), PINN (201), Controllers (219), Viz/Docs (140), Trajectories (101), ESKF (75), Specialized (82)
- Status: Expansion in hybrid_ctrl_v2 (853), controller/estimator/ESKF runs, utilities; all subsystems operationally intact

2026-08-13 (automated hierarchical layout sync)
- Updated file counts: code_insights 7,049 (↑1,007 from 6,042), data 23,311 (↑8 from 23,303)
- Major categories reorganized: Infrastructure (2,850), Hybrid Ctrl (1,017), Observer (757), Utilities (760), Estimators (413), PINN (218), Controllers (281), Viz/Docs (140), Trajectories (101), ESKF (97), Specialized (110), Root files (138), Comparisons (3)
- Growth driven by expanded hybrid_ctrl_v2 (898), utilities (_tmp: 706), estimator runs (413), and controller runs (281); .git updated to 2,825 files; all subsystems operationally intact
