# Handoff — A1 + A2 PINN experiments (parallel-exec, max-norm, window study) → continuation

## 1. Title + lineage
Continues the long session that built the **parallel-execution layer** for both PINN
approaches and ran the **A2 observer window ablation under max-normalization**. Project:
Mecanum PINN digital twin (IMECE 2026); Julia 39-D ODE → Arrow → PyTorch. A1 = force-recon
forward-inverse PINN (`Mecanum_PINN_Mamba_ForceRecon_v1/`); A2 = supervised SSM/Mamba state
observer (`observer_v1_py/`). All runs are on the **6 GB RTX 3060 laptop** (project under
OneDrive); the **24 GB Quadro box** (no OneDrive) is for A1 bulk.

## 2. Context the new chat depends on (exact)
**Envs (run everything from `code_insights/`):** torch = `C:\Users\vishv\miniforge3\envs\myenv\python.exe`;
torch-free (numpy/pandas/pyarrow/matplotlib) = `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`.
Can't `conda activate` in tool shells — call by full path. Workers need `-u` (unbuffered) + PYTHONUTF8=1.
**Data:** `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/` — 5949 `.arrow`, 238 GB, 2000 Hz
(decimated to **500 Hz**, DECIM=4). μ∈{0.3,0.5,0.8}, χ∈{0,0.002,0.005,0.008} (4180 at 0.005).
Whitelist = `diagnostics_combined.csv` non-`reject` (5345). Regimes shared: `observer_v1_py/regimes/*.toml`.
**Decimated cache (local, OFF OneDrive):** `C:/Users/vishv/mecanum_cache_decim` — float32 .npz, 500 Hz,
**normalization-agnostic** (stores raw arrays). Always pass `--cache-dir C:/Users/vishv/mecanum_cache_decim`
on the laptop; on the 24 GB box the in-tree default is fine (no OneDrive).
**Scaler (max-norm, frozen p95):** `…/Simulation_Data_…/variable_scaler_percentiles.csv` (built by
`observer_v1_py/build_variable_percentiles.py`, all 5345 files). Used scales: Vx 1.92, Vy 0.673,
psi_dot 2.408, Msat 5.279, w 39.089, **sin_tt/cos_tt = 1.0 (unscaled)**, gamma 82.81, zx 3.06e-4,
zy 2.08e-4; **offset 0 (no centering)**.
**Shared infra:** `code_insights/parallel_sweep.py` (Job/run_sweep/heartbeat/resume-skip via
`run_dir/metrics.json`/ranking CSV); `dataset_chunker.py` (≤9 GB transfer chunks + reassemble);
each package's `launch_parallel.py` (thin adapter). A1 entry `train.py`; A2 entry `train_observer.py`.

## 3. Purpose
Continue the A2 training program from the chosen window, and run A1 on the box. Immediate next:
**A2 physics (#A) + measurable-consistency (#B) phases**, warm-started from the w32 max-norm base.
Success = a measurably-better (or characterized) observer under the deployment normalization, plus
A1's force/μ-ID results.

## 4. Key design decisions (already made — defend)
1. **Max-normalization (frozen p95 scaler), not var-norm (z-score).** p95 is robust to adding future
   test cases (no recenter on train mean); sin/cos left unscaled (stretching breaks the sin²+cos²
   pair); offset 0. Implemented as a `Normalizer` with mean=0, std=p95 → rest of pipeline unchanged.
2. **A2 observer window = w32** (seq_len 32 @500 Hz = 64 ms, stride 16) under max-norm. Cross-subset
   elbow at w32; w64 turns over (worse on all γ/zx/zy/ω_z). **Var-norm had picked w16 → window choice
   is normalization-dependent.** Same-subset val is misleading (windows ~tied); the decider is
   cross-subset transfer + same-vs-cross gap from `make_observability_report.py`.
3. **Cache on local disk, off OneDrive.** Concurrent `.npz` writes under OneDrive caused `WinError 5`
   (sync locks); also avoids syncing GB of regenerable cache. Cache writes are best-effort (try/except).
4. **Physics-ONLY refinement of the supervised observer DEGRADES reconstruction** (drifts the
   non-unique zx/zy). A2's curriculum keeps a `W_SUP_MIN=0.1` supervised floor — never pure physics-only.
   A2 has NO L-BFGS (A1 does). So #A/#B keep supervision.
5. **Batch-2048 "edge" is a fixed-epoch step-count confound, not generalization** (both plateau ~ep84,
   same-vs-cross gap unchanged; b2048 = 2× steps/epoch). A real batch ablation must step-match or LR-scale.
6. **A1 normalization switched to the same p95 caps** (hardcoded in `mecanum_pinn/data.py`: state_max
   Vx/Vy/psi_dot/w + control_max Msat from p95; theta unscaled; **force_max kept = F_MAX=87.309**, a
   physical bound, since forces aren't in the CSV). Old hand-set caps under-sized the data.
7. **Detached launch = PowerShell `Start-Process -WindowStyle Hidden -Redirect…`** + a `keep_awake.py`
   process (Modern Standby kills idle laptop compute). Stop keep_awake when sweeps finish.

## 5. Open decisions / blocking relationships
- **#A physics phase:** flip `physics_loss=true` (5-phase ramp, W_SUP_MIN floor) — mostly a config/launch
  change. **#B measurable-consistency:** NEW loss term to implement — observer states → LuGre forces →
  **A1's analytical Newton-Euler integrator** → predicted velocities → match measured Vx/Vy/ψ̇. Test BOTH
  L_phys variants: (1) current roller+wheel torque-balance, (2) the new force→integrator→velocity.
- **A1 sweep** not yet run — `launch_a1_24gb.bat` is ready for the box (reassemble data there first:
  `python dataset_chunker.py --mode reassemble`).
- **Batch ablation** (step-matched/LR-scaled) — deferred, optional.
- Hand-back: best A2 observer (per-state cross-subset RMSE + ω_z bound for A1's χ channel); A1 winning
  hyperparameters + μ̂/χ̂ to the respective training threads.

## 6. Deliverables (where things live / go)
- **A2 runs:** `observer_v1_py/runs/<label>/` (checkpoint.pt, metrics.json, norm.npz, split.json,
  LOSS_AND_NORM.md). Labels: `S{1,2}_train_w{8,16,32,64}_non_phys_max_norm` (b4096, done),
  `…_non_phys_var_norm` (done), `…_w32_non_phys_max_norm_b2048` (done). Warm-start #A/#B from
  **`S1_train_w32_non_phys_max_norm` / `S2_…`**. Reports: `observer_v1_py/report_max_norm/` (max-norm,
  8 runs), `observer_v1_py/report/` (var-norm). Ranking CSVs: `runs/sweep_results_max_norm*.csv`.
- **A2 launch:** `python observer_v1_py/launch_parallel.py --regimes S1_train,S2_train --windows 32
  --norm max --scaler-csv <csv> --tag-suffix <_phys_max_norm> --cache-dir C:/Users/vishv/mecanum_cache_decim
  --phase-epochs 100 …` (per-run knobs: `--per-run-batch`, `--max-parallel`, `--heartbeat`, `--warm-cache`).
- **A1:** sweep via `launch_a1_24gb.bat` → checkpoints `checkpoints_mamba_v1/<tag>/`, CSV
  `Mecanum_PINN_Mamba_ForceRecon_v1/runs/sweep_results.csv`.
- New #B code goes in `observer_v1_py/mecanum_observer/` (losses.py + physics.py + a port of A1's
  `forward_integrate`/`ne_rhs` from `Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/physics.py`).

## 7. Conventions to respect
- **base.toml is physics authority** — never invent a constant; code-verify every numeric value.
- **Surgical edits; confirm design before coding** (this user pins decisions first). Decimated cache is
  normalization-agnostic — switching norm reuses it (only the scaler changes), no re-decimation.
- Scratch/tests in `code_insights/_tmp/`, deleted after. **Static matplotlib only.** **Chat math:
  Unicode + code blocks, never LaTeX `$…$`** (terminal has no MathJax). Hard cap **≤8 workers/parallel**.
- Memory has the settled findings: `project_a2_observer_window_choice.md`, `project_parallel_exec_pattern.md`.
  Separate physics-engine benchmark handoff exists: `chat-handoff/engine_fidelity_benchmark_handoff.md`.
