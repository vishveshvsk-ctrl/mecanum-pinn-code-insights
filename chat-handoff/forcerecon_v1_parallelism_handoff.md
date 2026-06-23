# Handoff — A1 (force_recon_v1) code orientation + parallelism enablement

## 1. Title + lineage
This is the **Approach-1 (A1)** handoff into a session that will build **parallelism /
sweep tooling for BOTH A1 and A2 in one sitting**. A1 = the force-reconstruction
forward-inverse PINN package `code_insights/Mecanum_PINN_Mamba_ForceRecon_v1/`
("mamba_force_recon_v1"). A separate A2 session (supervised SSM/Mamba **state observer**,
package `observer_v1_py/`) is sending you its own companion handoff — read both; your job
is the shared experiment-parallelism layer that drives them. KUKA youBot 4-Mecanum digital
twin, IMECE 2026: Julia 39-D stiff ODE → Arrow → PyTorch PINN.

## 2. Context the new chat depends on

**Environment (hard):**
- **Python = `C:\Users\vishv\miniforge3\envs\myenv\python.exe`** (torch 2.6.0+cu124, CUDA ok,
  pyarrow 24.0.0, numpy 2.4.2, py3.13). `claude-venv` has **no torch**. Both env prefixes are
  allow-listed in `.claude/check_project_scope.py`.
- **Run invocation:** from `code_insights/` with `PYTHONPATH=Mecanum_PINN_Mamba_ForceRecon_v1`,
  then `python Mecanum_PINN_Mamba_ForceRecon_v1/train.py both`. **cwd MUST be `code_insights/`**
  (so `..\data\...`, `project_root='.'`, `trajectory_files_run_0p5_main/profiles`,
  `diagnostics_combined.csv` resolve); the script dir lands on `sys.path` so `mecanum_pinn`
  imports. **In standalone scripts `import pyarrow.feather` BEFORE `torch`** or you get a
  `0xC0000005` access-violation segfault on first `read_feather`. `train.py` is already safe
  (its import chain hits `data`→feather before torch).
- **Two machines:** (1) laptop **RTX 3060, 6.4 GB / 16 GB RAM**; (2) lab PC **Quadro RTX 6000,
  24 GB / 64 GB RAM** (Turing, **no bf16**). Lab PC OS (Windows vs Linux) **unknown** — gates MPS.

**A1 package layout (`mecanum_pinn/`, all py_compile-clean) — purpose of each file:**
- `physics.py` — `RobotParams`/geometry; verified roller→body transform; Heun integrator
  `forward_integrate` (O(dt³), force at both endpoints) + `ne_rhs` (plant RHS verified vs
  `run_one.jl:716-722`; p1=0.11, Mz dropped). `F_MAX=87.309`, `N_per_roller=[79.57,105.07,69.55,95.05]`.
- `models.py` — **`SelectiveSSM` (rewritten THIS session → Mamba-S6 lean core, see §4.1)**;
  `ForceHead` (4-term law, 10 outputs/wheel); `MecanumForwardModel`; `MecanumInverseModel`
  (Δ-window MLP); `mu_readout_residual` (TEST-TIME μ-ID); `set_grad`; `maybe_compile_pinn`.
- `losses.py` — `forward_losses{grnd,phys}`; `inverse_losses{grnd,cons(monitor),phys}`; `ne_residual`.
- `training.py` — 5-phase schedule (grounding→rampup→overlap→rampdown→physics-only),
  Adam-per-phase + `EarlyStopper`, **L-BFGS** refine (60 full-dataset passes), checkpoint I/O.
- `data.py` — Arrow loader (new regex, Fpar/Fperp targets, **500 Hz** downsample), `stratified_split`
  (by `(profile,combo)`), `build_loaders_*`. Trajectories load **once into RAM** (~1.5 GB for S1).
- `regime_split.py` — **verbatim port of A2's selection**; reads `observer_v1_py/regimes/*.toml`
  + `diagnostics_combined.csv` + `trajectory_files_run_0p5_main/profiles`.
- `evaluation.py` — `evaluate_on_test`; `estimate_mu`; `evaluate_mu_id` → **MAE(inv), MAE(fwd),
  inv-fwd div** (the deliverable metric).
- `stages.py` — `run_main` (modes `forward|inverse|both|figures`), GPU flags, regime/internal branch.
- `config.py` — `build_config` (vram tiers 6/12/24 → batch 1024/512/4096), `build_run_tag`,
  `apply_dummy_overrides`. `plotting.py`, `manifest.py`. Top-level `train.py`, `smoke_test.py`.

**Data + scale (drives the parallelism math):**
- `data/Simulation_Data_MecanumSlipSpin_LugreAdamov/`: **5670 arrow** (1890 each μ=0.3/0.5/0.8);
  χ=0.005 → 4455 files. Each traj = 44,022 rows @2000 Hz (22 s) → **~11,006 windows/traj @500 Hz,
  stride 1**. Whitelist = `diagnostics_combined.csv` non-`reject` = **5,345 / 5,670**.
- Regime `S1_train.toml`: **train=1393, val=155, test=1579** (from 3127 selected). → full S1 ≈
  **15.3M windows/epoch @ stride 1**. Forward schedule has **110 fixed (non-ES) epochs** + L-BFGS.
- **Measured (laptop 3060):** forward model **5,498 params** (encoder 3,040 + head 2,442), inverse
  **2,226**. Batch 1024 → **0.04 GB** GPU. Throughput **~5,000 win/s @1024, ~6,000 @2048** (sublinear,
  launch/small-GEMM bound). → at stride 1 a full `both` run is weeks; stride 8 ~1.5 days.

## 3. Purpose
Build, in one session spanning A1+A2, the **experiment-level parallelism layer**: (a) a concurrency
benchmark and (b) an approach-agnostic sweep supervisor, so the model-hyperparameter × regime grid
can run unattended. Success = supervisor launches N independent runs with a `--max-concurrent K` cap,
resume-safe, logging each run's final MAE numbers to one CSV for ranking, working on both machines.

## 4. Key design decisions (already made — defend)
1. **`SelectiveSSM` is now Mamba-S6 lean core (rewritten this session).** Lift `feat(f_in=11)→D` channels,
   N-dim state each; **selective B, C, Δ** (linear in lifted u), **fixed physical A** (τ ladder 40→0.4 ms);
   Δ **physically anchored** to [0.5,8] ms. No conv1d / SiLU-gate / out_proj (the ForceHead MLP is the
   readout). Config knob `ssm_state_dim=6` → **`ssm_d_model=32` + `ssm_d_state=16`**. ForceHead reads the
   **SSM latent only** (no raw-`feat` concat — user judged that skip would cause optimization redundancy;
   instantaneous channels still reach it via the encoder's `D_skip·u`). Smoke test passes.
2. **Parallelize at the EXPERIMENT level (many independent runs), NOT DDP/data-parallel.** Model is 5.5k
   params; one run can't saturate even the 6 GB GPU → DDP sync overhead would dominate. Wrong tool.
3. **VRAM tiering is nearly irrelevant for memory** — batch barely fills the GPU; the tier only sets
   batch/amp and how many runs you pack. Don't size runs by VRAM.
4. **Per-GPU concurrency K is found empirically** — SM saturation caps it, not memory; CUDA context
   (~0.4–0.7 GB/proc) is the floor (6 GB ≈ room for ~8, 24 GB ≈ ~30, but useful K is lower).
5. **Machine roles:** laptop = interactive iteration (≤4 concurrent, L-BFGS off); lab PC = unattended
   sweep (K≈8, L-BFGS on). 64 GB RAM holds the in-RAM traj set even ×8.
6. **Stride is the dominant per-run cost lever** (windows/epoch ∝ 1/stride), not batch.
7. **Two parameter classes:** *runtime* (batch/stride/amp/L-BFGS/K — deterministic from the throughput
   numbers above) vs *model hyperparameters* (`ssm_d_model, ssm_d_state, shape_hidden, lr, w_phys_max,
   force_four_term` — the actual sweep, judged on **val MAE(inv)** subject to **MAE(fwd) staying low in
   the high-slip tail** (the bristle-D caveat), plus inv-fwd divergence + force `grnd`).
8. **Hard cap ≤8 concurrent processes** unless the user explicitly raises it (prior RAM-OOM history).
9. **`seq_len`/`stride` is a SWEEP AXIS, not an A1 code change.** Verified: `seq_len` is
   sequence-length-agnostic in A1 (SSM scan loops L; ForceHead pointwise; only coupling is
   `inv_window=3 ≤ seq_len`, plus dataset `n≥seq_len+2`, n≈11,006). **`k_steps` is DEAD config**
   (set in `config.py:138`, printed in `stages._summary`, but UNUSED — the physics loss `ne_residual`
   is a *one-step* Heun residual over the window, not a k-step rollout), so the prior brief's
   "seq_len constrained by k_steps lookahead" does NOT hold here. **Ablation: `seq_len ∈ {32,16,8}`,
   `stride = seq_len//2 = {16,8,4}`** — supplied as `cfg['seq_len']`/`cfg['stride']` overrides by the
   supervisor; **no package edits**. The rule is **iso-compute**: windows·L ≈ 2n ≈ 22.0k rows/traj for
   all three (vs ~55k at the legacy seq_len=5/stride=1 default), so only temporal context varies.

## 5. Open decisions / blocking relationships
- **Lab PC OS** (Windows vs Linux) — Linux enables NVIDIA **MPS** for efficient concurrent GPU sharing;
  Windows uses default time-slicing. Confirm before tuning K.
- **Exact sweep grid values** — undecided. ~6–8 model-knob configs × {S1, S2, S3×χ} regimes ≈ 20–50 runs.
- **Real-run scale** (stride / schedule trims / L-BFGS) — an earlier "Standard pass ~18–36 h, stride 8,
  L-BFGS on" pick **predates** the architecture change and the parallelism plan; treat as tentative.
- **A2 coexistence (critical):** A2 = `observer_v1_py/` state observer. **Shared with A1:** the regime
  TOMLs (`observer_v1_py/regimes/{S1,S2,S3}*.toml`), the `diagnostics_combined.csv` whitelist, the
  `trajectory_files_run_0p5_main/profiles` dir, and the data dir. A1's `regime_split.py` is a verbatim
  port of A2's selection → **identical trajectory selection**. Make the supervisor **approach-agnostic**:
  a config row carries `{entry_script, pkg_dir (PYTHONPATH), kwargs}`; both run via myenv from `code_insights/`.
- **Hand-back:** the benchmark + supervisor + chosen K-per-machine + sweep grid cross back to the parent;
  the actual training sweep runs after, and the winning A1 hyperparameters return to the A1 training thread.

## 6. Deliverables
1. **Concurrency benchmark** (e.g. `_tmp/bench_concurrency.py`) — aggregate win/s + peak GPU mem at
   K = 1,2,4,8 concurrent training procs; report the scaling knee. Run on both machines.
2. **Sweep supervisor** (e.g. `sweep_runner.py`) — takes a config list, `--max-concurrent K` cap,
   vram-tier-aware, resume-safe (skip runs whose checkpoint/tag exists), distinct `run_tag` per config,
   one CSV with each run's final MAE(inv)/MAE(fwd)/div for ranking. Model on `_tmp/mu_supervisor.py`.
3. **Approach-agnostic dispatch** — same supervisor drives A1 `train.py` and the A2 entry point.
4. **Short note:** chosen K per machine + recommended sweep grid.

## 7. Conventions to respect
- **myenv python only**; run from `code_insights/` with `PYTHONPATH=<pkg dir>`; `pyarrow.feather`
  before `torch` in standalone scripts; **≤8 parallel** procs (hard); `keep_awake.py` in background for
  long runs (Modern Standby kills idle compute).
- **Confirm design before coding** — this user pins architecture decisions before implementation, prefers
  surgical edits over rewrites, code-verifies every numeric value, wants **Unicode math in chat (no LaTeX)**.
- Scratch/tests in `_tmp/`, deleted after. Seeded determinism; distinct `run_tag` per run. Never reintroduce
  the legacy `beta`/`amp` filename scheme. Don't reopen w_cons=0 (consistency is monitor-only) or the
  measurable-only-inputs / μ-ID-is-test-time-only decisions (they predate this session; see prior brief).
