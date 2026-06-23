# Handoff — A2 (observer_v1_py) code map + parallelism plan

## 1. Title + lineage
Continues the **Approach-2 state-observer** session (the SSM/Mamba causal observer
in `observer_v1_py/`). The next session implements **parallel-execution code edits
for BOTH approaches** — it will receive a companion **A1 handoff** (force-recon
PINN, `train_GPU_PINN_v14_py/`). This brief covers **A2**: the observer's layout +
the agreed parallelism plan. Project: PINN digital twin of a KUKA youBot 4-Mecanum
platform (IMECE 2026); Julia sim → Arrow → PyTorch.

## 2. Context the new chat depends on (exact)
- **Package `observer_v1_py/`** (authoritative A2 code). Components:
  - `train_observer.py` — single-run worker CLI. `--vram {6,24}` preset, `--regime`,
    `--window`, `--stride-frac`, `--phase-epochs`, `--cache-dir`, `--test-chi`.
    Precedence: dataclass defaults < `--vram` preset < regime TOML < explicit CLI.
  - `launch_parallel.py` — orchestrator: runs N `train_observer --vram 24` jobs
    concurrently (process pool + `--max-parallel`, Linux `taskset` affinity, MPS
    note, per-job logs, `--dry-run`). Default sweep = 18 jobs (W∈{8,16,32} ×
    {S1_train, S2_train, S3×4χ}).
  - `mecanum_observer/config.py` — `ObserverConfig` dataclass + all constants:
    plant geometry, LuGre params, `VRAM_PRESETS`, `PHASE_SCHEDULE`, `eff_stride`.
  - `mecanum_observer/features.py` — `sawtooth_tanh` fold; builds inputs + targets;
    computes γ=0 slip surrogate (physics-loss only, NOT an input).
  - `mecanum_observer/data.py` — discovery/whitelist, `select_regime`,
    `assign_folds` (S1/S2), `split_files` (excitation-fold + χ-fold modes),
    `read_arrays(path, cache_dir)` **decimated 500 Hz float32 .npz cache**, windowing.
  - `mecanum_observer/models.py` — `MambaLiteSSM` (selective Δ/B/C, plain unrolled
    scan) + `GRUBaseline` + `WheelObserver`; wheel embedding zero+frozen; SiLU.
  - `mecanum_observer/physics.py` — torch LuGre port (verified vs Arrow to 2e-16) +
    roller/wheel torque-balance residuals (Mz term dropped).
  - `mecanum_observer/losses.py` — per-state normalised MSE + `physics_loss`.
  - `mecanum_observer/training.py` — `WindowStream` IterableDataset + 5-phase
    curriculum (`_phase_plan`, scalable via `phase_total_epochs`).
  - `mecanum_observer/evaluation.py` — per-state/-wheel/-bin RMSE, **same-subset
    (val) vs cross-subset (test)**, derived ω_z.
  - `regimes/` — `base.toml` + A_ranking, D_chi_study, E_learning_curve_25,
    S1_train, S2_train, S3_chi_kfold.
- **Model (current):** SSM `d_model=32`, `state_dim=6`, `head_hidden=32`,
  `emb_dim=4` (frozen-zero); **3 targets** {γ, zx, zy} → 12 heads; inputs 11-dim
  (Vx,Vy,ψ̇ + per-wheel Msat,ω,sinθ̃,cosθ̃ + emb). **~6,277 params** (6,261 trainable).
- **Data:** μ∈{0.3,0.5,0.8}, χ∈{0,0.002,0.005,0.008} (χ=0.005 dominant); whitelist
  = `diagnostics_combined.csv` `combined_reco` ≠ `reject*`. Avg **15,837 samples/
  file @500 Hz**. S1/S2 (χ=0.005, all μ, non-multisine): **S1=1548, S2=1579** files;
  S1_train ≈ train 1393 / val 155 / test(S2) 1579. S3 (matched χ-quads, backbone
  only): 1548 files; per χ-fold train 1098 / val 63 / test 387. Splits doc:
  `code_insights/training_data_split_design.md` (shared A1/A2, authoritative).
- **Windows/epoch (S1 train):** stride 1 → **20.2M**; stride 16 (=0.5·W=32) → **1.26M**.
- **Machines:** 6 GB RTX 3060 Mobile (Ampere, **bf16**, 16 GB RAM, laptop) and
  24 GB Quadro RTX 6000 (Turing, **fp32**, workstation). Torch env = conda `myenv`
  (`C:/Users/vishv/miniforge3/envs/myenv/python.exe`); numpy/pandas-only tests in
  `C:/Users/vishv/claude-venv/mecanum/`.

## 3. Purpose
Make parallel execution the primary throughput lever for **both** approaches: one
launcher per package, **N = degree of parallelism** as the only machine-dependent
knob. Success = the 18-job A2 ablation (and A1's sweep) run concurrently, cache-
backed, with N tunable per machine; wall-clock for the full sweep drops to hours.

## 4. Key design decisions (made — defend, don't reopen)
1. **Parallelism is the lever on BOTH 6 GB and 24 GB**, not just 24 GB. The model
   is tiny (~6 k params) + launch-bound (Python unrolled scan), so one run under-
   utilises the GPU; concurrent runs fill it. N just scales with the machine.
2. **N is capped by GPU-saturation + CPU cores + disk I/O — NOT VRAM.** At batch
   1024 a run is ~0.3–0.5 GB, so VRAM fits 4–6; CPU/IO bind first. Set N by
   `nvidia-smi` util (likely ~2 on the laptop, ~3–6 on the workstation).
3. **Decimated cache is the prerequisite for parallelism.** `read_arrays(cache_dir)`
   memoises 500 Hz arrays (float32 .npz, reused across W and regimes). Without it,
   N runs re-decimate + re-read and thrash CPU/disk. Cache first, then fan out.
4. **stride = 0.5·W** (`--stride-frac 0.5`). Stride-1 windows are ~97% redundant;
   total scan-steps ≈ constant across W (2·samples), so the 3 W-runs cost ~equally.
5. **Curriculum scaled to ~120 epochs** (`--phase-epochs 120` → 48/14/24/14/19);
   the 250-epoch A1 schedule is overkill for the simpler observer.
6. **zs dropped** (Mz low-SNR/unused) → 3 targets. **Wheel embedding zero+frozen**
   (identical sim wheels; hook for asymmetry/sim-to-real). **d_model=32** (small-
   encoder/anti-fingerprint; let the S1/S2 cross-subset gap decide further shrink).
   **SiLU** (Mamba-idiomatic).
7. **6 GB laptop is a poor host for the full sweep** (thermal throttle + Modern
   Standby kills background compute — needs `keep_awake.py`). Bulk on the 24 GB box.

## 5. Open decisions / next steps (the parallelism work)
- **Unify launcher for A2**: add `--per-run-batch` (passed to each job, overrides
  the preset) and `--warm-cache` (single-process pre-build of the decimated cache
  **before** fan-out, so N jobs don't race to write the same .npz). Drop the
  "6=single / 24=parallel" framing in README — one launcher, N knob.
- **Mirror the pattern to A1**: A1 has its own data loader/entry; the new session
  should give A1 the **same decimated-cache + launcher + MPS/affinity** pattern so
  both approaches share one parallel-execution idiom. Reconcile with the A1 handoff.
- **Hand-back:** the unified launcher + cache pattern usable by both A1 and A2;
  and the measured N per machine (from `nvidia-smi`) to fix the sweep defaults.

## 6. Deliverables
1. `observer_v1_py/launch_parallel.py` updated: `--per-run-batch`, `--warm-cache`;
   machine-agnostic (N is the knob). Keep `--dry-run`, affinity, MPS, per-job logs.
2. A1's equivalent launcher + decimated cache (per the A1 handoff), same idiom.
3. README/doc edits dropping the 6-vs-24 dichotomy; "build cache → fan out N" flow.

## 7. Conventions to respect
- **Test envs:** `claude-venv` for numpy/pandas/pyarrow (no torch); `myenv` for
  torch/CUDA. Cannot `conda activate` in tool shells — call python by full path.
- Run everything from `code_insights/` (cwd). `--jobs ≤ 8` per run (machine OOMs).
- Scratch in `_tmp/`, delete after. Static matplotlib only (no widgets).
- **Chat math: Unicode + code blocks, never LaTeX `$...$`** (terminal has no MathJax).
- Surgical edits; verify numbers in code before asserting; an `.arrow` = a finished
  job (never hand-edit). Handoffs live in `code_insights/chat-handoff/`.
