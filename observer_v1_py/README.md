# Approach 2 — Supervised Causal State Observer (`observer_v1_py`)

A measurable-only neural **state observer** that reconstructs the unobservable
per-wheel contact states of the Mecanum plant. Companion to Approach 1 (the
force-reconstruction forward-inverse PINN); see
`approach2_state_observer_handoff.md`.

## What it estimates

| Target | Source (Arrow col) | State idx | Expectation |
|---|---|---|---|
| `gamma` (roller rate) | `gamma1..4` | [13:16] | near-algebraic → strongly observable |
| `zx`, `zy` (linear bristle) | `zx_1..4`, `zy_1..4` | [22:29] | non-unique from I/O → irreducible floor |
| `omega_z` (contact spin) | **derived** | — | `psi_dot + gamma·sin(θ̃)·cosδ`; bounds the χ-channel |

→ **12 heads** = 3 states × 4 wheels. `omega_z` is **not** a trained head — only
`gamma` is hidden in its definition, so it's computed analytically from the
`gamma` prediction. The **spin bristle `zs` is dropped**: its only footprint is
`Mz`, which is low-SNR and unused, so it is unobservable-by-design.

## Inputs (measurable-only — HARD RULE)

Globals `Vx, Vy, psi_dot`; per wheel `Msat_i`, `w_i`, `sin(θ̃_i)`, `cos(θ̃_i)`.
The fold `θ̃ = atan2(60·sin12θ, 60·cos12θ+1)/12` replicates the simulator's
`sawtooth_tanh` exactly, so input geometry matches label geometry with zero
offset. The γ=0 slip surrogate (`Vpx0/Vpy0`) is **not** an input — `Vpy0` is a
pure linear combo of `{Vy, psi_dot}` and `Vpx0` is linear save one
`psi_dot·tan(θ̃)` product the nonlinearity learns, so feeding them just
duplicates inputs (kept inputs as independent as possible). They are still
computed for the physics loss. **Never inputs:** any force, `mu`, `chi`, hidden state.

## Design

- **Causal** filter (past-only) — deployable for sim-to-real, not a smoother.
- **Wheel-shared** encoder; **SSM** (Mamba-lite selective diagonal scan, plain-
  PyTorch unrolled — full AD, deterministic, **SiLU**) vs **GRU** baseline (control).
- **Wheel embedding zero-init + frozen** for this run (identical sim wheels →
  fully symmetric). A structural hook: unfreeze later for wheel asymmetry /
  sim-to-real (`freeze_wheel_emb=False`).
- Hyperparameters: `d_model=32` (encoder width/channels), `state_dim=6` (SSM
  latent/modes), `head_hidden=32` (one hidden layer per per-state head), `emb_dim=4`.
  ~6.3k params.
- 500 Hz (stride-4 decimation of the 2000 Hz sim grid).
- Whitelist via `diagnostics_combined.csv` (`combined_reco` not `reject*`).
- Splits are **regime-driven** (see `regimes/` and `training_data_split_design.md`):
  grouped by (profile, combo) — no trajectory leakage; S1/S2 excitation 2-fold;
  S3 χ k-fold. Eval reports same-subset (val) vs cross-subset (test).
- Per-state normalised MSE; `z` is **not** up-weighted (its floor is a result).
  Optional physics loss (5-phase curriculum): roller + wheel torque-balance
  residuals from predicted states, χ carried per-sample (loss-side only).

## Usage — build the cache, then fan out N (run from `code_insights/`)

**Parallelism is the throughput lever on every machine, not just the 24 GB box.**
The model is ~6 k params and launch-bound, so a single run under-utilises *any*
GPU; concurrent runs fill it. There is one machine knob: **N = `--max-parallel`**
(the degree of parallelism). Pick N from `nvidia-smi` GPU utilisation, **not from
VRAM** — at batch ~4 k a run is ~0.3–0.5 GB, so CPU cores / disk I/O bind long
before memory (likely N≈2 on the laptop, N≈3–6 on the workstation).

The prerequisite is the **decimated cache**: `read_arrays(cache_dir)` memoises the
2000→500 Hz arrays as float32 `.npz` (W- and regime-independent, so one cache
serves every window size and regime). Without it, N jobs re-decimate the same
Arrows and thrash CPU/disk. So always **warm the cache once, then fan out**:

```bash
# (Linux only) NVIDIA MPS gives true concurrent kernels; Windows time-slices:
nvidia-cuda-mps-control -d

# build the decimated cache ONCE (single process, no race), then run N at a time.
# Default sweep = W∈{8,16,32} × {S1_train, S2_train, S3_chi_kfold×4χ} = 18 jobs.
python observer_v1_py/launch_parallel.py --warm-cache --max-parallel 3

echo quit | nvidia-cuda-mps-control          # (Linux) stop MPS

# single run (cache auto-builds on first touch; later runs/W reuse it):
python observer_v1_py/train_observer.py \
    --regime observer_v1_py/regimes/S1_train.toml --window 32 --stride-frac 0.5 \
    --phase-epochs 120 --cache-dir observer_v1_py/cache_decim

# aggregate -> state_observability.csv + figures (same-vs-cross)
python observer_v1_py/make_observability_report.py

# physics-ablation: final checkpoint vs last Adam-only phase snapshot
python observer_v1_py/physics_ablation_eval.py \
    --final-ckpt observer_v1_py/runs/S1_train_w32/checkpoint.pt

# aggregate physics-ablation records
python observer_v1_py/physics_ablation_report.py
# -> observer_v1_py/report/physics_ablation_report.csv
```

Launcher knobs (shared idiom with A1 — see `code_insights/parallel_sweep.py`):
`--max-parallel N`, `--per-run-batch` (throughput knob, not memory), `--warm-cache`
(single-process pre-build before fan-out), `--cache-dir`, `--stride-frac 0.5`
(stride = 0.5·W), `--phase-epochs 120` (scales the 5-phase curriculum), `--dry-run`,
`--force`. **Resume-safe**: a run whose `runs/<label>/metrics.json` exists is
skipped; an interrupted run epoch-resumes from `runs/<label>/checkpoint.pt`. Each
sweep writes one ranking CSV (`runs/sweep_results.csv`). On Linux each job is
pinned to a disjoint CPU-core block (`taskset`); keep N·`--dl-workers` ≤ cores and
`--jobs ≤ 8`. Auto-precision → bf16 on Ampere, fp32 on Turing (no native bf16).
The per-run `--vram {6,24}` preset still exists for one-off manual runs (sets
batch/jobs/precision/cache together), but the launcher is machine-agnostic.

## Physics-ablation study (final vs Adam-only)

`physics_ablation_eval.py` compares the final `checkpoint.pt` against the last
fully-Adam phase snapshot (default: the latest `phase_ckpts/physics_*.pt`) on both
the same-subset (val) and cross-subset (test) splits. This answers whether the
future physics-only refinement tail is worth the extra compute.

```bash
python observer_v1_py/physics_ablation_eval.py \
    --final-ckpt observer_v1_py/runs/S1_train_w32/checkpoint.pt

# explicit Adam-only snapshot
python observer_v1_py/physics_ablation_eval.py \
    --final-ckpt observer_v1_py/runs/S1_train_w32/checkpoint.pt \
    --adam-ckpt observer_v1_py/runs/S1_train_w32/phase_ckpts/grnd_rampdown_ep119.pt
```

The script writes `physics_ablation_metrics.json` in the run directory. Aggregate
all records with `physics_ablation_report.py` (negative deltas mean the final
model is better).

## Layout

```
observer_v1_py/
├── train_observer.py            single-run worker (+ --warm-cache-only, --run-tag)
├── launch_parallel.py           machine-agnostic launcher (N concurrent jobs, warm-cache,
│                                MPS + affinity); thin adapter over parallel_sweep.py
├── make_observability_report.py CSV + static figures (same-vs-cross)
├── physics_ablation_eval.py     final checkpoint vs Adam-only phase snapshot
├── physics_ablation_report.py   aggregate physics-ablation metrics
├── regimes/                    A/D/E/S1/S2/S3 + base.toml (data-subset configs)
└── mecanum_observer/
    ├── config.py      constants, column contract, hyperparameters
    ├── features.py    sawtooth_tanh fold, targets (γ,zx,zy), γ=0 slip (physics-only)
    ├── data.py        discover/select_regime/assign_folds/split_files, windows, norm
    ├── models.py      MambaLiteSSM + GRUBaseline + wheel-shared WheelObserver (SiLU)
    ├── physics.py     torch LuGre port + roller/wheel torque-balance residuals
    ├── losses.py      per-state normalised MSE + physics loss
    ├── training.py    WindowStream + 5-phase curriculum
    └── evaluation.py  per-state/-wheel/-bin RMSE, same-vs-cross, derived ω_z,
                       plus model-scoring helper for ablation studies
```

## Output to hand back to Approach 1

The `omega_z_derived` RMSE (from the `gamma` head) is the fidelity bound on
Approach-1's χ-identification, since χ enters only via `c_t=(8/3π)|ω_z|χ`.
