# Approach 1 — Force-Reconstruction Forward–Inverse PINN (`Mecanum_PINN_Mamba_ForceRecon_v1`)

Measurable-only PINN that learns the **forward dynamics** (per-wheel roller-frame
forces `Fpar/Fperp` via a Mamba-S6 selective-SSM encoder + structured force head)
and recovers friction parameters by **test-time inverse identification** (`μ̂`, `χ̂`
residual readout from `F_inv`, never trained). Companion to Approach 2 (the state
observer, `observer_v1_py/`); the two **share** the regime TOMLs
(`observer_v1_py/regimes/*.toml`), the `diagnostics_combined.csv` whitelist, the
`trajectory_files_run_0p5_main/profiles` dir, and the data dir. `regime_split.py`
is a verbatim port of A2's selection → **identical trajectory selection**.

## Run (myenv python, from `code_insights/`)

```bash
python Mecanum_PINN_Mamba_ForceRecon_v1/train.py both     # forward then inverse
```

`cwd` **must** be `code_insights/` (so `..\data\...`, `project_root='.'`,
`trajectory_files_run_0p5_main/profiles`, `diagnostics_combined.csv` resolve). In
standalone scripts import `pyarrow.feather` before `torch` (`train.py` is already
safe). CLI overrides (consumed by the launcher; `train.py` parses them):

```
--vram {6,12,24}  --regime <toml>  --test-chi <f>  --batch-size/--per-run-batch <n>
--cache-dir <dir>  --run-tag <tag>  --no-lbfgs  --warm-cache-only  --set KEY=VALUE
```

`--set` overrides any top-level config key, typed (`--set ssm_d_model=48
--set lr=2e-3 --set force_four_term=false`) — that's the model-hyperparameter sweep
surface.

## Parallel execution — build the cache, then fan out N

Same idiom as A2, same shared core (`code_insights/parallel_sweep.py`). The model
is tiny (~5.5 k params) and launch-bound, so **parallelise at the experiment level**
(many independent runs), **not** DDP — one run can't saturate even the 6 GB GPU, and
DDP sync would dominate. **N = `--max-parallel`** is the only machine knob; size it
from `nvidia-smi` SM utilisation, **not VRAM** (batch 1024 ≈ 0.04 GB; the CUDA
context ~0.4–0.7 GB/proc is the floor).

The **decimated cache** is the prerequisite: `read_trajectory(cache_dir)` memoises
the 2000→500 Hz arrays as float32 `.npz` (seq_len-/stride-/regime-independent, so
one cache serves every run and every concurrent worker). **Warm it once, then fan
out** so N jobs never race to write the same `.npz`:

```bash
# build the cache once (single process), then run 4 at a time. Default sweep =
# the A2-like window ablation seq_len∈{8,16,32} (stride=0.5·W) × {S1,S2,S3×4χ} = 18 runs:
python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --warm-cache --max-parallel 4
python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --dry-run        # print the plan
# narrow the window axis, or cross it with a model sweep:
python Mecanum_PINN_Mamba_ForceRecon_v1/launch_parallel.py --windows 16 --ssm-dims 32x16,48x16
```

**Resume-safe**: a run whose `checkpoints_mamba_v1/<label>/metrics.json` exists is
skipped (`--force` overrides). An interrupted A1 run restarts from scratch (A1 has
no mid-run epoch resume — the marker is run-level). Each sweep writes one ranking
CSV (`runs/sweep_results.csv`) with every run's **MAE(inv) / MAE(fwd) / inv-fwd
divergence** + final forward/inverse test losses. On Linux each job is pinned to a
disjoint CPU-core block (`taskset`); keep N·`--dl-workers` ≤ cores and ≤ 8 total.

### Default sweep grid (tentative — exact model knobs undecided)

- **Window ablation** (default, the A2 analog): `--windows 8,16,32` with
  `stride = round(--stride-frac · window)` (`stride-frac` = 0.5 → stride {4,8,16}).
  A1's **`seq_len` IS the lookback window** (A2's `window`); the launcher sets it
  per job via `--set seq_len=W --set stride=…`. `windows/epoch ∝ 1/stride`, so the
  three window runs cost ~equally (same trick A2 uses). Note `seq_len` is coupled to
  the physics-loss lookahead `k_steps` (currently **4**, kept fixed — all windows ≥
  k_steps); pass `--extra --set k_steps=…` if you want the rollout to scale too.
- **Regime axis** (default): `S1_train`, `S2_train`, `S3_chi_kfold` × 4 χ-folds.
- Window × regime at the current architecture = **18 runs** (mirrors A2's 18-job
  W-ablation). Add the **model axis** (`--ssm-dims 32x16,48x16,…`) to multiply it.
  The sweep judges **val MAE(inv)** subject to **MAE(fwd) staying low in the
  high-slip tail**, plus inv-fwd divergence and force `grnd`. Other knobs:
  `--extra --set lr=… --set shape_hidden=… --set w_phys_max=…`.

### K per machine (set after running `_tmp/bench_concurrency.py`)

| Machine | role | suggested start | notes |
|---|---|---|---|
| RTX 3060, 6.4 GB / 16 GB | interactive iteration | `--max-parallel 2–4`, `--no-lbfgs` | thermal/Modern-Standby limited; keep `keep_awake.py` running |
| Quadro RTX 6000, 24 GB / 64 GB (Turing, no bf16) | unattended sweep | `--max-parallel ≈ 8` (≤ cap), L-BFGS on | 64 GB RAM holds the in-RAM traj set even ×8; Linux → enable MPS |

Run `python _tmp/bench_concurrency.py --approach a1 --warm-cache` on each box to
find the speedup knee, then set `--max-parallel` near it (back off once efficiency
drops below ~0.6).

## Layout

```
Mecanum_PINN_Mamba_ForceRecon_v1/
├── train.py              entry (cache ON by default; CLI overrides -> run_main)
├── launch_parallel.py    machine-agnostic launcher (thin adapter over parallel_sweep.py)
├── smoke_test.py         shape/wiring smoke test (CPU-OK, no data)
└── mecanum_pinn/
    ├── config.py     build_config (+ cache_dir knob), build_run_tag, vram tiers
    ├── data.py       Arrow loader + read_trajectory/.npz cache + warm_cache, windows, split
    ├── regime_split.py   verbatim port of A2 selection (shared regime TOMLs)
    ├── models.py     SelectiveSSM (Mamba-S6 lean core) + ForceHead + inverse readout
    ├── physics.py    RobotParams, Heun NE integrator + ne_rhs (verified vs run_one.jl)
    ├── losses.py     forward/inverse losses (w_cons=0 monitor-only)
    ├── training.py   5-phase schedule + EarlyStopper + L-BFGS refine
    ├── stages.py     run_main (forward|inverse|both|figures), CLI overrides, metrics.json
    ├── evaluation.py evaluate_on_test, estimate_mu, evaluate_mu_id (the deliverable metric)
    └── plotting.py / manifest.py
```
