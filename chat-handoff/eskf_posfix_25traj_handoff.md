# ESKF PosFix-VelRef Retuning — 25-Trajectory Ensemble

## Context

The original frozen ESKF was tuned **without** absolute pose updates (`use_pose_fix=false`) on VelRef trajectories. Even though the estimator supports pose fix, it was suboptimal. The controller comparison later showed that controllers fed by this ESKF drifted in world position because the controllers were only tracking velocity references.

Goal: re-tune the frozen ESKF so it actually uses absolute pose updates (`use_pose_fix=true`, docking tier) on the original VelRef estimator training/eval set, then use that re-tuned estimator for controller re-tuning.

The first 5-seed run on a 5-trajectory subset produced a wide score spread (training best 27–116, validation best seed 42 @ 11.4) — strong evidence of overfitting on only 5 trajectories. The training set was therefore expanded to **25 trajectories** stratified by profile.

## What is already done

- Code fixes applied:
  - `tuning/harness.jl`: `pose_hat` extracted as `[X, Y, ψ]` from `bus.xhat[5,6,4]`.
  - `tuning/objectives.jl`: `estimator_objective_abs` includes pose/heading error for **all** logs, so pose-aided VelRef tuning is driven by pose accuracy.
- Manifest created and validated: `_tmp/subset_manifest_25traj.json`
  - 25 entries, all `ref_type=velref`, `run_mode=velocity`, `pose_fix_tier=docking`.
  - Profile split: octagon 5, spin_creep 5, long_circle 4, coupled_vomega 4, spiral_orbit 4, multisine_75percent_cap 3.
  - Combos drawn from `diagnostics_combined.csv` (`combined_reco=keep`), excluding ellipse and docking.
- Launch scripts written:
  - `_tmp/run_estimator_posfix_velref_25traj.sh` (WSL)
  - `_tmp/run_estimator_posfix_velref_25traj.bat` (Windows)
- A prior 5-seed/5-traj run completed in `runs_estimator_posfix_velref/seed_*`. Best validation: **seed 42** (`runs_estimator_posfix_velref/seed_42/eskf_dxnes/best_config.json`), validation `long_circle` score 11.40.
- The 25-trajectory ensemble was launched once, then killed for a PC reboot. No results exist yet in `runs_estimator_posfix_velref_25traj/`.

## How to launch (after reboot)

### Option A: Windows command prompt

```bat
cd "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
_tmp\run_estimator_posfix_velref_25traj.bat
```

### Option B: WSL

```bash
cd /mnt/c/Users/vishv/OneDrive/Desktop/Vishvesh_Data/VNIT/mecanum_pinn_head/code_insights
bash _tmp/run_estimator_posfix_velref_25traj.sh
```

This starts 5 independent `dxnes` runs in parallel:

| seed | output dir | Julia threads | BBO workers | budget |
|---|---|---|---|---|
| 42 | `runs_estimator_posfix_velref_25traj/seed_42` | 3 | 2 | 50 |
| 43 | `runs_estimator_posfix_velref_25traj/seed_43` | 3 | 2 | 50 |
| 44 | `runs_estimator_posfix_velref_25traj/seed_44` | 3 | 2 | 50 |
| 45 | `runs_estimator_posfix_velref_25traj/seed_45` | 3 | 2 | 50 |
| 46 | `runs_estimator_posfix_velref_25traj/seed_46` | 3 | 2 | 50 |

Total CPU: ~15 threads.

## Expected runtime

Each objective evaluation runs 25 closed-loop simulations serially. At ~40 s per trajectory, one eval ≈ **17 min**. With 2 BBO workers per seed, budget 50 ≈ **6–8 hours wall time** for the full ensemble.

## How to monitor

Per-seed live traces:

```text
runs_estimator_posfix_velref_25traj/seed_<NN>/eskf_dxnes/fitness_trace.csv
```

Columns: `iteration,score,best_so_far`

Per-seed logs:

```text
runs_estimator_posfix_velref_25traj/seed_<NN>/tune.log
```

A quick status check in WSL:

```bash
cd /mnt/c/Users/vishv/OneDrive/Desktop/Vishvesh_Data/VNIT/mecanum_pinn_head/code_insights
for d in runs_estimator_posfix_velref_25traj/seed_*; do
  seed=$(basename "$d" | sed 's/seed_//')
  pid=$(cat "$d/tune.pid" 2>/dev/null || echo "?")
  ps -p "$pid" -o pid= > /dev/null 2>&1 && st="alive" || st="DEAD"
  n=$(wc -l < "$d/eskf_dxnes/fitness_trace.csv" 2>/dev/null | awk '{print $1-1}')
  best=$(tail -1 "$d/eskf_dxnes/fitness_trace.csv" 2>/dev/null | cut -d, -f3)
  echo "seed=$seed pid=$pid status=$st evals=${n:-0} best=${best:-N/A}"
done
```

Plateau rule of thumb: if `best_so_far` has not improved for ≥15 evaluations and ≥40 evals are done, that seed has likely converged.

## How to pick the final frozen ESKF

After all seeds finish, compare:

1. **Training best score** across seeds: lowest `best_so_far` in each `fitness_trace.csv`.
2. **Validation `long_circle` score**: printed at the end of each `tune.log` as:
   ```text
   [eskf] validation long_circle: score=... vel_rmse=... rate_rmse=...
   ```
3. The recommended final config is the seed with the **best validation score**, not necessarily the best training score (the 5-traj run showed overfitting).

The chosen config lives at:

```text
runs_estimator_posfix_velref_25traj/seed_<NN>/eskf_dxnes/best_config.json
```

## Next steps after ESKF retuning

1. Copy/link the chosen `best_config.json` to the estimator dir used by controller comparison.
2. Generate PosRef sidecars for VelRef trajectories:
   - Read `Vx_des`, `Vy_des`, `psi_des` from existing Arrow files in `data/Simulation_Data_MecanumSlipSpin_LugreAdamov/`.
   - Compute:
     - `Vxo_des = Vx_des·cos(ψ_des) − Vy_des·sin(ψ_des)`
     - `Vyo_des = Vx_des·sin(ψ_des) + Vy_des·cos(ψ_des)`
     - `xo_des = ∫ Vxo_des dt`
     - `yo_des = ∫ Vyo_des dt`
   - Store as `.arrow` in `data/tuner_estimator_data/` (do **not** modify the LugreAdamov files).
3. Re-tune ASMC and PID controllers in `:pose` mode using the new frozen ESKF + PosRef references.

## Files involved

- Manifest: `_tmp/subset_manifest_25traj.json`
- Launch scripts: `_tmp/run_estimator_posfix_velref_25traj.sh`, `_tmp/run_estimator_posfix_velref_25traj.bat`
- Manifest generator: `_tmp/make_posfix_25traj_manifest.py`
- Validation script: `_tmp/validate_25traj_manifest.jl`
- Tuning entry point: `tune_estimator.jl`
- Tuning spine: `tuning/harness.jl`, `tuning/objectives.jl`
