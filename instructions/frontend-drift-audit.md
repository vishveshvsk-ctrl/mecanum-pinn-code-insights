# Velocity Front-End Drift Audit (complementary filter, Euler vs rotation+AB2)

> **Generated:** 2026-07-09
> **Stack:** Python 3.11 (torch-free venv `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`), numpy 2.4, pandas 3.0, matplotlib 3.11, pyarrow 24.0
> **Scope:** Analysis / audit (no training; pre-training gate for Observer v2 — §8 step 3(d) of `instructions/observer-gamma-only-5phase-retrain.md`)
> **Prerequisite:** `instructions/arrow-accel-augmentation.md` completed (`accel/<stem>_accel.arrow` sidecars present; read via its `load_with_accel` join contract), or run on the sidecar pilot subset

## 1. Overview

Quantify the velocity-estimate error of the deployable sensor front-end BEFORE any v2
training: build the accelerometer observable from the exact-dynamics columns, run the
complementary filter (wheel-odometry LPF anchor + strapdown-mechanized accel HPF path)
over real trajectories, and report `V̂ − V_true` across a sweep matrix of
**integrator ∈ {Euler, exact-rotation+AB2}**, crossover frequency, filter rate
∈ {500, 2000} Hz, and noise stage ∈ {none, real}. Also run two attribution
references: odometry-only (anchor alone) and pure mechanized integration (anchor OFF —
the only configuration with true unbounded drift; report its drift RATE). Output: one
tidy CSV + static matplotlib figures + a pass/fail table against the pinned acceptance
thresholds (overall V̂ RMSE ≤ 2% of p95 scale ≈ 0.04 m/s; ≤ v_str = 0.01 m/s in
low-slip bins), and the selected `vel_filter_crossover_hz` for the v2 campaigns.

## 2. Architecture Pattern

Config-matrix sweep over a pure-function filter core — every (integrator, crossover,
rate, stage, branch-mode) cell is one deterministic pass of the same causal recursion
over each trajectory, so the filter functions are written once, deployment-identical,
and the audit is embarrassingly parallel over files.

## 3. Technology Constraints

- **Python:** 3.11 torch-free venv (this must run without the GPU env)
- **Required libraries:** numpy (filter recursions), pyarrow (augmented Arrow reads), pandas (tidy results), matplotlib (static figures ONLY — no interactive widgets)
- **Device targets:** CPU; ≤8 workers
- **Explicit exclusions:** NO learned components; NO torch; filter functions must be the SAME code contract later imported/mirrored by `sensor_frontend_v2.py` (write them import-cleanly); constants (v_str = 0.01, p95 scales, geometry) from the existing config/scaler sources — never invented

## 4. Component Breakdown

### `imu_observable.py` (module, `code_insights/tools_accel/`)
- **Type:** functions (numpy)
- **Responsibility:** From the original + `accel/<stem>_accel.arrow` sidecar pair (via `load_with_accel`), build the clean accelerometer observable per the code-verified convention (`a_x = dVx − ψ̇·Vy`, `a_y = dVy + ψ̇·Vx` — or the sim-EOM-consistent form settled in the augmentation brief), the gyro channel, and stage-2 corruption (accel white noise + constant bias + mounting-tilt gravity leakage; gyro noise + bias; seeded `SensorNoiseSpec` mirroring the rev-4 A2 brief), plus optional anti-alias LPF + decimation 2000 → 500 Hz.
- **Inputs:** joined trajectory arrays (`Vx, Vy, psi_dot` from the original; `dVx, dVy` from the sidecar; [T2k]), `SensorNoiseSpec`, target rate
- **Outputs:** `a_meas [T,2]`, `gyro [T]` at the requested rate
- **Depends on:** augmented Arrow columns

### `odometry.py` (module, same dir)
- **Type:** functions (numpy)
- **Responsibility:** Wheel-odometry body-velocity estimate: kinematic least-squares map from `(w, theta) [T,4]` to `V_odom [T,2]` using the standard mecanum inverse kinematics (geometry constants from `observer_v1_py/mecanum_observer/config.py`); slip-corrupted by nature — that is the point of the audit.
- **Inputs:** `w [T,4]`, `theta [T,4]` (+ optional ψ̇ for the yaw row)
- **Outputs:** `V_odom [T,2]`
- **Depends on:** geometry constants

### `comp_filter.py` (module, same dir — the deployment-contract core)
- **Type:** functions (numpy, strictly causal recursions)
- **Responsibility:** (a) strapdown mechanization step in both integrator variants — Euler (`V̂ += Δt·(a + ω×-correction with current V̂)`) and exact-rotation+AB2 (rotate V̂ by −ψ̇Δt via cos/sin, accel term `Δt·(1.5a[k] − 0.5a[k−1])`); (b) first-order complementary blend of the mechanized path (HPF) with `V_odom` (LPF) at a given crossover; (c) branch modes: `full` (blend), `odom_only` (anchor alone), `mech_only` (anchor off — unbounded-drift reference).
- **Inputs:** `a_meas [T,2]`, `gyro [T]`, `V_odom [T,2]`, `crossover_hz`, `integrator`, `mode`, `dt`
- **Outputs:** `V_hat [T,2]`
- **Depends on:** nothing (pure)

### `frontend_audit.py` (CLI script, same dir)
- **Type:** entry-point script
- **Responsibility:** Sweep runner: whitelisted file sample (`--sample N` per profile, default stratified ~200 files, ALWAYS including every `spin_creep` file — the weak-anchor stress case) × the config matrix (2 integrators × crossover grid (e.g. 0.2/0.5/1/2/5 Hz) × 2 rates × 2 stages × 3 modes); per (file, cell): run filter, compute error metrics with a transient exclusion of `3/(2π·crossover)` seconds; aggregate to tidy CSV; emit figures + acceptance table; recommend the crossover.
- **Outputs (files):** `observer_v1_py/report_frontend/frontend_audit.csv`, `fig_err_vs_crossover.png`, `fig_err_by_slip.png`, `fig_integrator_delta.png`, `fig_drift_rate_mech_only.png`, `ACCEPTANCE.md`
- **Depends on:** all above

### Metrics definition (inside `frontend_audit.py`)
- Per (file, cell): RMSE and max of `|V̂ − V_true|` per axis and combined, overall and binned by slip speed (reuse the report bin edges: 0.005/0.02/0.065/0.2/0.65/1.5 m/s) and by profile; for `mech_only` additionally the **drift rate** (least-squares slope of `|V̂ − V_true|` vs time, m/s per s) — the number that shows why the anchor is mandatory; for `odom_only` the slip-binned error — the number that shows why the accel path is mandatory.

## 5. File & Directory Structure

```
code_insights/tools_accel/
├── imu_observable.py        # accel observable + noise stage + decimation
├── odometry.py              # mecanum kinematic velocity estimate
├── comp_filter.py           # mechanization (2 integrators) + complementary blend
└── frontend_audit.py        # sweep CLI → report_frontend/ outputs
observer_v1_py/report_frontend/   # CSV + figures + ACCEPTANCE.md (generated)
```

## 6. Key Interfaces

```python
# tools_accel/comp_filter.py
def mechanize(a_meas: np.ndarray, gyro: np.ndarray, dt: float,
              integrator: str, v0: np.ndarray) -> np.ndarray:
    """
    Causal body-frame strapdown velocity integration.
    Args:  a_meas [T,2]; gyro [T]; integrator in {"euler", "rot_ab2"}; v0 [2]
    Returns: V_mech [T,2]
    """
    ...

def complementary(V_mech_step: Callable, V_odom: np.ndarray,
                  crossover_hz: float, dt: float, mode: str) -> np.ndarray:
    """
    First-order complementary blend; mode in {"full", "odom_only", "mech_only"}.
    Returns: V_hat [T,2]
    """
    ...

# tools_accel/frontend_audit.py
def audit_file(path: Path, cell: SweepCell, spec: SensorNoiseSpec) -> pd.DataFrame:
    """One trajectory × one config cell -> tidy metric rows (transient-excluded)."""
    ...
```

## 7. Data Flow

1. Original + sidecar pair (`load_with_accel`, alignment asserted by metadata) → arrays at 2000 Hz (`Vx, Vy, psi_dot, w, theta` ++ `dVx, dVy`).
2. `imu_observable` → `a_meas, gyro` at the cell's rate (+ noise if stage="real").
3. `odometry` → `V_odom`; `comp_filter` → `V̂` for the cell's integrator/crossover/mode.
4. Metrics vs stored `Vx, Vy` (truth), transient-excluded, slip-binned (slip speed from
   the stored Vpx/Vpy aux columns) → tidy rows → aggregate CSV → figures + ACCEPTANCE.md
   evaluating: overall RMSE ≤ 0.04 m/s; low-slip-bin RMSE ≤ 0.01 m/s; integrator delta
   (does rot_ab2 buy anything at 500 Hz?); recommended crossover.

## 8. Implementation Sequence

1. `comp_filter.py` — pure functions first; unit-test: noiseless constant-ψ̇ circular
   motion has a closed-form V(t); both integrators must converge to it, rot_ab2 at
   higher order.
2. `imu_observable.py` + `odometry.py` — validate on one original+sidecar pair: noiseless
   `mechanize` output tracks stored Vx/Vy (sanity for the convention), `V_odom` error
   grows with the file's slip bins.
3. `frontend_audit.py` — pilot on ~10 files incl. spin_creep, inspect, then full sample.

## 9. Operational Considerations

- **Numerical:** recursions in float64; transient exclusion mandatory (HPF settling);
  seed the noise per (file, cell) deterministically for reproducibility.
- **Initial condition:** `v0` from the first V_odom sample (deployment-realistic), and
  report sensitivity to `v0 = 0` (cold start) as a supplementary row.
- **Runtime:** sample × matrix is ~200 files × ~120 cells of cheap O(T) recursions —
  minutes on 4–8 workers; full-fleet is unnecessary (stratified sample + all
  spin_creep suffices; state the sample in the report header — no silent caps).
- **Figures:** static matplotlib; one figure per question, no dashboards.

## 10. Success Criteria

- [ ] Integrator unit test passes (closed-form circular-motion convergence orders)
- [ ] Noiseless `full`-mode V̂ tracks truth to integration tolerance on a real file
- [ ] `mech_only` shows measurable positive drift rate (stage "real"), `odom_only`
      shows slip-correlated error — both attribution references present in the CSV
- [ ] ACCEPTANCE.md renders a pass/fail against 0.04 m/s overall / 0.01 m/s low-slip
      thresholds per (integrator, crossover, rate, stage) and names the recommended
      crossover + integrator for the v2 campaigns
- [ ] Euler-vs-rot_ab2 delta is quantified (expected: small at 500 Hz — verify, don't assume)

## 11. Out of Scope

- The `sensor_frontend_v2.py` training-pipeline module itself (Observer-v2 brief owns
  it; it must import/mirror `comp_filter.py`'s contract)
- Kalman/learned estimators (fixed complementary filter only, per pinned decision)
- IMU noise-level calibration from a real datasheet (stage-2 uses placeholder spec
  values; calibration is follow-up)
- Any model training or checkpoint evaluation
