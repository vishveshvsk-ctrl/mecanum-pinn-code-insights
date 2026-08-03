# PCRLB Estimation Bound Without Wheel Odometry — Ellipse Friction-Circle Study

> **Generated:** 2026-07-29
> **Stack:** Julia 1.12.5 (simulation + trajectory screening) · Python 3.11.3 (bound recursion + figures, `claude-venv/mecanum`: numpy 2.4 / scipy 1.17 / pandas 3.0 / matplotlib 3.11)
> **Scope:** Offline analysis — no controller, estimator, or plant changes
> **Fixed conditions:** μ = 0.5, χ = 0.005 (both already the values in `trajectory_files_run_0p5_main/base.toml` — no new config required)

---

## 1. Overview

Compute the **posterior Cramér–Rao lower bound (PCRLB)** on state-estimation error for the
mecanum platform using **only the IMU and the exteroceptive pose fix — no wheel odometry**.
The bound is estimator-independent: it depends only on the motion model, the sensor noise
specification, and the true trajectory, so it answers "how well could *any* filter of this
model class do?" rather than "how well does our filter do?"

The study runs on **10 ellipse trajectories at μ = 0.5, χ = 0.005**, stratified 5/5 by
friction-circle utilization (near-saturating vs. lightly loaded), all in tangent heading mode
so the platform is in continuous rotation — the regime where the current ESKF degrades.

Three things come out of this:

1. **A reference floor.** The gap between the current ESKF's achieved RMSE and this bound
   says how much of the ellipse error is recoverable by better estimation versus how much is
   the sensor suite's information limit.
2. **The value of the wheel channel.** Computing the bound with and without wheel rows
   (§7.4 variants B0–B3) measures what wheel odometry contributes, and whether it contributes
   *negatively* once slip is accounted for.
3. **A decision on estimator selection** (the second question in the request) — see §11.

**System-level contract:** given a trajectory config directory and a combo list, emit per-
trajectory time series of the lower-bound standard deviations on position, velocity and
heading, for four measurement-model variants, alongside the achieved error of the frozen
ESKF, plus a summary table and figures grouped by friction-circle utilization.

---

## 2. Architecture Pattern

**Two-stage offline pipeline: Julia produces truth traces, Python computes the bound.**

This mirrors the existing repository convention (`save_controller_eskf_traces.jl` →
`plot_controller_eskf_traces.py`) — the simulation stack lives in Julia, the numerical
analysis and figures in the Python venv. The bound recursion itself is a small dense matrix
recursion over 8×8 (or 10×10) matrices; NumPy is more than adequate and keeps the analysis
and plotting in one place.

**The bound is a Riccati recursion, not an optimization.** Because the model is
additive-Gaussian and the Jacobians are evaluated along the *known* true trajectory, the
Tichavský PCRLB recursion reduces to a covariance propagation that runs in seconds per
trajectory. There is nothing to tune and nothing to converge — which is exactly what makes it
a defensible reference.

---

## 3. Technology Constraints

**Julia side**
- Julia 1.12.5, `Pkg.activate(code_insights/)` — existing project
- Uses: `Profiles` (trajectory builders + `PosRef` accessors), `SchedulerMod.run_hybrid`,
  `EstimatorMod.OracleEstimator`, `SensorMod.SensorModel`, `TOML`, `JSON`
- `LinearAlgebra.BLAS.set_num_threads(1)` — retain the existing determinism guarantee
- Process-level parallelism only (`Profiles.ACTIVE_KIND` is a global `Ref`,
  `SchedulerMod.ESTIMATOR_PROBE_LOG` an unlocked global `Dict`)

**Python side**
- Python 3.11.3 at `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`
- `numpy` — matrix recursion; `scipy` (`linalg.solve`, `signal` for autocorrelation fitting);
  `pandas` — summary tables; `matplotlib` — figures
- **No PyTorch, no GPU.** This is a deterministic linear-algebra recursion, not learning.

**Explicit exclusions**
- No changes to `estimators.jl`, `plant.jl`, `sensors.jl`, `controllers.jl`, or any tuner
- No re-tuning of anything
- No wheel-odometry rows in the primary bound (B0) — that is the point of the exercise
- Single μ and single χ — no friction sweep

---

## 4. Component Breakdown

### `select_ellipse_combos.jl`
- **Type:** Julia script
- **Responsibility:** Screen all ellipse combos by friction-circle utilization and emit the
  stratified 10-trajectory selection.
- **Inputs:** `run_dir::String`, `mu::Float64`, `n_per_group::Int`, `psi_mode_filter::String`
- **Outputs:** `ellipse_selection.json` — for each of 96 combos: `combo_idx`, `a`, `ratio`,
  `worbit`, `psi_mode`, `u_peak`, `u_rms`, `v_peak`, `psidot_peak`, `T_total`, `feasible`,
  `group` (`:loaded` / `:unloaded` / `:unselected`)
- **Depends on:** `Profiles` only — no simulation needed at this stage (utilization is read
  off the reference's analytic acceleration getters)

### `export_truth_traces.jl`
- **Type:** Julia script
- **Responsibility:** Run one closed-loop simulation per selected trajectory and export the
  true state, the simulated IMU stream, and the measured slip signal.
- **Inputs:** `ellipse_selection.json`, controller config (frozen ASMC), estimator selector
  (`:oracle` for the truth pass, `:eskf` for the achieved-error pass)
- **Outputs:** one `.npz` per trajectory containing, on a uniform 500 Hz grid of length `T`:
  - `t [T]`, `psi [T]`, `psidot [T]`, `Vx [T]`, `Vy [T]`, `X [T]`, `Y [T]` — truth
  - `a_x [T]`, `a_y [T]`, `g_z [T]` — simulated IMU stream
  - `slip [T]` — the `slip_detect` indicator (used only for the B2/B3 variants)
  - `eskf_Vx [T]`, `eskf_Vy [T]`, `eskf_psi [T]`, `eskf_X [T]`, `eskf_Y [T]` — achieved
    estimate from the frozen ESKF, for the overlay
- **Depends on:** `select_ellipse_combos.jl` output, `SchedulerMod`, `EstimatorMod`

### `ImuKinematicModel` (`bound_analysis/model.py`)
- **Type:** `dataclass`
- **Responsibility:** Hold the IMU-driven kinematic error model — state layout, the
  continuous Jacobian, the input-noise mapping, and the measurement Jacobians for each
  variant.
- **Inputs (construction):** sensor spec (`sigma_acc`, `sigma_gyro`, `acc_bias_rw`,
  `gyro_bias_rw`), pose-fix spec (`sigma_pos`, `sigma_psi`, `fix_rate_hz`), optional wheel
  spec (`sigma_wheel_vel`), optional slip Gauss–Markov spec (`tau_slip`, `sigma_slip`)
- **Outputs:** Jacobian and noise matrices as described in §6
- **Key constructor params:** `variant: str` — one of `"B0"`, `"B1"`, `"B2"`, `"B3"`
- **Depends on:** nothing (leaf)

### `pcrlb_recursion` (`bound_analysis/pcrlb.py`)
- **Type:** function
- **Responsibility:** Run the Tichavský information recursion along a true trajectory and
  return the per-step bound covariance.
- **Inputs:** `model: ImuKinematicModel`, truth arrays `[T]`, `dt: float`, `J0: ndarray [n,n]`
- **Outputs:** `J: ndarray [T, n, n]` (information) and `P_bound: ndarray [T, n, n]`
- **Depends on:** `ImuKinematicModel`

### `SlipStatistics` (`bound_analysis/slip_stats.py`)
- **Type:** `dataclass` + fitting function
- **Responsibility:** Fit a first-order Gauss–Markov description (time constant and
  stationary variance) to the measured slip signal, for the honest wheel variant B3.
- **Inputs:** `slip: ndarray [T]`, `dt: float`
- **Outputs:** `tau: float`, `sigma: float`, `fit_quality: float`
- **Depends on:** `scipy.signal`

### `run_bound.py`
- **Type:** script (driver)
- **Responsibility:** Loop trajectories × variants, run the recursion, assemble the summary
  table, write results.
- **Inputs:** trace directory, `ellipse_selection.json`
- **Outputs:** `bound_results.npz` (per-trajectory per-variant bound series),
  `bound_summary.csv` (one row per trajectory × variant)
- **Depends on:** all Python components above

### `plot_bound.py`
- **Type:** script
- **Responsibility:** Produce the figures described in §7.5.
- **Inputs:** `bound_results.npz`, `bound_summary.csv`
- **Outputs:** PNG/PDF figures under `reports/`
- **Depends on:** `run_bound.py` outputs

---

## 5. File & Directory Structure

```
code_insights/
├── bound_analysis/
│   ├── select_ellipse_combos.jl      # NEW — friction-circle screening of all 96 combos
│   ├── export_truth_traces.jl        # NEW — 10 closed-loop sims → .npz truth traces
│   ├── model.py                      # NEW — ImuKinematicModel: state, Jacobians, noise
│   ├── pcrlb.py                      # NEW — Tichavsky information recursion
│   ├── slip_stats.py                 # NEW — Gauss-Markov fit of the slip signal (B3)
│   ├── run_bound.py                  # NEW — driver: trajectories x variants
│   ├── plot_bound.py                 # NEW — figures
│   ├── traces/                       # NEW — one .npz per selected trajectory
│   └── reports/                      # NEW — figures + summary CSV + selection JSON
├── run_bound_analysis.bat            # NEW — end-to-end launcher
└── instructions/
    └── pcrlb-no-odometry-ellipse-bound.md   # this brief
```

---

## 6. Key Interfaces

Signatures only — bodies are `...`.

```python
# ---- bound_analysis/model.py -----------------------------------------------

@dataclass
class ImuKinematicModel:
    """
    IMU-driven kinematic error model for the PCRLB.

    State (n = 8 for B0/B1/B2, n = 10 for B3):
        [Vx, Vy, psi, X, Y, bx, by, bg]           (+ [sx, sy] for B3)
         0   1    2   3  4   5   6   7               8   9

    The IMU is treated as a NOISY INPUT to the dynamics, not as a measurement.
    This is what makes the process noise Q a property of the SENSOR SPEC rather
    than a tuning knob -- and therefore what makes the result a bound.
    """
    variant: str
    sigma_acc: float
    sigma_gyro: float
    acc_bias_rw: float
    gyro_bias_rw: float
    sigma_pos: float
    sigma_psi: float
    fix_rate_hz: float
    sigma_wheel_vel: Optional[float] = None
    tau_slip: Optional[float] = None
    sigma_slip: Optional[float] = None

def state_jacobian(model: ImuKinematicModel, psi: float, psidot: float,
                   Vx: float, Vy: float, dt: float) -> np.ndarray:
    """
    Discrete state transition Jacobian F = I + dt * A, evaluated at the TRUE
    state. A carries: Coriolis coupling between Vx and Vy through psidot, the
    heading integrator, the body-to-world rotation on the position rows, and the
    bias injection rows (accel biases enter velocity, gyro bias enters heading).

    Returns:
        F: [n, n]
    """
    ...

def process_noise(model: ImuKinematicModel, psi: float, dt: float) -> np.ndarray:
    """
    Q = G * Sigma_imu * G^T * dt + bias random-walk terms.

    Sigma_imu = diag(sigma_acc^2, sigma_acc^2, sigma_gyro^2) is the SENSOR spec;
    G maps input noise into the state. This is the load-bearing modelling choice:
    Q is DERIVED, not fitted.

    Returns:
        Q: [n, n]
    """
    ...

def measurement_model(model: ImuKinematicModel, k: int, slip_k: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measurement Jacobian and noise for step k, per variant (see section 7.4).
    Returns a zero-row H when no measurement is scheduled at this step (the pose
    fix is at fix_rate_hz, the state recursion at the trace rate).

    Args:
        k:      step index (decides pose-fix scheduling)
        slip_k: measured slip magnitude at step k (used by B2/B3 only)
    Returns:
        H: [m, n]   (m = 0, 3, or 5 depending on variant and schedule)
        R: [m, m]
    """
    ...


# ---- bound_analysis/pcrlb.py -----------------------------------------------

def pcrlb_recursion(model: ImuKinematicModel, truth: Dict[str, np.ndarray],
                    dt: float, J0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tichavsky posterior CRLB information recursion along a known true trajectory:

        J_{k+1} = inv( Q_k + F_k @ inv(J_k) @ F_k.T ) + H_{k+1}.T @ inv(R) @ H_{k+1}

    with F_k, H_k evaluated at the TRUE state. The bound on any unbiased
    estimator's error covariance is P >= inv(J_k).

    Args:
        model: variant-configured model
        truth: dict of [T] arrays -- t, psi, psidot, Vx, Vy, X, Y, slip
        dt:    trace timestep (uniform)
        J0:    initial information [n, n] (inverse of the initial-uncertainty prior)
    Returns:
        J:       [T, n, n]
        P_bound: [T, n, n]   (= inv(J), symmetrised)
    """
    ...

def bound_sigmas(P_bound: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Reduce the bound covariance to reportable scalar standard deviations.

    Returns dict of [T] arrays:
        sigma_pos  -- sqrt(P[3,3] + P[4,4])   horizontal position
        sigma_vel  -- sqrt(P[0,0] + P[1,1])   body velocity
        sigma_psi  -- sqrt(P[2,2])            heading
        sigma_bg   -- sqrt(P[7,7])            gyro bias
    """
    ...


# ---- bound_analysis/slip_stats.py ------------------------------------------

def fit_gauss_markov(slip: np.ndarray, dt: float) -> Tuple[float, float, float]:
    """
    Fit a first-order Gauss-Markov process to the measured slip signal by its
    autocorrelation decay.

    Rationale: treating slip as WHITE measurement noise (variant B2) understates
    its harm, because real slip is strongly correlated over the slip interval -- a
    correlated error is not averaged away by more samples. B3 augments the state
    with the slip pair so the bound sees the correlation honestly.

    Returns:
        tau:         correlation time constant [s]
        sigma:       stationary standard deviation [m/s]
        fit_quality: R^2 of the exponential fit, for reporting
    """
    ...
```

```julia
# ---- bound_analysis/select_ellipse_combos.jl -------------------------------

"""
    utilization(ref::Main.Profiles.PosRef, mu::Float64; n_grid::Int) -> NamedTuple

Body-level friction-circle utilization of a reference trajectory:
u(t) = hypot(Axo(t), Ayo(t)) / (mu * g), evaluated on a uniform grid over the
trajectory duration. Uses the PosRef's analytic acceleration getters -- NO
simulation is required for screening.

NOTE this is a BODY-level proxy. The exact per-wheel check would distribute the
wrench through the O-config allocation and compare against mu * N_per_roller;
the proxy is adequate for STRATIFICATION and should be labelled as such in the
report.

Returns: (u_peak, u_rms, v_peak, psidot_peak, T_total)
"""
function utilization(ref, mu::Float64; n_grid::Int) end

"""
    select_stratified(run_dir, mu; n_per_group, psi_mode_filter) -> Vector{NamedTuple}

Screen every ellipse combo, apply the feasibility filter, then take the top
`n_per_group` by u_peak as the LOADED group and the bottom `n_per_group` as the
UNLOADED group.

Feasibility filter (both must hold, per the established envelope finding):
  - peak commanded |Vy| within the platform velocity ceiling
  - combo present in the diagnostics whitelist (combined_reco starts with "keep")

If no feasible combo reaches u_peak >= 0.7, take the highest available and label
it `hardest_available` -- the same convention already used in
runs_controller/subset_manifest.json.
"""
function select_stratified(run_dir::String, mu::Float64;
                           n_per_group::Int, psi_mode_filter::String) end
```

---

## 7. Data Flow

### 7.1 Trajectory selection (no simulation)

1. Parse `ellipse_mu_0p5.toml` — 96 combos over `(a, ratio, worbit, psi_mode, n_laps, Twarp,
   theta_e_deg, psi0_deg)`.
2. **Filter to `psi_mode == "tangent"` (combos 1–72).** Tangent mode makes heading track the
   path tangent, so the platform completes one full rotation per lap — this is the
   continuous-rotation regime under investigation. (Crab mode, combos 73–96, holds heading
   fixed; see §11 for the optional contrast group.)
3. For each surviving combo: `Profiles.build("ellipse", cfg)` → `PosRef`, then evaluate
   `u(t) = hypot(Axo(t), Ayo(t)) / (mu*g)` on a uniform grid.
4. Apply the feasibility filter, rank by `u_peak`, take top 5 (LOADED) and bottom 5
   (UNLOADED). Write `ellipse_selection.json`.

Expected structure: `u_peak` scales roughly with `a * worbit^2` modulated by `ratio`, so the
LOADED group should concentrate on large `a` with high `|worbit|` and the UNLOADED group on
small `a` with low `|worbit|`. Verify this holds — if the two groups are not clearly
separated in `u_peak`, the stratification is not meaningful and the selection must be
revisited before proceeding.

### 7.2 Truth-trace export

For each of the 10 selected trajectories, run **two** closed-loop simulations:
- **Truth pass** — `OracleEstimator(:clean)`, frozen ASMC controller. Provides the true state
  trajectory along which the bound Jacobians are evaluated, plus the simulated IMU stream and
  the `slip_detect` signal.
- **Achieved pass** — the frozen ESKF (`runs_eskf_noellipse_v2/eskf_dxnes/best_config.json`),
  same controller and seed. Provides the achieved estimate for the overlay.

Both resampled onto a uniform 500 Hz grid (`saveat_hz` already 500) and written to one `.npz`
per trajectory. 20 simulations total at roughly 10–25 s each — a few minutes.

### 7.3 The bound recursion

State `x = [Vx, Vy, psi, X, Y, bx, by, bg]`. The IMU enters as a **noisy input**:

- body velocity is driven by `(a_x - bx)`, `(a_y - by)` plus Coriolis coupling through
  `psidot = (g_z - bg)`
- heading is driven by `(g_z - bg)`
- world position integrates body velocity through the rotation at `psi`
- the three bias states are random walks

`Q` is formed by mapping the IMU input noise `diag(sigma_acc^2, sigma_acc^2, sigma_gyro^2)`
through the input matrix, plus the bias random-walk intensities. **This is the load-bearing
choice that makes the result a bound**: `Q` is derived from the sensor spec in `sensors.jl`,
not fitted. The values come directly from `SensorModel` — `sigma_acc = 0.05`,
`sigma_gyro = 0.005` for `:default`, or the `:realistic` constructor's values for the
realistic case. Run both sensor grades.

Measurement: the pose fix at the `:docking` tier — 100 Hz, `sigma_pos = 0.01 m`,
`sigma_psi ≈ 0.5°` — measuring `[X, Y, psi]`. (The filter's actual 4-D `[X, Y, cos psi, sin psi]`
form is equivalent to first order; the 3-D form is used here for clarity and should be noted
in the report.) The recursion runs at the trace rate with a zero-row measurement on steps
where no fix is scheduled.

Then iterate the Tichavský recursion of §6 and invert to get `P_bound`.

### 7.4 The four variants

| Variant | Measurement rows | Purpose |
|---|---|---|
| **B0** | pose fix only | **The requested bound.** Best achievable with no wheel odometry. |
| **B1** | pose fix + wheel velocity with encoder noise only | Upper limit of what wheels could contribute if slip were known exactly. |
| **B2** | pose fix + wheel velocity, R inflated by the empirical slip variance (white) | Naive wheel use. **Optimistic** — treats correlated slip as white. |
| **B3** | pose fix + wheel velocity, slip augmented as a Gauss–Markov pair `[sx, sy]` (n = 10) | Honest wheel use. Correlation makes slip far harder to average away. |

The comparisons that matter:
- **B0 vs. achieved ESKF** — is the current filter beating a filter that ignores the wheels?
- **B0 vs. B1** — the theoretical maximum value of the wheel channel.
- **B0 vs. B3** — the realistic value of the wheel channel under correlated slip. **If B3 is
  worse than B0 on the loaded group, wheel odometry is net-harmful during sustained slip.**
- **B2 vs. B3** — how much the white-noise slip assumption flatters the wheels. This is the
  quantitative case for or against the current filter's `slip_R_inflate` mechanism, which
  makes exactly the white-noise assumption.

### 7.5 Outputs

- **Figure 1** — `sigma_pos(t)` bound curves for B0/B1/B2/B3 with the achieved ESKF error
  overlaid, one panel per trajectory, LOADED and UNLOADED groups in separate rows.
- **Figure 2** — headline: achieved-to-bound ratio versus `u_peak` across all 10
  trajectories. If the ratio grows with friction-circle utilization, the estimator's deficit
  is slip-driven — the central claim.
- **Figure 3** — `sigma_bg(t)`, the gyro-bias bound, to show how quickly the 100 Hz heading
  measurement pins the bias. This settles the gyro-bias observability question directly.
- **`bound_summary.csv`** — one row per trajectory × variant × sensor grade:
  `combo_idx, group, u_peak, u_rms, variant, sensor_grade, sigma_pos_rms,
   sigma_vel_rms, sigma_psi_rms, achieved_pos_rms, ratio_achieved_to_bound`.

---

## 8. Implementation Sequence

1. **`select_ellipse_combos.jl`** — pure `Profiles` work, no simulation, no dependencies.
   Inspect the utilization table before going further; if the LOADED/UNLOADED groups do not
   separate cleanly, everything downstream is meaningless.
2. **`export_truth_traces.jl`** — depends on #1. Verify one trace end-to-end (truth position
   matches the reference to within tracking error; the IMU stream is consistent with the
   differentiated truth) before running all 10.
3. **`model.py`** — leaf; the state layout and Jacobians. **Validate against a straight-line
   constant-velocity case where the bound has a known closed form** before trusting curved
   trajectories.
4. **`pcrlb.py`** — depends on #3. Validate the recursion: with no measurements the bound must
   grow monotonically; with a perfect measurement (`R → 0`) it must collapse to zero on the
   measured states.
5. **`slip_stats.py`** — depends on #2 (needs the slip traces). Only required for B3.
6. **`run_bound.py`** — depends on #2–#5. Start with B0 only; add variants once B0 is
   validated.
7. **`plot_bound.py`** — depends on #6.

---

## 9. Numerical & Determinism Considerations

*(Domain-adapted from the template's ML section — this is a deterministic linear-algebra
recursion, not a learning task.)*

- **Information-form conditioning.** The recursion inverts `J` every step. `J` becomes
  ill-conditioned for weakly-informed states early on (the bias states before enough fixes
  have accumulated). Use `scipy.linalg.solve` / Cholesky rather than explicit `inv`, and
  symmetrize (`0.5*(J + J.T)`) every step. If Cholesky fails, that is a genuine signal that
  the prior `J0` is too weak — do not silently regularize past it.
- **`J0` is a modelling choice and must be reported.** It encodes the initial uncertainty.
  Set it from the same prior the ESKF uses (`P0_scale`, with the position and bias entries as
  in `init_eskf!`) so the comparison is fair, and state the choice in the report.
- **The bound is only a bound for its own model class.** This is the most important caveat
  and must appear in the report: it bounds filters that treat the IMU as their only dynamic
  input and the pose fix as their only correction. An estimator that additionally knows the
  slip physics — a learned slip model — is **not** bounded by this, because it has strictly
  more information. Do not present B0 as a universal limit.
- **Sensor spec must be read from `sensors.jl`, not hardcoded.** `sigma_acc`, `sigma_gyro`,
  `gyro_bias_rw`, `acc_bias` all live in `SensorModel`. Any divergence silently invalidates
  the bound.
- **Flag the `gyro_bias_rw` accumulation.** `sensors.jl:91` increments the gyro bias by
  `gyro_bias_rw * sqrt(t + 0.001) * randn` — the increment scales with `sqrt(t)`, not
  `sqrt(dt)`, so this is not a standard Brownian walk and its variance grows faster than one.
  Log the realized bias trace and fit the *empirical* random-walk intensity for the bound
  rather than assuming `gyro_bias_rw` is a per-`sqrt(s)` intensity. Getting this wrong shifts
  the whole heading bound.
- **Heteroscedastic sensor noise.** `sensors.jl` redraws the scale-factor term every sample,
  so gyro noise std is approximately `sqrt(sigma_gyro^2 + (sf_gyro * psidot)^2)` and the
  wheel noise scales with wheel speed. On a continuously-rotating ellipse this is a real
  effect. Use the `psidot`-dependent form in `Q` for the `:realistic` grade — it is the
  correct model and it is trajectory-dependent, which is part of the point.
- **Determinism.** Fixed sensor seed across all runs; `Xoshiro(0)` for combo resolution per
  the existing convention; single-threaded BLAS.
- **Checkpointing.** The Julia trace export is the expensive stage (20 simulations). Write
  each `.npz` as it completes and skip existing files on re-run so an interrupted export
  resumes. The Python recursion is seconds — no checkpointing needed.

---

## 10. Success Criteria

- [ ] **Validation:** on a synthetic straight-line constant-velocity trajectory with no
      measurements, `sigma_pos(t)` grows as expected for a double integrator driven by
      accelerometer noise (`t^{3/2}` scaling) — confirms `Q` and `F` are correct
- [ ] **Validation:** with `R → 0` on the pose fix, `sigma_pos` and `sigma_psi` collapse to
      approximately zero at every fix instant
- [ ] LOADED and UNLOADED groups are cleanly separated in `u_peak` (report both ranges); if
      the loaded group cannot reach `u_peak >= 0.7` under the feasibility filter, this is
      reported explicitly rather than papered over
- [ ] B0 bound computed for all 10 trajectories at both sensor grades, no numerical failures
- [ ] `sigma_bg(t)` converges — demonstrating the gyro bias is well determined by the 100 Hz
      heading measurement, or showing it is not
- [ ] Achieved-to-bound ratio reported per trajectory; the relationship between that ratio
      and `u_peak` is stated as a finding **whatever its sign** — including if the ratio turns
      out to be flat, which would falsify the slip-driven-deficit hypothesis
- [ ] B0/B1/B2/B3 comparison yields an explicit verdict on whether wheel odometry is net
      positive or net negative on the loaded group

---

## 11. Out of Scope

- **Any estimator, controller, or plant modification.** This is a measurement exercise.
- **Friction sweep.** Single μ = 0.5, χ = 0.005, as specified.
- **Crab-mode ellipses.** Combos 73–96 hold heading fixed and would isolate the rotation axis
  from the loading axis. A worthwhile follow-up (a 2×2 design over spin × loading), but out of
  scope for this iteration, which follows the requested 10-trajectory single-axis split.
- **Per-wheel friction-circle computation.** The body-level proxy is used for stratification;
  the exact per-roller normal-load version is a refinement, not a requirement.
- **Building a fuzzy estimator selector.** The second question in the request is *answered* by
  this study rather than implemented: the B0-versus-B3 comparison establishes whether the
  wheel channel carries positive information during sustained slip. If it does not, the
  correct response is to drop or continuously de-weight that channel inside a single filter —
  not to arbitrate between filters. See the accompanying discussion; a multiple-model bank
  over the same dynamics was already tried and abandoned (the `IMMKalmanEstimator` →
  `ESKFEstimator` pivot, motivated in `estimators.jl:871` by IMM lockout-divergence
  bistability), and a fuzzy selector is a less principled version of that same architecture.
