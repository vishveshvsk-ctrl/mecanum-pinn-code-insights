# Sensor-Suite Consolidation, Optical Flow Addition, and Physically-Grounded Noise Models

> **Generated:** 2026-07-29
> **Stack:** Julia 1.12.5 — `code_insights/` project. No Python, no new packages.
> **Scope:** Refactor + model addition in `hybrid_ctrl/`. Touches `sensors.jl`, `estimators.jl`, `scheduler.jl`, `config.jl`, `tuning/param_space.jl`.
> **Mode:** Pose-tracking only. Velocity-mode paths are removed from the estimator/harness layer.

---

## 1. Overview

Four changes to the sensing and estimation stack, in one coherent pass:

1. **Consolidate all sensor models into `sensors.jl`.** Today `SensorModel` produces only the IMU and wheel encoders, while `PoseFixModel` — unambiguously a sensor — is defined in `estimators.jl`. Move it, and make `sensors.jl` the single source of truth for every measurement the estimator consumes.
2. **Add an optical-flow sensor.** A downward-facing ground-relative velocity measurement, immune to wheel slip. This is the measurement that makes the slip states observable *algebraically* rather than through slow inference against the pose fix.
3. **Pin every `R` from the sensor datasheet.** Measurement noise is a physical property of the sensor, not a tuning knob. The current tuner searches `Rn_diag` over four orders of magnitude and converged on a 2400× Vx/Vy asymmetry that contradicts a closed-form result (see §7.2).
4. **Derive every `Q` from a physical time constant.** `bias_Qn` and `slip_Qn` currently sit within a factor of 2–7 of each other, when accelerometer bias drifts over minutes and slip over ~0.1 s — roughly six orders of magnitude apart. This requires a model change, not just a value change: the slip states must become mean-reverting rather than pure random walks.

Plus: **remove the velocity-mode harness** from the estimator layer, since every trajectory now runs `:pose`.

**System-level contract:** given a true plant state and a time, `sensors.jl` emits the measurements of four independent modalities at three independent rates, each corrupted by a datasheet-grounded error model. `estimators.jl` consumes them through pose-mode-only code paths, with `Q` and `R` derived from physical parameters rather than searched.

**Net effect on tuning:** the `eskf_param_space` should shrink from 21 tunable dimensions to roughly 4 (see §10).

---

## 2. Architecture Pattern

**Per-modality sensor models behind a suite container, with per-modality rates and RNG streams.**

The current design has one `SensorModel` struct, one `simulate_measurement` call, and one shared RNG — which works because there is effectively one sampling rate. With four modalities at three rates that breaks down in a specific and dangerous way: **a shared RNG makes the noise draw order depend on callback scheduling**, so reproducibility silently depends on the order the scheduler fires its periodic callbacks. The fix is one RNG stream per modality, seeded independently.

For `Q` and `R`, the pattern is **derive, don't search**. Every noise parameter gets a documented physical origin — a datasheet figure, a propagated covariance, or a time constant. Anything that cannot be traced to one of those is a modelling admission and must be named as such.

---

## 3. Technology Constraints

- **Julia 1.12.5**, existing `code_insights/` project. **No new packages.**
- Uses only what is already imported: `StaticArrays`, `LinearAlgebra`, `Random`, `Statistics`
- `LinearAlgebra.BLAS.set_num_threads(1)` — retain
- Determinism is a hard requirement: same seed ⇒ same measurement sequence, independent of callback firing order
- **Explicit exclusions:**
  - No changes to `plant.jl` or the physics
  - No changes to `controllers.jl` (its `:velocity` branches are out of scope here — the controller v2 brief handles that side)
  - No re-tuning in this brief; it only *shrinks the space* that a later run will search
  - No new estimator type (no UKF, no IEKF) — this is model and plumbing work

---

## 4. Component Breakdown

### `ImuModel` (`hybrid_ctrl/sensors.jl`)
- **Type:** `Base.@kwdef struct`
- **Responsibility:** Accelerometer + gyroscope measurement synthesis.
- **Inputs:** true state `u`, true derivative `du`, time `t`
- **Outputs:** `(a_x::Float64, a_y::Float64, g_z::Float64)`
- **Key params:** `sigma_acc`, `sigma_gyro`, `acc_bias`, `gyro_bias_rw`, `sf_gyro`, `bias_gyro`, `rate_hz`, `rng`
- **Depends on:** nothing

### `EncoderModel` (`hybrid_ctrl/sensors.jl`)
- **Type:** `Base.@kwdef struct`
- **Responsibility:** Wheel-angle and wheel-speed measurement synthesis, including quantization.
- **Inputs:** true state `u`
- **Outputs:** `(theta::SVector{4}, omega::SVector{4})`
- **Key params:** `enc_cpr`, `gear`, `sigma_omega`, `sf_omega`, `bias_omega::SVector{4}`, `rate_hz`, `rng`
- **Depends on:** nothing

### `FlowModel` (`hybrid_ctrl/sensors.jl`) — **NEW**
- **Type:** `Base.@kwdef struct`
- **Responsibility:** Downward optical-flow ground-relative velocity measurement at a body-frame offset.
- **Inputs:** true state `u` (needs `Vx`, `Vy`, `psidot`), time `t`
- **Outputs:** `Union{Nothing, SVector{2,Float64}}` — `nothing` on dropout or speed saturation
- **Key params:** `r_offset::SVector{2}` (lever arm), `sigma_flow`, `sf_flow` (height-induced scale error, drawn once per run), `quantum` (metric per count), `v_max_track`, `dropout_prob`, `rate_hz`, `rng`
- **Depends on:** nothing

### `PoseFixModel` (`hybrid_ctrl/sensors.jl`) — **MOVED** from `estimators.jl`
- **Type:** `Base.@kwdef mutable struct` (unchanged fields)
- **Responsibility:** Intermittent absolute pose measurement, two tiers.
- **Inputs:** true state `u`, time `t`
- **Outputs:** `Union{Nothing, SVector{3,Float64}}` — `(x, y, psi)`
- **Key params:** `tier::Symbol` (`:transit` / `:docking`), `fix_rate_hz`, `sigma_pos`, `sigma_psi`, `rng`
- **Depends on:** nothing. **Must not reference `EstimatorMod`.**

### `SensorSuite` (`hybrid_ctrl/sensors.jl`) — **NEW**
- **Type:** `Base.@kwdef struct`
- **Responsibility:** Container for the four modality models plus the independent-RNG seeding contract.
- **Key params:** `imu::ImuModel`, `enc::EncoderModel`, `flow::Union{FlowModel,Nothing}`, `fix::Union{PoseFixModel,Nothing}`, `seed::Int`
- **Depends on:** all four models above

### `NoiseSpec` (`hybrid_ctrl/sensors.jl`) — **NEW**
- **Type:** module-level functions (not a struct)
- **Responsibility:** Derive every estimator `R` block from the sensor models, so the estimator never invents a noise value. **This is the single most important new component.**
- **Inputs:** the relevant sensor model, plus operating-point values where the noise is state-dependent (`psidot`, wheel speeds)
- **Outputs:** `R` blocks as `SMatrix` of the right shape
- **Depends on:** `ImuModel`, `EncoderModel`, `FlowModel`, `PoseFixModel`, and `params` (for the wheel Jacobian)

### `TimeConstantQ` (`hybrid_ctrl/estimators.jl`)
- **Type:** module-level functions
- **Responsibility:** Derive the nuisance-state process-noise entries and decay rates from physical time constants.
- **Inputs:** `tau_bias`, `sigma_bias`, `tau_slip`, `sigma_slip`, `tau_gyro_bias`, `sigma_gyro_bias`, `dt`
- **Outputs:** per-state `(q_tick, decay_per_tick)` pairs
- **Depends on:** nothing

### `ESKFEstimator` (`hybrid_ctrl/estimators.jl`) — **MODIFIED**
- **Responsibility (changes):** consume the new flow channel; take `Q`/`R` from the derived sources; mean-revert the slip states; drop velocity-mode paths.
- **Removed fields:** `Rn_base`, `Qn`, `bias_Qn`, `slip_Qn`, `gyro_bias_Qn` as *free* parameters — replaced by derived values
- **Retained tunable fields:** `P0_scale`, `slip_threshold`, `pose_Qn`, and whichever adaptive terms survive §7.5

---

## 5. File & Directory Structure

```
code_insights/hybrid_ctrl/
├── sensors.jl        # HEAVILY MODIFIED — four modality models + SensorSuite + NoiseSpec
├── estimators.jl     # MODIFIED — PoseFixModel removed; flow channel added;
│                     #            Q from time constants; velocity-mode paths deleted
├── scheduler.jl      # MODIFIED — new flow callback; pose-fix callback re-pointed to SensorMod
└── config.jl         # MODIFIED — use_flow / f_flow / flow tier; `tracking` pinned to :pose

code_insights/tuning/
└── param_space.jl    # MODIFIED — eskf_param_space shrinks (R and Q dims removed)

code_insights/_tmp/
└── sensor_suite_validation.jl   # NEW — the §10 validation checks
```

No new files in `hybrid_ctrl/` — this is consolidation, so the module count should go *down* in responsibility spread, not up in file count.

---

## 6. Key Interfaces

Signatures only; bodies are stubs.

```julia
# ---- sensors.jl : per-modality synthesis ------------------------------------

"""
    simulate_imu(m::ImuModel, u, du, t) -> NTuple{3,Float64}

Accelerometer + gyro measurement. Body proper acceleration is formed from the
true RHS (a_x = V̇x − ψ̇·Vy, a_y = V̇y + ψ̇·Vx), then corrupted by white noise, a
constant bias, the gyro random walk, and the ψ̇-proportional scale-factor term.

Returns (a_x, a_y, g_z).
"""
function simulate_imu(m::ImuModel, u::AbstractVector, du::AbstractVector, t::Real) end

"""
    simulate_encoders(m::EncoderModel, u) -> NamedTuple

Wheel angles (quantized to motor-shaft counts) and wheel speeds (white noise +
speed-proportional scale factor + per-wheel constant bias).

Returns (theta::SVector{4}, omega::SVector{4}).

NOTE the per-wheel constant `bias_omega` is a BIAS, not noise. It maps through
the wheel Jacobian to a constant body-velocity offset — which is precisely what
the estimator's slip states see. It must NOT be folded into R (see section 7.2).
"""
function simulate_encoders(m::EncoderModel, u::AbstractVector) end

"""
    simulate_flow(m::FlowModel, u, t) -> Union{Nothing,SVector{2,Float64}}

Optical-flow ground-relative velocity at the sensor's body-frame offset r:

    v_sensor = [Vx − ψ̇·r_y,  Vy + ψ̇·r_x]

Error chain: height-induced scale factor (drawn ONCE per run, constant
thereafter) → white noise → quantization to the sensor's metric count size.

Returns `nothing` when either
  (a) |v_sensor| exceeds `v_max_track` (the sensor loses lock above its rated
      speed), or
  (b) the per-sample dropout draw fires (low-texture / reflective floor).

`nothing` means NO measurement this tick — the estimator must skip the update,
not substitute a zero.
"""
function simulate_flow(m::FlowModel, u::AbstractVector, t::Real) end

"""
    sample_pose_fix(m::PoseFixModel, u, t) -> Union{Nothing,SVector{3,Float64}}

Absolute pose measurement (x, y, psi) at the tier's noise level. Unchanged in
behaviour from the version currently in estimators.jl; relocated only.
"""
function sample_pose_fix(m::PoseFixModel, u::AbstractVector, t::Real) end

"""
    build_suite(kind::Symbol; seed::Int, flow::Bool, fix_tier::Symbol) -> SensorSuite

Construct all four modality models with INDEPENDENT RNG streams.

`kind` selects the grade: :default (legacy modest noise) or :realistic
(MEMS/encoder/LiDAR-grade, per the existing SensorModel(:realistic) values).

DETERMINISM CONTRACT: each modality's RNG is seeded from hash((seed, :modality)),
NOT from a shared stream. A shared RNG would make the draw sequence depend on the
order the scheduler fires its periodic callbacks, so reproducibility would depend
on callback scheduling. This is the single most important requirement in this file.
"""
function build_suite(kind::Symbol; seed::Int, flow::Bool, fix_tier::Symbol) end


# ---- sensors.jl : NoiseSpec — R derived, never searched ---------------------

"""
    R_wheel(enc::EncoderModel, params) -> SMatrix{2,2,Float64,4}

Measurement covariance of the wheel-derived body-velocity pseudo-measurement,
obtained by propagating per-wheel encoder noise through the O-config wheel
Jacobian Hω.

CLOSED-FORM RESULT — implement this, don't fit it. The four O-config columns are
mutually orthogonal, giving

    HωᵀHω = diag( 4/R², 4/R², 4(l+h)²/R² )

so the least-squares noise covariance σ_ω²(HωᵀHω)⁻¹ is EXACTLY DIAGONAL and the
Vx and Vy entries are EXACTLY EQUAL:

    R_wheel[1,1] = R_wheel[2,2] = σ_ω² · R² / 4

Any tuned value that makes these two differ is contradicting the kinematics. The
frozen config's 2400× asymmetry is the specific pathology this replaces.

Add the speed-proportional scale-factor contribution at the operating point when
`sf_omega` is nonzero.
"""
function R_wheel(enc::EncoderModel, params) end

"""
    R_gyro(imu::ImuModel, psidot::Real) -> Float64

Gyro measurement variance at the current operating point. HETEROSCEDASTIC: the
scale-factor term in the sensor model is redrawn every sample, so it behaves as
ψ̇-proportional white noise rather than a fixed bias:

    var = sigma_gyro² + (sf_gyro · psidot)²

A constant gyro R makes the filter overconfident exactly on the
continuously-rotating trajectories that currently fail.
"""
function R_gyro(imu::ImuModel, psidot::Real) end

"""
    R_flow(m::FlowModel, v_mag::Real) -> SMatrix{2,2,Float64,4}

Flow measurement covariance: white-noise floor plus the speed-proportional term,
plus the quantization variance (quantum²/12 for uniform quantization).

The height-induced SCALE error is deliberately NOT included here — it is a
persistent multiplicative bias, not noise, and belongs either in a dedicated
scale state or in the reported model-error budget. Putting it in R would let the
filter average away an error that does not average away.
"""
function R_flow(m::FlowModel, v_mag::Real) end

"""
    R_posefix(m::PoseFixModel) -> SMatrix{4,4,Float64,16}

Pose-fix covariance in the 4-D (x, y, cosψ, sinψ) form the ESKF uses, with the
σ_ψ² entries mapped onto the unit-circle tangent. Straight from the tier spec.
"""
function R_posefix(m::PoseFixModel) end


# ---- estimators.jl : Q from time constants ---------------------------------

"""
    gauss_markov_q(tau::Real, sigma::Real, dt::Real) -> Tuple{Float64,Float64}

Per-tick process-noise increment and per-tick decay factor for a first-order
Gauss-Markov (Ornstein-Uhlenbeck) state with correlation time `tau` and
stationary standard deviation `sigma`.

Returns (q_tick, decay_per_tick) where decay_per_tick is the value to place on
the state's diagonal of A (i.e. the mean-reversion rate), and q_tick is the Q
diagonal entry.

WHY BOTH: the current filter models slip as a PURE RANDOM WALK — sx, sy are held
constant in prediction, with only Q driving them. A random walk has no stationary
variance; it wanders without bound. Physically, slip returns to zero when grip is
restored. Setting Q correctly is NOT sufficient — the mean-reversion term must be
added to A as well, or the "time constant" has no representation in the model.
"""
function gauss_markov_q(tau::Real, sigma::Real, dt::Real) end

"""
    derive_process_noise(spec::NamedTuple, dt::Real) -> NamedTuple

Assemble all nuisance-state Q entries and decay rates from physical time
constants. `spec` carries tau/sigma for each of: accel bias, slip, gyro bias.

TARGET SEPARATION (the entire point of this change):

    accel bias   tau ~ 100-1000 s     (MEMS bias stability)
    gyro bias    tau ~ 100-1000 s
    slip         tau ~ 0.05-0.2 s     (MEASURE IT -- see below)

which puts q_slip / q_bias around 1e5-1e6. The frozen config has that ratio at
about 2. That six-order-of-magnitude error is why the filter cannot decide
whether a residual is a slow bias or a fast slip.

tau_slip MUST be measured, not assumed: fit the autocorrelation decay of the
`slip_detect` signal from existing simulation traces. It is also the quantity
that sets the required optical-flow sampling rate, so it is worth measuring
carefully and reporting.
"""
function derive_process_noise(spec::NamedTuple, dt::Real) end


# ---- estimators.jl : the new flow update -----------------------------------

"""
    apply_flow!(bus, est::ESKFEstimator, m::FlowModel, z_flow, params) -> Bool

Optical-flow measurement update. The 2×12 Jacobian selects body velocity and the
lever-arm coupling to yaw rate:

    H_flow = [ 1  0  -r_y   0 …0 |  0  0 |  0
               0  1   r_x   0 …0 |  0  0 |  0 ]

CRITICAL STRUCTURAL POINT: the slip columns (10, 11) are ZERO. The wheels measure
Vx + sx; the flow measures Vx. That zero is what makes the (velocity, slip) pair
non-degenerate, and it is the entire reason for adding this sensor.

Gated by the covariance-aware NIS test at the χ² value for a 2-D measurement --
NOT a tuned threshold. The gate is what catches flow dropout and low-texture
garbage, which is why flow is a MEASUREMENT rather than a prediction input.

Returns true if the measurement was accepted.
"""
function apply_flow!(bus, est::ESKFEstimator, m::FlowModel, z_flow::SVector{2}, params) end
```

---

## 7. Data Flow

### 7.1 Per-tick flow, four modalities, three rates

```
   1000 Hz  ── simulate_imu     ──► PREDICTION  (accel as input; P grows by Q)
                                        │
   1000 Hz  ── simulate_encoders ──► UPDATE: wheels (rows 1-2) + gyro (row 3)
                                        │        R from R_wheel + R_gyro
                                        │
 100-1000 Hz── simulate_flow    ──► UPDATE: 2-D body velocity, R from R_flow
                                        │        skipped entirely when `nothing`
                                        │
   10-100 Hz── sample_pose_fix  ──► UPDATE: 4-D (x, y, cosψ, sinψ), R from R_posefix
```

Each modality gets its own `PeriodicCallback` in `scheduler.jl`, mirroring the
existing pose-fix pattern. The filter propagates at 1 kHz and applies whichever
updates have arrived — asynchronous measurement handling is native to a Kalman
filter and needs no special machinery.

### 7.2 The R derivation, per channel

| Channel | R source | Notes |
|---|---|---|
| Wheels (2×2) | **closed form** `σ_ω²·R²/4·I` | exactly isotropic, exactly diagonal — see §6 |
| Gyro (scalar) | `σ_gyro² + (sf_gyro·ψ̇)²` | heteroscedastic; recompute per tick |
| Flow (2×2) | `σ_flow² + (sf·v)² + quantum²/12` | scale error excluded on purpose |
| Pose fix (4×4) | tier spec | already datasheet-driven |

Two subtleties that must not be got wrong:

- **`bias_omega` is not noise.** The per-wheel constant encoder bias maps through
  the wheel Jacobian to a constant body-velocity offset. That offset is what the
  slip states legitimately see and absorb. Folding it into `R` would model a
  persistent bias as something that averages away.
- **The flow height-scale error is not noise either**, for the same reason. Either
  give it a dedicated scale state (observable against the pose fix) or leave it in
  the reported model-error budget — but do not hide it in `R`.

### 7.3 The Q derivation and the slip model change

For each nuisance state, `tau` and `sigma` come from physics, and **both** the Q
entry and the A-matrix decay term are set from them. The slip states change from
pure random walks to mean-reverting processes — this is a model change, and it is
the part that actually encodes "slip is fast, bias is slow."

Order-of-magnitude target, for orientation (the implementation must compute these,
not hardcode them):

Note the three nuisance states are NOT the same kind of process, and must not be
parameterized the same way:

| State | Process type in the sim | Correct Q derivation |
|---|---|---|
| `bx, by` accel bias | **constant** (`acc_bias`, no walk) | Q ≈ 0; a small nonzero value only as robustness margin. `tau → inf`, no decay term. |
| `bg` gyro bias | **pure random walk** (`gyro_bias +=`) | `Q_tick = sigma_rw^2 * dt` — directly from the intensity, AFTER the step-0 fix. No `tau`, no decay term, no stationary variance. |
| `sx, sy` slip | **mean-reverting** (physically decays as grip returns) | Gauss-Markov: `gauss_markov_q(tau_slip, sigma_slip, dt)`, and a decay term in `A`. |

So `gauss_markov_q` applies to the slip states ONLY. `derive_process_noise` must
take `sigma_rw` for the gyro bias (not a tau/sigma pair) and must derive it from
the SAME `gyro_bias_rw` the corrected sensor model uses — a single shared constant,
not two independently-set numbers that can drift apart.

Order-of-magnitude target, for orientation:

```
accel bias : constant                          →  q ~ 0
gyro bias  : sigma_rw ~ 1e-4 /sqrt(s)          →  q_tick = 1e-8 * dt  = 1e-11 at dt=1ms
slip       : tau ~ 0.1 s, sigma ~ 0.1-0.5 m/s  →  q_tick ~ 1e-3

q_slip / q_bias  ~  1e5 - 1e6        (frozen config: ~2)
```

`tau_slip` is measured from existing traces, not assumed. It is also the number
that determines the minimum sufficient optical-flow rate, so measure it properly
and report it.

### 7.4 Velocity-mode removal

Delete from the **estimator/harness layer** only:

- `tuning/harness.jl` — the `use_pose_fix = tracking == :pose || haskey(...)`
  branch and the velref-defaults-to-`:transit` logic. Pose fix is now
  unconditional; tier is explicit.
- `HybridConfig.tracking` — pin to `:pose`, or remove the field and its branches.
- Any `run_mode == :velocity` dispatch in the estimator construction path.
- The `zupt_threshold` field, which is documented as dead ("kept for config
  compatibility") — either implement it or remove it; do not leave it.

**Out of scope:** `controllers.jl` retains its `:velocity` branches. Those belong
to the controller v2 brief.

### 7.5 What survives in the tuner

After §7.2 and §7.3, most of `eskf_param_space` is determined rather than searched:

| Parameter | Fate |
|---|---|
| `Rn_diag` (3) | **removed** — derived from encoder + gyro spec |
| `Qn_diag` (3) | **removed** — derived from accelerometer noise |
| `bias_Qn` (2), `slip_Qn` (2), `gyro_bias_Qn` (1) | **removed** — from time constants |
| `nis_thresh` (1) | **pinned** at the χ² value for the measurement dimension |
| `slip_threshold` (1) | **pinned** from the physical slip definition |
| `P0_scale` (1) | retained — genuine initialization choice |
| `pose_Qn` (1) | retained — but see the note below |
| `slip_R_inflate`, `r_boost`, `alpha_acc`, `alpha_yaw`, `grip_slip_scale`, `pose_slip_gain` | **re-examine** — several of these exist to compensate for wrong R/Q and may become unnecessary |

`pose_Qn` deserves scrutiny rather than retention by default: it represents
unmodelled pose error, and with a correct velocity model most of that error should
already be accounted for. It is also the parameter that saturates the pose-fix
gain in the frozen config, so whatever bound it keeps must be justified.

---

## 8. Implementation Sequence

0. **Fix the `gyro_bias_rw` discretization bug** (see §9) and re-measure the
   baseline `rate_rmse` on one ellipse. Do this first: it is one line, and it may
   materially change how hard the problem is that the rest of this brief addresses.
1. **`sensors.jl` restructure** — split `SensorModel` into `ImuModel` +
   `EncoderModel`, add `SensorSuite` with independent RNG streams. Validate the
   split is behaviour-preserving against a snapshot of the pre-split measurement
   sequence; this is the regression gate for everything downstream.
2. **Move `PoseFixModel`** into `sensors.jl`. Mechanical, but it must stop
   referencing `EstimatorMod`.
3. **`NoiseSpec` functions** — `R_wheel`, `R_gyro`, `R_posefix`. Leaf functions,
   pure. Validate `R_wheel` against the closed form analytically.
4. **Wire derived R into the ESKF**, remove `Rn_base` as a free field. Re-run an
   existing trajectory and record the change; expect it to differ from the frozen
   config, since the frozen config's R was unphysical.
5. **`gauss_markov_q` + `derive_process_noise`**, and add the mean-reversion terms
   to `A` for the slip states. This is the substantive model change — do it after
   R is settled so the two effects are separable.
6. **Measure `tau_slip`** from existing traces and feed it in. Do not skip to an
   assumed value.
7. **`FlowModel` + `simulate_flow`** — new sensor, no estimator changes yet.
   Validate the lever-arm term against truth on a pure-rotation trajectory.
8. **`apply_flow!` + scheduler callback + config flags** — depends on #7.
9. **Velocity-mode removal** — last, so it doesn't complicate debugging of #1–#8.
10. **Shrink `eskf_param_space`** — depends on #4, #5.

---

## 9. Numerical, Determinism & Reproducibility Considerations

*(Domain-adapted — this is a Julia simulation refactor, not a learning task.)*

- **Every noise field carries its provenance as an inline comment.** This is a hard
  requirement, not a style preference: these numbers end up in a publication, and a
  bare `sigma_gyro = 0.003` is unciteable. Each field in `ImuModel`, `EncoderModel`,
  and `FlowModel` gets a comment giving (a) the value in **datasheet units**, (b) the
  **reference part**, and (c) the **grade**. Pattern:

  ```
  sigma_gyro = 0.003     # rad/s @1kHz = 0.0077 deg/s/sqrt(Hz); ADIS16470 spec 0.008 -- industrial
  acc_bias   = 0.02      # m/s^2 = 2.04 mg; consumer uncalibrated (ADIS16470 in-run 13 ug) -- consumer
  quantum    = 31.75e-6  # m/count; ADNS-3080 @800cpi
  ```

  Appendix A holds the full table with conversions. Also record the datasheet
  revision consulted, since these get re-verified before publication. A module-level
  comment block should state the overall grade claim in one sentence: *industrial
  white-noise levels with uncalibrated consumer bias terms.*
- **RNG independence is the top risk.** One stream per modality, seeded from
  `hash((seed, :modality))`. With a shared stream, the noise sequence depends on
  the order `scheduler.jl` fires its callbacks — so adding the flow callback would
  silently change every existing result. Write a test that reorders the callbacks
  and asserts the measurement sequences are unchanged.
- **The restructure in step 1 must be behaviour-preserving in itself.** Splitting
  one `SensorModel` into three structs is pure refactoring; if the noise sequence
  changes, that is a bug in the split, not an intended model change. Snapshot the
  pre-split measurement sequence and diff against it. (The `if iszero(sm.sf_ω)`
  branches in v1 exist to control draw order — understand them before moving code.)
- **`nothing` means no measurement.** Flow dropout and pose-fix gaps must skip the
  update entirely. Substituting a zero, or a stale value, injects a fabricated
  measurement — a subtle and destructive bug class.
- **Heteroscedastic R must be recomputed per tick**, not cached. `R_gyro` depends
  on the current `ψ̇`; caching it defeats the purpose.
- **`gyro_bias_rw` is a discretization BUG and must be fixed** (`sensors.jl:91`
  in v1). The increment scales with `sqrt(t + 0.001)` instead of `sqrt(dt)` — the
  closed-form growth law of a Wiener process has been written into the per-step
  increment. Consequences:

  ```
  correct random walk :  std[b(T)] = sigma_rw * sqrt(T)          ~ 0.45 mrad/s at T=20 s
  v1 code             :  std[b(T)] = sigma_rw * T / sqrt(2*dt)   ~ 45   mrad/s at T=20 s
  inflation factor    :  sqrt(T / (2*dt))  =  100x  at T=20 s, dt=1 ms
  ```

  Two defects: the magnitude is ~100x too large, and the growth is LINEAR in T
  rather than sqrt(T), so it behaves as a deterministic drift rather than a random
  walk. Correct form: accumulate `sigma_rw * sqrt(dt) * randn` per step, so the
  increment std is constant and the accumulated std recovers `sigma_rw*sqrt(T)`.
  This requires `simulate_imu` to receive `dt` (or derive it from `rate_hz`).

  **Why this matters beyond tidiness:** the reported `rate_rmse` values in the v1
  diagnostics (0.010-0.045 rad/s) are the same order as the 45 mrad/s this bug
  produces, so some fraction of the observed yaw/heading difficulty may be a
  simulation artifact rather than an estimation limitation. Fix this FIRST and
  re-measure the baseline before drawing conclusions from any downstream result —
  it may change the size of the problem.

  Whichever form is used, the estimator's `tau_gyro_bias`/`sigma_gyro_bias` must
  match the sensor's actual behaviour, or the derived Q is wrong at its source.
- **Joseph-form covariance update and PSD symmetrization** must be retained on the
  new flow update, matching the existing pattern.
- **Mean-reversion changes `A`, which changes `F = I + dt·A`.** Verify the slip
  rows still give a stable discrete transition at `dt = 1 ms`: `|1 − dt/tau_slip| < 1`
  requires `tau_slip > dt/2`, comfortably satisfied at `tau ~ 0.1 s`, but assert it
  rather than assume it.

---

## 10. Success Criteria

- [ ] **Provenance:** every noise field in `ImuModel` / `EncoderModel` / `FlowModel`
      carries an inline comment with its datasheet-unit conversion, reference part,
      and grade; module header states the overall grade claim (Appendix A)
- [ ] **Gyro bias:** accumulated bias std scales as `sigma_rw*sqrt(T)` (verify by
      Monte Carlo over seeds at several T), and the baseline `rate_rmse` on one
      ellipse is re-measured and reported before and after the fix
- [ ] **Regression:** the step-1 struct split reproduces the pre-split measurement
      sequence bit-identically
- [ ] **Determinism:** permuting the scheduler's callback order leaves every
      modality's measurement sequence unchanged
- [ ] **Closed form:** `R_wheel` matches `σ_ω²·R²/4·I` analytically, and
      `R_wheel[1,1] == R_wheel[2,2]` exactly
- [ ] **Lever arm:** on a pure-rotation trajectory (`Vx = Vy = 0`, `ψ̇ ≠ 0`), the
      synthesized flow measurement equals `ω × r` to within its noise
- [ ] **Dropout handling:** with `dropout_prob` forced to 1, the filter runs to
      completion with no flow updates applied and no NaNs
- [ ] **Q separation:** the derived `q_slip / q_bias` ratio is ≥ 1e4
- [ ] **`tau_slip` measured** from traces and reported, with the autocorrelation fit
      quality
- [ ] **Slip observability:** with flow enabled, the `P[1,10]` correlation
      (velocity ↔ slip) drops substantially versus flow disabled on the same
      trajectory — the direct evidence that the degeneracy is broken
- [ ] **Space reduction:** `n_params(eskf_param_space())` drops from 22 to roughly 4–6
- [ ] No `run_mode == :velocity` path remains reachable in `estimators.jl`,
      `scheduler.jl`, or `tuning/harness.jl`

---

## 11. Out of Scope

- **Re-tuning.** This brief changes what *can* be tuned; a later run does the tuning.
- **`controllers.jl` velocity-mode branches** — controller v2 brief.
- **New estimator architectures** (UKF, invariant EKF, VB/Student-t robust filtering).
- **The PCRLB bound study** — separate brief; it consumes this one's noise
  specifications, so it should run *after* R is pinned.
- **Physics, plant, LuGre model** — untouched.
- **Hardware procurement and sensor selection.** No part is being chosen here; the
  reference parts below exist to give every noise number a citable provenance.
- **Backward compatibility with v1.** This work lands in a fresh v2 folder; the v1
  tree and its frozen configs are left untouched and are not a constraint on any
  design choice here.

---

## Appendix A — Reference parts and noise provenance

Every noise field in `sensors.jl` must trace to a line in this table, and must
carry that trace as an inline comment (see §9). Conversions used:
`N = sigma * sqrt(2/f_s)` for discrete white noise at sample rate `f_s`;
`1 rad/s = 57.30 deg/s`; bias std after T seconds `= sigma_rw * sqrt(T)`.

### IMU — reference part: **Analog Devices ADIS16470**

Stated grade for the paper: *industrial-grade MEMS white-noise levels with
UNCALIBRATED bias terms, representative of an uncompensated consumer
installation.* Say this explicitly — the white-noise and bias figures below come
from different grades, and that is a deliberate modelling choice, not an
inconsistency.

| Field | Sim value | Datasheet units | Reference | Grade |
|---|---|---|---|---|
| `sigma_gyro` | 0.003 rad/s @1 kHz | **0.0077 °/s/√Hz** | ADIS16470 rate noise density **0.008 °/s/√Hz** | industrial (near-exact match) |
| `sigma_acc` | 0.05 m/s² @1 kHz | **228 µg/√Hz** | typical MEMS 60–300 µg/√Hz | industrial/consumer |
| `acc_bias` | 0.02 m/s² | **2.04 mg** | consumer accel bias stability 1–10 mg (ADIS16470 in-run: 13 µg) | consumer, uncalibrated |
| `bias_gyro` | 0.003 rad/s | **0.172 °/s = 619 °/hr** | uncalibrated consumer turn-on bias | consumer, uncalibrated |
| `gyro_bias_rw` | 1e-4 /√s → **92 °/hr** at T=20 s | rate random walk | ADIS16470 in-run stability 8 °/hr; consumer ≳50 °/hr | consumer |

**Sanity check that the step-0 fix is mandatory, not cosmetic:**

```
corrected :  4.5e-4 rad/s at T=20 s  =  0.026 °/s  =    92 °/hr   → consumer grade, citable
v1 bug    :  4.47e-2 rad/s at T=20 s =  2.56  °/s  =  9220 °/hr   → OFF SCALE for any MEMS grade
```

9220 °/hr is roughly 100× worse than consumer grade and ~1000× worse than the part
whose white-noise spec the model otherwise matches. **No commercial device behaves
that way**, so the unfixed model cannot be defended in print at any grade.

### Optical flow — reference parts: **PixArt PMW3901** and **PixArt/Avago ADNS-3080**

| Field | Value | Source |
|---|---|---|
| `rate_hz` | 121 fps (PMW3901) / 6400 fps (ADNS-3080) | datasheets |
| `v_max_track` | 7.4 rad/s × height (PMW3901); 40 ips ≈ **1.02 m/s** (ADNS-3080) | datasheets |
| `quantum` | **31.75 µm/count** at 800 cpi (ADNS-3080) | datasheet |
| min height | 80 mm (PMW3901) | datasheet |
| `sf_flow` | **14.7 %/mm** height sensitivity (conventional optics); 0.1 %/mm (afocal) | measured, Sensors PMC4481980 |
| `sigma_flow` | ~2% of speed (std dev 17.6–25.7 mm over 1 m trials at 35 cm/s) | same |
| `dropout` | field-confirmed on reflective/low-texture floors | NZPRG mecanum trial |

Note `v_max_track` for both parts lands at or just below the platform's 0.3–1.0 m/s
operating range — this is a design constraint, not a rounding detail, and the
`nothing`-on-saturation return in `simulate_flow` is how it gets represented.

The afocal 0.1 %/mm figure is a research optical assembly, not an off-the-shelf
part. Use the conventional 14.7 %/mm for a commodity sensor and cite the afocal
number only as what a corrected design achieves.

### Verification requirement

All figures above were gathered from published datasheets and papers for
**scoping**. Re-verify against current datasheet revisions before any of them
appear in a publication, and record the revision in the `sensors.jl` comment.
