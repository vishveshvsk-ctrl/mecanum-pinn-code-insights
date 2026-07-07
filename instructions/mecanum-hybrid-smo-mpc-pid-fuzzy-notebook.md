# Mecanum Hybrid Control Notebook — SMO / MPC / Kalman-PID / Fuzzy over a Roller+LuGre Plant

> **Generated:** 2026-07-07
> **Stack:** Julia 1.10+, OrdinaryDiffEq.jl, DiffEqCallbacks.jl, StaticArrays.jl, OSQP.jl (or JuMP+OSQP), DataFrames/Arrow (reuse `datastore.jl`)
> **Scope:** Simulation notebook — true plant + multi-rate hybrid controller with run-time block selection
> **Downstream target:** A coding model that will implement `Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1.ipynb` from this spec. No code is written here.

---

## 1. Overview

Build a new Julia simulation notebook that wraps a **hybrid, run-time-selectable controller** around the existing high-fidelity Mecanum plant (four-wheel roller kinematics + dynamic-LuGre bristle friction with Adamov spin→translation coupling). The plant is reused essentially verbatim from `run_one.jl` (`dynamics_full_mf_asmc!`, lines 658–763) but is **converted to a torque-input plant**: the controller is no longer computed inside the ODE RHS; instead a zero-order-hold (ZOH) wheel-torque vector is supplied on a shared mutable bus and updated by scheduled discrete controllers.

The controller reproduces the Pham & Han (2025) SMO–MPC–PID–Fuzzy hybrid, **adapted to two hard constraints of this project**: (1) only IMU + wheel encoders are measured — absolute pose (x, y, ψ) is *never* measured, only dead-reckoned, so it drifts; (2) actuation goes through a **youBot DC-motor + gearbox model** (parameters from `youbot_driver/config/youbot-base.cfg`): the control variable is **motor voltage**, mapped through the motor model to wheel torque, so voltage/current saturation is physical (matching the paper's voltage-domain constraints). The carried-over adaptive sliding-mode controller (ASMC, with its cubic + linear-leakage gain dampers) is promoted to a peer control law alongside MPC and Kalman-PID; a fuzzy supervisor arbitrates the blend weights so any subset of controllers can be enabled and compared.

**System-level contract:**
- **Input:** a `HybridConfig` (which blocks are active, sampling rates, tracking mode) + an active reference trajectory (`PosRef` or `VelRef` from `profiles.jl`) + physics point (μ, χ, friction case, LuGre params).
- **Output:** a solution object + an Arrow/DataFrame log of true state, estimated state, per-controller wrenches, blend weights, applied wheel torques, and tracking errors over the run — written through the existing `DataStore` schema so downstream diagnostics/PINN tooling keep working.

---

## 2. Architecture Pattern

**Plant-as-input ODE + multi-rate discrete supervisory controller (ZOH bus), with a fuzzy-weighted control-allocation mixer.**

Justification:
- The stiff LuGre bristle ODE (σ₀ ≈ 1.6e3) demands a stiff continuous integrator (`TRBDF2`/`RadauIIA5`, `dtmax≈1e-3`, `reltol 1e-8`) — keep **one** continuous integrator for the plant.
- The paper's blocks are discrete and multi-rate (SMO 1 kHz, MPC 100 Hz, PID 100 Hz, fuzzy 50 Hz) and MPC needs a QP solve — these cannot live inside the RHS. Implement them as `PeriodicCallback`s that read sensors from `integrator.u`, run their update, and write results onto a `ControllerBus`; the RHS only reads `bus.tau_wheel` (ZOH). This mirrors how `run_one.jl` already used `PeriodicCallback` for progress.
- Run-time selection = toggling which callbacks are installed and how the mixer normalizes weights; a disabled block simply contributes nothing and costs nothing.

---

## 3. Technology Constraints

- **Language:** Julia 1.10+ (match `Project.toml`/`Manifest.toml`; run from `code_insights/`).
- **ODE:** `OrdinaryDiffEq.jl` — stiff solver (`TRBDF2` default; `RadauIIA5` fallback). Continuous plant integrated with discrete `PeriodicCallback`s via `DiffEqCallbacks.jl`.
- **QP (MPC):** `OSQP.jl` directly (warm-started, sparse) — preferred for 100 Hz real-time feel. `JuMP` + `OSQP` acceptable for readability at a small speed cost.
- **Static data:** `StaticArrays.jl` for all 3- and 4-vectors/matrices (stack allocation, as in `run_one.jl`).
- **I/O:** reuse `profiles.jl` (`Profiles`, `current_ref`/`current_posref`, `global_to_local_frame`, `is_velref`, `active_ref`) and `datastore.jl` (`DataStore` Arrow schema + label extraction) unchanged.
- **Device targets:** CPU only (single-threaded per run; the sweep driver parallelizes across runs, ≤8 threads per project convention).
- **Explicit exclusions:** **no super-twisting/linear DOB** (removed); no absolute-pose sensor (IMU+encoder only); no PyTorch/ML in this notebook.

### youBot actuator parameters (from `youbot_driver/config/youbot-base.cfg`)
Hard-code these as the `MotorParams` defaults (§4.4b); mark the electrical pair as calibratable.

| Param | Symbol | Value | Notes |
|-------|--------|-------|-------|
| Torque constant | Kt | 0.0335 N·m/A | `TorqueConstant_[newton_meter_divided_by_ampere]` |
| Gear reduction (motor:wheel) | G | 9405/364 ≈ 25.84 | `GearRatio_numerator=364`, `_denominator=9405` (ratio = wheel/motor) |
| Encoder resolution | cpr | 4000 ticks/round | `EncoderTicksPerRound`; on the **motor** shaft ⇒ effective wheel resolution = 4000·G |
| Wheel radius (youBot) | r_yb | 0.0475 m | `WheelRadius_[meter]` — cross-check against `base.toml` `platform.geometry.R` (0.04); keep the plant's R, note the discrepancy |
| Commutation current | i_comm | 0.2 A | `CommutationCurrent_[ampere]` |
| Back-EMF constant | Kb | ≈ Kt = 0.0335 V·s/rad | ideal-motor SI equality; refine from datasheet |
| Winding resistance | Ra | **calibrate** (≈ a few Ω) | not in cfg — from maxon/TMCM datasheet |
| Winding inductance | La | **calibrate** (small) | not in cfg — default **quasi-static La→0** |
| Bus voltage limit | V_max | ≈ 24 V (youBot battery) | not in cfg — verify; controllers saturate here |
| (reference only) joint PID | P,I,D | 50, 20, 0 (I_max 1000) | `trajectory_controller_*` — youBot's own joint controller, not reused |

---

## 4. Component Breakdown

### 4.1 `plant_rhs!` / plant state layout
- **Type:** in-place ODE RHS `function`
- **Responsibility:** advance the physical plant given a ZOH **motor-voltage** input; contains **no** controller logic. Wheel torque is produced *inside* the RHS by the youBot motor model (§4.4b) from the commanded voltage and the current motor speed.
- **Inputs:** `du, u`, `p::PlantODEParams`, `t`
- **Outputs:** writes `du` in place
- **State layout — two variants, pick via `MotorParams.dynamic_electrical`:**
  - **Quasi-static electrical (default, La→0): 30-D** — no motor current states; wheel torque is algebraic `τ_wheel,i = G·Kt·i_i`, `i_i = clamp((V_i − Kb·G·ω_i)/Ra, ±i_max)`.
  - **Full electrical (La calibrated): 34-D** — append `[31:34] = i₁..i₄` motor currents with `L_a·di/dt = V − R_a·i − K_b·G·ω`.

  | idx | symbol | meaning |
  |-----|--------|---------|
  | 1–3 | Vx, Vy, ψ̇ | body velocities |
  | 4 | ψ | heading |
  | 5–8 | θ₁..θ₄ | wheel angles (encoder truth) |
  | 9–12 | ω₁..ω₄ | wheel speeds (encoder truth) |
  | 13–16 | γ₁..γ₄ | roller spin rates |
  | 17–18 | Xo, Yo | world position (evaluation only) |
  | 19–22 | zx₁..zx₄ | LuGre translational bristle x |
  | 23–26 | zy₁..zy₄ | LuGre translational bristle y |
  | 27–30 | zs₁..zs₄ | LuGre spin bristle |
  | (31–34) | i₁..i₄ | motor currents — **full-electrical variant only** |

- **Physics source:** copy the roller-contact velocities (`Vpi_x`, `Vpi_y`, `wzi`), `lugre_dyn_rates` calls, and `dv`/`dwi`/`dgi` expressions **verbatim** from `run_one.jl:675–742`; replace the internal `Mi_sat = smooth_sat.(ctrl(...))` with `Mi_sat = motor_torque(p.bus.v_cmd, ω, motor)` (§4.4b) — i.e. the applied wheel torque now comes from the youBot motor model driven by the ZOH voltage. World-position and bristle rows carry over unchanged. Note the wheel viscous term `p1·wi` in `dwi` (line 722) is the *load-side* friction; motor shaft friction (`τ_f`≈0.01 N·m/paper) can be folded into the motor torque if desired.
- **Depends on:** `PlatformParams`, `LuGreParams`, `MotorParams`, `sawtooth_approx`, `ControllerBus` (read-only).

### 4.2 `PlantODEParams`
- **Type:** immutable `struct` (tuple-like param pack)
- **Responsibility:** hold everything the RHS needs that is *not* state: `params::PlatformParams`, `chi`, `p1`, `p2`, `coupling::Symbol`, `lugre::LuGreParams`, `motor::MotorParams`, `bus::ControllerBus`.
- **Note:** `bus` is a **mutable** struct shared with the callbacks; the RHS only reads `bus.tau_wheel`.

### 4.3 `PlatformParams`, `LuGreParams`
- **Type:** reuse **unchanged** from `run_one.jl` (struct + `PlatformParams(base; mu_friction)` constructor, `LuGreParams` kwdef). `M_aug`/`M_aug_inv` are still needed by MPC and the estimator model.

### 4.4 `SensorModel` + `simulate_measurement`
- **Type:** `struct` (config) + `function`
- **Responsibility:** map true plant state → realistic **IMU + encoder** measurement vector at the sensor rate. This is the *only* information any controller/estimator may use.
- **Measurement vector `y`:**
  - Encoders: θ₁..θ₄ (quantized to counts/rev), ω₁..ω₄ (either direct or finite-difference of θ).
  - IMU: body proper accelerations a_x = V̇x − ψ̇·Vy, a_y = V̇y + ψ̇·Vx (needs `du` at sample time or a re-evaluation of the RHS accel block), and yaw rate g_z = ψ̇.
- **Corruptions:** additive Gaussian noise, constant + random-walk bias (gyro bias drift, accel bias), encoder quantization; all seed-controlled for reproducibility.
- **Key params:** `enc_cpr::Int`, `σ_ω`, `σ_acc`, `σ_gyro`, `gyro_bias_rw`, `acc_bias`, `seed`.
- **Depends on:** `PlatformParams` (wheel map), plant `u` and instantaneous `du`.
- **Critical rule:** true pose (Xo, Yo, ψ) and true Vx,Vy are **forbidden** as controller/estimator inputs — they are logged for evaluation only.

### 4.4b `MotorParams` + `motor_torque` (youBot DC-motor + gearbox actuator)
- **Type:** `Base.@kwdef struct` (params, §3 table) + `function`
- **Responsibility:** map commanded motor voltage `V_i` and wheel speed `ω_i` → applied wheel torque, with saturation. This is the actuator between the mixer's voltage command and the plant's roller/friction dynamics.
- **Model (quasi-static default):** motor speed `ω_m = G·ω_i`; current `i = clamp((V_i − Kb·ω_m)/Ra, ±i_max)`; motor torque `τ_m = Kt·i − τ_f·sign(ω_m)`; wheel torque `τ_wheel = G·η·τ_m` (η gear efficiency). Voltage pre-saturated to ±V_max by the mixer.
- **Model (full-electrical, optional):** as above but `i` is a plant state (`[31:34]`) with `L_a·i̇ = V − Ra·i − Kb·ω_m`; `motor_torque` then just reads `i` and returns `G·η·(Kt·i − τ_f·sign(ω_m))`.
- **Key params:** `Kt`, `G`, `Kb`, `Ra`, `La`, `i_max`, `V_max`, `eta`, `tau_f`, `cpr`, `dynamic_electrical::Bool`.
- **Depends on:** nothing (pure map); consumed by `plant_rhs!` and by the MPC/mixer for the voltage↔torque conversion.
- **Rationale:** gives the paper's voltage/current constraints a physical basis and lets MPC optimize in the true actuation variable; also makes the encoder resolution (`cpr`, on the motor shaft) concrete for the sensor model.

### 4.5 `Estimator` (shared) — `KalmanEstimator` (default) and optional `SMOEstimator`
- **Type:** mutable `struct` per kind + `estimator_update!` `function`
- **Responsibility:** fuse encoder-derived body velocity + IMU (yaw rate, accel) into an estimated body state, and **dead-reckon** pose (drifts). Output is the state every controller consumes.
- **Estimated state x̂:** `[V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]` (subset selectable).
- **`KalmanEstimator` (default):** discrete EKF/KF. Process model = simplified body dynamics `M_aug⁻¹·F` (or constant-velocity) with IMU accel as input; measurement = body velocity pseudo-measurement (invert the O-config wheel map, `profiles`-style eq. analog of paper eq. 3) + gyro yaw rate. Pose obtained by integrating V̂ through R(ψ̂) — unobservable, so covariance on X̂o,Ŷo grows without bound (expected, documented).
- **`SMOEstimator` (optional, "paper SMO if necessary"):** sliding-mode observer adapted to the **velocity** measurement model (not the paper's pose measurement). Smooth switching `s/(|s|+δ)`; gains L, K as tunables. Selected via `config.estimator == :smo`. **Additionally outputs a lumped disturbance estimate `d̂ = [d̂_x, d̂_y, d̂_ψ]`** (equivalent-injection / integral term of the SMO) — this is the paper-faithful reason to pick the SMO over the plain KF, and it is the DOB-replacement referenced in §4.6a (the super-twisting DOB was dropped, but the SMO can supply the same disturbance feedforward when `config.use_dhat=true`).
- **`d̂` feedforward:** written to `bus.d_hat`; consumed by the controllers' equivalent-control block (§4.6a), added as `ΔW = −M_aug·d̂` in task space — structurally the same compensation the removed DOB applied, now sourced from the SMO instead of a separate observer. Zeroed when `config.estimator≠:smo` or `use_dhat=false`.
- **Key params:** `Qn`, `Rn` (KF); `L`, `K`, `δ` (SMO); `rate_hz`; `use_dhat::Bool`.
- **Depends on:** `SensorModel` output, `PlatformParams` (wheel map, `M_aug`).

### 4.6 Control laws (each returns a **task-space wrench** `W = [Wx, Wy, Wψ]` in body frame)
Uniform interface so the mixer can blend them. Each reads only x̂ (estimated) and the active reference.

#### 4.6a `ASMCController` + `asmc_wrench!`  (carried over)
- **Responsibility:** adaptive sliding-mode control, position **and** velocity variants (mirror `asmc_torques` / `asmc_torques_vel`). Keep the smoothed λ(e), tanh boundary layer, and the **adaptive-gain update law with cubic pushback `−c·(K/Kmax)³` and linear leakage `−σ·(K−K₀)·exp(...)`** (from `run_one.jl:521–523`). Gains K_x,K_y,K_ψ now live in the controller struct and are integrated by forward-Euler at the ASMC rate.
- **Change vs original:** the *super-twisting DOB* is removed, but the same compensation slot remains: when `config.use_dhat=true` (SMO estimator active), add `ΔW = −M_aug·bus.d_hat` to the equivalent-control wrench — the disturbance feedforward is now sourced from the **SMO's d̂** (§4.5), not a dedicated observer. With `use_dhat=false` the wrench is the inverse-dynamics part only. Output is a task-space wrench (0.25 O-config wheel mapping happens in the mixer, not here). The same `d̂` term may optionally be added to the MPC/PID wrenches for a fair comparison.
- **Params:** reuse `ASMCParams` (minus the DOB-coupled `γ` reductions — set `use_dob=false` defaults).

#### 4.6b `MPCController` + `mpc_wrench!`
- **Responsibility:** finite-horizon QP tracking on a **linear discrete model** of the body (state `[X̂o,Ŷo,ψ̂,V̂x,V̂y,ψ̂̇]`, or velocity subset in velocity mode), decision variable = task-space wrench sequence (or wheel-torque sequence), horizon Np≈10.
- **Cost:** ‖x−x_ref‖²_Q + ‖U‖²_R + ‖ΔU‖²_S + terminal (paper eqs. 18–23), with `U` the **motor-voltage** sequence.
- **Decision variable:** per-wheel motor voltage `U=[V₁..V₄]` over the horizon (paper's true control), OR task-space wrench if a simpler MPC is wanted — pick via `mpc.decision::Symbol`. With the voltage form, the paper's constraints apply directly.
- **Constraints:** `−V_max ≤ Vᵢ ≤ V_max` (≈±24 V, paper eq. 16), voltage rate limit `|ΔVᵢ|≤ΔV_max` (paper eq. 17), derived current limit `|iᵢ|≤i_max`, and wheel-torque box via the motor map. Optional body-velocity box (paper eq. 15).
- **Model:** discretize `ẋ = A x + B U` where B folds the motor map (`τ_wheel = f(V,ω)`) and the O-config allocation and `M_aug⁻¹`; A from the linearized friction (viscous p1/p2 + back-EMF `Kb·G²/Ra` damping) about the operating point. Relinearize each MPC tick or use a fixed nominal.
- **Solver:** OSQP warm-started; if infeasible/te-out, fall back to previous command and flag.
- **Output:** if the mixer expects wrenches, return the equivalent task-space wrench; if voltage-native, write `bus.W_mpc` as the first-step voltage command (mixer handles the blend in voltage space).
- **Params:** `Np`, `Q`, `R`, `S`, `Qf`, `rate_hz`, `V_max`, `ΔV_max`, `i_max`, `decision`.

#### 4.6c `PIDController` + `pid_wrench!`  (Kalman-PID)
- **Responsibility:** velocity-error PID on the **estimated** body velocity (its "Kalman" part is the shared `KalmanEstimator`), with anti-windup saturation on the integral (paper eq. 32).
- **Params:** `Kp`, `Ki`, `Kd`, `I_max`, `rate_hz`.

### 4.7 `FuzzySupervisor` + `fuzzy_update!`
- **Type:** mutable `struct` (rule base + membership params) + `function`
- **Responsibility:** from tracking-error features computed on the **estimated** state — e_p (position or velocity error magnitude, mode-dependent), e_v, ė_p — output **normalized blend weights** over the *enabled* controllers `w = (w_ASMC, w_MPC, w_PID)` plus optional per-controller gain scales (α). Generalizes the paper's single β (MPC↔PID) to the 3-way arbitration; when only MPC+PID are enabled it reduces exactly to the paper's β(k) (eqs. 34–35, Algorithm 1 lines 15–18).
- **Type-1 triangular MFs** (paper §3.4): ep∈[0,0.5], ev∈[0,0.3], ėp∈[−0.2,0.2], {SMALL,MEDIUM,LARGE}; singleton outputs.
- **Selection interaction:** if `config.fuzzy=false`, weights come from `config.fixed_weights` (e.g., ASMC-only, or paper's fixed β).
- **Rate:** 50 Hz.
- **Depends on:** estimated state, active reference, `HybridConfig`.

### 4.8 `mix_and_allocate!`
- **Type:** `function`
- **Responsibility:** blend enabled controllers' wrenches by fuzzy weights → task-space wrench W; map W to 4 wheel torques via the **O-config allocation** (`run_one.jl:504–512` / `627–635`: `lever=R/(l+h)`, the 0.25 mix); **invert the youBot motor map** (§4.4b) to per-wheel voltage `V_i` at the current ω_i; saturate to ±`V_max` with `|iᵢ|≤i_max` and a per-tick `ΔV` rate limit; write `bus.v_cmd`.
- **Blending space:** keep the fuzzy blend in **wrench space** so ASMC/MPC/PID combine uniformly; do the single voltage conversion here. (Voltage-native MPC, §4.6b, is only for MPC-solo runs — then the mixer passes its voltage through with saturation.)
- **Depends on:** per-controller wrench slots on the bus, fuzzy weights, `PlatformParams`, `MotorParams`.

### 4.9 `ControllerBus`
- **Type:** `mutable struct` (shared state between callbacks and RHS)
- **Fields:** `v_cmd::SVector{4}` (ZOH motor-voltage input to plant), latest `x̂`, `d_hat::SVector{3}` (SMO disturbance estimate, zero if unused), latest per-controller wrenches (`W_asmc`, `W_mpc`, `W_pid`), fuzzy `weights`, ASMC gains `K`, PID integral accumulator, MPC warm-start & last solution, measurement cache, last applied wheel torque (diagnostic), diagnostic scratch.
- **Responsibility:** the single mutable rendezvous; RHS reads `v_cmd` (and computes wheel torque via the motor map), callbacks write everything.

### 4.10 `HybridConfig`
- **Type:** `Base.@kwdef struct`
- **Fields (selection + rates):** `tracking::Symbol` (`:pose`|`:velocity`), `estimator::Symbol` (`:kalman`|`:smo`|`:none`), `use_dhat::Bool` (SMO disturbance feedforward; requires `estimator==:smo`), `use_asmc::Bool`, `use_mpc::Bool`, `use_pid::Bool`, `fuzzy::Bool`, `fixed_weights::NTuple{3,Float64}`, rates `f_est`, `f_mpc`, `f_pid`, `f_fuzzy`, `f_mix`, sensor `seed`, solver knobs.
- **Responsibility:** single source of truth for *which architectures are active* and at what rate; drives which callbacks the scheduler installs.

### 4.11 `build_callbacks` / `run_hybrid`
- **Type:** `function`s
- **Responsibility:** `build_callbacks` assembles a `CallbackSet` of `PeriodicCallback`s (sensor+estimator, ASMC, MPC, PID, fuzzy, mixer, progress) **conditioned on `HybridConfig`**. `run_hybrid` builds `u0`, `PlantODEParams` (with a fresh `ControllerBus`), the `ODEProblem`, and calls `solve` with the callback set + stiff solver + `tstops = active_ref().tstops`, then post-processes into a log.
- **Depends on:** everything above.

### 4.12 `log_run` / `save_run`
- **Type:** `function`s
- **Responsibility:** assemble the per-timestep DataFrame (true state, estimated state, errors, wrenches, weights, τ) and persist via `DataStore` (Arrow), reusing the existing filename contract so downstream tooling is unaffected.

---

## 5. File & Notebook-Cell Structure

Delivered as one notebook that `include`s the reused modules; optional `.jl` extraction later (mirrors the `run_one.jl` pattern).

```
code_insights/
├── Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1.ipynb   # main notebook (cells below)
├── profiles.jl        # REUSED unchanged (references, frame transforms)
├── datastore.jl       # REUSED unchanged (Arrow schema, I/O)
└── hybrid_ctrl/       # optional extracted modules (post-notebook)
    ├── plant.jl        # §4.1–4.3  torque-input plant + params
    ├── sensors.jl      # §4.4      IMU + encoder model
    ├── estimators.jl   # §4.5      Kalman + optional SMO
    ├── controllers.jl  # §4.6      ASMC / MPC / PID wrench laws
    ├── fuzzy.jl        # §4.7      supervisor
    ├── mixer.jl        # §4.8      blend + O-config allocation + saturation
    ├── bus.jl          # §4.9      ControllerBus
    ├── config.jl       # §4.10     HybridConfig
    └── scheduler.jl    # §4.11–12  callbacks, run_hybrid, logging
```

**Notebook cell order:** (1) imports + `include(profiles/datastore)`; (2) `HybridConfig` (the *only* parametrization cell — tag it `parameters` to match project convention); (3) plant struct + RHS; (4) sensor model; (5) estimators; (6) control laws; (7) fuzzy; (8) mixer + bus; (9) scheduler/`run_hybrid`; (10) a demo run + tracking/estimation plots; (11) selection-matrix comparison (loop over configs); (12) save.

---

## 6. Key Interfaces

Signatures + docstrings only; bodies are placeholders (`# …`), no logic.

```julia
"""
    plant_rhs!(du, u, p, t)

Voltage-input plant RHS (30-D quasi-static / 34-D full-electrical, §4.1). Reads
ZOH motor voltage from `p.bus.v_cmd`, forms wheel torque via `motor_torque`,
then integrates physics verbatim from run_one.jl:675–742. No controller logic.
  u, du :: Vector{Float64}(30|34)   layout §4.1
  p     :: PlantODEParams           (…, motor::MotorParams, bus)
  t     :: Float64
"""
function plant_rhs!(du, u, p, t)
    # …
end

"""
    motor_torque(v_cmd, ω, motor) -> SVector{4}

youBot DC-motor + gearbox map (§4.4b): per wheel, ω_m=G·ω; i=clamp((V−Kb·ω_m)/Ra,±i_max);
τ_wheel = G·η·(Kt·i − τ_f·sign(ω_m)). Full-electrical variant instead reads i from state.
"""
function motor_torque(v_cmd, ω, motor::MotorParams) end

"""
    simulate_measurement(u, du, sm, t) -> y::NamedTuple

IMU+encoder measurement from true state. `y = (θ, ω, a_x, a_y, g_z)` with noise,
bias-drift, quantization. NEVER exposes true pose/body velocity.
  a_x = V̇x − ψ̇·Vy ,  a_y = V̇y + ψ̇·Vx ,  g_z = ψ̇   (+ corruptions)
"""
function simulate_measurement(u, du, sm::SensorModel, t) end

"""
    estimator_update!(bus, y, est, params, dt)

Advance the estimator one tick; fuse y → bus.x̂ = [V̂x,V̂y,ψ̂̇,ψ̂,X̂o,Ŷo].
Pose block is dead-reckoned (unobservable ⇒ growing covariance, expected).
`est` is KalmanEstimator or SMOEstimator (dispatch on type).
"""
function estimator_update!(bus::ControllerBus, y, est, params::PlatformParams, dt) end

"""
    asmc_wrench!(bus, x̂, ref, params, asmc, dt; mode) -> SVector{3}

Adaptive SMC task-space wrench [Wx,Wy,Wψ]. Keeps cubic + linear-leakage gain
update (run_one.jl:521–523); DOB term removed. `mode ∈ (:pose,:velocity)`.
Integrates bus.K by forward-Euler at the ASMC rate.
"""
function asmc_wrench!(bus, x̂, ref, params::PlatformParams, asmc::ASMCParams, dt; mode) end

"""
    mpc_wrench!(bus, x̂, ref, params, mpc) -> SVector{3}

QP-MPC task-space wrench. Linear discrete body model (B via M_aug⁻¹), horizon Np,
torque-domain box + rate constraints, OSQP warm-started. Falls back to prev on
infeasible.
"""
function mpc_wrench!(bus, x̂, ref, params::PlatformParams, mpc::MPCController) end

"""
    pid_wrench!(bus, x̂, ref, pid, dt) -> SVector{3}

Kalman-PID velocity wrench on estimated body velocity, anti-windup on integral.
"""
function pid_wrench!(bus, x̂, ref, pid::PIDController, dt) end

"""
    fuzzy_update!(bus, x̂, ref, cfg)

Set bus.weights = (w_ASMC,w_MPC,w_PID) (normalized over ENABLED controllers) and
optional α scales, from (e_p,e_v,ė_p) on the estimated state. Reduces to paper
β(k) when only MPC,PID enabled. If cfg.fuzzy=false, use cfg.fixed_weights.
"""
function fuzzy_update!(bus, x̂, ref, cfg::HybridConfig) end

"""
    mix_and_allocate!(bus, params, motor, cfg)

Blend enabled wrenches by bus.weights → W; O-config map to 4 wheel torques
(lever=R/(l+h), 0.25 mix; run_one.jl:504–512); invert motor map to voltages at
current ω; saturate to ±V_max with |i|≤i_max + ΔV rate limit; write bus.v_cmd.
"""
function mix_and_allocate!(bus, params::PlatformParams, motor::MotorParams, cfg::HybridConfig) end

"""
    run_hybrid(cfg, params, refname; chi, friction_case, lugre) -> (sol, log)

Build u0 + bus + ODEProblem, install config-gated PeriodicCallbacks, solve with
stiff solver + ref.tstops, post-process to a DataFrame log.
"""
function run_hybrid(cfg::HybridConfig, params::PlatformParams, refname::Symbol;
                    chi, friction_case, lugre) end
```

---

## 7. Data Flow (one master control window)

Continuous plant integration is interrupted by config-gated `PeriodicCallback`s. In rate order per window:

1. **Sensor + Estimator tick (f_est, ~1 kHz):** read `integrator.u` (+ RHS accel block for IMU) → `simulate_measurement` → `y`. `estimator_update!(bus, y, est, …)` writes `bus.x̂`. Pose in `x̂` drifts (only encoder-odometry + gyro integrate it).
2. **MPC tick (100 Hz):** if `use_mpc`, relinearize model at `bus.x̂`, build/warm-start QP over Np, solve → `bus.W_mpc`.
3. **PID tick (100 Hz):** if `use_pid`, velocity error on `bus.x̂` → `bus.W_pid` (anti-windup).
4. **ASMC tick (≈1 kHz):** if `use_asmc`, compute sliding surfaces on `bus.x̂`, update `bus.K` (cubic+leakage), → `bus.W_asmc`.
5. **Fuzzy tick (50 Hz):** if `fuzzy`, features from `bus.x̂` vs ref → `bus.weights`; else fixed weights.
6. **Mixer tick (f_mix ≥ max control rate):** `mix_and_allocate!` blends enabled `W_*` → task wrench → O-config wheel torques → **invert youBot motor map to voltages** → saturate (±V_max, i_max, ΔV) → `bus.v_cmd`.
7. **Plant RHS (continuous):** integrates physical state; each RHS eval forms wheel torque from `bus.v_cmd` (ZOH) via `motor_torque(v_cmd, ω, motor)` — so back-EMF makes the applied torque speed-dependent within the hold — until the next callback.

**Reference handling:** `is_velref()` / `current_ref()` vs `current_posref()` select the target set exactly as in `run_one.jl`; `mode` in ASMC/MPC/PID follows `cfg.tracking`.

**Evaluation vs control separation (enforce):** controllers/estimator consume **only** `bus.x̂`; true `u` (pose, Vx,Vy) is read solely by `log_run` for error metrics. This is the whole point of the IMU+encoder-only constraint.

**Loss/optimization chains:** MPC = OSQP QP (tracking + effort + Δeffort + terminal); ASMC = Lyapunov sliding surface + adaptive-gain ODE; PID = filtered velocity error; fuzzy = Mamdani/singleton defuzzification to weights. No gradients/backprop (no ML here).

---

## 8. Implementation Sequence

Topological build order (each unblocked before the next):

1. **`ControllerBus` + `HybridConfig`** — the shared contracts everything references.
2. **`MotorParams` + `motor_torque`** — youBot actuator map; unit-test voltage→torque and its inverse round-trip at a few ω.
3. **`plant_rhs!` + `PlantODEParams`** (reuse `PlatformParams`/`LuGreParams`) — regression gate: drive `bus.v_cmd` with voltages that invert to the *original* ASMC wheel torque and confirm it reproduces `run_one.jl` trajectories to tolerance.
4. **`SensorModel` + `simulate_measurement`** — check noiseless measurements invert back to true velocity via the wheel map; encoder quantization uses `cpr`·G.
5. **`KalmanEstimator`** (then optional `SMOEstimator`) — confirm velocity estimate tracks truth; document pose drift.
6. **`ASMCController`** (carried over, DOB removed) — first closed-loop controller; should recover near-original tracking on a reused profile.
7. **`PIDController`** — simplest peer controller; validates the wrench→mixer→motor→plant path with blending.
8. **`MPCController`** — QP + voltage/current constraints; validate feasibility and warm-start stability at 100 Hz.
9. **`FuzzySupervisor`** — weights; verify MPC+PID-only reduces to paper β.
10. **`mix_and_allocate!`** — finalize allocation + motor inversion + saturation (partly stubbed in steps 6–8).
11. **`build_callbacks` / `run_hybrid` / `log_run` / `save_run`** — assemble, run the selection matrix, persist.

---

## 9. Numerical / Solver / Real-time Considerations

- **Stiffness:** keep the LuGre bristle absolute tolerances tight (`bristle≈1e-10`, `bristle_rot≈1e-7`) as in `run_one.jl:879–892`, and a stiff solver with `dtmax ≤ 1e-3`. Rebuild `ABSTOL` for the new layout: 30 entries quasi-static (3+1+4+4+4+2+12), or 34 with motor currents (append `current≈1e-6`). The full-electrical variant adds fast electrical modes (`Ra/La`) → **stiffer**; if `La` is tiny keep quasi-static to avoid a needless stiffness penalty.
- **Motor map / back-EMF:** `motor_torque` is re-evaluated every RHS call, so back-EMF `−Kb·G·ω` damping and voltage saturation act continuously within each ZOH voltage hold — do **not** precompute torque once per control tick. Watch the sign convention of `τ_f·sign(ω_m)` near ω=0 (use a `tanh` smoothing to keep the RHS C∞ for the stiff solver, consistent with the existing smoothed laws).
- **Motor inversion consistency:** the mixer inverts the *same* `motor_torque` model the plant uses, at the *measured/estimated* ω — a mismatch between the inversion ω and the plant ω (sensor noise) shows up as a torque error; that is realistic, but keep the model identical so the only gap is the ω estimate.
- **ZOH consistency:** `dtmax` must be ≤ the fastest control period so the ZOH torque never lags more than one control tick; set `f_mix` ≥ `max(f_mpc,f_pid,f_asmc)`.
- **Callback ordering:** within a shared tick, ensure estimator → controllers → fuzzy → mixer ordering (use distinct `PeriodicCallback` phases or a single composite callback at the mix rate that calls sub-updates in order) to avoid using stale `x̂`/weights.
- **IMU accel term:** `a_x,a_y` need `V̇x,V̇y`. Either evaluate the RHS accel block at the sample point or expose it from the bus after the last RHS eval — do **not** finite-difference velocity (noisy). Document the choice.
- **Pose unobservability:** X̂o,Ŷo,ψ̂ have no absolute correction ⇒ KF covariance grows; this is correct, not a bug. In `:pose` tracking, expect steady drift — report drift rate as a metric and make the demo short enough that it stays bounded, or add a clearly-labeled optional pseudo-pose fix behind a config flag for A/B only.
- **QP robustness:** OSQP may return inaccurate/te-out; always have a feasible fallback (previous wrench) and log solver status; scale the QP (units) so it conditions well.
- **Saturation coherence:** saturate once, in the mixer (`smooth_sat`, `Max_torque=10`), so MPC's torque constraints and the plant input agree; MPC should predict with the same saturation model to avoid windup.
- **Reproducibility:** thread the RNG seed through `SensorModel` and set `Random.seed!` per run (as `run_one.jl:45`); log the seed.
- **State I/O for downstream:** keep the Arrow columns the existing loaders expect (per-wheel forces/slips, bristle states, torques). New columns (x̂, weights, per-controller wrenches) are additive; verify `datastore.jl` schema extension doesn't break the filename/label contract.

---

## 10. Success Criteria

- [ ] **Regression:** feeding the original ASMC torque as `bus.tau_wheel` reproduces `run_one.jl` trajectories to solver tolerance (validates the plant refactor).
- [ ] **Selection matrix runs:** every combination in {ASMC, MPC, PID} × {fuzzy on/off} × {pose, velocity} × {kalman, smo} executes without error and logs weights consistent with the enabled set.
- [ ] **Paper-β reduction:** MPC+PID-only with fuzzy on matches the paper's β(k) blend behavior (β∈[0.1,1.0], rises with tracking error).
- [ ] **Estimator:** velocity NRMSE bounded (report %); pose drift rate quantified and monotone (no false convergence).
- [ ] **Actuator validity:** |Vᵢ| ≤ V_max and |iᵢ| ≤ i_max at all times (youBot limits); wheel torque stays within the motor-map envelope; no QP infeasibility left unhandled.
- [ ] **Motor round-trip:** `motor_torque` ∘ inverse = identity to tolerance across the ω range (mixer↔plant consistency).
- [ ] **Closed loop stable** on at least circular + figure-eight (or your `Profiles` equivalents) for the full horizon under each enabled controller.
- [ ] **Reproducible:** identical seed ⇒ identical log.

---

## 11. Out of Scope

- Detailed BLDC commutation / three-phase electrical modeling (the youBot motor is modeled as an equivalent DC motor; full-electrical variant is single-current per wheel, not phase-resolved).
- Super-twisting / linear DOB (removed by decision).
- Absolute-pose sensing (mocap/UWB/GPS) — except an explicitly-flagged A/B-only pseudo-fix.
- Any PyTorch / PINN training — this notebook only *generates* runs; the existing PINN/observer packages consume its Arrow output separately.
- Hardware/real-time deployment, `torch.compile`, GPU.
- Multi-robot coordination and obstacle avoidance (paper's future work).
```
