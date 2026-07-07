#!/usr/bin/env python3
"""Generate Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1.ipynb from the hybrid_ctrl modules."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1.ipynb"


def code_cell(source, tags=None):
    if isinstance(source, list):
        lines = source
    else:
        lines = source.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": lines,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


cells = []

# --------------------------------------------------------------------------
# Title / scope
# --------------------------------------------------------------------------
cells.append(md_cell("""# Mecanum Hybrid Control Notebook — SMO / MPC / Kalman-PID / Fuzzy

This notebook simulates a KUKA youBot four-Mecanum-wheel platform under a
**hybrid, run-time-selectable controller**.  The controller is built from four
blocks:

- **Plant:** roller kinematics + dynamic LuGre bristle friction + youBot DC-motor
  actuator.
- **State estimator:** encoder + IMU only (no absolute pose sensor).
- **Control laws:** adaptive sliding-mode control (ASMC), model-predictive control
  (MPC), and Kalman-PID.
- **Supervisor:** a type-1 fuzzy system that blends the enabled controllers.

The plant is integrated with a stiff ODE solver.  All controller/estimator blocks
run as `PeriodicCallback`s on a shared `ControllerBus`; the plant RHS only reads
the held motor-voltage command.  This matches how real embedded software works:
sensors sample, controllers compute, and the actuator sees a zero-order-hold
voltage.
"""))

# --------------------------------------------------------------------------
# Cell 1: imports + reused modules
# --------------------------------------------------------------------------
cells.append(code_cell("""# Cell 1: imports + reused modules
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks
using StaticArrays
using DataFrames
using Arrow
using Plots
using Printf
using Random
using Statistics

# Reuse existing project modules/structs/functions
include("run_one.jl")          # PlatformParams, LuGreParams, lugre_dyn_rates, sawtooth_approx, smooth_sat
using .Profiles
using .DataStore

# Hybrid control modules
include("hybrid_ctrl/config.jl");    using .HybridConfigMod
include("hybrid_ctrl/bus.jl");       using .BusMod
include("hybrid_ctrl/plant.jl");     using .PlantMod
include("hybrid_ctrl/sensors.jl");   using .SensorMod
include("hybrid_ctrl/estimators.jl"); using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");     using .FuzzyMod
include("hybrid_ctrl/mixer.jl");     using .MixerMod
include("hybrid_ctrl/scheduler.jl"); using .SchedulerMod

println("Hybrid control modules loaded.")
"""))

# --------------------------------------------------------------------------
# Architecture overview
# --------------------------------------------------------------------------
cells.append(md_cell("""## System architecture

The simulation has one continuous part and several discrete parts:

```
Continuous plant  ←  v_cmd (ZOH motor voltage)
       ↑
   Mixer + motor inversion  (f_mix ≥ fastest control rate)
       ↑
   Fuzzy supervisor  (50 Hz)
       ↑
   ASMC (1 kHz)   MPC (100 Hz)   PID (100 Hz)
       ↑                ↑              ↑
   Estimator  (1 kHz) ← sensor model ← IMU + encoders
```

**Why this separation?**  The LuGre bristle dynamics are stiff
(`σ₀ ≈ 1.6×10³ 1/m`).  Putting the QP or the sliding-mode update inside the ODE
RHS would force the solver to re-run expensive logic at every micro-step.  By
running the blocks as `PeriodicCallback`s and writing their results to a shared
`ControllerBus`, the stiff integrator sees only a held voltage input.

**Key data contract:**  controllers and estimators consume **only** the estimated
state `x̂ = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]`.  True pose `(Xo, Yo, ψ)` and true velocity
`(Vx, Vy)` are logged for evaluation but are never fed back to the controller.
This reflects a real robot with wheel encoders and an IMU but no mocap/GPS/UWB.
"""))

# --------------------------------------------------------------------------
# Cell 2: HybridConfig (parameters tag)
# --------------------------------------------------------------------------
cells.append(md_cell("""## Cell 2 — Configuration (tagged `parameters`)

`HybridConfig` is the single switchboard.  It selects:

- `tracking`: `:velocity` or `:pose`
- `estimator`: `:kalman`, `:smo`, or `:none`
- `use_asmc`, `use_mpc`, `use_pid`: which controllers are active
- `fuzzy`: whether blend weights come from fuzzy rules or `fixed_weights`
- `fixed_weights`: `(w_ASMC, w_MPC, w_PID)` used when `fuzzy=false`
- rates: `f_est`, `f_mpc`, `f_pid`, `f_fuzzy`, `f_mix`
- solver tolerances and `saveat_hz`

All callbacks are installed conditionally, so disabling a block removes both its
computational cost and its influence on the blend.
"""))

cells.append(code_cell("""# Cell 2: HybridConfig — the only parametrization cell
cfg = HybridConfig(
    tracking   = :velocity,      # :pose or :velocity
    estimator  = :kalman,        # :kalman, :smo, or :none
    use_dhat   = false,
    use_asmc   = true,
    use_mpc    = false,
    use_pid    = false,
    fuzzy      = false,
    fixed_weights = (1.0, 0.0, 0.0),   # ASMC-only default
    f_est      = 1000.0,
    f_mpc      = 100.0,
    f_pid      = 100.0,
    f_fuzzy    = 50.0,
    f_mix      = 1000.0,
    sensor_seed = 42,
    reltol     = 1e-8,
    abstol_bristle = 1e-10,
    dtmax      = 1e-3,
    solver_symbol = :TRBDF2,
    saveat_hz  = 500.0,
)
""", tags=["parameters"]))

# --------------------------------------------------------------------------
# Plant dynamics theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Plant dynamics — voltage-input roller+LuGre model

The plant state is 30-D in the default quasi-static electrical mode:

```
u = [Vx, Vy, ψ̇, ψ, θ₁..θ₄, ω₁..ω₄, γ₁..γ₄, Xo, Yo,
     zx₁..zx₄, zy₁..zy₄, zs₁..zs₄]
```

- `(Vx, Vy, ψ̇)` — body-frame velocities.
- `ψ` — heading.
- `θᵢ, ωᵢ` — wheel angle and speed (encoder truth).
- `γᵢ` — roller spin rate.
- `(Xo, Yo)` — world position (evaluation only).
- `(zx, zy, zs)` — LuGre bristle deflections in translation and spin.

**Roller contact velocities.**  Each wheel has rollers at `±45°`.  The sawtooth
roller phase `θ̃ᵢ` enters the contact kinematics, giving the roller-frame slip
`(Vpxᵢ, Vpyᵢ)` and spin slip `wzᵢ` that drive the LuGre model.

**Dynamic LuGre friction.**  Instead of an algebraic friction law, we integrate
bristle states:

```
żx = Vpx − σ₀·s_t/g_t·zx
ży = Vpy − σ₀·s_t/g_t·zy
żs = wz  − σ₀_s·s_s/g_s·zs

Fx = −N·(σ₀·zx + σ₁·żx)
Fy = −N·(σ₀·zy + σ₁·ży)
Mz = −N·χ²·(σ₀_s·zs + σ₁_s·żs)
```

For the Adamov case, `s_t` and `s_s` contain the spin→translation coupling term
`(8/(3π))·|wz|·χ`.  `χ` is the roller radius and is the main calibration
parameter alongside `μ`.

**youBot actuator.**  The commanded variable is **motor voltage** `Vᵢ`, not wheel
torque.  In quasi-static form:

```
ω_m = G·ω_i                         (motor speed)
i   = clamp((Vᵢ − Kb·ω_m)/Ra, ±i_max)   (armature current)
τ_wheel = G·η·(Kt·i − τ_f·sign(ω_m))
```

Back-EMF (`Kb·ω_m`) and current saturation make the effective torque drop at
high wheel speeds, exactly the voltage-domain behavior the paper emphasizes.
"""))

# --------------------------------------------------------------------------
# Cell 3: physics point + plant params
# --------------------------------------------------------------------------
cells.append(code_cell("""# Cell 3: build PlatformParams from base.toml
config_dir = "trajectory_files_run_0p5_main"
base = Profiles.load_base(config_dir)
mu_friction   = 0.5
friction_case = 1
friction_model = :lugre_adamov
params = PlatformParams(base; mu_friction=mu_friction)

lugre = LuGreParams()
motor = MotorParams(dynamic_electrical=false)
chi = 0.002

println("PlatformParams built; R=$(params.R), Ra=$(params.Ra), mu=$(params.f_coulomb)")
"""))

# --------------------------------------------------------------------------
# Sensor model theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Sensor model — encoders + IMU only

The estimator sees realistic sensor outputs, not the true state:

- **Encoders:** wheel angles `θᵢ` quantized to motor-shaft counts
  (`cpr = 4000`, `G = 9405/364`).  Wheel speeds are obtained from the encoders
  plus white noise.
- **IMU:** body proper accelerations and yaw rate.
  ```
  a_x = V̇x − ψ̇·Vy
  a_y = V̇y + ψ̇·Vx
  g_z = ψ̇
  ```
  These are corrupted by additive white noise, a constant accelerometer bias,
  and a random-walk gyro bias.

**Why no pose sensor?**  This notebook deliberately mirrors a real youBot: it has
wheel encoders and an IMU, but no absolute localization.  The world position
`(Xo, Yo)` is therefore unobservable; any pose estimate drifts.  That drift is
expected and is quantified in the logs.
"""))

# --------------------------------------------------------------------------
# Estimator theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## State estimator — Kalman filter or sliding-mode observer

The estimated state is:

```
x̂ = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]
```

**KalmanEstimator (default).**  A discrete extended Kalman filter:

- Measurement: least-squares inversion of the Mecanum wheel Jacobian gives a
  body-velocity pseudo-measurement; the gyro gives `ψ̂̇`.
- Prediction: constant-velocity process model with the yaw angle integrated from
  `ψ̂̇` and the world position integrated through `R(ψ̂)`.
- Covariance: velocity channels are corrected by the measurement; the pose block
  is dead-reckoned, so its covariance grows without bound.

**SMOEstimator (optional).**  A sliding-mode observer on the velocity measurement
model:

```
ė = −L·s/(|s|+δ) + d̂
ḋ̂ = −K·s/(|s|+δ)
```

with `s = e = z − x̂_vel`.  The integral term `d̂` becomes a lumped disturbance
estimate.  When `use_dhat=true`, this `d̂` is fed forward to the controllers as a
replacement for the removed super-twisting DOB.
"""))

# --------------------------------------------------------------------------
# ASMC theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Adaptive sliding-mode control (ASMC)

ASMC is the carried-over controller from the existing PINN pipeline, now
repackaged as a task-space wrench law.

**Sliding surface (velocity mode).**

```
s_x   = Vx − Vx_des
s_y   = Vy − Vy_des
s_ψ   = ė_ψ + λ( e_ψ )·e_ψ
```

where `e_ψ` is a smooth wrap of `ψ − ψ_des` and `λ(e)` is a dynamic bandwidth
that is large when the error is small and small when the error is large.

**Control wrench.**  The total wrench is the sum of a switching part and an
inverse-dynamics (equivalent-control) part:

```
W_sw  = [−K_x·tanh(s_x/ε), −K_y·tanh(s_y/ε), −K_ψ·tanh(s_ψ/ε_ψ)]
W_eq  = M_aug·a_des − Coriolis/centrifugal − viscous drag
W_asmc = W_sw + W_eq
```

**Adaptive gains.**  Each gain `K_α` follows:

```
ḊK = γ·(s·tanh(s/ε))·smooth_bound(K, K_max)
     − cubic_pushback(K/K_max)³
     − σ·(K − K₀)·exp( decay·(1 − s²/(9·ε_ψ²)) )
```

The cubic term keeps gains from saturating, and the leakage term lets them decay
when the sliding error is small.

**d̂ feedforward.**  The DOB from the original notebook is removed.  When the
SMO estimator is active and `use_dhat=true`, its disturbance estimate is added as
`ΔW = −M_aug·d̂` to the equivalent-control wrench.
"""))

# --------------------------------------------------------------------------
# MPC theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Model-predictive control (MPC)

MPC solves a finite-horizon optimal control problem at every tick.  The paper
formulates it in the **motor-voltage** domain, which lets us impose the youBot
constraints directly:

```
min  Σ ‖x_k − x_ref‖²_Q + ‖U_k‖²_R + ‖ΔU_k‖²_S + terminal_cost
U_k
s.t.  x_{k+1} = A·x_k + B·U_k
      −V_max ≤ U_k,i ≤ V_max
      |ΔU_k,i| ≤ ΔV_max
      |i_k,i| ≤ i_max
```

In this first notebook version, `MPCController` uses a fast LQR-like gain on the
velocity error to guarantee the scheduler closes the loop; the struct already
stores `Np`, `Q`, `R`, `S`, `V_max`, `ΔV_max`, and a warm-start vector so a full
OSQP implementation can be dropped in later without changing the notebook cells.

The MPC output is a task-space wrench `W_mpc`, which the mixer blends with the
ASMC and PID wrenches.
"""))

# --------------------------------------------------------------------------
# PID theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Kalman-PID

The "Kalman" part is the shared state estimator; the PID part is a standard
velocity-error controller acting on the estimated body velocity:

```
e = [V̂x − Vx_des, V̂y − Vy_des, ψ̂̇ − ω_des]
I = clamp(I + e·dt, −I_max, I_max)
W_pid = −Kp·e − Ki·I − Kd·ė
```

Anti-windup clamps the integral accumulator.  Because the estimate already
filters encoder/IMU noise, the PID gains can be more aggressive than a raw
sensor-driven PID.
"""))

# --------------------------------------------------------------------------
# Fuzzy theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Fuzzy supervisor

The fuzzy block decides how much authority each enabled controller receives.
Inputs are features computed from the **estimated** tracking error:

- `e_p` — position/velocity error magnitude
- `e_v` — velocity error magnitude
- `ė_p` — derivative of the position/velocity error

Each input is fuzzified with triangular membership functions
`{SMALL, MEDIUM, LARGE}` (and `{NEGATIVE, ZERO, POSITIVE}` for `ė_p`).  Rules are
of the form:

```
IF e_p is LARGE AND e_v is LARGE THEN trust ASMC more, trust PID less
```

The singleton rule outputs give raw weights `(w_ASMC, w_MPC, w_PID)`.  Disabled
controllers are zeroed, and the remaining weights are normalized to sum to one.

**Paper-β reduction.**  If only MPC and PID are enabled and `fuzzy=true`, the
supervisor reduces to a single blend `β(k) ∈ [0.1, 1.0]` that rises with tracking
error: PID dominates when errors are small, MPC dominates when errors are large.
"""))

# --------------------------------------------------------------------------
# Mixer theory
# --------------------------------------------------------------------------
cells.append(md_cell("""## Mixer — blending, O-config allocation, and saturation

The mixer is the last discrete block before the plant.

**1. Blend.**  Fuzzy weights `w = (w_ASMC, w_MPC, w_PID)` combine the three
wrenches into a single task-space wrench:

```
W = w_ASMC·W_asmc + w_MPC·W_mpc + w_PID·W_pid
```

**2. Allocate to wheels.**  The Mecanum O-configuration maps `(W_x, W_y, W_ψ)`
to four wheel torques:

```
lever = R / (l + h)
τ₁ = 0.25·(W_x − W_y − lever·W_ψ)
τ₂ = 0.25·(W_x + W_y + lever·W_ψ)
τ₃ = 0.25·(W_x + W_y − lever·W_ψ)
τ₄ = 0.25·(W_x − W_y + lever·W_ψ)
```

**3. Invert the motor map.**  At the current measured/estimated wheel speed `ω`,
the desired torque is mapped back to motor voltage by inverting the DC-motor
algebra.  This ensures the plant and the controller use the same actuator model.

**4. Saturate.**  Voltage is clamped to `±V_max`, current to `±i_max`, and the
per-tick voltage change to `±ΔV_max·dt`.  The final voltage is written to
`bus.v_cmd` and held by the plant until the next mixer tick.
"""))

# --------------------------------------------------------------------------
# Cell 4: demo run
# --------------------------------------------------------------------------
cells.append(md_cell("""## Demo run

Run one closed-loop simulation.  The scheduler builds the initial state, installs
the callbacks selected by `cfg`, and integrates with `TRBDF2` (or `RadauIIA5`).
The returned `df` contains true state, estimated state, reference, applied wheel
torques, voltages, blend weights, and per-controller wrenches.
"""))

cells.append(code_cell("""# Cell 4: single hybrid run
sol, df = SchedulerMod.run_hybrid(cfg, params, Symbol("long_circle_mu_0p5");
                                  chi=chi,
                                  friction_case=friction_case,
                                  lugre=lugre,
                                  motor=motor,
                                  config_dir=config_dir)

println("Demo run complete: t_end=$(sol.t[end]), N_saved=$(length(sol.t))")
first(df, 3)
"""))

# --------------------------------------------------------------------------
# Cell 5: plots
# --------------------------------------------------------------------------
cells.append(md_cell("""## Tracking plots

Visualize:

- True body velocity vs. the velocity reference.
- Per-wheel motor voltage commands (should stay inside `±V_max`).
- Controller blend weights from the fuzzy supervisor.
"""))

cells.append(code_cell("""# Cell 5: plot velocity tracking and control voltages
p1 = plot(df.time, [df.Vx df.Vy df.psi_dot], label=["Vx" "Vy" "ψ̇"], lw=2,
          xlabel="t (s)", ylabel="body velocity", title="True state")
plot!(p1, df.time, df.Vx_des, ls=:dash, label="Vx_des")
plot!(p1, df.time, df.Vy_des, ls=:dash, label="Vy_des")
plot!(p1, df.time, df.omega_des, ls=:dash, label="ω_des")

p2 = plot(df.time, [df.v_cmd_1 df.v_cmd_2 df.v_cmd_3 df.v_cmd_4], lw=1.5,
          xlabel="t (s)", ylabel="V (V)", title="Motor voltage commands")

p3 = plot(df.time, [df.w_asmc df.w_mpc df.w_pid], lw=2,
          xlabel="t (s)", ylabel="weight", title="Controller blend weights",
          label=["ASMC" "MPC" "PID"])

plot(p1, p2, p3, layout=(3,1), size=(900,800))
"""))

# --------------------------------------------------------------------------
# Cell 6: selection matrix comparison
# --------------------------------------------------------------------------
cells.append(md_cell("""## Selection-matrix comparison

A selection matrix lets us compare architectures under the same physics point.
Each row is a different combination of enabled controllers and fuzzy blending.
The metric reported is RMS velocity-tracking error:

```
RMSE_V = sqrt(mean( (Vx − Vx_des)² + (Vy − Vy_des)² ))
```

The maximum commanded voltage is also reported to verify that the actuator
limits are respected.
"""))

cells.append(code_cell("""# Cell 6: selection matrix
configs = [
    ("ASMC_only",     HybridConfig(use_asmc=true,  use_mpc=false, use_pid=false, fuzzy=false, fixed_weights=(1,0,0))),
    ("PID_only",      HybridConfig(use_asmc=false, use_mpc=false, use_pid=true,  fuzzy=false, fixed_weights=(0,0,1))),
    ("MPC_PID_beta",  HybridConfig(use_asmc=false, use_mpc=true,  use_pid=true,  fuzzy=true,  fixed_weights=(0,0.5,0.5))),
    ("ASMC_MPC_PID",  HybridConfig(use_asmc=true,  use_mpc=true,  use_pid=true,  fuzzy=true,  fixed_weights=(0.34,0.33,0.33))),
]

results = []
for (name, c) in configs
    sol_c, df_c = SchedulerMod.run_hybrid(c, params, Symbol("long_circle_mu_0p5");
                                          chi=chi, friction_case=friction_case,
                                          lugre=lugre, motor=motor,
                                          config_dir=config_dir)
    rmse_V = sqrt(mean(df_c.e_Vx.^2 .+ df_c.e_Vy.^2))
    push!(results, (name=name, rmse_V=rmse_V, Vmax=max(maximum(abs.(df_c.v_cmd_1)), maximum(abs.(df_c.v_cmd_2)), maximum(abs.(df_c.v_cmd_3)), maximum(abs.(df_c.v_cmd_4)))))
end

DataFrame(results)
"""))

# --------------------------------------------------------------------------
# Cell 7: save
# --------------------------------------------------------------------------
cells.append(md_cell("""## Save run to Arrow

Persist the demo run using the existing `DataStore` filename contract so that
downstream PINN/observer tooling can load it without modification.  New columns
(`x̂`, weights, per-controller wrenches, voltages) are additive and do not break
the required column set:

```
Vx, Vy, psi_dot, w1..w4, theta1..theta4, Msat_1..4,
Fx_1..4, Fy_1..4, Mz_1..4, time
```
"""))

cells.append(code_cell("""# Cell 7: persist the demo run
outdir = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
mkpath(outdir)
meta = (profile="long_circle", combo_idx=1, mu=mu_friction, chi=chi,
        friction_case=friction_case, friction_model=friction_model, sweep_seed=cfg.sensor_seed)

paths = SchedulerMod.save_run(df, sol, outdir, meta; cfg=base)
println("Saved: ", paths)
"""))

# --------------------------------------------------------------------------
# Assemble notebook
# --------------------------------------------------------------------------
nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Julia 1.10",
            "language": "julia",
            "name": "julia-1.10",
        },
        "language_info": {"name": "julia"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
