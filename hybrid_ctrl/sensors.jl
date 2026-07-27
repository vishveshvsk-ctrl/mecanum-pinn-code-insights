# =============================================================================
# hybrid_ctrl/sensors.jl — IMU + wheel encoder measurement model
# =============================================================================
module SensorMod

using StaticArrays
using Random

export SensorModel, simulate_measurement, reset_sensor_bias!

"""
    SensorModel

Maps true plant state + acceleration to realistic IMU + encoder measurements.
Encoders are on the motor shaft: effective wheel resolution = cpr·G.
"""
Base.@kwdef struct SensorModel
    enc_cpr::Int        = 4000        # counts/rev on motor shaft
    gear::Float64       = 9405.0/364.0
    σ_ω::Float64        = 0.01        # wheel-speed noise [rad/s]
    σ_acc::Float64      = 0.05        # accel noise [m/s²]
    σ_gyro::Float64     = 0.005       # gyro noise [rad/s]
    gyro_bias_rw::Float64 = 1e-4      # gyro random-walk per sqrt(s)
    acc_bias::Float64   = 0.02        # constant accel bias [m/s²]
    # Optional realistic error model (scale-factor + constant bias).  These are
    # applied ON TOP of the base σ_* white noise above.
    sf_ω::Float64       = 0.0         # wheel-speed scale-factor error [-]
    bias_ω::SVector{4,Float64} = @SVector zeros(4)   # per-wheel bias [rad/s]
    sf_gyro::Float64    = 0.0         # gyro scale-factor error [-]
    bias_gyro::Float64  = 0.0         # constant gyro bias [rad/s]
    seed::Int           = 42
    rng::MersenneTwister = MersenneTwister(seed)
    gyro_bias::MVector{1,Float64} = MVector(0.0)
end

"""
    SensorModel(kind::Symbol; seed::Int=42)

Convenience constructors:
  :default  — existing modest noise levels.
  :realistic — MEMS/encoder/LiDAR-grade noise used for robustness ablations:
               wheel-speed noise ≈ 10 mm/s body-velocity equivalent + 2 % SF +
               per-wheel bias equivalent to ~5 mm/s body-velocity bias;
               gyro = 3 mrad/s + 0.5 % SF + 3 mrad/s constant bias.
"""
function SensorModel(kind::Symbol; seed::Int=42)
    rng = MersenneTwister(seed)
    if kind == :default
        return SensorModel(seed=seed, rng=rng)
    elseif kind == :realistic
        # Wheel-noise scaling: body-velocity std ≈ R * σ_ω * sqrt(trace/3).
        # For the youBot O-config (R≈0.05 m, l+h≈0.3 m) this gives ≈ 0.01 m/s.
        R_eff = 0.05
        σ_ω = 0.01 / (R_eff * sqrt(1.09))
        # Per-wheel bias std chosen so body-velocity bias std ≈ 5 mm/s.
        σ_bω = 0.005 * 2.0 / R_eff
        bω = σ_bω * @SVector [randn(rng), randn(rng), randn(rng), randn(rng)]
        return SensorModel(
            σ_ω=σ_ω, σ_gyro=0.003,
            sf_ω=0.02, bias_ω=bω,
            sf_gyro=0.005, bias_gyro=0.003 * randn(rng),
            seed=seed, rng=rng)
    else
        error("SensorModel: unknown kind '$kind'; use :default or :realistic")
    end
end

function reset_sensor_bias!(sm::SensorModel)
    sm.gyro_bias[1] = 0.0
    Random.seed!(sm.rng, sm.seed)
    return sm
end

"""
    simulate_measurement(u, du, sm, t)

Return `(θ, ω, a_x, a_y, g_z)` as NamedTuple.  Uses true `du` for body
acceleration (`a_x = V̇x − ψ̇·Vy`, `a_y = V̇y + ψ̇·Vx`) plus corruptions.
Pose and true body velocity are NOT exposed.
"""
function simulate_measurement(u::AbstractVector, du::AbstractVector,
                              sm::SensorModel, t::Real)
    Vx, Vy, psi_dot = u[1], u[2], u[3]
    Vxdot, Vydot    = du[1], du[2]

    # Body proper acceleration from true RHS acceleration
    a_x_true = Vxdot - psi_dot * Vy
    a_y_true = Vydot + psi_dot * Vx

    # Gyro bias random walk (Euler integrate)
    sm.gyro_bias[1] += sm.gyro_bias_rw * sqrt(t + 0.001) * randn(sm.rng)

    # Encoder quantization (motor-shaft counts -> wheel angle)
    counts_per_rad = sm.enc_cpr * sm.gear / (2π)
    θ_true  = SVector(u[5], u[6], u[7], u[8])
    ω_true  = SVector(u[9], u[10], u[11], u[12])
    θ_quant = floor.(θ_true .* counts_per_rad .+ 0.5) ./ counts_per_rad
    # Wheel-speed corruption: white noise + scale-factor + per-wheel bias.
    # Only draw random scale-factor samples when enabled so :default stays
    # bit-identical to the original noise sequence.
    if iszero(sm.sf_ω)
        ω_noisy = ω_true .+ sm.bias_ω .+
                  sm.σ_ω .* @SVector [randn(sm.rng), randn(sm.rng), randn(sm.rng), randn(sm.rng)]
    else
        ω_sf = @SVector [1.0 + sm.sf_ω * randn(sm.rng),
                         1.0 + sm.sf_ω * randn(sm.rng),
                         1.0 + sm.sf_ω * randn(sm.rng),
                         1.0 + sm.sf_ω * randn(sm.rng)]
        ω_white = sm.σ_ω .* @SVector [randn(sm.rng), randn(sm.rng), randn(sm.rng), randn(sm.rng)]
        ω_noisy = ω_sf .* ω_true .+ sm.bias_ω .+ ω_white
    end

    a_x = a_x_true + sm.acc_bias + sm.σ_acc * randn(sm.rng)
    a_y = a_y_true + sm.acc_bias + sm.σ_acc * randn(sm.rng)
    # Gyro corruption: white noise + optional scale-factor + optional constant bias
    if iszero(sm.sf_gyro) && iszero(sm.bias_gyro)
        g_z = psi_dot + sm.gyro_bias[1] + sm.σ_gyro * randn(sm.rng)
    else
        g_z = psi_dot * (1.0 + sm.sf_gyro * randn(sm.rng)) + sm.bias_gyro +
              sm.gyro_bias[1] + sm.σ_gyro * randn(sm.rng)
    end

    return (; θ = θ_quant, ω = ω_noisy, a_x, a_y, g_z)
end

end # module
