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
    seed::Int           = 42
    rng::MersenneTwister = MersenneTwister(seed)
    gyro_bias::MVector{1,Float64} = MVector(0.0)
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
    ω_noisy = ω_true .+ sm.σ_ω .* SVector(randn(sm.rng), randn(sm.rng), randn(sm.rng), randn(sm.rng))

    a_x = a_x_true + sm.acc_bias + sm.σ_acc * randn(sm.rng)
    a_y = a_y_true + sm.acc_bias + sm.σ_acc * randn(sm.rng)
    g_z = psi_dot + sm.gyro_bias[1] + sm.σ_gyro * randn(sm.rng)

    return (; θ = θ_quant, ω = ω_noisy, a_x, a_y, g_z)
end

end # module
