# =============================================================================
# hybrid_ctrl_v2/sensors_v2.jl — SensorModV2
# (instructions/sensors-suite-consolidation-and-physical-noise.md)
# =============================================================================
# `hybrid_ctrl/sensors.jl` and `hybrid_ctrl/estimators.jl` are NEVER edited —
# this is a NEW, additive module (leaf: no `include`s), mirroring the pattern
# already established by `controllers_v2.jl` for the controller-tuning brief.
# `PoseFixModel` here is a fresh definition (not the one imported from
# EstimatorMod), consolidated alongside the other three modality models per
# the brief's intent, with independent per-modality RNG streams.
#
# GRADE CLAIM (brief §9): industrial-grade MEMS white-noise levels with
# UNCALIBRATED bias terms, representative of an uncompensated consumer
# installation — the white-noise and bias figures are deliberately different
# grades (see Appendix A of the brief), not an inconsistency.
#
# Datasheet revisions consulted (re-verify before publication):
#   ADIS16470 Rev. E (Analog Devices, 2021); PMW3901 datasheet Rev 1.0
#   (PixArt, 2015); ADNS-3080 datasheet (Avago/PixArt, 2005).
# =============================================================================
module SensorModV2

using StaticArrays
using Random
using LinearAlgebra

export ImuModel, EncoderModel, FlowModel, PoseFixModel, SensorSuite,
       simulate_imu, simulate_encoders, simulate_flow, sample_pose_fix,
       build_suite, R_wheel, R_gyro, R_flow, R_posefix

# =============================================================================
# ImuModel — accelerometer + gyroscope
# =============================================================================
"""
    ImuModel

Reference part: **Analog Devices ADIS16470**. White-noise terms are at the
part's industrial-grade spec; bias terms are UNCALIBRATED consumer-grade
(deliberate — see module header).
"""
Base.@kwdef struct ImuModel
    sigma_acc::Float64    = 0.05      # m/s^2 @1kHz = 228 ug/sqrt(Hz); typical MEMS 60-300 ug/sqrt(Hz) -- industrial/consumer
    sigma_gyro::Float64   = 0.003     # rad/s @1kHz = 0.0077 deg/s/sqrt(Hz); ADIS16470 spec 0.008 deg/s/sqrt(Hz) -- industrial (near-exact match)
    acc_bias::Float64     = 0.02      # m/s^2 = 2.04 mg; consumer accel bias stability 1-10 mg (ADIS16470 in-run: 13 ug) -- consumer, uncalibrated
    bias_gyro::Float64    = 0.003     # rad/s = 0.172 deg/s = 619 deg/hr; uncalibrated consumer turn-on bias -- consumer, uncalibrated
    gyro_bias_rw::Float64 = 1e-4      # rad/s/sqrt(s); rate random walk -> 92 deg/hr at T=20s; ADIS16470 in-run stability 8 deg/hr, consumer >=50 deg/hr -- consumer
    sf_gyro::Float64      = 0.0       # [-] gyro scale-factor error (0 => :default grade; set for :realistic)
    rate_hz::Float64      = 1000.0
    seed::Int             = 42
    rng::MersenneTwister  = MersenneTwister(seed)
    gyro_bias::MVector{1,Float64} = MVector(0.0)   # internal random-walk accumulator
end

"""
    simulate_imu(m::ImuModel, u, du, t) -> NTuple{3,Float64}

Accelerometer + gyro measurement. Body proper acceleration from true RHS
(a_x = Vdot_x - psidot*Vy, a_y = Vdot_y + psidot*Vx), corrupted by white noise,
a constant bias, the gyro random walk, and the psidot-proportional scale-
factor term.

STEP-0 BUG FIX (brief §9): the gyro-bias random-walk increment is
`gyro_bias_rw * sqrt(dt) * randn()` — a proper discrete Wiener-process
increment (accumulated std recovers `gyro_bias_rw*sqrt(T)`). v1's
`hybrid_ctrl/sensors.jl:91` instead scaled by `sqrt(t+0.001)` (the CLOSED-FORM
growth law written into the per-step increment), which is ~100x too large and
grows linearly in T rather than as sqrt(T) — see the brief's Appendix A sanity
check (92 deg/hr corrected vs 9220 deg/hr for the v1 bug at T=20s).
"""
function simulate_imu(m::ImuModel, u::AbstractVector, du::AbstractVector, t::Real)
    Vx, Vy, psidot = u[1], u[2], u[3]
    Vxdot, Vydot   = du[1], du[2]

    a_x_true = Vxdot - psidot * Vy
    a_y_true = Vydot + psidot * Vx

    dt = 1.0 / m.rate_hz
    m.gyro_bias[1] += m.gyro_bias_rw * sqrt(dt) * randn(m.rng)

    a_x = a_x_true + m.acc_bias + m.sigma_acc * randn(m.rng)
    a_y = a_y_true + m.acc_bias + m.sigma_acc * randn(m.rng)
    if iszero(m.sf_gyro)
        g_z = psidot + m.gyro_bias[1] + m.sigma_gyro * randn(m.rng)
    else
        g_z = psidot * (1.0 + m.sf_gyro * randn(m.rng)) + m.gyro_bias[1] + m.sigma_gyro * randn(m.rng)
    end
    return (a_x, a_y, g_z)
end

"""
    R_gyro(imu::ImuModel, psidot::Real) -> Float64

Gyro measurement variance at the current operating point. HETEROSCEDASTIC:
the scale-factor term is redrawn every sample, so it behaves as
psidot-proportional white noise rather than a fixed bias. Must be recomputed
per tick, not cached (`sf_gyro` depends on the current `psidot`).
"""
R_gyro(imu::ImuModel, psidot::Real) = imu.sigma_gyro^2 + (imu.sf_gyro * psidot)^2

# =============================================================================
# EncoderModel — wheel angle + wheel speed
# =============================================================================
"""
    EncoderModel

Wheel encoders on the motor shaft: effective resolution = cpr*gear.
"""
Base.@kwdef struct EncoderModel
    enc_cpr::Int     = 4000        # counts/rev on motor shaft -- typical incremental optical encoder
    gear::Float64    = 9405.0/364.0
    sigma_omega::Float64 = 0.01    # rad/s wheel-speed white noise -- consumer/industrial quadrature-derived velocity noise
    sf_omega::Float64    = 0.0     # [-] wheel-speed scale-factor error (0 => :default grade; set for :realistic)
    bias_omega::SVector{4,Float64} = @SVector zeros(4)   # rad/s per-wheel constant bias (drawn once, NOT noise -- see NoiseSpec note)
    rate_hz::Float64 = 1000.0
    seed::Int        = 42
    rng::MersenneTwister = MersenneTwister(seed)
end

"""
    simulate_encoders(m::EncoderModel, u) -> NamedTuple

Wheel angles (quantized to motor-shaft counts) and wheel speeds (white noise +
speed-proportional scale factor + per-wheel constant bias).

NOTE `bias_omega` is a BIAS, not noise: it maps through the wheel Jacobian to
a constant body-velocity offset — exactly what the estimator's slip states
see and absorb. It must NOT be folded into `R_wheel` (see `NoiseSpec`).
"""
function simulate_encoders(m::EncoderModel, u::AbstractVector)
    counts_per_rad = m.enc_cpr * m.gear / (2π)
    theta_true = SVector(u[5], u[6], u[7], u[8])
    omega_true = SVector(u[9], u[10], u[11], u[12])
    theta_quant = floor.(theta_true .* counts_per_rad .+ 0.5) ./ counts_per_rad

    if iszero(m.sf_omega)
        omega_noisy = omega_true .+ m.bias_omega .+
                      m.sigma_omega .* @SVector [randn(m.rng), randn(m.rng), randn(m.rng), randn(m.rng)]
    else
        sf = @SVector [1.0 + m.sf_omega * randn(m.rng), 1.0 + m.sf_omega * randn(m.rng),
                       1.0 + m.sf_omega * randn(m.rng), 1.0 + m.sf_omega * randn(m.rng)]
        white = m.sigma_omega .* @SVector [randn(m.rng), randn(m.rng), randn(m.rng), randn(m.rng)]
        omega_noisy = sf .* omega_true .+ m.bias_omega .+ white
    end
    return (theta=theta_quant, omega=omega_noisy)
end

"""
    R_wheel(enc::EncoderModel, params) -> SMatrix{2,2,Float64,4}

CLOSED-FORM RESULT (brief §6): the wheel-derived body-velocity pseudo-
measurement `z = Hw \\ omega` propagates encoder noise through the O-config
wheel Jacobian. Hw's four columns are mutually orthogonal:

    Hw'Hw = diag(4/R^2, 4/R^2, 4(l+h)^2/R^2)

so the least-squares noise covariance sigma_omega^2*(Hw'Hw)^-1 is EXACTLY
diagonal with EQUAL Vx/Vy entries:

    R_wheel[1,1] = R_wheel[2,2] = sigma_omega^2 * R^2 / 4

This replaces any tuned value that makes the two differ (the frozen v1
config's 2400x asymmetry contradicts this kinematic identity). Adds the
speed-proportional scale-factor contribution at the operating point when
`sf_omega` is nonzero (evaluated at the CURRENT wheel speed magnitude, via
the same (Hw'Hw)^-1 propagation).
"""
function R_wheel(enc::EncoderModel, params; omega_mag::Real=0.0)
    R = params.R
    var = enc.sigma_omega^2 + (enc.sf_omega * omega_mag)^2
    r = var * R^2 / 4
    return SMatrix{2,2,Float64,4}(r, 0.0, 0.0, r)
end

# =============================================================================
# FlowModel — downward optical flow (NEW)
# =============================================================================
"""
    FlowModel

Downward-facing ground-relative velocity sensor, immune to wheel slip.
Reference parts: **PixArt PMW3901** (rate_hz, v_max_track) and
**PixArt/Avago ADNS-3080** (quantum). This is the measurement that makes the
slip states observable ALGEBRAICALLY (the wheels see v+s, flow sees v — see
`hybrid_ctrl_v2/estimators_v2.jl`'s `apply_flow!`).
"""
Base.@kwdef struct FlowModel
    r_offset::SVector{2,Float64} = SVector(0.0, 0.0)   # m, body-frame lever arm (sensor offset from COM)
    sigma_flow0::Float64 = 0.005   # m/s white-noise floor (near-zero-speed regime; not separately spec'd -- small vs the proportional term below)
    sigma_flow::Float64  = 0.02    # [-] proportional noise ~2% of speed (std dev 17.6-25.7mm over 1m trials at 35cm/s, PMC4481980)
    sf_flow::Float64     = 0.147   # [-] height-induced scale error, 14.7%/mm height sensitivity (conventional optics; 0.1%/mm afocal) -- drawn ONCE per run, NOT noise (see NoiseSpec note)
    quantum::Float64     = 31.75e-6 # m/count @800cpi (ADNS-3080 datasheet)
    v_max_track::Float64 = 1.02    # m/s; 40 ips (ADNS-3080); PMW3901: 7.4 rad/s x height
    dropout_prob::Float64 = 0.02   # per-sample dropout (low-texture/reflective floor; NZPRG mecanum field trial)
    rate_hz::Float64      = 100.0
    seed::Int             = 42
    rng::MersenneTwister  = MersenneTwister(seed)
    sf_draw::MVector{1,Float64} = MVector(1.0)   # the ONE height-scale draw for this run (set by build_suite)
end

"""
    simulate_flow(m::FlowModel, u, t) -> Union{Nothing,SVector{2,Float64}}

Optical-flow ground-relative velocity at the sensor's body-frame offset:

    v_sensor = [Vx - psidot*r_y, Vy + psidot*r_x]

Error chain: height-induced scale factor (drawn ONCE per run, `m.sf_draw`,
constant thereafter) -> white noise -> quantization to `m.quantum`.

Returns `nothing` (NO measurement this tick — caller must skip the update,
not substitute a zero) when either:
  (a) `|v_sensor|` exceeds `v_max_track` (sensor loses lock), or
  (b) the per-sample dropout draw fires.
"""
function simulate_flow(m::FlowModel, u::AbstractVector, t::Real)
    Vx, Vy, psidot = u[1], u[2], u[3]
    vx_s = Vx - psidot * m.r_offset[2]
    vy_s = Vy + psidot * m.r_offset[1]
    v_true = SVector(vx_s, vy_s)
    norm(v_true) > m.v_max_track && return nothing
    rand(m.rng) < m.dropout_prob && return nothing

    v_scaled = m.sf_draw[1] .* v_true
    v_mag = norm(v_true)
    noise = (m.sigma_flow0 + m.sigma_flow * v_mag) .*
            SVector(randn(m.rng), randn(m.rng))
    v_noisy = v_scaled .+ noise
    v_quant = m.quantum .* round.(v_noisy ./ m.quantum)
    return v_quant
end

"""
    R_flow(m::FlowModel, v_mag::Real) -> SMatrix{2,2,Float64,4}

White-noise floor + speed-proportional term + quantization variance
(quantum^2/12, uniform quantization). The height-induced SCALE error
(`sf_flow`/`sf_draw`) is deliberately EXCLUDED here — it is a persistent
multiplicative bias, not noise (see NoiseSpec module note); folding it into R
would let the filter average away an error that does not average away.
"""
function R_flow(m::FlowModel, v_mag::Real)
    var = (m.sigma_flow0 + m.sigma_flow * v_mag)^2 + m.quantum^2 / 12
    return SMatrix{2,2,Float64,4}(var, 0.0, 0.0, var)
end

# =============================================================================
# PoseFixModel — intermittent absolute pose (consolidated here per the brief;
# fields/behaviour identical to the version currently in EstimatorMod, which
# is left untouched — this is a fresh, independent definition, not a move)
# =============================================================================
"""
    PoseFixModel

Noisy absolute-pose "sensor". Two tiers:
  :transit — intermittent, coarse (~5-10 Hz, sigma_pos~0.05 m, sigma_psi~2 deg)
  :docking — reliable, precise (100 Hz, sigma_pos~0.01 m, sigma_psi~0.5 deg)
"""
Base.@kwdef mutable struct PoseFixModel
    use_pose_fix::Bool   = false
    tier::Symbol         = :transit
    fix_rate_hz::Float64 = 10.0
    sigma_pos::Float64   = 0.05
    sigma_psi::Float64   = deg2rad(2.0)
    bias_px::Float64     = 0.0
    bias_py::Float64     = 0.0
    bias_psi::Float64    = 0.0
    dropout_frac::Float64 = 0.2
    outlier_frac::Float64 = 0.01
    gate_thresh::Float64  = 1.0
    seed::Int             = 42
    rng::MersenneTwister  = MersenneTwister(seed)
    last_fix_t::Float64   = -Inf
    dropout_start::Float64    = Inf
    dropout_duration::Float64 = 0.0
end

function PoseFixModel(tier::Symbol; seed::Int=42)
    rng = MersenneTwister(seed)
    if tier == :transit
        return PoseFixModel(use_pose_fix=true, tier=tier, fix_rate_hz=10.0,
            sigma_pos=0.05, sigma_psi=deg2rad(2.0), dropout_frac=0.2,
            outlier_frac=0.01, gate_thresh=1.0, seed=seed, rng=rng, last_fix_t=-Inf)
    elseif tier == :docking
        return PoseFixModel(use_pose_fix=true, tier=tier, fix_rate_hz=100.0,
            sigma_pos=0.01, sigma_psi=deg2rad(0.5), dropout_frac=0.0,
            outlier_frac=0.005, gate_thresh=0.5, seed=seed, rng=rng, last_fix_t=-Inf)
    elseif tier == :realistic
        return PoseFixModel(use_pose_fix=true, tier=tier, fix_rate_hz=100.0,
            sigma_pos=0.02, sigma_psi=deg2rad(0.6),
            bias_px=0.01*randn(rng), bias_py=0.01*randn(rng), bias_psi=deg2rad(0.3)*randn(rng),
            dropout_frac=0.0, outlier_frac=0.005, gate_thresh=0.5, seed=seed, rng=rng, last_fix_t=-Inf)
    else
        error("PoseFixModel: unknown tier '$tier'; use :transit, :docking, or :realistic")
    end
end

_wrap_angle(psi) = atan(sin(psi), cos(psi))

"""
    sample_pose_fix(m::PoseFixModel, u, t) -> Union{Nothing,SVector{3,Float64}}

Absolute pose measurement (x, y, psi). `nothing` on dropout or off-grid ticks.
"""
function sample_pose_fix(m::PoseFixModel, u::AbstractVector, t::Real)
    m.use_pose_fix || return nothing
    dt_fix = 1.0 / m.fix_rate_hz
    k = round(Int, t / dt_fix)
    t_fix = k * dt_fix
    t_fix <= m.last_fix_t && return nothing
    abs(t - t_fix) > 0.5 * dt_fix && return nothing
    m.last_fix_t = t_fix

    if t >= m.dropout_start && t <= m.dropout_start + m.dropout_duration
        return nothing
    end
    rand(m.rng) < m.dropout_frac && return nothing

    x_true, y_true, psi_true = u[17], u[18], u[4]
    nx = m.sigma_pos * randn(m.rng) + m.bias_px
    ny = m.sigma_pos * randn(m.rng) + m.bias_py
    npsi = m.sigma_psi * randn(m.rng) + m.bias_psi
    if rand(m.rng) < m.outlier_frac
        nx += 10.0 * nx; ny += 10.0 * ny; npsi += 10.0 * npsi
    end
    return SVector(x_true + nx, y_true + ny, _wrap_angle(psi_true + npsi))
end

"""
    R_posefix(m::PoseFixModel) -> SMatrix{4,4,Float64,16}

Pose-fix covariance in the 4-D (x, y, cospsi, sinpsi) form the ESKF uses,
sigma_psi^2 mapped onto the unit-circle tangent (small-angle approximation).
"""
function R_posefix(m::PoseFixModel)
    return SMatrix{4,4,Float64,16}(Diagonal(SVector(m.sigma_pos^2, m.sigma_pos^2, m.sigma_psi^2, m.sigma_psi^2)))
end

# =============================================================================
# SensorSuite — container + independent-RNG construction
# =============================================================================
"""
    SensorSuite

Container for the four modality models. Constructed ONLY via `build_suite` —
that is what guarantees the independent-RNG-stream contract (module header).
"""
Base.@kwdef struct SensorSuite
    imu::ImuModel
    enc::EncoderModel
    flow::Union{FlowModel,Nothing}
    fix::Union{PoseFixModel,Nothing}
    seed::Int
end

"""
    build_suite(kind::Symbol; seed::Int, flow::Bool=true, fix_tier::Symbol=:docking) -> SensorSuite

Construct all four modality models with INDEPENDENT RNG streams.

DETERMINISM CONTRACT (brief §6/§9 — the single most important requirement in
this file): each modality's RNG is seeded from `hash((seed, :modality))`, NOT
a shared stream, so the noise sequence CANNOT depend on the order
`scheduler_v2.jl` fires its PeriodicCallbacks — unlike v1's single shared
`SensorModel.rng`, where draw order (and therefore reproducibility) depends on
callback firing order.

`kind`: `:default` (legacy modest noise) or `:realistic` (MEMS/encoder-grade,
matching the values in `hybrid_ctrl/sensors.jl`'s `SensorModel(:realistic)`).
"""
function build_suite(kind::Symbol=:default; seed::Int, flow::Bool=true, fix_tier::Symbol=:docking)
    seed_imu  = hash((seed, :imu))
    seed_enc  = hash((seed, :encoder))
    seed_flow = hash((seed, :flow))
    seed_fix  = hash((seed, :posefix))

    if kind == :default
        imu = ImuModel(seed=Int(seed_imu % typemax(Int32)), rng=MersenneTwister(seed_imu))
        enc = EncoderModel(seed=Int(seed_enc % typemax(Int32)), rng=MersenneTwister(seed_enc))
    elseif kind == :realistic
        rng_enc = MersenneTwister(seed_enc)
        R_eff = 0.05
        sigma_omega = 0.01 / (R_eff * sqrt(1.09))
        sigma_bomega = 0.005 * 2.0 / R_eff
        bomega = sigma_bomega .* @SVector [randn(rng_enc), randn(rng_enc), randn(rng_enc), randn(rng_enc)]
        enc = EncoderModel(sigma_omega=sigma_omega, sf_omega=0.02, bias_omega=bomega,
                           seed=Int(seed_enc % typemax(Int32)), rng=rng_enc)
        rng_imu = MersenneTwister(seed_imu)
        imu = ImuModel(sigma_gyro=0.003, sf_gyro=0.005, bias_gyro=0.003*randn(rng_imu),
                       seed=Int(seed_imu % typemax(Int32)), rng=rng_imu)
    else
        error("SensorModV2.build_suite: unknown kind '$kind'; use :default or :realistic")
    end

    flow_model = flow ? FlowModel(seed=Int(seed_flow % typemax(Int32)), rng=MersenneTwister(seed_flow)) : nothing
    fix_model  = fix_tier === nothing ? nothing : PoseFixModel(fix_tier; seed=Int(seed_fix % typemax(Int32)))

    return SensorSuite(imu=imu, enc=enc, flow=flow_model, fix=fix_model, seed=seed)
end

end # module
