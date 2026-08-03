# =============================================================================
# hybrid_ctrl_v2/estimators_v2.jl — EstimatorModV2
# (instructions/sensors-suite-consolidation-and-physical-noise.md)
# =============================================================================
# `hybrid_ctrl/estimators.jl` is NEVER edited — ESKFEstimatorV2 is a NEW
# struct (not the original ESKFEstimator with fields removed), with its own
# `estimator_update!`/`apply_pose_fix!` methods added via multiple-dispatch
# extension of the ORIGINAL generic functions (same pattern as
# `controllers_v2.jl`'s `MPCControllerV2`/`mpc_wrench!`). The original
# ESKFEstimator, KalmanEstimator, SMOEstimator, IMMKalmanEstimator, and
# EstimatorMod.PoseFixModel are completely untouched and keep working exactly
# as before for every existing caller.
#
# Must be `include`d after `sensors_v2.jl` (needs `SensorModV2`) and after
# `tune_controller.jl`/`hybrid_ctrl/estimators.jl` (extends
# `Main.EstimatorMod.estimator_update!`/`apply_pose_fix!`, reuses
# `Main.EstimatorMod._wheel_jacobian`/`slip_detect`/`_wrap_angle`).
# =============================================================================
module EstimatorModV2

using StaticArrays
using LinearAlgebra
using Random

export ESKFEstimatorV2, gauss_markov_q, derive_process_noise, apply_flow!

# =============================================================================
# TimeConstantQ — derive nuisance-state Q entries + decay rates from physics
# =============================================================================
"""
    gauss_markov_q(tau, sigma, dt) -> (q_tick, decay_per_tick)

Per-tick process-noise increment and decay factor for a first-order
Gauss-Markov (Ornstein-Uhlenbeck) state with correlation time `tau` and
stationary std `sigma`, discretized over one tick of length `dt`
(`dt << tau` assumed, matching `tau ~ 0.1 s` vs `dt = 1 ms`):

    x_{k+1} = (1 - decay_per_tick)*x_k + w_k,   Var(w_k) = q_tick

    decay_per_tick = dt/tau                  (goes on A's diagonal, NEGATIVE
                                               self-term: A[i,i] = -decay)
    q_tick         = 2*sigma^2/tau * dt      (preserves the stationary
                                               variance sigma^2 in the small-
                                               dt/tau limit)

WHY BOTH (brief §6): a pure random walk has no stationary variance and
wanders unboundedly; physically, slip returns to zero when grip is restored.
Setting Q alone is not sufficient — the decay term must ALSO go into A, or
the "time constant" has no representation in the model.
"""
function gauss_markov_q(tau::Real, sigma::Real, dt::Real)
    decay_per_tick = dt / tau
    q_tick = 2 * sigma^2 / tau * dt
    return q_tick, decay_per_tick
end

"""
    derive_process_noise(spec::NamedTuple, dt::Real) -> NamedTuple

Assemble the three nuisance-state Q entries + decay rates from physical
parameters. The three states are NOT the same kind of process (brief §7.3):

  accel bias (bx,by) : CONSTANT in the sim (no walk)      -> q ~ 0, tau->inf, no decay
  gyro bias  (bg)     : PURE random walk (`gyro_bias_rw`)  -> q_tick = sigma_rw^2*dt, no decay
  slip       (sx,sy)  : MEAN-REVERTING (grip returns)      -> gauss_markov_q(tau_slip, sigma_slip, dt)

`spec.sigma_gyro_bias_rw` MUST be the SAME constant the sensor model's
`ImuModel.gyro_bias_rw` uses (a single shared source — brief §7.3: "not two
independently-set numbers that can drift apart").
"""
function derive_process_noise(spec::NamedTuple, dt::Real)
    q_slip, decay_slip = gauss_markov_q(spec.tau_slip, spec.sigma_slip, dt)
    q_gyro_bias = spec.sigma_gyro_bias_rw^2 * dt
    q_accel_bias = get(spec, :q_accel_bias_margin, 1e-10)   # robustness margin only; constant process
    return (q_slip=q_slip, decay_slip=decay_slip,
            q_gyro_bias=q_gyro_bias, q_accel_bias=q_accel_bias)
end

# =============================================================================
# ESKFEstimatorV2 — single-model ESKF, R/Q derived from sensor physics
# =============================================================================
"ESKF nominal/error state dimension (heading-vector formulation) -- same as v1."
const ESKF_DIM = 12

"""
    ESKFEstimatorV2

Same state content and heading-vector formulation as `EstimatorMod.ESKFEstimator`
(x = [Vx,Vy,psidot,c,s,X,Y,bx,by,sx,sy,bg]), but:
  - `R` for the wheel+gyro update is DERIVED per tick from `enc`/`imu` via
    `SensorModV2.R_wheel`/`R_gyro` (heteroscedastic on `imu`'s gyro; NEVER a
    free/searched field).
  - `Q` for bx,by,sx,sy,bg is DERIVED from `derive_process_noise` (physical
    time constants), NOT free fields. `sx,sy` are now MEAN-REVERTING (a decay
    term is added to `A`), not a pure random walk.
  - `slip_threshold` is PINNED from the physical wheel-noise floor (3-sigma
    of `R_wheel`'s diagonal), not tuned.
  - `nis_thresh` is PINNED at chi^2_3,0.99 = 11.34 (v1's default 9.21 is
    chi^2_2,0.99 — mismatched against the 3-D wheel+gyro measurement).
  - An optional `flow::FlowModel` channel is consumed via `apply_flow!`.

Retained TUNABLE fields (per-user design pass, supersedes the brief §7.5
single-scalar draft): `P0_scale` is split into SEVEN independent per-group
initial-variance scales (one per physically distinct quantity/unit — tying,
e.g., accelerometer-bias uncertainty to translational-velocity uncertainty
was judged physically wrong, since they're different units and different
epistemic states at t=0), and `pose_Qn` is split into TWO (heading vs.
position — different units, different unmodelled-error sources). See
`hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl` for the resulting 10-dim
space. `pose_slip_gain` is REMOVED entirely (not pinned, deleted): with flow
providing a slip-immune, independent fix on Vx/Vy, the velocity estimate
feeding the pose prediction is already accurate during slip, so inflating
pose process noise specifically on slip activity is now redundant — the
thing it used to compensate for is handled at the velocity-measurement
level. The remaining fields below (`slip_R_inflate`, `grip_slip_scale`,
`alpha_acc`, `alpha_yaw`, `r_boost`) keep the v1 slip-aware-Q architecture's
STRUCTURE; `slip_R_inflate` is tunable (wheel-channel-only — see
`estimator_update!` below, it never touches the gyro/flow/pose-fix `R`s),
the rest stay pinned at v1 defaults.
"""
Base.@kwdef mutable struct ESKFEstimatorV2
    imu::Any    # SensorModV2.ImuModel
    enc::Any    # SensorModV2.EncoderModel
    flow::Any = nothing   # Union{SensorModV2.FlowModel,Nothing}

    # Physical specification for derive_process_noise (brief §7.3).
    # MEASURED (not assumed) from a closed-loop trace: spin_creep combo 178,
    # mu=0.5, ASMC default gains, 25s/25000 ticks. Ground-truth slip
    # s = Hw\omega_true - v_true (signed, per-axis), autocorrelation of sx/sy:
    # ACF decays cleanly (0.98 -> 0.13) over the first ~8ms, then goes
    # NEGATIVE and oscillates (a resonance coupled to spin_creep's own
    # periodic excitation, not intrinsic slip relaxation) -- this oscillatory
    # tail biases a naive multi-lag log-linear fit low (tau=0.006s, R^2=0.70).
    # The lag-1 AR(1) estimator tau=-dt/log(rho_1) is far less sensitive to
    # that tail (uses only the first step) and gives tau_sx=0.061s,
    # tau_sy=0.064s (rho_1~0.984) -- squarely inside the brief's own a-priori
    # order-of-magnitude expectation (0.05-0.2s) -- so THAT estimate is used.
    # sigma_slip = std(s) = 0.0132(sx)/0.0139(sy) m/s on this trajectory;
    # single-trajectory point estimate, likely trajectory-dependent -- worth
    # broadening to more traces in a follow-up (brief §11 leaves re-tuning
    # out of scope for THIS iteration).
    tau_slip::Float64             = 0.06     # s -- measured (lag-1 AR(1), see above)
    sigma_slip::Float64           = 0.014    # m/s -- measured stationary slip std (same measurement)
    sigma_gyro_bias_rw::Float64   = 1e-4    # rad/s/sqrt(s) -- MUST match imu.gyro_bias_rw (shared source)

    # Retained tunable — split by physically distinct quantity/unit (struct
    # docstring above). Defaults reproduce the old single-scalar draft's
    # numbers per-group so behaviour is unchanged until the optimizer moves
    # them apart.
    P0_vel::Float64        = 1e-2   # Vx,Vy initial variance [(m/s)^2]
    P0_yaw::Float64        = 1e-2   # psidot initial variance [(rad/s)^2]
    P0_heading::Float64    = 1e-2   # c,s initial variance [unitless, unit-circle]
    P0_bias_acc::Float64   = 1e-2   # bx,by initial variance [(m/s^2)^2]
    P0_bias_gyro::Float64  = 1e-2   # bg initial variance [(rad/s)^2]
    P0_slip::Float64       = 1e-5   # sx,sy initial variance [(m/s)^2] -- was P0_scale*1e-3
    P0_pos::Float64        = 1.0    # X,Y initial variance [m^2] -- was HARDCODED to 1.0, now a real tunable
    pose_Qn_heading::Float64 = 1e-6 # c,s per-tick process noise
    pose_Qn_pos::Float64     = 1e-6 # X,Y per-tick process noise

    # Kept-structure, no-longer-searched (brief §7.5 "re-examine"; defaults
    # unchanged from v1 so behaviour is comparable, just not free dims)
    slip_R_inflate::Float64  = 10.0
    grip_slip_scale::Float64 = 1e-3
    alpha_acc::Float64 = 1.0
    alpha_yaw::Float64 = 0.5
    r_boost::Float64   = 10.0

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x::MVector{12,Float64} = MVector{12}(0.0,0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0)
    P::MMatrix{12,12,Float64,144} = MMatrix{12,12}(I)
    div_count::Int = 0
    initialized::Bool = false

    # PINNED (derived at init, not tuned)
    slip_threshold::Float64 = NaN     # set in init_eskf_v2! from R_wheel (physical definition)
    nis_thresh::Float64       = 11.34 # chi^2_3,0.99 -- 3-D wheel+gyro measurement
    nis_thresh_flow::Float64  = 13.82 # chi^2_2,0.999 -- 2-D flow measurement (same 99.9% convention as pose-fix gate)
end

_wrap_angle(psi) = atan(sin(psi), cos(psi))
_renorm_heading!(x) = (n = hypot(x[4], x[5]); x[4] /= n; x[5] /= n; x)

function init_eskf_v2!(est::ESKFEstimatorV2, params)
    est.wheel_H = Main.EstimatorMod._wheel_jacobian(params)
    est.x .= 0.0
    est.x[4] = 1.0
    P0 = zeros(ESKF_DIM, ESKF_DIM)
    P0[1,1] = est.P0_vel;      P0[2,2] = est.P0_vel
    P0[3,3] = est.P0_yaw
    P0[4,4] = est.P0_heading;  P0[5,5] = est.P0_heading
    P0[6,6] = est.P0_pos;      P0[7,7] = est.P0_pos
    P0[8,8] = est.P0_bias_acc; P0[9,9] = est.P0_bias_acc
    P0[10,10] = est.P0_slip;   P0[11,11] = est.P0_slip
    P0[12,12] = est.P0_bias_gyro
    est.P .= P0
    est.div_count = 0

    # PINNED slip_threshold: 3-sigma of the wheel-pseudo-velocity noise floor
    # (R_wheel's diagonal) -- "the physical slip definition" (brief §7.5): a
    # slip signal is distinguishable from encoder noise once it exceeds this.
    Rw = Main.SensorModV2.R_wheel(est.enc, params)
    est.slip_threshold = 3.0 * sqrt(Rw[1,1])

    dt = 1.0 / est.rate_hz
    @assert est.tau_slip > dt/2 "ESKFEstimatorV2: tau_slip=$(est.tau_slip) too small for dt=$dt -- " *
        "the discrete mean-reversion transition |1-dt/tau_slip| would exceed 1 (unstable)"

    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::ESKFEstimatorV2, params, dt)

Single-model error-state KF tick — same exact-nonlinear-kinematics/heading-
vector formulation as `ESKFEstimator`, but `R`/`Q` are DERIVED per tick from
`est.enc`/`est.imu`/`derive_process_noise` instead of free fields, and the
slip states (sx,sy) get a mean-reversion decay term in `A` (brief §7.3's
model change — Q alone is not sufficient for a Gauss-Markov process).
"""
function Main.EstimatorMod.estimator_update!(bus, y, est::ESKFEstimatorV2, params, dt)
    !est.initialized && init_eskf_v2!(est, params)

    ax, ay = y.a_x, y.a_y
    x = est.x

    # --- 1. Nominal propagation (exact nonlinear kinematics, no wrap) -------
    v1, v2, v3 = x[1], x[2], x[3]
    cpsi, spsi = x[4], x[5]
    bx, by = x[8], x[9]
    x[1] = v1 + dt * (ax + v3*v2 - bx)
    x[2] = v2 + dt * (ay - v3*v1 - by)
    x[4] = cpsi + dt * (-v3*spsi)
    x[5] = spsi + dt * ( v3*cpsi)
    x[6] += dt * (v1*cpsi - v2*spsi)
    x[7] += dt * (v1*spsi + v2*cpsi)
    # bx,by,bg constant in prediction; sx,sy mean-revert via A below (not held constant)

    # --- 2. Error-covariance propagation ------------------------------------
    z = Main.EstimatorMod._wheel_body_velocity(y, est.wheel_H)
    v_pred = SVector(x[1], x[2], x[3])
    slip_meas = Main.EstimatorMod.slip_detect(y, v_pred, params)
    gripping = slip_meas < est.slip_threshold

    w1, w2, w3 = x[1], x[2], x[3]
    cp, sp = x[4], x[5]

    spec = (tau_slip=est.tau_slip, sigma_slip=est.sigma_slip, sigma_gyro_bias_rw=est.sigma_gyro_bias_rw)
    pn = derive_process_noise(spec, dt)

    A = zeros(MMatrix{12,12,Float64,144})
    A[1,2] = w3;  A[1,3] = w2;  A[1,8] = -1.0
    A[2,1] = -w3; A[2,3] = -w1; A[2,9] = -1.0
    A[4,3] = -sp; A[4,5] = -w3
    A[5,3] =  cp; A[5,4] =  w3
    A[6,1] = cp;  A[6,2] = -sp; A[6,4] = w1; A[6,5] = -w2
    A[7,1] = sp;  A[7,2] =  cp; A[7,4] = w2; A[7,5] =  w1
    A[10,10] = -pn.decay_slip     # mean-reversion: sx -> 0 as grip returns
    A[11,11] = -pn.decay_slip     # (THE model change -- brief §7.3)
    F = SMatrix{12,12}(I) + dt * A

    # Velocity/yaw-rate process noise DERIVED from accelerometer/gyro white
    # noise (brief: "Qn_diag removed -- derived from accelerometer noise"):
    # one Euler step v_{k+1}=v_k+dt*a_meas propagates a_meas's variance as
    # (dt*sigma)^2; psidot uses the gyro's noise floor as its sensor analog
    # (psidot itself isn't integrated in prediction -- x[3] is corrected only
    # by the measurement -- so this represents unmodelled yaw-accel between
    # ticks, bounded by the gyro channel).
    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(w3)
    q_vx = (dt * est.imu.sigma_acc)^2 * q_scale
    q_vy = q_vx
    q_yr = (dt * est.imu.sigma_gyro)^2 * q_scale
    slip_q_scale = gripping ? est.grip_slip_scale : 1.0

    Q = zeros(MMatrix{12,12,Float64,144})
    Q[1,1] = q_vx; Q[2,2] = q_vy; Q[3,3] = q_yr
    Q[4,4] = est.pose_Qn_heading; Q[5,5] = est.pose_Qn_heading
    Q[6,6] = est.pose_Qn_pos;     Q[7,7] = est.pose_Qn_pos
    Q[8,8] = pn.q_accel_bias
    Q[9,9] = pn.q_accel_bias
    Q[10,10] = pn.q_slip * slip_q_scale
    Q[11,11] = pn.q_slip * slip_q_scale
    Q[12,12] = pn.q_gyro_bias
    est.P .= F * est.P * F' + Q

    # --- 3. Measurement update (error-state), R DERIVED per tick -----------
    H = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
    e = z - H * SVector{12}(x)

    Rw = Main.SensorModV2.R_wheel(est.enc, params; omega_mag=norm(SVector(y.ω[1],y.ω[2],y.ω[3],y.ω[4])))
    r_gyro = Main.SensorModV2.R_gyro(est.imu, w3)
    R = MMatrix{3,3,Float64,9}(Diagonal(SVector(Rw[1,1], Rw[2,2], r_gyro)))
    if !gripping
        R[1,1] *= est.slip_R_inflate
        R[2,2] *= est.slip_R_inflate
    end

    S = H * est.P * H' + R
    Sinv = inv(S)
    nu = dot(e, Sinv * e)
    if nu > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = H * est.P * H' + R
        Sinv = inv(S)
        nu = dot(e, Sinv * e)
    end

    K = MMatrix{12,3,Float64,36}(est.P * H' * Sinv)
    if !gripping
        K[8, :] .= 0.0
        K[9, :] .= 0.0
    end

    dx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (R * K')
    est.P .= 0.5 * (Pn + Pn')

    x .+= dx
    _renorm_heading!(x)

    nu > 1e4 ? (est.div_count += 1) : (est.div_count = 0)
    if est.div_count >= 10
        est.P .*= 10.0
        est.div_count = 0
    end

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return bus.xhat
end

"""
    apply_flow!(bus, est::ESKFEstimatorV2, m::FlowModel, z_flow, params) -> Bool

Optical-flow measurement update. The 2x12 Jacobian selects body velocity and
the lever-arm coupling to yaw rate:

    H_flow = [ 1  0  -r_y  0 ... 0 |  0  0 |  0
               0  1   r_x  0 ... 0 |  0  0 |  0 ]

CRITICAL STRUCTURAL POINT (brief §6): the slip columns (10,11) are ZERO. The
wheels measure Vx+sx; the flow measures Vx. That zero is what makes the
(velocity, slip) pair non-degenerate -- the entire reason for this sensor.

Gated by the covariance-aware NIS test at chi^2_2,0.999 = 13.82 (NOT a tuned
threshold). Returns true if the measurement was accepted.
"""
function apply_flow!(bus, est::ESKFEstimatorV2, m, z_flow::SVector{2}, params)
    !est.initialized && return false
    x = est.x
    r_x, r_y = m.r_offset[1], m.r_offset[2]
    H = @SMatrix [1.0 0.0 -r_y 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0  r_x 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    e = z_flow - H * SVector{12}(x)
    v_mag = norm(SVector(x[1], x[2]))
    Rf = Main.SensorModV2.R_flow(m, v_mag)
    S = H * est.P * H' + Rf
    Sinv = inv(S)
    nu = dot(e, Sinv * e)
    nu > est.nis_thresh_flow && return false

    K = est.P * H' * Sinv
    dx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (Rf * K')   # Joseph form
    est.P .= 0.5 * (Pn + Pn')                    # PSD symmetrization

    x .+= dx
    _renorm_heading!(x)
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

"""
    apply_pose_fix!(bus, est::ESKFEstimatorV2, fix::PoseFixModel, z_fix) -> Bool

Identical to `ESKFEstimator`'s pose-fix update (4-D (x,y,cospsi,sinpsi), NIS
gate at chi^2_4,0.999=16.27, Joseph form) -- unchanged behaviour, just
dispatching on the new type since `fix::SensorModV2.PoseFixModel` is a
different (though field-identical) type from `EstimatorMod.PoseFixModel`.
"""
function Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFEstimatorV2, fix, z_fix::SVector{3})
    !est.initialized && return false
    H_fix = @SMatrix [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    R_fix = Main.SensorModV2.R_posefix(fix)

    x = est.x
    z4 = SVector(z_fix[1], z_fix[2], cos(z_fix[3]), sin(z_fix[3]))
    e = z4 - H_fix * SVector{12}(x)
    S_fix = H_fix * est.P * H_fix' + R_fix
    nu = dot(e, inv(S_fix) * e)
    nu > 16.27 && return false

    K_fix = est.P * H_fix' * inv(S_fix)
    x .+= K_fix * e
    _renorm_heading!(x)
    I_KH = SMatrix{12,12}(I) - K_fix * H_fix
    Pn = I_KH * est.P * I_KH' + K_fix * (R_fix * K_fix')
    est.P .= 0.5 * (Pn + Pn')

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

end # module
