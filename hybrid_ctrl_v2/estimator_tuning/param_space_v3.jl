# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/param_space_v3.jl
# (pruned "update-rate" space — supersedes param_space_v2.jl for retune)
# =============================================================================
# `param_space_v2.jl` is NEVER edited — the mu0p5/train12 3-seed runs under
# `runs_estimator_v2_mu0p5_train12/` reference it. This is a NEW, additive
# space for the SAME `ESKFEstimatorV2` struct, reusing
# `ParamDim`/`ParamSpace`/`n_params`/`_log_bounds` unchanged.
#
# Must be `include`d after `tuning/param_space.jl`.
#
# RATIONALE (k3 identifiability analysis + cross_eval_mu0p5.jl decisive check,
# see chat-handoff/eskf_v2_mu0p5_train12_retune_handoff.md):
#
# The v2 10-dim space tuned mostly INITIAL COVARIANCES (P0_*). The cross-eval
# (3 tuned configs x {orig, p0-reset} x 3 common noise seeds x train12+test)
# showed:
#   * P0_vel / P0_yaw / P0_slip are BIT-IDENTICAL in score between tuned and
#     default values -- on full-trajectory replay P collapses to its steady-
#     state Riccati solution within a fraction of a second at 1 kHz, so the
#     objective cannot see initial covariances of directly-measured states.
#   * P0_heading / P0_pos matter only through a ONE-SHOT event: whether the
#     heading locks before pose-fix innovations exceed the NIS gate (16.27)
#     -- a lock/no-lock bistability, not a continuous performance knob.
#   * P0_bias_acc / P0_bias_gyro are the exception that proves the rule:
#     with near-zero process noise their P collapses early and STAYS
#     collapsed, so their P0 sets the bias learning rate for the whole run.
#     The optimizer independently converged to the PHYSICAL PRIORS
#     (tuned 3.6-6.5e-4 vs (0.02 m/s^2)^2 = 4e-4; tuned 0.7-7e-6 vs
#     (0.003 rad/s)^2 = 9e-6).
#
# Conclusion (user direction): P0 is a STATEMENT OF KNOWLEDGE about t=0, not
# a performance knob -> pin all seven at physically-derived priors. The
# honest tunables are the UPDATE-RATE quantities: how fast P grows (Q-side
# q_scale coefficients, slip-Q downscale) and how hard measurements are
# distrusted (R-side policy multipliers). These were PINNED at v1 defaults in
# v2; they are what actually shapes steady-state filter responsiveness.
#
# PINNED (derived physical priors, never searched):
#
# | dim             | value | justification |
# |-----------------|-------|---------------|
# | P0_vel          | 1e-4  | rest start + +-1 cm/s nudge allowance [(m/s)^2] |
# | P0_yaw          | 1e-3  | rest start + +-0.03 rad/s disturbance [(rad/s)^2] |
# | P0_heading      | 0.5   | unit-circle ceiling: Var(cos psi) for uniform unknown heading |
# | P0_bias_acc     | 4e-4  | (0.02 m/s^2)^2 -- matches sim draw AND v2 tuned optimum |
# | P0_bias_gyro    | 9e-6  | (0.003 rad/s)^2 -- matches sim draw AND v2 tuned optimum |
# | P0_slip         | 2e-4  | sigma_slip^2 -- measured OU stationary variance |
# | P0_pos          | 0.25  | +-0.5 m "roughly known start" scenario [m^2] |
# | pose_Qn_pos     | 1e-7  | boundary optimum, all three v2 seeds (docking fix pins position) |
#
# RETAINED TUNABLE (6 dims, all update-rate / policy):
#
# | dim              | len | scale | bounds        | why |
# |------------------|-----|-------|---------------|-----|
# | alpha_acc        | 1   | log   | 1e-2 .. 1e2   | q_scale boost per |a| [s^2/m] -- velocity-channel responsiveness; v1-pinned at 1.0. Upper widened 1e1 -> 1e2 after the first v3 seed pinned BOTH alpha_* at the 1e1 bound (score 8.84): the lag-responsiveness trade was not yet bracketed |
# | alpha_yaw        | 1   | log   | 1e-2 .. 1e2   | q_scale boost per |omega| [s/rad] -- THE yaw-lag knob: rate_rmse (~46% of score) is structural ramp lag, q_yr gain is the only in-struct lever short of a yaw-accel state; v1-pinned at 0.5. Upper widened 1e1 -> 1e2 (same reason) |
# | grip_slip_scale  | 1   | log   | 1e-5 .. 1e0   | slip-Q downscale while GRIPPING -- with the decay fix (gauss_markov_q now 1/tau), sets how fast slip covariance relaxes between slip events; v1-pinned at 1e-3 |
# | r_boost          | 1   | log   | 1.0 .. 100.0  | NIS-rejection R boost (wheel channels) -- outlier-rejection aggressiveness; v1-pinned at 10 |
# | pose_Qn_heading  | 1   | log   | 1e-8 .. 1e-3  | continuous heading-covariance inflation -- the seed1-mode robustness mechanism (keeps pose fixes inside the NIS gate); v2 bounds retained |
# | slip_R_inflate   | 1   | log   | 1.0 .. 100.0  | wheel-R inflation once slip DETECTED -- v2 bounds retained |
#
# `use_dhat` fixed false; `tau_slip`/`sigma_slip`/`sigma_gyro_bias_rw` remain
# measured constants (see estimators_v2.jl struct docstring).
# =============================================================================
module ParamSpaceV3Mod

using StaticArrays
using LinearAlgebra

export eskf_param_space_v3, apply_params_v3!, PINNED_V3

# Physically-derived pinned values (see header rationale). Kept as a module
# constant so diagnostics can report exactly what was pinned.
const PINNED_V3 = (
    P0_vel        = 1e-4,
    P0_yaw        = 1e-3,
    P0_heading    = 0.5,
    P0_bias_acc   = 4e-4,
    P0_bias_gyro  = 9e-6,
    P0_slip       = 2e-4,
    P0_pos        = 0.25,
    pose_Qn_pos   = 1e-7,
)

"""
    eskf_param_space_v3() -> ParamSpace

Pruned 6-dim update-rate space for ESKFEstimatorV2 (see module header).
"""
function eskf_param_space_v3()
    T = Main.TuningParamSpaceMod
    dims = T.ParamDim[
        T.ParamDim("alpha_acc",       1, T._log_bounds(1e-2, 1e2)...,   :log, false),
        T.ParamDim("alpha_yaw",       1, T._log_bounds(1e-2, 1e2)...,   :log, false),
        T.ParamDim("grip_slip_scale", 1, T._log_bounds(1e-5, 1e0)...,   :log, false),
        T.ParamDim("r_boost",         1, T._log_bounds(1.0,  100.0)..., :log, false),
        T.ParamDim("pose_Qn_heading", 1, T._log_bounds(1e-8, 1e-3)...,  :log, false),
        T.ParamDim("slip_R_inflate",  1, T._log_bounds(1.0,  100.0)..., :log, false),
    ]
    return T.ParamSpace(:eskf_v3, dims)
end

"""
    apply_params_v3!(theta, space) -> NamedTuple

Decode a raw theta vector into ESKFEstimatorV2 kwargs: the 6 tuned update-rate
dims plus the physically-pinned P0/pose_Qn_pos values (PINNED_V3). Same decode
contract as `ParamSpaceV2Mod.apply_params_v2!` (log-scale exp, clamped).
"""
function apply_params_v3!(theta::AbstractVector{<:Real}, space)
    T = Main.TuningParamSpaceMod
    length(theta) == T.n_params(space) ||
        error("apply_params_v3!: theta length $(length(theta)) != $(T.n_params(space))")

    decoded = Vector{Float64}(undef, length(theta))
    for i in eachindex(theta)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[i] = space.flat_scale[i] == :log ? exp(v) : v
    end
    vals = Dict(d.name => decoded[i] for (i, d) in enumerate(space.dims))

    return merge(PINNED_V3, (
        alpha_acc       = vals["alpha_acc"],
        alpha_yaw       = vals["alpha_yaw"],
        grip_slip_scale = vals["grip_slip_scale"],
        r_boost         = vals["r_boost"],
        pose_Qn_heading = vals["pose_Qn_heading"],
        slip_R_inflate  = vals["slip_R_inflate"],
        use_dhat        = false,
    ))
end

end # module
