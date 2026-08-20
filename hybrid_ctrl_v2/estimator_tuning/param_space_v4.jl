# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/param_space_v4.jl
# (ESKFEstimatorV3 — 13-dim yaw-accel state; q_alpha TUNABLE per user direction)
# =============================================================================
# `param_space_v3.jl` is NEVER edited — the v3/v3-wide runs under
# `runs_estimator_v3_mu0p5_train12/` and `runs_estimator_v3wide_mu0p5_train12/`
# reference it. This is a NEW, additive space for the NEW `ESKFEstimatorV3`
# struct (estimators_v3.jl), reusing `ParamDim`/`ParamSpace`/`n_params`/
# `_log_bounds` and `PINNED_V3` unchanged.
#
# Must be `include`d after `tuning/param_space.jl` and `param_space_v3.jl`.
#
# The 7th dim `q_alpha` is the per-tick random-walk variance of the new yaw-
# acceleration state [(rad/s^2)^2 per tick]. Bounds rationale: over a 1 s
# horizon (1000 ticks) the random walk wanders by std sqrt(1000*q_alpha) --
# the range 1e-8..1e-1 spans "alpha frozen" (~1e-2 rad/s^2 over 1 s) to
# "alpha fully re-learned within a tick" (~3 rad/s^2 over 1 s), bracketing
# any plausible maneuver/jerk scale for these profiles.
#
# P0_alpha is PINNED at 0.25 (+-0.5 rad/s^2 initial yaw-accel uncertainty,
# profile-scale) per the PINNED_V3 convention ("P0 = statement of knowledge").
# All other pins inherit PINNED_V3 verbatim.
# =============================================================================
module ParamSpaceV4Mod

using StaticArrays
using LinearAlgebra

export eskf_param_space_v4, apply_params_v4!, PINNED_V4

const PINNED_V4 = merge(Main.ParamSpaceV3Mod.PINNED_V3, (
    P0_alpha = 0.25,   # [(rad/s^2)^2] -- +-0.5 rad/s^2 initial yaw-accel uncertainty
))

"""
    eskf_param_space_v4() -> ParamSpace

ESKFEstimatorV3 search space: v3's 6 update-rate dims + `q_alpha`.
"""
function eskf_param_space_v4()
    T = Main.TuningParamSpaceMod
    dims = T.ParamDim[
        T.ParamDim("alpha_acc",       1, T._log_bounds(1e-2, 1e2)...,   :log, false),
        T.ParamDim("alpha_yaw",       1, T._log_bounds(1e-2, 1e2)...,   :log, false),
        T.ParamDim("grip_slip_scale", 1, T._log_bounds(1e-5, 1e0)...,   :log, false),
        T.ParamDim("r_boost",         1, T._log_bounds(1.0,  100.0)..., :log, false),
        T.ParamDim("pose_Qn_heading", 1, T._log_bounds(1e-8, 1e-3)...,  :log, false),
        T.ParamDim("slip_R_inflate",  1, T._log_bounds(1.0,  100.0)..., :log, false),
        T.ParamDim("q_alpha",         1, T._log_bounds(1e-8, 1e-1)...,  :log, false),
    ]
    return T.ParamSpace(:eskf_v4, dims)
end

"""
    apply_params_v4!(theta, space) -> NamedTuple

Decode a raw theta vector into ESKFEstimatorV3 kwargs: the 7 tuned dims plus
the physically-pinned values (PINNED_V4). Same decode contract as
`ParamSpaceV3Mod.apply_params_v3!`.
"""
function apply_params_v4!(theta::AbstractVector{<:Real}, space)
    T = Main.TuningParamSpaceMod
    length(theta) == T.n_params(space) ||
        error("apply_params_v4!: theta length $(length(theta)) != $(T.n_params(space))")

    decoded = Vector{Float64}(undef, length(theta))
    for i in eachindex(theta)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[i] = space.flat_scale[i] == :log ? exp(v) : v
    end
    vals = Dict(d.name => decoded[i] for (i, d) in enumerate(space.dims))

    return merge(PINNED_V4, (
        alpha_acc       = vals["alpha_acc"],
        alpha_yaw       = vals["alpha_yaw"],
        grip_slip_scale = vals["grip_slip_scale"],
        r_boost         = vals["r_boost"],
        pose_Qn_heading = vals["pose_Qn_heading"],
        slip_R_inflate  = vals["slip_R_inflate"],
        q_alpha         = vals["q_alpha"],
        use_dhat        = false,
    ))
end

end # module
