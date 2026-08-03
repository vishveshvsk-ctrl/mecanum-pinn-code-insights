# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl
# (instructions/sensors-suite-consolidation-and-physical-noise.md §7.5)
# =============================================================================
# `tuning/param_space.jl` is NEVER edited — every existing caller
# (`eval_profile_breakdown.jl`, `rerank_topk.jl`, `tune_estimator.jl`,
# `evaluate_seed44_*.jl`) keeps using the original 22-dim `eskf_param_space()`
# completely unaffected. This is a NEW, additive space for ESKFEstimatorV2,
# reusing `ParamDim`/`ParamSpace`/`n_params`/`_log_bounds` from the original
# module unchanged (plain leaf types, no reason to duplicate them).
#
# Must be `include`d after `tuning/param_space.jl`.
# =============================================================================
module ParamSpaceV2Mod

using StaticArrays
using LinearAlgebra

export eskf_param_space_v2, apply_params_v2!

"""
    eskf_param_space_v2() -> ParamSpace

ESKFEstimatorV2 search space, per a user design pass that (a) splits the
single `P0_scale` into SEVEN per-group initial-variance scales (translational
velocity, yaw rate, heading, accel bias, gyro bias, slip, position — each a
physically distinct quantity/unit, so tying them to one scalar was judged
wrong; position in particular was previously HARDCODED to 1.0 and not tunable
at all), (b) splits `pose_Qn` into heading vs. position (different units,
different unmodelled-error sources), and (c) REMOVES `pose_slip_gain`
entirely: with the optical-flow channel giving a slip-immune fix on Vx/Vy,
the velocity feeding the pose prediction is already accurate during slip, so
inflating pose process noise specifically on slip activity is now redundant.
10 dims total (up from the brief §7.5 draft's 4 — see `estimators_v2.jl`'s
struct docstring for the full reasoning trail).

Removed entirely (DERIVED, never searched — see `hybrid_ctrl_v2/estimators_v2.jl`):
    Rn_diag(3), Qn_diag(3), bias_Qn(2), slip_Qn(2), gyro_bias_Qn(1),
    slip_threshold(1), nis_thresh(1)

Retained tunable:

| dim              | len | scale | bounds        | why |
|------------------|-----|-------|---------------|-----|
| P0_vel           | 1   | log   | 1e-4 .. 1e0   | Vx,Vy init variance [(m/s)^2] |
| P0_yaw           | 1   | log   | 1e-4 .. 1e0   | psidot init variance [(rad/s)^2] |
| P0_heading       | 1   | log   | 1e-4 .. 1e0   | c,s init variance [unitless] |
| P0_bias_acc      | 1   | log   | 1e-6 .. 1e-1  | bx,by init variance [(m/s^2)^2] -- wide: bias is a genuine unknown, not a well-constrained rest state |
| P0_bias_gyro     | 1   | log   | 1e-8 .. 1e-3  | bg init variance [(rad/s)^2] |
| P0_slip          | 1   | log   | 1e-6 .. 1e-1  | sx,sy init variance [(m/s)^2] |
| P0_pos           | 1   | log   | 1e-3 .. 1e1   | X,Y init variance [m^2] -- newly tunable, was hardcoded 1.0 |
| pose_Qn_heading  | 1   | log   | 1e-8 .. 1e-3  | unmodelled heading-drift rate |
| pose_Qn_pos      | 1   | log   | 1e-8 .. 1e-3  | unmodelled position-drift rate |
| slip_R_inflate   | 1   | log   | 1.0 .. 100.0  | how hard to distrust wheels once slip is DETECTED -- wheel-channel only, policy not sensor spec |

`alpha_acc`/`alpha_yaw`/`r_boost`/`grip_slip_scale` remain PINNED at their v1
defaults (not searched) — they compensate for mis-specified R/Q, which is no
longer the failure mode once R/Q are derived; kept fixed so a later ablation
can show whether they still matter at all.
"""
function eskf_param_space_v2()
    T = Main.TuningParamSpaceMod
    dims = T.ParamDim[
        T.ParamDim("P0_vel",          1, T._log_bounds(1e-4, 1e0)...,   :log, false),
        T.ParamDim("P0_yaw",          1, T._log_bounds(1e-4, 1e0)...,   :log, false),
        T.ParamDim("P0_heading",      1, T._log_bounds(1e-4, 1e0)...,   :log, false),
        T.ParamDim("P0_bias_acc",     1, T._log_bounds(1e-6, 1e-1)...,  :log, false),
        T.ParamDim("P0_bias_gyro",    1, T._log_bounds(1e-8, 1e-3)...,  :log, false),
        T.ParamDim("P0_slip",         1, T._log_bounds(1e-6, 1e-1)...,  :log, false),
        T.ParamDim("P0_pos",          1, T._log_bounds(1e-3, 1e1)...,   :log, false),
        T.ParamDim("pose_Qn_heading", 1, T._log_bounds(1e-8, 1e-3)...,  :log, false),
        T.ParamDim("pose_Qn_pos",     1, T._log_bounds(1e-8, 1e-3)...,  :log, false),
        T.ParamDim("slip_R_inflate",  1, T._log_bounds(1.0,  100.0)..., :log, false),
    ]
    return T.ParamSpace(:eskf_v2, dims)
end

"""
    apply_params_v2!(theta, space) -> NamedTuple

Decode a raw theta vector into ESKFEstimatorV2 kwargs. `tau_slip`/
`sigma_slip`/`sigma_gyro_bias_rw` are NOT part of `theta` -- they're measured
constants (see `estimators_v2.jl`'s struct docstring), passed through as the
struct's own defaults unless explicitly overridden by the caller.
"""
function apply_params_v2!(theta::AbstractVector{<:Real}, space)
    T = Main.TuningParamSpaceMod
    length(theta) == T.n_params(space) ||
        error("apply_params_v2!: theta length $(length(theta)) != $(T.n_params(space))")

    decoded = Vector{Float64}(undef, length(theta))
    for i in eachindex(theta)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[i] = space.flat_scale[i] == :log ? exp(v) : v
    end
    vals = Dict(d.name => decoded[i] for (i, d) in enumerate(space.dims))

    return (
        P0_vel          = vals["P0_vel"],
        P0_yaw          = vals["P0_yaw"],
        P0_heading      = vals["P0_heading"],
        P0_bias_acc     = vals["P0_bias_acc"],
        P0_bias_gyro    = vals["P0_bias_gyro"],
        P0_slip         = vals["P0_slip"],
        P0_pos          = vals["P0_pos"],
        pose_Qn_heading = vals["pose_Qn_heading"],
        pose_Qn_pos     = vals["pose_Qn_pos"],
        slip_R_inflate  = vals["slip_R_inflate"],
        use_dhat        = false,
    )
end

end # module
