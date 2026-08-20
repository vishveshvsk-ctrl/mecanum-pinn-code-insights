# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/param_space_v2_slipobs.jl — ParamSpaceV2SlipObsMod
# =============================================================================
# Search spaces for the slip-observer-extended ESKF.  The first 10 dimensions
# are reused verbatim from `ParamSpaceV2Mod.eskf_param_space_v2()`; only the
# observer-specific dimensions are appended.
#
# Include-after: param_space_v2.jl.
# =============================================================================
module ParamSpaceV2SlipObsMod

using StaticArrays
using LinearAlgebra

export eskf_slipobs_param_space_v2, apply_params_v2_slipobs!,
       apply_params_v2_smo!, apply_params_v2_eso!

"""
    eskf_slipobs_param_space_v2(kind::Symbol) -> ParamSpace

Return the search space for `:smo` (13 dims) or `:eso` (12 dims).
The shared 10 ESKF dims are taken directly from
`ParamSpaceV2Mod.eskf_param_space_v2()`; variant-specific dims are appended:

| dim         | len | scale | bounds        | used by |
|-------------|-----|-------|---------------|---------|
| smo_k1      | 1   | log   | 0.01 .. 2.0   | :smo |
| smo_k2      | 1   | log   | 0.5 .. 50.0   | :smo |
| eso_omega_o | 1   | log   | 5.0 .. 200.0  | :eso |
| rho_s       | 1   | log   | 1e-3 .. 0.2   | both |
"""
function eskf_slipobs_param_space_v2(kind::Symbol)
    base = Main.ParamSpaceV2Mod.eskf_param_space_v2()
    T = Main.TuningParamSpaceMod

    if kind == :smo
        extra = T.ParamDim[
            T.ParamDim("smo_k1", 1, T._log_bounds(0.01, 2.0)...,   :log, false),
            T.ParamDim("smo_k2", 1, T._log_bounds(0.5,  50.0)...,  :log, false),
            T.ParamDim("rho_s",  1, T._log_bounds(1e-3, 0.2)...,   :log, false),
        ]
        estimator = :eskf_slipobs_smo_v2
    elseif kind == :eso
        extra = T.ParamDim[
            T.ParamDim("eso_omega_o", 1, T._log_bounds(5.0,  200.0)..., :log, false),
            T.ParamDim("rho_s",       1, T._log_bounds(1e-3, 0.2)...,   :log, false),
        ]
        estimator = :eskf_slipobs_eso_v2
    else
        error("eskf_slipobs_param_space_v2: kind must be :smo or :eso, got $kind")
    end

    dims = vcat(base.dims, extra)
    return T.ParamSpace(estimator, dims)
end

"""
    _decode_theta(theta, space) -> Dict{String,Float64}

Mirror of `ParamSpaceV2Mod.apply_params_v2!`'s decoding step: clamp to the
flat box, exponentiate log-scale components, and map by dimension name.
"""
function _decode_theta(theta::AbstractVector{<:Real}, space)
    T = Main.TuningParamSpaceMod
    length(theta) == T.n_params(space) ||
        error("apply_params_v2_slipobs!: theta length $(length(theta)) != $(T.n_params(space))")

    decoded = Vector{Float64}(undef, length(theta))
    for i in eachindex(theta)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[i] = space.flat_scale[i] == :log ? exp(v) : v
    end
    vals = Dict(d.name => decoded[i] for (i, d) in enumerate(space.dims))
    return vals
end

"""
    apply_params_v2_smo!(theta, space) -> NamedTuple

Decoder for the `:smo` variant.  Returns the 10 shared ESKF params plus
`observer_kind=:smo`, `smo_k1`, `smo_k2`, `rho_s`, `use_slipobs=true`, and
`use_dhat=false`.
"""
function apply_params_v2_smo!(theta::AbstractVector{<:Real}, space)
    space.estimator == :eskf_slipobs_smo_v2 ||
        error("apply_params_v2_smo!: space is $(space.estimator), expected :eskf_slipobs_smo_v2")
    vals = _decode_theta(theta, space)
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
        observer_kind   = :smo,
        smo_k1          = vals["smo_k1"],
        smo_k2          = vals["smo_k2"],
        rho_s           = vals["rho_s"],
        use_slipobs     = true,
        use_dhat        = false,
    )
end

"""
    apply_params_v2_eso!(theta, space) -> NamedTuple

Decoder for the `:eso` variant.  Returns the 10 shared ESKF params plus
`observer_kind=:eso`, `eso_omega_o`, `rho_s`, `use_slipobs=true`, and
`use_dhat=false`.
"""
function apply_params_v2_eso!(theta::AbstractVector{<:Real}, space)
    space.estimator == :eskf_slipobs_eso_v2 ||
        error("apply_params_v2_eso!: space is $(space.estimator), expected :eskf_slipobs_eso_v2")
    vals = _decode_theta(theta, space)
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
        observer_kind   = :eso,
        eso_omega_o     = vals["eso_omega_o"],
        rho_s           = vals["rho_s"],
        use_slipobs     = true,
        use_dhat        = false,
    )
end

"""
    apply_params_v2_slipobs!(theta, space) -> NamedTuple

Unified decoder that dispatches on `space.estimator` to the `:smo` or `:eso`
decoder.
"""
function apply_params_v2_slipobs!(theta::AbstractVector{<:Real}, space)
    if space.estimator == :eskf_slipobs_smo_v2
        return apply_params_v2_smo!(theta, space)
    elseif space.estimator == :eskf_slipobs_eso_v2
        return apply_params_v2_eso!(theta, space)
    else
        error("apply_params_v2_slipobs!: unknown space estimator $(space.estimator)")
    end
end

end # module
