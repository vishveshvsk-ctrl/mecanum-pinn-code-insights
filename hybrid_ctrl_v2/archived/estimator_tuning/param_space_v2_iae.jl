# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/param_space_v2_iae.jl — ParamSpaceV2IAEMod
# (instructions/estimator-v2-iae-adaptive.md §5)
# =============================================================================
# `hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl` is NEVER edited. This
# module adds the 12-dimensional search space for `ESKFIAEEstimatorV2`: the
# 10 shared ESKF-v2 dims reused verbatim plus `tau_iae` and `kappa_iae`.
#
# Must be `include`d after `hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl`.
# =============================================================================
module ParamSpaceV2IAEMod

using StaticArrays
using LinearAlgebra

export eskf_iae_param_space_v2, apply_params_v2_iae!

"""
    eskf_iae_param_space_v2() -> ParamSpace

ESKFIAEEstimatorV2 search space. The first 10 dimensions are reused verbatim
from `ParamSpaceV2Mod.eskf_param_space_v2()`; two new dimensions control the
IAE adaptation loop (brief §2):

| dim       | len | scale | bounds       | why |
|-----------|-----|-------|--------------|-----|
| tau_iae   | 1   | log   | 0.05 .. 5.0  | adaptation timescale [s] — must stay slower than filter dynamics |
| kappa_iae | 1   | lin   | 0.1 .. 1.0   | adaptation exponent — response aggressiveness |

`gamma_min`/`gamma_max` remain PINNED at 1e-2/1e4 and are NOT searched.
"""
function eskf_iae_param_space_v2()
    T = Main.TuningParamSpaceMod
    base = Main.ParamSpaceV2Mod.eskf_param_space_v2()
    dims = copy(base.dims)
    push!(dims, T.ParamDim("tau_iae",   1, T._log_bounds(0.05, 5.0)..., :log, false))
    push!(dims, T.ParamDim("kappa_iae", 1, 0.1, 1.0,                       :lin, false))
    return T.ParamSpace(:eskf_iae_v2, dims)
end

"""
    apply_params_v2_iae!(theta, space) -> NamedTuple

Decode a raw 12-dim theta vector for `ESKFIAEEstimatorV2`. The first 10 dims
are delegated to `ParamSpaceV2Mod.apply_params_v2!`; `tau_iae` is log-scale
and `kappa_iae` is linear scale (the existing `:log` vs linear branch in
`apply_params_v2!` handles both). Returns `use_iae=true` and `use_dhat=false`.
"""
function apply_params_v2_iae!(theta::AbstractVector{<:Real}, space)
    T = Main.TuningParamSpaceMod
    base = Main.ParamSpaceV2Mod.eskf_param_space_v2()
    length(theta) == T.n_params(space) ||
        error("apply_params_v2_iae!: theta length $(length(theta)) != $(T.n_params(space))")

    # First 10 dims: identical to v2.
    base_cfg = Main.ParamSpaceV2Mod.apply_params_v2!(theta[1:10], base)

    # Last 2 dims: IAE adaptation params.
    decoded = Vector{Float64}(undef, 2)
    for (j, i) in enumerate(11:12)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[j] = space.flat_scale[i] == :log ? exp(v) : v
    end
    tau_iae   = decoded[1]
    kappa_iae = decoded[2]

    return merge(base_cfg, (tau_iae=tau_iae, kappa_iae=kappa_iae, use_iae=true))
end

end # module
