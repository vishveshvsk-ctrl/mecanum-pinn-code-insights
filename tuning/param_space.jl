# =============================================================================
# tuning/param_space.jl — estimator parameter spaces + theta decoder
# =============================================================================
module TuningParamSpaceMod

using StaticArrays
using LinearAlgebra

export ParamDim, ParamSpace, n_params, kf_param_space, smo_param_space,
       imm_kf_param_space, eskf_param_space, apply_params!

"""
    ParamDim(name, len, lower, upper, scale, is_diag)

One named dimension of a search space.  For `scale == :log` the stored bounds
are interpreted as *log-space* bounds and `apply_params!` exponentiates the
corresponding raw theta components.  For `scale == :lin` the bounds are used
directly.
"""
struct ParamDim
    name::String
    len::Int
    lower::Float64
    upper::Float64
    scale::Symbol
    is_diag::Bool
end

"""
    ParamSpace(estimator, dims)

Search space for an estimator.  `estimator` is `:kalman` or `:smo`.  `dims`
is a vector of `ParamDim`; the constructor also caches flattened lower/upper
bounds and per-scalar parameter names.
"""
struct ParamSpace
    estimator::Symbol
    dims::Vector{ParamDim}
    flat_lower::Vector{Float64}
    flat_upper::Vector{Float64}
    flat_scale::Vector{Symbol}
    flat_name::Vector{String}
end

function ParamSpace(estimator::Symbol, dims::Vector{ParamDim})
    flat_lower = Float64[]
    flat_upper = Float64[]
    flat_scale = Symbol[]
    flat_name  = String[]
    for d in dims
        for i in 1:d.len
            push!(flat_lower, d.lower)
            push!(flat_upper, d.upper)
            push!(flat_scale, d.scale)
            push!(flat_name,  d.len == 1 ? d.name : "$(d.name)_$i")
        end
    end
    return ParamSpace(estimator, dims, flat_lower, flat_upper, flat_scale, flat_name)
end

n_params(sp::ParamSpace) = length(sp.flat_lower)

function _log_bounds(lo, hi)
    lo > 0 && hi > 0 || error("log-scale bounds must be positive")
    return log(lo), log(hi)
end

"""
    kf_param_space() -> ParamSpace

Tunable hyperparameters for `KalmanEstimator` (9 tunable scalar dims + 1 fixed
`rate_hz` dimension for completeness = 10 scalar params).

| dim            | len | scale | bounds        |
|----------------|-----|-------|---------------|
| Qn_diag        | 3   | log   | 1e-4 .. 1e0   |
| Rn_diag        | 3   | log   | 1e-5 .. 1e-1  |
| P0_scale       | 1   | log   | 1e-4 .. 1e0   |
| slip_R_inflate | 1   | log   | 1.0 .. 100.0  |
| slip_threshold | 1   | log   | 1e-3 .. 1.0   |
| rate_hz        | 1   | lin   | 1000 .. 1000  |
"""
function kf_param_space()
    dims = ParamDim[
        ParamDim("Qn_diag",        3, _log_bounds(1e-4, 1e0)...,   :log, true),
        ParamDim("Rn_diag",        3, _log_bounds(1e-5, 1e-1)...,   :log, true),
        ParamDim("P0_scale",       1, _log_bounds(1e-4, 1e0)...,   :log, false),
        ParamDim("slip_R_inflate", 1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("slip_threshold", 1, _log_bounds(1e-3, 1.0)...,    :log, false),
        ParamDim("rate_hz",        1, 1000.0, 1000.0,               :lin, false),
    ]
    return ParamSpace(:kalman, dims)
end

"""
    smo_param_space() -> ParamSpace

Tunable hyperparameters for `SMOEstimator` (9 tunable scalar dims + 1 fixed
`rate_hz` dimension for completeness = 10 scalar params).

| dim            | len | scale | bounds        |
|----------------|-----|-------|---------------|
| L              | 3   | log   | 0.1 .. 50.0   |
| K              | 3   | log   | 0.1 .. 200.0  |
| delta          | 1   | log   | 1e-4 .. 0.5   |
| slip_gate      | 1   | log   | 1e-3 .. 1.0   |
| zupt_threshold | 1   | log   | 1e-3 .. 0.5   |
| rate_hz        | 1   | lin   | 1000 .. 1000  |
"""
function smo_param_space()
    dims = ParamDim[
        ParamDim("L",              3, _log_bounds(0.1,  50.0)...,   :log, true),
        ParamDim("K",              3, _log_bounds(0.1,  200.0)...,  :log, true),
        ParamDim("delta",          1, _log_bounds(1e-4, 0.5)...,    :log, false),
        ParamDim("slip_gate",      1, _log_bounds(1e-3, 1.0)...,    :log, false),
        ParamDim("zupt_threshold", 1, _log_bounds(1e-3, 0.5)...,    :log, false),
        ParamDim("rate_hz",        1, 1000.0, 1000.0,               :lin, false),
    ]
    return ParamSpace(:smo, dims)
end

"""
    imm_kf_param_space() -> ParamSpace

Tunable hyperparameters for `IMMKalmanEstimator` (22 tunable scalar dims + 1
fixed `rate_hz` dimension for completeness = 23 scalar params).

| dim            | len | scale | bounds        |
|----------------|-----|-------|---------------|
| Qn_diag        | 3   | log   | 1e-4 .. 1e0   |
| Rn_diag        | 3   | log   | 1e-5 .. 1e-1  |
| P0_scale       | 1   | log   | 1e-4 .. 1e0   |
| slip_R_inflate | 1   | log   | 1.0 .. 100.0  |
| slip_threshold | 1   | log   | 1e-3 .. 1.0   |
| alpha_acc      | 1   | log   | 1e-2 .. 100.0 |
| alpha_yaw      | 1   | log   | 1e-2 .. 100.0 |
| slip_Qn_diag   | 2   | log   | 1e-4 .. 1e0   |
| r_boost        | 1   | log   | 1.0 .. 100.0  |
| p_stay_grip    | 1   | lin   | 0.5 .. 0.999  |
| p_stay_slip    | 1   | lin   | 0.5 .. 0.999  |
| bias_Qn_diag   | 2   | log   | 1e-6 .. 1e-2  |
| gyro_bias_Qn   | 1   | log   | 1e-8 .. 1e-2  |
| pose_Qn        | 1   | log   | 1e-8 .. 1e-3  |
| grip_slip_scale| 1   | log   | 1e-6 .. 1.0   |
| nis_thresh     | 1   | log   | 1.0 .. 100.0  |
| rate_hz        | 1   | lin   | 1000 .. 1000  |
"""
function imm_kf_param_space()
    dims = ParamDim[
        ParamDim("Qn_diag",        3, _log_bounds(1e-4, 1e0)...,   :log, true),
        ParamDim("Rn_diag",        3, _log_bounds(1e-5, 1e-1)...,   :log, true),
        ParamDim("P0_scale",       1, _log_bounds(1e-4, 1e0)...,   :log, false),
        ParamDim("slip_R_inflate", 1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("slip_threshold", 1, _log_bounds(1e-3, 1.0)...,    :log, false),
        ParamDim("alpha_acc",      1, _log_bounds(1e-2, 100.0)...,  :log, false),
        ParamDim("alpha_yaw",      1, _log_bounds(1e-2, 100.0)...,  :log, false),
        ParamDim("slip_Qn_diag",   2, _log_bounds(1e-4, 1e0)...,   :log, true),
        ParamDim("r_boost",        1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("p_stay_grip",    1, 0.5, 0.999,                   :lin, false),
        ParamDim("p_stay_slip",    1, 0.5, 0.999,                   :lin, false),
        ParamDim("bias_Qn_diag",   2, _log_bounds(1e-6, 1e-2)...,  :log, true),
        ParamDim("gyro_bias_Qn",   1, _log_bounds(1e-8, 1e-2)...,  :log, false),
        ParamDim("pose_Qn",        1, _log_bounds(1e-8, 1e-3)...,  :log, false),
        ParamDim("grip_slip_scale",1, _log_bounds(1e-6, 1.0)...,   :log, false),
        ParamDim("nis_thresh",     1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("rate_hz",        1, 1000.0, 1000.0,               :lin, false),
    ]
    return ParamSpace(:kalman_imm, dims)
end

"""
    eskf_param_space() -> ParamSpace

Tunable hyperparameters for `ESKFEstimator` (21 tunable scalar dims + 1 fixed
`rate_hz` dimension for completeness = 22 scalar params).

| dim            | len | scale | bounds        |
|----------------|-----|-------|---------------|
| Qn_diag        | 3   | log   | 1e-4 .. 1e0   |
| Rn_diag        | 3   | log   | 1e-5 .. 1e-1  |
| P0_scale       | 1   | log   | 1e-4 .. 1e0   |
| slip_R_inflate | 1   | log   | 1.0 .. 100.0  |
| slip_threshold | 1   | log   | 1e-3 .. 1.0   |
| alpha_acc      | 1   | log   | 1e-2 .. 100.0 |
| alpha_yaw      | 1   | log   | 1e-2 .. 100.0 |
| slip_Qn_diag   | 2   | log   | 1e-4 .. 1e0   |
| r_boost        | 1   | log   | 1.0 .. 100.0  |
| bias_Qn_diag   | 2   | log   | 1e-6 .. 1e-2  |
| gyro_bias_Qn   | 1   | log   | 1e-8 .. 1e-2  |
| pose_Qn        | 1   | log   | 1e-8 .. 1e-3  |
| grip_slip_scale| 1   | log   | 1e-6 .. 1.0   |
| nis_thresh     | 1   | log   | 1.0 .. 100.0  |
| pose_slip_gain | 1   | log   | 1e-1 .. 1000  |
| rate_hz        | 1   | lin   | 1000 .. 1000  |
"""
function eskf_param_space()
    dims = ParamDim[
        ParamDim("Qn_diag",        3, _log_bounds(1e-4, 1e0)...,   :log, true),
        ParamDim("Rn_diag",        3, _log_bounds(1e-5, 1e-1)...,   :log, true),
        ParamDim("P0_scale",       1, _log_bounds(1e-4, 1e0)...,   :log, false),
        ParamDim("slip_R_inflate", 1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("slip_threshold", 1, _log_bounds(1e-3, 1.0)...,    :log, false),
        ParamDim("alpha_acc",      1, _log_bounds(1e-2, 100.0)...,  :log, false),
        ParamDim("alpha_yaw",      1, _log_bounds(1e-2, 100.0)...,  :log, false),
        ParamDim("slip_Qn_diag",   2, _log_bounds(1e-4, 1e0)...,   :log, true),
        ParamDim("r_boost",        1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("bias_Qn_diag",   2, _log_bounds(1e-6, 1e-2)...,  :log, true),
        ParamDim("gyro_bias_Qn",   1, _log_bounds(1e-8, 1e-2)...,  :log, false),
        ParamDim("pose_Qn",        1, _log_bounds(1e-8, 1e-3)...,  :log, false),
        ParamDim("grip_slip_scale",1, _log_bounds(1e-6, 1.0)...,   :log, false),
        ParamDim("nis_thresh",     1, _log_bounds(1.0,  100.0)...,  :log, false),
        ParamDim("pose_slip_gain", 1, _log_bounds(1e-1, 1000.0)..., :log, false),
        ParamDim("rate_hz",        1, 1000.0, 1000.0,               :lin, false),
    ]
    return ParamSpace(:eskf, dims)
end

"""
    apply_params!(theta, space) -> NamedTuple

Decode a raw parameter vector from the optimizer into a concrete estimator
config (a NamedTuple of kwargs ready for `KalmanEstimator`/`SMOEstimator`).
Log-scale dimensions are exponentiated; all values are bounds-clamped.
"""
function apply_params!(theta::AbstractVector{<:Real}, space::ParamSpace)
    length(theta) == n_params(space) ||
        error("apply_params!: theta length $(length(theta)) != $(n_params(space))")

    # Clamp + decode.
    decoded = Vector{Float64}(undef, length(theta))
    for i in eachindex(theta)
        v = clamp(Float64(theta[i]), space.flat_lower[i], space.flat_upper[i])
        decoded[i] = space.flat_scale[i] == :log ? exp(v) : v
    end

    # Group decoded scalars back into named dims.
    idx = 1
    vals = Dict{String,Any}()
    for d in space.dims
        v = decoded[idx:idx+d.len-1]
        vals[d.name] = d.len == 1 ? v[1] : SVector{d.len}(v)
        idx += d.len
    end

    if space.estimator == :kalman
        return (
            estimator       = :kalman,
            Qn              = Diagonal(vals["Qn_diag"]),
            Rn_base         = Diagonal(vals["Rn_diag"]),
            bias_Qn         = Diagonal(SVector(1e-4, 1e-4)),
            P0_scale        = vals["P0_scale"],
            slip_R_inflate  = vals["slip_R_inflate"],
            slip_threshold  = vals["slip_threshold"],
            zupt_threshold  = 0.02,
            rate_hz         = vals["rate_hz"],
            use_dhat        = false,
        )
    elseif space.estimator == :kalman_imm
        return (
            estimator       = :kalman_imm,
            Qn              = Diagonal(vals["Qn_diag"]),
            Rn_base         = Diagonal(vals["Rn_diag"]),
            bias_Qn         = Diagonal(vals["bias_Qn_diag"]),
            slip_Qn         = Diagonal(vals["slip_Qn_diag"]),
            gyro_bias_Qn    = vals["gyro_bias_Qn"],
            pose_Qn         = vals["pose_Qn"],
            P0_scale        = vals["P0_scale"],
            slip_R_inflate  = vals["slip_R_inflate"],
            slip_threshold  = vals["slip_threshold"],
            zupt_threshold  = 0.02,
            grip_slip_scale = vals["grip_slip_scale"],
            alpha_acc       = vals["alpha_acc"],
            alpha_yaw       = vals["alpha_yaw"],
            r_boost         = vals["r_boost"],
            nis_thresh      = vals["nis_thresh"],
            p_stay_grip     = vals["p_stay_grip"],
            p_stay_slip     = vals["p_stay_slip"],
            rate_hz         = vals["rate_hz"],
            use_dhat        = false,
        )
    elseif space.estimator == :eskf
        return (
            estimator       = :eskf,
            Qn              = Diagonal(vals["Qn_diag"]),
            Rn_base         = Diagonal(vals["Rn_diag"]),
            bias_Qn         = Diagonal(vals["bias_Qn_diag"]),
            slip_Qn         = Diagonal(vals["slip_Qn_diag"]),
            gyro_bias_Qn    = vals["gyro_bias_Qn"],
            pose_Qn         = vals["pose_Qn"],
            pose_slip_gain  = vals["pose_slip_gain"],
            P0_scale        = vals["P0_scale"],
            slip_R_inflate  = vals["slip_R_inflate"],
            slip_threshold  = vals["slip_threshold"],
            zupt_threshold  = 0.02,
            alpha_acc       = vals["alpha_acc"],
            alpha_yaw       = vals["alpha_yaw"],
            r_boost         = vals["r_boost"],
            nis_thresh      = vals["nis_thresh"],
            grip_slip_scale = vals["grip_slip_scale"],
            rate_hz         = vals["rate_hz"],
            use_dhat        = false,
        )
    elseif space.estimator == :smo
        return (
            estimator       = :smo,
            L               = vals["L"],
            K               = vals["K"],
            delta           = vals["delta"],
            slip_gate_thresh= vals["slip_gate"],
            zupt_threshold  = vals["zupt_threshold"],
            bias_gain       = SVector(0.5, 0.5),
            rate_hz         = vals["rate_hz"],
            use_dhat        = true,
        )
    else
        error("apply_params!: unknown estimator $(space.estimator)")
    end
end

end # module
