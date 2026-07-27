# =============================================================================
# tuning/results.jl — persist subset manifest, trials, best config, diagnostics
# =============================================================================
module TuningResultsMod

using DataFrames
using Arrow
using CSV
using JSON
using LinearAlgebra
using StaticArrays
using Plots

using Main.TuningParamSpaceMod: apply_params!
using Main.TuningHarnessMod: EstimatorLog

export save_trials, save_best_config, save_diagnostics

_json_value(x::Number)        = x
_json_value(x::Symbol)        = string(x)
_json_value(x::AbstractVector)= collect(x)
_json_value(x::SVector)       = collect(x)
_json_value(x::SMatrix)       = collect(x)
_json_value(x::Diagonal)      = collect(diag(x))
_json_value(x)                = string(x)

function _nt_to_dict(nt::NamedTuple)
    d = Dict{String,Any}()
    for (k, v) in pairs(nt)
        d[string(k)] = _json_value(v)
    end
    return d
end

"""
    save_trials(trials, outdir)

Persist the ranked per-theta results table as `outdir/trials.arrow`.
Returns the written path.
"""
function save_trials(trials::Vector{<:NamedTuple}, outdir::AbstractString)
    mkpath(outdir)
    df = DataFrame(trials)
    path = joinpath(outdir, "trials.arrow")
    Arrow.write(path, df)
    return path
end

"""
    save_best_config(best_theta, best_score, space, est_name, outdir)

Write `outdir/best_config.json` containing the best raw theta, its score, and
the decoded estimator config (loadable as kwargs for `KalmanEstimator` or
`SMOEstimator`).
"""
function save_best_config(best_theta::AbstractVector{<:Real},
                          best_score::Real,
                          space,
                          est_name::AbstractString,
                          outdir::AbstractString)
    mkpath(outdir)
    est_cfg = apply_params!(best_theta, space)
    data = Dict(
        "estimator"  => est_name,
        "best_score" => Float64(best_score),
        "theta"      => collect(best_theta),
        "config"     => _nt_to_dict(est_cfg)
    )
    path = joinpath(outdir, "best_config.json")
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
    return path
end

"""
    save_diagnostics(logs, outdir)

Write per-trajectory CSVs under `outdir/`.  If `Plots.jl` is available, also
write PNG diagnostic plots (true vs estimated velocity, estimation error,
slip vs time, disturbance estimate).  Returns `outdir`.
"""
function save_diagnostics(logs::Vector{EstimatorLog}, outdir::AbstractString)
    mkpath(outdir)

    # Always write CSV diagnostics.
    for log in logs
        df = DataFrame(
            time      = log.time,
            Vx_true   = log.v_true[1,:],
            Vy_true   = log.v_true[2,:],
            Vpsi_true = log.v_true[3,:],
            Vx_hat    = log.v_hat[1,:],
            Vy_hat    = log.v_hat[2,:],
            Vpsi_hat  = log.v_hat[3,:],
            Xo_true   = log.pose_true[1,:],
            Yo_true   = log.pose_true[2,:],
            psi_true  = log.pose_true[3,:],
            Xo_hat    = log.pose_hat[1,:],
            Yo_hat    = log.pose_hat[2,:],
            psi_hat   = log.pose_hat[3,:],
            dx_hat    = log.d_hat[1,:],
            dy_hat    = log.d_hat[2,:],
            dpsi_hat  = log.d_hat[3,:],
            slip      = log.slip,
        )
        CSV.write(joinpath(outdir, "log_$(log.traj_name).csv"), df)
    end

    # Attempt PNG plots; gracefully degrade if Plots.jl is missing.
    png_ok = false
    try
        @eval using Plots
        png_ok = true
    catch e
        @warn "Plots.jl not available; skipping PNG diagnostics" exception=e
    end

    if png_ok
        for log in logs
            _plot_log(log, outdir)
        end
    end

    return outdir
end

function _plot_log(log::EstimatorLog, outdir::AbstractString)
    t = log.time
    labels = ["Vx", "Vy", "ψ̇"]

    # True vs estimated velocity.
    p1 = Plots.plot(title="Velocity: $(log.traj_name)", xlabel="time [s]", ylabel="vel")
    for i in 1:3
        Plots.plot!(p1, t, log.v_true[i,:], label="$(labels[i]) true", lw=2)
        Plots.plot!(p1, t, log.v_hat[i,:], label="$(labels[i]) hat", ls=:dash)
    end

    # Velocity estimation error.
    err = log.v_hat .- log.v_true
    p2 = Plots.plot(title="Velocity error: $(log.traj_name)", xlabel="time [s]", ylabel="err")
    for i in 1:3
        Plots.plot!(p2, t, err[i,:], label=labels[i])
    end

    # Slip indicator.
    p3 = Plots.plot(title="Slip: $(log.traj_name)", xlabel="time [s]", ylabel="slip [m/s]")
    Plots.plot!(p3, t, log.slip, label="slip")

    # Disturbance estimate.
    p4 = Plots.plot(title="d_hat: $(log.traj_name)", xlabel="time [s]", ylabel="d_hat")
    for i in 1:3
        Plots.plot!(p4, t, log.d_hat[i,:], label=labels[i])
    end

    fig = Plots.plot(p1, p2, p3, p4; layout=(2,2), size=(1000,800))
    Plots.savefig(fig, joinpath(outdir, "diagnostics_$(log.traj_name).png"))
    return nothing
end

end # module
