# =============================================================================
# tuning/metrics.jl — Pham-&-Han-style comparison metrics per closed-loop run
# =============================================================================
module TuningMetricsMod

using LinearAlgebra
using Statistics: mean, std, median

export MetricSet, compute_metrics

"""
    MetricSet

Per-run comparison metrics for one controller / regime / seed.  Designed to
match the Pham & Han comparison variables where applicable.
"""
Base.@kwdef struct MetricSet
    controller::String
    regime::String          # "transit" or "docking"
    seed::Int
    pose_rmse::Float64      # eq. 49: √(mean[(x−x_ref)²+(y−y_ref)²+(ψ−ψ_ref)²])
    velocity_nrmse::Float64 # averaged per-channel NRMSE
    inslip_nrmse::Float64   # velocity NRMSE during high-slip samples
    ce::Float64             # control effort: mean Σ|v_cmd_i| per timestep
    emax::Float64           # max pose or velocity error
    settling_time::Float64  # time to enter ±5% band (pose mode)
    overshoot::Float64      # max relative overshoot (pose mode)
    pose_drift_rate::Float64 # terminal pose error / path length
    dropout_drift::Float64  # max pose error during a fix dropout window
end

"Smooth unwrap for heading error."
function _heading_error(ψ, ψ_ref)
    d = ψ .- ψ_ref
    return atan.(sin.(d), cos.(d))
end

function _per_channel_nrmse(y, y_des)
    n_ch = size(y, 1)
    # Guard against near-zero reference RMS (e.g. dwell at zero velocity).
    floors = n_ch >= 3 ? [0.05, 0.05, 0.01] : fill(0.05, n_ch)
    vals = Float64[sqrt(mean((y[i,:] .- y_des[i,:]).^2)) / max(sqrt(mean(y_des[i,:].^2)), floors[i])
                   for i in 1:n_ch]
    return mean(vals)
end

"""
    compute_metrics(df::DataFrame, ref, mode::Symbol; controller="", regime="", seed=0, slip=nothing)

Compute the comparison metric set from a closed-loop log DataFrame.  `mode` is
`:velocity` or `:pose`.  Optional `slip` is the per-sample ground-truth slip
vector for in-slip weighting.
"""
function compute_metrics(df, ref, mode::Symbol; controller::String="", regime::String="", seed::Int=0, slip=nothing, dropout_window::Union{Tuple{Float64,Float64},Nothing}=nothing)
    t = df.time
    N = length(t)

    # Control effort CE = mean Σ|v_cmd_i|
    ce = 0.0
    for i in 1:4
        col = Symbol("v_cmd_$i")
        if hasproperty(df, col)
            ce += mean(abs.(df[!, col]))
        end
    end

    if mode == :pose
        Xo_ref = [ref.xo(ti) for ti in t]
        Yo_ref = [ref.yo(ti) for ti in t]
        psi_ref = [ref.psi(ti) for ti in t]
        pose_err = sqrt.((df.Xo .- Xo_ref).^2 .+ (df.Yo .- Yo_ref).^2 .+ _heading_error(df.psi, psi_ref).^2)
        pose_rmse = sqrt(mean(pose_err.^2))
        emax = maximum(pose_err)

        # Settling time: enter ±5% of final error band and stay there
        final_err = pose_err[end]
        band = 0.05 * max(final_err, 1e-3)
        settled = pose_err .< band
        settling_time = 0.0
        for i in N:-1:1
            if !settled[i]
                settling_time = t[min(i+1, N)]
                break
            end
        end

        # Overshoot relative to initial error
        overshoot = N > 1 ? maximum(pose_err) / max(pose_err[1], 1e-3) - 1.0 : 0.0
        overshoot = max(overshoot, 0.0)

        # Pose drift rate
        path_len = sum(sqrt.(df.Vx.^2 .+ df.Vy.^2)[2:end] .* diff(t))
        pose_drift_rate = norm([df.Xo[end]-Xo_ref[end], df.Yo[end]-Yo_ref[end], _heading_error([df.psi[end]], [psi_ref[end]])[1]]) / max(path_len, 1e-3)

        # Dropout drift: max pose error during the controlled dropout window
        if dropout_window !== nothing
            t0, t1 = dropout_window
            mask = (t .>= t0) .& (t .<= t1)
            dropout_drift = any(mask) ? maximum(pose_err[mask]) : emax
        else
            dropout_drift = emax
        end

        velocity_nrmse = NaN
        inslip_nrmse = NaN

    else
        y = hcat(df.Vx, df.Vy, df.psi_dot)'
        y_des = hcat(df.Vx_des, df.Vy_des, df.omega_des)'
        velocity_nrmse = _per_channel_nrmse(y, y_des)

        if slip !== nothing && length(slip) == N
            med_slip = median(slip)
            mask = slip .> med_slip
            if any(mask)
                inslip_nrmse = _per_channel_nrmse(y[:,mask], y_des[:,mask])
            else
                inslip_nrmse = velocity_nrmse
            end
        else
            inslip_nrmse = velocity_nrmse
        end

        pose_rmse = NaN
        emax = maximum(sqrt.(df.e_Vx.^2 .+ df.e_Vy.^2 .+ df.e_omega.^2))
        settling_time = NaN
        overshoot = NaN
        pose_drift_rate = NaN
        dropout_drift = NaN
    end

    return MetricSet(
        controller = controller,
        regime = regime,
        seed = seed,
        pose_rmse = pose_rmse,
        velocity_nrmse = velocity_nrmse,
        inslip_nrmse = inslip_nrmse,
        ce = ce,
        emax = emax,
        settling_time = settling_time,
        overshoot = overshoot,
        pose_drift_rate = pose_drift_rate,
        dropout_drift = dropout_drift,
    )
end

end # module
