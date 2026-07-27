#!/usr/bin/env julia
# =============================================================================
# eval_profile_breakdown.jl — profile-wise error breakdown for the current
# checkpoint_best.json theta of two ESKF tuning runs (runs_eskf_full,
# runs_eskf_noellipse), tagged so results don't collide (both are "eskf").
# =============================================================================

using Pkg
Pkg.activate(".")

include("run_one.jl")
using .Profiles, .DataStore

include("hybrid_ctrl/config.jl");    using .HybridConfigMod
include("hybrid_ctrl/bus.jl");       using .BusMod
include("hybrid_ctrl/plant.jl");     using .PlantMod
include("hybrid_ctrl/sensors.jl");   using .SensorMod
include("hybrid_ctrl/estimators.jl"); using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");     using .FuzzyMod
include("hybrid_ctrl/mixer.jl");     using .MixerMod
include("hybrid_ctrl/scheduler.jl"); using .SchedulerMod

include("tuning/subset.jl");      using .TuningSubsetMod
include("tuning/param_space.jl"); using .TuningParamSpaceMod
include("tuning/harness.jl");     using .TuningHarnessMod
include("tuning/objectives.jl");  using .TuningObjectivesMod
include("tuning/metrics.jl");     using .TuningMetricsMod

using DataFrames, Arrow, CSV, JSON, TOML, Statistics

const RUNS = [
    ("full_v2",      "runs_eskf_full_v2"),
    ("noellipse_v2", "runs_eskf_noellipse_v2"),
]
const SEEDS = [42, 43]

function load_theta(run_dir::String)
    path = joinpath(run_dir, "eskf_dxnes", "checkpoint_best.json")
    data = JSON.parse(read(path, String))
    return Float64.(data["theta"]), Float64(data["score"]), Int(data["iteration"])
end

function load_manifest_entries(run_dir::String)
    data = JSON.parse(read(joinpath(run_dir, "subset_manifest.json"), String))
    return [
        (name=Symbol(e["name"]),
         profile_toml=e["profile_toml"],
         ref_type=Symbol(e["ref_type"]),
         mu=Float64(e["mu"]),
         config_dir=e["config_dir"],
         run_mode=Symbol(e["run_mode"]),
         combo_idx=get(e, "combo_idx", nothing),
         pose_fix_tier=get(e, "pose_fix_tier", nothing) === nothing ? nothing : Symbol(e["pose_fix_tier"]))
        for e in data["entries"]
    ]
end

function build_cfg(est_cfg::NamedTuple, entry::NamedTuple, seed::Int)
    tracking = entry.run_mode
    use_pose_fix = tracking == :pose || entry.pose_fix_tier !== nothing
    pose_fix_tier = entry.pose_fix_tier === nothing ?
        (tracking == :pose ? :docking : :transit) : entry.pose_fix_tier
    return HybridConfig(
        tracking       = tracking,
        estimator      = :eskf,
        use_dhat       = false,
        use_asmc       = true,
        use_mpc        = false,
        use_pid        = false,
        fuzzy          = false,
        fixed_weights  = (1.0, 0.0, 0.0),
        use_pose_fix   = use_pose_fix,
        pose_fix_tier  = pose_fix_tier,
        f_est          = est_cfg.rate_hz,
        f_mpc          = 100.0,
        f_pid          = 100.0,
        f_fuzzy        = 50.0,
        f_mix          = 1000.0,
        sensor_seed    = seed,
        reltol         = 1e-8,
        abstol_bristle = 1e-10,
        dtmax          = 1e-3,
        solver_symbol  = :TRBDF2,
        saveat_hz      = 500.0,
    )
end

const V_TOL = 1e-3    # m/s
const RATE_TOL = 1e-2 # rad/s
const POS_TOL = 1e-2  # m
const HEAD_TOL = 1e-2 # rad

_rms(x) = sqrt(mean(x .^ 2))
_wrap_diff(a, b) = atan.(sin.(a .- b), cos.(a .- b))

function run_one_eval(est_cfg::NamedTuple, entry::NamedTuple, seed::Int)
    cfg = build_cfg(est_cfg, entry, seed)
    est = TuningHarnessMod._build_estimator(est_cfg)

    base = Profiles.load_base(entry.config_dir)
    chi = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(entry.mu))

    prof = TOML.parsefile(joinpath(entry.config_dir, "profiles", entry.profile_toml))["profile"]
    pcfg = Profiles.resolve_profile(prof; combo_idx=entry.combo_idx)
    ref = Profiles.build(String(prof["builder"]), pcfg)

    sol, df, bus = SchedulerMod.run_hybrid(
        cfg, params, entry.name;
        chi=chi, friction_case=1, ref=ref, est=est,
        config_dir=entry.config_dir, profile_toml=entry.profile_toml,
        return_bus=true)

    slip = [TuningHarnessMod.slip_indicator(sol.u[k], params) for k in eachindex(sol.t)]

    probe = SchedulerMod.build_estimator_log(bus, params; decimation=1)

    e_vx = probe.v_hat[1,:] .- probe.v_true[1,:]
    e_vy = probe.v_hat[2,:] .- probe.v_true[2,:]
    e_ps = probe.v_hat[3,:] .- probe.v_true[3,:]
    vel_mag = sqrt.(e_vx.^2 .+ e_vy.^2)
    vel_rmse  = 0.5 * (_rms(e_vx) + _rms(e_vy))     # m/s, absolute
    rate_rmse = _rms(e_ps)                           # rad/s, absolute
    vel_max   = maximum(vel_mag)                      # m/s, absolute
    rate_max  = maximum(abs.(e_ps))                    # rad/s, absolute

    in_mask = probe.slip .> median(probe.slip)
    inslip_vel_rmse = any(in_mask) ?
        0.5 * (_rms(e_vx[in_mask]) + _rms(e_vy[in_mask])) : vel_rmse

    is_posref = entry.ref_type == :posref
    if is_posref
        e_x = probe.pose_hat[1,:] .- probe.pose_true[1,:]
        e_y = probe.pose_hat[2,:] .- probe.pose_true[2,:]
        e_h = _wrap_diff(probe.pose_hat[3,:], probe.pose_true[3,:])
        pos_mag = sqrt.(e_x.^2 .+ e_y.^2)
        pos_rmse     = _rms(pos_mag)                    # m, absolute
        heading_rmse = _rms(e_h)                         # rad, absolute
        pos_max      = maximum(pos_mag)                  # m, absolute
        heading_max  = maximum(abs.(e_h))                 # rad, absolute
    else
        pos_rmse = NaN
        heading_rmse = NaN
        pos_max = NaN
        heading_max = NaN
    end

    m = compute_metrics(df, ref, cfg.tracking;
                        controller="eskf", regime=string(entry.name), seed=seed, slip=slip)
    return (m=m, vel_rmse=vel_rmse, rate_rmse=rate_rmse, inslip_vel_rmse=inslip_vel_rmse,
            pos_rmse=pos_rmse, heading_rmse=heading_rmse,
            vel_max=vel_max, rate_max=rate_max, pos_max=pos_max, heading_max=heading_max)
end

_skipnan_mean(x) = (v = filter(!isnan, collect(x)); isempty(v) ? NaN : mean(v))
_skipnan_std(x)  = (v = filter(!isnan, collect(x)); length(v) < 2 ? NaN : std(v))

function main()
    space = eskf_param_space()
    rows = NamedTuple[]

    for (tag, run_dir) in RUNS
        theta, ckpt_score, ckpt_iter = load_theta(run_dir)
        est_cfg = apply_params!(theta, space)
        entries = load_manifest_entries(run_dir)
        println("== $tag ($run_dir): checkpoint iter=$ckpt_iter score=$(round(ckpt_score,digits=4)), $(length(entries)) profiles ==")

        for entry in entries
            for seed in SEEDS
                print("  [$tag] $(entry.name) seed=$seed ... ")
                try
                    r = run_one_eval(est_cfg, entry, seed)
                    m = r.m
                    push!(rows, (
                        run              = tag,
                        traj_name        = string(entry.name),
                        mode             = string(entry.run_mode),
                        seed             = seed,
                        vel_rmse_mps     = r.vel_rmse,
                        vel_tol_ratio    = r.vel_rmse / V_TOL,
                        inslip_vel_rmse_mps = r.inslip_vel_rmse,
                        rate_rmse_radps  = r.rate_rmse,
                        rate_tol_ratio   = r.rate_rmse / RATE_TOL,
                        pos_rmse_m       = r.pos_rmse,
                        pos_tol_ratio    = r.pos_rmse / POS_TOL,
                        heading_rmse_rad = r.heading_rmse,
                        heading_tol_ratio = r.heading_rmse / HEAD_TOL,
                        vel_max_mps      = r.vel_max,
                        vel_max_tol_ratio = r.vel_max / V_TOL,
                        rate_max_radps   = r.rate_max,
                        rate_max_tol_ratio = r.rate_max / RATE_TOL,
                        pos_max_m        = r.pos_max,
                        pos_max_tol_ratio = r.pos_max / POS_TOL,
                        heading_max_rad  = r.heading_max,
                        heading_max_tol_ratio = r.heading_max / HEAD_TOL,
                        emax             = m.emax,
                        ce               = m.ce,
                        pose_drift_rate  = m.pose_drift_rate,
                    ))
                    println("ok (vel_rmse=$(round(r.vel_rmse,digits=5)) m/s = $(round(r.vel_rmse/V_TOL,digits=1))x tol; " *
                            "vel_max=$(round(r.vel_max,digits=5)) m/s = $(round(r.vel_max/V_TOL,digits=1))x tol)")
                catch e
                    println("FAILED: $e")
                    push!(rows, (
                        run=tag, traj_name=string(entry.name), mode=string(entry.run_mode), seed=seed,
                        vel_rmse_mps=NaN, vel_tol_ratio=NaN, inslip_vel_rmse_mps=NaN,
                        rate_rmse_radps=NaN, rate_tol_ratio=NaN, pos_rmse_m=NaN, pos_tol_ratio=NaN,
                        heading_rmse_rad=NaN, heading_tol_ratio=NaN,
                        vel_max_mps=NaN, vel_max_tol_ratio=NaN, rate_max_radps=NaN, rate_max_tol_ratio=NaN,
                        pos_max_m=NaN, pos_max_tol_ratio=NaN, heading_max_rad=NaN, heading_max_tol_ratio=NaN,
                        emax=NaN, ce=NaN, pose_drift_rate=NaN,
                    ))
                end
            end
        end
    end

    df = DataFrame(rows)
    mkpath("runs_eskf_compare_v2")
    Arrow.write(joinpath("runs_eskf_compare_v2", "profile_breakdown.arrow"), df)
    CSV.write(joinpath("runs_eskf_compare_v2", "profile_breakdown.csv"), df)

    summary = combine(
        groupby(df, [:run, :traj_name, :mode]),
        :vel_rmse_mps        => _skipnan_mean => :vel_rmse_mps_mean,
        :vel_tol_ratio       => _skipnan_mean => :vel_tol_ratio_mean,
        :inslip_vel_rmse_mps => _skipnan_mean => :inslip_vel_rmse_mps_mean,
        :rate_rmse_radps     => _skipnan_mean => :rate_rmse_radps_mean,
        :rate_tol_ratio      => _skipnan_mean => :rate_tol_ratio_mean,
        :pos_rmse_m          => _skipnan_mean => :pos_rmse_m_mean,
        :pos_tol_ratio       => _skipnan_mean => :pos_tol_ratio_mean,
        :heading_rmse_rad    => _skipnan_mean => :heading_rmse_rad_mean,
        :heading_tol_ratio   => _skipnan_mean => :heading_tol_ratio_mean,
        :vel_max_mps         => _skipnan_mean => :vel_max_mps_mean,
        :vel_max_tol_ratio   => _skipnan_mean => :vel_max_tol_ratio_mean,
        :rate_max_radps      => _skipnan_mean => :rate_max_radps_mean,
        :rate_max_tol_ratio  => _skipnan_mean => :rate_max_tol_ratio_mean,
        :pos_max_m           => _skipnan_mean => :pos_max_m_mean,
        :pos_max_tol_ratio   => _skipnan_mean => :pos_max_tol_ratio_mean,
        :heading_max_rad     => _skipnan_mean => :heading_max_rad_mean,
        :heading_max_tol_ratio => _skipnan_mean => :heading_max_tol_ratio_mean,
        :ce                  => _skipnan_mean => :ce_mean,
        nrow                 => :n,
    )
    CSV.write(joinpath("runs_eskf_compare_v2", "profile_breakdown_summary.csv"), summary)
    println("\n=== Summary (mean over seeds) ===")
    show(summary, allrows=true, allcols=true)
    println()
    println("\nWrote runs_eskf_compare_v2/profile_breakdown.csv and profile_breakdown_summary.csv")
end

main()
