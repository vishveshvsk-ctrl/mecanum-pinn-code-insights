#!/usr/bin/env julia
# =============================================================================
# save_controller_eskf_traces.jl
#   Run the locked ASMC and PID pose-mode controllers through the frozen seed-44
#   ESKF for one sensor seed, save full time-series traces for later plotting.
#
#   Conditions: clean (default) and realistic sensor/pose-fix noise.
#   Trajectories: octagon, spin_creep, coupled_vomega, spiral_orbit.
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg
Pkg.activate(ROOT)

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(ROOT, "tune_controller.jl"))
include(joinpath(ROOT, "tuning/harness.jl")); using .TuningHarnessMod

using StaticArrays
using Random
using TOML
using JSON
using DataFrames
using Arrow

const USAGE = """
Usage: save_controller_eskf_traces.jl [options]

Save full time-series traces for ASMC vs PID through the frozen ESKF.

Options:
  --estimator MODE      Feedback source: eskf | oracle (default: eskf)
  --estimator-dir DIR   Frozen ESKF dir (default: runs_estimator_posfix_velref_mu05chi005_10traj_replay/seed_44/eskf_dxnes)
  --controllers SPEC    Comma list of ctrl:path pairs
                        (default: asmc:runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json,
                                  pid:runs_controller_pid_pose_5seed/seed2/pid_clean/best_config.json)
  --run-dir DIR         Trajectory config dir (default: trajectory_files_run_0p5_main)
  --traj-spec SPEC      Override trajectory subset with a single spec:
                        name:profile_toml:combo_idx (default: use default pose subset)
  --seed SEED           Sensor seed (default: 42)
  --out DIR             Output root for traces (default: hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose_traces)
  --help                Show this message
"""

function parse_args(argv)
    a = Dict{String,Any}(
        "estimator"     => "eskf",
        "estimator-dir" => "runs_estimator_posfix_velref_mu05chi005_10traj_replay/seed_44/eskf_dxnes",
        "controllers"   => "asmc:runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json," *
                           "pid:runs_controller_pid_pose_5seed/seed2/pid_clean/best_config.json",
        "run-dir"       => "trajectory_files_run_0p5_main",
        "traj-spec"     => nothing,
        "seed"          => 42,
        "out"           => "hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose_traces",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE); exit(0)
        elseif arg == "--estimator";     a["estimator"]     = argv[i+1]; i += 2
        elseif arg == "--estimator-dir"; a["estimator-dir"] = argv[i+1]; i += 2
        elseif arg == "--controllers";   a["controllers"]   = argv[i+1]; i += 2
        elseif arg == "--run-dir";       a["run-dir"]       = argv[i+1]; i += 2
        elseif arg == "--traj-spec";     a["traj-spec"]     = argv[i+1]; i += 2
        elseif arg == "--seed";          a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out";           a["out"]           = argv[i+1]; i += 2
        else; error("unknown arg $arg\n$USAGE"); end
    end
    a["estimator"] in ("eskf", "oracle") ||
        error("--estimator must be eskf or oracle")
    return a
end

function parse_controllers(spec::AbstractString)
    out = Tuple{Symbol,String}[]
    for pair in split(spec, ',')
        isempty(strip(pair)) && continue
        ci = findfirst(==(':'), pair)
        ci === nothing && error("bad --controllers entry: $pair")
        ctrl = Symbol(strip(pair[1:ci-1]))
        path = String(strip(pair[ci+1:end]))
        push!(out, (ctrl, path))
    end
    return out
end

function load_frozen_estimator(est_dir::String)
    path = joinpath(est_dir, "best_config.json")
    isfile(path) || error("Frozen estimator config not found: $path")
    data = JSON.parse(read(path, String))
    cfg = data["config"]
    est_name = cfg["estimator"]
    est_name == "eskf" || error("This script expects a frozen ESKF config; got estimator=$est_name")
    return (
        estimator       = :eskf,
        Qn              = Diagonal(SVector{3}(Float64.(cfg["Qn"]))),
        Rn_base         = Diagonal(SVector{3}(Float64.(cfg["Rn_base"]))),
        bias_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "bias_Qn", [1e-4, 1e-4])))),
        slip_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "slip_Qn", [1e-2, 1e-2])))),
        gyro_bias_Qn    = Float64(get(cfg, "gyro_bias_Qn", 1e-6)),
        pose_Qn         = Float64(get(cfg, "pose_Qn", 1e-6)),
        pose_slip_gain  = Float64(get(cfg, "pose_slip_gain", 10.0)),
        P0_scale        = Float64(cfg["P0_scale"]),
        slip_R_inflate  = Float64(cfg["slip_R_inflate"]),
        slip_threshold  = Float64(cfg["slip_threshold"]),
        zupt_threshold  = Float64(get(cfg, "zupt_threshold", 0.02)),
        alpha_acc       = Float64(get(cfg, "alpha_acc", 1.0)),
        alpha_yaw       = Float64(get(cfg, "alpha_yaw", 0.5)),
        r_boost         = Float64(get(cfg, "r_boost", 10.0)),
        nis_thresh      = Float64(get(cfg, "nis_thresh", 9.21)),
        grip_slip_scale = Float64(get(cfg, "grip_slip_scale", 1e-3)),
        rate_hz         = Float64(cfg["rate_hz"]),
        use_dhat        = false,
    )
end

function load_controller_kw(path::String, ctrl::Symbol)
    isfile(path) || error("controller config not found: $path")
    d = JSON.parse(read(path, String))
    haskey(d, "best_gains") || error("no `best_gains` in $path")
    pairs = Pair{Symbol,Any}[]
    for (k, v) in d["best_gains"]
        if v isa AbstractVector
            vv = Float64.(v)
            push!(pairs, Symbol(k) => SVector{length(vv)}(vv))
        else
            push!(pairs, Symbol(k) => Float64(v))
        end
    end
    return (; pairs...)
end

"Parse a single trajectory spec `name:profile_toml:combo_idx`."
function parse_traj_spec(spec::AbstractString, run_dir::String)
    parts = split(spec, ':')
    length(parts) == 3 || error("bad --traj-spec (need name:profile_toml:combo_idx): $spec")
    name, toml, cidx = strip.(parts)
    return (name=Symbol(name), profile_toml=toml, combo_idx=parse(Int, cidx),
            ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:pose, adapt=true)
end

"Pose subset from experiment_noise_eval_pose.jl, ellipse entries removed."
function build_subset(run_dir::String, traj_spec)
    traj_spec === nothing && return filter(tr -> !startswith(string(tr.name), "ellipse"),
                                           default_trajs_pose(run_dir))
    return [parse_traj_spec(traj_spec, run_dir)]
end

function run_one_trace(ctrl::Symbol, kw::NamedTuple, est_cfg, tr;
                       seed::Int=42, sensor_noise::Symbol=:default,
                       feedback::Symbol=:eskf)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    is_oracle = feedback == :oracle
    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = is_oracle ? :oracle : est_cfg.estimator,
        use_dhat      = is_oracle ? false : get(est_cfg, :use_dhat, false),
        use_asmc      = ctrl == :asmc,
        use_mpc       = ctrl == :mpc,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = !is_oracle,        # oracle supplies pose directly
        pose_fix_tier = is_oracle ? :none : :docking,
        f_est         = is_oracle ? 1000.0 : est_cfg.rate_hz,
        sensor_seed   = seed,
    )

    path  = joinpath(tr.config_dir, "profiles", tr.profile_toml)
    prof  = TOML.parsefile(path)["profile"]
    cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
    ref   = Profiles.build(prof["builder"], cfg_r)
    if get(tr, :adapt, false)
        ref = Profiles.velref_to_posref(ref)
    end
    Profiles.publish!(ref)

    asmc_o, mpc_o, pid_o = build_controller(ctrl, kw)

    if is_oracle
        oracle_kind = sensor_noise == :realistic ? :noisy : :clean
        oracle = OracleEstimator(oracle_kind; seed=seed, scale=1.0)
        sol, _df, bus = SchedulerMod.run_hybrid(
            cfg, params, Symbol(tr.name);
            chi=chi, friction_case=1, config_dir=tr.config_dir,
            profile_toml=tr.profile_toml, return_bus=true, est=oracle, ref=ref,
            asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)
    else
        est = TuningHarnessMod._build_estimator(est_cfg)
        fix_override = sensor_noise == :realistic ?
            Main.EstimatorMod.PoseFixModel(:realistic; seed=seed) : nothing
        sol, _df, bus = SchedulerMod.run_hybrid(
            cfg, params, Symbol(tr.name);
            chi=chi, friction_case=1, config_dir=tr.config_dir,
            profile_toml=tr.profile_toml, return_bus=true, est=est, ref=ref,
            fix_override=fix_override,
            asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)
    end

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref
end

_wrapdiff(a, b) = atan(sin(a - b), cos(a - b))

function extract_trace(probe, ref)
    isempty(probe) && return DataFrame()
    n = length(probe)
    t      = Vector{Float64}(undef, n)
    x_ref  = Vector{Float64}(undef, n); y_ref  = Vector{Float64}(undef, n); psi_ref = Vector{Float64}(undef, n)
    vx_ref = Vector{Float64}(undef, n); vy_ref = Vector{Float64}(undef, n); w_ref   = Vector{Float64}(undef, n)
    x_true = Vector{Float64}(undef, n); y_true = Vector{Float64}(undef, n); psi_true = Vector{Float64}(undef, n)
    vx_true= Vector{Float64}(undef, n); vy_true= Vector{Float64}(undef, n); w_true  = Vector{Float64}(undef, n)
    x_est  = Vector{Float64}(undef, n); y_est  = Vector{Float64}(undef, n); psi_est = Vector{Float64}(undef, n)
    vx_est = Vector{Float64}(undef, n); vy_est = Vector{Float64}(undef, n); w_est   = Vector{Float64}(undef, n)
    pos_err_true = Vector{Float64}(undef, n)
    head_err_true = Vector{Float64}(undef, n)
    pos_err_est   = Vector{Float64}(undef, n)
    head_err_est  = Vector{Float64}(undef, n)

    for (i, p) in enumerate(probe)
        tt = p.t
        t[i] = tt
        x_ref[i]  = ref.xo(tt);  y_ref[i]  = ref.yo(tt);  psi_ref[i] = ref.psi(tt)
        vx_ref[i] = ref.Vxo(tt); vy_ref[i] = ref.Vyo(tt); w_ref[i]   = ref.om(tt)

        vx_b, vy_b, w_b = p.u[1], p.u[2], p.u[3]
        psi = p.u[4]
        c, s = cos(psi), sin(psi)
        vx_true[i] = vx_b * c - vy_b * s
        vy_true[i] = vx_b * s + vy_b * c
        w_true[i]  = w_b
        psi_true[i]= psi
        x_true[i]  = p.u[17]
        y_true[i]  = p.u[18]

        vx_est[i] = p.xhat[1]; vy_est[i] = p.xhat[2]; w_est[i] = p.xhat[3]
        psi_est[i]= p.xhat[4]; x_est[i]  = p.xhat[5]; y_est[i] = p.xhat[6]

        pos_err_true[i]  = sqrt((x_true[i] - x_ref[i])^2 + (y_true[i] - y_ref[i])^2)
        head_err_true[i] = abs(_wrapdiff(psi_true[i], psi_ref[i]))
        pos_err_est[i]   = sqrt((x_est[i] - x_ref[i])^2 + (y_est[i] - y_ref[i])^2)
        head_err_est[i]  = abs(_wrapdiff(psi_est[i], psi_ref[i]))
    end

    return DataFrame(
        t=t,
        x_ref=x_ref, y_ref=y_ref, psi_ref=psi_ref,
        vx_ref=vx_ref, vy_ref=vy_ref, w_ref=w_ref,
        x_true=x_true, y_true=y_true, psi_true=psi_true,
        vx_true=vx_true, vy_true=vy_true, w_true=w_true,
        x_est=x_est, y_est=y_est, psi_est=psi_est,
        vx_est=vx_est, vy_est=vy_est, w_est=w_est,
        pos_err_true=pos_err_true, head_err_true=head_err_true,
        pos_err_est=pos_err_est, head_err_est=head_err_est,
    )
end

function main()
    a = parse_args(ARGS)
    feedback = Symbol(a["estimator"])
    est_cfg = feedback == :eskf ? load_frozen_estimator(a["estimator-dir"]) : nothing
    ctrl_specs = parse_controllers(a["controllers"])
    ctrl_kw = [(ctrl, load_controller_kw(path, ctrl)) for (ctrl, path) in ctrl_specs]
    trajs = build_subset(a["run-dir"], a["traj-spec"])
    out = a["out"]; mkpath(out)
    seed = a["seed"]

    if feedback == :eskf
        println("Frozen estimator: $(est_cfg.estimator) @ $(est_cfg.rate_hz) Hz")
    else
        println("Feedback: oracle (true state + optional noise)")
    end
    println("Controllers: $(join([string(c) for (c,_) in ctrl_specs], ", "))")
    println("Trajectories: $(join([string(t.name) for t in trajs], ", "))")
    println("Seed: $seed  Output: $out")

    sensor_modes = [(:default, "clean"), (:realistic, "realistic")]

    for (ctrl, kw) in ctrl_kw
        for tr in trajs
            for (sn, sn_label) in sensor_modes
                println("  [run] $(rpad(string(ctrl),4)) $(rpad(string(tr.name),16)) $(rpad(sn_label,10)) seed=$seed")
                probe, ref = run_one_trace(ctrl, kw, est_cfg, tr;
                                           seed=seed, sensor_noise=sn, feedback=feedback)
                df = extract_trace(probe, ref)
                if isempty(df)
                    @warn "empty probe" controller=ctrl traj=tr.name sensor=sn_label
                    continue
                end
                fname = "$(string(ctrl))_$(string(tr.name))_$(sn_label)_seed$(seed).arrow"
                Arrow.write(joinpath(out, fname), df)
                println("        -> $(fname)  (rows=$(nrow(df)))")
            end
        end
    end
    println("Trace saving complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
