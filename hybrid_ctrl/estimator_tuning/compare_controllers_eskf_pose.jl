#!/usr/bin/env julia
# =============================================================================
# compare_controllers_eskf_pose.jl
#   Evaluate locked-in ASMC (seed4) and PID (seed2) pose-mode controllers when
#   closed-loop feedback comes from the real seed-44 ESKF estimator.
#
#   Profiles: default_trajs_pose (octagon, spin_creep, coupled_vomega,
#             spiral_orbit) — ellipse entries excluded per user spec.
#
#   Outputs per-row:
#     - controller error : reference pose/velocity  vs  simulated true state
#     - estimator error  : simulated true state     vs  ESKF estimate
#   so deviations can be attributed to controller vs estimator.
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg
Pkg.activate(ROOT)

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# Pull in run_one, hybrid_ctrl, controller builder/metrics, trajectory subsets.
include(joinpath(ROOT, "tune_controller.jl"))
# _build_estimator for the frozen ESKF.
include(joinpath(ROOT, "tuning/harness.jl")); using .TuningHarnessMod

using StaticArrays
using Statistics
using Random
using TOML
using JSON
using DataFrames
using CSV
using Arrow

const USAGE = """
Usage: compare_controllers_eskf_pose.jl [options]

Evaluate locked pose-mode controllers through the frozen seed-44 ESKF.

Options:
  --estimator-dir DIR   Dir holding the frozen ESKF best_config.json
                        (default: runs_estimator_posfix_velref_mu05chi005_10traj_replay/seed_44/eskf_dxnes)
  --controllers SPEC    Comma list of ctrl:path pairs
                        (default: asmc:runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json,
                                  pid:runs_controller_pid_pose_5seed/seed2/pid_clean/best_config.json)
  --run-dir DIR         Trajectory config dir (default: trajectory_files_run_0p5_main)
  --seeds SEEDS         Comma-separated sensor seeds (default: 42)
  --sensor-noise MODE   Sensor/pose-fix noise: default | realistic
                        (default: default)
  --out DIR             Output root (default: hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose)
  --smoke               One ASMC/octagon run, then exit
  --help                Show this message
"""

const FINALIZED = Dict(:asmc => true, :pid => true)

function parse_args(argv)
    a = Dict{String,Any}(
        "estimator-dir" => "runs_estimator_posfix_velref_mu05chi005_10traj_replay/seed_44/eskf_dxnes",
        "controllers"   => "asmc:runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json," *
                           "pid:runs_controller_pid_pose_5seed/seed2/pid_clean/best_config.json",
        "run-dir"       => "trajectory_files_run_0p5_main",
        "seeds"         => [42],
        "sensor-noise"  => "default",
        "out"           => "hybrid_ctrl/estimator_tuning/reports/controller_eskf_pose",
        "smoke"         => false,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE); exit(0)
        elseif arg == "--estimator-dir"; a["estimator-dir"] = argv[i+1]; i += 2
        elseif arg == "--controllers";   a["controllers"]   = argv[i+1]; i += 2
        elseif arg == "--run-dir";       a["run-dir"]       = argv[i+1]; i += 2
        elseif arg == "--seeds";         a["seeds"] = [parse(Int, s) for s in split(argv[i+1], ',')]; i += 2
        elseif arg == "--sensor-noise";  a["sensor-noise"]  = argv[i+1]; i += 2
        elseif arg == "--out";           a["out"]           = argv[i+1]; i += 2
        elseif arg == "--smoke";         a["smoke"] = true; i += 1
        else; error("unknown arg $arg\n$USAGE"); end
    end
    a["sensor-noise"] in ("default", "realistic") ||
        error("--sensor-noise must be default or realistic")
    return a
end

"Parse `asmc:path,pid:path,...` into an ordered Vector of (Symbol, String)."
function parse_controllers(spec::AbstractString)
    out = Tuple{Symbol,String}[]
    for pair in split(spec, ',')
        isempty(strip(pair)) && continue
        ci = findfirst(==(':'), pair)
        ci === nothing && error("bad --controllers entry (need ctrl:path): $pair")
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

"Pose subset from experiment_noise_eval_pose.jl, ellipse entries removed."
function build_subset(run_dir::String)
    base = default_trajs_pose(run_dir)
    return filter(tr -> !startswith(string(tr.name), "ellipse"), base)
end

"Return the sensor/pose-fix noise tier to use in run_hybrid."
function sensor_noise_kind(mode::String)
    mode == "realistic" && return :realistic
    return :default
end

function run_controller_on_estimator(ctrl::Symbol, kw::NamedTuple, est_cfg, tr;
                                     seed::Int=42, sensor_noise::Symbol=:default)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    cfg = HybridConfig(
        tracking      = tr.run_mode,        # :pose
        estimator     = est_cfg.estimator,  # :eskf
        use_dhat      = get(est_cfg, :use_dhat, false),
        use_asmc      = ctrl == :asmc,
        use_mpc       = ctrl == :mpc,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = true,               # pose-mode requires/fix benefits from fixes
        pose_fix_tier = :docking,
        f_est         = est_cfg.rate_hz,
        sensor_seed   = seed,
    )

    # Deterministic ref build (combo_idx pinned).
    path  = joinpath(tr.config_dir, "profiles", tr.profile_toml)
    prof  = TOML.parsefile(path)["profile"]
    cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
    ref   = Profiles.build(prof["builder"], cfg_r)
    if get(tr, :adapt, false)
        ref = Profiles.velref_to_posref(ref)
    end
    Profiles.publish!(ref)

    est = TuningHarnessMod._build_estimator(est_cfg)   # fresh ESKF per run
    asmc_o, mpc_o, pid_o = build_controller(ctrl, kw)

    fix_override = sensor_noise == :realistic ?
        Main.EstimatorMod.PoseFixModel(:realistic; seed=seed) : nothing

    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=est, ref=ref,
        fix_override=fix_override,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, tr.run_mode
end

# -----------------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------------

_wrapdiff(a, b) = atan(sin(a - b), cos(a - b))

"Estimator error: true state - ESKF estimate.  Units SI."
function estimator_error(probe)
    isempty(probe) && return (
        vx=NaN, vy=NaN, psidot=NaN, psi=NaN, X=NaN, Y=NaN,
        pos=NaN, heading=NaN)
    rmse(h, t) = sqrt(mean((h .- t) .^ 2))
    tvx  = [p.u[1]  for p in probe]; hvx  = [p.xhat[1] for p in probe]
    tvy  = [p.u[2]  for p in probe]; hvy  = [p.xhat[2] for p in probe]
    tpd  = [p.u[3]  for p in probe]; hpd  = [p.xhat[3] for p in probe]
    tpsi = [p.u[4]  for p in probe]; hpsi = [p.xhat[4] for p in probe]
    tX   = [p.u[17] for p in probe]; hX   = [p.xhat[5] for p in probe]
    tY   = [p.u[18] for p in probe]; hY   = [p.xhat[6] for p in probe]
    psi_err = _wrapdiff.(hpsi, tpsi)
    pos_err = sqrt.((hX .- tX).^2 .+ (hY .- tY).^2)
    return (
        vx     = rmse(hvx, tvx),
        vy     = rmse(hvy, tvy),
        psidot = rmse(hpd, tpd),
        psi    = sqrt(mean(psi_err.^2)),
        X      = rmse(hX, tX),
        Y      = rmse(hY, tY),
        pos    = sqrt(mean(pos_err.^2)),
        heading= sqrt(mean(psi_err.^2)),
    )
end

"Controller error: reference - true plant state.  Units SI."
function controller_error(probe, ref, mode::Symbol)
    isempty(probe) && return (
        vx=NaN, vy=NaN, psidot=NaN, psi=NaN, X=NaN, Y=NaN,
        pos=NaN, heading=NaN, pos_final=NaN, heading_final=NaN)
    if mode == :velocity
        ex = [p.u[1] - ref.Vx(p.t) for p in probe]
        ey = [p.u[2] - ref.Vy(p.t) for p in probe]
        ep = [p.u[3] - ref.Wz(p.t) for p in probe]
        rms_vx = sqrt(mean(ex.^2)); rms_vy = sqrt(mean(ey.^2)); rms_w = sqrt(mean(ep.^2))
        return (vx=rms_vx, vy=rms_vy, psidot=rms_w, psi=NaN, X=NaN, Y=NaN,
                pos=NaN, heading=NaN, pos_final=NaN, heading_final=NaN)
    else  # :pose
        poserr  = Float64[]
        headerr = Float64[]
        X_err   = Float64[]
        Y_err   = Float64[]
        vx_g_err = Float64[]
        vy_g_err = Float64[]
        w_err    = Float64[]
        for p in probe
            t = p.t
            x, y, psi = p.u[17], p.u[18], p.u[4]
            Vx_b, Vy_b, psidot_b = p.u[1], p.u[2], p.u[3]
            c, s = cos(psi), sin(psi)
            Vx_g = Vx_b * c - Vy_b * s
            Vy_g = Vx_b * s + Vy_b * c

            push!(poserr,  sqrt((x - ref.xo(t))^2 + (y - ref.yo(t))^2))
            push!(headerr, abs(_wrapdiff(psi, ref.psi(t))))
            push!(X_err,   x - ref.xo(t))
            push!(Y_err,   y - ref.yo(t))
            push!(vx_g_err, Vx_g - ref.Vxo(t))
            push!(vy_g_err, Vy_g - ref.Vyo(t))
            push!(w_err,    psidot_b - ref.om(t))
        end
        return (
            vx        = sqrt(mean(vx_g_err.^2)),
            vy        = sqrt(mean(vy_g_err.^2)),
            psidot    = sqrt(mean(w_err.^2)),
            psi       = sqrt(mean(headerr.^2)),
            X         = sqrt(mean(X_err.^2)),
            Y         = sqrt(mean(Y_err.^2)),
            pos       = sqrt(mean(poserr.^2)),
            heading   = sqrt(mean(headerr.^2)),
            pos_final = poserr[end],
            heading_final = headerr[end],
        )
    end
end

function run_row(ctrl::Symbol, kw::NamedTuple, est_cfg, tr, seed::Int, sensor_noise::Symbol)
    probe, ref, mode = run_controller_on_estimator(ctrl, kw, est_cfg, tr;
                                                   seed=seed, sensor_noise=sensor_noise)
    m = controller_metrics(probe, ref, mode)
    e = estimator_error(probe)
    c = controller_error(probe, ref, mode)
    return (
        controller      = string(ctrl),
        finalized       = get(FINALIZED, ctrl, true),
        trajectory      = string(tr.name),
        mode            = string(mode),
        seed            = seed,
        sensor_noise    = string(sensor_noise),
        ok              = m.ok,
        tracking        = m.tracking,
        ce              = m.ce,
        chatter         = m.chatter,
        # Estimator error: true - estimate
        est_rmse_vx     = e.vx,
        est_rmse_vy     = e.vy,
        est_rmse_psidot = e.psidot,
        est_rmse_psi    = e.psi,
        est_rmse_X      = e.X,
        est_rmse_Y      = e.Y,
        est_rmse_pos    = e.pos,
        est_rmse_heading= e.heading,
        # Controller error: ref - true
        ctrl_rmse_vx    = c.vx,
        ctrl_rmse_vy    = c.vy,
        ctrl_rmse_psidot= c.psidot,
        ctrl_rmse_psi   = c.psi,
        ctrl_rmse_pos   = c.pos,
        ctrl_rmse_heading=c.heading,
        ctrl_pos_final  = c.pos_final,
        ctrl_head_final = c.heading_final,
    )
end

function main()
    a = parse_args(ARGS)
    est_cfg = load_frozen_estimator(a["estimator-dir"])
    println("Frozen estimator: $(est_cfg.estimator) @ $(est_cfg.rate_hz) Hz  ($(a["estimator-dir"]))")

    ctrl_specs = parse_controllers(a["controllers"])
    ctrl_kw = [(ctrl, load_controller_kw(path, ctrl)) for (ctrl, path) in ctrl_specs]
    for ((ctrl, _), (_, path)) in zip(ctrl_kw, ctrl_specs)
        fin = get(FINALIZED, ctrl, true) ? "finalized" : "NOT finalized"
        println("Controller: $(rpad(string(ctrl),4)) [$fin]  ($path)")
    end

    trajs = build_subset(a["run-dir"])
    println("Pose subset ($(length(trajs)) trajs): $(join([string(t.name) for t in trajs], ", "))")
    sensor_noise = sensor_noise_kind(a["sensor-noise"])
    println("Sensor noise mode: $sensor_noise")

    if a["smoke"]
        println("\n=== SMOKE: ASMC on octagon through the frozen ESKF ===")
        octagon = trajs[1]
        asmc_kw = first(kw for (c, kw) in ctrl_kw if c == :asmc)
        probe, ref, mode = run_controller_on_estimator(:asmc, asmc_kw, est_cfg, octagon;
                                                       seed=first(a["seeds"]), sensor_noise=sensor_noise)
        m = controller_metrics(probe, ref, mode)
        e = estimator_error(probe)
        c = controller_error(probe, ref, mode)
        println("  ticks=$(length(probe))  mode=$mode")
        println("  tracking=$(round(m.tracking,digits=4))  ce=$(round(m.ce,digits=3))  chatter=$(round(m.chatter,digits=4))  ok=$(m.ok)")
        println("  est_rmse:  vx=$(round(e.vx,digits=4)) vy=$(round(e.vy,digits=4)) ψ̇=$(round(e.psidot,digits=4)) " *
                "ψ=$(round(e.psi,digits=4)) X=$(round(e.X,digits=4)) Y=$(round(e.Y,digits=4))")
        println("  ctrl_rmse: vx=$(round(c.vx,digits=4)) vy=$(round(c.vy,digits=4)) ψ̇=$(round(c.psidot,digits=4)) " *
                "pos=$(round(c.pos,digits=4)) head=$(round(c.heading,digits=4))")
        if !isempty(probe)
            dmax = maximum(abs(p.xhat[1] - p.u[1]) for p in probe)
            println("  max|x̂_Vx - Vx_true| = $(round(dmax, digits=5))  " *
                    "($(dmax > 0 ? "OK: real ESKF" : "WARNING: looks like oracle feed"))")
        end
        println("=== SMOKE OK ===")
        return
    end

    out = a["out"]; mkpath(out)
    println("\nSeeds: $(a["seeds"])   Output root: $out")

    rows = NamedTuple[]
    for (ctrl, kw) in ctrl_kw
        for tr in trajs
            for seed in a["seeds"]
                println("  [run] $(rpad(string(ctrl),4)) $(rpad(string(tr.name),16)) seed=$(lpad(seed,2))")
                row = try
                    run_row(ctrl, kw, est_cfg, tr, seed, sensor_noise)
                catch err
                    @warn "run failed" controller=ctrl traj=tr.name seed=seed exception=err
                    (controller=string(ctrl), finalized=get(FINALIZED, ctrl, true),
                     trajectory=string(tr.name), mode=string(tr.run_mode), seed=seed,
                     sensor_noise=string(sensor_noise), ok=false, tracking=NaN, ce=NaN, chatter=NaN,
                     est_rmse_vx=NaN, est_rmse_vy=NaN, est_rmse_psidot=NaN, est_rmse_psi=NaN,
                     est_rmse_X=NaN, est_rmse_Y=NaN, est_rmse_pos=NaN, est_rmse_heading=NaN,
                     ctrl_rmse_vx=NaN, ctrl_rmse_vy=NaN, ctrl_rmse_psidot=NaN, ctrl_rmse_psi=NaN,
                     ctrl_rmse_pos=NaN, ctrl_rmse_heading=NaN, ctrl_pos_final=NaN, ctrl_head_final=NaN)
                end
                push!(rows, row)
            end
        end
    end

    df = DataFrame(rows)
    CSV.write(joinpath(out, "runs.csv"), df)
    Arrow.write(joinpath(out, "runs.arrow"), df)

    # Mean ± std grouped by controller + trajectory.
    _skipnan_mean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
    _skipnan_std(x)  = (v = filter(isfinite, x); length(v) < 2 ? NaN : std(v))
    summary = combine(groupby(df, [:controller, :trajectory, :sensor_noise]),
        :tracking         => _skipnan_mean => :tracking_mean,
        :tracking         => _skipnan_std  => :tracking_std,
        :ce               => _skipnan_mean => :ce_mean,
        :chatter          => _skipnan_mean => :chatter_mean,
        :est_rmse_pos     => _skipnan_mean => :est_rmse_pos_mean,
        :est_rmse_heading => _skipnan_mean => :est_rmse_heading_mean,
        :ctrl_rmse_pos    => _skipnan_mean => :ctrl_rmse_pos_mean,
        :ctrl_rmse_heading=> _skipnan_mean => :ctrl_rmse_heading_mean,
        :ctrl_pos_final   => _skipnan_mean => :ctrl_pos_final_mean,
        :ctrl_head_final  => _skipnan_mean => :ctrl_head_final_mean,
        nrow              => :n,
    )
    CSV.write(joinpath(out, "summary.csv"), summary)
    Arrow.write(joinpath(out, "summary.arrow"), summary)

    println("\nComparison complete. Outputs in $out/{runs,summary}.csv")
    println("Columns:")
    println("  est_rmse_*  = estimator error (true - estimate)")
    println("  ctrl_rmse_* = controller error (reference - true plant state)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
