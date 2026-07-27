#!/usr/bin/env julia
# Dump time-series pose data for one controller × trajectory × use_pose_fix case
# so Python can plot reference vs true vs estimated trajectory.
using Pkg; Pkg.activate(".")

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include("tune_controller.jl")
include("tuning/harness.jl"); using .TuningHarnessMod

using StaticArrays
using Random
using TOML
using JSON
using DelimitedFiles

const EST_DIR = "runs_eskf_noellipse_v2/eskf_dxnes"
const CTRL_CFG = Dict(
    :asmc => "runs_controller_asmc_pin/asmc_FINAL_seed3.json",
    :pid  => "runs_controller_pid_5seed/pid_FINAL_seed2.json",
)

function load_frozen_estimator(est_dir::String)
    path = joinpath(est_dir, "best_config.json")
    data = JSON.parse(read(path, String))
    cfg = data["config"]
    est_name = cfg["estimator"]
    est_name == "eskf" || error("expected eskf")
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

function load_controller_kw(path::String)
    d = JSON.parse(read(path, String))
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

function run_one(ctrl::Symbol, tr, use_pose_fix::Bool; seed::Int=1)
    est_cfg = load_frozen_estimator(EST_DIR)
    kw = load_controller_kw(CTRL_CFG[ctrl])

    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = :eskf,
        use_dhat      = false,
        use_asmc      = ctrl == :asmc,
        use_mpc       = false,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = use_pose_fix,
        f_est         = est_cfg.rate_hz,
        sensor_seed   = seed,
    )

    prof  = TOML.parsefile(joinpath(tr.config_dir, "profiles", tr.profile_toml))["profile"]
    cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
    ref   = Profiles.build(prof["builder"], cfg_r)
    Profiles.publish!(ref)

    est = TuningHarnessMod._build_estimator(est_cfg)
    asmc_o, mpc_o, pid_o = build_controller(ctrl, kw)

    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=est, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref
end

function dump_csv(ctrl::Symbol, tr_name::String, use_pose_fix::Bool, outpath::String)
    tr = only(t for t in default_trajs_3("trajectory_files_run_0p5_main") if string(t.name) == tr_name)
    probe, ref = run_one(ctrl, tr, use_pose_fix)
    isempty(probe) && error("empty probe")

    rows = Matrix{Float64}(undef, length(probe), 12)
    for (i, p) in enumerate(probe)
        t = p.t
        rows[i, 1]  = t
        rows[i, 2]  = ref.Vx(t)
        rows[i, 3]  = ref.Vy(t)
        rows[i, 4]  = ref.psi(t)
        rows[i, 5]  = p.u[17]      # true X
        rows[i, 6]  = p.u[18]      # true Y
        rows[i, 7]  = p.u[4]       # true psi
        rows[i, 8]  = p.xhat[5]    # est X
        rows[i, 9]  = p.xhat[6]    # est Y
        rows[i, 10] = p.xhat[4]    # est psi
        rows[i, 11] = p.u[1]       # true Vx
        rows[i, 12] = p.u[2]       # true Vy
    end
    header = ["time", "ref_Vx", "ref_Vy", "ref_psi",
              "true_X", "true_Y", "true_psi",
              "est_X", "est_Y", "est_psi",
              "true_Vx", "true_Vy"]
    mkpath(dirname(outpath))
    open(outpath, "w") do io
        println(io, join(header, ','))
        writedlm(io, rows, ',')
    end
    println("wrote $(length(probe)) samples to $outpath")
end

ctrl = Symbol(ARGS[1])
tr_name = ARGS[2]
pose_fix = parse(Bool, ARGS[3])
out = ARGS[4]
dump_csv(ctrl, tr_name, pose_fix, out)
