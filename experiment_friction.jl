#!/usr/bin/env julia
# Friction-circle feasibility test: run the TUNED ASMC across octagon cruise
# speeds (combos 1-4 = vcru 0.4/0.6/0.8/1.0) and feasible-vs-aggressive ellipse.
using Pkg; Pkg.activate(".")
include("run_one.jl"); using .Profiles, .DataStore
include("hybrid_ctrl/config.jl");      using .HybridConfigMod
include("hybrid_ctrl/bus.jl");         using .BusMod
include("hybrid_ctrl/plant.jl");       using .PlantMod
include("hybrid_ctrl/sensors.jl");     using .SensorMod
include("hybrid_ctrl/estimators.jl");  using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");       using .FuzzyMod
include("hybrid_ctrl/mixer.jl");       using .MixerMod
include("hybrid_ctrl/scheduler.jl");   using .SchedulerMod
using StaticArrays, JSON, Random, LinearAlgebra, TOML
using Statistics: mean
LinearAlgebra.BLAS.set_num_threads(1)

g = JSON.parse(read("runs_controller/asmc_clean/best_config.json", String))["best_gains"]
asmc = ControllerMod.ASMCController(
    gamma_x=g["gamma_x"], gamma_y=g["gamma_y"], gamma_psi=g["gamma_psi"],
    eps=g["eps"], eps_psi=g["eps_psi"],
    K_max_x=g["K_max_x"], K_max_y=g["K_max_y"], K_max_psi=g["K_max_psi"], rate_hz=1000.0)

const DIR = "trajectory_files_run_0p5_main"
const MU = 0.5
const A_FRIC = MU * 9.81   # friction-circle accel limit ≈ 4.9 m/s²

function build_ref(toml, cidx)
    path = joinpath(DIR, "profiles", toml)
    prof = TOML.parsefile(path)["profile"]
    cfg  = Profiles.resolve_profile(prof; combo_idx=cidx, rng=Random.Xoshiro(0))
    ref  = Profiles.build(prof["builder"], cfg)
    Profiles.publish!(ref)
    return ref, cfg
end

function run_asmc(toml, cidx, mode, name)
    base = Profiles.load_base(DIR); chi = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=MU)
    ref, cfg = build_ref(toml, cidx)
    cfgh = HybridConfig(tracking=mode, estimator=:oracle, use_dhat=false,
        use_asmc=true, use_mpc=false, use_pid=false, fuzzy=false,
        fixed_weights=(1.0,0.0,0.0), use_pose_fix=false, sensor_seed=42)
    oracle = OracleEstimator(:clean; seed=42)
    _sol, _df, bus = SchedulerMod.run_hybrid(cfgh, params, Symbol(name);
        chi=chi, config_dir=DIR, profile_toml=toml, return_bus=true, est=oracle, ref=ref,
        asmc_override=asmc)
    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, cfg
end

println("friction-circle accel limit ≈ $(round(A_FRIC,digits=2)) m/s²  (μ=$MU)\n")
println("=== OCTAGON — cruise-speed sweep (tuned ASMC) ===")
for cidx in [1, 2, 3, 4]
    probe, ref, cfg = run_asmc("octagon_mu_0p5.toml", cidx, :velocity, "octagon")
    ts = [p.t for p in probe]
    vxd = [ref.Vx(t) for t in ts]; vyd = [ref.Vy(t) for t in ts]
    axd = [ref.Ax(t) for t in ts]; ayd = [ref.Ay(t) for t in ts]
    evx = [probe[i].u[1] - vxd[i] for i in eachindex(probe)]
    evy = [probe[i].u[2] - vyd[i] for i in eachindex(probe)]
    accpk = maximum(sqrt.(axd.^2 .+ ayd.^2))
    println("  vcru=$(cfg["vcru"]) m/s: peak|Vy_des|=$(round(maximum(abs,vyd),digits=2)) m/s " *
            "peak_accel=$(round(accpk,digits=2)) m/s² | " *
            "Vx_rms=$(round(sqrt(mean(evx.^2))*1e3,digits=1)) Vy_rms=$(round(sqrt(mean(evy.^2))*1e3,digits=1)) mm/s")
end

println("\n=== ELLIPSE — feasible (a=0.8) vs aggressive (a=4.0), tuned ASMC ===")
for cidx in [1, 55]
    probe, ref, cfg = run_asmc("ellipse_mu_0p5.toml", cidx, :pose, "ellipse")
    ts = [p.t for p in probe]
    sp = [sqrt(ref.Vxo(t)^2 + ref.Vyo(t)^2) for t in ts]
    acc = [sqrt(ref.Axo(t)^2 + ref.Ayo(t)^2) for t in ts]
    perr = [sqrt((probe[i].u[17]-ref.xo(ts[i]))^2 + (probe[i].u[18]-ref.yo(ts[i]))^2) for i in eachindex(probe)]
    println("  a=$(cfg["a"]) worbit=$(round(cfg["worbit"],digits=2)): peak_speed=$(round(maximum(sp),digits=2)) m/s " *
            "peak_accel=$(round(maximum(acc),digits=2)) m/s² | " *
            "pos_err final=$(round(perr[end]*100,digits=3)) cm max=$(round(maximum(perr)*100,digits=2)) cm")
end
println("\nDONE")
