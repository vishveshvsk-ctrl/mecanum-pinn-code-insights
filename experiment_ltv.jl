#!/usr/bin/env julia
# Clean A/B: MPC-linear (use_ltv=false) vs MPC-LTV (use_ltv=true) at IDENTICAL
# gains, on the feasible trajectories — isolates the model effect from optimizer noise.
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

g = JSON.parse(read("runs_controller/mpc_clean/best_config.json", String))["best_gains"]  # linear-tuned gains
mkmpc(useltv) = ControllerMod.MPCController(
    Q      = SVector{3}(Float64.(g["Q"])),
    Q_pose = SVector{6}(Float64.(g["Q_pose"])),
    R      = SVector{4}(fill(Float64(g["R_scale"]), 4)),
    S      = SVector{4}(fill(Float64(g["S_scale"]), 4)),
    rate_hz = 100.0, use_ltv = useltv)

const DIR = "trajectory_files_run_0p5_main"
function build_ref(toml, cidx)
    prof = TOML.parsefile(joinpath(DIR, "profiles", toml))["profile"]
    cfg  = Profiles.resolve_profile(prof; combo_idx=cidx, rng=Random.Xoshiro(0))
    ref  = Profiles.build(prof["builder"], cfg); Profiles.publish!(ref); return ref
end

function run_mpc(mpc, toml, cidx, mode, name)
    base = Profiles.load_base(DIR); chi = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=0.5)
    ref = build_ref(toml, cidx)
    cfgh = HybridConfig(tracking=mode, estimator=:oracle, use_dhat=false,
        use_asmc=false, use_mpc=true, use_pid=false, fuzzy=false,
        fixed_weights=(0.0,1.0,0.0), use_pose_fix=false, sensor_seed=42)
    _s, _d, bus = SchedulerMod.run_hybrid(cfgh, params, Symbol(name);
        chi=chi, config_dir=DIR, profile_toml=toml, return_bus=true,
        est=OracleEstimator(:clean; seed=42), ref=ref, mpc_override=mpc)
    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    ce = mean(sum(abs.(Vector(p.v_cmd))) for p in probe)
    if mode == :pose
        perr = [sqrt((probe[i].u[17]-ref.xo(probe[i].t))^2 + (probe[i].u[18]-ref.yo(probe[i].t))^2) for i in eachindex(probe)]
        return "pos final=$(round(perr[end]*100,digits=2))cm max=$(round(maximum(perr)*100,digits=1))cm  ce=$(round(ce,digits=1))"
    else
        evx = [probe[i].u[1]-ref.Vx(probe[i].t) for i in eachindex(probe)]
        evy = [probe[i].u[2]-ref.Vy(probe[i].t) for i in eachindex(probe)]
        return "Vx_rms=$(round(sqrt(mean(evx.^2))*1e3,digits=1)) Vy_rms=$(round(sqrt(mean(evy.^2))*1e3,digits=1))mm/s  ce=$(round(ce,digits=1))"
    end
end

for (label, useltv) in [("LINEAR (frozen)", false), ("LTV (along ref)", true)]
    mpc = mkmpc(useltv)
    println("=== $label — same gains ===")
    println("  octagon: ", run_mpc(mpc, "octagon_mu_0p5.toml", 2, :velocity, "octagon"))
    println("  ellipse: ", run_mpc(mpc, "ellipse_mu_0p5.toml", 1, :pose, "ellipse"))
end
println("DONE")
