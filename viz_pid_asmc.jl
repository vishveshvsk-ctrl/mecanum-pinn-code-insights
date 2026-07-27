#!/usr/bin/env julia
# Run tuned ASMC + PID on ellipse (pose) and octagon (velocity), dump decimated
# trajectories to CSV for plotting.
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
using StaticArrays, JSON, Random, LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

g_asmc = JSON.parse(read("runs_controller/asmc_clean/best_config.json", String))["best_gains"]
g_pid  = JSON.parse(read("runs_controller/pid_clean/best_config.json", String))["best_gains"]
V3(x) = SVector{3,Float64}(Float64.(x))

asmc = ControllerMod.ASMCController(
    gamma_x=g_asmc["gamma_x"], gamma_y=g_asmc["gamma_y"], gamma_psi=g_asmc["gamma_psi"],
    eps=g_asmc["eps"], eps_psi=g_asmc["eps_psi"],
    K_max_x=g_asmc["K_max_x"], K_max_y=g_asmc["K_max_y"], K_max_psi=g_asmc["K_max_psi"],
    rate_hz=1000.0)
pid = ControllerMod.PIDController(
    Kp=V3(g_pid["Kp"]), Ki=V3(g_pid["Ki"]), Kd=V3(g_pid["Kd"]),
    Kp_pos=V3(g_pid["Kp_pos"]), Kd_pos=V3(g_pid["Kd_pos"]), rate_hz=100.0)

const DIR = "trajectory_files_run_0p5_main"
const STRIDE = 20

function run_dump(cname, ctrl, tname, toml, mode, out)
    base = Profiles.load_base(DIR); chi = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=0.5)
    cfg = HybridConfig(tracking=mode, estimator=:oracle, use_dhat=false,
        use_asmc=(cname==:asmc), use_mpc=false, use_pid=(cname==:pid),
        fuzzy=false, fixed_weights=(cname==:asmc ? (1.0,0.0,0.0) : (0.0,0.0,1.0)),
        use_pose_fix=false, sensor_seed=42)
    ref = Profiles.pick_and_build(DIR, [toml]; rng=Random.Xoshiro(hash(toml)))[1]
    oracle = OracleEstimator(:clean; seed=42)
    a_o = cname==:asmc ? ctrl : nothing
    p_o = cname==:pid  ? ctrl : nothing
    sol, _df, bus = SchedulerMod.run_hybrid(cfg, params, Symbol(tname);
        chi=chi, config_dir=DIR, profile_toml=toml, return_bus=true, est=oracle, ref=ref,
        asmc_override=a_o, pid_override=p_o)
    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    open(out, "w") do io
        if mode == :pose
            println(io, "t,Xo,Yo,psi,xo_ref,yo_ref,psi_ref")
            for i in 1:STRIDE:length(probe)
                p = probe[i]; t = p.t
                println(io, join([t, p.u[17], p.u[18], p.u[4],
                                  ref.xo(t), ref.yo(t), ref.psi(t)], ","))
            end
        else
            println(io, "t,Xo,Yo,Vx,Vy,Vx_des,Vy_des")
            for i in 1:STRIDE:length(probe)
                p = probe[i]; t = p.t
                println(io, join([t, p.u[17], p.u[18], p.u[1], p.u[2],
                                  ref.Vx(t), ref.Vy(t)], ","))
            end
        end
    end
    println("wrote $out ($(cld(length(probe),STRIDE)) rows)")
end

mkpath("runs_controller/viz")
run_dump(:asmc, asmc, "ellipse", "ellipse_mu_0p5.toml", :pose, "runs_controller/viz/asmc_ellipse.csv")
run_dump(:pid,  pid,  "ellipse", "ellipse_mu_0p5.toml", :pose, "runs_controller/viz/pid_ellipse.csv")
run_dump(:asmc, asmc, "octagon", "octagon_mu_0p5.toml", :velocity, "runs_controller/viz/asmc_octagon.csv")
run_dump(:pid,  pid,  "octagon", "octagon_mu_0p5.toml", :velocity, "runs_controller/viz/pid_octagon.csv")
println("DONE")
