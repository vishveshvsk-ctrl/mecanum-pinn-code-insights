# =============================================================================
# hybrid_ctrl/scheduler.jl — callbacks, run_hybrid, logging, save
# =============================================================================
module SchedulerMod

using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks
using DataFrames
using ProgressMeter

export build_callbacks, run_hybrid, log_run, save_run

function _make_sensor_callback(est, sm, p, cfg)
    return function (integrator)
        u = integrator.u
        du = similar(u)
        p_new = Main.PlantMod.PlantODEParams(p.params, p.chi, p.p1, p.p2,
                                              p.coupling, p.lugre, p.motor,
                                              p.bus)
        Main.PlantMod.plant_rhs!(du, u, p_new, integrator.t)
        y = Main.SensorMod.simulate_measurement(u, du, sm, integrator.t)
        p.bus.y_last = y
        p.bus.t_now[] = integrator.t
        dt = 1.0 / cfg.f_est
        Main.EstimatorMod.estimator_update!(p.bus, y, est, p.params, dt)
        return nothing
    end
end

function _make_asmc_callback(asmc, ref, p, cfg)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        W = Main.ControllerMod.asmc_wrench!(p.bus, p.bus.xhat, ref, p.params,
                                            asmc, 1.0/cfg.f_est; mode=cfg.tracking)
        p.bus.W_asmc = W
        return nothing
    end
end

function _make_mpc_callback(mpc, ref, p, cfg)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        W = Main.ControllerMod.mpc_wrench!(p.bus, p.bus.xhat, ref, p.params, mpc)
        p.bus.W_mpc = W
        return nothing
    end
end

function _make_pid_callback(pid, ref, p, cfg)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        W = Main.ControllerMod.pid_wrench!(p.bus, p.bus.xhat, ref, pid, 1.0/cfg.f_pid)
        p.bus.W_pid = W
        return nothing
    end
end

function _make_fuzzy_callback(fz, ref, p, cfg)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        Main.FuzzyMod.fuzzy_update!(p.bus, p.bus.xhat, ref, cfg, fz)
        return nothing
    end
end

function _make_mixer_callback(p, motor, cfg)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        Main.MixerMod.mix_and_allocate!(p.bus, p.params, motor, cfg)
        return nothing
    end
end

function build_callbacks(cfg, est, sm, asmc, mpc, pid, fz, ref, p, motor, T)
    cbs = []

    # Sensor + estimator at f_est
    push!(cbs, PeriodicCallback(_make_sensor_callback(est, sm, p, cfg), 1.0/cfg.f_est; initial_affect=true))

    # Controllers
    cfg.use_asmc && push!(cbs, PeriodicCallback(_make_asmc_callback(asmc, ref, p, cfg), 1.0/cfg.f_est; initial_affect=true))
    cfg.use_mpc  && push!(cbs, PeriodicCallback(_make_mpc_callback(mpc, ref, p, cfg),  1.0/cfg.f_mpc; initial_affect=false))
    cfg.use_pid  && push!(cbs, PeriodicCallback(_make_pid_callback(pid, ref, p, cfg),  1.0/cfg.f_pid; initial_affect=false))

    # Fuzzy supervisor (also needed when fuzzy=false to apply fixed_weights)
    push!(cbs, PeriodicCallback(_make_fuzzy_callback(fz, ref, p, cfg), 1.0/cfg.f_fuzzy; initial_affect=true))

    # Mixer: must run at least as fast as fastest controller
    push!(cbs, PeriodicCallback(_make_mixer_callback(p, motor, cfg), 1.0/cfg.f_mix; initial_affect=true))

    # Progress bar
    prog = Progress(100; desc="hybrid run")
    last_pct = Ref(0)
    pbar = function (integrator)
        pct = clamp(floor(Int, 100 * integrator.t / T), 0, 100)
        if pct > last_pct[]
            update!(prog, pct); last_pct[] = pct
        end
        return nothing
    end
    push!(cbs, PeriodicCallback(pbar, T/100; initial_affect=false))

    return CallbackSet(cbs...), prog
end

"Build absolute tolerance vector for the plant state dimension."
function _build_abstol(motor::Main.PlantMod.MotorParams, bristle_tol::Real)
    n = Main.PlantMod.nstate(motor)
    at = fill(1e-7, n)
    at[1:3]   .= 1e-8    # body velocities
    at[4]      = 1e-7    # psi
    at[5:8]   .= 1e-8    # theta
    at[9:12]  .= 1e-7    # omega
    at[13:16] .= 1e-7    # gamma
    at[17:18] .= 1e-7    # Xo,Yo
    at[19:26] .= bristle_tol
    at[27:30] .= 1e-7
    if n == 34
        at[31:34] .= 1e-6
    end
    return at
end

"""
    run_hybrid(cfg, params, refname; chi, friction_case, lugre,
               config_dir, outdir=nothing)

Build plant, install config-gated PeriodicCallbacks, solve, return `(sol, log_df)`.
"""
function run_hybrid(cfg::Main.HybridConfigMod.HybridConfig,
                    params::Main.PlatformParams,
                    refname::Symbol;
                    chi::Real,
                    friction_case::Int=1,
                    lugre::Main.LuGreParams=Main.LuGreParams(),
                    motor::Main.PlantMod.MotorParams=Main.PlantMod.MotorParams(),
                    config_dir::AbstractString="trajectory_files_run_0p5_main",
                    outdir::Union{AbstractString,Nothing}=nothing)

    ref, _prof_cfg, _prof_name = Main.Profiles.pick_and_build(config_dir, [string(refname)*".toml"])
    T = ref.T_total

    p1, p2 = friction_case == 1 ? (params.p1_case1, params.p2_case1) :
                                   (params.p1_case2, params.p2_case2)
    coupling = friction_case == 1 ? :adamov : :uncoupled

    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = cfg.use_dhat
    plant_p = Main.PlantMod.PlantODEParams(params, chi, p1, p2, coupling, lugre, motor, bus)

    # Instantiate blocks
    sm   = Main.SensorMod.SensorModel(seed=cfg.sensor_seed)
    est  = cfg.estimator == :kalman ? Main.EstimatorMod.KalmanEstimator(rate_hz=cfg.f_est, use_dhat=cfg.use_dhat) :
           cfg.estimator == :smo    ? Main.EstimatorMod.SMOEstimator(rate_hz=cfg.f_est, use_dhat=cfg.use_dhat) :
           error("Unsupported estimator: $(cfg.estimator)")
    asmc = Main.ControllerMod.ASMCController(rate_hz=cfg.use_asmc ? cfg.f_est : 1000.0)
    mpc  = Main.ControllerMod.MPCController(rate_hz=cfg.f_mpc)
    pid  = Main.ControllerMod.PIDController()
    fz   = Main.FuzzyMod.FuzzySupervisor(rate_hz=cfg.f_fuzzy)

    # Initial state
    nstate = Main.PlantMod.nstate(motor)
    u0 = zeros(nstate)
    u0[1:3] .= 0.0
    u0[4] = ref.psi(0.0)
    u0[5:8]  .= 0.1
    u0[9:12] .= 0.0
    u0[13:16].= 0.0

    cbs, prog = build_callbacks(cfg, est, sm, asmc, mpc, pid, fz, ref, plant_p, motor, T)

    solver = cfg.solver_symbol == :TRBDF2 ? TRBDF2() :
             cfg.solver_symbol == :RadauIIA5 ? RadauIIA5() : TRBDF2()

    t_eval = collect(range(0.0, T; length=round(Int, T * cfg.saveat_hz) + 1))
    prob = ODEProblem(Main.PlantMod.plant_rhs!, u0, (0.0, T), plant_p)
    sol = solve(prob, solver;
                reltol=cfg.reltol, abstol=_build_abstol(motor, cfg.abstol_bristle),
                saveat=t_eval, tstops=ref.tstops, callback=cbs,
                dtmax=cfg.dtmax, maxiters=10^7)
    finish!(prog)

    df = log_run(sol, plant_p, cfg, ref, motor)
    return sol, df
end

"""
    log_run(sol, p, cfg, ref, motor)

Assemble a DataFrame of true state, estimate, errors, wrenches, weights, and
applied wheel torques.  Columns are additive to the existing DataStore schema.
"""
function log_run(sol, p, cfg, ref, motor)
    N = length(sol.t)
    st = hcat(sol.u...)
    df = DataFrame()
    df.time = sol.t

    # True state
    df.Vx, df.Vy, df.psi_dot, df.psi = st[1,:], st[2,:], st[3,:], st[4,:]
    for i in 1:4
        df[!, Symbol("theta$i")] = st[4+i,:]
        df[!, Symbol("w$i")]     = st[8+i,:]
        df[!, Symbol("gamma$i")] = st[12+i,:]
    end
    df.Xo, df.Yo = st[17,:], st[18,:]
    for i in 1:4
        df[!, Symbol("zx_$i")] = st[18+i,:]
        df[!, Symbol("zy_$i")] = st[22+i,:]
        df[!, Symbol("zs_$i")] = st[26+i,:]
    end

    # Reference (velocity mode; extend for pose if needed)
    df.Vx_des    = [ref.Vx(t)  for t in sol.t]
    df.Vy_des    = [ref.Vy(t)  for t in sol.t]
    df.psi_des   = [ref.psi(t) for t in sol.t]
    df.omega_des = [ref.Wz(t)  for t in sol.t]

    # Recompute applied torque at each saved time by re-evaluating RHS torque path
    tau_log = zeros(4, N)
    for k in 1:N
        t = sol.t[k]; u = sol.u[k]
        p.bus.t_now[] = t
        wi = SVector(u[9], u[10], u[11], u[12])
        tau_log[:,k] .= Main.PlantMod.motor_torque(p.bus.v_cmd, wi, motor)
    end
    for i in 1:4
        df[!, Symbol("tau_wheel_$i")] = tau_log[i,:]
        df[!, Symbol("v_cmd_$i")]     = fill(p.bus.v_cmd[i], N)
    end

    # Errors (true vs reference)
    df.e_Vx = df.Vx .- df.Vx_des
    df.e_Vy = df.Vy .- df.Vy_des
    df.e_omega = df.psi_dot .- df.omega_des

    # Final weights / wrenches (constant between mixer ticks; log final bus values)
    df.w_asmc = fill(p.bus.weights[1], N)
    df.w_mpc  = fill(p.bus.weights[2], N)
    df.w_pid  = fill(p.bus.weights[3], N)
    df.Wx_asmc = fill(p.bus.W_asmc[1], N); df.Wy_asmc = fill(p.bus.W_asmc[2], N); df.Wpsi_asmc = fill(p.bus.W_asmc[3], N)
    df.Wx_mpc  = fill(p.bus.W_mpc[1],  N); df.Wy_mpc  = fill(p.bus.W_mpc[2],  N); df.Wpsi_mpc  = fill(p.bus.W_mpc[3],  N)
    df.Wx_pid  = fill(p.bus.W_pid[1],  N); df.Wy_pid  = fill(p.bus.W_pid[2],  N); df.Wpsi_pid  = fill(p.bus.W_pid[3],  N)

    return df
end

"""
    save_run(df, sol, outdir, meta)

Persist using DataStore.write_outputs if `outdir` is given; otherwise return
paths=nothing.
"""
function save_run(df, sol, outdir, meta; cfg=nothing)
    paths = Main.DataStore.write_outputs(df, sol, nothing, nothing, nothing,
                                         meta; outdir=outdir, cfg=cfg, write_jld2=false)
    return paths
end

end # module
