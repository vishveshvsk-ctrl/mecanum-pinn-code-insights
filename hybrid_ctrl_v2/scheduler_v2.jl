# =============================================================================
# hybrid_ctrl_v2/scheduler_v2.jl — SchedulerModV2
# (instructions/sensors-suite-consolidation-and-physical-noise.md)
# =============================================================================
# `hybrid_ctrl/scheduler.jl` is NEVER edited. `run_hybrid`/`build_callbacks`
# hardcode a call to `Main.SensorMod.SensorModel(...)` and dispatch estimator
# construction on `cfg.estimator` internally — both only fixable by editing
# the function bodies, which the preservation constraint forbids. So
# `run_hybrid_v2`/`build_callbacks_v2` below are a faithful duplicate that (a)
# takes a `SensorModV2.SensorSuite` + a pre-built estimator directly instead
# of constructing them internally, and (b) adds a flow `PeriodicCallback`.
#
# Controller/fuzzy/mixer callbacks are UNCHANGED — reused via qualification
# (`Main.SchedulerMod._make_asmc_callback` etc.), not duplicated, since they
# don't touch sensors/estimators at all (`controllers.jl` is out of scope for
# this brief). `tune_controller.jl` / `hybrid_ctrl_v2`'s controller-tuning
# pipeline keeps using the ORIGINAL `run_hybrid` untouched — it only ever
# passes `OracleEstimator`, which never consumes a sensor suite anyway.
#
# Must be `include`d after `tune_controller.jl` (brings in SchedulerMod,
# BusMod, PlantMod, ControllerMod, MixerMod, FuzzyMod, Profiles, DataStore,
# HybridConfigMod, EstimatorMod) and after `sensors_v2.jl`/`estimators_v2.jl`.
# =============================================================================
module SchedulerModV2

using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks

export run_hybrid_v2, build_callbacks_v2

"""
    _make_sensor_callback_v2(est, suite, p, cfg)

Same role as `SchedulerMod._make_sensor_callback`, but sources the IMU/encoder
measurement from `SensorModV2.simulate_imu`/`simulate_encoders` (independent
per-modality RNGs) instead of the single-RNG `SensorMod.simulate_measurement`.
Assembles the SAME `y` NamedTuple shape (`θ`, `ω`, `a_x`, `a_y`, `g_z`) so
`EstimatorMod._wheel_body_velocity`/`slip_detect` (unchanged, reused as-is)
keep working unmodified.
"""
function _make_sensor_callback_v2(est, suite, p, cfg)
    return function (integrator)
        u = integrator.u
        du = similar(u)
        p_new = Main.PlantMod.PlantODEParams(p.params, p.chi, p.p1, p.p2,
                                              p.coupling, p.lugre, p.motor, p.bus)
        Main.PlantMod.plant_rhs!(du, u, p_new, integrator.t)

        a_x, a_y, g_z = Main.SensorModV2.simulate_imu(suite.imu, u, du, integrator.t)
        enc = Main.SensorModV2.simulate_encoders(suite.enc, u)
        y = (θ=enc.theta, ω=enc.omega, a_x=a_x, a_y=a_y, g_z=g_z)

        p.bus.y_last = y
        p.bus.t_now[] = integrator.t
        dt = 1.0 / cfg.f_est
        if est isa Main.EstimatorMod.OracleEstimator
            Main.EstimatorMod.oracle_feed!(p.bus, u, est, dt)
        else
            Main.EstimatorMod.estimator_update!(p.bus, y, est, p.params, dt)
        end

        probe = get!(Main.SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(p.bus)) do
            Vector{NamedTuple}()
        end
        push!(probe, (t=integrator.t, xhat=p.bus.xhat, d_hat=p.bus.d_hat,
                     v_cmd=p.bus.v_cmd, u=copy(integrator.u)))
        return nothing
    end
end

"""
    _make_flow_callback_v2(est, flow, p)

NEW callback (brief component 8): samples the optical-flow modality and
applies it via `EstimatorModV2.apply_flow!`. `nothing` (dropout/saturation)
means no update this tick -- never substitutes a zero.
"""
function _make_flow_callback_v2(est, flow, p)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        z_flow = Main.SensorModV2.simulate_flow(flow, integrator.u, integrator.t)
        if z_flow !== nothing
            Main.EstimatorModV2.apply_flow!(p.bus, est, flow, z_flow, p.params)
        end
        return nothing
    end
end

"""
    _make_pose_fix_callback_v2(fix, est, p)

Same role as `SchedulerMod._make_pose_fix_callback`, re-pointed to
`SensorModV2.PoseFixModel`/`sample_pose_fix` (brief: "pose-fix callback
re-pointed to SensorMod"). `apply_pose_fix!` dispatches on `est`'s type via
the SAME generic function as v1 (multiple dispatch), so this works
unmodified whether `est` is an original `EstimatorMod` estimator or the new
`ESKFEstimatorV2`.
"""
function _make_pose_fix_callback_v2(fix, est, p)
    return function (integrator)
        p.bus.t_now[] = integrator.t
        z_fix = Main.SensorModV2.sample_pose_fix(fix, integrator.u, integrator.t)
        if z_fix !== nothing
            Main.EstimatorMod.apply_pose_fix!(p.bus, est, fix, z_fix)
        end
        return nothing
    end
end

"""
    build_callbacks_v2(cfg, est, suite, asmc, mpc, pid, fz, ref, p, motor, T)

Same callback set as `SchedulerMod.build_callbacks`, plus a flow callback
when `suite.flow !== nothing`. Controller/fuzzy/mixer callbacks are the
UNCHANGED v1 closures (reused via qualification -- they never touch sensors).
"""
function build_callbacks_v2(cfg, est, suite, asmc, mpc, pid, fz, ref, p, motor, T)
    cbs = []

    push!(cbs, PeriodicCallback(_make_sensor_callback_v2(est, suite, p, cfg), 1.0/cfg.f_est; initial_affect=true))

    if cfg.use_pose_fix && suite.fix !== nothing
        push!(cbs, PeriodicCallback(_make_pose_fix_callback_v2(suite.fix, est, p), 1.0/suite.fix.fix_rate_hz; initial_affect=false))
    end

    if suite.flow !== nothing
        push!(cbs, PeriodicCallback(_make_flow_callback_v2(est, suite.flow, p), 1.0/suite.flow.rate_hz; initial_affect=false))
    end

    # Controllers/fuzzy/mixer: UNCHANGED v1 closures, out of scope for this brief.
    cfg.use_asmc && push!(cbs, PeriodicCallback(Main.SchedulerMod._make_asmc_callback(asmc, ref, p, cfg), 1.0/cfg.f_est; initial_affect=true))
    cfg.use_mpc  && push!(cbs, PeriodicCallback(Main.SchedulerMod._make_mpc_callback(mpc, ref, p, cfg),  1.0/cfg.f_mpc; initial_affect=false))
    cfg.use_pid  && push!(cbs, PeriodicCallback(Main.SchedulerMod._make_pid_callback(pid, ref, p, cfg),  1.0/cfg.f_pid; initial_affect=false))
    push!(cbs, PeriodicCallback(Main.SchedulerMod._make_fuzzy_callback(fz, ref, p, cfg), 1.0/cfg.f_fuzzy; initial_affect=true))
    push!(cbs, PeriodicCallback(Main.SchedulerMod._make_mixer_callback(p, motor, cfg), 1.0/cfg.f_mix; initial_affect=true))

    prog = Main.ProgressMeter.Progress(100; desc="hybrid run (v2)")
    last_pct = Ref(0)
    pbar = function (integrator)
        pct = clamp(floor(Int, 100 * integrator.t / T), 0, 100)
        if pct > last_pct[]
            Main.ProgressMeter.update!(prog, pct); last_pct[] = pct
        end
        return nothing
    end
    push!(cbs, PeriodicCallback(pbar, T/100; initial_affect=false))

    return CallbackSet(cbs...), prog
end

"""
    run_hybrid_v2(cfg, params, refname; chi, suite, est, friction_case=1, ...)

Duplicate of `SchedulerMod.run_hybrid`, parameterized on a `SensorSuite` +
pre-built estimator directly (see module header for why). `ref`/controller
overrides follow the same contract as `run_hybrid`.
"""
function run_hybrid_v2(cfg::Main.HybridConfigMod.HybridConfig,
                       params,
                       refname::Symbol;
                       chi::Real,
                       suite,
                       est,
                       friction_case::Int=1,
                       lugre=Main.LuGreParams(),
                       motor=Main.PlantMod.MotorParams(),
                       config_dir::AbstractString="trajectory_files_run_0p5_main",
                       profile_toml::Union{AbstractString,Nothing}=nothing,
                       return_bus::Bool=false,
                       ref=nothing,
                       asmc_override=nothing,
                       mpc_override=nothing,
                       pid_override=nothing)

    if ref === nothing
        profile_file = profile_toml === nothing ? string(refname)*".toml" : string(profile_toml)
        ref, _prof_cfg, _prof_name = Main.Profiles.pick_and_build(config_dir, [profile_file])
    end
    T = ref.T_total

    if cfg.tracking == :pose && !(ref isa Main.Profiles.PosRef)
        error("run_hybrid_v2: tracking=:pose requires a PosRef profile (got $(typeof(ref)))")
    end

    p1, p2 = friction_case == 1 ? (params.p1_case1, params.p2_case1) :
                                   (params.p1_case2, params.p2_case2)
    coupling = friction_case == 1 ? :adamov : :uncoupled

    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = cfg.use_dhat
    Main.SchedulerMod.clear_probe_log!(bus)
    plant_p = Main.PlantMod.PlantODEParams(params, chi, p1, p2, coupling, lugre, motor, bus)

    asmc = asmc_override === nothing ? Main.ControllerMod.ASMCController(rate_hz=cfg.use_asmc ? cfg.f_est : 1000.0) : asmc_override
    mpc  = mpc_override  === nothing ? Main.ControllerMod.MPCController(rate_hz=cfg.f_mpc) : mpc_override
    pid  = pid_override  === nothing ? Main.ControllerMod.PIDController() : pid_override
    fz   = Main.FuzzyMod.FuzzySupervisor(rate_hz=cfg.f_fuzzy)

    nstate = Main.PlantMod.nstate(motor)
    u0 = zeros(nstate)
    u0[4] = ref.psi(0.0)
    u0[5:8]  .= 0.1

    cbs, prog = build_callbacks_v2(cfg, est, suite, asmc, mpc, pid, fz, ref, plant_p, motor, T)

    solver = if cfg.solver_symbol == :FBDF
        FBDF()
    elseif cfg.solver_symbol == :TRBDF2
        TRBDF2()
    elseif cfg.solver_symbol == :RadauIIA5
        RadauIIA5()
    else
        error("Unsupported solver_symbol: $(cfg.solver_symbol)")
    end

    abstol = cfg.abstol === nothing ? Main.SchedulerMod._build_abstol(motor, cfg.abstol_bristle) : cfg.abstol

    t_eval = collect(range(0.0, T; length=round(Int, T * cfg.saveat_hz) + 1))
    prob = ODEProblem(Main.PlantMod.plant_rhs!, u0, (0.0, T), plant_p)
    sol = solve(prob, solver;
                reltol=cfg.reltol, abstol=abstol,
                saveat=t_eval, tstops=ref.tstops, callback=cbs,
                dtmax=cfg.dtmax, maxiters=cfg.maxiters)
    Main.ProgressMeter.finish!(prog)

    Main.SchedulerMod.LAST_PROBE_LOG[] = get(Main.SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    if return_bus
        return sol, bus
    else
        return sol
    end
end

end # module
