#!/usr/bin/env julia
# =============================================================================
# bound_analysis/export_truth_traces.jl
#
#   For each of the 10 trajectories selected by select_ellipse_combos.jl
#   (bound_analysis/reports/ellipse_selection.json), run TWO closed-loop
#   simulations at mu=0.5, chi=0.005 (trajectory_files_run_0p5_main/base.toml):
#
#     TRUTH pass    -- OracleEstimator(:clean), frozen ASMC, :pose tracking,
#                      no pose fix (oracle supplies pose directly). Provides
#                      the true state trajectory + a diagnostic IMU stream.
#     ACHIEVED pass -- frozen ESKF (runs_eskf_noellipse_v2/eskf_dxnes/
#                      best_config.json), :docking pose-fix tier, same
#                      controller/seed. Provides the achieved-estimate overlay
#                      and the measured slip signal.
#
#   Both resampled onto the uniform 500 Hz grid (saveat_hz=500, already the
#   HybridConfig default) and written to ONE Arrow file per trajectory under
#   bound_analysis/traces/.
#
#   FORMAT DEVIATION FROM THE BRIEF: instructions/pcrlb-no-odometry-ellipse-
#   bound.md §5/§6 specifies ".npz". This script writes ".arrow" instead,
#   because NPZ.jl is not a resolved dependency in this repo's Manifest.toml,
#   while Arrow.jl already is, AND Arrow is the established Julia->Python
#   convention this pipeline is explicitly modelled on (see
#   hybrid_ctrl/estimator_tuning/save_controller_eskf_traces.jl /
#   plot_controller_eskf_traces.py). bound_analysis/run_bound.py reads Arrow
#   via pyarrow (already in the claude-venv). This is a deliberate, documented
#   substitution, not an oversight -- see the recon report in this task's
#   history for the reasoning.
#
#   IMU STREAM CAVEAT: a_x/a_y/g_z are NOT consumed by the PCRLB recursion
#   itself (pcrlb.py's model works from psi/psidot/Vx/Vy analytically, with Q
#   DERIVED from the sensor spec -- see model.py). They are exported here as a
#   diagnostic/reference field only.
#
#   GRID CAVEAT: run_hybrid's solve() is called with `saveat=t_eval` but
#   PeriodicCallback's default save_positions=(true,true) ALSO stores a point
#   at every callback trigger (sensor/asmc/mixer/fuzzy/pose-fix), so the
#   returned `sol.t`/`sol.u` are NOT the intended uniform 500 Hz grid -- they
#   are a much denser, non-uniform union (confirmed empirically: ~223k points
#   for a 34 s trajectory, vs. the 16.9k a uniform 500 Hz grid would give).
#   This script therefore reconstructs t_eval exactly as run_hybrid does
#   internally and reads the state OFF THE SOLUTION OBJECT via `sol(t)` /
#   `sol(t, Val{1})` (continuous interpolation between the densely-recorded
#   points -- `sol.dense` is false here since `saveat` disables the true
#   Hermite dense output, but linear interpolation between ~6.6 kHz-recorded
#   points is more than accurate enough for a proper-acceleration estimate).
#   `sol(0.0, Val{1})` returns NaN at the exact left endpoint (boundary
#   artifact of the interpolant) -- guarded by nudging t=0 to a tiny epsilon.
#
#   Checkpointed: skips any trajectory whose output .arrow already exists.
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, ".."))
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

const SELECTION_PATH = joinpath(@__DIR__, "reports", "ellipse_selection.json")
const OUT_DIR         = joinpath(@__DIR__, "traces")
const DEFAULT_RUN_DIR = "trajectory_files_run_0p5_main"
const DEFAULT_PROFILE_TOML = "ellipse_mu_0p5.toml"
const DEFAULT_ASMC_CFG = "runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json"
const DEFAULT_ESKF_DIR = "runs_eskf_noellipse_v2/eskf_dxnes"
const SENSOR_SEED = 42

# -----------------------------------------------------------------------------
# Frozen config loaders (identical convention to save_controller_eskf_traces.jl
# / compare_controllers_eskf.jl -- kept local so this script has no dependency
# beyond tune_controller.jl + tuning/harness.jl).
# -----------------------------------------------------------------------------
function load_frozen_eskf(est_dir::String)
    path = joinpath(est_dir, "best_config.json")
    isfile(path) || error("Frozen ESKF config not found: $path")
    data = JSON.parse(read(path, String))
    cfg = data["config"]
    cfg["estimator"] == "eskf" || error("expected estimator=eskf in $path, got $(cfg["estimator"])")
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

function load_asmc_kw(path::String)
    isfile(path) || error("ASMC config not found: $path")
    d = JSON.parse(read(path, String))
    haskey(d, "best_gains") || error("no `best_gains` in $path")
    pairs = Pair{Symbol,Any}[]
    for (k, v) in d["best_gains"]
        push!(pairs, Symbol(k) => (v isa AbstractVector ? SVector{length(v)}(Float64.(v)) : Float64(v)))
    end
    return (; pairs...)
end

"Build the world PosRef for one ellipse combo. Xoshiro(0) convention matches
tune_controller.jl / save_controller_eskf_traces.jl for reproducible resolution."
function build_ellipse_ref(run_dir::String, profile_toml::String, combo_idx::Int)
    path = joinpath(run_dir, "profiles", profile_toml)
    prof = TOML.parsefile(path)["profile"]
    cfg  = Profiles.resolve_profile(prof; combo_idx=combo_idx, rng=Random.Xoshiro(0))
    ref  = Profiles.build(prof["builder"], cfg)
    ref isa Profiles.PosRef || error("ellipse builder did not return a PosRef (got $(typeof(ref)))")
    return ref
end

"Nearest-neighbour index in sorted `probe_t` for each element of sorted `grid_t`."
function nearest_indices(grid_t::Vector{Float64}, probe_t::Vector{Float64})
    idxs = Vector{Int}(undef, length(grid_t))
    j = 1
    np = length(probe_t)
    for (i, t) in enumerate(grid_t)
        while j < np && abs(probe_t[j+1] - t) <= abs(probe_t[j] - t)
            j += 1
        end
        idxs[i] = j
    end
    return idxs
end

# -----------------------------------------------------------------------------
function run_truth_pass(params, chi, ref, asmc_kw)
    cfg = HybridConfig(
        tracking      = :pose,
        estimator     = :oracle,
        use_dhat      = false,
        use_asmc      = true, use_mpc = false, use_pid = false,
        fuzzy         = false,
        fixed_weights = weights_for(:asmc),
        use_pose_fix  = false,          # oracle supplies pose directly
        sensor_seed   = SENSOR_SEED,
        saveat_hz     = 500.0,
    )
    oracle = OracleEstimator(:clean; seed=SENSOR_SEED)
    asmc_o, mpc_o, pid_o = build_controller(:asmc, asmc_kw)
    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, :ellipse;
        chi=chi, friction_case=1, config_dir=DEFAULT_RUN_DIR, profile_toml=DEFAULT_PROFILE_TOML,
        return_bus=true, est=oracle, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)
    return sol, bus
end

function run_achieved_pass(params, chi, ref, asmc_kw, est_cfg)
    cfg = HybridConfig(
        tracking      = :pose,
        estimator     = est_cfg.estimator,
        use_dhat      = get(est_cfg, :use_dhat, false),
        use_asmc      = true, use_mpc = false, use_pid = false,
        fuzzy         = false,
        fixed_weights = weights_for(:asmc),
        use_pose_fix  = true,
        pose_fix_tier = :docking,
        f_est         = est_cfg.rate_hz,
        sensor_seed   = SENSOR_SEED,
        saveat_hz     = 500.0,
    )
    est = TuningHarnessMod._build_estimator(est_cfg)
    asmc_o, mpc_o, pid_o = build_controller(:asmc, asmc_kw)
    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, :ellipse;
        chi=chi, friction_case=1, config_dir=DEFAULT_RUN_DIR, profile_toml=DEFAULT_PROFILE_TOML,
        return_bus=true, est=est, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)
    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return sol, bus, probe
end

# -----------------------------------------------------------------------------
"""
    assemble_trace(sol_truth, T_total, probe_eskf, params) -> DataFrame

Build the combined per-trajectory trace on the CANONICAL uniform 500 Hz grid
(reconstructed here, NOT read off sol_truth.t/.u -- see header GRID CAVEAT):
truth kinematics interpolated off the truth-pass solution object, a diagnostic
IMU stream synthesised from a fresh SensorModel driven by the solution's own
derivative interpolant, the measured slip indicator and achieved estimate
nearest-matched from the ESKF pass's ~1 kHz probe log.
"""
function assemble_trace(sol_truth, T_total::Float64, probe_eskf, params)
    N = round(Int, T_total * 500.0) + 1
    t = collect(range(0.0, T_total; length=N))

    Vx = Vector{Float64}(undef, N); Vy = similar(Vx)
    psidot = similar(Vx); psi = similar(Vx)
    X = similar(Vx); Y = similar(Vx)
    a_x = similar(Vx); a_y = similar(Vx); g_z = similar(Vx)

    sm = Main.SensorMod.SensorModel(:default; seed=SENSOR_SEED)
    eps0 = 1e-6   # guards sol(t, Val{1}) NaN at the exact left endpoint t=0
    for k in 1:N
        tk = t[k]
        u  = sol_truth(tk)
        du = sol_truth(k == 1 ? tk + eps0 : tk, Val{1})
        Vx[k], Vy[k], psidot[k], psi[k] = u[1], u[2], u[3], u[4]
        X[k], Y[k] = u[17], u[18]
        y = Main.SensorMod.simulate_measurement(u, du, sm, tk)
        a_x[k] = y.a_x; a_y[k] = y.a_y; g_z[k] = y.g_z
    end

    Hw = Main.EstimatorMod._wheel_jacobian(params)
    slip     = Vector{Float64}(undef, N)
    eskf_Vx  = similar(slip); eskf_Vy  = similar(slip); eskf_psi = similar(slip)
    eskf_X   = similar(slip); eskf_Y   = similar(slip)
    if isempty(probe_eskf)
        @warn "empty ESKF probe log — eskf_*/slip columns will be NaN"
        fill!(slip, NaN); fill!(eskf_Vx, NaN); fill!(eskf_Vy, NaN)
        fill!(eskf_psi, NaN); fill!(eskf_X, NaN); fill!(eskf_Y, NaN)
    else
        probe_t = Float64[p.t for p in probe_eskf]
        idxs = nearest_indices(collect(t), probe_t)
        for (k, j) in enumerate(idxs)
            pr = probe_eskf[j]
            wv = Hw \ SVector(pr.u[9], pr.u[10], pr.u[11], pr.u[12])
            slip[k]    = norm(SVector(wv[1], wv[2]) - SVector(pr.u[1], pr.u[2]))
            eskf_Vx[k]  = pr.xhat[1]; eskf_Vy[k]  = pr.xhat[2]
            eskf_psi[k] = pr.xhat[4]
            eskf_X[k]   = pr.xhat[5]; eskf_Y[k]   = pr.xhat[6]
        end
    end

    return DataFrame(
        t=t, psi=psi, psidot=psidot, Vx=Vx, Vy=Vy, X=X, Y=Y,
        a_x=a_x, a_y=a_y, g_z=g_z, slip=slip,
        eskf_Vx=eskf_Vx, eskf_Vy=eskf_Vy, eskf_psi=eskf_psi, eskf_X=eskf_X, eskf_Y=eskf_Y,
    )
end

# -----------------------------------------------------------------------------
function main()
    isfile(SELECTION_PATH) ||
        error("$SELECTION_PATH not found — run select_ellipse_combos.jl first.")
    selection = JSON.parse(read(SELECTION_PATH, String))
    combos = filter(c -> c["group"] != "unselected", selection["combos"])
    isempty(combos) && error("ellipse_selection.json has no selected combos.")

    run_dir = String(selection["run_dir"])
    mu = Float64(selection["mu"])

    mkpath(OUT_DIR)

    base = Profiles.load_base(run_dir)
    chi  = Float64(get(base, "physics", Dict())["chi"])
    params = PlatformParams(base; mu_friction=mu)

    asmc_kw = load_asmc_kw(DEFAULT_ASMC_CFG)
    est_cfg = load_frozen_eskf(DEFAULT_ESKF_DIR)

    println("mu=$mu  chi=$chi")
    println("Frozen ASMC: $DEFAULT_ASMC_CFG")
    println("Frozen ESKF: $DEFAULT_ESKF_DIR ($(est_cfg.rate_hz) Hz)")
    println("Trajectories: $(length(combos))")

    for c in sort(combos, by = c -> Int(c["combo_idx"]))
        combo_idx = Int(c["combo_idx"])
        group = c["group"]
        fname = "ellipse_c$(lpad(combo_idx, 3, '0')).arrow"
        outpath = joinpath(OUT_DIR, fname)
        if isfile(outpath)
            println("  [skip] combo $combo_idx ($group) — $fname already exists")
            continue
        end

        println("  [run]  combo $combo_idx ($group) a=$(round(c["a"],digits=2)) " *
                "worbit=$(round(c["worbit"],digits=3)) T=$(round(c["T_total"],digits=1))s")

        ref = build_ellipse_ref(run_dir, DEFAULT_PROFILE_TOML, combo_idx)
        Profiles.publish!(ref)

        sol_truth, _bus_truth = run_truth_pass(params, chi, ref, asmc_kw)
        _sol_eskf, _bus_eskf, probe_eskf = run_achieved_pass(params, chi, ref, asmc_kw, est_cfg)

        df = assemble_trace(sol_truth, ref.T_total, probe_eskf, params)
        Arrow.write(outpath, df; metadata = [
            "combo_idx" => string(combo_idx), "group" => group, "role" => c["role"],
            "a" => string(c["a"]), "ratio" => string(c["ratio"]), "worbit" => string(c["worbit"]),
            "u_peak" => string(c["u_peak"]), "v_peak" => string(c["v_peak"]),
            "T_total" => string(c["T_total"]), "mu" => string(mu), "chi" => string(chi),
        ])
        println("        -> $fname  (rows=$(nrow(df)))")
    end
    println("Truth-trace export complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
