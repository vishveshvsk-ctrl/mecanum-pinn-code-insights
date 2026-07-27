#!/usr/bin/env julia
# =============================================================================
# compare_controllers_eskf.jl
#   ASMC vs PID controller comparison on a single FROZEN ESKF estimator.
#
# Implements: instructions/controller-comparison-frozen-eskf.md
#
# The culmination of the estimator-first methodology: the controllers were tuned
# against an OracleEstimator (true state + noise); here they are re-evaluated
# closed-loop when fed a REAL, imperfect frozen-ESKF estimate. Two ellipse-
# excluded subset variants are run — with and without `coupled_vomega` — so the
# controller ranking's sensitivity to the coupled-dynamics case can be checked.
#
# This is a THIN driver: it composes existing, validated halves —
#   * load_frozen_estimator + _build_estimator   (frozen ESKF, per-run fresh)
#   * build_controller + *_override kwargs        (varying controller)
#   * run_hybrid                                  (closed loop)
#   * controller_metrics                          (scoring on the TRUE state)
# No physics / controller / estimator logic is rewritten.
# =============================================================================

using Pkg
Pkg.activate(".")

using LinearAlgebra
# Determinism: multithreaded BLAS gives run-to-run FP variation (reduction
# order) that a chaotically-sensitive closed loop amplifies. Pin to 1 thread.
LinearAlgebra.BLAS.set_num_threads(1)

# Reuse the tuning stack. tune_controller.jl is main-guarded, so `include` pulls
# in run_one.jl + all hybrid_ctrl modules + build_controller / controller_metrics
# / default_trajs_3 / weights_for / _wrapang / fmt_abs WITHOUT running its main().
include("tune_controller.jl")
# _build_estimator (fresh ESKF struct from the frozen config NamedTuple).
include("tuning/harness.jl"); using .TuningHarnessMod

using StaticArrays
using Statistics
using Random
using TOML
using JSON
using DataFrames
using CSV
using Arrow

const USAGE = """
Usage: compare_controllers_eskf.jl [options]

Compare controllers closed-loop on a frozen ESKF estimator, on two ellipse-
excluded trajectory-subset variants (with / without coupled_vomega).

Options:
  --estimator-dir DIR   Dir holding the frozen ESKF best_config.json
                        (default: runs_eskf_noellipse_v2/eskf_dxnes)
  --controllers SPEC    Comma list of ctrl:path pairs
                        (default: asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,
                                  pid:runs_controller_pid_5seed/pid_FINAL_seed2.json)
  --run-dir DIR         Trajectory config dir (default: trajectory_files_run_0p5_main)
  --seeds SEEDS         Comma-separated sensor-noise seeds (default: 1)
  --out DIR             Output root (default: runs_controller_compare_eskf)
  --smoke               One ASMC/octagon run through the frozen ESKF, then exit
  --help                Show this message
"""

const FINALIZED = Dict(:asmc => true, :pid => true)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
function parse_args(argv)
    a = Dict{String,Any}(
        "estimator-dir" => "runs_eskf_noellipse_v2/eskf_dxnes",
        "controllers"   => "asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json," *
                           "pid:runs_controller_pid_5seed/pid_FINAL_seed2.json",
        "run-dir"       => "trajectory_files_run_0p5_main",
        "seeds"         => [1],
        "out"           => "runs_controller_compare_eskf",
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
        elseif arg == "--out";           a["out"]           = argv[i+1]; i += 2
        elseif arg == "--smoke";         a["smoke"] = true; i += 1
        else; error("unknown arg $arg\n$USAGE"); end
    end
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

# -----------------------------------------------------------------------------
# load_frozen_estimator — read best_config.json → estimator config NamedTuple.
# (Mirrors compare_estimators.jl; that file auto-runs main() on include, so the
# :eskf branch is replicated here rather than included.)
# -----------------------------------------------------------------------------
function load_frozen_estimator(est_dir::String)
    path = joinpath(est_dir, "best_config.json")
    isfile(path) || error("Frozen estimator config not found: $path")
    data = JSON.parse(read(path, String))
    cfg = data["config"]
    est_name = cfg["estimator"]
    if est_name == "eskf"
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
    else
        error("compare_controllers_eskf expects a frozen ESKF config; got estimator=$est_name")
    end
end

# -----------------------------------------------------------------------------
# load_controller_kw — read a controller best_config.json → kwargs NamedTuple.
# List-valued gains (Kp[3], Q_pose[6], ...) are reconstructed as SVectors so
# build_controller can splat them straight into the controller struct. ASMC
# already folds pinned K_max into best_gains, so no injection is needed.
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# run_controller_on_estimator — the one genuinely new run path.
# Mirrors tune_controller.jl::run_controller but injects a FRESH frozen estimator
# instead of an OracleEstimator, and sets estimator=:eskf in the HybridConfig.
# Everything else (controller build, ref build, solver/tolerances) matches the
# tuning conditions so the only change vs tuning is oracle -> real ESKF.
# -----------------------------------------------------------------------------
function run_controller_on_estimator(ctrl::Symbol, kw::NamedTuple, est_cfg, tr; seed::Int=42)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = est_cfg.estimator,            # :eskf
        use_dhat      = get(est_cfg, :use_dhat, false),
        use_asmc      = ctrl == :asmc,
        use_mpc       = ctrl == :mpc,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = true,         # simulation pose is available even for velref tracking
        f_est         = est_cfg.rate_hz,
        sensor_seed   = seed,
    )

    # Deterministic ref: pinned combo_idx + Xoshiro(0) (all subset entries carry
    # combo_idx); the global-RNG pick_and_build path is non-deterministic.
    if get(tr, :combo_idx, nothing) === nothing
        ref = Profiles.pick_and_build(tr.config_dir, [tr.profile_toml];
                                      rng=Random.Xoshiro(hash(tr.profile_toml)))[1]
    else
        prof  = TOML.parsefile(joinpath(tr.config_dir, "profiles", tr.profile_toml))["profile"]
        cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
        ref   = Profiles.build(prof["builder"], cfg_r)
        Profiles.publish!(ref)
    end

    est = TuningHarnessMod._build_estimator(est_cfg)   # FRESH per (traj, seed) — no state leak
    asmc_o, mpc_o, pid_o = build_controller(ctrl, kw)

    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=est, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, tr.run_mode
end

# -----------------------------------------------------------------------------
# estimator_error — per-channel RMSE of xhat vs the TRUE plant state.
# Context metric only (ESKF quality is closed-loop-coupled, so it varies by
# controller). xhat = [Vx, Vy, ψ̇, X, Y, ψ]; true state u has Vx,Vy,ψ̇=u[1:3],
# ψ=u[4], X=u[17], Y=u[18].  Units are SI (m/s, rad/s, m, rad).
# -----------------------------------------------------------------------------
function estimator_error(probe)
    isempty(probe) && return (vx=NaN, vy=NaN, psidot=NaN, psi=NaN, X=NaN, Y=NaN)
    rmse(h, t) = sqrt(mean((h .- t) .^ 2))
    # ESKF bus.xhat layout: [Vx, Vy, ψ̇, ψ, X, Y]
    tvx  = [p.u[1]  for p in probe]; hvx  = [p.xhat[1] for p in probe]
    tvy  = [p.u[2]  for p in probe]; hvy  = [p.xhat[2] for p in probe]
    tpd  = [p.u[3]  for p in probe]; hpd  = [p.xhat[3] for p in probe]
    tpsi = [p.u[4]  for p in probe]; hpsi = [p.xhat[4] for p in probe]
    tX   = [p.u[17] for p in probe]; hX   = [p.xhat[5] for p in probe]
    tY   = [p.u[18] for p in probe]; hY   = [p.xhat[6] for p in probe]
    # Heading: wrap the error onto (-π, π] before the RMS.
    psi_err = _wrapang.(hpsi .- tpsi)
    return (
        vx     = rmse(hvx, tvx),
        vy     = rmse(hvy, tvy),
        psidot = rmse(hpd, tpd),
        psi    = sqrt(mean(psi_err .^ 2)),
        X      = rmse(hX, tX),
        Y      = rmse(hY, tY),
    )
end

# -----------------------------------------------------------------------------
# build_subset_variants — two ellipse-excluded velref subsets from default_trajs_3.
#   with_coupled = [octagon, spin_creep, coupled_vomega, spiral_orbit]
#   no_coupled   = [octagon, spin_creep, spiral_orbit]
# (default_trajs_3 also contains the two ellipse posref entries — excluded here.)
# -----------------------------------------------------------------------------
function build_subset_variants(run_dir::String)
    base   = default_trajs_3(run_dir)
    byname = Dict(string(t.name) => t for t in base)
    getn(n) = haskey(byname, n) ? byname[n] : error("expected trajectory '$n' in default_trajs_3")
    with_coupled = [getn("octagon"), getn("spin_creep"), getn("coupled_vomega"), getn("spiral_orbit")]
    no_coupled   = [getn("octagon"), getn("spin_creep"), getn("spiral_orbit")]
    return with_coupled, no_coupled
end

# -----------------------------------------------------------------------------
# Output: one tidy row per (controller, trajectory, seed).
# -----------------------------------------------------------------------------
function run_row(ctrl::Symbol, kw::NamedTuple, est_cfg, tr, seed::Int)
    probe, ref, mode = run_controller_on_estimator(ctrl, kw, est_cfg, tr; seed=seed)
    m = controller_metrics(probe, ref, mode)
    e = estimator_error(probe)
    a = m.abs
    return (
        controller     = string(ctrl),
        finalized      = get(FINALIZED, ctrl, true),
        trajectory     = string(tr.name),
        mode           = string(mode),
        seed           = seed,
        ok             = m.ok,
        tracking       = m.tracking,
        ce             = m.ce,
        chatter        = m.chatter,
        est_rmse_vx     = e.vx,
        est_rmse_vy     = e.vy,
        est_rmse_psidot = e.psidot,
        est_rmse_psi    = e.psi,
        est_rmse_X      = e.X,
        est_rmse_Y      = e.Y,
        abs_rms_vx     = get(a, :rms_vx, NaN),
        abs_rms_vy     = get(a, :rms_vy, NaN),
        abs_rms_w      = get(a, :rms_w,  NaN),
        abs_max_vx     = get(a, :max_vx, NaN),
        abs_max_vy     = get(a, :max_vy, NaN),
        abs_max_w      = get(a, :max_w,  NaN),
    )
end

_skipnan_mean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
_skipnan_std(x)  = (v = filter(isfinite, x); length(v) < 2 ? NaN : std(v))

"mean±std grouped by [controller, trajectory]."
function summarize(df::DataFrame)
    combine(groupby(df, [:controller, :trajectory]),
        :tracking         => _skipnan_mean => :tracking_mean,
        :tracking         => _skipnan_std  => :tracking_std,
        :ce               => _skipnan_mean => :ce_mean,
        :ce               => _skipnan_std  => :ce_std,
        :chatter          => _skipnan_mean => :chatter_mean,
        :chatter          => _skipnan_std  => :chatter_std,
        :est_rmse_vx     => _skipnan_mean => :est_rmse_vx_mean,
        :est_rmse_vy     => _skipnan_mean => :est_rmse_vy_mean,
        :est_rmse_psidot => _skipnan_mean => :est_rmse_psidot_mean,
        :est_rmse_psi    => _skipnan_mean => :est_rmse_psi_mean,
        :est_rmse_X      => _skipnan_mean => :est_rmse_X_mean,
        :est_rmse_Y      => _skipnan_mean => :est_rmse_Y_mean,
        nrow              => :n,
    )
end

"Overall controller ranking for a variant: mean tracking across its trajectories."
function rank_controllers(summary::DataFrame)
    r = combine(groupby(summary, :controller),
                :tracking_mean => mean => :tracking_mean_over_trajs)
    sort!(r, :tracking_mean_over_trajs)
    return r
end

function run_variant(name::String, trajs, ctrl_kw, est_cfg, seeds, outroot::String)
    println("\n===== VARIANT: $name  ($(length(trajs)) trajs: " *
            "$(join([string(t.name) for t in trajs], ", "))) =====")
    rows = NamedTuple[]
    for (ctrl, kw) in ctrl_kw
        for tr in trajs
            tr.run_mode == :velocity || @warn "expected velocity mode for $(tr.name); got $(tr.run_mode)"
            for seed in seeds
                row = try
                    run_row(ctrl, kw, est_cfg, tr, seed)
                catch err
                    @warn "run failed" controller=ctrl traj=tr.name seed=seed exception=err
                    (controller=string(ctrl), finalized=get(FINALIZED, ctrl, true),
                     trajectory=string(tr.name), mode=string(tr.run_mode), seed=seed,
                     ok=false, tracking=NaN, ce=NaN, chatter=NaN,
                     est_rmse_vx=NaN, est_rmse_vy=NaN, est_rmse_psidot=NaN,
                     est_rmse_psi=NaN, est_rmse_X=NaN, est_rmse_Y=NaN,
                     abs_rms_vx=NaN, abs_rms_vy=NaN, abs_rms_w=NaN,
                     abs_max_vx=NaN, abs_max_vy=NaN, abs_max_w=NaN)
                end
                push!(rows, row)
                println("  [$name] $(rpad(string(ctrl),4)) $(rpad(string(tr.name),16)) " *
                        "seed=$(lpad(seed,2)) : track=$(round(row.tracking,digits=3)) " *
                        "ce=$(round(row.ce,digits=2)) est_vx=$(round(row.est_rmse_vx,digits=3))")
            end
        end
    end

    df = DataFrame(rows)
    outdir = joinpath(outroot, name)
    mkpath(outdir)
    CSV.write(joinpath(outdir, "runs.csv"), df)
    Arrow.write(joinpath(outdir, "runs.arrow"), df)

    summary = summarize(df)
    CSV.write(joinpath(outdir, "summary.csv"), summary)
    Arrow.write(joinpath(outdir, "summary.arrow"), summary)

    ranking = rank_controllers(summary)
    println("\n  --- $name ranking (mean tracking over trajectories, lower=better) ---")
    for r in eachrow(ranking)
        flag = get(FINALIZED, Symbol(r.controller), true) ? "" : "  (NOT finalized — coarse MPC)"
        println("    $(rpad(r.controller,5)) $(round(r.tracking_mean_over_trajs, digits=3))$flag")
    end
    return df, summary, ranking
end

# -----------------------------------------------------------------------------
function main()
    a = parse_args(ARGS)
    est_cfg = load_frozen_estimator(a["estimator-dir"])
    println("Frozen estimator: $(est_cfg.estimator) @ $(est_cfg.rate_hz) Hz  ($(a["estimator-dir"]))")

    ctrl_specs = parse_controllers(a["controllers"])
    ctrl_kw = [(ctrl, load_controller_kw(path, ctrl)) for (ctrl, path) in ctrl_specs]
    for ((ctrl, _), (_, path)) in zip(ctrl_kw, ctrl_specs)
        fin = get(FINALIZED, ctrl, true) ? "finalized" : "NOT finalized (coarse)"
        println("Controller: $(rpad(string(ctrl),4)) [$fin]  ($path)")
    end

    if a["smoke"]
        println("\n=== SMOKE: ASMC on octagon through the frozen ESKF ===")
        with_coupled, _ = build_subset_variants(a["run-dir"])
        octagon = with_coupled[1]
        asmc_kw = first(kw for (c, kw) in ctrl_kw if c == :asmc)
        probe, ref, mode = run_controller_on_estimator(:asmc, asmc_kw, est_cfg, octagon; seed=1)
        m = controller_metrics(probe, ref, mode)
        e = estimator_error(probe)
        println("  ticks=$(length(probe))  mode=$mode")
        println("  tracking=$(round(m.tracking,digits=4))  ce=$(round(m.ce,digits=3))  chatter=$(round(m.chatter,digits=4))  ok=$(m.ok)")
        println("  est_rmse: vx=$(round(e.vx,digits=3)) vy=$(round(e.vy,digits=3)) ψ̇=$(round(e.psidot,digits=3)) " *
                "ψ=$(round(e.psi,digits=3)) X=$(round(e.X,digits=3)) Y=$(round(e.Y,digits=3))")
        # Confirm the ESKF branch (real estimator) ran: xhat must differ from true u.
        if !isempty(probe)
            dmax = maximum(abs(probe[k].xhat[1] - probe[k].u[1]) for k in eachindex(probe))
            println("  max|x̂_Vx - Vx_true| = $(round(dmax, digits=5))  " *
                    "($(dmax > 0 ? "OK: real ESKF (≠ oracle)" : "WARNING: looks like oracle feed"))")
        end
        println("=== SMOKE OK ===")
        return
    end

    with_coupled, no_coupled = build_subset_variants(a["run-dir"])
    out = a["out"]; mkpath(out)
    println("\nSeeds: $(a["seeds"])   Output root: $out")

    run_variant("with_coupled", with_coupled, ctrl_kw, est_cfg, a["seeds"], out)
    run_variant("no_coupled",   no_coupled,   ctrl_kw, est_cfg, a["seeds"], out)

    println("\nComparison complete. Outputs in $out/{with_coupled,no_coupled}/{runs,summary}.csv")
    println("Note: `tracking` uses the TRUE plant state vs ref (real tracking under an imperfect")
    println("estimate); est_rmse_* is separate context (frozen-ESKF quality, closed-loop-coupled).")
    println("Cross-reference with the oracle-clean baseline (runs_controller/RESULTS_controller_tuning.md)")
    println("to quantify the oracle->real-estimator gap per controller.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
