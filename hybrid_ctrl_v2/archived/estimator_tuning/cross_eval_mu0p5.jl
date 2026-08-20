#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/cross_eval_mu0p5.jl — decisive identifiability check
# =============================================================================
# Cross-evaluates the three ESKFEstimatorV2 mu=0.5/train12 tuned configs
# (runs_estimator_v2_mu0p5_train12/seed{1,2,3}/best_config.json) under a
# COMMON set of measurement-noise seeds, removing the per-run noise confounder
# (--seed reseeds the sensor suite, so the three tuning runs optimized
# slightly different objectives). For each config two variants are evaluated:
#
#   orig       -- gains exactly as tuned
#   p0default  -- the 5 transient-only P0 dims (P0_vel, P0_yaw, P0_heading,
#                 P0_slip, P0_pos) reset to the ESKFEstimatorV2 struct defaults;
#                 the identified dims (P0_bias_acc, P0_bias_gyro,
#                 pose_Qn_heading, pose_Qn_pos, slip_R_inflate) kept as tuned
#
# Hypothesis under test (k3 identifiability analysis): on full-trajectory
# replay the 5 transient P0 dims are structurally invisible (P collapses to
# its steady-state Riccati solution within a fraction of a second at 1 kHz
# with 4 measurement channels), so orig vs p0default scores should be nearly
# identical, and the three configs should collapse to nearly the same score
# once noise is held common.
#
# Both the :train12 (12 traj, tuning set) and :test (9 traj, held out) tiers
# are evaluated. Parallelized with Threads over (config, variant, noise seed,
# tier, trajectory) tasks; run with `julia -t 8` (project parallelism cap).
#
# Output: runs_estimator_v2_mu0p5_train12/cross_eval/cross_eval_results.json
# plus a printed summary table.
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
using JSON
using Statistics: mean

const RUNS_DIR = joinpath(ROOT, "hybrid_ctrl_v2", "runs_estimator_v2_mu0p5_train12")
const NOISE_SEEDS = [101, 102, 103]   # common noise set; distinct from tuning seeds 1-3
const GAIN_KEYS = (:P0_vel, :P0_yaw, :P0_heading, :P0_bias_acc, :P0_bias_gyro,
                   :P0_slip, :P0_pos, :pose_Qn_heading, :pose_Qn_pos, :slip_R_inflate)
# ESKFEstimatorV2 struct defaults for the transient-only P0 dims
const P0_DEFAULTS = (P0_vel=1e-2, P0_yaw=1e-2, P0_heading=1e-2, P0_slip=1e-5, P0_pos=1.0)

function load_configs()
    cfgs = NamedTuple[]
    for s in 1:3
        gains = JSON.parsefile(joinpath(RUNS_DIR, "seed$s", "best_config.json"))["best_gains"]
        orig = NamedTuple{GAIN_KEYS}(Tuple(Float64(gains[string(k)]) for k in GAIN_KEYS))
        p0d  = merge(orig, P0_DEFAULTS)
        push!(cfgs, (label="seed$(s)_orig",      gains=orig))
        push!(cfgs, (label="seed$(s)_p0default", gains=p0d))
    end
    return cfgs
end

function replay_trajset_for(tier::Symbol)
    base = Main.TrajSetsMod.trajset(tier, "trajectory_files_run_0p5_main")
    return [merge(tr, (profile=Main.HarnessV2Mod._toml_to_profile(tr.profile_toml),)) for tr in base]
end

function main()
    cfgs = load_configs()
    tiers = (:train12, :test)
    trajsets = Dict(t => replay_trajset_for(t) for t in tiers)

    tasks = [(ci=ci, ns=ns, tier=tier, tr=tr)
             for (ci, _) in enumerate(cfgs)
             for ns in NOISE_SEEDS
             for tier in tiers
             for tr in trajsets[tier]]
    n_tasks = length(tasks)
    results = Vector{Any}(undef, n_tasks)
    done = Threads.Atomic{Int}(0)
    t_start = time()

    println("=== cross_eval_mu0p5: $(length(cfgs)) config-variants x $(length(NOISE_SEEDS)) noise seeds x " *
            "$(sum(length(trajsets[t]) for t in tiers)) trajectories = $n_tasks replays on $(Threads.nthreads()) threads ===")

    Threads.@threads for i in 1:n_tasks
        tk = tasks[i]
        cfg = cfgs[tk.ci]
        est_cfg = merge(cfg.gains, (use_dhat=false,))
        res = try
            suite = Main.SensorModV2.build_suite(:realistic; seed=tk.ns, flow=true, fix_tier=:docking)
            log = Main.HarnessV2Mod.run_and_log_replay_v2(est_cfg, tk.tr, suite; seed=tk.ns)
            m = Main.EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log])
            (ok=true, metrics=m)
        catch e
            (ok=false, metrics=nothing, err=sprint(showerror, e))
        end
        results[i] = (label=cfg.label, noise_seed=tk.ns, tier=tk.tier,
                      traj=string(tk.tr.name), combo=tk.tr.combo_idx, res=res)
        d = Threads.atomic_add!(done, 1)
        if d % 10 == 0 || d == n_tasks
            el = time() - t_start
            println("  [$d/$n_tasks] elapsed=$(round(el, digits=1))s eta=$(round(el/d*(n_tasks-d), digits=1))s")
        end
    end

    # Checkpoint raw per-trajectory results BEFORE aggregation, so a failure in
    # the reporting code below never loses the replay compute.
    mfields = (:vel_rmse, :inslip_vel_rmse, :rate_rmse, :pos_rmse, :heading_rmse, :smoothness)
    outdir = joinpath(RUNS_DIR, "cross_eval")
    mkpath(outdir)
    per_traj_records = [Dict("label"=>r.label, "noise_seed"=>r.noise_seed, "tier"=>string(r.tier),
                             "traj"=>r.traj, "combo"=>r.combo, "ok"=>r.res.ok,
                             "metrics"=>r.res.ok ? Dict(string(f)=>getfield(r.res.metrics, f) for f in (mfields..., :score)) : nothing,
                             "err"=>r.res.ok ? nothing : get(r.res, :err, nothing)) for r in results]
    open(joinpath(outdir, "cross_eval_raw.json"), "w") do io
        JSON.print(io, per_traj_records, 2)
    end

    # Aggregate: estimator_objective_abs_v2 averages per-log RMS terms, so the
    # group aggregate is exactly the mean of per-trajectory metric fields.
    summary = Dict{String,Any}()
    n_fail = 0
    for cfg in cfgs, tier in tiers
        for ns in NOISE_SEEDS
            rows = filter(r -> r.label == cfg.label && r.tier == tier && r.noise_seed == ns, results)
            oks = [r for r in rows if r.res.ok]
            n_fail += length(rows) - length(oks)
            isempty(oks) && continue
            agg = Dict(string(f) => mean([getfield(r.res.metrics, f) for r in oks]) for f in mfields)
            agg["score"] = agg["vel_rmse"]/1e-3 + agg["rate_rmse"]/1e-2 + agg["inslip_vel_rmse"]/1e-3 +
                           agg["pos_rmse"]/1e-2 + agg["heading_rmse"]/1e-2 + 0.05*agg["smoothness"]
            agg["n_ok"] = length(oks)
            summary["$(cfg.label)|$tier|ns$ns"] = agg
        end
    end

    # Per config/tier mean over the common noise seeds (the decisive numbers)
    println("\n===== AGGREGATE (mean over common noise seeds $(NOISE_SEEDS)) =====")
    println(rpad("config", 20), rpad("tier", 9), rpad("score", 9), rpad("vel", 10),
            rpad("rate", 10), rpad("inslip", 10), rpad("pos", 10), "heading")
    decisive = Dict{String,Any}()
    for cfg in cfgs, tier in tiers
        keys = ["$(cfg.label)|$tier|ns$ns" for ns in NOISE_SEEDS if haskey(summary, "$(cfg.label)|$tier|ns$ns")]
        isempty(keys) && continue
        m = Dict(string(f) => mean([summary[k][string(f)] for k in keys]) for f in (mfields..., "score"))
        decisive["$(cfg.label)|$tier"] = m
        println(rpad(cfg.label, 20), rpad(string(tier), 9),
                rpad(round(m["score"], digits=3), 9),
                rpad(round(m["vel_rmse"], sigdigits=4), 10),
                rpad(round(m["rate_rmse"], sigdigits=4), 10),
                rpad(round(m["inslip_vel_rmse"], sigdigits=4), 10),
                rpad(round(m["pos_rmse"], sigdigits=4), 10),
                round(m["heading_rmse"], sigdigits=4))
    end
    println("n_fail=$n_fail / $n_tasks")

    outdir = joinpath(RUNS_DIR, "cross_eval")
    mkpath(outdir)
    open(joinpath(outdir, "cross_eval_results.json"), "w") do io
        JSON.print(io, Dict(
            "noise_seeds" => NOISE_SEEDS,
            "configs" => [Dict("label" => c.label, "gains" => Dict(string(k) => v for (k, v) in pairs(c.gains))) for c in cfgs],
            "per_group" => summary,
            "decisive_mean_over_noise" => decisive,
            "n_fail" => n_fail,
            "per_trajectory" => [Dict("label"=>r.label, "noise_seed"=>r.noise_seed, "tier"=>string(r.tier),
                                      "traj"=>r.traj, "combo"=>r.combo, "ok"=>r.res.ok,
                                      "metrics"=>r.res.ok ? Dict(string(f)=>getfield(r.res.metrics, f) for f in (mfields..., :score)) : nothing,
                                      "err"=>r.res.ok ? nothing : get(r.res, :err, nothing)) for r in results],
        ), 2)
    end
    println("-> $(joinpath(outdir, "cross_eval_results.json"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
