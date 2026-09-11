# =============================================================================
# hybrid_ctrl_v2/eval_v3_s100.jl — eval mirror for the safety_margin = 1.0 run
# =============================================================================
# Supervisor-requested ablation: ASMC + PID-CT retuned with the (E56) friction-
# circle safety margin REMOVED (0.9 -> 1.0, --safety-margin 1.0, output trees
# runs_asmc_v3_s100/ and runs_pid_v3_s100/). This script evaluates the converged
# s100 configs EXACTLY the way the 0.9 campaign was evaluated, so the comparison
# is single-variable:
#
#   --mode oracle : clean + noisy-oracle evals on sensor seeds 101-105
#                   (mirror of eval_v3_noisytuned.jl), jls out to
#                   results_v3/s100_oracle_eval[_tier].jls
#   --mode eskf   : frozen-ESKF closed loop, per noise seed 1..5
#                   (mirror of eval_v3_eskf.jl), CSVs to runs_eskf_v3_s100_<tier>/
#
# --tier train14_v3 (default) | test_v3. V3_TIER env var honoured like the
# campaign scripts. Serialise before formatting; REPORT_ONLY=1 dry-runs the
# formatting on dummy data.
#
# The safety margin is read FROM EACH CONFIG ("safety_margin" key written by
# run_stage.jl) and rebuilt as a PhysicalLimits via physical_limits_with_margin,
# merged into the controller kwargs -- the eval can never silently run the 0.9
# gate against 1.0-tuned gains.
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization, JSON, StaticArrays, CSV, DataFrames

const TIER = Symbol(get(ENV, "V3_TIER", "train14_v3"))
const TSFX = TIER === :train14_v3 ? "" : "_" * String(TIER)
const NOISE = (101, 102, 103, 104, 105)

# ---- config discovery: best seed per (family, stage) by best_score -----------
function best_cfg(dir_glob::Vector{String}; expect::Int)
    cfgs = filter(isfile, dir_glob)
    if length(cfgs) != expect
        @warn "expected $expect converged configs, found $(length(cfgs))" pattern=dir_glob
    end
    isempty(cfgs) && error("no best_config.json found: $dir_glob")
    scored = [(JSON.parsefile(c)["best_score"], c) for c in cfgs]
    sort!(scored, by=first)
    return scored[1][2]
end

function s100_configs()
    A(st) = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_asmc_v3_s100", "seed$s", "asmc_v2_$st", "best_config.json") for s in 1:5]
    P(st) = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3_s100", "seed$s", "pid_v2_ct_$st", "best_config.json") for s in 1:5]
    return (
        ("ASMC s100 ct",   :asmc, best_cfg(A("clean"); expect=5)),
        ("ASMC s100 nt",   :asmc, best_cfg(A("noisy")[1:4]; expect=4)),
        ("PID-CT s100 ct", :pid,  best_cfg(P("clean"); expect=5)),
        ("PID-CT s100 nt", :pid,  best_cfg(P("noisy")[1:4]; expect=4)),
    )
end

# ---- gain decode + margin (same decode as eval_v3_eskf.jl / _noisytuned.jl) --
function _kw_from_config(path)
    doc = JSON.parsefile(path)
    g = doc["best_gains"]
    sm = Float64(get(doc, "safety_margin", 0.9))
    lim = Main.physical_limits_with_margin(sm)
    kw = if haskey(g, "lam_x_max")
        (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
         rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
         use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
         kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false, lim=lim)
    else
        (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
         N=g["N"], feedforward=g["feedforward"], lim=lim)
    end
    return kw, sm
end

function parse_args_s100(argv)
    a = Dict{String,Any}("mode" => "oracle", "tier" => String(TIER), "seed" => 0,
                         "aggregate" => false, "out" => "")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if     arg == "--mode";      a["mode"] = argv[i+1]; i += 2
        elseif arg == "--tier";      a["tier"] = argv[i+1]; i += 2
        elseif arg == "--seed";      a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out";       a["out"]  = argv[i+1]; i += 2
        elseif arg == "--aggregate"; a["aggregate"] = true; i += 1
        else; error("unknown arg $arg"); end
    end
    a
end

# ---- report (dummy-data dry-run first, per house rule) ------------------------
function report_s100(rows)
    println("\n=== s100 (safety_margin=1.0) vs archived 0.9 campaign ===")
    @printf("%-16s %10s %10s %10s %10s\n", "config", "score", "track", "chatter", "ce")
    for r in rows
        @printf("%-16s %10.5f %10.5f %10.5f %10.3f\n", r.label, r.score, r.track, r.chat, r.ce)
    end
end

if get(ENV, "REPORT_ONLY", "") == "1"
    report_s100([(label="x", score=0.1, track=0.1, chat=0.5, ce=37.0)])
    println("FORMAT DRY-RUN OK (no simulation performed)")
    exit(0)
end

# eval_controllers_eskf_v3.jl FIRST and ALONE among the drivers: it transitively
# includes tune_controller_v2.jl, trajsets.jl (+`using .TrajSetsMod`), sensors,
# estimators, scheduler, harness. Including tune_controller_v2.jl separately
# BEFORE it (the earlier draft) double-instanced every module and made
# HybridConfig/trajset ambiguous in Main; including it INSIDE main_s100 (the
# draft before that) hit world-age MethodErrors. This is the include order of
# the campaign's own eval_v3_eskf.jl, which is no accident.
include(joinpath(@__DIR__, "eval_controllers_eskf_v3.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
score3(t, c, h) = t + 0.05*(c/Main.V_MAX) + LAM*(h/Main.CHATTER_REF)

function main_s100()
    a = parse_args_s100(ARGS)
    tier = Symbol(a["tier"])
    # Suffix must follow the --tier ARGUMENT, not just the V3_TIER env var:
    # deriving it from the load-time TIER const meant `--tier test_v3` wrote to
    # the train14 filename and clobbered it (caught live 2026-09-11; test rows
    # were rescued by hand to s100_oracle_eval_test_v3.jls).
    tsfx = tier === :train14_v3 ? "" : "_" * String(tier)
    if isempty(a["out"])
        a["out"] = joinpath(ROOT, "hybrid_ctrl_v2", "runs_eskf_v3_s100$tsfx")
    end
    trs = collect(TrajSetsMod.trajset(tier, "trajectory_files_run_0p5_main"))
    CFG = s100_configs()
    for (lab, _, path) in CFG
        _, sm = _kw_from_config(path)
        @printf("config %-14s <- %s (safety_margin=%.2f)\n", lab, path, sm)
    end
    flush(stdout)

    if a["mode"] == "oracle"
        rows = NamedTuple[]
        for (lab, fam, path) in CFG
            kw, _ = _kw_from_config(path)
            # clean baseline + 5 noisy realisations, exactly as eval_v3_noisytuned.jl
            function evalrun(oracle, sd)
                t = Float64[]; ce = Float64[]; h = Float64[]
                for tr in trs
                    ao, _, po = build_controller_v2(fam, kw)
                    probe, ref, mode, bus = run_controller_v2(fam, oracle, tr;
                        asmc_o=(fam === :asmc ? ao : nothing), pid_o=(fam === :pid ? po : nothing), seed=sd)
                    m2 = Main.controller_metrics(probe, ref, mode)
                    m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
                    Main.SchedulerMod.clear_probe_log!(bus)
                    push!(t, m3.tracking); push!(ce, m2.ce); push!(h, m2.chatter)
                end
                (track=mean(t), ce=mean(ce), chat=mean(h))
            end
            cl = evalrun(:clean, 4)
            for s in NOISE
                r = evalrun(:noisy, s)
                push!(rows, (label=lab, regime="noisy$s", track=r.track, chat=r.chat, ce=r.ce,
                             score=score3(r.track, r.ce, r.chat)))
                @printf("done %s noisy %d score=%.5f\n", lab, s, rows[end].score); flush(stdout)
            end
            push!(rows, (label=lab, regime="clean", track=cl.track, chat=cl.chat, ce=cl.ce,
                         score=score3(cl.track, cl.ce, cl.chat)))
        end
        out = joinpath(ROOT, "hybrid_ctrl_v2", "results_v3", "s100_oracle_eval$(tsfx).jls")
        serialize(out, rows)
        println("serialised $(length(rows)) rows to $out")
        df = DataFrame(rows)
        println("\n=== s100 oracle, tier=$tier, seeds 101-105 (mean over realisations) ===")
        for g in groupby(df, [:label, :regime])
            @printf("%-16s %-8s score=%.5f track=%.5f chat=%.5f ce=%.3f\n",
                    g.label[1], g.regime[1], mean(g.score), mean(g.track), mean(g.chat), mean(g.ce))
        end
        return
    end

    # --mode eskf (eval_controllers_eskf_v3.jl already included at top level)
    a["aggregate"] && (aggregate_s100(a["out"]); return)
    ns = 100 + max(a["seed"], 1)
    outrows = NamedTuple[]
    for (lab, fam, path) in CFG, tr in trs
        kw, _ = _kw_from_config(path)
        row = try
            probe, ref, mode, bus = run_closed_loop_eskf(fam, kw, tr; noise_seed=ns)
            m2 = controller_metrics(probe, ref, mode)
            m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
            e = estimator_error(probe)
            Main.SchedulerMod.clear_probe_log!(bus)
            (controller=lab, trajectory=string(tr.name), noise_seed=ns, ok=(m3.ok && m2.ok),
             tracking=m3.tracking, ce=m2.ce, chatter=m2.chatter,
             score=score3(m3.tracking, m2.ce, m2.chatter),
             est_vel=e.vel, est_rate=e.rate, est_pos=e.pos, est_heading=e.heading)
        catch err
            @warn "run failed" config=lab traj=tr.name seed=ns err
            (controller=lab, trajectory=string(tr.name), noise_seed=ns, ok=false,
             tracking=NaN, ce=NaN, chatter=NaN, score=NaN,
             est_vel=NaN, est_rate=NaN, est_pos=NaN, est_heading=NaN)
        end
        push!(outrows, row)
        @printf("%-14s %-28s score=%.5f %s\n", row.controller, row.trajectory, row.score,
                row.ok ? "" : "[NOT OK]"); flush(stdout)
        mkpath(a["out"])
        CSV.write(joinpath(a["out"], "runs_seed$(a["seed"] == 0 ? 1 : a["seed"]).csv"),
                  DataFrame(outrows))
    end
    println("wrote $(length(outrows)) rows to $(a["out"])")
end

"Fold per-seed ESKF CSVs into per-config and per-trajectory summaries."
function aggregate_s100(outdir)
    files = filter(f -> occursin(r"^runs_seed\d+\.csv$", f), readdir(outdir))
    isempty(files) && error("no runs_seed*.csv in $outdir")
    df = reduce(vcat, [CSV.read(joinpath(outdir, f), DataFrame) for f in files])
    ok = df[df.ok .== true, :]
    CSV.write(joinpath(outdir, "summary_by_config.csv"),
              combine(groupby(ok, :controller),
                      :score => mean => :score_mean, :score => std => :score_std,
                      :tracking => mean => :tracking_mean, :chatter => mean => :chatter_mean,
                      :ce => mean => :ce_mean, :est_pos => mean => :est_pos_mean, nrow => :n))
    CSV.write(joinpath(outdir, "summary_by_traj.csv"),
              combine(groupby(ok, [:controller, :trajectory]),
                      :score => mean => :score_mean, :tracking => mean => :tracking_mean))
    println("\n=== ESKF s100, $outdir ===")
    for g in groupby(ok, :controller)
        @printf("%-16s score=%.5f (std %.5f) track=%.5f chat=%.5f ce=%.3f est_pos=%.5f n=%d\n",
                g.controller[1], mean(g.score), std(g.score), mean(g.tracking), mean(g.chatter),
                mean(g.ce), mean(g.est_pos), nrow(g))
    end
    println("wrote summary_by_config.csv / summary_by_traj.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_s100()
end
