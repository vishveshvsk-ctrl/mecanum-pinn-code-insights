# =============================================================================
# hybrid_ctrl_v2/eval_docking_oracle.jl — controller comparison on the docking
# (step-hold) regime, ORACLE feedback only.
# =============================================================================
# Motivation (chat, 2026-08): the v3 campaign tiers contain no sustained
# preslip operating time — TRAIN12 dropped both docking entries, so every
# PID-CT > ASMC number is a statement about rolling regimes only. Docking
# (quintic approach + explicit hold, profiles.jl:896) is the one profile with
# a sustained zero-velocity setpoint: the preslip/stiction band, where the
# friction band descends into the loop (slip ~< 2 mm/s crossing).
#
# This script evaluates the CONVERGED v3 configs on both docking references
# exactly the way the campaign evaluated train14_v3 (eval_v3_s100.jl oracle
# branch): same decode, same objective (score3 = tracking + ce + LAM*chatter),
# clean + noisy-oracle seeds 101-105. Configs: archived 0.9 campaign
# (runs_asmc_v3 / runs_pid_v3 clean-tuned) AND the s100 clean-tuned configs,
# so the regime comparison inherits both tuning generations.
#
# No new tuning — this is an eval-only probe. If scores here are low for BOTH
# controllers, the follow-up is the estimator question (docking was originally
# excluded partly because the estimator performed badly in this regime).
#
# Modes:
#   --mode oracle (default): clean + noisy-oracle, as the campaign does; per-run
#                            time series stored under results_v3/docking_runs/.
#   --mode eskf            : frozen tuned ESKF v4 (seed-4 config, realistic
#                            sensor suite, pose_fix_tier=:docking), noise seeds
#                            101-105 — the estimator-on-docking measurement.
#                            Series stored the same way; report carries
#                            estimator RMSE (vel/rate/pos/heading) alongside
#                            controller metrics.
#
# Rates: controller/estimator callbacks and mixer at 1000 Hz (HybridConfig
# defaults), plant integrated continuously (FBDF, dtmax=1 ms), probe series
# saved at 500 Hz (saveat_hz). (The 2 kHz figure belongs to the dataset-
# generation pipeline, a different harness.)
#
# Output: hybrid_ctrl_v2/results_v3/docking_oracle_eval.jls (+ .md report)
#         hybrid_ctrl_v2/results_v3/docking_runs/<mode>/*.csv  (time series)
# Usage:  julia --project=. hybrid_ctrl_v2/eval_docking_oracle.jl [--mode oracle|eskf]
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization, JSON, StaticArrays, DataFrames

const NOISE = (101, 102, 103, 104, 105)

# ---- traj entries (shape matches TrajSetsMod._mk output) ---------------------
const DOCKING_TRAJS = [
    (name = :docking_a,
     profile_toml = "docking_mu_0p5.toml",
     combo_idx = 1, ref_type = :posref, mu = 0.5,
     config_dir = "trajectory_files_run_0p5_main",
     run_mode = :pose, adapt = false, role = :step_hold),
    (name = :docking_step,
     profile_toml = "docking_step_mu_0p5.toml",
     combo_idx = 1, ref_type = :posref, mu = 0.5,
     config_dir = "trajectory_files_run_0p5_main",
     run_mode = :pose, adapt = false, role = :step_hold),
]

# ---- config discovery: best clean-tuned config per family --------------------
function best_cfg(dir_glob::Vector{String}; expect::Int)
    cfgs = filter(isfile, dir_glob)
    length(cfgs) != expect && @warn "expected $expect configs, found $(length(cfgs))" pattern=dir_glob
    isempty(cfgs) && error("no best_config.json found: $dir_glob")
    scored = [(JSON.parsefile(c)["best_score"], c) for c in cfgs]
    sort!(scored; by = first)
    return scored[1][2]
end

function campaign_configs()
    A = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_asmc_v3", "seed$s", "asmc_v2_clean", "best_config.json") for s in 1:5]
    P = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3", "seed$s", "pid_v2_ct_clean", "best_config.json") for s in 1:5]
    A1 = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_asmc_v3_s100", "seed$s", "asmc_v2_clean", "best_config.json") for s in 1:5]
    P1 = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3_s100", "seed$s", "pid_v2_ct_clean", "best_config.json") for s in 1:5]
    return [
        ("ASMC 0.9 ct",   :asmc, best_cfg(A; expect=5)),
        ("PID-CT 0.9 ct", :pid,  best_cfg(P; expect=5)),
        ("ASMC s100 ct",   :asmc, best_cfg(A1; expect=5)),
        ("PID-CT s100 ct", :pid,  best_cfg(P1; expect=5)),
    ]
end

# ---- gain decode + safety margin (verbatim from eval_v3_s100.jl) -------------
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

function report_dry()
    println("FORMAT DRY-RUN OK (no simulation performed)")
    @printf("%-16s %-12s %-8s %9s %9s %9s %9s\n",
            "config", "traj", "regime", "score", "track", "chatter", "ce")
    @printf("%-16s %-12s %-8s %9.5f %9.5f %9.5f %9.3f\n",
            "dummy", "docking_a", "clean", 0.1, 0.1, 0.5, 37.0)
end

# ---- per-run time-series storage (probe log @ 500 Hz) ------------------------
"Store one run's probe series: true state, estimate, command, reference."
function store_series(probe, ref, outdir, tag)
    mkpath(outdir)
    t   = [p.t for p in probe]
    sel(i) = [p.u[i] for p in probe]
    hat(i) = [p.xhat[i] for p in probe]
    cmd(i) = [p.v_cmd[i] for p in probe]
    df = DataFrame(
        t = t,
        vx = sel(1), vy = sel(2), psidot = sel(3), psi = sel(4), X = sel(17), Y = sel(18),
        vx_hat = hat(1), vy_hat = hat(2), psidot_hat = hat(3), psi_hat = hat(4),
        X_hat = hat(5), Y_hat = hat(6),
        v_cmd1 = cmd(1), v_cmd2 = cmd(2), v_cmd3 = cmd(3), v_cmd4 = cmd(4),
        x_ref = [ref.xo(p.t) for p in probe], y_ref = [ref.yo(p.t) for p in probe],
        psi_ref = [ref.psi(p.t) for p in probe])
    CSV.write(joinpath(outdir, tag * ".csv"), df)
    return nothing
end

if get(ENV, "REPORT_ONLY", "") == "1"
    report_dry()
    exit(0)
end

# eval_controllers_eskf_v3.jl FIRST and ALONE (see eval_v3_s100.jl header for
# why this include order is load-bearing).
include(joinpath(@__DIR__, "eval_controllers_eskf_v3.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
score3(t, c, h) = t + 0.05*(c/Main.V_MAX) + LAM*(h/Main.CHATTER_REF)

san(s::String) = replace(s, r"[^A-Za-z0-9_.-]" => "_")

function main_docking()
    mode = "oracle"
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--mode"; mode = ARGS[i+1]; i += 2
        else; error("unknown arg $(ARGS[i])"); end
    end
    mode in ("oracle", "eskf") || error("--mode must be oracle|eskf, got $mode")

    CFG = campaign_configs()
    for (lab, _, path) in CFG
        _, sm = _kw_from_config(path)
        @printf("config %-16s <- %s (safety_margin=%.2f)\n", lab, path, sm)
    end
    flush(stdout)

    run_dir = joinpath(ROOT, "hybrid_ctrl_v2", "results_v3", "docking_runs", mode)
    rows = NamedTuple[]

    if mode == "oracle"
        for (lab, fam, path) in CFG
            kw, _ = _kw_from_config(path)
            for (oracle, sd) in ((:clean, 4), [(:noisy, s) for s in NOISE]...)
                for tr in DOCKING_TRAJS
                    ao, _, po = build_controller_v2(fam, kw)
                    probe, ref, mode_, bus = run_controller_v2(fam, oracle, tr;
                        asmc_o=(fam === :asmc ? ao : nothing),
                        pid_o=(fam === :pid ? po : nothing), seed=sd)
                    m2 = Main.controller_metrics(probe, ref, mode_)
                    m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode_)
                    store_series(probe, ref, run_dir,
                                 "$(san(lab))__$(tr.name)__$(oracle)$(oracle === :noisy ? "_s$(sd)" : "")")
                    Main.SchedulerMod.clear_probe_log!(bus)
                    push!(rows, (config=lab, traj=String(tr.name),
                                 regime=(oracle === :noisy ? "noisy$(sd)" : "clean"),
                                 track=m3.tracking, chat=m2.chatter, ce=m2.ce,
                                 est_pos=NaN, est_vel=NaN,
                                 score=score3(m3.tracking, m2.ce, m2.chatter)))
                    @printf("%-16s %-12s %-8s track=%.5f chatter=%.5f ce=%.2f score=%.5f\n",
                            lab, tr.name, oracle, m3.tracking, m2.chatter, m2.ce, rows[end].score)
                    flush(stdout)
                end
            end
        end
        sfx, title = "", "# Docking (step-hold) oracle eval — controller comparison in the preslip regime\n"
    else
        # --mode eskf: frozen tuned ESKF v4 + realistic sensors, seeds 101-105
        for (lab, fam, path) in CFG
            kw, _ = _kw_from_config(path)
            for sd in 1:5
                for tr in DOCKING_TRAJS
                    probe, ref, mode_, bus = run_closed_loop_eskf(fam, kw, tr; noise_seed=100 + sd)
                    m2 = Main.controller_metrics(probe, ref, mode_)
                    m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode_)
                    e = estimator_error(probe)
                    store_series(probe, ref, run_dir,
                                 "$(san(lab))__$(tr.name)__seed$(sd)")
                    Main.SchedulerMod.clear_probe_log!(bus)
                    push!(rows, (config=lab, traj=String(tr.name), regime="eskf$(sd)",
                                 track=m3.tracking, chat=m2.chatter, ce=m2.ce,
                                 est_pos=e.pos, est_vel=e.vel,
                                 score=score3(m3.tracking, m2.ce, m2.chatter)))
                    @printf("%-16s %-12s seed=%d track=%.5f est_pos=%.5f est_vel=%.5f score=%.5f\n",
                            lab, tr.name, sd, m3.tracking, e.pos, e.vel, rows[end].score)
                    flush(stdout)
                end
            end
        end
        sfx, title = "_eskf", "# Docking (step-hold) ESKF eval — estimator performance in the preslip regime\n"
    end

    out = joinpath(ROOT, "hybrid_ctrl_v2", "results_v3", "docking_oracle_eval$(sfx).jls")
    serialize(out, rows)
    println("\nserialised $(length(rows)) rows to $out")

    # ---- report ----
    df = DataFrame(rows)
    rep = IOBuffer()
    print(rep, title)
    if mode == "oracle"
        println(rep, "\nOracle feedback (clean + noisy seeds 101-105), campaign scoring (score3),",
                " converged v3 configs (0.9 archived + s100 clean-tuned), no re-tuning.\n")
    else
        println(rep, "\nFrozen tuned ESKF v4 (seed-4 config) + realistic sensor suite",
                " (pose_fix_tier=:docking), noise seeds 101-105, campaign scoring (score3).\n")
    end
    hdr = mode == "eskf" ?
        "| config | traj | regime | score | tracking | chatter | est_pos [m] | est_vel [m/s] |" :
        "| config | traj | regime | score | tracking | chatter | ce |"
    println(rep, hdr)
    println(rep, "|---|---|---|---|---|---|---|---|")
    for r in rows
        if mode == "eskf"
            @printf(rep, "| %s | %s | %s | %.5f | %.5f | %.5f | %.5f | %.5f |\n",
                    r.config, r.traj, r.regime, r.score, r.track, r.chat, r.est_pos, r.est_vel)
        else
            @printf(rep, "| %s | %s | %s | %.5f | %.5f | %.5f | %.2f |\n",
                    r.config, r.traj, r.regime, r.score, r.track, r.chat, r.ce)
        end
    end
    println(rep, "\n## Per-config mean (both trajectories, noise seeds averaged)\n")
    noisy = df[df.regime .!= "clean", :]
    agg = combine(groupby(noisy, [:config, :traj]),
                  :score => mean => :score, :track => mean => :track,
                  :chat => mean => :chat,
                  :est_pos => (x -> mean(filter(!isnan, x))) => :est_pos,
                  :est_vel => (x -> mean(filter(!isnan, x))) => :est_vel)
    if mode == "eskf"
        println(rep, "| config | traj | score | tracking | chatter | est_pos [m] | est_vel [m/s] |")
        println(rep, "|---|---|---|---|---|---|---|")
        for r in eachrow(agg)
            @printf(rep, "| %s | %s | %.5f | %.5f | %.5f | %.5f | %.5f |\n",
                    r.config, r.traj, r.score, r.track, r.chat, r.est_pos, r.est_vel)
        end
    else
        println(rep, "| config | traj | score | tracking | chatter |")
        println(rep, "|---|---|---|---|---|")
        for r in eachrow(agg)
            @printf(rep, "| %s | %s | %.5f | %.5f | %.5f |\n", r.config, r.traj, r.score, r.track, r.chat)
        end
    end
    open(joinpath(ROOT, "hybrid_ctrl_v2", "results_v3", "docking_oracle_report$(sfx).md"), "w") do io
        write(io, String(take!(rep)))
    end
    println(String(take!(rep)))
    println("wrote docking_oracle_report$(sfx).md")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_docking()
end
