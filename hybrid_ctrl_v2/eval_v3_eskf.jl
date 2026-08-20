#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/eval_v3_eskf.jl
#
# The v3 comparison table with STATE FEEDBACK FROM THE TUNED ESKF instead of the
# oracle — the deployable-performance counterpart of RESULTS_v3.md §4.
#
# EVERY number in RESULTS_v3.md §1-§5 is oracle-state feedback: run_controller_v2
# hardcodes `estimator = :oracle`, and its `:clean`/`:noisy` argument selects the
# SENSOR realisation, not the state source. So those results answer "how well does
# the control law track given perfect state", and this script answers "...given the
# state the frozen ESKF actually delivers".
#
# WHAT IS HELD FIXED, and why each matters:
#   * ESKF = runs_estimator_v4_mu0p5_train12/seed4 (ESKFEstimatorV3, 13-dim), FROZEN.
#     Estimator-first methodology: tune and freeze the observer, THEN compare control
#     laws, so a control-law difference cannot be an observer difference in disguise.
#     NOTE the estimator was tuned on train12 at mu=0.5, NOT on train14_v3. It is
#     deliberately not retuned here -- refreezing per trajectory tier would reintroduce
#     exactly the confound the methodology exists to remove -- but it does mean the
#     estimator is slightly off-distribution for two of the 14 entries.
#   * Sensor suite / estimator rate / pose-fix tier: inherited UNCHANGED by including
#     eval_controllers_eskf_v3.jl and calling its `run_closed_loop_eskf`. Those
#     conditions must match the ESKF's own tuning conditions exactly
#     (build_suite(:realistic; seed, flow=true, fix_tier=:docking), f_est = 1000 Hz,
#     use_dhat=false); copying the wiring instead of reusing it would let the two
#     drift apart silently. That file's `main()` is guarded by a PROGRAM_FILE check,
#     so including it is side-effect free.
#   * Noise seeds 101-105 -- THE SAME FIVE REALISATIONS as the oracle evals
#     (--seed N maps to noise seed 100+N). Common random numbers: the same
#     (trajectory, noise seed) pair produces the same sensor realisation for every
#     controller, so the six configs are paired, not merely averaged.
#
# GAIN DECODE is copied from eval_v3_noisytuned.jl, NOT from the included driver's
# `load_gains`: that one does `Float64(v)` per entry and would throw on ASMC's
# `kmax_sched_floor` (a 3-vector), and it would not set `use_cubic=false`. Building
# the controllers identically to the oracle runs is what makes the ESKF-vs-oracle
# difference attributable to the estimator rather than to a different decode.
#
# METRIC: v3 tracking + the FULL v3 score including the chatter term --
#   score = tracking_v3 + 0.05*(ce/V_MAX) + 0.36*(chatter/CHATTER_REF)
# The included driver scores `tracking + 0.05*ce/V_MAX` on the v2 metric with NO
# chatter term, which is not comparable with anything in RESULTS_v3.md.
# All metrics are computed on the TRUE state (probe `u`), never on `xhat` -- a
# controller must not be rewarded for tracking its own estimate.
#
# CLI:
#   --seed N       noise seed 100+N, all 6 configs x 14 trajectories   (default 0 = 101)
#   --tier NAME    trajectory tier (default train14_v3)
#   --out DIR      output root (default hybrid_ctrl_v2/runs_eskf_v3_train14)
#   --smoke        1 trajectory x 6 configs x 1 seed, printed, no write
#   --aggregate    fold runs_seed*.csv into summary + the paired oracle comparison
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))

# Reuse the ESKF wiring verbatim (its main() is PROGRAM_FILE-guarded).
include(joinpath(@__DIR__, "eval_controllers_eskf_v3.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl"))
using .StageObjectiveMod
using StaticArrays, Serialization

const LAM_CHAT = StageObjectiveMod.LAMBDA_CHATTER_V3
v3_score(t, ce, ch) = t + Main.LAMBDA_CE*(ce/Main.V_MAX) + LAM_CHAT*(ch/Main.CHATTER_REF)

# ---- gain decode: identical to eval_v3_noisytuned.jl -------------------------
function asmc_kw_v3(path)
    g = JSON.parsefile(path)["best_gains"]
    (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
     rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
     use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
     kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false)
end
function pid_kw_v3(path)
    g = JSON.parsefile(path)["best_gains"]
    (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
     N=g["N"], feedforward=g["feedforward"])
end

_A(seed, stage) = joinpath(ROOT, "hybrid_ctrl_v2", "runs_asmc_v3", "seed$seed", "asmc_v2_$stage", "best_config.json")
_P(seed, var, stage) = joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3", "seed$seed", "pid_v2_$(var)_$stage", "best_config.json")

# Best seed per (controller, stage), exactly as quoted in RESULTS_v3.md §4.
const CONFIGS_V3 = [
    ("ASMC ct(s2)",   :asmc, asmc_kw_v3(_A(2, "clean"))),
    ("ASMC nt(s4)",   :asmc, asmc_kw_v3(_A(4, "noisy"))),
    ("PID-CT ct(s5)", :pid,  pid_kw_v3(_P(5, "ct", "clean"))),
    ("PID-CT nt(s4)", :pid,  pid_kw_v3(_P(4, "ct", "noisy"))),
    ("PID-FB ct(s3)", :pid,  pid_kw_v3(_P(3, "fb", "clean"))),
    ("PID-FB nt(s4)", :pid,  pid_kw_v3(_P(4, "fb", "noisy"))),
]

"One (config, trajectory, noise seed) closed-loop run through the frozen ESKF."
function run_row_v3(label, family, ctrl_kw, tr, noise_seed)
    probe, ref, mode, bus = run_closed_loop_eskf(family, ctrl_kw, tr; noise_seed=noise_seed)
    m2 = controller_metrics(probe, ref, mode)                       # ce / chatter
    m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)  # tracking_v3
    e  = estimator_error(probe)
    c  = mode == :pose ? controller_error_pose(probe, ref) :
           (pos=NaN, heading=NaN, pos_final=NaN, heading_final=NaN)
    Main.SchedulerMod.clear_probe_log!(bus)
    (controller=label, trajectory=string(tr.name), mode=string(mode), noise_seed=noise_seed,
     ok=(m3.ok && m2.ok), tracking=m3.tracking, ce=m2.ce, chatter=m2.chatter,
     score=v3_score(m3.tracking, m2.ce, m2.chatter),
     est_vel=e.vel, est_rate=e.rate, est_pos=e.pos, est_heading=e.heading,
     ctrl_pos=c.pos, ctrl_heading=c.heading,
     ctrl_pos_final=c.pos_final, ctrl_head_final=c.heading_final)
end

_fail_row_v3(label, tr, ns) = (controller=label, trajectory=string(tr.name),
    mode=string(tr.run_mode), noise_seed=ns, ok=false, tracking=NaN, ce=NaN, chatter=NaN,
    score=NaN, est_vel=NaN, est_rate=NaN, est_pos=NaN, est_heading=NaN,
    ctrl_pos=NaN, ctrl_heading=NaN, ctrl_pos_final=NaN, ctrl_head_final=NaN)

function parse_args_v3(argv)
    a = Dict{String,Any}("seed" => 0, "tier" => "train14_v3",
                         "out" => joinpath(ROOT, "hybrid_ctrl_v2", "runs_eskf_v3_train14"),
                         "smoke" => false, "aggregate" => false)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if     arg == "--seed";      a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--tier";      a["tier"] = argv[i+1]; i += 2
        elseif arg == "--out";       a["out"]  = argv[i+1]; i += 2
        elseif arg == "--smoke";     a["smoke"] = true; i += 1
        elseif arg == "--aggregate"; a["aggregate"] = true; i += 1
        else; error("unknown arg $arg"); end
    end
    a
end

"Fold per-seed CSVs into a summary and set it beside the ORACLE numbers."
function aggregate_v3(outdir)
    files = filter(f -> occursin(r"^runs_seed\d+\.csv$", f), readdir(outdir))
    isempty(files) && error("no runs_seed*.csv in $outdir")
    df = reduce(vcat, [CSV.read(joinpath(outdir, f), DataFrame) for f in files])
    ok = df[df.ok .== true, :]
    @printf("\n=== ESKF state feedback, %s, noise seeds 101-105 ===\n", basename(outdir))
    @printf("%-14s %10s %10s %10s %10s %10s %9s %s\n",
            "config", "score", "std", "track", "chatter", "ce", "est_pos", "n")
    means = Dict{String,Float64}()
    for lab in unique(ok.controller)
        g = ok[ok.controller .== lab, :]
        means[lab] = mean(g.score)
        @printf("%-14s %10.5f %10.5f %10.5f %10.5f %10.3f %9.5f %d\n", lab,
                mean(g.score), std(g.score), mean(g.tracking), mean(g.chatter),
                mean(g.ce), mean(g.est_pos), nrow(g))
    end
    nb = nrow(df) - nrow(ok)
    nb > 0 && @printf("\n[warn] %d of %d runs did NOT converge and are excluded\n", nb, nrow(df))

    # Both oracle baselines (RESULTS_v3.md §4), same 5 realisations.
    #
    # THE TWO BASELINES ARE NOT THE SAME KIND OF THING, and comparing against the
    # wrong one inverts the conclusion:
    #   ORACLE_CLEAN — perfect state. The BEST ACHIEVABLE, an upper bound.
    #   ORACLE_NOISY — `OracleEstimator(:noisy)` writes true state + UNFILTERED
    #                  per-tick sensor noise straight into xhat (sig_vel 0.010 m/s,
    #                  sig_pos 0.020 m, sig_psi 0.010 rad, + scale-factor + turn-on
    #                  bias). There is NO estimator in that path. It is a
    #                  no-filtering pessimistic bound, NOT a deployment model.
    # The ESKF delivers 0.94 mm/s velocity and 3.25 mm position RMSE, so it beats
    # ORACLE_NOISY by construction — a tuned filter beats raw noise, which is not a
    # finding. The meaningful quantities are the COST against clean state and the
    # FRACTION OF THE GAP the estimator recovers.
    ORACLE_CLEAN = Dict("ASMC ct(s2)"=>0.15318, "ASMC nt(s4)"=>0.15697,
                        "PID-CT ct(s5)"=>0.14340, "PID-CT nt(s4)"=>0.14422,
                        "PID-FB ct(s3)"=>0.14953, "PID-FB nt(s4)"=>0.15492)
    ORACLE_NOISY = Dict("ASMC ct(s2)"=>0.53168, "ASMC nt(s4)"=>0.51838,
                        "PID-CT ct(s5)"=>0.50372, "PID-CT nt(s4)"=>0.49799,
                        "PID-FB ct(s3)"=>0.53135, "PID-FB nt(s4)"=>0.53199)
    # Those baselines were measured on train14_v3. `k_traj` and the trajectory
    # mix both differ per tier, so comparing a test_v3 ESKF score against a
    # train14_v3 oracle score is meaningless -- suppress the table off-tier
    # rather than print a difference of two different populations.
    if !occursin("train14", basename(outdir))
        println("\n[baseline comparison suppressed: the stored oracle baselines are train14_v3.")
        println(" Scores across tiers are NOT comparable in magnitude (different k_traj and")
        println(" trajectory mix) -- compare RANKING and each controller's train->test delta.]")
        CSV.write(joinpath(outdir, "summary_by_config.csv"),
                  combine(groupby(ok, :controller),
                          :score => mean => :score_mean, :score => std => :score_std,
                          :tracking => mean => :tracking_mean, :chatter => mean => :chatter_mean,
                          :ce => mean => :ce_mean, :est_pos => mean => :est_pos_mean,
                          nrow => :n))
        CSV.write(joinpath(outdir, "summary_by_traj.csv"),
                  combine(groupby(ok, [:controller, :trajectory]),
                          :score => mean => :score_mean, :tracking => mean => :tracking_mean))
        println("\nwrote summary_by_config.csv / summary_by_traj.csv to $outdir")
        return
    end
    println("\n=== ESKF against BOTH oracle baselines (same 5 realisations) ===")
    println("clean = perfect state (upper bound); noisy-oracle = UNFILTERED sensor noise as")
    println("state, no estimator (pessimistic bound). ESKF is the deployable path between them.")
    @printf("%-14s %10s %10s %10s %12s %12s\n",
            "config", "clean", "ESKF", "noisyOrc", "cost vs clean", "gap recov.")
    for (lab, _, _) in CONFIGS_V3
        haskey(means, lab) || continue
        cl, ns, ek = ORACLE_CLEAN[lab], ORACLE_NOISY[lab], means[lab]
        @printf("%-14s %10.5f %10.5f %10.5f %+12.5f %11.1f%%\n",
                lab, cl, ek, ns, ek - cl, 100*(ns - ek)/(ns - cl))
    end
    CSV.write(joinpath(outdir, "summary_by_config.csv"),
              combine(groupby(ok, :controller),
                      :score => mean => :score_mean, :score => std => :score_std,
                      :tracking => mean => :tracking_mean, :chatter => mean => :chatter_mean,
                      :ce => mean => :ce_mean, :est_pos => mean => :est_pos_mean,
                      nrow => :n))
    CSV.write(joinpath(outdir, "summary_by_traj.csv"),
              combine(groupby(ok, [:controller, :trajectory]),
                      :score => mean => :score_mean, :tracking => mean => :tracking_mean))
    println("\nwrote summary_by_config.csv / summary_by_traj.csv to $outdir")
end

function main_v3()
    a = parse_args_v3(ARGS)
    a["aggregate"] && (aggregate_v3(a["out"]); return)

    trajs = collect(trajset(Symbol(a["tier"]), "trajectory_files_run_0p5_main"))
    ns = 100 + max(a["seed"], 1)
    println("ESKF config : $ESKF_CFG_PATH  (FROZEN, tuned on train12 mu=0.5)")
    println("tier        : $(a["tier"])  ($(length(trajs)) trajectories)")
    println("noise seed  : $ns")
    println("configs     : $(join([c[1] for c in CONFIGS_V3], ", "))")
    flush(stdout)

    a["smoke"] && (trajs = trajs[1:1])
    rows = NamedTuple[]
    for (label, family, kw) in CONFIGS_V3, tr in trajs
        row = try
            run_row_v3(label, family, kw, tr, ns)
        catch err
            @warn "run failed" config=label traj=tr.name seed=ns err
            _fail_row_v3(label, tr, ns)
        end
        push!(rows, row)
        @printf("%-14s %-28s score=%.5f track=%.5f chat=%.5f est_pos=%.5f%s\n",
                row.controller, row.trajectory, row.score, row.tracking, row.chatter,
                row.est_pos, row.ok ? "" : "   [NOT OK]"); flush(stdout)
        # Write after EVERY row: 84 ESKF runs per seed is too much to lose to a late crash.
        if !a["smoke"]
            mkpath(a["out"])
            CSV.write(joinpath(a["out"], "runs_seed$(a["seed"] == 0 ? 1 : a["seed"]).csv"),
                      DataFrame(rows))
        end
    end
    a["smoke"] ? println("\nSMOKE OK (nothing written)") :
                 println("\nwrote $(length(rows)) rows to $(a["out"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v3()
end
