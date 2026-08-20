# NOISY evaluation (not a retune) of the best converged v3 clean PID configs.
#   FB best = seed3 (0.14901)    CT best = seed5 (0.14319)
# Same objective as the clean tuning: v3 metric, lambda_chatter = 0.36,
# train14_v3. 5 NOISE REALISATIONS per config, sensor seeds 101-105 -- the seeds
# the earlier campaign used, so these sit alongside its PID-CT +88% / PID-FB
# +28% / ASMC +330% degradations (NOT directly comparable: different metric,
# chatter price and trajectory set -- same realisations though).
#
# The sensor seed is fixed per run (run_controller_v2 sets cfg.sensor_seed and
# the OracleEstimator seed from it), so each of the 5 is distinct but
# deterministic; the spread across them is REALISATION VARIANCE, the thing that
# flagged ASMC as fragile last campaign (24.9%).
#
# STRUCTURE, deliberately: compute -> SERIALISE -> report, with `report` a
# function that can be exercised on dummy data via REPORT_ONLY=1 WITHOUT running
# any simulation. Two earlier runs of this script completed all 168 simulations
# and then lost every number to a @printf argument-count typo in the reporting.
# Compute must survive a formatting bug, and formatting must be testable without
# paying for compute.
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization
# Trajectory tier, overridable via the V3_TIER env var for the held-out
# :test_v3 evaluation. Outputs are tier-SUFFIXED whenever the tier is not the
# default, so a test run can never overwrite the train14_v3 results the campaign
# tables in RESULTS_v3.md are built from.
# NOTE scores are NOT comparable in magnitude across tiers -- k_traj and the
# trajectory mix both differ. Compare RANKING and each config's train->test delta.
const TIER = Symbol(get(ENV, "V3_TIER", "train14_v3"))
const TSFX = TIER === :train14_v3 ? "" : "_" * String(TIER)
const JLS = "hybrid_ctrl_v2/results_v3/pid_v3_noisy_eval$(TSFX).jls"
const NOISE = (101, 102, 103, 104, 105)
const CFGNAMES = ("FB (seed3)", "CT (seed5)")

function report(R, LAM, V_MAX_, CHATTER_REF_)
    for nm in CFGNAMES
        r = R[nm]
        @printf("\n=== %s ===\n", nm)
        @printf("%-14s %10s %10s %10s %10s\n", "", "track_v3", "chatter", "ce", "SCORE")
        @printf("%-14s %10.5f %10.5f %10.3f %10.5f\n", "clean",
                r.clean.track, r.clean.chat, r.clean.ce, r.clean.sc)
        for (i, s) in enumerate(NOISE)
            q = r.noisy[i]
            @printf("%-14s %10.5f %10.5f %10.3f %10.5f\n", "noise $s", q.track, q.chat, q.ce, q.sc)
        end
        sc = [q.sc for q in r.noisy]; tk = [q.track for q in r.noisy]
        @printf("%-14s %10.5f %10s %10s %10.5f\n", "mean", mean(tk), "", "", mean(sc))
        @printf("%-14s %10.5f %10s %10s %10.5f\n", "std", std(tk), "", "", std(sc))
        @printf("%-14s %9.1f%% %10s %10s %9.1f%%\n", "degradation",
                100*(mean(tk)/r.clean.track - 1), "", "", 100*(mean(sc)/r.clean.sc - 1))
    end
    println("\n=== side by side ===")
    @printf("%-12s %10s %10s %10s %11s %11s\n",
            "", "clean sc", "noisy sc", "degrad", "realis.var", "head share")
    for nm in CFGNAMES
        r = R[nm]; sc = [q.sc for q in r.noisy]
        @printf("%-12s %10.5f %10.5f %9.1f%% %10.1f%% %10.1f%%\n", nm,
                r.clean.sc, mean(sc), 100*(mean(sc)/r.clean.sc - 1),
                100*std(sc)/mean(sc), 100*mean(q.head/q.track for q in r.noisy))
    end
end

if get(ENV, "REPORT_ONLY", "") == "1"
    # Formatting dry-run: exercises every @printf above on dummy data, no sim.
    dummy(x) = (track=x, ce=10x, chat=x/2, head=x/4, sc=2x)
    R = Dict(nm => (clean=dummy(0.1), noisy=[dummy(0.1 + 0.01i) for i in 1:5]) for nm in CFGNAMES)
    report(R, 0.36, 24.0, 0.8)
    println("\nFORMAT DRY-RUN OK (no simulation performed)")
    exit(0)
end

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using JSON
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
score(t, c, h) = t + 0.05*(c/V_MAX) + LAM*(h/CHATTER_REF)
function kw_of(v, s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_pid_v3/seed$s/pid_v2_$(v)_clean/best_config.json")["best_gains"]
    (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
     N=g["N"], feedforward=g["feedforward"])
end
CFG = (CFGNAMES[1] => kw_of("fb", 3), CFGNAMES[2] => kw_of("ct", 5))
trs = collect(trajset(TIER, "trajectory_files_run_0p5_main"))

"aggregate one (config, oracle, seed) over the 14 trajectories"
function evalrun(kw, oracle, sd)
    t = Float64[]; c = Float64[]; h = Float64[]; hp = Float64[]
    for tr in trs
        _, _, p = build_controller_v2(:pid, kw)
        probe, ref, mode, bus = run_controller_v2(:pid, oracle, tr; pid_o=p, seed=sd)
        m2 = Main.controller_metrics(probe, ref, mode)
        m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
        SchedulerMod.clear_probe_log!(bus)
        push!(t, m3.tracking); push!(c, m2.ce); push!(h, m2.chatter)
        q = m3.abs
        push!(hp, (q.final_head/Main.TOL.head_final + q.max_head/Main.TOL.head_max +
                   q.iae_head/StageObjectiveMod.TOL_V3.head_iae) / (6*q.k_traj))
    end
    (track=mean(t), ce=mean(c), chat=mean(h), head=mean(hp), sc=score(mean(t), mean(c), mean(h)))
end

R = Dict{String,Any}()
for (nm, kw) in CFG
    cl = evalrun(kw, :clean, 4)
    ns = [evalrun(kw, :noisy, s) for s in NOISE]
    R[nm] = (clean=cl, noisy=ns)
    @printf("done %s\n", nm); flush(stdout)
end
serialize(JLS, R)                      # BEFORE any formatting
println("results serialised to $JLS")
report(R, LAM, V_MAX, CHATTER_REF)
println("""
REFERENCE, v2-era campaign (tracking, seeds 101-105): PID-CT +88%, PID-FB +28%,
ASMC +330% with 24.9% realisation variance. NOT directly comparable -- different
metric, chatter price and trajectory set -- but the same noise realisations.""")
