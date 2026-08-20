# Do the NOISY-TUNED configs actually beat the CLEAN-TUNED ones under noise?
#
# The 12 noisy tuning stages each scored on their OWN stage realisation, so their
# best_scores cannot be compared with each other's absolute level, nor with the
# clean-tuned noisy EVAL (hybrid_ctrl_v2/results_v3/pid_v3_noisy_eval.jls, hybrid_ctrl_v2/results_v3/asmc_v3_noisy_eval.jls)
# which averaged sensor seeds 101-105. This script puts the noisy-tuned configs on
# those SAME five realisations, so all six rows (3 clean-tuned + 3 noisy-tuned)
# finally sit on one footing. That is deliverable 5's table.
#
# CONFIG SELECTION CAVEAT, stated because it is load-bearing: the best seed of
# every controller is seed4, and seed4 is also the EASIEST realisation for all
# three (identical seed ordering across controllers -- the tuning seed sets the
# noise draw, not just the optimizer start). So "best seed" here partly selects
# the luckiest draw, not the best gains. It is applied IDENTICALLY to all three
# controllers, so the cross-controller comparison is unaffected; what it does mean
# is that a noisy-tuned config could look worse than its clean-tuned counterpart
# on 101-105 through realisation over-fit -- a real effect worth measuring, not a
# bug in this script.
#
# Serialise before formatting; REPORT_ONLY=1 dry-runs the formatting on dummy data.
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization
const JLS   = "hybrid_ctrl_v2/results_v3/v3_noisytuned_eval.jls"
const NOISE = (101, 102, 103, 104, 105)
const NAMES = ("ASMC nt(s4)", "PID-CT nt(s4)", "PID-FB nt(s4)")
# clean-tuned counterparts, already evaluated on these same realisations
const PRIOR = (("ASMC nt(s4)",   "ASMC ct(s2)",   "hybrid_ctrl_v2/results_v3/asmc_v3_noisy_eval.jls", "ASMC (seed2)"),
               ("PID-CT nt(s4)", "PID-CT ct(s5)", "hybrid_ctrl_v2/results_v3/pid_v3_noisy_eval.jls",  "CT (seed5)"),
               ("PID-FB nt(s4)", "PID-FB ct(s3)", "hybrid_ctrl_v2/results_v3/pid_v3_noisy_eval.jls",  "FB (seed3)"))

function report(R, LAM, V_MAX_, CHATTER_REF_)
    for nm in NAMES
        r = R[nm]
        @printf("\n=== %s ===\n", nm)
        @printf("%-12s %10s %10s %10s %10s\n", "", "track_v3", "chatter", "ce", "SCORE")
        @printf("%-12s %10.5f %10.5f %10.3f %10.5f\n", "clean",
                r.clean.track, r.clean.chat, r.clean.ce, r.clean.sc)
        for (i, s) in enumerate(NOISE)
            q = r.noisy[i]
            @printf("%-12s %10.5f %10.5f %10.3f %10.5f\n", "noise $s", q.track, q.chat, q.ce, q.sc)
        end
        sc = [q.sc for q in r.noisy]
        @printf("%-12s %10s %10s %10s %10.5f  (std %.5f)\n", "mean", "", "", "", mean(sc), std(sc))
    end

    # Clean-tuned rows measured earlier on the SAME realisations.
    P = Dict{String,Any}()
    for (ntn, ctn, jls, key) in PRIOR
        isfile(jls) || continue
        D = deserialize(jls)
        haskey(D, key) && (P[ctn] = D[key])
    end

    println("\n=== v3 NOISY ROUND on sensor seeds 101-105 -- ALL SIX CONFIGS ===")
    println("score = tracking_v3 + 0.05*(ce/V_MAX) + 0.36*(chatter/CHATTER_REF), train14_v3")
    println("ct = clean-tuned, nt = noisy-tuned.  ABSOLUTE values (handoff decision 7).")
    @printf("%-14s %10s %10s %10s %10s %11s\n",
            "", "clean sc", "noisy sc", "track", "chatter", "realis.var")
    for (ntn, ctn, _, _) in PRIOR
        for (label, r) in ((ctn, get(P, ctn, nothing)), (ntn, get(R, ntn, nothing)))
            r === nothing && continue
            sc = [q.sc for q in r.noisy]
            @printf("%-14s %10.5f %10.5f %10.5f %10.5f %10.1f%%\n", label,
                    r.clean.sc, mean(sc), mean(q.track for q in r.noisy),
                    mean(q.chat for q in r.noisy), 100*std(sc)/mean(sc))
        end
    end

    # THE QUESTION: paired per-realisation, does noisy tuning beat clean tuning?
    println("\n=== DID NOISY TUNING HELP?  paired by realisation (negative = nt better) ===")
    @printf("%-16s %9s %9s %9s %9s %9s %10s\n", "", "n101", "n102", "n103", "n104", "n105", "mean")
    for (ntn, ctn, _, _) in PRIOR
        haskey(P, ctn) || continue
        d = [R[ntn].noisy[i].sc - P[ctn].noisy[i].sc for i in 1:5]
        verdict = all(x -> x < 0, d) ? "nt better 5/5" :
                  all(x -> x > 0, d) ? "ct better 5/5" : "MIXED"
        @printf("%-16s %+9.5f %+9.5f %+9.5f %+9.5f %+9.5f %+10.5f  %s\n",
                split(ntn)[1], d[1], d[2], d[3], d[4], d[5], mean(d), verdict)
    end

    println("\nWhat did noisy tuning cost on CLEAN operation?")
    for (ntn, ctn, _, _) in PRIOR
        haskey(P, ctn) || continue
        @printf("%-16s clean sc  ct=%.5f -> nt=%.5f   (%+.5f)\n",
                split(ntn)[1], P[ctn].clean.sc, R[ntn].clean.sc, R[ntn].clean.sc - P[ctn].clean.sc)
    end
end

if get(ENV, "REPORT_ONLY", "") == "1"
    dummy(x) = (track=x, ce=10x, chat=x/2, head=x/4, sc=2x)
    R = Dict(nm => (clean=dummy(0.1), noisy=[dummy(0.1 + 0.01i) for i in 1:5]) for nm in NAMES)
    report(R, 0.36, 24.0, 0.8)
    println("\nFORMAT DRY-RUN OK (no simulation performed)")
    exit(0)
end

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using JSON, StaticArrays
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
score(t, c, h) = t + 0.05*(c/V_MAX) + LAM*(h/CHATTER_REF)

function asmc_kw(path)
    g = JSON.parsefile(path)["best_gains"]
    (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
     rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
     use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
     kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false)
end
function pid_kw(path)
    g = JSON.parsefile(path)["best_gains"]
    (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
     N=g["N"], feedforward=g["feedforward"])
end
CFG = ((NAMES[1], :asmc, asmc_kw("hybrid_ctrl_v2/runs_asmc_v3/seed4/asmc_v2_noisy/best_config.json")),
       (NAMES[2], :pid,  pid_kw("hybrid_ctrl_v2/runs_pid_v3/seed4/pid_v2_ct_noisy/best_config.json")),
       (NAMES[3], :pid,  pid_kw("hybrid_ctrl_v2/runs_pid_v3/seed4/pid_v2_fb_noisy/best_config.json")))
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))

"aggregate one (config, oracle, seed) over the 14 trajectories"
function evalrun(c, kw, oracle, sd)
    t = Float64[]; ce = Float64[]; h = Float64[]; hp = Float64[]
    for tr in trs
        a, _, p = build_controller_v2(c, kw)
        probe, ref, mode, bus = run_controller_v2(c, oracle, tr;
            asmc_o=(c === :asmc ? a : nothing), pid_o=(c === :pid ? p : nothing), seed=sd)
        m2 = Main.controller_metrics(probe, ref, mode)
        m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
        SchedulerMod.clear_probe_log!(bus)
        push!(t, m3.tracking); push!(ce, m2.ce); push!(h, m2.chatter)
        q = m3.abs
        push!(hp, (q.final_head/Main.TOL.head_final + q.max_head/Main.TOL.head_max +
                   q.iae_head/StageObjectiveMod.TOL_V3.head_iae) / (6*q.k_traj))
    end
    (track=mean(t), ce=mean(ce), chat=mean(h), head=mean(hp), sc=score(mean(t), mean(ce), mean(h)))
end

R = Dict{String,Any}()
for (nm, c, kw) in CFG
    cl = evalrun(c, kw, :clean, 4)                   # seed 4 clean: the prior scripts' baseline
    @printf("done %s clean\n", nm); flush(stdout)
    ns = NamedTuple[]
    for s in NOISE
        push!(ns, evalrun(c, kw, :noisy, s))
        R[nm] = (clean=cl, noisy=copy(ns)); serialize(JLS, R)   # partial, crash-resilient
        @printf("done %s noise %d\n", nm, s); flush(stdout)
    end
end
serialize(JLS, R)
println("results serialised to $JLS")
report(R, LAM, V_MAX, CHATTER_REF)
