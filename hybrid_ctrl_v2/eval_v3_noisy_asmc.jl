# NOISY evaluation (not a retune) of the best converged v3 clean ASMC config.
#   ASMC best = seed2 (0.15233)
# The missing third row of the v3 noisy round: `hybrid_ctrl_v2/eval_v3_noisy_pid.jl`
# already produced PID-FB (seed3) and PID-CT (seed5) over the SAME five sensor
# seeds 101-105, under the SAME objective (v3 metric, lambda_chatter = 0.36,
# train14_v3). Nothing here may differ from that script except the controller,
# or the three rows do not belong in one table.
#
# WHAT THIS TESTS (v3_campaign_handoff.md §4): the clean gap ASMC-vs-CT is 72.7%
# tracking / 25.0% chatter, but the noisy PID scores are 64% CHATTER -- and
# ASMC's chatter penalty is the highest of the three and UNIFORM across
# trajectories (+0.0024 over CT on every slice). Prediction: ASMC loses further
# under noise for a chatter reason unrelated to its tracking concentration on
# coupled_vomega_stress. A tracking-dominated noisy gap would falsify it.
#
# The sensor seed is fixed per run (run_controller_v2 sets cfg.sensor_seed and
# the OracleEstimator seed from it), so each of the 5 is distinct but
# deterministic; the spread across them is REALISATION VARIANCE -- the statistic
# that flagged ASMC as fragile in the v2-era campaign (24.9%).
#
# STRUCTURE, deliberately (same as the PID script): compute -> SERIALISE ->
# report, with `report` a function exercisable on dummy data via REPORT_ONLY=1
# WITHOUT running any simulation. Two earlier runs of the PID script completed
# all 168 simulations and then lost every number to a @printf argument-count
# typo. Compute must survive a formatting bug, and formatting must be testable
# without paying for compute.
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization
const JLS     = "hybrid_ctrl_v2/results_v3/asmc_v3_noisy_eval.jls"
const PID_JLS = "hybrid_ctrl_v2/results_v3/pid_v3_noisy_eval.jls"     # for the combined round table
const NOISE   = (101, 102, 103, 104, 105)
const CFGNAMES = ("ASMC (seed2)",)

"one controller block: clean baseline, the 5 realisations, mean/std/degradation"
function report_one(nm, r)
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

function report(R, LAM, V_MAX_, CHATTER_REF_)
    for nm in CFGNAMES
        report_one(nm, R[nm])
    end

    # The whole point of this run: ASMC's row next to the two PID rows the
    # earlier script produced over the identical realisations. If the PID .jls
    # is missing the ASMC block above still stands on its own.
    ALL = Dict{String,Any}(nm => R[nm] for nm in CFGNAMES)
    order = collect(CFGNAMES)
    if isfile(PID_JLS)
        P = deserialize(PID_JLS)
        for nm in ("CT (seed5)", "FB (seed3)")
            haskey(P, nm) || continue
            ALL[nm] = P[nm]; push!(order, nm)
        end
    else
        @printf("\n[warn] %s absent -- PID rows omitted from the combined table\n", PID_JLS)
    end

    println("\n=== v3 NOISY ROUND, all controllers, seeds 101-105 ===")
    println("objective: tracking_v3 + 0.05*(ce/V_MAX) + 0.36*(chatter/CHATTER_REF), train14_v3")
    println("ABSOLUTE values -- do NOT quote % degradation (handoff decision 7).")
    @printf("%-14s %10s %10s %10s %10s %11s %11s\n",
            "", "clean sc", "noisy sc", "track", "chatter", "realis.var", "head share")
    for nm in order
        r = ALL[nm]; sc = [q.sc for q in r.noisy]; tk = [q.track for q in r.noisy]
        @printf("%-14s %10.5f %10.5f %10.5f %10.5f %10.1f%% %10.1f%%\n", nm,
                r.clean.sc, mean(sc), mean(tk), mean(q.chat for q in r.noisy),
                100*std(sc)/mean(sc), 100*mean(q.head/q.track for q in r.noisy))
    end

    # Where the noisy score sits: the prediction under test is "chatter, not
    # tracking". Split each mean noisy score into its three priced terms.
    println("\n=== noisy SCORE composition (mean over the 5 realisations) ===")
    @printf("%-14s %10s %10s %10s %10s\n", "", "track", "ce-term", "chat-term", "SCORE")
    for nm in order
        r = ALL[nm]
        tt = mean(q.track for q in r.noisy)
        cc = 0.05*mean(q.ce for q in r.noisy)/V_MAX_
        hh = LAM*mean(q.chat for q in r.noisy)/CHATTER_REF_
        @printf("%-14s %10.5f %10.5f %10.5f %10.5f\n", nm, tt, cc, hh, tt+cc+hh)
        @printf("%-14s %9.1f%% %9.1f%% %9.1f%%\n", "  share",
                100*tt/(tt+cc+hh), 100*cc/(tt+cc+hh), 100*hh/(tt+cc+hh))
    end

    # The open question, answered directly: is the ASMC-vs-CT noisy gap a
    # chatter gap or a tracking gap?
    if haskey(ALL, "CT (seed5)")
        a = ALL[CFGNAMES[1]]; c = ALL["CT (seed5)"]
        term(r) = (mean(q.track for q in r.noisy),
                   0.05*mean(q.ce for q in r.noisy)/V_MAX_,
                   LAM*mean(q.chat for q in r.noisy)/CHATTER_REF_)
        at, ac, ah = term(a); ct, cc, ch = term(c)
        d = (at-ct) + (ac-cc) + (ah-ch)
        println("\n=== ASMC vs PID-CT, NOISY (positive = ASMC worse) ===")
        @printf("dTrack=%+9.5f  dCE=%+9.5f  dChat=%+9.5f   total=%+9.5f\n",
                at-ct, ac-cc, ah-ch, d)
        @printf("share of the gap: track %5.1f%%   ce %5.1f%%   chatter %5.1f%%\n",
                100*(at-ct)/d, 100*(ac-cc)/d, 100*(ah-ch)/d)
        println("clean reference (three_way_components): track 72.7%, chatter 25.0%, ce 2.3%")
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
using JSON, StaticArrays
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
score(t, c, h) = t + 0.05*(c/V_MAX) + LAM*(h/CHATTER_REF)

# Same decode as hybrid_ctrl_v2/eval_v3_clean_components.jl -- the flags are stored in
# best_config.json but the derived-parameter defaults (use_cubic) are not.
function asmc_kw(s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_asmc_v3/seed$s/asmc_v2_clean/best_config.json")["best_gains"]
    (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
     rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
     use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
     kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false)
end
CFG = (CFGNAMES[1] => asmc_kw(2),)
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))

"aggregate one (config, oracle, seed) over the 14 trajectories"
function evalrun(kw, oracle, sd)
    t = Float64[]; c = Float64[]; h = Float64[]; hp = Float64[]
    for tr in trs
        a, _, _ = build_controller_v2(:asmc, kw)
        probe, ref, mode, bus = run_controller_v2(:asmc, oracle, tr; asmc_o=a, seed=sd)
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
    cl = evalrun(kw, :clean, 4)                      # seed 4: the clean baseline
    @printf("done %s clean\n", nm); flush(stdout)    # the PID script's seed too
    ns = NamedTuple[]
    for s in NOISE
        push!(ns, evalrun(kw, :noisy, s))
        serialize(JLS, Dict(nm => (clean=cl, noisy=copy(ns))))   # partial, resumable read
        @printf("done %s noise %d\n", nm, s); flush(stdout)
    end
    R[nm] = (clean=cl, noisy=ns)
end
serialize(JLS, R)                      # BEFORE any formatting
println("results serialised to $JLS")
report(R, LAM, V_MAX, CHATTER_REF)
