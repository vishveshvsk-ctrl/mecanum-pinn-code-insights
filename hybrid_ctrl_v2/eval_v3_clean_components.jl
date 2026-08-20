# WHICH TERM costs ASMC the v3 clean comparison against PID?
# Best converged seed of each: ASMC seed2 (0.15233), PID-CT seed5 (0.14319),
# PID-FB seed3 (0.14901). Same objective all three were tuned under:
#   score = tracking + 0.05*(ce/V_MAX) + 0.36*(chatter/CHATTER_REF)
# decomposed into its three terms, then tracking into its six sub-terms, then
# per trajectory. Serialise before formatting (two earlier scripts lost full
# sweeps to a @printf typo).
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using JSON, Printf, Statistics, StaticArrays, Serialization
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
const T = Main.TOL
const V = StageObjectiveMod.TOL_V3

function asmc_kw(s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_asmc_v3/seed$s/asmc_v2_clean/best_config.json")["best_gains"]
    (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
     rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
     use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
     kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false)
end
function pid_kw(v, s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_pid_v3/seed$s/pid_v2_$(v)_clean/best_config.json")["best_gains"]
    (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
     N=g["N"], feedforward=g["feedforward"])
end
CFG = (("ASMC",   :asmc, asmc_kw(2)),
       ("PID-CT", :pid,  pid_kw("ct", 5)),
       ("PID-FB", :pid,  pid_kw("fb", 3)))
# Trajectory tier, overridable via the V3_TIER env var for the held-out
# :test_v3 evaluation. Outputs are tier-SUFFIXED whenever the tier is not the
# default, so a test run can never overwrite the train14_v3 results the campaign
# tables in RESULTS_v3.md are built from.
# NOTE scores are NOT comparable in magnitude across tiers -- k_traj and the
# trajectory mix both differ. Compare RANKING and each config's train->test delta.
const TIER = Symbol(get(ENV, "V3_TIER", "train14_v3"))
const TSFX = TIER === :train14_v3 ? "" : "_" * String(TIER)
trs = collect(trajset(TIER, "trajectory_files_run_0p5_main"))
NM  = [String(t.name) for t in trs]

R = Dict{String,Any}()
for (nm, c, kw) in CFG
    tk=Float64[]; ce=Float64[]; ch=Float64[]; pp=Any[]
    for tr in trs
        a, _, p = build_controller_v2(c, kw)
        probe, ref, mode, bus = run_controller_v2(c, :clean, tr;
            asmc_o=(c === :asmc ? a : nothing), pid_o=(c === :pid ? p : nothing), seed=4)
        m2 = Main.controller_metrics(probe, ref, mode)
        m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
        SchedulerMod.clear_probe_log!(bus)
        push!(tk, m3.tracking); push!(ce, m2.ce); push!(ch, m2.chatter); push!(pp, m3.abs)
    end
    R[nm] = (tk=tk, ce=ce, ch=ch, pp=pp)
    @printf("done %s\n", nm); flush(stdout)
end
serialize("hybrid_ctrl_v2/results_v3/three_way_components$(TSFX).jls", R)
println("serialised")

println("\n=== SCORE DECOMPOSITION (mean over 14 trajectories) ===")
@printf("%-8s %10s %10s %10s %10s\n", "", "tracking", "ce-term", "chat-term", "SCORE")
for (nm, _, _) in CFG
    r = R[nm]
    tt = mean(r.tk); cc = 0.05*mean(r.ce)/V_MAX; hh = LAM*mean(r.ch)/CHATTER_REF
    @printf("%-8s %10.5f %10.5f %10.5f %10.5f\n", nm, tt, cc, hh, tt+cc+hh)
end
println("\ndeltas vs PID-CT (positive = ASMC/FB worse):")
b = R["PID-CT"]
bt = mean(b.tk); bc = 0.05*mean(b.ce)/V_MAX; bh = LAM*mean(b.ch)/CHATTER_REF
for nm in ("ASMC", "PID-FB")
    r = R[nm]
    tt = mean(r.tk); cc = 0.05*mean(r.ce)/V_MAX; hh = LAM*mean(r.ch)/CHATTER_REF
    d = (tt-bt) + (cc-bc) + (hh-bh)
    @printf("%-8s dTrack=%+9.5f dCE=%+9.5f dChat=%+9.5f  total=%+9.5f\n", nm, tt-bt, cc-bc, hh-bh, d)
    @printf("%-8s   share of the gap: track %5.1f%%  ce %5.1f%%  chatter %5.1f%%\n",
            "", 100*(tt-bt)/d, 100*(cc-bc)/d, 100*(hh-bh)/d)
end
@printf("\nraw: ")
for (nm, _, _) in CFG
    @printf("%s ce=%.2f chat=%.4f   ", nm, mean(R[nm].ce), mean(R[nm].ch))
end
println()

println("\n=== TRACKING SUB-TERMS (mean, each already /(6*k_traj)) ===")
@printf("%-8s %9s %9s %9s %9s %9s %9s %9s %9s\n",
        "", "fin_pos", "max_pos", "iae_pos", "fin_hd", "max_hd", "iae_hd", "POS-sum", "HEAD-sum")
for (nm, _, _) in CFG
    p = R[nm].pp
    f(g) = mean(g(q)/(6*q.k_traj) for q in p)
    a1 = f(q->q.final_pos/T.pos_final); a2 = f(q->q.max_pos/T.pos_max);  a3 = f(q->q.iae_pos/V.pos_iae)
    b1 = f(q->q.final_head/T.head_final); b2 = f(q->q.max_head/T.head_max); b3 = f(q->q.iae_head/V.head_iae)
    @printf("%-8s %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f\n",
            nm, a1, a2, a3, b1, b2, b3, a1+a2+a3, b1+b2+b3)
end

println("\n=== PER-TRAJECTORY tracking ===")
@printf("%-28s %10s %10s %10s %12s\n", "trajectory", "ASMC", "PID-CT", "PID-FB", "ASMC-CT")
for i in eachindex(NM)
    a = R["ASMC"].tk[i]; c = R["PID-CT"].tk[i]; f = R["PID-FB"].tk[i]
    @printf("%-28s %10.5f %10.5f %10.5f %+12.5f\n", NM[i], a, c, f, a-c)
end
