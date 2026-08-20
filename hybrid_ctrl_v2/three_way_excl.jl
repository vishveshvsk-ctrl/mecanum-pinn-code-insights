# Re-slice the three-way comparison with trajectories excluded. Reads the
# serialised per-trajectory data from hybrid_ctrl_v2/results_v3/three_way_components.jls -- NO
# re-simulation, so the numbers are identical to the full-set run by construction.
#
# coupled_vomega_stress carried 66% of the ASMC-vs-PID-CT tracking gap and
# spiral_orbit_stress a further 12%. The question is whether ASMC's deficit is a
# broad property or those two cases.
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using Printf, Statistics, Serialization
const LAM = StageObjectiveMod.LAMBDA_CHATTER_V3
const T = Main.TOL
const V = StageObjectiveMod.TOL_V3
R  = deserialize("hybrid_ctrl_v2/results_v3/three_way_components.jls")
NM = [String(t.name) for t in trajset(:train14_v3, "trajectory_files_run_0p5_main")]
CFG = ("ASMC", "PID-CT", "PID-FB")

function slice(drop::Vector{String}, label::String)
    idx = [i for i in eachindex(NM) if !(NM[i] in drop)]
    @printf("\n=== %s  (%d trajectories) ===\n", label, length(idx))
    @printf("%-8s %10s %10s %10s %10s %10s\n", "", "tracking", "ce-term", "chat-term", "SCORE", "vs CT")
    base = 0.0
    scores = Dict{String,Float64}()
    for nm in CFG
        r = R[nm]
        tt = mean(r.tk[idx]); cc = 0.05*mean(r.ce[idx])/V_MAX; hh = LAM*mean(r.ch[idx])/CHATTER_REF
        scores[nm] = tt + cc + hh
        nm == "PID-CT" && (base = scores[nm])
    end
    for nm in CFG
        r = R[nm]
        tt = mean(r.tk[idx]); cc = 0.05*mean(r.ce[idx])/V_MAX; hh = LAM*mean(r.ch[idx])/CHATTER_REF
        @printf("%-8s %10.5f %10.5f %10.5f %10.5f %9.2f%%\n",
                nm, tt, cc, hh, scores[nm], 100*(scores[nm]/base - 1))
    end
    # gap attribution vs PID-CT
    b = R["PID-CT"]
    bt = mean(b.tk[idx]); bc = 0.05*mean(b.ce[idx])/V_MAX; bh = LAM*mean(b.ch[idx])/CHATTER_REF
    for nm in ("ASMC", "PID-FB")
        r = R[nm]
        tt = mean(r.tk[idx]); cc = 0.05*mean(r.ce[idx])/V_MAX; hh = LAM*mean(r.ch[idx])/CHATTER_REF
        d = (tt-bt) + (cc-bc) + (hh-bh)
        @printf("  %-7s gap %+8.5f  =  track %+8.5f (%5.1f%%)  chat %+8.5f (%5.1f%%)  ce %+8.5f\n",
                nm, d, tt-bt, 100*(tt-bt)/d, hh-bh, 100*(hh-bh)/d, cc-bc)
    end
    # position vs heading
    @printf("  %-7s %9s %9s\n", "", "POS-sum", "HEAD-sum")
    for nm in CFG
        p = R[nm].pp[idx]
        f(g) = mean(g(q)/(6*q.k_traj) for q in p)
        P = f(q->q.final_pos/T.pos_final) + f(q->q.max_pos/T.pos_max) + f(q->q.iae_pos/V.pos_iae)
        H = f(q->q.final_head/T.head_final) + f(q->q.max_head/T.head_max) + f(q->q.iae_head/V.head_iae)
        @printf("  %-7s %9.5f %9.5f\n", nm, P, H)
    end
end

slice(String[], "ALL 14 (reference)")
slice(["coupled_vomega_stress"], "EXCLUDING coupled_vomega_stress")
slice(["coupled_vomega_stress", "spiral_orbit_stress"], "EXCLUDING both stress outliers")
