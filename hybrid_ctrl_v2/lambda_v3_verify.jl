# Verify LAMBDA_CHATTER_V3 = 0.36 and its guard.
const ROOT = abspath(joinpath(@__DIR__, "..")); cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using Printf, StaticArrays
L = StageObjectiveMod.LAMBDA_CHATTER_V3
@printf("LAMBDA_CHATTER_V3 = %.3f\n\n", L)
# (1) the balance claim: does 0.36 reproduce v2's 29.9% tracking share?
tr2, tr3, ch = 0.21824, 0.02603, 0.13675     # measured at lam_psi=30 (converged 32.5)
sh(t, lam) = 100t / (t + lam*ch/CHATTER_REF)
@printf("at the converged operating point (chatter=%.5f):\n", ch)
@printf("  v2 lambda=3.00  track=%.5f  chatter-term=%.5f  -> tracking %.1f%%\n", tr2, 3.0*ch/CHATTER_REF, sh(tr2,3.0))
@printf("  v3 lambda=%.2f  track=%.5f  chatter-term=%.5f  -> tracking %.1f%%\n", L, tr3, L*ch/CHATTER_REF, sh(tr3,L))
@printf("  v3 lambda=3.00 (WRONG) track=%.5f chatter-term=%.5f -> tracking %.1f%%\n", tr3, 3.0*ch/CHATTER_REF, sh(tr3,3.0))
@assert abs(sh(tr3,L) - sh(tr2,3.0)) < 1.0 "0.36 does not reproduce the v2 balance"
println("  balance preserved to within 1 percentage point  OK")
# (2) the guard
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))
kf = ControllerV2Mod.ASMCControllerV2(lim=default_physical_limits()).K_floor
fz = (eps_floor_xy=0.02, eps_floor_psi=0.08, use_demand_k=true, kmax_contact_b=true,
      enforce_k_floor=true, kmax_sched_floor=kf)
println("\nguard:")
fired = false
try; make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; freeze=fz, lambda_chatter=3.0, metric=:v3)
catch e; global fired = true; println("  v3 + lambda=3.0 REJECTED (correct)"); end
@assert fired "guard did not fire on the v2 lambda under v3"
make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; freeze=fz, lambda_chatter=L, metric=:v3)
println("  v3 + lambda=0.36 accepted (correct)")
make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; freeze=fz, lambda_chatter=3.0, metric=:v2)
println("  v2 + lambda=3.0 accepted -- existing callers unaffected (correct)")
# (3) end-to-end score
o = make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; seed=1, freeze=fz,
                         lambda_chatter=L, metric=:v3)
lo, hi = flat_bounds(ASMC_SPACE_V2); r = o((lo .+ hi) ./ 2)
@printf("\nend-to-end: score=%.4f track=%.4f chatter=%.4f  -> tracking %.0f%% of the two\n",
        r.score, r.tracking, r.chatter, 100*r.tracking/(r.tracking + L*r.chatter/CHATTER_REF))
@assert isfinite(r.score) && r.score < 1e5 "objective returned the divergence sentinel"
println("\nVERIFY PASS")
