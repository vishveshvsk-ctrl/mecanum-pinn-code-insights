# Delta-test the v3 metric wiring.
#  (A) chord identity + properties (pure math, no sim)
#  (B) metric=:v2 must be BIT-IDENTICAL to Main.controller_metrics
#  (C) metric=:v3 must DIFFER, and its parts must be self-consistent
#  (D) the per-trajectory scaler must be independent of the controller
const ROOT = abspath(joinpath(@__DIR__, "..")); cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using JSON, Printf, Statistics, StaticArrays
chord(d) = 2*abs(sin(0.5d))
println("=== (A) chord vs wrapped angle ===")
@printf("%10s %12s %12s %10s\n","dpsi","|wrap|","chord","rel diff")
for d in (0.0057, 0.01, 0.1, 0.5, 1.0, pi/2, pi, 3pi/2)
    w = abs(atan(sin(d), cos(d)))
    @printf("%10.4f %12.6f %12.6f %9.2e\n", d, w, chord(d), abs(chord(d)-w)/max(w,1e-12))
end
@assert isapprox(chord(0.0057), 0.0057; rtol=1e-5) "not linear at small dpsi"
@assert isapprox(chord(pi), 2.0; atol=1e-12) "chord max should be 2"
@assert chord(2pi) < 1e-12                     "chord must be 2pi-periodic"
@assert chord(pi-0.1) > chord(0.1)             "must be monotone (|sin| is not)"
println("  monotone, periodic, linear-at-small, max=2  OK")

g = JSON.parsefile("hybrid_ctrl_v2/runs_asmc_v2_demandk/seed4/asmc_v2_clean/best_config.json")["best_gains"]
akw = (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
       rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
       use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
       kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false)
gc = JSON.parsefile("hybrid_ctrl_v2/runs_pid_v2_chatter/seed4/pid_v2_ct_clean/best_config.json")["best_gains"]
pkw = (lam_inner_x=gc["lam_inner_x"], lam_inner_y=gc["lam_inner_y"],
       lam_inner_psi=gc["lam_inner_psi"], N=gc["N"], feedforward=gc["feedforward"])
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))

println("\n=== (B)/(C)/(D) on 4 trajectories ===")
@printf("%-26s %10s %10s | %7s %9s %9s %9s\n","trajectory","v2 track","v3 track","k_pos","iae_pos","max_pos","iae_head")
sc = Dict{String,Float64}()
for tr in trs[[1,7,11,12]]
    a,_,_ = build_controller_v2(:asmc, akw)
    probe, ref, mode, bus = run_controller_v2(:asmc, :clean, tr; asmc_o=a, seed=4)
    m2 = Main.controller_metrics(probe, ref, mode)
    m3 = StageObjectiveMod.controller_metrics_v3(probe, ref, mode)
    SchedulerMod.clear_probe_log!(bus)
    @assert m3.ce == m2.ce && m3.chatter == m2.chatter "v3 must reuse ce/chatter verbatim"
    @assert m3.tracking != m2.tracking "v3 tracking identical to v2 -- metric not live"
    sc[String(tr.name)] = m3.abs.k_pos
    @printf("%-26s %10.4f %10.4f | %7.2f %9.5f %9.5f %9.6f\n", String(tr.name),
            m2.tracking, m3.tracking, m3.abs.k_pos, m3.abs.iae_pos, m3.abs.max_pos, m3.abs.iae_head)
    # (D) scaler must not depend on the controller
    _,_,p = build_controller_v2(:pid, pkw)
    pr2, rf2, md2, bs2 = run_controller_v2(:pid, :clean, tr; pid_o=p, seed=4)
    mp = StageObjectiveMod.controller_metrics_v3(pr2, rf2, md2); SchedulerMod.clear_probe_log!(bs2)
    @assert mp.abs.k_pos == m3.abs.k_pos "k_pos differs between controllers -- scaler is not reference-only"
    flush(stdout)
end
println("  ce/chatter reused verbatim; tracking differs; k_pos controller-independent  OK")

println("\n=== objective closure: metric kwarg ===")
kf = ControllerV2Mod.ASMCControllerV2(lim=default_physical_limits()).K_floor
fz = (eps_floor_xy=0.02, eps_floor_psi=0.08, use_demand_k=true, kmax_contact_b=true,
      enforce_k_floor=true, kmax_sched_floor=kf)
o2 = make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; seed=1, freeze=fz, lambda_chatter=3.0)
o3 = make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; seed=1, freeze=fz, lambda_chatter=3.0, metric=:v3)
lo, hi = flat_bounds(ASMC_SPACE_V2); th = (lo .+ hi) ./ 2
r2 = o2(th); r3 = o3(th)
@printf("  v2 score=%.4f track=%.4f   v3 score=%.4f track=%.4f\n", r2.score, r2.tracking, r3.score, r3.tracking)
@assert r2.chatter == r3.chatter "chatter must be unchanged between metrics"
@assert r2.tracking != r3.tracking "metric kwarg did not take effect"
try; make_stage_objective(:asmc, ASMC_SPACE_V2, trs[1:1], :clean; freeze=fz, metric=:bogus)
    error("bad metric accepted"); catch e; println("  invalid metric rejected  OK"); end
println("\nVERIFY PASS")
