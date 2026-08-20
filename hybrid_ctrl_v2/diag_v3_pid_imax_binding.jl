# IS `I_max` A LIVE PARAMETER? -- i.e. would the optimizer have any gradient on it?
#
# In PID v2 `I_max` is DERIVED, not searched (PID_SPACE_V2 is lam_inner_{x,y,psi}
# only). imax_from_measured sets
#     I_max = wrench_budget ./ Ki,   Ki = d_eff ./ lam_inner
# so the integral's actual wrench contribution is clamped at
#     Ki .* I_max = wrench_budget      <- INVARIANT of lam_inner
# The bound therefore only has an effect on ticks where the clamp ACTUALLY FIRES.
# If it never fires, it is a dead dimension (the tau_ceiling / Kd_pos precedent:
# a quarter of a search budget spent on a zero-gradient coordinate).
#
# Its own docstring flags the CT value as unfinished -- "EMPIRICAL ADJUSTMENT ...
# still needs re-checking once real tuning happens", the doubling 2.389 -> 4.778
# never rechecked -- and pid_ct_heading_weakness_handoff.md §3 names a too-tight
# yaw integral bound as candidate mechanism #1 for CT's heading collapse on
# sustained-yaw trajectories. Both questions are answered by the same measurement.
#
# Reports PER AXIS (x, y, psi), per trajectory, on the CONVERGED v3 configs:
#   frac_sat  fraction of ticks at the clamp   -> is there any gradient at all
#   p95/max   integral utilisation |I|/I_max   -> headroom if it never saturates
# `i_util_log` (added behind the existing default-off `log_diag`) is what makes
# the per-axis split possible; `i_sat_log` is an `any` over the three axes and
# cannot attribute saturation to yaw.
#
# Serialise before formatting; REPORT_ONLY=1 dry-runs the formatting.
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, Statistics, Serialization
const JLS = "hybrid_ctrl_v2/results_v3/pid_imax_binding.jls"
const AX  = ("x", "y", "psi")
const CFGNAMES = ("PID-CT (seed5)", "PID-FB (seed3)")
const SATTOL = 1 - 1e-6

function report(R, NM)
    for nm in CFGNAMES
        r = R[nm]
        @printf("\n=== %s ===   I_max = %s  (Ki*I_max = %s N*m, the invariant)\n", nm,
                string(round.(r.I_max, digits=4)), string(round.(r.budget, digits=4)))
        @printf("%-26s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "trajectory",
                "sat_x", "sat_y", "sat_psi", "p95_x", "p95_y", "p95_psi",
                "max_x", "max_y", "max_psi")
        for i in eachindex(NM)
            s = r.sat[i]; p = r.p95[i]; m = r.mx[i]
            @printf("%-26s %7.1f%% %7.1f%% %7.1f%% %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
                    NM[i], 100*s[1], 100*s[2], 100*s[3], p[1], p[2], p[3], m[1], m[2], m[3])
        end
        @printf("%-26s %7.1f%% %7.1f%% %7.1f%% %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", "ALL (mean / max)",
                100*mean(s[1] for s in r.sat), 100*mean(s[2] for s in r.sat), 100*mean(s[3] for s in r.sat),
                mean(p[1] for p in r.p95), mean(p[2] for p in r.p95), mean(p[3] for p in r.p95),
                maximum(m[1] for m in r.mx), maximum(m[2] for m in r.mx), maximum(m[3] for m in r.mx))
    end

    println("\n=== VERDICT: does the optimizer have a gradient on I_max? ===")
    for nm in CFGNAMES
        r = R[nm]
        for a in 1:3
            tot = sum(s[a] for s in r.sat) / length(r.sat)
            top = maximum(m[a] for m in r.mx)
            live = tot > 0
            @printf("%-16s %-4s  frac_sat=%6.3f%%  max_util=%.3f  -> %s\n", nm, AX[a],
                    100*tot, top,
                    live ? "LIVE (clamp fires; a bound here changes the trajectory)" :
                           "DEAD (clamp never fires; zero gradient, headroom $(round(100*(1-top),digits=1))%)")
        end
    end
end

if get(ENV, "REPORT_ONLY", "") == "1"
    d(x) = [ [x, x/2, x/3] for _ in 1:3 ]
    NM = ["traj_a", "traj_b", "traj_c"]
    R = Dict(nm => (I_max=[1.0,2.0,3.0], budget=[4.0,5.0,6.0],
                    sat=d(0.1), p95=d(0.5), mx=d(0.9)) for nm in CFGNAMES)
    report(R, NM)
    println("\nFORMAT DRY-RUN OK (no simulation performed)")
    exit(0)
end

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
using JSON

function pid_kw(v, s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_pid_v3/seed$s/pid_v2_$(v)_clean/best_config.json")["best_gains"]
    (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"], lam_inner_psi=g["lam_inner_psi"],
     N=g["N"], feedforward=g["feedforward"], log_diag=true)
end
CFG = (CFGNAMES[1] => pid_kw("ct", 5), CFGNAMES[2] => pid_kw("fb", 3))
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))
NM  = [String(t.name) for t in trs]

R = Dict{String,Any}()
for (nm, kw) in CFG
    sat = Vector{Float64}[]; p95 = Vector{Float64}[]; mx = Vector{Float64}[]
    Imax = zeros(3); budget = zeros(3)
    for tr in trs
        _, _, p = build_controller_v2(:pid, kw)
        _, _, _, bus = run_controller_v2(:pid, :clean, tr; pid_o=p, seed=4)
        U = p.i_util_log                              # per tick, per axis, |I|/I_max
        push!(sat, [mean(u[a] >= SATTOL for u in U) for a in 1:3])
        push!(p95, [quantile([u[a] for u in U], 0.95) for a in 1:3])
        push!(mx,  [maximum(u[a] for u in U) for a in 1:3])
        Imax .= p.I_max; budget .= p.Ki .* p.I_max    # the invariant, for the header
        SchedulerMod.clear_probe_log!(bus)
    end
    R[nm] = (I_max=Imax, budget=budget, sat=sat, p95=p95, mx=mx)
    @printf("done %s\n", nm); flush(stdout)
end
serialize(JLS, R); serialize("hybrid_ctrl_v2/results_v3/pid_imax_binding_names.jls", NM)
println("serialised to $JLS")
report(R, NM)
