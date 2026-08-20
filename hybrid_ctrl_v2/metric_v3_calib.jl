# v3 tracking metric: per-trajectory scaler + time-normalised integral term.
#
#   S_pos_i  = max(radius_var(ref_i), TOL.pos_max)     # 0.1 m floor
#   S_head_i = max(dpsi_var(ref_i),   TOL.head_max)    # 0.1 rad floor
#   k_pos_i  = S_pos_i / TOL.pos_max                   # >= 1, the trajectory factor
#
# The three position terms keep their existing 1:10 tolerance RATIO (final 1 cm,
# max 10 cm) and are all scaled by k_pos_i, so `max_pos/(TOL.pos_max*k) ==
# max_pos/S_pos` exactly as specified, while `final` keeps its tighter standard.
# Same structure for heading. iae is the TIME-NORMALISED integral, mean(|e|),
# which for uniform probe sampling IS (1/T)*integral|e|dt.
#
# radius_var := max radial excursion of the reference path from its own centroid.
#
# TOL.pos_iae / TOL.head_iae are NOT guessed -- this script measures what the
# converged controllers actually achieve so the new term lands comparable in
# magnitude to final/max. Precedent for why: `chatter` was normalised by V_MAX=24
# (a voltage dividing a slew rate), leaving it ~30x too weak, and every run before
# that was found was effectively chatter-unpriced.
const ROOT = abspath(joinpath(@__DIR__, "..")); cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
using JSON, Printf, Statistics, StaticArrays

_wrap(a) = atan(sin(a), cos(a))
"raw per-channel error stats + the reference-derived scalers"
function metric_parts(probe, ref)
    pe = Float64[]; he = Float64[]; xr = Float64[]; yr = Float64[]; pr = Float64[]
    for p in probe
        t = p.t
        rx = ref.xo(t); ry = ref.yo(t); rp = ref.psi(t)
        push!(xr, rx); push!(yr, ry); push!(pr, rp)
        push!(pe, sqrt((p.u[17]-rx)^2 + (p.u[18]-ry)^2))
        push!(he, abs(_wrap(p.u[4]-rp)))
    end
    cx = mean(xr); cy = mean(yr)
    rad  = sqrt(maximum((xr .- cx).^2 .+ (yr .- cy).^2))
    dpsi = maximum(pr) - minimum(pr)
    (final_pos=pe[end], max_pos=maximum(pe), iae_pos=mean(pe),
     final_head=he[end], max_head=maximum(he), iae_head=mean(he),
     rad=rad, dpsi=dpsi)
end

function cfg_asmc(s)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_asmc_v2_demandk/seed$s/asmc_v2_clean/best_config.json")["best_gains"]
    (:asmc, (lam_x_max=g["lam_x_max"], lam_y_max=g["lam_y_max"], lam_psi_max=g["lam_psi_max"],
     rho_auth=g["rho_auth"], eps_floor_xy=g["eps_floor_xy"], eps_floor_psi=g["eps_floor_psi"],
     use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
     kmax_sched_floor=SVector{3,Float64}(g["kmax_sched_floor"]), use_cubic=false))
end
function cfg_pid(v)
    g = JSON.parsefile("hybrid_ctrl_v2/runs_pid_v2_chatter/seed4/pid_v2_$(v)_clean/best_config.json")["best_gains"]
    (:pid, (lam_inner_x=g["lam_inner_x"], lam_inner_y=g["lam_inner_y"],
            lam_inner_psi=g["lam_inner_psi"], N=g["N"], feedforward=g["feedforward"]))
end
CONFIGS = vcat([("ASMC s$s", cfg_asmc(s)...) for s in 1:5],
               [("PID-CT", cfg_pid("ct")...), ("PID-FB", cfg_pid("fb")...)])
trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))

R = Dict{String,Vector{Any}}()
for (nm,c,kw) in CONFIGS
    parts = Any[]
    for tr in trs
        a,_,p = build_controller_v2(c, kw)
        probe, ref, mode, bus = run_controller_v2(c, :clean, tr;
            asmc_o=(c==:asmc ? a : nothing), pid_o=(c==:pid ? p : nothing), seed=4)
        push!(parts, metric_parts(probe, ref)); SchedulerMod.clear_probe_log!(bus)
    end
    R[nm] = parts
    @printf("done %s\n", nm); flush(stdout)
end

println("\n=== per-trajectory scalers (reference-derived, config-independent) ===")
@printf("%-28s %8s %8s | %8s %8s\n","trajectory","radius","S_pos","dpsi","S_head")
p1 = R["PID-CT"]
for (i,tr) in enumerate(trs)
    q = p1[i]
    @printf("%-28s %8.3f %8.3f | %8.3f %8.3f  %s\n", String(tr.name), q.rad,
            max(q.rad, TOL.pos_max), q.dpsi, max(q.dpsi, TOL.head_max),
            (q.rad < TOL.pos_max || q.dpsi < TOL.head_max) ? "<- floor binds" : "")
end

println("\n=== achieved magnitudes, pooled over 7 configs x 14 trajectories ===")
for f in (:final_pos,:max_pos,:iae_pos,:final_head,:max_head,:iae_head)
    v = [getfield(q,f) for nm in keys(R) for q in R[nm]]
    @printf("  %-11s p50=%.5f  p90=%.5f  max=%.5f\n", f, quantile(v,0.5), quantile(v,0.9), maximum(v))
end
# ratio iae/max tells us the natural tolerance ratio
rp = [q.iae_pos/max(q.max_pos,1e-12) for nm in keys(R) for q in R[nm]]
rh = [q.iae_head/max(q.max_head,1e-12) for nm in keys(R) for q in R[nm]]
@printf("\n  iae_pos /max_pos   p50=%.3f  (=> TOL.pos_iae  ~ %.4f if matched to pos_max=%.2f)\n",
        quantile(rp,0.5), quantile(rp,0.5)*TOL.pos_max, TOL.pos_max)
@printf("  iae_head/max_head  p50=%.3f  (=> TOL.head_iae ~ %.4f if matched to head_max=%.2f)\n",
        quantile(rh,0.5), quantile(rh,0.5)*TOL.head_max, TOL.head_max)
using Serialization; serialize("hybrid_ctrl_v2/results_v3/metric_v3_parts.jls", (R=R, names=[String(t.name) for t in trs]))
println("\nraw parts saved to hybrid_ctrl_v2/results_v3/metric_v3_parts.jls")
