# Per-trajectory metric scalers on the FINAL v3 tiers, each over its own T_total.
#   radius_var = max || p_ref(t) - centroid ||      centroid = time-mean of p_ref
#   dpsi_var   = max psi_ref - min psi_ref          (unwrapped)
#   S_pos = max(radius_var, TOL.pos_max), S_head = max(dpsi_var, TOL.head_max)
# Also reports max|psi - mean(psi)| as the TRUE analogue of the radius definition,
# so the heading choice (cumulative range vs deviation) can be compared.
const ROOT = abspath(joinpath(@__DIR__, "..")); cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
using Printf, Statistics, Random, TOML
function mkref(tr)
    prof = TOML.parsefile(joinpath(tr.config_dir,"profiles",tr.profile_toml))["profile"]
    r = Profiles.build(prof["builder"], Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0)))
    get(tr,:adapt,false) ? Profiles.velref_to_posref(r) : r
end
for tier in (:train14_v3, :test_v3)
    @printf("\n=== %s ===\n%-28s %7s | %8s %8s %6s | %8s %8s %8s %6s\n", tier,
            "trajectory","T_total","radius","S_pos","k_pos","dpsi_rng","dpsi_dev","S_head","k_head")
    println("-"^104)
    for tr in trajset(tier, "trajectory_files_run_0p5_main")
        r = mkref(tr); T = r.T_total; ts = range(0,T;length=4000)
        x = [r.xo(t) for t in ts]; y = [r.yo(t) for t in ts]; p = [r.psi(t) for t in ts]
        cx, cy = mean(x), mean(y)
        rad  = sqrt(maximum((x .- cx).^2 .+ (y .- cy).^2))
        rng_ = maximum(p) - minimum(p)
        dev  = maximum(abs.(p .- mean(p)))
        Sp, Sh = max(rad, TOL.pos_max), max(rng_, TOL.head_max)
        @printf("%-28s %7.2f | %8.3f %8.3f %6.1f | %8.3f %8.3f %8.3f %6.1f  %s\n",
                String(tr.name), T, rad, Sp, Sp/TOL.pos_max, rng_, dev, Sh, Sh/TOL.head_max,
                (rad < TOL.pos_max || rng_ < TOL.head_max) ? "<- floor" : "")
    end
end
println("""

k_pos = S_pos/TOL.pos_max and k_head = S_head/TOL.head_max are the per-trajectory
weights the scaler applies. k=1 means the floor bound and the entry is scored on
the ABSOLUTE tolerance; k>1 means the trajectory's own size took over.""")
