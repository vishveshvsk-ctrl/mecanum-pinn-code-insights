# Final verification of the v3 tiers, every reference over its OWN ref.T_total.
const ROOT = abspath(joinpath(@__DIR__, "..")); cd(ROOT)
include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
using Printf, Statistics, Random, TOML
lim = default_physical_limits(); mot = Main.PlantMod.MotorParams()
const PHI_NL = mot.V_max/(mot.Kb*mot.G); const L = lim.l + lim.h
const RD = "trajectory_files_run_0p5_main"
function mkref(tr)
    prof = TOML.parsefile(joinpath(tr.config_dir,"profiles",tr.profile_toml))["profile"]
    r = Profiles.build(prof["builder"], Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0)))
    get(tr,:adapt,false) ? Profiles.velref_to_posref(r) : r
end
function audit(r; n=4000)
    T=r.T_total; m,ms,Is,aX,aY = lim.m,lim.ms,lim.Is,lim.aX,lim.aY
    wh=0.0; ut=0.0; vx=0.0; vy=0.0; wz=0.0
    for t in range(0,T;length=n)
        psi=r.psi(t); Vx,Vy,om = Profiles.global_to_local_frame(t,psi,r.Vxo,r.Vyo,r.om)
        ax,ay,al = Profiles.global_to_local_frame(t,psi,r.Axo,r.Ayo,r.al)
        vx=max(vx,abs(Vx)); vy=max(vy,abs(Vy)); wz=max(wz,abs(om))
        for (sy,so) in ((-1,-1),(1,1),(1,-1),(-1,1)); wh=max(wh,abs(Vx+sy*Vy+so*L*om)/lim.R); end
        b_x = ms*ax - m*aY*al - ms*om*Vy - m*aX*om^2
        b_y = ms*ay + m*aX*al + ms*om*Vx - m*aY*om^2 - 110.0*Vy
        b_O = -m*aY*ax + m*aX*ay + Is*al + m*om*(aX*Vx+aY*Vy) - 2.20*om
        ut = max(ut, sqrt((0.354*(b_x+b_y)-0.918*b_O)^2 + (lim.kappa*(Vy-lim.h*om))^2)/lim.mu_N3)
    end
    (T=T, w=100*wh/PHI_NL, u=ut, vx=vx, vy=vy, wz=wz)
end
function report(tier)
    set = trajset(tier, RD)
    @printf("\n=== %s (%d entries) ===\n%-28s %7s %8s %8s | %8s %7s  %s\n", tier, length(set),
            "trajectory","T_total","max|Vy|","max|wz|","wheel %%","util","flag")
    println("-"^92)
    ww=0.0; wn=""; uu=0.0
    for tr in set
        a = audit(mkref(tr))
        if a.w > ww; ww=a.w; wn=String(tr.name); end
        uu = max(uu, a.u)
        @printf("%-28s %7.2f %8.3f %8.3f | %7.1f%% %7.3f  %s\n", String(tr.name), a.T, a.vy, a.wz, a.w, a.u,
                a.w>=85 ? "<<< VEL-INFEASIBLE" : (a.w>70 ? "(marginal)" : (a.u>1 ? "(over circle)" : "")))
    end
    @printf("worst: wheel %.1f%% (%s)   util %.3f\n", ww, wn, uu)
    (ww, uu)
end
w1,_ = report(:train14_v3); w2,_ = report(:test_v3)
# additivity: originals untouched
o12 = trajset(:train12, RD); ot = trajset(:test, RD)
@assert only(t.combo_idx for t in o12 if String(t.name)=="spiral_orbit_stress")==37
@assert only(t.combo_idx for t in o12 if String(t.name)=="ellipse_stress_tangent")==55
@assert only(t.combo_idx for t in ot  if String(t.name)=="long_circle_profile_stress")==102
println("\n:train12 / :test untouched (37 / 55 / 102)  OK")
n14 = trajset(:train14_v3, RD)
@assert only(t.combo_idx for t in n14 if String(t.name)=="spiral_orbit_stress")==198
@assert length(n14)==14 && length(trajset(:test_v3,RD))==8
@assert w1 < 70.0 "train14_v3 worst wheel $(w1)% exceeds the 70% target"
@assert w2 < 70.0 "test_v3 worst wheel $(w2)% exceeds the 70% target"
@printf("\nworst wheel: train14_v3 %.1f%%   test_v3 %.1f%%   (both < 70%%)\n", w1, w2)
println("VERIFY PASS")
