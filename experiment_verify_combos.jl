#!/usr/bin/env julia
# Verify whether the PINNED combos (octagon 2, ellipse 1, spin_creep 7) are the
# hardest-FEASIBLE ones, by sweeping combos with the tuned ASMC (clean oracle)
# and reporting peak demand + tracking offset per combo.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")
using JSON, StaticArrays, Statistics

const DIR = "trajectory_files_run_0p5_main"
asmc_kw = let g = JSON.parse(read("runs_controller/asmc_clean/best_config.json", String))["best_gains"]
    (; (Symbol(k) => (v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...)
end

function offset(profile, combo, mode)
    tr = (name=split(profile,"_")[1], profile_toml=profile, combo_idx=combo, mu=0.5, config_dir=DIR, run_mode=mode)
    probe, ref, m = run_controller(:asmc, asmc_kw, :clean, tr; seed=42)
    ts = [p.t for p in probe]
    mm = controller_metrics(probe, ref, m)
    if mode == :pose
        sp = maximum(sqrt(ref.Vxo(t)^2 + ref.Vyo(t)^2) for t in ts)
        return (demand="peakSpd=$(round(sp,digits=2))m/s", err="posMax=$(round(mm.abs.max_pos*100,digits=2))cm")
    else
        pvy = maximum(abs(ref.Vy(t)) for t in ts)
        pw  = maximum(abs(ref.Wz(t)) for t in ts)
        return (demand="peak|Vy|=$(round(pvy,digits=2)) peak|ω|=$(round(pw,digits=2))",
                err="Vy_rms=$(round(mm.abs.rms_vy*1e3,digits=1))mm/s ω_rms=$(round(mm.abs.rms_w*1e3,digits=1))mrad/s")
    end
end

println("Tuned ASMC (clean) combo sweep — is the PINNED combo the hardest FEASIBLE?\n")
println("=== octagon [velocity]  (PINNED = combo 2) ===")
for c in 1:5; r=offset("octagon_mu_0p5.toml", c, :velocity); println("  combo $c $(c==2 ? "<PIN>" : "     "): $(r.demand)  →  $(r.err)"); end
println("\n=== ellipse [pose]  (PINNED = combo 1) ===")
for c in [1,2,3,10,55]; r=offset("ellipse_mu_0p5.toml", c, :pose); println("  combo $c $(c==1 ? "<PIN>" : "     "): $(r.demand)  →  $(r.err)"); end
println("\n=== spin_creep [velocity]  (PINNED = combo 7) ===")
for c in [4,8,7,2,1]; r=offset("spin_creep_mu_0p5.toml", c, :velocity); println("  combo $c $(c==7 ? "<PIN>" : "     "): $(r.demand)  →  $(r.err)"); end
println("\nDONE")
