#!/usr/bin/env julia
# Build the tuning/eval subset: for each curated profile, sample combos, run the
# tuned ASMC (clean), and report peak demand + tracking offset — sorted by offset
# so the EASY (low offset) and STRESS (highest still-feasible offset, before the
# cliff) combos are read off directly. ellipse split into tangent / crab modes.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")
using JSON, StaticArrays, Statistics

const DIR = "trajectory_files_run_0p5_main"
asmc_kw = let g = JSON.parse(read("runs_controller/asmc_clean/best_config.json", String))["best_gains"]
    (; (Symbol(k) => (v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...)
end

spread(n, k) = unique(round.(Int, range(1, n, length=k)))

# (label, toml, mu, run_mode, combos-to-sample)
const PROBES = [
    ("long_circle",    "long_circle_mu_0p5.toml",            0.5, :velocity, spread(254, 16)),
]

function probe1(toml, mu, combo, mode)
    tr = (name=split(toml,"_")[1], profile_toml=toml, combo_idx=combo, mu=mu, config_dir=DIR, run_mode=mode)
    probe, ref, m = run_controller(:asmc, asmc_kw, :clean, tr; seed=42)
    ts = [p.t for p in probe]
    mm = controller_metrics(probe, ref, m)
    if mode == :pose
        spd = maximum(sqrt(ref.Vxo(t)^2 + ref.Vyo(t)^2) for t in ts)
        return (spd=spd, yaw=0.0, off=mm.abs.max_pos*100, unit="cm(posMax)")
    else
        spd = maximum(sqrt(ref.Vx(t)^2 + ref.Vy(t)^2) for t in ts)
        yaw = maximum(abs(ref.Wz(t)) for t in ts)
        vy  = mm.abs.rms_vy*1e3    # mm/s
        wr  = mm.abs.rms_w*1e3     # mrad/s  (FIXED: was raw rad/s)
        return (spd=spd, yaw=yaw, off=max(vy, wr), vy=vy, wr=wr, unit="max(Vy_rms mm/s, ω_rms mrad/s)")
    end
end

open("runs_controller/subset_probe_longcircle.txt", "w") do io
    for (label, toml, mu, mode, combos) in PROBES
        rows = NamedTuple[]
        for c in combos
            try
                r = probe1(toml, mu, c, mode)
                push!(rows, (combo=c, r...))
            catch e
                # skip out-of-range / build failures
            end
        end
        sort!(rows, by = x -> x.off)
        hdr = "==================== $label [$mode]  (offset = $(isempty(rows) ? "?" : rows[1].unit)) ===================="
        for out in (stdout, io)
            println(out, hdr)
            for x in rows
                println(out, "  combo $(rpad(x.combo,4)): peakSpd=$(rpad(round(x.spd,digits=2),5)) peak|ω|=$(rpad(round(x.yaw,digits=2),5)) → Vy_rms=$(rpad(round(x.vy,digits=1),6))mm/s  ω_rms=$(round(x.wr,digits=1))mrad/s")
            end
            println(out)
        end
    end
end
println("DONE -> runs_controller/subset_probe.txt")
