#!/usr/bin/env julia
# Noise-doubling stress test on ONE trajectory (ellipse / pose — the most
# noise-sensitive case). ASMC vs PID(cap80) vs PID(hi800) at clean / 1× / 2×
# physical sensor noise.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")
using JSON, StaticArrays

const DIR = "trajectory_files_run_0p5_main"
const CONFIGS = [
    ("asmc",      :asmc, "runs_controller/asmc_clean/best_config.json"),
    ("pid_cap80", :pid,  "runs_controller_pid_trajset2/pid_clean/best_config.json"),
    ("pid_hi800", :pid,  "runs_controller_pid_trajset2_hibound/pid_clean/best_config.json"),
]
function load_kw(path)
    g = JSON.parse(read(path, String))["best_gains"]
    pairs = Pair{Symbol,Any}[]
    for (k, v) in g
        push!(pairs, Symbol(k) => (v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)))
    end
    return (; pairs...)
end

const TR = (name="ellipse", profile_toml="ellipse_mu_0p5.toml", combo_idx=1,
            mu=0.5, config_dir=DIR, run_mode=:pose)

posfmt(m) = "pos final=$(round(m.abs.final_pos*100,digits=2))cm max=$(round(m.abs.max_pos*100,digits=1))cm | head max=$(round(m.abs.max_head,digits=3))rad | ce=$(round(m.ce,digits=1))"

println("ellipse [pose] — noise sweep (clean, 1×, 2×)\n")
for (label, ctrl, path) in CONFIGS
    kw = load_kw(path)
    println(label)
    for (tag, noise, sc) in [("clean", :clean, 1.0), ("noisy 1x", :noisy, 1.0), ("noisy 2x", :noisy, 2.0)]
        probe, ref, mode = run_controller(ctrl, kw, noise, TR; seed=42, noise_scale=sc)
        m = controller_metrics(probe, ref, mode)
        println("   ", rpad(tag, 9), ": ", posfmt(m))
    end
    println()
end
println("DONE")
