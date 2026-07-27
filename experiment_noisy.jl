#!/usr/bin/env julia
# ASMC vs PID (actuation-limited Kp≤80  AND  raised-bounds Kp≤800) under
# physically-consistent sensor noise (white ⊕ scale-factor ⊕ bias per channel).
# Clean vs noisy, on octagon (transl), ellipse (pose), spin_creep (yaw).
# Gains are read from best_gains directly (bound-independent), so the raised
# PID_SPACE does not affect how the capped config decodes.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")   # main-guarded: run_controller, controller_metrics
using JSON, StaticArrays

const DIR = "trajectory_files_run_0p5_main"
# (label, ctrl, config path)
const CONFIGS = [
    ("asmc",       :asmc, "runs_controller/asmc_clean/best_config.json"),
    ("pid_cap80",  :pid,  "runs_controller_pid_trajset2/pid_clean/best_config.json"),         # Kd fixed, Kp≤80 (actuation-limited)
    ("pid_hi800",  :pid,  "runs_controller_pid_trajset2_hibound/pid_clean/best_config.json"), # Kd fixed, Kp≤800 (raised)
]

# Build the controller-gain NamedTuple straight from the stored best_gains.
function load_kw(path)
    g = JSON.parse(read(path, String))["best_gains"]
    pairs = Pair{Symbol,Any}[]
    for (k, v) in g
        push!(pairs, Symbol(k) => (v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)))
    end
    return (; pairs...)
end

const TRAJS = [
    (name="octagon",    profile_toml="octagon_mu_0p5.toml",    combo_idx=2, mu=0.5, config_dir=DIR, run_mode=:velocity),
    (name="ellipse",    profile_toml="ellipse_mu_0p5.toml",    combo_idx=1, mu=0.5, config_dir=DIR, run_mode=:pose),
    (name="spin_creep", profile_toml="spin_creep_mu_0p5.toml", combo_idx=7, mu=0.5, config_dir=DIR, run_mode=:velocity),
]

fmt(m, mode) = mode == :pose ?
    "pos final=$(round(m.abs.final_pos*100,digits=2))cm max=$(round(m.abs.max_pos*100,digits=1))cm | head max=$(round(m.abs.max_head,digits=3))rad" :
    "Vx_rms=$(round(m.abs.rms_vx*1e3,digits=1)) Vy_rms=$(round(m.abs.rms_vy*1e3,digits=1))mm/s | ω_rms=$(round(m.abs.rms_w*1e3,digits=1)) max_ω=$(round(m.abs.max_w*1e3,digits=1))mrad/s"

for tr in TRAJS
    println("\n==================== $(tr.name) [$(tr.run_mode)] ====================")
    for (label, ctrl, path) in CONFIGS
        kw = load_kw(path)
        print(rpad(label, 10))
        for noise in (:clean, :noisy)
            probe, ref, mode = run_controller(ctrl, kw, noise, tr; seed=42)
            m = controller_metrics(probe, ref, mode)
            print("  | ", rpad(string(noise), 5), ": ", fmt(m, mode), " (ce=", round(m.ce, digits=1), ")")
        end
        println()
    end
end
println("\nDONE")
