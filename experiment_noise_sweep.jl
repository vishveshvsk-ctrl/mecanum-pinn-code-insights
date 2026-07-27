#!/usr/bin/env julia
# Noise sweep {clean, 1×, 2×, 5×} × {ASMC, PID cap80, PID hi800} × all 3 trajectories.
# One seed (42). Physical sensor-noise model (white ⊕ scale-factor ⊕ bias per channel).
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
const TRAJS = [
    (name="octagon",    profile_toml="octagon_mu_0p5.toml",    combo_idx=2, mu=0.5, config_dir=DIR, run_mode=:velocity),
    (name="ellipse",    profile_toml="ellipse_mu_0p5.toml",    combo_idx=1, mu=0.5, config_dir=DIR, run_mode=:pose),
    (name="spin_creep", profile_toml="spin_creep_mu_0p5.toml", combo_idx=7, mu=0.5, config_dir=DIR, run_mode=:velocity),
]
const SCALES = [("clean", :clean, 1.0), ("1x", :noisy, 1.0), ("2x", :noisy, 2.0), ("5x", :noisy, 5.0)]

fmt(m, mode) = mode == :pose ?
    "posF=$(rpad(round(m.abs.final_pos*100,digits=2),6))cm posMax=$(rpad(round(m.abs.max_pos*100,digits=1),6))cm headMax=$(rpad(round(m.abs.max_head,digits=3),6))rad ce=$(round(m.ce,digits=1))" :
    "Vx=$(rpad(round(m.abs.rms_vx*1e3,digits=1),5)) Vy=$(rpad(round(m.abs.rms_vy*1e3,digits=1),5))mm/s ωrms=$(rpad(round(m.abs.rms_w*1e3,digits=1),6)) ωmax=$(rpad(round(m.abs.max_w*1e3,digits=1),6))mrad/s ce=$(round(m.ce,digits=1))"

open("runs_controller/noise_sweep_seed42.txt", "w") do io
    for out in (stdout, io)
        println(out, "Noise sweep — seed 42, physical sensor-noise model, all values LOWER=better\n")
    end
    for tr in TRAJS
        for out in (stdout, io); println(out, "==================== $(tr.name) [$(tr.run_mode)] ===================="); end
        for (label, ctrl, path) in CONFIGS
            kw = load_kw(path)
            for (tag, noise, sc) in SCALES
                probe, ref, mode = run_controller(ctrl, kw, noise, tr; seed=42, noise_scale=sc)
                m = controller_metrics(probe, ref, mode)
                line = "  $(rpad(label,10)) $(rpad(tag,6)): $(fmt(m, mode))"
                for out in (stdout, io); println(out, line); end
            end
            for out in (stdout, io); println(out); end
        end
    end
end
println("DONE  (saved -> runs_controller/noise_sweep_seed42.txt)")
