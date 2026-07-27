#!/usr/bin/env julia
# Append CLEAN-feedback rows (scale=0, seed=0) to the noisy-eval CSV, same columns.
# Clean is deterministic (no noise), so one row per controller×trajectory suffices.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")
using JSON, StaticArrays

const DIR = "trajectory_files_run_0p5_main"
trajs = default_trajs_3(DIR)
load_kw(p) = (g=JSON.parse(read(p,String))["best_gains"];
    (; (Symbol(k)=>(v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...))
const CTRL = [("asmc",:asmc,"runs_controller_asmc_pin/asmc_FINAL_seed3.json"),
              ("pid", :pid, "runs_controller_pid_5seed/pid_FINAL_seed2.json")]

rows = String[]
for (label,ctrl,path) in CTRL
    kw = load_kw(path)
    for tr in trajs
        probe, ref, mode = run_controller(ctrl, kw, :clean, tr; seed=0)
        m = controller_metrics(probe, ref, mode); a = m.abs
        if mode == :velocity
            push!(rows, "$label,$(tr.name),vel,0.0,0,$(a.rms_vx*1e3),$(a.rms_vy*1e3),$(a.rms_w*1e3),,,$(m.ce),$(m.chatter)")
        else
            push!(rows, "$label,$(tr.name),pose,0.0,0,,,,$(a.max_pos*100),$(a.max_head),$(m.ce),$(m.chatter)")
        end
    end
end
open("runs_controller/noise_eval_10seed.csv","a") do io
    for r in rows; println(io, r); end
end
println("appended ", length(rows), " clean rows (scale=0.0, seed=0) to noise_eval_10seed.csv")
