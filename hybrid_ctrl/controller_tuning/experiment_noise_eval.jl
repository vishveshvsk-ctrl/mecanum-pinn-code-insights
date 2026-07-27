#!/usr/bin/env julia
# Noisy robustness eval: ASMC(final seed3) vs PID(candidate seed2) on the 6-traj
# training set, physical sensor-noise at scales {1x,2x,5x}, 10 noise seeds each.
# Writes tidy per-seed CSV (figures) + mean±std summary. 2*6*3*10 = 360 runs.
#
# Relocated into hybrid_ctrl/controller_tuning/ (v1 script, logic unchanged) --
# the ROOT/cd(ROOT) preamble below is the only addition, so every relative path
# below (Pkg.activate, include, and the runs_controller/... I/O paths) keeps
# resolving against code_insights/ regardless of where this script is invoked from.
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg; Pkg.activate(ROOT)
include(joinpath(ROOT, "tune_controller.jl"))
using JSON, StaticArrays, Statistics

const DIR = "trajectory_files_run_0p5_main"
trajs = default_trajs_3(DIR)
load_kw(p) = (g=JSON.parse(read(p,String))["best_gains"];
    (; (Symbol(k)=>(v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...))
const CTRL = [("asmc",:asmc,"runs_controller_asmc_pin/asmc_FINAL_seed3.json"),
              ("pid", :pid, "runs_controller_pid_5seed/pid_FINAL_seed2.json")]
const SCALES = [1.0, 2.0, 5.0]
const SEEDS = 1:10

rows = String["controller,trajectory,mode,scale,seed,rms_vx_mm,rms_vy_mm,rms_w_mrad,max_pos_cm,max_head_rad,ce,chatter"]
s_prim = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # primary tracking metric
s_ce   = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # control effort
s_chat = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # chatter (TV of v_cmd)

for (label,ctrl,path) in CTRL
    kw = load_kw(path)
    for tr in trajs
        for sc in SCALES
            for sd in SEEDS
                probe, ref, mode = run_controller(ctrl, kw, :noisy, tr; seed=sd, noise_scale=sc)
                m = controller_metrics(probe, ref, mode)
                a = m.abs
                if mode == :velocity
                    push!(rows, "$label,$(tr.name),vel,$sc,$sd,$(a.rms_vx*1e3),$(a.rms_vy*1e3),$(a.rms_w*1e3),,,$(m.ce),$(m.chatter)")
                    prim = a.rms_w*1e3   # yaw-rate mrad/s
                else
                    push!(rows, "$label,$(tr.name),pose,$sc,$sd,,,,$(a.max_pos*100),$(a.max_head),$(m.ce),$(m.chatter)")
                    prim = a.max_pos*100 # cm
                end
                k=(label,tr.name,sc)
                push!(get!(s_prim,k,Float64[]), prim)
                push!(get!(s_ce,k,Float64[]),   m.ce)
                push!(get!(s_chat,k,Float64[]),  m.chatter)
            end
        end
    end
end

write("runs_controller/noise_eval_10seed.csv", join(rows,"\n"))

fmt3(d,label,tr) = join(["$(sc)x: $(round(mean(d[(label,tr,sc)]),digits=1))±$(round(std(d[(label,tr,sc)]),digits=1))" for sc in SCALES], "  ")
open("runs_controller/noise_eval_10seed_summary.txt","w") do io
    for out in (stdout, io)
        println(out, "Noisy eval — mean±std over 10 seeds, scales 1x/2x/5x\n")
        for (metricname, d) in [("TRACKING (yaw ω_rms mrad/s | posMax cm)", s_prim),
                                ("CONTROL EFFORT ce", s_ce),
                                ("CHATTER (TV of v_cmd)", s_chat)]
            println(out, "################ $metricname ################")
            for (label,_,_) in CTRL
                println(out, "==== $label ====")
                for tr in trajs
                    println(out, "  $(rpad(tr.name,16)) $(fmt3(d,label,tr.name))")
                end
            end
            println(out)
        end
    end
end
println("wrote noise_eval_10seed.csv (", length(rows)-1, " rows) + summary")
