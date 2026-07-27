#!/usr/bin/env julia
# Pose-mode noisy robustness eval (controller-posref-retune-reeval.md).
# ASMC(chatterpen seed4, score 1.465) vs PID(chatterpen seed2, score 1.758) on the
# all-pose 6-traj subset (default_trajs_pose), clean + physical sensor-noise at
# {1x,2x,5x}, 10 noise seeds each. Emits POSE metrics (final_pos_cm, max_pos_cm,
# final_head_rad, max_head_rad) + ce + chatter per row.
#
# PID seed2 (not the clean-score winner seed4) is used deliberately: seed4's
# Kd was 8x higher (9.30 vs 1.13 sum) and degraded 2-5x worse under noise in a
# side-by-side check (_tmp/noise_eval_pid_seed2.jl) -- seed2 is the more honest
# PID representative for a noisy-feedback comparison against ASMC.
#
# NEW FILE — a copy-and-adapt of experiment_noise_eval.jl (the v1 velocity-mode
# script, which loads default_trajs_3 and mixed vel/pose CTRL configs). The
# original is UNTOUCHED; this one is pose-only throughout (every trajectory here
# runs mode=:pose, so no mixed-schema branching is needed).
#
# Relocated into hybrid_ctrl/controller_tuning/ alongside the v1 script -- the
# ROOT/cd(ROOT) preamble is the only addition, so relative paths (Pkg.activate,
# include, and the runs_controller.../ I/O paths) keep resolving against
# code_insights/ regardless of where this script is invoked from.
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg; Pkg.activate(ROOT)
include(joinpath(ROOT, "tune_controller.jl"))
using JSON, StaticArrays, Statistics

const DIR = "trajectory_files_run_0p5_main"

function parse_args(argv)
    a = Dict{String,Any}("out" => "runs_controller", "exclude-ellipse" => false)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--out"; a["out"] = argv[i+1]; i += 2
        elseif arg == "--exclude-ellipse"; a["exclude-ellipse"] = true; i += 1
        else; error("unknown arg $arg"); end
    end
    return a
end
const CLI = parse_args(ARGS)

trajs_all = default_trajs_pose(DIR)
trajs = CLI["exclude-ellipse"] ? filter(tr -> !startswith(string(tr.name), "ellipse"), trajs_all) : trajs_all

load_kw(p) = (g=JSON.parse(read(p,String))["best_gains"];
    (; (Symbol(k)=>(v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...))
const CTRL = [("asmc",:asmc,"runs_controller_asmc_pose_5seed_chatterpen/seed4/asmc_clean/best_config.json"),
              ("pid", :pid, "runs_controller_pid_pose_5seed/seed2/pid_clean/best_config.json")]
const SCALES = [1.0, 2.0, 5.0]
const SEEDS = 1:10

const OUT_ROOT = CLI["out"]
mkpath(OUT_ROOT)

rows = String["controller,trajectory,scale,seed,final_pos_cm,max_pos_cm,final_head_rad,max_head_rad,ce,chatter"]
s_prim = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # primary tracking metric (max_pos_cm)
s_ce   = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # control effort
s_chat = Dict{Tuple{String,String,Float64},Vector{Float64}}()  # chatter (TV of v_cmd)

for (label,ctrl,path) in CTRL
    kw = load_kw(path)
    for tr in trajs
        # CLEAN pass: OracleEstimator(:clean) is all-zero noise regardless of
        # seed (scale ignored too), so a single run suffices -- no seed loop.
        probe, ref, mode = run_controller(ctrl, kw, :clean, tr; seed=1)
        m = controller_metrics(probe, ref, mode)
        a = m.abs
        push!(rows, "$label,$(tr.name),0.0,0,$(a.final_pos*100),$(a.max_pos*100),$(a.final_head),$(a.max_head),$(m.ce),$(m.chatter)")
        k = (label, tr.name, 0.0)
        push!(get!(s_prim,k,Float64[]), a.max_pos*100)
        push!(get!(s_ce,k,Float64[]),   m.ce)
        push!(get!(s_chat,k,Float64[]), m.chatter)

        for sc in SCALES
            for sd in SEEDS
                probe, ref, mode = run_controller(ctrl, kw, :noisy, tr; seed=sd, noise_scale=sc)
                m = controller_metrics(probe, ref, mode)
                a = m.abs
                push!(rows, "$label,$(tr.name),$sc,$sd,$(a.final_pos*100),$(a.max_pos*100),$(a.final_head),$(a.max_head),$(m.ce),$(m.chatter)")
                k = (label, tr.name, sc)
                push!(get!(s_prim,k,Float64[]), a.max_pos*100)
                push!(get!(s_ce,k,Float64[]),   m.ce)
                push!(get!(s_chat,k,Float64[]), m.chatter)
            end
        end
    end
end

write(joinpath(OUT_ROOT, "noise_eval_pose_10seed.csv"), join(rows,"\n"))

fmt(d,label,tr) = "clean: $(round(d[(label,tr,0.0)][1],digits=1))  " *
    join(["$(sc)x: $(round(mean(d[(label,tr,sc)]),digits=1))±$(round(std(d[(label,tr,sc)]),digits=1))" for sc in SCALES], "  ")
open(joinpath(OUT_ROOT, "noise_eval_pose_10seed_summary.txt"),"w") do io
    for out in (stdout, io)
        println(out, "Pose noisy eval — clean (1 run) + mean±std over 10 seeds, scales 1x/2x/5x\n")
        for (metricname, d) in [("TRACKING max_pos_cm", s_prim),
                                ("CONTROL EFFORT ce", s_ce),
                                ("CHATTER (TV of v_cmd)", s_chat)]
            println(out, "################ $metricname ################")
            for (label,_,_) in CTRL
                println(out, "==== $label ====")
                for tr in trajs
                    println(out, "  $(rpad(tr.name,16)) $(fmt(d,label,tr.name))")
                end
            end
            println(out)
        end
    end
end
println("wrote ", joinpath(OUT_ROOT, "noise_eval_pose_10seed.csv"), " (", length(rows)-1, " rows) + summary")
