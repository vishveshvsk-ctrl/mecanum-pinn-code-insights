#!/usr/bin/env julia
# Measure chatter/effort of each 5-seed ASMC optimum on the training set.
# Confirms low-K_max => low-chatter (the robust manifold corner) and calibrates
# the chatter/K_max penalty weights for the penalized re-run.
using Pkg; Pkg.activate(".")
include("tune_controller.jl")
using JSON, StaticArrays, Statistics

const DIR = "trajectory_files_run_0p5_main"
trajs = default_trajs_3(DIR)

function load_kw(path)
    g = JSON.parse(read(path, String))["best_gains"]
    (; (Symbol(k) => (v isa Vector ? SVector{length(v)}(Float64.(v)) : Float64(v)) for (k,v) in g)...)
end

function meanchat(kw, noise, sc)
    ch_sum=0.0; tr_sum=0.0
    for tr in trajs
        probe, ref, mode = run_controller(:asmc, kw, noise, tr; seed=42, noise_scale=sc)
        m = controller_metrics(probe, ref, mode)
        ch_sum += m.chatter; tr_sum += m.tracking
    end
    return ch_sum/length(trajs), tr_sum/length(trajs)
end

println("seed | ΣK_max | chatter_clean | chatter_noisy1x | chatter_noisy2x | track_noisy1x")
for s in 1:5
    p = "runs_controller_asmc_5seed_hiK/seed$s/asmc_clean/best_config.json"
    isfile(p) || (println("seed $s: (missing)"); continue)
    kw = load_kw(p)
    ksum = kw.K_max_x + kw.K_max_y + kw.K_max_psi
    chc,_  = meanchat(kw, :clean, 1.0)
    ch1,t1 = meanchat(kw, :noisy, 1.0)
    ch2,_  = meanchat(kw, :noisy, 2.0)
    println("  $s  | $(rpad(round(ksum,digits=0),6)) | " *
            "$(rpad(round(chc,digits=2),13)) | $(rpad(round(ch1,digits=2),15)) | " *
            "$(rpad(round(ch2,digits=2),15)) | $(round(t1,digits=2))  " *
            "(Kx=$(round(kw.K_max_x,digits=0)))")
end
println("\nnote: chatter = mean per-tick TV of v_cmd (V); does low ΣK_max => less noise chatter?")
println("DONE")
