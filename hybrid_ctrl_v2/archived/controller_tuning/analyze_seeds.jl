#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/controller_tuning/analyze_seeds.jl — cross-seed + held-out
# TEST-tier report (brief §7.4, §8 steps 9/14/19/20)
# =============================================================================
# Reads the `seed1..seed5/<ctrl>_<oracle>/best_config.json` files a `run_stage.jl`
# stage root produced, computes per-seed converged flags + score median/min/cv +
# per-parameter cv, a finite-difference Hessian of the score at the best seed's
# optimum (eigenvalues quantify the sloppy directions), and evaluates the
# winning gains on the held-out TEST tier. Writes seed_report.json,
# seed_report.md, test_grid_eval.csv.
#
#   --stage-root   directory containing seed1..seedN subdirs (required)
#   --controller   asmc | pid | mpc
#   --noise        clean | noisy (must match what run_stage.jl was run with)
#   --run-dir      config dir (default trajectory_files_run_0p5_main)
#   --n-seeds      how many seed<N> subdirs to look for (default 5)
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
using JSON, Statistics

function parse_args(argv)
    a = Dict{String,Any}("controller" => "asmc", "noise" => "clean",
                         "run-dir" => "trajectory_files_run_0p5_main", "n-seeds" => 5)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--stage-root"; a["stage-root"] = argv[i+1]; i += 2
        elseif arg == "--controller"; a["controller"] = argv[i+1]; i += 2
        elseif arg == "--noise"; a["noise"] = argv[i+1]; i += 2
        elseif arg == "--run-dir"; a["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--n-seeds"; a["n-seeds"] = parse(Int, argv[i+1]); i += 2
        else; error("analyze_seeds.jl: unknown arg $arg"); end
    end
    haskey(a, "stage-root") || error("analyze_seeds.jl: --stage-root is required")
    return a
end

"Flatten a JSON best_gains Dict into a sorted (names, values) pair for cv/Hessian work."
function _flatten_gains(gains::AbstractDict)
    names = String[]; vals = Float64[]
    for k in sort(collect(keys(gains)))
        v = gains[k]
        if v isa AbstractVector
            for (i, vi) in enumerate(v); push!(names, "$(k)_$i"); push!(vals, Float64(vi)); end
        elseif v isa Real
            push!(names, k); push!(vals, Float64(v))
        end
        # matrices (e.g. Stage-3 gains that carry no P_terminal here) skipped --
        # cv/Hessian are reported over the scalar/vector tuned gains only.
    end
    return names, vals
end

function _cv(x::Vector{Float64})
    m = mean(x)
    abs(m) < 1e-12 && return NaN
    return std(x) / abs(m) * 100
end

"""Rebuild the `kw` NamedTuple `run_controller_v2`/`build_controller(_v2)`
expects from a `best_config.json`'s `best_gains` Dict (round-trips the JSON
back into SVector/scalar fields keyed exactly like `build_controller`'s
:mpc branch for `ctrl==:mpc`, or the flat gain fields otherwise)."""
function _rebuild_kw(ctrl::Symbol, gains::AbstractDict)
    if ctrl == :mpc
        Pt = haskey(gains, "P_terminal") ? Matrix(hcat(Float64.(gains["P_terminal"])...)') : zeros(6, 6)
        return (Q_pose=SVector{6}(Float64.(gains["Q_pose"])), R=SVector{4}(Float64.(gains["R"])),
               S=SVector{4}(Float64.(gains["S"])), Np_pose=Int(gains["Np_pose"]), P_terminal=Pt)
    else
        pairs = Pair{Symbol,Any}[]
        for (k, v) in gains
            push!(pairs, Symbol(k) => (v isa AbstractVector ? SVector{length(v)}(Float64.(v)) : Float64(v)))
        end
        return (; pairs...)
    end
end

"Score `kw` directly (no decode) against `trajs` -- reuses run_controller_v2/controller_metrics."
function score_at_gains(ctrl::Symbol, kw::NamedTuple, trajs, oracle::Symbol; seed::Int=1)
    asmc_o, mpc_o, pid_o = ctrl == :mpc ? build_controller_v2(ctrl, kw) : build_controller(ctrl, kw)
    track_sum = 0.0; ce_sum = 0.0; chat_sum = 0.0; n_fail = 0
    for tr in trajs
        try
            probe, ref, mode, bus = run_controller_v2(ctrl, oracle, tr; asmc_o=asmc_o, mpc_o=mpc_o, pid_o=pid_o, seed=seed)
            m = controller_metrics(probe, ref, mode)
            SchedulerMod.clear_probe_log!(bus)
            if m.ok && isfinite(m.tracking)
                track_sum += m.tracking; ce_sum += m.ce; chat_sum += m.chatter
            else
                n_fail += 1
            end
        catch
            n_fail += 1
        end
    end
    n_fail > 0 && return 1e6
    n = length(trajs)
    return track_sum/n + LAMBDA_CE*(ce_sum/n/V_MAX)
end

"""Finite-difference Hessian of `f` at `x0` (central differences, step `h`
relative to |x0| -- brief §7.4: d(d+1)/2 extra evals via symmetry)."""
function fd_hessian(f::Function, x0::Vector{Float64}; h_rel::Float64=0.05)
    d = length(x0)
    h = max.(abs.(x0) .* h_rel, 1e-6)
    f0 = f(x0)
    fp = [f(x0 .+ (1:d .== i) .* h[i]) for i in 1:d]
    fm = [f(x0 .- (1:d .== i) .* h[i]) for i in 1:d]
    H = zeros(d, d)
    for i in 1:d
        H[i,i] = (fp[i] - 2f0 + fm[i]) / h[i]^2
    end
    for i in 1:d, j in (i+1):d
        xpp = copy(x0); xpp[i] += h[i]; xpp[j] += h[j]
        xpm = copy(x0); xpm[i] += h[i]; xpm[j] -= h[j]
        xmp = copy(x0); xmp[i] -= h[i]; xmp[j] += h[j]
        xmm = copy(x0); xmm[i] -= h[i]; xmm[j] -= h[j]
        Hij = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h[i]*h[j])
        H[i,j] = Hij; H[j,i] = Hij
    end
    return H
end

function main()
    a = parse_args(ARGS)
    ctrl = Symbol(a["controller"]); oracle = Symbol(a["noise"])
    stage_root = a["stage-root"]

    seed_dirs = [(n, joinpath(stage_root, "seed$n", "$(ctrl)_$(a["noise"])")) for n in 1:a["n-seeds"]]
    seed_dirs = filter(p -> isfile(joinpath(p[2], "best_config.json")), seed_dirs)
    isempty(seed_dirs) && error("analyze_seeds.jl: no best_config.json found under $stage_root/seed*/$(ctrl)_$(a["noise"])/")

    configs = [(n, JSON.parsefile(joinpath(dir, "best_config.json"))) for (n, dir) in seed_dirs]
    scores = [c["best_score"] for (_, c) in configs]
    converged_flags = [get(c, "converged", false) for (_, c) in configs]
    n_converged = count(converged_flags)

    println("===== analyze_seeds: $ctrl / $(a["noise"]) / $stage_root =====")
    println("seeds found: ", length(configs), "  converged: $n_converged/$(length(configs))")
    println("scores: ", [round(s, digits=4) for s in scores])
    println("median=$(round(median(scores),digits=4))  min=$(round(minimum(scores),digits=4))  cv=$(round(_cv(Float64.(scores)),digits=2))%")

    # Per-parameter cv across seeds (gains that appear in every seed's best_gains).
    all_names = [Set(keys(c["best_gains"])) for (_, c) in configs]
    common = reduce(intersect, all_names)
    param_cv = Dict{String,Float64}()
    for k in sort(collect(common))
        v0 = configs[1][2]["best_gains"][k]
        if v0 isa Real
            vals = Float64[c["best_gains"][k] for (_, c) in configs]
            param_cv[k] = _cv(vals)
        elseif v0 isa AbstractVector
            for i in 1:length(v0)
                vals = Float64[c["best_gains"][k][i] for (_, c) in configs]
                param_cv["$(k)_$i"] = _cv(vals)
            end
        end
    end

    # Best seed (lowest score) -> Hessian + held-out TEST tier.
    best_i = argmin(scores)
    best_seed, best_config = configs[best_i]
    best_gains = best_config["best_gains"]
    kw = _rebuild_kw(ctrl, best_gains)
    names, x0 = _flatten_gains(best_gains)

    test_trajs = trajset(:test, a["run-dir"])
    test_score = score_at_gains(ctrl, kw, test_trajs, oracle)
    println("best seed=$best_seed  score=$(round(scores[best_i],digits=4))  TEST-tier score=$(round(test_score,digits=4))")

    # coupled_vomega c12 anchor: also in TRAIN_FULL, so this is the train/test gap probe.
    anchor = filter(t -> t.name == "coupled_vomega_anchor", test_trajs)
    anchor_score = isempty(anchor) ? NaN : score_at_gains(ctrl, kw, anchor, oracle)
    println("coupled_vomega c12 anchor score=$(round(anchor_score,digits=4))")

    println("computing finite-difference Hessian at the best seed's optimum ($(length(x0)) params, " *
            "$(length(x0)*(length(x0)+1)) extra evals)...")
    Hf = θ -> begin
        kwθ = (; (Symbol(n) => v for (n, v) in zip(names, θ))...)
        # Re-fold flattened per-axis names ("Kp_1","Kp_2","Kp_3") back to
        # SVectors when the original gain was a vector (mirrors PIDCascadeMod's
        # regroup_joint pattern, generalized to arbitrary flattened names).
        grouped = Dict{Symbol,Any}()
        for (n, v) in zip(names, θ)
            m = match(r"^(.*)_(\d+)$", n)
            if m !== nothing && haskey(kw, Symbol(m.captures[1]))
                base = Symbol(m.captures[1])
                grouped[base] = get(grouped, base, Float64[])
                push!(grouped[base], v)
            else
                grouped[Symbol(n)] = v
            end
        end
        kwf = (; (k => (v isa Vector ? SVector{length(v)}(v) : v) for (k, v) in grouped)...)
        return score_at_gains(ctrl, kwf, test_trajs, oracle)
    end
    H = fd_hessian(Hf, x0)
    eigs = try; eigvals(H); catch; fill(NaN, length(x0)); end

    report = Dict(
        "controller" => string(ctrl), "noise" => a["noise"], "stage_root" => stage_root,
        "n_seeds" => length(configs), "n_converged" => n_converged,
        "scores" => scores, "score_median" => median(scores), "score_min" => minimum(scores),
        "score_cv_pct" => _cv(Float64.(scores)),
        "param_cv_pct" => param_cv,
        "best_seed" => best_seed, "best_score" => scores[best_i],
        "test_grid_score" => test_score, "coupled_vomega_c12_anchor_score" => anchor_score,
        "hessian_param_names" => names, "hessian_eigenvalues" => real.(eigs),
    )
    open(joinpath(stage_root, "seed_report.json"), "w") do io
        JSON.print(io, report, 2)
    end

    rows = String["trajectory,role,tracking,ce,chatter"]
    for tr in test_trajs
        try
            asmc_o, mpc_o, pid_o = ctrl == :mpc ? build_controller_v2(ctrl, kw) : build_controller(ctrl, kw)
            probe, ref, mode, bus = run_controller_v2(ctrl, oracle, tr; asmc_o=asmc_o, mpc_o=mpc_o, pid_o=pid_o, seed=1)
            m = controller_metrics(probe, ref, mode)
            SchedulerMod.clear_probe_log!(bus)
            push!(rows, "$(tr.name),$(tr.role),$(m.tracking),$(m.ce),$(m.chatter)")
        catch e
            push!(rows, "$(tr.name),$(tr.role),FAILED,FAILED,FAILED")
        end
    end
    write(joinpath(stage_root, "test_grid_eval.csv"), join(rows, "\n"))

    open(joinpath(stage_root, "seed_report.md"), "w") do io
        println(io, "# Cross-seed report — $ctrl / $(a["noise"])\n")
        println(io, "Seeds: $(length(configs))  Converged: $n_converged/$(length(configs))\n")
        println(io, "| seed | score | converged |")
        println(io, "|---|---|---|")
        for ((n, c), s) in zip(configs, scores)
            println(io, "| $n | $(round(s,digits=4)) | $(get(c,"converged",false)) |")
        end
        println(io, "\nScore median=$(round(median(scores),digits=4)) min=$(round(minimum(scores),digits=4)) " *
                    "cv=$(round(_cv(Float64.(scores)),digits=2))%\n")
        println(io, "## Per-parameter cv (%)\n")
        for (k, v) in sort(collect(param_cv))
            println(io, "- $k: $(round(v,digits=2))%")
        end
        println(io, "\n## Held-out TEST tier\n")
        println(io, "Best seed $best_seed TEST-tier score: $(round(test_score,digits=4))")
        println(io, "coupled_vomega c12 anchor (train/test gap probe): $(round(anchor_score,digits=4))\n")
        println(io, "## Hessian eigenvalues at the best seed's optimum (sloppy directions)\n")
        println(io, "Params: ", join(names, ", "))
        println(io, "Eigenvalues: ", join(round.(real.(eigs), sigdigits=4), ", "))
    end

    println("wrote $(joinpath(stage_root,"seed_report.json")), seed_report.md, test_grid_eval.csv")
end

main()
