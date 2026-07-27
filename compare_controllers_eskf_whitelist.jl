#!/usr/bin/env julia
# =============================================================================
# compare_controllers_eskf_whitelist.jl
#   {ASMC, PID} controller comparison on a single FROZEN ESKF estimator,
#   using N trajectories per profile selected from diagnostics_combined.csv.
#
# This wraps compare_controllers_eskf.jl: it includes that script (which
# defines all reusable helpers) and only replaces trajectory selection and
# the CLI/main flow. MPC is intentionally omitted from the default controller
# set per user request; only finalized ASMC and PID are compared.
# =============================================================================

# Reuse helpers from the original ESKF controller comparison.
include("compare_controllers_eskf.jl")

# Override the finalized dict so the output reflects ASMC/PID only.
empty!(FINALIZED)
FINALIZED[:asmc] = true
FINALIZED[:pid]  = true

const USAGE_WHITELIST = """
Usage: compare_controllers_eskf_whitelist.jl [options]

Compare ASMC and PID closed-loop on a frozen ESKF estimator, using N trajectories
per profile selected from diagnostics_combined.csv.

Options:
  --estimator-dir DIR      Dir holding frozen ESKF best_config.json
                           (default: runs_eskf_noellipse_v2/eskf_dxnes)
  --controllers SPEC       Comma list of ctrl:path pairs
                           (default: asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,
                                     pid:runs_controller_pid_5seed/pid_FINAL_seed2.json)
  --run-dir DIR            Trajectory config dir
                           (default: trajectory_files_run_0p5_main)
  --whitelist CSV          diagnostics_combined.csv path
                           (default: diagnostics_combined.csv)
  --mu MU                  Friction coefficient to select (default: 0.5)
  --n-per-profile N        Trajectories sampled per profile (default: 10)
  --whitelist-col COL      Column to filter on (default: combined_reco)
  --whitelist-vals VALS    Comma-separated allowed values (default: keep)
  --exclude-profiles PROFS Comma-separated profile names to skip
                           (default: ellipse)
  --sample-seed SEED       RNG seed for deterministic trajectory sampling
                           (default: 1234)
  --seeds SEEDS            Comma-separated sensor-noise seeds
                           (default: 1)
  --out DIR                Output root
                           (default: runs_controller_compare_eskf_whitelist)
  --smoke                  One ASMC run on the first sampled trajectory, then exit
  --help                   Show this message
"""

function parse_args_whitelist(argv)
    a = Dict{String,Any}(
        "estimator-dir"     => "runs_eskf_noellipse_v2/eskf_dxnes",
        "controllers"       => "asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json," *
                               "pid:runs_controller_pid_5seed/pid_FINAL_seed2.json",
        "run-dir"           => "trajectory_files_run_0p5_main",
        "whitelist"         => "diagnostics_combined.csv",
        "mu"                => 0.5,
        "n-per-profile"     => 10,
        "whitelist-col"     => "combined_reco",
        "whitelist-vals"    => "keep",
        "exclude-profiles"  => "ellipse",
        "sample-seed"       => 1234,
        "seeds"             => [1],
        "out"               => "runs_controller_compare_eskf_whitelist",
        "smoke"             => false,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE_WHITELIST); exit(0)
        elseif arg == "--estimator-dir";     a["estimator-dir"] = argv[i+1]; i += 2
        elseif arg == "--controllers";       a["controllers"]   = argv[i+1]; i += 2
        elseif arg == "--run-dir";           a["run-dir"]       = argv[i+1]; i += 2
        elseif arg == "--whitelist";         a["whitelist"]     = argv[i+1]; i += 2
        elseif arg == "--mu";                a["mu"]            = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--n-per-profile";     a["n-per-profile"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--whitelist-col";     a["whitelist-col"] = argv[i+1]; i += 2
        elseif arg == "--whitelist-vals";    a["whitelist-vals"] = argv[i+1]; i += 2
        elseif arg == "--exclude-profiles";  a["exclude-profiles"] = argv[i+1]; i += 2
        elseif arg == "--sample-seed";       a["sample-seed"]   = parse(Int, argv[i+1]); i += 2
        elseif arg == "--seeds";             a["seeds"] = [parse(Int, s) for s in split(argv[i+1], ',')]; i += 2
        elseif arg == "--out";               a["out"]           = argv[i+1]; i += 2
        elseif arg == "--smoke";             a["smoke"] = true; i += 1
        else; error("unknown arg $arg\n$USAGE_WHITELIST"); end
    end
    return a
end

"""
    build_whitelist_trajs(run_dir, whitelist_csv;
                          mu=0.5, whitelist_col="combined_reco",
                          whitelist_values=["keep"], n_per_profile=10,
                          sample_seed=1234, exclude_profiles=["ellipse"])

Read `diagnostics_combined.csv`, keep rows with the requested `mu` and whitelist
value, exclude the named profiles, then deterministically sample
`n_per_profile` combo rows per profile. Returns a Vector of trajectory entries
compatible with `run_controller_on_estimator`.
"""
function build_whitelist_trajs(run_dir::String, whitelist_csv::String;
                               mu::Float64=0.5,
                               whitelist_col::String="combined_reco",
                               whitelist_values::Vector{String}=["keep"],
                               n_per_profile::Int=10,
                               sample_seed::Int=1234,
                               exclude_profiles::Vector{String}=["ellipse"])
    df = CSV.read(whitelist_csv, DataFrame)
    mu_col = hasproperty(df, :mu) ? :mu : error("whitelist CSV missing 'mu' column")
    wl_col = Symbol(whitelist_col)
    wl_col in propertynames(df) || error("whitelist CSV missing column '$whitelist_col'")

    sub = df[(df[!, mu_col] .== mu) .& (df[!, wl_col] .∈ Ref(whitelist_values)), :]
    isempty(sub) && error("no whitelist entries for mu=$mu and $whitelist_col in $(join(whitelist_values, ", "))")

    if !isempty(exclude_profiles)
        sub = sub[.!(sub[!, :profile] .∈ Ref(exclude_profiles)), :]
    end
    isempty(sub) && error("all whitelist entries excluded by profile filter")

    # Deterministic sampling per profile.
    rng = Random.Xoshiro(sample_seed)
    trajs = NamedTuple[]
    profiles = sort(unique(sub[!, :profile]))

    for prof in profiles
        prof_rows = sub[sub[!, :profile] .== prof, :]
        n = min(n_per_profile, nrow(prof_rows))
        if n < n_per_profile
            @warn "profile '$prof' has only $(nrow(prof_rows)) whitelist entries; using all $n"
        end
        idx = sortperm(prof_rows[!, :combo_idx])  # stable order
        prof_rows = prof_rows[idx, :]
        # Sample without replacement deterministically.
        chosen = if nrow(prof_rows) <= n
            collect(1:nrow(prof_rows))
        else
            Random.randperm(rng, nrow(prof_rows))[1:n]
        end
        for r in eachrow(prof_rows[chosen, :])
            toml = "$(prof)_mu_$(mu_string(mu)).toml"
            entry = (
                name         = Symbol(prof * "_c" * lpad(r.combo_idx, 3, '0')),
                profile_toml = toml,
                combo_idx    = r.combo_idx,
                run_mode     = :velocity,   # ellipse excluded; all remaining are velref
                mu           = mu,
                config_dir   = run_dir,
                profile      = prof,
            )
            push!(trajs, entry)
        end
    end
    return trajs
end

"mu → filename fragment (0.5 -> 0p5, etc.)."
function mu_string(mu::Float64)
    # The project uses 0p3 / 0p5 / 0p8 naming.
    mu == 0.3 ? "0p3" : mu == 0.5 ? "0p5" : mu == 0.8 ? "0p8" :
        error("unsupported mu=$mu for filename fragment")
end

"Mean±std grouped by [controller, profile]."
function summarize_by_profile(df::DataFrame)
    combine(groupby(df, [:controller, :profile]),
        :tracking         => _skipnan_mean => :tracking_mean,
        :tracking         => _skipnan_std  => :tracking_std,
        :ce               => _skipnan_mean => :ce_mean,
        :ce               => _skipnan_std  => :ce_std,
        :chatter          => _skipnan_mean => :chatter_mean,
        :chatter          => _skipnan_std  => :chatter_std,
        :est_nrmse_vx     => _skipnan_mean => :est_nrmse_vx_mean,
        :est_nrmse_vy     => _skipnan_mean => :est_nrmse_vy_mean,
        :est_nrmse_psidot => _skipnan_mean => :est_nrmse_psidot_mean,
        nrow              => :n_traj,
    )
end

"Mean±std grouped by [controller, profile, trajectory]."
function summarize_by_traj(df::DataFrame)
    combine(groupby(df, [:controller, :profile, :trajectory]),
        :tracking         => _skipnan_mean => :tracking_mean,
        :tracking         => _skipnan_std  => :tracking_std,
        :ce               => _skipnan_mean => :ce_mean,
        :ce               => _skipnan_std  => :ce_std,
        :chatter          => _skipnan_mean => :chatter_mean,
        :chatter          => _skipnan_std  => :chatter_std,
        nrow              => :n,
    )
end

"Overall controller ranking: mean tracking across all sampled trajectories."
function rank_controllers_overall(df::DataFrame)
    r = combine(groupby(df, :controller),
                :tracking => _skipnan_mean => :tracking_mean_overall,
                :tracking => _skipnan_std  => :tracking_std_overall,
                nrow      => :n_runs)
    sort!(r, :tracking_mean_overall)
    return r
end

# Override run_row to include the profile field for per-profile summarization.
function run_row(ctrl::Symbol, kw::NamedTuple, est_cfg, tr, seed::Int)
    probe, ref, mode = run_controller_on_estimator(ctrl, kw, est_cfg, tr; seed=seed)
    m = controller_metrics(probe, ref, mode)
    e = estimator_error(probe)
    a = m.abs
    prof = get(tr, :profile, string(tr.name))
    return (
        controller     = string(ctrl),
        finalized      = get(FINALIZED, ctrl, true),
        profile        = prof,
        trajectory     = string(tr.name),
        mode           = string(mode),
        seed           = seed,
        ok             = m.ok,
        tracking       = m.tracking,
        ce             = m.ce,
        chatter        = m.chatter,
        est_nrmse_vx     = e.vx,
        est_nrmse_vy     = e.vy,
        est_nrmse_psidot = e.psidot,
        est_nrmse_psi    = e.psi,
        est_nrmse_X      = e.X,
        est_nrmse_Y      = e.Y,
        abs_rms_vx     = get(a, :rms_vx, NaN),
        abs_rms_vy     = get(a, :rms_vy, NaN),
        abs_rms_w      = get(a, :rms_w,  NaN),
        abs_max_vx     = get(a, :max_vx, NaN),
        abs_max_vy     = get(a, :max_vy, NaN),
        abs_max_w      = get(a, :max_w,  NaN),
    )
end

# Override main() so that running this file executes the whitelist workflow.
function main()
    a = parse_args_whitelist(ARGS)
    est_cfg = load_frozen_estimator(a["estimator-dir"])
    println("Frozen estimator: $(est_cfg.estimator) @ $(est_cfg.rate_hz) Hz  ($(a["estimator-dir"]))")

    ctrl_specs = parse_controllers(a["controllers"])
    ctrl_kw = [(ctrl, load_controller_kw(path, ctrl)) for (ctrl, path) in ctrl_specs]
    for ((ctrl, _), (_, path)) in zip(ctrl_kw, ctrl_specs)
        fin = get(FINALIZED, ctrl, true) ? "finalized" : "NOT finalized"
        println("Controller: $(rpad(string(ctrl),4)) [$fin]  ($path)")
    end

    wl_vals = String.(strip.(split(a["whitelist-vals"], ',')))
    exclude = String.(strip.(split(a["exclude-profiles"], ',')))
    trajs = build_whitelist_trajs(
        a["run-dir"], a["whitelist"];
        mu=a["mu"], whitelist_col=a["whitelist-col"],
        whitelist_values=wl_vals, n_per_profile=a["n-per-profile"],
        sample_seed=a["sample-seed"], exclude_profiles=exclude)

    byprof = Dict{String,Vector{NamedTuple}}()
    for tr in trajs
        push!(get!(byprof, tr.profile, NamedTuple[]), tr)
    end
    println("\nSampled $(length(trajs)) trajectories across $(length(byprof)) profiles:")
    for prof in sort(collect(keys(byprof)))
        println("  $prof: $(length(byprof[prof])) trajectories")
    end

    out = a["out"]; mkpath(out)
    println("\nSeeds: $(a["seeds"])   Output root: $out")

    if a["smoke"]
        println("\n=== SMOKE: ASMC on first sampled trajectory ===")
        tr = first(trajs)
        asmc_kw = first(kw for (c, kw) in ctrl_kw if c == :asmc)
        probe, ref, mode = run_controller_on_estimator(:asmc, asmc_kw, est_cfg, tr; seed=1)
        m = controller_metrics(probe, ref, mode)
        e = estimator_error(probe)
        println("  $(tr.name) [$(mode)]: tracking=$(round(m.tracking,digits=3)) ce=$(round(m.ce,digits=3))")
        println("  est_nrmse: vx=$(round(e.vx,digits=3)) vy=$(round(e.vy,digits=3)) ψ̇=$(round(e.psidot,digits=3))")
        println("=== SMOKE OK ===")
        return
    end

    # Run one flat variant containing all sampled trajectories.
    println("\n===== Running all sampled trajectories =====")
    rows = NamedTuple[]
    for (ctrl, kw) in ctrl_kw
        for tr in trajs
            tr.run_mode == :velocity || @warn "expected velocity mode for $(tr.name); got $(tr.run_mode)"
            for seed in a["seeds"]
                row = try
                    run_row(ctrl, kw, est_cfg, tr, seed)
                catch err
                    @warn "run failed" controller=ctrl traj=tr.name seed=seed exception=err
                    (controller=string(ctrl), finalized=get(FINALIZED, ctrl, true),
                     trajectory=string(tr.name), profile=tr.profile, mode=string(tr.run_mode),
                     seed=seed, ok=false, tracking=NaN, ce=NaN, chatter=NaN,
                     est_nrmse_vx=NaN, est_nrmse_vy=NaN, est_nrmse_psidot=NaN,
                     est_nrmse_psi=NaN, est_nrmse_X=NaN, est_nrmse_Y=NaN,
                     abs_rms_vx=NaN, abs_rms_vy=NaN, abs_rms_w=NaN,
                     abs_max_vx=NaN, abs_max_vy=NaN, abs_max_w=NaN)
                end
                push!(rows, row)
                println("  $(rpad(string(ctrl),4)) $(rpad(string(tr.name),20)) " *
                        "seed=$(lpad(seed,2)) : track=$(round(row.tracking,digits=3)) " *
                        "ce=$(round(row.ce,digits=2)) est_vx=$(round(row.est_nrmse_vx,digits=3))")
            end
        end
    end

    df = DataFrame(rows)
    CSV.write(joinpath(out, "runs.csv"), df)
    Arrow.write(joinpath(out, "runs.arrow"), df)

    sum_prof = summarize_by_profile(df)
    CSV.write(joinpath(out, "summary_by_profile.csv"), sum_prof)
    Arrow.write(joinpath(out, "summary_by_profile.arrow"), sum_prof)

    sum_traj = summarize_by_traj(df)
    CSV.write(joinpath(out, "summary_by_trajectory.csv"), sum_traj)
    Arrow.write(joinpath(out, "summary_by_trajectory.arrow"), sum_traj)

    ranking = rank_controllers_overall(df)
    CSV.write(joinpath(out, "overall_ranking.csv"), ranking)

    println("\n  --- Overall controller ranking (mean tracking over all sampled trajs) ---")
    for r in eachrow(ranking)
        println("    $(rpad(r.controller,5)) $(round(r.tracking_mean_overall, digits=3)) " *
                "± $(round(r.tracking_std_overall, digits=3))  ($(r.n_runs) runs)")
    end

    println("\nComparison complete. Outputs in $out/")
    println("  runs.csv                — one row per (controller, trajectory, seed)")
    println("  summary_by_profile.csv  — mean±std grouped by [controller, profile]")
    println("  summary_by_trajectory.csv — mean±std grouped by [controller, profile, trajectory]")
    println("  overall_ranking.csv     — overall controller ranking")
    println("Note: `tracking` uses the TRUE plant state vs ref.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
