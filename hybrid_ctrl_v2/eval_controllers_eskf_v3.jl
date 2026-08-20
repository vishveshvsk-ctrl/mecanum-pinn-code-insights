#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/eval_controllers_eskf_v3.jl
#
# Closed-loop controller comparison through the TUNED ESKF V3 (13-dim,
# yaw-accel state, seed-4 v4 config) on the :realistic sensor suite — the
# deployable-performance counterpart of the clean-oracle tuning numbers.
#
#   Controllers (best seed each, clean-oracle tuned, current code):
#     ASMC v2  — runs_asmc_v2_relaunch/seed2      (score 3.690 clean)
#     PID-FB   — runs_pid_v2_relaunch/seed1 fb    (score 1.723 clean)
#     PID-CT   — runs_pid_v2_relaunch/seed1 ct    (score 0.579 clean)
#   Estimator: runs_estimator_v4_mu0p5_train12/seed4 (best_score 2.5818 replay)
#
# Conditions match the ESKF's tuning conditions EXACTLY
# (run_estimator_replay_mu0p5_v4.jl / cross_eval_mu0p5_v4.jl):
#   SensorModV2.build_suite(:realistic; seed, flow=true, fix_tier=:docking),
#   estimator rate 1000 Hz (= f_est), pose mode, use_dhat=false.
# Sensor noise seeds 101..105 (CLI --seed N -> noise seed 100+N), the same
# family as the estimator cross-eval (101-103). Common random numbers: the
# same (trajectory, noise seed) pair produces the same sensor realization for
# every controller (per-modality RNGs seeded by hash((seed, modality))).
#
# Metrics per run: controller_metrics (tracking/ce/chatter on TRUE state —
# identical to the tuning objective, so score = tracking + 0.05*ce/24 is
# directly comparable to the clean-oracle scores), plus estimator_error
# (xhat vs true, context) and controller_error (ref vs true, SI units).
#
# CLI:
#   --seed N        run noise seed 100+N over 3 controllers x 12 train12 trajs
#   --tier NAME     trajectory tier (default train12)
#   --out DIR       output root (default hybrid_ctrl_v2/runs_controller_eskf_v3)
#   --smoke         one trajectory x 3 controllers x one seed, verbose, no write
#   --aggregate     fold runs_seed*.csv into summary.csv + printed table
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v3.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "scheduler_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl")); using .HarnessV2Mod

using JSON
using DataFrames
using CSV
using Statistics
using TOML
using Random

const ESKF_CFG_PATH = joinpath(ROOT, "hybrid_ctrl_v2", "runs_estimator_v4_mu0p5_train12", "seed4", "best_config.json")
const CONTROLLERS = [
    ("asmc_v2", :asmc, joinpath(ROOT, "hybrid_ctrl_v2", "runs_asmc_v2_relaunch", "seed2", "asmc_v2_clean", "best_config.json")),
    ("pid_fb",  :pid,  joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v2_relaunch", "seed1", "pid_v2_fb_clean", "best_config.json")),
    ("pid_ct",  :pid,  joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v2_relaunch", "seed1", "pid_v2_ct_clean", "best_config.json")),
]

"JSON best_gains -> NamedTuple; Bool values stay Bool (feedforward), rest Float64."
function load_gains(path::String)
    d = JSON.parsefile(path)
    haskey(d, "best_gains") || error("no best_gains in $path")
    pairs = Pair{Symbol,Any}[]
    for (k, v) in d["best_gains"]
        push!(pairs, Symbol(k) => (v isa Bool ? v : Float64(v)))
    end
    return (; pairs...)
end

const EST_CFG = load_gains(ESKF_CFG_PATH)   # flat v4 schema: best_gains IS the est_cfg

"""
    run_closed_loop_eskf(family, ctrl_kw, tr; noise_seed) -> (probe, ref, mode, bus)

One closed-loop run: controller `family` (:asmc/:pid) with gains `ctrl_kw`,
state feedback from a FRESH ESKF V3 (tuned seed-4 config) fed by a FRESH
:realistic SensorSuite. Mirrors tune_controller_v2.run_controller_v2's cfg /
reference-build discipline, routed through SchedulerModV2.run_hybrid_v2.
"""
function run_closed_loop_eskf(family::Symbol, ctrl_kw::NamedTuple, tr; noise_seed::Int)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    suite = Main.SensorModV2.build_suite(:realistic; seed=noise_seed, flow=true, fix_tier=:docking)
    est   = HarnessV2Mod.build_estimator_v3(EST_CFG, suite)   # fresh estimator per run

    asmc_o, _, pid_o = build_controller_v2(family, ctrl_kw)   # fresh controller per run

    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = :eskf,
        use_dhat      = false,             # matches all three controllers' tuning conditions
        use_asmc      = family == :asmc,
        use_mpc       = false,
        use_pid       = family == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(family),
        use_pose_fix  = true,
        pose_fix_tier = :docking,          # matches ESKF tuning (objective_v2 default)
        sensor_seed   = noise_seed,
    )

    # Reference: pinned combo_idx + Xoshiro(0), velref->posref adaptation —
    # byte-identical to run_controller_v2's deterministic path.
    path  = joinpath(tr.config_dir, "profiles", tr.profile_toml)
    prof  = TOML.parsefile(path)["profile"]
    cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
    ref   = Profiles.build(prof["builder"], cfg_r)
    if get(tr, :adapt, false)
        ref = Profiles.velref_to_posref(ref)
    end
    Profiles.publish!(ref)

    sol, bus = Main.SchedulerModV2.run_hybrid_v2(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=est, suite=suite, ref=ref,
        asmc_override=asmc_o, pid_override=pid_o)

    probe = get(Main.SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, tr.run_mode, bus
end

# ---- metrics (patterns from hybrid_ctrl/estimator_tuning/compare_controllers_eskf_pose.jl)
_wrapdiff(a, b) = atan(sin(a - b), cos(a - b))

function estimator_error(probe)
    rmse(h, t) = sqrt(mean((h .- t) .^ 2))
    tvx  = [p.u[1]  for p in probe]; hvx  = [p.xhat[1] for p in probe]
    tvy  = [p.u[2]  for p in probe]; hvy  = [p.xhat[2] for p in probe]
    tpd  = [p.u[3]  for p in probe]; hpd  = [p.xhat[3] for p in probe]
    tpsi = [p.u[4]  for p in probe]; hpsi = [p.xhat[4] for p in probe]
    tX   = [p.u[17] for p in probe]; hX   = [p.xhat[5] for p in probe]
    tY   = [p.u[18] for p in probe]; hY   = [p.xhat[6] for p in probe]
    psi_err = _wrapdiff.(hpsi, tpsi)
    pos_err = sqrt.((hX .- tX).^2 .+ (hY .- tY).^2)
    vel_err = sqrt.((hvx .- tvx).^2 .+ (hvy .- tvy).^2)
    return (vel=sqrt(mean(vel_err.^2)), rate=rmse(hpd, tpd),
            pos=sqrt(mean(pos_err.^2)), heading=sqrt(mean(psi_err.^2)))
end

function controller_error_pose(probe, ref)
    poserr = Float64[]; headerr = Float64[]
    for p in probe
        t = p.t
        push!(poserr,  sqrt((p.u[17] - ref.xo(t))^2 + (p.u[18] - ref.yo(t))^2))
        push!(headerr, abs(_wrapdiff(p.u[4], ref.psi(t))))
    end
    return (pos=sqrt(mean(poserr.^2)), heading=sqrt(mean(headerr.^2)),
            pos_final=poserr[end], heading_final=headerr[end])
end

function run_row(label::String, family::Symbol, ctrl_kw::NamedTuple, tr, noise_seed::Int)
    probe, ref, mode, bus = run_closed_loop_eskf(family, ctrl_kw, tr; noise_seed=noise_seed)
    m = controller_metrics(probe, ref, mode)
    e = estimator_error(probe)
    c = mode == :pose ? controller_error_pose(probe, ref) :
          (pos=NaN, heading=NaN, pos_final=NaN, heading_final=NaN)
    Main.SchedulerMod.clear_probe_log!(bus)
    score = m.tracking + Main.LAMBDA_CE * (m.ce / Main.V_MAX)   # tuning-objective scalarization
    return (controller=label, trajectory=string(tr.name), mode=string(mode),
            noise_seed=noise_seed, ok=m.ok,
            tracking=m.tracking, ce=m.ce, chatter=m.chatter, score=score,
            est_vel=e.vel, est_rate=e.rate, est_pos=e.pos, est_heading=e.heading,
            ctrl_pos=c.pos, ctrl_heading=c.heading,
            ctrl_pos_final=c.pos_final, ctrl_head_final=c.heading_final)
end

# ---- CLI ---------------------------------------------------------------------
function parse_args(argv)
    a = Dict{String,Any}("seed" => 0, "tier" => "train12",
                         "out" => joinpath(ROOT, "hybrid_ctrl_v2", "runs_controller_eskf_v3"),
                         "smoke" => false, "aggregate" => false)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed";           a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--tier";       a["tier"] = argv[i+1]; i += 2
        elseif arg == "--out";        a["out"] = argv[i+1]; i += 2
        elseif arg == "--smoke";      a["smoke"] = true; i += 1
        elseif arg == "--aggregate";  a["aggregate"] = true; i += 1
        else; error("unknown arg $arg"); end
    end
    return a
end

_fail_row(label, tr, ns) = (controller=label, trajectory=string(tr.name), mode=string(tr.run_mode),
    noise_seed=ns, ok=false, tracking=NaN, ce=NaN, chatter=NaN, score=NaN,
    est_vel=NaN, est_rate=NaN, est_pos=NaN, est_heading=NaN,
    ctrl_pos=NaN, ctrl_heading=NaN, ctrl_pos_final=NaN, ctrl_head_final=NaN)

function main()
    a = parse_args(ARGS)
    trajs = trajset(Symbol(a["tier"]), "trajectory_files_run_0p5_main")
    ctrl_kws = [(label, family, load_gains(path)) for (label, family, path) in CONTROLLERS]
    println("ESKF V3 config: $ESKF_CFG_PATH")
    println("Tier $(a["tier"]): $(length(trajs)) trajs; controllers: $(join([c[1] for c in ctrl_kws], ", "))")

    if a["smoke"]
        tr = trajs[1]   # octagon_easy
        for (label, family, kw) in ctrl_kws
            probe, ref, mode, bus = run_closed_loop_eskf(family, kw, tr; noise_seed=101)
            m = controller_metrics(probe, ref, mode)
            e = estimator_error(probe)
            Main.SchedulerMod.clear_probe_log!(bus)
            dmax = maximum(abs(p.xhat[1] - p.u[1]) for p in probe)
            println("  $label: ticks=$(length(probe)) tracking=$(round(m.tracking, digits=4)) " *
                    "ce=$(round(m.ce, digits=2)) chatter=$(round(m.chatter, digits=3)) ok=$(m.ok) | " *
                    "est vel=$(round(e.vel, digits=4)) rate=$(round(e.rate, digits=4)) " *
                    "pos=$(round(e.pos, digits=4)) heading=$(round(e.heading, digits=4)) | " *
                    "max|x̂Vx−Vx|=$(round(dmax, digits=5)) $(dmax > 0 ? "(real ESKF)" : "(WARNING: oracle-like)")")
        end
        # reproducibility: repeat the first controller, must be bit-identical
        (label, family, kw) = ctrl_kws[1]
        p1, r1, m1, b1 = run_closed_loop_eskf(family, kw, tr; noise_seed=101)
        mm1 = controller_metrics(p1, r1, m1); Main.SchedulerMod.clear_probe_log!(b1)
        p2, r2, m2, b2 = run_closed_loop_eskf(family, kw, tr; noise_seed=101)
        mm2 = controller_metrics(p2, r2, m2); Main.SchedulerMod.clear_probe_log!(b2)
        println("  repro check ($label x2): $(mm1.tracking) vs $(mm2.tracking) -> $(mm1.tracking === mm2.tracking ? "BIT-IDENTICAL" : "MISMATCH")")
        println("=== SMOKE DONE ===")
        return
    end

    if a["aggregate"]
        out = a["out"]
        files = filter(f -> startswith(f, "runs_seed") && endswith(f, ".csv"), readdir(out))
        isempty(files) && error("no runs_seed*.csv under $out")
        df = reduce(vcat, [CSV.read(joinpath(out, f), DataFrame) for f in files])
        _m(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
        _s(x) = (v = filter(isfinite, x); length(v) < 2 ? NaN : std(v))
        summ = combine(groupby(df, [:controller, :trajectory]),
            :score => _m => :score_mean, :score => _s => :score_std,
            :tracking => _m => :tracking_mean, :ce => _m => :ce_mean, :chatter => _m => :chatter_mean,
            :est_vel => _m => :est_vel_mean, :est_rate => _m => :est_rate_mean,
            :est_pos => _m => :est_pos_mean, :ok => (x -> sum(x)) => :n_ok, nrow => :n)
        per_ctrl = combine(groupby(df, :controller),
            :score => _m => :score_mean, :score => _s => :score_std,
            :tracking => _m => :tracking_mean, :tracking => _s => :tracking_std,
            :ce => _m => :ce_mean, :chatter => _m => :chatter_mean,
            :est_vel => _m => :est_vel_mean, :est_rate => _m => :est_rate_mean,
            :est_pos => _m => :est_pos_mean, :est_heading => _m => :est_heading_mean,
            :ok => (x -> sum(x)) => :n_ok, nrow => :n)
        CSV.write(joinpath(out, "summary_by_traj.csv"), summ)
        CSV.write(joinpath(out, "summary_by_controller.csv"), per_ctrl)
        println(per_ctrl)
        println("=== AGGREGATE done -> $out ===")
        return
    end

    a["seed"] in 1:5 || error("--seed N in 1..5 required (got $(a["seed"]))")
    ns = 100 + a["seed"]
    out = a["out"]; mkpath(out)
    rows = NamedTuple[]
    for (label, family, kw) in ctrl_kws
        for tr in trajs
            println("  [run] $(rpad(label, 8)) $(rpad(string(tr.name), 24)) ns=$ns")
            row = try
                run_row(label, family, kw, tr, ns)
            catch err
                @warn "run failed" controller=label traj=tr.name seed=ns exception=err
                _fail_row(label, tr, ns)
            end
            push!(rows, row)
        end
    end
    CSV.write(joinpath(out, "runs_seed$(a["seed"]).csv"), DataFrame(rows))
    println("[seed $(a["seed"]) / ns=$ns] done -> $(joinpath(out, "runs_seed$(a["seed"]).csv"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
