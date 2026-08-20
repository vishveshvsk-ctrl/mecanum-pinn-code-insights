#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/controller_tuning/run_stage.jl — CLI entry point
# =============================================================================
# Parses stage/controller/seed/tier/cap arguments, assembles the v2 modules,
# executes ONE seed of ONE stage, writes best_config.json / trials.json /
# trace.csv / phase_summary.json under a fresh per-stage/per-seed output root.
#
# RESUMABILITY (brief §9): true phase-2-resume-from-phase-1 would require
# serializing BlackBoxOptim's population/step-size state across process
# restarts, which is out of scope here. What IS implemented is the cheaper,
# still load-bearing half of the contract: writing into a non-empty existing
# seed directory is REFUSED unless `--force` -- so an interrupted run never
# gets silently overwritten by an unrelated restart; the operator must
# explicitly opt into a fresh run via `--force` (which fully re-runs both
# phases, it does not resume).
#
#   --stage        0 | 1 | 2 | 3
#   --controller   asmc | pid | mpc
#   --seed         1..5
#   --trajset-screen / --trajset-full   tier names (default :screen / :train_full)
#   --noise        clean | noisy
#   --p1-cap / --p2-cap                 hard eval caps per phase
#   --rel-tol / --window                convergence-gate parameters
#   --refiner      bobyqa | local
#   --out          output root
#   --pin-kmax     ASMC only -- unchanged ASMC_SPACE_PIN + K_MAX_PIN
#   --smoke        one eval per tier (SCREEN/TRAIN_FULL/TEST), no optimization
#   --force        allow writing into a non-empty seed dir
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl"));       using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "pid_cascade.jl"));    using .PIDCascadeMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "mpc_design.jl"));     using .MPCDesignMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "stage_objective.jl")); using .StageObjectiveMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "optimizer_stage.jl")); using .StageOptimizerMod
using JSON
using Dates

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
function parse_args(argv)
    a = Dict{String,Any}(
        "stage" => "0", "controller" => "asmc", "seed" => 1, "run-dir" => "trajectory_files_run_0p5_main",
        "trajset-screen" => "screen", "trajset-full" => "train_full", "noise" => "clean",
        "p1-cap" => 250, "p2-cap" => 60, "rel-tol" => 1e-3, "window" => 20,
        "refiner" => "bobyqa", "refine-halfwidth" => 0.25,
        "out" => "hybrid_ctrl_v2/runs_controller_v2_stage0", "pin-kmax" => false,
        "smoke" => false, "force" => false,
        "lambda-chatter" => 0.0, "lambda-kmax" => 0.0, "lambda-gamma" => 0.0, "recovery-weight" => 0.1,
        "pid-v2" => false, "pid-variant" => "fb", "noise-replicates" => 1,
        "asmc-v2" => false, "warm-from" => "", "np-grid" => "15,30",
        "eps-floor-xy" => 0.0, "eps-floor-psi" => 0.0,
        "tau-ceiling-hi" => 100.0, "lam-psi-hi" => 20.0,
        # rho_auth box (ASMC_SPACE_V2 dim 4, replacing tau_ceiling). Defaults
        # reproduce the space exactly; both ends are physically anchored, see
        # the run_asmc_v2 comment.
        "rho-lo" => 0.0, "rho-hi" => 0.0,
        # Tracking metric: "v2" (final+max, the default) or "v3" (adds a
        # time-normalised integral term, a per-trajectory position scaler and a
        # trigonometric heading error). See StageObjectiveMod.
        "metric" => "v2",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--stage"; a["stage"] = argv[i+1]; i += 2
        elseif arg == "--controller"; a["controller"] = argv[i+1]; i += 2
        elseif arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--run-dir"; a["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--trajset-screen"; a["trajset-screen"] = argv[i+1]; i += 2
        elseif arg == "--trajset-full"; a["trajset-full"] = argv[i+1]; i += 2
        elseif arg == "--noise"; a["noise"] = argv[i+1]; i += 2
        elseif arg == "--p1-cap"; a["p1-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p2-cap"; a["p2-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--rel-tol"; a["rel-tol"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--window"; a["window"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--refiner"; a["refiner"] = argv[i+1]; i += 2
        elseif arg == "--refine-halfwidth"; a["refine-halfwidth"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        elseif arg == "--pin-kmax"; a["pin-kmax"] = true; i += 1
        elseif arg == "--smoke"; a["smoke"] = true; i += 1
        elseif arg == "--force"; a["force"] = true; i += 1
        elseif arg == "--lambda-chatter"; a["lambda-chatter"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--lambda-kmax"; a["lambda-kmax"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--lambda-gamma"; a["lambda-gamma"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--recovery-weight"; a["recovery-weight"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--pid-v2"; a["pid-v2"] = true; i += 1
        elseif arg == "--pid-variant"; a["pid-variant"] = argv[i+1]; i += 2
        elseif arg == "--noise-replicates"; a["noise-replicates"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--asmc-v2"; a["asmc-v2"] = true; i += 1
        elseif arg == "--eps-floor-xy"; a["eps-floor-xy"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--eps-floor-psi"; a["eps-floor-psi"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--tau-ceiling-hi"; a["tau-ceiling-hi"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--lam-psi-hi"; a["lam-psi-hi"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--metric"; a["metric"] = argv[i+1]; i += 2
        elseif arg == "--rho-lo"; a["rho-lo"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--rho-hi"; a["rho-hi"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--warm-from"; a["warm-from"] = argv[i+1]; i += 2
        elseif arg == "--np-grid"; a["np-grid"] = argv[i+1]; i += 2
        else; error("run_stage.jl: unknown arg $arg"); end
    end
    return a
end

_jsonable(kw::NamedTuple) = Dict(string(k) => (v isa AbstractVector || v isa StaticArrays.SVector ? collect(v) :
                                               v isa AbstractMatrix ? [collect(r) for r in eachrow(v)] : v)
                                 for (k, v) in pairs(kw))

# CHECKPOINT_EVERY -- how often (in evals) StageOptimizerMod.optimize_staged
# invokes the checkpoint callback below. Per-user direction, added after the
# PID v2 FB 5-seed run demonstrated the gap: nothing about a run's progress
# (decoded gains, not just the trace.csv score) was persisted to disk until
# the ENTIRE run finished, so a crash mid-run (or just wanting to inspect
# where things stand) had no recovery point. 10 evals x ~90-120s/eval on the
# 12-trajectory train12 tier is a a few-minutes-of-compute risk window, not
# hours.
const CHECKPOINT_EVERY = 10

"""
    _make_checkpoint_cb(checkpoint_path, space, freeze; expand=identity) -> Function

Builds the `checkpoint_cb` `StageOptimizerMod.optimize_staged` calls every
`CHECKPOINT_EVERY` evals. `optimizer_stage.jl` itself knows nothing about
`decode`/`space`/gain field names -- it only hands back the RAW theta
vectors -- so decoding happens HERE, at the call site that already knows
`space`/`freeze`/`expand` (MPC's ratio-space `expand_ratio_space`, in
particular, is what turns `q_vel_ratio`/`S_scale` into actual `Q_pose`/`R`/
`S`; PID v2 and ASMC don't need it, `expand` defaults to `identity`).
Overwrites a SINGLE rolling JSON file (not one file per checkpoint, per-user
direction: "logging... for every 10... for not losing the compute", read as
one up-to-date checkpoint, not an unbounded pile of numbered files) with both
the just-evaluated ("current") and best-so-far ("best") decoded gains, so a
crash loses at most `CHECKPOINT_EVERY` evals of compute, not the whole run.
Write failures are caught and warned, never allowed to crash the run they're
protecting.
"""
function _make_checkpoint_cb(checkpoint_path::String, space, freeze::NamedTuple; expand::Function=identity)
    return function (n::Int, tier::Symbol, theta::Vector{Float64}, score::Float64,
                     best_theta::Vector{Float64}, best_score::Float64)
        try
            cur_gains  = _jsonable(expand(merge(freeze, decode(theta, space))))
            best_gains = isempty(best_theta) ? cur_gains :
                         _jsonable(expand(merge(freeze, decode(best_theta, space))))
            open(checkpoint_path, "w") do io
                JSON.print(io, Dict(
                    "iter" => n, "tier" => string(tier), "timestamp" => string(Dates.now()),
                    "current_score" => score, "current_gains" => cur_gains,
                    "best_score" => best_score, "best_gains" => best_gains,
                ), 2)
            end
        catch e
            @warn "checkpoint write failed (run continues -- this only affects crash recovery)" exception = e
        end
    end
end

# -----------------------------------------------------------------------------
# Stage 1 — ASMC: single-phase (space unchanged from tune_controller.jl).
# -----------------------------------------------------------------------------
function run_asmc(trajs_screen, trajs_full, a, outdir::String)
    space = space_for(:asmc; pin_kmax=a["pin-kmax"])
    lo, hi = flat_bounds(space)
    oracle = Symbol(a["noise"])
    mkobj(trajs) = make_stage_objective(:asmc, space, trajs, oracle; seed=a["seed"],
        lambda_chatter=a["lambda-chatter"], lambda_kmax=a["lambda-kmax"], lambda_gamma=a["lambda-gamma"],
        noise_replicates=a["noise-replicates"])
    checkpoint_cb = _make_checkpoint_cb(joinpath(outdir, "checkpoint.json"), space, (;))
    best, best_score, trials, diag = optimize_staged(mkobj(trajs_screen), mkobj(trajs_full), lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"),
        checkpoint_every=CHECKPOINT_EVERY, checkpoint_cb=checkpoint_cb)
    kw = decode(best, space)
    gains = _jsonable(kw)
    if a["pin-kmax"]
        gains["K_max_x"] = K_MAX_PIN.K_max_x; gains["K_max_y"] = K_MAX_PIN.K_max_y; gains["K_max_psi"] = K_MAX_PIN.K_max_psi
    end
    phase_summary = Dict("phase1" => Dict("evals"=>diag.phase1_evals), "phase2" => Dict("evals"=>diag.phase2_evals))
    return gains, best_score, trials, diag, phase_summary
end

# -----------------------------------------------------------------------------
# Stage 1 (v2) — ASMC v2: single-phase over ASMC_SPACE_V2 (instructions/asmc-
# v2-physical-gain-bounds.md §7.3). Same 6-dim dxNES shape as v1's run_asmc --
# ASMC v2 keeps K_max_*/lam_*_min/mu_xy/mu_psi/K_x0/K_y0/K_psi0 all DERIVED
# (ControllerV2Mod.ASMCControllerV2, via default_physical_limits()), so unlike
# PID v2 there is no freeze needed: every field the tuner doesn't search
# already has a physically-derived default wired into build_controller_v2.
# Routed through StageObjectiveMod.make_stage_objective's `is_asmc_v2` check
# (decoded kw carries lam_x_max, ASMC_SPACE_V2's discriminating key -- v1's
# ASMC_SPACE has none), so `run_asmc` (v1) above is untouched by this addition.
# -----------------------------------------------------------------------------
function run_asmc_v2(trajs_screen, trajs_full, a, outdir::String; warm_start=nothing)
    # Box overrides (--tau-ceiling-hi / --lam-psi-hi; defaults reproduce
    # ASMC_SPACE_V2 exactly). Added for the eps-floor runs, where tau_ceiling
    # railed at 100 on 5/5 seeds and lam_psi_max at 20 on 2/5 (the brief's
    # "box truncates the optimum" case) -- the widened run checks whether the
    # optimum is interior beyond the rail. Acceptance of any result past the
    # old rail is gated on the max(K)/K_max_base diagnostic (launch brief §3/§6).
    # rho_auth's box is PHYSICALLY ANCHORED at both ends, not chosen to bracket
    # degenerate limits. rho is defined by z* = 1/2 at |s| = rho*eps, so both ends
    # follow from the measured sliding-surface excursion (_tmp/asmc_surface_scale.jl,
    # |s|/eps recovered exactly as atanh(|Msw/K|) over train12):
    #
    #   LOWER, rho >= tanh(1) = 0.762. At exactly one boundary layer -- the width
    #   the controller is DESIGNED to hold, i.e. it is sliding, not struggling --
    #   z*(|s|=eps) = tanh(1)/(tanh(1)+rho). Requiring the adaptation to claim no
    #   more than HALF its authority at the design tolerance gives rho >= tanh(1).
    #   Below that, tolerance-level error alone buys majority authority, which is
    #   the parking regime this law exists to remove. (The as-designed sigma leak
    #   re-expressed in this form is rho = 0.383-0.831, straddling exactly this
    #   line -- consistent with it having pushed K to the ceiling.)
    #
    #   UPPER, rho <= 9*3 = 27. The reference excursion is 3*eps, which is NOT a
    #   number picked for this box: it is the scale the as-designed sigma leak
    #   itself uses, its gate being exp(decay_k*(1 - s^2/(9 eps^2))), unity at
    #   s = 3 eps. Since z* = s/(s+rho), z*(3eps) = 0.9 at rho = 3/9 and 0.1 at
    #   rho = 27, so the box spans the full usable authority range at the design's
    #   own excitation scale and no more.
    #
    #   REJECTED ANCHOR: the p99 of the measured |s|/eps distribution. It is
    #   dominated by lost-surface ticks -- |tanh| saturates numerically on 16-33%
    #   of ticks for spiral_orbit_stress and coupled_vomega_stress y, and
    #   ellipse_stress_tangent reaches 13.2 eps with ZERO censoring only because
    #   the 1e-12 cutoff is arbitrary; at 13 eps, tanh is saturated to ten decimals
    #   and the regime is identical. Sizing an authority scale off that tail sizes
    #   it to a failure mode, where more gain was measured to buy nothing (pose
    #   error grew 0.093 -> 0.230 m with K flat at its ceiling).
    #
    # An earlier draft used 2*v_max/eps = 63 from lambda_schedule's lam*|e| <= v_max
    # guarantee. That is a valid bound on |s| but a useless box edge: it puts half
    # authority at |s_psi| = 5.04 rad/s against a measured max of ~5 eps = 0.4, so
    # the top ~90% of the box was dead space where the objective is flat in rho.
    lo_rho = a["rho-lo"] > 0 ? a["rho-lo"] : nothing
    hi_rho = a["rho-hi"] > 0 ? a["rho-hi"] : nothing
    space = map(ASMC_SPACE_V2) do (n, l, sc, lo_, hi_)
        lo2 = (n == "rho_auth" && lo_rho !== nothing) ? lo_rho : lo_
        hi2 = n == "rho_auth"    ? something(hi_rho, hi_) :
              n == "tau_ceiling" ? a["tau-ceiling-hi"] :
              n == "lam_psi_max" ? a["lam-psi-hi"] : hi_
        (n, l, sc, lo2, hi2)
    end
    lo, hi = flat_bounds(space)
    oracle = Symbol(a["noise"])
    # Boundary-layer floors are OUTSIDE the 4-dim search space; inject as a freeze
    # (default 0.0 = struct default, so existing callers are unaffected). The
    # widened configuration (eps_floor_xy=0.02, eps_floor_psi=0.08) makes the layer
    # reachable at operating K -- see chat-handoff/asmc_v2_relaunch_pid_revalidation_handoff.md.
    #
    # The four gain-law flags are frozen ON together. They are not independent
    # options: rho_auth is inert without use_demand_k (stage_objective.jl's
    # reachability guard ERRORS on that combination), and use_demand_k normalises z
    # against kmax_schedule's output, which is only physically meaningful with
    # kmax_contact_b (the uncorrected schedule feeds ACTUATOR-side W_ff into a
    # CONTACT-side gate and collapses to its floor 38-67% of ticks).
    # kmax_sched_floor=K_floor keeps that denominator off its degenerate floor;
    # enforce_k_floor guarantees K >= K_floor at all times.
    kfloor = ControllerV2Mod.ASMCControllerV2(lim=default_physical_limits()).K_floor
    freeze = (eps_floor_xy=a["eps-floor-xy"], eps_floor_psi=a["eps-floor-psi"],
              use_demand_k=true, kmax_contact_b=true, enforce_k_floor=true,
              kmax_sched_floor=kfloor)
    mkobj(trajs) = make_stage_objective(:asmc, space, trajs, oracle; seed=a["seed"], freeze=freeze,
        lambda_chatter=a["lambda-chatter"], lambda_kmax=a["lambda-kmax"], lambda_gamma=a["lambda-gamma"],
        noise_replicates=a["noise-replicates"], metric=Symbol(a["metric"]))
    checkpoint_cb = _make_checkpoint_cb(joinpath(outdir, "checkpoint.json"), space, freeze)
    best, best_score, trials, diag = optimize_staged(mkobj(trajs_screen), mkobj(trajs_full), lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"),
        checkpoint_every=CHECKPOINT_EVERY, checkpoint_cb=checkpoint_cb, warm_start=warm_start)
    kw = merge(freeze, decode(best, space))
    gains = _jsonable(kw)
    phase_summary = Dict("phase1" => Dict("evals"=>diag.phase1_evals), "phase2" => Dict("evals"=>diag.phase2_evals))
    return gains, best_score, trials, diag, phase_summary
end

# -----------------------------------------------------------------------------
# Stage 2 — PID: sequential cascade :inner -> :outer -> :joint.
# -----------------------------------------------------------------------------
function run_pid(trajs_screen, trajs_full, a, outdir::String)
    oracle = Symbol(a["noise"])
    all_trials = NamedTuple[]
    phase_summary = Dict{String,Any}()
    frozen = (;)
    result_kw = (;)
    final_score = Inf
    for phase in (:inner, :outer, :joint)
        space, freeze = pid_phase_space(phase, frozen)
        lo, hi = flat_bounds(space)
        mkobj(trajs) = make_stage_objective(:pid, space, trajs, oracle; seed=a["seed"], freeze=freeze,
            lambda_chatter=a["lambda-chatter"],   # was dropped -- see run_pid_v2
            recovery_weight=a["recovery-weight"], noise_replicates=a["noise-replicates"])
        checkpoint_cb = _make_checkpoint_cb(joinpath(outdir, "checkpoint_$(phase).json"), space, freeze)
        best, best_score, trials, diag = optimize_staged(mkobj(trajs_screen), mkobj(trajs_full), lo, hi;
            method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
            p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
            refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace_$(phase).csv"),
            checkpoint_every=CHECKPOINT_EVERY, checkpoint_cb=checkpoint_cb)
        kw_raw = decode(best, space)
        kw_raw = phase == :joint ? regroup_joint(kw_raw) : kw_raw
        result_kw = merge(freeze, kw_raw)
        append!(all_trials, [merge(t, (cascade_phase=phase,)) for t in trials])
        phase_summary[string(phase)] = Dict("best_score"=>best_score, "phase1_evals"=>diag.phase1_evals,
                                            "phase2_evals"=>diag.phase2_evals, "stop_reason"=>string(diag.stop_reason))
        frozen = result_kw
        final_score = best_score
    end
    gains = _jsonable((Kp=result_kw.Kp, Ki=result_kw.Ki, Kd=result_kw.Kd, I_max=result_kw.I_max,
                       Kp_pos=result_kw.Kp_pos, Kd_pos=result_kw.Kd_pos))
    diag_overall = (converged=true, stop_reason=:cascade_complete,
                    phase1_evals=sum(v["phase1_evals"] for v in values(phase_summary)),
                    phase2_evals=sum(v["phase2_evals"] for v in values(phase_summary)))
    return gains, final_score, all_trials, diag_overall, phase_summary
end

# -----------------------------------------------------------------------------
# Stage 2 (v2) — PID v2: single-phase over PID_SPACE_V2 (instructions/pid-v2-
# imc-cascade-two-variants.md). No cascade: Kp/Ki are IMC-DERIVED from the
# searched lam_inner, not free parameters, so the v1 :inner->:outer->:joint
# sequencing has nothing to sequence. `N=4.0` (critical damping, brief §7.2)
# and `feedforward` (FB vs CT variant) are FROZEN, not searched -- passed via
# `freeze`, picked up by `build_controller_v2`'s `:pid` branch. Routed through
# `StageObjectiveMod.make_stage_objective`'s `is_pid_v2` check (decoded kw
# carries `lam_inner_x`, not `Kp`), so `run_pid` (v1 cascade) above is
# untouched by this addition.
# -----------------------------------------------------------------------------
function run_pid_v2(trajs_screen, trajs_full, a, outdir::String; feedforward::Bool, warm_start=nothing)
    space = PID_SPACE_V2
    lo, hi = flat_bounds(space)
    oracle = Symbol(a["noise"])
    freeze = (N=4.0, feedforward=feedforward)
    # lambda_chatter MUST be forwarded here. It was omitted originally, so
    # --lambda-chatter parsed fine and was then silently dropped for PID v2 --
    # make_stage_objective defaulted it to 0.0 and every PID run to date was
    # UNPRICED regardless of the flag. Same omission existed on run_pid (v1) and
    # run_mpc; both fixed alongside. Only the two ASMC paths ever honoured it.
    mkobj(trajs) = make_stage_objective(:pid, space, trajs, oracle; seed=a["seed"], freeze=freeze,
        lambda_chatter=a["lambda-chatter"],
        recovery_weight=a["recovery-weight"], noise_replicates=a["noise-replicates"],
        metric=Symbol(a["metric"]))
    checkpoint_cb = _make_checkpoint_cb(joinpath(outdir, "checkpoint.json"), space, freeze)
    best, best_score, trials, diag = optimize_staged(mkobj(trajs_screen), mkobj(trajs_full), lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"),
        checkpoint_every=CHECKPOINT_EVERY, checkpoint_cb=checkpoint_cb, warm_start=warm_start)
    kw = merge(freeze, decode(best, space))
    gains = _jsonable(kw)
    phase_summary = Dict("phase1" => Dict("evals"=>diag.phase1_evals), "phase2" => Dict("evals"=>diag.phase2_evals))
    return gains, best_score, trials, diag, phase_summary
end

# -----------------------------------------------------------------------------
# Stage 3 — MPC: outer Np_pose grid {10,15,20,30}, ratio-space search per grid point.
# -----------------------------------------------------------------------------
function run_mpc(trajs_screen, trajs_full, a, outdir::String)
    oracle = Symbol(a["noise"])

    # SWITCHED to MPC_SPACE_V2 (tau_cl per axis). This previously ran
    # MPCDesignMod.normalized_mpc_space() -- the SUPERSEDED 2-dim ratio search
    # (S_scale, q_vel_ratio) -- with `expand_ratio_space`, which rebuilt Q_pose/R
    # from bryson_weights(TOL, V_MAX) and thereby BYPASSED MPCControllerV2's own
    # derived bryson_Q_pose/bryson_R entirely. Worse, it searched S_scale in v1's
    # [1e-3, 1] bracket, which MPC_SPACE_V2's own docstring states "does NOT
    # transfer -- different Q normalization" (the v2 bracket was [0.25, 25], barely
    # overlapping). So MPC_SPACE_V2 was defined, documented as superseding, cited
    # in MPCControllerV2's docstring as "SEARCHED" -- and never actually used.
    #
    # Also dropped: the precomputed `Pt = terminal_weight(...)`, which was frozen
    # into the controller and then OVERWRITTEN on the first tick by
    # `mpc.P_terminal = P` inside mpc_wrench! whenever use_terminal=true (and
    # unused when false) -- dead in both branches.
    space = MPC_SPACE_V2
    lo, hi = flat_bounds(space)

    best_overall = nothing; best_overall_score = Inf; best_np = 15
    all_trials = NamedTuple[]
    phase_summary = Dict{String,Any}()
    # Np grid is a CLI arg (--np-grid, default "15,30"). It was hardcoded
    # (10,15,20,30); 10 and 20 are dropped by default because the solve-time
    # question that motivated a fine grid is now MEASURED and rules nothing out:
    # per-tick wall-clock is 1.4 / 1.9 / 2.8 / 5.6 ms mean for Np=10/15/20/30
    # against a 10 ms budget at 100 Hz (_tmp/mpc_solve_time.jl, taken WITH 5
    # competing processes, so upper bounds). Every horizon is admissible, so the
    # grid only has to separate the CONTROL effect, and {15,30} brackets it.
    #
    # NOTE the grid is no longer a pure horizon sweep: Q accumulates Np times
    # while P applies once, and P is now built from the TERMINAL tolerances
    # (100x heavier than the running ones), so changing Np also shifts the
    # running-vs-terminal balance. Real effect to interpret, not a confound.
    #
    # COST: each Np gets its OWN full p1-cap/p2-cap budget, so a seed runs
    # length(np_grid) COMPLETE searches back to back.
    np_grid = Tuple(parse.(Int, split(a["np-grid"], ",")))
    for Np in np_grid
        # lambda_chatter is frozen INTO the controller, not just the objective:
        # MPCControllerV2 derives S = lambda_chatter/(dV_max/rate_hz)^2, so the QP
        # prices slew internally at the same rate the objective prices chatter
        # externally. The two MUST carry the same value.
        freeze = (Np_pose=Np, lambda_chatter=a["lambda-chatter"])
        mkobj(trajs) = make_stage_objective(:mpc, space, trajs, oracle; seed=a["seed"], freeze=freeze,
            lambda_chatter=a["lambda-chatter"],   # was dropped -- see run_pid_v2
            # recovery_weight IS now forwarded, a deliberate reversal of the
            # earlier "flagged, not fixed" note. Rationale: a cross-controller
            # comparison is only meaningful if all three controllers are scored by
            # the IDENTICAL objective -- the same argument that keeps `chatter`
            # (an outcome metric, common to all) in the shared score and keeps
            # controller-specific parameter penalties OUT of it. MPC silently
            # omitting a term PID and ASMC both carry made its scores
            # non-comparable by construction.
            #
            # COST, stated explicitly: every MPC run predating this change was
            # scored with recovery_weight=0 and its `best_score` is NOT comparable
            # to runs after it. That cost is already sunk -- MPC has to be
            # re-tuned regardless, because every controller tuned to date was
            # chatter-unpriced (no launch bat ever passed --lambda-chatter, and
            # the V_MAX normaliser made any value ~30x too weak).
            recovery_weight=a["recovery-weight"],
            noise_replicates=a["noise-replicates"], metric=Symbol(a["metric"]))
        checkpoint_cb = _make_checkpoint_cb(joinpath(outdir, "checkpoint_Np$(Np).json"), space, freeze)
        best, best_score, trials, diag = optimize_staged(mkobj(trajs_screen), mkobj(trajs_full), lo, hi;
            method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
            p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
            refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace_Np$(Np).csv"),
            checkpoint_every=CHECKPOINT_EVERY, checkpoint_cb=checkpoint_cb)
        append!(all_trials, [merge(t, (Np_pose=Np,)) for t in trials])
        phase_summary["Np_pose=$Np"] = Dict("best_score"=>best_score, "phase1_evals"=>diag.phase1_evals,
                                            "phase2_evals"=>diag.phase2_evals, "stop_reason"=>string(diag.stop_reason))
        if best_score < best_overall_score
            best_overall_score = best_score; best_overall = best; best_np = Np
        end
    end
    kw = expand(merge((Np_pose=best_np, P_terminal=Pt), decode(best_overall, space)))
    gains = _jsonable((Q_pose=kw.Q_pose, R=kw.R, S=kw.S, Np_pose=best_np, P_terminal=Pt))
    diag_overall = (converged=true, stop_reason=:np_grid_complete,
                    phase1_evals=sum(v["phase1_evals"] for v in values(phase_summary)),
                    phase2_evals=sum(v["phase2_evals"] for v in values(phase_summary)))
    return gains, best_overall_score, all_trials, diag_overall, phase_summary
end

# -----------------------------------------------------------------------------
# Smoke mode: one default-gains eval per tier, no optimization.
# -----------------------------------------------------------------------------
function run_smoke(a)
    for tier in (:screen, :train_full, :test)
        trs = trajset(tier, a["run-dir"])
        println("=== SMOKE [$tier] default $(a["controller"]) gains, noise=$(a["noise"]) ($(length(trs)) trajs) ===")
        ctrl = Symbol(a["controller"])
        pid_v2_kw = ctrl == :pid && a["pid-v2"] ? (N=4.0, feedforward=(a["pid-variant"]=="ct")) : (;)
        is_v2 = ctrl == :mpc || (ctrl == :pid && a["pid-v2"]) || (ctrl == :asmc && a["asmc-v2"])
        asmc_o, mpc_o, pid_o = is_v2 ? build_controller_v2(ctrl, pid_v2_kw) : build_controller(ctrl, (;))
        n_fail = 0
        for tr in trs
            try
                probe, ref, mode, bus = run_controller_v2(ctrl, Symbol(a["noise"]), tr;
                    asmc_o=asmc_o, mpc_o=mpc_o, pid_o=pid_o, seed=a["seed"])
                m = controller_metrics(probe, ref, mode)
                SchedulerMod.clear_probe_log!(bus)
                m.ok || (n_fail += 1)
                println("  $(tr.name) [role=$(tr.role)]: tracking=$(round(m.tracking,digits=4)) ce=$(round(m.ce,digits=2)) ok=$(m.ok)")
            catch e
                n_fail += 1
                println("  $(tr.name): FAILED -- $e")
            end
        end
        n_fail == 0 || error("run_stage.jl --smoke: $n_fail/$(length(trs)) trajectories failed on tier $tier")
    end
    println("=== SMOKE OK (stage=$(a["stage"]) controller=$(a["controller"])) ===")
end

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
function main()
    a = parse_args(ARGS)

    if a["smoke"]
        run_smoke(a)
        return
    end

    ctrl = Symbol(a["controller"])
    trajs_screen = trajset(Symbol(a["trajset-screen"]), a["run-dir"])
    trajs_full   = trajset(Symbol(a["trajset-full"]),   a["run-dir"])

    ctrl_label = (ctrl == :pid && a["pid-v2"]) ? "pid_v2_$(a["pid-variant"])" :
                 (ctrl == :asmc && a["asmc-v2"]) ? "asmc_v2" : string(ctrl)
    outdir = joinpath(a["out"], "seed$(a["seed"])", "$(ctrl_label)_$(a["noise"])")
    if isdir(outdir) && !isempty(readdir(outdir)) && !a["force"]
        error("run_stage.jl: $outdir already exists and is non-empty. Pass --force to overwrite " *
              "(this does NOT resume -- both phases re-run from scratch; see module docstring).")
    end
    mkpath(outdir)

    println("\n===== Stage $(a["stage"]) / $ctrl_label / seed $(a["seed"]) — screen=$(length(trajs_screen)) trajs, " *
            "full=$(length(trajs_full)) trajs, noise=$(a["noise"]), metric=$(a["metric"]), lambda_chatter=$(a["lambda-chatter"]) =====")


    # metric + chatter price are echoed above because getting them wrong is
    # SILENT: every controller tuned before the chatter fix was scored unpriced
    # (no launch bat passed --lambda-chatter, default 0.0) and that survived
    # three box-widening rounds unnoticed.
    a["metric"] in ("v2","v3") || error("run_stage.jl: --metric must be v2 or v3, got $(a["metric"])")
    if a["metric"] == "v3" && a["lambda-chatter"] == 0.0
        @warn "run_stage.jl: --metric v3 with --lambda-chatter 0.0 -- chatter is UNPRICED. " *
              "The calibrated v3 value is LAMBDA_CHATTER_V3 = $(StageObjectiveMod.LAMBDA_CHATTER_V3). " *
              "Pass it explicitly unless an unpriced ablation is intended."
    end
    # --warm-from (PID v2 and ASMC v2): read a prior best_config.json's decoded
    # gains and hand them to optimize_staged's warm_start -- skips phase 1's
    # global dxNES search entirely and goes straight to a local BOBYQA refine
    # around the supplied point (per-user direction: FINE-TUNING off an
    # already-converged optimum -- e.g. re-checking a rail-bound result inside
    # a widened box -- not a fresh cold-start search). Key order MUST match
    # each space's declaration order (PID_SPACE_V2: lam_inner_x/_y/_psi;
    # ASMC_SPACE_V2: lam_x_max/lam_y_max/lam_psi_max/rho_auth).
    #
    # DIMENSION 4 CHANGED MEANING (tau_ceiling -> rho_auth). Rather than hardcode
    # a key list that can silently drift from the space again, read the key names
    # FROM the space itself: warm_theta is then wrong-by-construction impossible
    # as long as the archive carries the same names. A pre-swap archive (which has
    # tau_ceiling and no rho_auth) now fails LOUDLY with a message naming the
    # cause, instead of KeyError-ing or -- far worse -- silently mapping a
    # tau_ceiling value into rho_auth's range.
    warm_theta = nothing
    if !isempty(a["warm-from"]) && ((ctrl == :pid && a["pid-v2"]) || (ctrl == :asmc && a["asmc-v2"]))
        wc = JSON.parsefile(a["warm-from"])
        g  = wc["best_gains"]
        wspace = ctrl == :asmc ? ASMC_SPACE_V2 : PID_SPACE_V2
        keys_needed = [r[1] for r in wspace]
        missing_keys = [k for k in keys_needed if !haskey(g, k)]
        if !isempty(missing_keys)
            error("run_stage.jl --warm-from: $(a["warm-from"]) is missing $(missing_keys), " *
                  "which the current search space requires. If it carries `tau_ceiling`, it " *
                  "predates the ASMC_SPACE_V2 dimension-4 swap (tau_ceiling -> rho_auth) and " *
                  "CANNOT be warm-started from -- the coordinates mean different things. " *
                  "Start that run cold.")
        end
        warm_theta = log.([Float64(g[k]) for k in keys_needed])
    end

    gains, best_score, trials, diag, phase_summary =
        ctrl == :asmc && a["asmc-v2"] ? run_asmc_v2(trajs_screen, trajs_full, a, outdir; warm_start=warm_theta) :
        ctrl == :asmc ? run_asmc(trajs_screen, trajs_full, a, outdir) :
        ctrl == :pid && a["pid-v2"] ? run_pid_v2(trajs_screen, trajs_full, a, outdir; feedforward=(a["pid-variant"]=="ct"), warm_start=warm_theta) :
        ctrl == :pid  ? run_pid(trajs_screen, trajs_full, a, outdir) :
        ctrl == :mpc  ? run_mpc(trajs_screen, trajs_full, a, outdir) :
        error("run_stage.jl: unknown controller $ctrl")

    open(joinpath(outdir, "best_config.json"), "w") do io
        JSON.print(io, Dict(
            "controller" => ctrl_label, "noise" => a["noise"], "stage" => a["stage"], "seed" => a["seed"],
            "best_score" => best_score, "best_gains" => gains,
            "converged" => diag.converged, "stop_reason" => string(diag.stop_reason),
            "phase1_evals" => diag.phase1_evals, "phase2_evals" => diag.phase2_evals,
            "trajset_screen" => a["trajset-screen"], "trajset_full" => a["trajset-full"],
        ), 2)
    end
    open(joinpath(outdir, "trials.json"), "w") do io
        JSON.print(io, [Dict("iter"=>t.iter, "tier"=>string(t.tier), "score"=>t.score,
                             "tracking"=>get(t, :tracking, missing), "ce"=>get(t, :ce, missing))
                        for t in trials], 2)
    end
    open(joinpath(outdir, "phase_summary.json"), "w") do io
        JSON.print(io, phase_summary, 2)
    end
    println("[$ctrl_label seed$(a["seed"])] best score=$(round(best_score,digits=4)) stop_reason=$(diag.stop_reason) → $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
