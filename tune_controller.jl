#!/usr/bin/env julia
# =============================================================================
# tune_controller.jl — controller tuning on an ORACLE state feed
# =============================================================================
# Bootstrap controller tuning while the real KF/SMO is still being fixed.
# The controllers are fed the TRUE plant state (+ optional relative noise) via
# OracleEstimator, so the estimator is a clean controlled variable.
#
#   --noise clean  → true labels (upper bound)
#   --noise noisy  → 10% relative noise on position, 5% on IMU/gyro states
#
# Objective (per trajectory): velocity NRMSE (velref) or pose RMSE (posref),
# plus a mild control-effort penalty (mean Σ|v_cmd| normalised by V_max).
# Handles both VelRef and PosRef trajectories in one subset.
# =============================================================================

using Pkg
Pkg.activate(".")

include("run_one.jl")
using .Profiles, .DataStore

include("hybrid_ctrl/config.jl");      using .HybridConfigMod
include("hybrid_ctrl/bus.jl");         using .BusMod
include("hybrid_ctrl/plant.jl");       using .PlantMod
include("hybrid_ctrl/sensors.jl");     using .SensorMod
include("hybrid_ctrl/estimators.jl");  using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");       using .FuzzyMod
include("hybrid_ctrl/mixer.jl");       using .MixerMod
include("hybrid_ctrl/scheduler.jl");   using .SchedulerMod

using StaticArrays
using Statistics: mean
using Random
using LinearAlgebra
using TOML
using JSON

# Reproducibility: multithreaded BLAS gives run-to-run FP variation (different
# reduction order), which a chaotically-sensitive closed loop amplifies into
# divergent-vs-stable outcomes.  Pin to 1 thread so each run is deterministic.
LinearAlgebra.BLAS.set_num_threads(1)

# Global optimizer backend (BlackBoxOptim dxNES / adaptive DE).  Loaded at
# top-level (before main) so its methods are in-world when main runs.
const BBO = Base.require(Main, :BlackBoxOptim)

const V_MAX = 24.0          # for control-effort normalisation
const LAMBDA_CE = 0.05      # mild control-effort penalty weight

# -----------------------------------------------------------------------------
# ABSOLUTE-ERROR tolerances (physical units) — the tuning targets.
# The objective normalises each error metric by its tolerance, so a term = 1.0
# means "exactly at target"; < 1 is better.  Score is the mean of these ratios.
# -----------------------------------------------------------------------------
const TOL = (
    vel_rms     = 1e-3,   # m/s    RMS translational velocity error (Vx, Vy)   [1 mm/s]
    yawrate_rms = 1e-3,   # rad/s  RMS yaw-rate error (ψ̇)                       [1 mrad/s]
    pos_final   = 1e-2,   # m      final position error                        [1 cm]
    pos_max     = 1e-1,   # m      max position error                          [10 cm]
    head_final  = 1e-2,   # rad    final heading error
    head_max    = 1e-1,   # rad    max heading error
)

_wrapang(a) = atan(sin(a), cos(a))

# -----------------------------------------------------------------------------
# Per-controller search spaces: (name, len, scale, lo, hi)
# -----------------------------------------------------------------------------
const ASMC_SPACE = [
    ("gamma_x",   1, :log, 1.0,  100.0),
    ("gamma_y",   1, :log, 1.0,  100.0),
    ("gamma_psi", 1, :log, 1.0,  100.0),
    ("eps",       1, :log, 5e-3, 0.1),
    ("eps_psi",   1, :log, 0.02, 0.3),
    ("K_max_x",   1, :log, 20.0, 150.0),
    ("K_max_y",   1, :log, 20.0, 150.0),
    ("K_max_psi", 1, :log, 20.0, 150.0),
]

# Upper bounds raised (Kp 80→800, Ki 10→150, Kd 5→50): the physical limiter is
# the motor model's |i|≤i_max / |V|≤V_max saturation in the mixer, NOT an
# artificial gain cap. The set-2 run saturated Kp on all axes + Ki on y/ψ, so
# the old caps were starving the controller of authority it is physically
# allowed to use. Wider box is a strict superset — prior optima stay reachable.
const PID_SPACE = [
    ("Kp",     3, :log, 5.0,  800.0),
    ("Ki",     3, :log, 0.1,  150.0),
    ("Kd",     3, :log, 0.05, 50.0),
    ("Kp_pos", 3, :log, 0.2,  5.0),
    ("Kd_pos", 3, :log, 0.1,  3.0),
]

const MPC_SPACE = [
    ("Q",       3, :log, 1.0,  200.0),   # velocity-mode state weights
    ("Q_pose",  6, :log, 0.5,  200.0),   # pose-mode state weights (x,y,ψ,Vx,Vy,ψ̇)
    ("R_scale", 1, :log, 1e-3, 1.0),
    ("S_scale", 1, :log, 1e-3, 1.0),
]

# Pinned-K_max ASMC variant: K_max removed from the search and FIXED at the
# ceiling (max robustness headroom — cost-free per the chatter analysis), with
# the YAW axis higher (body-frame heading carries the spin/coupled disturbances,
# and the seeds consistently drove K_max_psi highest). Only gamma + eps tuned.
# ASMC_SPACE[1:5] = gamma_x, gamma_y, gamma_psi, eps, eps_psi.
const ASMC_SPACE_PIN = ASMC_SPACE[1:5]
const K_MAX_PIN = (K_max_x=150.0, K_max_y=150.0, K_max_psi=300.0)

space_for(ctrl::Symbol; pin_kmax::Bool=false) =
                          ctrl == :asmc ? (pin_kmax ? ASMC_SPACE_PIN : ASMC_SPACE) :
                          ctrl == :pid  ? PID_SPACE :
                          ctrl == :mpc  ? MPC_SPACE :
                          error("unknown controller $ctrl")

function flat_bounds(space)
    lo = Float64[]; hi = Float64[]
    for (_, len, scale, l, h) in space
        for _ in 1:len
            push!(lo, scale == :log ? log(l) : l)
            push!(hi, scale == :log ? log(h) : h)
        end
    end
    return lo, hi
end

"Decode a raw theta vector into a NamedTuple of controller kwargs."
function decode(theta, space)
    idx = 1
    pairs = Pair{Symbol,Any}[]
    for (name, len, scale, lo, hi) in space
        vals = Float64[]
        for _ in 1:len
            lb = scale == :log ? log(lo) : lo
            ub = scale == :log ? log(hi) : hi
            v = clamp(theta[idx], lb, ub)
            push!(vals, scale == :log ? exp(v) : v)
            idx += 1
        end
        push!(pairs, Symbol(name) => (len == 1 ? vals[1] : SVector{len}(vals)))
    end
    return (; pairs...)
end

# -----------------------------------------------------------------------------
# Build a controller struct override from decoded kwargs.
# -----------------------------------------------------------------------------
function build_controller(ctrl::Symbol, kw::NamedTuple)
    # Qualify with ControllerMod: PIDController is also exported by OrdinaryDiffEq
    # (the solver step controller), so the bare name is ambiguous in Main.
    if ctrl == :asmc
        # Pinned-space decode has gains but no K_max → inject the fixed ceiling.
        # (Empty-kw smoke path keeps the struct defaults.)
        kwa = (haskey(kw, :gamma_x) && !haskey(kw, :K_max_x)) ? merge(kw, K_MAX_PIN) : kw
        return ControllerMod.ASMCController(; kwa..., rate_hz=1000.0), nothing, nothing
    elseif ctrl == :pid
        return nothing, nothing, ControllerMod.PIDController(; kw..., rate_hz=100.0)
    elseif ctrl == :mpc
        d = ControllerMod.MPCController(rate_hz=100.0)   # defaults (for empty-kw smoke)
        Q  = get(kw, :Q, d.Q)
        Qp = get(kw, :Q_pose, d.Q_pose)
        R  = haskey(kw, :R_scale) ? SVector{4}(fill(kw.R_scale, 4)) : d.R
        S  = haskey(kw, :S_scale) ? SVector{4}(fill(kw.S_scale, 4)) : d.S
        return nothing, ControllerMod.MPCController(Q=Q, Q_pose=Qp, R=R, S=S, rate_hz=100.0), nothing
    else
        error("unknown controller $ctrl")
    end
end

weights_for(ctrl::Symbol) = ctrl == :asmc ? (1.0, 0.0, 0.0) :
                            ctrl == :mpc  ? (0.0, 1.0, 0.0) :
                                            (0.0, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Run one closed-loop sim with an oracle feed + controller override.
# -----------------------------------------------------------------------------
function run_controller(ctrl::Symbol, kw::NamedTuple, oracle_kind::Symbol, tr; seed::Int=42, noise_scale::Float64=1.0)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = :oracle,
        use_dhat      = false,
        use_asmc      = ctrl == :asmc,
        use_mpc       = ctrl == :mpc,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = false,          # oracle supplies pose directly
        sensor_seed   = seed,
    )

    # Build the reference.  Pin combo_idx to a FEASIBLE trajectory (within the
    # platform's velocity/friction envelope) instead of a random draw — random
    # combos can demand Vy>0.6 m/s (infeasible) → error dominated by infeasibility,
    # not control.  Fixed Xoshiro(0) makes the sweep draw deterministic too.
    if get(tr, :combo_idx, nothing) === nothing
        ref = Profiles.pick_and_build(tr.config_dir, [tr.profile_toml];
                                      rng=Random.Xoshiro(hash(tr.profile_toml)))[1]
    else
        path  = joinpath(tr.config_dir, "profiles", tr.profile_toml)
        prof  = TOML.parsefile(path)["profile"]
        cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
        ref   = Profiles.build(prof["builder"], cfg_r)
    end
    # POSE-SUBSET-ONLY adapter gate: entries flagged adapt=true (the 4 former-
    # velref trajectories under --trajset pose) get lifted VelRef->PosRef here.
    # Untouched for every other trajset (default_trajs/_2/_3: adapt absent ->
    # false), so --trajset 3's :velocity path stays byte-for-byte reproducible.
    if get(tr, :adapt, false)
        ref = Profiles.velref_to_posref(ref)
    end
    Profiles.publish!(ref)
    oracle = OracleEstimator(oracle_kind; seed=seed, scale=noise_scale)
    asmc_o, mpc_o, pid_o = build_controller(ctrl, kw)

    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=oracle, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, tr.run_mode
end

# -----------------------------------------------------------------------------
# Tracking + effort metrics from the per-tick probe.
# -----------------------------------------------------------------------------
function controller_metrics(probe, ref, mode::Symbol)
    isempty(probe) && return (tracking=Inf, ce=Inf, chatter=Inf, ok=false, abs=NamedTuple())
    ce = mean(sum(abs.(Vector(p.v_cmd))) for p in probe)
    # Chatter = mean per-tick total variation of the wheel-voltage command (sum
    # over 4 wheels of |Δv_cmd|). Targets high-frequency switching activity, the
    # sliding-mode chatter that a low K_max ceiling suppresses — distinct from ce
    # (magnitude). Proxy at the probe/log rate.
    vc = [Vector(p.v_cmd) for p in probe]
    chatter = length(vc) < 2 ? 0.0 : mean(sum(abs.(vc[k] .- vc[k-1])) for k in 2:length(vc))

    if mode == :velocity
        ex = Float64[]; ey = Float64[]; ep = Float64[]
        for p in probe
            t = p.t
            push!(ex, p.u[1] - ref.Vx(t))
            push!(ey, p.u[2] - ref.Vy(t))
            push!(ep, p.u[3] - ref.Wz(t))
        end
        rms_vx = sqrt(mean(ex.^2)); rms_vy = sqrt(mean(ey.^2)); rms_w = sqrt(mean(ep.^2))
        max_vx = maximum(abs, ex);  max_vy = maximum(abs, ey);  max_w = maximum(abs, ep)
        # tolerance-normalised: 1.0 == exactly at target
        tracking = (rms_vx / TOL.vel_rms + rms_vy / TOL.vel_rms + rms_w / TOL.yawrate_rms) / 3
        absm = (rms_vx=rms_vx, rms_vy=rms_vy, rms_w=rms_w,
                max_vx=max_vx, max_vy=max_vy, max_w=max_w)
    else  # :pose
        poserr = Float64[]; headerr = Float64[]
        for p in probe
            t = p.t
            pe = sqrt((p.u[17] - ref.xo(t))^2 + (p.u[18] - ref.yo(t))^2)
            he = abs(_wrapang(p.u[4] - ref.psi(t)))
            push!(poserr, pe); push!(headerr, he)
        end
        final_pos = poserr[end]; max_pos = maximum(poserr)
        final_head = headerr[end]; max_head = maximum(headerr)
        tracking = (final_pos / TOL.pos_final + max_pos / TOL.pos_max +
                    final_head / TOL.head_final + max_head / TOL.head_max) / 4
        absm = (final_pos=final_pos, max_pos=max_pos,
                final_head=final_head, max_head=max_head)
    end

    ok = isfinite(tracking) && isfinite(ce)
    return (tracking=tracking, ce=ce, chatter=chatter, ok=ok, abs=absm)
end

"Format the absolute-error sub-metrics in physical units for reporting."
function fmt_abs(mode::Symbol, a)
    isempty(a) && return "(no data)"
    if mode == :velocity
        return "vel_err rms=[$(round(a.rms_vx*1e3,digits=2)),$(round(a.rms_vy*1e3,digits=2))] mm/s " *
               "max=[$(round(a.max_vx*1e3,digits=1)),$(round(a.max_vy*1e3,digits=1))] mm/s | " *
               "ψ̇_err rms=$(round(a.rms_w*1e3,digits=2)) mrad/s max=$(round(a.max_w*1e3,digits=1)) mrad/s"
    else
        return "pos_err final=$(round(a.final_pos*100,digits=2)) cm max=$(round(a.max_pos*100,digits=1)) cm | " *
               "head_err final=$(round(a.final_head,digits=4)) rad max=$(round(a.max_head,digits=3)) rad"
    end
end

# -----------------------------------------------------------------------------
# Objective over the trajectory subset.
# -----------------------------------------------------------------------------
function make_objective(ctrl::Symbol, space, trajs, oracle_kind::Symbol;
                        seed::Int=42, lambda_chatter::Float64=0.0, lambda_kmax::Float64=0.0,
                        lambda_gamma::Float64=0.0)
    return function (theta)
        kw = decode(theta, space)
        track_sum = 0.0; ce_sum = 0.0; chat_sum = 0.0
        for tr in trajs
            m = try
                probe, ref, mode = run_controller(ctrl, kw, oracle_kind, tr; seed=seed)
                controller_metrics(probe, ref, mode)
            catch e
                @warn "run failed for $(tr.name)" exception=e
                (tracking=Inf, ce=Inf, chatter=Inf, ok=false)
            end
            (!m.ok || !isfinite(m.tracking)) && return (score=1e6, tracking=Inf, ce=Inf, chatter=Inf, kmax_pen=0.0, gamma_pen=0.0)
            track_sum += m.tracking; ce_sum += m.ce; chat_sum += m.chatter
        end
        n = length(trajs)
        tracking = track_sum / n
        ce = ce_sum / n
        chatter = chat_sum / n
        # K_max HEADROOM reward (ASMC only): penalize the SHORTFALL below the
        # switching-gain ceiling → rewards HIGH K_max. Chatter/tracking are
        # invariant to K_max here (the ceiling is non-binding — K rarely reaches
        # it — so it costs nothing nominally), but a high ceiling is robustness
        # RESERVE: it lets the adaptive gain climb to meet OOD disturbances a low
        # ceiling would saturate against. Also breaks the sloppy-manifold
        # degeneracy → gains converge to the high-K_max corner. 150 = K_max bound.
        kmax_pen = (ctrl == :asmc && lambda_kmax > 0) ?
            lambda_kmax * (3 * 150.0 - (kw.K_max_x + kw.K_max_y + kw.K_max_psi)) / (3 * 150.0) : 0.0
        # gamma regularization (ASMC only): LOWER adaptation rate preferred —
        # smoother/parsimonious adaptation, and breaks the remaining degeneracy
        # (with K_max pinned) so gamma converges to the min-sufficient value.
        gamma_pen = (ctrl == :asmc && lambda_gamma > 0) ?
            lambda_gamma * (kw.gamma_x + kw.gamma_y + kw.gamma_psi) / (3 * 100.0) : 0.0
        score = tracking + LAMBDA_CE * (ce / V_MAX) +
                lambda_chatter * (chatter / V_MAX) + kmax_pen + gamma_pen
        return (score=score, tracking=tracking, ce=ce, chatter=chatter,
                kmax_pen=kmax_pen, gamma_pen=gamma_pen)
    end
end

# -----------------------------------------------------------------------------
# Simple random-coarse + local-refine optimizer (bootstrap-grade).
# -----------------------------------------------------------------------------
function optimize(objfn, lo, hi; n_coarse::Int, n_local::Int)
    d = length(lo)
    best = Float64[]; best_score = Inf
    trials = NamedTuple[]
    for i in 1:n_coarse
        θ = [lo[k] + rand() * (hi[k] - lo[k]) for k in 1:d]
        r = objfn(θ)
        push!(trials, merge((iter=i, phase=:coarse, theta=copy(θ)), r))
        if r.score < best_score; best_score = r.score; best = copy(θ); end
        println("coarse $i/$n_coarse  score=$(round(r.score,digits=4))  " *
                "track=$(round(r.tracking,digits=4))  ce=$(round(r.ce,digits=2))  " *
                "best=$(round(best_score,digits=4))")
    end
    step = 0.4
    for i in 1:n_local
        θ = [clamp(best[k] + step * (hi[k] - lo[k]) * randn(), lo[k], hi[k]) for k in 1:d]
        r = objfn(θ)
        push!(trials, merge((iter=n_coarse+i, phase=:local, theta=copy(θ)), r))
        if r.score < best_score; best_score = r.score; best = copy(θ); end
        i % d == 0 && (step *= 0.7)
        println("local  $i/$n_local  score=$(round(r.score,digits=4))  best=$(round(best_score,digits=4))")
    end
    return best, best_score, trials
end

# BlackBoxOptim backend (dxNES default; adaptive DE alternative).  Covariance-
# adapting global search — handles the coupled, ill-conditioned gain landscape
# that the random+coordinate search above can't.
function optimize_bbo(objfn, lo, hi; budget::Int, method::Symbol, seed::Int,
                      trace_path::AbstractString="")
    Random.seed!(seed)
    trials = NamedTuple[]
    n = 0
    best_so_far = Ref(Inf)
    # Live convergence trace: one CSV row per eval, FLUSHED every write so the
    # descend-vs-plateau curve is readable while the run is in flight (Julia
    # block-buffers stdout to a redirected file, so the println below may lag by
    # minutes — this explicit flush is the reliable live signal).
    trace_io = isempty(trace_path) ? nothing : open(trace_path, "w")
    trace_io !== nothing && (println(trace_io, "iter,score,best_so_far"); flush(trace_io))
    fitness = function (θ)
        n += 1
        r = objfn(Vector{Float64}(θ))
        push!(trials, merge((iter=n, phase=method, theta=Vector{Float64}(θ)), r))
        s = Float64(r.score)
        s = isfinite(s) ? min(s, 1e6) : 1e6
        best_so_far[] = min(best_so_far[], s)
        if trace_io !== nothing
            println(trace_io, "$n,$s,$(best_so_far[])"); flush(trace_io)
        end
        if n % 10 == 0
            println("  [$method] eval $n  score=$(round(s,digits=4))  " *
                    "best=$(round(best_so_far[],digits=4))")
            flush(stdout)
        end
        return s
    end
    # TraceMode=:compact makes BBO print per-generation best-fitness AND its
    # internal step-size to stderr (unbuffered → visible live); a collapsing
    # step-size is the definitive "converged" signal.
    res = BBO.bboptimize(fitness;
        SearchRange   = collect(zip(lo, hi)),
        NumDimensions = length(lo),
        Method        = method,
        MaxFuncEvals  = budget,
        TraceMode     = :compact,
        TraceInterval = 1.0)
    trace_io !== nothing && close(trace_io)
    return collect(BBO.best_candidate(res)), Float64(BBO.best_fitness(res)), trials
end

# -----------------------------------------------------------------------------
# Trajectory subset: one VelRef + one PosRef.
# -----------------------------------------------------------------------------
function default_trajs(run_dir::String)
    # combo_idx pins FEASIBLE trajectories (within the velocity envelope):
    #   octagon combo 2 = vcru 0.6 m/s (peak |Vy|=0.6, feasible; combo 3-4 exceed it)
    #   ellipse combo 1 = a=0.8, peak speed ~0.31 m/s (feasible)
    return [
        (name="octagon", profile_toml="octagon_mu_0p5.toml", combo_idx=2,
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
        (name="ellipse", profile_toml="ellipse_mu_0p5.toml", combo_idx=1,
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose),
    ]
end

# Set 2 = set 1 + a FEASIBLE spin_creep (yaw) entry, so the ψ-axis gains are
# actually exercised.  spin_creep combo 7 = yaw_spin [2.0,-2.0] rad/s: within the
# ~2 rad/s spin feasibility ceiling (combo 1 @3.1 rad/s gross-slips even under
# tuned ASMC); VelRef → velocity mode (tracks ω, not ψ). Kept separate from
# default_trajs so set-1 results stay comparable.
function default_trajs_2(run_dir::String)
    return vcat(default_trajs(run_dir), [
        (name="spin_creep", profile_toml="spin_creep_mu_0p5.toml", combo_idx=7,
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
    ])
end

# Set 3 = the finalized WHITELISTED training set (6 trajectories, worst-feasible
# per profile; ellipse split tangent(55)/crab(83); straightline dropped — absent
# from the PINN dataset). All combos are keep*/keep_flagged in
# diagnostics_combined.csv. This is the set the publishing-ready ASMC/PID tuning
# optimizes on. Eval grid (easy anchors + multisine + long_circle) lives in the
# separate evaluation harness, NOT here.
function default_trajs_3(run_dir::String)
    v(name, toml, c) = (name=name, profile_toml=toml, combo_idx=c,
                        ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity)
    p(name, toml, c) = (name=name, profile_toml=toml, combo_idx=c,
                        ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose)
    return [
        v("octagon",        "octagon_mu_0p5.toml",        206),
        v("spin_creep",     "spin_creep_mu_0p5.toml",     178),
        v("coupled_vomega", "coupled_vomega_mu_0p5.toml", 12),
        v("spiral_orbit",   "spiral_orbit_mu_0p5.toml",   37),
        p("ellipse_tangent","ellipse_mu_0p5.toml",        55),
        p("ellipse_crab",   "ellipse_mu_0p5.toml",        83),
    ]
end

# Set "pose" (controller-posref-retune-reeval.md): same 6 combos as set 3, but
# EVERY entry run_mode=:pose. The 4 former-velref trajectories (octagon,
# spin_creep, coupled_vomega, spiral_orbit) carry adapt=true, routing them
# through velref_to_posref in run_controller; the 2 ellipse entries are
# already native PosRef and pass through unchanged (adapt=false/absent).
# ADDITIVE ONLY: default_trajs_3/--trajset "3" above are untouched, so the
# v1 velocity-mode results stay byte-for-byte reproducible.
function default_trajs_pose(run_dir::String)
    v(name, toml, c) = (name=name, profile_toml=toml, combo_idx=c,
                        ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:pose, adapt=true)
    p(name, toml, c) = (name=name, profile_toml=toml, combo_idx=c,
                        ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose, adapt=false)
    return [
        v("octagon",        "octagon_mu_0p5.toml",        206),
        v("spin_creep",     "spin_creep_mu_0p5.toml",     178),
        v("coupled_vomega", "coupled_vomega_mu_0p5.toml", 12),
        v("spiral_orbit",   "spiral_orbit_mu_0p5.toml",   37),
        p("ellipse_tangent","ellipse_mu_0p5.toml",        55),
        p("ellipse_crab",   "ellipse_mu_0p5.toml",        83),
    ]
end

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
function parse_args(argv)
    a = Dict{String,Any}(
        "controller" => "asmc", "noise" => "noisy", "budget" => 60,
        "optimizer" => "dxnes", "seed" => 42, "run-dir" => "trajectory_files_run_0p5_main",
        "out" => "runs_controller", "smoke" => false, "trajset" => "1",
        "lambda-chatter" => 0.0, "lambda-kmax" => 0.0, "lambda-gamma" => 0.0,
        "pin-kmax" => false,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--controller"; a["controller"] = argv[i+1]; i += 2
        elseif arg == "--noise";  a["noise"] = argv[i+1]; i += 2
        elseif arg == "--budget"; a["budget"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--optimizer"; a["optimizer"] = argv[i+1]; i += 2
        elseif arg == "--seed";   a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--run-dir"; a["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--out";    a["out"] = argv[i+1]; i += 2
        elseif arg == "--trajset"; a["trajset"] = argv[i+1]; i += 2
        elseif arg == "--lambda-chatter"; a["lambda-chatter"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--lambda-kmax"; a["lambda-kmax"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--lambda-gamma"; a["lambda-gamma"] = parse(Float64, argv[i+1]); i += 2
        elseif arg == "--pin-kmax"; a["pin-kmax"] = true; i += 1
        elseif arg == "--smoke";  a["smoke"] = true; i += 1
        else; error("unknown arg $arg"); end
    end
    return a
end

function tune_one(ctrl::Symbol, oracle_kind::Symbol, trajs, budget::Int, seed::Int,
                  outroot::String, optimizer::String;
                  lambda_chatter::Float64=0.0, lambda_kmax::Float64=0.0,
                  lambda_gamma::Float64=0.0, pin_kmax::Bool=false)
    space = space_for(ctrl; pin_kmax=pin_kmax)
    lo, hi = flat_bounds(space)
    objfn = make_objective(ctrl, space, trajs, oracle_kind;
                           seed=seed, lambda_chatter=lambda_chatter, lambda_kmax=lambda_kmax,
                           lambda_gamma=lambda_gamma)
    outdir = joinpath(outroot, "$(ctrl)_$(oracle_kind)")
    mkpath(outdir)
    println("\n===== Tuning $ctrl on oracle=$oracle_kind, optimizer=$optimizer, budget=$budget " *
            "($(length(trajs)) trajs/eval, $(length(lo)) params) =====")

    if optimizer == "coarse"
        n_coarse = max(1, div(budget, 2)); n_local = budget - n_coarse
        best, best_score, trials = optimize(objfn, lo, hi; n_coarse=n_coarse, n_local=n_local)
    else
        method = optimizer == "de" ? :adaptive_de_rand_1_bin_radiuslimited : :dxnes
        best, best_score, trials = optimize_bbo(objfn, lo, hi; budget=budget, method=method,
                                                seed=seed, trace_path=joinpath(outdir, "trace.csv"))
    end
    best_kw = decode(best, space)

    # Final eval of the best config: report absolute errors per trajectory.
    println("[$ctrl] best absolute errors (best gains, oracle=$oracle_kind):")
    abs_report = Dict{String,Any}()
    for tr in trajs
        probe, ref, mode = run_controller(ctrl, best_kw, oracle_kind, tr; seed=seed)
        m = controller_metrics(probe, ref, mode)
        println("   $(tr.name) [$(mode)] ce=$(round(m.ce,digits=2)): $(fmt_abs(mode, m.abs))")
        abs_report[tr.name] = Dict(string(k) => v for (k, v) in pairs(m.abs))
    end

    gains_out = Dict(string(k) => (v isa SVector ? collect(v) : v) for (k, v) in pairs(best_kw))
    # Pinned-K_max run: fold the fixed ceiling into the saved gains so best_gains
    # is a COMPLETE controller spec (5 tuned + 3 pinned = the full ASMCController).
    if pin_kmax && ctrl == :asmc
        gains_out["K_max_x"]   = K_MAX_PIN.K_max_x
        gains_out["K_max_y"]   = K_MAX_PIN.K_max_y
        gains_out["K_max_psi"] = K_MAX_PIN.K_max_psi
    end
    open(joinpath(outdir, "best_config.json"), "w") do io
        JSON.print(io, Dict(
            "controller" => string(ctrl), "noise" => string(oracle_kind),
            "best_score" => best_score,
            "best_gains" => gains_out,
            "pin_kmax" => pin_kmax,
            "abs_errors" => abs_report,
            "theta" => best,
        ), 2)
    end
    open(joinpath(outdir, "trials.json"), "w") do io
        JSON.print(io, [Dict("iter"=>t.iter, "phase"=>string(t.phase),
                             "score"=>t.score, "tracking"=>t.tracking, "ce"=>t.ce)
                        for t in trials], 2)
    end
    println("[$ctrl] best score=$(round(best_score,digits=4))  gains=$best_kw  → $outdir")
    return best_score
end

function main()
    a = parse_args(ARGS)
    Random.seed!(a["seed"])
    oracle_kind = Symbol(a["noise"])
    trajs = get(a, "trajset", "1") == "pose" ? default_trajs_pose(a["run-dir"]) :
            get(a, "trajset", "1") == "3" ? default_trajs_3(a["run-dir"]) :
            get(a, "trajset", "1") == "2" ? default_trajs_2(a["run-dir"]) :
            default_trajs(a["run-dir"])
    println("Trajectory set: ", get(a, "trajset", "1"), "  (", join([t.name for t in trajs], ", "), ")")
    controllers = a["controller"] == "all" ? [:asmc, :pid, :mpc] : [Symbol(a["controller"])]

    if a["controller"] == "det"
        # Determinism check: run the same config repeatedly in ONE process.
        for tr in trajs
            println("=== determinism check: $(tr.name) [$(tr.run_mode)] (asmc default, clean) ===")
            for rep in 1:3
                probe, ref, mode = run_controller(:asmc, (;), :clean, tr; seed=a["seed"])
                m = controller_metrics(probe, ref, mode)
                key = mode == :pose ? "final_pos=$(round(m.abs.final_pos*100,digits=3))cm max=$(round(m.abs.max_pos*100,digits=2))cm" :
                                      "vel_rms=$(round(m.abs.rms_vx*1e3,digits=3))mm/s"
                println("  rep $rep: ticks=$(length(probe)) $key")
            end
        end
        return
    end

    if a["smoke"]
        for ctrl in controllers
            println("=== SMOKE: default $ctrl gains, oracle=$oracle_kind ===")
            for tr in trajs
                probe, ref, mode = run_controller(ctrl, (;), oracle_kind, tr; seed=a["seed"])
                m = controller_metrics(probe, ref, mode)
                println("  $(tr.name) [$(mode)]: score=$(round(m.tracking,digits=2)) ce=$(round(m.ce,digits=2))")
                println("      $(fmt_abs(mode, m.abs))")
            end
        end
        println("=== SMOKE OK ===")
        return
    end

    for ctrl in controllers
        tune_one(ctrl, oracle_kind, trajs, a["budget"], a["seed"], a["out"], a["optimizer"];
                 lambda_chatter=a["lambda-chatter"], lambda_kmax=a["lambda-kmax"],
                 lambda_gamma=a["lambda-gamma"], pin_kmax=a["pin-kmax"])
    end
    println("\nAll tuning complete. Outputs in $(a["out"]).")
end

# Only auto-run when executed as a script; `include(...)` from another script
# (e.g. an eval harness) reuses run_controller/controller_metrics without tuning.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
