# =============================================================================
# hybrid_ctrl_v2/controller_tuning/optimizer_stage.jl — StageOptimizerMod
# =============================================================================
# Two-phase driver: global search (dxNES/DE via BlackBoxOptim) on the SCREEN
# objective, plateau-gated + hard-capped, then local refinement (BOBYQA via
# NLopt) on the FULL objective inside a box centred on the phase-1 optimum.
# `Sobol`/`NLopt`/`MatrixEquations` are now direct Project.toml dependencies
# (added via `Pkg.add`, per-user direction -- tune_controller.jl/controllers.jl
# themselves are still never edited). A pattern-search `:local` refiner is
# kept as an explicit-opt-in alternative (`--refiner local`), matching the
# brief's "must degrade gracefully" contract for environments without NLopt.
# =============================================================================
module StageOptimizerMod

using Random
using Sobol
using NLopt

export optimize_staged, plateau_reached, sobol_start

const BBO = Base.require(Main, :BlackBoxOptim)

"""
    sobol_start(lo, hi, index) -> Vector{Float64}

Deterministic point `index` (>= 1) of the `d`-dimensional Sobol low-
discrepancy sequence, scaled into the box `[lo, hi]`. A fresh `SobolSeq(d)` is
constructed and skipped to position `index-1` each call, so `sobol_start` is a
pure function of `(lo, hi, index)` -- exactly the "seed N's starting mean"
contract `optimize_staged` needs for its 5 independent multi-start seeds.
"""
function sobol_start(lo::Vector{Float64}, hi::Vector{Float64}, index::Int)
    d = length(lo)
    s = Sobol.SobolSeq(d)
    buf = Vector{Float64}(undef, d)
    index > 1 && Sobol.skip!(s, index - 1, buf; exact=true)
    Sobol.next!(s, buf)
    return [lo[k] + (hi[k] - lo[k]) * buf[k] for k in 1:d]
end

"""
    plateau_reached(best_curve, rel_tol, window) -> Bool

True when the best-so-far curve (non-increasing, one entry per eval) has
improved by LESS than `rel_tol` (relative to the current best magnitude) over
the trailing `window` evaluations. Gates on improvement MAGNITUDE, not just
numerical position, guarding against the known false-positive noted in the
brief (a curve that is still creeping down at generation boundaries reads as
"flat" under a naive positional check).
"""
function plateau_reached(best_curve::Vector{Float64}, rel_tol::Float64, window::Int)
    length(best_curve) <= window && return false
    ref = best_curve[end-window]
    cur = best_curve[end]
    denom = max(abs(cur), 1e-12)
    improvement = (ref - cur) / denom
    return improvement < rel_tol
end

# Chunk size for the incremental BBO run!/update_parameters! loop: small
# enough that plateau checks are responsive, large enough to amortize BBO's
# per-run! setup cost. `OptController.max_fevals` is checked as a CUMULATIVE
# total against `num_func_evals` (BlackBoxOptim/src/opt_controller.jl), so
# bumping it via `update_parameters!` and calling `run!`/`bboptimize` again
# correctly continues the same population/step-size state rather than
# restarting.
_chunk_size(d::Int) = max(10, 2 * d)

function _run_phase(objfn::Function, lo::Vector{Float64}, hi::Vector{Float64}, x0::Vector{Float64};
                    method::Symbol, cap::Int, rel_tol::Float64, window::Int, seed::Int,
                    tier_label::Symbol, trace_io=nothing,
                    checkpoint_every::Int=0, checkpoint_cb=nothing)
    d = length(lo)
    trials = NamedTuple[]
    n = Ref(0)
    best_so_far = Ref(Inf)
    best_theta = Ref(Float64[])
    best_curve = Float64[]
    last_improve = Ref(0)

    fitness = function (theta)
        n[] += 1
        theta_v = Vector{Float64}(theta)
        r = objfn(theta_v)
        push!(trials, merge((iter=n[], tier=tier_label, theta=theta_v), r))
        s = isfinite(r.score) ? min(Float64(r.score), 1e6) : 1e6
        if s < best_so_far[] - 1e-15
            best_so_far[] = s
            best_theta[] = theta_v
            last_improve[] = n[]
        end
        push!(best_curve, best_so_far[])
        if trace_io !== nothing
            println(trace_io, "$(n[]),$(tier_label),$s,$(best_so_far[])"); flush(trace_io)
        end
        if checkpoint_cb !== nothing && checkpoint_every > 0 && n[] % checkpoint_every == 0
            checkpoint_cb(n[], tier_label, theta_v, s, best_theta[], best_so_far[])
        end
        return s
    end

    Random.seed!(seed)
    chunk = _chunk_size(d)
    optctrl = BBO.bbsetup(fitness; SearchRange=collect(zip(lo, hi)), NumDimensions=d,
                          Method=method, MaxFuncEvals=chunk, TraceMode=:silent)
    res = BBO.bboptimize(optctrl, x0)   # singular x0 -> set_candidate! (defined for
                                       # DXNESOpt/PopulationOptimizer): seeds the
                                       # initial mean/candidate with the Sobol start.

    stop_reason = :cap
    while n[] < cap
        BBO.update_parameters!(optctrl, Dict{Symbol,Any}(:MaxFuncEvals => n[] + chunk))
        res = BBO.bboptimize(optctrl)
        if plateau_reached(best_curve, rel_tol, window)
            stop_reason = :plateau
            break
        end
    end

    best_vec = Vector{Float64}(collect(BBO.best_candidate(res)))
    best_score = Float64(BBO.best_fitness(res))
    return (best=best_vec, best_score=best_score, trials=trials, evals=n[],
            last_improvement_eval=last_improve[], stop_reason=stop_reason)
end

# ---- Local (NLopt-absent) fallback refiner: shrinking compass search -------
function _local_refine(objfn::Function, lo::Vector{Float64}, hi::Vector{Float64}, x0::Vector{Float64};
                       cap::Int, rel_tol::Float64, window::Int, tier_label::Symbol, trace_io=nothing,
                       checkpoint_every::Int=0, checkpoint_cb=nothing)
    d = length(lo)
    trials = NamedTuple[]
    n = Ref(0)
    best = copy(x0)
    r0 = objfn(best)
    push!(trials, merge((iter=1, tier=tier_label, theta=copy(best)), r0))
    n[] = 1
    best_score = isfinite(r0.score) ? min(Float64(r0.score), 1e6) : 1e6
    best_curve = Float64[best_score]
    last_improve = Ref(1)
    step = 0.25 .* (hi .- lo)
    stop_reason = :cap
    while n[] < cap
        improved = false
        for k in 1:d, sgn in (1.0, -1.0)
            n[] >= cap && break
            cand = copy(best)
            cand[k] = clamp(cand[k] + sgn * step[k], lo[k], hi[k])
            r = objfn(cand)
            n[] += 1
            s = isfinite(r.score) ? min(Float64(r.score), 1e6) : 1e6
            push!(trials, merge((iter=n[], tier=tier_label, theta=copy(cand)), r))
            if s < best_score - 1e-15
                best_score = s; best = cand; improved = true
                last_improve[] = n[]
            end
            push!(best_curve, best_score)
            if trace_io !== nothing
                println(trace_io, "$(n[]),$(tier_label),$s,$best_score"); flush(trace_io)
            end
            if checkpoint_cb !== nothing && checkpoint_every > 0 && n[] % checkpoint_every == 0
                checkpoint_cb(n[], tier_label, cand, s, best, best_score)
            end
        end
        improved || (step .*= 0.6)
        if plateau_reached(best_curve, rel_tol, window)
            stop_reason = :plateau
            break
        end
    end
    return (best=best, best_score=best_score, trials=trials, evals=n[],
            last_improvement_eval=last_improve[], stop_reason=stop_reason)
end

# ---- BOBYQA refiner (NLopt) -------------------------------------------------
function _bobyqa_refine(objfn::Function, lo::Vector{Float64}, hi::Vector{Float64}, x0::Vector{Float64};
                        cap::Int, rel_tol::Float64, window::Int, tier_label::Symbol, trace_io=nothing,
                        checkpoint_every::Int=0, checkpoint_cb=nothing)
    d = length(lo)
    opt = NLopt.Opt(:LN_BOBYQA, d)
    opt.lower_bounds = lo; opt.upper_bounds = hi
    opt.maxeval = cap
    n = Ref(0)
    trials = NamedTuple[]
    best = Ref(copy(x0)); best_score = Ref(Inf)
    last_improve = Ref(0)
    best_curve = Float64[]
    plateaued = Ref(false)
    opt.min_objective = (x, grad) -> begin
        n[] += 1
        x_v = Vector{Float64}(x)
        r = objfn(x_v)
        push!(trials, merge((iter=n[], tier=tier_label, theta=x_v), r))
        s = isfinite(r.score) ? min(Float64(r.score), 1e6) : 1e6
        if s < best_score[] - 1e-15
            best_score[] = s; best[] = x_v; last_improve[] = n[]
        end
        push!(best_curve, best_score[])
        if trace_io !== nothing
            println(trace_io, "$(n[]),$(tier_label),$s,$(best_score[])"); flush(trace_io)
        end
        if checkpoint_cb !== nothing && checkpoint_every > 0 && n[] % checkpoint_every == 0
            checkpoint_cb(n[], tier_label, x_v, s, best[], best_score[])
        end
        # EXPLICIT score-plateau check (per-user direction, found via the PID v2
        # FB seed 2 trace: BOBYQA's own xtol/ftol criterion did NOT fire even
        # though the SCORE had plateaued by eval 30/60 -- it kept taking tiny
        # trust-region steps that moved x without moving the score, burning
        # ~half the phase-2 budget after convergence). Throwing NLopt.ForcedStop
        # from inside the objective is NLopt's documented mechanism for halting
        # from the callback; NLopt.optimize catches it and returns normally
        # rather than propagating it.
        if plateau_reached(best_curve, rel_tol, window)
            plateaued[] = true
            throw(NLopt.ForcedStop())
        end
        return s
    end
    NLopt.optimize(opt, copy(x0))
    stop_reason = plateaued[] ? :plateau : (n[] < cap ? :plateau : :cap)
    return (best=best[], best_score=best_score[], trials=trials, evals=n[],
            last_improvement_eval=last_improve[], stop_reason=stop_reason)
end

"""
    optimize_staged(obj_screen, obj_full, lo, hi; kwargs...)
        -> (best::Vector{Float64}, best_score::Float64,
            trials::Vector{NamedTuple}, diag::NamedTuple)

Two-phase driver (brief §7.3). Phase 1: global search (`method` = :dxnes or
BlackBoxOptim's adaptive-DE) against `obj_screen`, Sobol-offset start, until
`plateau_reached` or `p1_cap`. Phase 2: local refine (`refiner` = :bobyqa via
NLopt, or :local -- the compass-search fallback) against `obj_full` inside a
box of half-width `refine_halfwidth*(hi-lo)` centred on the phase-1 optimum
(clipped to [lo,hi]), until `plateau_reached` fires (BOTH refiners now check
the score curve directly -- `:bobyqa` used to rely solely on NLopt's own
xtol/ftol criterion, which a real run showed does NOT reliably fire even
after the score has clearly flatlined; fixed per-user direction by throwing
`NLopt.ForcedStop` from the objective once `plateau_reached` is true) or
`p2_cap`.

`best_score` is ALWAYS a FULL-tier score: if phase 2 never improves on the
phase-1 point re-evaluated at full tier, that re-evaluated point is returned
and `diag.stop_reason = :no_refine_gain`.

`checkpoint_every`/`checkpoint_cb` (per-user direction, added so a crash mid-
run doesn't lose the compute already spent): when `checkpoint_every > 0` and
`checkpoint_cb !== nothing`, every `checkpoint_every` evals in EITHER phase
`checkpoint_cb(n, tier_label, current_theta, current_score, best_theta,
best_score)` is called with the RAW theta vectors -- this module knows
nothing about `decode`/`space`/gain field names, so the callback (built by
the caller, e.g. `run_stage.jl`'s `_make_checkpoint_cb`) is responsible for
decoding and persisting them (typically to a single rolling JSON file, not
one file per checkpoint). Disabled by default (`checkpoint_every=0`,
`checkpoint_cb=nothing`) -- existing callers (`run_asmc`/`run_mpc`/`run_pid`)
are unaffected unless they opt in.

`warm_start` (per-user direction, added for noisy-oracle FINE-TUNING off an
already-converged clean-oracle point): when given a decoded-space theta
vector, phase 1's global dxNES search is SKIPPED ENTIRELY (0 evals against
`obj_screen`) -- the supplied point stands in for `p1.best` (`phase1_evals=0` in the returned
`diag`) and everything downstream (phase-2 box construction, the full-tier
baseline re-eval, the refiner) proceeds exactly as it would after a phase 1
that happened to land there -- `diag.stop_reason` is `:no_refine_gain` if
phase 2 never beats the (clamped) warm point re-evaluated at full tier,
`:plateau`/`:cap` otherwise, same as the non-warm-started path. Default
`nothing` = existing behaviour, unchanged for every current caller.
"""
function optimize_staged(obj_screen::Function, obj_full::Function,
                         lo::Vector{Float64}, hi::Vector{Float64};
                         method::Symbol=:dxnes, refiner::Symbol=:bobyqa,
                         seed::Int=42, start_offset::Int=1,
                         p1_cap::Int=250, p2_cap::Int=60,
                         rel_tol::Float64=1e-3, window::Int=20,
                         refine_halfwidth::Float64=0.25,
                         trace_path::AbstractString="",
                         checkpoint_every::Int=0, checkpoint_cb=nothing,
                         warm_start::Union{Nothing,Vector{Float64}}=nothing)
    trace_io = isempty(trace_path) ? nothing : open(trace_path, "w")
    trace_io !== nothing && (println(trace_io, "iter,tier,score,best_so_far"); flush(trace_io))

    if warm_start === nothing
        x0 = sobol_start(lo, hi, start_offset)
        p1 = _run_phase(obj_screen, lo, hi, x0; method=method, cap=p1_cap, rel_tol=rel_tol,
                        window=window, seed=seed, tier_label=:screen, trace_io=trace_io,
                        checkpoint_every=checkpoint_every, checkpoint_cb=checkpoint_cb)
    else
        p1 = (best=clamp.(warm_start, lo, hi), best_score=NaN, trials=NamedTuple[], evals=0,
              last_improvement_eval=0, stop_reason=:warm_start)
    end

    # Phase-2 box: centred on the phase-1 optimum, clipped to [lo,hi].
    halfw = refine_halfwidth .* (hi .- lo)
    lo2 = clamp.(p1.best .- halfw, lo, hi)
    hi2 = clamp.(p1.best .+ halfw, lo, hi)

    use_bobyqa = refiner == :bobyqa

    # Re-evaluate the phase-1 optimum at FULL tier as the phase-2 baseline --
    # the returned best_score is always a full-tier score (brief §7.3).
    r_seed_full = obj_full(p1.best)
    seed_full_score = isfinite(r_seed_full.score) ? min(Float64(r_seed_full.score), 1e6) : 1e6
    n2_seed = 1
    trials2_seed = NamedTuple[merge((iter=1, tier=:full, theta=copy(p1.best)), r_seed_full)]
    if trace_io !== nothing
        println(trace_io, "1,full,$seed_full_score,$seed_full_score"); flush(trace_io)
    end

    p2 = use_bobyqa ?
        _bobyqa_refine(obj_full, lo2, hi2, p1.best; cap=p2_cap, rel_tol=rel_tol, window=window,
                       tier_label=:full, trace_io=trace_io,
                       checkpoint_every=checkpoint_every, checkpoint_cb=checkpoint_cb) :
        _local_refine(obj_full, lo2, hi2, p1.best; cap=p2_cap, rel_tol=rel_tol, window=window,
                      tier_label=:full, trace_io=trace_io,
                      checkpoint_every=checkpoint_every, checkpoint_cb=checkpoint_cb)

    trace_io !== nothing && close(trace_io)

    all_trials = vcat(p1.trials, trials2_seed, p2.trials)
    if p2.best_score < seed_full_score
        best, best_score, stop_reason = p2.best, p2.best_score, p2.stop_reason
    else
        best, best_score, stop_reason = p1.best, seed_full_score, :no_refine_gain
    end

    diag = (converged = stop_reason in (:plateau, :no_refine_gain),
            phase1_evals = p1.evals, phase2_evals = n2_seed + p2.evals,
            last_improvement_eval = p1.last_improvement_eval + n2_seed + p2.last_improvement_eval,
            plateau_window = window, stop_reason = stop_reason)
    return best, best_score, all_trials, diag
end

end # module
