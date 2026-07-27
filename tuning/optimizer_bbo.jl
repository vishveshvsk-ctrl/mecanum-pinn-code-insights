# =============================================================================
# tuning/optimizer_bbo.jl — driver-style global optimizer backends
#                            (BlackBoxOptim: dxNES / adaptive DE; Surrogates: BO)
# =============================================================================
module TuningBBOMod

using Random

using Main.TuningParamSpaceMod: ParamSpace, n_params

export run_backend

# Finite clamp for failed / non-finite scores: keeps the fitness GP- and
# population-safe (kills the Inf-poisoning seen with raw sentinel scores).
const PENALTY = 1e6

"""
    run_backend(kind, theta_objective, space; budget, nthreads=1, seed=42)
        -> (best_theta, best_score, history)

Driver-style optimizer backends that own the evaluation loop (unlike the
ask/tell `AbstractOptimizer` interface in `tuning/optimizer.jl`).

- `kind::Symbol` — one of `:dxnes` (BlackBoxOptim dxNES, primary),
  `:de` (BlackBoxOptim `adaptive_de_rand_1_bin_radiuslimited`),
  or `:bo` (Surrogates.jl Kriging + DYCORS, SRBF fallback).
- `theta_objective(theta::Vector{Float64}) -> NamedTuple` returns the full
  metrics NamedTuple (`score`, `overall_nrmse`, ...).  It is wrapped here into
  a scalar fitness: the `score` field, clamped to `PENALTY` when non-finite
  (sentinel / NaN / Inf) and always capped at `PENALTY`.
- `budget` — approximate total number of theta evaluations.
- `nthreads` — thread count forwarded to BlackBoxOptim (`NThreads`); ignored
  by `:bo` (serial).  Parallelism therefore lives inside the backend, so
  `theta_objective` itself is expected to evaluate serially.
- `history` — every evaluated `(theta, metrics)` pair as a NamedTuple shaped
  like the coarse-path trials (`iteration`, `theta`, plus the metrics fields),
  ready for `save_trials`.  Appends are lock-protected so the wrapper is safe
  under BlackBoxOptim's `NThreads > 1` multithreaded evaluator.

BlackBoxOptim.jl and Surrogates.jl are OPTIONAL heavy dependencies and are
loaded lazily (only inside the branch that needs them), so this file is
include-able with neither installed.
"""
function run_backend(kind::Symbol, theta_objective::Function, space;
                     budget::Int, nthreads::Int=1, seed::Int=42)
    kind in (:dxnes, :de, :bo) ||
        error("run_backend: unknown backend :$kind (expected :dxnes, :de or :bo)")
    budget >= 1 || error("run_backend: budget must be >= 1, got $budget")

    # Degenerate (fixed) dims — lower == upper, e.g. rate_hz pinned at 1000 —
    # break population optimizers: a zero-width SearchRange makes BBO emit NaN
    # candidates and poisons its internal NES covariance.  Search only over
    # genuinely-varying dims and expand candidates back to full length (fixed
    # dims filled with their lower bound) before evaluation.
    vary    = findall(i -> space.flat_upper[i] > space.flat_lower[i], 1:n_params(space))
    lower_v = space.flat_lower[vary]
    upper_v = space.flat_upper[vary]
    function expand(theta_sub)
        full = copy(space.flat_lower)
        full[vary] .= theta_sub
        return full
    end

    # History of every evaluated (theta, metrics) pair, shaped like the
    # ask/tell trials in tune_estimator.jl (iteration counter + theta column).
    history    = NamedTuple[]
    hist_lock  = ReentrantLock()
    eval_count = Ref(0)

    function fitness(theta)
        local metrics
        theta_sub = Vector{Float64}(theta)  # BBO gives Vector, Surrogates gives Tuple
        theta_vec = expand(theta_sub)
        try
            metrics = theta_objective(theta_vec)
        catch e
            @error "run_backend: objective threw for theta $theta_vec" exception=(e, catch_backtrace())
            return PENALTY
        end
        lock(hist_lock) do
            eval_count[] += 1
            push!(history, merge((iteration=eval_count[], theta=theta_vec), metrics))
        end
        s = Float64(metrics.score)
        return isfinite(s) ? min(s, PENALTY) : PENALTY
    end

    (best_sub, best_score) = if kind == :bo
        _run_surrogates(fitness, lower_v, upper_v, history; budget=budget, seed=seed)
    else
        _run_blackboxoptim(kind, fitness, lower_v, upper_v, history;
                           budget=budget, nthreads=nthreads, seed=seed)
    end
    return (expand(best_sub), best_score, history)
end

"""
    _run_blackboxoptim(kind, fitness, lower, upper, history; budget, nthreads, seed)

BlackBoxOptim driver (`bboptimize`) for `:dxnes` and `:de`, searching the
`lower`/`upper` box (varying dims only — see `run_backend`).  Returns
`(best_theta_sub, best_score)`.
"""
function _run_blackboxoptim(kind::Symbol, fitness::Function,
                            lower::Vector{Float64}, upper::Vector{Float64},
                            history;
                            budget::Int, nthreads::Int, seed::Int)
    # Optional heavy dependency: loaded on demand so this file stays
    # include-able without it.  Base.require returns the loaded module.
    local BBO
    try
        BBO = Base.require(Main, :BlackBoxOptim)
    catch
        error("Backend :$kind requires BlackBoxOptim.jl — install with: Pkg.add(\"BlackBoxOptim\")")
    end

    # NOTE (verified against BlackBoxOptim master): the :RngSeed and
    # :RandomizeRngSeed parameters are obsolete (init_rng! warns they have no
    # effect) and there is no `rng` kwarg — seeding must be done globally
    # before calling bboptimize.
    Random.seed!(seed)

    # Base.require above may have just loaded BlackBoxOptim, so its methods
    # live in a newer world age than this function.  Wrap the rest in a
    # closure run via invokelatest (which executes in the latest world).
    body = function ()
        method = kind == :dxnes ? :dxnes : :adaptive_de_rand_1_bin_radiuslimited
        params = Dict{Symbol,Any}(
            :SearchRange   => collect(zip(lower, upper)),
            :NumDimensions => length(lower),
            :Method        => method,
            # MaxFuncEvals caps total theta evaluations at `budget` — one optimizer
            # "step" evaluates a whole population, so MaxSteps alone would
            # overshoot the budget badly.  MaxSteps is kept as a secondary cap.
            :MaxFuncEvals  => budget,
            :MaxSteps      => budget,
            :TraceMode     => :compact,
        )
        # NThreads selects the MultithreadEvaluator (batch population/NES methods
        # such as :dxnes and :adaptive_de_rand_1_bin_radiuslimited support it).
        # BBO requires NThreads < Threads.nthreads() (one thread stays with the
        # driver), so clamp; skipped entirely at 1 → serial ProblemEvaluator.
        if nthreads > 1
            nworkers = min(nthreads, Base.Threads.nthreads() - 1)
            if nworkers >= 1
                nworkers < nthreads &&
                    @warn "run_backend: requested nthreads=$nthreads but Julia has " *
                          "$(Base.Threads.nthreads()) threads; using NThreads=$nworkers"
                params[:NThreads] = nworkers
            else
                @warn "run_backend: only $(Base.Threads.nthreads()) Julia thread; " *
                      "falling back to serial evaluation"
            end
        end

        res = BBO.bboptimize(fitness; params...)

        best_theta = collect(BBO.best_candidate(res))
        best_score = Float64(BBO.best_fitness(res))
        return (best_theta, best_score)
    end
    return Base.invokelatest(body)
end

"""
    _run_surrogates(fitness, lower, upper, history; budget, seed)

Surrogates.jl Bayesian optimization: Kriging surrogate + `surrogate_optimize!`
with the DYCORS strategy (SRBF fallback), initial Sobol DOE of
`min(20, div(budget, 3))` points, `maxiters` set so total evals ≈ `budget`.
Searches the `lower`/`upper` box (varying dims only — see `run_backend`).
Serial — no lock needed on the history wrapper here.  Returns
`(best_theta_sub, best_score)`.
"""
function _run_surrogates(fitness::Function,
                         lower::Vector{Float64}, upper::Vector{Float64},
                         history;
                         budget::Int, seed::Int)
    # Optional heavy dependency: loaded on demand (see _run_blackboxoptim).
    local S
    try
        S = Base.require(Main, :Surrogates)
    catch
        error("Backend :bo requires Surrogates.jl — install with: Pkg.add(\"Surrogates\")")
    end

    lb = copy(lower)
    ub = copy(upper)

    # Deterministic initial DOE: Sobol is quasi-random (no RNG state); the
    # global seed covers the stochastic candidate perturbations inside
    # DYCORS/SRBF.
    Random.seed!(seed)
    n_init   = max(2, min(20, div(budget, 3)))
    maxiters = max(1, budget - n_init)   # total evals ≈ n_init + maxiters ≈ budget

    # Base.require above may have just loaded Surrogates (newer world age than
    # this function) — run the Surrogates calls in a closure via invokelatest.
    body = function ()
        xys  = S.sample(n_init, lb, ub, S.SobolSample())
        zs   = [fitness(xy) for xy in xys]
        krig = S.Kriging(xys, zs, lb, ub)

        # surrogate_optimize! refits `krig` while appending new points to xys/zs
        # and returns (best_x, best_f).
        result = try
            S.surrogate_optimize!(fitness, S.DYCORS(), lb, ub, krig, S.SobolSample();
                                  maxiters=maxiters)
        catch e
            @warn "surrogate_optimize! with DYCORS failed; falling back to SRBF" exception=(e, catch_backtrace())
            S.surrogate_optimize!(fitness, S.SRBF(), lb, ub, krig, S.SobolSample();
                                  maxiters=maxiters)
        end

        best_theta = collect(result[1])
        best_score = Float64(result[2])
        return (best_theta, best_score)
    end
    return Base.invokelatest(body)
end

end # module
