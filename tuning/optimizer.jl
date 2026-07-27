# =============================================================================
# tuning/optimizer.jl — pluggable derivative-free optimizer interface
# =============================================================================
module TuningOptimizerMod

using Main.TuningParamSpaceMod: ParamSpace, n_params

export AbstractOptimizer, CoarseThenLocal, ask, tell!, best_theta, best_score, history

"""
    AbstractOptimizer

Base type for optimizers that propose parameter vectors and consume scores
under a fixed evaluation budget.  Implement `ask` and `tell!`.
"""
abstract type AbstractOptimizer end

"""
    CoarseThenLocal(space, n_coarse, n_local)

Simple two-phase optimizer:

1. Coarse: `n_coarse` independent uniform random samples in the (possibly
   log-transformed) theta box.
2. Local:  coordinate-descent refinement around the best coarse point for
   `n_local` iterations, shrinking the step every full coordinate cycle.

Total budget = `n_coarse + n_local`.
"""
mutable struct CoarseThenLocal <: AbstractOptimizer
    space::ParamSpace
    n_coarse::Int
    n_local::Int
    lower::Vector{Float64}
    upper::Vector{Float64}
    best_theta::Vector{Float64}
    best_score::Float64
    history::Vector{NamedTuple}
    phase::Symbol          # :coarse or :local
    idx::Int               # iteration counter within current phase
    local_center::Vector{Float64}
    local_step::Vector{Float64}
end

function CoarseThenLocal(space::ParamSpace, n_coarse::Int, n_local::Int)
    n_coarse >= 0 && n_local >= 0 || error("n_coarse and n_local must be non-negative")
    d = n_params(space)
    lower = copy(space.flat_lower)
    upper = copy(space.flat_upper)
    return CoarseThenLocal(space, n_coarse, n_local, lower, upper,
                           zeros(d), Inf, NamedTuple[], :coarse, 0,
                           zeros(d), ones(d))
end

"""
    ask(opt) -> Vector{Float64}

Return the next parameter vector to evaluate.  For `CoarseThenLocal` this
proposes one theta at a time; the caller iterates until the budget is exhausted.
"""
function ask(opt::CoarseThenLocal)
    d = length(opt.lower)

    # Transition from coarse to local phase.
    if opt.phase == :coarse && opt.idx >= opt.n_coarse && opt.n_local > 0
        opt.phase = :local
        opt.local_center = copy(opt.best_theta)
        opt.local_step   = 0.1 .* (opt.upper .- opt.lower)
        opt.idx = 0
    end

    if opt.phase == :coarse
        opt.idx += 1
        return [opt.lower[i] + rand() * (opt.upper[i] - opt.lower[i]) for i in 1:d]
    else
        # Local coordinate descent: cycle coordinates, alternate +/- step.
        opt.idx += 1
        coord = mod1(opt.idx, d)
        theta = copy(opt.local_center)
        sign = (div(opt.idx - 1, d) % 2 == 0) ? 1.0 : -1.0
        theta[coord] += sign * opt.local_step[coord]
        theta[coord] = clamp(theta[coord], opt.lower[coord], opt.upper[coord])
        # Shrink step every full sweep (both signs for every coordinate).
        if mod(opt.idx, 2 * d) == 0
            opt.local_step .*= 0.5
        end
        return theta
    end
end

"""
    tell!(opt, theta, score)

Record the score for a proposed theta and update the best point seen so far.
"""
function tell!(opt::CoarseThenLocal, theta::AbstractVector{<:Real}, score::Real)
    push!(opt.history, (theta=Vector{Float64}(theta), score=Float64(score)))
    if score < opt.best_score
        opt.best_score = Float64(score)
        opt.best_theta = Vector{Float64}(theta)
        if opt.phase == :local
            opt.local_center = Vector{Float64}(theta)
        end
    end
    return opt
end

best_theta(opt::CoarseThenLocal) = opt.best_theta
best_score(opt::CoarseThenLocal) = opt.best_score
history(opt::CoarseThenLocal)    = opt.history

end # module
