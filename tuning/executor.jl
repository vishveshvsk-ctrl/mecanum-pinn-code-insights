# =============================================================================
# tuning/executor.jl — parallel evaluation of proposed parameter vectors
# =============================================================================
module TuningExecutorMod

using Base.Threads

using Main.TuningParamSpaceMod: apply_params!
using Main.TuningSubsetMod: TuningSubset

export parallel_evaluate

const SENTINEL = (
    vel_rmse        = Inf,
    inslip_vel_rmse = Inf,
    rate_rmse       = Inf,
    pos_rmse        = Inf,
    heading_rmse    = Inf,
    smoothness      = Inf,
    score           = Inf,
)

"""
    parallel_evaluate(thetas, subset, space, objective; max_parallel=1) -> Vector

Evaluate each proposed `theta` by decoding it through `apply_params!` and calling
`objective(est_cfg, subset)`.  Each theta runs the whole subset.  When
`max_parallel > 1` and Julia was started with enough threads, the evaluations are
run with `Threads.@threads`.  Failures return a sentinel `(score=Inf, ...)` and
are logged to stderr without aborting the batch.
"""
function parallel_evaluate(thetas::Vector{Vector{Float64}},
                           subset::TuningSubset,
                           space,
                           objective;
                           max_parallel::Int=1)
    n = length(thetas)
    results = Vector{NamedTuple}(undef, n)

    if max_parallel > 1 && nthreads() >= max_parallel
        # Evaluate in batches of size max_parallel to respect the cap.
        for batch_start in 1:max_parallel:n
            batch_end = min(batch_start + max_parallel - 1, n)
            Threads.@threads for i in batch_start:batch_end
                results[i] = _eval_one(thetas[i], subset, space, objective)
            end
        end
    else
        if max_parallel > 1
            @warn "max_parallel=$max_parallel requires at least $max_parallel threads; " *
                  "only $(nthreads()) available. Running sequentially."
        end
        for i in eachindex(thetas)
            results[i] = _eval_one(thetas[i], subset, space, objective)
        end
    end

    return results
end

function _eval_one(theta::Vector{Float64},
                   subset::TuningSubset,
                   space,
                   objective)
    try
        est_cfg = apply_params!(theta, space)
        return objective(est_cfg, subset)
    catch e
        @error "parallel_evaluate: evaluation failed for theta $theta" exception=(e, catch_backtrace())
        return SENTINEL
    end
end

end # module
