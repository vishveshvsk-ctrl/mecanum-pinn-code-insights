# =============================================================================
# hybrid_ctrl/config.jl — HybridConfig: run-time selection + rates
# =============================================================================
module HybridConfigMod

export HybridConfig, enabled_controllers

"""
    HybridConfig

Single source of truth for which controller/estimator blocks are active and at
what rates.  All discrete callbacks are gated by the Boolean flags.

  tracking   :: :pose or :velocity
  estimator  :: :kalman, :kalman_imm, :eskf, :smo, or :none
  use_dhat   :: add SMO disturbance feedforward (requires estimator==:smo)
  use_asmc   :: enable adaptive sliding-mode controller
  use_mpc    :: enable QP-MPC controller
  use_pid    :: enable Kalman-PID controller
  fuzzy      :: enable fuzzy supervisor; if false, fixed_weights are used
  fixed_weights :: (w_ASMC, w_MPC, w_PID) used when fuzzy=false
  rates      :: f_est, f_mpc, f_pid, f_fuzzy, f_mix
  sensor     :: seed for measurement RNG
  solver     :: reltol, abstol, dtmax, solver_symbol
"""
Base.@kwdef struct HybridConfig
    tracking::Symbol      = :velocity
    estimator::Symbol     = :kalman
    use_dhat::Bool        = false
    use_asmc::Bool        = true
    use_mpc::Bool         = false
    use_pid::Bool         = false
    fuzzy::Bool           = false
    fixed_weights::NTuple{3,Float64} = (1.0, 0.0, 0.0)
    use_pose_fix::Bool    = false
    pose_fix_tier::Symbol = :transit
    f_est::Float64        = 1000.0
    f_mpc::Float64        = 100.0
    f_pid::Float64        = 100.0
    f_fuzzy::Float64      = 50.0
    f_mix::Float64        = 1000.0
    sensor_seed::Int      = 42
    reltol::Float64       = 1e-9
    abstol_bristle::Float64 = 1e-10
    dtmax::Float64        = 1e-3
    maxiters::Int         = 10_000_000
    solver_symbol::Symbol = :FBDF
    saveat_hz::Float64    = 500.0
    abstol::Union{Nothing, Vector{Float64}} = nothing
end

# Helper used by scheduler to decide which blocks are enabled.
enabled_controllers(cfg::HybridConfig) =
    (asmc = cfg.use_asmc, mpc = cfg.use_mpc, pid = cfg.use_pid)

end # module
