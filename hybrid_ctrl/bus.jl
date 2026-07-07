# =============================================================================
# hybrid_ctrl/bus.jl — ControllerBus: shared mutable state between callbacks
# =============================================================================
module BusMod

using StaticArrays

export ControllerBus, reset_bus!

"""
    ControllerBus

Mutable rendezvous between the discrete callbacks and the continuous plant RHS.
The RHS only reads `v_cmd`; callbacks write everything else.
"""
mutable struct ControllerBus
    v_cmd::SVector{4,Float64}      # ZOH motor-voltage command to plant
    xhat::SVector{6,Float64}       # [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]
    d_hat::SVector{3,Float64}      # SMO lumped disturbance estimate
    W_asmc::SVector{3,Float64}     # ASMC task-space wrench
    W_mpc::SVector{3,Float64}      # MPC task-space wrench
    W_pid::SVector{3,Float64}      # PID task-space wrench
    weights::SVector{3,Float64}    # blend weights (ASMC, MPC, PID)
    K::SVector{3,Float64}          # ASMC adaptive gains
    I_pid::SVector{3,Float64}      # PID integral accumulator
    mpc_warm::Vector{Float64}      # MPC warm-start (previous solution)
    mpc_last_u::SVector{4,Float64} # MPC last-applied voltage (fallback)
    tau_wheel::SVector{4,Float64}  # last applied wheel torque (diagnostic)
    use_dhat::Bool                 # SMO disturbance feedforward flag
    t_now::Base.RefValue{Float64}  # current integration time (set by callbacks)
    # Scratch for callbacks
    y_last::NamedTuple             # last sensor measurement
    v_cmd_prev::SVector{4,Float64} # for rate-limiting
end

function ControllerBus(; v_cmd = zeros(SVector{4}), xhat = zeros(SVector{6}),
                         d_hat = zeros(SVector{3}), K = zeros(SVector{3}),
                         I_pid = zeros(SVector{3}), mpc_warm = Float64[],
                         mpc_last_u = zeros(SVector{4}))
    return ControllerBus(v_cmd, xhat, d_hat, zeros(SVector{3}), zeros(SVector{3}),
                         zeros(SVector{3}), zeros(SVector{3}), K, I_pid,
                         mpc_warm, mpc_last_u, zeros(SVector{4}), false, Ref(0.0),
                         (; θ=zeros(4), ω=zeros(4), a_x=0.0, a_y=0.0, g_z=0.0),
                         zeros(SVector{4}))
end

function reset_bus!(bus::ControllerBus)
    bus.v_cmd       = zeros(SVector{4})
    bus.xhat        = zeros(SVector{6})
    bus.d_hat       = zeros(SVector{3})
    bus.W_asmc      = zeros(SVector{3})
    bus.W_mpc       = zeros(SVector{3})
    bus.W_pid       = zeros(SVector{3})
    bus.weights     = zeros(SVector{3})
    bus.K           = zeros(SVector{3})
    bus.I_pid       = zeros(SVector{3})
    bus.mpc_warm    = Float64[]
    bus.mpc_last_u  = zeros(SVector{4})
    bus.tau_wheel   = zeros(SVector{4})
    bus.use_dhat    = false
    bus.t_now[]     = 0.0
    bus.y_last      = (; θ=zeros(4), ω=zeros(4), a_x=0.0, a_y=0.0, g_z=0.0)
    bus.v_cmd_prev  = zeros(SVector{4})
    return bus
end

end # module
