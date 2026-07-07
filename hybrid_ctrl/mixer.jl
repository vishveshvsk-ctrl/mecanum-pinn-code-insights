# =============================================================================
# hybrid_ctrl/mixer.jl — blend + O-config allocation + motor inversion
# =============================================================================
module MixerMod

using StaticArrays

export mix_and_allocate!

"""
    mix_and_allocate!(bus, params, motor, cfg)

Blend enabled controller wrenches by `bus.weights` → task-space wrench W;
map W to 4 wheel torques via the O-config allocation; invert the motor map to
per-wheel voltage at current ω; saturate to ±V_max, |i|≤i_max, and ΔV rate
limit; write `bus.v_cmd`.
"""
function mix_and_allocate!(bus, params, motor, cfg)
    w = bus.weights
    W = w[1]*bus.W_asmc + w[2]*bus.W_mpc + w[3]*bus.W_pid

    # O-configuration allocation (run_one.jl:504–512)
    lever = params.R / (params.l + params.h)
    M1 = 0.25 * (W[1] - W[2] - lever * W[3])
    M2 = 0.25 * (W[1] + W[2] + lever * W[3])
    M3 = 0.25 * (W[1] + W[2] - lever * W[3])
    M4 = 0.25 * (W[1] - W[2] + lever * W[3])
    τ_wheel = SVector(M1, M2, M3, M4)

    # Saturate wheel torque (smooth) before inversion
    τ_max = params.Max_torque
    τ_sat = Main.smooth_sat.(τ_wheel, τ_max, 4)

    # Invert motor map: desired torque -> voltage at current estimated wheel speed
    ω = SVector(bus.xhat[1], bus.xhat[2], bus.xhat[3])  # placeholder: should use ω estimate
    # Better: use measured wheel speeds from last sensor sample if available
    if hasproperty(bus.y_last, :ω)
        ω = bus.y_last.ω
    end
    V = Main.PlantMod.motor_torque_inverse(τ_sat, ω, motor)

    # Rate limit on voltage
    dt_mix = 1.0 / cfg.f_mix
    dV = V - bus.v_cmd_prev
    dV_lim = clamp.(dV, -motor.dV_max*dt_mix, motor.dV_max*dt_mix)
    V = bus.v_cmd_prev + dV_lim

    bus.v_cmd = V
    bus.v_cmd_prev = V
    bus.tau_wheel = Main.PlantMod.motor_torque(V, ω, motor)
    return V
end

end # module
