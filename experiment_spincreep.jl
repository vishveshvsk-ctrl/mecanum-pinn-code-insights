#!/usr/bin/env julia
# Evaluate the ALREADY-TUNED controllers (no re-tuning) on spin_creep — a
# near-stationary, high-yaw VelRef.  Only the yaw-rate (ω) and heading (ψ)
# numbers are physically meaningful here; creep Vx/Vy are tiny by construction.
# Reuses tune_controller.jl's run_controller + controller_metrics verbatim
# (included with the main-guard, so no tuning runs).
using Pkg; Pkg.activate(".")
include("tune_controller.jl")   # main-guarded: brings run_controller, space_for, decode, controller_metrics
using JSON, TOML, Random
using Statistics: mean

const DIR = "trajectory_files_run_0p5_main"
const CFG = Dict(
    :asmc => "runs_controller/asmc_clean/best_config.json",                      # dxNES 1.28
    :pid  => "runs_controller_pid_b300_kdposfix/pid_clean/best_config.json",     # Kd_pos-fixed 16.93
    :mpc  => "runs_controller/mpc_clean/best_config.json",                       # dxNES LTV 67.3
)

# Reconstruct the tuned controller from the saved theta (exactly reproduces tuning).
load_kw(ctrl) = decode(Float64.(JSON.parse(read(CFG[ctrl], String))["theta"]), space_for(ctrl))

# Reference heading by integrating the yaw-rate command (velmode tracks ω, not ψ).
function heading_ref(ref, ts)
    psi = zeros(length(ts)); acc = 0.0
    for i in 2:length(ts)
        acc += 0.5 * (ref.Wz(ts[i]) + ref.Wz(ts[i-1])) * (ts[i] - ts[i-1])
        psi[i] = acc
    end
    return psi
end

function eval_ctrl(ctrl, combo)
    tr = (name="spin_creep", profile_toml="spin_creep_mu_0p5.toml", combo_idx=combo,
          ref_type=:velref, mu=0.5, config_dir=DIR, run_mode=:velocity)
    probe, ref, mode = run_controller(ctrl, load_kw(ctrl), :clean, tr; seed=42)
    m  = controller_metrics(probe, ref, mode)
    ts = [p.t for p in probe]
    # yaw rate: u[3]=ψ̇ actual, ref.Wz = command
    wz_des = [ref.Wz(t) for t in ts]
    # heading drift: u[4]=ψ actual vs integrated command
    psi_ref = heading_ref(ref, ts)
    eψ = [Main.EstimatorMod._wrap_angle(probe[i].u[4] - psi_ref[i]) for i in eachindex(probe)]
    return (peak_wz=maximum(abs, wz_des),
            rms_w=m.abs.rms_w, max_w=m.abs.max_w,
            head_final=abs(eψ[end]), head_max=maximum(abs, eψ),
            rms_vx=m.abs.rms_vx, rms_vy=m.abs.rms_vy, ce=m.ce)
end

# combos span yaw magnitude: 4=[-1.0,1.5] low, 7=[2.0,-2.0] mid, 1=[3.1,3.1] high
const COMBOS = [4, 7, 1]
println("spin_creep @ μ=0.5, oracle=clean — yaw-rate (ω) & heading (ψ) are the meaningful axes\n")
for combo in COMBOS
    println("──── combo $combo ────")
    for ctrl in (:asmc, :pid, :mpc)
        r = eval_ctrl(ctrl, combo)
        println(rpad(string(ctrl), 5), " | peak|ωdes|=", rpad(round(r.peak_wz, digits=2), 5),
                " rad/s | ω_rms=", rpad(round(r.rms_w*1e3, digits=1), 6), " max_ω_err=", rpad(round(r.max_w*1e3, digits=1), 6),
                " mrad/s | ψ final=", rpad(round(r.head_final, digits=3), 6), " max=", rpad(round(r.head_max, digits=3), 6),
                " rad | creep rms=(", round(r.rms_vx*1e3, digits=1), ",", round(r.rms_vy*1e3, digits=1),
                ")mm/s ce=", round(r.ce, digits=1))
    end
    println()
end
println("DONE")
