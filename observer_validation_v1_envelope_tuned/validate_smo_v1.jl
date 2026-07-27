# =============================================================================
# validate_smo_v1.jl  —  SMO observer convergence validation (envelope-tuned)
#
# Validates the SMO estimator from hybrid_ctrl/estimators.jl against the v1
# plant model (30-D LuGre + roller dynamics) with envelope-tuned parameters:
#   - MotorParams: i_max=12.8 A, Max_torque=6.0 N·m (v1)
#   - SMO: L=[6,6,20], K=[9,9,100], δ=0.05 (per acceleration envelope doc)
#   - Sensor: enc_cpr=4000, σ_gyro=0.005, seed=42 (v1)
#
# Tests the SMO against the full plant dynamics with noisy sensor measurements
# only — the SMO must recover true velocity from slip-corrupted wheel-speed
# pseudo-measurements.
#
# Run:  julia --project=.. validate_smo_v1.jl
# =============================================================================

using OrdinaryDiffEq, StaticArrays, LinearAlgebra, Plots, Printf, Random, Statistics
using TOML

outdir = @__DIR__

# =============================================================================
# 1. Include run_one.jl for PlatformParams, LuGreParams, sawtooth_approx,
#    lugre_dyn_rates, etc. (all defined at Main level)
# =============================================================================
include(joinpath(@__DIR__, "..", "run_one.jl"))

# =============================================================================
# 2. Include hybrid_ctrl modules (plant, sensors, estimators, bus)
# =============================================================================
include(joinpath(@__DIR__, "..", "hybrid_ctrl", "plant.jl"))
include(joinpath(@__DIR__, "..", "hybrid_ctrl", "sensors.jl"))
include(joinpath(@__DIR__, "..", "hybrid_ctrl", "estimators.jl"))
include(joinpath(@__DIR__, "..", "hybrid_ctrl", "bus.jl"))

using .PlantMod, .SensorMod, .EstimatorMod, .BusMod

# =============================================================================
# 3. Parameters
# =============================================================================
base = TOML.parsefile(joinpath(@__DIR__, "..", "trajectory_files_run_0p3_main", "base.toml"))
params = PlatformParams(base; mu_friction=0.3)

# Override Max_torque to v1 value (6.0 N·m)
params = PlatformParams(params.h, params.l, params.R, params.Ra,
    params.m, params.m_wheel, params.J_wheel, params.J_roller,
    params.ms, params.Is,
    params.p1_case1, params.p2_case1, params.p1_case2, params.p2_case2,
    params.f_coulomb, params.N_total, params.rollers_per_wheel,
    params.delta, params.wc_x, params.wc_y,
    params.aX, params.aY, params.N_per_roller,
    params.M_inv, params.M_aug, params.M_aug_inv,
    6.0)

motor = MotorParams(i_max=12.8, dynamic_electrical=false)
lugre = LuGreParams()
sm = SensorModel(enc_cpr=4000, σ_gyro=0.005, seed=42)

# Envelope-tuned SMO: L proportional to acceleration caps (2× margin)
smo = SMOEstimator(
    L = SVector(6.0, 6.0, 20.0),
    K = SVector(9.0, 9.0, 100.0),
    δ = 0.05,
    rate_hz = 1000.0,
    use_dhat = true
)

bus = ControllerBus()

chi = 0.005
p1  = params.p1_case1
p2  = params.p2_case1
p = PlantODEParams(params, chi, p1, p2, :lugre_adamov, lugre, motor, bus)

# =============================================================================
# 4. Open-loop voltage command (excites all three axes)
# =============================================================================
T_total = 4.0
dt_smo  = 1.0 / 1000.0

function voltage_command(t)
    v1 = 0.8 * sin(2π * 0.5 * t) + 0.4 * sin(2π * 1.2 * t)
    v2 = 0.8 * sin(2π * 0.7 * t + 1.2) + 0.4 * sin(2π * 1.5 * t + 0.8)
    v3 = 0.6 * sin(2π * 0.3 * t + 2.1) + 0.3 * sin(2π * 0.9 * t + 1.5)
    v4 = 0.6 * sin(2π * 0.4 * t + 0.5) + 0.3 * sin(2π * 1.1 * t + 2.8)
    return SVector(v1, v2, v3, v4)
end

# =============================================================================
# 5. Logging arrays
# =============================================================================
t_log     = Float64[]
true_Vx   = Float64[]; true_Vy   = Float64[]; true_psi_dot = Float64[]
est_Vx    = Float64[]; est_Vy    = Float64[]; est_psi_dot  = Float64[]
dhat_x    = Float64[]; dhat_y    = Float64[]; dhat_psi     = Float64[]
z_Vx      = Float64[]; z_Vy      = Float64[]; z_gyro       = Float64[]
meas_ω1   = Float64[]; meas_ω2   = Float64[]; meas_ω3      = Float64[]; meas_ω4 = Float64[]
error_Vx  = Float64[]; error_Vy  = Float64[]; error_psi_dot = Float64[]

# =============================================================================
# 6. Callback: sensor + SMO update at 1000 Hz
# =============================================================================
function sensor_smo_callback(integrator)
    u = integrator.u; t = integrator.t
    bus.v_cmd = voltage_command(t)
    du = similar(u); plant_rhs!(du, u, p, t)
    y = simulate_measurement(u, du, sm, t)
    push!(meas_ω1, y.ω[1]); push!(meas_ω2, y.ω[2])
    push!(meas_ω3, y.ω[3]); push!(meas_ω4, y.ω[4])
    estimator_update!(bus, y, smo, params, dt_smo)
    Hω = smo.wheel_H; z_vel = Hω \ y.ω
    push!(z_Vx, z_vel[1]); push!(z_Vy, z_vel[2]); push!(z_gyro, y.g_z)
    push!(t_log, t)
    push!(true_Vx, u[1]); push!(true_Vy, u[2]); push!(true_psi_dot, u[3])
    push!(est_Vx, bus.xhat[1]); push!(est_Vy, bus.xhat[2]); push!(est_psi_dot, bus.xhat[3])
    push!(dhat_x, bus.d_hat[1]); push!(dhat_y, bus.d_hat[2]); push!(dhat_psi, bus.d_hat[3])
    push!(error_Vx, u[1] - bus.xhat[1])
    push!(error_Vy, u[2] - bus.xhat[2])
    push!(error_psi_dot, u[3] - bus.xhat[3])
end

# =============================================================================
# 7. Solve
# =============================================================================
u0 = zeros(30)
prob = ODEProblem(plant_rhs!, u0, (0.0, T_total), p)
cb = PeriodicCallback(sensor_smo_callback, dt_smo, initial_affect=true)
println("Solving plant ODE + SMO observer (T_total = $(T_total) s)...")
@time sol = solve(prob, TRBDF2(), reltol=1e-8, abstol=1e-9, dtmax=1e-3,
    maxiters=1e7, callback=cb, saveat=dt_smo)
println("Done. $(length(t_log)) steps logged.")

# =============================================================================
# 8. Plots
# =============================================================================
p1 = plot(t_log, true_Vx,  label="True Vx",  lw=1.5, color=:black)
plot!(p1, t_log, est_Vx,   label="SMO V̂x",   lw=1.5, ls=:dash, color=:blue)
plot!(p1, t_log, z_Vx,     label="z_Vx",     lw=0.8, ls=:dot, color=:red, alpha=0.6)
xlabel!("Time (s)"); ylabel!("Vx (m/s)"); title!("Vx Tracking")
hline!([0], lw=0.5, ls=:dot, color=:gray)
p2 = plot(t_log, true_Vy,  label="True Vy",  lw=1.5, color=:black)
plot!(p2, t_log, est_Vy,   label="SMO V̂y",   lw=1.5, ls=:dash, color=:blue)
plot!(p2, t_log, z_Vy,     label="z_Vy",     lw=0.8, ls=:dot, color=:red, alpha=0.6)
xlabel!("Time (s)"); ylabel!("Vy (m/s)"); title!("Vy Tracking")
hline!([0], lw=0.5, ls=:dot, color=:gray)
p3 = plot(t_log, true_psi_dot, label="True ψ̇", lw=1.5, color=:black)
plot!(p3, t_log, est_psi_dot,  label="SMO ψ̂̇",  lw=1.5, ls=:dash, color=:blue)
plot!(p3, t_log, z_gyro,       label="Gyro",    lw=0.8, ls=:dot, color=:red, alpha=0.6)
xlabel!("Time (s)"); ylabel!("ψ̇ (rad/s)"); title!("Yaw Rate Tracking")
hline!([0], lw=0.5, ls=:dot, color=:gray)
plt_tracking = plot(p1, p2, p3, layout=(3,1), size=(1000, 800))
savefig(plt_tracking, joinpath(outdir, "01_velocity_tracking.png"))
println("Saved: 01_velocity_tracking.png")

p4 = plot(t_log, error_Vx,  label="e_Vx",  lw=1.5, color=:red)
plot!(p4, t_log, error_Vy,  label="e_Vy",  lw=1.5, color=:green)
plot!(p4, t_log, error_psi_dot, label="e_ψ̇", lw=1.5, color=:blue)
xlabel!("Time (s)"); ylabel!("Error"); title!("Estimation Error")
hline!([0], lw=0.5, ls=:dot, color=:gray); ylims!(-0.5, 0.5)
p5 = plot(t_log, abs.(error_Vx),  label="|e_Vx|",  lw=1.5, color=:red, yscale=:log10)
plot!(p5, t_log, abs.(error_Vy),  label="|e_Vy|",  lw=1.5, color=:green, yscale=:log10)
plot!(p5, t_log, abs.(error_psi_dot), label="|e_ψ̇|", lw=1.5, color=:blue, yscale=:log10)
xlabel!("Time (s)"); ylabel!("|Error|"); title!("Absolute Error (log)")
hline!([smo.δ], lw=0.5, ls=:dash, color=:gray, label="δ")
plt_error = plot(p4, p5, layout=(2,1), size=(1000, 600))
savefig(plt_error, joinpath(outdir, "02_estimation_error.png"))
println("Saved: 02_estimation_error.png")

zoom_idx = min(findfirst(t_log .> 0.5) || length(t_log), length(t_log))
p6 = plot(t_log[1:zoom_idx], true_Vx[1:zoom_idx],  label="True Vx",  lw=2, color=:black)
plot!(p6, t_log[1:zoom_idx], est_Vx[1:zoom_idx],   label="SMO V̂x",   lw=2, ls=:dash, color=:blue)
xlabel!("Time (s)"); ylabel!("Vx (m/s)"); title!("Vx Convergence — First 0.5s")
p7 = plot(t_log[1:zoom_idx], true_Vy[1:zoom_idx],  label="True Vy",  lw=2, color=:black)
plot!(p7, t_log[1:zoom_idx], est_Vy[1:zoom_idx],   label="SMO V̂y",   lw=2, ls=:dash, color=:blue)
xlabel!("Time (s)"); ylabel!("Vy (m/s)"); title!("Vy Convergence — First 0.5s")
p8 = plot(t_log[1:zoom_idx], true_psi_dot[1:zoom_idx], label="True ψ̇", lw=2, color=:black)
plot!(p8, t_log[1:zoom_idx], est_psi_dot[1:zoom_idx],  label="SMO ψ̂̇",  lw=2, ls=:dash, color=:blue)
xlabel!("Time (s)"); ylabel!("ψ̇ (rad/s)"); title!("Yaw Rate Convergence — First 0.5s")
plt_zoom = plot(p6, p7, p8, layout=(3,1), size=(1000, 800))
savefig(plt_zoom, joinpath(outdir, "03_convergence_zoom.png"))
println("Saved: 03_convergence_zoom.png")

p9 = plot(t_log, dhat_x,  label="d̂_x",  lw=1.5, color=:red)
plot!(p9, t_log, dhat_y,  label="d̂_y",  lw=1.5, color=:green)
plot!(p9, t_log, dhat_psi, label="d̂_ψ", lw=1.5, color=:blue)
xlabel!("Time (s)"); ylabel!("d̂ (m/s², rad/s²)"); title!("SMO Disturbance Estimate")
hline!([0], lw=0.5, ls=:dot, color=:gray)
savefig(p9, joinpath(outdir, "04_disturbance_estimate.png"))
println("Saved: 04_disturbance_estimate.png")

settle_idx = findfirst(t_log .> 0.2) || 1
rmse_Vx = sqrt(mean(abs2.(error_Vx[settle_idx:end])))
rmse_Vy = sqrt(mean(abs2.(error_Vy[settle_idx:end])))
rmse_psi = sqrt(mean(abs2.(error_psi_dot[settle_idx:end])))
println("\n========== SMO Convergence Metrics ==========")
println(@sprintf("  RMSE (after 0.2s):  Vx=%.4f m/s  Vy=%.4f m/s  ψ̇=%.4f rad/s", rmse_Vx, rmse_Vy, rmse_psi))
println(@sprintf("  Max error: Vx=%.4f  Vy=%.4f  ψ̇=%.4f", maximum(abs.(error_Vx[settle_idx:end])), maximum(abs.(error_Vy[settle_idx:end])), maximum(abs.(error_psi_dot[settle_idx:end]))))
println("=============================================")

p_metrics = plot(p1, p2, p3, p4, p9, p5, layout=(3,2), size=(1400, 1000),
    plot_title="SMO Observer Validation — envelope-tuned (L=[6,6,20], K=[9,9,100], δ=0.05)")
savefig(p_metrics, joinpath(outdir, "05_summary_dashboard.png"))
println("Saved: 05_summary_dashboard.png")
println("\nAll figures in: $(outdir)")
