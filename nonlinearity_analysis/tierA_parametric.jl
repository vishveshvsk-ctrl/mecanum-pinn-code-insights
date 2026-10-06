#!/usr/bin/env julia
# =============================================================================
# tierA_parametric.jl — parametric near-linearity scan: what does it take to
# push the plant into nonlinear regimes?
#
# Cases (all against the legacy envelope caps, exact plant_rhs! steady states,
# radial continuation + clamped-motor verification — same machinery as
# tierAC_static_jacobian.jl, parameterised on a plant pack):
#
#   base        mu=0.5, chi=0.005, case 1, COM as built
#   com2x/com3x COM offset x2 / x3           (static N skew + inertial coupling)
#   chi4x       chi=0.020                    (Adamov spin coupling x4)
#   mu03        mu=0.3                       (friction circle x0.6)
#   case2       low-viscous                  (damping down 10x -> friction-dominated)
#   extreme     mu=0.3 + chi=0.020 + COM x3  (all levers at once)
#
# Output: nonlinearity_analysis/results/tierA_parametric_report.md (+ .jld2)
# =============================================================================

using Pkg; Pkg.activate(".")

const ROOT = @__DIR__() |> dirname
cd(ROOT)

include(joinpath(ROOT, "run_one.jl")); using .Profiles
include(joinpath(ROOT, "hybrid_ctrl", "bus.jl"));   using .BusMod
include(joinpath(ROOT, "hybrid_ctrl", "plant.jl")); using .PlantMod

using StaticArrays
using LinearAlgebra
using ForwardDiff
using Statistics
using Printf
using JLD2

LinearAlgebra.BLAS.set_num_threads(1)

const OUTDIR = joinpath(ROOT, "nonlinearity_analysis", "results")
mkpath(OUTDIR)

const BASE0 = Profiles.load_base("trajectory_files_run_0p5_main")
const P_PAT = SMatrix{4,3,Float64}([ 0.5 -0.5 -0.5;
                                     0.5  0.5  0.5;
                                     0.5  0.5 -0.5;
                                     0.5 -0.5  0.5])
const VCAP = SVector(0.63, 0.63, 3.8)
const PHASES = [(k - 1) / 8 * (π / 6) for k in 1:8]
const ROWS23 = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

struct PlantPack
    params::PlatformParams
    chi::Float64
    p1::Float64
    p2::Float64
    lugre::LuGreParams
    motor::PlantMod.MotorParams
    bw::Float64          # G*eta*Kt/(Ra*J_wheel) — unclamped voltage gain, wheel rows
end

function make_pack(; mu=0.5, chi=0.005, case=1, com_scale=1.0)
    b = deepcopy(BASE0)
    b["platform"]["com_offset"]["aX"] *= com_scale
    b["platform"]["com_offset"]["aY"] *= com_scale
    params = PlatformParams(b; mu_friction = mu)
    p1, p2 = case == 1 ? (params.p1_case1, params.p2_case1) :
                         (params.p1_case2, params.p2_case2)
    motor = PlantMod.MotorParams()
    PlantPack(params, chi, p1, p2, LuGreParams(), motor,
              motor.G * motor.eta * motor.Kt / motor.Ra / params.J_wheel)
end

function mech_rows(pack::PlantPack, v, omega, gamma, z, theta, Vcmd)
    bus = BusMod.ControllerBus()
    bus.v_cmd = Vcmd
    plant_p = PlantMod.PlantODEParams(pack.params, pack.chi, pack.p1, pack.p2,
                                      :adamov, pack.lugre, pack.motor, bus)
    T = promote_type(eltype(v), eltype(omega), eltype(gamma), eltype(z), eltype(theta))
    u = zeros(T, 30)
    u[1:3] = v; u[5:8] = theta; u[9:12] = omega; u[13:16] = gamma
    u[19:22] = z[1:4]; u[23:26] = z[5:8]; u[27:30] = z[9:12]
    du = zeros(T, 30)
    PlantMod.plant_rhs!(du, u, plant_p, zero(T))
    return du[ROWS23]
end

function ss_residual(pack, y, vstar, theta)
    a = y[1:3]; omega = y[4:7]; gamma = y[8:11]; z = y[12:23]
    r0 = mech_rows(pack, vstar, omega, gamma, z, theta, zeros(SVector{4,Float64}))
    BV = pack.bw .* (P_PAT * a)
    return vcat(r0[1:3], r0[4:7] .+ BV, r0[8:23])
end

function solve_ss(pack, vstar, theta; y0 = zeros(23), maxit = 60, tol = 1e-9)
    y = copy(y0)
    F = ss_residual(pack, y, vstar, theta)
    rn = norm(F, Inf); it = 0
    while rn > tol && it < maxit
        J = ForwardDiff.jacobian(yy -> ss_residual(pack, yy, vstar, theta), y)
        step = J \ F
        rho = 1.0; ok = false
        while rho > 1e-4
            yt = y .- rho .* step
            Ft = ss_residual(pack, yt, vstar, theta)
            if norm(Ft, Inf) < rn
                y, F = yt, Ft; ok = true; break
            end
            rho /= 2
        end
        ok || break
        rn = norm(F, Inf); it += 1
    end
    return y, rn < 1e-7, rn
end

function solve_ss_path(pack, vstar, theta; rhos = (0.1, 0.25, 0.5, 0.75, 1.0))
    y = zeros(23); rn = Inf
    for rho in rhos
        y, conv, rn = solve_ss(pack, rho .* vstar, theta; y0 = y)
        conv || return y, false, rn
    end
    V = P_PAT * y[1:3]
    rex = mech_rows(pack, vstar, y[4:7], y[8:11], y[12:23], theta, V)
    rxn = norm(rex, Inf)
    return y, rxn < 1e-6, max(rn, rxn)
end

function feasible(pack, y, omega)
    V = P_PAT * y[1:3]
    i = (V .- pack.motor.Kb .* pack.motor.G .* omega) ./ pack.motor.Ra
    return all(abs.(V) .<= pack.motor.V_max + 1e-9) &&
           all(abs.(i) .<= pack.motor.i_max + 1e-9)
end

# --- per-case analysis ---------------------------------------------------------
const FRACS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
const FR3  = [-1.0, -0.5, 0.0, 0.5, 1.0]
const FR3P = [-0.75, -0.375, 0.0, 0.375, 0.75]

function analyze_case(pack)
    # axis sweeps (continuation over FRACS)
    sweeps = Dict{Int, Vector{NamedTuple}}()
    for ax in 1:3
        rows = NamedTuple[]
        for sgn in (1.0, -1.0)
            yprev = zeros(23)
            for f in FRACS
                vstar = SVector(sgn .* (f .* VCAP .* [ax == j for j in 1:3])...)
                ok_all = true; amean = zeros(3); nok = 0
                for ph in PHASES
                    theta = SVector{4}(fill(ph, 4))
                    if f == first(FRACS)
                        y, conv, rn = solve_ss_path(pack, vstar, theta)
                    else
                        y, conv, rn = solve_ss(pack, vstar, theta; y0 = yprev)
                        if conv
                            V = P_PAT * y[1:3]
                            rex = mech_rows(pack, vstar, y[4:7], y[8:11], y[12:23], theta, V)
                            conv = norm(rex, Inf) < 1e-6
                        end
                    end
                    yprev = y
                    if conv && feasible(pack, y, y[4:7])
                        amean .+= y[1:3]; nok += 1
                    end
                end
                ok_all = nok == length(PHASES)
                push!(rows, (frac = sgn * f, feas = ok_all,
                             a = ok_all ? amean ./ nok : nothing))
            end
        end
        sweeps[ax] = rows
    end

    # 3-D grid for nu
    U = Float64[]; Y = Float64[]
    for fx in FR3, fy in FR3, fp in FR3P
        fr = [fx, fy, fp]
        norm(fr) < 1e-12 && continue
        vstar = SVector(fr .* VCAP...)
        aa = zeros(3); nok = 0
        for ph in PHASES
            theta = SVector{4}(fill(ph, 4))
            y, conv, rn = solve_ss_path(pack, vstar, theta)
            if conv && feasible(pack, y, y[4:7])
                aa .+= y[1:3]; nok += 1
            end
        end
        nok == length(PHASES) || continue
        append!(U, fr .* VCAP); append!(Y, aa ./ nok)
    end
    nsamp = div(length(U), 3)
    if nsamp >= 6
        Um = reshape(U, 3, :); Ym = reshape(Y, 3, :)
        G0 = Ym * Um' * inv(Um * Um')
        res0 = Ym .- G0 * Um
        rn_rel = vec(sqrt.(sum(res0 .^ 2; dims=1)) ./ sqrt.(sum(Ym .^ 2; dims=1)))
        nu0 = maximum(rn_rel); nu0p = quantile(rn_rel, 0.95)
        sst = sum((Ym .- mean(Ym; dims=2)) .^ 2)
        R2 = 1 - sum(res0 .^ 2) / sst
    else
        nu0 = NaN; nu0p = NaN; R2 = NaN
    end

    # per-axis secant gain spread and max feasible fraction
    gains = NamedTuple[]
    for ax in 1:3
        gs = Float64[]
        for r in sweeps[ax]
            (r.frac > 0 && r.feas) || continue
            push!(gs, r.a[ax] / (r.frac * VCAP[ax]))
        end
        feas_max = maximum([abs(r.frac) for r in sweeps[ax] if r.feas]; init = 0.0)
        push!(gains, (axis = ax,
                      spread = length(gs) > 1 ? maximum(gs) / minimum(gs) : NaN,
                      feas_max = feas_max))
    end
    return (nu = nu0, nu_p95 = nu0p, R2 = R2, nsamp = nsamp, gains = gains,
            sweeps = sweeps)
end

# --- run all cases --------------------------------------------------------------
const CASES = [
    ("base",    ()),
    ("com2x",   (com_scale = 2.0,)),
    ("com3x",   (com_scale = 3.0,)),
    ("chi4x",   (chi = 0.020,)),
    ("mu03",    (mu = 0.3,)),
    ("case2",   (case = 2,)),
    ("extreme", (mu = 0.3, chi = 0.020, com_scale = 3.0)),
]

results = Dict{String, Any}()
for (name, kw) in CASES
    t0 = time()
    pack = make_pack(; kw...)
    results[name] = analyze_case(pack)
    r = results[name]
    @printf("%-8s  nu=%.4f (p95 %.4f)  R2=%.5f  n=%2d  spreads=[%s]  feas_max=[%s]  (%.0fs)\n",
            name, r.nu, r.nu_p95, r.R2, r.nsamp,
            join([@sprintf("%.2f", g.spread) for g in r.gains], ","),
            join([@sprintf("%.2f", g.feas_max) for g in r.gains], ","),
            time() - t0)
end

jldsave(joinpath(OUTDIR, "tierA_parametric_results.jld2"); results)

rep = IOBuffer()
println(rep, "# Parametric near-linearity scan (Tier A machinery)\n")
println(rep, "Steady-state map v -> V_req against the exact plant; envelope caps are the ",
        "LEGACY ones (0.63, 0.63, 3.8). nu = Allgoewer residual of the through-origin ",
        "best linear map (max / p95); spread = per-axis secant gain spread; ",
        "feas_max = max feasible fraction of cap per axis (x, y, psi).\n")
println(rep, "| case | nu max | nu p95 | R2 | grid pts | gain spread x/y/psi | feas max x/y/psi |")
println(rep, "|---|---|---|---|---|---|---|")
for (name, _) in CASES
    r = results[name]
    @printf(rep, "| %s | %.4f | %.4f | %.5f | %d | %s | %s |\n",
            name, r.nu, r.nu_p95, r.R2, r.nsamp,
            join([@sprintf("%.2f", g.spread) for g in r.gains], " / "),
            join([@sprintf("%.2f", g.feas_max) for g in r.gains], " / "))
end
open(joinpath(OUTDIR, "tierA_parametric_report.md"), "w") do io
    write(io, String(take!(rep)))
end
println("\n", String(take!(rep)))
println("saved: $(joinpath(OUTDIR, "tierA_parametric_report.md"))")
