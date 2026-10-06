#!/usr/bin/env julia
# =============================================================================
# tierB_multisine.jl — Tier B: open-loop odd-multisine Best-Linear-Approximation
# (Schoukens/Pintelon framework) for the voltage -> body-velocity plant.
#
# Excitation: task-space (one axis at a time, using the codebase's own O-config
# allocation patterns), random-phase multisine with ONLY bins k ≡ 1 (mod 4)
# excited — so non-excited odd bins (k ≡ 3 mod 4) measure ODD nonlinearity and
# even bins measure EVEN nonlinearity. Period-to-period variance at excited
# bins measures stochastic nonlinear distortion (no measurement noise here).
#
# OPERATING-POINT DESIGN: the near-linearity claim under test concerns the
# regime the trajectories actually occupy — SUSTAINED motion, not oscillation
# about rest. Runs are therefore biased: a constant task-space voltage holds a
# cruising operating point (bias_frac of the envelope cap, per Tier A's
# steady-state calibration) and the multisine probes AROUND it. A zero-bias
# reference run per axis is kept to document the velocity-reversal regime.
# Each analysis period is detrended (mean + linear trend) so open-loop drift
# does not smear power into the detection bins.
#
# Usage:  julia --project=. -t 6 nonlinearity_analysis/tierB_multisine.jl [--smoke]
# Output: nonlinearity_analysis/results/tierB_run_ax{ax}_b{bias}_e{exc}.jld2
#         nonlinearity_analysis/results/tierB_results.jld2
#         nonlinearity_analysis/results/tierB_report.md
# =============================================================================

using Pkg; Pkg.activate(".")

const ROOT = @__DIR__() |> dirname
cd(ROOT)

include(joinpath(ROOT, "run_one.jl")); using .Profiles
include(joinpath(ROOT, "hybrid_ctrl", "bus.jl"));   using .BusMod
include(joinpath(ROOT, "hybrid_ctrl", "plant.jl")); using .PlantMod

using StaticArrays
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks
using Random
using Statistics
using Printf
using JLD2

LinearAlgebra.BLAS.set_num_threads(1)

const SMOKE = "--smoke" in ARGS
const OUTDIR = joinpath(ROOT, "nonlinearity_analysis", "results")
mkpath(OUTDIR)

# --- plant point (same as Tier A: campaign physics) --------------------------
const BASE   = Profiles.load_base("trajectory_files_run_0p5_main")
const CHI    = Float64(BASE["physics"]["chi"])
const PARAMS = PlatformParams(BASE; mu_friction = Float64(BASE["physics"]["mu_friction"]))
const LUGRE  = LuGreParams()
const MOTOR  = PlantMod.MotorParams()
const P1, P2 = PARAMS.p1_case1, PARAMS.p2_case1

const P_PAT = SMatrix{4,3,Float64}([ 0.5 -0.5 -0.5;
                                     0.5  0.5  0.5;
                                     0.5  0.5 -0.5;
                                     0.5 -0.5  0.5])
const VCAP = SVector(0.63, 0.63, 3.8)

# --- multisine design ---------------------------------------------------------
# f0 = 0.1 Hz, band 0.1..10 Hz (control bandwidth ~1.5 Hz; rolloff visible to 10)
const F0   = 0.1
const FMAX = 10.0
const K_ALL = collect(1:round(Int, FMAX / F0))                 # 1..100
const K_EXC = filter(k -> k % 4 == 1, K_ALL)                   # 1,5,9,...,97
const K_ODD_DET  = filter(k -> k % 4 == 3, K_ALL)              # odd nonlinearity
const K_EVEN_DET = filter(k -> k % 2 == 0, K_ALL)              # even nonlinearity
const T_PER  = 1.0 / F0                                        # 10 s
const N_PER  = SMOKE ? 3 : 6                                   # periods total
const N_DROP = SMOKE ? 1 : 2                                   # transient periods
const FS_SAVE = 500.0                                          # saveat (Hz)
const SPP  = round(Int, T_PER * FS_SAVE)                       # samples per period

function multisine_coeffs(seed::Int)
    rng = Random.Xoshiro(seed)
    return 2π .* rand(rng, length(K_EXC))
end

"Unit-RMS multisine (random-phase, fixed per axis) evaluated at time t."
function make_multisine(phases)
    amps = fill(1.0, length(K_EXC))
    amps ./= sqrt(sum(amps .^ 2) / 2)
    f = let K = K_EXC, a = amps, ph = phases, f0 = F0
        t -> sum(a[k] * sin(2π * f0 * K[k] * t + ph[k]) for k in eachindex(K))
    end
    return f
end

# --- abstol vector matching the scheduler's 30-state quasi-static layout ------
function abstol30()
    at = fill(1e-7, 30)
    at[1:3]   .= 1e-8     # body vel
    at[5:8]   .= 1e-8     # theta
    at[9:12]  .= 1e-7     # omega
    at[13:16] .= 1e-7     # gamma
    at[17:18] .= 1e-7     # Xo, Yo
    at[19:26] .= 1e-10    # zx, zy bristles
    at[27:30] .= 1e-7     # zs
    return at
end

# --- one open-loop biased multisine run ---------------------------------------
function run_openloop(axis::Int, A_bias::Float64, A_exc::Float64, seed::Int)
    sfun = make_multisine(multisine_coeffs(1000 + axis))   # same waveform per axis
    pat  = P_PAT[:, axis]

    bus = BusMod.ControllerBus()
    plant_p = PlantMod.PlantODEParams(PARAMS, CHI, P1, P2, :adamov, LUGRE, MOTOR, bus)

    function set_v!(integrator)
        bus.v_cmd = (A_bias + A_exc * sfun(integrator.t)) .* pat
        return nothing
    end
    cb = PeriodicCallback(set_v!, 1e-3; initial_affect = true)

    u0 = zeros(30)
    u0[5:8] .= 0.1                     # match scheduler convention
    T = N_PER * T_PER
    t_eval = collect(range(0.0, T; length = round(Int, T * FS_SAVE) + 1))
    prob = ODEProblem(PlantMod.plant_rhs!, u0, (0.0, T), plant_p)
    sol = solve(prob, FBDF(); reltol = 1e-9, abstol = abstol30(),
                saveat = t_eval, callback = cb, dtmax = 1e-3, maxiters = 10_000_000)

    # NOTE: sol.t is polluted by callback/stop times — sample the dense
    # interpolant at exactly the saveat grid instead.
    t = t_eval
    vu = sol(t_eval)
    v = reduce(hcat, [vu[k][1:3] for k in eachindex(vu)])            # 3 x N
    a_in = [A_bias + A_exc * sfun(tk) for tk in t]                   # task channel
    return t, v, a_in
end

# --- spectral analysis --------------------------------------------------------
# Explicit DFT at the analysis bins only (K_ALL << SPP), so no FFT dependency.
const DFT_E = let N = SPP
    exp.(-2π * im ./ N .* (K_ALL * (0:N-1)')) ./ N
end

function detrend(x)
    n = length(x)
    t̄ = (n + 1) / 2                       # centred index => intercept = mean
    x̄ = mean(x)
    slope = sum(((1:n) .- t̄) .* (x .- x̄)) / sum(((1:n) .- t̄) .^ 2)
    return x .- (x̄ .+ slope .* ((1:n) .- t̄))
end

function analyze(t, v, a_in)
    n_keep = N_PER - N_DROP
    N = n_keep * SPP
    idx = (length(t) - N + 1):length(t)
    length(idx) == N || error("sample-count mismatch: have $(length(t)), need $N")

    # per-period DFTs of the DETRENDED series: row = bin (K_ALL), col = period
    Ain_p = zeros(ComplexF64, length(K_ALL), n_keep)
    V_p   = [zeros(ComplexF64, length(K_ALL), n_keep) for _ in 1:3]
    for p in 1:n_keep
        sl = idx[(p - 1) * SPP + 1 : p * SPP]
        Ain_p[:, p] = DFT_E * detrend(a_in[sl])
        for c in 1:3
            V_p[c][:, p] = DFT_E * detrend(v[c, sl])
        end
    end
    bidx(k) = findfirst(==(k), K_ALL)

    # BLA at excited bins: mean over periods; stochastic distortion: std/|mean|
    bla = [zeros(ComplexF64, length(K_EXC)) for _ in 1:3]
    bla_std_ratio = [zeros(length(K_EXC)) for _ in 1:3]
    for (j, k) in enumerate(K_EXC)
        for c in 1:3
            Gs = [V_p[c][bidx(k), p] / Ain_p[bidx(k), p] for p in 1:n_keep]
            bla[c][j] = mean(Gs)
            bla_std_ratio[c][j] = abs(std(Gs) / mean(Gs))
        end
    end

    # distortion power per output channel (summed over kept periods)
    chan = NamedTuple[]
    for c in 1:3
        pexc  = sum(abs2, V_p[c][bidx.(K_EXC), :])
        podd  = sum(abs2, V_p[c][bidx.(K_ODD_DET), :])
        peven = sum(abs2, V_p[c][bidx.(K_EVEN_DET), :])
        push!(chan, (channel=c,
                     odd_db  = 10 * log10(podd / pexc),
                     even_db = 10 * log10(peven / pexc),
                     tot_db  = 10 * log10((podd + peven) / pexc)))
    end
    stoch = [10 * log10(mean(bla_std_ratio[c] .^ 2)) for c in 1:3]

    return (k_exc = K_EXC, f_exc = K_EXC .* F0, bla = bla,
            bla_std_ratio = bla_std_ratio, chan = chan, stoch_db = stoch,
            v_mean = [mean(v[c, idx]) for c in 1:3],
            v_exc_rms = [std(v[c, idx]) for c in 1:3],
            a_exc_rms = std(a_in[idx]))
end

# --- amplitude calibration from Tier A ----------------------------------------
function calibrate()
    f = joinpath(OUTDIR, "tierAC_results.jld2")
    isfile(f) || error("run tierAC_static_jacobian.jl first (need calibration)")
    d = load(f)
    calib = Dict{Tuple{Int,Float64}, Float64}()
    for ax in 1:3
        for r in d["axis_sweep"][ax]
            r.frac <= 0 && continue
            ok = [p for p in r.phases if p.conv && p.feas]
            length(ok) == length(r.phases) || continue
            calib[(ax, r.frac)] = mean([p.a[ax] for p in ok])
        end
    end
    return calib
end

# --- driver -------------------------------------------------------------------
# (bias_frac, exc_frac) per run; a zero-bias reference at exc 0.25 per axis.
const JOBSPEC = SMOKE ? [(0.25, 0.25)] :
                      [(0.25, 0.10), (0.25, 0.25), (0.50, 0.10), (0.50, 0.25), (0.0, 0.25)]
const AXES_B  = SMOKE ? [1] : [1, 2, 3]

function main()
    calib = calibrate()
    jobs = Tuple{Int,Float64,Float64}[]
    for ax in AXES_B, (bf, ef) in JOBSPEC
        if !haskey(calib, (ax, ef))
            println("SKIP ax=$ax exc=$ef — no feasible Tier A calibration")
            continue
        end
        A_bias = bf == 0.0 ? 0.0 :
                 (haskey(calib, (ax, bf)) ? calib[(ax, bf)] :
                  (println("SKIP ax=$ax bias=$bf — no feasible Tier A calibration"); continue))
        push!(jobs, (ax, A_bias, calib[(ax, ef)]))
    end
    results = Vector{Any}(nothing, length(jobs))

    Threads.@threads for j in eachindex(jobs)
        ax, A_bias, A_exc = jobs[j]
        t0 = time()
        t, v, a_in = run_openloop(ax, A_bias, A_exc, 1000 + ax)
        spec = analyze(t, v, a_in)
        results[j] = (axis=ax, A_bias=A_bias, A_exc=A_exc, spec=spec, runtime=time() - t0)
        tag = "tierB_run_ax$(ax)_b$(round(A_bias; digits=2))_e$(round(A_exc; digits=2)).jld2"
        jldsave(joinpath(OUTDIR, tag); t, v, a_in, axis=ax, A_bias, A_exc)
        @printf("done ax=%d bias=%.2fV exc=%.2fV  v_mean=%s  (%.0f s)\n", ax, A_bias, A_exc,
                string(round.(spec.v_mean ./ VCAP; sigdigits=2)), time() - t0)
    end

    jldsave(joinpath(OUTDIR, "tierB_results.jld2"); results)

    rep = IOBuffer()
    println(rep, "# Tier B — open-loop biased odd-multisine BLA\n")
    println(rep, "f0=$F0 Hz, excited bins k≡1 mod 4 up to $FMAX Hz, $N_PER periods ",
            "(drop $N_DROP), Fs=$FS_SAVE Hz, per-period detrended. Bias = task-space ",
            "voltage holding the cruising operating point (Tier A calibration); ",
            "excitation = multisine RMS voltage around it. No measurement noise: ",
            "period variance = stochastic nonlinear distortion.\n")
    println(rep, "| axis | bias [V] | exc [V] | v_mean/cap | odd NL [dB] | even NL [dB] | total NL [dB] | stochastic [dB] |")
    println(rep, "|---|---|---|---|---|---|---|---|")
    for r in results
        r === nothing && continue
        dr = r.spec.chan[r.axis]
        @printf(rep, "| %d | %.2f | %.2f | %.2f | %.1f | %.1f | %.1f | %.1f |\n",
                r.axis, r.A_bias, r.A_exc, r.spec.v_mean[r.axis] / VCAP[r.axis],
                dr.odd_db, dr.even_db, dr.tot_db, r.spec.stoch_db[r.axis])
    end
    println(rep, "\n(dB = 10*log10(distortion power / excited-bin power); more negative = more linear.)\n")
    println(rep, "## BLA gain (driven channel)\n")
    println(rep, "| axis | bias [V] | exc [V] | band-median \\|G\\| | \\|G(0.5 Hz)\\| |")
    println(rep, "|---|---|---|---|---|")
    for r in results
        r === nothing && continue
        g = abs.(r.spec.bla[r.axis])
        g05 = abs(r.spec.bla[r.axis][findfirst(==(5), r.spec.k_exc)])
        @printf(rep, "| %d | %.2f | %.2f | %.4f | %.4f |\n", r.axis, r.A_bias, r.A_exc,
                median(g), g05)
    end
    open(joinpath(OUTDIR, "tierB_report.md"), "w") do io
        write(io, String(take!(rep)))
    end
    println(String(take!(rep)))
    println("saved: $(joinpath(OUTDIR, "tierB_results.jld2")), tierB_report.md")
end

main()
