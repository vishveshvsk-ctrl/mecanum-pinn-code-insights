#!/usr/bin/env julia
# =============================================================================
# tierF_gamma_probe.jl — brute-force gamma (Eker-Nikolaou incremental norm,
# Elkhalil-Zribi 2-norm surrogate) over baseline closed-loop windows.
#
# DOMAIN: commanded-wrench. The real loop's plant input is the task wrench
# M (N m, controller R-scaled units): the mixer + motor_torque_inverse maps
# M -> wheel volts WITH back-EMF cancellation (hybrid_ctrl/plant.jl), so the
# actuator+plant operator N: M -> v has the linearization 1/(m_eff s + d_eff)
# that the IMC design model L uses. The earlier volts-domain formulation was
# wrong: raw-volts N includes back-EMF, making (N - L) mismatch ~4x at DC and
# gamma ~ 8 regardless of true nonlinearity.
#
#   gamma_hat = sup over windows/probes of || T(delta) - T(0) ||_2 / || delta ||_2
#   T(delta)  = V_N(M_ff + Q[delta]) - L * (M_ff + Q[delta])
#             -> increment = (V_N(M_ff+Q delta) - V_N(M_ff)) - lowpass_lambda[delta]
#               (since L*Q = 1/(1+lambda s) per axis, exactly)
#
# M_ff: reconstructed per sample from the recorded v_cmd + wheel speeds via
# the exact inverse map (motor_torque + the run_one un-map) — no null-space
# loss. Q = (m_eff s + d_eff)/(lambda(1+lambda s)) per axis (wrench domain,
# tuned lam_inner from the best archived PID-CT 0.9 config).
#
# Probes: random-phase multisines 0.1-3 / 3-10 / 10-30 Hz, steps, chirps,
# single-axis and mixed; amplitude ladder (small amplitudes double as the
# linearity check: they must converge to the analytic gamma_lin backstop).
# Probes driving any wheel toward voltage/current/slew limits are REJECTED.
#
# Sanity checks: restart fidelity (delta=0 replay vs recorded V); small-
# amplitude convergence to gamma_lin; docking hold expected near 1 (preslip
# DC gain 0 vs L*Q(0)=1).
#
# Usage: julia --project=. -t 6 nonlinearity_analysis/tierF_gamma_probe.jl \
#          [--smoke] [--traj NAME]
# Output: nonlinearity_analysis/results/gamma_probe[_smoke]/...
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, JSON, Statistics, LinearAlgebra, ForwardDiff, JLD2, Random
using StaticArrays

# whole controller/eval stack (PlantMod, run_one plant pieces, lim via
# physical_limits_with_margin). Include FIRST and ALONE (load-bearing order).
include(joinpath(ROOT, "hybrid_ctrl_v2", "eval_controllers_eskf_v3.jl"))
LinearAlgebra.BLAS.set_num_threads(1)   # threaded replays: deterministic FP

const SMOKE = "--smoke" in ARGS
const OUTDIR = joinpath(ROOT, "nonlinearity_analysis", "results",
                        SMOKE ? "gamma_probe_smoke" : "gamma_probe")
const BASE_DIR = joinpath(ROOT, "nonlinearity_analysis", "results", "gamma_baseline")
mkpath(OUTDIR)

# ---- plant point (campaign physics) ------------------------------------------
const PARAMS = Main.PlatformParams(Profiles.load_base("trajectory_files_run_0p5_main");
                                   mu_friction = 0.5)
const CHI    = 0.005
const P1, P2 = PARAMS.p1_case1, PARAMS.p2_case1
const MOTOR  = Main.PlantMod.MotorParams()
const LUGRE  = Main.LuGreParams()
const DTADV  = MOTOR.G * MOTOR.eta * MOTOR.Kt / MOTOR.Ra     # N m per V (wheel)
const J_W    = PARAMS.J_wheel

const LEVER  = PARAMS.R / (PARAMS.l + PARAMS.h)
# controller allocation  M(3) -> wheel torques(4)  (run_one.jl asmc_torques)
const W2T = SMatrix{4,3,Float64}(0.25 .* [ 1 -1 -LEVER;
                                           1  1  LEVER;
                                           1  1 -LEVER;
                                           1 -1  LEVER])
# inverse map wheel torques(4) -> task wrench(3)  (run_one.jl 708-710)
const T2W = SMatrix{3,4,Float64}([1.0  1.0  1.0  1.0;
                                 -1.0 1.0  1.0 -1.0;
                                 -1.0 1.0 -1.0 1.0] ./ [1.0, 1.0, LEVER])

const ROWS23 = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# IMC design model + Q (wrench domain), from tuned lambda
const LIM = Main.physical_limits_with_margin(0.9)
function tuned_lambda()
    paths = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3", "seed$s",
                      "pid_v2_ct_clean", "best_config.json") for s in 1:5]
    best = sort([(JSON.parsefile(c)["best_score"], c) for c in filter(isfile, paths)];
                by = first)[1][2]
    g = JSON.parsefile(best)["best_gains"]
    @printf("gamma probe: PID-CT config %s\n", best)
    return SVector(g["lam_inner_x"], g["lam_inner_y"], g["lam_inner_psi"])
end
const LAM = tuned_lambda()

"Q state per axis: (b1 s + b0)/(s + a0), wrench domain.
Q(s) = C/(I+L C) = (m_eff s + d_eff)/(1 + lambda s)  [NOT lambda(1+lambda s) —
an algebra slip here inflates every gain by 1/lambda ~ 10x]."
function q_coeffs(i::Int)
    m, d, lam = LIM.m_eff[i], LIM.d_eff[i], LAM[i]
    return (a0 = 1.0 / lam, b1 = m / lam, b0 = d / lam)
end

# ---- exact-plant residual + composite linearization ---------------------------
function mech_rows(v, omega, gamma, z, theta, Vcmd)
    bus = Main.BusMod.ControllerBus()
    bus.v_cmd = Vcmd
    plant_p = Main.PlantMod.PlantODEParams(PARAMS, CHI, P1, P2, :adamov, LUGRE, MOTOR, bus)
    T = promote_type(eltype(v), eltype(omega), eltype(gamma), eltype(z), eltype(theta))
    u = zeros(T, 30)
    u[1:3] = v; u[5:8] = theta; u[9:12] = omega; u[13:16] = gamma
    u[19:22] = z[1:4]; u[23:26] = z[5:8]; u[27:30] = z[9:12]
    du = zeros(T, 30)
    Main.PlantMod.plant_rhs!(du, u, plant_p, zero(T))
    return du[ROWS23]
end

"Composite wrench->velocity linearization at full state x, evaluated at the
ACTUAL operating voltage V_op (the recorded window-start v_cmd — freezing a
wrong voltage, e.g. a torque value, linearizes a fictitious plant: garbage
poles, cond(A) ~ 1e10). The composite folds in the replay loop's motor
inverse map (back-EMF compensation) via A_eff = A + Bv*Kc, B_eff = Bv*Km."
function lin_map(x, V_op)
    v = x[1:3]; omega = x[9:12]; gam = x[13:16]; z = x[19:30]; theta = x[5:8]
    fwrap = xx -> mech_rows(xx[1:3], xx[4:7], xx[8:11], xx[12:23],
                            SVector{4}(theta), SVector{4}(V_op))
    xx = vcat(v, omega, gam, z)
    A = ForwardDiff.jacobian(fwrap, xx)          # 23x23, d/d(mechanical state)
    Bv = zeros(23, 4)                            # d/d per-wheel volts
    for j in 1:4
        Bv[3 + j, j] = DTADV / J_W
    end
    Km = (MOTOR.Ra / (MOTOR.G * MOTOR.eta * MOTOR.Kt)) * Matrix(W2T)   # 4x3
    Kc = zeros(4, 23)                          # back-EMF feedback on wheel rows
    for j in 1:4
        Kc[j, 3 + j] = MOTOR.Kb * MOTOR.G
    end
    Aeff = A + Bv * Kc
    Beff = Bv * Km
    C3 = [Matrix{Float64}(I, 3, 3) zeros(3, 20)]
    return Aeff, Beff, C3
end

function gamma_lin(x, V_op, fgrid)
    A, B, C3 = lin_map(x, V_op)
    gmax, fstar = 0.0, first(fgrid)
    for f in fgrid
        w = 2π * f
        Gw = C3 * (((im * w) * I - A) \ B)           # wrench -> velocity, 3x3
        M = zeros(ComplexF64, 3, 3)
        for k in 1:3
            Lk = 1.0 / (im * w * LIM.m_eff[k] + LIM.d_eff[k])
            qc = q_coeffs(k)
            Qk = (qc.b1 * im * w + qc.b0) / (im * w + qc.a0)
            for i in 1:3
                M[i, k] = (Gw[i, k] - (i == k ? Lk : 0.0)) * Qk
            end
        end
        g = maximum(svdvals(M))
        if g > gmax
            gmax, fstar = g, f
        end
    end
    return (g = gmax, fstar = fstar)
end

# ---- window replay -------------------------------------------------------------
const FS_REC = 500.0
const FS_CB  = 1000.0

function abstol30()
    at = fill(1e-7, 30)
    at[1:3] .= 1e-8; at[5:8] .= 1e-8
    at[19:26] .= 1e-10
    return at
end

"Replay one window from x0 with M(t) = M_ff(t) + Q[delta](t); the per-tick
 voltage applies the exact motor inverse map (back-EMF cancellation) and the
 mixer's per-wheel rate limit, as the real loop does.
 Returns (t, V, stats): stats counts the ticks RIDING each limit (>98% of
 slew/voltage/current) — the saturation screen is counting-based, since the
 limits are enforced by construction inside the replay."
function replay(x0, M_ff_fun, delta_fun, W::Float64)
    qc = q_coeffs.(1:3)
    xq = zeros(3)
    d0 = delta_fun(0.0)
    M0 = M_ff_fun(0.0)
    M_now = [M0[k] + qc[k].b1 * d0[k] for k in 1:3]
    t_prev = 0.0
    V_prev = zeros(4)
    nslew = nvolt = ncurr = ntick = 0

    bus = Main.BusMod.ControllerBus()
    plant_p = Main.PlantMod.PlantODEParams(PARAMS, CHI, P1, P2, :adamov, LUGRE, MOTOR, bus)

    function tick!(integ)
        t = integ.t
        dt = t - t_prev
        if dt > 0
            for k in 1:3
                e = delta_fun(t)[k]
                xq[k] += dt * (-qc[k].a0 * xq[k] + e)
                M_now[k] = M_ff_fun(t)[k] + (qc[k].b0 - qc[k].b1 * qc[k].a0) * xq[k] + qc[k].b1 * e
            end
        end
        t_prev = t
        ntick += 1
        omw = integ.u[9:12]
        V_des = Main.PlantMod.motor_torque_inverse(W2T * SVector{3}(M_now), SVector{4}(omw), MOTOR)
        if dt > 0
            step = clamp.(Vector(V_des) .- V_prev, -MOTOR.dV_max * dt, MOTOR.dV_max * dt)
            V4 = SVector{4}(V_prev .+ step)
        else
            V4 = V_des
        end
        bus.v_cmd = V4
        i_now = (Vector(V4) .- MOTOR.Kb .* MOTOR.G .* omw) ./ MOTOR.Ra
        nvolt += maximum(abs.(Vector(V4))) > 0.98 * MOTOR.V_max ? 1 : 0
        ncurr += maximum(abs.(i_now)) > 0.98 * MOTOR.i_max ? 1 : 0
        if dt > 0 && maximum(abs.(Vector(V4) .- V_prev)) > 0.98 * MOTOR.dV_max * dt
            nslew += 1
        end
        V_prev .= V4
        return nothing
    end
    cb = PeriodicCallback(tick!, 1 / FS_CB; initial_affect = true)

    saveat = collect(0.0:1 / FS_REC:W)
    prob = ODEProblem(Main.PlantMod.plant_rhs!, copy(x0), (0.0, W), plant_p)
    sol = solve(prob, FBDF(); reltol = 1e-9, abstol = abstol30(),
                saveat = saveat, callback = cb, dtmax = 1e-3, maxiters = 10_000_000)
    # NOTE: sol.t is polluted by callback stop times — sample the dense
    # interpolant at exactly the saveat grid (same lesson as tierB).
    t = saveat
    V = reduce(hcat, [sol(tt)[1:3] for tt in saveat])
    stats = (nslew = nslew, nvolt = nvolt, ncurr = ncurr, ntick = ntick)
    return t, V, stats
end

# ---- probes ---------------------------------------------------------------------
struct Probe
    kind::Symbol
    freqs::Vector{Float64}
    phases::Vector{Float64}
    amp::Float64
    axes::Vector{Int}
    tc::Float64
end

function probe_signal(p::Probe, t)
    s = zeros(3)
    if p.kind == :multisine
        for (f, ph) in zip(p.freqs, p.phases)
            for k in p.axes
                s[k] += p.amp / sqrt(length(p.freqs)) * sin(2π * f * t + ph)
            end
        end
    elseif p.kind == :step
        for k in p.axes
            s[k] = t >= 0.1 ? p.amp : 0.0
        end
    elseif p.kind == :chirp
        f0, f1 = p.freqs
        for k in p.axes
            s[k] = p.amp * sin(2π * (f0 * t + (f1 - f0) * t^2 / (2 * p.tc)))
        end
    end
    return s
end

"Exact per-axis low-pass 1/(1+lambda s) on a sampled signal (L*Q = this)."
function lowpass_lam(sig, lam, dt)
    a = dt / (lam + dt)
    out = similar(sig)
    y = 0.0
    for n in eachindex(sig)
        y += a * (sig[n] - y)
        out[n] = y
    end
    return out
end

# ---- per-window gamma --------------------------------------------------------------
function window_gamma(traj, w, W, probes)
    x0    = Vector(traj.state[:, w.istart])
    Mseg  = traj.Mrec[:, w.istart:w.iend]          # 3 x M recorded wrench
    dt_rec = 1 / FS_REC
    M_ff_fun = let Mseg = Mseg, dt = dt_rec
        t -> begin
            x = clamp(t / dt + 1.0, 1.0, size(Mseg, 2) - 1.0)
            i0 = floor(Int, x); f = x - i0
            return Mseg[:, i0] .* (1 - f) .+ Mseg[:, i0 + 1] .* f
        end
    end

    t, V0, st0 = replay(x0, M_ff_fun, tt -> zeros(3), W)

    # restart fidelity vs recorded segment. NOTE: the gamma RATIO itself is
    # insensitive to replay-vs-record mismatch (Vd and V0 share x0 and the
    # replay machinery, so the error cancels in the difference); this is a
    # pure sanity diagnostic. Relative error is unstable on low-signal
    # windows (docking hold: v ~ 0), so report absolute error there.
    vrec = traj.state[1:3, w.istart:w.iend]
    n = min(size(V0, 2), size(vrec, 2))
    abs_err = sqrt(mean((V0[:, 1:n] .- vrec[:, 1:n]) .^ 2))
    rms_v = sqrt(mean(vrec[:, 1:n] .^ 2))
    fid = rms_v > 0.05 ? abs_err / rms_v : NaN     # NaN = low-signal window

    best = (ratio = 0.0, probe = nothing)
    results = NamedTuple[]
    for p in probes
        dfun = let p = p
            tt -> probe_signal(p, tt)
        end
        _, Vd, stp = replay(x0, M_ff_fun, dfun, W)
        # saturation screen (counting-based): reject the probe if it pushes
        # the actuator INTO a limit noticeably more than the baseline does
        ride(p) = p.nslew + p.nvolt + p.ncurr
        if ride(stp) > ride(st0) + 0.02 * st0.ntick
            push!(results, (kind = string(p.kind), amp = p.amp, ax = p.axes,
                            ratio = NaN, rejected = true))
            continue
        end
        m = min(size(Vd, 2), size(V0, 2))
        dtsig = reduce(hcat, [probe_signal(p, tt) for tt in t[1:m]])
        lq = zeros(3, m)
        for k in 1:3
            lq[k, :] = lowpass_lam(dtsig[k, :], LAM[k], 1 / FS_REC)
        end
        DT = (Vd[:, 1:m] .- V0[:, 1:m]) .- lq
        ratio = sqrt(mean(DT .^ 2)) / max(1e-12, sqrt(mean(dtsig .^ 2)))
        push!(results, (kind = string(p.kind), amp = p.amp, ax = p.axes,
                        ratio, rejected = false))
        if ratio > best.ratio
            best = (ratio = ratio, probe = p)
        end
    end
    return (window = w, fidelity = fid, stats0 = st0, best = best, results)
end

# ---- main ---------------------------------------------------------------------------
const W_WIN = 2.0
const FGRID = exp10.(range(log10(0.05), log10(30.0); length = 40))

"Reconstruct the commanded wrench per sample from recorded volts + wheel speeds."
function wrench_reconstruct(v_cmd, omega)
    N = size(v_cmd, 2)
    Mrec = zeros(3, N)
    for n in 1:N
        tau = Main.PlantMod.motor_torque(SVector{4}(v_cmd[:, n]),
                                         SVector{4}(omega[:, n]), MOTOR)
        Mrec[:, n] = T2W * tau
    end
    return Mrec
end

function load_traj(name)
    d = load(joinpath(BASE_DIR, name * ".jld2"))
    v_cmd = d["v_cmd"]
    om = d["state"][9:12, :]
    Mrec = wrench_reconstruct(v_cmd, om)
    i_now = (v_cmd .- MOTOR.Kb .* MOTOR.G .* om) ./ MOTOR.Ra
    # baseline exclusion: HARD plant actuator limits only (voltage/current).
    # The 200 V/s slew limiter lives in the MIXER (controller side), screened
    # per-probe inside the replay, not here.
    sat = (vec(maximum(abs.(v_cmd); dims = 1)) .> (MOTOR.V_max - 0.05)) .|
          (vec(maximum(abs.(i_now); dims = 1)) .> (MOTOR.i_max - 0.25))
    t = d["t"]
    Wsamp = round(Int, W_WIN * FS_REC)
    windows = NamedTuple[]
    istart = 1
    while istart + Wsamp <= length(t)
        iend = istart + Wsamp
        push!(windows, (istart = istart, iend = iend,
                        excluded = any(sat[istart:iend]), t0 = t[istart]))
        istart = iend
    end
    return (name = name, t = t, state = d["state"], Mrec = Mrec,
            v_cmd = v_cmd, windows = windows)
end

function make_probes(rng_seed = 7)
    rng = Random.Xoshiro(rng_seed)
    probes = Probe[]
    bands = [(0.1, 3.0), (3.0, 10.0), (10.0, 30.0)]
    amps = SMOKE ? [1e-3, 0.01, 0.05] : [1e-3, 0.005, 0.02, 0.06, 0.12]
    for (f0, f1) in bands
        nfr = 5
        fr = collect(range(f0, f1; length = 2 * nfr)[1:2:end])
        for amp in amps
            for axes in ([1], [2], [3], [1, 2, 3])
                ph = 2π .* rand(rng, length(fr))
                push!(probes, Probe(:multisine, fr, ph, amp, axes, 0.0))
            end
        end
    end
    push!(probes, Probe(:step, [0.0], [0.0], 0.03, [1, 2, 3], 0.0))
    push!(probes, Probe(:chirp, [0.1, 12.0], [0.0], 0.03, [1, 2, 3], W_WIN))
    return probes
end

function main_probe()
    names = SMOKE ? ["octagon_easy", "coupled_vomega_stress", "docking_step"] :
        replace.(filter(n -> endswith(n, ".jld2"), readdir(BASE_DIR)), ".jld2" => "")
    if "--traj" in ARGS
        names = [ARGS[findfirst(==("--traj"), ARGS) + 1]]
    end
    probes = make_probes()
    @printf("probe set: %d probes, window %.1f s, LAM=%s\n",
            length(probes), W_WIN, string(round.(LAM; sigdigits = 3)))
    flush(stdout)

    allrows = NamedTuple[]
    for name in names
        traj = load_traj(name)
        if SMOKE
            m = min(8, length(traj.windows))
            wins = occursin("docking", name) ? traj.windows[end - m + 1:end] :
                                               traj.windows[1:m]
        else
            wins = traj.windows
        end
        # per-window, threaded over windows (each replay is self-contained).
        # gamma_lin runs FIRST per window and contributes backstop-driven
        # single-sine probes at its argmax frequency f* (probes alone only
        # lower-bound a sup; f* closes the loop).
        rowbuf = Vector{Any}(nothing, length(wins))
        Threads.@threads for wi in eachindex(wins)
            w = wins[wi]
            gl = gamma_lin(Vector(traj.state[:, w.istart]),
                           Vector(traj.v_cmd[:, w.istart]), FGRID)
            fprobes = [Probe(:multisine, [gl.fstar], [0.0], amp, axes, 0.0)
                       for amp in (SMOKE ? [1e-3, 0.01, 0.05] : [1e-3, 0.005, 0.02, 0.06])
                       for axes in ([1], [2], [3], [1, 2, 3])]
            r = window_gamma(traj, w, W_WIN, vcat(probes, fprobes))
            ride0 = (r.stats0.nslew + r.stats0.nvolt + r.stats0.ncurr) / r.stats0.ntick
            excluded = w.excluded || r.stats0.nslew > 0.05 * r.stats0.ntick
            rowbuf[wi] = (row = (traj = name, t0 = w.t0, excluded = excluded,
                                 fidelity = r.fidelity, ride0 = ride0,
                                 gamma_hat = r.best.ratio, gamma_lin = gl.g,
                                 fstar = gl.fstar,
                                 flag = gl.g - r.best.ratio),
                          results = r.results)
        end
        for (wi, w) in enumerate(wins)
            rb = rowbuf[wi]
            push!(allrows, rb.row)
            tiny = [x for x in rb.results
                    if x.kind == "multisine" && x.amp <= 1e-3 && x.ax == [1] && !x.rejected]
            tinystr = isempty(tiny) ? "n/a" : @sprintf("%.4f", first(tiny).ratio)
            # backstop flag: gamma_lin is INFINITESIMAL — at low-slip windows it
            # captures micro-slip creep resonances (real at ~1e-5 m/s scales, far
            # below any probe amplitude) and over-prints. Only flag when the
            # linearized gain exceeds BOTH the tiny-probe ratio and 1.
            tinyr = isempty(tiny) ? rb.row.gamma_hat : first(tiny).ratio
            flag = rb.row.gamma_lin > 1.0 && (rb.row.gamma_lin - tinyr) > 0.2
            fidstr = isnan(rb.row.fidelity) ? "lowsig" : @sprintf("%.4f", rb.row.fidelity)
            @printf("%-24s win t=%6.1fs excl=%d fid=%7s ride=%.3f g_hat=%.4f g_lin=%.4f f*=%5.1fHz tiny=%s %s\n",
                    name, w.t0, w.excluded, fidstr, rb.row.ride0,
                    rb.row.gamma_hat, rb.row.gamma_lin, rb.row.fstar, tinystr,
                    flag ? "<-- PROBE HARDER" : "")
        end
        flush(stdout)
        jldsave(joinpath(OUTDIR, name * "_gamma.jld2");
                rows = filter(r -> r.traj == name, allrows),
                probes = [(k = string(p.kind), f = p.freqs, a = p.amp, ax = p.axes)
                          for p in probes])
    end

    ok = filter(r -> !r.excluded, allrows)
    good = filter(r -> !isnan(r.fidelity) && r.fidelity < 0.05, allrows)
    println("\n=== gamma smoke summary ===")
    if !isempty(ok)
        @printf("windows in E: %d/%d (excl %d)   max gamma_hat = %.4f\n",
                length(ok), length(allrows),
                count(r.excluded for r in allrows),
                maximum(r.gamma_hat for r in ok))
        @printf("max gamma_lin backstop = %.4f\n", maximum(r.gamma_lin for r in ok))
    end
    lows = count(r -> !r.excluded && isnan(r.fidelity), allrows)
    lows > 0 && println("low-signal windows (hold, fidelity n/a): ", lows)
    bad = filter(r -> !r.excluded && !isnan(r.fidelity) && r.fidelity >= 0.05, allrows)
    isempty(bad) || println("WARNING: poor restart fidelity in ", length(bad), " windows")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_probe()
end
