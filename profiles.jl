# =============================================================================
# Profiles.jl  —  velocity-tracking excitation reference library.
#
# Each builder returns a VelRef. The MIXED-degree velocity controller
# (asmc_torques_vel: degree-1 velocity surfaces on (Vx,Vy), degree-2 heading
# surface on ψ) tracks BOTH the body-frame velocity AND an explicit heading
# reference psi_des(t). Accordingly every VelRef now carries a heading getter
# `psi`, and the yaw-rate / yaw-accel feedforwards (Wz, al) are the FIRST and
# SECOND derivatives of that single psi closure (ForwardDiff) — so they are
# consistent by construction (Wz ≡ d(psi)/dt, al ≡ d²(psi)/dt²). This matters:
# the degree-2 yaw surface uses e_ψ = ψ - psi_des and ė_ψ = ψ̇ - Wz; if Wz and
# psi_des came from independent expressions the controller would fight itself.
#
# Usage:
#   include("profiles.jl"); using .Profiles
#   case = TOML.parsefile("configs/profiles/multisine.toml")
#   ref  = Profiles.build(case["profile"]["builder"], case["profile"]["params"])
#   # body velocity:   ref.Vx(t), ref.Vy(t),  feedforward accel ref.Ax(t), ref.Ay(t)
#   # heading channel: ref.psi(t),  ref.Wz(t) (=ψ̇_d),  ref.al(t) (=ψ̈_d)
#   # ref.tstops, ref.T_total          # pass to solve(...; tstops=ref.tstops)
#
# ALL profiles are VelRef now: under the mixed controller the heading is bounded
# by the degree-2 yaw surface, so even a circle is velocity-tracked (no drift,
# no PosRef needed). PosRef/_posref are kept defined but unused for a possible
# future profile that genuinely wants world-position tracking.
#
# Deps: FunctionWrappers, ForwardDiff  (add to the project's Project.toml).
# =============================================================================
module Profiles

using FunctionWrappers: FunctionWrapper
using ForwardDiff

export VelRef, PosRef, build, BUILDERS

const Getter = FunctionWrapper{Float64, Tuple{Float64}}   # t::Float64 -> Float64, concrete type

# VelRef now carries a heading getter `psi`. Wz and al are d/dt and d²/dt² of psi.
struct VelRef
    Vx::Getter;  Vy::Getter                   # body-frame translational velocity references
    psi::Getter                               # heading reference ψ_des(t)        (PRIMARY)
    Wz::Getter                                # yaw-rate feedforward  = d(psi)/dt
    Ax::Getter;  Ay::Getter                   # translational accel feedforward   = d/dt(Vx,Vy)
    al::Getter                                # yaw-accel feedforward = d²(psi)/dt²
    tstops::Vector{Float64}                   # phase-boundary times for the solver
    T_total::Float64
end

# Kept for a possible future world-position-tracked profile; UNUSED by the
# current registry (all builders return VelRef under the mixed controller).
struct PosRef
    xo::Getter;  yo::Getter;  psi::Getter     # global-frame position + heading references
    Vxo::Getter; Vyo::Getter; om::Getter      # global velocity + yaw rate (= d/dt)
    Axo::Getter; Ayo::Getter; al::Getter      # global acceleration + yaw accel (= d²/dt²)
    tstops::Vector{Float64}
    T_total::Float64
end

# ---- private helpers --------------------------------------------------------
_S(x)   = 10x^3 - 15x^4 + 6x^5                 # quintic smoothstep (S, S', S'' = 0 at 0 and 1)
_SI(x)  = 2.5x^4 - 3x^5 + x^6                  # ∫₀ˣ S  (used to integrate yaw rate → heading)
_sat(x) = min(max(x, 0.0), 1.0)
_deg(x) = x * (pi / 180)
# trapezoidal pulse: 0 →(ramp tu)→ 1 →(hold th)→ 1 →(ramp td)→ 0
_pulse(t, t0, tu, th, td) = _S(_sat((t - t0) / tu)) - _S(_sat((t - t0 - tu - th) / td))
# startup ramp 0 → 1 over tw, then holds
_ramp(t, t0, tw) = _S(_sat((t - t0) / tw))
# integral of a speed ramp: smooth ramp to unit slope (C²) — for orbit angle warp-in
_g(x) = _SI(_sat(x)) + max(x - 1.0, 0.0)
# heading displacement from one trapezoidal yaw-rate pulse = ∫ yawr·_pulse dt (closed form).
# Mirrors the old long_circle yaw_disp; verified against numeric integration.
function _yaw_disp(t, t0, tu, th, td, yawr)
    a = _sat((t - t0) / tu)
    c = _sat((t - t0 - tu) / th)
    d = _sat((t - t0 - tu - th) / td)
    return yawr * (tu * _SI(a) + th * c + td * (d - _SI(d)))
end
_req(cfg, k) = haskey(cfg, k) ? cfg[k] : error("Profiles: missing required key '$k'")

# Build a VelRef from translational-velocity closures (fVx,fVy) and a heading
# closure (fpsi). Accel feedforwards are ForwardDiff: Ax,Ay = d/dt(Vx,Vy);
# Wz = d/dt(psi); al = d²/dt²(psi). Raw closures (not wrapped Getters) are handed
# to ForwardDiff so they accept Dual t; wrappers fix Float64->Float64 for the loop.
function _velref(fVx, fVy, fpsi, tstops, T_total)
    dWz(t)  = ForwardDiff.derivative(fpsi, t)
    dal(t)  = ForwardDiff.derivative(dWz, t)          # nested ⇒ second derivative of psi
    VelRef(Getter(fVx), Getter(fVy),
           Getter(fpsi),
           Getter(dWz),
           Getter(t -> ForwardDiff.derivative(fVx, t)),
           Getter(t -> ForwardDiff.derivative(fVy, t)),
           Getter(dal),
           sort(collect(Float64, tstops)), Float64(T_total))
end

# Build a PosRef from three GLOBAL-frame position/heading closures (UNUSED; kept
# for a future world-position-tracked profile under the position controller).
function _posref(fxo, fyo, fpsi, tstops, T_total)
    d1(f) = t -> ForwardDiff.derivative(f, t)
    d2(f) = t -> ForwardDiff.derivative(d1(f), t)
    PosRef(Getter(fxo), Getter(fyo), Getter(fpsi),
           Getter(d1(fxo)), Getter(d1(fyo)), Getter(d1(fpsi)),
           Getter(d2(fxo)), Getter(d2(fyo)), Getter(d2(fpsi)),
           sort(collect(Float64, tstops)), Float64(T_total))
end

# =============================================================================
# OCTAGON — per-leg start-cruise-stop body velocity in n directions, with a
# confined lateral velocity wiggle during each cruise. Heading is held fixed
# (ψ_des ≡ θ0, Ω ≡ 0): the legs are body-frame velocity vectors, the platform
# does not rotate. The degree-2 yaw surface simply holds heading at θ0.
# =============================================================================
function build_octagon(cfg)::VelRef
    n   = get(cfg, "n_sides", 8)
    v   = get(cfg, "vcru", 0.8)
    th0 = _deg(get(cfg, "theta0_deg", 0.0))
    af  = get(cfg, "accel_frac", 0.25)
    df  = get(cfg, "dwell_frac", 0.10)
    cf  = 0.90 - 2 * af                       # cruise_frac is DERIVED: 2·af + cf + df = 1, df=0.10
    @assert df ≈ 0.10 "octagon: this builder assumes dwell_frac=0.10 (cruise_frac=0.90-2·accel_frac)"
    @assert cf > 0 "octagon: accel_frac too large — cruise_frac = 0.90-2·accel_frac ≤ 0"
    lA  = get(cfg, "lat_vamp", 0.25)          # lateral VELOCITY amplitude [m/s]
    Nw  = get(cfg, "n_waves", 2)
    Tp  = get(cfg, "T_period", 10.0)
    psid = _deg(get(cfg, "psi_des_deg", 0.0)) # held heading reference [rad]

    Ttot = 4.0 * Tp
    tau  = Ttot / n
    tu, tc, td = af * tau, cf * tau, af * tau
    leg  = tu + tc + td + df * tau
    th   = [th0 + (k - 1) * 2pi / n for k in 1:n]
    T0   = [(k - 1) * leg for k in 1:n]

    fwd(t, k) = v * _pulse(t, T0[k], tu, tc, td)
    function lat(t, k)
        c = _sat((t - T0[k] - tu) / tc)        # cruise progress 0..1
        lA * sin(pi * c)^3 * sin(2pi * Nw * c) # sin^3 envelope ⇒ vanishes (C²) at cruise edges
    end
    fVx(t)  = sum(fwd(t, k) * cos(th[k]) - lat(t, k) * sin(th[k]) for k in 1:n)
    fVy(t)  = sum(fwd(t, k) * sin(th[k]) + lat(t, k) * cos(th[k]) for k in 1:n)
    fpsi(t) = psid + zero(t)                   # Ω ≡ 0 (zero(t) keeps it ForwardDiff-safe)

    kinks = reduce(vcat, [[T0[k], T0[k] + tu, T0[k] + tu + tc, T0[k] + tu + tc + td] for k in 1:n])
    _velref(fVx, fVy, fpsi, kinks, Ttot)
end

# =============================================================================
# LONG_CIRCLE — VELOCITY-TRACKED heading-aligned circular orbit. The body drives
# straight ahead (Vx = R·ψ̇_des, Vy = 0) while the heading sweeps WITH the orbit
# tangent: ψ_des(t) = ψ0 + φ(t), with φ the orbit angle (speed warm-up over Twarp,
# then constant ψ̇ = worbit). Because the heading is the orbit angle and the body
# speed is R·ψ̇, world velocity R(ψ)(Vx,0) traces a clean circle — the degree-2
# yaw surface keeps ψ on φ, so the orbit is repeatable with no drift, NO position
# tracking required. The old independent yaw-rate pulse schedule is retired: the
# orbit itself supplies the heading content.
# =============================================================================
function build_long_circle(cfg)::VelRef
    R    = get(cfg, "R", 1.0)                 # orbit radius [m]
    worb = get(cfg, "worbit", 0.5)            # orbital rate [rad/s] ⇒ steady |V| = R·worbit
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))    # heading offset added to orbit angle [rad]
    Tw   = get(cfg, "Twarp", 3.0)             # orbit speed warm-up (0 → worbit)
    nlap = get(cfg, "n_laps", 2)              # number of full revolutions after warm-up

    # orbit angle φ(t): speed ramps 0 → worb over Tw (C² via _g), then constant.
    phi(t) = worb * Tw * _g(t / Tw)
    # n_laps = STEADY revolutions AFTER warm-up (may be non-integer). The steady
    # phase sweeps exactly 2π·n_laps of heading regardless of |worbit| or Twarp;
    # the warm-up arc (|worb|·Tw/2) is additional. Durations from the MAGNITUDE —
    # worb may be negative (CW orbit, driven in reverse since Vx = R·φ̇ < 0).
    #   Ttot = Tw + 2π·n_laps/|worb|  ⇒  total heading sweep = |worb|·Tw/2 + 2π·n_laps.
    wa = abs(worb)
    wa > 0 || error("long_circle: worbit must be nonzero")
    nlap > 0 || error("long_circle: n_laps must be positive")
    Ttot = Tw + 2pi * nlap / wa

    # heading is the orbit tangent angle (PRIMARY); Wz = φ̇, al = φ̈ via ForwardDiff.
    fpsi(t) = psi0 + phi(t)
    # body velocity: drive straight ahead at the instantaneous orbit speed R·φ̇.
    # Vx tied to ψ_des through R (Vx = R·Wz), Vy ≡ 0. Consistent by construction.
    fVx(t)  = R * ForwardDiff.derivative(phi, t)
    fVy(t)  = zero(t)

    kinks = [Tw]
    _velref(fVx, fVy, fpsi, kinks, Ttot)
end

# =============================================================================
# SPIN_CREEP — small constant creep translation + scheduled HIGH yaw-rate
# pulses. Low |V| frees the wheel-speed budget for large |Ω|. Heading ψ_des is
# the running integral of the yaw-rate pulse schedule (closed form via _yaw_disp),
# so Wz recovers the pulse train exactly and al its derivative.
# =============================================================================
function build_spin_creep(cfg)::VelRef
    vc   = get(cfg, "v_creep", 0.05)
    dc   = _deg(get(cfg, "delta_creep_deg", 0.0))
    yaws = Vector{Float64}(get(cfg, "yaw_spin", [3.1, -3.1]))   # default: 2-pulse alternating
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))
    Tw   = get(cfg, "Twarp", 2.0)
    Tpre = get(cfg, "T_pre", 3.0)
    tu = get(cfg, "tu", 1.5); th = get(cfg, "th", 4.0)
    td = get(cfg, "td", tu); tdw = get(cfg, "tdwell", 2.0)   # td defaults to tu (symmetric ramp)
    @assert Tpre >= Tw "spin_creep: T_pre must be ≥ Twarp"

    np   = length(yaws)
    Tsc  = [Tpre + (k - 1) * (tu + th + td + tdw) for k in 1:np]
    Ttot = Tsc[end] + (tu + th + td + tdw)

    env(t)  = _ramp(t, 0.0, Tw)
    fVx(t)  = vc * cos(dc) * env(t)
    fVy(t)  = vc * sin(dc) * env(t)
    # heading = ψ0 + Σ ∫ yaw-rate pulses  (PRIMARY); Wz = Σ pulses, al = d/dt via ForwardDiff
    fpsi(t) = psi0 + sum(_yaw_disp(t, Tsc[k], tu, th, td, yaws[k]) for k in 1:np)

    kinks = vcat(Tw, reduce(vcat, [[Tsc[k], Tsc[k] + tu, Tsc[k] + tu + th, Tsc[k] + tu + th + td]
                                    for k in 1:np]))
    _velref(fVx, fVy, fpsi, kinks, Ttot)
end

# =============================================================================
# SPIRAL — independent |V| and Ω profiles, each either "const" (held, on/off
# ramped at the trajectory edges) or "ramp" (a single bump in the middle).
#   kind 1: V_mode="ramp",  Om_mode="const"   (sweep |V|, low const Ω)
#   kind 2: V_mode="const", Om_mode="ramp"    (const |V|, sweep Ω)
#   kind 3: V_mode="ramp",  Om_mode="ramp"    (synchronized — rides a (V,Ω) ray)
# Heading ψ_des is the analytic integral of the Ω profile (PRIMARY); Wz recovers
# Ω(t), al its derivative.
# =============================================================================
function build_coupled_vomega(cfg)::VelRef
    d   = _deg(get(cfg, "delta_deg", 0.0))
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))
    Tw  = get(cfg, "Twarp", 2.0)
    t0  = get(cfg, "bump_t0", 4.0)
    tu  = get(cfg, "bump_tu", 4.0); th = get(cfg, "bump_th", 6.0); td = get(cfg, "bump_td", 4.0)
    Ttot = get(cfg, "T_total", t0 + tu + th + td + 4.0)
    Vmode = get(cfg, "V_mode", "ramp");  Vc = get(cfg, "V_const", 0.5);  Vp = get(cfg, "V_peak", 0.8)
    Omode = get(cfg, "Om_mode", "const"); Oc = get(cfg, "Om_const", 0.2); Op = get(cfg, "Om_peak", 0.8)

    # ---- minimum total heading sweep (e.g. 540°): stretch HOLDS to reach it. ----
    # Stretching holds leaves every ramp rate — hence every validated acceleration
    # — untouched, so the TOML's friction-circle pre-validation stays in force.
    #   "const": H = |Oc|·(Ttot − Tw)            → extend Ttot
    #   "ramp" : H = |Op|·(th + (tu+td)/2)        → extend the bump hold th
    # (if V_mode == "ramp" too, the shared th stretches the V hold equally: longer
    # cruise at V_peak, zero new acceleration; if V_mode == "const" the flattop
    # follows Ttot automatically.)
    Hmin = _deg(get(cfg, "min_heading_deg", 0.0))
    if Hmin > 0
        if Omode == "ramp"
            abs(Op) > 0 || error("coupled_vomega: min_heading_deg requires Om_peak ≠ 0")
            H0 = abs(Op) * (th + 0.5 * (tu + td))
            if H0 < Hmin
                dth = (Hmin - H0) / abs(Op)
                th += dth;  Ttot += dth
            end
        else
            abs(Oc) > 0 || error("coupled_vomega: min_heading_deg requires Om_const ≠ 0")
            H0 = abs(Oc) * (Ttot - Tw)
            H0 < Hmin && (Ttot = Tw + Hmin / abs(Oc))
        end
    end

    flattop(t) = _S(_sat(t / Tw)) * _S(_sat((Ttot - t) / Tw))   # held value, on/off ramps at edges
    bump(t)    = _pulse(t, t0, tu, th, td)                      # single 0→peak→0 bump
    Vmag(t) = Vmode == "ramp" ? Vp * bump(t) : Vc * flattop(t)
    fVx(t)  = Vmag(t) * cos(d)
    fVy(t)  = Vmag(t) * sin(d)

    # ψ_des = ψ0 + ∫₀ᵗ Ω dt (PRIMARY); Wz = d(ψ_des)/dt recovers Ω(t) exactly (FTC).
    # Two closed-form cases for the integral, both ForwardDiff-clean:
    #   "ramp":  Ω = Op·_pulse(t)        ⇒ ∫ = Op·_yaw_disp(t, t0,tu,th,td, 1.0)
    #   "const": Ω = Oc·flattop(t)       ⇒ ∫ = Oc·(on-ramp + linear middle + off-ramp),
    #            each piece the analytic _SI-based integral of the quintic flattop.
    if Omode == "ramp"
        fpsi_r(t) = psi0 + Op * _yaw_disp(t, t0, tu, th, td, 1.0)
        kinks = [Tw, Ttot - Tw, t0, t0 + tu, t0 + tu + th, t0 + tu + th + td]
        return _velref(fVx, fVy, fpsi_r, kinks, Ttot)
    else
        # const Ω with C² on/off ramps: Ω(t)=Oc·flattop(t).
        # ∫₀ᵗ flattop = [on-ramp: Tw·_SI(_sat(t/Tw))] + [middle: linear] + [off-ramp].
        # Piecewise closed form (Tw on-ramp at start, Tw off-ramp before Ttot):
        function fpsi_c(t)
            on  = Tw * _SI(_sat(t / Tw))                              # ∫ of rising _S over [0,Tw]
            mid = max(0.0, min(t, Ttot - Tw) - Tw)                    # linear unit-height middle
            # off-ramp ∫_{Ttot-Tw}^{t} _S(_sat((Ttot-τ)/Tw)) dτ = Tw·(0.5 − _SI(_sat((Ttot-t)/Tw)))
            off = t <= (Ttot - Tw) ? 0.0 : Tw * (0.5 - _SI(_sat((Ttot - t) / Tw)))
            return psi0 + Oc * (on + mid + off)
        end
        kinks = [Tw, Ttot - Tw, t0, t0 + tu, t0 + tu + th, t0 + tu + th + td]
        return _velref(fVx, fVy, fpsi_c, kinks, Ttot)
    end
end

# =============================================================================
# MULTISINE — zero-mean harmonic sums on Vx, Vy (and optionally Ω). Frequencies
# are integer harmonics of f0 = 1/T_fund ⇒ periodic (bounded CoM, no leakage).
# Amplitudes ∝ 1/f^shape (shape=1 ⇒ ~flat accel); scaled to target peak |V|/|Ω|.
# Heading ψ_des is PRIMARY: the closed-form antiderivative of the yaw rate, with the
# t=0 rate offset removed by a C² decay so Ω(0)=0 (see Om0/Rdecay below). Wz =
# d(ψ_des)/dt then recovers Ω(t) exactly (FTC), al = d²/dt² its angular accel.
# NOTE: per the agreed simplification, the Vx/Vy are envelope-ramped but the YAW
# integral uses the unenveloped multisine (minus the start offset) — exact except in
# the short Tw window where Ω is small. multisine heading is not load-bearing.
# =============================================================================
function build_multisine(cfg)::VelRef
    Tf   = get(cfg, "T_fund", 20.0); nper = get(cfg, "n_periods", 3)
    flo  = get(cfg, "f_lo", 0.1);    fhi  = get(cfg, "f_hi", 3.0); shp = get(cfg, "shape", 1.0)
    pyseed = get(cfg, "pyseed", 0);  Vpk = get(cfg, "Vpk", 0.5);  Ompk = get(cfg, "Ompk", 0.6)  # pyseed: label only (phases are explicit)
    yaw  = get(cfg, "excite_yaw", true); Tw = get(cfg, "Twarp", 2.0)
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))

    f0   = 1.0 / Tf
    kmin = max(1, ceil(Int, flo / f0)); kmax = floor(Int, fhi / f0)

    # Per-channel harmonic combs. ZIPPERED design (kx/ky/kw integer index columns
    # in the TOML): the three channels get DISJOINT interleaved combs (round-robin
    # k mod 3), exactly orthogonal over T_fund — within-run decorrelated joint
    # (Vx,Vy,Ω) coverage, exact channel attribution per excited line, and nonlinear
    # cross-coupling readable on each channel's unexcited lines.
    # LEGACY (kx/ky/kw absent): all channels share the full grid kmin:kmax —
    # pre-zipper TOMLs keep building exactly as validated.
    kx = Vector{Int}(get(cfg, "kx", Int[]))
    ky = Vector{Int}(get(cfg, "ky", Int[]))
    kw = Vector{Int}(get(cfg, "kw", Int[]))
    if isempty(kx)
        kx = collect(kmin:kmax);  ky = copy(kx);  kw = copy(kx)
    end
    (length(kx) >= 2 && length(ky) >= 2 && length(kw) >= 2) ||
        error("multisine: each channel comb needs ≥2 harmonics — widen band or lengthen T_fund")
    fx = f0 .* kx;  fy = f0 .* ky;  fw = f0 .* kw
    wxk = 2pi .* fx; wyk = 2pi .* fy; wwk = 2pi .* fw

    px = Vector{Float64}(get(cfg, "px", Float64[]))
    py = Vector{Float64}(get(cfg, "py", Float64[]))
    pw = Vector{Float64}(get(cfg, "pw", Float64[]))
    (length(px) == length(kx) && length(py) == length(ky) && length(pw) == length(kw)) ||
        error("multisine: phase vectors must match comb lengths " *
              "(px $(length(px))/$(length(kx)), py $(length(py))/$(length(ky)), pw $(length(pw))/$(length(kw))); " *
              "check kx/ky/kw and f_lo/f_hi/T_fund against the generator.")
    ampx = 1.0 ./ (fx .^ shp);  ampy = 1.0 ./ (fy .^ shp);  ampw = 1.0 ./ (fw .^ shp)

    tg  = range(0, Tf; length = 4000)
    vx0(t) = sum(ampx[k] * sin(wxk[k] * t + px[k]) for k in eachindex(kx))
    vy0(t) = sum(ampy[k] * sin(wyk[k] * t + py[k]) for k in eachindex(ky))
    wz0(t) = sum(ampw[k] * sin(wwk[k] * t + pw[k]) for k in eachindex(kw))
    sV = Vpk / maximum(sqrt(vx0(t)^2 + vy0(t)^2) for t in tg)
    sW = yaw ? Ompk / maximum(abs(wz0(t)) for t in tg) : 0.0
    ampx .*= sV; ampy .*= sV; ampw .*= sW

    Ttot   = nper * Tf
    env(t) = _S(_sat(t / Tw)) * _S(_sat((Ttot - t) / Tw))
    fVx(t) = env(t) * sum(ampx[k] * sin(wxk[k] * t + px[k]) for k in eachindex(kx))
    fVy(t) = env(t) * sum(ampy[k] * sin(wyk[k] * t + py[k]) for k in eachindex(ky))

    # Heading is PRIMARY and must be a genuine antiderivative of the yaw rate we
    # want, so that Wz = d(ψ_des)/dt is EXACTLY that rate (consistency). The raw
    # multisine Ω_ms(t)=Σ ampw·sin(wk t+pw) has Ω_ms(0)=Σ ampw·sin(pw) ≠ 0 — a
    # yaw-rate step at t=0. We remove it with a smooth C² decay so the commanded
    # yaw rate starts at zero, and integrate the whole thing in closed form:
    #   Ω(t)   = Ω_ms(t) − Ω0·r(t),     r(t) = 1 − _S(_sat(t/Tw))   (1→0, C²)
    #   ψ_des  = ψ0 + ∫₀ᵗ Ω = ψ0 + [Σ ampw(cos pw − cos(wk t+pw))/wk] − Ω0·R(t)
    # where R(t)=∫₀ᵗ r = t − Tw·_SI(_sat(t/Tw)) on [0,Tw], then constant (Tw·(1−0.5))
    # afterwards. ForwardDiff of ψ_des returns Ω(t) exactly (FTC), and Ω(0)=0.
    Om0  = yaw ? sum(ampw[k] * sin(pw[k]) for k in eachindex(kw)) : 0.0
    psi_ms(t)  = sum(ampw[k] * (cos(pw[k]) - cos(wwk[k] * t + pw[k])) / wwk[k] for k in eachindex(kw))
    Rdecay(t)  = t <= Tw ? (t - Tw * _SI(_sat(t / Tw))) : (Tw * 0.5)   # ∫₀ᵗ r, C¹, flat after Tw
    fpsi(t) = psi0 + psi_ms(t) - Om0 * Rdecay(t)

    _velref(fVx, fVy, fpsi, [Tw, Ttot - Tw], Ttot)
end

# =============================================================================
# ELLIPSE — POSITION-TRACKED orbit (the PosRef profile; runs under the pure
# degree-2 position controller asmc_torques, keeping that path alive).
# World path: P(φ) = Rot(θe)·[(a·cosφ − a), b·sinφ] — starts AT THE ORIGIN
# (major-axis end), matching the zero position IC. Orbit parameter φ ramps
# 0 → worbit over Twarp (C² via _g), then runs at constant rate; unlike a
# circle, constant φ̇ still gives time-varying speed AND curvature — that
# modulation is the excitation an ellipse adds over long_circle.
#
# Two heading modes (combo column psi_mode):
#   "tangent" — heading sweeps with the path tangent (forward-drive-rich).
#     The tangent angle is computed in UNWRAPPED closed form: with
#     u = φ + π/2 and k = (b−a)/(b+a),
#         θ_t(u) = u + atan( k·sin(2u) / (1 − k·cos(2u)) )
#     (derived from tan(θ−u) = (r−1)·sinu·cosu/(cos²u + r·sin²u), r = b/a;
#     verified to machine precision against unwrapped atan2 over multiple
#     laps down to ratio 0.05). |k| < 1 ⇒ the denominator stays positive, so
#     the correction term is smooth, periodic, and single-valued: NO ±π wrap
#     across laps, which a raw atan2 would inject straight into the degree-2
#     yaw surface. θ_t is strictly increasing (θ_t' = r/(cos²u + r²sin²u) > 0).
#   "fixed"   — crab ellipse: heading held at psi0 while the platform traces
#     the orbit by strafing (gated by the strafe cap in the TOML generator).
# =============================================================================
function build_ellipse(cfg)::PosRef
    a    = _req(cfg, "a")                            # semi-major axis [m]
    rho  = _req(cfg, "ratio");  b = rho * a          # b/a ∈ (0,1]
    worb = _req(cfg, "worbit")                       # signed orbital rate [rad/s]
    nlap = get(cfg, "n_laps", 1)
    Tw   = get(cfg, "Twarp", 3.0)
    the  = _deg(get(cfg, "theta_e_deg", 0.0))        # ellipse orientation in world
    mode = get(cfg, "psi_mode", "tangent")           # "tangent" | "fixed"
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))           # heading for "fixed" mode

    (0.0 < rho <= 1.0) || error("ellipse: ratio must be in (0,1], got $rho")
    wa = abs(worb); wa > 0 || error("ellipse: worbit must be nonzero")

    # orbit parameter: rate ramps 0 → worb over Tw (C²), then constant
    phi(t) = worb * Tw * _g(t / Tw)
    ang_warm = wa * Tw * 0.5
    Ttot = Tw + (2pi * nlap - ang_warm) / wa

    cth, sth = cos(the), sin(the)
    fxo(t) = (p = phi(t);  cth * (a * cos(p) - a) - sth * (b * sin(p)))
    fyo(t) = (p = phi(t);  sth * (a * cos(p) - a) + cth * (b * sin(p)))

    if mode == "tangent"
        k    = (b - a) / (b + a)
        flip = worb < 0 ? pi : 0.0                   # face the direction of travel
        fpsi_t(t) = (u = phi(t) + pi/2;
                     the + flip + u + atan(k * sin(2u) / (1 - k * cos(2u))))
        return _posref(fxo, fyo, fpsi_t, [Tw], Ttot)
    elseif mode == "fixed"
        fpsi_f(t) = psi0 + zero(t)                   # zero(t) keeps it Dual-safe
        return _posref(fxo, fyo, fpsi_f, [Tw], Ttot)
    else
        error("ellipse: psi_mode must be \"tangent\" or \"fixed\", got \"$mode\"")
    end
end

# =============================================================================
# SPIRAL_ORBIT — heading-locked geometric spiral (VelRef): ψ = ψ0 + ∫Ω,
# Vx = |Ω|·R(t) (ALWAYS forward-driving; chirality enters only through Ω's
# sign), Vy ≡ 0. R(t) sweeps, so the path curls through ≥ min_heading_deg of
# heading (540° default ⇒ ≥1.5 loops). Three modes (combos column `mode`):
#
#   "om_const" — Ω held; R(t) C²-sweeps R0→R1 ⇒ V = |Ω|·R(t) sweeps.
#       The V-marginal probe at frozen yaw rate. Creep-approach rows are this
#       mode with small |Ω| and small inward terminal R.
#   "v_const"  — V held at Vc; Ω(t) C²-sweeps Om0→Om1 (same sign) ⇒ R = V/Ω
#       sweeps implicitly. The Ω-marginal probe at frozen speed. NOTE: Ω is
#       the primary specified function (not R) so ψ stays closed-form.
#   "iso_accel"— centripetal utilization held: u* = |Ω|·V const while Ω sweeps
#       Om0→Om1 ⇒ V = u*/|Ω|, R = u*/Ω². Sustained near-boundary operation
#       with (V,Ω)-composition sweep — unique in the profile set.
#
# Timing: Ω ramps in over Twarp on the initial circle (R0 = u*/Om0² for iso);
# the sweep runs Tsw (computed below); a Ttail steady tail settles the final
# circle. Tsw is DERIVED per row (not a TOML column): the larger of the
# min_heading_deg floor and the pacing floors
#   om_const: a_tan = |Ω|·Ṙ ≤ a_tan_cap  ⇒ Tsw ≥ 1.875·|Ω|·|ΔR|/a_tan_cap
#   sweeps:   α = Ω̇    ≤ alpha_cap      ⇒ Tsw ≥ 1.875·|ΔΩ|/alpha_cap
# Pacing caps are gentleness knobs; SAFETY is certified independently by the
# master per-wheel gate in gen_spiral_orbit_toml.py (which replicates these
# exact kinematics — keep the two in sync if formulas change).
#
# Heading integral (closed form): with _g(x) = _SI(_sat(x)) + max(x−1,0) and
# the identity _g′(x) = _S(_sat(x)),
#   Ω(t)  = Om0·_S(_sat(t/Tw)) + ΔΩ·_S(_sat((t−Tw)/Tsw))
#   ψ(t)  = ψ0 + Om0·Tw·_g(t/Tw) + ΔΩ·Tsw·_g((t−Tw)/Tsw)
# differentiate to verify: ψ̇ = Ω exactly. om_const is the ΔΩ = 0 case.
# =============================================================================
function build_spiral_orbit(cfg)::VelRef
    mode  = get(cfg, "mode", "om_const")
    Tw    = get(cfg, "Twarp", 3.0)
    Ttail = get(cfg, "Ttail", 2.0)
    Hmin  = _deg(get(cfg, "min_heading_deg", 540.0))
    atan_cap  = get(cfg, "a_tan_cap", 1.5)
    alpha_cap = get(cfg, "alpha_cap", 0.5)
    psi0  = _deg(get(cfg, "psi0_deg", 0.0))

    if mode == "om_const"
        Om = _req(cfg, "Om");  R0 = _req(cfg, "R0");  R1 = _req(cfg, "R1")
        wa = abs(Om);  wa > 0 || error("spiral_orbit: Om must be nonzero")
        (R0 > 0 && R1 > 0) || error("spiral_orbit: radii must be positive")
        Tsw = max(1.875 * wa * abs(R1 - R0) / atan_cap,
                  Hmin / wa - Tw / 2 - Ttail, 4.0)
        Ttot = Tw + Tsw + Ttail
        Rfun(t)   = R0 + (R1 - R0) * _S(_sat((t - Tw) / Tsw))
        fVx_o(t)  = wa * _S(_sat(t / Tw)) * Rfun(t)       # forward: |Ω|·R(t)
        fVy_o(t)  = zero(t)
        fpsi_o(t) = psi0 + Om * Tw * _g(t / Tw)           # ψ̇ = Om·_S ramp → const
        return _velref(fVx_o, fVy_o, fpsi_o, [Tw, Tw + Tsw], Ttot)
    elseif mode == "v_const" || mode == "iso_accel"
        Om0 = _req(cfg, "Om0");  Om1 = _req(cfg, "Om1")
        (abs(Om0) > 0 && sign(Om0) == sign(Om1)) ||
            error("spiral_orbit: Om0, Om1 must be nonzero and same-signed (no Ω zero-crossing)")
        s   = sign(Om0)
        dOm = Om1 - Om0
        Tsw = max(1.875 * abs(dOm) / alpha_cap,
                  (Hmin - abs(Om0) * Tw / 2 - abs(Om1) * Ttail) / (0.5 * abs(Om0 + Om1)),
                  4.0)
        Ttot = Tw + Tsw + Ttail
        omega(t)  = Om0 * _S(_sat(t / Tw)) + dOm * _S(_sat((t - Tw) / Tsw))
        fpsi_s(t) = psi0 + Om0 * Tw * _g(t / Tw) + dOm * Tsw * _g((t - Tw) / Tsw)
        fVy_s(t)  = zero(t)
        if mode == "v_const"
            Vc = _req(cfg, "Vc");  Vc > 0 || error("spiral_orbit: Vc must be positive")
            fVx_v(t) = Vc * _S(_sat(t / Tw))
            return _velref(fVx_v, fVy_s, fpsi_s, [Tw, Tw + Tsw], Ttot)
        else
            u = _req(cfg, "ustar");  u > 0 || error("spiral_orbit: ustar must be positive")
            R0i = u / Om0^2                               # initial circle radius
            # warm-up rides the initial circle (utilization ramps 0→u* with Ω²);
            # after Tw, V = u*/|Ω| holds u* exactly. C² at the joint: Ω̇(Tw±)=0.
            fVx_i(t) = t <= Tw ? s * omega(t) * R0i : u / (s * omega(t))
            return _velref(fVx_i, fVy_s, fpsi_s, [Tw, Tw + Tsw], Ttot)
        end
    else
        error("spiral_orbit: mode must be \"om_const\", \"v_const\", or \"iso_accel\", got \"$mode\"")
    end
end

# =============================================================================
# STRAIGHTLINE — pure heading-FIXED translation along a body-frame ray at angle
# β to the heading axis (no yaw: ψ_des ≡ ψ0 ⇒ Ω = al = 0). Two kinds:
#   "const_vel": hold |V| = V_start (= V_end) for sim_time after warm-up.
#   "const_acc": accelerate |V| from V_start to V_end at constant a; the duration
#                is DERIVED, Tsim = (V_end − V_start)/a (V_end reached at the end).
# A quintic warm-up ramps |V| from 0 → V_start over Twarp and is ADDED before the
# stated sim_time (T_total = Twarp + Tsim), mirroring the other profiles' on-ramp.
# β = 0 ⇒ pure forward Vx; β = 90° ⇒ pure lateral Vy; β = 45° ⇒ diagonal — so the
# grid maps the per-direction feasibility envelope (forward vs lateral mobility).
# =============================================================================
function build_straightline(cfg)::VelRef
    beta = _deg(_req(cfg, "beta_deg"))
    psi0 = _deg(get(cfg, "psi0_deg", 0.0))
    Tw   = Float64(get(cfg, "Twarp", 3.0))
    kind = String(get(cfg, "kind", "const_vel"))
    V0   = Float64(_req(cfg, "V_start"))
    Ve   = Float64(_req(cfg, "V_end"))
    Tw > 0 || error("straightline: Twarp must be > 0")

    if kind == "const_vel"
        isapprox(V0, Ve; atol = 1e-9) ||
            error("straightline const_vel: V_start must equal V_end ($V0 ≠ $Ve)")
        a, Tsim = 0.0, Float64(get(cfg, "sim_time", 10.0))
    elseif kind == "const_acc"
        a = Float64(_req(cfg, "accel"))
        a > 0 || error("straightline const_acc: accel must be > 0")
        Ve > V0 || error("straightline const_acc: V_end ($Ve) must exceed V_start ($V0)")
        Tsim = (Ve - V0) / a
    else
        error("straightline: unknown kind '$kind' (use \"const_vel\" | \"const_acc\")")
    end
    Ttot = Tw + Tsim

    # |V|(t): quintic warm-up 0 → V_start over Tw (S(sat) stays 1 afterwards, so it
    # HOLDS V_start), then a constant-a linear ramp V_start → V_end. const_vel sets
    # a = 0, so |V| simply holds V_start. Kink (accel step) at t = Tw ⇒ one tstop.
    Vmag(t) = V0 * _S(_sat(t / Tw)) + (t <= Tw ? zero(t) : a * (t - Tw))
    fVx(t)  = Vmag(t) * cos(beta)
    fVy(t)  = Vmag(t) * sin(beta)
    fpsi(t) = psi0 + zero(t)            # heading held ⇒ Wz = al = 0

    _velref(fVx, fVy, fpsi, [Tw], Ttot)
end

# =============================================================================
# Registry + dispatch
# =============================================================================
const BUILDERS = Dict{String, Function}(
    "octagon"     => build_octagon,
    "long_circle" => build_long_circle,
    "spin_creep"  => build_spin_creep,
    "coupled_vomega" => build_coupled_vomega,   # ex-"spiral": coupled V–Ω excitation
    "spiral_orbit"   => build_spiral_orbit,     # heading-locked geometric spiral
    "multisine"   => build_multisine,
    "ellipse"     => build_ellipse,
    "straightline"   => build_straightline,     # heading-fixed β-ray (feasibility probe)
)

function build(name::AbstractString, cfg::AbstractDict)
    haskey(BUILDERS, name) ||
        error("Profiles: unknown profile '$name'; known: $(join(sort(collect(keys(BUILDERS))), ", "))")
    BUILDERS[name](cfg)
end

# =============================================================================
# CURRENT_REF — the single reference the controller reads.
#
# Thread-safety contract (same as the old Symbolics-getter design): the driver
# sets the reference ONCE per trajectory, single-threaded, BEFORE spinning up
# the inner (mu, chi) parallel loop. All inner threads only READ it. Never call
# set_reference! from inside a threaded region.
# =============================================================================
export set_reference!, current_ref, set_pos_reference!, current_posref,
       publish!, active_ref, is_velref, global_to_local_frame

const CURRENT_REF  = Ref{VelRef}()
const ACTIVE_KIND  = Ref{Symbol}(:none)     # :vel | :pos | :none

"""Publish `ref` as the active trajectory. Call single-threaded, per trajectory."""
set_reference!(ref::VelRef) = (CURRENT_REF[] = ref; ACTIVE_KIND[] = :vel; ref)

"""The active VelRef. Errors with guidance if no trajectory has been built yet."""
@inline function current_ref()::VelRef
    isassigned(CURRENT_REF) ||
        error("Profiles.current_ref(): no reference set — call set_reference!/" *
              "build_reference! (or pick_and_build) before solving.")
    return CURRENT_REF[]
end

# --- Parallel slot for the position controller (asmc_torques). --------------
# PosRef profiles (e.g. "ellipse") publish here; asmc_torques reads
# current_posref() the same way asmc_torques_vel reads current_ref().
const CURRENT_POSREF = Ref{PosRef}()

set_pos_reference!(ref::PosRef) = (CURRENT_POSREF[] = ref; ACTIVE_KIND[] = :pos; ref)

@inline function current_posref()::PosRef
    isassigned(CURRENT_POSREF) ||
        error("Profiles.current_posref(): no position reference set — build a " *
              "PosRef profile (e.g. ellipse) and publish it first.")
    return CURRENT_POSREF[]
end

"""Type-dispatching publish — works for both reference kinds."""
publish!(ref::VelRef) = set_reference!(ref)
publish!(ref::PosRef) = set_pos_reference!(ref)

"""Whichever reference was published last (VelRef or PosRef). Both kinds share
the fields the solver scaffolding needs: psi, tstops, T_total."""
function active_ref()
    ACTIVE_KIND[] === :vel && return current_ref()
    ACTIVE_KIND[] === :pos && return current_posref()
    error("Profiles.active_ref(): no trajectory published yet")
end

"""true ⇒ active trajectory is a VelRef (mixed velocity controller);
false ⇒ PosRef (position controller asmc_torques)."""
is_velref() = ACTIVE_KIND[] === :vel

# Global → local frame rotation helper (used by asmc_torques and plotting;
# formerly defined in the deleted Symbolics reference cell). The x/y/psi
# arguments are callables of t — VelRef/PosRef Getters work directly.
@inline function global_to_local_frame(t, current_psi, x_func, y_func, psi_func)
    Axo_d   = x_func(t)
    Ayo_d   = y_func(t)
    alpha_d = psi_func(t)
    c_psi, s_psi = cos(current_psi), sin(current_psi)
    Ax_local =  Axo_d * c_psi + Ayo_d * s_psi
    Ay_local = -Axo_d * s_psi + Ayo_d * c_psi
    return Ax_local, Ay_local, alpha_d
end

# =============================================================================
# TOML config handling — base.toml + per-profile files.
#
# Per-profile TOML layout (all five profiles):
#   [profile]            builder = "octagon"            (required)
#   [profile.params]     fixed scalars / structural lists  → copied verbatim
#   [profile.sweep]      key = [v1, v2, ...]   → ONE random value per key
#   [profile.combos]     parallel arrays       → ONE row index i; element i of
#                        every column (scalar, string, or list-valued alike)
# =============================================================================
import TOML
import Random

export load_base, resolve_profile, enumerate_jobs, build_job, pick_and_build

"""Read configs/base.toml (shared sweep grid + solver + fixed settings)."""
function load_base(config_dir::AbstractString)
    path = joinpath(config_dir, "base.toml")
    isfile(path) || error("Profiles.load_base: $path not found")
    return TOML.parsefile(path)
end

"""Resolve a profile TOML path: prefer <config_dir>/profiles/<f>, fall back to
<config_dir>/<f> (flat layout). Errors listing both tried paths."""
function _profile_path(config_dir::AbstractString, f::AbstractString)
    p1 = joinpath(config_dir, "profiles", f)
    isfile(p1) && return p1
    p2 = joinpath(config_dir, f)
    isfile(p2) && return p2
    error("Profiles: profile file '$f' not found; tried\n  $p1\n  $p2\n" *
          "Place profile TOMLs in '$(joinpath(config_dir, "profiles"))' (canonical) " *
          "or flat beside base.toml.")
end

"""
    resolve_profile(prof; combo_idx=nothing, rng) -> Dict{String,Any}

Flatten a parsed `[profile]` table into the flat `cfg::Dict` a builder expects.

  - `params` keys are copied verbatim.
  - `sweep` keys each get ONE value drawn from their list via `rng`.
  - `combos` contributes element `i` of every parallel column, where `i` is
    `combo_idx` if given (deterministic sweep) or drawn from `rng` (interactive).
    Scalar, string, and list-valued columns all resolve via the same indexing.

The chosen row index is recorded as `cfg["combo_idx"]` for provenance.
"""
function resolve_profile(prof::AbstractDict;
                         combo_idx::Union{Integer,Nothing} = nothing,
                         rng::Random.AbstractRNG = Random.default_rng())
    cfg = Dict{String,Any}()

    if haskey(prof, "params")
        for (k, v) in prof["params"];  cfg[k] = v;  end
    end

    if haskey(prof, "sweep")
        for k in sort(collect(keys(prof["sweep"])))     # sorted ⇒ stable RNG consumption
            vals = prof["sweep"][k]
            vals isa AbstractVector || error("Profiles: [profile.sweep] key '$k' must be a list")
            cfg[k] = vals[Random.rand(rng, 1:length(vals))]
        end
    end

    if haskey(prof, "combos")
        combos = prof["combos"]
        cols   = sort(collect(keys(combos)))          # sorted ⇒ stable iteration
        n      = length(combos[cols[1]])
        for k in cols
            length(combos[k]) == n ||
                error("Profiles: [profile.combos] column '$k' has length " *
                      "$(length(combos[k])) ≠ $n — parallel arrays must match")
        end
        i = combo_idx === nothing ? Random.rand(rng, 1:n) : Int(combo_idx)
        1 <= i <= n || error("Profiles: combo_idx=$i out of range 1:$n")
        for k in cols;  cfg[k] = combos[k][i];  end
        cfg["combo_idx"] = i
    end

    return cfg
end

"""Number of combo rows in a parsed `[profile]` table (1 if no combos table)."""
n_combos(prof::AbstractDict) =
    haskey(prof, "combos") ? length(first(values(prof["combos"]))) : 1

# -----------------------------------------------------------------------------
# Deterministic full enumeration (sweep-driver path)
# -----------------------------------------------------------------------------

"""
    enumerate_jobs(config_dir, profile_files; sweep_seed=1234) -> Vector{NamedTuple}

One job per (profile, combo row), in file order then row order — the FULL
deterministic enumeration. Each job carries everything `build_job` needs:

    (profile, builder, file, combo_idx, n_combos, sweep_seed)

`sweep_seed` controls the per-job RNG used only for `[profile.sweep]` keys
(e.g. long_circle's psi0_deg): the RNG is seeded from
hash((sweep_seed, profile, combo_idx)), so re-running the sweep (resume) draws
identical values — no dependence on enumeration order or thread schedule.
"""
function enumerate_jobs(config_dir::AbstractString,
                        profile_files::AbstractVector{<:AbstractString};
                        sweep_seed::Integer = 1234)
    jobs = NamedTuple[]
    for f in profile_files
        path = _profile_path(config_dir, f)
        prof = TOML.parsefile(path)["profile"]
        haskey(prof, "builder") || error("Profiles: $f missing profile.builder")
        bname = String(prof["builder"])
        haskey(BUILDERS, bname) || error("Profiles: $f names unknown builder '$bname'")
        # Strip a trailing _mu_0pN tag so μ-versioned TOMLs (e.g. octagon_mu_0p3.toml)
        # emit the SAME output "profile" id (octagon) — matched across the μ grid.
        pname = replace(first(splitext(basename(f))), r"_mu_0p\d+$" => "")
        N = n_combos(prof)
        for i in 1:N
            push!(jobs, (profile = pname, builder = bname, file = String(path),
                         combo_idx = i, n_combos = N, sweep_seed = Int(sweep_seed)))
        end
    end
    return jobs
end

"""
    build_job(job) -> (ref::VelRef, cfg::Dict)

Parse + resolve + build one enumerated job. Deterministic given the job tuple.
Does NOT publish to CURRENT_REF — pair with `set_reference!(ref)` in the driver,
single-threaded, before the inner parallel loop.
"""
function build_job(job::NamedTuple)
    prof = TOML.parsefile(job.file)["profile"]
    rng  = Random.Xoshiro(hash((job.sweep_seed, job.profile, job.combo_idx)))
    cfg  = resolve_profile(prof; combo_idx = job.combo_idx, rng = rng)
    ref  = build(String(prof["builder"]), cfg)
    return ref, cfg
end

# -----------------------------------------------------------------------------
# Random single-pick (interactive notebook path)
# -----------------------------------------------------------------------------

"""
    pick_and_build(config_dir, profile_set; rng) -> (ref, cfg, profile_name)

Pick one profile file at random from `profile_set`, resolve a random combo row
(and random sweep values), build, and PUBLISH it to CURRENT_REF. For
interactive single runs in the notebook; the sweep driver uses
enumerate_jobs/build_job + set_reference! instead.
"""
function pick_and_build(config_dir::AbstractString,
                        profile_set::AbstractVector{<:AbstractString};
                        rng::Random.AbstractRNG = Random.default_rng())
    f    = profile_set[Random.rand(rng, 1:length(profile_set))]
    path = _profile_path(config_dir, f)
    prof = TOML.parsefile(path)["profile"]
    cfg  = resolve_profile(prof; rng = rng)
    ref  = build(String(prof["builder"]), cfg)
    publish!(ref)                       # dispatches VelRef/PosRef to the right slot
    return ref, cfg, replace(first(splitext(basename(f))), r"_mu_0p\d+$" => "")
end

end # module Profiles
