#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/tune_controller_v2.jl — v2 adapter over the UNTOUCHED
# tune_controller.jl (controller-tuning-v2-staged-multistart.md)
# =============================================================================
# PRESERVATION CONSTRAINT (brief §11): tune_controller.jl and controllers.jl
# are NEVER edited. Everything Stage 0-3 needs beyond what they already export
# is added here as NEW functions/types (multiple-dispatch extension, plain
# `include`, and one duplicated-and-adapted function) so the v1 behaviour of
# both files stays byte-for-byte reproducible.
#
#   - `run_controller` (original) hardcodes a call to `build_controller(ctrl, kw)`
#     internally, and does not return the `bus` it builds (needed here to call
#     `clear_probe_log!` after every eval -- see §9, the ESTIMATOR_PROBE_LOG
#     unbounded-growth note). Both are only fixable by editing the function
#     body, which the constraint forbids -- so `run_controller_v2` below is a
#     faithful duplicate that (a) takes pre-built controller overrides directly
#     instead of deriving them from `kw` via the hardcoded `build_controller`
#     call, and (b) returns `bus` alongside `(probe, ref, mode)`.
#   - `build_controller` (original) is reused AS-IS for :asmc/:pid (freeze
#     values are merged into `kw` by the CALLER -- StageObjectiveMod -- before
#     `decode`, so no `freeze` kwarg needs to be added to the original).
#   - MPCControllerV2 (Stage 3's P_terminal field) lives in controllers_v2.jl,
#     which extends `ControllerMod.mpc_wrench!` with a new method rather than
#     editing the MPCController struct.
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)

using Pkg; Pkg.activate(ROOT)
include(joinpath(ROOT, "tune_controller.jl"))
include(joinpath(@__DIR__, "controllers_v2.jl"))
using .ControllerV2Mod

"""
    run_controller_v2(ctrl, oracle_kind, tr; asmc_o=nothing, mpc_o=nothing,
                      pid_o=nothing, seed=42, noise_scale=1.0)
    -> (probe, ref, mode, bus)

Duplicate of `tune_controller.jl`'s `run_controller`, parameterized on
pre-built controller override objects instead of `(ctrl, kw)` +
`build_controller`. Lets callers hand in a `ControllerV2Mod.MPCControllerV2`
(Stage 3) exactly like an ordinary `ControllerMod.MPCController` override, and
returns `bus` so the caller can `SchedulerMod.clear_probe_log!(bus)` after
reading `probe` (the original `run_controller` never returns `bus`, so this
cleanup was structurally impossible without a duplicate).
"""
function run_controller_v2(ctrl::Symbol, oracle_kind::Symbol, tr;
                           asmc_o=nothing, mpc_o=nothing, pid_o=nothing,
                           seed::Int=42, noise_scale::Float64=1.0)
    base   = Profiles.load_base(tr.config_dir)
    chi    = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(tr.mu))

    cfg = HybridConfig(
        tracking      = tr.run_mode,
        estimator     = :oracle,
        use_dhat      = false,
        use_asmc      = ctrl == :asmc,
        use_mpc       = ctrl == :mpc,
        use_pid       = ctrl == :pid,
        fuzzy         = false,
        fixed_weights = weights_for(ctrl),
        use_pose_fix  = false,
        sensor_seed   = seed,
    )

    # Reference construction: byte-identical RNG discipline to run_controller
    # (Xoshiro(hash(profile_toml)) unpinned / Xoshiro(0) pinned combo_idx) --
    # see brief §9 RNG discipline note.
    if get(tr, :combo_idx, nothing) === nothing
        ref = Profiles.pick_and_build(tr.config_dir, [tr.profile_toml];
                                      rng=Random.Xoshiro(hash(tr.profile_toml)))[1]
    else
        path  = joinpath(tr.config_dir, "profiles", tr.profile_toml)
        prof  = TOML.parsefile(path)["profile"]
        cfg_r = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
        ref   = Profiles.build(prof["builder"], cfg_r)
    end
    if get(tr, :adapt, false)
        ref = Profiles.velref_to_posref(ref)
    end
    Profiles.publish!(ref)
    oracle = OracleEstimator(oracle_kind; seed=seed, scale=noise_scale)

    sol, _df, bus = SchedulerMod.run_hybrid(
        cfg, params, Symbol(tr.name);
        chi=chi, friction_case=1, config_dir=tr.config_dir,
        profile_toml=tr.profile_toml, return_bus=true, est=oracle, ref=ref,
        asmc_override=asmc_o, mpc_override=mpc_o, pid_override=pid_o)

    probe = get(SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    return probe, ref, tr.run_mode, bus
end

"""
    build_controller_v2(ctrl, kw) -> (asmc_o, mpc_o, pid_o)

:asmc/:pid reuse the original `build_controller` unchanged. :mpc builds a
`ControllerV2Mod.MPCControllerV2` (Stage 3's P_terminal-carrying type) instead
of the original `MPCController`. Two ways to specify `R`/`S` are supported:
a direct full `:R`/`:S` SVector (what `MPCDesignMod.expand_ratio_space`
produces for Stage 3's ratio search) takes priority; a scalar `:R_scale`/
`:S_scale` (the original build_controller's :mpc decode contract, for a
uniform-fill fallback) is used otherwise; else the struct default.
"""
# ASMC_SPACE_V2 -- FOUR tuned dimensions: surface slopes lam_{x,y,psi}_max plus
# tau_ceiling. K_max_*, lam_*_min, mu_xy/mu_psi and K_x0/K_y0/K_psi0 are all
# DERIVED (ControllerV2Mod.ASMCControllerV2) and dropped from the search.
#
# gamma_{x,y,psi} are NOT searched. asmc-v2-physical-gain-bounds.md §7.3 specified
# six dimensions including them (and its acceptance list still carries
# `n_params(ASMC_SPACE_V2) == 6`), but asmc-v2-tuning-launch.md §6 SUPERSEDES it:
# gamma was factored out of the gain-update bracket
#
#     dK = gamma * [ s*tanh(s/eps)*bound - c_tilde*(K/K_max)^3 - sigma_tilde*(...) ]
#
# so it became a pure time-scaling that no longer sets where K settles, and was
# then fixed and removed from the search. That is a MEASURED call, not a
# stylistic one: score varies only 0.5-1.2% across a 64x gamma sweep
# (octagon_stress 0.7075 -> 0.7042 for gamma = 25 -> 1600).
#
# (The stale "six dimensions" comment previously here is the reason this needed
# stating -- see also the reachability guard in stage_objective.jl, which now
# ERRORS if --lambda-gamma is passed against a space with no :gamma_x.)
#
# DIMENSION 4 WAS `tau_ceiling`, NOW `rho_auth`. tau_ceiling set K's settling
# level only through the CUBIC BARRIER (K_eq ~ tau_ceiling^(1/3)); with
# `use_cubic=false` it feeds `cubic_coeff` and nothing else, so it has been
# completely inert since the cubic was disabled -- a quarter of the search budget
# spent on a dimension with zero gradient. It is replaced by `rho_auth`, the
# demand-driven gain law's one scalar (ControllerV2Mod, `use_demand_k`):
#
#     z* = G/(G + rho_auth*eps),   G = s*tanh(s/eps)
#
# read as "how many boundary layers of sliding error buys HALF the available
# authority". BOTH BOX ENDS ARE PHYSICALLY ANCHORED (see run_asmc_v2 in
# run_stage.jl for the full derivation and the measurement behind it):
#   lo = tanh(1) = 0.762  -- at one boundary layer, the width the controller is
#        DESIGNED to hold, the adaptation must claim at most HALF its authority.
#   hi = 9*3 = 27         -- 3*eps is the excursion the as-designed sigma leak
#        ITSELF treats as "surface excited" (its gate is exp(decay_k*(1 -
#        s^2/(9 eps^2))), unity at s=3eps). z*(3eps) = 0.1 at rho = 27 and 0.9 at
#        rho = 3/9, so the box spans the full usable authority range at the
#        design's own excitation scale and no more.
# The measured-good region (rho 4-8 at frozen lam, mean tracking 0.6074/0.5977)
# is comfortably interior, so the box does not truncate the optimum.
#
# WARM-START HAZARD: a run warm-started from a PRE-SWAP archive will map that
# archive's dimension-4 coordinate (tau_ceiling in log[0.1,100]) into rho_auth's
# log[0.5,64] and get a wrong-but-plausible rho, silently. Warm starts must not
# cross this change; start such runs cold.
const ASMC_SPACE_V2 = [
    ("lam_x_max",   1, :log, 0.3, 10.0),
    ("lam_y_max",   1, :log, 0.3, 10.0),
    ("lam_psi_max", 1, :log, 1.0, 20.0),
    ("rho_auth",    1, :log, 0.762, 27.0),
]

# PID_SPACE_V2 -- three tuned dimensions (instructions/pid-v2-imc-cascade-two-
# variants.md §7.2): lam_inner per axis, GENUINELY per-axis upper bounds
# (=tau_open, beyond which the closed loop is slower than doing nothing) and
# a UNIFORM lower bound (~5 sample periods at 100 Hz; the saturation and
# noise-amplification alternative lower bounds are both slack, see brief).
# Three separate len=1 rows (not one len=3 row) because the bounds differ per
# axis; build_controller_v2 regroups lam_inner_{x,y,psi} into lam_inner::SVector{3}.
# LOWER BOUNDS ARE NO LONGER A DESIGN SPECIFICATION -- they are a numerical
# guard at 3*dt (dt = 1/f_pid = 1 ms now that PID is rate-matched to the ASMC),
# present only so a pathological eval cannot drive Kp = m_eff/lambda to infinity,
# diverge the plant ODE and burn evaluations against the 1e6 sentinel.
#
# The lower bound is instead supplied ENDOGENOUSLY by the chatter term (per-user
# direction): command noise scales as Kp*sigma_v = m_eff*sigma_v/lambda, so
# chatter diverges as lambda -> 0 and the optimizer prices the floor itself
# rather than inheriting one from the sample rate. Measured on the two archived
# FB families (_tmp/check_chatter_vs_lambda_pid_fb.jl, 12 trajectories x 5 seeds),
# the clean-stage elasticity d ln(chatter)/d ln(1/lambda) is 1.00 on psi
# (exact 1/lambda scaling), 0.70 on x and 0.65 on y -- a real gradient on all
# three axes, which is what makes this viable. REQUIRES --lambda-chatter > 0;
# with lambda_chatter = 0 these bounds are NOT a design floor and every axis
# will rail into the guard.
#
# The old floors were sample-rate derived (~5 periods at the inherited 100 Hz):
# 0.05 uniform, hand-lowered to 0.01 on psi after every FB seed railed. Both are
# obsolete at 1 kHz.
const PID_SPACE_V2 = [
    ("lam_inner_x",   1, :log, 0.003, 0.256),   # upper = tau_open, genuinely per-axis
    ("lam_inner_y",   1, :log, 0.003, 0.114),
    ("lam_inner_psi", 1, :log, 0.003, 0.155),
]

# MPC_SPACE_V2 -- THREE tuned dimensions: tau_cl per axis. Everything else in
# the QP is DERIVED, so this is the whole continuous search space:
#
#   Q_pose  = bryson_Q_pose(TOL.pos_max,   TOL.head_max,   tau_cl)   running
#   Q_final = bryson_Q_pose(TOL.pos_final, TOL.head_final, tau_cl)   terminal seed
#   R       = bryson_R(lim, motor)                                   physical
#   S       = lambda_chatter / (dV_max/rate_hz)^2                    shared weight
#   P       = DARE(A_end, Bm, Q_final, R)                            falls out
#
# WHY tau_cl AND NOT S_scale (this REPLACES the brief's §7.2 1-D S_scale space):
# only RATIOS matter in a QP, so {Q,R,S} carry two independent degrees of freedom.
# Bryson pins Q's and R's absolute scale from tolerances and limits, which used to
# leave S_scale as the one free ratio. But S is now derived from lambda_chatter
# (see MPCControllerV2), so that ratio is spoken for -- and searching it would
# have let MPC discover its own chatter trade while ASMC/PID are held to the
# shared one. What is NOT pinned is the POSITION-vs-VELOCITY weighting inside
# Q, which is set entirely by tau_cl (q_v = q_pos*tau_cl^2). That is the real
# remaining freedom, and it is what the older MPCDesignMod.normalized_mpc_space
# reached for with its `q_vel_ratio` scalar.
#
# tau_cl was previously FROZEN at (0.15, 0.11, 0.12) -- PID v2's design
# mid-window, described as the "shared v2 bandwidth". That is now stale: the
# chatter-priced PID retune converged to lam_inner_psi ~0.0145 (FB) / ~0.0199
# (CT), 6-8x away from 0.12, and tau_cl enters Q QUADRATICALLY, so MPC's yaw-rate
# weight was 36-68x heavier than a PID-matched value. Rather than re-freeze it at
# a number that will go stale again, it is searched.
#
# BOUNDS mirror PID_SPACE_V2 deliberately -- same quantity (a closed-loop time
# constant per axis), so the two controllers' bandwidth searches sit on the same
# footing. Upper = tau_open per axis (beyond it the loop is slower than doing
# nothing); lower = the same 0.003 numerical guard, NOT a design floor.
#
# UNLIKE PID, chatter is NOT known to bound tau_cl from below. For PID the
# mechanism is direct (Kp = m_eff/lambda, so command noise ~ 1/lambda). Here
# tau_cl only reshapes the Q weighting, and the sign of its effect on chatter is
# not established -- so the 0.003 guard may well bind. Watch it, as with PID.
const MPC_SPACE_V2 = [
    ("tau_cl_x",   1, :log, 0.003, 0.256),
    ("tau_cl_y",   1, :log, 0.003, 0.114),
    ("tau_cl_psi", 1, :log, 0.003, 0.155),
]

# Cache cell for default_physical_limits() below.
const _DEFAULT_LIM = Ref{Union{Nothing,ControllerV2Mod.PhysicalLimits}}(nothing)

"""
    default_physical_limits() -> ControllerV2Mod.PhysicalLimits

Lazily-constructed, cached `PhysicalLimits` for the standard
`trajectory_files_run_0p5_main` run_dir at `mu=0.5` -- every current v2
trajset uses exactly this (config_dir, mu) pair, and `PhysicalLimits`'
fields (kappa, mu_N3, m_tilde, I_psi, ...) depend only on base.toml's
geometry/mass/viscous entries, NOT on which `mu_friction` a given
`PlatformParams` happened to be built with -- so one cached instance is
correct for every :asmc trajectory in the current trajsets.
"""
function default_physical_limits()
    if _DEFAULT_LIM[] === nothing
        base = Profiles.load_base("trajectory_files_run_0p5_main")
        params0 = PlatformParams(base; mu_friction=0.5)
        _DEFAULT_LIM[] = ControllerV2Mod.PhysicalLimits(params0; mu=0.5)
    end
    return _DEFAULT_LIM[]
end

"""
    build_controller_v2(ctrl, kw) -> (asmc_o, mpc_o, pid_o)

:asmc builds `ControllerV2Mod.ASMCControllerV2` (this brief's physically-
derived-bounds controller) instead of the original `ASMCController`. `kw.lim`
overrides the default `PhysicalLimits` (e.g. a caller building it from a
non-default run_dir/mu); otherwise `default_physical_limits()` is used.
:pid is unchanged (reuses the original `build_controller`). :mpc builds
`ControllerV2Mod.MPCControllerV2` (instructions/mpc-v2-bryson-weights-
terminal-cost.md's extended type -- Bryson-derived `Q_pose`/`R`, `S_scale`-
searched `S`, `U_eq` re-centering, Riccati terminal cost); `kw.lim`/`kw.motor`
override the defaults the SAME way `kw.lim` does for :asmc/:pid.
"""
function build_controller_v2(ctrl::Symbol, kw::NamedTuple)
    if ctrl == :asmc
        lim = get(kw, :lim, default_physical_limits())
        asmc_kw = Dict{Symbol,Any}(:lim => lim)
        for k in (:lam_x_max, :lam_y_max, :lam_psi_max, :gamma_x, :gamma_y, :gamma_psi,
                 :eps, :eps_psi, :noise, :tol_v, :tol_rate, :k_eps,
                 :eps_floor_xy, :eps_floor_psi,
                 :tau_relax, :tau_ceiling, :decay_k, :v_max_axis,
                 :use_scheduled_kmax, :kmax_lpf_tau, :start_at_floor, :kmax_sched_floor,
                 :kmax_contact_b, :use_demand_k, :rho_auth, :z_den_frac, :enforce_k_floor,
                 # K_max_base is DERIVED (capability_wrench(lim)), NOT a tuning
                 # parameter -- it is the wrench the friction circle can deliver.
                 # WARNING, measured: overriding it is a NO-OP under the default
                 # use_scheduled_kmax=true, because the per-tick ceiling comes from
                 # kmax_schedule(asmc.lim, ...) which recomputes capability_wrench
                 # (lim) internally; K_max_base then survives only in the
                 # never-binding degenerate floor (0.05 .* K_max_base). A 4x sweep
                 # of it produced bit-identical chatter/tracking/score
                 # (_tmp/kmax_sweep.jl) for exactly this reason. To move the
                 # ceiling you must either set use_scheduled_kmax=false (which
                 # also disables the state-dependent scheduling) or scale `lim`
                 # itself. Forwarded for that first case only.
                 :K_max_base,
                 # log_K enables the per-tick K / K_max_sched / W / Msw logging
                 # (brief §10's "max(K) logged per axis" check, still outstanding).
                 # Zero cost when false, which is the default.
                 :log_K, :use_cubic)
            haskey(kw, k) && (asmc_kw[k] = getfield(kw, k))
        end
        return ControllerV2Mod.ASMCControllerV2(; asmc_kw...), nothing, nothing
    elseif ctrl == :pid
        lim = get(kw, :lim, default_physical_limits())
        # PID_SPACE_V2's decode gives 3 scalar keys (lam_inner_x/y/psi) -- regroup
        # into lam_inner::SVector{3} unless the caller already supplied it directly.
        lam_inner = haskey(kw, :lam_inner) ? kw.lam_inner :
                    SVector(get(kw, :lam_inner_x, 0.15), get(kw, :lam_inner_y, 0.11),
                           get(kw, :lam_inner_psi, 0.12))
        pid_kw = Dict{Symbol,Any}(:lim => lim, :lam_inner => lam_inner)
        for k in (:N, :feedforward, :use_rate_limit, :vcmd_clamp, :rate_hz, :log_diag)
            haskey(kw, k) && (pid_kw[k] = getfield(kw, k))
        end
        return nothing, nothing, ControllerV2Mod.PIDControllerV2(; pid_kw...)
    elseif ctrl == :mpc
        # instructions/mpc-v2-bryson-weights-terminal-cost.md: lim/motor/
        # tau_cl are construction-time deps of the DERIVED Q_pose/R defaults
        # (bryson_Q_pose/bryson_R), so they must be resolved and passed
        # explicitly -- MPCControllerV2() with no args now errors (lim has no
        # default, matching the ASMCControllerV2/PIDControllerV2 precedent).
        lim   = get(kw, :lim, default_physical_limits())
        motor = get(kw, :motor, Main.PlantMod.MotorParams())
        mpc_kw = Dict{Symbol,Any}(:lim => lim, :motor => motor)
        # MPC_SPACE_V2's decode gives 3 scalar keys (tau_cl_x/y/psi) -- regroup
        # into tau_cl::SVector{3} unless the caller supplied it directly. Exactly
        # the PID_SPACE_V2 lam_inner_x/y/psi precedent above.
        if !haskey(kw, :tau_cl) && haskey(kw, :tau_cl_x)
            mpc_kw[:tau_cl] = SVector(kw.tau_cl_x, kw.tau_cl_y, kw.tau_cl_psi)
        end
        for k in (:tau_cl, :Np, :Q, :Np_pose, :Q_pose, :Q_final, :R, :S_scale, :S,
                 :lambda_chatter, :P_terminal, :rate_hz, :use_ltv, :use_u_eq,
                 :use_terminal, :log_diag)
            haskey(kw, k) && (mpc_kw[k] = getfield(kw, k))
        end
        # Q_pose/Q_final/S are @kwdef defaults computed FROM tau_cl/lambda_chatter/
        # rate_hz. Passing a stale explicit value would silently override the
        # derivation the search is driving, so drop them unless the caller is
        # deliberately pinning (they are absent from MPC_SPACE_V2's decode).
        if haskey(mpc_kw, :tau_cl)
            for k in (:Q_pose, :Q_final); delete!(mpc_kw, k); end
        end
        return nothing, ControllerV2Mod.MPCControllerV2(; mpc_kw...), nothing
    else
        return build_controller(ctrl, kw)
    end
end
