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
# ASMC_SPACE_V2 -- six tuned dimensions (instructions/asmc-v2-physical-gain-
# bounds.md §7.3): surface slopes lam_{x,y,psi}_max + adaptation rates
# gamma_{x,y,psi}. K_max_*, lam_*_min, mu_xy/mu_psi, and K_x0/K_y0/K_psi0 are
# all DERIVED now (ControllerV2Mod.ASMCControllerV2) and dropped from the
# search -- same dimensionality as v1's ASMC_SPACE, spanning different
# directions.
const ASMC_SPACE_V2 = [
    ("lam_x_max",   1, :log, 0.3, 10.0),
    ("lam_y_max",   1, :log, 0.3, 10.0),
    ("lam_psi_max", 1, :log, 1.0, 20.0),
    ("gamma_x",     1, :log, 1.0, 100.0),
    ("gamma_y",     1, :log, 1.0, 100.0),
    ("gamma_psi",   1, :log, 1.0, 100.0),
]

# PID_SPACE_V2 -- three tuned dimensions (instructions/pid-v2-imc-cascade-two-
# variants.md §7.2): lam_inner per axis, GENUINELY per-axis upper bounds
# (=tau_open, beyond which the closed loop is slower than doing nothing) and
# a UNIFORM lower bound (~5 sample periods at 100 Hz; the saturation and
# noise-amplification alternative lower bounds are both slack, see brief).
# Three separate len=1 rows (not one len=3 row) because the bounds differ per
# axis; build_controller_v2 regroups lam_inner_{x,y,psi} into lam_inner::SVector{3}.
const PID_SPACE_V2 = [
    ("lam_inner_x",   1, :log, 0.05, 0.256),
    ("lam_inner_y",   1, :log, 0.05, 0.114),
    ("lam_inner_psi", 1, :log, 0.01, 0.155),   # floor lowered 0.05->0.01 (per-user
                                                # direction): every PID v2 FB seed
                                                # (clean and noisy) pinned lam_inner_psi
                                                # at the 0.05 floor -- box was binding,
                                                # not a converged interior optimum.
]

# MPC_SPACE_V2 -- ONE tuned dimension (instructions/mpc-v2-bryson-weights-
# terminal-cost.md §7.2/§9): S_scale only. Q_pose/R are DERIVED
# (bryson_Q_pose/bryson_R at MPCControllerV2 construction, brief §6) and
# dropped from the search entirely -- 11 -> 1 dimensions (brief §1), down
# from the OLDER MPCDesignMod.normalized_mpc_space's 2-dim ratio search
# (controller_tuning/mpc_design.jl, a prior Stage-3 iteration this brief
# supersedes but does not delete).
#
# S_scale bracket [0.25, 25] (brief §7.2): from the hard slew limit
# max ΔV per tick = dV_max*dt = 200*0.01 = 2V; Bryson S_jj=1/(ΔV)^2 gives
# S=0.25 at the limit (ΔV=2V) and S=25 at "very smooth" (ΔV=0.2V).
# v1's [1e-3, 1] bracket does NOT transfer -- different Q normalization.
#
# dxNES is the WRONG tool for a single log-scaled parameter (brief §9:
# "designed for coupled, ill-conditioned, covariance-adapting problems").
# `MPC_S_SCALE_GRID` is the deterministic 10-point log grid a future sweep
# script would iterate (optionally golden-section-refined) instead --
# NOT wired into any dxNES-based optimizer here. The `Np_pose` convergence
# sweep (brief §7.3, run BEFORE this grid, at S_scale=0.25) and the actual
# multi-seed S_scale grid run are both deferred (matching the ASMC v2/PID v2
# briefs' "re-tuning is a later run" precedent) -- this pass only wires the
# derivation + search-space definition so that a later run can launch both.
const MPC_SPACE_V2 = [
    ("S_scale", 1, :log, 0.25, 25.0),
]
const MPC_S_SCALE_GRID = 10.0 .^ range(log10(0.25), log10(25.0); length=10)

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
                 :eps, :eps_psi, :tau_relax, :tau_ceiling, :decay_k, :v_max_axis,
                 :use_scheduled_kmax, :kmax_lpf_tau, :start_at_floor)
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
        for k in (:tau_cl, :Np, :Q, :Np_pose, :Q_pose, :R, :S_scale, :S,
                 :P_terminal, :rate_hz, :use_ltv, :use_u_eq, :use_terminal, :log_diag)
            haskey(kw, k) && (mpc_kw[k] = getfield(kw, k))
        end
        return nothing, ControllerV2Mod.MPCControllerV2(; mpc_kw...), nothing
    else
        return build_controller(ctrl, kw)
    end
end
