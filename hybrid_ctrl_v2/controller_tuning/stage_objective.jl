# =============================================================================
# hybrid_ctrl_v2/controller_tuning/stage_objective.jl — StageObjectiveMod
# =============================================================================
# Tier-scoped objective closure over `run_controller_v2`/`controller_metrics`.
# Everything from `tune_controller.jl`/`tune_controller_v2.jl` is referenced
# via explicit `Main.` qualification (a `module` nested via `include` does NOT
# see its includer's top-level bindings) -- the caller (`run_stage.jl`) is
# responsible for `include`ing `tune_controller_v2.jl` (and, when `ctrl==:pid`,
# `pid_cascade.jl`) BEFORE constructing any objective closure with this module.
# =============================================================================
module StageObjectiveMod

export make_stage_objective, recovery_time

"""
    recovery_time(probe, ref, tol_pos::Float64) -> Float64

Time (s) from the end of a step/approach segment (`ref.tstops[end-1]`, the
docking builder's `T_approach` boundary) until `|position error|` re-enters
and STAYS inside `tol_pos` for the remainder of the trajectory (any later
excursion resets the clock). Returns `ref.T_total` if never (durably)
recovered. Callers gate this to `role == :step_hold` entries only (this
function itself has no `role` parameter -- see brief §6); it is meaningless
(and unused) for any other role.
"""
function recovery_time(probe, ref, tol_pos::Float64)
    isempty(probe) && return 0.0
    T_total = ref.T_total
    t_step_end = length(ref.tstops) >= 2 ? ref.tstops[end-1] : 0.0
    last_ok = nothing
    for p in probe
        t = p.t
        t < t_step_end && continue
        pe = sqrt((p.u[17] - ref.xo(t))^2 + (p.u[18] - ref.yo(t))^2)
        if pe > tol_pos
            last_ok = nothing
        elseif last_ok === nothing
            last_ok = t
        end
    end
    return last_ok === nothing ? T_total : (last_ok - t_step_end)
end

const _FAILED_METRICS = (tracking=Inf, ce=Inf, chatter=Inf, ok=false, abs=NamedTuple())

"""
    _derive_noise_seeds(seed, k) -> Vector{Int}

`k` sub-seeds deterministically derived from the top-level `seed` (per-user
direction: average the objective over `k` different `OracleEstimator(:noisy)`
noise realizations per eval, instead of the single fixed realization
`OracleEstimator(kind; seed=seed)` gives when called with the SAME `seed`
every eval -- which is what every prior run did, since `run_controller_v2`
was always invoked with the outer `seed` unchanged across the whole
optimization. Pure function of `(seed, k)`: same top-level seed always
derives the same `k` sub-seeds, so a run is still fully reproducible.
"""
_derive_noise_seeds(seed::Int, k::Int) = [Int(hash((seed, :noise_rep, i)) % 1_000_000_000) for i in 1:k]

"""
    make_stage_objective(ctrl, space, trajs, oracle_kind; kwargs...) -> Function

Build a tier-scoped objective closure `theta::Vector{Float64} -> NamedTuple`.

Keyword args:
    seed::Int
    freeze::NamedTuple          gains pinned outside this phase's search space
                                 (merged in BEFORE `expand`; decoded values win
                                 on any overlapping key)
    expand::Function=identity   post-freeze `kw -> kw` hook for controllers
                                 whose search space isn't decoded gains
                                 directly -- e.g. Stage 3's
                                 `MPCDesignMod.expand_ratio_space`, which turns
                                 the ratio params into `Q_pose`/`R`/`S`
    lambda_chatter/lambda_kmax/lambda_gamma::Float64
    recovery_weight::Float64    weight on the recovery-time term (step_hold
                                 entries only; the identical scalarization
                                 form tune_controller.jl's `make_objective`
                                 already uses, plus this one extra term)
    noise_replicates::Int=1    per-user direction: with `oracle_kind=:noisy`,
                                 average tracking/ce/chatter/recovery over
                                 this many INDEPENDENT noise realizations per
                                 trajectory per eval (sub-seeds derived from
                                 `seed` via `_derive_noise_seeds`, so the run
                                 stays reproducible), instead of the single
                                 fixed realization every prior run used
                                 (`OracleEstimator(:noisy; seed=seed)` re-seeds
                                 identically on every eval when `seed` never
                                 changes). Default 1 = old single-realization
                                 behavior, unchanged for every existing
                                 caller. COST: multiplies the per-eval
                                 simulation count by `noise_replicates` --
                                 e.g. 5 turns a 12-trajectory eval into 60
                                 plant-ODE solves.

Unlike `tune_controller.jl`'s `make_objective` (which short-circuits to the
`1e6` sentinel on the FIRST failed trajectory), every trajectory is always
run so `per_traj` is complete for diagnostics/held-out-grid reporting even
when some entries fail -- but the returned `score` is STILL the `1e6`
sentinel whenever `n_fail > 0` (brief: "On any failed/non-finite trajectory
the score is the existing 1e6 sentinel" -- preserved, just computed after a
full pass instead of short-circuiting on the first failure).

`ctrl == :pid` always applies `Main.PIDCascadeMod.regroup_joint` to the
decoded `kw` (a no-op pass-through for the :inner/:outer spaces, which already
decode straight to SVector fields -- see pid_cascade.jl; also a no-op for
`PID_SPACE_V2`'s `lam_inner_x/y/psi` keys, which have no `:Kp1` field).

`:mpc` always routes through `Main.build_controller_v2`. `:pid` routes there
too, but ONLY when the decoded `kw` actually carries `PID_SPACE_V2`'s
`lam_inner`/`lam_inner_x` keys (a v2 tuning call) -- the pre-existing v1
cascade (`run_pid`/`PIDCascadeMod`, `Kp`/`Ki`/`Kd`/... keys) must keep
reaching the original `Main.build_controller`, since `build_controller_v2`'s
`:pid` branch has no idea what to do with those keys and would silently
build a DEFAULT `PIDControllerV2` every eval instead (wasting the whole
cascade run). `:asmc` is routed the same way, via `is_asmc_v2` (decoded `kw`
carries `lam_x_max`, `ASMC_SPACE_V2`'s discriminating key -- v1's `ASMC_SPACE`
has none) -- a v1 `:asmc` call (`ASMC_SPACE`, `Kp`/`K_max_x`/... keys) keeps
reaching the original `Main.build_controller`/`ASMCController` unchanged.
"""
function make_stage_objective(ctrl::Symbol, space, trajs, oracle_kind::Symbol;
                              seed::Int=42, freeze::NamedTuple=(;),
                              expand::Function=identity,
                              lambda_chatter::Float64=0.0, lambda_kmax::Float64=0.0,
                              lambda_gamma::Float64=0.0, recovery_weight::Float64=0.0,
                              noise_replicates::Int=1)
    noise_seeds = noise_replicates > 1 ? _derive_noise_seeds(seed, noise_replicates) : [seed]
    return function (theta::Vector{Float64})
        kw_raw = Main.decode(theta, space)
        kw_raw = ctrl == :pid ? Main.PIDCascadeMod.regroup_joint(kw_raw) : kw_raw
        kw = expand(merge(freeze, kw_raw))

        is_pid_v2 = ctrl == :pid && (haskey(kw, :lam_inner) || haskey(kw, :lam_inner_x))
        # ASMC_SPACE_V2's decode always carries lam_x_max -- v1's ASMC_SPACE has no
        # such key -- so this discriminates a v2 tuning call exactly like is_pid_v2
        # does for :pid.
        is_asmc_v2 = ctrl == :asmc && haskey(kw, :lam_x_max)
        build_ctrl() = (ctrl == :mpc || is_pid_v2 || is_asmc_v2) ? Main.build_controller_v2(ctrl, kw) :
                                                                   Main.build_controller(ctrl, kw)

        track_sum = 0.0; ce_sum = 0.0; chat_sum = 0.0; recovery_sum = 0.0
        per_traj = Dict{String,NamedTuple}()
        n_fail = 0

        for tr in trajs
            # Run once per noise sub-seed (noise_replicates=1 -- the default,
            # and every pre-existing caller's behavior -- is exactly the old
            # single-run path, just written as a length-1 loop). A trajectory
            # is counted as FAILED for this eval if ANY replicate fails,
            # matching the existing single-replicate failure semantics rather
            # than silently averaging over a partial failure.
            #
            # FRESH CONTROLLER PER (trajectory, replicate) CALL -- per-user
            # direction, fixing a state-leak found while planning the
            # trajectory-loop parallelization: v2 controllers carry per-tick
            # MUTABLE state on the struct itself (PIDControllerV2's
            # prev_vcmd/vcmd_initialized feeding vcmd_limits's rate-limiter,
            # in particular -- prev_e/initialized and prev_e_pos/
            # pos_initialized are inert since Kd=Kd_pos=0, but prev_vcmd is
            # genuinely consumed). Building the controller ONCE per eval and
            # reusing it across all 12 trajectories x replicates meant every
            # trajectory after the first inherited the PREVIOUS trajectory's
            # final vcmd_limits state -- an artifact of iteration order, not
            # of the gains being scored. Rebuilding fresh here is cheap
            # (closed-form derivation, not a simulation) and also protects
            # any future ASMC v2 use of this same loop (Vy_filt/psidot_filt/
            # initialized have the identical risk, just never exercised yet).
            rep_tracking = Float64[]; rep_ce = Float64[]; rep_chatter = Float64[]; rep_recovery = Float64[]
            failed = false
            for ns in noise_seeds
                m, rec = try
                    asmc_o, mpc_o, pid_o = build_ctrl()
                    probe, ref, mode, bus = Main.run_controller_v2(ctrl, oracle_kind, tr;
                        asmc_o=asmc_o, mpc_o=mpc_o, pid_o=pid_o, seed=ns)
                    mm = Main.controller_metrics(probe, ref, mode)
                    rr = get(tr, :role, :none) == :step_hold ? recovery_time(probe, ref, Main.TOL.pos_max) : 0.0
                    Main.SchedulerMod.clear_probe_log!(bus)
                    mm, rr
                catch e
                    @warn "StageObjectiveMod: run failed for $(tr.name) (noise seed $ns)" exception = e
                    _FAILED_METRICS, 0.0
                end
                if !m.ok || !isfinite(m.tracking)
                    failed = true
                    break
                end
                push!(rep_tracking, m.tracking); push!(rep_ce, m.ce)
                push!(rep_chatter, m.chatter); push!(rep_recovery, rec)
            end
            m = failed ? _FAILED_METRICS :
                (tracking=sum(rep_tracking)/length(rep_tracking), ce=sum(rep_ce)/length(rep_ce),
                 chatter=sum(rep_chatter)/length(rep_chatter), ok=true)
            rec = failed ? 0.0 : sum(rep_recovery) / length(rep_recovery)
            per_traj[tr.name] = (tracking=m.tracking, ce=m.ce, chatter=m.chatter, ok=m.ok, recovery=rec)
            if !m.ok || !isfinite(m.tracking)
                n_fail += 1
            else
                track_sum += m.tracking; ce_sum += m.ce; chat_sum += m.chatter; recovery_sum += rec
            end
        end

        n_ok = length(trajs) - n_fail
        tracking = n_ok > 0 ? track_sum / n_ok : Inf
        ce       = n_ok > 0 ? ce_sum / n_ok    : Inf
        chatter  = n_ok > 0 ? chat_sum / n_ok  : Inf
        recovery = n_ok > 0 ? recovery_sum / n_ok : 0.0

        if n_fail > 0
            return (score=1e6, tracking=tracking, ce=ce, chatter=chatter, kmax_pen=0.0,
                    gamma_pen=0.0, recovery=recovery, per_traj=per_traj, n_fail=n_fail)
        end

        # ASMC-only regularization terms -- identical form to
        # tune_controller.jl's make_objective (K_max headroom reward + gamma
        # regularization); guarded by haskey so a pinned-K_max space (which
        # never has kw.K_max_x) doesn't error.
        kmax_pen = (ctrl == :asmc && lambda_kmax > 0 && haskey(kw, :K_max_x)) ?
            lambda_kmax * (3*150.0 - (kw.K_max_x + kw.K_max_y + kw.K_max_psi)) / (3*150.0) : 0.0
        gamma_pen = (ctrl == :asmc && lambda_gamma > 0 && haskey(kw, :gamma_x)) ?
            lambda_gamma * (kw.gamma_x + kw.gamma_y + kw.gamma_psi) / (3*100.0) : 0.0

        score = tracking + Main.LAMBDA_CE * (ce / Main.V_MAX) +
                lambda_chatter * (chatter / Main.V_MAX) + recovery_weight * recovery +
                kmax_pen + gamma_pen

        return (score=score, tracking=tracking, ce=ce, chatter=chatter,
                kmax_pen=kmax_pen, gamma_pen=gamma_pen, recovery=recovery,
                per_traj=per_traj, n_fail=n_fail)
    end
end

end # module
