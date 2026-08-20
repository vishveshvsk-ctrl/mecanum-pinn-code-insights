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

export make_stage_objective, recovery_time, controller_metrics_v3, TOL_V3, LAMBDA_CHATTER_V3

# =============================================================================
# v3 TRACKING METRIC
# =============================================================================
# Replaces ONLY the `tracking` term of `Main.controller_metrics`; ce / chatter /
# chatter_hf are reused from it verbatim, so the chatter pricing and its
# normaliser are untouched. `tune_controller.jl` is on the never-edit list, so
# this lives here and the v2 metric stays the default.
#
# THREE CHANGES, each for a stated reason.
#
# (1) INTEGRAL TERM, TIME-NORMALISED. The v2 pose metric is
#         (final_pos/1cm + max_pos/10cm + final_head/0.01 + max_head/0.1)/4
#     i.e. FOUR samples out of ~12k-60k: t=T and the argmax, per channel. A
#     controller that oscillates violently mid-run but ends inside 1 cm and never
#     breaches 10 cm scores identically to one that tracks cleanly throughout.
#     `iae = mean(|e|)` is the time-NORMALISED integral (1/T)*int|e|dt -- uniform
#     probe sampling makes mean() exactly that -- so trajectories of different
#     duration (9.5-85.3 s here) are comparable. Mean-absolute rather than RMS
#     deliberately: `max` already carries the peak, so mean-abs adds the distinct
#     "typical/sustained" dimension instead of duplicating it. The three terms
#     then read terminal / peak / typical.
#
# (2) PER-TRAJECTORY SCALER on the WHOLE tracking term (per-user direction).
#         S = max(radius_var(ref), TOL.pos_max),   k_traj = S/TOL.pos_max
#     radius_var = max ||p_ref(t) - centroid||, the reference path's own extent.
#     Chosen over the alternatives because it is rotation-invariant (octagon_stress
#     and octagon_stress_hdg30 are the same path rotated 15 deg; a bounding-box
#     measure would score them differently), it is a single ISOTROPIC scalar
#     matching the Euclidean position error, and it measures EXTENT rather than
#     distance travelled (path length would rate a tight circle repeated many
#     times as "large"). ALL SIX terms are divided by k_traj -- position AND
#     heading. Dividing position alone (the first version, now reverted) silently
#     re-weighted position DOWN against heading by k_traj, since v2 held both
#     channels absolute and equally weighted; that re-weighting on its own
#     inverted the PID FB/CT ranking and cost an 8.4 h run. Scaling the whole
#     term keeps v2's position:heading balance exactly, keeps tracking
#     trajectory-relative, and avoids dpsi_var (cumulative rotation, which would
#     discount heading error 130x on the very trajectories that rotate).
#     The TOL.pos_max floor matters: spin_creep_easy never translates (radius
#     0.000 exactly), so an unfloored scaler divides by zero. Floored, it falls
#     back to the absolute tolerance, which is the right treatment for a
#     spin-in-place reference.
#
# (3) TRIGONOMETRIC HEADING ERROR, ABSOLUTE SCALE.
#         e_head = ||u(psi) - u(psi_ref)|| = 2|sin(dpsi/2)|,  u = [cos, sin]
#     the chord between heading unit vectors. Monotone on [0,pi], SMOOTH at
#     dpsi = +-pi where |atan2(sin,cos)| has its branch kink, bounded by 2, and
#     LINEAR for small dpsi so the existing head tolerances carry over unchanged
#     (at the measured p90 of 0.0057 rad the two forms differ by 1.4e-6 relative;
#     they diverge only above ~0.5 rad, where compressing a gross error is the
#     desired behaviour). |sin(dpsi)| was rejected as non-monotone -- it cannot
#     distinguish 5 deg from 175 deg -- and 1-cos(dpsi) as quadratic in dpsi,
#     which would change the units and invalidate the tolerances.
#     NO trajectory scaler on heading. dpsi_var is CUMULATIVE ROTATION (rate x
#     duration), not an extent: spin_creep_stress_yaw accumulates 13 rad, so
#     scaling by it would discount heading error 130x on exactly the trajectories
#     that exercise heading control, leaving the channel owned by the four
#     octagons and ellipse_stress_crab -- the entries with ZERO commanded
#     rotation. A chord error is bounded by 2 regardless, so it has a fixed
#     natural scale and wants an absolute tolerance.
#
# TOL_V3's two new tolerances are MEASURED, not chosen: over 7 converged configs
# x 14 trajectories the achieved iae/max ratio is 0.249 (position) and 0.128
# (heading), so matching the integral term to the existing max terms gives
# 0.249*pos_max = 0.0249 m and 0.128*head_max = 0.0128. Calibrating rather than
# guessing because this stack's worst objective bug was `chatter` normalised by
# V_MAX=24 -- a voltage dividing a slew rate -- which left the term ~30x too weak
# and silently unpriced every run until someone checked.
const TOL_V3 = (pos_iae = 0.0249, head_iae = 0.0128)

"""
    LAMBDA_CHATTER_V3

Chatter price to use with `metric=:v3`. **`lambda_chatter=3.0` is calibrated for
the v2 tracking scale and MUST NOT be carried over.** v3 tracking is 5-10x
smaller (the k_pos division), so at 3.0 the chatter term is ~90% of the score and
the optimiser effectively stops pricing tracking -- the same class of silent
re-weighting as the old `chatter/V_MAX=24` bug, where one term's normaliser
changed while its exchange rate against the others did not.

BASIS: balance-preserving, NOT a re-derived exchange rate, and the distinction is
load-bearing. At the converged operating point (lam_psi 30, chatter 0.13675) the
v2 objective at lambda=3.0 puts tracking at 29.9% of (tracking + chatter term);
0.36 reproduces that same 29.9% split under v3. It preserves the weighting the
project already accepted rather than claiming to derive a new one.

WHY NOT DERIVED: the exchange-rate method that produced 3.0 (widen lam_psi_max,
price d(tracking)/d(chatter)) FAILS ITS OWN CONTROL here -- re-run on the v2
metric it returns 0.052 against the known bracket 0.91-9.5, because under the
corrected schedule and the demand-k law lam_psi_max is no longer a v2 trade-off
lever at all: over lam_psi 12->60, v2 tracking is flat to 2% and drifts the WRONG
way (0.2199 -> 0.2233) while chatter rises 0.126 -> 0.202. There is no v2 trade
to price. Under v3 the trade IS real (tracking 0.0361 -> 0.0229, -37%), which is
further evidence that final+max was blind to sustained error -- and retroactively
explains three box-widening rounds that kept railing on a lever the objective
could not price. A properly derived rate needs a lever that trades both under
BOTH metrics; eps_floor_psi (boundary-layer width) is the classical candidate and
is not yet measured. Revisit if the chatter weighting is ever contested.
"""
const LAMBDA_CHATTER_V3 = 0.36

_chord(dpsi) = 2 * abs(sin(0.5 * dpsi))

"""
    _radius_var(ref; n=2000) -> Float64

Max radial excursion of the reference path from its own time-mean position, over
`ref.T_total`. Computed from T_total and a FIXED sample count rather than from the
probe, so the scaler is a pure function of the trajectory and cannot shift with
run length (a diverged run yields a short probe). Reading T_total is not optional:
durations span 9.5-85.3 s and an earlier audit that assumed 12.0 s picked a
trajectory that looked feasible over its first 12 s and is 98.5% of no-load
across its full duration.
"""
function _radius_var(ref; n::Int=2000)
    T = ref.T_total
    xs = Vector{Float64}(undef, n); ys = Vector{Float64}(undef, n)
    @inbounds for (i, t) in enumerate(range(0.0, T; length=n))
        xs[i] = ref.xo(t); ys[i] = ref.yo(t)
    end
    cx = sum(xs)/n; cy = sum(ys)/n
    m = 0.0
    @inbounds for i in 1:n
        m = max(m, (xs[i]-cx)^2 + (ys[i]-cy)^2)
    end
    return sqrt(m)
end

"""
    controller_metrics_v3(probe, ref, mode) -> NamedTuple

Same shape as `Main.controller_metrics`; `tracking` recomputed per the header.
`mode == :velocity` falls through unchanged (that branch is already a
full-trajectory RMS, and no current tier uses it).
"""
function controller_metrics_v3(probe, ref, mode::Symbol)
    m = Main.controller_metrics(probe, ref, mode)
    (mode != :pose || !m.ok || isempty(probe)) && return m
    T = Main.TOL
    pe = Vector{Float64}(undef, length(probe)); he = similar(pe)
    @inbounds for (i, p) in enumerate(probe)
        t = p.t
        pe[i] = sqrt((p.u[17] - ref.xo(t))^2 + (p.u[18] - ref.yo(t))^2)
        he[i] = _chord(p.u[4] - ref.psi(t))
    end
    k_traj = max(_radius_var(ref), T.pos_max) / T.pos_max     # >= 1
    fp, mp, ip = pe[end], maximum(pe), sum(pe)/length(pe)
    fh, mh, ih = he[end], maximum(he), sum(he)/length(he)
    # k_traj divides the WHOLE term, not the position half. Scaling position
    # alone silently re-weighted position DOWN against heading by k_traj (1-29,
    # mean ~16), because v2 held both channels absolute and equally weighted.
    # That inverted the PID ranking on its own: CT was better on all three
    # position terms (0.0436 vs FB's 0.0889 summed) and lost on heading (0.2966
    # vs 0.0783), so heading became 87% of CT's tracking and decided the result.
    # See the ABANDONED run archive. Dividing the whole term keeps v2's
    # position:heading balance exactly while still making tracking
    # trajectory-relative, and never involves dpsi_var.
    tracking = (fp/T.pos_final + mp/T.pos_max + ip/TOL_V3.pos_iae +
                fh/T.head_final + mh/T.head_max + ih/TOL_V3.head_iae) / (6 * k_traj)
    return (tracking=tracking, ce=m.ce, chatter=m.chatter, chatter_hf=m.chatter_hf,
            ok=isfinite(tracking) && m.ok,
            abs=(final_pos=fp, max_pos=mp, iae_pos=ip, final_head=fh, max_head=mh,
                 iae_head=ih, k_traj=k_traj))
end


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

const _FAILED_METRICS = (tracking=Inf, ce=Inf, chatter=Inf, chatter_hf=NaN, ok=false, abs=NamedTuple())

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
# _assert_terms_reachable(ctrl, space, freeze, expand, trajs; kwargs...)
#
# Fail LOUDLY at closure-construction time when a non-zero weight is passed for a
# term that cannot possibly affect the score under this controller/space.
#
# This whole class of bug has bitten four times in this stack, always silently:
#
#   * `lambda_chatter` was never forwarded to the objective by `run_pid_v2`,
#     `run_pid` or `run_mpc` -- the CLI flag parsed and was then dropped, so every
#     PID/MPC run was chatter-UNPRICED regardless of the value passed.
#   * `kmax_pen`/`gamma_pen` are gated on `:K_max_x`/`:gamma_x`, which ASMC v2's
#     decoded gains do not contain (`lam_*_max`/`tau_ceiling`), so both are
#     structurally inert for v2 at ANY lambda.
#   * `ASMC_SPACE_V2`'s docstring described a 6-D space the array never implemented.
#
# None of these errored. A tuning run would burn hours optimizing an objective
# that quietly differed from the one requested, and the result looks perfectly
# normal. Cheap up-front check, no simulation: decode the box midpoint exactly the
# way the closure will, then assert every requested term is actually reachable.
function _assert_terms_reachable(ctrl::Symbol, space, freeze::NamedTuple,
                                 expand::Function, trajs;
                                 lambda_chatter::Float64, lambda_kmax::Float64,
                                 lambda_gamma::Float64, recovery_weight::Float64)
    lo, hi = Main.flat_bounds(space)
    kw_raw = Main.decode((lo .+ hi) ./ 2, space)
    kw_raw = ctrl == :pid ? Main.PIDCascadeMod.regroup_joint(kw_raw) : kw_raw
    kw = expand(merge(freeze, kw_raw))

    bad = String[]
    if lambda_kmax > 0 && !(ctrl == :asmc && haskey(kw, :K_max_x))
        push!(bad, "lambda_kmax=$lambda_kmax but kmax_pen requires ctrl==:asmc AND a " *
                   ":K_max_x gain. ASMC v2 DERIVES K_max_* (they are not searched), so " *
                   "the term would be constant w.r.t. theta -- zero gradient, not a " *
                   "regulariser. Applies to v1 ASMC_SPACE only.")
    end
    if lambda_gamma > 0 && !(ctrl == :asmc && haskey(kw, :gamma_x))
        push!(bad, "lambda_gamma=$lambda_gamma but gamma_pen requires ctrl==:asmc AND a " *
                   ":gamma_x gain. ASMC v2 FIXES gamma and drops it from the search " *
                   "(asmc-v2-tuning-launch.md: score varies only 0.5-1.2% across a 64x " *
                   "gamma sweep -- measured inert). Applies to v1 ASMC_SPACE only.")
    end
    # rho_auth replaced tau_ceiling as ASMC_SPACE_V2's dimension 4. It is read
    # ONLY inside the demand-driven gain law, which asmc_wrench! gates on
    # `use_demand_k` (default FALSE) -- so searching it without that flag is
    # exactly the failure mode above, a whole dimension with zero gradient, and
    # it is the SECOND time this dimension has been inert (tau_ceiling was inert
    # from the moment use_cubic went false, which is why it was replaced).
    if haskey(kw, :rho_auth) && !get(kw, :use_demand_k, false)
        push!(bad, "rho_auth is a searched/frozen gain but use_demand_k is not true. " *
                   "The demand-driven gain law is gated on use_demand_k (default FALSE), " *
                   "so rho_auth would have zero effect on the score. Pass " *
                   "use_demand_k=true in `freeze` (normally alongside " *
                   "kmax_contact_b=true, the corrected schedule).")
    end
    # The converse: running the demand law while the space still carries the
    # now-inert tau_ceiling means dimension 4 is being searched for nothing.
    if get(kw, :use_demand_k, false) && haskey(kw, :tau_ceiling) && !haskey(kw, :rho_auth)
        push!(bad, "use_demand_k=true but the space/freeze supplies tau_ceiling and not " *
                   "rho_auth. tau_ceiling feeds only the (disabled) cubic barrier and the " *
                   "demand law ignores it, so this searches an inert dimension while " *
                   "leaving the live one at its default. Use the updated ASMC_SPACE_V2.")
    end
    isempty(bad) || error("make_stage_objective: requested objective term(s) are " *
                          "structurally unreachable and would be SILENTLY IGNORED:\n  - " *
                          join(bad, "\n  - "))

    # recovery_time is gated to role==:step_hold entries; with none in the set the
    # term is identically 0. WARN rather than error: unlike the lambdas (which
    # default to 0.0, so any non-zero value is an explicit request), recovery_weight
    # defaults to 0.1, so a trajset without step_hold entries is a legitimate
    # configuration the caller never asked for -- erroring would break it.
    if recovery_weight > 0 && !any(get(tr, :role, :none) == :step_hold for tr in trajs)
        @warn "make_stage_objective: recovery_weight=$recovery_weight but no trajectory " *
              "in this set has role==:step_hold -- the recovery term is identically 0 here."
    end
    # lambda_chatter needs no check: `chatter` is computed from v_cmd for every
    # controller on every run, so the term is always reachable once forwarded.
    return nothing
end

function make_stage_objective(ctrl::Symbol, space, trajs, oracle_kind::Symbol;
                              seed::Int=42, freeze::NamedTuple=(;),
                              expand::Function=identity,
                              lambda_chatter::Float64=0.0, lambda_kmax::Float64=0.0,
                              lambda_gamma::Float64=0.0, recovery_weight::Float64=0.0,
                              noise_replicates::Int=1, metric::Symbol=:v2)
    _assert_terms_reachable(ctrl, space, freeze, expand, trajs;
                            lambda_chatter=lambda_chatter, lambda_kmax=lambda_kmax,
                            lambda_gamma=lambda_gamma, recovery_weight=recovery_weight)
    metric in (:v2, :v3) || error("make_stage_objective: metric must be :v2 or :v3, got $metric")
    # A v2-scale lambda under the v3 metric is not merely suboptimal, it silently
    # converts the objective into a chatter-only one (v3 tracking is 5-10x smaller,
    # so lambda=3.0 leaves tracking at ~4-11% of the score). Refuse it loudly --
    # this stack has already lost a whole tuning campaign to exactly this shape of
    # error (chatter normalised by V_MAX=24, ~30x too weak, unnoticed for months).
    if metric === :v3 && lambda_chatter > 4 * LAMBDA_CHATTER_V3
        error("make_stage_objective: lambda_chatter=$lambda_chatter with metric=:v3. " *
              "3.0 is calibrated for the v2 tracking scale; v3 tracking is 5-10x smaller, " *
              "so this leaves tracking at a few percent of the score and the run would " *
              "optimise chatter alone. Use LAMBDA_CHATTER_V3 = $LAMBDA_CHATTER_V3 " *
              "(balance-preserving; see its docstring for why it is not a derived " *
              "exchange rate). Pass a value <= $(4*LAMBDA_CHATTER_V3) to override.")
    end
    # :v2 (default) is Main.controller_metrics verbatim, so every existing caller
    # and archived run is byte-identical. :v3 replaces ONLY the tracking term.
    _metrics = metric === :v3 ? controller_metrics_v3 : Main.controller_metrics
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

        track_sum = 0.0; ce_sum = 0.0; chat_sum = 0.0; chat_hf_sum = 0.0; recovery_sum = 0.0
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
            rep_tracking = Float64[]; rep_ce = Float64[]; rep_chatter = Float64[]
            rep_chatter_hf = Float64[]; rep_recovery = Float64[]
            failed = false
            for ns in noise_seeds
                m, rec = try
                    asmc_o, mpc_o, pid_o = build_ctrl()
                    probe, ref, mode, bus = Main.run_controller_v2(ctrl, oracle_kind, tr;
                        asmc_o=asmc_o, mpc_o=mpc_o, pid_o=pid_o, seed=ns)
                    mm = _metrics(probe, ref, mode)
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
                push!(rep_chatter, m.chatter); push!(rep_chatter_hf, m.chatter_hf)
                push!(rep_recovery, rec)
            end
            m = failed ? _FAILED_METRICS :
                (tracking=sum(rep_tracking)/length(rep_tracking), ce=sum(rep_ce)/length(rep_ce),
                 chatter=sum(rep_chatter)/length(rep_chatter),
                 chatter_hf=sum(rep_chatter_hf)/length(rep_chatter_hf), ok=true)
            rec = failed ? 0.0 : sum(rep_recovery) / length(rep_recovery)
            per_traj[tr.name] = (tracking=m.tracking, ce=m.ce, chatter=m.chatter,
                                 chatter_hf=m.chatter_hf, ok=m.ok, recovery=rec)
            if !m.ok || !isfinite(m.tracking)
                n_fail += 1
            else
                track_sum += m.tracking; ce_sum += m.ce; chat_sum += m.chatter
                chat_hf_sum += m.chatter_hf; recovery_sum += rec
            end
        end

        n_ok = length(trajs) - n_fail
        tracking = n_ok > 0 ? track_sum / n_ok : Inf
        ce       = n_ok > 0 ? ce_sum / n_ok    : Inf
        chatter  = n_ok > 0 ? chat_sum / n_ok  : Inf
        # REPORTED ONLY -- chatter_hf does not enter `score`. It is the
        # band-limited chatter diagnostic (tune_controller.jl CHATTER_FC); pricing
        # is still done, if at all, on the raw TV `chatter` term below.
        chatter_hf = n_ok > 0 ? chat_hf_sum / n_ok : NaN
        recovery = n_ok > 0 ? recovery_sum / n_ok : 0.0

        if n_fail > 0
            return (score=1e6, tracking=tracking, ce=ce, chatter=chatter,
                    chatter_hf=chatter_hf, kmax_pen=0.0,
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

        # chatter is normalised by CHATTER_REF (the 0.8 V/ms slew ceiling), NOT
        # by V_MAX -- see the CHATTER_REF docstring in tune_controller.jl. No
        # prior run is affected: every caller to date left lambda_chatter at 0.0.
        score = tracking + Main.LAMBDA_CE * (ce / Main.V_MAX) +
                lambda_chatter * (chatter / Main.CHATTER_REF) + recovery_weight * recovery +
                kmax_pen + gamma_pen

        return (score=score, tracking=tracking, ce=ce, chatter=chatter,
                chatter_hf=chatter_hf, kmax_pen=kmax_pen, gamma_pen=gamma_pen,
                recovery=recovery, per_traj=per_traj, n_fail=n_fail)
    end
end

end # module
