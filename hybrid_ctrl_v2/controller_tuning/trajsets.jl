# =============================================================================
# hybrid_ctrl_v2/controller_tuning/trajsets.jl — TrajSetsMod
# =============================================================================
# Leaf module: single registry of the three v2 trajectory tiers (SCREEN /
# TRAIN_FULL / TEST), per controller-tuning-v2-staged-multistart.md §7.1.
#
# Depends on NOTHING (no `include`s) -- entries are plain data. Does NOT import
# tune_controller.jl (brief §4 constraint). Every entry this iteration runs
# mu=0.5, run_mode=:pose; `adapt=true` marks a VelRef builder lifted to PosRef
# via Profiles.velref_to_posref (done downstream, in run_controller_v2).
# =============================================================================
module TrajSetsMod

export trajset, tier_names

"""
    tier_names() -> NTuple{4,Symbol}

Canonical tier ordering.
"""
tier_names() = (:screen, :train_full, :test, :train12, :train14_v3, :test_v3)

_mk(name, toml, combo, ref_type, adapt, role, run_dir) =
    (name=name, profile_toml=toml, combo_idx=combo, ref_type=ref_type,
     mu=0.5, config_dir=run_dir, run_mode=:pose, adapt=adapt, role=role)

# ---- SCREEN — 4 trajectories, ~50 s/eval (brief §7.1 table 1) --------------
function _screen(run_dir::AbstractString)
    return [
        _mk("spin_creep_stress_yaw", "spin_creep_mu_0p5.toml",              178, :velref, true,  :stress_yaw,   run_dir),
        _mk("ellipse_stress_crab",   "ellipse_mu_0p5.toml",                  83, :posref, false, :stress_crab,  run_dir),
        _mk("multisine75_broadband", "multisine_75percent_cap_mu_0p5.toml",  55, :velref, true,  :broadband,    run_dir),
        _mk("docking_step",          "docking_step_mu_0p5.toml",              1, :posref, false, :step_hold,    run_dir),
    ]
end

# ---- TRAIN_FULL — 12 trajectories, superset of SCREEN (brief §7.1 table 2) -
function _train_full(run_dir::AbstractString)
    return [
        _mk("octagon_easy",            "octagon_mu_0p5.toml",                   1, :velref, true,  :easy,           run_dir),
        _mk("octagon_stress",          "octagon_mu_0p5.toml",                 206, :velref, true,  :stress,         run_dir),
        _mk("spin_creep_easy",         "spin_creep_mu_0p5.toml",              255, :velref, true,  :easy,           run_dir),
        _mk("spin_creep_stress_yaw",   "spin_creep_mu_0p5.toml",              178, :velref, true,  :stress_yaw,     run_dir),
        _mk("coupled_vomega_easy",     "coupled_vomega_mu_0p5.toml",          114, :velref, true,  :easy,           run_dir),
        _mk("coupled_vomega_stress",   "coupled_vomega_mu_0p5.toml",           12, :velref, true,  :stress,         run_dir),
        _mk("spiral_orbit_stress",     "spiral_orbit_mu_0p5.toml",             37, :velref, true,  :stress,         run_dir),
        _mk("ellipse_stress_tangent",  "ellipse_mu_0p5.toml",                  55, :posref, false, :stress_tangent, run_dir),
        _mk("ellipse_stress_crab",     "ellipse_mu_0p5.toml",                  83, :posref, false, :stress_crab,    run_dir),
        _mk("multisine75_broadband",   "multisine_75percent_cap_mu_0p5.toml",  55, :velref, true,  :broadband,      run_dir),
        _mk("docking_a",               "docking_mu_0p5.toml",                   1, :posref, false, :step_hold,      run_dir),
        _mk("docking_step",            "docking_step_mu_0p5.toml",              1, :posref, false, :step_hold,      run_dir),
    ]
end

# ---- TEST — 8 trajectories, HELD OUT (brief §7.1 table 3 minus straightline) -
# straightline_profile REMOVED (per-user direction): a pure zero-curvature
# translation profile with no representation anywhere in the training
# excitation cube (octagon/spin_creep/ellipse/spiral_orbit/coupled_vomega are
# all curved/rotating) -- both PID-FB and PID-CT blew up on it near-identically
# (tracking 144.25/143.48, ~98% of the aggregate test score), a shared
# training-coverage gap rather than a controller-specific or overfitting
# signal (every held_out_combo entry -- a different parameter draw of a
# profile FAMILY that WAS trained on -- generalized cleanly, 0.004-0.07
# tracking). Keeping it in the aggregate score buried a coverage gap inside
# what looked like a single number; report it separately if/when the training
# set gains a straight-line profile instead of silently reintroducing it here.
function _test(run_dir::AbstractString)
    return [
        _mk("long_circle_profile",        "long_circle_mu_0p5.toml",              52, :velref, true,  :held_out_profile, run_dir),
        _mk("long_circle_profile_stress", "long_circle_mu_0p5.toml",             102, :velref, true,  :held_out_profile, run_dir),
        _mk("multisine75_combo_gentle",   "multisine_75percent_cap_mu_0p5.toml",   1, :velref, true,  :held_out_combo,   run_dir),
        _mk("multisine50_combo",          "multisine_50percent_cap_mu_0p5.toml",  55, :velref, true,  :held_out_combo,   run_dir),
        _mk("spiral_orbit_easy",          "spiral_orbit_mu_0p5.toml",             27, :velref, true,  :held_out_combo,   run_dir),
        _mk("ellipse_easy_tangent",       "ellipse_mu_0p5.toml",                   1, :posref, false, :held_out_combo,   run_dir),
        _mk("ellipse_easy_crab",          "ellipse_mu_0p5.toml",                  73, :posref, false, :held_out_combo,   run_dir),
        _mk("coupled_vomega_anchor",      "coupled_vomega_mu_0p5.toml",           12, :velref, true,  :anchor,           run_dir),
    ]
end

# ---- TRAIN12 — ad hoc 12-trajectory set (per-user direction, "for now" --
# ADDITIVE ONLY, does not touch :screen/:train_full above so any future
# ASMC v2/MPC v2 tuning run against those tiers is unaffected). Derived from
# TRAIN_FULL by dropping both docking (:step_hold) entries and adding two new
# octagon combos:
#   octagon_mid           combo 2   -- vcru=0.6, documented feasible in
#                                      tune_controller.jl's default_trajs
#                                      ("peak |Vy|=0.6, feasible; combo 3-4
#                                      exceed it")
#   octagon_stress_hdg30  combo 309 -- EXACT heading-shifted twin of
#                                      octagon_stress (combo 206): identical
#                                      vcru=0.6/lat_vamp=0.26/accel_frac=0.16/
#                                      n_waves=1, only theta0_deg differs
#                                      (206: 15deg, 309: 30deg) -- confirmed
#                                      by resolving both combos directly,
#                                      not guessed from the raw sweep.
# Used for BOTH trajs_screen and trajs_full in this run (a single 12-entry
# training set, not the usual smaller-screen/larger-full split) -- per-user
# direction ("only 12 trajectories total in training").
function _train12(run_dir::AbstractString)
    return [
        _mk("octagon_easy",           "octagon_mu_0p5.toml",                   1, :velref, true,  :easy,           run_dir),
        _mk("octagon_mid",            "octagon_mu_0p5.toml",                   2, :velref, true,  :mid,            run_dir),
        _mk("octagon_stress",         "octagon_mu_0p5.toml",                 206, :velref, true,  :stress,         run_dir),
        _mk("octagon_stress_hdg30",   "octagon_mu_0p5.toml",                 309, :velref, true,  :stress,         run_dir),
        _mk("spin_creep_easy",        "spin_creep_mu_0p5.toml",              255, :velref, true,  :easy,           run_dir),
        _mk("spin_creep_stress_yaw",  "spin_creep_mu_0p5.toml",              178, :velref, true,  :stress_yaw,     run_dir),
        _mk("coupled_vomega_easy",    "coupled_vomega_mu_0p5.toml",          114, :velref, true,  :easy,           run_dir),
        _mk("coupled_vomega_stress",  "coupled_vomega_mu_0p5.toml",           12, :velref, true,  :stress,         run_dir),
        _mk("spiral_orbit_stress",    "spiral_orbit_mu_0p5.toml",             37, :velref, true,  :stress,         run_dir),
        _mk("ellipse_stress_tangent", "ellipse_mu_0p5.toml",                  55, :posref, false, :stress_tangent, run_dir),
        _mk("ellipse_stress_crab",    "ellipse_mu_0p5.toml",                  83, :posref, false, :stress_crab,    run_dir),
        _mk("multisine75_broadband",  "multisine_75percent_cap_mu_0p5.toml",  55, :velref, true,  :broadband,      run_dir),
    ]
end

# =============================================================================
# V3 TIERS — feasibility-screened :train14_v3 / :test_v3          [change log]
# =============================================================================
# INTENT. The v2 tiers contain trajectories the HARDWARE CANNOT FLY, and the
# tuner was spending most of its budget on them. spiral_orbit_stress alone
# contributed 72-81% of the training tracking term while sitting at 85.4% of
# no-load wheel speed -- so the converged gains were optimised against a
# trajectory no control law recovers (ASMC old 6.78, ASMC retuned 5.25, PID-CT
# 5.08, PID-FB 14.66; each 3-8x its own next-worst entry). The v3 tiers replace
# those entries with the most demanding combos that remain inside the envelope,
# so the optimiser spends its budget where control quality is actually decidable.
# Expect ALL scores to drop -- the easy wins are gone and the objective now
# reflects trajectories the controllers can genuinely differentiate on.
#
# SCREENING CRITERION — two walls, both evaluated on the reference alone:
#   (1) MOTOR / back-EMF : max per-wheel speed (Vx +- Vy +- (l+h)*psidot)/R
#       against no-load V_max/(Kb*G) = 27.73 rad/s. This is the HARD wall. At
#       90% of no-load the back-EMF leaves ~9% of torque authority and error
#       growth cannot be arrested by any gain law.
#   (2) FRICTION CIRCLE : max (E54) contact-side utilisation, bare mass and
#       contact drags only. This is a SOFT wall -- octagon_stress/hdg30 exceed it
#       (1.092/1.111) and are among the best-tracked trajectories in the set,
#       because a brief excursion is momentary slip the LuGre plant absorbs.
#   Measured separation: everything <=76.2% of no-load is flyable; >=85.4% is
#   catastrophic for every controller. v3 targets <=70% for margin.
#   NOTE the pre-existing screen (Vy <= 0.6) sees NEITHER case: spiral_orbit_
#   stress and long_circle_profile_stress both have Vy = 0.000 exactly.
#
# EVALUATE EVERY REFERENCE OVER ITS OWN ref.T_total (9.5-85.3 s here). An earlier
# audit hard-coded T=12.0 and therefore truncated the long references; it picked a
# spiral combo that looked like 67.7% over its first 12 s and is 98.5% across its
# full 20.36 s. The SIMULATIONS were never affected -- scheduler.jl integrates to
# ref.T_total -- the error was confined to offline analysis. Any future screening
# script must read T_total, never assume a window.
#
# CHANGES (originals are untouched; :train12/:test stay byte-identical so every
# archived run remains reproducible):
#
#   tier          entry                       combo      wheel %      util
#   ------------  --------------------------  ---------  -----------  -------------
#   train14_v3    spiral_orbit_stress          37 -> 198  85.4 -> 56.4  0.882 -> 0.985
#   train14_v3    ellipse_stress_tangent       55 ->   5  76.2 -> 67.3  0.298 -> 0.734
#   train14_v3    octagon_stress_lat08        ADD    120        60.3          0.983
#   train14_v3    octagon_stress_lat08_hdg30  ADD    223        60.3          0.961
#   test_v3       long_circle_profile_stress  102 ->  15  90.6 -> 59.2  0.580 -> 0.409
#
#   worst-case wheel speed:  train12 85.4% -> train14_v3 67.3%
#                            test    90.6% -> test_v3    59.2%
#
# Per-entry rationale (what each swap buys, and what it costs) is documented at
# each function below. Scans: _tmp/spiral_rescan_Ttotal.jl, _tmp/ellipse_scan.jl,
# _tmp/longcircle_scan.jl, _tmp/octagon_scan.jl; audit: _tmp/feasibility_audit_Ttotal.jl.
# =============================================================================

# ---- TEST_V3 — :test with the velocity-INFEASIBLE long_circle replaced --------
# ADDITIVE ONLY: :test above is untouched. Exactly one entry differs.
#
#   long_circle_profile_stress   combo 102 -> combo 15
#
# WHY. Screened on BOTH walls, not just the friction circle:
#   motor / back-EMF : max per-wheel speed (Vx +- Vy +- (l+h)*psidot)/R against
#                      no-load V_max/(Kb*G) = 27.73 rad/s
#   friction circle  : max (E54) contact-side utilisation, bare mass + contact drags
# Combo 102 (R=1.5, worbit=0.6667, drive_axis=x -> Vx=1.000) sits at 90.6% of
# no-load: back-EMF takes 21.7 V of the 24 V bus, leaving ~9% of torque authority.
# It is the single trajectory behind the "ASMC generalises 11-20x worse than PID"
# result -- every controller fails it (PID-CT 8.53, PID-FB 8.96, ASMC old 20.11,
# ASMC retuned 56.97-77.83) and it dominates the tier aggregate. Its friction
# utilisation is only 0.580, which is exactly why a friction-only screen missed it.
#
# Combo 15 (R=0.6, worbit=0.8333, drive_axis=x) keeps the family's tangential
# traversal and the SAME forward speed as the easy entry (Vx=0.500), stressing it
# through a 5x tighter orbit and 5x the yaw rate: 59.2% of no-load, utilisation
# 0.409 against the easy entry's 0.137. A single-variable intensity increase in
# the same mode -- 204/400 combos pass both walls, and the higher-utilisation ones
# (367/385 at 0.777) all have drive_axis=y, which would confound "stress" with a
# traversal-mode change in a tier whose purpose is profile-level generalisation.
# Recorded as the alternative if a friction-dominant held-out case is wanted later
# (_tmp/longcircle_scan.jl).
#
# UNCHANGED and worth knowing: coupled_vomega_anchor is combo 12, the SAME
# trajectory as train12's coupled_vomega_stress (that is what :anchor means -- a
# cross-tier consistency check, verified byte-identical: ASMC old 1.6101 and
# PID-CT 0.6655 in both tiers). So the genuinely held-out count is 7, not 8.
function _test_v3(run_dir::AbstractString)
    return [
        _mk("long_circle_profile",        "long_circle_mu_0p5.toml",              52, :velref, true,  :held_out_profile, run_dir),
        # REPLACED: combo 102 (90.6% of no-load) -> combo 15 (59.2%).
        _mk("long_circle_profile_stress", "long_circle_mu_0p5.toml",              15, :velref, true,  :held_out_profile, run_dir),
        _mk("multisine75_combo_gentle",   "multisine_75percent_cap_mu_0p5.toml",   1, :velref, true,  :held_out_combo,   run_dir),
        _mk("multisine50_combo",          "multisine_50percent_cap_mu_0p5.toml",  55, :velref, true,  :held_out_combo,   run_dir),
        _mk("spiral_orbit_easy",          "spiral_orbit_mu_0p5.toml",             27, :velref, true,  :held_out_combo,   run_dir),
        _mk("ellipse_easy_tangent",       "ellipse_mu_0p5.toml",                   1, :posref, false, :held_out_combo,   run_dir),
        _mk("ellipse_easy_crab",          "ellipse_mu_0p5.toml",                  73, :posref, false, :held_out_combo,   run_dir),
        _mk("coupled_vomega_anchor",      "coupled_vomega_mu_0p5.toml",           12, :velref, true,  :anchor,           run_dir),
    ]
end

# ---- TRAIN14_V3 — feasibility-screened training set (14 entries) --------------
# RENAMED from the 12-entry convention on purpose: two octagon entries are ADDED
# below, so a "train12" label would be a lie. :train12 above is untouched, so
# every archived run against it stays reproducible.
#
# Two entries REPLACED (velocity wall) and two ADDED (friction coverage).
#
#   spiral_orbit_stress   combo 37 -> combo 198
#
# WHY. Feasibility here is a VELOCITY (back-EMF) wall, not the friction circle,
# and the existing screen (Vy <= 0.6) cannot see it: the binding quantity is
# per-wheel speed (Vx +- Vy +- (l+h)*psidot)/R against the no-load speed
# V_max/(Kb*G) = 27.73 rad/s, which no per-axis cap captures. spiral_orbit_stress
# combo 37 reaches Vy=0.000 but psidot=2.038, putting it at 85.4% of no-load --
# back-EMF eats 21.1 V of the 24 V bus, leaving ~9% of torque authority. No
# control law recovers it: ASMC old 6.78, ASMC retuned 5.25, PID-CT 5.08, PID-FB
# 14.66, every one of them 3-8x its own next-worst trajectory. It then contributed
# 72-81% of the training tracking term, so the tuner spent most of its budget
# optimising against a trajectory the hardware cannot fly.
#
# EVALUATE EVERY REFERENCE OVER ITS OWN ref.T_total. An earlier pass hard-coded
# T=12.0 s for the audit; true durations run 9.5-85.3 s, so that truncated the
# long references and picked combo 128, which looks like 67.7% over its first 12 s
# but reaches 98.5% of no-load (util 1.063) across its full 20.36 s -- WORSE than
# the combo 37 it replaced. Nothing was wrong with the simulations, which always
# used ref.T_total (hybrid_ctrl/scheduler.jl); the error was confined to the
# offline audit. Rescanned properly, 46/200 spiral combos pass both walls
# (_tmp/spiral_rescan_Ttotal.jl).
#
# Combo 198 (T_total 22.4 s): 56.4% of no-load, utilisation 0.985, psidot=0.935 --
# the HIGHEST yaw rate of any feasible combo, which is the point since spin is what
# this entry exists to exercise. Combo 192 carries marginally more friction demand
# (0.999) but sits essentially ON the circle and at 66.4% of no-load, so 198 is
# taken for the margin.
#
# TWO CHARACTER CHANGES, deliberate and recorded: (1) every feasible spiral combo
# is lateral-dominant (Vx=0.000, Vy=0.632), so the entry becomes spin-with-strafe
# rather than spin-with-forward-motion; combo 37's psidot=2.038 is unreachable
# inside the envelope. (2) Vy=0.632 marginally exceeds the older Vy<=0.6
# heuristic, but passes both walls that were actually validated against outcomes.
#
#   ellipse_stress_tangent   combo 55 -> combo 5
#
# WHY. Combo 55 (a=4.00, worbit=0.2542) violates the documented ellipse a<=0.8
# rule by 5x. Peak tangential speed is a*worbit = 1.017 m/s and psi_mode=tangent
# puts ALL of it into body Vx, giving 76.2% of no-load -- the marginal band, ~24%
# of torque authority left. It tracks ~0.20 for every controller so it is not
# catastrophic, but it IS the entry whose chatter rose +157% under the corrected
# schedule and drove maxK to 3.87.
#
# Worse, it was not a stress case at all: its friction utilisation is 0.298
# against ellipse_easy_tangent's 0.299 -- IDENTICAL demand, differing only in
# speed. Combo 5 (a=0.80, ratio=0.40, worbit=0.7275) is rule-compliant, sits at
# 67.3% of no-load, and carries utilisation 0.734 -- 2.5x the demand of the entry
# it replaces, and a genuine stress counterpart to the easy tangent case rather
# than the same trajectory driven faster. 26/72 tangent combos pass wheel<=70%;
# combo 5 is the most demanding of them (_tmp/ellipse_scan.jl).
#
# ADDED (not replacing anything):
#   octagon_stress_lat08        combo 120  (theta0=15)
#   octagon_stress_lat08_hdg30  combo 223  (theta0=30)
#
# WHY ADDED RATHER THAN SWAPPED. octagon_stress (206) and octagon_stress_hdg30
# (309) exceed the friction circle -- utilisation 1.092 and 1.111 -- so they fail
# a strict util<=1.0 screen. But the measurement does not support treating that as
# infeasibility: both are tracked at ~0.10 by EVERY controller, among the best in
# the set, while long_circle_profile_stress at 58% circle utilisation is
# catastrophic for all of them. A brief circle excursion is momentary slip the
# LuGre plant absorbs; the back-EMF wall means no torque exists at all. Only the
# second is a real feasibility limit.
#
# Swapping them for the nearest compliant twin pair (120/223, utilisation 0.961)
# would also cost the discriminating axis: it drops lat_vamp 0.26 -> 0.08, a 3x
# cut in LATERAL excitation, and lateral is exactly where the control laws differ
# (on coupled_vomega, ASMC beats PID-CT 16x on x and 11x on psi while losing 5x on
# y). So both pairs are kept: 206/309 retain the lateral content and sit just over
# the circle, 120/223 sit just under it with the same vcru=0.6 and n_waves=1.
#
# CONSEQUENCE ON WEIGHTING, worth knowing before reading any score: the octagon
# family now carries 6/14 = 43% of the trajectory mean (was 4/12 = 33%). The
# objective is a plain mean over trajectories, so this is a real re-weighting
# toward octagon behaviour, not a neutral addition.
function _train14_v3(run_dir::AbstractString)
    return [
        _mk("octagon_easy",           "octagon_mu_0p5.toml",                   1, :velref, true,  :easy,           run_dir),
        _mk("octagon_mid",            "octagon_mu_0p5.toml",                   2, :velref, true,  :mid,            run_dir),
        _mk("octagon_stress",         "octagon_mu_0p5.toml",                 206, :velref, true,  :stress,         run_dir),
        _mk("octagon_stress_hdg30",   "octagon_mu_0p5.toml",                 309, :velref, true,  :stress,         run_dir),
        # ADDED: the compliant twin pair (util 0.961, vs 206/309's 1.092/1.111).
        _mk("octagon_stress_lat08",       "octagon_mu_0p5.toml",             120, :velref, true,  :stress,         run_dir),
        _mk("octagon_stress_lat08_hdg30", "octagon_mu_0p5.toml",             223, :velref, true,  :stress,         run_dir),
        _mk("spin_creep_easy",        "spin_creep_mu_0p5.toml",              255, :velref, true,  :easy,           run_dir),
        _mk("spin_creep_stress_yaw",  "spin_creep_mu_0p5.toml",              178, :velref, true,  :stress_yaw,     run_dir),
        _mk("coupled_vomega_easy",    "coupled_vomega_mu_0p5.toml",          114, :velref, true,  :easy,           run_dir),
        _mk("coupled_vomega_stress",  "coupled_vomega_mu_0p5.toml",           12, :velref, true,  :stress,         run_dir),
        # REPLACED: combo 37 (85.4% of no-load) -> combo 198 (56.4%).
        _mk("spiral_orbit_stress",    "spiral_orbit_mu_0p5.toml",            198, :velref, true,  :stress,         run_dir),
        # REPLACED: combo 55 (76.2%, util 0.298) -> combo 5 (67.3%, util 0.734).
        _mk("ellipse_stress_tangent", "ellipse_mu_0p5.toml",                   5, :posref, false, :stress_tangent, run_dir),
        _mk("ellipse_stress_crab",    "ellipse_mu_0p5.toml",                  83, :posref, false, :stress_crab,    run_dir),
        _mk("multisine75_broadband",  "multisine_75percent_cap_mu_0p5.toml",  55, :velref, true,  :broadband,      run_dir),
    ]
end

"""
    trajset(tier::Symbol, run_dir::AbstractString) -> Vector{NamedTuple}

Return the trajectory entries for one tier (`:screen`, `:train_full`, `:test`,
`:train12`). Shaped exactly like `tune_controller.jl`'s `default_trajs_pose`
output plus `tier::Symbol` and `role::Symbol`. Every entry: run_mode=:pose,
mu=0.5.
"""
function trajset(tier::Symbol, run_dir::AbstractString)
    entries = tier == :screen     ? _screen(run_dir) :
              tier == :train_full ? _train_full(run_dir) :
              tier == :test       ? _test(run_dir) :
              tier == :train12    ? _train12(run_dir) :
              tier == :train14_v3 ? _train14_v3(run_dir) :
              tier == :test_v3    ? _test_v3(run_dir) :
              error("TrajSetsMod.trajset: unknown tier $tier (expected :screen|:train_full|:test|:train12|:train14_v3|:test_v3)")
    return [merge(e, (tier=tier,)) for e in entries]
end

end # module
