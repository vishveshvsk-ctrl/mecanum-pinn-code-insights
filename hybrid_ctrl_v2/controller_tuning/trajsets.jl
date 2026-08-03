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
tier_names() = (:screen, :train_full, :test, :train12)

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

# ---- TEST — 9 trajectories, HELD OUT (brief §7.1 table 3) ------------------
function _test(run_dir::AbstractString)
    return [
        _mk("long_circle_profile",        "long_circle_mu_0p5.toml",              52, :velref, true,  :held_out_profile, run_dir),
        _mk("long_circle_profile_stress", "long_circle_mu_0p5.toml",             102, :velref, true,  :held_out_profile, run_dir),
        _mk("straightline_profile",       "straightline_mu_0p5.toml",              1, :velref, true,  :held_out_profile, run_dir),
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
              error("TrajSetsMod.trajset: unknown tier $tier (expected :screen|:train_full|:test|:train12)")
    return [merge(e, (tier=tier,)) for e in entries]
end

end # module
