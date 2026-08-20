# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl — ReplayTrajSetMod
# =============================================================================
# Frozen trajectory list for ESTIMATOR-ONLY replay tuning (per-user direction:
# tune against trajectories that ALREADY EXIST in data/, not fresh ODE sims —
# see `harness_v2.jl`'s `run_and_log_replay_v2`). Every entry below was
# verified present as a real Arrow file (+ accel sidecar where available) in
# `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/` at
# `trajectory_files_run_0p5_main`'s base.toml chi (0.005) before being listed
# here.
#
# 9 velocity-mode entries are the SAME combos as the already-vetted v1
# 10-trajectory replay manifest (`_tmp/subset_manifest_10traj.json`, used
# successfully in `runs_estimator_posfix_velref_10traj_replay`) minus one
# duplicate spiral_orbit/octagon entry, spanning mu=0.3/0.5/0.8 (slip->grip).
# 2 ellipse (PosRef, :pose) entries are reused from the PCRLB bound-analysis
# selection (`bound_analysis/reports/ellipse_selection.json`): c001
# (unloaded, low friction-circle utilization) and c005 (loaded, high
# utilization) — "ellipse cases included" per user direction. `docking`/
# `docking_step` are excluded: no Arrow data for them exists anywhere in the
# main sweep (verified: 0 files), so they cannot be replayed at all.
#
# `name` MUST be the literal profile string used in the Arrow filename
# contract (e.g. "ellipse", "octagon") — `DataStore.expected_output` resolves
# the path from it directly. No `profile_toml` field: `run_and_log_replay_v2`
# never reads it (only `config_dir` for `chi`, and `name`/`combo_idx`/`mu`
# for the Arrow path).
#
# Depends on NOTHING (no `include`s) — entries are plain data.
# =============================================================================
module ReplayTrajSetMod

export replay_trajset

_mk(name, combo, mu, ref_type, run_mode, role, run_dir) =
    (name=name, combo_idx=combo, mu=mu, config_dir=run_dir,
     ref_type=ref_type, run_mode=run_mode, role=role)

"""
    replay_trajset(run_dir::AbstractString="trajectory_files_run_0p5_main") -> Vector{NamedTuple}

11 trajectories: 9 velocity-mode (grip->slip spread, mu=0.8/0.5/0.3) + 2
ellipse PosRef (:pose, loaded + unloaded) entries. All confirmed present as
Arrow files in the main data/ sweep.
"""
function replay_trajset(run_dir::AbstractString="trajectory_files_run_0p5_main")
    return [
        _mk("octagon",                  288, 0.5, :velref, :velocity, :grip,       run_dir),
        _mk("octagon",                  213, 0.3, :velref, :velocity, :slip,       run_dir),
        _mk("coupled_vomega",            32, 0.8, :velref, :velocity, :grip,       run_dir),
        _mk("coupled_vomega",            72, 0.5, :velref, :velocity, :mid,        run_dir),
        _mk("spin_creep",                26, 0.3, :velref, :velocity, :slip,       run_dir),
        _mk("spin_creep",                46, 0.3, :velref, :velocity, :slip,       run_dir),
        _mk("long_circle",              115, 0.3, :velref, :velocity, :slip,       run_dir),
        _mk("spiral_orbit",              16, 0.8, :velref, :velocity, :grip,       run_dir),
        _mk("multisine_75percent_cap",   55, 0.8, :velref, :velocity, :broadband,  run_dir),
        _mk("ellipse",                     1, 0.5, :posref, :pose,     :unloaded,   run_dir),
        _mk("ellipse",                     5, 0.5, :posref, :pose,     :loaded,     run_dir),
    ]
end

end # module
