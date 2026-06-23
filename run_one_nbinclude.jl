# ============================================================================
# run_one_nbinclude.jl — INTERIM simulator loader for Data_Generation_Julia.jl.
#
# The committed run_one.jl (extract_run_one.py output, May 2025) is STALE: it
# predates the profile-based migration and lacks Profiles/DataStore, ESOParams,
# coupling_of, asmc_torques_vel, lugre_dyn_rates, the PlatformParams(BASE; ...)
# constructor and the 7-arg DataStore.compute_labels the driver calls. Until
# extract_run_one.py is rebuilt, load the simulator definitions DIRECTLY from the
# authoritative notebook via NBInclude — the same mechanism validated by
# Solver_Ablation_Multisine.ipynb.
#
# @nbinclude(...; counters = [1..6, 8..13]) runs the notebook's definition code
# cells in Main, which:
#   * includet("profiles.jl") / includet("datastore.jl") -> Profiles, DataStore
#   * defines PlatformParams, ASMCParams, ESOParams, LuGreParams, lugre,
#     coupling_of, lugre_dyn_rates, sawtooth_approx, asmc_torques,
#     asmc_torques_vel, dynamics_full_mf_asmc!, build_initial_state
#
# Code cell 7 (the trajectory-build cell) is DELIBERATELY SKIPPED: it is the only
# cell that declares a non-const global `BASE` (and builds a throwaway reference),
# which collides with the driver's `const BASE = Profiles.load_base(...)`. The
# driver builds + publishes its own reference per trajectory, so nothing in cells
# 8..13 needs cell 7 at include time (they only DEFINE functions).
#
# Usage:
#   julia --project=. -t 8 Data_Generation_Julia.jl --script run_one_nbinclude.jl ...
# ============================================================================
ENV["GKSwstype"] = "nul"   # GR offscreen — the notebook's diagnostic plot cell must not open a window
using NBInclude
@nbinclude("Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb"; counters = vcat(1:6, 8:13))
