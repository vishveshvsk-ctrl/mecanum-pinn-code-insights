#!/usr/bin/env julia
# =============================================================================
# bound_analysis/select_ellipse_combos.jl
#
#   Screen all 96 combos in trajectory_files_run_0p5_main/profiles/
#   ellipse_mu_0p5.toml by BODY-LEVEL friction-circle utilization, filter to
#   psi_mode == "tangent" (the continuous-rotation regime — combos 1-72),
#   apply the feasibility filter, and emit the stratified 5/5 LOADED/UNLOADED
#   selection consumed by export_truth_traces.jl.
#
#   Per instructions/pcrlb-no-odometry-ellipse-bound.md §4, §7.1.
#
#   No simulation required: utilization is read off the PosRef's own analytic
#   acceleration getters (Axo, Ayo), which are algebraic (ForwardDiff of the
#   closed-form ellipse path) — this is pure Profiles work.
#
#   DISCREPANCY NOTE (see instructions brief, flagged during implementation):
#   theta_e_deg / psi0_deg are NOT combo-table columns — they live in
#   [profile.sweep] and are drawn ONE value per key via the RNG passed to
#   resolve_profile. We use the same Xoshiro(0) convention already established
#   in tune_controller.jl / compare_controllers_eskf.jl / save_controller_eskf_traces.jl
#   for reproducible per-combo_idx resolution, so re-running this script
#   reproduces the identical selection.
#
#   FEASIBILITY FILTER: no ready-made "platform velocity ceiling" function
#   exists in this repo (see recon report). We implement the peak
#   body-lateral-speed check ourselves (matching the formula SchedulerMod.log_run
#   uses to rotate a PosRef's world velocity into body frame), against the
#   V_CEILING established in project history (project_trajectory_feasibility_envelope
#   memory: sharp lateral-velocity ceiling at Vy ≈ 0.6-0.7 m/s; ellipse a≤~0.8
#   was measured feasible, a=4.0 was degraded). We ALSO require
#   diagnostics_combined.csv's combined_reco to start with "keep" (the
#   established open-loop chatter/tracking whitelist) — for the ellipse/mu=0.5
#   rows currently in that CSV this is a no-op (all 96 rows are "keep"), but we
#   apply it anyway per the brief's explicit instruction.
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)

using Pkg
Pkg.activate(ROOT)

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(ROOT, "profiles.jl")); using .Profiles

using TOML
using JSON
using CSV
using DataFrames
using Random

const G_GRAV = 9.81
const V_CEILING_MPS = 0.6     # peak body-lateral-speed feasibility ceiling (see header note)
const U_PEAK_STRESS_TARGET = 0.7   # brief §4/§8: label LOADED "hardest_available" if unreached

# -----------------------------------------------------------------------------
"""
    utilization(ref::Profiles.PosRef, mu::Float64; n_grid::Int=2000) -> NamedTuple

Body-level friction-circle utilization u(t) = hypot(Axo(t), Ayo(t)) / (mu*g),
evaluated on a uniform grid over the trajectory duration. Uses the PosRef's
analytic acceleration getters -- NO simulation is required for screening.

NOTE this is a BODY-level proxy. The exact per-wheel check would distribute the
wrench through the O-config allocation and compare against mu * N_per_roller;
the proxy is adequate for STRATIFICATION and is labelled as such in the report.

Also returns v_peak (peak resultant body speed hypot(Vx_body,Vy_body)) and
psidot_peak. NOTE: in "tangent" psi_mode (the only mode this script selects
from — see §7.1), heading tracks the path tangent BY CONSTRUCTION, so
Vy_body ≡ 0 identically and all commanded speed is Vx_body: the platform never
commands sideways motion in this regime. The feasibility-relevant ceiling
(project_trajectory_feasibility_envelope memory: "ellipse a=0.8 -> peak
0.31 m/s feasible; a=4.0 -> peak 1.02 m/s degraded", ceiling ~0.6-0.7 m/s) is
therefore checked against peak RESULTANT body speed, not peak |Vy_body| (which
would be vacuously zero here and filter nothing).

Returns: (u_peak, u_rms, v_peak, psidot_peak, T_total)
"""
function utilization(ref::Profiles.PosRef, mu::Float64; n_grid::Int=2000)
    T = ref.T_total
    ts = range(0.0, T; length=n_grid)
    u      = Vector{Float64}(undef, n_grid)
    vbody  = Vector{Float64}(undef, n_grid)
    psidot = Vector{Float64}(undef, n_grid)
    for (i, t) in enumerate(ts)
        ax, ay = ref.Axo(t), ref.Ayo(t)
        u[i] = hypot(ax, ay) / (mu * G_GRAV)
        vxo, vyo = ref.Vxo(t), ref.Vyo(t)
        vbody[i]  = hypot(vxo, vyo)   # == peak body speed since Vy_body≡0 in tangent mode
        psidot[i] = ref.om(t)
    end
    return (u_peak = maximum(u), u_rms = sqrt(sum(abs2, u) / n_grid),
            v_peak = maximum(vbody), psidot_peak = maximum(abs.(psidot)),
            T_total = T)
end

# -----------------------------------------------------------------------------
"""
    select_stratified(run_dir, mu; n_per_group=5, psi_mode_filter="tangent",
                      profile_toml="ellipse_mu_0p5.toml",
                      whitelist_csv=joinpath(ROOT,"diagnostics_combined.csv")) -> (rows, hardest_available)

Screen every ellipse combo, apply the feasibility filter, then take the top
`n_per_group` by u_peak as the LOADED group and the bottom `n_per_group` as the
UNLOADED group.

Feasibility filter (both must hold):
  - peak resultant body speed within V_CEILING_MPS (see utilization() docstring
    for why this is checked on resultant speed, not |Vy_body|, in tangent mode)
  - combo present in the diagnostics whitelist (combined_reco starts with "keep")
    when whitelist data exists for this profile/mu; if the whitelist CSV has no
    rows for this profile/mu, the whitelist check is skipped (not silently
    passed as "reject" -- absence of data is not evidence of infeasibility).

If no feasible combo reaches u_peak >= U_PEAK_STRESS_TARGET, the top LOADED
combo is labelled `hardest_available` -- the same convention already used in
runs_controller/subset_manifest.json.
"""
function select_stratified(run_dir::String, mu::Float64;
                           n_per_group::Int=5,
                           psi_mode_filter::String="tangent",
                           profile_toml::String="ellipse_mu_0p5.toml",
                           whitelist_csv::String=joinpath(ROOT, "diagnostics_combined.csv"))
    path = joinpath(run_dir, "profiles", profile_toml)
    prof = TOML.parsefile(path)["profile"]
    builder = String(prof["builder"])
    combos = prof["combos"]
    n = length(first(values(combos)))

    wl_ellipse = DataFrame()
    if isfile(whitelist_csv)
        wl = CSV.read(whitelist_csv, DataFrame)
        if "profile" in names(wl) && "mu" in names(wl)
            wl_ellipse = wl[(wl.profile .== "ellipse") .& (wl.mu .== mu), :]
        end
    end
    keep_set = isempty(wl_ellipse) ? Set{Int}() :
        Set{Int}(r.combo_idx for r in eachrow(wl_ellipse) if startswith(String(r.combined_reco), "keep"))

    rows = NamedTuple[]
    for i in 1:n
        cfg = Profiles.resolve_profile(prof; combo_idx=i, rng=Random.Xoshiro(0))
        psi_mode = String(get(cfg, "psi_mode", "tangent"))
        psi_mode == psi_mode_filter || continue

        ref = Profiles.build(builder, cfg)
        stats = utilization(ref, mu)

        whitelisted = isempty(wl_ellipse) ? true : (i in keep_set)
        feasible = whitelisted && stats.v_peak <= V_CEILING_MPS

        push!(rows, (
            combo_idx = i,
            a = Float64(cfg["a"]), ratio = Float64(cfg["ratio"]), worbit = Float64(cfg["worbit"]),
            psi_mode = psi_mode,
            u_peak = stats.u_peak, u_rms = stats.u_rms,
            v_peak = stats.v_peak, psidot_peak = stats.psidot_peak,
            T_total = stats.T_total,
            whitelisted = whitelisted,
            feasible = feasible,
            group = "unselected",
            role = "unselected",
        ))
    end

    feas = sort(filter(r -> r.feasible, rows), by = r -> r.u_peak, rev = true)
    isempty(feas) && error("select_stratified: no feasible ellipse/tangent combos found — " *
                           "cannot build a LOADED/UNLOADED selection.")

    hardest_available = feas[1].u_peak < U_PEAK_STRESS_TARGET

    k = min(n_per_group, length(feas))
    loaded   = feas[1:k]
    unloaded = feas[(length(feas)-k+1):end]
    loaded_idx   = Set(r.combo_idx for r in loaded)
    unloaded_idx = Set(r.combo_idx for r in unloaded)

    out = NamedTuple[]
    for r in rows
        group = r.combo_idx in loaded_idx ? "loaded" :
                r.combo_idx in unloaded_idx ? "unloaded" : "unselected"
        role = if group == "unselected"
            "unselected"
        elseif group == "loaded" && hardest_available && r.combo_idx == loaded[1].combo_idx
            "hardest_available"
        else
            "keep"
        end
        push!(out, merge(r, (; group = group, role = role)))
    end
    return out, hardest_available
end

# -----------------------------------------------------------------------------
function main()
    run_dir     = length(ARGS) >= 1 ? ARGS[1] : "trajectory_files_run_0p5_main"
    mu          = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
    n_per_group = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

    rows, hardest_available = select_stratified(run_dir, mu; n_per_group=n_per_group)

    loaded   = filter(r -> r.group == "loaded",   rows)
    unloaded = filter(r -> r.group == "unloaded", rows)
    n_tangent = length(rows)
    n_feasible = count(r -> r.feasible, rows)

    println("Screened $n_tangent tangent-mode ellipse combos ($n_feasible feasible).")
    println("LOADED   (n=$(length(loaded))):   u_peak = ", extrema([r.u_peak for r in loaded]),
            "  combo_idx = ", sort([r.combo_idx for r in loaded]))
    println("UNLOADED (n=$(length(unloaded))): u_peak = ", extrema([r.u_peak for r in unloaded]),
            "  combo_idx = ", sort([r.combo_idx for r in unloaded]))
    if hardest_available
        @warn "No feasible combo reached u_peak >= $U_PEAK_STRESS_TARGET; " *
              "top LOADED combo labelled 'hardest_available' (subset_manifest.json convention)."
    end
    loaded_range   = extrema([r.u_peak for r in loaded])
    unloaded_range = extrema([r.u_peak for r in unloaded])
    if loaded_range[1] <= unloaded_range[2]
        @warn "LOADED and UNLOADED u_peak ranges OVERLAP ($loaded_range vs $unloaded_range) — " *
              "stratification is NOT cleanly separated; revisit before proceeding (brief §7.1)."
    end

    outdir = joinpath(@__DIR__, "reports")
    mkpath(outdir)
    outpath = joinpath(outdir, "ellipse_selection.json")
    payload = Dict(
        "mu" => mu, "run_dir" => run_dir, "psi_mode_filter" => "tangent",
        "n_per_group" => n_per_group, "hardest_available" => hardest_available,
        "v_ceiling_mps" => V_CEILING_MPS, "u_peak_stress_target" => U_PEAK_STRESS_TARGET,
        "loaded_u_peak_range" => collect(loaded_range),
        "unloaded_u_peak_range" => collect(unloaded_range),
        "combos" => [Dict{String,Any}(String(k) => v for (k, v) in pairs(r)) for r in rows],
    )
    open(outpath, "w") do io
        JSON.print(io, payload, 2)
    end
    println("Wrote $outpath")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
