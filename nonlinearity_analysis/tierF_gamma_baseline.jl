#!/usr/bin/env julia
# =============================================================================
# tierF_gamma_baseline.jl — baseline closed-loop runs for the brute-force
# gamma (Eker-Nikolaou incremental norm) computation.
#
# Re-runs the best archived PID-CT (0.9 campaign, clean-tuned) with ORACLE-
# CLEAN feedback over the full trajectory set (train14_v3 + test_v3 + the two
# docking references) and stores the complete probe series per trajectory:
#   t, full plant state, estimator output (oracle passthrough), v_cmd,
#   reference (world pose + velocities).
# Everything downstream (windowing, saturation flags, a_ff decomposition,
# epsilon_0 construction, replay probes) is derived OFFLINE from these files.
#
# Rates (same stack as the campaign): FBDF reltol 1e-9 dtmax 1e-3, controller
# callbacks 1 kHz, probe saveat 500 Hz.
#
# Output: nonlinearity_analysis/results/gamma_baseline/<traj>.jld2
# Usage:  julia --project=. -t 2 nonlinearity_analysis/tierF_gamma_baseline.jl
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)
using Printf, JSON, StaticArrays, JLD2

# eval_controllers_eskf_v3.jl FIRST and ALONE (load-bearing include order, see
# eval_v3_s100.jl header): transitively provides tune_controller_v2, trajsets,
# sensors/estimators v2+v3, scheduler_v2, and run_controller_v2.
include(joinpath(ROOT, "hybrid_ctrl_v2", "eval_controllers_eskf_v3.jl"))

const RUN_DIR = "trajectory_files_run_0p5_main"
const OUTDIR  = joinpath(ROOT, "nonlinearity_analysis", "results", "gamma_baseline")
mkpath(OUTDIR)

# ---- trajectory set: train14_v3 + test_v3 + docking ----------------------------
const DOCKING = [
    (name = "docking_a",    profile_toml = "docking_mu_0p5.toml",
     combo_idx = 1, ref_type = :posref, mu = 0.5, config_dir = RUN_DIR,
     run_mode = :pose, adapt = false, role = :step_hold),
    (name = "docking_step", profile_toml = "docking_step_mu_0p5.toml",
     combo_idx = 1, ref_type = :posref, mu = 0.5, config_dir = RUN_DIR,
     run_mode = :pose, adapt = false, role = :step_hold),
]

const SMOKE = "--smoke" in ARGS
const SMOKE_TRAJS = ["octagon_easy", "coupled_vomega_stress", "docking_step"]

function all_trajs()
    trs = collect(Any, TrajSetsMod.trajset(:train14_v3, RUN_DIR))
    append!(trs, collect(TrajSetsMod.trajset(:test_v3, RUN_DIR)))
    append!(trs, DOCKING)
    SMOKE && return filter(tr -> string(tr.name) in SMOKE_TRAJS, trs)
    return trs
end

# ---- best archived PID-CT (0.9) clean-tuned config -----------------------------
function best_pidct_config()
    paths = [joinpath(ROOT, "hybrid_ctrl_v2", "runs_pid_v3", "seed$s",
                      "pid_v2_ct_clean", "best_config.json") for s in 1:5]
    cfgs = filter(isfile, paths)
    isempty(cfgs) && error("no pid_v2_ct_clean best_config.json found")
    scored = sort([(JSON.parsefile(c)["best_score"], c) for c in cfgs]; by = first)
    return scored[1][2]
end

function kw_from_config(path)
    g = JSON.parsefile(path)["best_gains"]
    lim = Main.physical_limits_with_margin(0.9)
    return (lam_inner_x = g["lam_inner_x"], lam_inner_y = g["lam_inner_y"],
            lam_inner_psi = g["lam_inner_psi"], N = g["N"],
            feedforward = g["feedforward"], lim = lim)
end

function main_baseline()
    cfg_path = best_pidct_config()
    kw = kw_from_config(cfg_path)
    @printf("PID-CT baseline config: %s\n", cfg_path)
    flush(stdout)

    for tr in all_trajs()
        out = joinpath(OUTDIR, string(tr.name) * ".jld2")
        if isfile(out)
            println("skip $(tr.name) (already stored)"); continue
        end
        t0 = time()
        try
            _, _, po = build_controller_v2(:pid, kw)
            probe, ref, mode, bus = run_controller_v2(:pid, :clean, tr;
                                                      pid_o = po, seed = 4)
            t    = [p.t for p in probe]
            U    = reduce(hcat, [Vector(p.u) for p in probe])
            H    = reduce(hcat, [Vector(p.xhat) for p in probe])
            VC   = reduce(hcat, [Vector(p.v_cmd) for p in probe])
            jldsave(out; traj = string(tr.name), mode = string(mode),
                    t, state = U, xhat = H, v_cmd = VC,
                    Xo_ref = [ref.xo(tt) for tt in t],
                    Yo_ref = [ref.yo(tt) for tt in t],
                    psi_ref = [ref.psi(tt) for tt in t],
                    Vxo_ref = [ref.Vxo(tt) for tt in t],
                    Vyo_ref = [ref.Vyo(tt) for tt in t],
                    om_ref  = [ref.om(tt) for tt in t],
                    T_total = t[end])
            Main.SchedulerMod.clear_probe_log!(bus)
            @printf("done %-28s N=%6d T=%6.1fs (%.0fs wall)\n",
                    tr.name, length(t), t[end], time() - t0)
        catch err
            @warn "baseline run failed" traj = tr.name err
        end
        flush(stdout)
    end
    println("all baselines stored in $OUTDIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_baseline()
end
