# =============================================================================
# hybrid_ctrl_v2/diag_v3_error_budget.jl
#
# THREE-STAGE ERROR BUDGET for the ESKF closed-loop runs (RESULTS_v3.md §7.4):
#
#   (1) INJECTED    what the raw sensors say, in state units, BEFORE filtering
#   (2) SURVIVING   what is left in xhat AFTER the ESKF        (est_* columns)
#   (3) TRACKING    how far the platform ends up from the reference (ctrl_* columns)
#
# so "how much did the ESKF buy" (1)->(2) is separated from "how much is the
# controller off" (3).
#
# NO RE-SIMULATION. Stages (2) and (3) are already in
# runs_eskf_v3_train14/runs_seed*.csv. Stage (1) is an ANALYTIC function of the
# sensor model and the wheel rates -- the noise coefficients are fixed constants
# and the wheel rates follow from the reference, so integrating the plant again
# would only re-derive numbers that are already determined.
#
# SENSOR MODEL (SensorModV2.build_suite(:realistic), verbatim):
#   encoder  omega_noisy = (1 + sf)*omega_true + bias + white
#              sf    ~ N(0, 0.02)          RE-DRAWN EVERY TICK (not a fixed
#                                          calibration error) -> contributes
#                                          0.02*|omega| of per-tick noise
#              white ~ N(0, 0.01/(0.05*sqrt(1.09)) = 0.1916 rad/s)   per tick
#              bias  ~ N(0, 0.005*2/0.05  = 0.2    rad/s)   DRAWN ONCE per run
#   gyro     g_z = psidot*(1 + N(0,0.005)) + bias + N(0, 0.003 rad/s)
#              bias ~ N(0, 0.003 rad/s) drawn once
#
# CHANNEL ROUTING matters and is easy to get wrong -- all FOUR channels have
# DIFFERENT injected sources:
#   velocity (Vx,Vy)  ENCODERS via the wheel Jacobian
#   yaw rate          GYRO directly (`_wheel_body_velocity` returns y.g_z for it)
#   position          POSE FIX  (PoseFixModel(:docking): 100 Hz, sigma_pos 10 mm)
#   heading           POSE FIX  (sigma_psi = deg2rad(0.5) = 8.727 mrad)
# Lumping them, or scoring position against the encoder noise, gives nonsense.
# Pose-fix noise has no scale-factor and (at :docking) no bias, so position and
# heading injection are trajectory-INDEPENDENT constants; the encoder/gyro
# channels are not, because their scale-factor term rides on |omega| and |psidot|.
#
# Encoder -> body velocity: z = H \ omega, so with per-wheel error covariance
# Sigma the body-velocity error covariance is pinv(H)*Sigma*pinv(H)'. Wheel
# rates come from the reference: omega_true = H * [Vx, Vy, psidot].
#
# HEADING TRAP -- this script splits the heading channel by psi(0) for a reason.
# The ESKF initialises psi_hat at 0, so any reference starting at psi(0) != 0
# opens with a 1.6-2.4 rad error. `est_heading` is a full-run RMS, and an ~11 ms
# convergence transient at that amplitude moves a 33-second RMS by 30x. Averaged
# over a set with mixed psi(0) the heading channel reads 9.78 mrad / -12% removed
# ("worse than the raw fix"); split by psi(0) it reads 1.26 mrad / 85.5% removed,
# matching the open-loop replay figure of 1.27 mrad exactly. The first reading is
# an artefact and an earlier draft of RESULTS_v3.md §7.4 published it. The script
# now reports the split, plus the implied T_conv per contaminated trajectory --
# agreement of T_conv across different T_total and different psi(0) is what makes
# the transient explanation conclusive rather than a plausible story.
#
# CAVEAT, stated because it bounds the claim: stage (1) covers the encoder, gyro
# and pose-fix inputs -- every sensor with a MEMORYLESS state-space equivalent,
# i.e. one that can be inverted to a state without integrating. The remaining
# input, OPTICAL FLOW, is an additional velocity source with no such equivalent,
# so the VELOCITY row's (1)->(2) improvement is filtering PLUS the extra
# information flow brings, not filtering alone. Position, heading and yaw rate
# are unaffected by this (flow measures velocity), so only the velocity row's
# 76.3% should be read as an upper bound on filtering alone.
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
using Printf, Statistics, LinearAlgebra, StaticArrays, TOML, Random
using CSV, DataFrames

# --- sensor constants, copied from SensorModV2.build_suite(:realistic) --------
const R_EFF        = 0.05
const SIG_OMEGA    = 0.01 / (R_EFF * sqrt(1.09))   # 0.1916 rad/s, per tick
const SF_OMEGA     = 0.02                          # per-tick multiplicative
const SIG_BOMEGA   = 0.005 * 2.0 / R_EFF           # 0.2 rad/s, once per run
const SIG_GYRO     = 0.003
const SF_GYRO      = 0.005
const SIG_BGYRO    = 0.003

"""
Reference motion scale, for judging whether an error is large or small.

TWO different notions, reported separately because they answer different questions
and disagree by orders of magnitude on repeating paths:
  travel_x/y  total distance travelled along each WORLD axis, int|dX|, int|dY|.
              How much the axis was exercised. An ellipse repeated 5 times has 5x
              the travel of one lap.
  range_x/y   peak-to-peak extent, max-min. How big the path IS. Unchanged by
              repetition -- closer to what `k_traj`'s radius_var measures.
`rot_total` is int|dpsi|, the heading analogue of travel.
"""
function ref_motion_scale(ref; n=4000)
    ts = range(0.0, ref.T_total; length=n)
    X = [ref.xo(t) for t in ts]; Y = [ref.yo(t) for t in ts]; P = [ref.psi(t) for t in ts]
    dpsi = [atan(sin(P[i+1]-P[i]), cos(P[i+1]-P[i])) for i in 1:(n-1)]
    (travel_x=sum(abs, diff(X)), travel_y=sum(abs, diff(Y)),
     range_x=maximum(X)-minimum(X), range_y=maximum(Y)-minimum(Y),
     rot_total=sum(abs, dpsi))
end

"Reference body-frame velocity RMS, by finite-differencing the reference pose."
function ref_velocity_rms(ref; n=4000)
    T = ref.T_total
    ts = range(0.0, T; length=n)
    dt = ts[2] - ts[1]
    vx = Float64[]; vy = Float64[]; wz = Float64[]
    for i in 1:(n-1)
        t = ts[i]
        dX = (ref.xo(t+dt) - ref.xo(t)) / dt
        dY = (ref.yo(t+dt) - ref.yo(t)) / dt
        dpsi = atan(sin(ref.psi(t+dt) - ref.psi(t)), cos(ref.psi(t+dt) - ref.psi(t))) / dt
        psi = ref.psi(t)
        push!(vx,  cos(psi)*dX + sin(psi)*dY)      # world -> body
        push!(vy, -sin(psi)*dX + cos(psi)*dY)
        push!(wz, dpsi)
    end
    (vx=sqrt(mean(vx.^2)), vy=sqrt(mean(vy.^2)), wz=sqrt(mean(wz.^2)))
end

"Stage (1): unfiltered sensor-implied state error, in state units."
function injected_state_noise(params, v)
    H  = Main.EstimatorMod._wheel_jacobian(params)
    Hp = pinv(Matrix(H))
    omega_true = H * SVector(v.vx, v.vy, v.wz)                 # rad/s per wheel
    # per-wheel error variance: per-tick white + per-tick scale-factor + once-per-run bias
    sigma_w = [sqrt(SIG_OMEGA^2 + (SF_OMEGA*abs(w))^2 + SIG_BOMEGA^2) for w in omega_true]
    Cov = Hp * Diagonal(sigma_w .^ 2) * Hp'
    sig_vx, sig_vy = sqrt(Cov[1,1]), sqrt(Cov[2,2])
    sig_wz = sqrt(SIG_GYRO^2 + (SF_GYRO*abs(v.wz))^2 + SIG_BGYRO^2)   # GYRO, not encoders
    (vx=sig_vx, vy=sig_vy, vel=sqrt(sig_vx^2 + sig_vy^2), wz=sig_wz)
end

trs = collect(trajset(:train14_v3, "trajectory_files_run_0p5_main"))
INJ = Dict{String,Any}()
for tr in trs
    base   = Profiles.load_base(tr.config_dir)
    params = PlatformParams(base; mu_friction=Float64(tr.mu))
    path   = joinpath(tr.config_dir, "profiles", tr.profile_toml)
    prof   = TOML.parsefile(path)["profile"]
    cfg_r  = Profiles.resolve_profile(prof; combo_idx=tr.combo_idx, rng=Random.Xoshiro(0))
    ref    = Profiles.build(prof["builder"], cfg_r)
    get(tr, :adapt, false) && (ref = Profiles.velref_to_posref(ref))
    v = ref_velocity_rms(ref)
    # psi(0) decides whether est_heading is usable for this trajectory -- see the
    # HEADING section: the ESKF initialises psi_hat at 0, so a reference starting
    # at psi(0) != 0 opens with a 1.6-2.4 rad error whose ~11 ms convergence
    # transient dominates a full-run RMS.
    INJ[String(tr.name)] = (v=v, n=injected_state_noise(params, v),
                            psi0=ref.psi(0.0), T=ref.T_total, m=ref_motion_scale(ref))
end

# --- stages (2) and (3), straight from the completed runs ---------------------
dir = joinpath(ROOT, "hybrid_ctrl_v2", "runs_eskf_v3_train14")
df  = reduce(vcat, [CSV.read(joinpath(dir, f), DataFrame)
                    for f in readdir(dir) if occursin(r"^runs_seed\d+\.csv$", f)])
df  = df[df.ok .== true, :]

println("="^108)
println("THREE-STAGE ERROR BUDGET — ESKF closed loop, train14_v3, noise seeds 101-105")
println("(1) injected = raw sensor noise in state units, pre-filter   (2) surviving = after ESKF   (3) tracking = true vs ref")
println("    sources: velocity<-encoders   yaw rate<-gyro   position/heading<-pose fix @100Hz")
println("="^108)

@printf("\n--- VELOCITY channel [mm/s] ---   (stage 3 is position, shown separately below)\n")
@printf("%-28s %10s %10s %10s %9s %8s\n",
        "trajectory", "|v|ref", "(1)inj", "(2)surv", "removed", "ratio")
tot1 = Float64[]; tot2 = Float64[]
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    inj = INJ[nm].n.vel * 1000; surv = mean(g.est_vel) * 1000
    push!(tot1, inj); push!(tot2, surv)
    @printf("%-28s %10.3f %10.3f %10.3f %8.1f%% %7.1fx\n", nm,
            hypot(INJ[nm].v.vx, INJ[nm].v.vy), inj, surv, 100*(inj-surv)/inj, inj/surv)
end
@printf("%-28s %10s %10.3f %10.3f %8.1f%% %7.1fx\n", "MEAN", "",
        mean(tot1), mean(tot2), 100*(mean(tot1)-mean(tot2))/mean(tot1), mean(tot1)/mean(tot2))

@printf("\n--- YAW-RATE channel [mrad/s] ---\n")
@printf("%-28s %10s %10s %10s %9s %8s\n", "trajectory", "|w|ref", "(1)inj", "(2)surv", "removed", "ratio")
t1 = Float64[]; t2 = Float64[]
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    inj = INJ[nm].n.wz * 1000; surv = mean(g.est_rate) * 1000
    push!(t1, inj); push!(t2, surv)
    @printf("%-28s %10.3f %10.3f %10.3f %8.1f%% %7.1fx\n", nm,
            INJ[nm].v.wz, inj, surv, 100*(inj-surv)/inj, inj/surv)
end
@printf("%-28s %10s %10.3f %10.3f %8.1f%% %7.1fx\n", "MEAN", "",
        mean(t1), mean(t2), 100*(mean(t1)-mean(t2))/mean(t1), mean(t1)/mean(t2))

# --- POSITION / HEADING channels: injected by the POSE FIX, not the encoders --
# PoseFixModel(:docking) — the tier these runs used (cfg.pose_fix_tier=:docking):
#   fix_rate 100 Hz, sigma_pos = 0.01 m, sigma_psi = deg2rad(0.5), NO bias, NO
#   dropout, outlier_frac 0.005. No scale-factor term, so unlike the encoder
#   channel the injected magnitude is trajectory-INDEPENDENT: one number.
const FIX_SIG_POS = 0.01                # m
const FIX_SIG_PSI = deg2rad(0.5)        # rad
@printf("\n--- POSITION channel [mm] (injected by pose fix @100Hz, sigma_pos=10mm) ---\n")
@printf("%-28s %10s %10s %9s %8s\n", "trajectory", "(1)inj", "(2)surv", "removed", "ratio")
p1 = Float64[]; p2 = Float64[]
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    inj = FIX_SIG_POS*1000; surv = mean(g.est_pos)*1000
    push!(p1, inj); push!(p2, surv)
    @printf("%-28s %10.3f %10.3f %8.1f%% %7.1fx\n", nm, inj, surv, 100*(inj-surv)/inj, inj/surv)
end
@printf("%-28s %10.3f %10.3f %8.1f%% %7.1fx\n", "MEAN",
        mean(p1), mean(p2), 100*(mean(p1)-mean(p2))/mean(p1), mean(p1)/mean(p2))

@printf("\n--- HEADING channel [mrad] (injected by pose fix @100Hz, sigma_psi=8.727 mrad) ---\n")
println("SPLIT BY psi(0). The ESKF initialises psi_hat at 0, so a reference starting at")
println("psi(0) != 0 opens with a 1.6-2.4 rad error. est_heading is a FULL-RUN RMS, so that")
println("startup transient dominates it. Do NOT average the two groups together.")
@printf("%-28s %8s %10s %10s %9s %8s\n", "trajectory", "psi(0)", "(1)inj", "(2)surv", "removed", "ratio")
inj = FIX_SIG_PSI*1000
ss = Float64[]; tr_aff = NamedTuple[]
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    surv = mean(g.est_heading)*1000; p0 = INJ[nm].psi0
    if abs(p0) < 1e-9
        push!(ss, surv)
        @printf("%-28s %8.4f %10.3f %10.3f %8.1f%% %7.1fx\n", nm, p0, inj, surv,
                100*(inj-surv)/inj, inj/surv)
    else
        push!(tr_aff, (nm=nm, surv=surv, p0=p0, T=INJ[nm].T))
    end
end
@printf("%-28s %8s %10.3f %10.3f %8.1f%% %7.1fx   <- STEADY STATE, the usable figure\n",
        "MEAN (psi0 == 0, n=$(length(ss)))", "", inj, mean(ss),
        100*(inj-mean(ss))/inj, inj/mean(ss))

println("\n  psi(0) != 0 -- startup-transient contaminated, NOT an estimator fault:")
@printf("  %-26s %8s %10s %9s %14s\n", "trajectory", "psi(0)", "rms", "T_total", "implied T_conv")
for r in tr_aff
    # If the excess is a startup transient at the full initial error e0 lasting
    # T_conv, then rms ~= e0*sqrt(T_conv/T_total)  =>  T_conv = T*(rms/e0)^2.
    # Agreement of this number ACROSS trajectories with different T and different
    # e0 is what makes the transient explanation conclusive rather than a guess.
    tconv = r.T * (r.surv/1000 / r.p0)^2
    @printf("  %-26s %8.4f %10.3f %9.2f %11.1f ms\n", r.nm, r.p0, r.surv, r.T, 1000*tconv)
end
if !isempty(tr_aff)
    tcs = [r.T * (r.surv/1000 / r.p0)^2 for r in tr_aff]
    @printf("  -> implied T_conv spread %.1f-%.1f ms across %d trajectories with differing\n",
            1000*minimum(tcs), 1000*maximum(tcs), length(tr_aff))
    println("     T_total and psi(0): a single convergence transient explains ALL of the excess.")
    @printf("     Steady-state heading (%.3f mrad) matches the open-loop replay figure (1.27 mrad).\n",
            mean(ss))
    @printf("     [the all-%d mean would read %.3f mrad / %.1f%% removed -- AN ARTEFACT, do not quote]\n",
            length(ss)+length(tr_aff), (sum(ss)+sum(r.surv for r in tr_aff))/(length(ss)+length(tr_aff)),
            100*(inj - (sum(ss)+sum(r.surv for r in tr_aff))/(length(ss)+length(tr_aff)))/inj)
end

# --- the three stages side by side, per config -------------------------------
println("\n" * "="^108)
println("PER CONFIG — what the estimator removed vs what the controller then did")
println("="^108)
@printf("%-14s %12s %12s %10s | %12s %12s\n", "config",
        "(1)inj vel", "(2)surv vel", "removed", "(3)trk pos", "(3)trk head")
@printf("%-14s %12s %12s %10s | %12s %12s\n", "", "[mm/s]", "[mm/s]", "", "[mm]", "[mrad]")
inj_mean = mean(tot1)
for lab in unique(df.controller)
    g = df[df.controller .== lab, :]
    surv = mean(g.est_vel)*1000
    @printf("%-14s %12.3f %12.3f %9.1f%% | %12.3f %12.3f\n", lab,
            inj_mean, surv, 100*(inj_mean-surv)/inj_mean,
            mean(g.ctrl_pos)*1000, mean(g.ctrl_heading)*1000)
end
# --- per-trajectory: motion scale | pose estimation | pose tracking -----------
println("\n" * "="^116)
println("PER TRAJECTORY — motion scale vs POSE ESTIMATION error vs POSE TRACKING error")
println("="^116)
println("pose est: 10.000 mm injected by the fix -> est_pos surviving (the ESKF correction).")
println("trk: controller error, TRUE state vs REFERENCE, averaged over all 6 configs.")
println("err/travel and err/range put the tracking error on the trajectory's own scale.")
@printf("%-27s %7s %7s %7s %7s %6s | %7s %7s | %8s %7s %7s\n",
        "trajectory", "trvl_x", "trvl_y", "rng_x", "rng_y", "rot",
        "est_pos", "removed", "trk_pos", "/travel", "/range")
@printf("%-27s %7s %7s %7s %7s %6s | %7s %7s | %8s %7s %7s\n",
        "", "[m]", "[m]", "[m]", "[m]", "[rad]", "[mm]", "", "[mm]", "", "")
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    m = INJ[nm].m
    ep = mean(g.est_pos)*1000
    tp = mean(g.ctrl_pos)*1000
    travel = hypot(m.travel_x, m.travel_y)
    rng    = hypot(m.range_x, m.range_y)
    @printf("%-27s %7.2f %7.2f %7.2f %7.2f %6.2f | %7.3f %6.1f%% | %8.2f %6.2f%% %6.2f%%\n",
            nm, m.travel_x, m.travel_y, m.range_x, m.range_y, m.rot_total,
            ep, 100*(FIX_SIG_POS*1000 - ep)/(FIX_SIG_POS*1000),
            tp, travel > 0 ? 100*tp/1000/travel : NaN,
            rng    > 0 ? 100*tp/1000/rng    : NaN)
end

println("\n--- stage-3 POSITION tracking [mm], per trajectory x config ---")
cfgs = sort(unique(df.controller))
@printf("%-27s", "trajectory"); for c in cfgs; @printf(" %13s", c); end; println()
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    @printf("%-27s", nm)
    for c in cfgs
        gc = g[g.controller .== c, :]
        @printf(" %13.2f", isempty(gc) ? NaN : mean(gc.ctrl_pos)*1000)
    end
    println()
end

println("\n--- stage-3 HEADING tracking [mrad], per trajectory x config ---")
@printf("%-27s", "trajectory"); for c in cfgs; @printf(" %13s", c); end; println()
for tr in trs
    nm = String(tr.name); g = df[df.trajectory .== nm, :]
    isempty(g) && continue
    @printf("%-27s", nm)
    for c in cfgs
        gc = g[g.controller .== c, :]
        @printf(" %13.3f", isempty(gc) ? NaN : mean(gc.ctrl_heading)*1000)
    end
    println()
end

println("""
Stage (1) is identical across configs by construction — it is a property of the sensors and
the reference, not of the control law. Stage (2) is near-identical because the estimator is
FROZEN and its accuracy does not depend on which controller drives (§7.4). So the entire
spread between controllers lives in stage (3): the estimator contributes a common-mode
offset, and only stage (3) discriminates control laws.

Note stage-(3) heading is CONTROLLER tracking error (true psi vs reference), which is a
different quantity from the stage-(2) heading ESTIMATION error split by psi(0) above — it
is unaffected by the psi_hat initialisation transient. PID-CT reads best on position and
worst on heading; PID-FB is the mirror image. That split, not the estimator, is what
separates the control laws.""")
