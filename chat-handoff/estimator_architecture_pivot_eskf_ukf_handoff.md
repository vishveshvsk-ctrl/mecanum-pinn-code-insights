# Handoff — Estimator architecture pivot: IMM tuning exhausted → ESKF / UKF (2026-07-23)

## Why this document exists (the intent)

The 11-state IMM Kalman estimator (`IMMKalmanEstimator`, `:kalman_imm`) has been
tuned to its structural floor and **did not reach the proposed absolute
tolerances**: position ~1 cm, velocity ~1 mm/s, heading ~0.01 rad (user spec:
absolute errors, NOT signal-normalized NRMSE, so quiet trajectories aren't
penalized). This handoff records what was achieved, the diagnosis of why tuning
stopped paying off, the literature check on alternatives, and the two agreed
paths forward (ESKF, UKF).

## State of play (frozen artifacts)

- Frozen winner: `runs_estimator/frozen/best_config.json`
  (backups in same dir: `best_config_imm10_prev.bak.json`,
  `best_config_legacy_kalman.bak.json`)
- Tuning chain: dxNES budget-400 pinned run (plateau 294.46 @ eval 102, flat
  160 evals, stopped at 270) → warm-started Kriging/DYCORS BO (budget 150,
  best 286.66) → top-10 re-rank on held-out seeds 44,45 → froze candidate 10
  (search 297.4, robust 320.7, lowest variance) instead of the peak
  (286.7, robust 375.3 — knife-edge).
- Winners' diagnostics (seed 42, absolute RMS):
  - velocity: octagon 3.9/10.9 mm/s (4× better after whitelist combo pinning),
    docking 5.4/12.9, ellipse 61/54, long_circle (held-out) 2.3 mm/s,
    multisine ψ̇ 84 mrad/s (worst channel)
  - yaw-rate: docking 8.3 mrad/s AT TARGET; long_circle 11.9
  - pose: ellipse 381 cm / 1.41 rad (BISTABLE), docking 64 cm / 0.019 rad
    (heading at target)
- Objective: `estimator_objective_abs` (tuning/objectives.jl) — score = sum of
  RMS-error/tolerance ratios; ≈5 means "at target everywhere"; winner ≈ 290–320
  ⇒ ~40–60× off, dominated by pose terms.
- Subset: whitelist-pinned combos (diagnostics_combined.csv keep-list +
  envelope feasibility from docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope):
  ellipse c24 (a=1.5 band), octagon c2 (0.36 m/s < 0.379 cap @ μ=0.3), others c1.
  combo_idx now flows through subset.jl/harness/tune/rerank/compare (deterministic).

## Diagnosis — why the IMM cannot be tuned further

1. **Pose bistability is structural, not parametric.** The SAME config tracks
   ellipse on some seeds and diverges on others (score pose terms are
   mathematically incompatible with a uniformly-tracked run). Mechanism:
   covariance inconsistency in slip — pose block of P grows only via F-coupling
   + tiny pose_Qn, so in slip the filter is overconfident about pose, the NIS
   fix-gate (correctly, given the wrong covariance) rejects the pose fixes that
   would rescue it, and pose free-runs. Which side a run lands on is chaos/seed
   dependent (this also caused the 294-vs-308 same-config re-score discrepancy).
2. **All top-10 tuned candidates share it** (re-rank std 13–42 across seeds).
   Two different optimizers (dxNES global, Kriging/DYCORS local) agree on the
   basin ⇒ the ~290–320 composite is the floor of this architecture+Q-structure,
   not of the search.
3. Heading channels prove the sensing is sufficient (docking ψ at 0.019 rad,
   ψ̇ at 8 mrad/s with fix σψ=0.009 rad): information for 1 cm exists; the
   filter's consistency machinery is what fails.
4. Verdict: KF line is NOT dead, but hyperparameter tuning of the current
   IMM-with-patches is exhausted. The fix must be architectural: honest
   covariance behavior (error-state formulation or sigma points) + slip-aware
   process noise baked into the structure.

## Noise-covariance (Q/R) tuning — methods survey (user asked)

- Black-box metric optimization = what we do (= "discriminative filter
  training", Abbeel et al. RSS 2005, from memory).
- Innovation-based adaptive estimation (IAE, Mehra '70-'72 lineage);
  modern: Akhlaghi et al. 2017 (arXiv:1702.00884).
- Covariance matching closed loops: Jiang et al. 2021 (arXiv:2112.12082).
- ML/EM batch identification (Åkesson et al., from memory).
- MMAE/IMM (ours), Variational-Bayes ADF (Särkkä), learned (KalmanNet 2021).
- Practical combo: offline optimization + light online IAE; pure online
  Q-estimation on 11 states with slip transients tends to self-inflate.

## Literature check: UKF & PF on Mecanum platforms

UKF (usable at our layer):
- MDPI 2025: EKF on SO(2) Lie group, Mecanum, wheel+IMU, real-robot validated
  (mdpi.com/2673-4591/115/1/3) — supports the geometric/error-state direction.
- WPILib MecanumDrivePoseEstimator — production UKF (encoder+gyro+vision) on
  Mecanum drivetrains (frcdocs.wpi.edu, 2021 docs).
- Mutti & Pedrocchi 2021 (SPIE 11785) — UKF kinematic calibration, 4-Mecanum.
- hrčak EKF-vs-UKF real-robot comparison (hrcak.srce.hr/file/10284): UKF
  smoother/more accurate. Nugraha et al. 2025 (arXiv:2509.22693): UKF better
  for stronger nonlinearities, more compute.

PF (right tool one level up, wrong tool here):
- Standard usage is map-based AMCL on Mecanum bases (Neaz et al. 2023,
  PMC10056260; typical ROS packages). Nagatani IROS 2000 fused Mecanum odometry
  + visual dead-reckoning by ML.
- Nobody uses PF for 1 kHz wheel/IMU odometric fusion: our posterior is
  unimodal near-Gaussian, PF costs 100s of particles/tick and starves exactly
  during slip transients. Reserve PF for a future lidar/map localization layer.

## Two steps forward

### Step 1 — ESKF (error-state Kalman filter), primary path

Port the 11-state content to nominal+error-state form:
- Nominal integrator: full nonlinear kinematics (IMU accel − bias, Coriolis,
  rotating-frame pose integration), no filter assumptions.
- Error state δx (11-dim: δV, δψ̇, δψ, δX, δY, δbx, δby, δsx, δsy, δbg):
  predict P with the error-state Jacobian (re-evaluated at the current nominal
  each tick), update with wheel(v+s)/gyro(ψ̇+bg) measurements, inject δx →
  nominal, reset δx ← 0, Joseph-form P.
- Slip-aware Q STRUCTURE (the actual medicine): pose-error block of Q inflated
  by slip activity (new knob pose_slip_gain); wheel-R inflated in slip; slip
  states frozen in grip (grip_slip_scale), free in slip (slip_Qn); accel-bias
  learning off in slip, gyro-bias learning always on.
- NIS-gated pose fix (χ²₃,₉₉.₉=14.16), velocity updated via cross-covariance.
- Simplify: DROP the IMM wrapper initially (its mode-switching was compensating
  for the missing covariance discipline); keep slip/gyro-bias states. If the
  single adaptive ESKF underperforms the frozen IMM head-to-head, re-add modes.
- Reuse all machinery: param_space (new eskf_param_space), harness, objective,
  optimizer backends, rerank, compare. Estimator symbol :eskf.
- Acceptance: beat frozen IMM on the SAME protocol (abs objective, seeds
  42,43 search + 44,45 re-rank); target trajectory: pose RMS < 10 cm ellipse /
  < 3 cm docking, no bistability across seeds; velocity ≤ 2× current.

### Step 2 — UKF (sigma-point), parallel/backup path

- Same state/measurement/Q-structure; replace Jacobian predict + linear update
  with 2n+1 = 23 sigma-point propagations through the exact closed-form
  transition (cost trivial here — no ODE solves, 23 closed-form evals/tick).
- UKF attacks the same linearization disease as ESKF without a rewrite of the
  state meaning; slightly heavier per tick, no nominal/reset bookkeeping.
- Implement as :ukf behind the same estimator interface; tune with the same
  warm-started BO; compare three-way (frozen IMM vs ESKF vs UKF).
- Either path that fails to beat the frozen IMM head-to-head is abandoned, not
  iterated — the frozen IMM remains the deployable fallback.

## Reproduce / resume pointers

- Code: hybrid_ctrl/estimators.jl (IMM section ~431-763), tuning/{param_space,
  harness,objectives,optimizer_bbo}.jl, tune_estimator.jl (--optimizer,
  --obj-seeds, --warm-start), rerank_topk.jl, compare_estimators.jl.
- Bats: tune_kalman_imm_abs2_dxnes.bat (dxNES), tune_kalman_imm_abs2_bo.bat
  (BO warm start), rerank_abs2.bat, compare_estimators.bat.
- Traces: runs_estimator_abs2*/**/fitness_trace.csv + checkpoint_best.json
  (crash-proof incumbent, written every improvement).
- Objective/eval-noise note: ~few-% run-to-run wobble from BLAS thread
  scheduling amplified by slip chaos — never read meaning into <5% score diffs.
- Cron monitors used for long runs (:09/:29/:49 cadence); keep_awake.py before
  any run >20 min.
