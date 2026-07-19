# Velocity Front-End Drift Audit — Slip-Gating Design Question

Continues work in the mecanum_pinn_head IMECE-2026 digital-twin project: the
velocity front-end audit specified in `instructions/frontend-drift-audit.md`
(a pre-training gate for Observer-v2's sensor-real input contract, per
`instructions/observer-gamma-only-5phase-retrain.md` §8 step 3(d)). The audit
itself is DONE and its outputs are final; this handoff is about a follow-on
design question raised while interpreting the results — whether the fixed
complementary-filter crossover should be replaced with a slip-gated one.

## Context this task depends on

**Code** (`code_insights/tools_accel/`): `accel_dynamics.py` (exact-dynamics
EOM), `make_accel_sidecars.py` (`load_with_accel` join contract),
`imu_observable.py` (accel/gyro synthesis + `SensorNoiseSpec` + decimation),
`odometry.py` (wheel-only kinematic least-squares `V_odom`), `comp_filter.py`
(mechanization + complementary blend — THE file any slip-gating change would
land in), `frontend_audit.py` (sweep CLI + report generator).

**Audit outputs** (`code_insights/observer_v1_py/report_frontend/`):
 `acceptance_table.csv`,
`ACCEPTANCE.md`, 4 PNG figures. Full approved fleet, no sampling: 5,345 files
across 8 profiles (spin_creep 1692, octagon 1422, long_circle 762,
coupled_vomega 647, ellipse 288, spiral_orbit 204, multisine_50%cap 165,
multisine_75%cap 165), swept over 120 cells/file (2 integrators × 5
crossovers [0.2/0.5/1/2/5 Hz] × 2 rates [500/2000 Hz] × 2 noise stages × 3
modes) + a cold-start v0 supplement.
`frontend_audit.csv` (6,841,600 rows), moved to "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\data\IMU_frontend_audit\frontend_audit.csv"

**Thresholds** (never invented — see `mecanum_observer/config.py`):
overall RMSE ≤ 0.0384 m/s (`2% * PRED_P95['Vx']=1.919952`); low-slip RMSE
≤ 0.01 m/s (`LG_V_STR`). Slip speed = `max` over 4 wheels of
`hypot(Vpx_i, Vpy_i)`. Slip bins `0/0.005/0.02/0.065/0.2/0.65/1.5/inf` m/s
taken verbatim from the brief (no other repo source matched, confirmed by
exhaustive grep).

**Filter math**: `alpha = tau/(tau+dt)`, `tau = 1/(2*pi*crossover_hz)`. At
`crossover_hz=5`, `rate=500Hz` (`dt=0.002s`): `tau=31.83ms`, `alpha=0.9409`,
`beta=0.0591`.

## Purpose for the next thread

Decide whether and how to replace the fixed `crossover_hz` in `full` mode
with a slip-gated (adaptive) blend, before writing any code. Success
criterion: a concrete gating design (what signal to gate on, hard-switch vs.
continuous) that can be implemented in `comp_filter.py` and validated by
re-running `frontend_audit.py`.

## Key design decisions already made (defend, don't relitigate)

1. **Per-profile-equal-weighted RMSE is the primary acceptance metric**, not
   whole-sample pooling. Rationale: spin_creep is deliberately over-sampled
   (every approved file, per the brief) as the stress case; raw pooling would
   let its ~32% sample share silently become "the overall number."
   Whole-sample-pooled and spin_creep-only numbers are kept as secondary
   context columns in `ACCEPTANCE.md`.
2. **Ran the full approved fleet (5,345 files), not a stratified sample.**
   User's explicit choice after being shown timing estimates (~2hr on 8
   workers), to maximize statistical power and avoid sample-size ambiguity.
3. **`SensorNoiseSpec` values are placeholders**, borrowed from
   `hybrid_ctrl/sensors.jl` (a different subsystem, same physical units) —
   `accel_noise_std=0.05 m/s²`, `accel_bias_std=0.02 m/s²`,
   `mount_tilt_deg=1.0°`, `gyro_noise_std=0.005 rad/s`,
   `gyro_bias_std=1e-4 rad/s`. No real IMU datasheet exists in-repo; this is
   explicitly acknowledged as acceptable by the brief's own §11.
4. **`odometry.py` has zero slip-awareness by construction**: standard
   rigid-body no-slip inverse kinematics (O-config wheel Jacobian, geometry
   from `mecanum_observer/config.py`: `H=0.235, L=0.15, R=0.05`), solved by
   ordinary least squares (`np.linalg.pinv`) over the 4-equation/3-unknown
   system. No wheel is down-weighted even if it's the only one slipping.
5. **Prior recommendation — `crossover_hz=5.0`, integrator either (delta
   <0.0007 m/s at 500Hz) — is now known to be incomplete.** It passes 6/8
   profiles cleanly (long_circle, ellipse, spiral_orbit, both multisine
   variants, spin_creep) but fails on `octagon` (RMSE 0.231, 12.8% of samples
   >0.65 m/s slip) and `coupled_vomega` (RMSE 0.203, 9.2% extreme-slip,
   worst single-file max error 1.87 m/s). This recommendation should be
   revisited once/if a gated design changes the picture.

## The core finding motivating slip-gating

On a worked example (`octagon_c123_mu_0.8_case1_lugre_adamov_chi_0.002.arrow`,
500Hz, stage=real, noise seed=42): this file has 26 high-slip episodes,
durations ranging 0.001s–2.41s, several sustained (0.52, 0.53, 1.62, 1.65,
2.38, 2.41s). During high slip (>0.65 m/s): `odom_only` RMSE=0.5325,
`full` (fixed crossover=5Hz) RMSE=0.5317 — **statistically no benefit from
the IMU path at all**, despite the per-step blend nominally favoring the
mechanized term 94/6.

Verified by isolated recursion test (constant `a=1.0 m/s²`, constant
`Vo=0`): the filter converges to a *bounded* steady state
`V* = Vo + tau*a = 0.03183` within ~0.1–0.5s and stays there through 4s —
it does NOT track the accel-integrated trajectory long-term. Mechanism:
`alpha` is a per-step retention coefficient on the previous *blended* state,
not a weight on independent long-horizon paths — its effective memory is
`1/(1-alpha) ≈ 16.9 steps ≈ 34ms` at this crossover. `crossover_hz` is
literally the corner frequency: content varying slower than ~3*tau≈95ms gets
routed to odometry regardless of the nominal 94/6 split. Octagon's sustained
episodes (500ms–2.4s) are 5–75x longer than that boundary.

A "reset-to-truth every N seconds, open-loop mechanize in between" experiment
on the same file/window (simulating an idealized, perfectly-timed gate)
showed: at N=0.2s, mech RMSE during high slip = 0.0222 (**24x better** than
odometry's 0.5325) and during low slip = 0.0220 (2.5x worse than odometry's
0.0086) — the crossover point where mech stops winning is around N≈2–4s
(mech RMSE grows to ≈0.44–0.50, matching odometry's high-slip badness).

Separately confirmed `mech_only` (anchor fully off) is bad for a *different*,
slip-independent reason: it's driven by elapsed-time drift, not local slip.
Proof: `multisine` profiles have **zero** samples in any high-slip bin yet
have the *worst* `mech_only` RMSE of all 8 profiles (~6.0 m/s, drift rate
~0.17 m/s/s vs. 0.03–0.04 for most other profiles) — its badness has nothing
to do with slip.

## Open decisions for the next thread

1. **What signal to gate on.** Ground-truth slip (`Vpx_i, Vpy_i`) isn't
   measurable on real hardware. Candidate raised but NOT validated: the
   odometry least-squares fit residual (already a free byproduct of the
   4-equation/3-unknown solve in `odometry.py` — near-zero under pure
   rolling, grows when wheels disagree). Needs checking whether this residual
   actually correlates with true slip speed on real trajectory data before
   committing to it.
2. **Hard switch vs. continuous modulation** of `alpha`/`crossover_hz` based
   on the gate signal, and what time constant(s) govern the gate's own
   response (too fast = chattery, too slow = same problem as now).
3. **Whether this belongs in `comp_filter.py` at all** — it's the shared
   contract `sensor_frontend_v2.py` (owned by the Observer-v2 brief) will
   later import/mirror, so a gating change here has downstream scope.
4. **Re-audit cost**: any change requires re-running `frontend_audit.py`
   (full fleet ≈2hr on 8 workers) to get a new `ACCEPTANCE.md` verdict —
   confirm with the user before launching that again given the runtime.

**Hand-back**: whatever gating design (or explicit decision not to gate) is
settled on should update `crossover_hz`'s status in
`instructions/observer-gamma-only-5phase-retrain.md`, which currently expects
a single pinned value + integrator from this audit's `ACCEPTANCE.md`.

## Deliverables for the next thread

1. A decided slip-gating design (signal, response curve/time-constants,
   hard-switch vs. continuous) — written down before touching code.
2. If proceeding: a gated blend mode added to `comp_filter.py` (new function
   or a parameterized extension of `complementary()`), unit-tested against
   the same closed-form-convergence style check used for the two existing
   integrators.
3. A re-run of `frontend_audit.py` (pilot first, `--limit-files`, before any
   full-fleet re-run) and an updated `ACCEPTANCE.md`.

## Conventions to respect

- Never invent constants — pull from `mecanum_observer/config.py` or flag
  explicitly as a placeholder (as done for `SensorNoiseSpec`).
- Validate claims numerically before asserting them — the user pushed back
  on an unverified claim mid-session and the isolated-recursion test above is
  what resolved it; keep doing that rather than reasoning from formulas alone.
- Confirm with the user before any fleet-scale (multi-hour) run.
- `ACCEPTANCE.md` is the pass/fail source of truth; keep it self-contained
  (the per-profile and cumulative-slip-bin breakdown tables were added
  specifically so findings don't live only in chat).
