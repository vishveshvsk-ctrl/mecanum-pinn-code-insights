# Forward Model v2 — Stick/Slip Redesign (Design Summary)

> **Date:** 2026-07-09
> **Scope:** A1 FORWARD model only (`Mecanum_PINN_Mamba_ForceRecon_v1`). Inverse-model
> redesign is deferred to a follow-up discussion. Sim-to-real transfer mechanics are
> summarized only as architecture hooks (§8); the full transfer recipe is a separate doc.
> **Companion brief:** `instructions/mecanum-forcerecon-forward-v2-stickslip.md`

---

## 1. Why v1 forward fails (evidence)

From `checkpoints_mamba_v1/a1_*/cross_metrics.json` (6 runs, S1/S2 × w8/16/32):

| Metric (normalized by F_MAX = 87.309 N) | Value | Physical |
|---|---|---|
| Forward force recon (grnd MSE, same+cross) | 0.039–0.046 | RMSE ≈ 17–19 N (~40% of μN at μ=0.5) |
| Inverse force recon (grnd MSE) | 0.0028–0.0038 | RMSE ≈ 4.6–5.4 N |
| Forward physics residual | 5–7e-5 | near-exact |
| μ MAE from F_inv readout | 0.25–0.41 | uninformative (grid span 0.5) |

Two root causes:

1. **Null-space drift under the physics-only tail.** The NE residual constrains the
   net wrench + wheel torque balances but not the per-wheel force decomposition.
   The v1 curriculum ends physics-only, so the forward satisfies physics near-exactly
   (5e-5) while the decomposition walks away from truth (0.04). The μ-readout projects
   F_inv onto the *forward's* shape basis, so a drifted forward destroys μ-ID even
   though F_inv itself is good.
2. **Stick regime is architecturally unrepresentable.** With LuGre constants
   (σ0 = 1640, v_str = 0.01 m/s), the bristle time constant is τ = g/(σ0·|v_s|):
   quasi-static (τ ≲ T_s = 2 ms) for |v_s| ≳ 0.15 m/s, but 30 ms at v_str and
   ~0.3 s at 1 mm/s. Bristle states are NOT reset at roller handover (confirmed:
   roller dynamics are slaved), so stick memory is unbounded. v1 zero-inits the SSM
   state every window (w32 = 64 ms) and its A-ladder inits to τ ∈ [0.4, 40] ms
   (Δ-selectivity stretches this ×4 at most). In stick, z is an integrator of
   micro-slip that (a) exceeds any window, (b) is below input SNR (micro-slip is a
   ~0.05%-scale cancellation of O(1) normalized features), and (c) is observable
   only through the force it exerts — i.e. through accelerations, which v1 does not
   receive.

**Slip regime is fine by construction:** for |v_s| ≳ 0.1–0.3 m/s the force law is a
memoryless bounded map of slip velocity (z locks to quasi-static within one sample),
and slip velocity is a near-static function of the fed measurables. This is consistent
with the window insensitivity (w8↔w32 ~2%) and the inverse hitting ~6% with 6 ms of
memory.

---

## 2. Pinned design decisions (user-confirmed)

1. **No LuGre-structured recurrence, no bristle-state supervision or inputs.**
   Generalizability for sim-to-real: real friction need not be LuGre. Inductive
   biases are limited to model-agnostic facts: timescale structure, load balance,
   friction-circle caps, passivity, continuity.
2. **IMU + encoder-rate channels become inputs** (a_x, a_y, ψ̈, ẇ_i). Synthetic IMU
   in sim: **exact-dynamics accelerations from the `accel/` sidecar files**
   (`instructions/arrow-accel-augmentation.md`; body EOM with stored forces — NOT
   finite differencing, which is demoted to the sidecar generator's cross-check),
   converted to the accelerometer observable, anti-alias LPF before decimation;
   the noise/bias/mounting-tilt model applies only in stage 2 (decision 9).
3. **Wrench-restructured output**: predict 6 measured force combinations by fusion +
   **2 null scalars** (s1, s2) along fixed roller-frame directions (§4). Output stays
   in the roller frame [Fpar_1..4, Fperp_1..4].
4. **Two SSMs per wheel** — one slip (fast band), one stick (slow band + integrator
   channels) — with carried state / burn-in training (§5).
5. **Two-branch head with physical gate α** (§6). Slip multiplier cap = **1.0**
   (μ_c·N asymptote); stick cap = **μ_s/μ_c = 1.1** (stiction bound). The Stribeck
   overshoot/decay band is represented by the α-blend, not inside either head.
6. **v_str = 0.01 m/s is pinned** as an order-of-magnitude gate scale (used in gate
   feature normalization and in sim-side α warm-up labels). It is NOT a trainable
   architecture parameter; the learnable gate bias moves the effective boundary.
7. **From A2, take only γ̂ (roller spin)** as a derived-measurable input feature.
   Do NOT import ẑx/ẑy (would re-smuggle LuGre structure and correlate sim priors).
8. **Keep a supervised floor throughout sim training** (W_SUP_MIN idiom; drop the
   physics-only tail). Physics-loss-only *fine-tuning* remains possible later because
   the wrench loss is well-posed under the restructured output (§8).
9. **Sensor-real input contract (mirrors Observer v2 rev 4).** Direct Vx/Vy are NOT
   inputs: V̂x/V̂y come from the same FIXED complementary filter (wheel-odometry
   anchor + strapdown-mechanized IMU path, `V̂ += Δt·(a ± ψ̇·V̂-coupling)`; crossover
   and integrator selected by `instructions/frontend-drift-audit.md`); ψ̇ is the
   gyro; accel inputs are the accelerometer observable (`V̇x − ψ̇·Vy`, `V̇y + ψ̇·Vx`
   — EOM convention code-verified vs run_one.jl). Staged noise: campaign 1
   `noise_stage="none"` (exact-by-construction wrench — architecture validated
   free of sensor realism), campaign 2 flips the toggle with no other change.
   **Single-realization rule:** the forward consumes the IMU twice — encoder
   features AND measured wrench combos — one physical sensor ⇒ ONE corruption
   realization per trajectory shared by both paths, never independent draws.
   Sensor-real applies to model INPUTS only; ground-truth kinematics stay legal on
   the loss side. Yaw angle is needed nowhere (body-frame formulation). The fixed
   filter is the v2 BASELINE front-end; decision 11 records the planned PINN
   velocity-observer successor behind the same V̂ interface.
10. **Condition-adaptive forward estimation — fault correction MANIFESTS in the
    forward law (indirect adaptation; detection alone is not the deliverable).**
    Three feedback paths by fault locus:
    (a) **Contact-law faults** (wear → μ_i, stiction ratio, χ): conditioning promoted
    from scalar μ to a PER-WHEEL condition vector θ_c,i = (μ_i, stiction_ratio_i, …).
    Valid without retraining because the architecture is wheel-factorized (shared
    weights, per-wheel inputs; cross-wheel coupling only via measured body states):
    mixed per-wheel μ is pointwise evaluation of the trained per-wheel map, not
    extrapolation. Updated μ̂_i coherently shifts friction circle, breakaway cap
    (μ_s,i = 1.1·μ_i), gate proximity ρ, and slip-head scale — structurally.
    (b) **Measurement-map faults** (p1_i, actuator gain_i, Jw, IMU calibration):
    estimated and updated IN THE EQUATIONS (`wrench.measured_combos`, load-balance
    features) — an un-updated p1 biases the "measured" drive force; no network
    adaptation can fix a lying measurement equation.
    (c) **Unparameterized drift**: the reserved per-wheel embedding (4-dim, frozen
    hook) unfrozen for the implicated wheel ONLY, fit on the measured-combo loss
    under trust-region + sim rehearsal — the bounded escalation tier.
    **Detector–estimator handshake:** CUSUM/GLR detectors on per-wheel residuals
    CONTROL the recursive estimators — long forgetting factor while nominal,
    covariance reset on detection, regime-gated updates (slip → μ_i; low-slip →
    p1/p2; spin → χ). Fusion innovation corrects FILTER mode per step; θ_c is what
    corrects TWIN/ROLLOUT mode (F_blend), where no measurements exist.
    **Validation = sim fault-injection campaign** (per-wheel μ perturbation e.g.
    wheel 1 at 0.35 vs 0.5, p2 increase, roller-radius perturbation; new combo IDs —
    immutability respected; small run_one.jl extension for per-wheel μ). Headline
    metric: forward/rollout force error BEFORE vs AFTER correction; explicit test of
    the factorization claim (uniform-μ-trained model under mixed per-wheel μ) — if
    it fails, add a small mixed-μ training set.
    **Implementation hook now:** the v2 model accepts μ as `[B]` or `[B,4]`
    (broadcast), so the conditioning interface never breaks when adaptation lands.
11. **Estimated-input interface (PINN observers feed the forward net).** The forward
    consumes ESTIMATES, never privileged states: γ̂ from Observer v2 (pinned,
    decision + rev-2 brief), and V̂ from the velocity front-end. v2 campaigns use the
    fixed complementary filter (decision 9, drift-audited); a **PINN-based velocity
    observer is the planned successor**, dropping in behind the SAME V̂ interface —
    the swap changes the input error statistics, not the contract. On swap: re-run
    the sensor-real ablation and re-check the drift-audit acceptance with the new
    estimator's error profile.

---

## 3. New inputs and features

Per wheel i, on top of the v1 measurables [w_i, sin(12θ_i), cos(12θ_i), Msat_i] and
the sensor-real body globals [V̂x, V̂y, ψ̇_gyro] (decision 9 — sim Vx/Vy never inputs):

- **IMU body channels:** a_x, a_y, ψ̈ — accelerometer observables (shared across wheels).
- **Wheel acceleration:** ẇ_i (encoder-differentiated; from ω at 2000 Hz in sim).
- **Load-balance features** (friction-model-agnostic stick physics — below breakaway
  stick force ≈ applied tangential load):
  - per-wheel pinned drive combo: (Msat_i − Jw·ẇ_i − p1·w_i)/R
  - body net-wrench features from IMU: M_body·accel terms (Qx, Qy, Mz_net)
- **γ̂ (roller spin) feature**: during A1 training, ground-truth γ + injected noise
  matched to A2's binned error statistics (γ̂ error grows with slip: ~0.04 → 0.15
  norm across slip bins); at deployment, A2 supplies γ̂. Rule-compliant because
  γ̂ = A2(measurable window) is a derived function of measurables.
- **Slip surrogate** |v̂_s0| (γ=0 Vpx0/Vpy0 magnitude) retained as a gate feature and
  fallback path (complementary error profile to γ̂: surrogate is best in gross slip,
  where γ̂ is worst).

Hard rule preserved: μ, χ, per-wheel forces, and bristle states are never inputs.
Vpx/Vpy/γ ground truth appear only as (a) auxiliary *targets* for readout heads,
(b) noise-injected stand-ins for the deployment-time γ̂ estimate.

---

## 4. Wrench-restructured output (kills the null-space drift by construction)

Measurable constraints on the 8 roller-frame components:

- 4 wheel torque balances pin each wheel's **drive diagonal**
  (cos δ_i·Fpar_i − sin δ_i·Fperp_i) = (Msat_i − Jw·ẇ_i − p1·w_i)/R.
- Body-y and yaw equations pin 2 combinations of the remaining lateral components.
- Body-x is **redundant** given the wheel balances → **6 independent constraints,
  2-D null space**.

Null basis (roller frame, δ = (−45°, +45°, +45°, −45°)): each wheel's free direction
is the diagonal (sin δ_i, cos δ_i) — the roller-frame image of body-y. The two null
modes are equal-and-opposite body-lateral force within a wheel pair:

```
n1 (front pair):  wheel1 += s1·(−1,+1)/√2   wheel2 += s1·(−1,−1)/√2
n2 (rear pair):   wheel3 += s2·(+1,+1)/√2   wheel4 += s2·(+1,−1)/√2
```

**Output parameterization:** F = assemble(6 fusion outputs anchored to the measured
combos) + n1·s1 + n2·s2, with (s1, s2) from a dedicated null head. Consequences:

- The 6 measured dims become measurement fusion (network predicts small corrections
  around sensor-derived values) — near-noise-floor accuracy expected.
- The whole modeling burden concentrates in (s1, s2) + the stick/slip decomposition.
- The wrench loss has exactly **zero direct gradient** on (s1, s2) (null space), so
  physics-only training cannot push them — the v1 drift mode becomes a named,
  anchorable module.
- Regime closure of the null modes: in gross slip a sliding wheel's force is on its
  friction circle opposite v̂_s (pins that wheel's free diagonal) — a pair's mode is
  open only while BOTH its wheels stick. Stick↔slip transitions are **anchor events**
  that momentarily pin the null coordinates (used for eval and later for real-world
  supervision).

---

## 5. Encoders: two SSMs per wheel, carried state

Same lean selective-SSM core as v1 (diagonal, selective Δ/B/C, trainable A_log),
instantiated twice per wheel with disjoint τ bands:

- **Slip SSM:** fast band, τ init ≲ 10 ms; D_skip path carries the quasi-static map.
- **Stick SSM:** slow band, τ init ∈ [50 ms, 5 s], plus a reserved group of a ≈ 0
  (pure integrator) channels whose write/leak is controlled by the input-dependent
  Δ and B — a strict superset of LuGre/Dahl/GMS leak laws with no law hard-coded.

**Stateful training (mandatory for stick):** burn-in windows — long sequences
(≈ 0.5–1 s) where the first portion builds state with no loss and the tail carries
the loss. Streaming (carried-h) inference at deployment. The stick SSM must never be
zero-init evaluated on short windows.

---

## 6. Two-branch head + physical gate

```
F_i = α_i · F_stick,i + (1 − α_i) · F_slip,i        (composed with §4 assembly)
```

- **Slip head:** F_slip,i = μ_c·N_i·softcircle(m_i), multiplier m dimensionless with
  cap 1.0; direction anchored to −v̂_s with a small learned angular correction.
  μ enters multiplicatively (preserves the test-time μ-readout).
- **Stick head:** reads the stick SSM latent + load-balance features; predicts
  deviation from load balance (breakaway onset, presliding transients); bounded by
  ‖F_stick,i‖ ≤ (μ_s/μ_c)·μ_c·N_i = μ_s·N_i (cap 1.1).
- **Gate (per wheel):**
  α_i = σ( w1·(1 − ρ_i) − w2·ŝ_i − w3·|v̂_s0,i|/v_str + b ), with w1..w3 ≥ 0 via
  softplus (structural monotonicity), where
  - ρ_i = ‖F̂_stick,i‖/(μ_s·N_i) — breakaway proximity (v_str-free physical driver),
  - ŝ_i — auxiliary log-slip readout from the slip SSM latent, supervised in sim
    from the Vpx/Vpy aux label columns (readout target, never an input),
  - |v̂_s0,i|/v_str — kinematic surrogate, v_str = 0.01 m/s pinned,
  - b — learnable bias = effective boundary (real-world adaptable: the gate affects
    the measurable wrench, so it receives genuine gradients at deployment).
  Evaluation is a clean per-step DAG: h_slip → ŝ; h_stick (carried) → F_stick → ρ;
  then α; then blend. Hysteresis emerges from the carried stick state.
- **Gate warm-up:** supervise α on α* = exp(−(|v_s|/v_str)²) from sim ground truth
  (also prevents branch/gate collapse), then anneal so the total-force loss owns it.
  Initializing with the Stribeck weight makes the blend α·(≤1.1) + (1−α)·(≤1.0)
  reproduce g(v)'s μ_s → μ_c decay at init.

---

## 7. Losses and curriculum (sim training)

- **Supervised force loss** (roller-frame, F_MAX-normalized) — with a permanent
  W_SUP_MIN-style floor; **no physics-only tail**.
- **Wrench-measurement loss**: predicted vs IMU/encoder-derived 6 combos (this is
  supervision in 6 of 8 dims, not a prior).
- **Gate warm-up loss** (annealed) + **aux slip-readout loss** (ŝ vs log|v_s|).
- **Passivity regularizer**: ⟨F·v_s⟩ ≤ 0 over windows (model-agnostic; replaces any
  LuGre-specific prior).
- **Sampling**: upweight low-|v_s| / low-|w| segments (reversals, creep, holds);
  the slip majority must not dominate gradients.
- **Evaluation**: always binned by (|v_s|, |w|) so stick progress is visible;
  overall + per-bin force RMSE, gate calibration (α vs true regime), null-coordinate
  error at anchor events.

---

## 8. Sim-to-real hooks (summary only — full recipe is a separate doc)

- Null head (s1, s2): zero direct wrench gradient (automatic); indirect
  representation drift handled by sim rehearsal (interleave labeled sim batches) or
  a frozen-encoder tap; genuine real-world labels arrive only at **anchor events**
  (stick↔slip transitions pin the coordinates) and via offline smoothing between
  anchors.
- Adaptable at deployment: gate bias b, physical calibration scalars (μ̂ init, p1,
  IMU bias/misalignment, actuator gain), then optional low-rank adapters under an
  L2-SP/EWC trust region with Fisher from labeled sim data.
- Load transfer (a_x, a_y → N_i shifts) exists on the real robot but NOT in the sim
  data (static N_per_roller) — a known structural sim-real difference; model it
  explicitly rather than letting the network discover it.

---

## 9. Expected outcome / acceptance

- The 6 measured force combinations reconstruct at sensor-noise level (they are
  fused measurements, not predictions).
- Forward total force RMSE at least matches the v1 *inverse* (grnd MSE ≤ ~0.004,
  ≈ 5 N) — the previous 17–19 N was dominated by exactly the two failure modes this
  design removes.
- Stick-bin (|v_s| < 0.05 m/s) force error becomes the tracked headline metric.
- μ-readout from F_inv becomes meaningful again (drift-free forward shape basis);
  target μ MAE well below the 0.17 predict-the-mean floor.
