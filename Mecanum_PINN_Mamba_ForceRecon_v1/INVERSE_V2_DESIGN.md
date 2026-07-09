# Inverse Model v2 — Null-Space Estimator + Model-Free μ Readout (Design Summary)

> **Date:** 2026-07-09
> **Scope:** A1 INVERSE model only (`Mecanum_PINN_Mamba_ForceRecon_v1`). Forward v2
> (`FORWARD_V2_STICKSLIP_DESIGN.md`) and Observer v2 are pinned siblings and proceed in
> parallel — zero model coupling in either direction. Sim-to-real mechanics appear only
> as hooks (§9).
> **Companion brief:** `instructions/mecanum-forcerecon-inverse-v2-brief.md` (generated
> after this document is user-confirmed)

---

## 1. Why v1 inverse μ-ID fails, and what forward v2 changed (evidence)

From `checkpoints_mamba_v1/a1_*/cross_metrics.json` (6 runs, F_MAX = 87.309 N):

| Metric | Value | Physical |
|---|---|---|
| Inverse force recon (grnd MSE) | 0.0028–0.0038 | RMSE ≈ 4.6–5.4 N — GOOD |
| Forward force recon (grnd MSE) | 0.039–0.046 | RMSE ≈ 17–19 N — the failure |
| μ MAE from F_inv readout | 0.25–0.41 | uninformative (predict-mean floor ≈ 0.17) |

**Root cause of the μ failure:** `mu_readout_residual` projects F_inv onto the FORWARD
model's learned shape basis (Phi_A/Phi_C/…). The v1 forward's null-space drift corrupted
that basis, so a good F_inv still produced a useless μ̂. F_inv itself was never the
problem.

**What forward v2 changed:** its wrench restructuring establishes that 6 of the 8
roller-frame force dims are *measured* (4 wheel torque balances
`(Msat_i − Jw·ẇ_i − p1·w_i)/R` + IMU body-y + yaw; body-x redundant), leaving a fixed
2-D null space (pair-antisymmetric lateral forces along the per-wheel free diagonals
`(sin δ_i, cos δ_i)`, δ = (−45°, +45°, +45°, −45°)). This obsoletes most of the v1
inverse's job (reading forces from Δ-states) and makes the following observation
load-bearing:

**In gross slip, μ is observable nearly model-free.** A sliding wheel's force lies on
the friction circle opposite its slip velocity, and its drive diagonal is measured:

```
c_i = (Msat_i − Jw·ẇ_i − p1·w_i)/R = F_i · ê_d,i        ê_d,i = (cos δ_i, −sin δ_i)
slip:  F_i ≈ μ·N_i·û_i ,   û_i = −v̂_s,i                 (measurables + γ̂)
⇒      c_i ≈ μ · N_i·(û_i · ê_d,i)  =  μ · x_i           x_i sensor-computable
```

The learned forward basis — the exact thing that drifted in v1 — is not needed anywhere
in the readout. The only learned inputs are γ̂ (Observer v2; allowed derived-measurable)
and a regime gate, neither of which touches the forward model.

What remains for a network to do: (a) the 2 null coordinates (the only unmeasured force
content — needed for stick-regime force completeness and change detection), (b) regime /
anchor-event detection, (c) nothing else. The readout is closed-form.

---

## 2. Pinned design decisions (user-confirmed 2026-07-09)

1. **Hybrid architecture.** `F_inv = wrench.assemble(combos_meas, ŝ_inv)`. The 6
   measured combos are taken as given (no learned re-estimation); the inverse network
   predicts only the null coordinates ŝ = (ŝ1, ŝ2) and per-wheel regime probabilities.
   Rejected alternatives: full 8-dim standalone network (re-learns measured dims, its
   errors pollute the readout); thin readout on forward trunk features (re-couples
   inverse to forward — the v1 failure mode).
2. **Forward ⊥ inverse preserved.** Zero model coupling; `w_cons = 0` stays
   monitor-only. Shared *modules* only: `wrench.py`, `data_v2.py`, `imu_features.py`,
   and the Observer-v2 γ̂ cache (`precompute_gamma_hat.py` artifacts).
3. **μ-ID is a test-time readout, never a training objective** (rule unchanged).
   Readout = gated scalar recursive least squares on the *measured* combos against the
   physics-defined slip-direction regressor (§5). No learned basis anywhere in it.
4. **No χ regressor.** No χ̂ deliverable in v2. Consequences (pinned): the μ RLS gains
   a **spin-exclusion gate** — high-|ω_z| samples are down-weighted so the unmodeled
   spin term c_t = (8/3π)·|ω_z|·χ (≈ 2.3 N at high spin) cannot alias into μ̂;
   χ-change detection is delegated entirely to the forward-innovation channel (§6);
   the S3 χ k-fold matched-quad data goes unused by this design (reserved for a future
   χ-ID revival).
5. **Null head = gated blend with a per-pair stick-band SSM (v2.0, not deferred):**
   `ŝ_k = g_k · s_pinned,k + (1 − g_k) · ŝ_stick,k` where `g_k` = probability at least
   one wheel of pair k slips, `s_pinned` is the analytic slip-regime value (§4), and
   `ŝ_stick` comes from a minimal per-pair stick-band SSM with **carried state,
   burn-in training, and anchor-injection inputs**. The SSM must never be zero-init
   evaluated on short windows (v1 forward stick lesson).
6. **Trunk memory stays short.** The Δ-window MLP trunk (window 3–8 samples,
   default 3 = 6 ms @ 500 Hz, v1 idiom: newest + diffs) serves the regime head and the
   stick SSM's instantaneous features. The stick SSM is the ONLY stateful component.
   Long-horizon μ integration lives in the RLS accumulator, not in network state.
   (This amends the session's earlier "short Δ-window only" pin — scope of the
   amendment is exactly the null head's stick branch.)
7. **Measurable-only inputs; μ-agnostic everywhere.** No μ/χ inputs to any module;
   trained across the full μ×χ grid. γ̂ is allowed (derived measurable); ẑx/ẑy are
   forbidden. No LuGre structure; v_str = 0.01 m/s appears only as a fixed gate scale
   (shared α* label convention with forward v2 — label reuse, not model reuse).
8. **Purely supervised training — no physics loss exists for this inverse.**
   Newton–Euler is satisfied identically by construction (the measured combos ARE the
   wrench), so the v1 null-drift/physics-tail hazard is structurally absent and the
   W_SUP floor rule is trivially met. The handoff's two-branch caps (1.0/1.1) do not
   apply — this is not a stick/slip generative head; the μ-agnostic cap is §4's
   feasibility bound (per-wheel ‖F_i‖ ≤ N_i, v1 idiom).
9. **v2 = parallel files, v1 byte-identical** (package pattern shared with forward v2
   and Observer v2).

---

## 3. Inputs and components

**Per-wheel features (v2 set, shared with forward v2 via `data_v2`/`wrench`):** v1
measurables [w_i, sin(12θ_i), cos(12θ_i), Msat_i, Vx, Vy, ψ̇] ++ IMU (a_x, a_y, ψ̈) ++
ẇ_i ++ γ̂_i (Observer-v2 cache; gt-noise fallback for ablation) ++ load-balance combos
++ slip surrogate |v̂_s0| and γ̂-corrected v̂_s. Δ-window stacking as in v1
(`[feat_t, Δfeat, …]`).

**Regime head.** Per-wheel slip probability β_i from the Δ-window trunk (small MLP),
trained on the shared regime labels (slip prob = 1 − α*, α* = exp(−(|v_s|/v_str)²)).
Provides: (a) readout gating, (b) pair gate g_k = 1 − (1−β_i)(1−β_j), (c) the
breakaway/anchor detector (β_i threshold crossings).

**Null head.**
- *Slip-pinned branch* `s_pinned,k`: when a pair wheel slips, its direction û_i and
  measured diagonal c_i close the full force vector
  (`F_i = (c_i / (û_i·ê_d,i))·û_i`, guarded by a conditioning weight on |û_i·ê_d,i|),
  and the null coordinate follows from the wrench null-basis inner product (exact
  coefficients live in `wrench.py`, the single verified source). If both pair wheels
  slip, the two pins are reconciled by confidence-weighted averaging.
- *Stick branch* `ŝ_stick,k`: one minimal stick-band SSM per pair (2 instances;
  reuse forward-v2's `SelectiveSSM` class — import, independent weights): d_model
  8–16, τ init ∈ [50 ms, 5 s] plus a ≈ 0 integrator channels; inputs = pair-pooled
  Δ-window features, load-balance features, β_i/β_j, and the **anchor injection**
  g_k·s_pinned,k (lets the SSM latch the pinned value at stick↔slip transitions and
  carry/drift through both-stick intervals — the estimation problem is "propagate from
  the last anchor", not "integrate from cold start"). Explicit h0/hT carried-state
  API; burn-in windows in training; streaming at deployment.
- *Cap*: soft feasibility bound from ‖F_i‖ ≤ N_i given the measured diagonal:
  |s_k| ≤ √2·min over pair wheels of sqrt(max(N_i² − c_i², 0)), applied as a smooth
  (tanh-scaled) clamp. μ-agnostic by construction.

**Assembly.** `F_inv = wrench.assemble(combos_meas, ŝ)` — exact linear map, shared
verified algebra.

---

## 4. Roles (what each output is for)

| Output | Consumer |
|---|---|
| ŝ (null coords) | F_inv completeness in stick; null cross-check channel (§6); anchor-event eval |
| β (regime) | readout gates; anchor detection; breakaway μ_s readout |
| F_inv (assembled) | change-detection monitor; stick/transition force analysis; secondary readout diagnostics |
| μ̂, μ_conf (readout, §5) | the deliverable — friction identification + drift flag |

---

## 5. The μ̂ readout (test-time only, never trained)

**Primary: gated scalar RLS on the 4 wheel-diagonal combos** (rows used per wheel,
only while that wheel is gated slipping; the body-y/yaw rows mix stick-wheel
contributions and are used only as an optional secondary system gated on all-4-slip):

```
row (wheel i, time t):   c_i,t ≈ μ · x_i,t ,    x_i,t = N_i·(û_i,t · ê_d,i)

μ̂      = Σ_t Σ_i g_i,t·x_i,t·c_i,t  /  Σ_t Σ_i g_i,t·x_i,t²
μ_conf  = Σ_t Σ_i g_i,t·x_i,t²                    (gated slip energy, Gram scalar)
```

Deployment form: recursive with forgetting factor λ (running μ̂ + running Gram);
eval form: windowed batch LS. Regressors normalized by N_i so μ̂ is dimensionless.

**Gate g_i,t (product of four factors):**
1. β_i,t — regime head slip probability;
2. quasi-static ramp on |v̂_s,i| (≈ 0 below 0.1 m/s, 1 above 0.15 m/s — the regime-map
   band where the force law is memoryless);
3. **spin exclusion**: smooth down-weight in |ω_z| (form: weight = 1/(1 + (c_t_max(|ω_z|)/(ε·μ_lo·N))²)
   with c_t_max = (8/3π)|ω_z|·χ_max; the threshold constants are calibrated in the
   step-0 audit, §7 — not invented here);
4. direction conditioning |û_i·ê_d,i| (a slip direction orthogonal to the measured
   diagonal carries no μ information in that row).

**Secondary: breakaway events.** At detected stick→slip transitions of wheel i, the
pinned reconstruction gives ‖F_i‖ at breakaway ⇒ μ̂_s,i = ‖F_i‖/N_i. With the pinned
stiction ratio μ_s/μ_c = 1.1, breakaway events yield an independent estimate of the
same μ; agreement `μ̂_s/(1.1·μ̂_RLS) ≈ 1` is a free self-consistency check and enters
the reported confidence.

**Deployment claim gating:** μ̂ is reported only with μ_conf above a floor calibrated
on sim (per-trajectory percentile); below floor, report "insufficient excitation".

---

## 6. Change-detection contract (redefined for the fused forward)

Both models' 6 measured dims track sensors by construction, so the v1 notion of
"F_fwd↔F_inv divergence" collapses there. The v2 contract has three gated,
EWMA-filtered signals (all monitor-only; `w_cons = 0` forever):

1. **Forward innovation (primary μ- and χ-change signal):** forward v2's pre-fusion
   `combos_pred` (μ/χ-conditioned branch output, already in its diagnostics dict)
   minus `combos_meas`, slip-gated. A stale conditioning μ/χ pulls the branch
   prediction away from the measurement. The spin-gated part of this innovation is the
   ONLY χ-change channel (decision 4).
2. **Readout drift (primary μ signal):** |μ̂_RLS − μ_cond| with μ_conf above floor,
   where μ_cond is the value conditioning the forward. Health check: the forward's own
   self-consistency readout (recover μ_cond from its pre-fusion slip-branch
   multipliers — already defined in forward-v2 evaluation) must stay clean; if it
   drifts, the flag indicts the forward, not the plant.
3. **Null cross-check (secondary):** ŝ_inv vs forward's (s1, s2) — the only genuinely
   unmeasured dims; divergence flags model disagreement in the stick-null space.

Implementation lands in inverse-v2 evaluation tooling reading forward diagnostics —
no forward code change, no training coupling.

---

## 7. Step-0 audits (mandated before implementation)

1. **`null_audit.py`** (Arrow labels only, analog of Observer v2's `roller_audit.py`):
   (a) s_lab = `wrench.decompose(F_sim)` statistics; both-stick dwell-time
   distribution per pair/profile; s drift within dwells vs a zero-order-hold-from-anchor
   baseline (the non-learned baseline `ŝ_stick` must beat);
   (b) validate the s_pinned formula on gross-slip segments vs s_lab (residual ≈ label
   noise floor — the slip-pin correctness proof);
   (c) **noiseless readout recovery:** batch LS on noiseless combos + ground-truth
   gates must recover the grid μ near-exactly (proposed MAE ≤ 0.02) — the readout's
   unit test — and quantify the μ̂ bias vs |ω_z| (calibrates the spin-exclusion gate
   constants of §5).
2. **Carried-state equivalence** test for the stick SSM (one long scan == chained
   scans) — same test as forward v2.
3. `wrench.py` round-trip and `measured_combos` residual verification is owned by
   forward v2 (shared dependency); inverse implementation blocks on it existing.

---

## 8. Losses and curriculum (sim training — purely supervised)

Computed on the loss tail only (`[:, L_burn:]`, reusing `data_v2` burn-in windows,
default 384 + 128 @ 500 Hz):

- **L_s** — MSE on ŝ vs s_lab (own p95 scale entries appended to the scaler CSV).
  Supervise s directly, NOT assembled F, so measured-combo sensor noise never enters
  the loss floor.
- **L_regime** — BCE on β vs (1 − α*) labels, per wheel.
- **L_anchor** — auxiliary MSE on the stick branch: `ŝ_stick` vs `s_pinned`, gated by
  g_k (teaches the latch; without it the SSM branch receives no gradient where the
  blend is dominated by the pin).
- **F_inv vs F_sim** — logged as a *monitor metric only* (redundant with L_s up to
  measurement noise).
- **Gate curriculum:** phase 1 (grounding) uses ground-truth regime labels (noise-
  injected) for the blend gate g and anchor injection; phase 2 (consolidation) anneals
  to the model's own β — mirroring the γ̂ noise-injection idiom. Two phases only; no
  physics phases exist. Plateau LR (patience 10, Observer-v2 idiom), single scheduler.
- Sampling: upweight both-stick and transition-dense windows (the null head's hard
  content); slip-majority windows must not dominate.

---

## 9. Sim-to-real hooks (summary)

- **The readout transfers by construction** — it is training-free and needs only
  calibrated physical scalars (N_i, R, Jw, p1, IMU bias/misalignment — same
  calibration list as forward v2) plus real-robot γ̂ from Observer v2.
- **Anchor events are real-world supervision** for ŝ (identical mechanism to forward
  v2's null anchoring): stick↔slip transitions pin the null coordinates with zero
  labels required.
- Parameter groups named so the stick SSM and regime head are individually freezable;
  regime head adaptable via gate-consistency against the kinematic surrogate.
- Load transfer (a_x, a_y → N_i shifts) is absent in sim data (static N) — the N_i
  entering both the cap and the readout regressor must be an injectable function, not
  a baked constant (structural sim-real difference, same note as forward v2).

---

## 10. Expected outcome / acceptance

- [ ] `null_audit.py`: s_pinned residual ≈ noise floor on slip segments; noiseless
      readout recovers grid μ with MAE ≤ 0.02; spin-gate constants calibrated.
- [ ] Carried-state equivalence passes for the stick SSM.
- [ ] Trained F_inv grnd MSE ≤ v1 inverse (≤ ~0.003, ≈ 5 N) overall — the measured
      dims should make this easy; the honest metric is the null part:
- [ ] Both-stick binned ŝ RMSE beats the ZOH-from-anchor baseline (the stick SSM's
      justification) and the null contribution to stick-bin F_inv error is reported.
- [ ] Regime head calibrated (β vs true regime, per bin); breakaway ratio
      μ̂_s/(1.1·μ̂_RLS) ≈ 1 within tolerance on sim.
- [ ] **Headline:** with realistic sensor noise + Observer-v2 γ̂, conf-gated μ MAE
      ≤ 0.05 (proposed target), and everywhere-reported μ MAE well below the 0.17
      predict-the-mean floor — vs v1's 0.25–0.41.
- [ ] μ-swap eval: forward conditioned on wrong μ over a held-out trajectory →
      readout-drift and forward-innovation channels both flag, with μ_conf above
      floor, within a bounded slip-time budget.
