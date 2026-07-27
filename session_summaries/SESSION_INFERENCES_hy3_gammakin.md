# Session inferences — hy3 observer: decimation, bugs, gamma-kin, ΔV, slip ablation

Key conclusions from this working session. Excludes the log-loss redesign (see
`chat-handoff/log_domain_slip_loss_handoff.md`) and any conclusion drawn from a run that still
carried a bug (superseded). All numbers are physical RMSE, cross-fold TEST unless noted.

## 1. Decimation to 500 Hz is NOT the reconstruction limiter
- Native 2000 Hz is exact by construction (`contact_from_gamma` == the ODE's contact formula).
- Per-component 500 Hz decimation floor (true states, LPF@200 + downsample-4): **Vpx 0.8→2.4→4.2
  mm/s, Vpy 0.5→1.5→5.5 mm/s** (stick/slip/high). Spin-driven at the top end.
- ~10× below the 10 mm/s gate boundary and ≫ below head error. **Keep 500 Hz; do not go to 1000 Hz.**
- The 200 Hz LPF stays (anti-alias below the 250 Hz decimated Nyquist; blocks LuGre chatter folding).

## 2. Two latent bugs found and fixed (both raised the gate off the floor)
- **Base double-count** (`losses_v2hy3`): slip base was `Vpx0_hat + (V̂+ΔV̂)` but `Vpx0_hat` already
  carries V̂ (`sensor_frontend_v2.py:302`). Correct = `Vpx0_hat + ΔV̂`. Identity floor (true γ,ΔV →
  true slip) dropped **0.857 → 0.0025 m/s (340×)**; verified through the real code path (0.0016).
- **γ double de-normalization** (`physics_loss_hy3` + `eval_hy3_endpoints`): callers passed γ already
  in rad/s to a fn that de-normalizes in-graph → γ scaled ×82.8 twice → gate pinned at g=1
  (`stick_frac=0`). This — not the base bug — was the run-1 "physics slip-only" gate saturation.
  Fix = pass NORMALIZED γ. (Grounding-only gammakin runs never called the physics path, so they were
  unaffected — bug only bit physics-phase / eval.)

## 3. Gradient-conflict diagnostic: γ ⊥ ΔV → keep the SHARED encoder
- cos(∇Lγ, ∇LΔV) at the shared trunk ≈ **0** across a 40-ep grounding run (|mean| ≤ 0.05, neg_frac
  ~0.5). Orthogonal, not conflicting → a split/per-wheel encoder is NOT warranted by this evidence.
- The heads converge at wildly different *rates* (|∇ΔV| collapses ~42×, |∇γ| ~3.7×) — that magnitude
  gap, not conflict, is all GradNorm sees. **GradNorm REJECTED:** it crushes the priority (γ) task
  purely for having the larger gradient; the `r^α` rate term mildly opposes but doesn't save it.
- Encoder = one selective diagonal SSM block (`MambaLiteSSM`, d_model=32, state_dim=6, physical-τ
  init), shared per-wheel, LayerNorm-only (deterministic).

## 4. gamma-kin residual parametrization is the big win (nd711 §5.1 Model 1)
- γ̂ = γ_noslip(V_y_used) + Δγ̂·dgamma_scale, where `γ_noslip = (V_Y+Ω·ρ_X)/((Rd−R)cos δ)` (V_Y & Ω
  only). Measured: γ_noslip alone predicts γ to **0.285 rad/s in stick** vs the direct head's 5.56.
- Result vs baselines (grounding, cross-fold test, λ=1 label / λ=0 deployable):
  ```
  version            γ_all (rad/s)   |Vp| gate err   stick_frac (true ~0.22)
  v1 (3-state,TRUE)  3.9–5.2         —               —
  v2-Hy3 baseline    4.2–5.4         6.5 cm/s        0.03–0.08   (gate dead)
  gamma-kin λ=1      0.72–0.88       1.1–1.4         0.13–0.20
  gamma-kin λ=0      0.98–1.26       1.2–1.5         0.17–0.20   (gate WORKS)
  ```
- **v1 ≈ v2-Hy3 on γ despite v1 using TRUE velocity inputs** → the ~5 rad/s wall was NEVER an
  input-quality limit; it was direct prediction of a p95=84 rad/s signal. The residual breaks it
  ~4–7× while on the HARDER estimated inputs — physics prior, not data/capacity (model unchanged,
  8,357 params). Also: **uniform `dv_scale`=1.2967 repaired S2's ΔV_x (R² −0.32 → +0.90+)**; per-axis
  scaling had starved x to ~2% of the ΔV gradient.

## 5. ΔV is now the binding constraint, and it's ARCHITECTURAL
- Attribution: fixing γ alone recovers 82–87% of the gate; fixing ΔV alone ~0–5%. But post-gamma-kin,
  γ and ΔV contributions to |Vp| are co-limiting (~1 cm/s each).
- ΔV_x captured only 45–77% of the front-end error (ΔV_y 90%+). ΔV floored by ~ep30–40 and never
  improved with more epochs — limit is the **32-sample window (64 ms) vs the complementary filter's
  ~80-sample (1 Hz) memory**. Lever is longer window / stateful inference, not more training.

## 6. Deployable λ phase-out: γ is ΔV-floored, ~1.0/1.2 rad/s
- Freeze encoder+ΔV, ramp λ 1→0 over 40 ep, fine-tune γ head only on the deployable base. Reaches its
  floor by end-of-ramp; the hold and LR do NOT help further. γ can't undo ΔV's residual error (it's
  the part of V_y the ΔV head couldn't extract from the same window). Every path (eval-only, longer
  λ=1, dedicated fine-tune) converges to the same ~1.0 (S1) / ~1.2 (S2) rad/s deployable floor.
- **Method fix (general):** freeze LR scheduler + ES + best-ckpt during ANY ramp — a plateau watcher
  misreads the non-stationary objective's expected metric rise as "no progress" and anneals early.

## 7. Slip-consistency term = a GATE-CALIBRATION knob only
- Ablation `w_slip ∈ {0, 0.02, 1.0}` (× λ{0,1} × in-dist/OOD). **γ, ΔV, and the V_slip components move
  ≤0.5% of p95 across the whole sweep** — negligible. The ONLY metric that responds is stick_frac.
- More w_slip → better gate, monotone on the hard fold: S2 gate error **noslip 23% → slip02 22% →
  wslip1 4%**. On S1, 0.02 saturates it. But w_slip=1.0 overfits the large-slip tail → worst γ and
  worst OOD (multisine γ 0.55→0.82). Slip "falls out for free" from γ+ΔV supervision: noslip's derived
  slip is within ~1–10% of slip02's despite zero slip term in the loss.
- **Decision (this session): keep w_slip = 0.02** for the gate at negligible cost. Open question →
  a log/relative slip loss may capture wslip1's S2 gate gain without the tail-overfit (handoff doc).
- Scale caveat that motivates it: vpm is heavily skewed (median 2.47 cm/s, p95 17.5). vs p95 the ~1.7
  cm/s slip error is ~10% (small); vs the MEDIAN it's ~68% (large). γ/ΔV are ~10–13% of *their*
  medians (small). So slip is the exception where the gate — thresholding at 1 cm/s near the median —
  is genuinely delicate, and the linear-MSE slip loss is dominated by the large-slip tail.

## Artifacts (repo-persisted)
- `runs/S{1,2}_..._gammakin_grnd80_{noslip,slip02,wslip1}/regime_attrib_lam{0,1}.json` (+ multisine).
- `runs/S{1,2}_..._gammakin_grnd{80,150}_slip02/`, `_ftphaseout/` — the main gamma-kin + phase-out.
- `observer_v1_py/OBSERVER_GROUNDING_EVOLUTION.md` — v1→v2hy3→gamma-kin narrative + results table.
- `runs/COSINE_DIAG_FINDINGS_40ep.md`; probes `regime_split_attrib.py`, `eval_v1_gamma_test.py`.
