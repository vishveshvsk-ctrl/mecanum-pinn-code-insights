# Observer evolution: v1 → v2-Hy3 (slip baseline) → gamma-kin

Roller-spin observer for the KUKA youBot 4-Mecanum platform. Grounding-phase
(supervised, physics OFF) results and the reasoning behind each architecture change.
All numbers are **physical RMSE on the cross-fold TEST split** (S1 model → S2 data and
vice versa), from `runs/*/regime_attrib*.json` and `runs/*/physical_rmse_trend*.json`.

---

## 1. The three versions

### v1 — supervised state observer (`train_observer.py`, `ObserverConfig`)
- **Inputs (3 global):** `[Vx, Vy, ψ̇]` body velocity + yaw rate + per-wheel `[Msat, w, sinθ̃, cosθ̃]`.
- **Targets (3 per wheel):** `[γ, zx, zy]` — roller spin **plus the LuGre bristle
  deflection states** `zx, zy`.
- **Encoder:** shared per-wheel MambaLite SSM (wheel-batched, wheel embedding), one MLP
  head per state.
- **Intent:** learn the full friction internal state so forces could be reconstructed.
- **Problem that forced the redesign:** `zx, zy` are LuGre *bristle* states — an
  un-observable modeling fiction that does not survive sim-to-real (no sensor sees them),
  and the global inputs used near-true `Vx, Vy` that a real robot never measures directly.

### v2-Hy3 — the slip baseline (`train_observer_v2hy3.py`, `ObserverConfigV2Hy3`)
The "hybrid v3" redesign. Three structural moves:
1. **Sensor-real front-end (5 global):** inputs become `[V̂x, V̂y, ψ̇_gyro, a_x, a_y]` — a
   deployable causal **complementary filter** (wheel-odometry LPF anchor + IMU strapdown
   HPF, 1 Hz crossover) plus raw IMU accel/gyro. Nothing true-state enters at inference.
2. **Dropped the bristle states**; added a **ΔV head** that predicts the body-velocity
   correction `ΔV = V_true − V̂` (per-axis, normalized by `dv_scale`), alongside the γ head.
3. **Regime-split steady-state physics loss + derived contact-slip gate.** A GT-free
   `|Vp|(γ̂, ΔV̂)` is formed from the two heads and a sigmoid gate `g = σ((|Vp|−c)/w)`
   blends a stick residual and a slip (roller-torque-balance) residual.
- **Intent:** a deployable, sim-to-real-honest observer whose slip regime is *derived*
  from measurable-only quantities (no label gate at the endpoint).

### gamma-kin — residual parametrization (`train_observer_v2hy3_gammakin.py`)
Same encoder and ΔV head; **only how γ is produced changes.** From nd711 §5.1 "Model 1"
(nonholonomic, no slip), the two contact conditions split: `VPiX=0` fixes the *wheel*
spin (which we measure), `VPiY=0` fixes the *roller* spin in closed form:

```
γ_noslip_i = (V_Y + Ω·ρ_Xi) / ((Rd − R)·cos δ_i)          # V_Y and Ω only
γ̂ = γ_noslip(V_y_used) + Δγ̂ · dgamma_scale               # network learns the DEVIATION
V_y_used = V̂_y + (λ·ΔV_true + (1−λ)·ΔV̂)_y                 # λ: label → deployable phase-out
```

Three coupled changes, each measurement-justified (§2):
1. **γ as a residual off the analytic no-slip base.** Measured over 9.94 M samples,
   `γ_noslip` alone predicts γ to **0.285 rad/s RMS in stick** vs the v2-Hy3 head's 5.56
   (~19×). The network now learns only the small slip deviation (`dgamma_scale = γ_p95/10
   = 8.28`, vs re-using γ_p95 = 82.8 which would starve it — the same normalization trap
   as ΔV below).
2. **Uniform `dv_scale`** = mean(p95 Vx, p95 Vy) = 1.297 m/s, replacing per-axis
   (1.92, 0.673). `∂|Vp|/∂ΔV ≈ 1` on both axes, so the ΔV loss should track the physical
   vector error isotropically. Per-axis normalized the ~(5, 14) cm/s correction by the
   *velocity* p95 → ΔV_x got ~2 % of the head's gradient at init.
3. **`gamma_high_slip_upweight` 3.0 → 1.0** — the old 3× weight on the ~2 % high-slip tail
   pulled effort away from the 20–28 % **stick** bin where the gate actually decides.

**Two latent bugs fixed en route** (both in the baseline files, both raising the gate off
the floor):
- **Base double-count:** slip base was `Vpx0_hat + (V̂+ΔV̂)` but `Vpx0_hat` already carries
  V̂ → correct is `Vpx0_hat + ΔV̂`. Identity floor (true γ,ΔV → true slip) dropped
  **0.857 → 0.0025 m/s (340×)**.
- **γ double de-normalization** in `physics_loss_hy3`: callers passed γ already in rad/s
  to a function that de-normalizes in-graph → γ scaled ×82.8 twice, pinning the gate at
  `g=1` (`stick_frac=0`). This was the run-1 "physics slip-only" symptom.

---

## 2. Grounding results (cross-fold TEST, physical RMSE)

| version (best grounding) | γ all | γ stick | dVx R² | dVy R² | \|Vp\| gate err | stick_frac (true≈0.22) |
|---|---|---|---|---|---|---|
| **v1** (γ,zx,zy · TRUE input) S1 | 4.37 | — | — | — | — | — |
| **v1** (γ,zx,zy · TRUE input) S2 | 3.87 | — | — | — | — | — |
| **v2-Hy3 baseline** S1 | 5.45 | 5.43 | 0.76 | 0.99 | 6.51 cm/s | 0.076 |
| **v2-Hy3 baseline** S2 | 4.20 | 5.06 | **−0.32** | 0.98 | 6.47 | 0.034 |
| **gamma-kin λ=0** (deploy) S1 | 0.98 | 0.62 | 0.96 | 0.99 | 1.21 | 0.204 |
| **gamma-kin λ=0** (deploy) S2 | 1.26 | 0.91 | 0.93 | 0.99 | 1.54 | 0.179 |
| **gamma-kin λ=1** (label) S1 | 0.72 | 0.34 | 0.96 | 0.99 | 1.09 | 0.201 |
| **gamma-kin λ=1** (label) S2 | 0.88 | 0.47 | 0.93 | 0.99 | 1.41 | 0.125 |

(γ in rad/s, cross-fold TEST; λ=1 uses the true-V_y label in the base, λ=0 the deployable
`V̂+ΔV̂`. v1 has no ΔV head / gate, so those columns are N/A. v1 γ is a **direct** checkpoint
eval — the stored `val_loss` is `0.1·sup + physics`, unusable for γ. v1 numbers are the best
per fold across the w{8,16,32,64}/batch/epoch grid; the full grid spans 3.9–7.0 rad/s.)

**Full-lineage γ story (the headline):**
```
v1 (3-state, TRUE Vx,Vy,ψ̇)      ~3.9–5.2 rad/s
v2-Hy3 (γ+ΔV, sensor-real V̂)    ~4.2–5.4 rad/s
gamma-kin λ=0 (deployable)       ~0.98–1.26 rad/s      ← ~4× better
gamma-kin λ=1 (label-assisted)   ~0.72–0.88 rad/s
```
Two things this makes explicit that the v2-Hy3→gamma-kin row alone hides:
- **v1 ≈ v2-Hy3 on γ despite v1 using TRUE velocity inputs.** Moving to the harder
  sensor-real front-end did *not* cost γ accuracy — so the ~5 rad/s wall was **not** an
  input-quality limit; it was the difficulty of predicting a p95 = 84 rad/s signal
  **directly**. Both versions hit the same wall from opposite input qualities.
- **gamma-kin breaks that wall ~4× while on the HARDER (estimated) inputs**, purely by
  changing γ from a direct target to a residual off the analytic no-slip base. The gain is
  the physics prior, not better data and not more capacity (model unchanged at 8,357 params;
  v1 was 6,277). This is the strongest single justification for the residual parametrization.

**What the changes bought, and why they were made:**

- **The gate now works.** `stick_frac` 0.03–0.08 → **0.18–0.20** against a true ~0.22.
  The baseline's |Vp| error (6.5 cm/s) sat 6× above the 1 cm/s gate width, so `g→1`
  everywhere. gamma-kin brings it to ~1.2–1.5 cm/s. **Attribution: γ was 82–87 % of the
  gate error** (swapping true-γ recovers `stick_frac` 0.08→0.18; swapping true-ΔV recovers
  nothing) — which is exactly why the effort went into the γ parametrization.
- **γ improved 4–7×** overall and 6–9× in the stick regime, because the residual replaces
  "predict a p95 = 84 rad/s signal to 1 %" with "predict ≈ 0 in stick, a small deviation
  in slip." This is a physics prior doing the heavy lifting, not more capacity — the model
  is unchanged at 8,357 params.
- **ΔV_x repaired.** Per-axis normalization left S2's x-channel at **R² = −0.32** (worse
  than predicting zero — it *added* error on held-out data). Uniform `dv_scale` alone
  (the ΔV path is detached from γ) took it to **R² = 0.90–0.96**. The raw front-end error
  (~4.3 cm/s x, ~10.1 cm/s y) is corrected to ~1 cm/s, capturing 77 % of x and 92 % of y.
- **Decimation is not the limiter.** The 500 Hz per-component reconstruction floor is
  **0.16 cm/s** — 1 order of magnitude below the gate and ~10× below head error. The
  reconstruction "noise floor" was the two bugs, then the γ head — never the sample rate.
- **Shared encoder is right.** A gradient-cosine diagnostic showed γ and ΔV gradients are
  **orthogonal** (mean |cos| ≤ 0.05) — no representational conflict — so a split encoder is
  unwarranted. GradNorm was rejected: it crushes the priority (γ) task purely because γ has
  the larger gradient magnitude, which the orthogonality shows is not a conflict signal.

**Convergence / budget:** grounding was extended 80 → 150 epochs; γ gained a further ~12 %
(diminishing returns), while **velocity floored by ~ep30–40** and never improved after —
its limit is the 32-sample window vs the filter's ~80-sample memory, i.e. architectural
(longer window / stateful inference), not more epochs.

**In progress — deployable phase-out (`*_ftphaseout`):** fork from ep80, **freeze encoder +
ΔV head**, ramp λ 1→0 over 40 epochs then hold 40, fine-tuning **only** the γ head. Because
ΔV is frozen, `dv_hat` is a fixed base, so the γ head re-fits to the deployable base without
chasing a moving target — closing the λ=1→λ=0 mismatch (the 0.72→0.98 / 0.88→1.26 gap above)
without risking the ~1 cm/s ΔV. Results pending.

---

## 3. Evidence index (repo-persisted)

```
runs/S{1,2}_..._phys_max_norm_grnd80_slip02/          v2-Hy3 baseline
    regime_attrib.json                                per-regime γ/ΔV, |Vp| attribution, stick_frac
    physical_rmse_trend.json                          per-epoch physical RMSE
runs/S{1,2}_..._gammakin_grnd80_slip02/               gamma-kin, 80 grounding ep
    regime_attrib_lam{0,1}.json                       deployable vs label-assisted
runs/S{1,2}_..._gammakin_grnd150_slip02/              gamma-kin, resumed to 150 ep
    regime_attrib_lam{0,1}.json
    physical_rmse_trend_full_0to150.json
runs/COSINE_DIAG_FINDINGS_40ep.md                     γ⊥ΔV orthogonality + GradNorm rejection
observer_v1_py/regime_split_attrib.py                 the attribution probe (reproducible)
```
