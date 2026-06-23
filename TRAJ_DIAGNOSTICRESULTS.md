# Diagnostics results — detailed observation → inference → explanation

Four diagnostic screens were run over the **μ-grid friction-cap sweep — 5,670
Arrow files = 3 batches (μ ∈ {0.3, 0.5, 0.8}) × 1,890**, each batch being the
~1,485 single-χ production sweep (χ=0.005) plus a matched χ-grid {0, 0.002, 0.008}
on the χ-quad combos; friction_model=lugre_adamov, case 1 throughout. This note
explains each result in plain terms, then ties them together. **Headline: whitelist
5,345 / 5,670 (94.3 %)**, ~94 % at every μ (§6). The per-μ friction-cap findings and
the octagon feasibility fix are in **§7**. The model-free **μ-multiplicativity**
check (do the forces scale as `A·μ + B`?) is in **§8**. **§9** collects three other
model-free inferences (slip-regime coverage, gate-variable observability, force
anisotropy); **§10** tests the gated multiplicative force law `μ(A+Cχ)+B` as the PINN's
forward friction head.

**Reading guide — the physics in one paragraph.** The recorded wheel forces look
noisy, but the "noise" has four distinct origins, two of which are signal:
(1) **roller ripple** — periodic in wheel angle θ at `f_roll = 12·|ω|/2π ≈ 1.91·|ω|` Hz
(physical, the network must learn it); (2) **LuGre stick–slip** — bristle
oscillation whose frequency scales with slip, `f_lugre ≈ 653·|Vp|` Hz (physical);
(3) **ASMC chatter** — sliding-mode switching that leaks from the controller
(`Msw`) into the forces (unwanted but coherent); (4) **numerical hash** —
broadband solver noise (unwanted, incoherent). Every result below is ultimately
about telling these four apart, or about what happens to them under
parameter/rate changes.

---

## 1. Chatter diagnostics — `chatter_report.csv`

**What it measures (six spectral force metrics per trajectory, median over the 4
wheels; plus M7 — a separate control-torque burst flag, documented below):**

| metric | one-line meaning | clean = |
|---|---|---|
| `ridge_concentration` | fraction of force energy on the roller ridge `h·f_roll` | high |
| `hash_fraction` | fraction of energy *above* the physical ceiling `F_hash` | low |
| `control_chatter_index` | high-freq energy in the switching wrench `Msw` ÷ RMS(`Meq`) | low |
| `msw_force_coherence_hf` | coherence between `Msw` and force above `F_hash` | — |
| `hf_slip_modulation` | correlation of the above-ceiling force envelope with `|Vp|` | — |
| `theta_fold_tightness` | scatter of force folded over the roller pitch | low |

The classifier: **`hash`** = high `hash_fraction` + low coherence + not slip-modulated
(incoherent noise → reject); **`chatter`** = high `control_chatter_index` + high
coherence (control switching leaking into force → kept but flagged); else
`clean`/`marginal`. This verdict is purely spectral and **does not include M7** —
the burst flag is reported alongside it (see below) and fuses in only at §6.

### M7 — localized control-torque burst (`msat_burstiness`, separate flag)

A saturated controller that loses tracking can erupt in a brief, violent `Msat`
oscillation (an ASMC limit cycle), typically at the **end** of a run. The six
spectral metrics are **structurally blind** to it: it is *localized*, so the
whole-trajectory PSD dilutes it; and under gross slip `F_hash` inflates above the
Nyquist, so M2/M3 read exactly 0 there. M7 sidesteps both by working on the
*control* torque and measuring **localization, not amplitude**. Per wheel, on the
full 2000 Hz `Msat`:

1. **high-pass at 15 Hz** (4th-order Butterworth);
2. sliding-window **RMS over 0.25 s** windows of the high-passed torque;
3. `burstiness = max_window / (p10_window + FLOOR)`, with `FLOOR = 0.10 N·m`;
4. reduce over the 4 wheels by **max** — a burst on any one wheel counts.

**Why the 15 Hz high-pass.** It is ≈5× the wheel mechanical corner
`p1 / (2π·Jw) = 0.11 / (2π·5.87e-3) ≈ 3.0 Hz`. Above that corner the heavy wheel
inertia low-passes the applied torque, so HF content there can do **no useful
tracking work** — all *functional* control lives well below it (the velocity
references are ≤1 Hz). 15 Hz therefore sits above every functional control band
yet **below** both the measured ~29 Hz burst content and the ~77 Hz
roller-dynamics corner. The band `[15 Hz, Nyquist]` isolates exactly the
non-functional HF where a genuine give-up burst lives, without flagging
legitimate control effort (`5 × 2.98 ≈ 14.9 Hz`, rounded to 15).

**Why a ratio, not absolute amplitude.** Broadband control (multisine) has *flat*
HF (~0.1 N·m across the whole run) → `max ≈ p10` → burstiness ≈ 1; a localized
burst sits on a quiet baseline → `max ≫ p10` → burstiness ≫ 1. Absolute HF
amplitude cannot separate the two (multisine ≈ a real burst at ~0.1 N·m).

**Why p10, not the median.** When the disturbed fraction (startup chatter + an
end-of-run burst) exceeds ~50% of the run, the median is dragged up *into* the
disturbed population and the ratio collapses — strong coupled_vomega bursts scored
~93 under the old `median + tau=100` denominator and **escaped**. The
10th-percentile window-RMS estimates the quiet baseline robustly to a large
disturbed fraction; `FLOOR` (= 1% of `Max_torque = 10 N·m`) is a physical
silent-torque level so a truly-silent baseline (`p10 → 0`) doesn't degenerate the
ratio into `max/eps` and explode on a clean blip.

`burst_flag = msat_burstiness > tau_burst`, **`tau_burst = 10`** — tuned on the
sweep (p10 denominator, `FLOOR = 0.10`): clean trajectories ceil at ~5.5
(multisine 1.4), give-up bursts start ~13, so 10 sits in the >2× gap.

**Implementation.** The kernel is `chatter_diagnostics.m7_msat_burstiness`; the
batch driver records `msat_burstiness`, `msat_burst_abs` (= max-window RMS ÷
`Max_torque`, secondary context), and the boolean `burst_flag`. The deep-dive
notebook `Trajectory_Chatter_Diagnostics_Profiles.ipynb` visualizes M7 per wheel
by calling the **same** `cd.m7_msat_burstiness` + `cd.CLASSIFIER_THRESHOLDS['tau_burst']`,
so the notebook and the screen can never drift. How `burst_flag` enters the
training decision is in §6.

### Observation
5,670 files (3 μ pooled), **0 errors, 0 `hash`, 689 `chatter`, 4,981 `clean`.** By profile:

| profile | chatter | n | chatter % |
|---|---|---|---|
| octagon | 0 | 1548 | 0% |
| long_circle | 0 | 762 | 0% |
| coupled_vomega | 14 | 846 | 2% |
| ellipse | 30 | 288 | 10% |
| spiral_orbit | 14 | 204 | 7% |
| spin_creep | 344 | 1692 | 20% |
| multisine_50percent_cap | 165 | 165 | **100%** |
| multisine_75percent_cap | 122 | 165 | 74% |

(`multisine_75` is no longer a flat 100 % once μ varies — the chatter is μ-dependent,
reinforcing that it is a global-threshold artifact, not a data defect; see below.)

**M7 burst flag (separate from the above):** 415 trajectories (3 μ) trip
`msat_burstiness > 10` — spin_creep 198, coupled_vomega 154, octagon 63. They split
cleanly by tracking: the 209 coupled_vomega/octagon bursts coincide with **degraded
tracking** (give-up at the friction circle), while **all 198 spin_creep bursts (+8
others) are on well-tracked runs** — the by-design high-yaw-rate pulse edges, kept.
This split is exactly why the burst flag does not reject on its own (§6). Burst count
rises with μ (more torque authority → more give-up bursts); see §7.

### Inference
- **The integration is clean.** Zero `hash` across 5,670 files means no trajectory
  carries broadband numerical noise that is unexplained by any input. The solver
  + sweep produced physically self-consistent forces everywhere.
- **`spin_creep` and `multisine` carry the most control chatter** — exactly the two
  profiles that excite the controller hardest.
- **The `multisine` 100% flag is almost certainly a threshold artifact, not a
  data defect** — see explanation. So the current verdicts are *provisional*; a
  per-profile threshold-tuning pass is required before any training whitelist.

### Explanation
- **Why 0 hash is believable, not suspicious:** the hash test fires only on energy
  above the LuGre-aware ceiling `F_hash` that is *also* incoherent with `Msw` and
  *not* slip-modulated. A well-converged stiff solve simply doesn't produce that.
- **Why `spin_creep` chatters most:** the ASMC boundary-layer thickness is
  `eps = max(0.025·PEAK_V, 0.001)`. `spin_creep` runs at low translational speed,
  so `PEAK_V` is small and `eps` hits its 0.001 floor → thin boundary layer →
  high effective switching gain `K/eps` → the sliding variable limit-cycles. That
  switching shows up in `Msw` and propagates to the force. This is genuine,
  physically-expected control chatter.
- **Why `multisine` reads 100% chatter (and why that's a threshold issue):**
  multisine commands a zero-mean *broadband* reference on Vx/Vy/Ω. The controller
  faithfully produces broadband `Msw` to track it, and that `Msw` is genuinely
  coherent with the resulting broadband force — so `control_chatter_index` and
  `msw_force_coherence_hf` are both high, tripping the `chatter` rule. But this is
  *commanded* broadband control, not pathological switching chatter. The single
  global threshold can't tell "designed broadband excitation" from "limit-cycle
  chatter"; a multisine-specific cutoff will separate them. (This is the same
  per-motion-threshold lesson from the v1 diagnostics.)
- **Why `spin_creep`'s 66 bursts are kept, not rejected (the same lesson, for M7):**
  spin_creep schedules high-yaw-rate pulses; each pulse edge spikes `Msat` and trips
  `burst_flag` (a tight 17–19 cluster, far from the wide 13–50 scatter of real
  give-up bursts). But the platform tracks fine throughout — spin_creep has **zero**
  missed trajectories — so the spikes are by-design excitation, not give-up. Rather
  than a profile-specific `tau_burst`, the combined rule requires a burst to be
  *confirmed by lost tracking* (§6), which keeps all 66 automatically.

---

## 2. χ-identifiability — `chi_identifiability_mu{03,05,08}.csv`

**Question:** can the PINN recover χ from the *observable* forces **Fpar/Fperp**?
(`Mz` is dropped — χ²-dependent and unmeasurable at deployment.)

**Method.** χ reaches the forces only through the spin→translation coupling
`c_t = (8/3π)·|ω_z|·χ` — *gated by spin and LINEAR in χ*. **Per μ** (run separately
with `--mu`, so friction levels aren't pooled), take the **whitelisted matched
χ-quads** (same trajectory at χ ∈ {0, 0.002, 0.005, 0.008}, with *every*
counterpart well-tracked), bin by contact spin `|ω_z|`, stratify by `|psi_dot|`,
and within each cell regress the force on **χ (linear)** while controlling for
**slip (`Vpx, Vpy, |Vp|, |Vp|²`, nonlinear)** and per-wheel offset. Headline = the
**absolute χ-induced force swing** `|ΔF|_χ` over χ∈[0,0.008] — *not* R²/F (see explanation).

> **Model choice (validated).** χ enters **linearly**: the coupling is linear in χ,
> so the χ² term in the original design (needed only for the dropped `Mz`) is both
> unmotivated for forces *and* near-collinear with χ on the tiny χ grid — that
> collinearity was the μ=0.3 conditioning **blow-up** (swings → 1e17). Switching to
> linear-χ leaves the median swing **unchanged** (<5% on the robust μ=0.5/0.8 cells)
> while improving conditioning **~12 orders** and removing the blow-up. Dropping the
> nonlinear slip term |Vp|², by contrast, *shifts* the swing (−36% on Fpar at μ=0.5)
> — it carries real slip-confound — so **slip stays nonlinear**. A whitelist filter
> (every χ-counterpart well-tracked) plus a condition-number guard keep the fits clean.

### Observation
χ-induced |ΔF| in the high-spin bins, **per μ** (covariate-controlled median;
linear-χ, whitelisted quads):

| μ | Fpar | Fperp |
|---|---|---|
| 0.3 | 1.07 N (10.0% of RMS) | 1.60 N (8.4%) |
| 0.5 | 1.44 N (8.1%) | **2.84 N (9.2%)** |
| 0.8 | 1.27 N (3.3%) | 1.20 N (2.6%) |

The commanded reference is *bit-identical* across χ (differences are χ-only); the
χ-effect **rises with spin** (the coupling signature); the per-cell swing
distribution is **right-skewed** (median is the robust summary); and the signal
**leans toward `Fperp`** at μ=0.3/0.5 — consistent with χ's spin force being more
transverse to slip (a slip-aligned `F⊥` view would likely concentrate it further).

### Inference
- **χ is identifiable from the observable forces.** A ~1–3 N swing in Fpar/Fperp
  (3–10% of the force RMS) at high spin sits above a realistic force-noise floor,
  so the PINN can invert χ from the forces it sees at deployment. **Strongest at
  μ=0.5 / Fperp** (2.84 N, 9.2%).
- **rel-swing falls as μ rises** (Fperp 8.4% → 9.2% → 2.6%): at higher friction the
  force is larger, so the same-order χ swing is a smaller *fraction* of it — χ gets
  relatively harder to read at high μ.
- **The mechanism is confirmed:** the χ-effect growing with `|ω_z|` is the signature
  of the spin-gated coupling, not a spurious correlation.

### Explanation
- **Why we report absolute swing, not R²/F:** each cell holds ~10⁶ samples, so the
  F-statistic is astronomically "significant" for any effect — significance is
  meaningless here. And partial R²_χ is *relative to each channel's own variance*
  (the "units trap": `Mz` is ∝χ², so χ explains nearly all of its *tiny* variance →
  R²_χ ≈ 1 despite a negligible absolute effect — one reason `Mz` is dropped).
  Identifiability under real noise depends on the **absolute** signal (N) vs the
  noise floor, so the few-Newton Fpar/Fperp swing is what counts.
- **Why we couldn't difference forces in time (the binning was mandatory):** on a
  matched χ-pair the pointwise `RMS(Fpar(χ) − Fpar(0))` was ~32 N — *larger than
  Fpar itself* — because a tiny χ change makes the two trajectories drift out of
  phase over ~20 s (decoherence). Binning by operating point and comparing across χ
  *within* bins removes that artifact and exposes the true ~1–3 N χ-effect.
- **Conditioning:** linear-χ is well-conditioned (cond ~10⁶); the cond-number guard
  in `_fit_cell` rarely fires now and exists only to NaN any residual sparse-bin
  degeneracy (it was added when the old {χ,χ²} design blew up at μ=0.3).

### Deep dive — χ's force-variance contribution by spin regime (gated fit)

A complementary read of χ-identifiability falls out of the gated multiplicative force
law `F = μ·(A + C·χ) + B` (§10). Instead of the absolute swing, ask what *fraction of
force variance* the χ term explains, **resolved by contact spin `|ω_z|`**: within each
(slip-bin × `|ω_z|`-bin) cell, `partial R²_χ` is the variance the `C·μ·χ` term adds over
the μ-only model. (This is the *forward-model* adequacy question — how much of the force
the term explains, computed at fixed slip/spin — not the inverse-noise-floor / units-trap
question of the main result above.)

**Observation** (full dataset, `|ω_z|`-binned):

| `\|ω_z\|` [rad/s] | partial R²_χ (constant `C`) | partial R²_χ (explicit `\|ω_z\|`) | χ-swing [N] |
|---|---|---|---|
| ~1.2 (low spin) | 0.01 | 0.02 | ~1 |
| ~6.2 | 0.12 | 0.16 | ~4 |
| ~11.2 | 0.26 | 0.30 | ~6 |
| ~16.2 (high spin) | 0.42 | 0.71 | ~4–12 |

**Inference**
- **χ explains ~0–2 % of force variance at low spin, rising to ~40–70 % at high spin** —
  the same spin-gating §2's swing showed, now as variance-explained. The χ-swing (≈1 N →
  ≈12 N) brackets the §2 median of 1–3 N at moderate spin.
- **A constant `C` (blind to `|ω_z|`) cannot absorb the spin-gating.** Aggregate
  `partial R²_χ` with a single `C` is **0.08**; resolving `|ω_z|` — by binning, or putting
  `|ω_z|` explicitly in the regressor — lifts it to **0.12** overall and to **0.4–0.7** in
  the high-spin bins. So a forward model that does not resolve spin recovers only ~⅔ of the
  χ-variance and misses the strong high-spin signal entirely.
- **χ-ID is therefore coupled to spin reconstruction.** `|ω_z|` is ~87–90 % hidden roller
  rate (§9.2), so reading χ from the observable forces requires the encoder/observer to
  reconstruct contact spin — it is not available directly.
- **The single-χ imbalance halves it.** On the χ=0.005-dominant full set, `μ` and `μ·χ`
  are near-collinear in most cells, so `partial R²_χ` drops ~2× versus the balanced
  matched-χ-quad subset (0.039 vs 0.080); χ-leverage comes from the quads.

> **Figure (deck candidate, no PNG yet):** `partial R²_χ` vs `|ω_z|` (the climb from ~0.01
> at low spin to ~0.4–0.7 at high spin, constant-`C` vs explicit-`|ω_z|` curves) — from
> `force_mu_chi_gated.csv`.

---

## 3. Sampling-rate sensitivity — `sampling_sensitivity.csv`

**Question:** the PINN trains more cheaply on a coarse grid; is the difference
between **2000 / 1000 / 500 Hz** significant for the forces it learns from?

**Method:** for each file, downsample to 1000 and 500 Hz two ways — *anti-aliased*
(low-pass then decimate; loses high-freq content cleanly) and *naive* (index-slice;
high-freq content aliases into the band) — then measure the band-limited
**reconstruction error** back to 2000 Hz, per force family, plus whether the
chatter verdict flips.

### Observation
Per-profile median reconstruction error (normalized; → = 1000 Hz then 500 Hz,
anti-aliased; **pooled over all 3 μ-batches, 5,670 files**):

| profile | Fpar | Fperp | Mz | 500 Hz flip% |
|---|---|---|---|---|
| coupled_vomega | 0.011 → 0.011 | 0.012 → 0.012 | 0.064 → 0.096 | 0 |
| spiral_orbit | 0.013 → 0.016 | 0.016 → 0.021 | 0.036 → 0.055 | 0 |
| spin_creep | 0.012 → 0.019 | 0.014 → 0.018 | 0.163 → **0.235** | **8** |
| octagon | 0.011 → 0.011 | 0.011 → 0.012 | 0.069 → 0.102 | 0 |
| long_circle | 0.013 → 0.015 | 0.019 → 0.027 | 0.034 → 0.049 | 0 |
| ellipse | 0.010 → 0.012 | 0.014 → 0.017 | 0.044 → 0.068 | 0 |
| multisine (both) | 0.008 → 0.011 | 0.011 → 0.013 | 0.047 → 0.079 | 0 |

**μ-robust:** Fpar/Fperp stay ~1–2 % at *every* μ (0.3/0.5/0.8). The only μ-sensitivity is
`spin_creep`'s `Mz`/flip, which **peaks at μ=0.5** (flip 12.8 %) and is *lower* at μ=0.3
(4.4 %) and μ=0.8 (5.5 %) — and `Mz` is dropped anyway, so "train at 500 Hz" holds across
the whole grid.

### Inference
- **For the training targets (Fpar/Fperp), 2000 vs 1000 vs 500 Hz is
  insignificant — across every profile, including the high-slip ones.** Errors
  stay ~1–3% and barely move from 1000→500 Hz. **You can train at 500 Hz** with
  negligible force-fidelity loss — which solves the fine-grid practical-cost
  problem.
- **`Mz` is the only genuinely rate-sensitive channel** (spin_creep 0.16→0.24 at
  500 Hz) — but since `Mz` is being dropped (Result 2), this doesn't matter for
  training. The two results agree: the rate-fragile channel is the unneeded one.
- **The chatter *screen* should run at 2000 Hz, not at the training rate** —
  `spin_creep` flips 8% of verdicts at 500 Hz (3-μ pooled) because the screen goes blind
  above the 250 Hz Nyquist. Screen at 2000, train at 500 — no conflict.

### Explanation
- **Why my earlier "500 Hz breaks for high-slip" worry didn't materialize:** the
  first-principles LuGre upper bounds (e.g. "40% of energy above 250 Hz at
  Vp=0.3") assumed a flat/white slip input — pessimistic. In the real data the
  forces carry almost no energy above 250 Hz, so 500 Hz loses essentially nothing.
  The forces are dominated by the slow trajectory envelope + roller ripple
  (≤~86 Hz), all well below the 250 Hz Nyquist.
- **Why naive ≈ anti-aliased for the forces (and naive is even slightly better at
  1000 Hz):** with negligible content above the Nyquist there is nothing to alias,
  so naive subsampling is nearly lossless — while the anti-alias filter itself
  adds mild passband distortion. Aliasing only shows up in `Mz` (more high-freq
  content), where naive is worse. So the rate choice isn't hostage to perfect
  filtering; anti-aliased remains the safe default at no cost.

---

## 4. The three results together — recommended actions

| decision | basis |
|---|---|
| **Train the PINN at 500 Hz** | Result 3: force recon error ~1–3% at 500 Hz, flat vs 1000 Hz, all profiles |
| **Drop `Mz` from training; recover χ from `Fpar/Fperp`** | Result 2: forces carry a 2.3 N χ-signal; Mz tiny + unmeasurable |
| **Run the chatter screen at native 2000 Hz** | Result 3: 8% verdict flips for spin_creep at 500 Hz (3-μ pooled) |
| **Tune chatter thresholds per-profile before any whitelist** | Result 1: multisine 100% flagged is a global-threshold artifact |
| **Dataset integration quality is good** | Result 1: 0 hash across 5,670 files |

**Net:** the dataset is clean; χ is learnable from the observable forces alone;
and the PINN can train on a 4× cheaper 500 Hz grid without losing the forces it
needs — provided the chatter screen (run at 2000 Hz, with per-profile thresholds)
gates the whitelist first.

---

## 5. Tracking / friction-circle gate — `tracking_report.csv`

A fourth, orthogonal screen (added after Results 1–3): did the platform actually
*achieve* the commanded trajectory, or lose tracking / exceed the friction
circle? The chatter screen cannot catch this — a saturated controller rails
smoothly (no chatter), and `F_hash` inflates above the Nyquist under gross slip,
blinding the hash test. The verdict is driven by two **duration** metrics
(post-0.5 s-startup mask): `track_viol_frac` (fraction of the run the instantaneous
generalized-velocity error exceeds `track_inst = 0.30`) and `util_viol_frac`
(fraction of the run the max-wheel utilisation `|F|/μN` exceeds the **friction-circle
level `UTIL_VIOL_LEVEL = 1.10`**). A run is `missed` if either exceeds its gate —
`track_viol_frac > 0.10` **or** `util_viol_frac > 0.10`. `track_err` (RMS),
`util_budget_frac` (frac > 0.8), and `msat_rail_frac` are reported as descriptors only.

**Two data-driven util tunings.** (1) The duration gate `util_viol_frac` was raised
0.05 → 0.10 (sparing runs that briefly touch the circle). (2) The **per-instant
violation level was raised 1.0 → 1.10** (`tracking_gate.UTIL_VIOL_LEVEL`).
Inspection of the util-driven rejects showed the max-wheel util **rides AT the
circle** — in 1.0–1.05 for ~30 % of the run and **never above 1.05** — i.e.
**at-limit friction saturation (`|F| ≈ μN`), physically valid data the PINN should
learn, not loss of control**. "Gross slip" was a mis-read: there is none in this
dataset. At level 1.10, `util_viol_frac > 0.10` fires on **zero** trajectories, so
**every `missed` run is now track-driven** (a genuine off-command failure); util is
retained as a *dormant* safety net that would still catch real gross slip
(sustained util > 1.10). This recovered **66 trajectories**, including the entire
**spin_creep μ=0.3 / χ=0 matched-quad anchor** (44 combos) that previously
saturation-rejected — restoring the 4-point χ regression at μ=0.3.

**Result (3 μ-batches, 5,670 trajectories):** 5,340 `tracked` / 5 `marginal` /
**325 `missed`**, **all 325 track-driven**. The remaining rejections are **only
octagon** (the residual top-`vcru` production combos — see §7) and **coupled_vomega**
(a handful of infeasible (V,Ω) rays); every other profile is fully clean. The two
gates stay disjoint and complementary — chatter catches force contamination,
tracking catches genuine off-command runs.

> **Octagon lateral infeasibility (diagnosed → χ-quad fixed, production fix pending).**
> The top `vcru` tier (0.60/1.00/1.60 at μ=0.3/0.5/0.8) commands a lateral-leg velocity
> the Mecanum platform cannot reach (lateral caps ~0.7×vcru_max), so it misses on
> *tracking* at every μ and χ (0 % success, μ/χ-independent — confirmed by the new
> `straightline` β=90 probe). The matched **χ-quad** has been **relocated one tier
> below the cap and regenerated** (now 100 % feasible at all μ×χ, §7); the remaining
> octagon rejects are the top-tier *production* (χ=0.005) combos, pending the full
> `vcru_max` re-cap.

---

## 6. Regenerating and combining the reports

The four screens stream independently (each CSV is its own resume marker), then
`blend_reports.py` fuses them into one `diagnostics_combined.csv` with a single
`combined_reco` column. **Run order (from `code_insights/`):**

```bash
DATA=../data/Simulation_Data_MecanumSlipSpin_LugreAdamov
python chatter_diagnostics.py   --data-dir $DATA --out chatter_report.csv
python sampling_sensitivity.py  --data-dir $DATA --out sampling_sensitivity.csv --rates 1000,500
python tracking_gate.py         --data-dir $DATA --out tracking_report.csv
for mu in 0.3 0.5 0.8; do                                                        # χ: per-μ, whitelisted, linear-χ
  python chi_identifiability.py --data-dir $DATA --mu $mu --whitelist diagnostics_combined.csv \
         --out chi_identifiability_mu$(echo $mu | tr -d .).csv
done
python blend_reports.py         # -> diagnostics_combined.csv  (run AFTER any of the above re-run)
```

**`combined_reco` decision logic** (`blend_reports._combined_reco`), evaluated in
this precedence order:

1. `chatter_verdict == 'hash'` → **`reject_hash`** (contaminated forces).
2. `burst_flag` **and** `track_flag ∈ {marginal, missed}` → **`reject_burst`** — a
   give-up burst *confirmed by lost tracking*. A burst alone never rejects: M7 and
   the tracking gate must **agree**. This is what keeps spin_creep's 66 well-tracked
   pulse bursts (§1) while still rejecting coupled_vomega/octagon's give-up bursts.
3. `track_flag == 'missed'` → **`reject_missed`** (missed the commanded trajectory,
   no burst).
4. otherwise **kept**, with provenance sub-flags appended where present —
   `chatter` (spectral chatter), `marg_track` (marginal tracking), `burst` (a burst
   on an otherwise-tracked run) → `keep_flagged:<…>`, else plain `keep`.

The **whitelist is every trajectory whose `combined_reco` does not start with
`reject`**. So a run is excluded for exactly three reasons: `reject_hash`,
`reject_burst`, `reject_missed`.

Note rules 2 and 3 are complementary, not redundant: rule 3 uniquely rejects the
"quietly missed, no burst" runs, while rule 2 rejects nothing rule 3 wouldn't *by
count* (every give-up burst is also `missed`) — its value is the distinct **label**
and the latent burst-and-only-marginal case it would catch. Since the util level
moved to 1.10 (§5), **all rejections are now track-driven** (util drives none).

**Current tally (3 μ-batches, 5,670 trajectories):** keep 4,477, keep_flagged 868,
**reject_burst 209, reject_missed 116, reject_hash 0 → whitelist 5,345 / 5,670
(94.3 %).** All rejects are octagon (126) + coupled_vomega (199), both genuine
off-command track misses. Per μ the whitelist is **94.8 % / 94.2 % / 93.8 %** at
μ=0.3 / 0.5 / 0.8 (§7). `diagnostics_combined.csv` (65 columns) is the single source
of truth; the four per-screen CSVs are intermediates.

> **Threshold provenance.** M7 (`tau_burst = 10`, p10 denominator, `FLOOR = 0.10`)
> and the tracking gate (`track_viol_frac = 0.10`; `util_viol_frac = 0.10` *duration*
> gate at **friction-circle level `UTIL_VIOL_LEVEL = 1.10`**) are **data-tuned**. The
> six spectral chatter cutoffs (M2–M6) remain first-principles placeholders pending
> their own per-profile pass — the multisine 100%-chatter flag (§1) is the
> outstanding item there.

---

### Source artifacts
- `diagnostics_combined.csv` (**5,670 rows = 3 μ-batches × 1,890**, 65 cols — **the
  combined report**), blended from `chatter_report.csv`, `sampling_sensitivity.csv`,
  `tracking_report.csv` (all 5,670 rows) + per-μ `chi_identifiability_mu{03,05,08}.csv`
  (matched-quad subset, run with `--mu` to keep μ unpooled).
- Code: `chatter_diagnostics.py`, `sampling_sensitivity.py`, `tracking_gate.py`,
  `chi_identifiability.py`, `blend_reports.py` (+ `test_*.py` and deep-dive
  notebooks). Rendered per-profile deep-dives in `rendered_diagnostics/`.
- Threshold status: M7 burst (`tau_burst`, p10 denominator, `FLOOR`) and tracking
  (`THRESHOLDS` with `track_viol_frac = 0.10`, `util_viol_frac = 0.10`,
  `UTIL_VIOL_LEVEL = 1.10`) are **data-tuned**; the chatter spectral cutoffs
  (`CLASSIFIER_THRESHOLDS` M2–M6) and sampling `FLOOR` remain first-principles
  placeholders pending their own per-profile pass.

---

## 7. μ-grid friction-cap sweep + octagon feasibility (deck summary)

The single-μ (0.5) suite was extended to a **3-batch friction-cap grid, μ ∈ {0.3,
0.5, 0.8}** (1,890 trajectories each, identical profile/χ layout, velocity
amplitudes scaled with μ). Same tuned thresholds throughout.

**Whitelist is flat across μ — the gates generalise.**

| μ | tracked / marginal / missed | reject_burst | reject_missed | **whitelist** |
|---|---|---|---|---|
| 0.3 | 1787 / 4 / 99 | 32 | 67 | **1,791 / 1,890 (94.8 %)** |
| 0.5 | 1781 / 0 / 109 | 76 | 33 | **1,781 / 1,890 (94.2 %)** |
| 0.8 | 1772 / 1 / 117 | 101 | 16 | **1,773 / 1,890 (93.8 %)** |
| **all** | 5340 / 5 / 325 | 209 | 116 | **5,345 / 5,670 (94.3 %)** |

**The physics is monotonic where it should be.** As μ rises, the controller gains
torque authority: **reject_burst climbs (32 → 76 → 101)** and **reject_missed falls
(67 → 33 → 16)** — the *same* marginal trajectories fail by erupting in give-up
**bursts** at high μ instead of quietly **running out of friction** at low μ. Net
whitelist is therefore flat (~94 %). The friction circle is the dominant lever:
util-driven pressure (and burst count) scale directly with the μ-cap.

**Octagon lateral-infeasibility — found and (χ-quad) fixed.** ~1/3 of octagon combos
(top `vcru` tier) command a **lateral-leg velocity the platform cannot reach**
(lateral mobility caps at ~0.7×vcru_max), so they miss on *tracking* at **every μ
and every χ** (0 % success, μ/χ-independent — a reference-design bug, not friction).
The new **`straightline` β=90 probe** (heading-fixed pure-lateral sweep) measures the
lateral cap directly. The infeasible **matched χ-quad was relocated one `vcru` tier
below the cap and regenerated** (279 old files deleted, 621 new generated): the
octagon χ-quad is now **100 % feasible at all μ×χ**, lifting octagon yield 52.6 % →
**91.9 %**. The residual octagon rejects (126) are the top-tier *production* combos,
pending the full `vcru_max` re-cap.

**Other gate tunings that the μ-grid drove (see §5–§6):** the friction-circle
**violation level `UTIL_VIOL_LEVEL` was raised 1.0 → 1.10** — the util-driven
rejects merely *ride* the circle (util 1.0–1.05, never above) = at-limit saturation,
valid data, not loss of control. At 1.10 util drives **zero** rejects (all misses are
genuine track failures), and this **restored the spin_creep μ=0.3/χ=0 matched-quad
anchor** (44 combos) that low-μ + χ=0 saturation had been rejecting. χ-identifiability
is run **per-μ** (`--mu`) to avoid pooling friction levels (χ enters the force scaled
by μ).

---

## 8. μ-identifiability — `mu_identifiability.csv`

**Question:** do the *observable* friction forces **Fpar/Fperp** scale
**multiplicatively with μ** across {0.3, 0.5, 0.8} — i.e. is μ recoverable from the
forces the PINN sees? (The companion to §2's χ check.)

**Method.** μ's "dose" enters through the Coulomb/dissipative term `F ≈ μN·g(slip)`,
*gated by slip* (unlike χ, gated by spin), while the LuGre `σ₁ż` bristle-damping term
is the μ-**independent** affine `B`. The 3 μ-batches are **utilization-scaled, not
command-matched**, so there are no matched quads — μ is the *x-axis of a regression*,
matched **statistically by binning**. Pooling all whitelisted files, each per-wheel
force is **normalized by its static normal load `Nᵢ`** (`[79.6, 105.1, 69.6, 95.1] N`,
non-uniform from the CoM offset) so the four wheels collapse onto one
`F/N = μ·g(slip) + σ₁ż/N` relation — a multiplicative per-wheel difference an additive
dummy cannot remove. We **bin by signed co-directional slip** (`Vp∥ = Vpx·cosδ+Vpy·sinδ`
→ Fpar; `Vp⊥ = −Vpx·sinδ+Vpy·cosδ` → Fperp; δ = O-config per wheel) and **stratify by
total slip `|Vp|` vs the Stribeck velocity `v_str = 0.01`** into **pre-slip / transition
/ gross**. Within each (stratum, channel, bin) the per-μ force is slip-matched at the
pooled bin-mean slip, then fit `F̄(μ) = A·μ + B` (n-weighted); headline = the **absolute
μ-swing** `|ΔF| = |F̄(0.8) − F̄(0.3)|` in N (units-trap-safe — *not* R²/F), with the
**multiplicative fraction** `A·μ/(A·μ+B)` and the held-out-μ=0.5 `curvature_resid`.

> **Model choice (validated).** μ enters **linearly** (no μ²: unmotivated for forces
> and ill-conditioned on a 3-point grid — the §2 lesson). Held-out μ=0.5 is recovered
> to `curvature_resid ≈ 0.1–0.3 N` against 27 N swings (~1%), confirming linear-in-μ.
> **Ablation:** adding the co-directional quadratic `Vcod²` to the slip controls
> (`[1,Vcod,|Vp|,|Vp|²]` → `[1,Vcod,Vcod²,|Vp|,|Vp|²]`) moves the swing by `≤0.3 N`
> everywhere (mostly `<0.01 N`) — `Vcod²` is inert; linear-`Vcod` is the primary model.

### Observation

Per-regime verdict (n-weighted median over bins; QUAD model, identical to LIN):

| regime (`\|Vp\|`) | channel | `mult_fraction` | `\|ΔF\|` bulk [N] | `\|ΔF\|` high-slip [N] | verdict |
|---|---|---|---|---|---|
| **pre-slip** (<0.01) | Fpar / Fperp | — | — | — | **unfittable → μ blind** |
| **transition** (0.01–0.03) | Fpar | 0.37 | 2.9 | — | partial |
| | Fperp | 0.80 | 6.2 | — | mostly multiplicative |
| **gross** (≥0.03) | Fpar | 0.24 | 5.5 | **~28** | slip-gated (see below) |
| | Fperp | 0.96 | 6.4 | **~26** | strongly multiplicative |

The "bulk median" undersells the gross stratum: multiplicativity is itself
**co-directional-slip-gated**. Within gross slip, both channels rise from
affine-`B`-dominated at small `|Vcod|` (`mult ≈ 0.2`, `|ΔF| ≈ 3–7 N`) to nearly pure
multiplicative at large `|Vcod|` (`mult ≈ 0.95–0.98`, `|ΔF| ≈ 26–29 N`) — see
`images_and_plots/mu_identifiability_gross.png`. The median is low only because the
small-`|Vcod|` bins hold most of the samples (~150 M vs ~2 M). Signed bins are
sign-consistent (±`Vcod` mirror: force flips sign, swing/mult equal).

### Inference
- **μ is strongly identifiable at high co-directional slip** — `|ΔF| ≈ 26–29 N` (μ
  0.3→0.8), `rel_swing ≈ 0.75–0.91`, a huge signal vs any plausible force-noise
  floor. **Weakly identifiable at low co-directional slip** (`|ΔF| ≈ 3–7 N`, large
  μ-independent intercept) — even though that is where most of the data lives.
- **μ-multiplicativity is slip-gated, not regime-binary.** Each channel's *own* axis
  decides its regime: a wheel grossly sliding sideways can have a near-clean drive
  axis. This is exactly why the total-`|Vp|` **stratum** and the per-channel
  **`Vcod` bin** carry different information — keeping both axes was load-bearing.
- **The Fpar/Fperp asymmetry inverts the prior hypothesis.** The drive axis (Fpar)
  carries the **large μ-independent intercept** at low slip (`B ≈ 7–19 N`: structural
  /bristle/viscous drive force from the rolling constraint + actuator), while the
  free-roll axis (Fperp) has `B ≈ 0` so it reads cleanly multiplicative even at low
  slip. Fpar dominates the *absolute* swing at high slip; Fperp is the *cleaner*
  μ-reader at low slip.
- **μ is not identifiable in pre-slip** — the cells are unfittable (forces carry no
  slip-resolved μ-signal), consistent with bristle-dominated `F ≈ σ₀z`.

### Explanation
- **Why normalize by `Nᵢ`, not wheel dummies:** the per-wheel force difference is
  `μ·Nᵢ·g(slip)` — *multiplicative* in the signal, which an additive offset cannot
  remove. With CoM-offset loads (`Nᵢ` spanning 69.6–105.1 N), `÷Nᵢ` homogenizes all
  four wheels onto one relation, drops 4 regressors, and makes the within-bin
  "constant slope" assumption hold (raw `A = Nᵢ·g` still varies with `Nᵢ` in a bin).
- **Why bin by *signed* co-directional slip:** friction opposes slip (measured
  `corr(Vcod,F) ≈ −0.5…−0.9`), so binning by `|Vp|` would sign-cancel; signed bins
  preserve the drive/free-roll anisotropy that the A/B split lives in.
- **Why absolute swing, not R²/F:** identical units-trap to §2 — ~10⁶ samples/bin make
  F always "significant"; identifiability under real noise is the **absolute** N-swing
  vs the floor.
- **Why the changeover bins (`|Vcod|≈0.09–0.13`) show the only real curvature**
  (`curvature_resid ≈ 2.6–3.1 N`): that is the affine→multiplicative transition, an
  expected slip nonlinearity — *not* a μ² effect (it is along the slip axis, not μ).

**Hand-back (gates the PINN's Approach-1 μ-ID, §5 of the handoff):** μ̂ is recoverable
from observable forces **only at high co-directional slip**; the affine-`B` floor at
low slip (especially Fpar) means the **affine force-head hook must be active**, and μ
confidence should be **gated on `|Vcod|` / saturation** — the slip analogue of χ's
spin-gated confidence (§2). The noise-floor threshold for a hard identifiable/not
verdict remains deferred to the user; `|ΔF|` is reported in Newtons throughout.

### Deep dive — the identifiable tail (where μ is actually readable)

μ̂ is recoverable **only at high co-directional slip**, and that regime is a thin slice
of the data. Over the 5,345 whitelisted files (1.354 B wheel-samples; regimes on total
`|Vp|` — full breakdown in §9.1):

> **~45 % of data is gross slip, but only ~4.0 % (Fpar) / ~4.7 % (Fperp) sits at high
> co-directional slip `|Vcod| ≥ 0.15`** — ≈9–10 % of gross slip, ≈60 M samples.

- **Why the tail is thin.** High *co-directional* slip means the wheel slides along its
  **drive axis** — friction-circle saturation / aggressive maneuvers. Most gross slip is
  *perpendicular* (free-roll `V⊥`): the wheel slides sideways while its drive axis stays
  near-clean. So 45 % gross collapses to ~4–5 % at high `|Vcod|`.
- **The other ~95 % is μ-blind by construction** — the drive force there is the affine-`B`
  structural/bristle term (μ-independent), so a forward/inverse loss averaged over all
  samples is dominated by μ-invisible data and the μ head sees almost no gradient.
- **Consequence for training.** ≈60 M samples is ample in *count*, but a few-percent
  *fraction* — so μ confidence must be **gated on `|Vcod|`/saturation** (read μ only in the
  tail) or the high-`|Vcod|` data **tail-weighted**, else the affine-`B` bulk drowns the
  estimate. χ is thinner still: it needs high co-directional slip *and* high spin (§2 deep
  dive), so its identifiable intersection is a few percent. This is the quantitative basis
  for the confidence-gating in the Hand-back above.

### Source artifacts
- `mu_identifiability.csv` (40 fitted cells = 3 strata × 2 channels × signed-slip
  bins; pre-slip NaN). Code: `mu_identifiability.py` (streaming sufficient-stats, no
  `--mu` — pools all μ; `--whitelist diagnostics_combined.csv`). Figure:
  `images_and_plots/mu_identifiability_gross.png`.

---

## 9. Other interesting inferences

Three model-free results from the pre-training analytics that bear on the PINN's
input/output design — slip-regime coverage, gate-variable observability, and force
anisotropy. All over the 5,345 whitelisted files (1.354 B wheel-samples), streaming.

### 9.1 Slip-regime data coverage

Regime on total per-wheel slip `|Vp| = hypot(Vpx,Vpy)` vs the Stribeck velocity
`v_str = 0.01` (the gating §8 and §10 use):

| regime | `\|Vp\|` [m/s] | wheel-samples | share |
|---|---|---|---|
| pre-slip | < 0.01 | 437.4 M | **32.3 %** |
| transition | 0.01 – 0.03 | 309.0 M | **22.8 %** |
| gross | ≥ 0.03 | 608.0 M | **44.9 %** |

**Per μ (gross share):** μ=0.3 → 46.4 %, μ=0.5 → 44.7 %, μ=0.8 → 43.5 % — gross shrinks as
friction rises (more grip, less sliding; pre-slip grows 28 → 38 %). **Inference:** ~⅓ of
operation is pre-slip (bristle-dominated, μ/χ-blind, §8), ~45 % gross; the
friction-parameter signal lives in the high-co-directional-slip sub-slice of gross (§8
deep dive). The grids span every regime — the dataset is not regime-starved.

> **Figure (deck candidate, no PNG yet):** stacked bar of the three regime shares
> (32.3 / 22.8 / 44.9 %) with the per-μ split — sourced from the slip-coverage count.

### 9.2 Gate-variable observability — roller (hidden γ̇) fraction — `roller_slip_fraction.csv`

The μ-gate rides on co-directional slip `V∥` and the χ-gate on contact spin `ω_z`; both
contain the unmeasurable roller rate `γ̇`. Per wheel each decomposes **exactly** (validated
to machine precision against the stored `Vpx/Vpy/wz`) into a measurable part + a roller
part. Headline metric: roller **energy** fraction for spin (= squared error of the
measured-only approximation), roller **magnitude** (L1) fraction for slip (the velocities
are linear sums; energy fraction is degenerate under the near-cancellation of `V⊥`).

| gate variable | roller share | reading |
|---|---|---|
| `ω_z` (χ-gate) | **~87 % energy / ~90 % magnitude** | measured-only approx never beats 77 % RMS error in *any* regime (0 cells < 30 %) |
| `V∥` (μ-gate, drive) | **~0.2–0.4 %** | essentially all measurable |
| `V⊥` (free-roll) | ~24–48 % | partly hidden |

**Inference**
- **χ-gate roller reconstruction is mandatory** — `ω_z` is ~90 % hidden roller in every
  regime, so a deployable χ-reading requires the encoder/observer to reconstruct contact
  spin (ties to §2 deep dive: resolving `|ω_z|` is exactly what recovers the χ-signal).
- **μ-gate rides on measurables** — `V∥` is ~100 % body-velocity + wheel-rolling; roller
  reconstruction is a sanity check there, not a requirement.
- **Asymmetric:** the cleaner μ-reader `Fperp` is gated by the partly-hidden `V⊥`, while
  the fully-measurable `V∥` feeds the noisier `Fpar` — so the best μ-signal also benefits
  from the observer.

**Explanation:** `ω_z = ψ̇ + γ̇·sinθ̃·cosδ`, and the roller rate `γ̇` is large in every
operating regime, so body yaw `ψ̇` is a minor part of contact spin. Whereas
`V∥ = (Vx − ψ̇·(py+DY) − ω·R)·cosδ + …` is all measurable — the roller's surface velocity
projects onto the free-roll axis `V⊥` (it enters `V∥` only at O(θ̃²), the contact-geometry
curvature term).

> **Figure:** `images_and_plots/roller_slip_fraction.png` — roller share of contact spin
> `ω_z` (energy + magnitude vs `|ω_z|`) and of drive/free-roll slip `V∥`/`V⊥` (magnitude
> vs `|slip|`); shows `ω_z` ~90 % roller, `V∥` ~0 %.

### 9.3 Force anisotropy — `|Fperp|/|Fpar|` (the no-slip assumption)

Ideal / no-slip Mecanum models put all contact force on the roller axle (`Fpar`) and set
`Fperp ≈ 0`. The composite LuGre+Adamov model deliberately does not — how badly is that
violated? Distribution of `r = |Fperp| / (|Fpar| + 1e-6)`:

> **Figure (deck candidate, no PNG yet):** CDF/histogram of `r = |Fperp|/|Fpar|` split by
> slip regime (median 0.39; ~23 % above 1) — recompute the histogram from the streaming
> `force_aniso` pass (the `.npz` was scratch, not saved).

| regime | median `r` | % (`r` > 1) | RMS ratio |
|---|---|---|---|
| pre-slip | 0.40 | 17.1 % | 0.52 |
| transition | 0.34 | 22.5 % | 0.54 |
| gross | 0.42 | 26.9 % | 0.66 |
| **ALL** | **0.39** | **22.8 %** | **0.62** |

**Inference:** the free-roll force is *typically ~40 %* of the drive force, *exceeds* it in
~23 % of samples, and dropping it discards ~38 % of the force energy (RMS ratio 0.62) —
worst in gross slip. So "force only along the axle" is materially wrong; modelling both
channels (and the composite friction law itself) is justified, and it is why `Fperp` is a
usable — and per §8 the *cleaner* — μ-reader.

---

## 10. Force-law adequacy for the PINN forward head — `force_mu_chi_gated.csv`

**Question:** can the analytic form `F = μ·(A + C·χ) + B` — with `A, B, C` as functions of
slip/regime — approximate the per-wheel friction force well enough to serve as the PINN's
**forward** friction head?

**Method.** Gated, regime-resolved, **multiplicative** (an ungated *additive*
`A·μ + B + C·χ` global fit sees ~0 — sign-cancellation across signed slip + regime
averaging + additive mis-specification). Per (signed co-directional slip bin × `|Vp|`
stratum × channel), on the matched-χ-quad **and** full sets, regress `F/Nᵢ` on
`[1, μ, slip controls, μ·χ]` (and the spin-aware `μ·|ω_z|·χ`). `A = dF/dμ`, `B` = μ/χ-
independent intercept, `C` = χ's modulation of the μ-scaled shape. The χ-term contribution
is in §2's deep dive; here we report **form adequacy `R²_form`** (the `μ·A + B + slip`
backbone) and the recovered `A`.

> **Form choice.** `μ·(A + C·χ) + B` is the LuGre linearization: `F = μN·g(slip;χ) + σ₁ż`
> with `g ≈ g₀ + g₁·χ` gives `A = N·g₀`, `C = N·g₁`, `B = σ₁ż`. The additive
> `A·μ + B + C·χ` is wrong — χ cannot move the force without μ present to mobilize Coulomb.

### Observation
`A` recovers §8 exactly: per signed bin `|A| = 53–60 N` at high `|Vcod|` (sign-flipping
with slip direction), `|A|`-average ≈27 N — matching §8's 26–29 N μ-swing (`A ≈ swing/0.5`).

Form adequacy `R²_form` (fraction of force variance the algebraic form explains):

> **Figure (deck candidate, no PNG yet):** `R²_form` per channel vs `|Vcod|` — Fperp flat
> at ~0.8, Fpar rising 0.14 → 0.98 — from `force_mu_chi_gated.csv`. The clean visual of
> "algebraic vs history-dependent" force.

| channel | quad | full | vs slip (gross stratum) |
|---|---|---|---|
| Fperp (free-roll) | 0.76–0.80 | 0.71–0.78 | ~0.86 (low) → 0.98 (high) |
| Fpar (drive) | 0.26–0.35 | 0.11–0.16 | **0.14 (low) → 0.98 (high)** |

### Inference
- **The form is the right backbone, not the whole model.** Pooled `R²_form ≈ 0.45–0.59`;
  it recovers the μ-multiplicativity (§8) and a real χ effect (§2 deep dive) — but as an
  *instantaneous algebraic* map it caps at ~½ the force variance. The residual ~40–55 % is
  the LuGre bristle term `σ₁ż`, which depends on the bristle **state** `z(t)` — the history
  the recurrent encoder must supply.
- **The split is per-channel and tells the GRU where it is needed.** `Fperp` is ~75–80 %
  algebraic everywhere (clean `μ·g(slip)`, §8). `Fpar` is **bimodal**: ~98 % algebraic in
  gross drive-slip (Coulomb-dominated) but only ~14 % at low drive-slip — that low-slip
  structural drive force (the affine-`B`) is where the recurrent state is mandatory.
- **Net forward head:** `F = μ·(A(slip) + C(slip,|ω_z|)·χ) + B(slip)` is the correct
  analytic backbone supplying ~½ the force (most of `Fperp`, gross-slip `Fpar`); the
  GRU/SSM supplies the state-dependent remainder (concentrated in low-drive-slip `Fpar`);
  and the χ channel requires reconstructed `|ω_z|` (§2 deep dive, §9.2).

### Source artifacts
- `force_mu_chi_gated.csv` (612 cells = 2 datasets × 2 `|ω_z|`-schemes × 2 channels × 3
  strata × signed-slip bins [× `|ω_z|`-bins when binned]). Code: `force_mu_chi_gated.py`
  (streaming sufficient-stats; reduced `[1, μ, slip]` always yields `A,B`; two χ forms by
  column subset; pools all μ, no `--mu`; `--whitelist`).
