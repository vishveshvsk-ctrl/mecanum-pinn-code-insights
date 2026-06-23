# Trajectory chatter-diagnostics — execution plan (profiles era)

> **⚠ Status — execution plan, partly superseded. For latest relevance see
> [`TRAJ_DIAGNOSTICRESULTS.md`](TRAJ_DIAGNOSTICRESULTS.md).** This file is the
> original spec + physics derivations (the source `chatter_diagnostics.py` points
> to for "the physics derivation behind every band/threshold"). It remains the
> canonical derivation reference, but as-run findings, the final classifier logic,
> the tuned thresholds, and the whitelist live in `TRAJ_DIAGNOSTICRESULTS.md` —
> consult that first for anything current. Known drift to be aware of here:
> - The **burst metric `msat_burstiness`** (now the code's M7) was added *after*
>   this plan and is **absent below**. The "M7" in §3/§5 of this file is `wall_s`
>   (a report annotation), a different metric — the numbering does not match the
>   code. The burst metric is documented in `TRAJ_DIAGNOSTICRESULTS.md §1`.
> - **Thresholds have since been data-tuned** (M7 `tau_burst=10` on a p10
>   denominator + `FLOOR=0.10`; tracking `util_viol_frac=0.10`). Treat any
>   threshold values in this plan as first-principles placeholders.

> **Prerequisite task (sampling-rate sensitivity) — see §12 at the bottom.** It
> must run before the chatter threshold-tuning pass, because the sample rate the
> PINN trains on changes what every metric (and the network) ever sees.

Successor to `Mecanum_PINN_TrajectoryDiagnostics_1.ipynb` (chatter lane) for the
profiles-era pipeline. Scope is **narrow and specific**: flag trajectories whose
force/torque signals carry *unwanted chatter* — ASMC sliding-mode switching that
leaks into the physics, or solver/numerical hash — while **not** penalising the
two physical oscillations the PINN must learn: **roller-switching ripple** and
**LuGre stick–slip oscillation**.

This is the spectral/coherence lane. It is complementary to (not a replacement
for) the physics-residual lane (Diagnostic-2 successor) and the decoherence-curve
idea in the parent brief; label error stays the primary whitelist gate. Chatter
flagging catches a failure mode those don't isolate cleanly: high-frequency
control/numerical contamination that is statistically self-consistent yet
un-learnable.

---

## 0. What changed since Diagnostics-1 (must respect)

| Aspect | Diagnostics-1 (old) | This plan (profiles era) |
|---|---|---|
| Filename | `test_mu_..._{motion}_..._beta..._amp..._chi_....arrow` | `<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow` |
| Grouping var | `motion` ∈ {straightline,…} | `profile` ∈ {octagon, long_circle, spin_creep, coupled_vomega, spiral_orbit, multisine, ellipse} |
| Force channels | **projected in Python** from Fx/Fy | **`Fpar_i`, `Fperp_i` precomputed** in the Arrow |
| Control split | only `Msat_i` | `Msw_i` (switching), `Meq_i` (equivalent), `Msat_i` (applied) |
| Slip / friction state | `Vpx/Vpy/wz` (sometimes absent) | `Vpx/Vpy/wz_i` **+ bristle `zx/zy/zs_i` + `util_i`** always present |
| Save rate | 1000 Hz, decimated ×5 → 200 Hz | **2000 Hz** dense uniform grid |
| Categories | ripple (signal) vs chatter+hash (noise) | ripple **and LuGre** (signal) vs ASMC-chatter vs hash |
| Label re-injection | n/a | **none needed** — all labels are columns (no Julia round-trip) |

Two correction notes carried from the codebase, not the brief:
- **`naccept`/`nreject` are NOT logged, and the per-trajectory solver screen is
  dropped.** `jobs_log_profiles.jsonl` records only `profile, combo_idx, mu,
  chi, friction_case, friction_model, wall_s, sim_t, sim_T, retcode, err_msg,
  thread, timestamp`; the JLD2 sidecar stores `sol_t/sol_u/labels/params/asmc/
  meta/traj_cfg` — neither has `sol.stats`. **Decision:** the solver-ablation
  study already fixes the integrator globally, and the spectral `hash_fraction`
  metric (M2) catches the *consequence* of any badly-integrated trajectory
  directly from the recorded data, so a per-trajectory `nreject` screen is
  redundant. The driver is **not** patched. `wall_s` is kept only as a
  free annotation column in the report (already in the JSONL), never a gate.
- **`data.py` `_FNAME_RE` is still the old beta/amp scheme** (line ~80). The
  whitelist this diagnostic emits is only consumable once that regex is migrated
  to the profile scheme. The migrated regex is given in §6.

---

## 1. The three oscillation sources (physics intent)

All three look like "noisy force" by eye. They separate by **what sets their
frequency** and **where they originate**:

### 1a. Roller-switching ripple — SIGNAL (keep)
- Origin: `sawtooth_tanh(θ)` (notebook), `TANH_K=60`, period `2π/12` in wheel
  angle `θ_i`. Enters the slip kinematics via `DYᵢ = Ra·tanδ·tan(θ̃ᵢ)`.
- Signature: time-frequency **ridge** at `f_roll(t) = 12·|ωᵢ(t)|/(2π) =
  1.910·|ωᵢ|` Hz (+ harmonics 2·,3·), **sweeping with wheel speed**; and a
  **clean closed curve** when `Fpar/Fperp` are folded over `θ mod 2π/12`.
- Frequency reach (computed): f₁ up to 28.7 Hz, f₃ up to 86 Hz at |ω|=15 rad/s.

### 1b. LuGre stick–slip oscillation — SIGNAL (keep) **← the new category**
- Origin: bristle ODE `ż = Vp − σ₀·sₜ/gₜ·z` (σ₀=1.64e3 1/m, Stribeck `v_str=0.01`
  m/s, `stiction_ratio=1.1`). Force `F = −N(σ₀z + σ₁ż)`. The micro-damping
  `σ₁ż` term and the falling Stribeck curve near `v_str` produce genuine
  stick–slip oscillation.
- Signature: a **slip-driven low-pass corner** `f_lugre ≈ σ₀·Vp/(gₜ·2π) ≈ 653·Vp`
  Hz (computed) — the bristle force tracks slip below this corner. Its
  **amplitude envelope tracks the slip state** (`|Vpᵢ|`, `util_i`), it is **not**
  present in `Msw_i` (it is a friction-law output, not commanded), and it is
  **not** phase-locked to `θ`.
- Frequency reach (computed): 6.5 Hz at Vp=v_str, **65 Hz at Vp=0.1, 196 Hz at
  Vp=0.3.** This is the crux: **f_lugre can exceed the 3rd roller harmonic**, so
  the "above the roller band ⇒ hash" rule used by Diagnostics-1 would
  mis-flag high-slip LuGre. The noise floor must subtract a per-sample LuGre
  band computed from the saved `Vp` (§3, metric M2).

### 1c. ASMC chatter + numerical hash — UNWANTED (flag)
- ASMC chatter: switching wrench `M_sw = −K·tanh(s/eps)`, boundary layer
  `eps = max(0.025·PEAK_V, 0.001)` (driver). High effective gain `K/eps` →
  limit-cycling of the sliding variable at a frequency **set by the controller,
  unrelated to ωᵢ or Vp**. It is **present in `Msw_i`** and propagates
  Msw→Msat→wheel ODE→ωᵢ→Fᵢ. Low-speed profiles get the thinnest `eps` (floored
  at 0.001) and are the most chatter-prone — **spin_creep is the prime suspect.**
- Numerical hash: broadband, no periodic structure, no relation to ωᵢ/Msw/Vp;
  from stiff-solver step thrashing near friction-circle saturation.

**Discriminator summary**

| Property | Roller ripple | LuGre | ASMC chatter | Numerical hash |
|---|---|---|---|---|
| Freq set by | `|ω|` (sweeps) | `Vp` (slip) | controller `eps` (≈fixed) | solver (broadband) |
| Phase-locked to θ | **yes** | no | no | no |
| In `Msw_i`? | no | no | **yes** | no |
| Envelope tracks slip/util | partly | **yes** | no | no |
| Coherent with `Msat/ω`? | yes (via friction) | yes (via friction) | **yes** | **no** |
| Verdict | keep | keep | flag (learnable-but-undesirable) | flag (reject hard) |

The two "flag" columns are separated by `Msw` coherence: chatter is coherent
control energy; hash is incoherent. The two "keep" columns are separated from
chatter by the θ-fold (ripple) and the slip-envelope correlation (LuGre).

---

## 2. Data sources

Per trajectory, **everything is in the Arrow** (no Julia, no JLD2 needed for the
core metrics):

- Required contract: `Vx, Vy, psi_dot, w1..4, theta1..4, Msat_1..4,
  Fx_1..4, Fy_1..4, Mz_1..4, time`.
- Profiles-era bonus columns this diagnostic leans on:
  `Fpar_1..4, Fperp_1..4, util_1..4, Msw_1..4, Meq_1..4,
  Vpx_1..4, Vpy_1..4, wz_1..4, zx_1..4, zy_1..4, zs_1..4`.
- Per-job from `jobs_log_profiles.jsonl`: `wall_s`, `retcode` (join key
  `(profile, combo_idx, mu, chi, friction_model)`).

JLD2 sidecar: only if we later want `asmc.eps` / `params` per run for
diagnostics annotation. Not required for the metrics.

---

## 3. Metric set

Each metric is per-wheel then reduced (median over 4 wheels) unless noted.
**Two-rate processing** (see §5): ridge/fold metrics on a ×8-decimated copy
(250 Hz); HF/hash and Msw metrics on the **full 2000 Hz** signal.

**M0 — `util_sat_frac` (near-free pre-screen).**
Fraction of time `util_i > 0.8` (master-gate budget) and `> 1.0` (violation),
max over wheels. Uses the saved `util_i` directly — no spectral work. Sustained
`util≈1` is where stick-slip, ASMC chatter, and solver thrashing all peak (brief
finding 3). Not a reject on its own (a profile may legitimately ride the budget);
it gates which trajectories get the expensive spectral pass and contextualises a
high hash score.

**M1 — `ridge_concentration` (ripple = signal).** *[250 Hz]*
Fraction of off-DC STFT energy of `Fpar_i` within `±tol_rel·(h·f_roll)` of
harmonics h=1..3, `f_roll = 1.910·|ωᵢ|`, `tol_rel≈0.25`, 1 Hz floor. High ⇒
roller-dominated. Same idea as Diagnostics-1 but on the **precomputed `Fpar_i`**.
Keep the diag-1 caveat: near stall the 1 Hz floor inflates this — never a
standalone pass.

**M2 — `hash_fraction` (the core chatter metric, LuGre-aware).** *[2000 Hz]*
Welch PSD of `Fpar_i` and `Fperp_i`. Define a **per-sample physical ceiling**
```
f_phys(t) = max( 3·f_roll(t),  K_lugre·f_lugre(Vp_i(t)),  f_floor )
f_lugre   = σ₀·|Vp_i| / (g_t · 2π)  ≈ 653·|Vp_i|   (g_t≈μ; refine with stribeck_g)
```
`K_lugre≈1.5`, `f_floor≈40 Hz`. Convert to a single trajectory cutoff
`F_hash = percentile_95_t( f_phys(t) )` (robust to spikes). `hash_fraction` =
PSD energy above `F_hash` / total off-DC energy. **Low = clean.** This is the
metric that does NOT exist in Diagnostics-1 in usable form — diag-1 decimated to
200 Hz and thus filtered out the very hash it meant to detect, and used a fixed
`3·f_roll` ceiling that ignores LuGre.

**M3 — `control_chatter_index` (ASMC switching fingerprint).** *[2000 Hz]*
`HF_energy(Msw_i above F_hash) / (RMS(Meq_i) + ε)`. ASMC chatter lives in the
pure switching wrench `Msw_i`; the equivalent control `Meq_i` is smooth. High ⇒
the controller is chattering. Roller ripple and LuGre contribute **zero** here
(they are friction outputs, not commanded) — this is the cleanest separator the
new columns unlock.

**M4 — `msw_force_coherence_hf` (chatter vs hash split).** *[2000 Hz]*
Magnitude-squared coherence `γ²(Msw_i, Fpar_i)` (and vs `ωᵢ`) averaged over the
band `> F_hash`.
- high coherence + high M2/M3 ⇒ **coherent ASMC chatter** (marginal: it is a
  faithful, if undesirable, control response — borderline learnable).
- low coherence + high M2 ⇒ **numerical hash** (reject hard: HF energy not
  explained by any input).

**M5 — `hf_slip_modulation` (LuGre rescue).** *[2000 Hz]*
Pearson correlation between the HF-force envelope (Hilbert magnitude of `Fpar_i`
band-passed in `[F_hash_lugre_lo, F_hash]`) and `|Vp_i|` (or `util_i`).
- high ⇒ the HF energy is **slip-modulated LuGre** (signal) → suppress a hash
  flag even when M2 is elevated.
- low/flat ⇒ white hash (no slip envelope) → confirm reject.
This is the explicit guard that keeps genuine stick-slip out of the reject bin.

**M6 — `theta_fold_tightness` (ripple periodicity, keep).** *[250 Hz]*
Fold high-passed `Fperp_i` (and `Mz_i` for χ>0) over `θ mod 2π/12`; report
`median_bin IQR / range(median over bins)`. Low = clean periodic-in-θ ripple.
Carry diag-1's χ-branch (Mz over full pitch for χ>0; HP-Fperp at pitch edges for
χ=0).

**M7 — `wall_s` (report annotation only; NOT a gate).** *[scalar]*
Joined from `jobs_log_profiles.jsonl` and carried into the report as context.
The per-trajectory solver-stress *screen* is **dropped** (see §0): the ablation
fixes the integrator globally and M2 catches mis-integration consequences from
the data. `wall_s` (and its within-`(profile, friction_model)` z-score) is left
in the CSV purely so a human can eyeball whether a flagged trajectory was also a
slow solve — no classifier logic depends on it.

---

## 4. Classifier

Per trajectory, after computing M0–M6, assign `clean / chatter / hash / marginal`:

```
hash      if  M2 (hash_fraction) > τ_hash
              AND M4 (coherence)  < τ_coh_low
              AND M5 (slip_mod)   < τ_slipmod         # not LuGre  → REJECT
chatter   if  M3 (control_chatter_index) > τ_ctrl
              AND M4 (coherence)  > τ_coh_high        # coherent w/ Msw → KEEP, FLAGGED
marginal  if  exactly one soft check trips,
              or util_sat_frac high with mild M2
clean     otherwise (ripple + LuGre only)
```

LuGre is protected by construction: M5 high vetoes a hash verdict; M3 ignores
friction-only energy; M6 confirms ripple. A trajectory dominated by genuine
stick-slip lands in `clean`.

**Whitelist policy (decided):** `hash` is the **only** hard reject. `chatter`
(coherent ASMC switching) is **kept with a flag** — the PINN sees a faithful,
if undesirable, control→force map; the report carries a `chatter_flag` column so
it stays a per-profile knob. `clean` + `chatter(flagged)` + (opt) `marginal`
form the whitelist.

**Thresholds are data-driven, not asserted.** Run once, plot M2/M3/M4/M5
histograms **split by profile**, set cutoffs at the natural shoulders (diag-1
lesson). Expect per-profile differences: spin_creep (thin `eps`) shifts M3
right; high-slip profiles (coupled_vomega, spiral_orbit) shift M2's *physical*
content up — which M2's LuGre-aware ceiling already absorbs.

Whitelist = `clean` (optionally + `marginal`). Composite key parity with the new
filename: `(profile, combo_idx, mu, chi, friction_model)`.

---

## 5. Processing architecture

- **Streaming, per-file** (read → metrics → delete → advance), as in
  `scan_scaling_factors_windowed_v3.ipynb` — 2000 Hz × ~60 s × ~60 cols per file
  is large; never hold the corpus in RAM.
- **Two-rate**: anti-alias-decimate ×8 → 250 Hz for M1/M6 (ridge/fold need
  Nyquist > 86 Hz for the 3rd harmonic; 250 Hz Nyquist=125 Hz is enough).
  Run M2/M3/M4/M5 on the **full 2000 Hz** signal — decimating first would filter
  out the chatter. `theta_i` is index-sliced, never filtered (it accumulates).
- **Filename parse** (new regex, §6) → `(profile, combo, mu, fc, fm, chi)`;
  group thresholds/plots by `profile`. `ellipse` is **PosRef** (world-frame
  reference columns `xo_des,…`); the others are **VelRef** (`Vx_des,…`). The
  chatter metrics ignore the reference block, so this only matters for labelling.
- **Cost**: diag-1 was 384 s / 1525 files at 200 Hz. M2–M5 at 2000 Hz on Fpar/
  Fperp/Msw (a few Welch PSDs + 2 coherences + 1 Hilbert per wheel) is the
  dominant term. Budget ~0.5–1.5 s/file → ~15–35 min for ~1485 jobs,
  single-process. Parallelise per-file if needed. M0/M7 are ~free.

---

## 6. Outputs & integration

1. `trajectory_chatter_report.csv` — one row per Arrow file: parsed key + M0–M7
   + verdict. Written under the data tree / a diagnostics output dir (never
   inside `code_insights/`).
2. `pinn_training_whitelist.txt` — bare filenames with verdict ∈ {clean,
   chatter(flagged), +marginal opt-in}; only `hash` excluded. One per line;
   `#` comments allowed. The report's `chatter_flag` column lets the trainer
   later drop coherent-chatter runs per-profile without re-running the sweep.
3. **`data.py` migration (separate, required for the whitelist to be usable):**
   replace the old `_FNAME_RE`/`parse_arrow_filename`. Proposed regex (anchors
   `fm` to the known set so the underscore-bearing profile/fm don't collide):
   ```python
   _FNAME_RE = re.compile(
       r'^(?P<profile>.+?)_c(?P<combo>\d{3})_mu_(?P<mu>[0-9.eE+\-]+)'
       r'_case(?P<fc>\d+)_(?P<fm>lugre_adamov|lugre_uncoupled)'
       r'_chi_(?P<chi>[0-9.]+)\.arrow$')
   ```
   and add `Fpar/Fperp/util/Msw/Meq` to the optional column set if the trainer
   wants them.

---

## 7. Validation plan (before trusting the verdicts)

1. **Synthetic injection.** Take a known-clean trajectory; add (a) a fixed-freq
   tone into `Msw`→`Fpar` (synthetic ASMC chatter), (b) white HF noise into
   `Fpar` only (synthetic hash), (c) a slip-modulated HF burst (synthetic LuGre).
   Confirm the classifier labels them chatter / hash / clean respectively. This
   is the acceptance test for M3/M4/M5.
2. **Cross-check vs the physics-residual lane.** `hash` trajectories should also
   show elevated Diagnostic-2 `L_simpson`/`L_cont_FD` residuals; `chatter` need
   not. Disagreement is informative, not a failure.
3. **Known-bad regression.** Run on a slice of the legacy TRBDF2-era data (brief
   finding 1: errF≈28–31 N decoherence). Diagnostics-1 labelled that corpus
   0 NOISY — if this classifier also passes it as clean, the HF/coherence
   metrics are still blind and need retuning. (Caveat: legacy data is the old
   filename/column scheme; needs the old loader — treat as a separate smoke test.)

---

## 8. Decisions (resolved)

1. **Deliverable form** — **Python `chatter_diagnostics.py` (batch) + thin
   notebook (deep-dive & threshold histograms).** Rationale: the chatter
   diagnostic is pure DSP on already-recorded Arrow columns (no ODE re-solve, no
   label re-injection), so the "a `.jl` is cleaner" lesson from the solver
   ablation — which *was* native Julia ODE work — does not transfer. The Julia
   env has no `DSP.jl`/`FFTW.jl`, coherence has no batteries-included function,
   and scipy.signal + the existing diag-1/2 metric code are mature and portable.
   The execution-cleanliness the parent chat valued (script over notebook,
   streaming/resume/parallel) is honoured by making the **batch a CLI script**
   in the `Data_Generation_Julia.jl` mould; the notebook is only the visual
   deep-dive and the threshold-picking histograms.
2. **Solver screen** — **dropped** (see §0/M7). No driver patch. `wall_s` kept as
   a report annotation only.
3. **Coherent ASMC chatter** — **kept with flag** (see §4). Only `hash` is a hard
   reject.
4. **Thresholds** — **sweep first, you pick.** The script runs the full metric
   pass and emits per-profile histograms of M2/M3/M4/M5; cutoffs are chosen at
   the natural shoulders before the classifier is finalised. No guessed
   thresholds shipped.

---

## 9. Implementation steps

1. `chatter_diagnostics.py` — CLI batch script (mirrors `Data_Generation_Julia.jl`
   discipline: streaming per-file, resume on existing report rows, optional
   per-file parallelism). Imports the new filename regex + column names from
   `data.py` (no drift). Two-rate decimate; pulls
   `Fpar/Fperp/util/Msw/Meq/Vp*/wz/w*/theta*/Mz/time`.
2. Metric kernels M0–M6 (vectorised per wheel; full-rate vs decimated split),
   porting diag-1's ridge/fold/lag-corr where reusable.
3. Thin notebook — single-trajectory deep-dive (STFT+ridge, θ-fold, Msw↔F
   coherence, slip-envelope overlay with the LuGre band drawn), mirroring
   Diagnostics-1 §5–9 on the new columns.
4. Synthetic-injection acceptance test (§7.1) — the gate on M3/M4/M5 before
   trusting verdicts.
5. Batch sweep → report CSV (+`wall_s`, `chatter_flag`) → per-profile histograms.
6. Threshold confirmation pass with you → finalise classifier → whitelist.
7. (separate) `data.py` regex migration (§6).

---

## 12. Sampling-rate sensitivity study (prerequisite task)

**Question:** the PINN trains more cheaply on a coarser grid, but lower rates
risk losing or aliasing physical content. Is the difference between **2000 / 1000
/ 500 Hz** significant or insignificant for training? Decide it on evidence.

**Two downsampling modes, opposite failure modes** (study both):
- *anti-aliased decimation* — low-passes first; nothing folds, but genuine
  content above the new Nyquist is **lost**.
- *naive subsampling* (`x[::k]`) — content above the new Nyquist **folds back**
  into the band, manufacturing fake low-frequency structure. Since ASMC chatter
  and numerical hash live high, naive subsampling would *synthesize* contamination
  the chatter screen then (correctly) flags — so the screen is the right
  instrument for detecting it.

**First-principles bound (computed, data-independent):**
- Roller ripple energy above the Nyquist: ~0% for `|ω| ≤ 10`; ~2.4% above 250 Hz
  at `|ω|=15`, ~0% above 500 Hz at any realistic speed.
- LuGre roll-off (corner `f_c = 653·Vp`) — *upper bounds* (flat-input
  assumption): above 500 Hz reaches ~11% at Vp=0.15, ~21% at Vp=0.3; above 250 Hz
  ~23–40%.
- → **`2000→1000 Hz` is low-risk; `1000→500 Hz` is questionable only for
  high-slip / high-ω regimes** (`coupled_vomega`, `spiral_orbit`, fast `octagon`
  legs) — profile-dependent, not a blanket verdict. Needs empirical confirmation.

**Significance criteria (decided with user):**
1. *Force reconstruction error* — band-limited (`resample_poly`) reconstruction
   of the resampled force back to 2000 Hz vs native, per family, normalised;
   compared against the brief's interpolation floor (~2.7e-4 Mz). Band-limited
   (not linear) reconstruction is deliberate — it isolates genuine loss/aliasing
   from interpolator crudeness.
2. *Chatter-verdict flips* — does the chatter screen's verdict change across
   rate/mode? (naive should manufacture `hash`; anti-aliased should not.)
   Energy-above-Nyquist is reported as cheap physics-grounded context.

**Implementation:** `sampling_sensitivity.py` (reuses `chatter_diagnostics.
diagnose_columns` for the verdict-at-rate, so the two screens stay consistent) +
`test_sampling_sensitivity.py` (synthetic acceptance gate). Validated on
synthetic: clean trajectory faithful at every rate/mode (recon ~0.01, no flips);
HF content preserved at 1000 Hz (modes agree, recon ~0.01) but lost/aliased at
500 Hz (anti-aliased 0.51, naive 0.73 — modes diverge as predicted).

**Data:** runs on the production sweep output once it exists (no probe-set
generation). `python sampling_sensitivity.py --data-dir <sweep> --out samp.csv`.
The real run produces per-profile recon-error distributions and flip rates → the
rate decision (likely: 1000 Hz safe globally; 500 Hz only for low-slip profiles).
