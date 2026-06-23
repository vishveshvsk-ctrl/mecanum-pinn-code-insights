# Handoff — `mu_identifiability` diagnostic (model-free μ-multiplicativity check)

## 1. Title + lineage
Build **`mu_identifiability.py`**, a model-free diagnostic that tests whether per-wheel friction
forces scale **multiplicatively with μ** across the grid **{0.3, 0.5, 0.8}**, mirroring the existing
`chi_identifiability.py`. Continues the Forward-Inverse PINN / trajectory-diagnostics work; implement
in the diagnostics area (beside `chatter_diagnostics.py`, `chi_identifiability.py`, `tracking_gate.py`,
`blend_reports.py`). It is the **gate** that must pass before the PINN's μ-identification is trusted.

## 2. Context the task depends on
- **Data**: `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov` = μ=0.5 (1776 files). The μ=0.3 and
  μ=0.8 sets are being generated per `mu_trajectory_generation_handoff.md` with **utilization-scaled
  amplitudes** (so they are **NOT command-matched**, unlike the χ-quads), likely in separate dirs
  `..._mu0.3\` / `..._mu0.8\`. **This diagnostic is BLOCKED on those landing** — write + unit-test
  against μ=0.5 now, run fully when the data arrives.
- **Forces**: Fpar/Fperp only (Mz dropped). **Read everything from Arrow columns — JLD2 is not
  stored.** Confirm exact force/kinematic column names from a real Arrow file.
- **μ physics (key)**: μ is observed through **friction-circle utilization / the slip–force relation**
  — NOT spin like χ. `F = μN·Φ` to leading order; the μ-independent LuGre damping breaks it at low
  slip → that residual is the affine `B` to extract.
- **Mirror**: `chi_identifiability.py` (structure, streaming/resume CLI). Its CSV columns are
  `stratum, channel, wz_center, n, F, partial_R2_chi, slope_dF_dchi, chi_swing_abs, channel_rms,
  rel_swing` — produce the μ analogue.

## 3. Purpose / success criterion
A model-free verdict: **does Fpar/Fperp scale multiplicatively with μ across {0.3,0.5,0.8}?** Quantify
the affine `B` (μ-independent) fraction, test for curvature, and confirm μ̂ is separable by force
magnitude. Success = `mu_identifiability.csv` + a findings note with a clear
**multiplicative / affine / nonlinear** verdict per operating regime.

## 4. Key design decisions (already made — defend, don't reopen)
1. **Bin by operating point, regress force on μ within bins** — do NOT difference forces pointwise
   across μ (trajectories decohere; same artifact χ hit — pointwise `RMS(F(μ₁)-F(μ₂))` was larger
   than F itself). Mandatory here because the μ-sets aren't command-matched.
2. **Bin by SLIP / utilization, not |ω_z|.** μ's signal lives in friction-circle utilization and the
   slip–force relation, not spin. Use a measurable slip magnitude and/or realized `|F|/(μN)` as the
   covariate (+ per-axis `a_long/a_lat` if available), controlling per-wheel offset.
3. **Per-bin model: fit `F = μ·A + B` across the 3 μ values** (over-determined → separate the
   μ-scaled term `A` from the μ-independent `B`); also test the multiplicative collapse
   `Φ_total = F/(μN)` and the affine-fit residual (= curvature). The 0.5 midpoint doubles as a
   held-out interpolation check (fit {0.3,0.8} → predict 0.5).
4. **Report ABSOLUTE swing vs the noise floor**, not just R²/F-stat — the "units trap" from
   `TRAJ_DIAGNOSTICRESULTS.md` §2 (huge sample counts make R² meaningless; absolute `|ΔF|_μ` in N is
   what determines identifiability under real noise).
5. **Forces = Fpar/Fperp only** (Mz dropped), read from Arrow.

## 5. Open decisions / blocking
- Confirm the exact slip/utilization binning covariate available in (or derivable from) the Arrow
  columns — mirror how `chi_identifiability.py` obtained `|ω_z|`.
- **Blocked on μ=0.3/0.8 generation** (`mu_trajectory_generation_handoff.md`).
- Optional: fold a μ-provenance column into `blend_reports.py` / `diagnostics_combined.csv`.
- **Hand-back**: the verdict (multiplicative? affine-`B` fraction? μ̂ separable?) gates PINN
  Approach-1 μ-ID and decides whether the **affine force-head hook** must be activated.

## 6. Deliverables
1. `mu_identifiability.py` — streaming, resume-aware (CSV = resume marker), CLI mirroring
   `chi_identifiability.py` (`--data-dir`, `--out`); accept multiple data dirs (one per μ).
2. `mu_identifiability.csv` — per (stratum, channel∈{Fpar,Fperp}, slip/util bin): `n`, per-μ means,
   multiplicative-collapse residual, affine `A` & `B`, `mu_swing_abs`, `channel_rms`, `rel_swing`,
   `curvature_resid`.
3. Findings note (mirror `TRAJ_DIAGNOSTICRESULTS.md` §2 style: observation → inference → explanation)
   with the multiplicative/affine/nonlinear verdict.
4. (Optional) `test_mu_identifiability.py` (mirror `test_chatter_diagnostics.py`).

## 7. Conventions to respect
- Run from `code_insights/`; dedicated **claude-venv** python; **per-file streaming**
  (read→accumulate→delete→advance) for memory safety over the Arrow set; seeded/deterministic;
  **static matplotlib** figures + tables (no interactive widgets); CLI-script over notebook; never
  hand-edit outputs (CSV existence = done, for resume).
