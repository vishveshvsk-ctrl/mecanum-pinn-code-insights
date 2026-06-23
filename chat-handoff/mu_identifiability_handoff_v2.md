# Handoff — build `mu_identifiability.py` (model-free μ-multiplicativity check)

## 1. Title + lineage
Build **`mu_identifiability.py`**, a model-free diagnostic testing whether per-wheel
friction forces scale **multiplicatively with μ** across {0.3, 0.5, 0.8}, mirroring
`chi_identifiability.py`. Continues the Mecanum-PINN trajectory-diagnostics chat
(KUKA youBot Mecanum PINN digital twin, IMECE 2026; Julia 39-D stiff ODE sim →
Arrow → PyTorch PINN). This supersedes `mu_identifiability_handoff.md` (its "blocked
on μ=0.3/0.8 generation / separate dirs" premise is now resolved).

## 2. Context the task depends on
- **Data (all in ONE dir):** `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov`
  = **5,670 Arrow files = 3 μ-batches (0.3/0.5/0.8) × 1,890**. μ-sets are
  **utilization-scaled, NOT command-matched** (combo cXXX differs across μ).
- **Whitelist:** `diagnostics_combined.csv` (5,670 rows, col `combined_reco`;
  whitelisted = NOT startswith `reject`; 5,345 whitelisted).
- **Arrow columns (verified in `datastore.jl:106-125`):** `Fx_i, Fy_i` = raw
  friction force, **wheel frame**. `Fpar_i, Fperp_i` = force rotated into the
  **roller frame** (`Fpar=fx·cosδ+fy·sinδ`, drive/free-roll axes; δ = O-config
  `[-π/4, π/4, π/4, -π/4]`, per-wheel constant). `Vpx_i, Vpy_i` = contact slip,
  **wheel frame** (same frame as Fx/Fy, NOT rotated by δ). Also `wz_i`, `util_i`
  (`=|F|/(μN)`), `psi_dot`. **No JLD2** — read all from Arrow.
- **Mirror exactly:** `chi_identifiability.py` — streaming sufficient-stats
  (XᵀX/Xᵀy/Σy²/n, never pools raw samples), per-μ via `--mu`, `--whitelist`,
  conditioning guard (`cond(A)>1e10 → NaN`), `MIN_N=400`, `PSI_EDGES=(0,0.5,2,∞)`
  3 strata. Its design choices (linear-in-parameter, abs-swing headline, units
  trap) are the precedents to carry over.
- **μ physics:** F ≈ μN·g(slip) + σ₁ż (LuGre). μ scales the **Coulomb/dissipative**
  part (gated by **slip**, not spin like χ); the σ₁ż LuGre term is the
  μ-**independent** affine `B`.

## 3. Purpose / success criterion
Model-free verdict: **do Fpar/Fperp scale as `A·μ + B` across {0.3,0.5,0.8}?**
Quantify the μ-scaled slope `A`, the μ-independent intercept `B`, the multiplicative
fraction, and the curvature. **Success = `mu_identifiability.csv` (per-μ-batch is
N/A — it fits across μ) + findings written into `TRAJ_DIAGNOSTICRESULTS.md` (new
§8) with a multiplicative / affine / nonlinear verdict per channel & slip regime.**

## 4. Key design decisions (already made — defend, don't reopen)
1. **Bin by SLIP, not spin.** μ's "dose" is slip (`F≈μN·g(slip)`), unlike χ (spin).
   So the bin axis is slip magnitude `|Vp|`, NOT `|ω_z|`. Keep the `|psi_dot|`
   3-strata stratification (regime control + deployment-observable).
2. **Per-channel Fpar/Fperp in the ROLLER frame, binned by CO-DIRECTIONAL signed
   slip.** Compute `Vp∥ = Vpx·cosδ+Vpy·sinδ`, `Vp⊥ = −Vpx·sinδ+Vpy·cosδ`; bin Fpar
   by signed `Vp∥`, Fperp by signed `Vp⊥` (~16 signed bins). Rationale: Mecanum
   anisotropy — Fpar (drive, grips) carries the μ-multiplicative `A`; Fperp
   (free-roll) is mostly μ-independent `B`. Signed co-directional bins avoid the
   sign-cancellation that bins-by-`|Vp|` would cause. (Rejected: force magnitude
   |F| — hides the A/B anisotropy; rejected: slip-aligned F∥/F⊥ — cleaner physics
   but loses the drive/free-roll split and is ill-defined as |Vp|→0.)
3. **Fit `F̄(μ) = A·μ + B` across the 3 μ within each cell** (covariate-controlled
   regression: per-wheel dummies + μ as a regressor; slip held ~fixed by the bin).
   Headline = **multiplicative fraction `A·μ̄/(A·μ̄+B)`** (n-weighted median over
   gross-slip bins, per channel) + **`mu_swing_abs`=|F̄(0.8)−F̄(0.3)|** in N vs
   `channel_rms` (the absolute-vs-noise "units trap" — NOT R²/F, which are useless
   at ~10⁶ samples/bin). μ=0.5 is the **held-out check**: fit {0.3,0.8}, predict
   0.5, report `curvature_resid` (low ⇒ truly linear-multiplicative).
4. **Linear in μ (NO μ² term), nonlinear in slip.** Mirrors the χ decision just
   validated: a quadratic in the swept parameter is unmotivated for forces and
   ill-conditioned on a 3-point grid; keep the slip controls nonlinear (|Vp|²)
   because dropping them shifts the estimate (real confound).
5. **NO pointwise differencing** across μ (trajectories decohere — same artifact χ
   hit: pointwise RMS(F(μ₁)−F(μ₂)) was larger than F itself). Binning is mandatory.
6. **NO `--mu` filter** — the whole point is to pool all 3 μ (they are the x-axis
   of the A·μ+B fit). But DO use `--whitelist diagnostics_combined.csv`.

## 5. Open decisions / blocking relationships
- **Bin edges for signed `Vp∥`/`Vp⊥`** — confirm the slip range from a real Arrow
  (mirror how χ set `WZ_EDGES`); pick symmetric signed edges.
- **Verify the Fpar↔Vp∥ / Fperp↔Vp⊥ coupling sign** on one real file before locking
  the binning (datastore rotates force by δ but NOT slip — rotate slip yourself).
- **Noise floor** for the identifiability verdict is still unset (same open item as
  χ) — report `mu_swing_abs` in N and defer the floor to the user.
- **Hand-back:** the verdict (multiplicative? affine-`B` fraction? where μ̂ readable?)
  gates the PINN's Approach-1 μ-ID and whether the **affine force-head hook** must be
  activated.

## 6. Deliverables
1. `mu_identifiability.py` — streaming, resume-aware, CLI mirroring χ
   (`--data-dir`, `--out`, `--whitelist`); pools all μ from the one dir.
2. `mu_identifiability.csv` — per `(stratum, channel∈{Fpar,Fperp}, signed-slip bin)`:
   `n, F_mu03, F_mu05, F_mu08, affine_A, affine_B, mult_fraction, mu_swing_abs,
   channel_rms, rel_swing, curvature_resid`.
3. New **§8 in `TRAJ_DIAGNOSTICRESULTS.md`** (observation→inference→explanation;
   match §2's style) with the multiplicative/affine/nonlinear verdict.
4. (Optional) `test_mu_identifiability.py`.

## 7. Conventions to respect
- Run from `code_insights/`; **claude-venv** python `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`.
- **≤8 threads/workers** (hard cap, machine OOMs above). Per-file streaming
  (read→accumulate→advance). Static matplotlib only — **no interactive widgets**.
- **Confirm architecture/decisions BEFORE coding**; surgical edits over rewrites;
  compute every numeric value in code before stating it. Never hand-edit outputs
  (CSV existence = resume marker). Scratch in `_tmp/`, clean up after.
- Long runs (>~20 min): start `keep_awake.py` in background first; its bg task may
  report "exit 127" but the python child keeps running — kill it explicitly when done.
