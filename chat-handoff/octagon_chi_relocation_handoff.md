# Handoff — relocate octagon χ-grid one tier below the infeasible cap

**Lineage:** continues the Mecanum-PINN trajectory-diagnostics chat. For an
EXISTING data-generation pipeline session — pipeline mechanics already known, so
this is *only* the octagon χ-case change. Project: KUKA youBot Mecanum PINN digital
twin (IMECE 2026); Julia 39-D stiff ODE sim → Arrow → PyTorch PINN.

## Context the task depends on
- **octagon** = body-velocity legs in 8 directions (incl. the ±y **lateral** leg),
  heading held fixed. `vcru` is the per-leg cruise speed; on the lateral leg the
  platform must reach **Vy = vcru sideways**.
- **Finding driving this task:** the **top `vcru` tier is laterally infeasible at
  every μ**. Lateral mobility caps at ≈0.7×vcru_max (e.g. Vy saturates ~0.42 m/s at
  μ=0.3), so the top tier's lateral legs saturate → **0% success**. Lower 3 tiers
  100%. χ-independent in *outcome* (χ=0 only adds gross-slip util 0.32 vs ~0.05).
- **`vcru` tiers per μ** (4 tiers; the three μ TOMLs are scaled copies with the
  SAME combo layout/indices):
  - μ=0.3: {0.24, 0.36, **0.48**, ~~0.60~~}  — cap=0.60, **1-below = 0.48**
  - μ=0.5: {0.40, 0.60, **0.80**, ~~1.00~~}  — cap=1.00, **1-below = 0.80**
  - μ=0.8: {0.64, 0.96, **1.28**, ~~1.60~~}  — cap=1.60, **1-below = 1.28**
- **χ-grid = {0.000, 0.002, 0.005, 0.008}.** χ=0.005 is full production (all combos,
  all tiers). The χ-quad subset {0.000, 0.002, 0.008} currently sits **ONLY at the
  cap tier** → every octagon χ-identifiability sample is on a FAILED/saturated
  trajectory (0/31 per μ).
- **Files (authoritative):**
  - octagon TOMLs: `trajectory_files_run_0p3_main/profiles/octagon_mu_0p3.toml`,
    `…_run_0p8_main/…/octagon_mu_0p8.toml`,
    `…_run_0p5_main/…/octagon_mu_{0p3,0p5,0p8}.toml` (identical layout, scaled vcru).
  - χ-quad combo selection (the active artifact): `trajectory_files_chinc/`
    `selected_near_cap_combos.json` → key `octagon` = **31 indices, all vcru=cap**.
  - Output dir: `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov`
    (filename `octagon_c<combo:%03d>_mu_<mu:%g>_case1_lugre_adamov_chi_<chi:%.3f>.arrow`).

## Purpose
Generate the octagon **χ cases (all four χ, all three μ)** at the **highest
100%-success vcru tier — one below the cap (0.48 / 0.80 / 1.28)** — instead of the
infeasible cap tier, so the χ-identifiability data sits on trajectories the platform
actually tracks. **Success =** octagon χ-quad files exist at the target tier for all
3 μ × all χ AND pass the tracking gate (`track_viol_frac < 0.10`, `util_viol_frac < 0.10`).

## Key design decisions (already made)
1. **Move exactly ONE tier below the cap** (not lower). χ is spin-gated — its force
   signal is strongest near the friction cap — so stay as near the cap as is
   *tracking-feasible*; the cap tier itself is laterally infeasible (0% success).
2. **One index list for all three μ.** The μ TOMLs are scaled copies, so combo
   indices are shared. **Target-tier indices (vcru 0.48/0.80/1.28), 69 available:**
   `3,7,11,20,21,22,28,32,40,41,42,50,51,56,60,68,69,70,75,76,81,82,85,106,110,114,123,124,125,131,135,143,144,145,153,154,159,163,171,172,173,178,179,184,185,188,209,213,217,226,227,228,234,238,246,247,248,256,257,262,266,274,275,276,281,282,287,288,291`.
   **Cap-tier indices to REPLACE (current, 31):**
   `4,8,23,24,25,29,33,43,44,45,52,53,57,61,107,111,126,127,128,132,136,146,147,148,155,210,229,230,231,235,239`.
3. **χ-quad set stays {0.000, 0.002, 0.008}** (χ=0.005 already exists at the target
   tier from production — generate only the 3 extra χ there, unless a clean matched
   quad is wanted, then regenerate 0.005 too).
4. **Do NOT touch** the lower two tiers or the production χ=0.005 grid — feasible and correct.

## Open decisions / blocking relationships
- **Combo count at target tier:** match the current **31** (pick 31 of the 69, ideally
  matching the original lat_vamp/theta0 structure for comparability) OR use all 69.
- The infeasible **cap-tier octagon files** (135/μ + χ variants) stay in the data dir —
  purge vs leave is a separate cleanup call (diagnostics reject them either way).
- **Out of scope:** the full octagon `vcru_max` re-cap (making the whole top tier
  feasible) is blocked on the straightline β=90 lateral-cap measurement.
- **Hand-back:** once regenerated, the parent diagnostics chat re-runs chatter /
  tracking / sampling + `chi_identifiability.py --mu <μ>` on the new octagon χ files,
  confirms 100% success + a valid matched χ-quad, and appends `diagnostics_combined.csv`.

## Deliverables
1. Updated `trajectory_files_chinc/selected_near_cap_combos.json` — `octagon` list =
   target-tier indices (vcru one below cap).
2. Generated octagon χ `.arrow` files at the target tier, **all 3 μ × χ {0.000, 0.002,
   0.008}** (+0.005 if regenerating the quad), in the output dir above.

## Conventions to respect
- **≤8 threads / parallel workers** — hard cap (machine OOMs above) unless the user
  explicitly raises it.
- Arm `keep_awake.py` (background) before any run >~20 min; resume is `.arrow`-existence
  based and safe. Output files never hand-edited (an existing `.arrow` = done).
- **Deterministic:** keep `--sweep-seed 1234` so combos reproduce
  (`build_job` hashes `(sweep_seed, profile, combo_idx)`).
- Surgical edits; confirm the combo selection before launching generation.
