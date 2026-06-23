# Handoff — χ-grid trajectory regeneration (for the χ-identifiability analysis)

**Lineage:** continues the Mecanum-PINN chatter/sampling-diagnostics chat. This
spun-off session's job is to **run the simulator to regenerate a small set of
matched trajectories at additional χ values**, holding μ constant, so a later
analysis can test whether χ is recoverable from the observable forces. Project:
KUKA youBot Mecanum PINN digital twin (IMECE 2026); Julia simulator → Arrow/JLD2
→ PyTorch PINN.

## Context the task depends on
- **Driver:** `Data_Generation_Julia.jl` (run from project root). Config:
  `trajectory_files/base.toml` + per-profile TOMLs in `trajectory_files/profiles/`.
  Authoritative modules: `profiles.jl`, `datastore.jl`.
- **Current sweep (complete):** 1,485 Arrow+JLD2 files, **all χ=0.005, μ=0.5,
  friction_model=lugre_adamov, friction_case=1**, in
  `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov`. Per-profile counts:
  spin_creep 432, octagon 309, long_circle 254, coupled_vomega 216, ellipse 96,
  spiral_orbit 68, multisine_75percent_cap 55, multisine_50percent_cap 55.
- **Filename scheme** (`datastore.jl`, single source of truth):
  `<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow`.
  χ is in the name at **3-dp** precision → a new χ never clashes and never
  false-resume-skips. Use well-separated χ (3rd-decimal distinct).
- **Sweep seed:** 1234 (driver default; `build_job = hash((sweep_seed, profile,
  combo))`). Resume = `.arrow` existence via `DataStore.expected_output`.

## Purpose
Regenerate **matched** trajectories at **χ ∈ {0.000, 0.002, 0.008}** (0.005
already exists) for the spin-heavy profiles + a low-spin control, with everything
else identical, so the downstream analysis can difference forces across χ.
**Intent:** χ reaches the observable forces only through the spin→translation
coupling `c_t = (8/3π)·|ω_z|·χ` — gated by spin. The check asks whether that
χ-signal in `Fpar/Fperp` clears the noise (Mz ∝ χ² is too tiny/low-SNR to matter
and is excluded). **Success = for each chosen combo, files exist at all four χ
values, identical in (profile, combo, μ, fm) — a valid χ-only matched group.**

## Key design decisions (made — defend, don't reopen)
1. **Only `spin_creep` + `coupled_vomega`, plus a small `octagon` negative
   control.** Spin coupling is strongest in the high-Ω profiles, so χ is most
   identifiable there; low-spin profiles are bounded above → skipping them loses
   no identifiability info. octagon (low-spin) should show χ *not* identifiable,
   confirming the signal tracks spin (mechanism check, not artifact).
2. **Hold μ = 0.5 constant.** Force scales as O(μN); varying μ would confound the
   χ-attributable force change. The matched-group difference must be χ-only.
3. **χ grid {0.000, 0.002, 0.005, 0.008}.** Four points enable a `∂F/∂χ` slope;
   χ=0 is the no-coupling / `Mz≡0` anchor. Only the 3 missing values are run.
4. **Same `--sweep-seed 1234` and same combo indices as the existing files.**
   The seed+combo_idx is what makes the regenerated run reproduce the *identical
   commanded reference* — the precondition for "matched, χ-only." A different
   seed or renumbered combo = a different trajectory = not matched.
5. **Subset combos by TRUNCATING each profile's `[profile.combos]` arrays to the
   first N rows (≈10–15), never cherry-picking rows.** Running all combos of
   these profiles at 3 χ ≈ 2,600 jobs (> the whole original sweep). Truncating
   from row 1 preserves combo_idx 1..N → still matched with existing 0.005 files;
   cherry-picking non-contiguous rows renumbers them → breaks the match.
   **Verify combo_idx assignment in `profiles.jl` (`enumerate_jobs`) before
   trimming** to confirm row order maps to combo_idx.
6. **Mz is not a χ channel / training target.** Absent from the platform & wheel
   RHS (its only path is the excluded roller spin γ̇); χ²-scaled, low-SNR. The
   χ signal lives in `Fpar/Fperp` via the coupling. (Decided in parent chat.)

## Open decisions / blocking relationships
- **Combo-subset mechanism** (truncate TOML copies vs run-all) — resolve by
  reading `profiles.jl` first (decision 5). If compute is cheap, running all
  combos of the 3 profiles avoids any renumbering risk entirely.
- **Hand-back:** the regenerated Arrow+JLD2 files are the deliverable; they are
  consumed by the χ-identifiability analysis (bin forces by operating point
  `(|ω_z|, |Vp|)`, between-χ shift vs within-χ noise, slope `F` vs χ; compare
  `Fpar/Fperp` against `Mz`) run back in the diagnostics thread.

## Deliverables
Arrow (+JLD2) files at **χ ∈ {0.000, 0.002, 0.008}** for the chosen combos of
`spin_creep`, `coupled_vomega`, and a few `octagon` (control) — μ=0.5,
lugre_adamov, case 1 — written to
`..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov`, seed 1234, matched to the
existing `…_chi_0.005.arrow` files.

## Run recipe
1. Edit `trajectory_files/base.toml` `[physics]`: `chi = [0.0, 0.002, 0.005,
   0.008]`, keep `mu_friction = 0.5` (scalar), `friction_model = "lugre_adamov"`,
   `friction_case = 1`. (resume skips the existing 0.005 files.)
2. (subset) Make trimmed copies of `spin_creep.toml`, `coupled_vomega.toml`,
   `octagon.toml` with the first N combo rows — only after confirming row→idx in
   `profiles.jl`.
3. Dry-run, then run:
   ```
   julia --project=. -t auto Data_Generation_Julia.jl --profiles spin_creep.toml,coupled_vomega.toml,octagon.toml --sweep-seed 1234 --dry-run
   julia --project=. -t auto Data_Generation_Julia.jl --profiles spin_creep.toml,coupled_vomega.toml,octagon.toml --sweep-seed 1234
   ```

## Conventions to respect
- `base.toml` is read-or-die authoritative; edit it, never hand-edit outputs.
- Module edits (`profiles.jl`/`datastore.jl`) ⇒ kernel/process restart; re-read
  exact filenames before acting.
- Keep the **same sweep seed** across passes; don't renumber combos.
- Long runs die to Modern Standby (~30 min) even on AC — use the keep-awake
  daemon + powercfg lockdown before a multi-hour run.
