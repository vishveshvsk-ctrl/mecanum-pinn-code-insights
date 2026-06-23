# Handoff — PINN trajectory-diagnostics: per-profile threshold tuning

**Lineage:** continues the Mecanum-PINN diagnostics chat. A four-screen
trajectory-quality suite is **built, validated, and run over the full sweep**;
all thresholds are first-principles **placeholders**. This session's job is the
**per-profile threshold-tuning pass** → finalized whitelist. Project: KUKA youBot
Mecanum PINN digital twin (IMECE 2026); Julia 39-D stiff ODE sim → Arrow → PyTorch PINN.

## Context the task depends on
- **Data:** `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov` — **1,776 Arrow files**
  (1,485 production at χ=0.005 + 291 χ-grid {0.000,0.002,0.008} for 97 matched quads:
  spin_creep 44 + coupled_vomega 22 + octagon 31). All μ=0.5, lugre_adamov, case 1.
  Profiles: octagon 309, spin_creep 432, long_circle 254, coupled_vomega 216,
  ellipse 96 (PosRef), spiral_orbit 68, multisine_75/50 55+55 (counts pre-χ-grid).
- **Screens (all in `code_insights/`, streaming/resume/crash-safe-flush, CLI):**
  `chatter_diagnostics.py` (M0–M7), `sampling_sensitivity.py`, `tracking_gate.py`,
  `chi_identifiability.py`. Fused by `blend_reports.py` → **`diagnostics_combined.csv`**
  (the single source of truth; the 4 per-screen CSVs are intermediates).
- **Current combined tally:** keep 1182, keep_flagged 373, reject_burst 52,
  reject_missed 169 → **whitelist 1,555 / 1,776**. Chatter: 0 hash, 272 chatter
  (spin_creep 144, multisine **110 = 100%**, ellipse 14, spiral_orbit 4), 1504 clean.
  Tracking (duration metric): tracked 1454, marginal 101, missed 221 (coupled_vomega 86,
  octagon **135**; all other profiles fully tracked).
- **Placeholder thresholds to tune:**
  - chatter `CLASSIFIER_THRESHOLDS`: tau_hash=0.15, tau_coh_low=0.30, tau_coh_high=0.50,
    tau_slipmod=0.30, tau_ctrl=0.20, util_sat_frac=0.50, **tau_burst=100.0**.
  - tracking `THRESHOLDS`: **track_inst=0.30, track_viol_frac=0.10**, util_viol_frac=0.05,
    msat_rail_frac=0.50; `T_EXCLUDE=0.5` s startup mask.
  - sampling notebook `FLOOR=0.05`.
- **Env tooling (system matplotlib is BROKEN; recorded in memory `project_python_env_tooling`):**
  use venv `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe` (matplotlib 3.11,
  nbconvert), `tectonic.exe` at `...\mecanum\bin\`, Jupyter kernel `claude-mecanum`.
  Render notebooks: `<venv> -m nbconvert --to html --execute --ExecutePreprocessor.kernel_name=claude-mecanum`.

## Purpose
Replace every placeholder threshold with a **data-driven, per-profile** value:
plot each metric's distribution split by profile, pick cutoffs at the natural
shoulders, finalize the verdicts and the whitelist. **Success = tuned thresholds
committed in the modules + a defensible `diagnostics_combined.csv` and
`pinn_training_whitelist.txt`.**

## Key design decisions (made — defend, don't reopen)
1. **Two orthogonal gates: chatter (force-signal contamination) vs tracking
   (was the commanded trajectory achieved).** They are disjoint — every `missed`
   is chatter-`clean`; chatter is structurally blind to saturation misses.
2. **M7 `msat_burstiness` = max_window/(median_window+floor)** on `Msat`, HP at
   **15 Hz ≈ 5× wheel corner `p1/2πJw≈3Hz`**, 0.25 s window. **Ratio, not amplitude**:
   broadband multisine control is high-but-FLAT (~1.3); a real localized burst spikes
   (c125 = 3711). At tau_burst=100, burst ⊆ missed (52); **burst_flag → reject**.
3. **Tracking uses a DURATION metric**, not RMS: `track_viol_frac` = fraction of the
   (post-startup) run the instantaneous error `e(t)=‖v_err‖/scale` exceeds `track_inst`.
   RMS diluted sustained-moderate off-command; `track_err` (RMS) is kept as a reported
   descriptor only. **0.5 s startup mask** excludes the controller's convergence transient.
4. **Generalized velocity `v=(Vx,Vy,ψ̇·L_CHAR)`, `L_CHAR=0.279 m`** (yaw rate→edge speed,
   unit-consistent, normalized by ONE motion scale). PosRef (ellipse) uses path error.
5. **DTW rejected** for tracking: this is real-time tracking (a lag IS an error), DTW
   would warp-away and mask the dynamic-infeasibility (friction-circle) failure, refs are
   slow (≤1 Hz) so no lag artifact exists, and it's O(n²) at n≈80k.
6. **Drop `Mz` from PINN training; recover χ from `Fpar/Fperp`** — χ gives a ~2.3 N
   spin-gated force swing (rises with |ω_z|); Mz is χ²-tiny and unmeasurable.
   (`chi_identifiability.py`, 129 cells over the 97 quads.)
7. **Train at 500 Hz, run the chatter screen at 2000 Hz.** Force recon error ~1–3%
   and flat 1000→500 Hz; only Mz is rate-sensitive (and dropped). At 500 Hz the screen
   flips 13% of spin_creep and goes blind >250 Hz.
8. **`F_hash` LuGre-aware ceiling** `= pctile95(max(3·f_roll, 1.5·f_lugre, 40))`; under
   gross slip it inflates above the Nyquist → M2/M3 = 0 (blind) → this is WHY M7 exists.
9. **Combined reject precedence: hash → burst → missed.** Whitelist excludes all `reject_*`.

## Open decisions / blocking relationships
- **All thresholds are undecided** (this task sets them). Specific items to eyeball:
  (a) multisine 100% chatter — needs a profile-specific cutoff (designed broadband control
  ≠ pathological chatter); (b) octagon missed 77→135 jump under the duration metric —
  is `track_inst=0.30 / track_viol_frac=0.10` right for start-cruise-**stop** legs?
  (c) tau_burst 100 (burst⊆missed, sub-label) vs ~10–20 (catches ~18 independent mild bursts).
- **`data.py` `_FNAME_RE` is still the OLD beta/amp scheme** — the whitelist is NOT
  consumable by the trainer until it is migrated to the profile scheme (regex in PLAN §6).
- **Hand-back:** the finalized `pinn_training_whitelist.txt` crosses back to the PINN
  training pipeline (WSL2, `~/mecanum_pinn_main/`).

## Deliverables
1. Tuned thresholds committed in `chatter_diagnostics.py`, `tracking_gate.py`,
   (sampling notebook `FLOOR`).
2. Re-run the affected screens (purge the CSV first — metric/threshold change), re-blend
   → updated `diagnostics_combined.csv`.
3. `pinn_training_whitelist.txt` (bare filenames kept), keyed by the profile filename scheme.
4. Per-profile metric histograms (optional figure, via the venv).

## Conventions to respect
- Architecture decisions confirmed BEFORE code; surgical edits over rewrites; compute
  every numeric value in code before stating it; verify on real data + the synthetic
  acceptance tests (`test_chatter_diagnostics.py` etc.).
- **Purge the per-screen CSV before re-running after any metric/threshold change**
  (resume preserves stale rows).
- Use the **venv python** for anything that imports matplotlib / executes notebooks.
- **NotebookEdit-by-position pitfall:** inserts shift the position-based cell IDs, so
  subsequent edits hit the wrong cells — after an insert, edit cells by direct JSON, or
  re-Read to get current positions. (Verify a notebook by rendering it and counting plots.)
- Scratch in `_tmp/` (auto-allowed); clean up. Reference files as clickable `path:line`.
