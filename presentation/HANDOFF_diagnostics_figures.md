# Handoff — Mecanum-PINN deck: add 3 diagnostics example figures

**Continues:** the IMECE/VNIT Mecanum-PINN slide-deck session. **Frame:** a 29-slide
academic deck (`presentation/deck.html` → `Mecanum_PINN_Deck.pdf`) on a PINN digital
twin of a 4-Mecanum-wheel platform; the diagnostics-results section (slides 23–27)
summarizes how the 1,776-trajectory sweep is screened into a 1,568-file PINN whitelist.

## Context the new chat needs
- **Editable master:** `code_insights/presentation/deck.html` (plain HTML; MathJax via
  local `assets/tex-svg.js`; Academic-Navy theme). Figures live in
  `code_insights/images_and_plots/`, referenced as `../images_and_plots/<name>`.
  Edits to `presentation/**` and `images_and_plots/**` are allow-listed (no prompt).
- **Render:** `& msedge --headless=new --disable-gpu --no-first-run --user-data-dir="$env:TEMP\edge_pdf_profile" --no-pdf-header-footer --run-all-compositor-stages-before-draw --virtual-time-budget=30000 --print-to-pdf="<dir>\Mecanum_PINN_Deck.pdf" "file:///<dir>/deck.html"`
- **QA-rasterize a slide** (no poppler): `myenv` python `fitz.open(pdf)[p-1].get_pixmap(dpi=120).save(...)`. Page p = slide p.
- **Tooling (see memory `project-python-env-tooling`):** matplotlib = **claude venv**
  `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`; `fitz` + the **`julia-1.12`**
  IJulia kernel = **myenv** `C:\Users\vishv\miniforge3\envs\myenv\python.exe`.
- **Figure source A — slipspin v4 notebook** via driver `_tmp/run_nb_figs.py` (myenv;
  appends a per-profile override to the `parameters` cell, runs headless, extracts by
  cell-match). It currently sets `COMBO_IDX = nothing` (random) — **change to a target int**
  to reproduce a specific trajectory. Already in `_tmp/nb_figs/` (random combos, PICK_SEED
  123): `tracking_*`, `controlsplit_*` (incl. spin_creep, octagon), `dwell_*`, `rollerbalance_*`.
- **Figure source B —** `Mecanum_PINN_TrajectoryDiagnostics_1.ipynb` (STFT/θ-fold) and the
  per-trajectory diagnostic notebook. **Pick trajectories from `diagnostics_combined.csv`**
  (1,776 rows; cols `profile, combo_idx, combined_reco, chatter_verdict, util_sat_frac_*,
  track_flag, msat_burstiness`). Data Arrow files: `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov`.

## Purpose / deliverables (3 figures + text, then re-render & verify)
1. **STFT, roller ripple (slide 24, chatter).** STFT of a force channel for a *clean,
   accepted* trajectory (`combined_reco`=keep; e.g. a long_circle/octagon with clear ripple);
   show the roller ridge `f_roll≈1.91·|ω|` Hz. No hash counterexample. → `diag_stft_clean.svg`.
2. **ASMC chatter (slide 24).** 4-wheel control-torque split (M_sw vs M_eq vs M_sat) for a
   **near-cap spin_creep whitelist** run — pick max `util_sat_frac_1p0` among
   `profile=spin_creep & combined_reco` not reject; get its `combo_idx`; regenerate via the
   driver (set that COMBO_IDX). `controlsplit_spin_creep.png` (random) already exists as a
   fallback. → e.g. `controlsplit_spincreep_nearcap.svg`.
3. **M7 burst, octagon (new slide after 24).** An octagon `reject_burst` run (high
   `util_sat_frac` = long disturbed fraction): **2×2 tracking (Vx/Vy/ψ̇ + path) beside the
   2×2 4-wheel control torque**. Find combo via CSV (profile=octagon, combined_reco startswith
   `reject_burst`, high util), run driver with that COMBO_IDX → `tracking_octagon_<idx>` +
   `controlsplit_octagon_<idx>`. Add text: **M7 quantification** — 15 Hz Butterworth high-pass
   → 0.25 s sliding-window RMS → `burstiness = max_window/(p10_window+FLOOR)`, FLOOR=0.10 N·m,
   max over 4 wheels, `tau_burst=10`; **HF-knob rationale** — 15 Hz ≈ 5× wheel corner
   `p1/(2π Jw)=0.11/(2π·5.87e-3)≈3.0 Hz` (above it wheel inertia low-passes torque → no
   tracking work), below the ~29 Hz burst content and ~77 Hz roller corner; ratio not absolute
   (multisine flat HF→≈1), p10 not median (robust to >50% disturbed). (All in
   `TRAJ_DIAGNOSTICRESULTS.md` §1.)
4. **Verify:** rasterize slides **25 (χ) and 26 (sampling)** — confirm the latest text panels
   didn't overflow; trim if clipped. Re-render the PDF after embedding.

## Key decisions already made (don't reopen)
1. **Deck order:** flowchart is the **second-last** slide (28); closing is 29. Diagnostics
   method (22) then results (23–27) then flowchart (28).
2. **Whitelist headline = 1,568/1,776** (keep 1172 + flagged 396; reject_burst 119 +
   reject_missed 89; 0 hash). A burst rejects **only if** tracking is also lost (rule 2).
3. **Embed pattern:** `<div class="figwrap"><img class="figimg" src="../images_and_plots/X"></div>`;
   two-figure slides use a flex-column col with stacked figure blocks + tiny captions (see slide 6).
4. **Mz dropped, train at 500 Hz, screen at 2000 Hz** (from the diagnostics results).
5. Figure-gen patches the notebook **in-memory only** — the authoritative slipspin notebook is never modified.

## Conventions
Surgical edits (not rewrites); compute exact numbers from source before writing them; verify
each rendered slide by rasterizing; scratch/intermediate files → `_tmp/` (auto-allowed);
call interpreters by full path (no `conda activate` in tool shells); credit external figures.
