# Handoff — add the identifiability + force-law analytics to the IMECE deck

## 1. Title + lineage
Add new result slides to the **existing IMECE 2026 deck** (`presentation/deck.html`,
already built — Academic Navy theme, MathJax, VNIT logo). Continues the Mecanum-PINN
trajectory-diagnostics work (KUKA youBot Mecanum digital twin; Julia 39-D stiff ODE →
Arrow → PyTorch PINN). This task **only adds the new pre-training analytics** — the
deck's existing dataset-quality content (chatter, sampling, tracking, whitelist, μ-grid
sweep = §1, §3–§7 of the source) is already there and is **not** touched.

> **One correction to carry over (not new content):** if the deck's §3 *sampling* slide
> quotes "spin_creep flips **13 %** of verdicts at 500 Hz", that is a μ=0.5-vintage figure.
> The source was refreshed to the 3-μ-pooled value (**8 %**) — fix that one number if the
> deck shows it. All other §1/§3–§7 numbers are current.

## 2. Context the task depends on
- **Single source of truth for all numbers/figures:** `code_insights/TRAJ_DIAGNOSTICRESULTS.md`.
  Pull every value verbatim from there — do not re-derive or approximate.
- **Sections to add (from "χ-identifiability onwards"):** **§2** (χ-identifiability +
  its new "χ force-variance contribution" deep dive), **§8** (μ-identifiability + its
  "identifiable tail" deep dive), **§9** (9.1 slip-regime coverage, 9.2 roller
  observability, 9.3 force anisotropy), **§10** (force-law adequacy for the PINN head).
- **Generated figures on disk (use directly):**
  - `images_and_plots/mu_identifiability_gross.png` → §8 (μ mult-fraction & swing vs co-dir slip)
  - `images_and_plots/roller_slip_fraction.png` → §9.2 (roller share of ω_z, V∥, V⊥)
- **Figure-not-generated sections** (§2 deep-dive, §9.1, §9.3, §10): each carries a
  `> **Figure (deck candidate…)**` note in the md saying what to plot and from which CSV.
  **Per the user, the deck session decides per slide whether to render a figure or just
  insert a table** — both are acceptable; do not block on missing PNGs.
- **Deck mechanics:** `presentation/deck.html` is the source; `presentation/make_figures.py`
  builds assets; deck → PDF (`Mecanum_PINN_Deck.pdf`). MathJax is active, so **LaTeX math
  is fine in the deck** (unlike the chat/md, which use Unicode).

## 3. Purpose
Insert a coherent **"Friction-parameter identifiability & forward force-law"** block of
slides covering §2, §8, §9, §10, placed after the existing dataset-quality slides.
Success = new slides match the deck's existing visual style, every number traces to
`TRAJ_DIAGNOSTICRESULTS.md`, and the narrative arc below is preserved.

## 4. Key design decisions (already made — present these, don't re-derive)
1. **μ enters multiplicatively, χ modulates it: `F = μ·(A + C·χ) + B`** (LuGre
   linearization; `B` = μ/χ-independent bristle term). The additive `A·μ+B+C·χ` form is
   wrong and is **excluded from the deck** (it was a diagnostic dead-end).
2. **Both parameters are gated, not global.** μ is **slip-gated** (readable only at high
   co-directional slip; `mult_fraction` 0.2→0.98, `|ΔF|≈28 N`). χ is **spin-gated**
   (`partial R²_χ` 0.01→0.71 as `|ω_z|` rises; χ-swing ~1–3 N at high spin).
3. **The identifiable regime is a thin tail.** ~45 % of data is gross slip, but only
   **~4–5 %** sits at high co-directional slip where μ/χ are readable → motivates
   confidence-gating / tail-weighting in the PINN.
4. **The χ-gate variable `ω_z` is ~87–90 % unmeasurable roller rate** (μ-gate `V∥` is
   ~0.2 % roller) → χ-ID is hard-coupled to the A2 state-observer; μ-gate rides on
   measurables. (§9.2)
5. **The form is the right backbone, ~½ the force.** `R²_form` ≈ 0.45–0.59 pooled
   (Fperp 0.71–0.80; Fpar bimodal 0.14 low-slip → 0.98 high-slip); the residual ~½ is the
   bristle **state** the GRU/SSM must supply. (§10)
6. **No-slip "force only along the axle" is materially wrong** — `|Fperp|/|Fpar|` median
   0.39, exceeds 1 in 23 % of samples (§9.3) → justifies the composite friction model.

## 5. Open decisions / blocking relationships
- **Noise-floor threshold** for a hard "identifiable / not" verdict is deferred — present
  swings in Newtons, do not assert a binary verdict.
- **Hand-back to the PINN-training session:** these slides state the forward-head design
  (`F = μ(A(slip) + C(slip,|ω_z|)χ) + B(slip)` + GRU for state; χ needs reconstructed
  `|ω_z|`). Keep that as the closing "implications for training" slide.
- Per-channel (Fperp vs Fpar) asymmetry in §10 is the key non-obvious point — keep it,
  don't collapse to the pooled R² only.

## 6. Deliverables
1. Updated `presentation/deck.html` with the new identifiability/force-law slide block
   (suggested order: §2 χ → §8 μ → §9 inferences → §10 force-law → implications).
2. Any new figures rendered into `presentation/assets/` (or reuse the two existing PNGs).
3. Rebuilt `presentation/Mecanum_PINN_Deck.pdf`.

## 7. Conventions to respect
- **Match the existing deck** — same theme, slide layout, fonts, footer/logo; additive
  edits only, do not restyle existing slides.
- **Numbers are authoritative in `TRAJ_DIAGNOSTICRESULTS.md`** — quote exactly; if a value
  isn't there, ask rather than invent.
- **LaTeX/MathJax is correct in the deck** (Unicode is only for chat/md).
- Static figures only; no interactive widgets. Tables are an acceptable substitute for any
  un-rendered figure (user's call, per slide).
- Surgical, reviewable edits; show the slide plan before bulk-editing `deck.html`.
