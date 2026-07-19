# Observer Grounding Results — Academic Navy Slide Deck

> **Generated:** 2026-07-16
> **Stack:** Single-file HTML5 + CSS3, MathJax (local `tex-svg.js`), no build step, no framework
> **Scope:** Static presentation deck (print-to-PDF capable), content-only — no app logic

## 1. Overview
Build a self-contained HTML slide deck that presents the roller-spin observer's
grounding-phase results, converting `observer_v1_py/OBSERVER_GROUNDING_EVOLUTION.md` into
~10–12 slides. The deck must be visually indistinguishable from the existing
`presentation/deck.html` ("Academic Navy" theme): same CSS variables, slide skeleton,
header/footer bands, panel/table/figure primitives, and local MathJax. Output is ONE
`.html` file that opens offline and prints cleanly to 16:9 PDF. The deck reuses the
existing theme system verbatim — it does **not** define a new visual language.

## 2. Architecture Pattern
**Template-cloned static deck.** Copy the theme layer (`:root` variables, base layout,
`.slide`/`.hdr`/`.body`/`.ftr`, and the component classes) from `presentation/deck.html`,
then author new `.slide` sections for this content. Rationale: the theme is already
authored, VNIT-aligned, and print-tuned; cloning guarantees pixel parity and avoids
theme drift. One slide = one `<section class="slide">`; a print stylesheet paginates one
slide per page.

## 3. Technology Constraints
- **HTML:** single self-contained file; all styles in one `<style>` block in `<head>`.
- **CSS:** reuse the exact `:root` custom properties from `presentation/deck.html`
  (`--navy:#10243E`, `--navy2:#1B3A5B`, `--steel:#3E6691`, `--gold:#B07D2B`,
  `--goldl:#D8AE5C`, `--ink:#1C2530`, `--light:#EEF2F7`, `--grid:#C9D3DF`,
  `--red:#9B2D30`, `--green:#2E6B4F`, serif/sans font stacks).
- **Math:** MathJax via the **local** `assets/tex-svg.js` (`<script id="MathJax-script"
  async>`), exactly as `deck.html` loads it — NOT a CDN (offline requirement).
- **Fonts:** system serif/sans stacks already in the theme (Palatino Linotype / Book
  Antiqua / Georgia serif; Segoe UI / system-ui sans). No web-font downloads.
- **Device targets:** desktop browser + print-to-PDF. No JS framework, no bundler.
- **Explicit exclusions:** no external network requests of any kind (CSP-safe, offline);
  no CDN; no build tooling; no reveal.js / impress.js — plain sectioned HTML like the
  reference deck; no client-side data fetching.

## 4. Component Breakdown

### `theme-layer` (CSS)
- **Type:** `<style>` block (cloned)
- **Responsibility:** the Academic Navy visual system — copied 1:1 from `deck.html`.
- **Inputs:** none. **Outputs:** the class vocabulary all slides consume.
- **Key pieces to preserve:** `:root` vars; `.slide` (16:9 page box + `counter-reset`/
  `counter-increment` slide numbering); `.hdr` (navy band, `.sec` gold uppercase section
  label, `.brand` VNIT block); `.body`; `.ftr` (`.vn` navy VNIT mark); `.rule` (gold-then-
  grid hairline); text roles `.g` (gold), `.k`/`b.key` (navy), `.st` (steel); `.panel`
  (+ `.gold`/`.red`/`.green` variants, `.panel h3` uppercase sans label); `table`/`th`
  (navy header); `.fph` figure-placeholder; `.title` cover-slide variant.
- **Depends on:** nothing (foundation).

### `title-slide`
- **Type:** `<section class="slide title">`
- **Responsibility:** cover — deck title, subtitle, VNIT / IMECE 2026 attribution bar.
- **Content:** "From LuGre States to a Working Slip Gate" (or similar); subtitle names the
  three versions; reuse `.title .tbar`, `.title h1 .hl`, `.title .sub`, `.title .inst`.

### `evolution-slides` (section 1 of the source doc)
- **Type:** 3–4 `<section class="slide">`
- **Responsibility:** v1 → v2-Hy3 → gamma-kin architecture story with intent per step.
- **Inputs:** §1 of `OBSERVER_GROUNDING_EVOLUTION.md`.
- **Layout:** one "lineage" overview slide (3-column or stacked `.panel`s: v1 / v2-Hy3 /
  gamma-kin, each with inputs→targets→intent); then one slide per transition highlighting
  the *why* (drop bristle states + sensor-real front-end; residual γ + uniform dv_scale).
- **Math:** render `γ_noslip`, `γ̂ = γ_noslip + Δγ̂·dgamma_scale`, `V_y_used` via MathJax.

### `results-slides` (section 2)
- **Type:** 3–4 `<section class="slide">`
- **Responsibility:** the grounding results table + the four justification points.
- **Inputs:** §2 table + bullets.
- **Layout:** one slide = the 6-row results `<table>` (version × [γ_all, γ_stick, dVx R²,
  |Vp| err, stick_frac]); highlight the gate row with `.g`/color. Then 1–2 slides for the
  narrative wins (gate un-saturated, γ 4–7×, ΔV_x repaired from R²=−0.32, decimation not
  the limiter, γ⊥ΔV orthogonality). Use `.panel.green` for "works"/accept and `.panel.red`
  for the baseline failure (stick_frac≈0, R²<0).

### `closing-slide`
- **Type:** `<section class="slide">`
- **Responsibility:** current status (deployable λ phase-out in progress) + evidence index.
- **Content:** §2 "In progress" paragraph + §3 repo paths in a `.panel`.

### `figure-placeholders`
- **Type:** `.fph` blocks
- **Responsibility:** where a plot belongs but isn't rendered yet (e.g., per-epoch γ/slip
  trend 0→150, |Vp| attribution bar). Use the theme's dashed-gold `.fph` with a `.lab`,
  `.desc`, and `.fn` filename hint — matches how `deck.html` marks pending figures.

## 5. File & Directory Structure
```
presentation/
├── deck.html                       # EXISTING reference — clone theme from here, do not edit
├── academic_navy_theme.md          # EXISTING theme spec (authority for colors/fonts)
├── assets/
│   └── tex-svg.js                  # EXISTING local MathJax — reference by relative path
└── observer_grounding_deck.html    # NEW — the deliverable
```

## 6. Key Interfaces
The per-slide skeleton every content slide follows (structure only — fill from the source doc):

```html
<!-- one content slide -->
<section class="slide">
  <div class="hdr">
    <div class="sec">…section label…</div>
    <div class="brand">…VNIT mark…</div>
  </div>
  <div class="body">
    <h1>…title with <span class="accent">…</span>…</h1>
    <div class="subttl">…one-line takeaway…</div>
    <div class="rule"></div>
    …panels / table / fph / MathJax…
  </div>
  <div class="ftr">…<span class="vn">VNIT</span> · IMECE 2026 · slide #…</div>
</section>
```

Results table contract (columns fixed; one row per version/regime):

```html
<table>
  <thead><tr><th>Version</th><th>γ all</th><th>γ stick</th>
    <th>dVx R²</th><th>|Vp| err</th><th>stick_frac</th></tr></thead>
  <tbody>…rows from §2, gate/failure cells emphasized…</tbody>
</table>
```

## 7. Content-to-Slide Mapping (the "data flow")
1. Source of truth = `observer_v1_py/OBSERVER_GROUNDING_EVOLUTION.md`, three sections.
2. §1 (evolution) → title + 3–4 evolution slides; every architecture claim keeps its
   *intent* line (why the change was made), not just the what.
3. §2 (results) → the results table slide + justification slides; numbers must be copied
   **exactly** from the table (physical RMSE, cross-fold test): do not round or restate.
4. §3 (evidence index) → closing slide panel.
5. Equations pass through MathJax; tables/panels through the cloned theme classes.
6. No meta/agenda slide (per request — the deck *is* the artifact).

## 8. Implementation Sequence
1. **Clone the theme layer** from `deck.html` into the new file's `<head>` — verify colors
   and one sample slide render identically before adding content.
2. **Wire local MathJax** (`assets/tex-svg.js`) and confirm one test equation renders.
3. **Title slide.**
4. **Evolution slides** (§1) — reuse `.panel`/`.stategrid` for the 3-version comparison.
5. **Results table slide** (§2) — the load-bearing slide; get the table exact.
6. **Justification slides** (§2 bullets) with accept/reject `.panel` coloring.
7. **Closing slide** (§3).
8. **Print stylesheet pass** — one `.slide` per page, colors preserved
   (`print-color-adjust:exact`, already in the theme).

## 9. Deck-Specific Considerations (replaces ML section)
- **Self-contained:** zero external requests. MathJax is the local `assets/tex-svg.js`;
  keep it a relative-path reference so the deck + `assets/` folder move together (or inline
  it if a single loose file is required — state which in the file header comment).
- **Fidelity over novelty:** do not restyle. Every color/spacing decision defers to the
  cloned `:root` and `academic_navy_theme.md`. New classes only if a layout truly has no
  existing analog.
- **Numbers are authoritative:** the results table and per-metric values are the paper's
  claims — transcribe verbatim from the MD; never invent or "clean up" a figure.
- **Print correctness:** verify PDF export shows exactly one slide per page with the navy
  header/footer bands and table header fills intact (the theme sets `print-color-adjust`).
- **Accessibility of emphasis:** keep the theme's semantic colors — `.green`=accept,
  `.red`=reject/limit, `.gold`=key term — consistent with how `deck.html` uses them.

## 10. Success Criteria
- [ ] Opens offline in a browser with **no** network requests (check devtools Network).
- [ ] Side-by-side with `deck.html`, header/footer/panels/tables are pixel-consistent.
- [ ] All equations render via local MathJax (no raw TeX visible).
- [ ] Results table values match `OBSERVER_GROUNDING_EVOLUTION.md` §2 exactly.
- [ ] Print-to-PDF yields one 16:9 slide per page with colors preserved.
- [ ] No agenda/meta slide; ~10–12 slides total.

## 11. Out of Scope
- Rendering the actual result figures (use `.fph` placeholders with filename hints).
- Editing `deck.html` or the theme spec.
- The deployable λ-phase-out numbers (still training — leave the "in progress" note).
- Any interactive/animated behavior; speaker notes; multi-theme support.
