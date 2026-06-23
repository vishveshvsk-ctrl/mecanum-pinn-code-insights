# Solver selection — decision record (Mecanum PINN data-generation sweep)

**Date:** 2026-06-14
**Scope:** which stiff ODE integrator the ~1,485-job production sweep
(`Data_Generation_Julia.jl`) should use.
**Verdict:** solver family = **RadauIIA5, QNDF, FBDF**; reject **TRBDF2** and **KenCarp47**.
Stage B (added 2026-06-14, below) pins the *tolerance*: all three clear margin ≤ 0.1 on
**all 7** cases at **FBDF 1e-9 / RadauIIA5 1e-8 / QNDF 1e-10** (dtmax 1e-3), re-confirmed
against an independent RadauIIA9 @ 1e-12 reference. **FBDF @ 1e-9 is the production choice
on cost** (232 s vs 283/290 s, lowest `nf`) = the current `base.toml [solver]`, no change
needed. The per-config margins are at different reltols, so they are not a cross-solver
accuracy ranking. See Stage B for the full tables.

---

## TL;DR

On the hardest excitation case, only four of six candidate solvers can produce
force/moment labels that sit ≥10× below the dataset's interpolation noise floor.
Ranked by work to get there:

| Rank | Solver | Passing reltol | nf (work) | wall (s) | margin | state err | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **RadauIIA5** | 1e-7 | 1,889,465 | 29.6 | 0.062 | 8.3e-5 | ✅ use |
| 2 | **QNDF** | 1e-9 | 1,945,713 | 31.7 | 0.015 | 3.4e-5 | ✅ use |
| 3 | **FBDF** | 1e-9 | 2,351,417 | 34.4 | 0.062 | 8.8e-5 | ✅ use (current `base.toml`) |
| 4 | RadauIIA9 | 1e-9 | 2,554,822 | 38.0 | 0.087 | 2.0e-4 | acceptable, but dominated |
| — | **KenCarp47** | none | — | — | best **1.08** | — | ❌ reject |
| — | **TRBDF2** | none | — | — | best **~594** | — | ❌ reject |

The three retained solvers are the basis for restricting future sweeps and
ablations to {RadauIIA5, QNDF, FBDF}.

---

## What was tested

- **Notebook:** `Solver_Ablation_Multisine.ipynb`, Stage A only, run headless
  via `nbclient` (`julia-1.12` kernel, project env). Executed notebook archived
  as `_stageA_run_executed.ipynb`; full per-config table in `stageA_results.arrow`.
- **Case:** `ms75_fhi3.5` — the hardest multisine row (75 %-cap amplitude ×
  `f_hi = 3.5 Hz` bandwidth): maximum slip-reversal density and deepest
  friction-circle excursions, so it stresses the integrator, not event handling.
- **Physics point:** μ = 0.5, χ = 0.005, `:lugre_adamov`, friction case 1, DOB off
  — the production parameters.
- **Matrix (QUICK_PASS):** 6 solvers × reltol ∈ {1e-5, 1e-7, 1e-9} × dtmax ∈ {∞, 1e-3},
  1 rep, per-solve wall guard.
- **Reference:** RadauIIA5 @ reltol 1e-10, abstol/100, 4 kHz grid (cached;
  self-validated against a 1e-11 reference — its own noise is far below the floor).

### Acceptance criterion

The PINN consumes the recomputed labels `Fx_i / Fy_i / Mz_i / Msat_i` on the
2 kHz grid. The metric is the **RMS error of those label arrays vs. the reference**,
grouped into three families (F, Mz, Msat), normalized by each family's
**force-interpolation floor** (the error the pipeline already commits by treating
2 kHz force labels as continuous through stick–slip):

```
margin = max( errF/floorF , errMz/floorMz , errMsat/floorMsat )
PASS  ⇔  margin ≤ 0.1        # solver error ≥ 10× below the existing noise floor
```

A passing solver contributes negligible label noise to the PINN's physics-residual
budget. A failing solver makes the dataset solver-noise-limited.

---

## Why TRBDF2 and KenCarp47 are rejected

Both **fail outright** — they never reach `margin ≤ 0.1` on this case, and not
marginally:

- **TRBDF2** — best margin **~594** (at reltol 1e-7), i.e. its labels are ~600×
  the interpolation floor; even at 1e-9 the margin is ~675. Tightening tolerance
  does not help.
- **KenCarp47** — best margin **1.08** (at reltol 1e-9, dtmax 1e-3); at dtmax = ∞
  it blows up to 600+. It gets within a factor of ~10 of the floor but never clears
  it within the tolerance ladder.

**Physical reading.** The slip-spin LuGre bristle model carries a ≥500× stiffness
(`tanh` smoothing factor) in its boundary layer. The L-stable / high-order methods
(RadauIIA5/9 and the BDF family FBDF/QNDF) damp that boundary layer correctly; the
lower-order A-stable TRBDF2 and the ESDIRK KenCarp47 ring / under-resolve it, so
their *labels* diverge from the reference far above the floor even when their
step-by-step state error looks plausible. This is the "stiffness order-reduction"
failure the ablation was designed to catch.

> Note: TRBDF2 is still listed as the documented default in `CLAUDE.md §10`. That
> is stale relative to this result — the active `base.toml [solver].name` is already
> `FBDF`, which this study confirms is a correct (passing) choice. Update §10 to
> reflect {RadauIIA5, QNDF, FBDF}.

## Why RadauIIA5, QNDF, FBDF are retained

- **RadauIIA5** — clears the floor at the **loosest** tolerance (1e-7), so it does
  the **least work** (lowest nf) and is the most efficient. First choice.
- **QNDF** — best raw accuracy (margin 0.015, lowest state error); needs 1e-9 but
  its work is essentially tied with RadauIIA5.
- **FBDF** — passes solidly at 1e-9; current production solver in `base.toml`.
- **RadauIIA9** passes too but is dominated by all three above on work — keep as a
  fallback, not a primary.

---

## Stage B — robustness across the 7 cases (added 2026-06-14)

Each config run on all 7 cases (both amplitude caps × `f_hi` ∈ {1.0, 2.5, 3.5} + the
`spiral_iso` spot-check), judged against each case's own interpolation floor. Two passes:
the initial finalists (cheapest passing clamped Stage-A config per solver + the two
on-disk FBDF configs), then a tolerance-extension pass (`RadauIIA5 @ 1e-8`, `QNDF @ 1e-10`)
to give RadauIIA5/QNDF a fair shot. Margin per case (PASS ⇔ ≤ 0.1), passing configs first:

| config | ms75_1.0 | ms75_2.5 | ms75_3.5 | ms50_1.0 | ms50_2.5 | ms50_3.5 | spiral | verdict | wall |
|---|---|---|---|---|---|---|---|---|---|
| **FBDF 1e-9 / 1e-3** | 0.037 | 0.047 | 0.062 | 0.032 | 0.040 | 0.015 | 0.004 | ✅ PASS (worst 0.062) | **214 s** |
| **QNDF 1e-10 / 1e-3** | 0.010 | 0.010 | 0.062 | 0.003 | 0.011 | 0.003 | 0.001 | ✅ PASS (worst 0.062) | 250 s |
| **RadauIIA5 1e-8 / 1e-3** | 0.014 | 0.021 | 0.010 | 0.010 | 0.008 | 0.023 | 0.003 | ✅ PASS (worst **0.023**) | 292 s |
| QNDF 1e-9 / 1e-3 | 0.004 | 0.027 | 0.015 | 0.027 | **0.275** | 0.004 | 0.006 | ✗ fails ms50_fhi2.5 | 183 s |
| RadauIIA5 1e-7 / 1e-3 | 0.078 | **0.145** | 0.062 | 0.095 | 0.041 | 0.056 | 0.011 | ✗ fails ms75_fhi2.5 | 210 s |
| FBDF 1e-8 / 1e-3 | 0.066 | 0.163 | 0.063 | 0.132 | **0.31** | 0.043 | 0.102 | ✗ fails 4 (too loose) | 156 s |

**All three solvers are production-robust at the right tolerance.** Choice:

- **FBDF @ 1e-9** — **cheapest robust config on both axes**, and already the production
  `base.toml`. Recommended default; no change needed.
- **RadauIIA5 @ 1e-8** / **QNDF @ 1e-10** — ~22–25 % more wall than FBDF, roughly tied with
  each other. All three pass with comfortable margin. **The per-config margins are at
  *different* reltols** (each solver's minimum-robust tolerance), so they are **not** a
  method-accuracy ranking — a solver run at a tighter reltol will show a smaller margin
  almost by construction. See the Reference-upgrade subsection.

**Clean wall / work comparison** (same-session, median-of-3, each solver at its robust
tolerance, summed over the 7 cases — from the reference-upgrade re-solve;
`stageB_radau9ref_results.arrow`). Earlier per-config wall figures were mixed-provenance —
prefer these:

| Finalist | total wall (7 cases) | mean s/case | mean `nf` |
|---|---|---|---|
| **FBDF @ 1e-9** | **232 s** | **33.1** | **2.16 M** |
| RadauIIA5 @ 1e-8 | 283 s | 40.4 | 2.37 M |
| QNDF @ 1e-10 | 290 s | 41.4 | 2.39 M |

FBDF is fastest *and* lowest `nf` (least work, so not a machine artifact), and fastest on
every individual case — the basis for choosing it as the production solver.

**Key Stage-B findings:**
- The **binding cases are the mid-bandwidth `f_hi = 2.5` rows**, not the nominal "hardest"
  `f_hi = 3.5`. At loose tolerance the solvers ring on the `2.5` rows more than on the
  primary case — this is exactly why Stage A on the hardest row alone is insufficient to
  fix a production tolerance; each solver needs one notch tighter than its Stage-A cheapest.
- The initial finalists (`RadauIIA5 1e-7`, `QNDF 1e-9`) each failed exactly one mid-bandwidth
  case; tightening to `1e-8` / `1e-10` respectively clears all 7 — confirming RadauIIA5 and
  QNDF as valid production alternatives to FBDF, not just family members.

Stage B data: `stageB_results.arrow` (42 rows). Prior FBDF rows preserved across both
passes (the notebook's destructive in-place filters were removed). **The `wall` column in
the table above is mixed-provenance (different sessions / rep counts) — superseded; use the
clean same-session median-of-3 wall/`nf` comparison in the Reference-upgrade subsection
below.** Margins in this table are exact (vs the RadauIIA5 1e-10 reference).

### Reference upgrade — re-scored vs RadauIIA9 @ 1e-12 (added 2026-06-14)

The Stage-A/B reference above is RadauIIA5 @ 1e-10 — only ~1 order tighter than the
1e-9 candidate, and *same-method* as the RadauIIA5 candidate (shared truncation error
can flatter it). To make the verdict unimpeachable, the 3 finalists were re-scored
against an **independent, higher-order, tighter reference: RadauIIA9 @ reltol 1e-12,
abstol ABSTOL/1000** (≥3 orders below the tightest candidate, different solver family).
Results in `stageB_radau9ref_results.arrow` (21 rows); new `ref_…_tol1e-12.jld2` (7 cases).

| finalist | worst margin (RadauIIA5 1e-10 ref) | worst margin (**RadauIIA9 1e-12 ref**) | all 7 ≤ 0.1? |
|---|---|---|---|
| QNDF @ 1e-10 | 0.062 | **0.0093** | ✅ |
| FBDF @ 1e-9 | 0.062 | **0.045** | ✅ |
| RadauIIA5 @ 1e-8 | 0.023 | **0.062** | ✅ |

- **Verdict holds, strengthened:** all three still pass `≤ 0.1` on every case against the
  stronger reference. The production choice **FBDF @ 1e-9 is validated** at the highest
  standard.
- **No cross-solver accuracy ranking is claimed.** The three finalists run at *different*
  reltols (FBDF 1e-9, RadauIIA5 1e-8, QNDF 1e-10 — each its minimum-robust tolerance), so
  their margins are **not** comparable as a method-accuracy statement: a tighter reltol
  yields a smaller margin almost by construction (QNDF's 0.009 largely reflects its 1e-10
  tolerance, not an intrinsic edge over the BDF-cousin FBDF). A fair method comparison would
  hold reltol fixed — not done here, and not needed, since all three already pass.
- **One reference-choice caveat (independent of tolerance):** under the *same-method*
  RadauIIA5 1e-10 reference, the RadauIIA5-candidate margin was artificially low (0.010 on
  the primary case); against the independent RadauIIA9 1e-12 reference it rises to 0.062
  (the candidate stayed at 1e-8 in both — only the reference changed). Lesson: don't score a
  solver against a reference that shares its method. This is why the upgraded reference is a
  different family.
- For IMECE: cite the RadauIIA9 @ 1e-12 reference (Hairer–Wanner reference-solution
  methodology; SciMLBenchmarks convention) — it removes both the thin-gap and
  same-method objections.

**Final finding (no over-claim):** all three finalists pass `margin ≤ 0.1` on all 7 cases
against the tight independent reference; per-config worst margins are FBDF 0.045 /
RadauIIA5 0.062 / QNDF 0.0093 (at their respective robust reltols). **FBDF @ 1e-9 is the
production choice on cost** — 232 s vs 283 s / 290 s and the lowest `nf` — and is the
current `base.toml`.

## Caveats (so this record is not over-read)

1. **Tolerance, not just solver.** Stage B shows passing the hardest Stage-A case does **not**
   imply passing the mid-bandwidth rows. Each solver needs one notch tighter than its Stage-A
   cheapest to clear all 7 cases: FBDF 1e-9, RadauIIA5 1e-8, QNDF 1e-10. All three are now
   confirmed robust at those tolerances.
2. **Timing provenance.** RadauIIA9/FBDF/RadauIIA5 rows were reused from a cached
   prior run at the identical config; TRBDF2/KenCarp47/QNDF were executed in this
   pass. Margins and `nf` are exact and fully comparable (deterministic per
   solver/tol/case); **wall-clock is same-machine but different sessions**, which is
   why the ranking is on `nf`, not wall.
3. **A known DNF.** RadauIIA5 @ reltol 1e-9 / dtmax = ∞ did **not** converge and ran
   ~6990 s before termination (the per-step wall guard cannot interrupt a stuck
   Newton step). Use `dtmax = 1e-3` with RadauIIA5 at tight tolerance, or keep it at
   1e-7 where it passes cheaply.

---

## Evidence files (this folder)

- `stageA_results.arrow` — full per-config table (all 6 solvers, 48 rows).
- `_stageA_run_executed.ipynb` — the executed Stage A notebook (0 cell errors).
- `stageA_workprecision_nf.png`, `stageA_workprecision_wall.png`,
  `stageA_dtmax_subsidy.png` — work–precision plots (from the earlier full run).
- Reproduce: `Solver_Ablation_Multisine.ipynb` with `QUICK_PASS = true` and the
  6-solver `SOLVER_TABLE`/`SOLVER_ORDER`.
