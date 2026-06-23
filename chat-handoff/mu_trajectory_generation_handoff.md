# Task brief — extend the sweep to μ = 0.3 and 0.8 (for the session that owns the μ=0.5 pipeline)

You generated the μ=0.5 dataset and know the driver, the `gen_*_toml.py` generators, the
`_chinc` config/output-dir mechanism, the filename scheme, and the conventions. This brief
is **only the new context + decisions** from the Forward-Inverse PINN session — it does not
re-explain the pipeline. The one thing to internalize: **do NOT just set `mu_friction` and
re-run the existing TOMLs.** Read §2.

## 1. Why this task exists
The PINN must **identify μ from reconstructed forces**, but with a single μ the map `F=μN·Φ`
is degenerate (μ is absorbable into the learned shape) — μ is unidentifiable. We need μ
variation. Grid decided = **{0.3, 0.5, 0.8}**: brackets the existing 0.5 in grip-space → gives
interpolation (predict held-out 0.5), both extrapolation directions, and low-grip coverage.

## 2. The non-obvious part — re-LOAD the friction circle, don't replicate envelopes
**μ and χ are observable through different mechanisms.** χ is a spin-gated *additive* coupling,
observable sub-cap (why the conservative, fixed-envelope χ-quads worked at μ=0.5). **μ is the
*scale of the friction circle itself*** — well below the cap, the forces are dynamics-set and
μ-blind; μ only shows up through friction-circle **utilization** / the slip–force relationship.

So if you keep the μ=0.5 envelopes and change μ:
- **μ=0.8** → cap rises to 0.8·N, utilization drops ~37% → μ becomes nearly unobservable.
- **μ=0.3** → cap falls → the existing loads over-drive the circle → saturation / lost tracking.

**Goal: scale the loading envelopes so realized friction-circle utilization matches the μ=0.5
baseline** (verify with `tracking_gate.py`). Comparable utilization = comparable μ-observability
across the grid, and it populates overlapping slip bins for the downstream collapse test.

## 3. Scaling rule
- Friction loading is set by **accelerations** (longitudinal, lateral, yaw-angular) and the
  centripetal **Ω·V** term — **not** forward velocity. **Hold `Vx_cap` fixed** (it's a
  wheel-speed/actuator budget, μ-independent). Scale only the accel / lateral / yaw amplitudes.
- Factor **k = μ/0.5 → 0.6 (μ=0.3), 1.6 (μ=0.8)** on accel/force-type limits; **√k** for
  centripetal-V² velocity amplitudes. Final values tuned empirically in the pilot (§5).
- Friction limits scale ∝ μ. From the current TOML headers (μ=0.5: long-accel 1.84, lat 1.17,
  ×0.9 band → 1.66/1.05; combined per-wheel force `0.9·μ·N_min = 37.6 N`, N_min≈83.6):

  | μ | long-accel (0.9·) | lat-accel (0.9·) | combined wheel force |
  |---|---|---|---|
  | 0.3 | 1.10 (0.99) | 0.70 (0.63) | 22.6 N |
  | 0.5 | 1.84 (1.66) | 1.17 (1.05) | 37.6 N |
  | 0.8 | 2.94 (2.65) | 1.87 (1.68) | 60.2 N |

- **Rebuild via `gen_*_toml.py` at the new μ-scaled limits** (so rows stay friction-circle
  re-validated) — don't hand-edit combo arrays. Keep row counts identical → same 1776 composition.

## 4. Per-profile (scaling is not uniform)
- **Friction-bound → scale**: octagon (`accel_frac`, `lat_vamp`, lateral part of `vcru`),
  coupled_vomega (`V_peak`, `V_const`, `Om_peak`, `Om_const`), and confirm ellipse / spiral_orbit.
- **Rate-bound → leave the rate caps**: spin_creep — `yaw_spin` is capped at |Ω|<3.15 rad/s (a
  wheel-speed limit, μ-independent) and its friction coupling is 0.155 ≪ 1.05, i.e. it barely
  loads the circle. long_circle / multisine likewise; classify each from its TOML header notes.
- **High-μ caution**: at μ=0.8 load via *sharper ramps*, not higher peak velocity, so scaled accel
  doesn't breach the fixed Vx / yaw-rate caps. Flag any profile that turns rate-bound.

## 5. Workflow
1. **Pilot first** (sims are expensive + standby-fragile): the near-cap combos in
   `trajectory_files_chinc/selected_near_cap_combos.json` (~22 coupled_vomega + 31 octagon) at each
   new μ, across a few candidate scale factors. Run `tracking_gate.py`; lock the factors whose
   realized utilization matches the μ=0.5 band with no `reject_missed`.
2. **Full sweep**: 1776 `.arrow` (+ JLD2) per μ, same profile/χ composition as μ=0.5, scaled
   amplitudes, into separate output dirs (`..._mu0.3\`, `..._mu0.8\`) to keep the baseline pristine.
   `keep_awake.py` running; same `--sweep-seed 1234`.
3. **χ is deferred** — keep χ at the existing distribution (full 1776 parity incl. the `_chinc`
   χ-quads; that gives μ×χ for free later). Don't analyze χ now; μ is the focus.

## 6. Deliverables + hand-back
- μ=0.3 and μ=0.8 datasets (1776 each), and `mu_generation_note.md` recording the final
  per-profile/per-cap scale factors and realized-utilization vs the μ=0.5 baseline.
- These cross back to the PINN session, which runs a model-free `mu_identifiability` analysis
  (mirrors `chi_identifiability.py`) and the μ-axis PINN validation — both blocked on this.
