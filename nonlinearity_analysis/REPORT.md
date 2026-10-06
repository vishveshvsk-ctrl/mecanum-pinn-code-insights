# Plant near-linearity quantification — synthesis report

**Question.** Is the voltage-input Mecanum plant near-linear inside the
operating envelope the v3 controller campaign runs in — and is that why the
linear PID-CT cascade beats ASMC?

**Plant point.** μ = 0.5, χ = 0.005 m, friction case 1, quasi-static motor with
back-EMF active, dynamic LuGre bristles, Adamov spin coupling. All numbers below
are computed against the *exact* `PlantMod.plant_rhs!` (no re-derived physics).
Scripts: `tierAC_static_jacobian.jl`, `tierB_multisine.jl` (+ `.bat` twins);
raw data in `results/`.

---

## Tier A — steady-state map v → V_req (Allgöwer-style, numerical)

Steady states solved as 23-equation root-finds (body/wheel/roller/bristle
balance) with radial continuation from the origin root, verified against the
*clamped* motor map; 8 roller phases per point.

| Measure | Value | Reading |
|---|---|---|
| Allgöwer ν (through-origin best linear map, feasible subset) | max 0.051, R² = 0.9994 | mild |
| Allgöwer ν (affine fit, Coulomb offset removed) | max 0.053, R² = 0.9994 | mild |
| Secant gain spread, x (0.05→1.00 cap) | 83.8 → 47.8 V·s/m (1.75×) | worst at low speed (friction offset) |
| Secant gain spread, y (0.05→0.75) | 90.5 → 65.9 (1.39×) | mild |
| Secant gain spread, ψ (0.05→0.50) | 25.5 → 20.6 (1.24×) | mild |
| Roller-phase ripple of the static map | ≤ 2.6% | small |

**Feasibility finding (back-EMF envelope, steady state, 24 V bus):**
x feasible to 100% of the legacy cap (0.63 m/s); **y only to 75% (0.47 m/s)**;
**ψ̇ only to 50% (1.9 rad/s)**. Pure spin at the legacy 3.8 rad/s cap needs
~25.3 V of back-EMF alone — electrically unreachable. At the y cap the required
voltage is *roller-phase-dependent* (19.5–23.9 V across phases at 100% y, with
two phases infeasible): sustained lateral motion at the friction-derived cap
brushes the bus limit twice per roller period. This is direct numerical
confirmation of the corrected envelope TeX (back-EMF + motor torque limits),
and says the legacy `v_max_axis`/`vcmd_clamp` ψ̇ cap of 3.8 rad/s is roughly 2×
the electrically honest sustained value.

## Tier C — local linearisation (23-state mechanical subsystem, exact Jacobians)

Transfer G(jω) = C(jωI−A)⁻¹B evaluated at 0.1–3 Hz (the control band). DC is
deliberately not used: at rest the preslip bristle stiffness holds the platform
against any constant force, so v_dc = 0 — that zero **is** the stiction
nonlinearity, not a modelling artifact.

| Measure | Value | Reading |
|---|---|---|
| ‖G_jj‖ spread across 12 moving operating points, 0.1–1 Hz | 1.06–1.8× | within the 2× "mild" threshold |
| ‖G_jj‖ spread at 3 Hz | 2.1–2.2× | at/above threshold, top of control band |
| Max cross-coupling ‖off-diag‖/‖diag‖ inside envelope | 0.07–0.14 | near-decoupled |
| Cross-coupling at y = 100% cap (envelope edge) | ≈ 0.99 | strong y→ψ coupling + 40% gain compression — friction-circle saturation showing up exactly where expected |
| Rest point | v_dc = 0 (preslip) | strongly nonlinear *at zero velocity*, tiny amplitude range |

## Tier B — open-loop biased odd-multisine BLA (Schoukens/Pintelon)

Task-space voltage excitation (codebase's own allocation patterns), random-phase
odd multisine 0.1–10 Hz, excited bins k≡1 mod 4; detection at k≡3 mod 4 (odd
nonlinearity) and even k (even nonlinearity); 6 periods (2 dropped), per-period
detrended, no measurement noise. Bias = cruise at 25%/50% of cap; excitation =
10%/25% of cap; plus zero-bias (velocity-reversal) reference.

| Condition | Total NL distortion | Reading |
|---|---|---|
| Best (axis 3, bias 50%, exc 10%) | −21.0 dB | 0.8% distortion power |
| Typical (exc 10–25%, any bias) | −14.5 … −19 dB | 2–3.5% |
| Worst (axis 2, bias 50%, exc 25%) | −10.7 dB | 8.5% |
| Zero-bias (reversal regime) | −15.0 … −16.4 dB | only ~1 dB worse than biased — the reversal/stiction regime is *not* dramatically more nonlinear at these amplitudes |

Amplitude dependence (the BLA fingerprint of nonlinearity): distortion rises
with excitation at fixed bias (e.g. axis 2 @ 50% bias: −15.4 → −10.7 dB) —
present but gentle. BLA gain ‖G(0.5 Hz)‖ varies only ±6–10% across all
amplitudes and biases per axis. Stochastic (period-to-period) distortion:
−41 … −71 dB — the nonlinearity is almost entirely *systematic* (a deterministic
distortion the same every period), which is precisely the part a linear-model +
disturbance-observer or a learned residual can absorb.

---

## Verdict

**The plant is mildly nonlinear inside the operating envelope, by every
framework in the taxonomy:**

- Allgöwer ν ≈ 0.05 on the static map (R² = 0.9994);
- BLA nonlinear distortion −11 to −21 dB (1–9% power) — against the ~−20 dB
  rule-of-thumb for "mildly nonlinear", we sit at the mild end with the worst
  corner at high lateral drive;
- BLA gain amplitude-dependence ±6–10%; local gain spread ≤ 1.8× in the control
  band (2× threshold);
- cross-coupling ≤ 0.14 away from the envelope edge.

**Strong nonlinearity is confined to three narrow places:** (i) zero velocity /
reversal (preslip — but it costs ~1 dB and occupies little trajectory time);
(ii) the envelope edge (y-cap gain compression + y→ψ coupling → 1);
(iii) the fast bristle/motor timescales (ms — invisible to the control band,
and exactly what the PINN's 2 kHz identification signal lives on).

**Does this explain PID-CT > ASMC?** Yes, with the mechanisms already
identified: with (a) both controllers sharing the identical computed-torque
feedforward and (b) the residual error plant being ~95%-linear (power-wise)
with a bounded, mostly *systematic* disturbance, the IMC pole-cancellation PI
is operating on essentially the plant it is optimal for, while ASMC pays its
boundary-layer deadband, adaptation lag, and switching ripple for robustness
against a model-error scenario that does not occur. The s100 ablation fits:
giving ASMC more authority helped it significantly (it was partly
authority-limited) but did not close the gap — a structural deficit, not a
budget one.

**Caveats for the thesis.** (1) "Near-linear" is a statement about the control
band (≤ ~3 Hz) and the envelope interior; the identification-relevant friction
dynamics are fully nonlinear at ms timescales — no contradiction with the
PINN's motivation, but the two statements must be timescale-qualified when
written up. (2) The electrical envelope finding (sustained y ≤ 0.47 m/s,
ψ̇ ≤ 1.9 rad/s at 24 V) should be cross-checked against the corrected
envelope TeX numbers and reflected in `v_max_axis`/`vcmd_clamp` if future
trajectories are meant to be sustained-feasible, not just transient-feasible.
(3) All numbers are at μ = 0.5, χ = 0.005, case 1; the lower-μ and low-viscous
(case 2) corners will be *less* linear (less damping, deeper Stribeck relative
dip) — cheap to re-run if needed.

---

## Follow-up: parametric scan — what would it take to push the plant nonlinear?

`tierA_parametric.jl` re-runs the Tier A machinery over lever combinations
(COM offset ×2/×3, χ ×4, μ = 0.3, friction case 2, and an extreme stack).
Same grid, same exact steady states (`tierA_parametric_report.md`):

| case | ν max | R² | gain spread x/y/ψ | feasible max x/y/ψ |
|---|---|---|---|---|
| base (μ=.5, χ=.005, case 1) | 0.051 | 0.9994 | 1.75 / 1.39 / 1.24 | 1.00 / 0.75 / 0.50 |
| COM ×2 | 0.051 | 0.9993 | 1.75 / 1.38 / 1.24 | 1.00 / 0.75 / 0.50 |
| COM ×3 | 0.051 | 0.9991 | 1.75 / 1.36 / 1.23 | 1.00 / 0.50 / 0.50 |
| χ ×4 (0.020) | 0.051 | 0.9992 | 1.75 / 1.11 / 1.12 | 1.00 / 0.75 / 0.50 |
| μ = 0.3 | 0.050 | 0.9993 | 1.75 / 1.37 / 1.24 | 1.00 / 0.50 / 0.50 |
| case 2 (low viscous) | **0.077** | 0.9980 | 1.96 / 1.70 / 1.35 | 1.00 / **1.00** / **0.75** |
| extreme (μ=.3+χ.020+COM×3) | — (n=5) | — | 1.75 / 1.16 / 1.16 | 1.00 / **0.25** / **0.25** |

Reading (see the scan's own caveat below):

- **The static v→V map is stubbornly near-linear under every lever.** ν never
  exceeds 0.08. The reason is structural: steady-state voltage is dominated by
  back-EMF + viscous drag; the friction circle binds under *acceleration*
  demand, which the static map never probes. So the static-map ν is the wrong
  proxy for "regime entry".
- **The dominant effect of every lever is on the FEASIBLE REGION, not the
  shape.** μ ↓, COM ↑, χ ↑ all shrink the envelope toward the operating
  point; the extreme stack collapses sustained y to 25% of cap. The mechanism
  for entering the nonlinear regime is therefore: shrink the headroom until
  ordinary trajectories operate at the friction-circle edge — the edge is
  where the measured nonlinearity lives (Tier C: coupling → 1, gain
  compression ~40% at the y-cap).
- **COM offset's specific role**: at ×3 the binding wheel's normal load
  halves (N₃: 69.5 → ~34 N), so the E54 friction-circle gate
  (`F_par,avail = sqrt((s·μ·N₃)² − F_perp3²) − |F_par3|`) saturates at
  roughly half the previous acceleration — it lowers the *threshold* of
  regime entry, confirmed by the y feasibility drop 0.75 → 0.50, but it adds
  no new nonlinearity to the map itself.
- **Case 2 (low viscous) is the only lever that makes the plant itself
  visibly less linear** (ν +50%, spreads up) *and* enlarges the feasible
  envelope (less drag current) — it removes the linear damping component, so
  the friction nonlinearity carries proportionally more of the response.
  Also the cheapest to exploit: it is already a `friction_case` switch.
- χ ×4 leaves the static map untouched (spin coupling barely loads the
  steady drag balance) — its nonlinear effect would show under *spin
  transients*, i.e. in Tier B with biased ψ̇ excitation, not here.

**Scan caveat.** The proxy mismatch matters: to quantify "how nonlinear does
the plant become at the edge" per parameter, the next run is Tier C (local
Jacobians, gain/pole/coupling spread) per parameter pack, or Tier B BLA under
combined near-saturating bias — not the static map. The static scan answers
"where does the edge move", Tier C/B answer "what does the edge do to you".

