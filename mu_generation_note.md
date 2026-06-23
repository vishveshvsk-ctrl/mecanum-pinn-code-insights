# μ-grid generation note — final scaling factors, justification, realized utilization

**Scope.** μ=0.3 and μ=0.8 trajectory datasets, scaled from the μ=0.5 reference, 1776
files each, matched 1-to-1 to μ=0.5 by `(profile, combo_idx, χ)`. Generator of record:
`_tmp/gen_mu_grid.jl` (supersedes the disagreeing `gen_mu_scaled.jl` / `setup_mu_full.jl`).
Verifier: `_tmp/verify_mu_grid.jl`. Generated 2026-06-16.

---

## 1. Scaling law (from `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex` §3.4)

`k = μ_new / 0.5`. The friction circle radius at each wheel is `μ·N_i`; under μ→μ_new it
scales by `k` (RHS of the master envelope, eq 27: `(A∥⁺b)_i² + F⊥,i² ≤ (μN_i)²`). Two
owners (the p1/p2 split), each scaled to hold realized utilization invariant:

- **Channel A — F⊥ (velocity-slaved, steady caps).** `F⊥ = κ·V_y`, κ = √2·p2/(R−Ra)² =
  38.88 (geometric, μ-independent). Caps are linear in μ: `V_y,crit = 1.266·μ`,
  `ψ̇_max = 7.618·μ`. ⇒ lateral-speed and yaw-rate amplitudes track the cap. Longitudinal
  V_x carries no F⊥ term (cap = τ_max·R/p1 = 4.55 m/s, μ-independent) ⇒ not circle-loaded.
- **Channel B — F∥ (traction = acceleration demand).** `F∥ = A∥⁺·b`, A∥⁺ geometric-constant.
  Holding utilization ⇒ `b` scales by k. Decomposing b (eqs 33–35):
  - order-1 acceleration (`M·v̇`) and order-1 velocity drag (`−110·V_y`, `−2.20·ψ̇`) → ∝ k
  - order-2 Coriolis/centripetal (`ψ̇²`, `ψ̇·V`) → ∝ k²

**Yaw rule = FOS/√k (≡ ×√k on yaw amplitudes).** The μ=0.5 TOMLs embed FOS s=0.8 on yaw
(`yaw_spin_max ≈ 3.1 ≈ 0.8·ψ̇_max(0.5)=3.047`). Setting `s_eff = s/√k` ⇒ `yaw_new = √k·yaw_ref`.
This pins the **order-2 ψ̇² centripetal fraction invariant across μ (a constant 12.8 % of the
circle)** — (√k)²=k matches the budget exactly — while the linear F⊥ yaw util moves as 1/√k.
No hard clamp needed. Consequence (code-verified):

| μ | k | √k | s_eff=0.8/√k | yaw scale | linear-yaw util | ψ̇² centripetal % |
|---|---|---|---|---|---|---|
| 0.3 | 0.6 | 0.775 | 1.033 | ×0.775 | 1.033 | 12.8 % |
| 0.5 | 1.0 | 1.000 | 0.800 | ×1.000 | 0.800 | 12.8 % |
| 0.8 | 1.6 | 1.265 | 0.632 | ×1.265 | 0.632 | 12.8 % |

For k<1 (μ=0.3) order-2 shrinks faster than budget (conservative); for k>1 (μ=0.8) the √k
rule absorbs it. μ=0.3 nudges linear-yaw util to ~1.03 (inside the Stribeck/stiction band
s_stat=1.1 and the original FOS margin) — the deliberate "slightly hotter μ=0.3" trade.

---

## 2. Final per-profile factors

| Profile | param(s) | factor | rationale |
|---|---|---|---|
| octagon | `vcru`, `lat_vamp` | **×k** | translational / accel (order-1) |
| coupled_vomega | `V_peak`, `V_const` | **×k** | translational |
| coupled_vomega | `Om_peak`, `Om_const` | **×√k** | yaw (FOS/√k) |
| spin_creep | `yaw_spin` | **×√k** | yaw (FOS/√k) |
| spin_creep | `v_creep` | **×1 (held)** | deliberate low-load creep |
| multisine 50/75 | `Vpk` | **×k** | translational (dominant load) |
| multisine 50/75 | `Ompk` | **×√k** | yaw (small) |
| ellipse | `worbit` | **×√k** | centripetal a_c=ρφ̇² → k |
| spiral_orbit | `ustar` | **×k** | iso-accel target |
| spiral_orbit | `Vc` | **×√k** | pins ψ̇·V Coriolis to k |
| spiral_orbit | `Om`, `Om0`, `Om1` | **×√k** | yaw (FOS/√k) |
| spiral_orbit | `R0`, `R1` | **×1 (held)** | radii; V follows Om·√k |
| long_circle | (all) | **×1 (held)** | low-load anchor (util 0.01–0.07) |

Verifier confirms: translational/accel cols ×{0.6,1.6}, yaw cols ×{0.7746,1.2649},
held cols unchanged, long_circle byte-identical, zeros preserved. (`verify_mu_grid.jl` → ALL PASS.)

---

## 3. Realized utilization vs μ=0.5 (post-solve `tracking_gate`)

**Composition:** 1776 each = main 1485 (χ=0.005) + quad 291 (97 near-cap × χ∈{0,0.002,0.008}).
Zero solver failures in either sweep.

**μ=0.3 row-preservation: 95.9 % flag-identical (1704/1776).** Every deviation is util-driven
(friction circle), zero tracking loss:
- octagon: 135 missed @μ=0.3 = 135 missed @μ=0.5 (pre-existing high-`vcru` lateral-leg combos,
  μ-invariant `vcru/V_y,crit`; not scaling-related).
- 46 tracked→missed, **all at χ=0**: 44 near-cap spin_creep + 2 coupled_vomega. The √k "hotter
  μ=0.3" effect (1/√k=1.29× + Stribeck, since v_creep held) tips the **undamped-spin (χ=0)**
  near-cap combos into gross slip (util_viol median 0.32). χ≥0.002 stay clean (util_viol ≤0.01) —
  the drilling friction `c_t=(8/3π)|ω_z|χ` absorbs the high-yaw pulses. **Decision: kept** as
  informative near-boundary data (2.5 % of μ=0.3; filterable by `track_flag`).

**μ=0.8 (conservative direction): 98.3 % flag-identical to μ=0.5 (1745/1776), ZERO new
tracked→missed.** The 31 changes are marginal⟷tracked shuffles around the 0.02 util boundary,
net improving (14 marginal→tracked + 3 other recoveries vs 14 slight nudges). util_budget drops
(e.g. spin_creep 0.97→0.31); the ×√k yaw rule makes μ=0.8 slightly *less* hot than μ=0.5.

**Flag totals:** μ=0.5 {tracked 1494, marginal 75, missed 207}; μ=0.3 {1444, 87, 245};
μ=0.8 {1503, 61, 212}. Evidence: `_tmp/gate_mu0p3_full.csv`, `_tmp/gate_mu0p8_full.csv`,
`tracking_report.csv` (μ=0.5).

---

## 4. Run provenance
- Solver FBDF, reltol 1e-9, dtmax 1e-3, saveat 2000 Hz, friction_case 1, lugre_adamov, seed 1234.
- Driver `Data_Generation_Julia.jl --script run_one_nbinclude.jl`, Arrow-only (`write_jld2=false`).
- 6-process single-thread pool, resume-safe (atomic tmp→rename), keep_awake held.
- Output: `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov` (μ disambiguated by filename).
