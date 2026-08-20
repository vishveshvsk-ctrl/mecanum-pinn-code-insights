# ASMC v2 switching-authority limiting — revisit the algebra

Continues the chatter-pricing / controller-comparison session (PID v2 rate-match →
chatter normalisation fix → ASMC v2 chatter-priced retune). Project: hybrid controller
comparison (ASMC / PID-CT / PID-FB / MPC) on the mecanum PINN digital twin, IMECE 2026.

**Task:** re-derive the ASMC / PID-CT / PID-FB algebra and find a better way to bound
ASMC's switching authority. Three attempts at a `vcmd_limits`-style budget guard failed;
all are reverted. Working tree is at the as-designed baseline plus the fixes below.

## 1. Context

Authoritative files (all under `code_insights/`):

| file | what |
|---|---|
| `hybrid_ctrl_v2/controllers_v2.jl` | `ASMCControllerV2`, `asmc_wrench!`, `kmax_schedule`, `vcmd_limits`, `PIDControllerV2`, `decay_parameters`, `_smooth_bound_v2`, `capability_wrench` |
| `hybrid_ctrl_v2/tune_controller_v2.jl` | `ASMC_SPACE_V2`, `PID_SPACE_V2`, `MPC_SPACE_V2`, `build_controller_v2` |
| `hybrid_ctrl_v2/controller_tuning/stage_objective.jl` | `make_stage_objective`, `_assert_terms_reachable` |
| `tune_controller.jl` | `controller_metrics`, `TOL`, `V_MAX=24`, `LAMBDA_CE=0.05`, `CHATTER_REF=0.8`, `CHATTER_FC=25` |
| `hybrid_ctrl/config.jl` | `f_pid = 1000.0` (was 100) |

Objective: `score = tracking + 0.05*(ce/24) + lambda_chatter*(chatter/0.8) + 0.1*recovery`.
All runs used `--lambda-chatter 3.0`, `train12`, clean oracle, 5 seeds.

**The adaptation law** (`asmc_wrench!`), cubic now gated off by `use_cubic=false`:
```
dK = gamma_i * [ s*tanh(s/eps) * _smooth_bound_v2(K, K_max_sched)
                 - (cubic, DISABLED)
                 - sigma*(K - 0.95*K0)*exp(decay_k*(1 - s^2/(9*eps^2))) ]
_smooth_bound_v2 = 0.5 - 0.5*tanh(50*(K/K_max - 0.98))        range [0,1], knee at 98%
sigma = 1/(gamma_ref*tau_relax*exp(decay_k)) = 1/(250*2*e^0.25) = 0.0015576
gamma_x/y/psi = 170/380/280,  gamma_ref = 250,  decay_k = 0.25,  tau_relax = 2.0
K0 = [1.642, 1.602, 13.926],  K_max_base = capability_wrench(lim) = [6.591, 6.434, 55.913]
```

**`kmax_schedule`** (the thing to replace): `F_perp3 = kappa*(V_y - h*psidot)`;
`F_par_avail = sqrt((0.9*mu_N3)^2 - F_perp3^2)`; `F_par3_ff = 0.354*(Fx_ff+Fy_ff) - 0.918*Mz_ff`;
`F_avail = F_par_avail - |F_par3_ff|`; `frac = F_avail/mu_N3`;
`K_max_sched = capability_wrench(lim)*frac`, floored at `0.05*K_max_base = [0.330, 0.322, 2.796]`.

**`vcmd_limits`** (PID's analogue, the model): `a_corr = correction/lam_inner`;
`F_par3_0 = Fpar3(b(0,0,0))`; `dF = Fpar3(b(a_corr)) - F_par3_0`;
`budget = (0.9*mu_N3)^2 - F_perp3^2`; solve `|F_par3_0 + gamma*dF| = sqrt(budget)`;
`V_cmd = V_ff + gamma*correction`. **`V_ff` is never charged.**

## 2. Purpose

Find a limiter for ASMC's switching authority that is **not a persistent cap on K**.
Success = adaptation active on the stress trajectories (K lifts off `0.95*K0`) **and**
`spiral_orbit_stress` does not run away, at chatter ≤ ~20% of the 0.8 V/ms cap.

## 3. Decisions already made (do not reopen)

1. **`lambda_chatter = 3.0`, normaliser `CHATTER_REF = 0.8 V/ms`** (`= 4*dV_max/f_mix`).
   The old `chatter/V_MAX` divided a slew rate by a voltage — any O(1) lambda was ~30x too
   weak. 3.0 was calibrated from the ASMC box-widening exchange rate (bracket 0.91–9.5).
2. **`f_pid = 1000`** — rate-matched to `f_est` so the comparison isolates the control law.
   MPC stays at 100 Hz; solve time measured 5.6 ms mean at `Np=30` vs a 10 ms budget, so
   this is a declared method constraint, not a limit.
3. **Cubic barrier removed** (`use_cubic=false`). Measured harmful: with the lazy clamp
   `K_max_eff = max(K, K_max_sched)`, `(K/K_max_eff)^3 == 1` on 53–61% of ticks so the cubic
   runs at full strength; at `tau_ceiling` 0.05/0.5 it drives K below its floor and tracking
   degrades 10–12x (9.78 / 11.76 vs 0.99 at 300). **Linear sigma leak KEPT** — the growth
   term `s*tanh(s/eps) >= 0` is a one-way ratchet, so the leak is the only thing that can
   lower K and cannot be deleted.
4. **`gamma`, `tau_ceiling`, `tau_relax` are all measured inert** — 64x sweep → 0.5–1.2%;
   16x (300→4800) → score −0.7%; 100x (2→200) → chatter ≤3%. Do not re-search them.
   `K_max_base` overrides are a **no-op** under `use_scheduled_kmax=true` (`kmax_schedule`
   recomputes `capability_wrench(lim)` internally).
5. **Simply disabling the schedule is not the fix.** `use_scheduled_kmax=false` opens the
   gate on 11/12 and improves tracking 11–39% on stress cases, but `spiral_orbit_stress`
   blows up 6.7754 → 34.2197 as K winds up to 4.52x. The schedule prevents that.
6. **Three `wrench_limits` scopings failed and are reverted.** (a) charge `W_eq` →
   `gamma` p50 = 0.000 on 7/12, tracking 15–150x worse; (b) state-only baseline + direct
   wrench projection → `gamma` p50 = 1.000 everywhere, identical to ceiling-off on 7/12;
   (c) acceleration route `a_sw = W_sw ./ m_eff` through the same `_b` → `gam<1%` 0–46%,
   `spiral_orbit_stress` = 34.2197 **bit-identical to ceiling-off**, mean tracking 3.0624
   vs 0.7404 as-designed at +28% chatter. **Conclusion: the wind-up is not budget-driven,
   so no budget guard reaches it. The protective mechanism and the suppressive mechanism
   are the same persistent K cap.**

## 4. Measured state (the numbers to design against)

**Adaptation is pinned.** K sits at `0.95*K0` to within 0.4% on all 12 trajectories
(measured p50 `[1.5608, 1.5275, 13.2321]` vs `0.95*K0 = [1.5599, 1.5219, 13.2297]`).
`K_max_sched` bottoms at its 5% floor on 8/12, giving `K/K_sched ≈ 4.7` and a growth gate
closed **16.0–73.8%** of ticks. Only excursion: `spiral_orbit_stress` yaw, `maxK/0.95K0 = 2.461`.

**PID's guard barely binds:** `vcmd_limits` `gamma` median = 1.0000 on all 12; `guard_hit`
0% on 6/12, 10.4–41.1% on the rest.

**Converged comparison** (clean; all three at ~16% of the 0.8 V/ms cap):

| | score (5 seeds) | tracking clean | tracking noisy (seeds 101–105) | chatter noisy |
|---|---|---|---|---|
| PID CT | 1.1069 | 0.5356 | 1.0085 ± 0.193 | 89.6% |
| ASMC | 1.4148 | 0.9626 | **4.1359 ± 1.030** | 92.7% |
| PID FB | 2.4419 | 1.8803 | 2.4123 ± 0.215 | 95.1% |

Ordering **flips under noise**: clean CT < ASMC < FB; noisy CT < FB < ASMC. ASMC degrades
+330% vs +88%/+28% and has the highest realisation variance (24.9%). ASMC seed5 is a
different basin (`lam_y = 0.74` vs ~1.97, score 1.5089) — report separately, don't average.

## 5. Open / blocking

- **`tau_ceiling` is still a searched dimension in `ASMC_SPACE_V2`** but feeds only the
  now-disabled cubic — it is completely inert. Remove it (space → 3 `lam` dims) before any
  ASMC re-tune, or the next run wastes a quarter of its budget.
- **MPC is unresolved and blocks the 3-way comparison.** It diverges on 4 velocity-demanding
  trajectories (`ellipse_stress_tangent` 201.8, `coupled_vomega_stress` 102.3,
  `coupled_vomega_easy` 53.9, `spiral_orbit_stress` 43.3; other 8 fine at 0.5–1.3).
  Pre-existing — the closest pre-change config scores *worse* (207.0 vs 173.4).
  **`use_ltv=false` improves it (163.2 vs 173.4), which is backwards** — start there.
- **`tau_cl` frozen at PID's pre-retune mid-window** `(0.15, 0.11, 0.12)` while PID converged
  to `(0.164, 0.102, 0.0145)` FB / `(0.197, 0.114, 0.0199)` CT. It enters `bryson_Q_pose`
  **quadratically**, so MPC's yaw-rate weight is 36–68x heavier than PID-matched.
- **Hand-back:** whatever limiter replaces `kmax_schedule` must be validated on the
  12-trajectory table in §4 before ASMC is re-tuned and the comparison is rewritten.

## 6. Deliverables

1. Design note: the limiter, its algebra, and why it avoids both failure modes (adaptation
   suppression AND `spiral_orbit_stress` wind-up).
2. Implementation in `controllers_v2.jl` behind a default-`false` flag, forwarded in
   `build_controller_v2`'s `:asmc` key list.
3. Paired 12-trajectory validation vs the as-designed baseline — reuse
   `_tmp/asmc_wl_accel.jl` (logs `gam p50/p10/min`, `gam<1 %`, `maxK/K0`, tracking, and the
   as-designed column inline).
4. Decision on removing `tau_ceiling` from `ASMC_SPACE_V2`.

## 7. Conventions

- **Measure before believing.** Four parameters were assumed live and measured inert.
  Every claim in §4 has a `_tmp/` script behind it; add to that pattern, don't replace it.
- **Silent-inertness is the house failure mode** — 8 instances found this session (flags
  parsed then dropped, penalties gated on absent keys, docstrings describing spaces the
  code doesn't implement, `train12` having zero `step_hold` so `recovery_weight` is inert).
  `_assert_terms_reachable` in `stage_objective.jl` now errors on unreachable weights;
  extend it rather than adding new silent paths.
- **Verify with delta-tests**, not parse checks: build the objective at weight 0 vs
  non-zero on ONE trajectory and assert the score differs by exactly the expected term
  (`_tmp/verify_objective_terms.jl`, `_tmp/verify_lambda_chatter_wired.jl`). A test that
  can't fail (both sides zero) is worse than no test.
- **New behaviour goes behind a default-off flag**, so running tunes and archived configs
  are undisturbed. `tune_controller.jl` and `hybrid_ctrl/` v1 files are never edited.
- Temp/scratch work in `_tmp/`; never write data into `code_insights/`.
- Thread limit ≤ 8–10 Julia processes (16 cores, 40 GB); run `keep_awake.py` for long sweeps.
