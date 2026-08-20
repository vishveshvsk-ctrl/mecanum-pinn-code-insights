# v3 campaign — three-way comparison COMPLETE (MPC deferred); test-tier outstanding

Continues the ASMC switching-authority session (parent: `asmc_v2_switching_authority_limiting_handoff.md`).
Project: hybrid controller comparison (ASMC / PID-CT / PID-FB / MPC) on the mecanum PINN digital twin, IMECE 2026.

## 1. Context

Authoritative files, all under `code_insights/`:

| file | what changed |
|---|---|
| `hybrid_ctrl_v2/controller_tuning/trajsets.jl` | NEW tiers `:train14_v3` (14), `:test_v3` (8). `:train12`/`:test` untouched. |
| `hybrid_ctrl_v2/controller_tuning/stage_objective.jl` | `controller_metrics_v3`, `TOL_V3=(pos_iae=0.0249, head_iae=0.0128)`, `LAMBDA_CHATTER_V3=0.36` |
| `hybrid_ctrl_v2/controller_tuning/run_stage.jl` | `--metric v2\|v3` forwarded to ASMC/PID/MPC v2 paths |
| `hybrid_ctrl_v2/controllers_v2.jl` | `use_demand_k`, `rho_auth`, `kmax_contact_b`, `enforce_k_floor`, `kmax_sched_floor` (all default-false/inert) |
| `hybrid_ctrl_v2/tune_controller_v2.jl` | `ASMC_SPACE_V2` dim 4: `tau_ceiling` → `rho_auth`, log[0.762, 27] |
| `hybrid_ctrl_v2/run_stage_asmc_v3.bat`, `run_stage_pid_v3.bat` | launchers |

Objective actually used: `score = tracking_v3 + 0.05*(ce/24) + 0.36*(chatter/0.8)`, `train14_v3`,
p1-cap 250 / p2-cap 60, eps floors 0.02/0.08, `--lam-psi-hi 60`, cold start.

**Converged clean results (all 5 seeds each, `stop_reason=plateau`):**

```
PID-CT   0.14319 0.14323 0.14328 0.14341 0.14341     best seed5, lam_inner=(0.09815, 0.11399, 0.03705)
PID-FB   0.14901 0.14931 0.14937 0.14938 0.14943     best seed3, lam_inner=(0.08092, 0.03736, 0.02404)
ASMC     0.15233 0.15235 0.15239 0.15305 0.15314     best seed2
```

Ordering **PID-CT < PID-FB < ASMC**, no family overlap, seed spread 0.15–0.5%.

**Score decomposition, best seeds, 14 trajectories, clean** (`hybrid_ctrl_v2/eval_v3_clean_components.jl`, data in `results_v3/`):

```
           tracking   ce-term  chat-term    SCORE
ASMC        0.01845   0.07797    0.05676   0.15318
PID-CT      0.01133   0.07774    0.05432   0.14340
PID-FB      0.01997   0.07768    0.05188   0.14953
ASMC-vs-CT gap +0.00978 = track +0.00711 (72.7%) + chat +0.00244 (25.0%) + ce +0.00023
```

**PID noisy, seeds 101–105** (`hybrid_ctrl_v2/eval_v3_noisy_pid.jl`, data in `results_v3/`):

```
          clean     noisy    realis.var   chatter clean->noisy (cap 0.8)
PID-CT   0.14340   0.50372      7.1%      0.12071 -> 0.71476  (89%)
PID-FB   0.14953   0.53135      6.2%      0.11528 -> 0.75851  (95%)
noisy score composition: chatter 64%, tracking 21%, ce 15%
```

## 2. Purpose

Finish the v3 campaign so a three-way comparison can be quoted on ONE objective.
Success = ASMC noisy + MPC v3 (clean & noisy) converged on `train14_v3`, and all
four evaluated on `:test_v3`, with every number produced under `--metric v3
--lambda-chatter 0.36`.

## 3. Decisions already made — do not reopen

1. **`train14_v3` replaces `train12`.** `train12` held two trajectories the hardware cannot fly:
   `spiral_orbit_stress` at 85.4% of no-load wheel speed and `ellipse_stress_tangent` at 76.2%.
   Spiral alone carried 72–81% of the v2 tracking term.
2. **Feasibility = per-wheel speed vs no-load `V_max/(Kb*G)=27.73 rad/s`, NOT the friction circle.**
   Measured: ≤76.2% flyable, ≥85.4% catastrophic for every controller. `octagon_stress`/`hdg30`
   exceed the friction circle (util 1.092/1.111) and are among the best-tracked — the circle is soft, the
   back-EMF wall is hard. The old `Vy<=0.6` screen sees neither case (both have `Vy=0.000`).
3. **Evaluate every reference over `ref.T_total`, never a fixed window.** Durations span 9.5–85.3 s.
   An audit assuming 12.0 s picked a spiral combo that is 98.5% of no-load over its real 20.36 s.
4. **`rho_auth` box log[0.762, 27] is physically anchored.** Lower = `tanh(1)` (at one boundary layer
   the adaptation claims ≤half its authority); upper = `9×3ε`, 3ε being the excitation scale the
   as-designed sigma leak itself uses. Do not widen without re-deriving.
5. **`LAMBDA_CHATTER_V3 = 0.36` is balance-preserving, NOT a derived exchange rate.** The method that
   produced 3.0 fails its own v2 control here (returns 0.052 vs a known 0.91–9.5 bracket) because
   `lam_psi_max` is no longer a v2 trade-off lever. 0.36 reproduces v2's 29.9% tracking share.
   `stage_objective.jl` REFUSES lambda > 1.44 under `--metric v3`.
6. **The v3 scaler `k_traj` divides the WHOLE tracking term, not position alone.** Scaling position
   only re-weighted heading up ~16× and inverted the PID FB/CT ranking; cost an 8.4 h run
   (archived, `hybrid_ctrl_v2/archived/runs_pid_v3_ABANDONED_imbalanced_metric/`). Not the trigonometric heading error, which
   is inert at these magnitudes (~5e-6 relative).
7. **Do not quote % degradation.** CT reads +815.6% tracking degradation vs FB's +460.4% only because
   CT's clean tracking is better; CT's ABSOLUTE noisy tracking is better (0.10377 vs 0.11194).
   Report absolute noisy performance.

## 4. Open / blocking

**STATUS 2026-08-19 — the three-way comparison is COMPLETE. Results: `hybrid_ctrl_v2/RESULTS_v3.md`.**
Read that file first; it supersedes the predictions below.

- ~~**ASMC noisy never run.**~~ DONE, and the prediction was half right. Noisy tuning ran for ALL THREE
  controllers at 4 seeds each (not 5 — user decision), plus a noisy EVAL of every clean-tuned config on
  sensor seeds 101–105. Chatter/tracking do swap dominance clean→noisy (25.0%→62.3% of the ASMC-vs-CT
  gap), so the chatter half holds — but the mechanism is NOT an ASMC-specific penalty: under noise all
  three saturate near the 0.8 cap and ASMC's chatter is LOWER than FB's. CT is best because CT alone
  stays off the cap. Tracking still contributes 37.7%, so "unrelated to its tracking concentration"
  was too strong.
- **MPC v3 — DEFERRED out of this iteration** (user decision, 2026-08-19). The `use_ltv` anomaly stays
  unresolved and no `run_stage_mpc_v3.bat` exists. It no longer blocks anything: the ASMC/PID-CT/PID-FB
  comparison is complete and quotable without it.
- **`:test_v3` has not been evaluated for any v3 config.** Generalization is unmeasured on the v3 tiers.
  This is now the largest open item — every number in `RESULTS_v3.md` is on the TRAINING tier.
- **`chat-handoff/pid_ct_heading_weakness_handoff.md` §1 needs revising.** It reads as though PID-CT has a
  broad heading deficit. Measured: CT's HEAD-sum is 0.00445 over 14 trajectories but 0.00085 over the 12
  excluding `coupled_vomega_stress`/`spiral_orbit_stress` — best of the three. The weakness is
  CONCENTRATED on high sustained-yaw trajectories. This fits its `imax_from_measured` hypothesis better.
- **`coupled_vomega_stress` carries 66% of ASMC's tracking gap.** Excluding it flips ASMC ahead of PID-FB
  (ASMC +3.61% vs CT, FB +4.88%). It is FEASIBLE (54.4% no-load, util 0.941) so exclusion is NOT
  justified — the slices are diagnostic only. Root cause measured earlier: pure-lateral trajectory where
  ASMC's bounded switching authority loses to PID's integral action (ASMC wins x 16×, ψ 11×, loses y 5×).

## 5. Deliverables

1. ~~ASMC noisy configs~~ **DONE** — and for all three controllers, 4 seeds each, all `plateau`:
   `runs_asmc_v3/seed{1..4}/asmc_v2_noisy/`, `runs_pid_v3/seed{1..4}/pid_v2_{ct,fb}_noisy/`.
   NOTE deliverable 1 as originally written would have produced an ASMC-only noisy stage quoted
   against clean-tuned PID — the asymmetry `run_stage_asmc_v3.bat`'s own header warns about.
   Noisy tuning is all-or-nothing across the controllers being compared.
2. ~~MPC v3~~ **DEFERRED** out of this iteration (see §4).
3. A `:test_v3` evaluation of the converged configs, extending `_tmp/test_set_scores.jl`. **OPEN** —
   now the largest gap; everything measured so far is on the training tier.
4. Revised `pid_ct_heading_weakness_handoff.md` §1 per the concentration finding. **OPEN**, but the
   measurement now exists: `RESULTS_v3.md` §6 — CT's yaw integral clamp fires 19.9% / 17.0% of ticks
   on `coupled_vomega_stress` / `spiral_orbit_stress` and 0.0% on the other twelve, matching where the
   heading deficit lives. Mechanism #1 confirmed.
5. ~~A single comparison table~~ **DONE** — `RESULTS_v3.md` §4, six configs (3 clean-tuned + 3
   noisy-tuned) on sensor seeds 101–105, one objective, absolute values. Test-tier column pending (3).

6. **ESKF closed-loop eval — DONE** (added beyond the original list). `eval_v3_eskf.jl` +
   `run_eval_v3_eskf.bat`, 6 configs × 14 trajectories × seeds 101–105 = 420 runs, zero failures,
   data in `runs_eskf_v3_train14/`. `RESULTS_v3.md` §7 also documents the ESKF's OWN tuning
   objective, 7-dim search space, converged gains and term decomposition.

**Headline:** PID-CT wins in ALL THREE feedback regimes — clean oracle (0.14340), noisy oracle
(0.50372), and tuned-ESKF (0.34127, paired 69/70 and 70/70). Noisy tuning helps ASMC under the
noisy oracle (−0.01330, 5/5), helps PID-CT less (−0.00572, 5/5), FAILS for PID-FB (+0.00064,
mixed), and **does not transfer to ESKF feedback at all** (ASMC −0.00103 at 58/70; both PIDs
actively worse) — so the CLEAN-tuned configs are the ones to deploy. ASMC vs PID-FB is a tie
under clean-oracle AND under ESKF; ASMC's 5/5 win exists only under noisy-tuned oracle feedback.

**TRAP, cost me a wrong conclusion for one draft:** `OracleEstimator(:noisy)` injects UNFILTERED
per-tick sensor noise straight into `xhat` (sig_vel 0.010 m/s, sig_pos 0.020 m) with NO estimator.
The ESKF (0.94 mm/s, 3.25 mm RMSE) beats it by construction. ESKF-minus-noisy-oracle is therefore
NOT "the cost of state estimation" — that cost is ESKF vs CLEAN oracle, ~+0.20, of which the
estimator recovers ~44% of the clean→unfiltered gap, uniformly across all six configs.

Scripts promoted from `_tmp/` into `hybrid_ctrl_v2/eval_v3_*.jl` + `diag_v3_*.jl`, data in
`hybrid_ctrl_v2/results_v3/`.

## 6. Conventions

- **Measure before believing.** Every claim above has a `_tmp/` script behind it; add to that pattern.
- **A guard that cannot fire is worse than none.** Assert both directions (fires when it should, quiet
  when it shouldn't) — see the v2/v3 lambda guard tests.
- **Cross-validate a derivation against a known answer** before adopting its output (see decision 5).
- **Serialise compute BEFORE formatting.** Two 168-run sweeps were lost to a `@printf` arg-count typo.
  the promoted `hybrid_ctrl_v2/eval_v3_*.jl` all support `REPORT_ONLY=1` to dry-run formatting on dummy data.
- New behaviour goes behind a **default-off flag**; `tune_controller.jl` and `hybrid_ctrl/` v1 are never
  edited. Abandoned runs are archived with a README, not deleted.
- Scratch in `_tmp/`; never write data into `code_insights/`. Long sweeps need `keep_awake.py`.
- ≤8–10 concurrent Julia processes; the limiter is the Windows COMMIT limit (55.4 GB), ~1.8 GB/process.
  10-wide held at ~6–7 GB headroom.
