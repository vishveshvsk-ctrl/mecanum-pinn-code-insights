# PID-CT heading weakness — measured, and the metric bug that overstated it

Continues the v3 metric / v3 trajectory-set session. This documents ONE finding
that outlived the bug it was discovered through, so the next session neither
loses it nor over-reads it.

## 1. The finding

PID-CT (computed torque, `feedforward=true`) tracks POSITION better than PID-FB
and HEADING substantially worse, on the same platform, same trajectories, same
objective. Measured on `train14_v3`, 14 trajectories, clean oracle, seed 4,
both at their own seed-1 mid-run optima (`_tmp/pid_v3_components.jl`):

```
tracking sub-terms, mean over 14 trajectories, tolerance-normalised
         fin_pos   max_pos   iae_pos  |  fin_hd    max_hd    iae_hd
FB        0.0069    0.0417    0.0403  |  0.0317    0.0183    0.0283
CT        0.0025    0.0189    0.0222  |  0.0076    0.1054    0.1836
```

CT wins all three position terms (0.0436 vs 0.0889 summed, ~2x better) and
loses heading 0.2966 vs 0.0783 (~3.8x worse). In physical units the heading gap
is `max_head` 0.0105 rad for CT against 0.0018 for FB (5.8x) and ~6.5x on the
sustained (time-normalised integral) heading error.

Per trajectory, CT is better on 10 of 14 and much worse on two:
`coupled_vomega_stress` (0.0716 -> 0.3243) and `spiral_orbit_stress`
(0.0599 -> 0.3519), plus moderately worse on `octagon_stress_hdg30`.

## 2. What this is NOT

**It is not why CT lost the abandoned run.** That was a metric bug, mine. The v3
tracking metric divided the three POSITION terms by the per-trajectory scaler
`k = max(radius_var(ref), TOL.pos_max)/TOL.pos_max` (1.0-29.0, mean ~16) and the
three HEADING terms by nothing. v2 held both channels absolute and equally
weighted, so this silently re-weighted position DOWN against heading by ~16x,
making heading 87% of CT's tracking term and inverting the FB/CT ranking.
Fixed by dividing the WHOLE tracking term by `k` (`stage_objective.jl`). After
the fix, CT is better again (0.00996 vs FB 0.01763) and consistent with v2
(0.11187 vs 0.22428). Heading share fell from 87%/74% to 27.2%/15.9%.

**It is not the trigonometric heading error.** `e_head = 2|sin(dpsi/2)|` replaced
`|atan2(sin,cos)|`; at these magnitudes (0.0018-0.0105 rad) the two agree to
~5e-6 relative. Inert here, and kept.

**It is not yet a converged result.** Both configs were mid-run (~70-87 evals of
a 250 cap) and were optimised against the IMBALANCED metric, so their gains were
being pulled toward heading. The weakness is a property of the runs measured, not
established for CT-at-its-optimum.

## 3. Why it might be real

CT's advantage is model feedforward; its heading channel is where the model is
weakest. Two candidate mechanisms, neither tested:

- **`imax_from_measured` anti-windup bound.** CT's per-wheel budget was set to
  `2.389*2 = 4.778` N*m against FB's `5.279` -- and that doubling is a
  hand-adjustment flagged in its own docstring as "EMPIRICAL ADJUSTMENT ...
  still needs re-checking once real tuning happens". It has never been rechecked.
  A too-tight integral bound on the yaw axis would show exactly as sustained
  heading error that the feedforward cannot cancel.
- **Feedforward model error on yaw.** The equivalent control carries `I_psi` and
  the yaw drag `4p1(l+h)^2/R^2 + 8p2h^2/(R-Ra)^2`; if the yaw model is the least
  accurate part, CT injects a systematic yaw wrench FB never commits to, and FB's
  pure integral action absorbs the residual instead.

The two trajectories where CT collapses (`coupled_vomega_stress`,
`spiral_orbit_stress`) are the two with the largest sustained yaw demand in the
set, which is consistent with both.

## 4. What to do next

1. **Re-measure after convergence.** The corrected-metric FB/CT runs relaunched
   2026-08-17 ~04:20 into `runs_pid_v3/`. Re-run `_tmp/pid_v3_components.jl`
   against their `best_config.json` -- it already reports the six sub-terms,
   per-trajectory tracking, and the score decomposition.
2. **If the heading gap survives**, test the `I_max` hypothesis first: it is a
   single constant, the recheck is overdue by its own docstring, and it is
   cheaper than instrumenting the feedforward.
3. **Report it either way.** "Computed torque buys position accuracy and costs
   heading accuracy on a mecanum platform" is a substantive result if it holds,
   and the per-axis decomposition is the evidence. Do not let it disappear into
   an aggregate score -- it is invisible in `tracking` alone, which is how it
   went unnoticed through the entire v2 campaign.

## 5. Conventions carried from the parent session

- **Measure before believing.** Every claim above has a `_tmp/` script behind it.
- **A guard that cannot fire is worse than no guard.** The v2/v3 lambda guard and
  the reachability assertions are asserted in BOTH directions in their tests.
- **Cross-validate a derivation against a known answer.** The lambda-recalibration
  procedure was run on the v2 metric first; it returned 0.052 against a known
  0.91-9.5 bracket, so its v3 output was NOT adopted and `LAMBDA_CHATTER_V3=0.36`
  is documented as balance-preserving rather than derived.
- **Evaluate references over `ref.T_total`, never a fixed window.** An audit that
  assumed 12.0 s picked a trajectory that is 98.5% of no-load across its real
  20.36 s. Durations here span 9.5-85.3 s.
- Abandoned runs are archived with a README stating why, not deleted:
  `hybrid_ctrl_v2/runs_pid_v3_ABANDONED_imbalanced_metric/`.
