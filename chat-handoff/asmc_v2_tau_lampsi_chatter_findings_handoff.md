# ASMC v2 `tau_ceiling` / `lam_psi_max` box-widening — chatter findings

Companion to `asmc-v2-tuning-launch.md`. That brief predicted a top rail on
`tau_ceiling`/`lam_psi_max` "would also be a maximum-chatter result" and
listed `chatter alongside tracking` as an unchecked during-run diagnostic
(§6). This closes that item — plus decomposes it per axis — across two
subsequent box-widening rounds the launch brief didn't anticipate, and adds
a PID v2 comparison.

## 1. Background — why this needed checking

`runs_asmc_v2_epsfloor` (`tau_ceiling` hi=100, `lam_psi_max` hi=20, the
launch config) converged with both parameters pinned at their box edge on
nearly every seed. The box was widened twice more:

| Run | `tau_ceiling` hi | `lam_psi_max` hi | seeds |
|---|---|---|---|
| `runs_asmc_v2_epsfloor` | 100 | 20 | 1–5 |
| `runs_asmc_v2_epsfloor_wide` | 200 | 40 | 1–4 (seed 5 diverged to a separate bad basin, excluded) |
| `runs_asmc_v2_epsfloor_wide2` | 300 | 60 (warm-started from `_wide`'s winners) | 1–4 |

Both widenings kept railing at the new edge. **The objective cannot see any
cost that would stop this**: `stage_objective.jl`'s `lambda_chatter` defaults
to `0.0` and none of the launch bat files override it, and the `kmax_pen`/
`gamma_pen` regularizers are gated on `haskey(kw, :K_max_x)`/`:gamma_x`, keys
that don't exist in ASMC v2's decoded gains (`lam_x_max/lam_y_max/
lam_psi_max/tau_ceiling`) — both are structurally inert for v2 regardless of
their lambda values. So the score only ever rewards more authority; nothing
was penalizing chatter as the box was widened twice.

## 2. Method

`controller_metrics` (`tune_controller.jl`) already computes `chatter =
mean(sum_i |Δv_cmd_i|)` over the 4 wheel-voltage commands — a single number
mixing all 3 task-space axes via the O-config allocation. It answers "how
much chatter" but not "which axis." Per-tick task-space wrench (`bus.W_asmc`)
is not logged anywhere (the `ESTIMATOR_PROBE_LOG` probe only carries `t,
xhat, d_hat, v_cmd, u`; `log_run`'s `Wx_asmc` etc. columns are a single
final-tick snapshot broadcast across the DataFrame, not a time history).

Rather than instrument a new probe (would require duplicating `run_hybrid`/
`build_callbacks`, both marked never-edited), each tick's already-logged
`Δv_cmd` (4-vector) is least-squares projected through the **exact** O-config
allocation matrix (`hybrid_ctrl/mixer.jl:mix_and_allocate!`):

```
A = 0.25 * [[1,-1,-λ], [1,1,λ], [1,1,-λ], [1,-1,λ]],   λ = R/(l+h)
ΔW_proj = A \ Δv_cmd     (least-squares, 4 eq / 3 unknown)
```

**Caveat:** this is exact for the *allocation* step but `v_cmd` sits after a
torque saturation, a nonlinear motor-torque→voltage inverse, and a
rate-limiter — none of which are linear in `W`. The projection is therefore
exact in the unsaturated/non-rate-limited regime and degrades (axes blend)
exactly where chatter is highest. Treat absolute per-axis numbers as
approximate; the **relative x:y:psi split and the cross-run trend are load-
bearing** — the effect described below is large and consistent across every
single seed, which a projection artifact would not produce.

Script: `_tmp/check_chatter_per_axis.jl` (read-only re-evaluation of each
seed's converged `best_config.json` gains against the same `train12`/clean
objective the tuner used; no retuning, no writes to any `runs_*` directory).

## 3. Results

Aggregate chatter (already-existing metric, % of the 0.8 V/ms slew ceiling):

| Run | chatter | % of ceiling |
|---|---|---|
| `epsfloor` (baseline) | 0.120–0.129 | ~15–16% |
| `epsfloor_wide` | 0.142 | ~18% |
| `epsfloor_wide2` | 0.188–0.189 | ~24% |
| PID v2 CT | 0.223–0.284 (seed4: 0.419) | 28–35% (seed4: 52%) |
| PID v2 FB | 0.452–0.474 | 56–59% |

Per-axis decomposition (mean of `|ΔW_proj|` per tick, averaged over the 12
`train12` trajectories):

| Config | chatter_x | chatter_y | chatter_psi | psi share |
|---|---|---|---|---|
| ASMC epsfloor (baseline) | ~0.065 | ~0.067 | 0.46–0.53 | 78–80% |
| ASMC epsfloor_wide (`lam_psi_max`→40) | ~0.069 | ~0.070 | 0.65 | 82–83% |
| ASMC epsfloor_wide2 (`lam_psi_max`→60) | ~0.079 | ~0.073 | **0.99** | 87% |
| PID CT (seeds 1,2,3,5) | ~0.066 | ~0.067 | 1.3–1.8 | 91–93% |
| PID CT seed4 (outlier) | 0.066 | 0.064 | **2.91** | 96% |
| PID FB (all seeds) | ~0.054 | ~0.050 | 3.25–3.42 | 97% |

## 4. Findings

1. **Chatter is almost entirely a yaw-axis (psi) phenomenon.** `chatter_x`/
   `chatter_y` stay essentially flat (0.05–0.08) across every run and
   controller. All movement in the aggregate number — the rail-chasing as
   the box widened, the PID-vs-ASMC gap — lives in `chatter_psi`.
2. **`lam_psi_max` is causally implicated, specifically.** As
   `lam_psi_max` went 20→40→60, `chatter_x`/`chatter_y` rose only ~20%
   while `chatter_psi` nearly doubled (0.53→0.99, +87%). This directly
   confirms the launch brief's prediction that `lam_psi_max`'s real ceiling
   is `λ|e| ≤ v_max` / yaw chatter (§2), not the box bound.
3. **PID CT seed4's aggregate chatter anomaly (0.419 vs 0.22–0.28 for the
   other 4 seeds, flagged separately) is a yaw-specific outlier**, not a
   general chatter increase: `chatter_psi=2.91` vs 1.3–1.8 for the other
   seeds, at matching tracking/score. Same tracking quality, converged to a
   noisier point on the yaw axis for no benefit.
4. **PID's chatter disadvantage vs ASMC is entirely a yaw story.** PID FB's
   yaw chatter (3.25–3.42) is 3–6x ASMC's *worst-case* yaw chatter (0.99 at
   wide2); PID CT is 1.3–2x. PID's x/y chatter is actually slightly *lower*
   than ASMC's. PID's better tracking score (0.58 CT / 1.7 FB vs ASMC's
   0.81–1.18) is bought almost entirely with yaw-axis actuator switching —
   consistent with the launch brief's note that ASMC's tuned yaw λ (16.8–
   19.8) undershoots what PID effectively runs on that axis (§2, "Reaching
   vs sliding" table).
5. **Not yet saturating the physical limit.** Even at wide2, aggregate
   chatter is ~24% of the 0.8 V/ms slew ceiling — comfortably inside budget
   in absolute terms, despite the real and compounding relative growth.

## 5. Implication / open question

Nothing in the current objective would have stopped any of this — chatter
is unpriced (`lambda_chatter=0`) for both ASMC v2 and PID v2 runs to date, so
the rail-chasing on `lam_psi_max`/`tau_ceiling` will continue however far the
box is widened again. Two ways to close this, not yet decided:

- Turn `--lambda-chatter` on (nonzero) for a future ASMC v2 / PID v2 run so
  the optimizer prices chatter itself instead of needing a manual box-width
  decision each round.
- Or treat `lam_psi_max`/`tau_ceiling` as physically bounded (per launch
  brief §2's `λ|e| ≤ v_max` actuation cap) and stop searching them past that
  derived limit, the way `gamma` was pulled out of the search and specified
  directly for the same reason ("since `lambda_chatter = 0` by default the
  optimizer would never see that cost — which is exactly why this is
  specified rather than searched," launch brief §2).

Either way, `max(K)/K_max_sched` (launch brief §5/§6, still unchecked for
the wide/wide2 winners) should be verified before any of the wide2 configs
are accepted as a real result — a rail is also a maximum-authority result.
