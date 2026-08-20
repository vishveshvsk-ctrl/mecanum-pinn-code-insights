# v3 campaign results — ASMC / PID-CT / PID-FB

Status as of **2026-08-19**. MPC is **deferred out of this iteration** (user decision);
everything below is a complete three-way comparison on one objective, under THREE feedback
regimes: clean oracle, noisy oracle (unfiltered, no estimator), and the tuned ESKF (§7).

**Headline: PID-CT wins in all three regimes AND on held-out trajectories** (§9, paired
35/35). ASMC and PID-FB are equivalent -- the apparent FB-over-ASMC edge on the training
tier is carried by one trajectory. Noisy tuning helps ASMC, helps PID-CT less, fails for
PID-FB, and does not transfer to ESKF feedback -- so the CLEAN-tuned configs are the ones
to deploy.

Every number here was produced under the **same** settings — mixing v2 and v3 scores in
one table is invalid, and so is mixing tuning-stage scores with eval scores (see
[Reading the numbers](#reading-the-numbers)):

```
objective   score = tracking_v3 + 0.05*(ce/V_MAX) + 0.36*(chatter/CHATTER_REF)
trajset     train14_v3   (14 trajectories, all <= 67.7% of no-load wheel speed)
flags       --metric v3 --lambda-chatter 0.36 --p1-cap 250 --p2-cap 60
            eps floors 0.02 / 0.08, --lam-psi-hi 60, cold start for clean stages
```

---

## 1. The objective — what is actually being minimised

Defined in `controller_tuning/stage_objective.jl` (`controller_metrics_v3`, `TOL_V3`,
`LAMBDA_CHATTER_V3`); `ce` and `chatter` are reused verbatim from
`tune_controller.jl`'s `controller_metrics`, so **v3 changes the tracking term only**.

```
score = tracking_v3  +  LAMBDA_CE*(ce/V_MAX)  +  LAMBDA_CHATTER_V3*(chatter/CHATTER_REF)
                        LAMBDA_CE = 0.05                LAMBDA_CHATTER_V3 = 0.36
                        V_MAX     = 24.0 V              CHATTER_REF        = 0.8 V/ms
```

Every term is a **ratio to a tolerance**, so a term of 1.0 means "exactly at target" and
< 1 is better. The score is therefore dimensionless and comparable across trajectories.

### 1.1 `tracking_v3` — six terms over two channels

```
                fp/pos_final + mp/pos_max + ip/pos_iae        <- position: terminal, peak, typical
tracking_v3 =  ------------------------------------------------------------------------
                + fh/head_final + mh/head_max + ih/head_iae   <- heading:  terminal, peak, typical
                                    6 * k_traj

  fp, fh = error at t = T_total          (terminal)
  mp, mh = max over the run              (peak)
  ip, ih = mean(|e|) over probe ticks    (typical)  = (1/T)*int|e|dt, uniform sampling
  pe = sqrt((x - x_ref)^2 + (y - y_ref)^2)      he = 2|sin(dpsi/2)|
```

| tolerance | value | source |
|---|---|---|
| `pos_final` | 1e-2 m | v2 `TOL`, design target |
| `pos_max` | 1e-1 m | v2 `TOL`, design target |
| `head_final` | 1e-2 rad | v2 `TOL`, design target |
| `head_max` | 1e-1 rad | v2 `TOL`, design target |
| `pos_iae` | **0.0249 m** | **measured**: achieved iae/max = 0.249 over 7 converged configs × 14 trajectories |
| `head_iae` | **0.0128 rad** | **measured**: achieved iae/max = 0.128, same sample |

The two new tolerances were *calibrated, not chosen*, so the integral term enters at the
same strength as the max term it sits beside. The precedent for insisting on this: the
worst objective bug in this stack was `chatter` normalised by `V_MAX = 24` — a voltage
dividing a slew rate — which left the term ~30× too weak and silently unpriced for months.

### 1.2 The three changes from v2, and why

**(1) A time-normalised integral term.** The v2 pose metric was
`(final_pos/1cm + max_pos/10cm + final_head/0.01 + max_head/0.1)/4` — **four samples out
of 12k–60k**: the value at `t = T` and at the argmax, per channel. A controller that
oscillated violently mid-run but ended inside 1 cm and never breached 10 cm scored
identically to one that tracked cleanly throughout. `iae = mean(|e|)` is the time-
*normalised* integral, so trajectories spanning 9.5–85.3 s stay comparable. Mean-absolute
rather than RMS deliberately: `max` already carries the peak, so mean-abs contributes the
distinct "sustained/typical" dimension instead of duplicating it. The three terms read
**terminal / peak / typical**.

**(2) A per-trajectory scaler `k_traj`, dividing the WHOLE term.**

```
radius_var(ref) = max || p_ref(t) - centroid ||        (the path's own extent)
k_traj          = max(radius_var, pos_max) / pos_max   >= 1
```

`radius_var` was chosen over alternatives because it is **rotation-invariant**
(`octagon_stress` and `octagon_stress_hdg30` are the same path rotated 15°; a bounding-box
measure would score them differently), it is a single **isotropic** scalar matching the
Euclidean position error, and it measures **extent** rather than distance travelled (path
length would rate a tight circle repeated many times as "large"). The `pos_max` floor is
load-bearing: `spin_creep_easy` never translates (radius exactly 0.000), so an unfloored
scaler divides by zero; floored, it falls back to the absolute tolerance, which is the
right treatment for a spin-in-place reference.

**`k_traj` divides all six terms, not the position half.** Dividing position alone — the
first version, since reverted — silently re-weighted position *down* against heading by
`k_traj` (range 1–29, mean ~16), because v2 held both channels absolute and equally
weighted. That re-weighting on its own **inverted the PID FB/CT ranking** and cost an 8.4 h
run, archived under `archived/runs_pid_v3_ABANDONED_imbalanced_metric/`. Scaling the whole term
preserves v2's position:heading balance exactly while still making tracking
trajectory-relative.

**(3) A trigonometric heading error.** `e_head = ||u(psi) − u(psi_ref)|| = 2|sin(dpsi/2)|`,
the chord between heading unit vectors. Monotone on [0, π], **smooth at dpsi = ±π** where
`|atan2(sin,cos)|` has a branch kink, bounded by 2, and **linear for small dpsi** so the
existing heading tolerances carry over unchanged — at the measured p90 of 0.0057 rad the
two forms differ by 1.4e-6 relative, diverging only above ~0.5 rad where compressing a
gross error is the desired behaviour. `|sin(dpsi)|` was rejected as non-monotone (it cannot
tell 5° from 175°) and `1−cos(dpsi)` as quadratic, which would change the units and
invalidate the tolerances.

**No trajectory scaler on heading.** `dpsi_var` is *cumulative rotation* (rate × duration),
not an extent: `spin_creep_stress_yaw` accumulates 13 rad, so scaling by it would discount
heading error ~130× on exactly the trajectories that exercise heading control — leaving the
channel owned by the four octagons and `ellipse_stress_crab`, the entries with **zero**
commanded rotation. A chord error is bounded by 2 regardless, so it has a fixed natural
scale and wants an absolute tolerance.

### 1.3 The `ce` and `chatter` terms

```
ce      = mean over ticks of sum(|v_cmd|) over the 4 wheels      -- voltage MAGNITUDE  [V]
chatter = mean per-tick total variation of the wheel-voltage command, summed over 4 wheels
                                                                 -- slew RATE       [V/ms]
```

`CHATTER_REF = 0.8 V/ms = 4 × dV_max/f_mix = 4 × 200/1000` — the per-tick wheel-voltage
slew clamp in `mixer.jl` summed over four wheels. It is a **hard cap**, not a soft target,
so `chatter/CHATTER_REF = 1.0` means "saturating the actuator's slew capability". This is
why the noisy configs sitting at 0.70–0.76 are described as *near the cap*.

**Why not normalise chatter by `V_MAX`:** chatter is a rate and `ce` is a magnitude, so
`chatter/24` is dimensionally meaningless and numerically crushing — it lands at
0.005–0.008 for every config measured, meaning `--lambda-chatter 1` would move a score of
~1.0 by 0.005. That, not merely the old 0.0 default, is why the chatter term never bit:
any O(1) value anyone would naturally try was ~100× too small.

### 1.4 Why `LAMBDA_CHATTER_V3 = 0.36`, and the guard

**v2's 3.0 must not be carried over.** v3 tracking is 5–10× smaller (the `k_traj`
division), so at 3.0 the chatter term becomes ~90% of the score and the optimiser
effectively stops pricing tracking — the same class of silent re-weighting as the
`chatter/V_MAX` bug.

**0.36 is balance-preserving, NOT a re-derived exchange rate**, and the distinction is
load-bearing. At the converged operating point the v2 objective at λ=3.0 puts tracking at
**29.9%** of (tracking + chatter); 0.36 reproduces that same 29.9% split under v3. It
preserves a weighting the project already accepted rather than claiming to derive a new one.

The derivation method that produced 3.0 **fails its own control** here: re-run on the v2
metric it returns 0.052 against a known bracket of 0.91–9.5, because under the corrected
schedule and the demand-k law `lam_psi_max` is no longer a v2 trade-off lever at all (over
lam_psi 12→60, v2 tracking is flat to 2% and drifts the *wrong* way, 0.2199 → 0.2233, while
chatter rises 0.126 → 0.202). There is no v2 trade to price. Under v3 the trade is real
(tracking 0.0361 → 0.0229, −37%) — further evidence that final+max was blind to sustained
error, and a retroactive explanation for three box-widening rounds that kept railing on a
lever the objective could not price. A properly derived rate needs a lever that trades
under *both* metrics; `eps_floor_psi` is the classical candidate and is not yet measured.

**`stage_objective.jl` REFUSES `lambda_chatter > 1.44` under `--metric v3`** rather than let
a carried-over v2 value pass silently. `lambda_v3_verify.jl` asserts the guard fires *and*
stays quiet when it should — a guard that cannot fire is worse than none.

---

## 2. Clean-tuned (5 seeds each, all `stop_reason=plateau`)

```
PID-CT   0.14319  0.14323  0.14328  0.14341  0.14341     best seed5
PID-FB   0.14901  0.14931  0.14937  0.14938  0.14943     best seed3
ASMC     0.15233  0.15235  0.15239  0.15305  0.15314     best seed2
```

Ordering **PID-CT < PID-FB < ASMC**, no family overlap, seed spread 0.15–0.5%.

Score decomposition at the best seeds (`eval_v3_clean_components.jl`):

```
           tracking   ce-term  chat-term    SCORE
ASMC        0.01845   0.07797    0.05676   0.15318
PID-CT      0.01133   0.07774    0.05432   0.14340
PID-FB      0.01997   0.07768    0.05188   0.14953

ASMC-vs-CT gap +0.00978 = track +0.00711 (72.7%) + chat +0.00244 (25.0%) + ce +0.00023
```

## 3. Noisy-tuned stages (4 seeds each, all `plateau`)

```
         seed1     seed2     seed3     seed4      mean      std
ASMC    0.57065   0.53392   0.55977   0.49273   0.53927   0.03464
PID-CT  0.55484   0.50927   0.54155   0.47081   0.51912   0.03746
PID-FB  0.58221   0.53847   0.56815   0.50723   0.54902   0.03329
```

**The seed ordering is identical for all three controllers** (seed4 < seed2 < seed3 <
seed1). The tuning seed sets the *noise realisation*, not merely the optimizer start, so
the ~16% seed spread is shared realisation variance and the marginal means understate the
signal. Paired across the common seeds:

```
CT - ASMC   -0.01581  -0.02465  -0.01822  -0.02192    mean -0.02015    4/4
FB - ASMC   +0.01156  +0.00455  +0.00838  +0.01450    mean +0.00975    4/4
```

Noisy-tuned ordering **PID-CT < ASMC < PID-FB**, consistent on every seed.

## 4. The comparison table — all six configs on sensor seeds 101–105

`ct` = clean-tuned, `nt` = noisy-tuned. Absolute values only; **do not quote %
degradation** (a controller with better clean tracking reads a worse % for the same
absolute result).

```
                 clean sc   noisy sc      track    chatter  realis.var
PID-CT ct(s5)     0.14340    0.50372    0.10377    0.71475       7.1%
PID-CT nt(s4)     0.14422    0.49799    0.10331    0.70347       7.1%
ASMC   ct(s2)     0.15318    0.53168    0.11433    0.75346       6.5%
ASMC   nt(s4)     0.15697    0.51838    0.11176    0.72952       6.6%
PID-FB ct(s3)     0.14953    0.53135    0.11194    0.75815       6.2%
PID-FB nt(s4)     0.15492    0.53199    0.11153    0.76067       6.4%
```

**Did noisy tuning help?** Paired by realisation (negative = noisy-tuned better):

```
ASMC     -0.01160  -0.01794  -0.00852  -0.01126  -0.01716   mean -0.01330   5/5 better
PID-CT   -0.00597  -0.00396  -0.00431  -0.00656  -0.00781   mean -0.00572   5/5 better
PID-FB   +0.00040  -0.00053  +0.00107  -0.00063  +0.00288   mean +0.00064   MIXED
```

Clean-side cost: ASMC +0.00379, PID-CT +0.00083, PID-FB +0.00538.

## 5. Findings

**PID-CT wins at every stage** — clean (0.14340), clean-tuned under noise (0.50372), and
noisy-tuned under noise (0.49799). Noisy tuning narrows ASMC's deficit to CT from
+0.02796 to +0.02039 (a 27% reduction) without changing the ordering.

**Noisy tuning fails for PID-FB.** Three realisations worse, two better, mean slightly
negative, and its chatter *rose* (0.75815 → 0.76067) under an objective that is ~64%
chatter — while it paid the largest clean-side cost. This is structural, not a tuning
accident: FB has no feedforward, so its integrator carries the whole model-compensation
load and sits at its anti-windup bound on 20.5% of ticks (§6). Its chatter is a
consequence of that structure, which `lam_inner` cannot reach; at 0.758 against a 0.8
reference there was nowhere left to go. ASMC gains 2.3× what CT does (chatter −0.02394 vs
−0.01128) precisely because it carried the most reducible chatter.

**The ASMC / PID-FB ranking flips, but state it carefully.** Paired:

```
ASMC vs FB   clean-tuned   -0.00173 +0.00381 -0.00090 -0.00394 +0.00439   mean +0.00032   MIXED
             noisy-tuned   -0.01373 -0.01360 -0.01049 -0.01457 -0.01565   mean -0.01361   ASMC 5/5
```

Clean-tuned, ASMC and FB are a genuine tie — a 3/2 split with a mean difference of 0.0003,
inside realisation noise. The defensible claim is **"indistinguishable when tuned clean,
ASMC clearly better when both are tuned for noise"**, not "ASMC overtakes FB".

**And that advantage does not survive real state estimation**: under ESKF feedback the two
are tied again (33/70, §7.4). The ASMC edge is specific to noisy-tuned *oracle* feedback,
where the injected noise is unfiltered and 10× larger than the ESKF's residual.

**All realisation variance lives in tracking.** Across 101–105 the chatter column moves
<0.3% while tracking swings 0.068–0.154. Noise changes how well a controller tracks, not
how much it chatters.

## 6. `I_max` is derived, and its yaw bound explains CT's heading weakness

`I_max` is **not** a tuning dimension in PID v2 (`PID_SPACE_V2` is `lam_inner_{x,y,psi}`).
`imax_from_measured` sets `I_max = wrench_budget ./ Ki` with `Ki = d_eff ./ lam_inner`, so
`Ki*I_max = wrench_budget` is invariant of `lam_inner` — searching it would mean searching
the measured wrench budget, not an anti-windup knob. (PID **v1** `pid_cascade.jl` does
search it, log[5,200]; that is a different path and does not carry over.)

Measured tick-fraction at the clamp (`diag_v3_pid_imax_binding.jl`, converged v3 configs):

```
                    sat_x   sat_y  sat_psi
PID-CT (mean)        0.4%    4.5%    2.6%
PID-FB (mean)        0.5%   20.5%    5.1%
```

The budget is sized at the **p95** of integral demand, so ~5% saturation is the design
point, not a fault. FB's higher rate is structural (no feedforward — its integral *is* the
model path) and is **not** evidence of a mis-sized bound; the two variants' budgets differ
because the control laws differ.

What *is* anomalous: **CT's yaw clamp fires on 19.9% of ticks on `coupled_vomega_stress`
and 17.0% on `spiral_orbit_stress`, and exactly 0.0% on the other twelve trajectories** —
against a design intent of residual-only integral action. Those are the same two
sustained-yaw trajectories carrying CT's entire heading deficit (HEAD-sum 0.00445 over 14
trajectories, 0.00085 over the 12 excluding them). This confirms candidate mechanism #1 in
`chat-handoff/pid_ct_heading_weakness_handoff.md` §3.

Note the clamp *binds*; that it *harms* is not established — a saturating integrator may be
doing its job. Establishing harm needs a budget-sensitivity run.

## 7. The frozen ESKF — its own tuning metric and converged result

Everything in §2–§6 is **oracle** state feedback (`run_controller_v2` hardcodes
`estimator = :oracle`; its `:clean`/`:noisy` argument selects the *sensor realisation*,
not the state source). The deployable counterpart routes state feedback through the tuned
ESKF. This section documents the estimator itself; §7.4 is the closed-loop result.

Estimator: **`ESKFEstimatorV3`** (13-dim, yaw-acceleration state), config
`runs_estimator_v4_mu0p5_train12/seed4/best_config.json`, **frozen**. Estimator-first
methodology — tune and freeze the observer, *then* compare control laws, so a control-law
difference cannot be an observer difference in disguise.

### 7.1 The estimator objective (`estimator_tuning/objective_v2.jl`)

A **replay** objective: the estimator is scored against ground truth on logged
trajectories, independent of any controller.

```
score = vel_rmse/v_tol + rate_rmse/rate_tol + lam_slip*inslip_vel_rmse/v_tol
        + pos_rmse/pos_tol + heading_rmse/heading_tol + lam_smooth*smoothness

  v_tol    = 1e-3 m/s      pos_tol     = 1e-2 m       lam_slip   = 1.0
  rate_tol = 1e-2 rad/s    heading_tol = 1e-2 rad     lam_smooth = 0.05
```

Same shape as the controller objective: **absolute physical-unit RMS errors normalised by
target tolerances**, so a term near 1.0 means "at target" — deliberately *not* normalised
by signal RMS, which would make an easy trajectory and a hard one look alike.

Two terms deserve note. `inslip_vel_rmse` is the velocity RMSE restricted to ticks where
slip exceeds its median, entered at full weight (`lam_slip = 1.0`) **in addition to** the
unrestricted `vel_rmse` — so velocity error during slip is priced twice, which is the
whole point of an estimator for a Mecanum platform whose model mismatch *is* slip.
`smoothness` is `mean((d|v_hat|/dt)^2)`, a regulariser against an estimator that achieves
low RMSE by chattering, priced weakly at 0.05.

### 7.2 Search space (`estimator_tuning/param_space_v4.jl`) — 7 dims, all log-scaled

```
alpha_acc        [1e-2, 1e2]     accelerometer update rate
alpha_yaw        [1e-2, 1e2]     yaw-rate update rate
grip_slip_scale  [1e-5, 1e0]     grip/slip transition scale
r_boost          [1.0,  100.0]   measurement-noise boost
pose_Qn_heading  [1e-8, 1e-3]    pose-update heading process noise
slip_R_inflate   [1.0,  100.0]   R inflation while slipping
q_alpha          [1e-8, 1e-1]    yaw-accel random-walk variance  [(rad/s^2)^2/tick]
```

Pinned, not searched (`PINNED_V3` + v4's `P0_alpha`): `P0_vel` 1e-4, `P0_yaw` 1e-3,
`P0_heading` 0.5, `P0_bias_acc` 4e-4, `P0_bias_gyro` 9e-6, `P0_slip` 2e-4, `P0_pos` 0.25,
`pose_Qn_pos` 1e-7, `P0_alpha` 0.25. The convention is "`P0` is a statement of knowledge",
i.e. initial covariances encode what is actually known at t=0 and are not free parameters.

Tuning conditions: `train12` at mu=0.5 / chi=0.005, 12 trajectories, `:realistic` sensor
suite, replay, `f_est` = 1000 Hz, `use_dhat=false`, `fix_tier=:docking`.

### 7.3 Converged result — 5 seeds, all `plateau`

```
seed1  2.66697     seed2  2.62510     seed3  2.62867
seed4  2.58176  <- adopted            seed5  2.63062
```

Converged gains, seed4:

```
alpha_acc 0.0228662   alpha_yaw 0.0145877   grip_slip_scale 3.65730e-4
r_boost   61.6003     slip_R_inflate 42.2166
pose_Qn_heading 1.0e-8 (at its lower bound)  q_alpha 0.0665666
```

**The seed spread is a noise artifact, not a quality difference.** Re-scored on a *common*
noise realisation (`train12 | ns101`) the five configs land within **0.23%**, against a
3.30% spread in their own tuning scores:

```
seed1 2.6064   seed2 2.6005   seed3 2.6029   seed4 2.6031   seed5 2.6041     range 0.23%
```

So adopting seed4 is inconsequential — any of the five gives the same estimator to within
a quarter of a percent. (Exactly the pattern seen in the controller noisy stages, §3: when
the seed sets the noise draw, ranking by tuning score ranks realisations, not configs.)

**Term decomposition** at seed4 (`cross_eval/cross_eval_v4_results.json`, mean over noise
seeds 101–103):

```
tier       vel   inslip    rate     pos    head  smooth      SUM
train12  0.935    0.950   0.245   0.321   0.127   0.086    2.6637
 share   35.1%    35.7%    9.2%   12.0%    4.8%    3.2%
test     1.026    1.045   0.228   0.321   0.126   0.103    2.8483
 share   36.0%    36.7%    8.0%   11.3%    4.4%    3.6%
```

In physical units on `train12`: velocity RMSE **0.94 mm/s** against a 1 mm/s tolerance,
in-slip velocity RMSE **0.95 mm/s**, yaw-rate **2.45 mrad/s** against 10, position
**3.21 mm** against 10 mm, heading **1.27 mrad** against 10 mrad.

Two things follow. The objective is **~71% a velocity-estimation objective** (vel + inslip),
with position at 12% and heading under 5% — so the ESKF is tuned primarily to deliver
velocity, which is what the inner loops consume. And **in-slip velocity error is
indistinguishable from overall velocity error** (0.950 vs 0.935) — the estimator does not
degrade during slip, which is the specific failure the `lam_slip` term exists to prevent.

Generalisation to the held-out `test` tier costs **+6.9%** (2.6637 → 2.8483), concentrated
entirely in the velocity terms; position, heading and rate are flat or better. That is a
well-behaved estimator, not one fitted to its training tier.

Note the tuning `best_score` 2.5818 and the cross-eval `train12` mean 2.6637 differ because
they are different noise realisations — the same caveat as everywhere else in this document.

### 7.4 Closed-loop result through the ESKF

420 runs (6 configs × 14 `train14_v3` trajectories × noise seeds 101–105), **zero
non-converged**. `eval_v3_eskf.jl`, data in `runs_eskf_v3_train14/`.

```
config              score        std      track    chatter         ce   est_pos    n
PID-CT ct(s5)     0.34127    0.03174    0.02393    0.53231     37.345   0.00325   70
PID-CT nt(s4)     0.34341    0.03340    0.02624    0.53218     37.288   0.00325   70
PID-FB ct(s3)     0.36054    0.03936    0.03315    0.55431     37.415   0.00324   70
PID-FB nt(s4)     0.36298    0.03777    0.03149    0.56338     37.425   0.00324   70
ASMC   nt(s4)     0.36320    0.05517    0.04128    0.54157     37.541   0.00325   70
ASMC   ct(s2)     0.36423    0.04829    0.03658    0.55437     37.528   0.00325   70
```

#### Read this against the RIGHT baseline

There are **two** oracle baselines in §4 and they are not the same kind of thing:

* **clean oracle** — perfect state. The best achievable; an upper bound.
* **noisy oracle** — `OracleEstimator(:noisy)` writes true state **+ unfiltered per-tick
  sensor noise** straight into `xhat` (`sig_vel` 0.010 m/s, `sig_pos` 0.020 m, `sig_psi`
  0.010 rad, plus scale-factor and turn-on bias). **There is no estimator in that path.**
  It is a no-filtering pessimistic bound, not a deployment model.

**ESKF beats the noisy oracle because the state it hands the controller is cleaner** — and
that is measured, not assumed. Delivered state error, per channel, over all 420 closed-loop
runs, against the noisy oracle's injected error:

```
channel        ESKF delivered      noisy-oracle injected        ratio
velocity       2.71 mm/s           ~13 mm/s   (0.010 + 0.02|v| + bias)      ~5x cleaner
position       3.25 mm             ~22 mm     (0.020 + bias N(0,0.01))      ~7x cleaner
yaw rate       2.33 mrad/s         ~4 mrad/s  (0.003 + 0.005|w| + bias)     ~2x cleaner
heading        1.26 mrad *         ~11 mrad   (0.010 + bias N(0,0.005))     ~9x cleaner
```

`*` **steady-state**, i.e. the 11 of 14 trajectories that start at psi(0)=0. The raw mean over
all 14 is 9.78 mrad, but that number is an artefact and must not be quoted — see the
startup-transient finding in §7.5. An earlier draft of this section read the 9.78 mean as
"heading no better than raw injection" and as a 7.7x degradation from replay; **both were
wrong**, and both were consequences of averaging across a bimodal set.

So the filtering advantage is real in every channel — largest in heading and position,
smallest in yaw rate (~2x). Yaw *rate* is the genuinely weak channel (§7.5), not heading.

Two consequences for how this must be read:

1. **This is not one noise process filtered to two levels.** The noisy oracle injects
   synthetic error directly into `xhat` with no sensors; the ESKF path runs a `:realistic`
   SensorSuite (encoder/IMU/flow/pose-fix) through the filter. They are two *different*
   state-degradation models, so the ratios above compare delivered state quality — which is
   what the controller actually experiences — not filter efficiency against a common input.
   The "gap recovered" column below positions the ESKF between two bounds; it is not a
   statement that the filter removed 44% of a fixed noise budget.
2. **A tuned filter delivering cleaner state than raw injection is expected**, so the
   ESKF-minus-noisy-oracle delta is not "the cost of state estimation". That cost is ESKF
   vs *clean* oracle.

**Replay performance transfers to closed loop on three channels of four.** §7.3's converged
replay figures against these closed-loop measurements:

```
channel      replay (train12, open loop)    closed loop (train14_v3)     change
velocity     0.94 mm/s                      2.71 mm/s                    2.9x worse
yaw rate     2.45 mrad/s                    2.33 mrad/s                  unchanged
position     3.21 mm                        3.25 mm                      unchanged
heading      1.27 mrad                      1.26 mrad (steady state)     unchanged
```

Position, yaw rate and heading hold up essentially exactly; only **velocity degrades (~2.9×)**,
which is a genuine closed-loop effect — the per-trajectory velocity errors are tightly
clustered (1.9–3.5 mm/s), not bimodal, so it is not a transient artefact. Replay is open-loop
on `train12`; closed loop adds a controller and a different trajectory tier.

```
config              clean       ESKF  noisyOrc   cost vs clean   gap recovered
ASMC   ct(s2)     0.15318    0.36423   0.53168        +0.21105          44.2%
ASMC   nt(s4)     0.15697    0.36320   0.51838        +0.20623          42.9%
PID-CT ct(s5)     0.14340    0.34127   0.50372        +0.19787          45.1%
PID-CT nt(s4)     0.14422    0.34341   0.49799        +0.19919          43.7%
PID-FB ct(s3)     0.14953    0.36054   0.53135        +0.21101          44.7%
PID-FB nt(s4)     0.15492    0.36298   0.53199        +0.20806          44.8%
```

Real state estimation **costs ~+0.20 of score** against perfect state, and the ESKF recovers
**~44% of the clean→unfiltered gap**. Both figures are strikingly uniform across all six
configs (42.9–45.1%), i.e. **the estimator's contribution is controller-independent** — it
is not rescuing one control law more than another. `est_pos` is likewise identical to three
significant figures across all six (0.00324–0.00325): the estimator's own accuracy does not
depend on which controller is driving, which is exactly the property that makes a frozen
observer a fair basis for comparison.

#### Paired rankings

Controller ranking, clean-tuned configs, paired per (trajectory, noise seed), n = 70:

```
ASMC - PID-CT   mean +0.02296    PID-CT better on 69/70
PID-FB - PID-CT mean +0.01926    PID-CT better on 70/70
ASMC - PID-FB   mean +0.00370    split 33/70          -> TIED
```

**PID-CT wins under ESKF feedback too** — 69/70 and 70/70. That makes it best in all three
regimes: clean oracle, noisy oracle, and real state estimation.

**ASMC and PID-FB are tied under the ESKF** (33/70, mean 0.0037 against a std of ~0.04).
This matches the clean-tuned oracle result (§5, 3/2 split) and *not* the noisy-tuned oracle
result where ASMC beat FB 5/5. The ASMC advantage found in §5 does not survive real state
estimation.

Does noisy tuning transfer to ESKF feedback? Paired per (trajectory, noise seed):

```
ASMC     nt - ct   mean -0.00103    nt better on 58/70
PID-CT   nt - ct   mean +0.00214    nt better on 21/70
PID-FB   nt - ct   mean +0.00244    nt better on 14/70
```

**Largely no.** ASMC keeps a small, consistent benefit (58/70) an order of magnitude smaller
than the −0.01330 it gained under the noisy oracle; for **both PID variants noisy tuning is
actively counterproductive** under the ESKF. The reason is visible in the setup: the noisy
stages were tuned against *unfiltered* injected noise, whose character (white, per-tick,
10× larger) is nothing like the ESKF's filtered, correlated, much smaller residual. Gains
tuned for one do not transfer to the other.

Practical consequence: **the clean-tuned configs are the ones to deploy.** The ~10 h of
noisy tuning bought −0.01330 for ASMC and nothing for the PIDs under the oracle, and under
the ESKF it is neutral-to-harmful.

**Caveat carried from §7.2:** the ESKF is frozen at a config tuned on `train12` at mu=0.5,
not on `train14_v3` — deliberately not retuned, since refreezing per tier would reintroduce
the confound the estimator-first methodology exists to remove, but it does leave the
estimator slightly off-distribution for the `train14_v3` entries `train12` did not contain.

Chatter through the ESKF runs **0.532–0.563** against ~0.12 on a clean oracle: estimator
residual passes into the command. It stays well below the noisy oracle's 0.70–0.76, which
is the filtering doing its job.

### 7.5 Three-stage error budget — what the ESKF bought vs what the controller did

`diag_v3_error_budget.jl`. Splits the chain into **(1) injected** — the raw sensor noise in
state units, before any filtering, **(2) surviving** — what is left in `x̂` after the ESKF,
and **(3) tracking** — how far the platform ends up from the reference. No re-simulation:
(2) and (3) are already in `runs_eskf_v3_train14/runs_seed*.csv`, and (1) is analytic in the
fixed sensor constants plus, for the encoder and gyro channels, the reference-implied wheel
rates.

All four channels, each against **its own** injected source — velocity from the encoders
through the wheel Jacobian, yaw rate from the gyro, and position/heading from the pose fix
(`PoseFixModel(:docking)`: 100 Hz, `sigma_pos` 10 mm, `sigma_psi` 8.727 mrad, no bias, no
dropout). Getting this routing right matters: `_wheel_body_velocity` takes Vx/Vy from the
encoders but psi-dot from the gyro, and position/heading come from neither.

```
channel        (1) injected     (2) surviving    removed    ratio    injected by
velocity       11.458 mm/s       2.712 mm/s       76.3%      4.2x    encoders
position       10.000 mm         3.248 mm         67.5%      3.1x    pose fix
heading         8.727 mrad       1.263 mrad *     85.5%      6.9x    pose fix
yaw rate        4.901 mrad/s     2.332 mrad/s     52.4%      2.1x    gyro
```

`*` steady state — see the transient finding below.

**Yaw rate is the weak channel**, at 52.4% removed against 67–86% elsewhere, and it is worst
exactly where yaw is exercised: 43.6% on `spin_creep_easy` and 43.0% on
`spin_creep_stress_yaw` against ~59% on the zero-rotation octagons. Unlike position and
heading, yaw rate has no absolute fix to lean on — the gyro is its only source, and the
estimator objective prices `rate` at 9.2% (§7.1).

#### The heading channel is bimodal, and it is a startup transient — not an estimator fault

Per trajectory, heading splits cleanly in two:

```
11 of 14 trajectories        1.221 - 1.286 mrad    ~85.5% removed, 6.9x
spiral_orbit_stress         41.829 mrad            4.8x WORSE than the raw fix
ellipse_stress_tangent      38.560 mrad
ellipse_stress_crab         42.657 mrad
```

Those three are **exactly** the three trajectories whose reference starts at a non-zero
heading — `psi(0)` = 1.5708, 1.5708, 2.3562 rad. All eleven others have `psi(0)` = 0.0000
exactly. The ESKF initialises `psi_hat` at zero, so those three open with a 1.6–2.4 rad
error, and `est_heading` is an RMS over the whole run.

Testing that quantitatively: if the excess is a startup transient at the full initial error
`e0` lasting `T_conv`, then `rms ≈ e0*sqrt(T_conv/T_total)`, so `T_conv = T*(rms/e0)^2`:

```
trajectory               rms[mrad]   e0[rad]     T[s]   implied T_conv
spiral_orbit_stress         41.829    1.5708    15.53         11.0 ms
ellipse_stress_tangent      38.560    1.5708    18.27         11.0 ms
ellipse_stress_crab         42.657    2.3562    33.59         11.0 ms
```

**All three imply the same 11.0 ms**, across three different durations and two different
initial errors. The ESKF converges heading from an arbitrary start in ~11 ms; nothing else
is happening. Steady-state heading estimation is ~1.26 mrad on every trajectory, matching
the replay figure of 1.27 mrad exactly.

Two consequences. **Heading is the estimator's *best* channel** (6.9× removal), not its
worst — the opposite of what the all-14 mean of 9.78 mrad suggests. And **`est_heading`
averaged over a set with mixed `psi(0)` is not a usable statistic**: one 11 ms transient
moves a 33-second RMS by 30×. Any future comparison should either exclude a startup window
or initialise `psi_hat` from the first pose fix.

Per config, with stage (3) split into its two channels:

```
config           (1)inj vel  (2)surv vel   removed |  (3)trk pos  (3)trk head
                     [mm/s]       [mm/s]           |        [mm]       [mrad]
ASMC   ct(s2)        11.458        2.610     77.2% |      34.106        1.936
ASMC   nt(s4)        11.458        2.825     75.3% |      29.213        4.677
PID-CT ct(s5)        11.458        2.941     74.3% |      15.313        6.007
PID-CT nt(s4)        11.458        2.954     74.2% |      16.593        7.411
PID-FB ct(s3)        11.458        2.483     78.3% |      30.188        1.733
PID-FB nt(s4)        11.458        2.462     78.5% |      32.178        1.400
```

**This is the attribution the score alone cannot give.** Stage (1) is identical across
configs by construction — a property of the sensors and the reference, not of the control
law. Stage (2) varies only 74.2–78.5%, because the estimator is frozen and its accuracy
barely depends on which controller drives it. **So essentially the entire spread between
controllers lives in stage (3).** The estimator contributes a large common-mode improvement;
only the tracking stage discriminates control laws.

And stage (3) shows the two PID variants failing in *opposite* channels:

* **PID-CT has the best position tracking by 2×** (15.3 mm vs 29–34 mm for everyone else)
  **and the worst heading tracking by 3–4×** (6.0–7.4 mrad vs 1.4–1.9 mrad for PID-FB).
* **PID-FB is the mirror image** — best heading (1.4–1.7 mrad), position roughly twice CT's.

CT wins the overall score because the v3 tracking term weights the two channels equally and
its position advantage is larger in tolerance-normalised terms than its heading deficit.
This is the same CT heading weakness that §6 traced to a yaw integral clamp firing on 19.9%
/ 17.0% of ticks on the two sustained-yaw trajectories — now confirmed a third time, in
closed loop with a real estimator, in physical units.

Note the two weaknesses **compound on the same channel**: the estimator removes least error
on yaw rate (52%, worst on high-yaw trajectories), and CT's controller-side heading error is
the largest of the field. Anything that improves heading — re-weighting the estimator
objective, or re-deriving CT's yaw integral budget — attacks both at once.

### 7.6 Per trajectory — motion scale, pose estimation, pose tracking

`trk_pos`/`trk_head` are **controller** error: true plant state vs reference, RMS over
ticks. Distinct from `est_pos` (stage 2, how wrong the *estimate* was) and not the same
object as `tracking_v3`, which is tolerance-normalised, six-term and `k_traj`-scaled.
`travel` is total distance along each world axis (`int|dX|`), `range` is peak-to-peak
extent; they disagree on repeating paths, so both are shown.

```
trajectory                   trvl_x  trvl_y   rng_x   rng_y    rot | est_pos removed |  trk_pos /travel  /range
                                [m]     [m]     [m]     [m]  [rad] |    [mm]         |     [mm]
octagon_easy                   5.79    5.79    2.90    2.90   0.00 |   3.235   67.6% |     3.07   0.04%   0.08%
octagon_mid                    8.69    8.69    4.35    4.35   0.00 |   3.244   67.6% |    15.00   0.12%   0.24%
octagon_stress                11.50   11.50    5.75    5.75   0.00 |   3.253   67.5% |    24.58   0.15%   0.30%
octagon_stress_hdg30          11.50   11.50    5.75    5.75   0.00 |   3.252   67.5% |    25.24   0.16%   0.31%
octagon_stress_lat08           9.33    9.33    4.66    4.66   0.00 |   3.242   67.6% |    11.80   0.09%   0.18%
octagon_stress_lat08_hdg30     9.33    9.33    4.66    4.66   0.00 |   3.240   67.6% |    11.50   0.09%   0.17%
spin_creep_easy                0.00    0.00    0.00    0.00  12.50 |   3.242   67.6% |     2.07     n/a     n/a
spin_creep_stress_yaw          0.95    0.51    0.55    0.10  26.00 |   3.247   67.5% |     2.03   0.19%   0.37%
coupled_vomega_easy            3.09    4.54    2.58    4.54   4.00 |   3.248   67.5% |     2.29   0.04%   0.04%
coupled_vomega_stress          3.79    3.90    2.88    2.80   8.00 |   3.280   67.2% |   160.36   2.95%   3.99%
spiral_orbit_stress            3.66    3.75    1.60    2.48   9.42 |   3.316   66.8% |   102.14   1.95%   3.46%
ellipse_stress_tangent         6.40    2.56    1.60    0.64  12.57 |   3.251   67.5% |     2.74   0.04%   0.16%
ellipse_stress_crab            7.20    2.88    1.80    0.72   0.00 |   3.214   67.9% |     2.40   0.03%   0.12%
multisine75_broadband          2.77    4.71    0.17    0.42   4.62 |   3.202   68.0% |     2.50   0.05%   0.56%
```

**The ESKF's pose correction is flat.** 3.20–3.32 mm surviving from 10 mm injected, 66.8–68.0%
removed, on every trajectory — independent of speed, path length and rotation. The pose fix is
an absolute 100 Hz measurement, so unlike the encoder channel it has no state-dependent term.
That is what makes it a clean common-mode offset (§7.5).

**Tracking error is 0.03–0.19% of distance travelled on 12 of 14 trajectories.** Only
`coupled_vomega_stress` (2.95%) and `spiral_orbit_stress` (1.95%) are outliers. The 160 mm on
the former is 3% of a 5.4 m path, not a gross failure.

**On 7 of 14 trajectories the controller tracks the reference more accurately than the
estimator knows where the robot is** — `spin_creep_stress_yaw` 2.03 mm tracking against
3.25 mm estimation, `ellipse_stress_crab` 2.40 vs 3.21, `multisine75_broadband` 2.50 vs 3.20,
and similarly for `octagon_easy`, `spin_creep_easy`, `coupled_vomega_easy`,
`ellipse_stress_tangent`. On those, **tracking error sits at or below the estimator's noise
floor**, so further control-law improvement there is unmeasurable with this sensor suite:
the controller is already correcting faster than the state estimate can resolve. Controller
comparison on this tier is therefore decided almost entirely by the handful of trajectories
where tracking error rises clear of ~3 mm.

#### Stage-3 position tracking [mm], per trajectory × config

```
trajectory                    ASMC ct(s2)   ASMC nt(s4) PID-CT ct(s5) PID-CT nt(s4) PID-FB ct(s3) PID-FB nt(s4)
octagon_easy                         3.26          2.13          2.16          2.48          4.89          3.51
octagon_mid                          8.35          4.81          4.85          5.52         32.05         34.44
octagon_stress                      20.01         12.60         11.39         16.05         41.63         45.80
octagon_stress_hdg30                19.32         11.72         11.05         14.06         45.80         49.49
octagon_stress_lat08                 9.02          5.19          5.52          6.71         21.30         23.03
octagon_stress_lat08_hdg30           9.02          5.02          4.20          5.25         21.78         23.72
spin_creep_easy                      2.42          1.92          1.93          2.13          2.00          2.03
spin_creep_stress_yaw                2.17          1.82          1.91          2.09          2.09          2.10
coupled_vomega_easy                  2.02          1.55          1.94          2.09          3.54          2.58
coupled_vomega_stress              255.16        231.65         97.33        101.11        131.46        145.45
spiral_orbit_stress                139.57        125.27         65.97         68.14        104.31        109.59
ellipse_stress_tangent               2.35          1.85          1.89          1.92          5.09          3.33
ellipse_stress_crab                  2.83          1.86          2.04          2.26          2.98          2.44
multisine75_broadband                1.99          1.61          2.22          2.51          3.70          2.98
```

**PID-FB's position deficit is an octagon problem** — 21–50 mm on the six octagons against
4–16 mm for CT and ASMC-nt, while being competitive everywhere else. **ASMC's is the opposite**:
fine on the octagons, catastrophic on `coupled_vomega_stress` (255 mm, 2.6× CT) and
`spiral_orbit_stress` (140 mm). PID-CT is the only config without a collapse mode on position.

#### Stage-3 heading tracking [mrad], per trajectory × config

```
trajectory                    ASMC ct(s2)   ASMC nt(s4) PID-CT ct(s5) PID-CT nt(s4) PID-FB ct(s3) PID-FB nt(s4)
octagon_easy                        1.555         3.251         1.151         1.224         1.144         1.158
octagon_mid                         1.684         3.625         1.312         1.575         1.131         1.144
octagon_stress                      1.775         3.994         2.434         3.142         1.188         1.173
octagon_stress_hdg30                1.881         4.194         2.876         3.340         1.268         1.218
octagon_stress_lat08                1.539         3.056         1.264         1.434         1.129         1.147
octagon_stress_lat08_hdg30          1.543         3.183         1.277         1.505         1.171         1.174
spin_creep_easy                     1.498         3.186         1.107         1.214         1.953         1.412
spin_creep_stress_yaw               1.363         2.719         1.036         1.049         5.399         3.036
coupled_vomega_easy                 1.239         2.175         1.065         1.067         1.115         1.139
coupled_vomega_stress               4.764        14.293        29.813        38.217         1.282         1.187
spiral_orbit_stress                 4.411        15.481        37.597        46.800         2.350         1.724
ellipse_stress_tangent              1.303         2.099         1.046         1.080         2.757         1.775
ellipse_stress_crab                 1.235         1.983         1.060         1.044         1.111         1.138
multisine75_broadband               1.312         2.234         1.059         1.069         1.262         1.175
```

**This is the cleanest statement of CT's heading weakness in the whole campaign.** PID-CT is
the *best* heading config on 9 of 14 trajectories (1.04–1.31 mrad) and then collapses to
**29.8 mrad on `coupled_vomega_stress` and 37.6 mrad on `spiral_orbit_stress`** — 25–30× its
own baseline, and ~25× PID-FB on the same two runs (1.28 / 2.35 mrad). Those are exactly the
two trajectories where §6 measured CT's yaw integral clamp firing on 19.9% / 17.0% of ticks
and 0.0% everywhere else. Three independent measurements — HEAD-sum concentration, clamp
saturation fraction, and now closed-loop heading tracking in physical units — land on the
same two trajectories.

Two further readings. **Noisy tuning damaged ASMC's heading everywhere**: ASMC-nt is ~2× worse
than ASMC-ct on all 14 (3.0–4.2 vs 1.2–1.9 mrad on the benign ones), which is part of why
noisy tuning fails to transfer (§7.4). And **PID-FB's only heading weakness is
`spin_creep_stress_yaw`** (5.40 mrad) — sustained pure yaw, the same regime, reached from the
opposite direction: FB has no feedforward to fight, but also nothing to help it.

## 8. Scripts and data

Promoted out of `_tmp/` (which is gitignored) into this directory. All take
`REPORT_ONLY=1` to dry-run their formatting without simulating — two earlier 168-run
sweeps were lost to a `@printf` arg-count typo, so **serialise before formatting** and
verify the formatting on dummy data first.

| script | what it produces | cost |
|---|---|---|
| `eval_v3_clean_components.jl` | clean score decomposition, 3 controllers, per-trajectory | 42 sims |
| `eval_v3_noisy_pid.jl` | PID-CT/FB clean-tuned configs on seeds 101–105 | 168 sims |
| `eval_v3_noisy_asmc.jl` | ASMC clean-tuned config on seeds 101–105 | 84 sims |
| `eval_v3_noisytuned.jl` | all 3 noisy-tuned configs on 101–105 + paired vs clean-tuned | 252 sims |
| `diag_v3_pid_imax_binding.jl` | per-axis integral saturation / utilisation | 28 sims |
| `eval_v3_eskf.jl` (+ `run_eval_v3_eskf.bat`) | 6 configs through the frozen ESKF, seeds 101-105 | 420 sims |
| `diag_v3_error_budget.jl` | 4-channel error budget + per-trajectory motion scale / pose est / pose tracking | 0 sims (analytic + CSV) |

Provenance and verification for the settings those evals run under — promoted alongside,
original basenames kept:

| script | establishes |
|---|---|
| `metric_v3_calib.jl` | the v3 tracking metric: per-trajectory scaler + time-normalised integral term |
| `metric_v3_verify.jl` | metric wiring delta-test, incl. `--metric v2` must stay BIT-IDENTICAL |
| `scalers_v3.jl` | per-trajectory scalers on the final v3 tiers, each over its own `T_total` |
| `lambda_v3_verify.jl` | `LAMBDA_CHATTER_V3 = 0.36` and that its `>1.44` guard fires BOTH ways |
| `v3_verify_final.jl` | v3 tier acceptance: every reference over its own `ref.T_total` |
| `three_way_excl.jl` | re-slices the clean comparison with trajectories excluded (reads the `.jls`, no re-simulation) |

Serialised results in `results_v3/*.jls`. Console logs stay in `_tmp/` — they are 0.2–2.3 MB
of progress-bar output each.

Left in `_tmp/` as superseded scratch, deliberately not promoted: `v3_check.jl`,
`v3_check2.jl`, `v3_final.jl` (all superseded by `v3_verify_final.jl`, which is the version
that reads `T_total` rather than assuming a window) and `pid_v3_components.jl` (asks why
PID-FB beat PID-CT — a question that only existed under the ABANDONED imbalanced metric;
`eval_v3_clean_components.jl` supersedes it for all three controllers).

### Directory layout

`hybrid_ctrl_v2/` holds **only** what reproduces the results above; everything superseded is
in `archived/` (876 files) with its own README. The split was computed by include-closure
from the four entry points, not by hand, and all four were re-run afterwards to confirm they
still load.

```
core modules      controllers_v2.jl  estimators_v2.jl  estimators_v3.jl
                  scheduler_v2.jl  sensors_v2.jl  tune_controller_v2.jl
                  eval_controllers_eskf_v3.jl   <- library: eval_v3_eskf.jl includes it
controller_tuning/  run_stage.jl  stage_objective.jl  trajsets.jl
                    pid_cascade.jl  optimizer_stage.jl  mpc_design.jl
estimator_tuning/   harness_v2.jl  objective_v2.jl  param_space_v3.jl  param_space_v4.jl
                    run_estimator_replay_mu0p5_v4.jl  cross_eval_mu0p5_v4.jl  (+ .bat)
evaluation        eval_v3_{clean_components,noisy_pid,noisy_asmc,noisytuned,eskf}.jl
diagnostics       diag_v3_{pid_imax_binding,error_budget}.jl
provenance        metric_v3_{calib,verify}.jl  scalers_v3.jl  lambda_v3_verify.jl
                  v3_verify_final.jl  three_way_excl.jl
launchers         run_stage_asmc_v3.bat  run_stage_pid_v3.bat  run_eval_v3_eskf.bat
results           runs_asmc_v3/  runs_pid_v3/  runs_estimator_v4_mu0p5_train12/
                  runs_eskf_v3_train14/  results_v3/
```

`param_space_v3.jl` is kept although the v3 estimator is superseded: `param_space_v4.jl`
builds `PINNED_V4` from `Main.ParamSpaceV3Mod.PINNED_V3`, so v4 does not load without it.

Tuning launchers (unchanged): `run_stage_asmc_v3.bat <clean|noisy> [nseeds]`,
`run_stage_pid_v3.bat <fb|ct> <clean|noisy> [nseeds]`.
Converged configs: `runs_asmc_v3/seed*/asmc_v2_{clean,noisy}/`,
`runs_pid_v3/seed*/pid_v2_{ct,fb}_{clean,noisy}/`.

Supporting instrumentation added to `controllers_v2.jl`: `i_util_log`, per-axis
`|I|/I_max` per tick, behind the pre-existing default-off `log_diag`. `i_sat_log` is an
`any()` across the three axes and cannot attribute saturation to yaw.

## 9. Held-out tier `:test_v3` — does any of this generalise?

Same structure as §4 (oracle), §7.4 (ESKF), §7.5 (error budget) and §7.6 (per trajectory),
generated by the same scripts under `V3_TIER=test_v3`, so the columns are directly
comparable line-for-line. Outputs are tier-suffixed (`*_test_v3.jls`,
`runs_eskf_v3_test/`) and never overwrite the training results.

**Scores are NOT comparable in magnitude across tiers** — `k_traj` and the trajectory mix
both differ. Read the ranking and each config's train→test delta.

### 9.1 What the tier actually holds — three levels of novelty

```
trajectory                    profile                          novelty
long_circle_profile           long_circle_mu_0p5.toml          PROFILE never seen
long_circle_profile_stress    long_circle_mu_0p5.toml          PROFILE never seen
multisine50_combo             multisine_50percent_cap_mu_0p5   PROFILE never seen
multisine75_combo_gentle      multisine_75percent_cap_mu_0p5   combo unseen
spiral_orbit_easy             spiral_orbit_mu_0p5.toml         combo unseen
ellipse_easy_tangent          ellipse_mu_0p5.toml              combo unseen
ellipse_easy_crab             ellipse_mu_0p5.toml              combo unseen
coupled_vomega_anchor         coupled_vomega_mu_0p5.toml       *** NOT HELD OUT ***
```

**`coupled_vomega_anchor` is `combo_idx=12` — the same TOML, combo, ref type and adapt flag
as `train14_v3`'s `coupled_vomega_stress`.** It is the same trajectory under a different
name: a deliberate anchor tying the tiers together, as its name says. It must be **excluded
from every generalisation claim**, and it matters more than one-in-eight suggests because it
is the worst-tracked entry in either tier (160 mm vs 2.0–2.7 mm) and therefore dominates any
mean it enters.

It also serves as a **determinism check, and it passes exactly**: across all six configs the
anchor's ESKF score reproduces between tiers to **±0.000000** (e.g. PID-CT ct 0.36179 in
both). Same combo and same noise seeds give bit-identical closed-loop results.

So the tier gives **7 genuinely held-out trajectories**, of which **3 use a profile the
controllers have never encountered**.

### 9.2 Oracle — all six configs, sensor seeds 101–105 (cf. §4)

```
                 clean sc   noisy sc      track    chatter  realis.var
PID-CT ct(s5)     0.09556    0.51186    0.11970    0.74571       9.3%
PID-CT nt(s4)     0.09604    0.50366    0.11716    0.73350       9.4%
ASMC   ct(s2)     0.10836    0.53472    0.13197    0.76925       9.1%
ASMC   nt(s4)     0.10778    0.51921    0.12713    0.74519       8.7%
PID-FB ct(s3)     0.10613    0.52980    0.12370    0.77661       8.8%
PID-FB nt(s4)     0.10736    0.52986    0.12357    0.77689       8.9%
```

**Did noisy tuning help?** Paired by realisation — the §4 test, re-run on held-out data:

```
ASMC     -0.01634  -0.01911  -0.01983  -0.00519  -0.01708   mean -0.01551   nt better 5/5
PID-CT   -0.00937  -0.00788  -0.00892  -0.00683  -0.00804   mean -0.00821   nt better 5/5
PID-FB   +0.00298  +0.00137  +0.00070  -0.00323  -0.00152   mean +0.00006   MIXED
```

Clean-side cost: ASMC −0.00058, PID-CT +0.00047, PID-FB +0.00123.

**The §4 noisy-tuning finding replicates exactly**: helps ASMC most (5/5), helps PID-CT less
(5/5), fails for PID-FB (mixed, ~zero mean). Same three verdicts, same ordering of benefit,
on trajectories none of the tuning ever saw.

One shape difference worth noting: clean scores are much better on this tier (0.096 vs
0.143 for PID-CT ct) while **noisy scores barely move** (0.512 vs 0.504). Under noise the
score is ~64% chatter, and chatter is near its cap regardless of trajectory — so an easier
tier shows up almost entirely in the clean column.

### 9.3 ESKF closed loop (cf. §7.4)

240 runs (6 configs × 8 trajectories × 5 seeds), zero non-converged.

```
config              score        std      track    chatter         ce   est_pos    n
PID-CT ct(s5)     0.31894    0.03630    0.02289    0.53312     26.952   0.00324   40
PID-CT nt(s4)     0.32132    0.03705    0.02581    0.53201     26.927   0.00324   40
PID-FB ct(s3)     0.33287    0.03644    0.02655    0.55546     27.053   0.00323   40
PID-FB nt(s4)     0.33623    0.03756    0.02550    0.56493     27.127   0.00323   40
ASMC   nt(s4)     0.34063    0.05354    0.04129    0.53934     27.186   0.00324   40
ASMC   ct(s2)     0.34485    0.05404    0.03818    0.55584     27.140   0.00323   40
```

### 9.4 Error budget (cf. §7.5) — the estimator is tier-invariant

```
channel        (1) injected     (2) surviving    removed    ratio    [train14_v3 for comparison]
velocity       10.813 mm/s       2.532 mm/s       76.6%      4.3x    76.3%  4.2x
position       10.000 mm         3.235 mm         67.7%      3.1x    67.5%  3.1x
heading         8.727 mrad       1.256 mrad *     85.6%      6.9x    85.5%  6.9x
yaw rate        4.576 mrad/s     2.027 mrad/s     55.7%      2.3x    52.4%  2.1x
```

`*` steady state, the 6 test trajectories with `psi(0)=0`.

Every channel lands within ~3 percentage points of its training value. **The frozen
estimator's contribution does not depend on the trajectory tier** — expected, since the pose
fix is an absolute 100 Hz measurement and the observer was never refit. Yaw rate remains the
weak channel in both.

### 9.5 Per trajectory (cf. §7.6)

```
trajectory                   trvl_x  trvl_y   rng_x   rng_y    rot | est_pos removed |  trk_pos /travel  /range
                                [m]     [m]     [m]     [m]  [rad] |    [mm]         |     [mm]
long_circle_profile           12.50   12.04    6.00    6.00   6.45 |   3.228   67.7% |     2.11   0.01%   0.02%
long_circle_profile_stress     2.84    2.60    1.20    1.20   7.12 |   3.347   66.5% |     2.30   0.06%   0.14%
multisine75_combo_gentle       3.43    6.02    0.26    0.52   6.61 |   3.201   68.0% |     2.65   0.04%   0.45%
multisine50_combo              1.77    2.93    0.12    0.27   4.36 |   3.200   68.0% |     2.32   0.07%   0.78%
spiral_orbit_easy              5.68    5.68    2.52    2.90   9.42 |   3.189   68.1% |     2.16   0.03%   0.06%
ellipse_easy_tangent           6.40    2.56    1.60    0.64  12.57 |   3.212   67.9% |     2.00   0.03%   0.12%
ellipse_easy_crab              4.00    1.60    1.00    0.40   0.00 |   3.220   67.8% |     2.24   0.05%   0.21%
coupled_vomega_anchor  [ANCH]  3.79    3.90    2.88    2.80   8.00 |   3.280   67.2% |   160.36   2.95%   3.99%
```

Stage-3 position tracking [mm], per trajectory × config:

```
trajectory                    ASMC ct(s2)   ASMC nt(s4) PID-CT ct(s5) PID-CT nt(s4) PID-FB ct(s3) PID-FB nt(s4)
long_circle_profile                  2.03          1.63          1.96          2.08          2.67          2.30
long_circle_profile_stress           2.11          1.68          1.71          1.83          3.82          2.62
multisine75_combo_gentle             2.17          1.71          2.22          2.50          4.17          3.11
multisine50_combo                    2.00          1.63          2.23          2.61          2.90          2.51
spiral_orbit_easy                    2.39          1.92          2.13          2.40          2.05          2.07
ellipse_easy_tangent                 1.91          1.56          1.87          1.98          2.48          2.19
ellipse_easy_crab                    2.17          1.65          2.20          2.53          2.64          2.26
coupled_vomega_anchor  [ANCHOR]    255.16        231.65         97.33        101.11        131.46        145.45
```

Stage-3 heading tracking [mrad], per trajectory × config:

```
trajectory                    ASMC ct(s2)   ASMC nt(s4) PID-CT ct(s5) PID-CT nt(s4) PID-FB ct(s3) PID-FB nt(s4)
long_circle_profile                 1.233         1.939         1.038         0.994         1.103         1.135
long_circle_profile_stress          1.226         1.839         1.067         1.062         1.217         1.181
multisine75_combo_gentle            1.305         2.195         1.061         1.067         1.217         1.146
multisine50_combo                   1.373         2.507         1.073         1.113         1.232         1.161
spiral_orbit_easy                   1.335         2.526         1.048         1.033         1.081         1.108
ellipse_easy_tangent                1.199         1.753         1.056         1.050         1.325         1.211
ellipse_easy_crab                   1.229         2.069         1.014         1.022         1.070         1.103
coupled_vomega_anchor  [ANCHOR]     4.764        14.293        29.813        38.217         1.282         1.187
```

**On the 7 held-out trajectories every controller tracks to 1.6–4.2 mm and 1.0–2.5 mrad.**
The 160 mm and 29.8/38.2 mrad outliers are entirely the anchor — i.e. a *training*
trajectory. PID-CT's yaw-clamp heading collapse (§6) appears only there, consistent with it
being a property of sustained-yaw trajectories rather than of held-out data.

Note **ASMC nt has the best position tracking on all 7 held-out trajectories** (1.56–1.92 mm)
while simultaneously having the worst heading (1.75–2.53 mrad) — the same
position/heading trade the two PID variants show in §7.6, here inside one controller family.

### 9.6 Verdict — what generalises and what does not

Excluding the anchor, on the 7 genuinely held-out trajectories (ESKF, paired per
(trajectory, seed), n=35):

```
ASMC ct   - PID-CT ct   mean +0.01560   PID-CT better 35/35
PID-FB ct - PID-CT ct   mean +0.01310   PID-CT better 35/35
ASMC ct   - PID-FB ct   mean +0.00250   MIXED 23/35
ASMC nt   - PID-FB ct   mean -0.00063   MIXED  8/35
```

**PID-CT generalises as the winner — unambiguously.** First and second place on both tiers,
35/35 paired against each rival on held-out data, and first on every individual held-out
trajectory. This is the campaign's most robust result.

**The ASMC-vs-PID-FB ordering does NOT generalise.** On `train14_v3` the mean ordering put
PID-FB ahead of ASMC; on held-out trajectories they are statistically indistinguishable
(23/35 and 8/35, means 0.0006–0.0025 against a std of ~0.04). The training-tier ordering was
carried by `coupled_vomega_stress` — the trajectory §5 identified as holding 66% of ASMC's
gap, and the very one duplicated into the test tier as the anchor. Remove it and the two
families tie.

That is consistent, not contradictory, with the rest of the campaign: §5 found ASMC and FB
tied clean-tuned (3/2 split), §7.4 found them tied under the ESKF (33/70). **The
defensible claim is that ASMC and PID-FB are equivalent, and PID-CT beats both** — the
apparent FB-over-ASMC edge is a single-trajectory artifact.

## Reading the numbers

1. **Never mix a tuning-stage score with an eval score.** Each noisy tuning stage scored on
   its own realisation; the eval table averages seeds 101–105. Their ranges overlap, which
   makes "noisy tuning made things worse" look supported when it is not.
2. **Pair, don't average, across realisations.** Realisation variance (~6–7% in eval, ~16%
   at tuning) swamps controller differences of ~2–4%. Every ordering claim above is a
   paired 5/5 or 4/4, never a difference of marginal means.
3. **Best-seed selection is confounded under noise.** All three controllers' best seed is
   seed4, which is also the easiest realisation for all three. It is applied identically to
   all three so cross-controller comparison holds, but absolute `nt` improvements may be
   flattered.
4. **Absolute values, never % degradation** — see §4.
5. **Three feedback regimes, not two, and they are not interchangeable.** `clean oracle` =
   perfect state; `noisy oracle` = *unfiltered* sensor noise written straight into `xhat`
   with **no estimator**; `ESKF` = the deployable path. The ESKF beats the noisy oracle by
   construction (a filter beats raw noise), so an ESKF-vs-noisy-oracle delta is **not** the
   cost of state estimation — that cost is ESKF vs *clean* (§7.4). Getting this wrong
   inverts the sign of the conclusion.

## Not done

- **MPC v3** — deferred out of this iteration. Its `use_ltv` anomaly (diverges on 4
  velocity-demanding trajectories; `use_ltv=false` improves it, which is backwards) remains
  unresolved from the parent handoff.
- ~~**`:test_v3` generalisation**~~ **DONE** -- §9. Both regimes evaluated (oracle 288 sims,
  ESKF 240 runs). Caveat recorded there: `coupled_vomega_anchor` is combo-identical to
  training's `coupled_vomega_stress`, so the tier yields 7 held-out trajectories, not 8.
- **`pid_ct_heading_weakness_handoff.md` §1** — still reads as a broad heading deficit; §6
  above supplies the measurement for the concentration finding that should replace it.
- **`I_max` budget-sensitivity** — whether CT's binding yaw clamp actually costs heading
  accuracy is unmeasured.
- **ESKF yaw-RATE channel** — §7.5 measures only 52.4% of injected yaw-rate error removed,
  against 67–86% on the other three channels, and worst on the high-yaw trajectories (43%).
  The gyro is its only source (no absolute fix to lean on) and the estimator objective prices
  `rate` at 9.2% (§7.1). Re-tuning with a heavier rate weight is the obvious next experiment;
  it interacts with the `I_max` yaw finding in §6 and with PID-CT's heading *tracking* error
  in §7.5 — all three concern yaw.
  *(An earlier draft named the HEADING channel here. That was the `psi(0)` startup artefact,
  resolved in §7.5: heading is the estimator's best channel at 6.9×, not its worst.)*
- **`psi_hat` initialisation** — the ESKF starts heading at zero rather than from the first
  pose fix, costing an ~11 ms transient that dominates a full-run RMS on any trajectory with
  `psi(0) != 0`. Cheap to fix, and it would make `est_heading` comparable across tiers with
  mixed initial headings.
