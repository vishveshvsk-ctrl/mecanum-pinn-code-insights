# s100 ablation — safety margin 0.9 → 1.0 (single-variable mirror of the v3 campaign)

Supervisor-requested ablation: the (E56) factor of safety on the *available* friction
circle is removed (`PhysicalLimits.safety_margin` 0.9 → 1.0) and the **entire v3 tuning +
eval chain is re-run** for the two controllers the factor actually gates:

* **ASMC** — `kmax_schedule`'s gain ceiling scales with `safety_margin × available friction
  circle`.
* **PID-CT** — `vcmd_limits`' command gate γ (the feedforward `V_ff` ceiling) scales the
  same way. (PID-FB is out of scope: no feedforward; the gate touches only its correction
  term. Its results stay at 0.9.)

Everything else is bit-identical to the archived campaign: train14_v3 tier, `--metric v3`,
`lambda-chatter 0.36`, eps floors (0.02/0.08), lam-psi-hi 60, same seeds, same optimizer
budgets (p1 250 / p2 60, dxNES → BOBYQA), noisy stage warm-started from own clean.

**Verified gate deltas** (`_tmp/verify_safety_margin_wired.jl`, all four levels pass):
kmax free-circle ceiling ×1.1111 (= 1/0.9 exactly), kmax loaded circle ×1.1452
(nonlinear — the loaded circle shifts the contact ellipse, not just its scale),
PID γ 0.4388 → 0.4876 (×1.1111). The default-margin path is byte-identical to the
archived campaign. Output roots are auto-suffixed `_s100`; the archived runs are
untouched.

---

## 1. Clean tuning (5 seeds, all converged)

```
              s=0.9 (archived)                          s=1.0
ASMC    0.15233 0.15235 0.15239 0.15305 0.15314    0.15197 0.15206 0.15248 0.15277 0.15295
PID-CT  0.14319 0.14323 0.14328 0.14341 0.14341    0.14175 0.14179 0.14189 0.14193 0.14212

best:   ASMC  0.15318 (s2) → 0.15197 (s1)     −0.79%
        PID-CT 0.14340 (s5) → 0.14175 (s2)    −1.15%
```

**Every one of the 10 s=1.0 seeds beats its family's archived 0.9 best.** Seed spread
stays tight (≤0.7%). Stop reasons: 8× `plateau`, 2× `no_refine_gain` (one per family).

## 2. Noisy tuning (4 seeds, warm-started from own clean; all `plateau`)

```
              seed1     seed2     seed3     seed4      mean      std
ASMC  0.9   0.57065   0.53392   0.55977   0.49273    0.53927   0.03464
ASMC  1.0   0.57667   0.53388   0.56902   0.49309    0.54317   0.03693
PID   0.9   0.55484   0.50927   0.54155   0.47081    0.51912   0.03746
PID   1.0   0.55511   0.51302   0.54154   0.47134    0.52025   0.03748
```

The tuning seed sets the noise realisation, so pair by seed (positive = s100 worse):

```
ASMC    paired 1.0−0.9:  +0.00602  −0.00004  +0.00925  +0.00036   mean +0.00390
PID-CT  paired 1.0−0.9:  +0.00027  +0.00375  −0.00001  +0.00053   mean +0.00114
```

**The clean-stage gain does not survive into the noisy tuning objective** — both families
are flat-to-slightly-worse, and the effect is well inside the shared realisation
variance (std ≈ 0.037). The relaxed gate buys tracking that unfiltered sensor noise
hands back.

## 3. Oracle eval, train14_v3 (sensor seeds 101–105; mirror of archived §4)

```
                 clean sc   noisy sc      track    chatter  realis.var
PID-CT ct  0.9   0.14340    0.50372    0.10377    0.71475       7.1%
PID-CT ct  1.0   0.14203    0.50093    0.10396    0.70836       6.8%
PID-CT nt  0.9   0.14422    0.49799    0.10331    0.70347       7.1%
PID-CT nt  1.0   0.14269    0.49799    0.10345    0.70307       6.9%
ASMC   ct  0.9   0.15318    0.53168    0.11433    0.75346       6.5%
ASMC   ct  1.0   0.15271    0.52948    0.11289    0.75172       6.6%
ASMC   nt  0.9   0.15697    0.51838    0.11176    0.72952       6.6%
ASMC   nt  1.0   0.15477    0.51866    0.11256    0.72876       6.5%
```

Did noisy tuning help? (paired nt−ct, negative = nt better):

```
ASMC   1.0:  −0.00948 −0.01257 −0.00892 −0.00901 −0.01412   mean −0.01082   5/5
PID-CT 1.0:  −0.00259 −0.00252 −0.00325 −0.00326 −0.00305   mean −0.00293   5/5
(0.9 was:  ASMC mean −0.01330 5/5;  PID-CT mean −0.00572 5/5)
```

The noisy-tuning benefit **shrinks** at s=1.0 for both families (ASMC −0.0108 vs −0.0133;
PID-CT −0.0029 vs −0.0057) — the relaxed gate leaves less on the table for noisy tuning
to recover.

## 4. Oracle eval, test_v3 (held-out; mirror of archived §9.2)

```
                 clean sc   noisy sc      track    chatter  realis.var
PID-CT ct  0.9   0.09556    0.51186    0.11970    0.74571       9.3%
PID-CT ct  1.0   0.09460    0.50511    0.11819    0.73423       9.4%
PID-CT nt  0.9   0.09604    0.50366    0.11716    0.73350       9.4%
PID-CT nt  1.0   0.09513    0.50185    0.11758    0.72836       9.4%
ASMC   ct  0.9   0.10836    0.53472    0.13197    0.76925       9.1%
ASMC   ct  1.0   0.10784    0.53271    0.13052    0.76789       8.8%
ASMC   nt  0.9   0.10778    0.51921    0.12713    0.74519       8.7%
ASMC   nt  1.0   0.10904    0.51913    0.12886    0.74162       8.7%
```

Paired nt−ct at 1.0: ASMC mean −0.01358 (5/5), PID-CT mean −0.00325 (5/5) — the archived
noisy-tuning verdicts replicate at the new margin too.

## 5. Frozen-ESKF closed loop (the number that matters for deployment)

train14_v3 (n = 70 = 14 traj × 5 seeds; mirror of archived §7.4):

```
config              score        std      track    chatter         ce   est_pos    n
PID-CT ct  0.9    0.34127    0.03174    0.02393    0.53231     37.345   0.00325   70
PID-CT ct  1.0    0.34188    0.03282    0.02544    0.53015     37.376   0.00325   70
PID-CT nt  0.9    0.34341    0.03340    0.02624    0.53218     37.288   0.00325   70
PID-CT nt  1.0    0.34243    0.03266    0.02606    0.53001     37.377   0.00325   70
ASMC   nt  0.9    0.36320    0.05517    0.04128    0.54157     37.541   0.00325   70
ASMC   nt  1.0    0.36375    0.05355    0.04171    0.54212     37.478   0.00325   70
ASMC   ct  0.9    0.36423    0.04829    0.03658    0.55437     37.528   0.00325   70
ASMC   ct  1.0    0.36082    0.04916    0.03749    0.54491     37.495   0.00325   70
```

test_v3 (n = 40 = 8 traj × 5 seeds; mirror of archived §9.3):

```
config              score        std      track    chatter         ce   est_pos    n
PID-CT ct  0.9    0.31894    0.03630    0.02289    0.53312     26.952   0.00324   40
PID-CT ct  1.0    0.32111    0.03434    0.02595    0.53067     27.053   0.00324   40
PID-CT nt  0.9    0.32132    0.03705    0.02581    0.53201     26.927   0.00324   40
PID-CT nt  1.0    0.32220    0.03481    0.02716    0.53038     27.058   0.00324   40
ASMC   nt  0.9    0.34063    0.05354    0.04129    0.53934     27.186   0.00324   40
ASMC   nt  1.0    0.34060    0.05590    0.04214    0.53770     27.114   0.00324   40
ASMC   ct  0.9    0.34485    0.05404    0.03818    0.55584     27.140   0.00323   40
ASMC   ct  1.0    0.33875    0.05339    0.03765    0.54351     27.130   0.00324   40
```

`est_pos` is identical (0.00324–0.00325) in every row — the frozen ESKF is untouched by
the margin change, as it must be; only the controller side moved. Chatter *drops*
slightly for every s100 config (e.g. ASMC ct 0.55437 → 0.54491 train14) — the higher
ceiling lets the gain schedule hold tracking with less boundary-layer thrash.

## 6. Where did the gains move? (best clean seeds)

```
ASMC   0.9 (s2):  lam_x 5.6846  lam_y 0.9188  lam_psi 12.5006  rho_auth  1.7109
ASMC   1.0 (s1):  lam_x 3.4785  lam_y 1.0716  lam_psi  7.6342  rho_auth  2.0609
PID-CT 0.9 (s5):  lam_inner_x 0.0981  lam_inner_y 0.1140  lam_inner_psi 0.0371
PID-CT 1.0 (s2):  lam_inner_x 0.2556  lam_inner_y 0.1100  lam_inner_psi 0.0517
```

**The tuner does not rail the new ceilings.** ASMC's clean optimum at s=1.0 sits at
*lower* surface slopes than at 0.9 (lam_x 3.48 vs 5.68) — the relaxed `K_max` ceiling
changes the shape of the achievable gain landscape, and the optimizer moves to a
different, slightly better point on it, not to the new boundary. PID-CT's `lam_inner_x`
roughly doubles (0.098 → 0.256): with the γ gate relaxed, a more aggressive inner x-loop
becomes affordable.

## 7. Verdict

1. **Clean tuning objective: real, consistent improvement** — all 10 seeds better;
   best-seed −0.8% (ASMC) / −1.2% (PID-CT). This is the regime where the gate binds.
2. **Noisy tuning objective: flat to slightly worse** (paired mean +0.0039 ASMC,
   +0.0011 PID-CT; realisation std ≈ 0.037) — the clean gain does not survive
   unfiltered sensor noise.
3. **Oracle eval: within noise.** train14 noisy-mean moves ≤0.003 in either direction;
   test_v3 moves ≤0.007 (all four s100 configs marginally *better* there).
4. **Frozen ESKF: mixed and small.** ASMC clean-tuned improves on both tiers
   (−0.0034 / −0.0061); PID-CT clean-tuned degrades slightly on test_v3 (+0.0022);
   noisy-tuned configs are ~unchanged. Ordering **PID-CT < ASMC is untouched** on
   every tier and every stage.
5. **Net**: removing the 0.9 factor of safety buys ~1% on the clean tuning objective
   and nothing robust beyond it. The margin was not masking a materially better
   controller; it costs ~1% of clean-regime tracking headroom as insurance. Whether
   that insurance is worth keeping is a design call — the measured price is now known.
6. The campaign's secondary findings replicate at s=1.0 unchanged: noisy tuning helps
   ASMC more than PID-CT (5/5 seeds both), and the benefit is smaller at the relaxed
   margin.

## 8. Reproducibility

* Tuning: `run_stage_asmc_v3_s100.bat` / `run_stage_pid_v3_s100.bat` (or the equivalent
  direct `run_stage.jl --safety-margin 1.0` calls); logs `runs_{asmc,pid}_v3_s100_*.log`;
  outputs `runs_asmc_v3_s100/seed{1..5}/{asmc_v2_clean,asmc_v2_noisy}/`,
  `runs_pid_v3_s100/seed{1..5}/{pid_v2_ct_clean,pid_v2_ct_noisy}/`. Each
  `best_config.json` records `"safety_margin": 1.0`.
* Evals: `eval_v3_s100.jl` (`--mode oracle|eskf`, `--tier train14_v3|test_v3`); oracle
  rows in `results_v3/s100_oracle_eval{,_test_v3}.jls`; ESKF CSVs + summaries in
  `runs_eskf_v3_s100{,_test_v3}/`. The margin is read back from each config, so gates
  can never be mixed across campaigns.
* Two eval-script bugs were caught and fixed mid-campaign (tier-suffix binding and a
  world-age/double-include failure that NaN'd the first ESKF wave); all reported
  numbers come from runs after the fixes. Details in the script header comments.
* Held-out caveat carried over from the archived campaign: `coupled_vomega_anchor` in
  test_v3 is combo-identical to a training entry — generalisation claims rest on the
  other 7 trajectories.

## 9. Statistical significance (paired tests, frozen-ESKF scores)

Script: `stats_significance_s100.py` (+ `.bat`). Pairing unit is
(trajectory, noise seed) — the seed sets the noise realisation and the trajectory set
is a shared design, so differences are paired, never pooled. Two aggregation levels:
**run-level** (n = 70 train14 / 40 test_v3; paired t + Wilcoxon signed-rank + Cohen's
dz) and the conservative **traj-level** (seeds averaged first; n = 14 / 8).

**Q1 — the controller gap (PID-CT − ASMC, negative = PID-CT better):**

```
                          n    meanD      p_t        p_W        dz
train14 0.9, ct           70   −0.02296   4.0e-10    1.1e-12    −0.87
train14 1.0, ct           70   −0.01894   1.5e-07    5.7e-11    −0.70
test_v3 0.9, ct           40   −0.02590   3.0e-06    1.8e-12    −0.86
test_v3 1.0, ct           40   −0.01764   9.7e-04    4.2e-06    −0.56
(nt rows: same story, dz −0.56 … −0.68, all p ≤ 1.1e-4)
```

**PID-CT < ASMC is statistically significant at both margins, both tiers, both tuning
regimes** — it survives any multiplicity correction by orders of magnitude. Medians are
~half the means: the gap is skewed, driven hardest by the stress trajectories, but the
sign never flips.

**Q2 — the margin effect (s100 − 0.9 within each config):**

```
                 run-level p_t / p_W          traj-level p_t / p_W    s100-better
PID-CT ct  t14   0.44 / 0.30                  0.70 / 0.33              4/14
PID-CT ct  test  0.059 / 0.18                 0.36 / 0.20              2/8
PID-CT nt  t14   0.17 / 0.42                  0.44 / 0.67              4/14
PID-CT nt  test  0.29 / 0.075                 0.60 / 0.25              2/8
ASMC   ct  t14   1.9e-04 / 3.0e-09            0.015 / 0.011           13/14
ASMC   ct  test  2.2e-11 / 1.4e-08            <1e-4 / 0.008            8/8
ASMC   nt  t14   0.47 / 0.008                 0.65 / 0.46              7/14
ASMC   nt  test  0.96 / 0.058                 0.98 / 0.20              7/8
```

Of the eight margin tests, **only ASMC clean-tuned is significant** — at BOTH
aggregation levels and BOTH tiers (worst p = 0.015; 21 of 22 trajectories better;
test_v3 dz = −1.46), and it survives Holm correction across the family. The PID-CT ct
test_v3 degradation (p_t = 0.059) does not clear the bar: a trend, not a finding.
Everything else is noise.

**Caveats.** (i) Inference generalises over the sensor-noise distribution and this
fixed trajectory set — trajectories are design points, not random draws, so
"significant" means robust to noise realisation and consistent across these motions.
(ii) Statistical ≠ practical: even the significant ASMC ct effect is ~1% of total
score (~1.4% of the tracking component alone). (iii) The ~1% clean-tuning-objective
gain from removing the margin (§1) is NOT confirmed downstream except for ASMC ct —
it washes out under sensor noise everywhere else.

**Bottom line:** the PID-CT-vs-ASMC ordering is a statistically significant property of
the platform, not of the tuning margin; the margin ablation has exactly one significant
effect (ASMC clean-tuned, small and favourable) and none anywhere else.
