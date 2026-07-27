# Session inferences — log-domain slip loss: gradient redistribution, ε choice, label-order fix

Continues `chat-handoff/log_domain_slip_loss_handoff.md` (the log-slip redesign that
`SESSION_INFERENCES_hy3_gammakin.md` explicitly excludes). No training run yet — this session is
the loss-shape analysis, the ε decision, and a label-pipeline bug found while validating it.

All gradient fractions below use ONE definition so the forms are comparable: a constant-**relative**
error model (residual = δ·v), under which a sample's gradient contribution ∝ (v·f'(v))², where
f is the loss's transform of |Vp|. vpm distribution from the frozen scaler (13.5 M samples) plus the
stick_frac knot: p50 0.0247, p95 0.17538, p99 0.98976, max 3.638, F(0.01)=0.224.

## 1. The linear loss spends 73.5% of its gradient on 1% of the data

```
gradient fraction by |Vp| band     stick<1cm/s   gate..p50   p50..p95   p95..p99    >p99
  (n% of samples)                       22.4%       27.6%      45.0%       4.0%     1.0%
---------------------------------------------------------------------------------------
linear MSE / p95^2                        0.0%        0.1%       6.7%      19.7%    73.5%
log additive   e=0.0100                   4.7%       20.2%      66.2%       7.1%     1.8%
log additive   e=0.0173                   2.8%       15.6%      70.8%       8.5%     2.2%
log additive   e=0.0247                   2.0%       12.6%      73.1%       9.8%     2.6%
half-square    e=0.0100                   2.5%       23.0%      66.7%       6.2%     1.5%
half-square    e=0.0173                   0.6%       13.3%      76.5%       7.6%     1.9%
half-square    e=0.0247                   0.2%        7.3%      81.4%       8.9%     2.2%
```

- **Linear MSE puts 93.2% of its gradient above p95 and 0.0% in the stick band** — the population the
  gate decision is actually about. This is the mechanism behind the observed w_slip=1.0 tradeoff
  (fixes S2 gate, worst γ, worst OOD γ): raising w_slip buys gate calibration only as a side effect
  of pouring 50× more weight onto the large-slip tail.
- **Any log form collapses the tail from 93.2% to ~8%** and moves 20–25% of the gradient into
  the gate-adjacent bands. That is the whole intervention; it is a redistribution, not a rescale.
  Confirms handoff decision #1 (shape, not normalizer) quantitatively.

## 2. Half-square `½·log(v²+ε²)` is a band-pass; additive `log(v+ε)` is a low-pass

```
additive:     f = log(v+ε)        f' = 1/(v+ε)       weight peaks at v = 0
half-square:  f = ½·log(v²+ε²)    f' = v/(v²+ε²)     weight peaks at v = ε/√3
```

The additive form's attention *saturates at maximum* below the gate — where `g = σ((|Vp|−0.01)/0.01)`
is already saturated and the error is inert. The half-square form rolls off to zero there instead.
At matched ε=0.01 the two spend the same total near-gate budget but distribute it differently:

```
                     below gate (wasted)   gate..p50 (useful)
  additive e=0.01           4.7%                20.2%
  half-square e=0.01        2.5%                23.0%     <- dominates
```

Half-square also drops the `sqrt` entirely (no hypot on either side, `LG_EPS_REG` guard unnecessary),
and is what the boundary-weight fallback (weight ∝ ∂g/∂|Vp|) was trying to achieve by hand.

**Cost:** `dL/ds → 0` as the *prediction* s → 0, so collapse-to-zero-slip has no restoring gradient.
Not judged live — the measured bias is over-prediction in stick on all 24 baseline rows, and γ
supervision pins s independently — but it is the failure mode to watch in the first run.

## 3. ε = 0.01 (the gate width). Placing the *peak* at the gate is the wrong objective.

An intermediate recommendation of ε = √3 × 0.01 = 0.0173 (which puts the sensitivity peak exactly at
the gate center) is **rejected**. Peak location is not the figure of merit — gradient *share* in the
gate band is, and ε larger than the gate drains that share monotonically into p50..p95:

```
half-square, gate..p50 gradient share:   e=0.0100 -> 23.0%   e=0.0173 -> 13.3%   e=0.0247 -> 7.3%
```

ε = 0.01 also keeps its other two jobs coherent: it is the log-singularity guard (vpm has exact
zeros, `raw_min = 0.0`) and it bounds the per-sample loss at `(log(3.648/0.01))² ≈ 34.8` versus
`(3.638/0.17538)² ≈ 430` for the linear form — a 12× smaller worst case, bounded by construction.
Do not go below ε = 0.005: gradient magnitude scales as 1/ε (worst case `2·log(ratio)/ε`), and the
loss starts fitting unidentifiable near-zero noise.

**Final form:**
```
L = mean( ( ½·log( (Vpx̂² + Vpŷ² + ε²) / (Vpx² + Vpy² + ε²) ) )² )        ε = 0.01 m/s
```
`w_log` still needs the early-epoch gradient-ratio measurement; the ¼ from the half-square plus the
changed E[w] means the additive-form estimate (≈0.02) does not carry over.

## 4. Label-pipeline bug: the vpm label applies hypot and the LPF in the wrong order

`data_v2hy3.py:105` stores `vpm = decimate(hypot(Vpx, Vpy))` — magnitude formed at 2 kHz, *then*
anti-alias filtered. `hypot` is convex, so by Jensen this is biased HIGH versus the
component-consistent `hypot(decimate(Vpx), decimate(Vpy))`, which is what the model path produces
(it builds |Vp| from already-decimated `Vpx0_hat`/`Vpy0_hat`). Label and prediction currently
commute the nonlinearity in opposite orders.

Measured over the **full** test splits (89.5 M samples, all 1548–1579 files, via the new sidecars):

```
                       S1                      S2
vpm p50      0.029314 -> 0.028813 (-1.71%)   0.026280 -> 0.025719 (-2.14%)
vpm p95      0.149058 -> 0.148810 (-0.17%)   0.132753 -> 0.132472 (-0.21%)
stick_frac   0.231645 -> 0.255996 (+10.51%)  0.284507 -> 0.306529 (+7.74%)
crossings    slip->stick 2.44 pp             slip->stick 2.20 pp
             stick->slip 0.00 pp             stick->slip 0.00 pp
```

- Jensen holds exactly — **zero** stick→slip crossings on either fold (max apparent violation 1.9e-08
  is float32 storage rounding in the sidecar).
- Bias is monotone in band, spanning three orders: median relative gap 0.24–0.57% in the stick band,
  0.003–0.004% above p95. **p95/p99 move < 0.25%, so the frozen scaler `vpm_scale = 0.17538` needs
  no revision** and nothing about normalization or the ε choice is affected.
- **A 50-file subsample underestimates this 4–8×** (gave +1.35%/+2.14%). Use the full split.

**Effect on the existing ablation — common-mode, ranking-preserving, but the headline is wrong.**
`true_stick_frac` rises; the model under-predicts stick in all 24 rows; so every run's error grows:

```
lam0 stick_frac err%     as-reported   relabelled
  S1/noslip                  14.1        22.3
  S1/slip02                   8.4        17.1   <- still wins S1
  S1/wslip1                  11.6        20.0
  S2/noslip                  22.8        28.3
  S2/slip02                  22.6        28.2
  S2/wslip1                   3.9        10.8   <- still wins S2
```

Ordering is preserved on both folds, so the log-vs-linear comparison remains valid. But
**w_slip=1.0's "4% gate calibration" is really ~11%** — nearly 3×. The handoff's success criterion
("match wslip1's ≈4%") targets a number that does not exist and must be restated. Note this is a
row-level population; the attribution probe uses windowed endpoints (n=176k vs 89.5M), so exact
figures need the sidecar wired into the loader and `regime_split_attrib` re-run.

## 5. Sidecar rebuild (done)

`observer_v1_py/build_vp_components.py` writes `Vpx_true`/`Vpy_true` [T,4] float32 per file as
`<arrow>.vpcomp.npz` in `C:/Users/vishv/mecanum_cache_decim`. **5,949 files, 2.83 GiB, 177 s** at
jobs=6. The existing cache is untouched — warm `hy3in` npz and the `_noslip/_slip02/_wslip1`
checkpoints stay valid.

Filter is `scipy.signal.lfilter` with the one-pole coefficients of
`sensor_frontend_v2._causal_lpf` (200 Hz cutoff), verified bit-identical to the reference Python
loop at **1.7e-16** max abs diff over 3 files × 8 channels.

A full cache rebuild was **not** needed: the front-end quantities are unchanged, only the true
labels were missing. Sidecar cost 3 min versus a ~15 GiB regeneration.

## 6. RESULT — the log slip loss wins on the gate, on both folds

Run `_grnd80_slipLOG`: half-square log, eps=0.01, **w_log = 0.283**, `use_vp_components=True`,
80 grounding epochs, otherwise byte-identical to `_wslip1`. w_log was set by measuring the
init gradient norm on the shared trunk via `--cosine-diag`:

```
kind=mse   |dgamma| 1.872   |dslip|  85.71      -> w_slip 1.0  runs the slip term at ~46x gamma's gradient
kind=log   |dgamma| 1.613   |dslip| 302.6       -> w_log = 1.0 * 85.71/302.6 = 0.283
```

Holding gradient magnitude at `_wslip1`'s level makes SHAPE the only variable. (Retro-finding:
`w_slip=0.02` = 1.872/85.71 is almost exactly the gradient-matched-to-gamma point, which is why
`_slip02` was the best-behaved linear run on gamma; `_wslip1` at ~46x is why it wrecked gamma/OOD.)

**lam=0 (deployable), cross-fold test, ALL runs re-scored on corrected labels:**

```
variant   fold   sf err%   g_rmse   g_stick   hi>0.6  |  OOD sf err%   OOD g_rmse
noslip    S1      -22.7     1.145    0.709     6.607  |     -34.2        0.590
noslip    S2      -30.4     1.211    0.919     5.033  |     -48.2        0.608
slip02    S1      -17.5     1.104    0.659     5.727  |     -37.5        0.525
slip02    S2      -30.2     1.411    0.859     6.174  |     -48.7        0.676
wslip1    S1      -20.4     1.304    0.794     5.091  |     -40.0        0.764
wslip1    S2      -13.4     1.601    0.726     5.706  |     -45.0        0.774
slipLOG   S1       -9.4     1.333    0.604     7.782  |     -23.8        0.665
slipLOG   S2       -9.6     1.475    0.646     7.481  |     -19.2        0.514
```

1. **Beats every baseline on BOTH folds** (-9.4/-9.6 vs best-per-fold -17.5/-13.4), and the
   FOLD DISAGREEMENT DISAPPEARS. Linear swung by fold (slip02 -17.5->-30.2, wslip1 -20.4->-13.4),
   which is why the handoff could not pick a winner; slipLOG lands within 0.2 points. The
   consistency is a stronger result than the margin.
2. **Best stick-band gamma of any variant** (0.604/0.646 vs 0.659-0.919) while OVERALL gamma got
   worse -- accuracy migrated to where the gate decides. Mechanism confirmation, not coincidence.
   Note this happened WITHOUT touching the gamma loss.
3. **OOD improved rather than regressed** (-23.8/-19.2 vs -34..-49; OOD gamma 0.665/0.514 vs
   noslip 0.590/0.608 -- S2 better). `_wslip1`'s OOD gamma damage does not reappear.

**Costs, stated plainly.** Overall gamma +16%/+22% vs noslip (1.145->1.333, 1.211->1.475) = 0.19
rad/s = **2.1% of the gamma median (9.18), 0.23% of p95**. Small on the quantity's own scale, but
the handoff's "keep gamma near the noslip/slip02 level" criterion FAILS. Gate and OOD criteria are
both exceeded. **The pre-registered tail trigger fired on S2**: `high>0.6` +48.6% vs noslip
(S1 +17.8%), past the >30% threshold set before the run. In absolute terms 7.5 cm/s on |Vp|>60
cm/s (~12% relative, 1.3% of samples, gate saturated there) -- judged not worth acting on, but
recorded because it was pre-committed. Remedy if ever needed: small linear term as a tail floor.

**lam=1 -> lam=0 costs almost nothing on the gate** (slipLOG -10.9 -> -9.4; it IMPROVES, since
pred and true shift together). Gamma is what pays for the phase-out (0.958 -> 1.333, ~39%),
uniformly across variants.

## 7. The relabel OVERTURNED the OOD conclusion (bigger effect than in-distribution)

Multisine `true_stick_frac` **0.0069 (old label) -> 0.0757 (corrected) = 11x**, far larger than
the +7.7..10.5% seen cross-fold. Every OOD number in the handoff flips sign:

```
multisine, lam0     OLD label err     CORRECTED err
noslip S1              +566%             -34.2%
slip02 S1              +464%             -37.5%
wslip1 S1              +452%             -40.0%
```

So the "gate is catastrophically miscalibrated OOD (350-570%)" finding was **largely a label
artifact**. Physically consistent: multisine drives rapid slip-direction reversals, which is
exactly when LPF(hypot) >> hypot(LPF). Anything quoting the old OOD gate numbers must be redone.

**Tooling:** `regime_split_attrib.py` gained `--use-vp-components` (+ `--no-...`). REQUIRED for
any comparison spanning the sidecar boundary: pre-sidecar checkpoints have no
`use_vp_components` field, so cfg rebuilds it as False and they would be scored against the OLD
`true_stick_frac` while slipLOG got the corrected one -- manufacturing a difference from the label
change alone. Output is tagged `_vpc` so the original baseline JSONs are preserved.

## 8. gamma will NOT benefit from a log loss (asked and checked)

Similar p95/p50 ratio (9.0 vs 7.1) but that is BULK spread, not tail. The pathology log fixes is
TAIL DOMINANCE, which comes from the tail ratios:

```
          abs_p50   abs_p95   abs_p99   abs_max   p99/p95   max/p95    raw_min
gamma      9.176     82.81     107.02    139.31     1.29      1.68     -139.31
vpm        0.0247     0.1754     0.990     3.638     5.64     20.74        0.0

LINEAR-MSE gradient share:   bulk<p50   p50-p95   p95-p99   top 1%
vpm                              0.1%      6.7%     19.6%    73.6%
gamma                            0.8%     68.6%     21.5%     9.0%
worst single sample vs p95:  vpm 428.9x   gamma 2.8x
```

gamma is effectively BOUNDED ABOVE; its gradient already sits where the data is. Nothing to fix.
Plus three blockers: (a) gamma is SIGNED (raw_min -139.31) so log is undefined -- arcsinh would be
needed; (b) the trained target is not gamma but the near-zero-centered residual dgamma
(RMS 3.24, p95 6.49), and relative accuracy is meaningless for a quantity that crosses zero;
(c) there is no decision boundary on gamma -- the whole log-slip justification was the gate at
0.01 m/s. **And the fix is already in place**: `gamma_noslip` subtracting the closed-form base is
structurally the SAME intervention (stop spending capacity predicting a big number precisely),
worth 19x in stick. Empirical corroboration: slipLOG produced the best stick-band gamma of any
variant without touching the gamma loss at all.

If a gamma-side experiment is wanted, the defensible one is `dgamma_scale = gamma_p95/10 = 8.28`,
which the config itself flags as discretionary.

## 9. Open / next

- Wire `.vpcomp.npz` into `data_v2hy3.py` behind a flag; re-run `regime_split_attrib` (EVAL-ONLY —
  loads `checkpoint_best.pt`, no training) over the 6 existing baseline run dirs
  (`_noslip/_slip02/_wslip1` × S1/S2) for exact relabelled reference numbers on the windowed
  population. **No baseline is retrained.** The only training this redesign needs is the one new
  slipLOG experiment = 2 fold-runs.

  Justification that the relabel does not confound that comparison: `vpm` enters training through
  exactly one path — `slip_consistency_loss`, gated on `w_slip > 0`
  (`training_v2hy3_gammakin.py:168,383`) — and `gamma_high_slip_upweight = 1.0` makes `_sup_weight`
  a no-op. So `_noslip` training never saw `vpm` at all, and for `_slip02`/`_wslip1` the bias sits
  in the stick band where the linear loss carries **0.0%** of its gradient (§1); where their
  gradient does live (>p95, 93.2%) the bias is 0.003–0.004%. Same label error is negligible for
  linear and material for log — log puts ~25% of its gradient in the bands the bias occupies. That
  asymmetry is why the sidecar had to precede the slipLOG run.
- **IN FLIGHT:** `_grnd80_slipLOG02eq`, S1+S2, `w_log = 0.00566` = 0.02 x (85.71/302.6), the
  slip02-equivalent gradient. With `_noslip` as the w=0 point this gives a 3-point weight ablation
  for the log loss: **0 / 0.00566 / 0.283**. `_noslip` is a VALID zero without retraining --
  `vpm` reaches training only via `slip_consistency_loss` gated on `w_slip > 0`, so that run never
  saw the label under either scheme.
- Then: is -9.5% the floor, or just where w_log=0.283 lands? The ablation answers this.
- Single seed per fold throughout. Before publication, repeat the winner at >=2 more seeds.
- Neither pre-registered failure mode fired: no `s -> 0` collapse, no gradient blowup, no NaN.
  eps=0.01 held. The `high>0.6` trigger DID fire on S2 (see section 6).
- The linear term is **not** retained alongside log: |Vp| is redundant-at-optimum (a function of the
  supervised γ̂, ΔV̂), and w_slip 0→1.0 does not consistently improve the >0.6 band anyway
  (S1 6.607→5.091 cm/s, but S2 5.039→5.710 — one fold each way, no monotone trend).
