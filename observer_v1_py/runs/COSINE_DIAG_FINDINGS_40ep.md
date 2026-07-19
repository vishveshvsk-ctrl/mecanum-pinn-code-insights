# Gradient-conflict (cosine) diagnostic — 40-epoch grounding samples

**Status:** reference log, not a conclusion. Two 40-epoch **grounding-only** runs
(physics OFF), stride 16, w32, batch 4096, lr 2e-3, S1 fold. Both runs were **still
descending at ep39** — no convergence claim is made here.

- Run A `S1_train_hy3_grnd_cosine` — w_dv=1, **w_slip=0** (no slip term)
- Run B `S1_train_hy3_grnd_cosine_slip` — w_dv=1, **w_slip=0.1**, base-bug FIXED

Raw per-heartbeat traces: `<run>/cosine_diag.json`. Enable with
`--cosine-diag [--cosine-diag-every N] [--cosine-diag-batches M]` (default OFF).

## What is measured

`cos(grad L_a, grad L_b)` over the flattened gradient w.r.t. the **shared trunk**
(encoder + feat + wheel_emb; both heads excluded — same param set GradNorm uses).
Cosine is **scale-invariant**, so loss weights cannot affect it: it isolates
representation geometry from the gradient-magnitude effect GradNorm reacts to.

## Run A (no slip) — γ vs ΔV

```
 ep  cos_mean  cos_med  negfrac      |dLg|      |dLdv|   ratio     g_mse    dv_mse
  0   -0.0015  +0.0634    0.450  1.0814e+00  2.9865e-01   3.62   0.07523  0.002962
 10   -0.0080  -0.0347    0.600  6.5235e-01  2.1903e-02  29.78   0.00695  0.000242
 20   +0.0303  +0.0563    0.350  3.3086e-01  7.2240e-03  45.80   0.00512  0.000145
 30   +0.0489  +0.0399    0.350  2.9494e-01  7.0317e-03  41.94   0.00413  0.000114
```

1. **No representation conflict.** |cos| <= 0.05 at every heartbeat, drifting slightly
   POSITIVE; neg_frac wanders 0.45/0.60/0.35/0.35 with no trend (the ep10 tilt reversed
   — it was noise on 20 batches). The heads are **orthogonal**. A real conflict would
   pin cos ~ -0.3..-0.5 with neg_frac >> 0.5. **=> keep the shared encoder; a split /
   per-wheel adapter is not justified by this evidence.**
2. **Extreme convergence asymmetry, magnitude-only.** |grad dV| collapsed 42x and
   plateaued by ep20; |grad gamma| fell only 3.7x and was still moving at ep30.
   Different *difficulty*, same features.
3. **This fully explains GradNorm crushing wg.** GradNorm equalizes weighted norms, so
   `w_g/w_dv = |grad dV|/|grad gamma| = 1/41.94` -> **wg ~ 0.047** from magnitude alone
   (sum-to-2 renorm), matching the observed ~0.03. The `r^alpha` rate term does NOT
   drive it — in normalized terms dV converged MORE (26.1x) than gamma (18.2x), so
   r_gamma=1.176 > r_dv=0.824 and the rate term *up-weights* gamma (wg 0.047 -> 0.078),
   i.e. it mildly OPPOSES the crush. The handoff's original read ("gamma has the larger
   gradient, GradNorm damps the priority task") is correct.
4. Explains the old `w_dv=10` result: dV's gradient is already ~42x smaller and
   orthogonal (harmless); forcing it up 10x steals trunk capacity for a task that
   finished at ep20 — hence gamma degraded 0.008 -> 0.027.

## Run B (w_slip=0.1, base FIXED) — 3-way

```
 ep     g_dv   g_slip  dv_slip       |dg|      |ddv|    |dslip|
  0  -0.0008  +0.4872  +0.6243  1.209e+00  4.359e-01  5.674e+01
 10  +0.2588  +0.8366  +0.2189  6.027e-01  4.970e-02  1.126e+01
 20  +0.3420  +0.7604  +0.2399  2.958e-01  2.517e-02  4.617e+00
 30  +0.3677  +0.8131  +0.2339  2.233e-01  1.554e-02  4.048e+00
```

5. **gamma<->dV stays non-conflicting** with the slip term present (ep0 -0.0008
   reproduces Run A), and turns cooperative (+0.37 by ep30, neg_frac 0.05).
6. **w_slip is NOT a plain fraction — |grad slip| is huge.** At ep0 |grad slip|=56.7 vs
   |grad gamma|=1.21 (47x). Amplification is structural: `slip_consistency_loss`
   divides by `vpm_scale^2 = 0.0308` (32x) and the chain rule runs through the gamma
   de-normalization (gamma_std ~ 82.8). The ratio |grad slip|/|grad gamma| decays with
   training: **ep0 46.9 -> ep10 18.7 -> ep20 15.6 -> ep30 18.2**, settling ~18.
   ```
   target                      w_slip @ep0     w_slip @steady-state
   slip ~ 1.0 x |grad gamma|      0.021               0.055        ("balanced")
   slip ~ 0.1 x |grad gamma|      0.0021              0.0055       (1/10th regularizer)
   ```
   The handoff's `w_slip ~ 0.02` (derived from early-epoch MSE) is confirmed as the
   EARLY balance point by an independent measurement. NOTE these are extrapolated from
   a w_slip=0.1 trajectory; at lower weight the slip gradient decays less.

7. **CORRECTION to the handoff's mechanism for "slip corrupts dV" (decision #3).**
   The corruption is REAL but it is **not gradient opposition** — `cos(dv,slip)` stayed
   **POSITIVE (+0.22..+0.24)** the whole run and never went negative. The mechanism is
   **weak alignment x large magnitude**:
   ```
   cos(gamma, slip) = +0.81  -> 81% aligned with gamma's optimum   -> HELPS gamma
   cos(dv,    slip) = +0.23  -> only 23% aligned with dV's optimum -> ~97% of a
                               DOMINANT gradient points off-axis and drags dV away
   ```
   **A positive cosine does not mean harmless.** Read `alignment x magnitude`, not the
   sign. Watching for a negative cosine would miss this entirely.

8. **Head-to-head at ep39 (both still descending, LR 2e-3 throughout):**
   ```
        metric        A(no slip)   B(w_slip=0.1)   B/A
     train g_mse        0.003720      0.003510     0.94x   slip HELPS gamma  -6%
    train dv_mse        0.000100      0.000430     4.30x
       val g_mse        0.004890      0.004490     0.92x   slip HELPS gamma  -8% (VAL)
      val dv_mse        0.000100      0.000420     4.20x   slip COSTS dV     4.2x
    val dV RMSE(x)      1.92 cm/s     3.93 cm/s
   ```
   The B/A dv_mse ratio is **flat ~4x from ep6 onward** (both descend in parallel), so
   the gap is a persistent offset, not a mid-descent artifact. gamma's advantage is
   LARGER on val (-8%) than train (-6%) => the slip term is a genuine **regularizer**
   for gamma. But dV *is* the contact base (`Vpx0_u = Vpx0_hat + dV`), so degrading dV
   feeds straight back into the slip reconstruction the term exists to improve.
   **w_slip=0.1 is self-defeating.** Caveat: neither run converged (40 ep); the
   converged gap is strictly unknown.

## Slip-reconstruction error budget (the original question)

```
base double-count bug   ~86 cm/s      <- FIXED (was swamping everything, ~340x)
dV head error          1.9-3.9 cm/s   <- NOW the binding limit
decimation floor        0.16 cm/s     <- 12-25x below dV error; NOT the limiter
```
Decimation at 500 Hz was never the constraint. 1000 Hz would shave a term already an
order of magnitude below what actually binds.

## LR

Run A/B held **lr 2e-3 for all 40 epochs; the scheduler never fired**, and both losses
were still descending ~ -27%/10 epochs at ep39. For contrast, the OLD reference
`S1_train_hy3_grnd` collapsed 2e-3 -> 1.56e-5 (128x) with reductions at ep
11/22/33/44/55/66/77 — exactly `sched_patience=10 + 1`, the signature of a metric that
never improves (it scheduled on the physics-dominated terminal metric during grounding;
its `val_loss=981.8` confirms). **This is the first empirical confirmation that the
supervised-metric scheduler fix works.** No evidence favors raising LR — the binding
constraint is the epoch budget, not the step size.

## Open

- Cosine under **PHYSICS** is untested (these are grounding-only). The physics/slip path
  adds coupling absent here; re-measure before generalizing "no conflict".
- Whether a w_slip sweet spot exists (gamma's regularization gain at a fraction of the
  dV cost) — untested below 0.1.
