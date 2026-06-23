# Run: S2_train_w8  (NON-PHYSICS, VAR-NORMALIZED)

## Loss regime
- **Supervised only** — `observer_loss` = mean over states of per-state MSE (z-scored units).
- `physics_loss = FALSE` -> physics weight 0 for all epochs. The 5-phase curriculum acted
  ONLY as an LR schedule (w_sup=1.0, w_phys=0.0 throughout).
- LR: AdamW base 2e-3 x {grounding 1.0, phys_rampup 0.25, overlap 0.5, grnd_rampdown 0.25,
  physics 0.10}; phase epochs 48/14/24/14/19 (--phase-epochs 120, total 119).
- AdamW (wd 1e-4), grad_clip 1.0. **No early stopping. No L-BFGS.**

## Data normalization
- **VAR-NORMALIZATION (z-score / standardization):** per-channel (x - mean)/std, fit on TRAIN
  files only (`norm.npz`). Applied to BOTH inputs (Vx,Vy,psi_dot; per-wheel Msat,w,sin_tt,cos_tt)
  AND targets (gamma,zx,zy).
- ** SUPERSEDED:** future runs use MAX-normalization (p95-based scaler). This checkpoint is
  NOT warm-start-compatible with max-norm runs (its weights + loss assume z-scored I/O).

## Result
- model=ssm window=8 stride=4 regime=S2_train epochs=119 params=6277
- final val_loss (z-scored mean per-state MSE) = 0.0341
- Cross-subset (held-out fold) per-state RMSE: observer_v1_py/report/state_observability.csv
