# Run: S1_train_w32_non_phys_max_norm_b1024

- model=ssm window=32 stride=16 regime=S1_train
- **normalization:** MAX (frozen p95; sin/cos unscaled) <- ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv
- **loss:** SUPERVISED ONLY (physics_loss=False)
- phases=a1_5phase phase_total_epochs=64; AdamW lr=0.002 wd=0.0001 grad_clip=1.0; no L-BFGS, no early stopping
