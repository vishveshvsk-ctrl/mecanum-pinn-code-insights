# Run: S2_train_w8_non_phys_max_norm

- model=ssm window=8 stride=4 regime=S2_train
- **normalization:** MAX (frozen p95; sin/cos unscaled) <- ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv
- **loss:** SUPERVISED ONLY (physics_loss=False)
- phases=a1_5phase phase_total_epochs=100; AdamW lr=0.002 wd=0.0001 grad_clip=1.0; no L-BFGS, no early stopping
