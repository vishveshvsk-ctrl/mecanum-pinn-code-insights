# Run: S2_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm

- model=ssm window=32 stride=16 regime=S2_train_hy3
- **normalization:** MAX (frozen p95, 5 global channels) <- ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv
- **loss:** γ + ΔV supervised + regime-split steady-state physics (5-phase ramp, W_SUP_MIN=0.1)
- w_dv=1.0 gate_center=0.01 gate_width=0.01 mindlin_iters=2
- vel_filter_crossover_hz=1.0 integrator=rot_ab2 noise_stage=none
- AdamW lr=0.002 wd=0.0001 grad_clip=1.0; ReduceLROnPlateau factor=0.5 patience=10 min_lr=1e-06
- warm_from=none
