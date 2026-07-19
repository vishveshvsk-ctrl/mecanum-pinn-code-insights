# Run: S1_train_hy3_w32_gamma_dv_v2hy3_gammakin_grnd150_slip02  (gamma-RESIDUAL variant)

- model=ssm window=32 stride=16 regime=S1_train_hy3
- **gamma parametrization:** residual off nd711 sec5.1 Model 1:
  `gamma_hat = gamma_noslip(V_y_used) + dgamma_hat * 8.2810`
  `gamma_noslip = -Vpy0_u / (cos_delta*(R*cos_tt - Rd))`  (V_Y and Omega only)
- **vy_label lambda:** 1.0 -> 0.0 (TRUE V_y label ramps out as physics ramps in); base_detach=True
- **dv_scale:** UNIFORM (1.296694143116474, 1.296694143116474) (isotropic; d|Vp|/ddV ~ 1 on both axes)
- **normalization:** MAX (frozen p95, 5 global) <- ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv
- w_dv=1.0 w_slip=0.02 upweight=1.0 gate=(0.01,0.01)
- AdamW lr=0.002 wd=0.0001; ReduceLROnPlateau factor=0.5 patience=10
