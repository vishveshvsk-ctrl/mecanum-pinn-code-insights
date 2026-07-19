# Energy / Power Balance Consistency Audit

Scripts for checking the energy consistency of the generated Arrow trajectory
data against the 39-D mecanum dynamics model
(`Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb`).

## What was investigated

The claim: energy lost to friction (`∫ F·dv`), viscous forces
(`∫ p1·ω² + p2·γ²`), and motor work (`∫ τ_motor·ω`) should balance against
ΔKE. Derived from the three KE reservoirs (platform, wheel, roller):

```
dE_KE/dt = P_fric_contact + P_motor − P_visc
P_fric_contact = Σ_i [ Fx_i·Vpx_i + Fy_i·Vpy_i + Mz_i·wz_i ]
P_motor        = Σ_i Msat_i · w_i
P_visc         = Σ_i [ p1·w_i² + p2·γ_i² ]
```

where `Vpx/Vpy/wz` are the STORED contact-point velocities (datastore.jl
computes them from the full `Vpi_x = Vx − ψ̇(py+DYi) − ω_i R + …`, so they
already include the DYi correction).

## Findings (on the active sweep, ~120-file samples)

1. **DYi arm omission is negligible.** In `dynamics_full_mf_asmc!` the
   platform yaw balance uses moment arm `py` (wheel center) instead of
   `py+DYi` (actual roller contact offset). The leaked power is
   `R_DYi(t) = Σ_i ψ̇·DYi_i·Fx_i`. Integrated over a trajectory it is
   **≤ 0.04% of motor energy**, peak **≤ 2 W**. Not the source of any
   meaningful imbalance. (`energy_audit_dyi.py`)

2. **Instantaneous power balance does NOT close to machine precision, even
   with DYi included.** Using the analytic `Ẇ_KE` from the acceleration
   sidecar (not finite-differenced velocity), the per-timestep relative
   power error `|Ẇ_KE − (P_fric+P_motor−P_visc)| / P_scale` has:
   - median **~0.15–0.3%**
   - integrated residual **~0.2–0.3%** of motor energy
   - p99 up to ~28%, isolated spikes (eps_max up to ~50%, a few files to
     ~370%) at torque-saturation / roller-handoff instants.

3. **The residual is a MODELING error, not integration error.** Dominant
   cause: `EOM_CONVENTION = "ne_rhs_v1_mz_dropped"` — the spin friction
   torque `Mz_i` is deliberately omitted from the platform yaw balance
   (`RHS2` in datastore.jl:180 and notebook Cell 22). Solver choice
   (TRBDF2 / Rodas5P / QNDF / FBDF) does NOT help: at `reltol 1e-9` the
   trajectory is already converged; the residual lives in the RHS, not the
   integrator.

4. **The training whitelist does NOT reduce the spikes.** Filtering to
   `combined_reco` starting with `keep` (4477 files) leaves the spike
   distribution statistically unchanged (median ~0.28% vs ~0.16% on the
   unfiltered sample). The whitelist gates chatter/burst/tracking, which is
   orthogonal to the power-balance residual.

## Scripts

- `power_balance_audit.py` — **primary tool.** Per-timestep power-balance
  audit using the accel sidecar for analytic `Ẇ_KE`. Reports eps_rms/med/
  p95/p99/max and integrated residual per file; `--trace` dumps resid(t).
- `energy_audit_dyi.py` — measures the DYi arm-leak term `R_DYi(t)` and its
  integral directly. Confirms it is negligible.
- `check_energy_closure.py` — earlier closure attempt (retained for
  reference; its reservoir-reconstruction path is buggy — use
  `power_balance_audit.py` instead).

## Usage

```powershell
# python env (CPU): C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe
# or Windows .bat launcher: energy_audit_dyi.bat  (mirrors the dye audit)

# Power-balance audit, first 120 files, write summary CSV
python power_balance_audit.py --limit 120 --out summary.csv

# Whitelist-filtered run (only 'keep*' rows in diagnostics_combined.csv)
python power_balance_audit.py --whitelist ..\diagnostics_combined.csv --limit 120 --out summary_wl.csv

# DYi leak measurement
python energy_audit_dyi.py --limit 120 --out dyi_summary.csv

# Dump resid(t) time series for one file
python power_balance_audit.py --trace "<path>/octagon_c042_....arrow"
```

Defaults resolve `DATA_DIR` to
`..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov` and `CONFIG_DIR` to
`trajectory_files_run_0p3_main` (for `p1/p2` + geometry from `base.toml`).
Override with `--data-dir` / `--config-dir`.

## What WOULD reduce the residual (not yet done)

1. Re-include `ΣMz_i` in `RHS2` (undo `ne_rhs_v1_mz_dropped`) — biggest fix.
2. Use `py + DYi` arm in `RHS2` — completeness (tiny effect).
3. Log the true applied `Mi_sat` at solver resolution instead of the
   2000 Hz recomputed `smooth_sat(Mi_sw+Mi_eq)` — kills the saturation
   spikes. This is a logging fix, not a solver fix.
