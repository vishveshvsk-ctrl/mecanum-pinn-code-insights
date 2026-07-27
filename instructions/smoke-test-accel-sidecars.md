# Smoke-test: Julia acceleration sidecars

After syncing the repo to the Windows live tree (`C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\`), run this minimal smoke test to confirm `datastore.jl` now emits `accel/<stem>_accel.arrow` sidecars.

## 1. Generate one trajectory with sidecar

From the WSL/Windows Julia environment, run the sweep driver on the smallest config set for one cheap profile:

```bash
cd /mnt/c/Users/vishv/OneDrive/Desktop/Vishvesh_Data/VNIT/mecanum_pinn_head/code_insights
julia --project=. -t 4 Data_Generation_Julia.jl \
  --config-dir trajectory_files_run_0p3_quad \
  --profiles octagon.toml \
  --limit 1 --no-resume
```

This writes one main Arrow file and, because `write_accel` defaults to `true`, one sidecar under `data/Simulation_Data_MecanumSlipSpin_LugreAdamov/accel/`.

## 2. Check the sidecar exists and has the right columns

```bash
ls data/Simulation_Data_MecanumSlipSpin_LugreAdamov/accel/octagon_c*_mu_0.3_case1_lugre_adamov_chi_*.arrow
```

Expected columns: `dVx`, `dVy`, `dpsidot`, `dw1`, `dw2`, `dw3`, `dw4`.

## 3. Run the Python verifier

Use the helper in the repo (`_tmp/verify_julia_sidecar.py`) against the file just generated:

```bash
python _tmp/verify_julia_sidecar.py \
  data/Simulation_Data_MecanumSlipSpin_LugreAdamov/octagon_c001_mu_0.3_case1_lugre_adamov_chi_0.000.arrow
```

It checks:
- sidecar exists at `accel/<stem>_accel.arrow`
- required metadata keys: `source_name`, `source_rows`, `source_fingerprint`, `eom_convention`, `generator_version`
- `source_fingerprint` matches Python's exact `os.stat(...).st_mtime_ns`
- row count matches the original
- all values are finite
- numerical agreement with exact-dynamics recomputation

Expected output: `RESULT: PASS`.

## 4. Cross-check against the Python reference verifier

The existing `tools_accel/make_accel_sidecars.py` verifier should also accept the Julia-generated sidecar because the fingerprint format matches:

```bash
python tools_accel/make_accel_sidecars.py --verify-only --limit 1 \
  --data-dir data/Simulation_Data_MecanumSlipSpin_LugreAdamov
```

Expected: the newly generated file reports `ok=1`.

## 5. If something fails

- **`stat failed` during sidecar write**: the Linux x86_64 `ccall(:stat, ...)` path did not work on your libc. The code falls back to `floor(Int, stat(path).mtime * 1e9)`, which can differ from Python's exact nanoseconds by a few hundred nanoseconds. In that case the Python verifier will report a fingerprint mismatch; switch `arrow_fingerprint` to a content-based scheme (e.g., hash of first/last MB) and update the Python verifier to match.
- **numerical mismatch**: almost certainly means `Mz_i` was not dropped from the body yaw balance or the wrong `p1` was selected. Check `meta.friction_case` is being passed correctly.
- **missing sidecar**: `write_outputs` was called with `write_accel=false`; check that `Data_Generation_Julia.jl` or the notebook cell uses the default keyword.
