# Arrow Acceleration Sidecar Generation (exact dynamics, fleet-wide)

> **Generated:** 2026-07-09 (rev 3: sidecars move to an `accel/` SUBFOLDER with `_accel.arrow` suffix — invisible to non-recursive `*.arrow` globs by construction; rev 2: sidecars replace in-place column addition — originals never rewritten)
> **Stack:** Python 3.11 (torch-free venv `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`), pyarrow 24.0, numpy 2.4
> **Scope:** Data pipeline (one-time batch sidecar generation + non-interference verification; no training)

## 1. Overview

For every Arrow file in the active sweep
(`../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/`, 5,949 files, ~238 GB,
2000 Hz native), write a **sidecar Arrow file** `accel/<stem>_accel.arrow` — inside a
NEW `accel/` subfolder of the sweep directory — containing ONLY the exact-dynamics
acceleration columns: body accelerations
(`dVx, dVy, dpsidot` [T]) evaluated from the body EOM with the STORED per-wheel forces
(all inputs are existing columns — no ODE re-run), and wheel accelerations
(`dw1..dw4` [T,4]) from the wheel torque balance. **Originals are never opened for
writing — the data-immutability rule holds fully.** Sidecars carry Arrow custom
schema metadata: source filename, source row count, a source fingerprint (size +
mtime or content hash), the EOM-convention tag, and a generator version — so
consumers can verify pairing at load. Expected sidecar footprint: 7 float32 columns
× T rows ≈ 3–5% of the original fleet (~10–15 GB total). The run is idempotent
(valid sidecar present = skip) and resumable. The subfolder placement makes sidecars
invisible to every non-recursive `*.arrow` glob by construction; the residual risk —
RECURSIVE globs (`**/*.arrow`, `rglob`, Julia `walkdir`) anywhere in the consumers —
**must be verified absent by test, not assumed**, before the fleet run.

## 2. Architecture Pattern

Idempotent map-over-files batch job producing derived sidecar artifacts — each file
is an independent unit of work (read original → compute → tmp-write sidecar →
`os.replace`), originals are read-only throughout, and a crash leaves at worst a
`.tmp` to clean up, never a corrupted trajectory.

## 3. Technology Constraints

- **Python:** 3.11 torch-free venv (numpy/pyarrow only — the GPU env is not involved)
- **Required libraries:** pyarrow (IPC read/write + custom schema metadata), numpy (EOM evaluation), tqdm (progress)
- **Device targets:** CPU only; disk-read-bound
- **Explicit exclusions:** originals NEVER opened for writing (assert read-only access pattern; spot-check mtimes unchanged after the fleet run); NO pandas round-trip; ≤8 workers (default 4); physics constants from the verified A1 `physics.py` values / `base.toml` — never invented; sidecar location + naming is EXACTLY `accel/<original stem>_accel.arrow` (e.g. `accel/octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002_accel.arrow`; the `accel/` folder is created once at run start)

## 4. Component Breakdown

### `accel_dynamics.py` (module, `code_insights/tools_accel/`)
- **Type:** functions (pure numpy)
- **Responsibility:** Evaluate the exact-dynamics accelerations from stored columns: body EOM `[V̇x, V̇y, ψ̈] = M_body⁻¹ · (generalized force from stored roller-frame Fpar/Fperp + coupling terms)` and wheel balance `ẇ_i = (Msat_i − Fx_i·R − p1·w_i)/Jw` — a numpy port of the ALREADY-VERIFIED torch implementation in `Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/physics.py::ne_rhs` ("residual 0.000" header). Port, don't re-derive; cross-validate against the torch version on one real file to fp32 tolerance; **resolve the EOM transport-term convention against `run_one.jl` here** (authority rule) and record the convention tag that goes into sidecar metadata.
- **Inputs:** column arrays `Vx, Vy, psi_dot [T]`, `w, theta, Msat [T,4]`, `Fpar, Fperp [T,4]` (float64 internally)
- **Outputs:** `dVx, dVy, dpsidot [T]`, `dw [T,4]` (float32 for storage)
- **Depends on:** nothing

### `build_sidecar` (function in `make_accel_sidecars.py`)
- **Type:** function (per-file worker)
- **Responsibility:** Per-file pipeline: if `accel/<stem>_accel.arrow` exists AND its metadata fingerprint matches the current original → SKIPPED (idempotency; a stale fingerprint = rebuild); read the original (read-only); compute via `accel_dynamics`; assemble a 7-column table (`dVx, dVy, dpsidot, dw1, dw2, dw3, dw4`, all float32, T rows — no time column; alignment is by row index, guaranteed by the metadata row-count + fingerprint pairing); attach custom schema metadata {source_name, source_rows, source_fingerprint, eom_convention, generator_version}; write `accel/<stem>_accel.arrow.tmp` → `os.replace`; run `verify_sidecar`; on any failure, remove tmp — the original is untouched by construction.
- **Inputs:** original path, config
- **Outputs:** status record (path, status ∈ {DONE, SKIPPED, REBUILT, FAILED}, wall time, rows, FD-check residual)
- **Depends on:** `accel_dynamics`, `verify_sidecar`

### `verify_sidecar` (function in `make_accel_sidecars.py`)
- **Type:** function
- **Responsibility:** Post-write checks: (a) sidecar opens and row count equals the original's; (b) all values finite; (c) metadata complete and fingerprint matches; (d) **FD cross-check** — central finite difference of the original's stored `Vx, Vy, psi_dot, w` at 2000 Hz matches the sidecar columns to dense-output tolerance (per-file max/rms residual to the manifest; a systematic residual of magnitude ψ̇·Vy exposes a misread EOM convention).
- **Depends on:** `accel_dynamics`

### `test_sidecar_noninterference.py` (script, run BEFORE the fleet)
- **Type:** test script
- **Responsibility:** The compat gate, on **copies in `code_insights/_tmp/`**: build a mock data dir (originals at top level + generated sidecars in `accel/`), then (a) run the A1 discovery/load path (`Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/data.py` — whitelist + filename-contract parsing) and the A2 path (`observer_v1_py/mecanum_observer/data.py` — diagnostics_combined whitelist) and assert the sidecar folder is NOT traversed and loader outputs are identical with/without it present; (b) grep-audit BOTH packages plus the Julia modules (`datastore.jl`, `Data_Generation_Julia.jl`, `run_one.jl`) for RECURSIVE enumeration (`**/*.arrow`, `rglob`, `os.walk`, Julia `walkdir`) — non-recursive `*.arrow` globs are safe by construction now; every recursive hit must exclude the `accel/` folder or be listed as a required fix; (c) verify the pairing join: load original + sidecar, assert row alignment via metadata. Only a full pass unlocks the fleet run.
- **Depends on:** `build_sidecar`

### `make_accel_sidecars.py` (CLI script, `code_insights/tools_accel/`)
- **Type:** entry-point script
- **Responsibility:** Fleet orchestration: enumerate original `*.arrow` under `--data-dir` (default the active sweep; exclude `*.accel.arrow` from enumeration by suffix, and exclude DEPRECATED legacy dirs), `ProcessPoolExecutor(--workers, default 4, cap 8)`, per-file `build_sidecar`, streaming manifest CSV (`tools_accel/sidecar_manifest.csv`), `--dry-run`, `--limit N` (pilot), `--verify-only` (fleet-wide `verify_sidecar` re-run), disk-headroom guard (free ≥ 5% of fleet size), summary on exit.
- **Depends on:** all above

## 5. File & Directory Structure

```
code_insights/tools_accel/
├── accel_dynamics.py                 # numpy EOM port (verified vs A1 physics.py)
├── make_accel_sidecars.py            # CLI orchestrator + build_sidecar + verify_sidecar
├── test_sidecar_noninterference.py   # compat gate (runs on _tmp copies + glob audit)
└── sidecar_manifest.csv              # streaming run manifest (generated)
../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/
├── <name>.arrow                      # originals — READ-ONLY, byte-untouched
└── accel/                            # NEW subfolder (invisible to top-level globs)
    └── <name>_accel.arrow            # sidecars (7 float32 cols + pairing metadata)
```

## 6. Key Interfaces

```python
# tools_accel/accel_dynamics.py
def body_wheel_accels(Vx: np.ndarray, Vy: np.ndarray, psi_dot: np.ndarray,
                      w: np.ndarray, Msat: np.ndarray,
                      Fpar: np.ndarray, Fperp: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact-dynamics accelerations from stored columns (no ODE re-run).
    Args:   Vx/Vy/psi_dot [T]; w/Msat/Fpar/Fperp [T,4]
    Returns: dVx [T], dVy [T], dpsidot [T], dw [T,4]  (float32)
    """
    ...

# tools_accel/make_accel_sidecars.py
def build_sidecar(path: Path, cfg: SidecarConfig, dry_run: bool) -> FileResult:
    """Idempotent sidecar generation; original opened read-only; tmp+replace write."""
    ...

def verify_sidecar(orig: Path, sidecar: Path) -> VerifyResult:
    """Row count, finiteness, metadata/fingerprint pairing, FD cross-check residual."""
    ...

def load_with_accel(orig: Path) -> pa.Table:
    """Reference join helper for downstream loaders: original + its sidecar
    (resolved as <orig.parent>/accel/<orig.stem>_accel.arrow), alignment asserted
    via sidecar metadata (source_rows + fingerprint). Raises on mismatch/missing."""
    ...
```

## 7. Data Flow

1. CLI enumerates originals at the sweep TOP LEVEL only (the `accel/` subfolder is never enumerated) → skips fingerprint-valid sidecars.
2. Worker reads the original (read-only, float64 promotion for computation).
3. `body_wheel_accels` → 7 float32 arrays; sidecar table + pairing metadata assembled.
4. `.accel.arrow.tmp` write → `os.replace` → `verify_sidecar` (incl. FD cross-check) → manifest row.
5. Downstream (Observer-v2 / Forward-v2 data pipelines, drift audit) consume pairs via
   the `load_with_accel` join contract and feed the decimated cache from there.
6. Exit summary: DONE/SKIPPED/REBUILT/FAILED counts, worst FD residuals, total sidecar bytes.

## 8. Implementation Sequence

1. `accel_dynamics.py` — port + cross-validate vs A1 torch `ne_rhs` on one real file (fp32 tolerance); settle the EOM convention tag.
2. `build_sidecar` + `verify_sidecar` + `load_with_accel` — exercised on `_tmp/` copies.
3. `test_sidecar_noninterference.py` — loader-invisibility + glob audit + join test; fleet blocked until it passes.
4. `make_accel_sidecars.py` CLI — pilot `--limit 10` on real files, inspect manifest, then fleet.

## 9. Operational Considerations

- **Numerical:** compute float64, store float32; FD cross-check tolerance calibrated on the pilot (dense-output interpolation sets the floor).
- **Write safety:** tmp + `os.replace` prevents partial sidecars; originals are read-only by construction — after the fleet run, spot-check a random sample of original mtimes/sizes as a paranoia gate.
- **OneDrive:** the job now only ADDS ~10–15 GB of small files under the synced tree — far milder than a rewrite, but concurrent tmp/replace churn under sync has bitten before (WinError 5); pausing OneDrive sync for the run remains recommended, not mandatory.
- **Endurance:** run `keep_awake.py` alongside; the job is read-bound (238 GB scanned once); manifest makes it resumable.
- **Staleness:** the fingerprint check means a regenerated original (new combo runs) automatically invalidates its sidecar — REBUILT status, no manual bookkeeping.

## 10. Success Criteria

- [ ] `accel_dynamics` matches A1 torch `ne_rhs` on a real file (fp32 tol); FD cross-check shows no systematic ψ̇·Vy-shaped residual (convention verified)
- [ ] `test_sidecar_noninterference.py` passes: `accel/` folder invisible to A1 and A2 discovery, loader outputs identical with/without it present, recursive-glob audit clean across Python AND Julia consumers (or exclusions itemized and fixed), pairing join verified
- [ ] Pilot (10 files): all DONE, verify passes
- [ ] Fleet: 5,949 sidecars DONE/SKIPPED, zero unresolved FAILED; immediate re-run yields 100% SKIPPED (idempotency); original mtime/size spot-check unchanged
- [ ] Total sidecar footprint within the ~3–5% estimate (sanity on schema bloat)

## 11. Out of Scope

- In-place column addition to originals — REJECTED (user decision: no full rewrite; immutability rule preserved outright)
- DEPRECATED legacy dirs (`SimulationDataSlipSpin_Julia*`) and `_mu_pilot2/`
- Accelerometer-observable conversion, noise, decimation (sensor front-end concerns — `instructions/observer-gamma-only-5phase-retrain.md`)
- Wiring `load_with_accel` into the v2 data pipelines and cache-version bump (owned by the v2 briefs; this brief only provides the join contract)
