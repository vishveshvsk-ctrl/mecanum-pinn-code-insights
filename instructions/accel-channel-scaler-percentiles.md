# Acceleration-Channel Scaler Percentiles — populate `variable_scaler_percentiles.csv`

> **Generated:** 2026-07-10
> **Stack:** Python 3.11 (claude-venv `mecanum`), numpy, pandas, pyarrow (torch-free)
> **Scope:** Data pipeline / offline analysis (one-shot statistics job; no training)

## 1. Overview

Build a standalone, torch-free analysis script that computes robust per-channel
max-normalization scaler statistics for the **acceleration channels** of the
LuGre+Adamov dataset and **upserts** them as new rows into the existing
`data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv`.
That CSV was produced by `observer_v1_py/build_variable_percentiles.py` **before** the
acceleration data existed, so it has 14 channel rows and **no acceleration scalers**.
The new acceleration data lives in the `accel/` subfolder as `<stem>_accel.arrow`
(columns `dVx, dVy, dpsidot, dw1..dw4`), **row-aligned at native 2000 Hz** to the root
`<stem>.arrow`. The script derives the IMU accelerometer observable `a_x, a_y` (which
needs `Vx, Vy, psi_dot` from the root file), applies the observer front-end's
anti-alias LPF, decimates 2000→500 Hz to match the training grid, and produces
p95-based scaler rows in the **identical CSV schema** so the observer's frozen-p95
normalizer can look them up exactly like every other channel. System contract:
`(root arrows + accel/ arrows + existing CSV) → same CSV with 6 acceleration rows added`.

## 2. Architecture Pattern

**Streaming map-reduce over trajectory files with an idempotent upsert.** Mirrors
`build_variable_percentiles.py` exactly: one file loaded at a time (memory-safe), abs
values row-subsampled into bounded accumulators for the percentile estimate, exact
min/max tracked over all decimated rows, then a reduce into one row per channel. The
only structural additions are (a) a **paired read** (root + accel joined by row index),
(b) the **accelerometer-observable derivation** and **anti-alias LPF** before
decimation, and (c) an **upsert writer** that preserves the existing 14 rows verbatim
instead of overwriting the file. Rationale: reusing the proven p95 methodology
guarantees the new rows are statistically consistent with the old ones (same file set,
same decimation, same subsampling), which is what makes them mixable in one scaler CSV.

## 3. Technology Constraints

- **Python:** 3.11 via `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`
- **PyTorch:** none — this is a numpy/pandas/pyarrow job (matches the v1 generator)
- **Required libraries:**
  - `numpy` — percentile/min/max reduction
  - `pandas` — CSV read/upsert/write
  - `pyarrow` — read `.arrow` (root + accel), via the existing loader helpers
  - `scipy.signal` — anti-alias LPF design/apply (Butterworth); the only new dep vs v1
- **Device targets:** CPU only
- **Reuse (do NOT reimplement):** import `_parse_name`, `load_whitelist` from
  `mecanum_observer.data`; import `SIM_HZ`, `DECIM` (=4), and per-wheel/geometry
  constants from `mecanum_observer.config as C`. Reuse the **statistics + row-assembly
  block** of `build_variable_percentiles.py` (p50/p95/p99, abs_max, raw_min/raw_max,
  `p99_over_p95`, `max_over_p95`) unchanged.
- **Explicit exclusions:** do NOT modify `build_variable_percentiles.py` or any v1
  module; do NOT overwrite the existing 14 CSV rows; do NOT read via the decimated
  `.npz` cache (`read_arrays`) — that cache does **not** contain the accel columns, so
  the accel channels must be read from the `accel/` arrows and joined to the root arrow
  directly; do NOT re-derive `dVx` by finite difference for production (it is a stored
  column) — FD is used only as a verification gate (§10).

## 4. Component Breakdown

### `build_accel_percentiles.py`
- **Type:** `script` (CLI, `argparse`; torch-free)
- **Responsibility:** end-to-end driver — select files, stream-accumulate accel-channel
  statistics, upsert rows into the CSV, print a summary + verification report.
- **Inputs:** CLI args (see §6 `parse_args`): `--data-dir`, `--whitelist-csv`,
  `--file-stride`, `--row-stride` (default 25, same as v1), `--limit`, `--out` (the CSV),
  `--lpf-cutoff-hz`, `--lpf-order`, `--no-lpf` (audit toggle).
- **Outputs:** the updated CSV (20 rows); a provenance sidecar (see §9); stdout report.
- **Depends on:** `accel_channels`, `paired_read`, `accumulate_stats`, `upsert_rows`.

### `paired_read`
- **Type:** `function`
- **Responsibility:** for one trajectory stem, memory-map both the root `<stem>.arrow`
  and `accel/<stem>_accel.arrow`, assert equal row counts (alignment gate), and return
  the native-2000 Hz numpy columns needed downstream.
- **Inputs:** `root_path: Path`, `accel_dir: Path`
- **Outputs:** `dict[str, np.ndarray]` with keys `Vx, Vy, psi_dot` (root) and
  `dVx, dVy, dpsidot, dw1, dw2, dw3, dw4` (accel), each `[T_native]` (float64).
- **Depends on:** pyarrow reader idiom (memory_map → `open_file`/`open_stream`).

### `accel_channels`
- **Type:** `function`
- **Responsibility:** transform one file's native-rate paired arrays into the flat
  per-channel 1-D arrays to be scored, applying **form-observable → LPF → decimate** in
  that order (matches `sensor_frontend_v2` §7 of the observer brief). Body-level
  channels stay `[T500]`; the per-wheel channel is raveled across the 4 wheels.
- **Inputs:** `paired: dict[str, np.ndarray]`, `lpf: LPFSpec`, `decim: int`
- **Outputs:** `dict[str, np.ndarray]` with keys
  `a_x, a_y, dVx, dVy, dpsi_dot` (each `[T500]`) and `dw` (`[4·T500]`, raveled).
- **Depends on:** `apply_antialias_lpf`; the transport-term formula (§7).

### `apply_antialias_lpf`
- **Type:** `function`
- **Responsibility:** zero-/low-distortion anti-alias low-pass applied at the **native
  2000 Hz** rate before stride decimation, so LuGre chatter above the 250 Hz decimated
  Nyquist cannot fold into the 0–250 Hz band. Cutoff sourced from `--lpf-cutoff-hz`
  (single source of truth = the observer `config_v2` anti-alias knob once it exists).
- **Inputs:** `x: np.ndarray [T_native]`, `lpf: LPFSpec`
- **Outputs:** `np.ndarray [T_native]` (filtered)
- **Key params:** `LPFSpec(cutoff_hz: float, order: int, fs_hz: float, enabled: bool)`
- **Depends on:** `scipy.signal` (Butterworth design + forward-backward apply).

### `accumulate_stats`
- **Type:** `function`
- **Responsibility:** the v1 reducer, verbatim in spirit — fold each file's channel
  arrays into running abs-value subsample lists (`[::row_stride]`) and exact
  running min/max over all decimated rows.
- **Inputs:** `acc: dict[str,list]`, `rmin: dict[str,float]`, `rmax: dict[str,float]`,
  `chans: dict[str,np.ndarray]`, `row_stride: int`
- **Outputs:** mutates the three accumulators in place (no return)
- **Depends on:** none (pure numpy)

### `finalize_rows`
- **Type:** `function`
- **Responsibility:** reduce accumulators to one CSV row per channel using the **exact
  v1 columns and formulas** (`abs_p50/p95/p99`, `abs_max = max(|rmin|,|rmax|)`,
  `raw_min`, `raw_max`, `p99_over_p95`, `max_over_p95`, `n_samples = subsample size`).
- **Inputs:** `acc`, `rmin`, `rmax`
- **Outputs:** `list[dict]` (rows in the CSV schema)
- **Depends on:** `numpy.percentile`

### `upsert_rows`
- **Type:** `function`
- **Responsibility:** merge the new accel rows into the existing CSV **without altering
  the existing 14 rows' values** — string-preserving read of untouched rows, replace any
  row whose `variable` is in the new set (idempotent re-run), append the rest, write back.
- **Inputs:** `csv_path: Path`, `new_rows: list[dict]`, `protected: set[str]`
- **Outputs:** writes the CSV; returns `(n_existing_kept, n_added, n_replaced)`
- **Depends on:** pandas (with the byte-preservation caveat in §9)

## 5. File & Directory Structure

```
code_insights/
├── observer_v1_py/
│   ├── build_variable_percentiles.py      # EXISTING — reference; unchanged
│   └── build_accel_percentiles.py         # NEW — this brief's deliverable
└── instructions/
    └── accel-channel-scaler-percentiles.md # this brief

data/Simulation_Data_MecanumSlipSpin_LugreAdamov/
├── variable_scaler_percentiles.csv         # EXISTING — 14 rows → upsert to 20
├── variable_scaler_percentiles.provenance.json  # NEW — accel-run provenance sidecar
├── <stem>.arrow                            # root trajectories (Vx,Vy,psi_dot,…)
└── accel/
    └── <stem>_accel.arrow                  # dVx,dVy,dpsidot,dw1..dw4 (2000 Hz)
```

## 6. Key Interfaces

```python
# build_accel_percentiles.py

# --- channel name mapping (the 6 new CSV rows) ---
# "a_x", "a_y"      : IMU accelerometer observable (body-level), network INPUTS
# "dVx", "dVy"      : raw stored body-frame velocity derivatives (body-level)
# "dpsi_dot"        : raw stored yaw acceleration (body-level; accel col = "dpsidot")
# "dw"              : per-wheel angular acceleration, raveled over wheels (cols dw1..dw4)

from dataclasses import dataclass

@dataclass
class LPFSpec:
    cutoff_hz: float          # < 250 Hz (decimated Nyquist at 500 Hz); default from config_v2 knob
    order: int                # Butterworth order
    fs_hz: float              # 2000.0 (SIM_HZ)
    enabled: bool             # False only for the --no-lpf aliasing-headroom audit

def paired_read(root_path: "Path", accel_dir: "Path") -> "dict[str, np.ndarray]":
    """Memory-map root + accel arrows for one stem; assert equal row counts.
    Returns native-2000 Hz float64 columns:
      root : Vx, Vy, psi_dot            [T_native]
      accel: dVx, dVy, dpsidot, dw1..4  [T_native]
    Raises on a missing/short accel sidecar (reported, file skipped by caller)."""
    ...

def apply_antialias_lpf(x: "np.ndarray", lpf: LPFSpec) -> "np.ndarray":
    """Forward-backward Butterworth low-pass at fs = lpf.fs_hz, cutoff lpf.cutoff_hz.
    Identity when lpf.enabled is False. Input/Output: [T_native]."""
    ...

def accel_channels(paired: "dict[str, np.ndarray]", lpf: LPFSpec, decim: int
                   ) -> "dict[str, np.ndarray]":
    """Form accelerometer observable at native rate, LPF, then stride-decimate.
      a_x = dVx - psi_dot*Vy ;  a_y = dVy + psi_dot*Vx     (transport term; see §7)
    Order per file: build a_x/a_y at 2000 Hz -> LPF all accel-derived channels ->
    x[::decim]. Body channels -> [T500]; 'dw' -> ravel(dw1..dw4 decimated) [4*T500].
    Returns {a_x, a_y, dVx, dVy, dpsi_dot, dw}."""
    ...

def accumulate_stats(acc: "dict[str, list]", rmin: "dict[str, float]",
                     rmax: "dict[str, float]", chans: "dict[str, np.ndarray]",
                     row_stride: int) -> None:
    """Fold one file: append abs(chan)[::row_stride] (float32) to acc[name];
    update rmin/rmax with exact min/max over ALL decimated rows. Mutates in place."""
    ...

def finalize_rows(acc: "dict[str, list]", rmin: "dict[str, float]",
                  rmax: "dict[str, float]") -> "list[dict]":
    """Reduce to CSV rows using the v1 schema/formulas exactly:
    variable, n_samples, abs_p50, abs_p95, abs_p99, abs_max, raw_min, raw_max,
    p99_over_p95, max_over_p95."""
    ...

def upsert_rows(csv_path: "Path", new_rows: "list[dict]", protected: "set[str]"
                ) -> "tuple[int, int, int]":
    """Read existing CSV preserving untouched rows' exact text; replace rows whose
    'variable' is in the new set; append the rest; write back. Never edits a row not
    in the new set. Returns (n_kept, n_added, n_replaced)."""
    ...

def parse_args() -> "argparse.Namespace":
    """--data-dir (default ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov),
    --accel-subdir (default 'accel'), --whitelist-csv (default diagnostics_combined.csv),
    --file-stride (1), --row-stride (25), --limit (0), --out (the CSV path),
    --lpf-cutoff-hz, --lpf-order, --no-lpf, --write-provenance (default True)."""
    ...
```

## 7. Data Flow

1. **File selection (identical to v1).** Glob `--data-dir/*.arrow` (root files only,
   NOT the `accel/` subdir); keep those where `_parse_name(name)` is truthy AND
   (`whitelist is None` OR `name in whitelist`), where `whitelist = load_whitelist(diagnostics_combined.csv)`;
   apply `[::file_stride]` and `--limit`. This yields the **same file set** that
   produced the existing 14 rows — essential for cross-row consistency.
2. **Paired read.** For each root file, locate `accel/<stem>_accel.arrow`; if missing
   or row-count-mismatched, log and skip. Load `Vx, Vy, psi_dot` (root) and
   `dVx, dVy, dpsidot, dw1..4` (accel) at native 2000 Hz.
3. **Accelerometer observable (native rate).** `a_x = dVx − psi_dot·Vy`,
   `a_y = dVy + psi_dot·Vx`. The `ψ̇×V` transport term is added because `dVx, dVy` are
   the **pure body-frame component derivatives** (verified in §10), whereas the sensor
   measures specific force resolved in body axes. Sign convention (2D, `ω = ψ̇ ẑ`):
   `ω×V = (−ψ̇·Vy, +ψ̇·Vx)`.
4. **Anti-alias LPF (native rate).** Apply `apply_antialias_lpf` to `a_x, a_y` and to
   `dVx, dVy, dpsidot, dw1..4` at 2000 Hz, cutoff `< 250 Hz`. This matches the observer
   front-end (`form observable → LPF → decimate`) so the scaler reflects exactly what
   the network ingests, and keeps `abs_max` from being inflated by >250 Hz LuGre
   chatter that would otherwise alias.
5. **Decimate.** `x[::DECIM]` (DECIM = 4 → 500 Hz), same phase as the v1 cache
   (`df.iloc[::C.DECIM]`), so the 500 Hz grid is row-comparable to the existing channels.
6. **Flatten.** Body channels stay `[T500]`; `dw = ravel(stack(dw1..4 decimated))`
   `[4·T500]` (one scale for the wheel-shared encoder, matching how `w`, `gamma`, etc.
   are pooled in v1 `channels()`).
7. **Accumulate → finalize.** Subsample `[::row_stride]` abs values for percentiles;
   track exact min/max over all decimated rows; reduce to 6 rows with the v1 columns.
8. **Upsert.** Merge the 6 rows into the CSV, preserving the existing 14 rows' bytes;
   write the provenance sidecar.

No gradients, no loss — this is a statistics job. The only "correctness graph" is the
verification chain in §10.

## 8. Implementation Sequence

1. **`paired_read` + alignment gate** — nothing else can run without a correct paired
   loader; validate on one known stem (`coupled_vomega_c001_mu_0.3_..._chi_0.002`,
   confirmed 44022 rows both sides).
2. **`apply_antialias_lpf` + `accel_channels`** — depends on #1; unit-check shapes
   (`[T500]` body, `[4·T500]` dw) and that `--no-lpf` is a true identity path.
3. **Verification gates (§10)** — depends on #2; run BEFORE any full sweep so a wrong
   transport-term sign or a non-pure `dVx` is caught on one file, not after hours.
4. **`accumulate_stats` + `finalize_rows`** — reuse v1's reducer; confirm on `--limit 5`.
5. **`upsert_rows`** — depends on #4; test idempotency (run twice → identical CSV) and
   byte-preservation of the 14 existing rows (diff the untouched rows).
6. **`build_accel_percentiles.py` driver + provenance sidecar** — wire together; full
   run (`--file-stride 1`) with the consistency checks in §10 asserted at the end.

## 9. ML-Specific Considerations

*(This is an offline stats job; the "ML" concerns are normalizer consistency and
reproducibility rather than gradients/precision.)*

- **Scaler consistency (the core risk).** The new rows must be comparable to the old
  ones: **same file set** (whitelist + `_parse_name` filter + `file_stride`), **same
  decimation phase** (`::DECIM`), **same `row_stride`** (default 25), **same percentile
  columns**. Any deviation makes the accel scalers silently mis-scaled relative to the
  velocity channels. Assert the file set size equals the v1 run's (or record both).
- **Anti-alias cutoff is a pinned parameter, not a free choice.** Source
  `--lpf-cutoff-hz` from the observer `config_v2` anti-alias knob (the single source of
  truth once `sensor_frontend_v2` exists). Until pinned, use a documented default below
  the 250 Hz decimated Nyquist (e.g. 200 Hz Butterworth order 4) and **record it in the
  provenance sidecar**. Note: p95/p99 are largely insensitive to the cutoff (robust by
  design); `abs_max`/`raw_min`/`raw_max` are not — so the cutoff mainly governs the tail
  columns, and its value must be reproducible.
- **Byte-preservation on upsert.** A naive `pd.read_csv → concat → to_csv` round-trips
  the existing 14 rows through float parsing and can change their last-digit
  representation. Preserve untouched rows as raw text (or assert the re-emitted values
  are bit-identical); only the 6 accel rows are newly formatted. "Do not disturb the
  existing rows" is a literal requirement.
- **Memory safety.** Never hold the whole dataset: one file at a time, subsample rows
  into bounded lists, exact min/max as running scalars — identical to the v1 generator's
  memory model. `dw` accumulators are ~4× the body channels; size accordingly.
- **Numerical detail.** Accumulate abs values as float32 (v1 parity) but compute the
  observable and LPF in float64 to avoid catastrophic cancellation in
  `dVx − psi_dot·Vy` when the two terms nearly cancel. `p95` guards against outliers;
  still clamp any degenerate all-zero channel before `p99/p95` division (v1 uses
  `max(p95, 1e-12)`).
- **Provenance/checkpointing.** Emit `variable_scaler_percentiles.provenance.json`:
  git-less run stamp (pass a timestamp in — do not call `Date.now()` equivalents if
  determinism is needed), `lpf_cutoff_hz`, `lpf_order`, `decim`, `row_stride`,
  `file_stride`, `n_files_used`, `whitelist_csv`, and the transport-term formula string.
  This makes the accel rows auditable against the (undated) v1 rows.

## 10. Success Criteria

- [ ] **Alignment gate:** every used `accel/<stem>_accel.arrow` has the same row count
      as its root `<stem>.arrow` (skip + log any that don't; report the skip count).
- [ ] **`dVx` is the pure component derivative (transport-term justification):** on a
      sample of files, stored `dVx` matches a central finite difference of root `Vx`
      (`d/dt Vx`) to solver dense-output tolerance, and stored `dVy` matches `d/dt Vy`.
      This confirms `dVx ≠ a_x` and that `a_x = dVx − ψ̇·Vy` is the correct observable
      (NOT already specific force). Same check for `dpsidot` vs `d/dt psi_dot`.
- [ ] **`n_samples` consistency:** `a_x, a_y, dVx, dVy, dpsi_dot` (body-level) each equal
      the existing `Vx`/`Vy`/`psi_dot` `n_samples` (3,389,512) under the identical file
      set + `row_stride`; `dw` equals the existing `w` `n_samples` (13,546,458). A
      mismatch means the file set or strides drifted from the v1 run.
- [ ] **Schema match:** the 6 new rows carry exactly the 10 existing columns, in order,
      with the same formulas (`abs_max = max(|raw_min|,|raw_max|)`, `p99_over_p95`,
      `max_over_p95`).
- [ ] **Sanity of magnitudes:** `a_x ≈ dVx` and `a_y ≈ dVy` in p95 (transport term is a
      modest correction), `dw` p95 ≫ body-accel p95 (wheel inertia; pilot `dw1` p95 ≈ 135
      vs `dVx` p95 ≈ 0.8), and `max_over_p95` for `dw` flags heavy chatter tails (LPF on
      should reduce it vs `--no-lpf`).
- [ ] **Upsert integrity:** CSV grows 14 → 20 rows; the 14 pre-existing rows are
      unchanged (byte-diff clean); re-running the script is idempotent (no duplicate
      rows, identical output).
- [ ] **Provenance emitted** with the LPF cutoff and all sampling parameters.

## 11. Out of Scope

- Rebuilding the decimated `.npz` cache to embed accel columns (an alternative,
  more-integrated route via extending v1 `channels()`); this brief takes the standalone
  companion-script path instead and leaves v1 untouched.
- Modifying `build_variable_percentiles.py` or any observer module.
- Recomputing / altering the existing 14 channel scalers.
- Sensor-noise corruption of the accel channels (`SensorNoiseSpec` stage-2) — the
  scaler is computed on the clean exact-dynamics accelerations.
- Wiring the new scalers into `config_v2`/`data_v2` lookup (consumer-side; separate
  observer work) — this job only populates the CSV the consumer reads.
- Choosing the final `vel_filter_crossover_hz` / complementary-filter design — unrelated
  to the accel scaler; only the anti-alias LPF cutoff is used here.
```
