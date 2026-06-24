# Mecanum trajectory visualizer widget — build instructions

## Objective
Build an interactive Jupyter (`ipywidgets`) widget that lets the user pick a simulation run by four dropdowns — `mu`, `chi`, `profile`, `combo` — and visualizes that run's **global (world-frame) trajectory**: the reconstructed reference path as a **dotted black line**, the KUKA youBot plant drawn as an **oriented rectangle with a heading pointer** tracking the plant pose, the actual path traced up to the current time, a **time scrub slider**, and **gif-style play/pause playback** (plus a GIF export). "Done" means: dropdowns are data-driven and cascade correctly, the rectangle and reference draw correctly in the world frame, the slider + Play animate the plant along the path, and a headless acceptance render + GIF can be produced reproducibly.

The work has four parts (catalog → extraction → widget → wire/verify), all delivered together. Build them in order; each part lists its own acceptance check.

---

## Shared context (applies to all parts)

### Data location and immutability
- Repo working dir is `code_insights/`. The repo root is its parent (the dir containing both `code_insights/` and `data/`).
- Simulation outputs: `data/Simulation_Data_MecanumSlipSpin_LugreAdamov/` (relative to repo root), ~5,951 Apache Arrow / Feather files, **READ-ONLY** — never write, rename, move, or delete anything under `data/`.
- Resolve the data dir robustly: walk up from `__file__` to the dir containing both `code_insights/` and `data/`; allow an env override `MECANUM_DATA_DIR`. Do NOT hardcode a per-machine absolute path with no fallback.

### Filename contract (single source of truth for parsing)
```
<profile>_c<combo:%03d>_mu_<mu:%g>_case1_lugre_adamov_chi_<chi:%.3f>.arrow
```
Example: `octagon_c001_mu_0.3_case1_lugre_adamov_chi_0.005.arrow`.
- `<profile>` itself contains underscores (`coupled_vomega`, `multisine_50percent_cap`, `spin_creep`, `spiral_orbit`), so parse with a regex anchored on the fixed substrings (`_c###_mu_`, `_case1_lugre_adamov_chi_`, `.arrow`) — NOT naive left-to-right `split("_")`.
- Observed profiles (confirm by scan, don't assume): `coupled_vomega`, `ellipse`, `long_circle`, `multisine_50percent_cap`, `multisine_75percent_cap`, `octagon`, `spin_creep`, `spiral_orbit`.
- Observed `mu`: `0.3`, `0.5`, `0.8`. Observed `chi`: `0.000`, `0.002`, `0.005`, `0.008`. Not every `(profile, mu, chi, combo)` combination exists — the sweep is partial, so the selection space MUST be data-driven from the catalog.

### Column contract (verified against `datastore.jl::assemble_dataframe`)
Each Arrow table has 97 columns; row count varies per file (tens of thousands of rows over a 0–40 s horizon at ~2000 Hz, native `time` step ≈ 0.0005 s). Required columns:
- `time` — seconds.
- `Xo`, `Yo` — **actual world-frame position** of the plant origin (state indices 20, 21).
- `psi` — **actual world heading** (radians; state index 4).
- `Vx_des`, `Vy_des`, `psi_des`, `omega_des` — **body-frame** desired velocities + desired heading.

**There is NO world-frame reference XY column.** All profiles are `VelRef` (body-frame velocity reference) except `ellipse`, which is a `PosRef` that DOES store `xo_des`/`yo_des`. So the reference path must be reconstructed (see Part 2). If `xo_des`/`yo_des` are present (ellipse), use them directly; otherwise integrate.

### Plant geometry (authoritative — `run_one.jl::PlatformParams`, fed from `trajectory_files_run_0p3_main/base.toml [platform.geometry]`)
- `h = 0.235` m = **half-LENGTH** (body-x, the forward/heading axis).
- `l = 0.15` m = **half-WIDTH** (body-y).
- `R = 0.05` m = wheel radius; `Ra = 0.0355` m.
- Tiebreaker confirming the axis mapping: wheel centers `wc_x = [h, h, -h, -h]` (±h along body-x), `wc_y = [l, -l, l, -l]` (±l along body-y).
- **Chassis footprint rectangle = `2*h = 0.47` m along body-x (heading) × `2*l = 0.30` m along body-y.** The heading pointer points along **body +x**, direction `(cos psi, sin psi)`. (The long axis must align with the pointer.)
- Read `h`, `l`, `R` from the data payload / TOML at runtime; do not hardcode in a way that can silently diverge.

### Environment
- Interpreter for all headless execution: `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe` (system matplotlib is broken; do NOT `conda activate`). It has `pyarrow`, `pandas`, `numpy`, `matplotlib` but **NOT** `ipywidgets`/`ipympl`/jupyter.
- The interactive widget runs in the user's **Jupyter kernel**, which must have `ipywidgets` + `matplotlib`. Do NOT depend on `ipympl` — redraw into an `Output`/`Image` widget so only `ipywidgets` is required.
- Read Arrow via `pyarrow.feather`. Do NOT import `torch`; if any dependency pulls it in, import `pyarrow.feather` first (Windows/WSL load-order gotcha).
- Keep scratch artifacts in `code_insights/_tmp/` and clean them up. No LaTeX in console output (Unicode only).

### Deliverable layout
```
code_insights/trajectory_widget/
  build_catalog.py          # Part 1
  catalog.json              # Part 1 (generated)
  contract.json             # Part 1 (generated)
  widget.py                 # Part 3 (front end + headless render core + gif export)
  trajectory_widget.ipynb   # Part 3 (thin launcher)
  verify_end_to_end.py      # Part 4 (non-interactive check + frame renderer)
  exports/                  # gif/frame outputs (generated)
code_insights/mecanum_widget/
  __init__.py               # Part 2
  data_extract.py           # Part 2 (extract() payload builder)
```

---

## Part 1 — File catalog + contract verification

Goal: index every available `(profile, combo, mu, chi)` → filename without opening ~6,000 files, and verify the column/geometry contract by opening ONE file.

1. Write `code_insights/trajectory_widget/build_catalog.py` that:
   - Lists `*.arrow` in the data dir (flat, non-recursive).
   - Parses each filename with the anchored regex; counts and reports any skipped (non-matching) names.
   - Writes `catalog.json` with: `data_dir`, `n_files`, sorted distinct `profiles`, `mus`, `chis`, `combos`, a `valid_combos_per_profile` map (profile → sorted combos present), and `entries` resolving each `(profile, combo, mu, chi)` to its exact filename (flat list of `{profile, combo, mu, chi, filename}` or nested dict — either is fine).
   - Does NOT open Arrow contents during the catalog build (filename parsing only — stay fast).
2. Add a `--verify` path (in the same script or a sibling) that opens ONE representative Arrow file via `pyarrow.feather.read_table` and asserts the required columns are present (`time, Xo, Yo, psi, Vx_des, Vy_des, psi_des, omega_des`), reads `[platform.geometry]` from `base.toml`, and writes `contract.json` with: the required-column list, the sampled file's row count and `time` min/max, the geometry dict (`R, Ra, h, l`) plus derived `chassis_length_x = 2*h = 0.47` and `chassis_width_y = 2*l = 0.30`, and a `notes` field stating the reference world path is NOT stored (must be integrated) except for `ellipse` (`xo_des`/`yo_des`).

**Acceptance:** `python build_catalog.py --verify` exits 0, prints matched + skipped counts; `catalog.json` has `n_files` within a few of 5951, `profiles` == the eight observed, non-empty `mus`/`chis`; every `entries` filename exists on disk (spot-check 50); `contract.json` lists all eight columns and `h=0.235, l=0.15, R=0.05, chassis_length_x=0.47, chassis_width_y=0.30`.

---

## Part 2 — Arrow extraction + reference reconstruction (`mecanum_widget/data_extract.py`)

Goal: given a selection, return a compact JSON-serializable payload the widget consumes.

1. Create the package `code_insights/mecanum_widget/` (`__init__.py` may be empty) and `data_extract.py`.
2. `build_filename(profile, combo, mu, chi)` → exact contract filename (combo zero-padded to 3 digits, `mu` via `%g`, `chi` 3 decimals, literal `case1_lugre_adamov`), joined under the resolved data dir.
3. Geometry loader: parse `base.toml [platform.geometry]` with `tomllib` (Python 3.11+ stdlib) and return `l, h, R`; safe fallback to `l=0.15, h=0.235, R=0.05` only if the file is unreadable (document the fallback).
4. Loader: `import pyarrow.feather as feather` at module top; `feather.read_table(path)`; pull the needed columns to NumPy. Raise a clear `FileNotFoundError` with the attempted path if missing.
5. Decimation: `max_points` arg (default ~1500). `stride = max(1, ceil(n_rows / max_points))`; slice every per-sample array with that stride; always include the final sample so the path end isn't clipped.
6. **Reference reconstruction** (document prominently in the module docstring):
   - If `xo_des`/`yo_des` columns exist (`ellipse`/PosRef): use them directly as the reference XY.
   - Else (VelRef): integrate desired body velocity into the world frame on the FULL-resolution `time` array, then decimate with the same stride:
     ```
     Xref += (Vx_des*cos(psi_des) - Vy_des*sin(psi_des)) * dt
     Yref += (Vx_des*sin(psi_des) + Vy_des*cos(psi_des)) * dt
     ```
     Use cumulative trapezoidal (or forward-Euler) integration; start at the actual initial position `(Xo[0], Yo[0])` so reference and actual share an origin.
7. `extract(profile, combo, mu, chi, max_points=...)` → plain dict (JSON-serializable; convert NumPy via `.tolist()`/`float(...)`):
   - `meta`: `{profile, combo, mu, chi, filename, n_rows_full, n_points, dt, t_start, t_end, ref_source}` (`ref_source` ∈ `{"integrated","posref"}`).
   - `geometry`: `{l, h, R}`.
   - `time`: decimated timestamps.
   - `actual`: `{x, y, psi}` (decimated `Xo, Yo, psi`).
   - `reference`: `{x, y}` (decimated reconstructed/stored reference).
8. Add `to_json(payload)` and a `__main__` CLI (`--profile --combo --mu --chi --max-points [--out FILE]`) that prints only `meta` + array lengths for a readable spot-check.

**Acceptance:** module imports clean under the venv; `build_filename('octagon',1,0.3,0.005)` basename == `octagon_c001_mu_0.3_case1_lugre_adamov_chi_0.005.arrow`; `extract('coupled_vomega',1,0.3,0.005,max_points=1500)` exits 0, JSON parses, all arrays equal length ≤ 1500, `geometry=={l:0.15,h:0.235,R:0.05}`, and `|reference.x[0]-actual.x[0]|<1e-6` (same for `y`); `json.dumps(extract(...))` works with no custom encoder.

---

## Part 3 — The widget front end (`trajectory_widget/widget.py` + launcher notebook)

Goal: the ipywidgets UI plus a headless-reusable render core.

1. In `widget.py`, import `extract` from Part 2 (add `code_insights/` to `sys.path`, `from mecanum_widget.data_extract import extract`; resolve `code_insights/` robustly from `__file__`).
2. Catalog helpers: `load_catalog(path=None)` reads `trajectory_widget/catalog.json`; expose `profiles()`, `mus()`, `chis()`, `valid_combos(profile)`, `has_entry(...)`, `resolve_filename(...)`. Match `mu`/`chi` with float tolerance (`abs(a-b)<1e-9`) so dropdown values map to catalog keys.
3. **Headless render core** (no `ipywidgets`/`IPython` imports — must run under Agg in the venv):
   - `setup_axes(ax, payload)` — once per selection: clear axes; plot `reference` as **dotted black** (`linestyle=":"`, `color="k"`, lw≈1.5, `label="reference"`); optionally plot full actual path faint grey; `ax.set_aspect("equal", adjustable="datalim")`; axis limits = union of reference+actual extents + margin ≈ `2*h`; labels `X (m)`/`Y (m)`, grid, title `f"{profile} c{combo} mu={mu} chi={chi}"`. Create and return the per-frame artists:
     - a solid `Line2D` for the **traced actual path** (e.g. tab:blue, `label="actual"`), empty init;
     - a `patches.Rectangle` anchored at local corner `(-h, -l)` with `width=2*h`, `height=2*l` (semi-transparent face, solid edge) — note width is along body-x (length 0.47), height along body-y (0.30);
     - a heading **pointer** (updatable `Line2D`/`FancyArrow`) from center along `(cos psi, sin psi)`;
     - a current-position dot and a `t = {t:.2f}s` text annotation.
   - `update_frame(artists, payload, idx)` — per frame: set traced-path data to `actual.x[:idx+1]`, `actual.y[:idx+1]`; set rectangle transform to `Affine2D().rotate(psi).translate(x, y) + ax.transData` with `x=actual.x[idx], y=actual.y[idx], psi=actual.psi[idx]` (rotates the `(-h,-l)`-anchored rect about the body origin → centered on the plant, long axis along heading); update pointer to start at `(x,y)` extending ≈`1.3*h` along `(cos psi, sin psi)`; update dot + time text; return changed artists.
4. **Interactive assembly** — `build_widget(catalog_path=None, max_points=1500)` (import `ipywidgets` lazily INSIDE the function):
   - `Dropdown`s `profile`/`mu`/`chi`/`combo` (combo options from `valid_combos(profile)`).
   - `Play` + `IntSlider` for the time index, linked with `jslink((play,"value"),(slider,"value"))` (interval ≈ 60 ms) for gif-style playback.
   - An `Output` (or `Image`) widget holding the matplotlib axes; an "Export GIF" `Button` and a status `Label`.
   - Callbacks: on `profile` change repopulate `combo.options` from `valid_combos(...)`; on any selection change, if `has_entry(...)` is False show a status message and skip (don't crash), else `extract(...)`, store payload, set `slider.max=len(time)-1`, `slider.value=0`, `setup_axes(...)`, `update_frame(...,0)`, redraw. On `slider` change, `update_frame(...)` + redraw.
   - **Redraw without ipympl:** keep ONE persistent `Figure`; after `update_frame`, push pixels to the widget — either `with output: clear_output(wait=True); display(fig)`, or (more flicker-free) `fig.canvas.draw()` → set an `ipywidgets.Image.value` from the canvas buffer. Do not create a new figure per frame.
   - Layout: dropdowns in an `HBox`, Play+slider in an `HBox`, stacked `VBox([controls, figure, export_row, status])`. Trigger one initial render before returning.
5. **GIF export** — `export_gif(payload, out_path, fps=15)` (headless-usable) builds `FuncAnimation` over the frames reusing `setup_axes`/`update_frame`, saves with `PillowWriter` to `trajectory_widget/exports/`. Cap at ~150–250 frames. Fallback: write numbered PNG frames if `Pillow`/`PillowWriter` is unavailable. The button handler calls this and shows the saved path (or error) in the status `Label`.
6. **Launcher notebook** `trajectory_widget/trajectory_widget.ipynb`: one markdown cell (which kernel/env, read-only data, dropdown meanings, "dotted black = reconstructed reference") + one code cell `from widget import build_widget; build_widget()`.
7. Module docstring: dropdown → catalog → `extract` → render flow; geometry mapping (rect 2*h along body-x/heading × 2*l along body-y, pointer +x); the no-ipympl redraw strategy; the headless entry points.

**Acceptance (headless, venv):** module imports WITHOUT `ipywidgets` (lazy import) and exposes `build_widget/setup_axes/update_frame/export_gif`; rendering a mid-frame for `extract('coupled_vomega',1,0.3,0.005)` to PNG produces a >5 KB image showing all four elements (dotted-black reference, traced actual path, oriented rectangle, heading pointer) with the rectangle long axis parallel to the pointer; `export_gif` for `('octagon',1,0.5,0.005)` writes a >10 KB gif (or numbered-frame fallback). **Interactive (notebook kernel):** `build_widget()` displays; changing `profile` repopulates `combo`; the slider advances the plant; Play animates it; Export GIF writes a file and reports its path.

---

## Part 4 — Wire together + verify end-to-end

Goal: prove the whole chain for ≥2 distinct selections, non-interactively.

1. Write `code_insights/trajectory_widget/verify_end_to_end.py` that: loads `catalog.json`; picks two selections that exist and differ in >1 field (e.g. A=`coupled_vomega,1,0.3,0.005`, B=`octagon,1,0.5,0.000` — verify each is in the catalog, substitute if not); for each, resolves the filename, calls `extract(...)`, asserts `len(time)==len(actual.x)==len(actual.y)==len(reference.x)==len(reference.y) > 1` and `<= max_points`, and `|reference.x[0]-actual.x[0]|<1e-6` (same for y); renders a mid-trajectory frame via `setup_axes`/`update_frame` to `trajectory_widget/acceptance_frame_A.png` and `_B.png`; and exports `trajectory_widget/exports/acceptance_playback.gif` (or 3+ numbered progression frames) to demonstrate animation.
2. Assert dropdown cascade is data-driven: `set(valid_combos_per_profile) == set(profiles)` and `valid_combos(p)` equals the catalog map per profile.

**Acceptance:** `python verify_end_to_end.py` exits 0 and prints, per selection, the resolved filename + decimated point count; both `acceptance_frame_*.png` exist and are >5 KB; the playback gif (or numbered frames) exists; the cascade assertion passes. Visual check of `acceptance_frame_A.png` shows the four required elements with the dotted-black reference clearly distinct from the actual path.

---

## Global constraints
- Never write/modify anything under `data/` (Arrow files are read-only). New outputs go under `code_insights/trajectory_widget/`.
- Do NOT import `torch`; if unavoidable, import `pyarrow.feather` first.
- Do NOT depend on `ipympl`; keep the render core free of `ipywidgets`/`IPython` so the venv can run it headless under Agg.
- Read `h`/`l`/`R` from the payload/TOML — do not hardcode geometry that can diverge from `base.toml`/`run_one.jl`. Correct mapping: length `2*h=0.47` along body-x (heading), width `2*l=0.30` along body-y.
- Build exactly ONE interactive widget (the deliverable); no extra dashboards. Keep scratch in `code_insights/_tmp/`; respect the ≤8 parallel-worker limit (this task is single-threaded).
