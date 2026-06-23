# CLAUDE.md — Mecanum PINN Digital Twin (IMECE 2026)

Orientation file for Claude Code. Explains *what this project is for*, *how the
pieces fit*, *what is authoritative vs. in-flux*, and *the conventions to
respect*. Read this before touching anything.

> **📁 Chat handoffs live in `code_insights/chat-handoff/`.** Cross-session task
> handoff briefs (`*_handoff.md`) are collected there — when a task says "continue
> per the handoff brief", "per `<name>_handoff.md`", or references a prior session's
> plan, look in **`chat-handoff/`** first (not the project root), and write any new
> handoff brief there. **Exception:** the deck/figure handoffs
> (`presentation/HANDOFF_*.md`) stay with the deck in `presentation/`.

---

## 1. What this project is

A **Physics-Informed Neural Network (PINN) digital twin** for a KUKA youBot
four-Mecanum-wheel omnidirectional platform, targeting an **IMECE 2026** paper.

The work splits into two halves joined by a file-format contract:

```
   JULIA                                              PYTHON
   high-fidelity ODE simulator  ──►  Arrow files  ──►  PyTorch PINN
   (39-D plant + ASMC + DOB)        (+ JLD2 sidecar)   (forward dynamics +
                                                        inverse friction ID)
```

The PINN has two jobs: learn the **forward dynamics**, and recover **friction
parameters** (`μ`, `χ`) by inverse identification. The simulator's role is to
manufacture richly-exciting, physically-valid trajectory data with ground-truth
labels (per-wheel friction forces, slips, controller torques) that the PINN
trains against.

---

## 2. Authority map — what to trust, what is being rewritten

| Component | Status | Notes |
|---|---|---|
| Julia simulator notebook | **authoritative** | the plant + controllers; source of `run_one.jl` |
| `profiles.jl` | **authoritative** | trajectory/excitation reference library |
| `datastore.jl` | **authoritative** | label extraction + DataFrame + disk I/O |
| `Data_Generation_Julia.jl` | **authoritative (migrated)** | profile-based parallel sweep driver |
| `base.toml` + per-profile TOMLs | **authoritative** | single source of truth for physics/solver/sweep |
| **Arrow filename + column schema** | **the cross-language contract** | defined in `datastore.jl`; Python must honor it |
| Python `mecanum_pinn` package (`*.py`) | **being restructured in Claude Code** | treat current module layout as provisional; the *intent* in §7 is what survives |

**The data pipeline overhaul is complete.** Earlier sessions had the driver
still on the old `(beta, amplitude)` grid; that migration is now done — the
driver enumerates profile × combo jobs and calls into `Profiles`/`DataStore`.
Do not reintroduce `beta`/`amp` anywhere.

---

## 3. The plant model (physics intent)

Based on **Adamov & Saypulaev (2021)** (`nd711.pdf` in the repo) with deliberate
modifications:

- **12 rollers per wheel** (paper uses 6); roller inertia `J_roller = 1e-6 kg·m²`.
- **Composite LuGre + Adamov friction** with slip–spin coupling, replacing the
  paper's algebraic multicomponent friction law. Two friction models are
  selectable: `:lugre_adamov` (coupled) and `:lugre_uncoupled`.
- Platform constants (all live in `base.toml`, no hard-coded fallbacks):
  `R = 0.05 m`, `Ra (Rd) = 0.0355 m`, roller angle `δ = 45°` (O-configuration
  `[-π/4, π/4, π/4, -π/4]`), platform mass `30 kg`, `Is = 4.42 kg·m²`.
  Identified viscous coefficients: `p1 = 0.11`, `p2 = 5.78e-3 N·m·s`
  (`friction_case = 1`); a low-viscous ablation is `friction_case = 2`.

**State is 39-dimensional**, in this canonical order (also encoded in
`base.toml [solver.abstol_counts]` and consumed by `compute_labels`):

```
[1:3]   body velocity  Vx, Vy, psi_dot      [20:21] world position x_o, y_o
[4]     heading psi                         [22:29] linear bristle zx(4), zy(4)
[5:8]   wheel angle theta_1..4              [30:33] rotational bristle zs(4)
[9:12]  wheel rate omega_1..4               [34:36] observer state
[13:16] roller rate gamma_1..4              [37:39] disturbance estimate delta_hat
[17:19] adaptive gains K_x, K_y, K_psi
```

**Roller dynamics (`γ̇ᵢ`) are simulated but excluded from the PINN** — they are
unmeasurable at deployment. The PINN's GRU encoder is meant to reconstruct them
implicitly (its legitimate role is bounded to ~4 hidden dims); excess capacity
causes trajectory fingerprinting, so keep it small.

---

## 4. The Julia data-generation subsystem

### Files and their boundaries

- **`profiles.jl` (`module Profiles`)** — the excitation reference library.
  Builders return a `VelRef` (body-frame velocity + heading reference, tracked by
  the mixed-degree velocity controller `asmc_torques_vel`) or a `PosRef`
  (world-frame, tracked by the degree-2 position controller `asmc_torques`).
  - Every `VelRef` carries a primary heading getter `psi`; `Wz` and `al` are the
    1st and 2nd `ForwardDiff` derivatives of `psi`, so yaw-rate / yaw-accel
    feedforwards are **consistent by construction** (the controller never fights
    itself). Same idea for `Ax,Ay` as derivatives of `Vx,Vy`.
  - Reference publishing is via `set_reference!`/`set_pos_reference!`/`publish!`
    into module-level slots `CURRENT_REF`/`CURRENT_POSREF`. **Thread-safety
    contract: publish once per trajectory, single-threaded, before the inner
    parallel loop. Inner threads only READ.**

- **`datastore.jl` (`module DataStore`)** — everything between `sol` and disk:
  `compute_labels`, `assemble_dataframe`, `write_outputs`, the **filename scheme**
  (single source of truth for writer *and* resume check), the streaming logger,
  and `reload_run`. Deliberately **physics-free**: the plant/controller callables
  it needs (`lugre_dyn_rates`, the controller, `sawtooth_approx`) are **injected**
  as keyword arguments, because they live in the notebook (`Main`), not here.
  CSV output is intentionally gone — Arrow is the consumed format.

- **`Data_Generation_Julia.jl`** — the parallel sweep driver. Structure:
  - `include(run_one.jl)` loads the simulator (which itself includes `profiles.jl`
    + `datastore.jl`, putting `Profiles`/`DataStore` in scope).
  - **Outer loop, sequential** over `Profiles.enumerate_jobs(...)` (one job per
    profile × combo row): `build_job` → `publish!` → pick controller
    (`asmc_torques_vel` for VelRef, `asmc_torques` for PosRef).
  - **Inner loop, `Threads.@threads :static`** over `(mu, chi, friction_model)`
    combos from `base.toml`. Each thread builds its own `PlatformParams`,
    `ODEProblem`, and `solve` locally.
  - **Resume** = `.arrow` existence via `DataStore.expected_output` (the same
    function the writer uses, so they can't drift). Atomic `tmp → rename` writes
    mean a killed job leaves only `.tmp` debris, never a partial `.arrow`.
  - Per-solve wall-clock timeout via a **per-step `DiscreteCallback`** (not
    `PeriodicCallback` — when `dt` collapses, sim-time callbacks stop firing).

### CLI

```bash
julia --project=. -t auto Data_Generation_Julia.jl
julia --project=. -t 8    Data_Generation_Julia.jl --dry-run
julia --project=. -t 8    Data_Generation_Julia.jl --profiles octagon.toml,ellipse.toml
julia --project=. -t 8    Data_Generation_Julia.jl --timeout 900   # straggler retry pass
```

Thread count is set by Julia's `-t` flag, not a script arg. `--dry-run` prints a
resume-aware per-profile pending/done table and exits. `--sweep-seed` (default
1234) **must match across resume passes** — `build_job` is deterministic
(`hash((sweep_seed, profile, combo))`), so a retry reproduces the identical
trajectory.

### Config schema

- **`base.toml`** — single source of truth for everything physics/solver:
  - `[physics]` — `mu_friction`, `chi`, `friction_model`, `friction_case`; each
    may be a **scalar (single point) or a list (swept)**.
  - `[platform.*]` — geometry / mass / inertia / viscous / contact constants.
  - `[solver]` — `name` (`TRBDF2|Rodas5P|RadauIIA5|FBDF|KenCarp47|QNDF`), `reltol`,
    `dtmax`, `maxiters`, `saveat_rate` (2000 Hz dense uniform grid).
  - `[solver.abstol]` + `[solver.abstol_counts]` — per-group tolerances; the
    driver rebuilds the flat 39-vector and **asserts the counts sum to 39**.
  - `[dob]` — *optional* observer table; defaults are the notebook's production
    values (super-twisting, yaw-only by default: `omega_o_psi = 6π`, `k1_psi = 15`,
    `k2_psi = 80`; x/y observer gains 0). Editing the solver/DOB never needs a
    driver change — it's all a `base.toml` edit.

- **Per-profile TOML** — three tables consumed by `resolve_profile`:
  - `[profile]` `builder = "..."` (required).
  - `[profile.params]` — fixed scalars / structural lists, copied verbatim.
  - `[profile.sweep]` — each key gets **one random value** per job (seeded RNG).
  - `[profile.combos]` — **parallel arrays**; one row index `i` selects element
    `i` of every column. Deterministic in the sweep (`combo_idx`), random in
    interactive use. `enumerate_jobs` emits one job per combo row.

### The profile set (excitation intent)

Seven builders, eight TOMLs (multisine ships a 50%- and a 75%-cap variant):

| Profile | Kind | Excitation purpose |
|---|---|---|
| `octagon` | VelRef | start-cruise-stop legs in N directions; `lat_vamp` adds a confined lateral wiggle. Heading held fixed. |
| `long_circle` | VelRef | heading-aligned orbit, `Vx = R·ψ̇`, `Vy = 0`; heading sweeps with the tangent — no drift, no position tracking. |
| `spin_creep` | VelRef | small creep translation + scheduled high yaw-rate pulses (low \|V\| frees wheel-speed budget for large \|Ω\|). |
| `coupled_vomega` | VelRef | independent \|V\| and Ω profiles (const/ramp); rides a (V,Ω) ray. (Formerly "spiral".) |
| `spiral_orbit` | VelRef | heading-locked geometric spiral; modes `om_const` / `v_const` / `iso_accel` probe V-, Ω-, and composition-marginals near the boundary. |
| `multisine` | VelRef | zero-mean harmonic sums on Vx/Vy/(Ω); zippered disjoint combs for decorrelated joint coverage. |
| `ellipse` | **PosRef** | world-position-tracked orbit (keeps the position-controller path alive); tangent or fixed ("crab") heading. |

**Multisine phases are Python-generated and written explicitly into the TOML** —
never rely on cross-language RNG seeding (`pyseed` is a label only).

---

## 5. The cross-language data contract (CRITICAL)

This is the durable interface the restructured Python code must honor.

**Filename scheme** (`DataStore.output_prefix` / `expected_output`):

```
<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
e.g.  octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
```

> **TODO for the Python restructure:** the loader's `parse_arrow_filename` /
> `_FNAME_RE` must parse this scheme — `profile` + `combo` **replace** the old
> `beta` + `amp` fields. This is flagged in `datastore.jl` and is a known
> migration point.

**Required Arrow columns** (exact names — the PINN loader depends on them):

```
Vx, Vy, psi_dot, w1..w4, theta1..theta4, Msat_1..4,
Fx_1..4, Fy_1..4, Mz_1..4, time
```

Reference channels are **body-frame** for VelRef runs (`Vx_des, Vy_des, psi_des,
omega_des, Ax_des, Ay_des, alpha_des`) and **world-frame** for PosRef runs
(`xo_des, yo_des, Vxo_des, Vyo_des, psi_des, omega_des, alpha_des`). The loader
ignores both reference blocks for the required-column contract.

Each run also writes a **JLD2 sidecar** with `sol_t/sol_u`, labels, params,
ASMC config, the resolved trajectory `cfg`, and `meta` — enough to rebuild a run
exactly without consulting the TOMLs.

---

## 6. Repository layout & rules

Root: `C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\`
_(Last updated: 2026-06-19)_

**Top-level structure:**

```
mecanum_pinn_head\
├── .claude/                              Claude Code settings
│   ├── settings.json
│   └── settings.local.json
├── CLAUDE.md                             Root project orientation
├── PROJECT_LAYOUT.md                     Detailed file hierarchy reference
├── code_insights\                        ← PROJECT ROOT (code only)
└── data\                                 ← ALL simulation data (read and write)
```

**`code_insights/` directory (working directory):**

Key subdirectories:
- **Julia simulation subsystem:** `profiles.jl`, `datastore.jl`, `Data_Generation_Julia.jl`, 
  `run_one.jl` — the parallel sweep driver
- **Configuration:** `trajectory_files_run_0p{3,5,8}_{main,quad}/` (μ-specific generated 
  configs) + `trajectory_files_chinc/` (χ-specific study). **No base `trajectory_files/` 
  directory.** Generated by μ-grid scaling; never hand-edit.
- **PINN training:** `train_GPU_PINN_v14_py/` (main package), 
  `Mecanum_PINN_Mamba_ForceRecon_v1/` (force reconstruction), `observer_v1_py/` (observer).
- **Diagnostics:** `chatter_diagnostics.py`, `chi_identifiability.py`, `sampling_sensitivity.py`,
  `tracking_gate.py`, `force_mu_chi_*.py`, `mu_identifiability.py`, `roller_slip_fraction.py`
  → output CSVs in root.
- **Notebooks:** main simulator (`Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb`),
  solver benchmarking (`Solver_Ablation_Multisine.ipynb`), trajectory diagnostics, 
  sampling analysis.
- **Solver benchmarking:** `julia_solver_benchmark_asmc4/` (~5 MB; ablation results
  tables, work-precision plots, notes — tracked in git). The heavy reference
  trajectories (`.jld2`, ~984 MB) now live in `../data/solver_ablation_studies/`;
  `Solver_Ablation_Multisine.ipynb` reads/writes them via `REF_DIR` (mirrors the
  PINN `../data` convention), while `BENCH_DIR` keeps the small in-repo outputs.
- **Outputs:** `rendered_traj_diagnostics/` (16 interactive HTML files), 
  `images_and_plots/` (consolidated figures), `presentation/` (IMECE 2026 deck).
- **Scratch:** `_tmp/` (test artifacts, logs, config sanity checks), 
  `_nb_read_eval/` (notebook reader skill evaluation).

Complete tree: see **[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)** for the detailed breakdown.

**`data/` directory (simulation outputs):**

```
data\
├── Simulation_Data_MecanumSlipSpin_LugreAdamov\  ACTIVE sweep output (~5670 files, 209 GB)
│   └── <profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
│        examples:
│          coupled_vomega_c001_mu_0.3_case1_lugre_adamov_chi_0.000.arrow
│          octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow
│
├── solver_ablation_studies/              solver-ablation reference trajectories (.jld2, ~984 MB)
├── _mu_pilot2/                           earlier pilot sweep (6.1 GB)
├── SimulationDataSlipSpin_Julia/         LEGACY — old beta/amp scheme (2.8 MB)
└── SimulationDataSlipSpin_Julia_3/       LEGACY — old beta/amp scheme (2.6 MB)
```

**Legend:**
- **ACTIVE** = currently maintained, loaded by recent sessions (5670 files)
- **LEGACY** = old `beta`/`amp` filename scheme; reference only
- **CONFIG_DIR** = `trajectory_files_run_0p{3,5,8}_{main,quad}/` or `trajectory_files_chinc/`
  (μ/χ-specific generated artifacts)
- **AUTHORITATIVE** = single source of truth; never edit manually except through tracked commits

See §5 for the active Arrow filename contract and column schema.
See Authority rules (below) for critical constraints.

PyTorch side runs on a **separate machine**: WSL2 Ubuntu, project root
`~/mecanum_pinn_main/`, data under `~/mecanum_pinn_main/data/...`. Not touched by
the Julia sweep.

**Authority rules (respect these):**
1. `base.toml` in each `trajectory_files_run_*` directory is the single source of truth 
   for physics/solver settings — no fallbacks.
2. Module copies (`profiles.jl`, `datastore.jl`) are authoritative; updates replace 
   both the notebook copy and the root copy. Two stale-copy incidents already 
   happened — this rule exists because of them.
3. Everything runs from the project root (notebook kernel pwd = driver pwd).
   `CONFIG_DIR` points to `trajectory_files_run_*` variants.
4. Output files are never hand-edited; an existing `.arrow` = a completed job.
5. Only the notebook's **Cell 2** carries the `parameters` tag — multiple tagged
   cells break `extract_run_one.py`.
6. **No `trajectory_files/` base directory exists.** The `trajectory_files_run_*` 
   directories are generated artifacts, rebuilt by the μ-grid scaling generator. 
   Never hand-edit them; re-tuning the scaling means regenerating all of them. 
   See `mu_generation_note.md`.

---

## 7. The PINN side (intent — current `.py` layout is provisional)

The Python `mecanum_pinn` package is being restructured in Claude Code. Preserve
this *intent*; the module breakdown is up for redesign.

- **Learning targets:** forward dynamics + inverse friction-parameter recovery
  (`μ`, `χ`). The network learns only the **dimensionless friction-law shape**
  (an encoder GRU + a friction-factor head); state integration is an **analytical
  Newton–Euler integrator**, not learned (the old learned `dec2`/`dec3` heads
  were dropped).
- **GRU encoder** reconstructs the unmeasurable hidden roller rates `γ̇ᵢ`; its
  capacity should stay ~4 effective dims (hidden_dim=16 validated optimal in a
  sweep; larger leaks trajectory fingerprints).
- **Single `χ` per training run** is enforced — mixing `χ` corrupts the
  one-to-many supervision. `χ` is a held-out split variable.
- `seq_len` is constrained by the physics-loss lookahead (`k_steps`), not by a
  genuine memory requirement; the GRU acts as a learned implicit solver for the
  `γ̇ᵢ` fixed point.
- Windows/WSL constraints that bit us before: import `pyarrow.feather` first;
  `from torch import _dynamo`; `torch.load(..., weights_only=False)`.
- Two GPUs in play: a 24 GB Quadro RTX 6000 (Turing, no native bf16) and a 6 GB
  RTX 3060 Mobile (Ampere, native bf16).

---

## 8. Working conventions (how this project likes to be worked on)

- **First-principles, code-verified.** Derive before asserting; compute every
  numerical value in code before writing it into a doc or LaTeX. Prefer
  **trajectory-independent, envelope-based bounds** over trajectory-specific
  numbers.
- **Surgical edits.** Provide specific cell/line diffs, not full-file
  regenerations, unless a rewrite is explicitly requested. Read exact filenames
  before acting.
- **Numerical safety:** hard clamps are solver-unsafe (Jacobian discontinuities
  for stiff implicit solvers) — use smooth `Lp` soft-clamps / `tanh` (smoothing
  factor ~200–500, not 10). Avoid additive transcendental terms in error
  functions (they create spurious stable equilibria); multiply by strictly
  positive functions instead.
- **Determinism & provenance:** seeded sweeps, explicit multisine phases in TOML,
  JLD2 sidecars carrying full `cfg` + `meta`.
- **Memory-safe data processing:** per-file streaming (read → accumulate →
  delete → advance) to avoid exhaustion when iterating Arrow files.
- **Scratch / test files go in `_tmp/`** — any dummy notebooks, throwaway
  scripts, smoke-test inputs, or intermediate artifacts that Claude generates
  for verification purposes must be written to `code_insights/_tmp/`. Clean up
  (delete) all generated files after the test is complete. This folder is
  inside `mecanum_pinn_head`, so all file operations there are auto-allowed
  with no permission prompts. **Never scatter test artifacts in the project
  root or in `trajectory_files/`.**
- Acknowledge numerical mistakes immediately and correct them; direct technical
  engagement without hedging is preferred.
- **Math in chat responses: Unicode + code blocks, never LaTeX.** The user's
  terminal client renders responses as plain markdown with **no MathJax/KaTeX**,
  so `$...$` / `$$...$$` (with `\dot`, `\partial`, `\frac`, `\psi`, …) display as
  raw backslash-soup. Write math with Unicode glyphs (γ, ψ̇, ω, δ, θ̃, μ, χ, ∂, ∇,
  →, ≈, ≤, ², ₁, |·|) inline, and put multi-line / aligned equations inside fenced
  code blocks (monospaced, preserves alignment). LaTeX is still correct when
  *authoring* `.tex` / docs / the slide deck (MathJax there) — this rule is for
  chat output only.
- **Long runs need a keep-awake daemon.** This laptop's Modern Standby kills
  background compute (~30 min into idle, even on AC) — Julia sweeps and long
  re-runs vanish with no error. Before launching anything expected to exceed
  ~20 min (a sweep, a full-dataset screen re-run, a generation pass), first start
  `keep_awake.py` in the **background** (`<venv python> keep_awake.py`), then kick
  off the run; stop it when done. It holds `SetThreadExecutionState(ES_CONTINUOUS
  | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)` and re-arms every 60 s. Short runs
  (a few minutes) don't need it. Resume is safe regardless (`.arrow` existence +
  atomic writes), so a standby-killed run relaunches cleanly.
- **Hard cap: never use more than 8 threads / parallel workers** (Julia `-t`,
  Python `--jobs`/`ProcessPoolExecutor`, any pool) unless the user *explicitly*
  specifies a higher number. This machine OOMs under heavier parallelism — 12
  chatter-diagnostics DSP workers exhausted RAM mid-run. Default to ≤8; when in
  doubt, fewer.

---

## 9. Headroom MCP (context compression)

Headroom MCP is configured for this project. Use it proactively when handling
large inputs to reduce token usage:

- **`headroom_compress`** — compress large file contents, logs, or Arrow data
  summaries before processing. Use when reading files >500 lines or large JSON.
- **`headroom_retrieve`** — retrieve original content from a compressed handle
  if the full text is needed after compression.
- **`headroom_stats`** — check compression statistics.

The Headroom proxy must be running (`headroom proxy` in `myenv` conda env) for
the MCP tools to function. If tools error, the proxy is likely not running.

---

## 10. Quick reference (defaults)

| Thing | Default |
|---|---|
| Config dir | `trajectory_files` |
| Output dir | `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov` (relative to project root) |
| Simulator script | `run_one.jl` (from `extract_run_one.py`) |
| Solver | `TRBDF2`, `reltol 1e-8`, `dtmax 1e-3`, `saveat 2000 Hz` |
| Per-solve timeout | 300 s (`--timeout`/`--no-timeout`) |
| Sweep seed | 1234 (must match across resume passes) |
| Job log | `<outdir>/jobs_log_profiles.jsonl` |
| DOB default | super-twisting, yaw-only (`omega_o_psi = 6π`, `k1_psi 15`, `k2_psi 80`) |
