# Agent Instructions — Mecanum PINN Digital Twin

Hard rules and quick reference for agents working on this repository. For full technical context, read [`CLAUDE.md`](CLAUDE.md) and [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md). For per-task handoffs, check `chat-handoff/`.

---

## 1. Project overview

PINN digital twin of a KUKA youBot four-Mecanum-wheel platform, targeting an **IMECE 2026** paper. Julia 39-D stiff ODE simulator → Arrow files (+ JLD2 sidecars) → PyTorch PINN. The PINN learns forward dynamics and recovers friction parameters (`μ`, `χ`).

---

## 2. Authority rules

1. **`base.toml` in each `trajectory_files_run_*`** — single source of truth for physics/solver settings.
2. **`profiles.jl` / `datastore.jl`** — if updated, replace all copies (notebook + root).
3. **Everything runs from the project root** — `CONFIG_DIR` resolves to `trajectory_files_run_*` directories.
4. **Output files are never hand-edited** — an existing `.arrow` = a completed simulation.
5. **Only Cell 2** in the simulator notebook carries the `parameters` tag.
6. **Data immutability** — Arrow files in `data/` are read-only; new outputs require new combo IDs.
7. **Profile-based enumeration** — old beta/amplitude grid is **DEPRECATED**; all configs use profile names.

### Authoritative artifacts

| File | Role |
|---|---|
| `base.toml` + per-profile TOMLs | Physics/solver/sweep config |
| `profiles.jl` | Trajectory/excitation library |
| `datastore.jl` | Labels, DataFrame, Arrow I/O, filename scheme |
| `Data_Generation_Julia.jl` | Parallel sweep driver |
| `diagnostics_combined.csv` | Training whitelist |

---

## 3. Cross-language data contract

**Filename scheme:**

```
<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
```

Example: `octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow`

**Required Arrow columns:** `Vx, Vy, psi_dot, w1..w4, theta1..theta4, Msat_1..4, Fx_1..4, Fy_1..4, Mz_1..4, time`.

Each run writes a **JLD2 sidecar** with full state, params, ASMC config, `cfg`, and `meta` for reproduction.

---

## 4. Working conventions

- **First-principles, code-verified.** Compute every numerical value in code before asserting it. Prefer trajectory-independent, envelope-based bounds.
- **Surgical edits.** Specific line/cell diffs, not full-file regenerations unless asked.
- **No hard clamps** in functions seen by stiff implicit solvers — use smooth `Lp` soft-clamps or `tanh` (smoothing ~200–500).
- **Determinism:** seeded sweeps, explicit multisine phases in TOML, JLD2 sidecars with `cfg` + `meta`.
- **Streaming data processing:** `read → accumulate → delete → advance` over Arrow/JLD2 corpora.
- **Temp/scratch files go in `_tmp/`**. Clean up after. Never scatter artifacts in the project root or `trajectory_files/`.
- **Never write data to the code root**; simulation outputs go in `../data/`.
- **Math in chat:** Unicode + fenced code blocks, not LaTeX `$...$`.
- **Long runs >20 min:** start `keep_awake.py` in the background first.
- **Parallelism cap: ≤8 threads/workers** unless explicitly requested.
- Acknowledge numerical mistakes immediately and correct them.

---

## 5. Quick reference defaults

| Thing | Default |
|---|---|
| Config dir | `trajectory_files_run_*` variants |
| Output dir | `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov` |
| Solver | `TRBDF2`, `reltol 1e-8`, `dtmax 1e-3`, `saveat 2000 Hz` |
| Sweep seed | `1234` (must match across resume passes) |
| Per-solve timeout | 300 s |
| DOB | super-twisting, yaw-only (`omega_o_psi = 6π`, `k1_psi = 15`, `k2_psi = 80`) |
| Training rate | 500 Hz; chatter screen at 2000 Hz |

---

## 6. Windows / WSL live-data execution model

The git working directory (`/home/vishveshvsk07/mecanum-pinn-code-insights`) is **not** the live data location. Because the large simulation dataset is kept in only one place, the authoritative runtime tree is the Windows-synced folder mounted under WSL:

```
/mnt/c/Users/vishv/OneDrive/Desktop/Vishvesh_Data/VNIT/mecanum_pinn_head/code_insights/
```

The project root is documented in [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md):

```
C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\
```

### Hard rules for live runs

1. **Live status checks** — always read from the `/mnt/c/.../code_insights/` counterpart, never from the git working directory. Relevant files include `sweep_status.txt`, per-run `.log` files, and any JSON status dumps in `_parallel_logs_*/` directories.
2. **Training / simulation / sweep code** — whenever the task involves a non-trivial runnable script (Python or Julia), produce **two artifacts**:
   - The script itself, written against the `/mnt/c/.../code_insights/` layout.
   - An accompanying `.bat` file that runs the same script from Windows (using the equivalent `C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\` path).
3. **No duplicate large datasets** — scripts must read/write data in the single Windows-mounted tree. Do not design workflows that copy `../data/` or `.arrow` corpora into the git working directory.
4. **Git as transport + post-run archive** — commit scripts and `.bat` files to the repo; the user will pull/transfer them into the `/mnt` tree for execution. While a run is **live**, logs/status files exist only in the `/mnt` tree. After a run **completes**, final logs and status files are pushed to git so the repo has the archived record.
