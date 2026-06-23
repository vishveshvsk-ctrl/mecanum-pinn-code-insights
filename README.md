# Mecanum PINN Digital Twin

Physics-Informed Neural Network (PINN) digital twin of a KUKA youBot four-Mecanum-wheel omnidirectional platform, targeting an **IMECE 2026** paper.

The project is split into two halves joined by a file-format contract:

```
JULIA                                  PYTHON
high-fidelity ODE simulator  ──►  Arrow files  ──►  PyTorch PINN
(39-D plant + ASMC + DOB)          (+ JLD2 sidecar)   (forward dynamics +
                                                      inverse friction ID)
```

The PINN has two jobs: learn the **forward dynamics** (per-wheel roller-frame forces), and recover friction parameters (`μ`, `χ`) by inverse identification. The Julia simulator manufactures richly-exciting, physically-valid trajectory data with ground-truth labels.

---

## Repository layout

The working tree is the `code_insights/` code root. Simulation data lives in a sibling `data/` directory (outside this repo). A detailed file-by-file map is in [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md); project orientation and authority rules are in [`CLAUDE.md`](CLAUDE.md).

```
.
├── CLAUDE.md                         # project orientation & authority map
├── PROJECT_LAYOUT.md                 # detailed file hierarchy
├── Data_Generation_Julia.jl          # parallel sweep driver (Julia)
├── profiles.jl                       # trajectory/excitation library (Julia)
├── datastore.jl                      # label extraction & Arrow I/O (Julia)
├── run_one.jl                        # simulator extracted from notebook
│
├── trajectory_files_run_0p5_main/    # μ=0.5 production configs
├── trajectory_files_run_0p3_main/    # μ=0.3 scaled configs
├── trajectory_files_run_0p8_main/    # μ=0.8 scaled configs
├── trajectory_files_run_*_quad/      # χ-quad near-cap subsets
│
├── chatter_diagnostics.py            # spectral chatter/hash screen
├── tracking_gate.py                  # tracking/friction-circle gate
├── chi_identifiability.py            # χ identifiability from forces
├── mu_identifiability.py             # μ multiplicativity check
├── sampling_sensitivity.py           # training-rate sensitivity
├── blend_reports.py                  # fuse screens → diagnostics_combined.csv
│
├── train_GPU_PINN_v14_py/            # modular GRU PINN (legacy v14)
├── Mecanum_PINN_Mamba_ForceRecon_v1/ # Approach 1: Mamba force-recon PINN
├── observer_v1_py/                   # Approach 2: SSM/GRU state observer
│
├── julia_solver_benchmark_asmc4/     # solver ablation results
├── rendered_traj_diagnostics/        # per-profile HTML reports
├── presentation/                     # IMECE 2026 slide deck
└── docs/                             # LaTeX derivations + reference PDFs
```

---

## The Julia simulator

A 39-D stiff ODE plant based on Adamov & Saypulaev (2021) with deliberate modifications:

- **12 rollers per wheel** (paper uses 6), roller inertia `J_roller = 1e-6 kg·m²`.
- **Composite LuGre + Adamov friction** with slip–spin coupling (`:lugre_adamov` or `:lugre_uncoupled`).
- Platform constants: `R = 0.05 m`, `Ra = 0.0355 m`, roller angle `δ = 45°` (O-configuration), mass `30 kg`, yaw inertia `Is = 4.42 kg·m²`.
- **ASMC + super-twisting DOB** controllers for velocity/position tracking.

State order (39-D): `[Vx, Vy, ψ̇, ψ, θ₁..₄, ω₁..₄, γ₁..₄, Kx, Ky, Kψ, xo, yo, zx₁..₄, zy₁..₄, zs₁..₄, observer, δ̂]`.

### Run a sweep

```bash
julia --project=. -t auto Data_Generation_Julia.jl
julia --project=. -t 8 Data_Generation_Julia.jl --dry-run
julia --project=. -t 8 Data_Generation_Julia.jl --timeout 900
```

Thread count is set by Julia's `-t` flag. `--dry-run` prints a resume-aware pending/done table. The sweep seed (`--sweep-seed`, default `1234`) must match across resume passes.

Configuration is read from `base.toml` and per-profile TOMLs under `trajectory_files_run_*/`. See [`mu_generation_note.md`](mu_generation_note.md) for how the μ=0.3 and μ=0.8 grids were scaled from the μ=0.5 reference.

---

## Data contract

**Arrow filename scheme:**

```
<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow
```

Example: `octagon_c042_mu_0.5_case1_lugre_adamov_chi_0.002.arrow`

**Required Arrow columns:** `Vx, Vy, psi_dot, w1..w4, theta1..theta4, Msat_1..4, Fx_1..4, Fy_1..4, Mz_1..4, time`. Additional columns (`Fpar/Fperp`, `util`, `Msw/Meq`, `Vpx/Vpy/wz`, bristle states) are used by diagnostics.

Each run also writes a **JLD2 sidecar** with full state, params, ASMC config, and trajectory metadata for exact reproduction.

---

## Diagnostics pipeline

Four screens are run over the sweep output, then fused into one whitelist:

```bash
DATA=../data/Simulation_Data_MecanumSlipSpin_LugreAdamov
python chatter_diagnostics.py   --data-dir $DATA --out chatter_report.csv
python sampling_sensitivity.py  --data-dir $DATA --out sampling_sensitivity.csv --rates 1000,500
python tracking_gate.py         --data-dir $DATA --out tracking_report.csv
for mu in 0.3 0.5 0.8; do
  python chi_identifiability.py --data-dir $DATA --mu $mu --whitelist diagnostics_combined.csv \
         --out chi_identifiability_mu$(echo $mu | tr -d .).csv
done
python blend_reports.py         # → diagnostics_combined.csv
```

`diagnostics_combined.csv` is the single source of truth for the training whitelist. Findings and thresholds are documented in [`TRAJ_DIAGNOSTICRESULTS.md`](TRAJ_DIAGNOSTICRESULTS.md); the original spec is in [`Trajectory_Chatter_Diagnostics_PLAN.md`](Trajectory_Chatter_Diagnostics_PLAN.md).

Headline results:

- **Whitelist: 5,345 / 5,670 trajectories (94.3%)** across μ ∈ {0.3, 0.5, 0.8}.
- `hash` (incoherent numerical noise): **0**.
- Train the PINN at **500 Hz**; run the chatter screen at native **2000 Hz**.
- Drop `Mz` from training; recover `χ` from `Fpar/Fperp`.

---

## PINN training

Two complementary approaches share the same regime TOMLs and whitelist.

### Approach 1 — Force-reconstruction forward-inverse PINN

`Mecanum_PINN_Mamba_ForceRecon_v1/`

A Mamba-S6 selective-SSM encoder learns forward dynamics (`Fpar/Fperp`) and recovers `μ̂`/`χ̂` at test time from the inverse model residual. See [`Mecanum_PINN_Mamba_ForceRecon_v1/README.md`](Mecanum_PINN_Mamba_ForceRecon_v1/README.md).

```bash
python Mecanum_PINN_Mamba_ForceRecon_v1/train.py both
```

### Approach 2 — Supervised neural state observer

`observer_v1_py/`

A small SSM/GRU observer reconstructs the unmeasurable per-wheel contact states (`γ`, `zx`, `zy`, derived `ω_z`) from measurable signals only. See [`observer_v1_py/README.md`](observer_v1_py/README.md).

```bash
# warm cache, then run 3 concurrent jobs
python observer_v1_py/launch_parallel.py --warm-cache --max-parallel 3

# aggregate results
python observer_v1_py/make_observability_report.py
```

### Legacy modular PINN

`train_GPU_PINN_v14_py/`

The earlier GRU-based modular training package. See [`train_GPU_PINN_v14_py/README.md`](train_GPU_PINN_v14_py/README.md) and [`train_GPU_PINN_v14_py/CLI_QUICKREF.md`](train_GPU_PINN_v14_py/CLI_QUICKREF.md).

---

## Training-data splits

Data-level train/val/test splits are defined in [`training_data_split_design.md`](training_data_split_design.md):

- **S1/S2** — excitation 2-fold (single χ = 0.005, all μ), stratified so both folds cover the 21 plausible excitation cells.
- **S3** — χ k-fold cross-validation (matched χ-quads on octagon/spin_creep/coupled_vomega).

Both PINN approaches consume the same partitions via `observer_v1_py/regimes/*.toml`.

---

## Documentation & references

- [`CLAUDE.md`](CLAUDE.md) — project orientation, authority map, conventions.
- [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) — complete directory tree and rules.
- [`TRAJ_DIAGNOSTICRESULTS.md`](TRAJ_DIAGNOSTICRESULTS.md) — diagnostic findings.
- [`training_data_split_design.md`](training_data_split_design.md) — split design.
- [`mu_generation_note.md`](mu_generation_note.md) — μ-grid scaling justification.
- `docs/` — LaTeX derivations and reference PDFs, including:
  - `Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex` — velocity/acceleration envelope.
  - `ASMC_Velocity_Tracking_Stability_PoseTrack_2.tex` — ASMC stability analysis.
- `chat-handoff/` — cross-session task handoff briefs.

---

## Conventions

- `base.toml` is the single source of truth for physics/solver constants.
- `profiles.jl` and `datastore.jl` are authoritative; keep copies in sync.
- Everything runs from the project root.
- Output files are never hand-edited; existing `.arrow` files are treated as completed jobs.
- Use ≤8 threads/parallel workers unless explicitly told otherwise.
- Long runs: start `keep_awake.py` in the background first (Modern Standby workaround).
- Scratch/tests go in `_tmp/` and are cleaned up after.
