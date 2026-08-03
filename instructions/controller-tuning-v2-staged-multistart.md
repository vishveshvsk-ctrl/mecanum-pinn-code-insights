# Controller Tuning v2 — Expanded Trajectory Set + Staged Multi-Start Optimization

> **Generated:** 2026-07-28
> **Stack:** Julia 1.12.5 · BlackBoxOptim 0.6.9 · OSQP 0.8.1 · OrdinaryDiffEq (FBDF) · StaticArrays · JSON
> **Scope:** Offline controller tuning pipeline (no plant/physics changes, no PINN, no training-data generation)
> **Iteration constraint:** μ = 0.5 ONLY. Friction diversity is explicitly deferred to a later iteration.

---

## 1. Overview

The current controller tuner (`tune_controller.jl`) runs a single dxNES search over a fixed
evaluation budget against a 6-trajectory all-pose training set. Trace analysis of the most
recent pose-mode runs shows every run was **terminated mid-descent**, not converged:

| Run | Budget → evals | Last improvement | Score across 5 seeds |
|---|---|---|---|
| ASMC (`runs_controller_asmc_pose_5seed`) | 50 → 61 | eval **55–58 of 61** | 1.17 / 1.18 / 1.60 / 2.73 / 2.99 (**2.6× spread**) |
| PID (`runs_controller_pid_pose_5seed`) | 150 → 161 | eval **143 of 161** | 1.45 – 2.56 |

Consequently the current inter-seed scatter cannot be attributed to a sloppy objective
manifold — it is indistinguishable from an unexplored one. The reported
non-identifiability of `gamma_x`/`gamma_y` is therefore **not yet a valid result**.

This brief specifies a v2 tuning pipeline that:

1. Replaces the 6-trajectory training set with a **12-trajectory diverse set** (adds
   broadband/multisine, step-and-hold/docking, and easy anchors) plus a **9-entry held-out
   test grid** covering the profile, combo, and task axes.
2. Introduces a **two-phase tiered objective** (cheap 4-trajectory SCREEN tier for global
   search → full 12-trajectory tier for refinement) so an expanded set does not multiply
   wall-clock linearly.
3. Replaces the fixed-budget stop with a **convergence gate** (plateau detection + hard cap)
   and records a `converged` flag in every result file.
4. Runs **exactly 5 seeds** per controller — 5 Sobol-offset starting points × 5 optimizer
   RNG seeds — as independent restarts, reported with cross-seed statistics.
5. Executes in **four stages of strictly increasing complexity**: pipeline-only (Stage 0) →
   ASMC (Stage 1) → PID (Stage 2) → MPC (Stage 3), each independently shippable.

**System-level contract:** given a controller symbol, a trajectory-set name, and a seed, the
pipeline emits a converged gain vector, its cross-seed reproducibility statistics, and its
score on a held-out test grid — writing `best_config.json` / `trials.json` / `trace.csv` /
`seed_report.json` under a fresh, per-stage output root.

---

## 2. Architecture Pattern

**Staged tiered-fidelity multi-start optimization with a preserved legacy path.**

- *Tiered fidelity* — a cheap SCREEN tier drives global exploration; the expensive FULL tier
  is entered only for local refinement of the screen-phase optimum. This is the standard
  successive-halving trade applied over trajectories rather than over epochs, and it is what
  makes a 12-trajectory training set affordable.
- *Multi-start* — 5 independent restarts replace one long run. This converts the seed spread
  from an embarrassment into the measurement: identical basins across starts ⇒ converged;
  distinct basins ⇒ a genuine multimodality finding.
- *Preserved legacy path* — every existing `--trajset` value and code path must remain
  byte-for-byte reproducible (see §11 Preservation Constraint). All new behaviour is additive
  and gated behind new flags.

Rationale for staging by controller rather than by feature: the three controllers currently
sit at very different quality levels (ASMC acceptable, PID moderately worse, MPC badly worse)
and require correspondingly different amounts of surgery. Stage 1 changes no controller math
at all and therefore validates the new pipeline against a known-good baseline before any
controller is touched.

---

## 3. Technology Constraints

- **Julia:** 1.12.5 (per `Manifest.toml`)
- **Environment:** `Pkg.activate(code_insights/)` — the existing project
- **Already available (no install):** `BlackBoxOptim` 0.6.9 (dxNES, adaptive DE), `Sobol`,
  `QuasiMonteCarlo`, `Surrogates`, `OSQP` 0.8.1, `OrdinaryDiffEq`, `StaticArrays`, `JSON`,
  `TOML`, `ForwardDiff`, `LinearAlgebra`
- **New dependencies — Stage 2/3 only, both optional with fallbacks:**
  - `NLopt.jl` — BOBYQA/NEWUOA derivative-free trust-region refiner. Reason: at 5–8
    dimensions with a deterministic, ~1-minute-per-eval objective, a quadratic-model trust
    region is far more eval-efficient than a population method. **Must degrade gracefully:**
    if `NLopt` is absent, fall back to the existing local-refine loop already inside
    `optimize()` in `tune_controller.jl`.
  - `MatrixEquations.jl` (or `ControlSystemsBase.jl`) — discrete algebraic Riccati solve
    (`dare`) for the MPC terminal weight. Stage 3 only. If absent, Stage 3 runs without the
    terminal cost and logs a warning.
- **Device targets:** CPU only. `LinearAlgebra.BLAS.set_num_threads(1)` MUST remain — it is
  the existing determinism guarantee and a chaotically-sensitive closed loop amplifies BLAS
  reduction-order variation into divergent-vs-stable outcomes.
- **Parallelism:** **process-level only.** In-process threading (BlackBoxOptim `NThreads`,
  `Threads.@threads` over trajectories) is **UNSAFE** — `Profiles.ACTIVE_KIND` is a global
  `Ref` and `SchedulerMod.ESTIMATOR_PROBE_LOG` is an unlocked global `Dict`. Seeds run as 5
  separate `julia` processes (already demonstrated: the existing `seed1..seed5` logs share a
  ~60 s mtime window).
- **Explicit exclusions:** no friction diversity (μ=0.3/0.8 profiles are OUT this iteration);
  no changes to `plant.jl`, `sensors.jl`, `estimators.jl`, or any physics; no Arrow/data
  reads (the tuner simulates live and reads no `../data/` file — this is what permits
  non-PINN-whitelist trajectories such as `docking`).

---

## 4. Component Breakdown

### `TrajSetsMod` (`hybrid_ctrl/controller_tuning/trajsets.jl`)
- **Type:** module
- **Responsibility:** Single registry of every named trajectory tier (SCREEN / FULL-TRAIN /
  TEST), returning entry NamedTuples in exactly the shape `run_controller` already consumes.
- **Inputs:** `run_dir::String`, tier name `Symbol`
- **Outputs:** `Vector{NamedTuple}` — each entry
  `(name::String, profile_toml::String, combo_idx::Int, ref_type::Symbol, mu::Float64,
    config_dir::String, run_mode::Symbol, adapt::Bool, tier::Symbol, role::Symbol)`
- **Key constructor params:** none (pure functions)
- **Depends on:** nothing (leaf module). Must NOT import `tune_controller.jl`.
- **Note:** `run_mode` is `:pose` for every entry in every tier this iteration; `adapt=true`
  for entries whose underlying builder returns a `VelRef` (lifted via
  `Profiles.velref_to_posref`), `adapt=false` for native `PosRef` builders.

### `StageObjectiveMod` (`hybrid_ctrl/controller_tuning/stage_objective.jl`)
- **Type:** module
- **Responsibility:** Wrap the existing `make_objective` into a tier-switchable objective and
  add the per-trajectory metric breakdown needed by the seed report.
- **Inputs:** controller `Symbol`, param space (`Vector{Tuple}`), tier trajectory vector,
  oracle kind `Symbol`, penalty weights
- **Outputs:** a closure `theta::Vector{Float64} -> NamedTuple` carrying at minimum
  `(score, tracking, ce, chatter, per_traj::Dict{String,NamedTuple}, n_fail::Int)`
- **Key constructor params:** `lambda_chatter`, `lambda_kmax`, `lambda_gamma`,
  `recovery_weight` (new — see §7), `seed`
- **Depends on:** `run_controller` / `controller_metrics` from `tune_controller.jl` (via
  `include`, exactly as `experiment_noise_eval_pose.jl` already does), `TrajSetsMod`

### `StageOptimizerMod` (`hybrid_ctrl/controller_tuning/optimizer_stage.jl`)
- **Type:** module
- **Responsibility:** Two-phase driver — global search on the SCREEN tier, local refine on
  the FULL tier — with plateau-based convergence gating and Sobol multi-start offsets.
- **Inputs:** two objective closures (screen, full), flat bounds `lo`/`hi`, phase budgets,
  convergence-gate parameters, seed
- **Outputs:**
  `(best_theta::Vector{Float64}, best_score::Float64, trials::Vector{NamedTuple},
    diag::NamedTuple)` where `diag` carries
  `(converged::Bool, phase1_evals::Int, phase2_evals::Int, last_improvement_eval::Int,
    plateau_window::Int, stop_reason::Symbol)`
- **Key constructor params:** `method::Symbol` (`:dxnes`/`:de`), `refiner::Symbol`
  (`:bobyqa`/`:local`), `rel_tol::Float64`, `window::Int`, `hard_cap::Int`
- **Depends on:** `BlackBoxOptim`, `Sobol`, optionally `NLopt`

### `PIDCascadeMod` (`hybrid_ctrl/controller_tuning/pid_cascade.jl`) — Stage 2
- **Type:** module
- **Responsibility:** Provide the masked/sequential PID parameter spaces (inner-loop-only,
  outer-loop-only, joint-polish) and the `I_max` extension.
- **Inputs:** phase `Symbol` (`:inner`/`:outer`/`:joint`), frozen-gain NamedTuple
- **Outputs:** `(space::Vector{Tuple}, freeze::NamedTuple)` — `freeze` supplies the values
  for parameters excluded from this phase's space
- **Depends on:** nothing (leaf); consumed by `StageObjectiveMod` via a decode hook

### `MPCDesignMod` (`hybrid_ctrl/controller_tuning/mpc_design.jl`) — Stage 3
- **Type:** module
- **Responsibility:** Analytic MPC weight construction — Bryson-rule seeding from the
  existing `TOL` tuple, scale normalization, and the Riccati terminal weight.
- **Inputs:** `TOL` NamedTuple, `params` (PlatformParams), `motor` (MotorParams), `dt`,
  horizon `Np`
- **Outputs:** `(Q_pose::SVector{6}, R::SVector{4}, S::SVector{4}, P_terminal::Matrix{Float64})`
- **Depends on:** `_mpc_body_matrices` from `ControllerMod`, optionally `MatrixEquations`

### `run_stage.jl` (`hybrid_ctrl/controller_tuning/run_stage.jl`)
- **Type:** script (CLI entry point)
- **Responsibility:** Parse stage/controller/seed arguments, assemble the modules above,
  execute one seed of one stage, write all artefacts.
- **Inputs:** CLI flags (see §6)
- **Outputs:** files under `runs_controller_v2_<stage>/seed<N>/<ctrl>_<oracle>/`
- **Depends on:** all modules above

### `analyze_seeds.jl` (`hybrid_ctrl/controller_tuning/analyze_seeds.jl`)
- **Type:** script
- **Responsibility:** Cross-seed convergence + identifiability report; evaluate the winning
  gains on the held-out TEST tier.
- **Inputs:** a stage output root
- **Outputs:** `seed_report.json`, `seed_report.md`, `test_grid_eval.csv`
- **Depends on:** `StageObjectiveMod`, `TrajSetsMod`

### `tune_controller.jl` — MODIFIED (additive only)
- **Type:** existing script
- **Responsibility (added):** expose `run_controller`, `controller_metrics`, `decode`,
  `flat_bounds`, `space_for`, `build_controller` for reuse; accept a `freeze` NamedTuple in
  `build_controller` so Stage 2 can pin parameters outside the active phase space.
- **Must not change:** `default_trajs`, `default_trajs_2`, `default_trajs_3`,
  `default_trajs_pose`, `TOL`, `LAMBDA_CE`, `V_MAX`, `ASMC_SPACE`, `PID_SPACE`,
  `K_MAX_PIN`, or the behaviour of any existing `--trajset` value.

### `controllers.jl` — MODIFIED (Stage 3 ONLY)
- **Type:** existing module
- **Responsibility (added):** `MPCController` gains a `P_terminal` field (terminal state
  weight, defaulting to a zero matrix so existing behaviour is unchanged) and `Np`/`Np_pose`
  become tunable via the constructor as they already are.
- **Stages 0–2 must not touch this file.**

---

## 5. File & Directory Structure

```
code_insights/
├── tune_controller.jl                         # MODIFIED — additive exports + freeze hook
├── hybrid_ctrl/
│   ├── controllers.jl                         # MODIFIED — Stage 3 only (P_terminal field)
│   └── controller_tuning/
│       ├── trajsets.jl                        # NEW — tier registry (SCREEN/FULL/TEST)
│       ├── stage_objective.jl                 # NEW — tier-switchable objective + breakdown
│       ├── optimizer_stage.jl                 # NEW — 2-phase driver, convergence gate, Sobol multistart
│       ├── pid_cascade.jl                     # NEW — Stage 2 masked/sequential PID spaces
│       ├── mpc_design.jl                      # NEW — Stage 3 Bryson + Riccati helpers
│       ├── run_stage.jl                       # NEW — CLI entry, one seed of one stage
│       └── analyze_seeds.jl                   # NEW — cross-seed + held-out test report
├── trajectory_files_run_0p5_main/profiles/
│   ├── docking_mu_0p5.toml                    # EXISTS — reused unchanged as DOCK-A
│   └── docking_step_mu_0p5.toml               # NEW — DOCK-B: larger offset + heading step
├── run_stage_asmc_5seed.bat                   # NEW — Stage 1 launcher (5 parallel processes)
├── run_stage_pid_5seed.bat                    # NEW — Stage 2 launcher
├── run_stage_mpc_5seed.bat                    # NEW — Stage 3 launcher
├── runs_controller_v2_stage0/                 # NEW — pipeline validation output
├── runs_controller_v2_asmc/                   # NEW — Stage 1 output (seed1..seed5)
├── runs_controller_v2_pid/                    # NEW — Stage 2 output
└── runs_controller_v2_mpc/                    # NEW — Stage 3 output
```

Per-seed output layout (unchanged schema plus two new files):

```
runs_controller_v2_<stage>/seed<N>/<ctrl>_<oracle>/
├── best_config.json      # + "converged", "stop_reason", "phase1_evals", "phase2_evals", "tier"
├── trials.json           # + "tier" per trial row
├── trace.csv             # + tier column: iter,tier,score,best_so_far
└── phase_summary.json    # NEW — per-phase best/evals/wall-clock
```

---

## 6. Key Interfaces

Julia signatures with docstrings. Bodies are stubs — no logic is specified here.

```julia
# ---- trajsets.jl -----------------------------------------------------------

"""
    trajset(tier::Symbol, run_dir::AbstractString) -> Vector{NamedTuple}

Return the trajectory entries for one tier.

Args:
    tier:    :screen | :train_full | :test
    run_dir: config directory, e.g. "trajectory_files_run_0p5_main"

Returns:
    Vector of entries shaped exactly like `default_trajs_pose` output, each with
    the extra fields `tier::Symbol` and `role::Symbol`. Every entry has
    run_mode=:pose and mu=0.5 this iteration.
"""
function trajset(tier::Symbol, run_dir::AbstractString) end

"""
    tier_names() -> NTuple{3,Symbol}

The canonical tier ordering (:screen, :train_full, :test). Used for validation.
"""
function tier_names() end


# ---- stage_objective.jl ----------------------------------------------------

"""
    make_stage_objective(ctrl, space, trajs, oracle_kind; kwargs...) -> Function

Build a tier-scoped objective closure.

Args:
    ctrl:        :asmc | :pid | :mpc
    space:       parameter space rows (name, len, scale, lo, hi)
    trajs:       trajectory entries from `trajset`
    oracle_kind: :clean | :noisy

Keyword Args:
    seed::Int
    freeze::NamedTuple        # gains pinned outside this phase's space (Stage 2/3)
    lambda_chatter::Float64
    lambda_kmax::Float64
    lambda_gamma::Float64
    recovery_weight::Float64  # weight on the post-step recovery-time term (docking only)

Returns:
    theta::Vector{Float64} -> NamedTuple with fields
      score, tracking, ce, chatter, kmax_pen, gamma_pen, recovery,
      per_traj::Dict{String,NamedTuple}, n_fail::Int
    On any failed/non-finite trajectory the score is the existing 1e6 sentinel.
"""
function make_stage_objective(ctrl::Symbol, space, trajs, oracle_kind::Symbol; kwargs...) end

"""
    recovery_time(probe, ref, tol_pos::Float64) -> Float64

Time (s) from the end of a step/approach segment until |position error| re-enters
and stays inside `tol_pos`. Returns the full trajectory duration if never
recovered. Defined only for step-and-hold entries (role == :step_hold); returns
0.0 for all other roles so it contributes nothing to their score.
"""
function recovery_time(probe, ref, tol_pos::Float64) end


# ---- optimizer_stage.jl ----------------------------------------------------

"""
    optimize_staged(obj_screen, obj_full, lo, hi; kwargs...)
        -> (best::Vector{Float64}, best_score::Float64,
            trials::Vector{NamedTuple}, diag::NamedTuple)

Two-phase driver: global search against `obj_screen`, then local refinement of
the phase-1 optimum against `obj_full`.

Keyword Args:
    method::Symbol         # :dxnes | :de           (phase 1)
    refiner::Symbol        # :bobyqa | :local       (phase 2; :local = no NLopt)
    seed::Int
    start_offset::Int      # Sobol index for this seed's start point
    p1_cap::Int            # hard eval cap, phase 1
    p2_cap::Int            # hard eval cap, phase 2
    rel_tol::Float64       # plateau threshold, relative improvement
    window::Int            # plateau window in evals
    refine_halfwidth::Float64  # phase-2 box as a fraction of (hi - lo)
    trace_path::AbstractString

Returns:
    diag carries converged::Bool, stop_reason::Symbol
    (:plateau | :cap | :failed), phase evals, and last_improvement_eval.

Note: the phase-2 box is centred on the phase-1 optimum and clipped to [lo, hi];
the returned best_score is ALWAYS a FULL-tier score, never a screen-tier score.
"""
function optimize_staged(obj_screen::Function, obj_full::Function,
                         lo::Vector{Float64}, hi::Vector{Float64}; kwargs...) end

"""
    plateau_reached(best_curve::Vector{Float64}, rel_tol::Float64, window::Int) -> Bool

True when the best-so-far curve has improved by less than `rel_tol` (relative to
the current best) over the trailing `window` evaluations. Guards against the
known false-positive in the existing diagnose_convergence.py verdict, which fires
on numerical position alone without gating on improvement MAGNITUDE.
"""
function plateau_reached(best_curve::Vector{Float64}, rel_tol::Float64, window::Int) end

"""
    sobol_start(lo, hi, index::Int) -> Vector{Float64}

Deterministic Sobol point `index` in the box, used as seed N's starting mean.
Sobol.jl is already in the Manifest (pulled in by Surrogates).
"""
function sobol_start(lo::Vector{Float64}, hi::Vector{Float64}, index::Int) end


# ---- pid_cascade.jl (Stage 2) ----------------------------------------------

"""
    pid_phase_space(phase::Symbol, frozen::NamedTuple) -> Tuple{Vector,NamedTuple}

Parameter space and freeze-set for one cascade tuning phase.

Args:
    phase:  :inner  -> Kp, Ki, Kd, I_max              (9 + 3 = 12 params)
            :outer  -> Kp_pos, Kd_pos                 (6 params)
            :joint  -> all 18, narrow box around the sequential optimum
    frozen: gains held fixed for parameters absent from this phase's space

Returns:
    (space, freeze) where `space` uses the same row format as PID_SPACE.
"""
function pid_phase_space(phase::Symbol, frozen::NamedTuple) end


# ---- mpc_design.jl (Stage 3) -----------------------------------------------

"""
    bryson_weights(tol::NamedTuple, v_max::Float64) -> (SVector{6}, SVector{4})

Bryson-rule state and input weights derived from the existing absolute-error
tolerance tuple and the motor voltage limit. Returns (Q_pose, R). These are
DESIGN values, not search variables — they remove 10 of the 11 current MPC
search dimensions.
"""
function bryson_weights(tol::NamedTuple, v_max::Float64) end

"""
    terminal_weight(params, motor, dt::Float64, Q::SVector{6}, R::SVector{4}) -> Matrix{Float64}

Discrete-algebraic-Riccati terminal state weight for the 6-state pose model built
by `ControllerMod._mpc_body_matrices`. Returns a 6x6 zero matrix (i.e. no terminal
cost) if MatrixEquations/ControlSystemsBase is unavailable, and logs a warning.
"""
function terminal_weight(params, motor, dt::Float64, Q, R) end

"""
    normalized_mpc_space() -> Vector{Tuple}

The Stage-3 MPC search space: the RATIO parameters only.
`R_scale` is PINNED to 1.0 because scaling (Q, R, S) by any c > 0 scales both H
and f in the OSQP problem by c and leaves the argmin and all constraints
unchanged -- an exactly flat direction that degenerates the dxNES covariance.
Velocity-mode `Q` is EXCLUDED because every trajectory in every tier runs
mode=:pose this iteration, so `mpc.Q` is never read.
"""
function normalized_mpc_space() end


# ---- run_stage.jl CLI ------------------------------------------------------
#   --stage        0 | 1 | 2 | 3
#   --controller   asmc | pid | mpc
#   --seed         1..5
#   --trajset-screen / --trajset-full   tier names (default :screen / :train_full)
#   --noise        clean | noisy
#   --p1-cap / --p2-cap                 hard eval caps per phase
#   --rel-tol / --window                convergence-gate parameters
#   --refiner      bobyqa | local
#   --out          output root
#   --smoke        one eval per tier, no optimization
```

---

## 7. Data Flow

### 7.1 Trajectory tiers (the substantive set change)

All entries: `run_dir = "trajectory_files_run_0p5_main"`, μ = 0.5, `run_mode = :pose`.
`adapt=true` marks a `VelRef` builder lifted to `PosRef` via `Profiles.velref_to_posref`.

**SCREEN tier — 4 trajectories (~50 s/eval).** Chosen to span the four directions that
identify the parameters: yaw, lateral/pose, bandwidth, transient.

| # | Profile | combo | adapt | role | Identifies |
|---|---|---|---|---|---|
| 1 | `spin_creep_mu_0p5.toml` | 178 | true | stress_yaw | yaw axis (most discriminating trajectory in v1) |
| 2 | `ellipse_mu_0p5.toml` | 83 | false | stress_crab | lateral / crab pose |
| 3 | `multisine_75percent_cap_mu_0p5.toml` | **55** | true | broadband | bandwidth-limited gains |
| 4 | `docking_step_mu_0p5.toml` | 1 | false | step_hold | transient recovery + steady-state |

**FULL-TRAIN tier — 12 trajectories (~3 min/eval).** Superset of SCREEN.

| # | Profile | combo | adapt | role |
|---|---|---|---|---|
| 1 | `octagon_mu_0p5.toml` | 1 | true | easy |
| 2 | `octagon_mu_0p5.toml` | 206 | true | stress |
| 3 | `spin_creep_mu_0p5.toml` | 255 | true | easy |
| 4 | `spin_creep_mu_0p5.toml` | 178 | true | stress_yaw |
| 5 | `coupled_vomega_mu_0p5.toml` | 114 | true | easy |
| 6 | `coupled_vomega_mu_0p5.toml` | 12 | true | stress |
| 7 | `spiral_orbit_mu_0p5.toml` | 37 | true | stress |
| 8 | `ellipse_mu_0p5.toml` | 55 | false | stress_tangent |
| 9 | `ellipse_mu_0p5.toml` | 83 | false | stress_crab |
| 10 | `multisine_75percent_cap_mu_0p5.toml` | **55** | true | broadband |
| 11 | `docking_mu_0p5.toml` | 1 | false | step_hold (DOCK-A) |
| 12 | `docking_step_mu_0p5.toml` | 1 | false | step_hold (DOCK-B) |

**TEST tier — 9 trajectories, HELD OUT, never seen by any optimizer.** Held out on three
independent axes so generalization claims are separable.

| # | Profile | combo | adapt | held-out axis |
|---|---|---|---|---|
| 1 | `long_circle_mu_0p5.toml` | 52 | true | profile (sustained yaw) |
| 2 | `long_circle_mu_0p5.toml` | 102 | true | profile (sustained yaw, stress) |
| 3 | `straightline_mu_0p5.toml` | 1 | true | profile (pure translation) |
| 4 | `multisine_75percent_cap_mu_0p5.toml` | 1 | true | combo (gentle end of the band) |
| 5 | `multisine_50percent_cap_mu_0p5.toml` | 55 | true | combo (lower amplitude, same band) |
| 6 | `spiral_orbit_mu_0p5.toml` | 27 | true | combo (easy) |
| 7 | `ellipse_mu_0p5.toml` | 1 | false | combo (easy tangent) |
| 8 | `ellipse_mu_0p5.toml` | 73 | false | combo (easy crab) |
| 9 | `coupled_vomega_mu_0p5.toml` | 12 | true | *anchor* — also in train, for train/test gap |

**Why these specific additions:**

- **`multisine_75` combo 55, not combo 1.** The profile has 55 combos with `f_hi` spanning
  1.0 → 3.5 Hz and `Vpk` 0.18 → 0.29 m/s. The v1 eval grid used combo 1 — the *gentlest of
  all 55*. Broadband content is the only excitation that makes bandwidth-limited gains
  (PID `Kd`, MPC horizon, ASMC `eps`) identifiable, and it was previously both eval-only
  and set to its least informative point.
- **`docking` (both variants).** Every v1 training trajectory is a continuously-moving
  reference. Adaptation rate (`gamma`) is observable only during transient recovery, and
  integral action (`Ki`, `I_max`) only during a sustained-offset hold. `build_docking`
  already exists and is registered in `Profiles.BUILDERS`; it is a native `PosRef` with a
  quintic approach followed by a hold. DOCK-B is a new TOML with a larger offset and a
  non-zero `psi_target_deg` so the heading axis also sees a step.
- **Easy anchors now in training.** With 12 trajectories the tolerance-normalized objective
  no longer needs the set to be all-stress to avoid being dominated by easy cases, and their
  presence prevents gains that are over-aggressive on gentle motion.
- **`straightline` and `docking` are permitted.** They are absent from the PINN dataset, but
  `run_controller` builds the reference from TOML and simulates live — it reads no Arrow
  file. The PINN whitelist constrains only the *reported comparison* set, not what may be
  tuned on.

**Implementation note (verified):** `Profiles.resolve_profile` ignores `combo_idx` entirely
when a profile has no `[profile.combos]` table. `docking` has only `[profile.params]`, so
passing `combo_idx=1` for docking entries is harmless and keeps the entry schema uniform.

### 7.2 Per-evaluation flow

1. `optimize_staged` proposes `theta::Vector{Float64}` (log-space for `:log` rows).
2. `decode(theta, space)` → gain NamedTuple; `freeze` values merged in for any parameter
   outside the active phase space.
3. For each trajectory in the active tier: `run_controller` builds the reference
   (`resolve_profile` with the pinned `combo_idx` and `Xoshiro(0)`; `velref_to_posref` when
   `adapt`), publishes it, and runs `SchedulerMod.run_hybrid` on an `OracleEstimator`.
4. `controller_metrics(probe, ref, :pose)` → tolerance-normalized `tracking`, plus `ce` and
   `chatter`. For `role == :step_hold` entries, `recovery_time` adds the transient term.
5. Scalarization (unchanged form): `score = mean(tracking) + LAMBDA_CE·(ce/V_MAX) +
   lambda_chatter·(chatter/V_MAX) + recovery_weight·recovery + kmax_pen + gamma_pen`.
6. Any non-finite or failed trajectory short-circuits the whole eval to the `1e6` sentinel
   (existing behaviour — preserve it; it keeps the population/trust-region safe).

### 7.3 Two-phase optimization flow

- **Phase 1 (global, SCREEN tier).** dxNES from a Sobol-offset start. Runs until
  `plateau_reached(rel_tol, window)` OR `p1_cap`. Note dxNES overshoots the cap at generation
  boundaries (observed 50 → 61, 300 → 311) — record actual evals, do not assume the cap.
- **Phase 2 (refine, FULL tier).** Box of `refine_halfwidth·(hi−lo)` centred on the phase-1
  optimum, clipped to `[lo, hi]`. BOBYQA if `NLopt` is present, else the existing local
  refine. Same plateau gate, cap `p2_cap`.
- **Reported `best_score` is always a FULL-tier score.** If phase 2 never improves on the
  phase-1 point re-evaluated at full tier, return that re-evaluated point and set
  `stop_reason = :no_refine_gain`.

### 7.4 Cross-seed flow

5 seeds run as 5 independent processes, `--seed N` and `start_offset = N`. `analyze_seeds.jl`
then computes: per-seed converged flag, score median/min/cv, per-parameter cv, and the
finite-difference Hessian of the score at the best seed's optimum (`d(d+1)/2` extra evals) —
its eigenvalues quantify the sloppy directions. This is what turns "the seeds scattered" into
a defensible identifiability statement.

---

## 8. Implementation Sequence

Strictly increasing complexity. Each stage is independently shippable and each is gated on
its own success criteria before the next begins.

### Stage 0 — Pipeline only (no controller math touched)
1. **`trajsets.jl`** — leaf module, no dependencies. Build first.
2. **New TOML `docking_step_mu_0p5.toml`** — copy `docking_mu_0p5.toml`, larger start offset,
   non-zero `psi_target_deg`. Verify it builds via `Profiles.build("docking", cfg)`.
3. **`tune_controller.jl` additive edits** — export the reused functions; add the `freeze`
   keyword to `build_controller` (default empty ⇒ current behaviour byte-identical).
4. **`stage_objective.jl`** — depends on #1 and #3.
5. **`optimizer_stage.jl`** — depends on #4 for its objective contract only.
6. **`run_stage.jl`** — depends on #1–#5.
7. **Validation:** `--stage 0 --smoke`, then re-run the legacy path and confirm
   `--trajset 3` and `--trajset pose` reproduce their prior `best_config.json` bit-identically.

### Stage 1 — ASMC (lowest complexity: existing 8-parameter space, run properly)
8. Launch 5 seeds on the new tiers with the **unchanged** `ASMC_SPACE` + `--pin-kmax`.
   This isolates the pipeline's effect from any controller change and yields a directly
   comparable number.
9. **`analyze_seeds.jl`** — cross-seed report + held-out TEST evaluation.
10. *Optional Stage 1b, only if Stage 1 converges cleanly:* extend the ASMC space with the
    sliding-surface slopes `lam_x_max`, `lam_y_max`, `lam_psi_max`, `mu_xy`, `mu_psi`.
    These set the error-decay rate on the sliding surface and are currently frozen at
    struct defaults — they are the steep directions the search has never seen. Keep the
    pinned `K_max`.

### Stage 2 — PID (moderate: cascade decomposition + `I_max`)
11. **`pid_cascade.jl`** — the three phase spaces.
12. Sequential driver: `:inner` (velocity loop, outer frozen at feedforward-only) →
    `:outer` (inner frozen) → `:joint` (narrow box polish). This replaces one 15-D blob with
    12-D → 6-D → 18-D-local, exploiting the near-diagonal body-frame plant and the
    time-scale separation the cascade already assumes.
13. Add `I_max` (3 params) to the `:inner` space — currently frozen at (50, 50, 30) despite
    hard `V_max`/`i_max` saturation making it a first-order performance parameter.
14. 5 seeds + `analyze_seeds.jl`.

### Stage 3 — MPC (highest complexity: structural repair before any search)
15. **`mpc_design.jl`** — `normalized_mpc_space` (pin `R_scale = 1`, drop velocity-mode `Q`),
    `bryson_weights`, `terminal_weight`.
16. **`controllers.jl`** — add the `P_terminal` field to `MPCController` and apply it to the
    final horizon step in the cost assembly. Default zero ⇒ existing behaviour unchanged.
17. Add `Np_pose` to the search as an outer grid `{10, 15, 20, 30}` (integer — grid it, do
    not round a continuous variable). It is currently fixed at 15 and is the dominant
    pose-tracking knob.
18. Run the reduced search: with Bryson weights fixed by design and `R_scale` pinned, what
    remains is a small ratio set plus the `Np` grid — a structured sweep, not a black-box
    search. 5 seeds only for the stochastic remainder.
19. `analyze_seeds.jl`.

### Stage 4 — Reporting
20. Cross-controller comparison on the held-out TEST tier; identifiability tables;
    train/test gap on the `coupled_vomega c12` anchor.

---

## 9. Numerical, Determinism & Reproducibility Considerations

*(Domain-adapted from the template's ML section — this is a Julia ODE/optimization pipeline,
not a PyTorch model.)*

- **BLAS threading:** `LinearAlgebra.BLAS.set_num_threads(1)` MUST be set before any solve.
  Removing it reintroduces run-to-run FP variation that the closed loop amplifies into
  divergent-vs-stable outcomes. Non-negotiable.
- **Global mutable state — parallelism hazard.** `Profiles.ACTIVE_KIND` is a global `Ref`
  and `SchedulerMod.ESTIMATOR_PROBE_LOG` is an unlocked global `Dict` keyed by
  `objectid(bus)`. **Do not** enable BlackBoxOptim `NThreads` or thread the trajectory loop.
  Seed parallelism is process-level only. Also call `clear_probe_log!(bus)` after each
  trajectory — the dict grows unboundedly across a multi-hundred-eval run otherwise.
- **RNG discipline:** reference construction must keep using `Xoshiro(0)` for pinned
  `combo_idx` and `Xoshiro(hash(profile_toml))` for the unpinned path. Drawing from the
  global RNG makes the closed loop non-deterministic across evals — a known past failure.
  The optimizer's RNG is seeded separately per seed and must not touch the reference path.
- **Sentinel handling:** keep the `1e6` clamp for non-finite scores. Trust-region refiners
  (BOBYQA) are more sensitive to sentinel cliffs than population methods — if phase 2 stalls
  immediately, check whether the refine box straddles a sentinel region and shrink
  `refine_halfwidth`.
- **Log-space parameters:** `flat_bounds` returns log-space bounds for `:log` rows. Sobol
  starts, refine boxes, and any Hessian finite differences must all be computed in the SAME
  space as the optimizer operates (log), not in physical units.
- **Trace flushing:** keep the explicit `flush` after every `trace.csv` write. Julia
  block-buffers stdout when redirected to a file, so the flushed trace is the only reliable
  live progress signal on a multi-hour run.
- **Checkpointing / resumability:** a 5-hour × 5-seed run must survive interruption. Write
  `trace.csv` and `phase_summary.json` incrementally; on restart with an existing output dir,
  either resume phase 2 from the recorded phase-1 optimum or fail loudly — never silently
  restart phase 1 and overwrite.
- **Output directory safety:** the v1 tuner overwrote `runs_controller/<ctrl>_<oracle>/` with
  no run id and lost configs once. Every stage writes to a fresh root; `run_stage.jl` must
  refuse to write into a non-empty existing seed directory unless `--force` is passed.
- **Modern Standby:** these runs exceed 30 minutes. `keep_awake.py` must be running or the
  sweep dies mid-flight even on AC power.

**Wall-clock budget (estimate, for planning):** SCREEN ≈ 50 s/eval, FULL ≈ 3 min/eval.
Phase 1 cap 250 evals ≈ 3.5 h; phase 2 cap 60 evals ≈ 3 h. ≈ 6.5 h per seed, 5 seeds in
parallel ⇒ **≈ 6.5 h wall clock per controller** — one overnight run per stage.

---

## 10. Success Criteria

**Stage 0 (pipeline)**
- [ ] `--trajset 3` and `--trajset pose` on the legacy `tune_controller.jl` path reproduce
      their prior `best_config.json` **bit-identically**
- [ ] `--stage 0 --smoke` completes one eval on each tier with no failed trajectories
- [ ] `docking_step_mu_0p5.toml` builds a valid `PosRef` and runs closed-loop to completion
- [ ] `trace.csv` gains a `tier` column and stays live-flushed

**Stage 1 (ASMC)**
- [ ] `converged == true` for **≥ 4 of 5 seeds** (`stop_reason == :plateau`, not `:cap`)
- [ ] Inter-seed score cv **< 10 %** (v1 pose-mode baseline: ~45 %)
- [ ] Best v2 gains score better than the current `asmc_pose_5seed` winner **when both are
      evaluated on the same 12-trajectory FULL tier** (apples-to-apples, not cross-set)
- [ ] Held-out TEST score reported for all 9 entries; no entry diverges

**Stage 2 (PID)**
- [ ] All three cascade phases complete; `:outer` phase changes only pose-mode metrics
- [ ] Inter-seed cv on the *inner-loop* gains materially below the v1 15-D figure
      (v1: up to 154 % on `Ki`)
- [ ] `I_max` inclusion is shown to matter or shown not to — either result is reportable

**Stage 3 (MPC)**
- [ ] **Unit test:** scaling `(Q_pose, R, S)` by any `c > 0` leaves the applied first-step
      voltage `u0` unchanged to solver tolerance — confirms the flat direction, and confirms
      pinning `R_scale` loses nothing
- [ ] **Unit test:** with all trajectories in `:pose` mode, mutating `mpc.Q` changes no
      output — confirms those 3 parameters are dead and correctly excluded
- [ ] Bryson + Riccati seeding alone (zero search) beats the current MPC score of ~67
- [ ] `Np_pose` grid shows reduced sensitivity once the terminal cost is active

**Stage 4**
- [ ] Cross-seed identifiability table with Hessian eigenvalues per controller
- [ ] Train/test gap quantified on the `coupled_vomega c12` anchor

---

## 11. Out of Scope

- **Friction diversity.** μ=0.3 / μ=0.8 profiles, μ-stratified training, warm-start transfer
  across μ, and friction-robustness claims are all **deferred to the next iteration** at the
  user's explicit direction. Every tier in this brief is μ=0.5.
- **More than 5 seeds.** Fixed at 5 by user direction; generalization budget goes into
  trajectory-set size instead.
- **Noise/robustness evaluation.** `experiment_noise_eval_pose.jl` stays as-is and runs after
  a stage completes; this brief covers clean-oracle tuning only.
- **Frozen-ESKF comparison.** `compare_controllers_eskf.jl` is downstream of this work and
  unchanged here.
- **Physics, plant, sensors, estimators.** No file outside `tune_controller.jl`,
  `controllers.jl` (Stage 3 only), and the new `controller_tuning/` modules is modified.
- **Relay auto-tuning, Pareto/NSGA-II fronts, Bayesian surrogate backends.** Discussed as
  options but excluded from this iteration to keep the complexity ladder clean.
- **PINN / observer work.** Entirely unrelated.

### PRESERVATION CONSTRAINT (hard)

All v1 results must remain reproducible. Specifically:

- `default_trajs`, `default_trajs_2`, `default_trajs_3`, `default_trajs_pose` — **unchanged**
- `TOL`, `LAMBDA_CE`, `V_MAX`, `ASMC_SPACE`, `ASMC_SPACE_PIN`, `PID_SPACE`, `K_MAX_PIN` —
  **unchanged**
- `--trajset 1|2|3|pose` behaviour — **unchanged**
- `build_controller` with no `freeze` argument — **byte-identical output**
- `runs_controller_v1_velocity_ARCHIVE/` and all existing `runs_controller*` directories —
  **never written to**

New behaviour is additive and reachable only through `run_stage.jl` and the new flags.
