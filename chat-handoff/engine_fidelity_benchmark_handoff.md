# Handoff — Physics-engine (PyBullet / ROS2) fidelity benchmark of the Mecanum plant

## 1. Title + lineage
New session: build an **independent, higher-fidelity physics-engine simulation**
(PyBullet+PyTorch and/or ROS2/Gazebo) of the same KUKA youBot 4-Mecanum platform
and the same trajectory families, with the **non-linear roller–ground contact**
modeled explicitly. Continues the **Mecanum PINN digital-twin** project (IMECE
2026), whose data today comes from a Julia 39-D stiff-ODE simulator. This task
exists to **cross-validate that ODE dataset against an engine that resolves the
roller contact geometrically**, and to characterize the fidelity gap. It does NOT
touch the PINN training threads (A1 force-recon, A2 observer) — it produces a
benchmark dataset + comparison.

## 2. Context the task depends on (exact)
**Plant** (Adamov & Saypulaev 2021, `nd711.pdf`; constants are authoritative in
each `trajectory_files_run_*/base.toml`, mirrored in `observer_v1_py/mecanum_observer/config.py`):
- 4 Mecanum wheels, **12 rollers/wheel** (paper uses 6), roller axis δ = 45°,
  **O-configuration** δ = [−π/4, +π/4, +π/4, −π/4].
- Wheel radius **R = 0.05 m**, roller axle distance **Rd = 0.0355 m**.
- Wheel centres: px = [+0.235, +0.235, −0.235, −0.235] m (half-length H=0.235),
  py = [+0.15, −0.15, +0.15, −0.15] m (half-width L=0.15).
- Masses/inertia: platform **30 kg**, yaw inertia **Is = 4.42 kg·m²**, wheel
  **1.4 kg**, **J_wheel = 5.87e-3**, **J_roller = 1e-6 kg·m²**. COM offset
  AX=1.6e-2, AY=−2.6e-2 m. Per-roller normal load N = [79.57, 105.07, 69.55,
  95.05] N; **F_MAX = 87.309 N**.
- Drivetrain viscous **p1 = 0.11**, roller-bearing viscous **p2 = 5.78e-3 N·m·s**
  (friction_case 1; case 2 = low-viscous ablation). Wheel actuator cap ±10 N·m.
- **Friction = composite LuGre + Adamov, slip–spin coupled** (`:lugre_adamov`,
  use_mindlin=true). Bristle: σ0=1.64e3, σ1=1.6, σ2=0; spin σ0_s=1.09e3,
  σ1_s=1.1, σ2_s=0; stiction ratio 1.1; v_str=w_str=0.01; eps_reg=1e-4. Spin
  enters via **c_t = (8/3π)·|ω_z|·χ**. Folded contact angle uses sawtooth_tanh
  (TANH_K=60). A **verified torch port** of this law lives in
  `observer_v1_py/mecanum_observer/physics.py` (roller + wheel residuals).
- Control: **ASMC + super-twisting DOB** (`asmc_torques_vel` for VelRef,
  `asmc_torques` for PosRef; DOB yaw-only: ω_o_ψ=6π, k1_ψ=15, k2_ψ=80).
- **39-D state order** (CLAUDE.md §3): [1:3] Vx,Vy,ψ̇ · [4] ψ · [5:8] θ1..4 ·
  [9:12] ω1..4 · [13:16] γ1..4 (roller rates) · [17:19] adaptive gains ·
  [20:21] x_o,y_o · [22:29] zx,zy bristles · [30:33] zs spin bristles ·
  [34:39] observer + disturbance est.
- Sweep grid: μ ∈ {0.3, 0.5, 0.8}, χ ∈ {0, 0.002, 0.005, 0.008}.

**Trajectory families** (`profiles.jl` builders + per-profile TOMLs in
`trajectory_files_run_0p5_main/profiles/`): octagon, long_circle, spin_creep,
coupled_vomega, spiral_orbit, multisine (50%- & 75%-cap), ellipse. VelRef = body-
frame (Vx,Vy,ψ + ForwardDiff feedforwards); ellipse is the lone **PosRef** (world
position). Multisine phases are explicit in the TOMLs.

**Reference dataset to benchmark against:** `data/Simulation_Data_MecanumSlipSpin_LugreAdamov/`
— **5949 Arrow files, 238 GB**, each a **22 s trajectory at 2000 Hz (44,022 rows)**.
Column/filename contract in CLAUDE.md §5 (`Vx,Vy,psi_dot, w1..4, theta1..4,
Msat_1..4, Fpar/Fperp_1..4 (& Fx/Fy/Mz), gamma1..4, zx_/zy_1..4, wz_1..4, time`).
Solver was TRBDF2, reltol 1e-8, dtmax 1e-3, saveat 2000 Hz. `variable_scaler_percentiles.csv`
(in the data dir) gives p50/p95/max of every channel — use it to sanity-check ranges.

## 3. Purpose
Stand up an engine sim that (a) reproduces the platform + 12-roller wheels with
**geometric roller-ground contact**, (b) runs the same trajectory families at
matched (μ, χ), and (c) emits trajectories in (or convertible to) the Arrow
contract so the existing loaders/diagnostics ingest them. **Success = matched-input
runs whose body (Vx,Vy,ψ̇) and per-wheel contact forces are quantitatively compared
to the Julia data per profile/μ/χ, with the fidelity gap reported.**

## 4. Key design decisions (already made — defend)
1. **Replicate the Julia plant exactly** (geometry/mass/inertia/loads from
   base.toml above). A benchmark is only valid if the plant matches.
2. **Rollers are explicit bodies, not a friction-anisotropy hack** — 1 chassis +
   4 driven wheels + 48 passive free-spinning rollers (revolute about the 45°
   axis). The geometric contact is the higher-fidelity contribution vs the Julia
   analytical roller transform.
3. **Engine-native Coulomb friction ≠ our LuGre+Adamov.** Either inject the LuGre
   force as an external contact wrench (full fidelity, harder) OR run native
   friction and **report the Coulomb-vs-LuGre gap as a measured result** — that
   gap IS part of the benchmark. Do not silently substitute Coulomb and call it equal.
4. **Drive with matched excitation:** preferred = re-create the VelRef/PosRef +
   ASMC controller; fallback = open-loop **replay of recorded `Msat_1..4` torques**
   from a chosen Arrow file. Matched input is mandatory for comparison.
5. **Compare on the PINN's labels** — body velocities + per-wheel forces at matched
   μ/χ — not on hidden bristle states (engine has no bristle analog).

## 5. Open decisions / blocking relationships
- **Engine: PyBullet vs ROS2/Gazebo (Gz/DART)** — undecided. PyBullet is lighter
  and scriptable for a research benchmark; ROS2/Gazebo if a full robotics stack /
  hardware path is wanted. Pick one and justify.
- **Friction realization** (native vs external LuGre) — see decision 3; pick after
  a feasibility spike on contact-force access in the chosen engine.
- **Drive mode** (controller re-impl vs torque-replay) — start with torque-replay
  (cheapest matched input), add the controller if closed-loop behavior is needed.
- **Hand-back to the parent thread:** a fidelity-comparison report (engine vs Julia
  per profile/μ/χ: RMSE of Vx,Vy,ψ̇ and forces, + qualitative slip/spin behavior).
  This informs whether the Julia ODE dataset is high-enough fidelity for the PINN
  and the IMECE paper — that conclusion crosses back here.

## 6. Deliverables
1. **Platform model** — URDF/SDF (or PyBullet programmatic) of chassis + 4 wheels +
   48 rollers, with the exact geometry/mass/inertia from §2.
2. **Sim driver** — runs a given profile at (μ, χ); torque-replay first. Scratch in
   `_tmp/`; package under a new dir, e.g. `engine_sim_v1/`.
3. **Arrow exporter** — emits the §2 column contract (at least Vx,Vy,psi_dot,
   w1..4, theta1..4, Msat_1..4, per-wheel forces, time) so existing tooling loads it.
4. **Comparison report** — `engine_vs_julia_fidelity.{csv,md}` + static matplotlib
   figures: per (profile, μ, χ) RMSE of body states & forces, engine vs Julia.

## 7. Conventions to respect
- **base.toml is the physics authority** — never invent a constant; pull from it.
  Code-verify every numeric value before asserting (this project's rule).
- **Match the Arrow filename + column contract** (CLAUDE.md §5) so the data is
  drop-in for the loaders/diagnostics.
- Surgical edits, decisions before code; **scratch/tests in `code_insights/_tmp/`,
  deleted after**; static matplotlib only (no widgets without permission).
- **Chat math: Unicode + code blocks, never LaTeX `$...$`** (terminal has no MathJax).
- Handoffs live in `code_insights/chat-handoff/`. Reference `nd711.pdf`,
  `code_insights/CLAUDE.md`, `profiles.jl`, `run_one.jl` (NE RHS ≈ lines 716–722),
  `datastore.jl`, and `observer_v1_py/mecanum_observer/physics.py` (verified LuGre port).
