# Handoff — A1 INVERSE model v2: architecture brainstorm (design-only)

## 1. Title + lineage
Continues the 2026-07-09 design session that produced the **forward-v2 stick/slip
redesign** and the **Observer-v2 γ-only rewire**. Project: Mecanum PINN digital twin
(IMECE 2026); Julia 39-D ODE → Arrow (500 Hz decim) → PyTorch. This new chat owns ONE
thing: **the A1 inverse-model v2 architecture and the (μ̂, χ̂) readout**. Forward v2 and
Observer v2 are pinned and proceed in parallel — do not reopen them.

## 2. Context the task depends on (exact)
**Envs:** torch = `C:\Users\vishv\miniforge3\envs\myenv\python.exe` (py 3.13, torch
2.6.0+cu124); torch-free = `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`.
No `conda activate` in tool shells. Data: `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/`,
500 Hz (DECIM=4), frozen p95 max-norm scaler, S1/S2 regimes (`observer_v1_py/regimes/`).
**v1 inverse (read first):** `Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/models.py`
`MecanumInverseModel` — per-wheel MLP over a causal Δ-state window, `inv_window=3`
(6 ms), `inv_hidden=32`, output soft-bounded ‖F_i‖ ≤ N_i, roller frame [Fpar_1..4,
Fperp_1..4]; `mu_readout_residual` = test-time LS projection of F onto the FORWARD
model's shape basis (μ_conf = slip energy); χ-ID deferred in v1.
**v1 results** (`checkpoints_mamba_v1/a1_*/cross_metrics.json`, 6 runs, F_MAX=87.309 N):
inv grnd MSE 0.0028–0.0038 (≈4.6–5.4 N RMSE — GOOD); fwd grnd 0.039–0.046 (≈17–19 N —
the failure); mu_mae_inv 0.25–0.41 on μ-span 0.5 (predict-mean floor ≈0.17). Root cause:
μ-readout projects onto the drifted FORWARD basis — F_inv itself was never the problem.
**Sibling designs (pinned, read before proposing):**
- `Mecanum_PINN_Mamba_ForceRecon_v1/FORWARD_V2_STICKSLIP_DESIGN.md` (authority) +
  `instructions/mecanum-forcerecon-forward-v2-stickslip.md`
- `instructions/observer-gamma-only-5phase-retrain.md` (Observer v2: γ-only, roller
  balance trained, 200-epoch 5-phase 80/24/40/24/32, plateau LR patience 10 + ramp freeze)
**Wrench facts (forward v2, reuse):** 6 measurable combos = 4 wheel torque balances
`(Msat_i − Jw·ẇ_i − p1·w_i)/R` + body-y + yaw from IMU (body-x redundant); 2-D null
space = pair-antisymmetric lateral forces, roller-frame free diagonals (sin δ_i, cos δ_i),
δ = (−45°,+45°,+45°,−45°). Regime map: τ = g/(σ0|v_s|), σ0=1640, v_str=0.01 m/s; slip
quasi-static for |v_s| ≳ 0.15 m/s; stick = integrator, observable only through wrench.
χ physics: c_t = (8/3π)·|ω_z|·χ, ≈2.3 N at high spin only (memory: spin-gated).

## 3. Purpose
Brainstorm and pin the inverse-v2 design: a μ-agnostic force/parameter-identification
path coherent with forward v2. Success = user-confirmed `INVERSE_V2_DESIGN.md` +
(after pinning) an /instructor brief. Design-only session — no implementation.

## 4. Key design decisions (already made — defend, don't reopen)
1. **Forward ⊥ inverse:** trained in parallel, zero coupling; `w_cons=0` stays
   monitor-only (training it clones inverse onto forward, killing μ-agnostic reading
   and F_fwd↔F_inv change detection).
2. **Forward v2 output = 6 measured combos + 2 null scalars** (IMU-fused). Implication
   the brainstorm must start from: the v1 inverse's job (reading forces from Δ-states ≈
   accelerations) is largely obsoleted — 6/8 force dims are now measured. Inverse v2's
   real job is what's left: μ-agnostic structure, null coords, and the (μ̂, χ̂) readout.
3. **μ-ID is a test-time readout, never a training objective** (unchanged rule).
4. **Measurable-only inputs.** Observer-v2 γ̂ is an allowed derived-measurable input;
   ẑx/ẑy are forbidden (LuGre-specific, no consumer, sim-prior contamination).
5. **No LuGre structure in deployed modules** (sim-to-real generalizability);
   v_str = 0.01 m/s appears only as a fixed order-of-magnitude gate scale.
6. **No physics-only training tails** — supervised floor 0.1 everywhere (v1 forward
   null-drift lesson: phys 5e-5 while grnd 0.04).
7. Slip-branch cap 1.0·μ_c·N, stick cap 1.1 (Stribeck decay lives in the gate blend) —
   applies if the inverse adopts the two-branch structure.
8. **v2 = parallel files, v1 byte-identical** (both packages follow this pattern).

## 5. Open decisions (this session's actual scope)
- **Architecture form:** separate inverse network (v1-style, upgraded) vs a thin
  μ-agnostic readout head on the measured wrench + forward-v2 trunk features vs a
  hybrid. This is the central question.
- **Memory:** keep short Δ-window (6 ms) or adopt burn-in/carried state — does μ/χ
  information in stick/transition regimes need long memory even though instantaneous
  force recon doesn't?
- **(μ̂, χ̂) readout v2:** what basis replaces the forward shape basis (which caused the
  v1 failure)? χ̂ design from the spin-gated channel (needs high-|ω_z| conditioning/
  confidence gating). How μ_conf/χ_conf gate deployment claims.
- **Change-detection contract:** how F_fwd↔F_inv divergence (the μ/χ-change signal)
  is defined once forward v2 fuses measurements.
- **Blocking:** nothing blocks on this session (forward v2 + Observer v2 proceed);
  hand-back = the two deliverables below feed the A1 implementation thread.

## 6. Deliverables
1. `Mecanum_PINN_Mamba_ForceRecon_v1/INVERSE_V2_DESIGN.md` — pinned-decision design doc
   (mirror FORWARD_V2_STICKSLIP_DESIGN.md's structure: evidence → pinned decisions →
   losses → hooks → acceptance).
2. `instructions/mecanum-forcerecon-inverse-v2-brief.md` — /instructor brief, no
   figures, only AFTER the user pins the design.

## 7. Conventions to respect
- **Decisions before code** — propose, get the user's pin, only then brief. Surgical
  edits over rewrites. Never invent physics constants (base.toml/run_one.jl authority;
  code-verify).
- Chat math: Unicode + code blocks, never `$…$` LaTeX. Static matplotlib only. No
  interactive widgets. Scratch in `code_insights/_tmp/`, cleaned after.
- ≤8 workers; laptop commit-limit caps parallel training at 2; long runs need
  `keep_awake.py`; run everything from `code_insights/`.
- Instructor briefs: exact section template, `...`-body signatures only, no figures.
- Memory files already hold the settled context: `project_a2_gamma_only_redesign.md`,
  `project_pinn_consistency_monitor_only.md`, `project_pinn_measurable_inputs_only.md`.
