# Handoff — Approach 2: SSM/Mamba neural observer for per-wheel unobservable contact states

## 1. Title + lineage
Build a **supervised neural state observer** (Mamba-like selective SSM) that reconstructs the
**unobservable per-wheel contact states `γ_i` (roller rate), `ω_z,i` (contact spin), `z_i` (LuGre
bristle deflection)** from **measurable signals only** — no forces, no μ/χ. Continues the
*Forward-Inverse PINN architecture* session (that session keeps **Approach 1** = the force-
reconstruction PINN; this is the parallel **Approach 2**). Project: PINN digital twin of a KUKA
youBot 4-Mecanum-wheel platform, IMECE 2026.

## 2. Context the task depends on (exact)
- **Plant**: 39-D ODE sim, composite LuGre+Adamov friction, 12 rollers/wheel, ASMC+DOB controller.
  State layout (1-indexed): `[1:3]` Vx,Vy,ψ̇ · `[4]` ψ · `[5:8]` θ₁..₄ · `[9:12]` ω₁..₄ ·
  **`[13:16]` γ₁..₄ (roller rate)** · `[17:19]` gains · `[20:21]` x,y · **`[22:25]` zx, `[26:29]` zy
  (linear bristle)** · **`[30:33]` zs (rotational bristle)** · `[34:36]` observer · `[37:39]` δ̂.
- **Data (same as Approach 1 — no new generation needed, unblocked now)**:
  `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov` — 1776 runs, **μ=0.5**, χ∈{0,0.002,**0.005**,
  0.008}. **Whitelist 1568** via `diagnostics_combined.csv` (`combined_reco` not starting `reject`).
  Each run = an **Arrow file** (measurables + forces + references) **+ a JLD2 sidecar** carrying
  `sol_t/sol_u` (the full 39-D state). **The JLD2 `sol_u` is the hidden-state supervision source.**
  Sim grid is 2000 Hz; **train at 500 Hz** (Fpar/Fperp recon error ~1–3% at 500 Hz — see
  `TRAJ_DIAGNOSTICRESULTS.md` §3).
- **Inputs (HARD RULE — sensor-measurable only)**: `Msat_1..4` (control torques, in Arrow),
  `Vx,Vy,ψ̇`, `ω_1..4`, wrapped `θ̃` fed as `sin(12θ),cos(12θ)` (folded to ±π/12), plus derived
  kinematic features. **Never** inputs: forces, μ, χ, or any hidden state.
- **Targets (from JLD2 `sol_u`)**: `γ_i [13:16]`, `zx_i [22:25]`, `zy_i [26:29]`, `zs_i [30:33]`;
  **`ω_z,i` is DERIVED** (compute per the contact-spin definition in `chi_identifiability.py`,
  which already bins by `|ω_z|`). → 5 targets/wheel × 4 wheels.
- **Key files**: `chi_identifiability.py` (ω_z formula + spin binning), `CLAUDE.md` (§3 plant, §5
  data contract), `base.toml` (constants: m=30, J_roller=1e-6, J_wheel=5.87e-3, R=0.05, rollers=12),
  `TRAJ_DIAGNOSTICRESULTS.md` (whitelist, 500 Hz, spin-gating). Existing PINN code for reference:
  `train_GPU_PINN_v14_py/` (data loaders, JVP-safe hand-unrolled GRU).

## 3. Purpose / success criterion
A supervised observer `measurable_history → {γ_i, ω_z,i, z_i}` per wheel, trained in sim (labels
from JLD2), **run from measurables only at inference** (virtual sensing). Primary scientific output:
**a per-state observability ranking** — the per-state reconstruction error *is* the observability
metric (low = observable; high irreducible error = non-unique/unobservable). Secondary: a reusable
state estimator that could feed Approach 1.

## 4. Key design decisions (already made — defend, don't reopen)
1. **Supervised observer using JLD2 hidden-state labels** (sim-trained, measurable-only inference).
   Rationale: directly quantifies what's recoverable; deployable since labels are only needed at
   training. This is the *distinct* value of Approach 2 vs Approach 1 (which never anchors to hidden
   states — it uses them only as a probe).
2. **Encoder = small "Mamba-lite" selective SSM** (diagonal selective scan, state-dim ~4–8, shared
   across wheels via a wheel embedding), **implemented as a plain-PyTorch unrolled scan — NOT the
   official Mamba CUDA kernel.** Rationale: (a) selectivity = input-dependent relaxation rate, which
   mirrors the LuGre bristle decay `σ₀|v|/g(v)` and lets the model infer the slip-dependent timescale
   from observables; (b) at short sequences (~5–15 samples) / tiny state the CUDA kernel gives no
   speedup and breaks forward-mode/double-backward AD, won't build cleanly on Windows, and is
   non-deterministic; plain PyTorch is full-AD, deterministic, portable. **GRU = fallback baseline**.
3. **Measurable-only inputs** (shared hard rule with Approach 1).
4. **Expected observability (test, don't assume)**: `z` is hardest — published PINN/SMC work shows
   the LuGre bristle state is **non-unique from input–output** (multiple z give identical I/O), so
   expect an irreducible-error floor / equivalence-class recovery. `γ` is near-algebraic
   (J_roller=1e-6 → fast → strongly observable). `ω_z` is mostly measurable-derived (ψ̇+geometry) +
   a small hidden correction → expected recoverable. **`ω_z` reconstruction quality bounds
   Approach 1's χ-channel** (χ enters only via `c_t=(8/3π)|ω_z|χ`).
5. **Train at 500 Hz; whitelist via `diagnostics_combined.csv`.**
6. **Evaluate** by per-state normalized RMSE (per wheel, and binned by slip / `|ω_z|` regime),
   SSM vs GRU. Report the irreducible floor as the observability signature.

## 5. Open decisions / blocking relationships
- Whether to add a self-supervised term (predict future measurables) to regularize the latent —
  intentionally open; start purely supervised.
- Confirm the exact **`ω_z,i` definition** and the **`γ` vs `γ̇`** target from `chi_identifiability.py`
  + the notebook before training.
- **Hand-back to Approach 1 (parent session)**: the observability ranking (which states are
  reliable) and, optionally, a trained observer checkpoint. ω_z fidelity directly informs whether
  χ-identification is viable; z's irreducible error confirms/quantifies the non-uniqueness caveat.

## 6. Deliverables
1. A **CLI training script** (streaming, resume-aware — per project convention) for the SSM observer
   + a GRU baseline, reading Arrow (inputs) + JLD2 (targets), 500 Hz, whitelist-gated.
2. **Per-state reconstruction report** (`state_observability.csv` + static matplotlib figures):
   normalized RMSE per hidden state per wheel, SSM vs GRU, binned by slip/`|ω_z|`.
3. A short findings note: **observability ranking** of {γ, ω_z, z}, with the z non-uniqueness
   result quantified.
4. (Optional) trained observer checkpoint for the Approach-1 hand-back.

## 7. Conventions to respect
- **Measurable-only inputs** is a hard rule; forces/μ/χ/hidden-states are never inputs.
- **PyTorch side runs on the separate WSL2 machine** (`~/mecanum_pinn_main/`, data under
  `~/mecanum_pinn_main/data/...`), two GPUs: 24 GB Quadro RTX 6000 (Turing, **no native bf16**) and
  6 GB RTX 3060 (Ampere, bf16). Gotchas: import `pyarrow.feather` first; `torch.load(...,
  weights_only=False)`; `from torch import _dynamo`.
- CLI-script over notebook for batch work (streaming + resume); per-file streaming
  (read→accumulate→delete→advance) for memory safety over the Arrow/JLD2 set.
- Scratch/tests → `_tmp/`, cleaned up after. Figures: static matplotlib + tables (no interactive
  widgets). Seeded/deterministic; surgical edits; verify numbers in code before asserting them.

## 8. Relevant prior art (precedents — your building blocks; the 4-wheel coupled version is novel)
- PINN estimates LuGre bristle `z` from velocity/position/torque without force, with transfer:
  arxiv **2504.12441**. (Reports the **z non-uniqueness from I/O** finding — §4.4 above.)
- Probabilistic SSM + Sequential Monte Carlo reconstructs LuGre `z` from measurables, no force:
  arxiv **2412.15756**.
- Mamba-as-observer (learns latent SSM dynamics, Kalman filter/smoother via parallel scan):
  **KalMamba**, arxiv 2406.15131. Classical LuGre observer: arxiv 2501.04793.
