# Handoff: Hy3 observer — decimation effects & derived-slip reconstruction

**Continues:** the hy3 observer session (gate/slip debugging → root-cause bug found).
**Frame:** PINN digital twin (KUKA youBot, 4 Mecanum wheels). Observer v2-Hy3 predicts
per-wheel roller spin **γ** + body-velocity correction **ΔV** from sensor-real inputs;
a regime-split physics loss uses a **derived contact slip** `|Vp|` to gate stick vs slip.

## Context the new chat needs
- **Env (torch/CUDA):** `C:\Users\vishv\miniforge3\envs\myenv\python.exe`. Run from `code_insights/`.
- **Decimated cache (off-OneDrive):** `C:/Users/vishv/mecanum_cache_decim`. hy3 entries keyed
  `<arrow>.hy3in.xo1_rot_ab2_noise-none_hz500.npz` (full μ/χ grid warmed, ~5,345 files).
- **Scales (frozen p95, from `variable_scaler_percentiles.csv`):** γ=82.806, dv_scale=[1.9200, 0.6734],
  vpm=0.17538, a_x=2.312, a_y=2.386.
- **Key files:**
  - `observer_v1_py/mecanum_observer/sensor_frontend_v2.py` — front-end. `Vpx0_hat = V̂ − ψ̇·(PY+DY) − w·R`
    (**already contains V̂**). `_decimate` = causal 1-pole LPF @ **200 Hz** + downsample 4 (2000→500 Hz).
  - `observer_v1_py/mecanum_observer/losses_v2hy3.py` — `derived_contact_slip`, `physics_loss_hy3`,
    `supervised_loss`, `slip_consistency_loss`. **← the bug lives here.**
  - `observer_v1_py/mecanum_observer/physics_v2hy3.py` / `physics.py` — `contact_from_gamma`
    (`Vpx = Vpx0 + γ·dVpx_dg`; `dVpx_dg` etc. are functions of folded wheel angle θt, **range ±15°**).
  - `training_v2hy3.py`, `data_v2hy3.py`, `config_v2hy3.py`, `train_observer_v2hy3.py`, `launch_parallel_v2.py`.
- **Scratchpad probes (reusable):** `scratchpad/test_slip_identity.py` (identity via CORRECT base),
  `scratchpad/slip_attribution.py` (swap true↔model per head), `dv_collapse_check.py`, `calibrate_gate.py`.

## THE ROOT-CAUSE BUG (central finding — verify, then finish the fix)
`physics_loss_hy3` (line ~104-108) and `derived_contact_slip` (line ~68-70) compute the corrected
base as `Vpx0_u = Vpx0_hat + V_used` where `V_used = V̂ + ΔV̂`. But `Vpx0_hat` **already carries V̂**,
so this **double-counts V̂ (~0.5–1 m/s)**. Brief always said `Vpx0_u = Vpx0_hat + ΔV̂` (add only the
correction). **Present since run-1.** Proof: identity `(true γ, true ΔV)` → true vpm gives **RMSE 0.69 m/s**
with the buggy base vs **0.002 m/s** with the correct base (`Vpx0_hat + ΔV̂`).
**Fix = `Vpx0_hat + dv_phys` (drop V_used)** in BOTH functions. ✅ **Code is at consistent-original**
(bug present in both `derived_contact_slip` line 69 and `physics_loss_hy3` line 108 — a temporary half-fix
was reverted so all prior runs are reproducible). **Step 1 for the new chat: apply the fix to BOTH, then
verify.** `dv_phys`/`dVx` here is a **velocity** (ΔV correction), NOT the sidecar `dVx` (an acceleration) —
naming collision caused confusion; consider renaming the local var when fixing.

## Purpose of the new chat
Understand the **effect of decimating to 500 Hz** on the derived-slip reconstruction, and decide whether
the 500 Hz rate is a binding limiter. Concretely: (1) finish/verify the base-bug fix; (2) measure the
corrected identity floor **split by stick / slip / high-slip**; (3) decide if a higher rate (e.g. 1000 Hz
data + ~400 Hz cutoff) is justified. Success = a data-backed answer on decimation, not intuition.

## Key decisions already made (don't relitigate)
1. **Gate must be measurable-derived by end of sim** (sim-to-real): labels (true vpm, μ, χ) must phase out,
   so a label-gate is a non-starter as endpoint.
2. **No separate slip head** — `v_slip` is a deterministic function of (γ,ΔV); it's a *redundant-at-optimum*
   consistency regularizer, not a third task. (Confirmed by the identity: true γ+ΔV ⇒ true slip.)
3. **Slip-loss leverage is ~70:1** (`∂Vp/∂ΔV ≈ 1` vs `∂Vp/∂γ ≈ 0.0141 m/rad·s over θt∈±15°`) — a *physics*
   property, so it preferentially corrupts ΔV. Weight sizing (from early-epoch MSE): balanced `w_slip ≈ 0.02`
   (γ=1); user's earlier 100 was ~1000× too high. User **does not want to detach ΔV**.
4. **GradNorm REJECTED** — it equalizes gradient magnitudes and structurally *down-weights γ* (the priority
   task, larger gradient); wg crushed to ~0.03, γ degraded. Also ~2× compute (2 extra `autograd.grad`/batch).
5. **Uncertainty weighting REJECTED** — gameable (inflate σ to lower loss without improving).
6. **Heads are fine, not the problem:** ΔV mm/s-accurate (corr 0.997); γ ~5.5 rad/s (norm MSE ~0.008).
   The >20% slip error was the base bug + (secondarily) decimation, NOT head error.

## Facts / results already in hand
- **Full 2-fold cross-run done** (buggy physics): `runs/S1_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm/`
  + S2. Best val 0.288/0.325; cross-test 0.288/0.366. LR collapsed by ep77 (bug: scheduler stepped on
  physics-dominated terminal metric during grounding — since fixed to use supervised metric in grounding).
- **True slip fractions (ground-truth vpm, pooled 4 wheels):** stick(<0.01) ~25–31%, slip(0.01–0.6) ~65–75%,
  high(>0.6) ~1–7%. Multisine (held-out OOD excitation) ~99% slip, v_max~0.13 (low-spin/easy).
- **Grounding runs (all `runs/S1_train_hy3_grnd*`):** baseline (w_dv=1) γ→0.008; w_dv=10 starved γ→0.027;
  w_slip=100 wrecked ΔV (std 16×, corr 0.09) — now explained as ΔV compensating for the base bug.
- **Implemented + working:** grounding early-stop (`--no-grounding-early-stop` to disable), LR/scheduler
  reset at grounding→physics, supervised-metric scheduler+best during grounding, checkpoint every 10 epochs,
  GradNorm (`--gradnorm`), `--w-slip`, `OBS_PHASE_PLAN='N,0,0,0,0'` env for grounding-only, resume clean.

## Open / next steps (numbered deliverables)
1. Finish base-bug fix in `physics_loss_hy3` (mirror the `derived_contact_slip` edit); verify with
   `slip_attribution.py` → identity floor should drop 0.69 → ~0.002.
2. New probe: corrected identity RMSE **binned by stick/slip/high-slip** (extend `slip_attribution.py`).
3. If high-slip decimation floor is large near the 0.01 gate boundary → evaluate 1000 Hz data + ~400 Hz
   anti-alias cutoff (note: **500 Hz cutoff at 1000 Hz aliases**; and rate change = 2× cache/train +
   changes window/SSM dt + deployment rate). Else leave rate at 500 Hz.
4. Only after the floor is understood: re-run grounding+slip (fixed physics, `w_slip≈0.02`) and re-check
   `calibrate_gate.py` — does the gate anti-correlation disappear once the base is correct?

## Conventions to respect
- **Go step-by-step; verify before killing runs.** (User flagged that a run was killed before a fix was
  verified.) Prefer measurement over assertion; the user reasons hard from physics and catches term-mixups.
- **Surgical edits, not rewrites.** Match existing code idiom; explain diffs before/after applying.
- **Compute:** laptop RTX 3060 6 GB; commit-limit ceiling ~2 concurrent torch processes; long runs need
  `keep_awake.py`. Thread/worker ≤ 8. A `keep_awake.py` daemon may still be running — stop it.
- **No LaTeX in chat** (Unicode + code blocks). Temp files in `_tmp/`. Data in `data/` is read-only.
