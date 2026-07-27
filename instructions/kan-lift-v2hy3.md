# KAN Lift for Observer v2-Hy3 — Execution Brief (for K2.7 session)

**Origin:** designed in a K3 planning session on 2026-07-20; full reasoning and citations in
`session_summaries/2026-07-20_kan_encoder_design_k3.md`. This brief is self-contained — you do
not need to read the summary to execute, but it has the "why".

**Goal:** replace the per-step **linear** state lift in front of the Mamba-lite SSM encoder with a
small **KAN (Kolmogorov-Arnold) lift**, with (a) Z₂ odd-parity-constrained edge bases and (b) a
learned sparse edge mask, and validate it against the existing v2hy3 baseline on the standard
S1/S2 folds.

---

## 1. Why KAN (one paragraph of physics grounding)

The simulator (`run_one.jl`, extracted from the authoritative notebook) computes contact slip as a
**linear** combination of measurables given `sin/cos(12θ)` roller-harmonic features
(`run_one.jl:680-684`), and all friction nonlinearity is **univariate in the scalar slip
magnitude**: Stribeck `g(s) = μ(1+(r−1)·exp(−(s/v_str)²))`, bristle rate `σ0·s/g(s)`, Mindlin
`1−(1−x)^{2/3}` (`run_one.jl:290-331`). That is exactly the Kolmogorov-Arnold compositional form
`F = Σ φ_k(⟨w_k, x⟩)`, so a KAN lift is structurally matched to the plant; FNO/WNO/DeepONet lifts
were considered and rejected (see session summary §2).

## 2. Scope

**In scope (this task):** the v2-Hy3 observer lift only.

- `observer_v1_py/mecanum_observer/models_v2hy3.py:43` — `self.feat = nn.Linear(raw_in, cfg.d_model)`,
  `raw_in = 5 + N_PERWHEEL(4) + emb_dim(4) = 13`, `d_model = 32`.

**Out of scope (do NOT touch in this task):** `MambaLiteSSM.in_proj` (`models.py:47`), the
ForceRecon `SelectiveSSM.in_proj` (`Mecanum_PINN_Mamba_ForceRecon_v1/.../models.py:93`), the
gammakin variant, all training-phase logic beyond the sparsification schedule hooks in §6. These
are follow-up phases after the v2hy3 lift is validated.

## 3. Deliverables

1. `observer_v1_py/mecanum_observer/kan_lift.py` — new module (see §4 spec).
2. `observer_v1_py/mecanum_observer/models_v2hy3kan.py` — `WheelObserverV2Hy3KAN`, a subclass/sibling
   of `WheelObserverV2Hy3` that swaps `feat` for the KAN lift. Reuse `MambaLiteSSM`, both heads,
   and `load_warm_start_v2hy3` unchanged.
3. `observer_v1_py/mecanum_observer/config_v2hy3kan.py` — config extending `ObserverConfigV2Hy3`
   with the knobs in §5.
4. `observer_v1_py/train_observer_v2hy3kan.py` — entry point mirroring `train_observer_v2hy3.py`
   (same CLI + regime TOMLs), plus the sparsification schedule (§6).
5. `observer_v1_py/test_kan_lift.py` — unit tests (§7).
6. Launcher + `.bat` per AGENTS.md §6 (live-tree execution model): a launch script AND a Windows
   `.bat` companion targeting
   `C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\`.
7. Smoke run + one full S1 run; metrics committed per the repo's run-archive convention
   (`observer_v1_py/runs/<tag>/metrics.json`, `LOSS_AND_NORM.md`).

## 4. `kan_lift.py` module spec

B-spline KAN in the efficient-KAN formulation (base weight + spline weight + grid), implemented in
plain PyTorch (no external KAN dependency — check `Manifest`/`pip list` first; do not add one
without flagging it). Requirements:

```python
class KANLift(nn.Module):
    """raw_in -> d_model, 1 or 2 KAN layers.

    Per edge e=(i,j):  out_j += base_w[j,i] * SiLU(x_i) + sum_k spline_w[j,i,k] * B_k(x_i)
    B_k = Cox-de Boor basis on a per-layer uniform grid over [grid_min, grid_max],
    G grid intervals, order k (cubic default).
    """
```

- **Parity split (Z₂):** each output channel is declared `odd` or `even` (config list or fraction,
  default: first half odd, second half even). Odd edges use `φ(x) = ψ(x) − ψ(−x)` (equivalently:
  fit basis on half-grid `x ≥ 0`, evaluate `sign(x)·ψ(|x|)`); even edges use `ψ(|x|)` form or a
  full grid with mirrored coefficient tying. **No biases, no LayerNorm anywhere in the parity
  layers** (both break parity — this is why FastKAN's LN is unacceptable here).
- **Grid adaptation:** after N optimizer steps (or on phase transitions), refit each layer's grid
  to the observed per-input-channel p01–p99 activation range. The Stribeck dip is sharp
  (`v_str = 0.01 m/s`), so grid placement matters; log grid-refit events to the run log.
- **Edge scoring:** method `edge_scores(x_calib) -> Tensor[in, out]` returning
  `mean(|φ_e(x)|)` over a calibration batch (used by the pruning step in §6).
- **Param report:** `count_parameters()` helper; print at startup. Expected: 13→32 spline-KAN
  (G=8, k=3) ≈ 5.1k params vs 448 for the current linear (whole model ≈ 8.4k → ≈13k, +55%).

## 5. Config knobs (`config_v2hy3kan.py`, extend `ObserverConfigV2Hy3`)

```python
lift_type: str = "kan"            # {"linear", "mlp", "kan"}  — "mlp" is the fairness control
kan_hidden: int = 32              # inner width if 2-layer; 1-layer if == 0
kan_grid: int = 8                 # G grid intervals
kan_spline_order: int = 3         # cubic
kan_grid_range: Tuple[float, float] = (-3.0, 3.0)   # inputs are max-norm normalized
kan_parity: str = "half_odd"      # {"none", "half_odd"} — odd/even output channel split
kan_grid_refit_every: int = 0     # epochs between grid refits; 0 = only at phase transitions
# --- sparsification (§6) ---
sparse_lambda: float = 0.0        # group-lasso strength on per-edge spline coeff vectors
sparse_l1_lambda: float = 0.0     # KAN-paper L1 on edge activations
prune_at_phase: str = "grnd_rampdown"   # phase boundary to hard-prune at
prune_threshold_rel: float = 0.01 # keep edges with score >= 1% of max edge score
```

`lift_type="mlp"` must implement a **parameter-matched** 2-layer MLP+SiLU lift (same param count
as the KAN, ±10%) — this is the fairness control arm (Yu et al., arXiv:2407.16674); results are
only meaningful against it, not against the 448-param linear.

## 6. Sparsification schedule (hook into the existing 5-phase curriculum)

The v2hy3 trainer already phases training (grounding → phys_rampup → overlap → physics →
grnd_rampdown; see `training_v2hy3.py` and the phase checkpoints in
`observer_v1_py/runs/S1_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm/phase_ckpts/`). Insert:

```
grounding … physics        : sparse_lambda = 0 (dense; redundancy aids optimization)
physics (last 25% epochs)  : ramp sparse_lambda 0 → cfg value; optional small L1
at grnd_rampdown start     : score edges on a val-window calibration batch,
                             hard-prune below prune_threshold_rel, freeze mask,
                             refit grids to surviving inputs' ranges
grnd_rampdown              : finetune with mask frozen
```

Log: edge count before/after, per-input-channel surviving-out-degree (that vector is the
"which measurables matter" output — save it to the run dir as `kan_edge_graph.json`).

## 7. Tests (`test_kan_lift.py`, run with the persistent venv `~/.kimi-code/venvs/mecanum_widget/bin/python -m pytest`)

1. **Parity:** with `kan_parity="half_odd"`, random `x`: odd output channels satisfy
   `f(−x) == −f(x)` to 1e-6; even channels satisfy `f(−x) == f(x)`. Also after a grid refit.
2. **Shape/dtype:** forward on `[B*4, W, 13]` matches the old `feat` output shape `[B*4, W, 32]`.
3. **Param count:** within 5% of the closed-form estimate `13·32·(G+k+1) + LN(0)` for 1-layer.
4. **Prune determinism:** `edge_scores` on a fixed seed batch is reproducible; pruning is
   idempotent (second prune removes nothing).
5. **Smoke train:** `--limit-files 1 --epochs 2` end-to-end through `train_observer_v2hy3kan.py`
   (CPU is fine), checkpoints save/load, `kan_edge_graph.json` written.

## 8. Runs and acceptance

Run from the live tree (`/mnt/c/.../code_insights/`), same data/scaler CSV as the v2hy3 baselines,
≤8 workers, `keep_awake.py` for anything >20 min:

1. **Smoke:** S1, `--limit-files 8`, tag suffix `_kansmoke8` (mirrors the existing `_smoke8`
   convention in `launch_parallel_v2.py`).
2. **Main:** S1 full run, `lift_type=kan`, `kan_parity=half_odd`, sparsification on. Tag:
   `S1_train_w32_gamma_dv_v2hy3kan_phys_max_norm`.
3. **Control:** S1 full run, `lift_type=mlp` (parameter-matched), same seed/protocol.

Acceptance (compare against `observer_v1_py/runs/S1_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm/`):
- γ̂ physical RMSE and ΔV̂ RMSE (val) ≤ baseline + 5%, at ≤ +60% params;
- KAN lift ≥ control MLP on the same metrics (else stop and report — do not iterate silently);
- `kan_edge_graph.json` shows sparse survival (expect: no surviving edges from other wheels'
  per-wheel features into a wheel's own lift — the ground-truth graph is wheel-local,
  `run_one.jl:680-684`);
- parity test passes on the trained checkpoint (load and re-run test 1).

## 9. Pitfalls (check these first when something looks wrong)

- **Grid range vs normalization:** inputs are max-norm normalized; confirm the actual p01/p99 per
  channel on a real batch before fixing `kan_grid_range`. Out-of-grid inputs get zero basis
  response in efficient-KAN — silent dead edges.
- **Scaler symmetry for parity:** Z₂ negation must commute with normalization. Max-norm is fine;
  any asymmetric (p05/p95) channel scaling is not — check the scaler CSV before enabling
  `kan_parity`. If asymmetric, either symmetrize the scaler or drop parity for that channel.
- **Warm-start:** `load_warm_start_v2hy3` already skips `feat.*` (raw_in mismatch), so the KAN
  lift trains fresh; keep the full grounding phase budget — do not shorten it.
- **Bias/LayerNorm creep:** any `nn.LayerNorm` or bias added inside the parity layers voids test 1.
- **Backward memory:** the per-edge basis cache is ~150 MB at B=64/W=32 (G=8,k=3); if the training
  box is memory-tight, drop G to 5 before dropping batch size.
- **Do not** modify shared files (`models.py`, `training_v2hy3.py`, `data_v2hy3.py`) in place —
  subclass/wrap in the new `*kan*` modules so the gammakin and baseline v2hy3 paths keep working.
- **Data rules:** Arrow data is read-only; all runs from the `/mnt/c` live tree; temp artifacts in
  `_tmp/`; new run dirs under `observer_v1_py/runs/` with new tags (never overwrite existing runs).

## 10. References

Design + citations: `session_summaries/2026-07-20_kan_encoder_design_k3.md`. Load-bearing ones:
KAN (arXiv:2404.19756), KAN 2.0 sparsify→prune→symbolify (arXiv:2408.10205), KAN-or-MLP fairness
(arXiv:2407.16674), EKAN/EMLP equivariance framework (arXiv:2410.00435; Finzi et al. ICML 2021),
group lasso (Yuan & Lin 2006), L0 gates (arXiv:1712.01312). Physics ground truth: `run_one.jl`
(slip kinematics 680-684, LuGre/Stribeck/Mindlin 290-331).
