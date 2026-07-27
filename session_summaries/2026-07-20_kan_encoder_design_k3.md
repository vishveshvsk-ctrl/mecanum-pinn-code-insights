# Session Summary — KAN Encoder Design for Mamba SSM Lifting (`_k3`)

**Date:** 2026-07-20 (session run on K3, 1M context)
**Scope:** Architecture research + planning for replacing the linear/MLP state-lifting stage in front of the Mamba-lite SSM encoders (Observer v2-Hy3/gammakin, Mamba ForceRecon v1). Execution deferred to a K2.7 session via `instructions/kan-lift-v2hy3.md`.

---

## 1. Session setup actions

- Read `kimi_start_prompt.txt`, `AGENTS.md`, `PROJECT_LAYOUT.md`. Confirmed WSL git working tree; live data tree at `/mnt/c/Users/vishv/OneDrive/Desktop/Vishvesh_Data/VNIT/mecanum_pinn_head/`.
- Model context: session runs `kimi-code/k3` (1M ctx). K2.7 Code = 256K ctx, ~1/3 token price ($0.95/$4 vs $3/$15 per M). Plan: **plan with K3 → hand off → execute with K2.7**, using this summary + the instruction brief as the bridge.
- **Git sync**: local was behind 2 commits (`b05e96f → 566c3bf`, 296 files, +94,628 lines) with stale July-10 local drafts of v2hy3 files conflicting with the incoming July-20 versions. Backed up 9 local files to `_tmp/git_sync_backup_20260720/`, fast-forwarded cleanly. Untracked local-only items kept: `instructions/smoke-test-accel-sidecars.md`, `observer_v1_py/runs/_smoke_v2hy3_limit1/`.
  - New in tree after sync: Observer v2-Hy3 + **gammakin** variants (trainers, configs, S1/S2 run dirs with checkpoints), `tools_accel/` IMU/accel sidecar pipeline, `energy_audit/`, new handoffs (`frontend_drift_audit_slip_gating_handoff.md`, `hy3_decimation_slip_reconstruction_handoff.md`, `inverse_v2_brainstorm_handoff.md`).

## 2. Core design question and conclusion

**Question:** which neural operator can outperform the MLP/linear lifting of states into the Mamba SSM hidden dimension?

**Current lifts (all purely linear):**
- `observer_v1_py/mecanum_observer/models_v2hy3.py:43` — `feat = nn.Linear(13, 32)` (v2hy3)
- `observer_v1_py/mecanum_observer/models.py:47` — `MambaLiteSSM.in_proj = nn.Linear(32, 32)`
- `Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/models.py:93` — `SelectiveSSM.in_proj = nn.Linear(11, 32)`

**Conclusion: B-spline KAN lift** (efficient-KAN style, ~2 layers, 13→32→d_model), chosen over FNO/spectral, WNO, DeepONet because the simulator's own structure is Kolmogorov-Arnold-shaped:

- Slip kinematics are **linear in measurables** given `sin/cos(12θ)` features (`run_one.jl:680-684`).
- All friction nonlinearity is **univariate in scalar slip magnitude**: Stribeck `g(s) = μ(1+(r−1)·exp(−(s/v_str)²))`, bristle rate `σ0·s/g(s)`, Mindlin `1−(1−x)^{2/3}` (`run_one.jl:290-331`).
- So the force law decomposes as linear lift → univariate maps → linear recombination — the KA compositional form exactly.
- Rejected alternatives: FNO (convolutional/global-frequency — friction law is pointwise in slip, not spectral; roller harmonics already hand-encoded), WNO (only if transient chatter focus), DeepONet (only if cross-rate generalization becomes a target).
- Bonus: learned edge functions can be plotted against the analytic Stribeck curve — paper-grade interpretability.

## 3. Parameter cost (computed, not estimated)

Lift 13→32: linear 448 → spline-KAN (G=8,k=3) 5,056 (~11×); FastKAN RBF G=8: 3,808 (~8.5×). Whole v2hy3 model: 8,357 → 12,965 params (**+55%**, ~44→52 kB checkpoint). Activation cache for backward ~150 MB at B=64/W=32. Mitigation if parity needed: KAN 13→16 ≈ 2.5k params; report vs parameter-matched MLP control (Yu et al. fairness critique).

## 4. Equivariance analysis

- **Z₂ friction oddness**: `F(−v) = −F(v)` — exact symmetry of `lugre_dyn_rates`; μ, χ, N all enter even. Implement as parity-split edge bases (`φ_odd(x) = ψ(x) − ψ(−x)` or half-grid + mirror). Doubles effective grid resolution on s≥0 (helps the sharp `v_str = 0.01` dip). Caveats: no biases/LayerNorm on parity edges (breaks FastKAN's LN); normalization must commute with sign flip (max-norm OK; asymmetric percentile scalers not).
- **Holds under per-wheel μ/χ heterogeneity** — parity is constitutive per wheel, independent of parameter values. μ/χ as conditioning inputs = even features.
- **Wheel permutation (S₄)/D₂**: shared encoder + frozen zero `wheel_emb` already gives S₄; survives per-wheel μ/χ because the per-wheel map "window → γ̂" is identical across wheels (μ inferred per window). Breaks only for properties not inferable from the window — the reserved `wheel_emb` unfreeze hook (sim-to-real). `head_dv`'s fixed concat order already spends geometry asymmetry.
- Already free: time-translation (per-step lift + scan), yaw-frame invariance (body-frame inputs).
- Literature: EKAN (arXiv:2410.00435), EMLP (Finzi et al., ICML 2021) — matrix-group framework; Z₂/D₂ are diagonal ±1 actions, its easiest case. Full EKAN machinery judged overkill; parity-by-construction preferred.

## 5. Sparse (non-dense) KAN encoder

- KAN edges are individually scoreable (`s_e = E_x[|φ_e(x)|]`) — "which edges to keep" is learnable, unlike MLP weights.
- Recommended: **dense → sparsify → prune → finetune**; never train sparse from scratch (lottery-ticket effect).
  - Mechanisms: group lasso on spline coefficient vectors (Yuan & Lin 2006); KAN-native L1+entropy (KAN paper §; KAN 2.0 workflow); L0 hard-concrete gates (Louizos et al.) if a sparsity budget must be targeted.
  - Schedule into v2hy3 5-phase curriculum: dense through grounding/physics → penalty late physics → prune before grnd_rampdown → finetune in rampdown; refit grids after pruning.
- Structured variant: input-channel group lasso = sensor/feature selection (ties into observability-report program).
- Physics payoff: ground-truth dependence graph is exactly sparse (`Vpx_i` depends only on `(Vx, ψ̇, ωᵢ, γᵢ, θᵢ)` — `run_one.jl:680-684`); surviving-edge graph is a falsifiable result. Pruned + symbolified KAN ≈ SINDY with learned coordinates (Brunton et al., PNAS 2016).
- Compatibility: whole-edge pruning preserves Z₂ parity automatically. Acceptance: mask stability across S1/S2 folds (and clean vs `noise_stage="real"`); edge-set differences = noise-fragility diagnostic.
- Honest caveat: pruned nets lose optimization slack — expect slightly worse raw RMSE at equal width; widen the pruned skeleton rather than un-prune if accuracy dips.

## 6. Key citations (verified)

KAN (arXiv:2404.19756, ICLR 2025) · KAN 2.0 (arXiv:2408.10205) · FastKAN/RBF (arXiv:2405.06721) · Chebyshev KAN (arXiv:2405.07200) · KAN-or-MLP fairness (arXiv:2407.16674) · Shukla et al. CMAME 431 (2024) · ss-Mamba (arXiv:2506.14802) · SMamba-KAN (Frontiers in Water, 2025) · TSKANMixer (arXiv:2502.18410) · FNO (arXiv:2010.08895) · DeepONet (Nat. Mach. Intell. 3:218, 2021) · WNO (CMAME 404:115783, 2023) · TimesNet (arXiv:2210.02186) · spectral bias (arXiv:1806.08734) · EKAN (arXiv:2410.00435) · EMLP (Finzi et al., ICML 2021, PMLR 139:3318) · L0 gates (arXiv:1712.01312) · group lasso (Yuan & Lin, JRSS-B 2006) · lottery ticket (arXiv:1803.03635) · SINDY (PNAS 113(15):3932, 2016) · KAN model discovery (PRE 7:023037, 2025) · Mamba (arXiv:2312.00752) · S4D (arXiv:2206.11893) · LuGre (IEEE TAC 40(3):419, 1995) · Revisiting LuGre (IEEE CSM 28(6):101, 2008 — in `docs/`) · Stribeck (Z. VDI 46, 1902) · Mindlin (J. Appl. Mech. 16:259, 1949) · Adamov & Saypulaev (IEEE NIR 2021, doi:10.1109/NIR52917.2021.9666053 — in `docs/`).

## 7. Outstanding engineering flags for execution

1. Sharp Stribeck dip (`v_str = 0.01`): prefer B-spline KAN with grid adaptation over fixed-center RBF FastKAN; verify normalized input range covers the dip.
2. Warm-start: `load_warm_start_v2hy3` already skips `feat.*`; KAN lift trains fresh — plan a full grounding phase.
3. Check `dv_scale`/scaler CSV symmetry before enabling Z₂ parity.
4. Report KAN lift against a parameter-matched MLP lift control, not just the 448-param linear.
5. Per AGENTS.md §6: any runnable training/launch script produced during execution needs a Windows `.bat` companion targeting the `/mnt/c` live tree; ≤8 workers; `keep_awake.py` for runs >20 min.

**Next step:** execute `instructions/kan-lift-v2hy3.md` in a K2.7 session.
