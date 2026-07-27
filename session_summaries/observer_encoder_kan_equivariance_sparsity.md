# Encoder Design Inferences — hy3-gammakin Observer Lift

**Context**: Replacing `nn.Linear(13, 64)` (the `feat` projection in
`observer_v1_py/mecanum_observer/models_v2hy3.py:43`) that lifts the per-wheel
sensor window into the MambaLiteSSM latent. Input `x ∈ ℝ¹³` per
`(batch, wheel, timestep)`:

```
[ V̂x, V̂y, ψ̇_gyro, a_x, a_y,          # 5 globals (sensor front-end)
  Msat_i, ω_i, sinθ̃_i, cosθ̃_i,      # 4 per-wheel measurables
  emb_i (4) ]                          # wheel embedding (currently learned, zero-init)
```

Output: `ℝ⁶⁴` → Mamba causal scan → final rep `[B*4, 64]` → `head_gamma`
(per-wheel, `[B,4]`) + `head_dv` (concat 4×latent → `[B,2]`).

The gammakin parametrization already subtracts dominant physics:
`γ_hat = γ_noslip(V_y_used) + dγ̂·dγ_scale` (`training_v2hy3_gammakin.py`,
`physics_v2hy3.py:92`). The encoder only needs to lift the **slip-corrector
residual** — so it should be cheap, smooth, and physics-aligned.

---

## 1. KAN choice over alternatives

**Decision**: KAN (Kolmogorov-Arnold Network) is the preferred lifting over
`nn.Linear`, DeepONet, FNO, RFF, or a plain MLP-hybrid.

### Why not the others
- **`nn.Linear(13,64)`** (current): single matmul, entangles globals↔wheel-meas↔emb
  all-to-all; cannot separate the smooth 1-D physics terms; no Koopman structure.
- **DeepONet**: learns operator `u(s)→v(y)` between function spaces. Our task is
  finite-dim pointwise lift; Mamba already owns the temporal/window operator. A
  literal branch+trunk DeepONet would need the full window → **non-causal**, and
  competes with Mamba. Only the *hypernetwork idea* (branch compresses window →
  modulates trunk weights) is salvageable — and that is covered under "dynamic
  sparsity" below.
- **FNO**: spectral conv on the `W` (time) axis — replaces Mamba, not the lift.
  Out of scope; don't touch the scanner.
- **RFF (Random Fourier Features)**: good cheap kernel-Koopman baseline, learnable
  `ω` adapts to wheel (10–100 Hz) / friction (100–1000 Hz) bands, but less
  interpretable than spline edges and no built-in separability.
- **Hybrid (fixed physics dict + residual MLP)**: strong, but the "fixed dict"
  requires manual monomial pruning; KAN gets the same effect *automatically* via
  spline edges.

### Why KAN wins
- KAN replaces linear weights with **learnable 1-D B-spline activations on each
  edge** + a sum. Additive separability = exactly the Koopman eigenfunction
  intuition (latent = sum of 1-D observables).
- Contact kinematics (`run_one.jl:680-684`) and
  `γ_noslip = -Vpy0/(cosδ·(R·cosθ̃-Rd))` are smooth 1-D/rational functions of
  `(Vpy0, cosθ̃)`. A KAN **edge** `φ(Vpy0)`, `φ(cosθ̃)` learns these directly
  with ~1 hidden layer — no need for a deep net.
- Interpretable edge functions; can inspect which observables matter.
- One hidden layer (13 → 32 → 64) keeps it **deployable at 500 Hz**.
- **KAE** (ensemble of KANs) gives per-wheel latent uncertainty for free — useful
  because the `|Vp|` gate decides stick/slip on a ~1 cm/s margin
  (`losses_v2hy3.py` gate `g = σ((|Vp|-center)/width)`).

### Recommended stack
1. Fixed analytic prefix (4–8 dims) from the front-end:
   `Vpx0_hat, Vpy0_hat, cti, sti, Msat, ω` — already computed in
   `sensor_frontend_v2.py`, zero-param Koopman observables.
2. `KAN([13, 32, 64])` for the learnable smooth residual lift.
3. Keep Mamba for the causal scan (untouched).

---

## 2. Equivariance of the O-configuration

### What the symmetry actually is
The 4 wheels are in an **O-configuration** with distinct
`δ = [-π/4, π/4, π/4, -π/4]` and positions `px, py`
(`PlatformParams` in `run_one.jl`). The robot is **NOT permutation-symmetric**:
wheels have distinct `δ_i, px_i, py_i`.

The correct inductive bias is therefore:

> **Same compute graph per wheel (weight-tied), each wheel fed its own local
> geometry.** Equivariant under *wheel relabeling* — swap wheel labels **and**
> their `(δ, px, py, θ̃)` inputs, and the output latents swap identically.

This is stronger and cleaner than the current learned `wheel_emb` (zero-init,
freezable), which is a weak proxy for geometry.

### Physics backing
- `γ_noslip_i` depends **only** on wheel-`i`'s own `(Vpy0_i, δ_i, θ̃_i)` plus the
  *shared* body `(V_y, Ω)` (`physics_v2hy3.py:92-113`). So per-wheel KAN edges
  need **no cross-wheel term** for the no-slip base.
- The only true cross-wheel coupling is through **body `V_y, Ω`** (shared) →
  handled by an **equivariant aggregator** (sum/mean-pool or equivariant
  attention), never by asymmetric weights.
- `head_dv` already preserves the asymmetric O-config by concatenating the 4
  wheel latents (`models_v2hy3.py:60-63`).

### Concrete structure
```
x_i,t = [5 globals ‖ 4 wheel-meas ‖ 3 explicit geometry (δ_i,px_i,py_i)] ∈ ℝ¹²
        (drop learned emb; make geometry explicit)
h_i,t = KAN_shared(x_i,t)              # weight-tied ∀i  → ℝ¹⁶
ĥ_i,t = EquivAgg({h_1..h_4}, V_y, Ω)   # permutation-equivariant
z_i,t = concat over i → ℝ⁶⁴ → Mamba
```
- `KAN_shared` has **one** param set used for all 4 wheels ⇒ relabel-equivariant
  by construction.
- `EquivAgg`: sum-pool the 4 latents + broadcast-modulate with shared body
  `(V_y, Ω)`, or `nn.MultiheadAttention(16,4)` over the 4 latents (Q,K,V from the
  same shared KAN ⇒ still equivariant).
- Params: KAN ~1.5k + pool ~0.4k ≈ 1.9k (vs dense 0.9k); causality preserved.

---

## 3. Learning sparsity of encoded states (for Δγ and ΔV heads)

The 13→64 lift feeds **both** heads: `head_gamma` (per-wheel `dγ̂`) and
`head_dv` (concat 4×latent → `ΔV̂`). Making the lift **non-dense** and
**learned-sparse** directly shapes what each head receives.

### Non-dense motivation (structured, not random)
Dense `W ∈ ℝ⁶⁴ˣ¹³` lets `Msat_3` write directly to wheel-1's latent — physically
wrong (coupling is only via body `V_y, Ω`). Block/pruned structure:
```
W = [ W_glob→all (5×64)  |  W_wheel→own (4×16 block-diag)  |  0 ]
```
Edges zeroed per `run_one.jl` causal graph:
- **Keep**: `V̂y, ψ̇ → γ_i`; `cosθ̃_i → γ_i`; `a_x,a_y → γ_i` (via V̂̇).
- **Drop**: `V̂x → γ_i` (γ depends on V_y, Ω only, Model 1); `Msat_j → γ_i (j≠i)`;
  direct cross-wheel force paths.

Params drop ~2.3× (896 → ~384), less overfit on the 9.94M-sample corpus, better
sim-to-real, faster at 500 Hz.

### Making the sparsity *learned* (three options)
1. **L0/L1 gate on edges** — `g = sigmoid(log_alpha)` multiplies `W`; penalty
   `(1-sigmoid(log_alpha-0.5)).sum()`. Edges collapse to zero → mask emerges from
   data (e.g. `V̂x→γ_i` drops, `V̂y,cosθ̃→γ_i` survive). Use as regularizer
   (`λ≈1e-4`), not a hard constraint (risk of underfit).
2. **Dynamic (input-dependent) gate** — *best fit*. Mask = `f(state)` via a
   hypernetwork: `g = sigmoid(gate(ctx))`, `ctx = [V_y, Ω, a_x, a_y, |Vp|]`.
   In **stick** the gate suppresses slip-term edges (`Msat, ω`→high-order); in
   **slip** it opens them. This makes the encoder's connectivity follow the *same*
   `|Vp|` gate the physics loss already uses (`losses_v2hy3.py`) — the latent
   representation auto-adapts to regime. Fully learnable, causal, 500 Hz-safe.
3. **Learned low-rank + sparse residual** — `z = x@(V@U.T).T + x@(W_sparse·gate).T`
   with `U∈ℝ⁶⁴ˣ⁸, V∈ℝ¹³ˣ⁸` (rank `r=8` SVD-init). Captures smooth global subspace
   + per-wheel transients; network decides the split.

### Effect on the two heads
- **`head_gamma` (dγ̂)**: receives a per-wheel latent built from sparse,
  physics-gated observables → the residual it predicts is small in stick (where
  `γ_noslip` already covers ~95%), large only in genuine slip. Matches the
  gammakin motivation (19× stick-error reduction vs v2hy3).
- **`head_dv` (ΔV̂)**: receives the concat of 4 sparse latents; the body-coupled
  (equivariant) part carries the shared `V_y, Ω` context, the per-wheel sparse
  part carries local `Msat, ω, θ̃` corrections. Uniform `dv_scale`
  (`config_v2hy3_gammakin.py:65`) stays valid because sparsity is per-wheel
  symmetric.

### Recommendation
Use **approach 2 (dynamic gate) + equivariant KAN** (spline edges are naturally
sparse). This yields a fully **learnable non-dense, Koopman-aligned,
physics-gated** lift with **no manual masking** — the connectivity follows the
stick/slip regime switch end-to-end, benefiting both `dγ̂` and `ΔV̂` heads.

---

## File pointers (for implementation)
- Lift location: `observer_v1_py/mecanum_observer/models_v2hy3.py:41-43`
  (`self.feat = nn.Linear(raw_in, cfg.d_model)`)
- Mamba internals: `observer_v1_py/mecanum_observer/models.py:27-77`
- γ_noslip / contact kin: `observer_v1_py/mecanum_observer/physics_v2hy3.py:92`,
  `physics.py:75`
- Physics loss / gate: `observer_v1_py/mecanum_observer/losses_v2hy3.py:87-141`
- Front-end observables: `observer_v1_py/mecanum_observer/sensor_frontend_v2.py:300-304`
- gammakin config: `observer_v1_py/mecanum_observer/config_v2hy3_gammakin.py`
