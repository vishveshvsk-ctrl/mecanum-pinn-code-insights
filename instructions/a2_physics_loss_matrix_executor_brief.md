# Executor brief — A2 physics-loss 2×2 matrix (implement body residual + run/evaluate)

**Role:** You are the EXECUTOR. Implement the changes below verbatim, run the matrix, evaluate,
and hand back the results. Do NOT redesign. If a numeric constant or a residual sign disagrees
with `base.toml` / `run_one.jl`, STOP and report — do not guess (authority rule #1).

**Planner intent (do not change):** A2's physics consistency is the SAME one-step plant EOM expressed
TWO ways — we compare them as two **scaling/normalisation regimes** (NOT different physics):
- **`residual`** — instantaneous force/accel residual: per-wheel **wheel torque balance** (`J·ẇ −
  (Msat − R·Fpar − p1·w)`) + **body accel** (`M·dv_meas − RHS`), each normalised by control-torque /
  per-wheel-weight scales.
- **`integrated`** — one **Heun step (held force, O(dt²))** from the MEASURED state at t using the
  predicted forces, yielding predicted `Vx,Vy,ψ̇,w₁..₄` at t+1, compared to the MEASURED next sample,
  each normalised by its **p95** (Vx 1.92, Vy 0.673, ψ̇ 2.408, w 39.089).
These are algebraically the same constraint up to constants (e.g. wheel: `integrated_err =
−(dt/J)·wheel_residual`); the experiment isolates **loss conditioning under the two scalings**. So:
run the cross `{none, residual, integrated}` — **never a cell that sums both** (double-counts the
constraint) — and compare on the (already-fixed) cross-subset metric. The integrated form couples
wheel+body in one step, so there is NO wheel/body split. **DROP the roller/γ torque-balance term from
BOTH losses** (ill-conditioned, non-unique γ kinematics) but KEEP computing it every step for
**monitoring only** (logged per-wheel, never backpropped).

---

## 0. Environment / conventions (HARD)

- Run everything from `code_insights/`.
- **torch python:** `C:\Users\vishv\miniforge3\envs\myenv\python.exe` (call by full path; cannot `conda activate` in tool shells).
- **torch-free python (numpy/pandas/pyarrow/matplotlib):** `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`.
- Workers need `-u` and `PYTHONUTF8=1`.
- **Decimated cache (warm, off OneDrive):** `C:/Users/vishv/mecanum_cache_decim` — always pass `--cache-dir C:/Users/vishv/mecanum_cache_decim`.
- **Scaler (max-norm):** `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv`.
- **Parallelism cap:** on THIS laptop the ceiling is `--max-parallel 2` (Windows commit-limit, not RAM/GPU — spawn re-imports torch per worker). On the 24 GB/128 GB box, raise `--max-parallel` until `nvidia-smi` GPU-util saturates (VRAM/RAM are not the limit there).
- Long runs need the sleep lockdown + `keep_awake.py`; stop keep_awake when done.
- Scratch in `_tmp/`, delete after. Static matplotlib only. No interactive widgets.

---

## 1. The two formulations (same one-step EOM, two scalings; both reuse the verified LuGre law)

`observer_v1_py/mecanum_observer/physics.py::lugre_forces(...)` already returns per-contact
**roller-frame** forces `(Fpar, Fperp, Mz)` from predicted states `(gamma, zx, zy, zs=0)` and is
verified against stored Arrow forces. `contact_from_gamma(...)` gives `Vpx, Vpy, w_z`.

**Existing residuals** (`physics.py::roller_residual`, `wheel_residual`; `losses.py::physics_loss`):
```
wheel  (TRAIN) :  J_wheel*w_dot - (Msat - R*Fpar - p1*w)        -> 0  (per wheel; / WHEEL_SCALE)
roller (MONITOR):  p2*gamma + Fpar*dVpx/dg + Fperp*dVpy/dg      -> 0  (per wheel; logged, NOT in loss)
```
(`Fpar` is the longitudinal/parallel roller force = `Fx` returned by `lugre_forces`.)

**UNITS NOTE (the roller residual is dimensionally correct — do not "fix" it; it's just monitor-only
now):** every roller term is N·m. `gamma` is an angular velocity [rad/s]; `dVpx/dg = ∂Vpx/∂gamma` is a
PARTIAL VELOCITY with units of LENGTH [m] (∂[m/s]/∂[rad/s]), NOT a velocity — the force→torque lever,
analogous to `R` in the wheel balance. So `p2*gamma` (N·m) + `Fpar*dVpx/dg` (N·[m]) are consistent;
the "force×velocity=power" reading is the trap. We drop it from training for being ill-conditioned
(non-unique γ kinematics), not for units.

Shared force→body assembly (both cells need it):
```
Fx_i = Fpar_i*cosδ_i - Fperp_i*sinδ_i           (roller -> body, per wheel)
Fy_i = Fpar_i*sinδ_i + Fperp_i*cosδ_i
RHS0 = ΣFx_i + ms*ψ̇*Vy + m*aX*ψ̇²                (body generalized force, x)
RHS1 = ΣFy_i - ms*ψ̇*Vx + m*aY*ψ̇²                (y)
RHS2 = Σ(px_i*Fy_i - py_i*Fx_i) - m*ψ̇*(aX*Vx + aY*Vy)   (yaw)
dVx,dVy,dψ̇ = M_inv @ [RHS0,RHS1,RHS2]            (M = body mass matrix; M_inv 3x3)
dw_i = (Msat_i - R*Fpar_i - p1*w_i) / J_wheel    (wheel; reaction = R*Fpar — A2 convention, VERIFY §3)
```
Per-wheel spin moment `Mz_i` DROPPED (chi-identifiability decision). `Vx,Vy,ψ̇,w` are the MEASURED
state at t; the forces come from the PREDICTED states.

**`residual` cell — YOU IMPLEMENT (force/accel residual).** Body: `M·dv_meas − [RHS0,RHS1,RHS2]`
per channel (expand `M·dv` so no inverse needed: `ms·dVx − m·aY·dψ̇ ; ms·dVy + m·aX·dψ̇ ; −m·aY·dVx +
m·aX·dVy + Is·dψ̇`), normalised r0,r1 /BODY_F_SCALE, r2 /BODY_M_SCALE. Wheel: the existing
`wheel_residual` /WHEEL_SCALE. (`dv_meas` = measured one-step velocity derivative.)

**`integrated` cell — YOU IMPLEMENT (Heun, held force, O(dt²)).** From the MEASURED state at t,
`s_t = [Vx,Vy,ψ̇,w₁..₄]`, with the predicted forces HELD across dt:
```
f1 = ne_rhs(s_t,   F_pred)          # = (dVx,dVy,dψ̇, dw1..4) above
s* = s_t + dt*f1                    # Euler predictor
f2 = ne_rhs(s*,    F_pred)          # corrector, SAME (held) force; recompute Coriolis/dw at s*
s_pred = s_t + 0.5*dt*(f1 + f2)     # Heun next-state prediction
err = s_pred - s_{t+1,measured}     # per output
```
Then divide each `err` by that output's p95 (`PRED_P95`: Vx 1.92, Vy 0.673, ψ̇ 2.408, w 39.089 for all
4 wheels) BEFORE squaring, and sum. dt = DECIM/SIM_HZ = 1/500 s. (Note: `f2` recomputes the body
Coriolis terms and `dw` at the predicted `s*` but with the SAME held `F_pred`; the forces depend on the
observer states, not on `s_t`, so they don't change between f1 and f2.)

**Reference:** body RHS + mass matrix + the Heun step are a 1:1 match of
`Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/physics.py::ne_rhs` (188-210) + `forward_integrate`
(219-241) + `RobotParams` (82-114). Read those; copy EXACT constants AND code-verify vs `base.toml` /
`run_one.jl`. **Wheel-reaction caveat:** A1's `ne_rhs` uses `R*Fx_body`, A2's `wheel_residual` uses
`R*Fpar`. Use `R*Fpar` (A2) and let §3 PROVE it (integrated `w_i` prediction must match measured).

---

## 2. Code changes (surgical; file by file)

### 2a. `observer_v1_py/mecanum_observer/config.py` — add body-EOM constants
Near the existing plant constants (M_PLATFORM, M_WHEEL, AX, AY, PX, PY, COS_DELTA, SIN_DELTA),
add — and **code-verify each value** against `base.toml` + A1 `RobotParams.finalize`:
```python
MS = M_PLATFORM + 4.0 * M_WHEEL        # 35.6 kg  (sprung+wheels; verify)
IS = 4.42                              # body yaw inertia (verify vs base.toml / run_one.jl)

# --- Physics-term non-dimensionalisation (PHYSICAL scales; diagonal, applied to the residual
# BEFORE squaring). Control-torque + per-wheel-weight scales -> each residual channel is a
# dimensionless fraction of an actuator-/weight-level quantity, so no channel dominates by scale. ---
MAX_TORQUE   = 10.0                    # base.toml actuator cap [N·m] (already in config)
WHEEL_SCALE  = MAX_TORQUE             # per-wheel torque residual / control-torque max
BODY_F_SCALE = MS * 9.81 / 4.0        # 87.3 N  (= m_s*g/4, per-wheel static weight) -> body x,y
BODY_M_SCALE = MAX_TORQUE             # body yaw (torque) residual / control-torque max
ROLLER_SCALE = 0.5                    # MONITOR-ONLY scale for the (dropped) roller term [N·m]

# Body 3x3 mass matrix + its inverse (the INTEGRATED cell needs M_inv to get dv from RHS).
# M = [[MS,0,-m*AY],[0,MS,m*AX],[-m*AY,m*AX,IS]] ; build M_inv once with np.linalg.inv and store.
import numpy as _np
M_BODY = _np.array([[MS, 0.0, -M_PLATFORM*AY],
                    [0.0, MS,  M_PLATFORM*AX],
                    [-M_PLATFORM*AY, M_PLATFORM*AX, IS]], dtype=_np.float64)
M_BODY_INV = _np.linalg.inv(M_BODY)   # 3x3; physics.py builds a torch copy on device

# INTEGRATED-cell output p95 scales (from variable_scaler_percentiles.csv) — per-variable, frozen.
PRED_P95 = dict(Vx=1.919951957464218, Vy=0.6734363287687299,
                psi_dot=2.4079599380493164, w=39.088775253295864)
```
(Verify the four p95 values against the scaler CSV rows `Vx,Vy,psi_dot,w`.)
(Old `ROLLER_TAU` was 0.5 and old wheel scale was `MAX_TORQUE` — same numbers, renamed. No per-term
weights: the physical scales ARE the balancing. If, after looking at the per-component logs, one term
still dominates the GRADIENT, add a single `W_BODY`/`W_WHEEL` knob then — not by default.)
Add to `ObserverConfig`:
```python
physics_variant: str = "integrated"    # {"residual","integrated"}; used only if physics_loss (roller is monitor-only)
warm_from: str = ""                    # weights-only warm-start checkpoint (refinement mode)
```
Also add a grounding-skipped schedule for the warm-start refinement (the supervised checkpoint IS
the grounding result), plus a PURE-PHYSICS tail:
```python
REFINE_SCHEDULE = PHASE_SCHEDULE[1:]   # drop "grounding"; [phys_rampup, overlap, grnd_rampdown, physics]
PURE_PHYSICS_EPOCHS = 10               # appended tail with w_sup=0 (NO supervised floor)
PURE_PHYSICS_LR = 0.10                 # lr_scale for the pure-physics tail
```
NOTE (deviation, intentional): the pure-physics tail sets `w_sup=0`, which breaks the `W_SUP_MIN=0.1`
floor that decision #4 mandates ("A2 never pure physics — drifts the non-unique zx/zy"). That decision
was established for the INTERNAL torque-balance physics only. The point of this tail — combined with
per-epoch checkpoints (§2e) — is to MEASURE, per epoch, whether the new MEASURABLE body residual
(the measurable wheel+body physics) sustains reconstruction without supervision. Decision #4's degradation was for the INTERNAL
(roller) balance physics, which we've now dropped — both trained terms here (wheel, body) are
measurable-grounded, so the open question is whether the pure tail HOLDS (good — supports physics-only
deployment) or still drifts the non-unique `zx`/`zy`. It is a characterization experiment, not a
deployment recipe.

### 2b. `observer_v1_py/mecanum_observer/physics.py` — add `body_residual`
Append (keep it backend-agnostic; RETURN A TUPLE to avoid the torch `dim=` vs numpy `axis=`
`stack` mismatch):
```python
def body_residual(xp, gamma, zx, zy, zs, mu, chi,
                  psi_dot, Vpx0, Vpy0, cti, sti,
                  Vx, Vy, dVx, dVy, dpsi_dot):
    """Body Newton-Euler residual, generalized-force form (physical units).
    gamma/zx/zy/cti/sti/Vpx0/Vpy0: [...,4]; mu/chi/Vx/Vy/psi_dot/dV*: [...].
    Returns (r0, r1, r2), each [...]  (normalised, dimensionless)."""
    Vpx, Vpy, w_z, *_ = contact_from_gamma(gamma, psi_dot[..., None] if False else psi_dot,
                                           Vpx0, Vpy0, cti, sti)
    N = _as(gamma, C.N_PER_ROLLER)
    Fpar, Fperp, _ = lugre_forces(xp, mu[..., None], N, chi[..., None],
                                  w_z, Vpx, Vpy, zx, zy, zs)
    cd = _as(gamma, C.COS_DELTA); sd = _as(gamma, C.SIN_DELTA)
    Fx = Fpar * cd - Fperp * sd
    Fy = Fpar * sd + Fperp * cd
    px = _as(gamma, C.PX); py = _as(gamma, C.PY)
    ms, m, aX, aY = C.MS, C.M_PLATFORM, C.AX, C.AY
    RHS0 = Fx.sum(-1) + ms * psi_dot * Vy + m * aX * psi_dot * psi_dot
    RHS1 = Fy.sum(-1) - ms * psi_dot * Vx + m * aY * psi_dot * psi_dot
    RHS2 = (px * Fy - py * Fx).sum(-1) - m * psi_dot * (aX * Vx + aY * Vy)
    Mdv0 = ms * dVx - m * aY * dpsi_dot
    Mdv1 = ms * dVy + m * aX * dpsi_dot
    Mdv2 = -m * aY * dVx + m * aX * dVy + C.IS * dpsi_dot
    return ((Mdv0 - RHS0) / C.BODY_F_SCALE,
            (Mdv1 - RHS1) / C.BODY_F_SCALE,
            (Mdv2 - RHS2) / C.BODY_M_SCALE)
```
NOTE: `contact_from_gamma` already does `psi_dot[..., None]` internally for w_z — pass plain
`psi_dot` ([...]) exactly as `roller_residual` does. (Remove the `if False` guard above; it is
only there to flag: pass psi_dot the SAME way roller_residual/wheel_residual do.)

Also add the INTEGRATED-cell helpers (`ne_rhs` + held-force Heun). Forces depend on the PREDICTED
states (gamma/zx/zy + measured cti/sti/Vpx0/Vpy0), NOT on the body state, so they're computed ONCE and
held across the Heun step:
```python
def _body_wheel_rates(xp, Fpar, Fperp, Vx, Vy, psi_dot, w, Msat, Minv):
    """Body+wheel accelerations from HELD roller forces and a (possibly mid-step) body state.
    Fpar/Fperp/w/Msat: [...,4]; Vx,Vy,psi_dot: [...]. Returns (dVx,dVy,dpd,[...4] dw)."""
    cd = _as(Fpar, C.COS_DELTA); sd = _as(Fpar, C.SIN_DELTA)
    Fx = Fpar * cd - Fperp * sd; Fy = Fpar * sd + Fperp * cd
    px = _as(Fpar, C.PX); py = _as(Fpar, C.PY)
    ms, m, aX, aY = C.MS, C.M_PLATFORM, C.AX, C.AY
    RHS0 = Fx.sum(-1) + ms * psi_dot * Vy + m * aX * psi_dot * psi_dot
    RHS1 = Fy.sum(-1) - ms * psi_dot * Vx + m * aY * psi_dot * psi_dot
    RHS2 = (px * Fy - py * Fx).sum(-1) - m * psi_dot * (aX * Vx + aY * Vy)
    dVx = Minv[0,0]*RHS0 + Minv[0,1]*RHS1 + Minv[0,2]*RHS2
    dVy = Minv[1,0]*RHS0 + Minv[1,1]*RHS1 + Minv[1,2]*RHS2
    dpd = Minv[2,0]*RHS0 + Minv[2,1]*RHS1 + Minv[2,2]*RHS2
    dw  = (Msat - C.R * Fpar - C.P1 * w) / C.J_WHEEL          # A2 convention: R*Fpar (VERIFY §3)
    return dVx, dVy, dpd, dw

def integrated_step(xp, gamma, zx, zy, zs, mu, chi, psi_dot, Vpx0, Vpy0, cti, sti,
                    Vx, Vy, w, Msat, Minv, dt):
    """Heun (held force, O(dt^2)) one-step prediction of [Vx,Vy,psi_dot,w1..4] at t+1.
    Returns (Vx_n, Vy_n, pd_n, w_n[...,4]) — PREDICTED next state (physical)."""
    Vpx, Vpy, w_z, *_ = contact_from_gamma(gamma, psi_dot, Vpx0, Vpy0, cti, sti)
    N = _as(gamma, C.N_PER_ROLLER)
    Fpar, Fperp, _ = lugre_forces(xp, mu[...,None], N, chi[...,None], w_z, Vpx, Vpy, zx, zy, zs)
    # f1 at s_t
    dVx1,dVy1,dpd1,dw1 = _body_wheel_rates(xp, Fpar, Fperp, Vx, Vy, psi_dot, w, Msat, Minv)
    # Euler predictor s* (forces HELD)
    Vxs=Vx+dt*dVx1; Vys=Vy+dt*dVy1; pds=psi_dot+dt*dpd1; ws=w+dt*dw1
    dVx2,dVy2,dpd2,dw2 = _body_wheel_rates(xp, Fpar, Fperp, Vxs, Vys, pds, ws, Msat, Minv)
    Vx_n = Vx + 0.5*dt*(dVx1+dVx2); Vy_n = Vy + 0.5*dt*(dVy1+dVy2)
    pd_n = psi_dot + 0.5*dt*(dpd1+dpd2); w_n = w + 0.5*dt*(dw1+dw2)
    return Vx_n, Vy_n, pd_n, w_n
```
`Minv` is a torch tensor copy of `C.M_BODY_INV` built once on device. `w_z` uses the contact kinematics
at the MEASURED psi_dot (held with the force). Note `contact_from_gamma` uses the measured psi_dot for
`w_z`; that's fine (forces held).

### 2c. `observer_v1_py/mecanum_observer/losses.py` — variant dispatch (residual vs integrated), per-component log
`physics_loss(pred_phys, phys, variant="integrated", Minv=None)`, variant ∈ {`residual`,`integrated`}.
Each channel is scaled (diagonal ND) BEFORE squaring; roller is computed EVERY call for MONITOR only
(never in `loss`). Log EACH component separately.
```python
gamma = pred_phys[:, :, 0]; zx = pred_phys[:, :, 1]; zy = pred_phys[:, :, 2]
zs = torch.zeros_like(gamma)
loss = pred_phys.new_zeros(()); log = {}

if variant == "residual":
    # wheel (4, /WHEEL_SCALE)
    r_wheel = P.wheel_residual(torch, gamma, zx, zy, zs, phys["mu"], phys["chi"],
                               phys["psi_dot"], phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"],
                               phys["Msat"], phys["w"], phys["w_dot"]) / C.WHEEL_SCALE      # [B,4]
    sw = (r_wheel**2).mean(0); loss = loss + sw.sum()
    for i in range(4): log[f"phys_wheel_w{i+1}"] = float(sw[i].detach())
    # body (3, x/y /BODY_F_SCALE, yaw /BODY_M_SCALE inside body_residual)
    r0, r1, r2 = P.body_residual(torch, gamma, zx, zy, zs, phys["mu"], phys["chi"],
        phys["psi_dot"], phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"],
        phys["Vx"], phys["Vy"], phys["dVx"], phys["dVy"], phys["dpsi_dot"])
    lx, ly, lyaw = (r0**2).mean(), (r1**2).mean(), (r2**2).mean()
    loss = loss + lx + ly + lyaw
    log.update(phys_body_x=float(lx.detach()), phys_body_y=float(ly.detach()),
               phys_body_yaw=float(lyaw.detach()))

elif variant == "integrated":
    Vx_n, Vy_n, pd_n, w_n = P.integrated_step(
        torch, gamma, zx, zy, zs, phys["mu"], phys["chi"], phys["psi_dot"],
        phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"],
        phys["Vx"], phys["Vy"], phys["w"], phys["Msat"], Minv, C.T_S)     # T_S = 1/500
    eVx  = (Vx_n - phys["Vx_next"]) / C.PRED_P95["Vx"]
    eVy  = (Vy_n - phys["Vy_next"]) / C.PRED_P95["Vy"]
    ePd  = (pd_n - phys["psi_dot_next"]) / C.PRED_P95["psi_dot"]
    eW   = (w_n  - phys["w_next"]) / C.PRED_P95["w"]                       # [B,4]
    lVx, lVy, lPd = (eVx**2).mean(), (eVy**2).mean(), (ePd**2).mean()
    sw = (eW**2).mean(0)                                                   # [4] per-wheel
    loss = loss + lVx + lVy + lPd + sw.sum()
    log.update(phys_int_Vx=float(lVx.detach()), phys_int_Vy=float(lVy.detach()),
               phys_int_psidot=float(lPd.detach()))
    for i in range(4): log[f"phys_int_w{i+1}"] = float(sw[i].detach())

# ROLLER — MONITOR ONLY (never in loss), per wheel:
r_roll = P.roller_residual(torch, gamma, zx, zy, zs, phys["mu"], phys["chi"],
                           phys["psi_dot"], phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"]) / C.ROLLER_SCALE
sr = (r_roll**2).mean(0)
for i in range(4): log[f"phys_roller_w{i+1}_MON"] = float(sr[i].detach())
return loss, log
```
Pass `Minv` (torch copy of `C.M_BODY_INV` on device, built once in `train()`) from the caller. The
per-component logs (`phys_wheel_w*`/`phys_body_*` for residual; `phys_int_Vx/Vy/psidot/w*` for
integrated; `phys_roller_w*_MON` always) are the diagnostics — surface in the print + `metrics.json`.

### 2d. `observer_v1_py/mecanum_observer/data.py` — emit the new phys keys
The physics cells need `ends-1` (residual: w_dot, dV) AND `ends+1` (integrated: next-state target),
so when `cfg.physics_loss` RESTRICT the window-end range to leave both valid:
```python
if cfg.physics_loss:
    ends = np.arange(W - 1, T - 1, st)     # drop the final sample so ends+1 < T (and ends-1 >= 0)
    starts = ends - (W - 1); idx = starts[:, None] + np.arange(W)[None, :]
    # rebuild Gw/Pw/Yt/aux on this trimmed `ends` (or compute ends once up top under this guard)
```
Then in the `if cfg.physics_loss:` block (~line 503) add (reuse `ends`, `dt`, `a["G"]`=`[Vx,Vy,psi_dot]`,
`a["P"][...,1]`=`w`):
```python
dG = (a["G"][ends] - a["G"][ends - 1]) / dt          # [M,3] backward diff (residual cell)
out.update(
    ph_Vx=a["G"][ends, 0].astype(np.float32),
    ph_Vy=a["G"][ends, 1].astype(np.float32),
    ph_dVx=dG[:, 0].astype(np.float32),
    ph_dVy=dG[:, 1].astype(np.float32),
    ph_dpsi_dot=dG[:, 2].astype(np.float32),
    # --- INTEGRATED cell: next-step MEASURED targets (t+1) ---
    ph_Vx_next=a["G"][ends + 1, 0].astype(np.float32),
    ph_Vy_next=a["G"][ends + 1, 1].astype(np.float32),
    ph_psi_dot_next=a["G"][ends + 1, 2].astype(np.float32),
    ph_w_next=a["P"][ends + 1, :, 1].astype(np.float32),   # [M,4]
)
```
(`ph_Vx/Vy`, `ph_Msat`, `ph_w` at t are already emitted; the integrated step uses those + the `_next`
targets. The trimmed `ends` keeps `ends-1>=0` and `ends+1<=T-1` both valid.)

### 2e. `observer_v1_py/mecanum_observer/training.py` — plumb keys, variant, warm-start
- Extend `_PHYS_KEYS` (line ~40) with: `"ph_Vx","ph_Vy","ph_dVx","ph_dVy","ph_dpsi_dot",
  "ph_Vx_next","ph_Vy_next","ph_psi_dot_next","ph_w_next"`.
- Extend `_phys_batch(batch, device)` to add (`.to(device)`): `Vx, Vy, dVx, dVy, dpsi_dot,
  Vx_next, Vy_next, psi_dot_next, w_next`, keyed as the loss expects (`phys["Vx_next"]`, etc.).
- Build `Minv = torch.tensor(C.M_BODY_INV, dtype=torch.float32, device=device)` once in `train()`.
- In the train loop where `physics_loss(...)` is called (line ~250), pass `variant=cfg.physics_variant,
  Minv=Minv`. Note `physics_loss` now returns a RICH log dict — `residual` cell: `phys_wheel_w1..4`,
  `phys_body_{x,y,yaw}`; `integrated` cell: `phys_int_Vx/Vy/psidot`, `phys_int_w1..4`; ALWAYS:
  `phys_roller_w1..4_MON`. Accumulate ALL of them across batches (epoch means), print a compact
  summary each epoch (residual cell: wheel-sum + body x/y/yaw; integrated cell: int Vx/Vy/ψ̇ + w-sum;
  both: roller-sum-MON), and write the per-component
  epoch means into `metrics.json` (and into each `epoch_ckpts/ep{NNN}.pt` if cheap) so the diagnostics
  survive for the ablation. The roller `_MON` terms are diagnostics only — never enter `loss`.
  IMPORTANT: physics terms require the de-normalized states; the existing loop already de-normalizes
  (`pred_phys = pred * y_std + y_mean`) before calling `physics_loss` — keep that. Compute `physics_loss`
  every step that `w_phys>0` (so the `_MON` roller log is populated through the physics-active phases);
  in pure supervised phases (`w_phys==0`) you may skip it (no monitor needed there).
- **Warm-start (weights-only) + refine schedule + pure-physics tail.** In `_phase_plan`, when
  `cfg.warm_from` is set use `C.REFINE_SCHEDULE` (skips grounding) instead of `C.PHASE_SCHEDULE`,
  scaled to `phase_total_epochs` as usual, THEN append the fixed pure-physics tail (NOT scaled):
  ```python
  for _ in range(C.PURE_PHYSICS_EPOCHS):
      plan.append(dict(phase="pure_physics", lr_scale=C.PURE_PHYSICS_LR, w_sup=0.0, w_phys=1.0))
  ```
  (So with `--phase-epochs 60` the plan is 60 refine + 10 pure = 70 epochs. The tail needs
  `physics_loss=True` — it is, for all physics cells — else `w_phys` is forced 0.)
  In `train()`, BEFORE the normal resume block, if `cfg.warm_from` and the run's own `checkpoint.pt`
  does NOT yet exist, load ONLY the weights and start fresh:
  ```python
  if cfg.warm_from and not ckpt.exists():
      stt = torch.load(cfg.warm_from, map_location=device, weights_only=False)
      model.load_state_dict(stt["model"])          # weights ONLY — fresh AdamW, start_epoch stays 0
      print(f"[train] warm-started weights from {cfg.warm_from} (refine schedule, epoch 0)")
  ```
  Do NOT load the optimizer or epoch from `warm_from`. The existing run-dir resume (mid-run restarts)
  is unchanged and still takes precedence once this run has written its own checkpoint.
- **Save EVERY epoch's checkpoint (for the final ablation).** Keep the existing `torch.save(... ckpt)`
  (latest, for resume), and ALSO snapshot per epoch (weights+epoch+cfg only — no optimizer, ~small):
  ```python
  snap_dir = run_dir / "epoch_ckpts"; snap_dir.mkdir(exist_ok=True)
  torch.save(dict(model=model.state_dict(), epoch=ge, phase=ph["phase"],
                  w_sup=ph["w_sup"], w_phys=ph["w_phys"], cfg=asdict(cfg)),
             snap_dir / f"ep{ge:03d}.pt")
  ```
  `norm.npz` / `split.json` in the run dir are shared across all epoch snapshots (the eval driver reuses
  them). Gitignore `epoch_ckpts/` — these are for the LOCAL ablation; only the final `checkpoint.pt`
  + `metrics.json` get committed. (~70 snapshots × ~50 KB × 4 runs ≈ 14 MB local.)

### 2f. `observer_v1_py/train_observer.py` + `observer_v1_py/launch_parallel.py` — CLI
- `train_observer.py`: add `--physics-loss` (store_true), `--physics-variant`
  (choices `residual,integrated`, default `integrated`), and `--warm-from` (str, default ""). Map to
  `physics_loss`, `physics_variant`, `warm_from` in the kwargs (same precedence block as the other
  CLI args). Update the `ObserverConfig.physics_variant` default to `"both"` and its choices doc.
- `launch_parallel.py`: add `--physics-loss` / `--physics-variant` passthrough. For `--warm-from`,
  the checkpoint differs PER FOLD (S1 vs S2), so add a `--warm-from-map` that resolves the right
  checkpoint per regime — simplest: add a CLI like `--warm-from-template` and substitute the regime,
  OR (preferred, least code) run ONE launcher invocation per fold with an explicit `--warm-from`
  pointing at that fold's supervised checkpoint (see §4).

---

## 3. MANDATORY verification gate (before any training run)

Write `_tmp/verify_physics.py` (torch-free python). For 5–10 random whitelisted Arrow files:
1. Read GROUND-TRUTH `gamma,zx,zy` (and `Vpx0,Vpy0,sin_tt,cos_tt,Msat,w,Vx,Vy,psi_dot,wz,mu,chi`)
   at 500 Hz (DECIM=4), exactly as `data.read_arrays` / `make_windows` do, INCLUDING the next sample
   `[Vx,Vy,psi_dot,w]_{t+1}`.
2. **Residual form:** compute `body_residual` + `wheel_residual` with numpy (`xp=np`) on the
   GROUND-TRUTH states and measured `dV/dt`/`ẇ` (backward diff). Must be ~0.
3. **Integrated form (the new, critical check):** run `integrated_step` (numpy) with GROUND-TRUTH
   states → predicted `[Vx,Vy,ψ̇,w]_{t+1}`; compare to the MEASURED next sample. Report median/95p of
   `|pred − meas|` per output (UNSCALED, physical), and as a fraction of each `PRED_P95`. Expect
   small (O(dt²) Heun + decimation + held-force), i.e. a few %% of p95 — NOT O(p95).
4. **WHEEL-REACTION CONVENTION (must resolve):** run step 3 BOTH ways — `dw=(Msat−R*Fpar−p1*w)/Jw`
   (A2) and `dw=(Msat−R*Fx_body−p1*w)/Jw` (A1). Whichever makes the integrated `w_{t+1}` match measured
   is correct; USE it in `_body_wheel_rates` and report which won. (Expect A2/`Fpar`.)
5. Sanity-check the monitor term: `roller_residual` ~0 on ground-truth states (unchanged).
6. **SCALE SANITY CHECK (optional but do it once).** All ND scales are PHYSICAL/p95 constants — nothing
   to fit. For the residual cell: `p95(|Msat|)` vs 10 ; `p95(|ΣFx_body|)` vs 87.3 ; `p95(|Σ(px·Fy−py·Fx)|)`
   vs 10. For the integrated cell confirm `PRED_P95` matches the scaler CSV rows. If a scale is off by
   >~5× from the data, flag it — do NOT change a physical constant without asking the planner.

Do NOT proceed to training until the gate passes (steps 1-5 — especially the integrated-form step 3 and
the wheel-reaction resolution step 4). Put the residual stats, the integrated pred-vs-measured stats,
and which wheel-reaction convention won in the hand-back.

---

## 4. The matrix run — WARM-START refinement (4 runs)

**Warm-start each physics cell from that fold's existing supervised b1024 w32 checkpoint** (weights
only, via `--warm-from`). The `none` (baseline) cell is NOT trained — it IS the warm-start origin
(the existing supervised runs, already evaluated); reuse it as the baseline row.

Origin checkpoints (must already exist; confirm before launching):
```
S1: observer_v1_py/runs/S1_train_w32_non_phys_max_norm_b1024/checkpoint.pt
S2: observer_v1_py/runs/S2_train_w32_non_phys_max_norm_b1024/checkpoint.pt
```

Recipe (all physics cells): `--warm-from <fold ckpt>` (⇒ grounding-skipped `REFINE_SCHEDULE`
+ pure-physics tail), `--per-run-batch 1024 --phase-epochs 60`, `--norm max --windows 32`,
`--cache-dir C:/Users/vishv/mecanum_cache_decim`, `--physics-loss`. **Full phase breakdown
(60 refine + 10 pure = 70 epochs):**

| phase | epochs (cum.) | LR× | w_sup | w_phys |
|-------|--------|-----|-------|--------|
| phys_rampup   | 12  (0–11)  | 0.25 | 1.00 | 0 → 1 |
| overlap       | 20  (12–31) | 0.50 | 1.00 | 1.00 |
| grnd_rampdown | 12  (32–43) | 0.25 | 1.00 → 0.10 | 1.00 |
| physics       | 16  (44–59) | 0.10 | 0.10 (W_SUP_MIN floor) | 1.00 |
| **pure_physics** | **10  (60–69)** | **0.10** | **0.00 (NO floor — see §2a deviation)** | **1.00** |

Cells (each run PER FOLD with the matching `--warm-from`). The roller term is monitored (logged) in
ALL physics cells but trained in NONE:

| cell | flags | tag-suffix |
|------|-------|-----------|
| none (baseline) | — reuse existing `S{1,2}_train_w32_non_phys_max_norm_b1024` — | (not run) |
| residual   | `--physics-loss --physics-variant residual --warm-from <fold>`   | `_phys_residual_b1024` |
| integrated | `--physics-loss --physics-variant integrated --warm-from <fold>` | `_phys_integrated_b1024` |

4 runs (2 cells × 2 folds). Because `--warm-from` differs per fold, run ONE launcher invocation per
(cell, fold) — example below = **integrated cell, S1**. On the laptop use `--max-parallel 1` per
invocation and run the two folds of a cell concurrently as two detached launchers (= 2 jobs, commit-
safe); on the 24/128 box raise parallelism per `nvidia-smi`. Start `keep_awake.py` + sleep lockdown first:
```
PYTHONUTF8=1 <torch-python> -u observer_v1_py/launch_parallel.py \
  --regimes S1_train --windows 32 --per-run-batch 1024 --phase-epochs 60 \
  --norm max --scaler-csv ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv \
  --cache-dir C:/Users/vishv/mecanum_cache_decim --dl-workers 2 --max-parallel 1 --heartbeat 120 \
  --physics-loss --physics-variant integrated \
  --warm-from observer_v1_py/runs/S1_train_w32_non_phys_max_norm_b1024/checkpoint.pt \
  --tag-suffix _phys_integrated_b1024 \
  --log-dir observer_v1_py/runs/_parallel_logs_phys_integrated_S1 \
  --csv observer_v1_py/runs/sweep_results_phys_integrated_S1.csv
```
Resume-safe (skip if `run_dir/metrics.json` exists). Sanity-check each log: the `phys_*` term(s)
should be a sensible fraction of `sup` once `wp>0`, and val loss must NOT blow up when physics ramps in
— if it does, `BODY_*_SCALE` is too small (term over-weighted); raise it in config and re-run that cell.
Expect the first epoch's val to sit near the supervised origin's (warm weights), then move as physics enters.

---

## 5. Evaluation (use the FIXED metric)

`evaluation.py` already normalises `omega_z_derived` by the frozen `config.WZ_P95` (do not revert).
Reuse the cross-subset driver pattern from `_tmp/cross_subset_batch_study.py` (it loops run dirs,
calls `evaluate_observer`, tags rows, prints the same-vs-cross gap per state, writes CSV+figs). Make a
copy `_tmp/cross_subset_phys_study.py` pointing at the **3 cells = the existing supervised b1024 runs
(the `none` baseline) + the 4 physics run dirs (residual/integrated × S1/S2)**, output to
`observer_v1_py/report_phys_matrix/`. Produce, per cell, the cross-subset RMSE + same-vs-cross gap for
`gamma, zx, zy, omega_z_derived` (mean over S1+S2 folds & 4 wheels), and the binned `zy`-vs-slip and
`omega_z`-vs-|wz| curves (the channels physics should move).

**Read it as:** the two physics cells are the SAME constraint under two scalings — does `integrated`
(state-p95) beat `residual` (force-scale) on cross-subset `zy`/`zx`/`omega_z` vs `none`, or are they
tied? A clear gap = loss-conditioning matters; a tie = the physics content, not the scaling, is the
lever. Also check the roller `_MON` diagnostic: does the (untrained) roller residual drop anyway when
the trained terms improve the states — is the dropped term consistent, or fighting them?

### 5b. Per-epoch checkpoint ablation (the "final ablation")
For each of the 4 physics runs, evaluate the saved `epoch_ckpts/ep{NNN}.pt` snapshots to trace
cross-subset RMSE per state ACROSS the curriculum — the point is to locate the best (early-stopping)
checkpoint and to characterize the **pure-physics tail (ep60–69)**: does reconstruction hold or drift
once `w_sup→0`? Driver `_tmp/ablate_epoch_ckpts.py`: for each run, build the model + load the run's
`norm.npz`/`split.json`, then for each chosen snapshot load `ep{NNN}.pt["model"]` and call the
per-split scorer (factor it out of `evaluate_observer`, or load the snapshot into `checkpoint.pt`-shape
and reuse it). To keep it tractable, evaluate **every epoch in the `physics`+`pure_physics` phases
(ep44–69, 26 each) plus the end-of-phase epochs (11,31,43)** — ~29 evals/run × 4 ≈ 2 h. Output a
per-state cross-subset-RMSE-vs-epoch table + line plot per cell (mark the phase boundaries and the
`w_sup→0` point at ep60). The headline result: the epoch (and variant) at which each state's
cross-subset RMSE is minimized, and whether the pure-physics tail improves or degrades each state.

---

## 6. Hand back

- Verification-gate residual stats (the acceptance numbers).
- Verification-gate stats AND which wheel-reaction convention won (step 4).
- The 3×(γ,zx,zy,ω_z) cross-subset + gap table (cells = none/residual/integrated), both folds.
- Per-component physics logs (residual: `phys_wheel_w*`,`phys_body_{x,y,yaw}`; integrated:
  `phys_int_{Vx,Vy,psidot,w*}`; both: `phys_roller_w*_MON`): which component carried the gradient,
  and whether the per-channel scaling kept them comparable (no single channel dominating).
- The per-epoch ablation (§5b): per-state cross-subset-RMSE-vs-epoch curve per cell, the best-epoch
  per state, and the **pure-physics-tail finding** — does the measurable physics sustain reconstruction
  without supervision (ep60–69), or drift? (Compare residual vs integrated on the tail.)
- One-paragraph verdict: residual vs integrated (does the scaling regime matter?); any
  cell where physics degraded a state (watch the non-unique `zy`/`zx` per decision #4).
- List of new/edited files and the run dirs + CSVs produced. Do NOT git-commit; leave that to the planner.

**Do not deviate from the residual equations or constants without code-verifying against base.toml /
run_one.jl and reporting the discrepancy first.**
