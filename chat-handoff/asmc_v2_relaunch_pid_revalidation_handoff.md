# ASMC v2 Re-launch + PID v2 Re-validation — Handoff

> **Date:** 2026-08-08/09 · **Session:** Kimi k3, Windows tree
> **Briefs:** `instructions/asmc-v2-tuning-launch.md` (executed), `instructions/pid-v2-imc-cascade-two-variants.md`
> **Runs:** `hybrid_ctrl_v2/runs_asmc_v2_relaunch/` (NEW), `hybrid_ctrl_v2/runs_pid_v2_relaunch/` (NEW)

---

## 1. What was done and why

1. **ASMC v2 5-seed clean tuning re-launch** per `asmc-v2-tuning-launch.md`. The prior
   run (`hybrid_ctrl_v2/runs_asmc_v2/`) is **VOID** (K exceeded its friction-circle
   ceiling by up to 11.7×; do not use as baseline). Pre-launch gates §5 verified first
   (`_tmp/asmc_v2_prelaunch_check.jl`): fractional smooth-bound knee identical on all
   axes (0.11920 at K=K_max), bit-identical reproducibility, `bus.K` init =
   0.95·K_floor with `start_at_floor=true` / ~0 with false, plus a 12-trajectory
   default-gains smoke (score 8.009, n_fail=0, 177.8 s/eval uncontended).
2. **PID archived scores proven stale, then re-tuned.** The launch brief §7 warned the
   PID v2 numbers predated the `vcmd_limits` restructure + acceleration-convention fix.
   Decisive check (`_tmp/pid_v2_staleness_check.jl`): archived FB seed-1 gains
   re-evaluate to **2.2184** on current code vs **0.9480** archived → stale, and the
   old gains got *worse* (they had co-adapted to the buggy gate). PID v2 FB and CT
   were then both re-tuned on current code, same protocol as archived (PID_SPACE_V2,
   ψ floor 0.01, train12 both phases, clean, p1-cap 250 / p2-cap 60).

## 2. Results (clean oracle, train12, current code, all scores = objective of
##    `stage_objective.jl`: tracking + 0.05·ce/24)

**ASMC v2** (`runs_asmc_v2_relaunch/seed{1..5}/asmc_v2_clean/`, 4-D space:
lam_x/y/psi_max + tau_ceiling; γ fixed derived 170/380/280, γ_ref=250):

| seed | score | lam_x | lam_y | lam_psi | tau_ceiling | stop |
|---|---|---|---|---|---|---|
| **2** | **3.690** | 3.04 | 1.21 | 10.60 | 0.194 | plateau |
| 1 | 4.012 | 2.51 | 1.21 | 11.53 | 0.368 | plateau |
| 5 | 4.195 | 2.46 | 1.13 | 7.99 | 0.204 | plateau |
| 3 | 4.862 | 0.30 | 1.98 | 19.91 (rail) | 99.996 (rail) | plateau |
| 4 | 5.251 | 1.65 | 9.89 | 1.09 | 0.316 | plateau |

**PID v2 FB** (`runs_pid_v2_relaunch/seed{1..5}/pid_v2_fb_clean/`) — NEW, replaces
stale 0.945–0.961:

| seed | 1 | 4 | 5 | 3 | 2 |
|---|---|---|---|---|---|
| score | **1.7234** | 1.7284 | 1.7321 | 1.7455 | 1.7590 |

All plateau stops, 2.1% spread. `lam_inner_psi` at the 0.01 floor on all 5 seeds.
Typical gains: lam_inner ≈ (0.14–0.16, 0.08–0.10, 0.011).

**PID v2 CT** (`runs_pid_v2_relaunch/seed{1..5}/pid_v2_ct_clean/`) — **reproduces the
archive digit-for-digit**: 0.5782 / 0.5786 / 0.5791 / 0.5791 / 0.6013, each seed's
gains matching its archived twin to 4 printed digits. **The archived CT numbers were
never stale** — the gate defect strangled CT's feedforward demands, but CT tuning had
already been re-done on effectively-fixed code; the staleness was FB-only.

**Corrected controller comparison (clean oracle, train12):**

```
PID-CT   0.578 – 0.601      (validated, unchanged)
PID-FB   1.723 – 1.759      (NEW — was 0.945–0.961 stale; fixed gate costs +83%)
ASMC v2  3.690 (best seed)  (2.1× PID-FB, 6.4× PID-CT)
```

## 3. Launch-brief §4 decision criteria — verdict

- **Convergence machinery**: 5/5 plateau stops (void run: 4/5 capped). Budget used
  85–151 of ~310 evals; total wall ~7.1 h for 5 parallel seeds.
- **4-D viable**: YES, with multimodality. Dominant basin = 3/5 seeds
  (3.69/4.01/4.20 = 13% spread, parameters within 1.9× on all dims, lam_y within 7%,
  tau_ceiling 0.19–0.37). Two exploration seeds found worse basins (railed corner;
  high-lam_y/low-lam_psi) — restart cost, not space-size problem.
- **No box widening triggered**: only seed 3 railed (lam_psi 19.91/20, tau_ceiling
  99.996/100, lam_x 0.305/0.3) and it converged WORST-on-merit → rails don't truncate
  the good optimum. tau_ceiling settled at 0.19–0.37, rejecting the sweep's
  higher-is-better trend; interior optimum near the original pinned 0.5.

## 4. Structural findings (carry into the paper carefully)

- ASMC's clean-oracle deficit is concentrated in `spiral_orbit_stress` (57.7) and
  `coupled_vomega_stress` (34.9) at default gains; the other 10 trajectories average
  ~0.26. Measured `|M_sw|/K` median = 1.0 on coupled_vomega → the ASMC operates
  bang-bang there; a relay limit-cycle floor exists that λ/tau_ceiling tuning cannot
  cross. Clean oracle is the ASMC's worst case by construction (no noise, no model
  error → adaptive-K buys nothing, switching floor costs accuracy).
- `lam_*_max` acts only BELOW the threshold v_max/lam (x: 0.21 m, y: 0.52 m, ψ:
  0.36 rad at seed-2 gains) — it is a terminal-precision knob; large-error reaching
  is governed by K adaptation (`tau_ceiling` level knob; γ is a pure rate multiplier,
  fixed).
- PID: the ψ-floor rail (0.01) binds on ALL 10 re-tuned seeds (FB+CT) on current
  code. A 0.01→0.005 floor probe has the 42% precedent (0.05→0.01) behind it —
  cheap, justified, not yet done.

## 5. Open items / next steps (priority order)

0. **ESKF-closed-loop eval DONE** (`hybrid_ctrl_v2/eval_controllers_eskf_v3.jl` + `.bat`;
   results `hybrid_ctrl_v2/runs_controller_eskf_v3/`): tuned ESKF V3 (seed-4 v4 config)
   in the loop, :realistic suite, 5 paired noise seeds (101–105), train12, all 180 runs ok.
   Score means (12 trajs × 5 seeds): **PID-CT 0.659, PID-FB 1.887, ASMC v2 8.43**.
   Per-9-benign-trajectories (excluding coupled_vomega_stress/ellipse_stress_tangent/
   spiral_orbit_stress): **CT 0.195, ASMC 0.595, FB 0.698 — ASMC beats FB under the ESKF**.
   The 3 stress-trajectory ASMC blowups are CONTROLLER-side, not estimator-side: on the
   worst realization (ellipse_tangent seed 104, score 217) the ESKF's in-loop errors
   stayed at the noise floor (est_rate 0.0019). Mechanism: clean-tuned eps (resolution
   floor 5e-4) vs ESKF xhat noise ~2-3 mm/s → tanh(s/eps) saturated by noise alone →
   switching fires spuriously → limit-cycle amplification on authority-edge trajectories.
   This is exactly the launch brief's "populate SurfaceNoise from the ESKF's measured
   error before the noisy stage" — the noisy-stage ASMC retune is now REQUIRED, with
   SurfaceNoise = (sig_vel≈0.0022, sig_rate≈0.004, sig_pos≈0.0032, sig_psi≈0.0065)
   measured in-loop this eval. ESKF in-loop errors matched replay floor on every seed
   (vel 0.0022, rate 0.0025–0.0042, pos 0.0032) — the estimator is deployable as-is.

1. **Held-out eval**: run ASMC seed-2 config + PID-FB seed-1 + PID-CT seed-1 on the
   `:test` tier (8 trajectories; `straightline` Arrow still missing — 8/9).
2. **§6 diagnostics on tuned ASMC** (log_K=true): max(K)/K_max_sched (invalidate if
   >1.02), |M_sw|/K median (is it still bang-bang at tuned gains?), chatter vs the
   0.8 V/ms slew ceiling, max_pos timing.
3. **Noisy stage**: populate `SurfaceNoise` from tuned ESKF V3 measured errors
   (vel 0.0009, pos 0.0032, rate 0.0025, heading 0.0013 — see
   `chat-handoff/eskf_v4_yawaccel_tuning_handoff.md`), NOT the oracle injection, then
   warm-started noisy re-tunes for all three controllers.
4. ψ-floor probe 0.01→0.005 for PID (both variants).
5. The eps-floor/chatter trade for ASMC remains deferred per the brief
   (eps_floor_xy=0.005, eps_floor_psi=0.02 reproduces the widened configuration,
   measured −33–45% chatter at no tracking cost).
6. Old VOID run `hybrid_ctrl_v2/runs_asmc_v2/` retained for reference only — never
   quote it.

## 6. Files

**2026-08-11 eps-floor retune (runs_asmc_v2_epsfloor/, all plateau stops):**
eps_floor_xy=0.02 / eps_floor_psi=0.08 frozen (new `--eps-floor-xy/--eps-floor-psi`
args on run_stage.jl; verified reaching the built controller). Finals:
seed1 **1.0680** lam=(3.85,1.59,20.00) tau_c=99.2 · seed2 1.0706 (3.55,1.70,20.00) 97.8 ·
seed3 1.0691 (3.61,1.72,20.00) 99.99 · seed4 1.1797 (3.51,0.69,16.11) 98.5 ·
seed5 3.6196 (1.58,2.20,1.01) 98.2.
vs floor-0 relaunch (3.69/4.01/4.20/4.86/5.25): **best 3.45× better; top-3 seeds
within 0.25% (multimodality in the good basin gone; one outlier basin remains)**.
`tau_ceiling` railed ~98-100 on 5/5, `lam_psi_max` at exactly 20 on 3/5 → box
truncates → WIDE-box follow-up (tau_ceiling [0.1,200], lam_psi [1,40]) launched via
new `--tau-ceiling-hi/--lam-psi-hi` args (`run_stage_asmc_v2_epsfloor_wide_5seed.bat`,
root `runs_asmc_v2_epsfloor_wide/`). Interpretation: with the layer reachable, the
switching term is a proportional corrector of stiffness K/eps and the sigma-leak
engages — high authority no longer self-sustains a limit cycle, so the optimum moved
to the high-K corner. Chatter NOT persisted in trials.json (only ce/tracking) — the
chatter + max(K)/K_max_base guard comes from the log_K diagnostics run on the
winner (pending, after the wide run). NOTE: ASMC 1.068 < PID-FB 1.723 — first time
ASMC beats FB on clean oracle; still 1.8× above PID-CT 0.579.

- New launcher bats: `hybrid_ctrl_v2/run_stage_asmc_v2_5seed.bat`,
  `hybrid_ctrl_v2/run_stage_pid_v2_relaunch.bat` (takes `fb`|`ct`).
- Verification scripts: `_tmp/asmc_v2_prelaunch_check.jl` (+log),
  `_tmp/pid_v2_staleness_check.jl` (+log).
- Results: `runs_asmc_v2_relaunch/`, `runs_pid_v2_relaunch/` — each seed dir has
  best_config.json / trials.json / trace.csv / checkpoint.json / phase_summary.json;
  per-seed stdout logs at `runs_*_relaunch*_seedN.log` in `hybrid_ctrl_v2/`.
- Unchanged/protected: `hybrid_ctrl/*`, `tune_controller.jl` — no edits this session.
