# PhD Research Plan — Physics-Informed Mamba Digital Twin for Mecanum-Wheeled Omnidirectional Robots

**Candidate:** Vishvesh Koranne  
**Guide:** Prof. Shital S. Chiddarwar  
**Institute:** VNIT Nagpur, Department of Mechanical Engineering  
**Target:** PhD Thesis + IMECE 2026 paper + 2 journal papers  
**Date:** 2026-07-17  

---

## AIM
**To develop a resilient, data-driven control architecture for 4WD Mecanum-Wheeled Omnidirectional Robots (FMWOR) for dynamic real-world conditions** by closing the per-wheel friction identifiability gap through a physics-informed Mamba-SSM digital twin.

---

## GOAL
**Design and deploy a Physics-Informed Mamba State-Space-Model Neural Network architecture for real-time FMWOR control, robust to nonlinear frictional interaction, unmodeled slipping, and mechanical wear of mecanum wheel rollers.**

---

## OBJECTIVES (4 Thesis Contributions)

| Obj | Description | Success Metric |
|-----|-------------|----------------|
| **O1** | **High-fidelity sim-to-real pipeline** — Julia 39-D LuGre-Adamov ODE → 274 GB dataset → MuJoCo cross-validation → custom 4WD platform sim | • 5,670 trajectories, 24 profiles × 3 μ × χ sweep<br>• MuJoCo state RMSE ≤ 5% vs Julia<br>• Custom sim calibrated to hardware ≤ 10% parameter error |
| **O2** | **Mamba-SSM PINN architecture** — selective SSM backbone + LuGre physics loss + per-wheel (μ, χ) heads for forward dynamics + inverse friction ID | • ≤ 1 ms inference on Jetson Orin (FP16)<br>• Forward dynamics RMSE ≤ 5% envelope<br>• μ RMSE ≤ 0.05, χ RMSE ≤ 0.001 (whitelisted) |
| **O4** | **Hardware validation** — zero-shot (floor μ) → one-shot (payload/CoM) → online ID (roller wear) on custom 4WD | • Phase 1: pose RMSE ≤ 2× sim (μ ∈ [0.3, 0.8])<br>• Phase 2: pose RMSE ≤ 3× sim (±11% payload, ±20% CoM)<br>• Phase 3: μ(t) tracking R² ≥ 0.8 vs wear progression |
| **O3** | **Certifiability-by-design framework** — output-bounded PINN + reachability + runtime monitor for SIL 1–2 evidence | • Lipschitz-constrained Mamba blocks (spectral norm ≤ 1)<br>• Runtime monitor FP ≤ 1%<br>• Reachable-set over-approx ≤ 2× true envelope |

---

## TASK HIERARCHY (Non-Linear Time, with Risk Forks)

### O1: SIM-TO-REAL PIPELINE
```
O1
├── T1.1  Julia ODE refinement & dataset generation (COMPLETE)
│   ├── 39-D stiff ODE (TRBDF2, reltol 1e-8, 2 kHz)
│   ├── 24 excitation profiles, μ ∈ {0.3, 0.5, 0.8}, χ ∈ [0.001, 0.01]
│   └── Output: Arrow + JLD2 (5345/5670 whitelisted)
│
├── T1.2  MuJoCo cross-validation (youBot)  [PARALLEL with T2.1]
│   ├── Replicate geometry + LuGre in MJCF
│   ├── Pass/Fail: State RMSE ≤ 5% AND Slip RMSE ≤ 8%
│   │
│   ├── 🍴 FORK: MuJoCo–Julia mismatch >5% [Risk R2]
│   │   ├── F1.2a  Debug contact model (roller handoff, Hertz params)
│   │   ├── F1.2b  If unresolved → Use Julia data ONLY for PINN training
│   │   └── F1.2c  Restrict MuJoCo to custom platform sys-ID only
│   │
│   └── Deliverable: Cross-validation report with pass/fail decision
│
├── T1.3  Custom 4WD platform system identification  [DEPENDS ON HARDWARE]
│   ├── MuJoCo model → hardware parameter fitting (mass, inertia, friction)
│   ├── Target: ≤ 10% parameter error vs physical measurements
│   │
│   ├── 🍴 FORK: Custom platform hardware delays [Risk R3]
│   │   ├── F1.3a  Parallel sim work continues (Julia + MuJoCo)
│   │   ├── F1.3b  Hardware-in-loop with recorded Julia data
│   │   └── F1.3c  Defer hardware-dependent tasks; advance O2/O4
│   │
│   └── Deliverable: Calibrated MuJoCo model for custom platform
```

### O2: MAMBA-PINN ARCHITECTURE  [STARTS AFTER T1.1, PARALLEL WITH T1.2/T1.3]
```
O2
├── T2.1  Architecture design & implementation  [CAN START IMMEDIATELY]
│   ├── Encoder: 2-layer BiGRU (hidden=128) → latent z (roller rates)
│   ├── Backbone: Mamba-SSM (4 layers, d_model=256, d_state=16, expand=2)
│   ├── Heads: Physics (μ, χ × 4) + Dynamics (39-D ẋ)
│   └── Export: ONNX → TensorRT for Jetson Orin
│
├── T2.2  Physics-informed training (3 stages)  [NEEDS T1.2 PASS/FAIL DECISION]
│   ├── Stage 1 Grounding: λ_phys=0, supervised on 5345 whitelisted
│   ├── Stage 2 Physics: λ_phys=1.0, physics loss on full 5670 + pseudo-labels
│   ├── Stage 3 Identification: λ_id=5.0, force-reconstruction + bounds
│   ├── Loss: L_data + λ_phys L_phys + λ_id L_id + λ_bounds L_bounds
│   └── Targets: dyn RMSE ≤ 5%, μ RMSE ≤ 0.05, χ RMSE ≤ 0.001
│
├── T2.3  Inference optimization & profiling  [NEEDS T2.2 MODEL]
│   ├── Target: ≤ 1 ms on Orin NX (FP16)
│   │
│   ├── 🍴 FORK: Mamba inference >1 ms [Risk R1]
│   │   ├── F2.3a  Profile early (TensorRT, FP16, batch=1)
│   │   ├── F2.3b  Knowledge distillation → 2-layer Mamba (d_model=128)
│   │   ├── F2.3c  Distillation loss: L_KD = MSE(student, teacher) + λ L_phys
│   │   └── F2.3c  Fallback target: ~0.35 ms, ~1.2M params
│   │
│   └── Deliverable: ONNX/TensorRT model + latency report
```

### O4: HARDWARE VALIDATION  [STARTS AFTER T2.3 MODEL READY + T1.3 CALIBRATED]
```
O4
├── T4.1  Phase 1 — Floor μ variation (zero-shot)
│   ├── Surfaces: epoxy (0.3), concrete (0.5), carpet (0.6), steel (0.3)
│   ├── Trajectories: octagon, coupled_vω, spiral_iso × 3 reps
│   ├── Metrics: pose RMSE (% traj length) ≤ 2× sim; μ RMSE ≤ 0.05
│   └── Control: MPC (horizon=20, 1 kHz) using twin + ASMC fallback
│
├── T4.2  Phase 2 — Payload/CoM shift (one-shot)
│   ├── Payload: +5.5% (1 kg), +11% (2 kg); CoM offset: ±13% h, ±20% l
│   ├── Trajectories: ellipse, figure8 × 2 reps
│   ├── Adaptation: 5 trajectories → fine-tune μ heads
│   ├── Metrics: pose RMSE ≤ 3× sim; μ drift ≤ 0.03 vs baseline
│
├── T4.3  Phase 3 — Roller wear (online ID)
│   ├── Wear stages: 0% → 25% → 50% → 75% → 100% (sandpaper abrasion)
│   ├── Trajectories: spin_creep, multisine50 × 2 reps (spin-rich)
│   ├── Primary: μ(t) tracking R² ≥ 0.8 vs measured progression
│   ├── Secondary: χ(t) correlation with μ(t) (optional, wear mechanism)
│   ├── Exploratory: Motor current FFT at 12× wheel freq → wear index
│   ├── Metrics: pose RMSE growth ≤ 2× per stage
│   │
│   ├── 🍴 FORK: Wear progression too slow [Risk R4]
│   │   ├── F4.3a  Accelerated wear rig: motorized abrasion, 24/7
│   │   ├── F4.3b  Simulated wear in MuJoCo (vary roller geometry, μ)
│   │   └── F4.3c  Pre/post caliper validation (12 rollers/wheel)
│   │
│   └── Deliverable: Hardware validation report + sim-to-real gap characterization
```

### O3: CERTIFIABILITY FRAMEWORK  [INFORMED BY O4 RESULTS]
```
O3
├── T3.1  Lipschitz-constrained Mamba blocks  [USES O4 MEASURED GAPS]
│   ├── Spectral normalization on Mamba projections (spectral norm ≤ 1)
│   ├── Compute Lipschitz constants per layer (power iteration)
│   ├── Calibrate κ in monitor using O4 Phase 1–2 disturbance statistics
│   └── Output bounding proof for 20-step horizon
│
├── T3.2  Reachability analysis  [VALIDATED ON O4 TRAJECTORIES]
│   ├── Zonotope propagation through PINN
│   ├── Target: reachable-set over-approximation ≤ 2× true envelope
│   ├── Validation: Monte Carlo on O4 hardware data vs analytical bounds
│   └── Adjust horizon/partitioning based on O4 chatter frequencies
│
├── T3.3  Runtime monitor design & validation  [DEPLOYED ON O4 HARDWARE]
│   ├── Monitor: ‖x_meas - x_pred‖ > κ·σ → safe stop trigger
│   ├── κ calibrated from O4 Phase 1–2 residual distributions
│   ├── False positive target: ≤ 1% (measured on O4 nominal runs)
│   ├── Fault injection: bit-flip, NaN, OOD input → detection rate
│   │
│   ├── 🍴 FORK: Certifiability evidence insufficient [Risk R5]
│   │   ├── F3.3a  Scope claim: "enables SIL 1–2 evidence for learned component"
│   │   ├── F3.3b  Explicitly exclude: full PL d, Category 3 hw, FSA
│   │   └── F3.3c  Produce: architecture diagram, bounds report, monitor FP rate
│   │
│   └── Deliverable: Certifiability evidence package (Lipschitz + reachability + monitor) validated on hardware
```

---

## TIMING INDEPENDENCE MAP

| Task | Can Start | Blocks | Parallel With |
|------|-----------|--------|---------------|
| T1.1 | Day 0 | T1.2, T2.1 | — |
| T1.2 | T1.1 done | T2.2 (data decision) | T2.1 |
| T2.1 | T1.1 done | T2.2 | T1.2, T1.3 |
| T1.3 | Hardware ready | T4.1 (calibrated sim) | T2.1, T2.2 |
| T2.2 | T1.2 decision + T2.1 | T2.3 | T1.3 |
| T2.3 | T2.2 done | T4.1 (model) | — |
| T4.1 | T2.3 + T1.3 | T4.2, T3.1 (gap data) | — |
| T4.2 | T4.1 | T4.3, T3.1/3.2 | — |
| T4.3 | T4.2 | T3.2/3.3 (validation) | — |
| T3.1 | T4.1 (initial gaps) | T3.2, T3.3 | T4.2, T4.3 |
| T3.2 | T3.1 + T4.2 data | T3.3 | T4.3 |
| T3.3 | T3.1 + T4.3 data | — | — |

**Critical path:** T1.1 → T1.2 → T2.2 → T2.3 → T4.1 → T4.2 → T4.3 → T3.3  
**Parallel tracks:** T2.1 runs during T1.2/T1.3; T3.1 starts early with T4.1 data and refines through T4.3

---

## RISK REGISTRY (Mapped to Task Forks)

| Risk ID | Risk | Likelihood | Impact | Primary Fork Location |
|---------|------|------------|--------|----------------------|
| **R1** | Mamba inference >1 ms on Orin | Medium | High | O2 → T2.3 🍴 |
| **R2** | MuJoCo–Julia mismatch >5% | Low | Medium | O1 → T1.2 🍴 |
| **R3** | Custom platform hardware delays | Medium | High | O1 → T1.3 🍴 |
| **R4** | Wear progression too slow | Medium | Medium | O4 → T4.3 🍴 |
| **R5** | Certifiability evidence insufficient | Low | Medium | O3 → T3.3 🍴 |

---

## OUT-OF-SCOPE (Explicit Boundaries)

| Item | Reason |
|------|--------|
| Full PL d / SIL 2 certification | Requires Category 3 hardware, safety PLC, FSA — platform-level, not learned-component |
| Hardware fault injection (IEC 61508-2) | Requires safety-rated test lab, fault injectors, months campaign — validates platform, not PINN architecture |
| Multi-robot coordination | Single-platform focus |
| End-to-end RL/IL policy | Model-based architecture (twin + MPC/SMC) |
| Battery/energy as primary objective | Byproduct of slip reduction |

---

## JUSTIFICATION: O4 BEFORE O3 ORDERING

| Argument | Detail |
|----------|--------|
| **Empirical grounding** | Certifiability artifacts (Lipschitz bounds, reachability over-approx, monitor FP rate) need *measured* sim-to-real gaps from O4 to be calibrated. Theoretical bounds without hardware validation are weak evidence. |
| **Thesis claim validation** | Claim: *"physics-informed structure enables SIL 1–2 evidence."* This is proven by showing the runtime monitor works on *real* hardware with real disturbances — not just in simulation. |
| **Design iteration** | O4 Phase 1–2 reveals actual OOD behavior, chatter, sensor noise. These inform monitor threshold κ, Lipschitz tightening, reachability horizon. Doing O3 first = designing blind. |
| **Risk reduction** | If O4 reveals fundamental sim-to-real gaps (e.g., unmodeled dynamics), O3 framework adapts. Reverse order risks building a framework for a simulator that doesn't match reality. |

---

## JUSTIFICATION: ARCHITECTURE (T2.1) BEFORE CUSTOM PLATFORM SYS-ID (T1.3)

- T2.1 depends only on Julia dataset (T1.1 complete) — independent of hardware
- T1.3 needs physical robot (fabrication, integration) — schedule risk
- Parallelizing T2.1 during T1.3 wait time keeps critical path moving

---

## JUSTIFICATION: HARDWARE FAULT INJECTION OUT OF SCOPE

| Requirement | What It Needs | Why Out of Scope |
|-------------|---------------|------------------|
| **IEC 61508-2 fault injection** | Safety-rated hardware platform, programmable fault injectors (clock glitch, power glitch, EMFI, pin-level), fault campaign documentation | Requires dedicated safety test lab, certified equipment, months of campaign time — not a research contribution |
| **ISO 13849-1 Category 3 validation** | Dual-channel architecture, cross-monitoring, diagnostic coverage ≥90% on *hardware* | Our contribution is the *learned component's* software evidence (Lipschitz, monitor); hardware architecture is a separate engineering effort |
| **Thesis scope** | We deliver: software evidence artifacts enabling SIL 1–2 *for the learned component* | Full hardware fault injection validates the *platform*, not the PINN architecture |

---

*Document version 2.0 — revised task hierarchy with O4 before O3, non-linear timing, risk forks, and explicit out-of-scope justifications.*