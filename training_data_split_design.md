# Training-data split design — excitation cube + χ k-fold

**Scope.** Authoritative record of the train/val/test split decisions for the
PINN digital-twin learning experiments. **Shared by both approaches** —
Approach 2 (this repo's `observer_v1_py`, the SSM/GRU state observer) and
Approach 1 (`train_GPU_PINN_v14_py`, the force-reconstruction forward-inverse
PINN). The splits are defined at the **data level** (which trajectories go where),
independent of model/controller/loss, so either approach can consume them.

Last updated: 2026-06-19.

---

## 1. Dataset facts that drive the design

- Active sweep: `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov`.
- **μ ∈ {0.3, 0.5, 0.8}** (~1780 whitelisted files each).
- **χ ∈ {0, 0.002, 0.005, 0.008}**, but **χ=0.005 is dominant** (~1390/μ);
  {0, 0.002, 0.008} are sparse satellites (~130/μ each).
- **Only 3 profiles carry χ variation** — `octagon`, `spin_creep`,
  `coupled_vomega`. The other five (`long_circle`, `ellipse`, `spiral_orbit`,
  `multisine_50/75percent_cap`) exist at **χ=0.005 only**.
- Within the χ-swept profiles, most combos exist *only* at χ=0.005; only a subset
  (the **matched χ-quads**) span all four χ: octagon 69, spin_creep 44,
  coupled_vomega 16 combos.
- Whitelist = `diagnostics_combined.csv`, rows where `combined_reco` does not
  start with `reject`.
- **multisine is held out of training entirely** — its controller chatter is
  M_sw-dominated, which risks teaching the observer controller signatures rather
  than plant physics. Reserved as a separate chatter-robustness eval.

Two consequences: (a) anything χ-dependent can only be studied on the 3 χ-swept
profiles; (b) the μ axis and the χ axis must be handled by *different* split
designs — hence the cube (single χ) and the χ k-fold below.

---

## 2. The excitation cube (basis for the S1/S2 partition)

Classify excitation by the **three body DOFs** `(Vx, Vy, ψ̇)`, each in one of
three states — **zero / const / accel** — giving a **3×3×3 = 27-cell cube**.
This is trajectory-*independent* and physics-grounded: the body DOFs are what
excite the wheel/roller/bristle states. "Cover all excitations" becomes a precise
checkable property (each plausible cell present in each subset).

Classification (reference/intended excitation, computed analytically from the
builders — no ODE, no Arrow scan needed):

```
per axis a ∈ {Vx, Vy, ψ̇}, over a short segment:
  zero   if  mean|a| < v_thr            (axis inactive)
  const  if  not zero and std(a) < s_thr (steady nonzero: cruise / steady spin)
  accel  if  not zero and std(a) ≥ s_thr (ramp, pulse, OR oscillation/wiggle)
v_thr ≈ 0.05 m/s (Vx,Vy), 0.05 rad/s (ψ̇)
```

### Plausible set: 21 of 27 cells

6 cells are **structurally impossible** given the profile set (not threshold
artifacts):

```
(Z,Z,Z)                       null excitation (nothing to observe)
(C,C,C)                       no Vconst·Omconst mode exists in coupled_vomega
(A,C,*),(C,A,*) with spin     translation axes in DIFFERENT modes WHILE spinning —
                              impossible: under spin, V is one magnitude+direction,
                              so Vx,Vy share a mode. Mixed-translation modes occur
                              ONLY in the ψ̇=0 plane (octagon, via axis-aligned leg
                              + perpendicular wiggle).
```

### Per-profile cell coverage and the backbone/redundant split

```
profile          feeds                                              role
octagon          entire ψ̇=0 plane (8 cells); heading fixed         BACKBONE (sole)
spin_creep       pure-spin (Z,Z,C/A), lateral-const+spin (Z,C,C/A) BACKBONE (sole)
coupled_vomega   coupled interior (A,A,*),(C,C,A),(Z,A,*-with-spin) BACKBONE (sole)
long_circle      (C,Z,C),(A,Z,A)  — also fed by spin_creep/coupled  redundant
spiral_orbit     (A,Z,C),(C,Z,A),(A,Z,A) — all fed by coupled       redundant
ellipse          (A,Z,A) tangent, (A,A,Z) crab — fed elsewhere      redundant
```

`octagon`, `spin_creep`, `coupled_vomega` are each the **unique** source of a
distinct cell group → they MUST appear in both folds. The other three add density
/ exercise the PosRef controller (ellipse) but no unique cells → free to assign.
(Note: spiral/ellipse are structurally confined to Vy=0 or ψ̇=0 faces, so they can
NOT substitute for coupled_vomega — only coupled moves Vy while spinning.)

---

## 3. S1/S2 — excitation 2-fold (single dominant χ = 0.005, all μ)

Two mutually-exclusive, **excitation-complete** subsets of the χ=0.005 data
(all μ, multisine excluded). Partition unit = whole **(profile, combo)** group
(μ-siblings kept together → no leakage).

```
backbone {octagon, spin_creep, coupled_vomega}:
    stratified 50/50 by excitation MODE (combo→mode read from the profile combo
    TOMLs): octagon → theta0_deg × lat-on/off; spin_creep → creep-dir × creep-
    zero/nonzero; coupled → (V_mode, Om_mode) × direction-class {x,y,diag}.
    Each (profile,mode) stratum is split 50/50 → BOTH folds get every mode →
    both folds cover all 21 plausible cells.

redundant {long_circle, spiral_orbit → S1;  ellipse → S2}:
    assigned wholesale (coverage-neutral). The over-represented S1 redundants are
    randomly downsampled to 1/4 so fold sizes match (~S1 1548 / S2 1579).
```

### Cross-subset training protocol (2 directions)

```
Run 1 (S1_train):  train = S1 (0.9) | val = S1 (0.1) | test = S2 (cross)
Run 2 (S2_train):  train = S2 (0.9) | val = S2 (0.1) | test = S1 (cross)
  val  = same-subset generalization  (new combos, same excitation coverage)
  test = cross-subset transfer       (the other excitation-complete fold)
  gap (val vs test) = instance memorization / fingerprinting vs true
                      excitation generalization
```

This is the strongest anti-fingerprinting test: a small encoder that learned
plant physics (not trajectory signatures) should have val ≈ test.

---

## 4. S3 — χ k-fold cross-validation (independent of S1/S2)

Tests **χ-generalization**: can the observer reconstruct hidden states at an
**unseen χ**? Each of the k=4 folds = one χ value.

```
restrict to matched χ-quads (combos present at ALL 4 χ) on the χ-swept profiles
  → 129 combos × 3 μ × 4 χ = 1548 files, BALANCED across folds.
for held-out χ* in {0, 0.002, 0.005, 0.008}:
    test  = χ*                       (387 files)
    train = other 3 χ (0.9)          (1098 files)
    val   = other 3 χ (0.1)          (63 files)
```

Matched-quads is what removes the χ=0.005-only combos that would otherwise make
the 0.005 fold huge — without it the folds are 5–7× imbalanced. Each χ-fold is
excitation-complete on its own (the χ-swept profiles ARE the backbone, which
covers all 21 cells). μ pooled; multisine/redundant profiles excluded (no χ
variation).

---

## 5. Reuse by Approach 1

A1 (force reconstruction) can consume the **same** partitions — they are
data-level file assignments, agnostic to the network/loss. The excitation cube
and the backbone/redundant classification are properties of the *trajectories*,
not the observer. A1's μ/χ are batch inputs already; the cube gives A1 a clean
excitation-stratified train/test split and the χ k-fold gives it a χ-extrapolation
test for its inverse μ/χ identification. Recommended: A1 reuse S1/S2 (excitation
generalization of forward dynamics) and S3 (χ-channel extrapolation).

---

## 6. Implementation pointers (Approach 2)

```
observer_v1_py/regimes/        A_ranking, D_chi_study, E_learning_curve_25,
                               S1_train, S2_train, S3_chi_kfold (+ base.toml)
observer_v1_py/mecanum_observer/data.py
   select_regime()    per-profile cap / χ-quad filter / stratified subsample
   _combo_mode_map()  combo_idx → excitation-mode (reads profile combo TOMLs)
   assign_folds()     stratified S1/S2 backbone + wholesale redundant
   split_files()      excitation-fold (train_fold) and χ-fold (chi_fold_test) modes
```

All splits are deterministic (seeded), record their exact file lists in each run
dir's `split.json`, and keep μ-/χ-siblings grouped to prevent leakage.
