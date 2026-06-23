# Handoff — build `roller_slip_fraction.py` (how much of slip/spin is roller-driven)

## 1. Title + lineage
Build **`roller_slip_fraction.py`**, a model-free diagnostic quantifying **what
fraction of the per-wheel slip velocity and contact spin is driven by the hidden
roller rate `γ̇ᵢ`** vs the measurable kinematics. Continues the Mecanum-PINN
trajectory-diagnostics chat (KUKA youBot digital twin, IMECE 2026; Julia 39-D stiff
ODE → Arrow → PyTorch PINN). Sibling to `chi_identifiability.py` /
`mu_identifiability.py`; same streaming/CSV/`TRAJ_DIAGNOSTICRESULTS.md` pattern.

**Why now (the question it answers):** the μ-gate rides on co-directional slip
`Vp∥` and the χ-gate rides entirely on contact spin `ω_z` — both of which are
*partly hidden* (they contain `γ̇`, the roller rate the PINN cannot measure). This
diagnostic puts a hard number on **how much of each gate variable is directly
measurable vs must be reconstructed**, BEFORE the PINN is built. It decides whether
the gates can run on measurable proxies (roller reconstruction = insurance) or
genuinely require it (roller reconstruction = mandatory; elevates the separate
state-observer track A2). It needs no model — just algebra on existing columns.

## 2. Context the task depends on
- **Data (one dir):** `..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov`
  (5,670 Arrow files = μ∈{0.3,0.5,0.8}×1,890). Whitelist = `diagnostics_combined.csv`
  `combined_reco` NOT startswith `reject` (5,345 kept). **No JLD2 — read from Arrow.**
- **Everything needed is an Arrow column** (the roller rate `γ̇ᵢ` is stored as
  `gamma_i`): `Vx, Vy, psi_dot, theta1..4, w1..4, gamma1..4`, and the stored totals
  `Vpx_i, Vpy_i, wz_i` (for the cross-check). Geometry from `base.toml`:
  `R=0.05`, `Rd=Ra=0.0355`, `delta=[-π/4,π/4,π/4,-π/4]`, `px=wc_x=[h,h,-h,-h]`,
  `py=wc_y=[l,-l,l,-l]`, `h=0.235`, `l=0.15`.
- **Mirror exactly:** `chi_identifiability.py` / `mu_identifiability.py` — streaming
  **sufficient statistics** (Σx², Σ, n per bin; never pool raw samples), `--whitelist`,
  `MIN_N`, `PSI_EDGES=(0,0.5,2,∞)`. NO `--mu` filter (pool all μ; fraction is
  μ-independent kinematics).

## 3. The decomposition (EXACT — from `run_one.jl:675-684`)
Per wheel, with `θ̃=sawtooth_approx(θᵢ)`, `s=sinδ, c=cosδ, DYᵢ=Rd·tanδ·tan θ̃`,
`sθ=sin θ̃, cθ=cos θ̃`, `gᵢ=gamma_i` (roller rate, HIDDEN):
```
Vpx_i = [Vx − ψ̇·(py+DYᵢ) − wᵢ·R]              +  gᵢ·s·(Rd·cθ − R) + DYᵢ·gᵢ·c·sθ
Vpy_i = [Vy + ψ̇·px]                            +  gᵢ·c·(R·cθ − Rd)
ω_z,i = [ψ̇]                                    +  gᵢ·sθ·c
        └──── MEASURABLE ────┘                    └──── ROLLER (∝ gᵢ) ────┘
```
**Gate axes** (datastore rotates FORCE by δ but NOT slip — rotate slip yourself):
`Vp∥ = Vpx·cosδ + Vpy·sinδ` (drive / Fpar / **μ-gate**),
`Vp⊥ = −Vpx·sinδ + Vpy·cosδ` (free-roll / Fperp). Split `Vp∥, Vp⊥` into
measurable/roller by rotating the measurable and roller parts of `Vpx, Vpy` above.

## 4. Key design decisions (already made — defend, don't reopen)
1. **Headline = energy fraction `Σ(roller²)/Σ(total²)` per bin** (streaming-friendly,
   robust). Pointwise `|roller|/|total|` blows up as total→0; report its bin **median**
   as a secondary descriptor only, computed as `|roller|/(|meas|+|roller|)` ∈ [0,1].
2. **Sufficient stats per bin:** `Σmeas², Σroll², Σtot², Σ(meas·roll), n`. From these:
   energy_frac, and the meas/roller alignment `Σ(meas·roll)/√(Σmeas²·Σroll²)`.
3. **The TWO numbers that decide it:** (a) roller energy-frac of **`ω_z`** in
   **high-|ω_z| bins** (χ-gate stakes — the spin term); (b) roller energy-frac of
   **`Vp∥`** in **gross-slip bins** (μ-gate stakes). Report these as the verdict.
4. **Stratify by the SAME regimes the gates use, NOT a global average** (a global mean
   over low-slip creep would mislead): for `Vp∥/Vp⊥`, bin by signed slip + `|psi_dot|`
   strata (mirror `mu_identifiability`); for `ω_z`, bin by `|ω_z|` (mirror
   `chi_identifiability` `WZ_EDGES`) + `|psi_dot|` strata.
5. **Validate the formula FIRST** on one real file: `meas + roller` must reconstruct the
   stored `Vpx_i / Vpy_i / wz_i` within solver tol — this pins the `gamma_i` sign/units
   before trusting any fraction. (If it mismatches, the decomposition or a sign is wrong.)
6. **Frame as a bound:** measurable energy-frac = the FLOOR on gate quality with zero
   reconstruction; roller energy-frac = what the encoder (A1) / observer (A2) must
   recover to do better. High measurable frac in the gate regimes ⇒ gates ride on
   measurables, A2 = sanity check; low ⇒ A2 graduates toward mandatory.

## 5. Open decisions / blocking
- Bin edges for signed `Vp∥/Vp⊥` (reuse `mu_identifiability`'s) and `|ω_z|` (reuse
  `chi_identifiability`'s `WZ_EDGES`) — confirm ranges from a real Arrow.
- Confirm `sawtooth_approx` (`atan2(40·sin12θ, 40·cos12θ+1)/12`, k=40) matches the
  notebook before computing `θ̃` (it sets `DYᵢ` and the roller terms).
- **Hand-back:** the two verdict numbers gate the PINN's μ/χ-gating design and decide
  how load-bearing the A2 state-observer is. Feeds the main PINN session.

## 6. Deliverables
1. `roller_slip_fraction.py` — streaming, resume-aware, CLI mirroring the others
   (`--data-dir`, `--out`, `--whitelist`); pools all μ.
2. `roller_slip_fraction.csv` — per `(stratum, quantity∈{Vp_par,Vp_perp,wz}, regime-bin)`:
   `n, meas_rms, roller_rms, total_rms, roller_energy_frac, roller_over_total_median,
   meas_roller_align`.
3. New **§9 in `TRAJ_DIAGNOSTICRESULTS.md`** (observation→inference→explanation, §2 style)
   with the verdict: roller fraction of `ω_z`@high-spin and `Vp∥`@gross-slip, and the
   "gates-on-measurables vs reconstruction-required" call.
4. (Optional) one static figure: roller energy-frac vs `|ω_z|` and vs `|Vp∥|`.

## 7. Conventions to respect
- Run from `code_insights/`; **claude-venv** python `C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`.
- **≤8 threads/workers** (hard cap — machine OOMs above). Per-file streaming
  (read→accumulate→advance; never hold all files). Static matplotlib only — **no
  interactive widgets**. No LaTeX in chat (Unicode/code blocks).
- Confirm decisions BEFORE coding; surgical edits; compute every numeric value in code
  before stating it. CSV existence = resume marker (never hand-edit). Scratch in `_tmp/`,
  clean up. Long runs (>~20 min): start `keep_awake.py` in background first.
