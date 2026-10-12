# Gamma metric (Eker–Nikolaou incremental norm) — full campaign report

**Quantity.** γ = ‖(N − L)Q‖ over the error-signal set E, brute-force
incremental-norm estimate (Elkhalil–Zribi 2-norm surrogate): per 2 s window
of closed-loop baseline (best archived PID-CT 0.9 config, oracle-clean,
seeds n/a), per probe δ, γ̂ = ‖(V_N(M_ff+Qδ) − V_N(M_ff)) − lowpass_λ[δ]‖₂ / ‖δ‖₂.
Domain: commanded wrench (the mixer's motor-inverse + back-EMF cancellation is
part of the loop, per hybrid_ctrl/plant.jl). L = diag(1/(m_eff s + d_eff)),
Q = (m_eff s + d_eff)/(1 + λ s) with the tuned λ = (0.098, 0.114, 0.037) s.
Probes: band multisines 0.1–3/3–10/10–30 Hz + steps + chirps + backstop-driven
single-sines at the window's γ_lin argmax; amplitude ladder 1e-3…0.12 (m/s,
rad/s). Windows riding the slew/voltage/current limits >5% of ticks are
EXCLUDED from E (109/794 = 13.7%); probes pushed INTO a limit are rejected.

## Headline

- **Median γ̂ over the unsaturated envelope: 0.685 (< 1). p90 = 1.97; max 6.86.**
- **30.5% of in-E windows exceed γ̂ = 1**, concentrated in identifiable regimes
  (below) — the certificate is regime-conditional, not envelope-uniform.
- Uniformly certified (max γ̂ ≤ 0.8): long_circle (0.48/0.69), coupled_vomega
  (0.45–0.81), octagon_stress family in its rolling segments (median 0.41–0.63).
- Uniformly NOT certified: docking hold (γ̂ 2.8–6.2), spin_creep (median > 1),
  multisine broadband tiers (up to 6.9), spiral_orbit_easy (median 0.81 but
  30/85 windows > 1).

## Where γ > 1 — and why it matches the rest of the thesis

1. **Standstill / hold (docking_a, docking_step):** γ_lin = 1.0000–1.011 —
   the analytic preslip boundary (plant DC velocity gain = 0, LQ(0) = 1),
   reproduced exactly. Finite-amplitude hold probes: γ̂ up to 6.2 (breakaway
   nonlinearity). Same regime Tier C flagged (zero DC gain, stiff-spring
   linearization) and the BLA flagged (even-order dominance at reversals).
2. **Velocity-reversal-dense motion (multisine 50/75% tiers):** every reversal
   samples the bristle/preslip regime; the incremental norm over E integrates
   this. γ̂ up to 6.9 — the certificate fails even though per-window tracking
   is fine, because γ measures worst-case model error, not average.
3. **Gentle low-slip cruise (spiral_orbit_easy, spin_creep):** operating slip
   ~1e-4 m/s puts the linearized bristle creep modes at 0.05–0.3 Hz; error
   signals of ~1e-3 m/s already sample the micro-slip boundary — γ̂ grows with
   probe amplitude and with time-on-trajectory (0.3 → 3+ for spiral_orbit_easy).
   The infinitesimal Jacobian backstop (γ_lin up to 55.8, flagged) captures
   this regime but is NOT the incremental norm at practical amplitudes.

## Methodological notes (for the thesis)

- Brute force vs Jacobians, resolved empirically: the Jacobian backstop is
  valid where the operating slip ≫ probe micro-structure (aggressive tiers:
  γ̂ ≈ γ_lin within 2×); at low-slip windows it over-prints (micro-slip creep
  resonances at scales far below any controller error signal). The incremental
  norm at realistic amplitudes is the control-relevant quantity — this is the
  user's brute-force argument, confirmed by measurement.
- Restart fidelity: 299 in-E windows flagged > 5% replay-vs-record divergence;
  177 of them have the baseline riding a limit (expected divergence). The γ̂
  RATIO is insensitive to this by construction (probe and baseline replays
  share the snapshot and machinery — the error cancels in the difference).
- Amplitude scope: γ̂ grows along the probe ladder; the claim covers error
  amplitudes up to ~0.12 m/s / 0.12 rad/s (20% of envelope cap).

## Claim as defensible in the thesis

> Over the unsaturated envelope of the tuned PID-CT loop (86.3% of trajectory
> time), the median incremental model-error gain is γ̂ = 0.685 < 1 — i.e. the
> closed loop is "virtually linear" (Eker–Nikolaou) in the median regime —
> and γ̂ ≤ 0.8 uniformly on sustained rolling trajectories. γ̂ exceeds 1 in
> three identifiable regimes: standstill (γ = 1 analytically, preslip),
> reversal-dense motion (multisine), and gentle low-slip creep — the same
> corners the open-loop near-linearity study localized. The tuned λ was chosen
> for tracking, not robustness margin: γ scales with controller aggressiveness,
> so a λ-relaxed PID-CT could extend the certified region (future work).

Scripts: tierF_gamma_baseline.jl / tierF_gamma_probe.jl (+ .bat twins).
Rows: results/gamma_probe/*_gamma.jld2 (per window). Baselines:
results/gamma_baseline/*.jld2 (full state, 500 Hz, v_cmd, refs).
