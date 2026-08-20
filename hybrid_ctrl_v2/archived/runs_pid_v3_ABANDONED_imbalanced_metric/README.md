# runs_pid_v3 — ABANDONED (imbalanced v3 metric)

PID v2 FB/CT clean retune, 5 seeds each, launched 2026-08-16 19:41:32 against
`train14_v3` with `--metric v3 --lambda-chatter 0.36`. Killed 2026-08-17 ~04:05
at 70–87 evals per run (of a 250 cap), **before convergence**. No
`best_config.json` was written; `checkpoint.json` + `trace.csv` are the record.

## Why abandoned

The v3 tracking metric divided the three POSITION terms by the per-trajectory
scaler `k_pos = max(radius_var(ref), TOL.pos_max)/TOL.pos_max` (range 1.0–29.0,
mean ~16) and divided the three HEADING terms by nothing:

    tracking = ( fin_pos/(0.01*k) + max_pos/(0.1*k) + iae_pos/(0.0249*k)
               + fin_hd/0.01      + max_hd/0.1      + iae_hd/0.0128      ) / 6

Under v2 both channels were absolute and equally weighted. So v3 silently
downweighted position ~16x relative to heading — a re-weighting that was not a
design decision, it fell out of scaling one channel and not the other.

Consequence, measured on these runs' own configs (`_tmp/pid_v3_components.jl`):

    tracking sub-terms, mean over 14 trajectories
             fin_pos  max_pos  iae_pos |  fin_hd   max_hd   iae_hd
    FB        0.0069   0.0417   0.0403 |  0.0317   0.0183   0.0283
    CT        0.0025   0.0189   0.0222 |  0.0076   0.1054   0.1836

CT is better on ALL THREE position terms (0.0436 vs 0.0889 summed) and loses on
heading (0.2966 vs 0.0783). Heading was 87% of CT's tracking term. That inverted
the controller ranking versus v2 (v2: CT 0.1134 better than FB 0.2243; v3 here:
CT 0.0567 worse than FB 0.0279) — i.e. the ranking was being driven by the
accidental re-weighting, not by control quality.

NOT implicated: the trigonometric heading error `2|sin(dpsi/2)|`. At the observed
magnitudes (max_head 0.0018–0.0105 rad) it agrees with the wrapped angle to
~5e-6 relative. It is inert here and was kept.

## Fix applied afterwards

Apply the single trajectory factor to the WHOLE tracking term rather than to
position alone (equivalently, divide both channels by `k`), restoring v2's
position:heading balance while keeping the term trajectory-relative. See
`stage_objective.jl`.

## What IS still usable here

The runs were healthy — no errors, no divergence sentinels, memory flat at ~7 GB
commit headroom across 10 concurrent processes for 8.4 h. The traces are a valid
record of optimiser behaviour under the imbalanced metric, and the CT heading
weakness they exposed (max_head 0.0105 vs FB 0.0018 rad, ~6.5x the sustained
heading error) is a REAL measured difference worth re-checking under the
corrected metric — it was simply weighted wrongly here.

Scores in this directory are NOT comparable to anything produced before or after.
