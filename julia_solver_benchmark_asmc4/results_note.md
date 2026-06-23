# Solver selection for the production sweep - results note (IMECE methods input)

**Setup.** Work-precision study on the 39-state stiff ODE (slip-spin LuGre, tanh factor >= 500),
multisine excitation: Stage A on the hardest row (75%-cap, f_hi = 3.5 Hz; mu = 0.4, chi = 0.005,
Adamov coupling), Stage B on six rows spanning both amplitude caps x f_hi in (1.0, 2.5, 3.5) Hz.
Error metric: RMS of the recomputed force/moment/torque labels on the 2 kHz grid against a
RadauIIA5 reference (rtol 1e-10, abstol/100, 4 kHz; self-validated against rtol 1e-11).

**Floor.** Force-interpolation floor on the primary case (midpoint reconstruction of the 2 kHz
grid): F = 0.0466 N, Mz = 0.000271 N*m,
Msat = 0.00385 N*m. Acceptance: solver label-RMS <= floor/10 per family.

**Result.** FBDF, reltol 1.0e-9, dtmax dtmax=0.001: margin
0.0622 on the primary case, passing all six Stage-B cases
(worst-case margins in stageB_results.arrow); median wall 36.6 s
per 60 s trajectory (n/ax vs the
previous TRBDF2 / rtol 1e-8 / dtmax 1e-3 setting). Solver integration error is therefore >= 10x
below the dataset's existing interpolation noise floor and does not contribute to the PINN
physics-residual budget.

**Caveats.** [edit: dtmax-subsidy observation; stiffness order-reduction observation;
concurrency ratio from the spot-check.]
