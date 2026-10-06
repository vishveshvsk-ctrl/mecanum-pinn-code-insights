# Parametric near-linearity scan (Tier A machinery)

Steady-state map v -> V_req against the exact plant; envelope caps are the LEGACY ones (0.63, 0.63, 3.8). nu = Allgoewer residual of the through-origin best linear map (max / p95); spread = per-axis secant gain spread; feas_max = max feasible fraction of cap per axis (x, y, psi).

| case | nu max | nu p95 | R2 | grid pts | gain spread x/y/psi | feas max x/y/psi |
|---|---|---|---|---|---|---|
| base | 0.0508 | 0.0508 | 0.99936 | 16 | 1.75 / 1.39 / 1.24 | 1.00 / 0.75 / 0.50 |
| com2x | 0.0509 | 0.0509 | 0.99931 | 16 | 1.75 / 1.38 / 1.24 | 1.00 / 0.75 / 0.50 |
| com3x | 0.0511 | 0.0511 | 0.99914 | 16 | 1.75 / 1.36 / 1.23 | 1.00 / 0.50 / 0.50 |
| chi4x | 0.0508 | 0.0508 | 0.99923 | 16 | 1.75 / 1.11 / 1.12 | 1.00 / 0.75 / 0.50 |
| mu03 | 0.0500 | 0.0500 | 0.99927 | 15 | 1.75 / 1.37 / 1.24 | 1.00 / 0.50 / 0.50 |
| case2 | 0.0771 | 0.0714 | 0.99797 | 50 | 1.96 / 1.70 / 1.35 | 1.00 / 1.00 / 0.75 |
| extreme | NaN | NaN | NaN | 5 | 1.75 / 1.16 / 1.16 | 1.00 / 0.25 / 0.25 |
