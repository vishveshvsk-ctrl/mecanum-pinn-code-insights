# Tier A+C — plant near-linearity quantification

Plant point: mu=0.5, chi=0.005, friction_case=1, quasi-static motor (back-EMF active), LuGre dynamic bristles, Adamov coupling.
Steady state solved against the exact `plant_rhs!` (23 eqs) at 8 roller phases; envelope caps Vx,Vy=0.63 m/s, psidot=3.8 rad/s.

## Tier A — steady-state map v -> V_req

- feasible grid points used for fit: 16 (of 125)
- Allgoewer nu (through-origin best linear map): max 0.0508, p95 0.0508, R2 0.99936
- Allgoewer nu (affine map, Coulomb offset removed):  max 0.0525, p95 0.0501, R2 0.99936
- roller-phase spread of the static map: 2.56% max
- per-axis secant gain spread (V_req/v, +sweep):
    axis 1: gain 83.825 -> 47.814 (frac 0.05->1.00), spread 1.753x
    axis 2: gain 90.474 -> 65.920 (frac 0.05->0.75), spread 1.388x
    axis 3: gain 25.469 -> 20.588 (frac 0.05->0.50), spread 1.237x
- per-axis feasible envelope (24 V bus + 12.8 A current, steady state):
    axis 1 +1 sweep: feasible to 1.00 of cap (0.630 m/s)
    axis 1 -1 sweep: feasible to 1.00 of cap (0.630 m/s)
    axis 2 +1 sweep: feasible to 0.75 of cap (0.473 m/s)
    axis 2 -1 sweep: feasible to 0.75 of cap (0.473 m/s)
    axis 3 +1 sweep: feasible to 0.50 of cap (1.900 rad/s)
    axis 3 -1 sweep: feasible to 0.50 of cap (1.900 rad/s)

## Tier C — local linearisation

Transfer G(jω) = C(jωI−A)⁻¹B evaluated at [0.1, 0.3, 1.0, 3.0] Hz (DC is meaningless at rest: preslip bristle stiffness holds the platform, v_dc = 0 — that zero IS the stiction nonlinearity, reported separately, not mixed into the spread).

- |G_jj| spread across the 12 moving operating points (max/min):
    f=0.1 Hz axis 1: [0.0206, 0.0218], ratio 1.06x (rest point: 0.0000)
    f=0.1 Hz axis 2: [0.0094, 0.0160], ratio 1.70x (rest point: 0.0000)
    f=0.1 Hz axis 3: [0.0366, 0.0559], ratio 1.53x (rest point: 0.0000)
    f=0.3 Hz axis 1: [0.0190, 0.0216], ratio 1.14x (rest point: 0.0000)
    f=0.3 Hz axis 2: [0.0094, 0.0159], ratio 1.70x (rest point: 0.0000)
    f=0.3 Hz axis 3: [0.0352, 0.0502], ratio 1.43x (rest point: 0.0000)
    f=1.0 Hz axis 1: [0.0131, 0.0203], ratio 1.55x (rest point: 0.0000)
    f=1.0 Hz axis 2: [0.0085, 0.0153], ratio 1.80x (rest point: 0.0000)
    f=1.0 Hz axis 3: [0.0292, 0.0477], ratio 1.63x (rest point: 0.0000)
    f=3.0 Hz axis 1: [0.0066, 0.0144], ratio 2.17x (rest point: 0.0000)
    f=3.0 Hz axis 2: [0.0055, 0.0121], ratio 2.20x (rest point: 0.0000)
    f=3.0 Hz axis 3: [0.0176, 0.0374], ratio 2.13x (rest point: 0.0000)
- max cross-coupling |offdiag|/|diag| per frequency: 0.990, 0.918, 0.831, 0.784
- slowest 6 poles per operating point (phase 1):
    [0.0, 0.0, 0.0]        [-0.4+0.0i, -1.0+0.0i, -1.0+0.0i, -1.0+0.0i, -1.0+0.0i, -4.6-87.4i]  modes <50/50-500/>500: 5/6/12
    [0.5, 0.0, 0.0]        [-1.0+0.0i, -1.0+0.0i, -1.0+0.0i, -1.1+0.0i, -16.2+0.0i, -21.2-0.1i]  modes <50/50-500/>500: 7/8/8
    [-0.5, 0.0, 0.0]       [-1.0+0.0i, -1.0+0.0i, -1.0+0.0i, -1.1+0.0i, -16.2+0.0i, -20.3+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.0, 0.5, 0.0]        [-1.0+0.0i, -1.1+0.0i, -1.1+0.0i, -1.2+0.0i, -16.1+0.0i, -20.7+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.0, -0.5, 0.0]       [-1.0+0.0i, -1.1+0.0i, -1.1+0.0i, -1.2+0.0i, -16.3+0.0i, -20.5+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.0, 0.0, 0.375]      [-16.3+0.0i, -20.3+0.0i, -21.5+0.0i, -29.4+0.0i, -30.1+0.0i, -32.9+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.0, 0.0, -0.375]     [-16.3+0.0i, -20.2+0.0i, -21.6+0.0i, -29.4+0.0i, -30.5+0.0i, -32.3+0.0i]  modes <50/50-500/>500: 7/8/8
    [1.0, 0.0, 0.0]        [-1.0+0.0i, -1.0+0.0i, -1.0+0.0i, -1.1+0.0i, -16.2+0.0i, -21.2-0.7i]  modes <50/50-500/>500: 7/8/8
    [0.0, 1.0, 0.0]        [-1.3+0.0i, -1.4+0.0i, -1.7+0.0i, -2.4+0.0i, -16.2+0.0i, -21.0+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.0, 0.0, 0.75]       [-17.1-0.4i, -17.1+0.4i, -19.3+0.0i, -95.0+0.0i, -106.4+0.0i, -87.2-148.4i]  modes <50/50-500/>500: 3/12/8
    [0.5, 0.5, 0.0]        [-1.0+0.0i, -1.1+0.0i, -1.1+0.0i, -1.1+0.0i, -18.8+0.0i, -45.6+0.0i]  modes <50/50-500/>500: 6/8/9
    [0.5, 0.0, 0.375]      [-16.3+0.0i, -20.6+0.0i, -21.2+0.0i, -30.1+0.0i, -31.2+0.0i, -34.8+0.0i]  modes <50/50-500/>500: 7/8/8
    [0.25, 0.25, 0.25]     [-16.3+0.0i, -18.2+0.0i, -18.9+0.0i, -19.3+0.0i, -21.3-0.5i, -21.3+0.5i]  modes <50/50-500/>500: 7/8/8

runtime 15.8 s
