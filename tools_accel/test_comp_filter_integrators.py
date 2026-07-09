#!/usr/bin/env python
"""Unit test for comp_filter.py's two mechanization integrators (Implementation
Sequence step 1 of instructions/frontend-drift-audit.md): a closed-form
constant-gyro, zero-specific-force "coasting rotation" — the body-frame
velocity vector traces a circle (in velocity space) over time purely from the
psi_dot*V transport-coupling term. Both integrators must converge to the
closed form as dt shrinks; "rot_ab2" (exact planar rotation for the coupling
term) must be dramatically more accurate than "euler" (linearized coupling) at
the SAME dt, and "euler"'s error must scale like O(dt) (halving dt halves it).

    a_x = a_y = 0, gyro = omega0 (constant)
    => dVx/dt = omega0*Vy, dVy/dt = -omega0*Vx
    => V(t) = R(-omega0*t) @ V(0)   (exact rotation, constant magnitude)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comp_filter import mechanize  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def closed_form(v0: np.ndarray, omega0: float, t: np.ndarray) -> np.ndarray:
    theta = -omega0 * t
    c, s = np.cos(theta), np.sin(theta)
    vx = c * v0[0] - s * v0[1]
    vy = s * v0[0] + c * v0[1]
    return np.stack([vx, vy], axis=-1)


def run(dt: float, omega0: float, duration: float, v0: np.ndarray):
    T = int(round(duration / dt)) + 1
    t = np.arange(T) * dt
    a_meas = np.zeros((T, 2), dtype=np.float64)
    gyro = np.full(T, omega0, dtype=np.float64)
    V_euler = mechanize(a_meas, gyro, dt, "euler", v0)
    V_rot = mechanize(a_meas, gyro, dt, "rot_ab2", v0)
    V_true = closed_form(v0, omega0, t)
    err_euler = np.linalg.norm(V_euler[-1] - V_true[-1])
    err_rot = np.linalg.norm(V_rot[-1] - V_true[-1])
    return err_euler, err_rot


def main() -> int:
    omega0 = 1.3  # rad/s
    duration = 2.0  # s -> ~0.41 revolutions, well within a single period
    v0 = np.array([0.8, -0.3])

    dt1 = 0.002   # 500 Hz
    dt2 = 0.001   # 1000 Hz (half)
    dt3 = 0.0005  # 2000 Hz (quarter)

    e1_euler, e1_rot = run(dt1, omega0, duration, v0)
    e2_euler, e2_rot = run(dt2, omega0, duration, v0)
    e3_euler, e3_rot = run(dt3, omega0, duration, v0)

    print(f"dt={dt1}: euler_err={e1_euler:.3e} rot_ab2_err={e1_rot:.3e}")
    print(f"dt={dt2}: euler_err={e2_euler:.3e} rot_ab2_err={e2_rot:.3e}")
    print(f"dt={dt3}: euler_err={e3_euler:.3e} rot_ab2_err={e3_rot:.3e}")

    check("rot_ab2 tracks the closed form to ~machine precision (exact rotation)",
          e1_rot < 1e-9, f"e1_rot={e1_rot:.3e}")
    check("rot_ab2 error does not grow when dt is coarsened",
          e1_rot < 1e-8 and e3_rot < 1e-8)

    check("euler shows O(dt) error at the coarsest step (clearly non-trivial)",
          e1_euler > 1e-4, f"e1_euler={e1_euler:.3e}")
    ratio_1_2 = e1_euler / e2_euler if e2_euler > 0 else float("inf")
    ratio_2_3 = e2_euler / e3_euler if e3_euler > 0 else float("inf")
    check("euler error halves when dt halves (dt1->dt2, first-order convergence)",
          1.7 < ratio_1_2 < 2.3, f"ratio={ratio_1_2:.3f}")
    check("euler error halves when dt halves again (dt2->dt3)",
          1.7 < ratio_2_3 < 2.3, f"ratio={ratio_2_3:.3f}")

    check("rot_ab2 is orders of magnitude more accurate than euler at the same dt",
          e1_rot < e1_euler * 1e-4, f"rot={e1_rot:.3e} euler={e1_euler:.3e}")

    check("both integrators converge toward the closed form as dt -> 0",
          e3_euler < e1_euler and e3_rot <= e1_rot + 1e-12)

    print()
    if FAILURES:
        print(f"COMP_FILTER INTEGRATOR TEST: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("COMP_FILTER INTEGRATOR TEST: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
