"""
Scaled, spin-gated modified-LuGre regime analysis for PU Mecanum rollers.

Model (calibration-note formulation + epsilon/xi spin gate from composite_v2):

    zdot   = V_slip - V_eff * (z/delta*) * mu_s / g(V_eff)
    F      = mu_s * N * ( z/delta* + tau * zdot/delta* )
    f      = F/(mu_s N) = xi + tau * xidot           (xi = z/delta*)

    V_eff  = ||Vp|| + R(xi) * (8/(3 pi)) * |w_z| * a      (Adamov mean slip)
    xi_safe= xi / sqrt(1+xi^2)
    R(xi)  = 1 - (1 - xi_safe)^(2/3)                      (Mindlin annulus, C-inf)
    g(s)   = mu_c [ 1 + (r_s - 1) exp(-(s/v_str)^2) ]     (dimensionless Stribeck)

Calibration identities:  delta* = mu_s/sigma0 ,  tau = sigma1/sigma0.

Worked example (a = 5 mm contact radius, Section 6 of the calibration note):
    sigma0 = 1.64e3 /m , sigma1 = tau*sigma0 = 1.64 s/m (tau = 1 ms) ,
    mu_c = 0.60 , mu_s = 0.66 , v_str = 0.01 m/s , N = 87 N/roller.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0.  Output location and parameters
# ----------------------------------------------------------------------
OUTDIR = r"C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\stick_slip_behavior"
os.makedirs(OUTDIR, exist_ok=True)

sigma0          = 1.64e3              # m^-1  (calibration note)
mu_c, mu_s     = 0.60, 0.66            # -
v_str          = 0.01                  # m s^-1  (discretionary Stribeck vel.)
N              = 87.0                  # N per roller
a              = 5.0e-3                # contact radius a = 5 mm
beta_roller    = 10.0e-3               # PU roller radius beta ~ 10 mm (note Sec.6)
tau_def        = 1.0e-3                # s  (explicit time constant = 1 ms)
sigma1         = tau_def * sigma0     # s m^-1  (derived so tau = 1 ms exactly)

tau    = tau_def                       # s  (= sigma1/sigma0)
dstar  = mu_s / sigma0                 # m  (break-away deflection)
r_s    = mu_s / mu_c
C_SPIN = 8.0 / (3.0 * np.pi)           # Adamov translational spin factor
C_ROT_CROSS = 5.0                       # drilling cross-term (hard-coded factor)

results = {}   # collect scalars for the markdown table


def g(s):
    """Dimensionless Stribeck friction ratio, s = effective slip speed."""
    return mu_c * (1.0 + (r_s - 1.0) * np.exp(-(s / v_str) ** 2))


def R_gate(xi):
    """Mindlin slip-annulus area fraction, C-inf in xi = |z|/delta*."""
    xi = np.abs(xi)
    xi_safe = xi / np.sqrt(1.0 + xi * xi)
    return 1.0 - (1.0 - xi_safe) ** (2.0 / 3.0)


def V_eff(z, Vp, wz):
    """Adamov pressure-weighted mean slip with xi-gated spin coupling."""
    xi = np.abs(z) / dstar
    return abs(Vp) + R_gate(xi) * C_SPIN * abs(wz) * a


def zdot(t, z, Vp_f, wz_f):
    z = z[0] if hasattr(z, "__len__") else z
    Vp, wz = Vp_f(t), wz_f(t)
    Ve = V_eff(z, Vp, wz)
    return Vp - Ve * (z / dstar) * mu_s / g(Ve)


def f_scaled(t, z, Vp_f, wz_f):
    """Normalised force f = F/(mu_s N) = xi + tau*xidot."""
    zd = zdot(t, [z], Vp_f, wz_f)
    return z / dstar + tau * zd / dstar


# ----------------------------------------------------------------------
# Rotational (drilling) channel -- built by direct analogy with z, but
# with the rotational V_eff of Eq. (12) and the sigma_0s = (2/3) sigma_0
# calibration.  We work in CONTACT-EDGE speeds (multiply the rad/s Eq. 12
# by a) so every term is in m/s and the scaling mirrors translation:
#
#   zdot_rot = a*wz - Veff_rot * (z_rot/dstar_rot) * mu_s/g(Veff_rot)
#   Veff_rot = (16/3pi)*(a|wz|) + R(xi_rot)*5*||Vp||       (edge, m/s)
#   m = M_z/(mu_s N a) = xi_rot + tau_rot * xidot_rot,  xi_rot=z_rot/dstar_rot
#
#   dstar_rot = mu_s/sigma_0s = 1.5 dstar ,  tau_rot = sigma_1s/sigma_0s = tau
#
# Note the native/effective coefficient mismatch (1 vs 16/3pi) makes the
# pure-spin ceiling xi_rot,ss = (3pi/16) g/mu_s ~= 0.589 g/mu_s < 1, unlike
# translation (ceiling 1). The cross term carries the factor 5 (not 8/3pi).
# ----------------------------------------------------------------------
sigma0_rot = (2.0 / 3.0) * sigma0        # m^-1
dstar_rot  = mu_s / sigma0_rot           # = 1.5 * dstar
tau_rot    = tau                          # (2/3) cancels in sigma1s/sigma0s
C_SPIN_ROT   = 16.0 / (3.0 * np.pi)       # native drilling factor on a*wz
C_CROSS_ROT  = 5.0                        # gated translation cross factor


def V_eff_rot(z_rot, Vp, wz):
    """Rotational Adamov mean slip (edge, m/s): native spin + gated transl."""
    xir = np.abs(z_rot) / dstar_rot
    return C_SPIN_ROT * abs(wz) * a + R_gate(xir) * C_ROT_CROSS * abs(Vp)


def zdot_rot(t, z, Vp_f, wz_f):
    z = z[0] if hasattr(z, "__len__") else z
    Vp, wz = Vp_f(t), wz_f(t)
    Ve = V_eff_rot(z, Vp, wz)
    return a * wz - Ve * (z / dstar_rot) * mu_s / g(Ve)


def m_scaled(t, z, Vp_f, wz_f):
    """Normalised drilling torque m = M_z/(mu_s N a) = xi_rot + tau*xidot_rot."""
    zd = zdot_rot(t, [z], Vp_f, wz_f)
    return z / dstar_rot + tau_rot * zd / dstar_rot


def run_rot(Vp_f, wz_f, tspan, z0, max_step):
    sol = solve_ivp(zdot_rot, tspan, [z0], args=(Vp_f, wz_f),
                    max_step=max_step, dense_output=True, rtol=1e-9, atol=1e-12)
    t = sol.t
    z = sol.y[0]
    xir = z / dstar_rot
    m   = np.array([m_scaled(ti, zi, Vp_f, wz_f) for ti, zi in zip(t, z)])
    Ve  = np.array([V_eff_rot(zi, Vp_f(ti), wz_f(ti)) for ti, zi in zip(t, z)])
    return sol, t, z, xir, m, Ve


# ----------------------------------------------------------------------
# 1.  The static gate table  R(xi)
# ----------------------------------------------------------------------
xi_tab = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 1e3])
R_tab  = R_gate(xi_tab)
results["gate_table"] = list(zip(xi_tab, R_tab))

# ----------------------------------------------------------------------
# helper to run one channel
# ----------------------------------------------------------------------
def run(Vp_f, wz_f, tspan, z0, max_step):
    sol = solve_ivp(zdot, tspan, [z0], args=(Vp_f, wz_f),
                    max_step=max_step, dense_output=True, rtol=1e-9, atol=1e-12)
    t = sol.t
    z = sol.y[0]
    xi = z / dstar
    f  = np.array([f_scaled(ti, zi, Vp_f, wz_f) for ti, zi in zip(t, z)])
    Ve = np.array([V_eff(zi, Vp_f(ti), wz_f(ti)) for ti, zi in zip(t, z)])
    return sol, t, z, xi, f, Ve


ZERO = lambda t: 0.0

# ======================================================================
# REGIME 1 & 2 : zero ICs, deep presliding at Vp = 5 mm/s
#                compare spin-free vs spin-on (|wz| a = 5 mm/s)
# ======================================================================
Vp1  = lambda t: 5e-3
wz0  = lambda t: 0.0
wzON = lambda t: 5e-3 / a          # |wz| a = 5 mm/s  (spin slip == transl slip)

_, t1a, z1a, xi1a, f1a, Ve1a = run(Vp1, wz0,  [0, 0.6], 0.0, 1e-3)
_, t1b, z1b, xi1b, f1b, Ve1b = run(Vp1, wzON, [0, 0.6], 0.0, 1e-3)

T_pre = (g(5e-3) / mu_s) * (dstar / 5e-3)
results["reg12"] = dict(
    dstar_mm=dstar * 1e3, tau_ms=tau * 1e3,
    xi_ss_free=g(5e-3) / mu_s, T_ms=T_pre * 1e3,
    fjump_theory=tau * 5e-3 / dstar, fjump_tauT=tau / T_pre,
    f_ss_free=f1a[-1], xi_ss_free_sim=xi1a[-1],
    xi_ss_spin=xi1b[-1], f_ss_spin=f1b[-1],
    Ve_end_spin=Ve1b[-1],
)

# --- Regime 1/2, ROTATIONAL channel: native a*wz = 5 mm/s; cross-coupling
#     translation off (pure spin) vs on (||Vp|| = 5 mm/s) ---------------
wzR   = lambda t: 5e-3 / a            # a*|wz| = 5 mm/s  (native drilling drive)
VpOFF = lambda t: 0.0
VpON  = lambda t: 5e-3
_, t1ra, z1ra, xr1a, m1a, Ver1a = run_rot(VpOFF, wzR, [0, 0.6], 0.0, 1e-3)
_, t1rb, z1rb, xr1b, m1b, Ver1b = run_rot(VpON,  wzR, [0, 0.6], 0.0, 1e-3)
results["reg12_rot"] = dict(
    dstar_rot_mm=dstar_rot * 1e3, ceiling_pure_spin=(3 * np.pi / 16) * g(5e-3) / mu_s,
    xir_ss_transfree=xr1a[-1], m_ss_transfree=m1a[-1],
    xir_ss_transon=xr1b[-1], m_ss_transon=m1b[-1],
)

# ======================================================================
# REGIME 3 : stick -> slip, ramp Vp 0 -> 0.3 m/s over 1 s
#            spin-free vs spin-on (|wz| a = 50 mm/s constant)
# ======================================================================
Vp3   = lambda t: 0.3 * np.clip(t, 0, 1.0)
wz3   = lambda t: 50e-3 / a
_, t3a, z3a, xi3a, f3a, Ve3a = run(Vp3, wz0, [0, 1.5], 0.0, 5e-4)
_, t3b, z3b, xi3b, f3b, Ve3b = run(Vp3, wz3, [0, 1.5], 0.0, 5e-4)

ia, ib = np.argmax(f3a), np.argmax(f3b)
results["reg3"] = dict(
    peak_free=f3a[ia], vpk_free=Vp3(t3a[ia]) * 1e3, xipk_free=xi3a[ia],
    peak_spin=f3b[ib], vpk_spin=Vp3(t3b[ib]) * 1e3, xipk_spin=xi3b[ib],
    fslip_free=f3a[-1], fslip_spin=f3b[-1], floor=1.0 / r_s,
    ximax_free=xi3a.max(), ximax_spin=xi3b.max(),
)

# --- Regime 3, ROTATIONAL channel: ramp native a*wz 0 -> 300 mm/s;
#     cross-coupling translation off vs on (||Vp|| = 50 mm/s) ----------
wz3R    = lambda t: (0.3 / a) * np.clip(t, 0, 1.0)   # a*wz: 0 -> 300 mm/s
Vp3ON   = lambda t: 50e-3
_, t3ra, z3ra, xr3a, m3a, Ver3a = run_rot(VpOFF, wz3R, [0, 1.5], 0.0, 5e-4)
_, t3rb, z3rb, xr3b, m3b, Ver3b = run_rot(Vp3ON, wz3R, [0, 1.5], 0.0, 5e-4)
awz3a = np.array([a * wz3R(ti) for ti in t3ra])
awz3b = np.array([a * wz3R(ti) for ti in t3rb])
jr = np.argmax(m3a)
results["reg3_rot"] = dict(
    peak_transfree=float(m3a.max()), m_gross_transfree=float(m3a[-1]),
    peak_transon=float(m3b.max()), m_gross_transon=float(m3b[-1]),
    ceiling_pure_spin_lowspeed=(3 * np.pi / 16),
    xir_max_transfree=float(xr3a.max()), xir_max_transon=float(xr3b.max()),
)

# --- Same-run COUNTERPARTS for Fig 2 (drives differ between the two rows,
#     so we integrate the *other* channel under each row's coupled drive) --
# drilling counterpart of the translational ramp (Vp ramp, |wz|a=50 const):
_, t3rc, z3rc, xr3c, m3c, _ = run_rot(Vp3, wz3, [0, 1.5], 0.0, 5e-4)
# translational counterpart of the drilling ramp (||Vp||=50 const, a*wz ramp):
_, t3d,  z3d,  xi3d, f3d, _ = run(Vp3ON, wz3R, [0, 1.5], 0.0, 5e-4)
awz3d = np.array([a * wz3R(ti) for ti in t3d])

# ======================================================================
# REGIME 4 : slip -> stick, decel 0.3 -> 0 m/s over 0.2 s then hold
# ======================================================================
def Vp4(t):
    return 0.3 * max(0.0, 1.0 - t / 0.2) if t < 0.2 else 0.0
z0_4 = (g(0.3) / mu_s) * dstar          # start from kinetic steady state
_, t4a, z4a, xi4a, f4a, Ve4a = run(Vp4, wz0, [0, 0.5], z0_4, 2e-4)

results["reg4"] = dict(
    xi0=xi4a[0], f0=f4a[0], xi_frozen=xi4a[-1], f_frozen=f4a[-1],
    xi_max=xi4a.max(),
    zdot_end=zdot(0.5, [z4a[-1]], Vp4, wz0),
)

# ======================================================================
# REGIME 4b : deceleration-RATE sweep -> Stribeck/rate gating of recovery
#   Slow stops give the state time to track the rising Stribeck target
#   xi_ss = g(Veff)/mu_s (-> 1 as v -> 0); fast stops freeze it lower.
# ======================================================================
decel_times = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
frozen_xi = []
sweep_curves = []
for td in decel_times:
    def Vp4b(t, td=td):
        return 0.3 * max(0.0, 1.0 - t / td) if t < td else 0.0
    _, tt, zz, xx, ff, VV = run(Vp4b, wz0, [0, td + 0.4], z0_4,
                                min(2e-4, td / 50.0))
    frozen_xi.append(float(xx[-1]))
    sweep_curves.append((td, tt.copy(), xx.copy()))
results["reg4_rate_sweep"] = dict(
    decel_times_s=decel_times, frozen_xi=frozen_xi,
    xi_target_at_rest=1.0, xi_start=z0_4 / dstar,
)

# ======================================================================
# REGIME 4c : translation stops but residual spin remains -> NO freeze
#   Veff = R(xi)*(8/3pi)*|wz|*a > 0, so the relaxation term stays active
#   and the bristle keeps evolving (no locked-in preload).
# ======================================================================
def Vp4c(t):
    return 0.3 * max(0.0, 1.0 - t / 0.2) if t < 0.2 else 0.0
wz_res = lambda t: 20e-3 / a          # residual |wz| a = 20 mm/s after stop
_, t4c, z4c, xi4c, f4c, Ve4c = run(Vp4c, wz_res, [0, 0.8], z0_4, 2e-4)
results["reg4_residual_spin"] = dict(
    xi_end=xi4c[-1], f_end=f4c[-1],
    zdot_end=zdot(0.8, [z4c[-1]], Vp4c, wz_res),
    Veff_end_mm_s=Ve4c[-1] * 1e3,
)
# drilling-channel counterpart of the SAME residual-spin run (gives the
# right-axis overlay for Fig 4 panel h):
_, t4cr, z4cr, xr4c, m4c, _ = run_rot(Vp4c, wz_res, [0, 0.8], 0.0, 2e-4)

# ======================================================================
# REGIME 5 : stick -> slip -> stick -> slip  (full oscillatory cycle)
#   Unlike Regimes 3/4 (single transitions poised at the break-away limit),
#   here the drive carries the contact DEEP into gross sliding (v >> v_str,
#   f -> kinetic floor 1/r_s) and back to a frozen stick dwell, repeatedly.
#   The result is a closed hysteresis loop in the f-|Vp| plane -- the
#   signature of a stick-slip limit cycle.
# ======================================================================
P5    = 0.5          # cycle period [s]
Vmax5 = 0.15         # peak slip speed [m/s] = 15 * v_str  (deep gross sliding)

def Vp5(t):
    tc = t % P5
    if tc < 0.10:                       # stick dwell (bristle frozen)
        return 0.0
    elif tc < 0.25:                     # accelerate 0 -> Vmax (stick -> slip)
        return Vmax5 * (tc - 0.10) / 0.15
    elif tc < 0.40:                     # decelerate Vmax -> 0 (slip -> stick)
        return Vmax5 * (1.0 - (tc - 0.25) / 0.15)
    else:                               # stick dwell (re-freeze)
        return 0.0

_, t5, z5, xi5, f5, Ve5 = run(Vp5, wz0, [0, 2 * P5], 0.0, 1e-4)
Vp5arr = np.array([Vp5(ti) for ti in t5])

# dwell (stick) samples: where Vp == 0 and t beyond the first build-up
dwell_mask = (Vp5arr == 0.0) & (t5 > 0.4)
results["reg5"] = dict(
    Vmax_mm_s=Vmax5 * 1e3, Vmax_over_vstr=Vmax5 / v_str,
    f_gross_floor=1.0 / r_s,
    f_min=float(f5.min()), f_max=float(f5.max()),
    xi_gross_min=float(xi5[Vp5arr > 5 * v_str].min()) if np.any(Vp5arr > 5 * v_str) else None,
    xi_dwell_frozen=float(xi5[dwell_mask][-1]) if np.any(dwell_mask) else None,
    f_dwell_frozen=float(f5[dwell_mask][-1]) if np.any(dwell_mask) else None,
    n_cycles=2,
)

# ======================================================================
# Velocity domains actually exercised (reported in the markdown)
# ======================================================================
results["velocity_domains"] = dict(
    v_str_mm_s=v_str * 1e3,
    reg12_Vp_mm_s=5.0, reg12_Vp_over_vstr=5.0 / (v_str * 1e3),
    reg12_spin_wz_a_mm_s=5.0,
    reg3_Vp_range_mm_s=[0.0, 300.0], reg3_Vp_over_vstr=[0.0, 300.0 / (v_str * 1e3)],
    reg3_spin_wz_a_mm_s=50.0,
    reg4_Vp_range_mm_s=[300.0, 0.0], reg4_Vp_over_vstr=[300.0 / (v_str * 1e3), 0.0],
    reg4c_residual_wz_a_mm_s=20.0,
    reg5_Vp_range_mm_s=[0.0, Vmax5 * 1e3], reg5_Vp_over_vstr=[0.0, Vmax5 / v_str],
)

# ----------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "lines.linewidth": 1.8})

CP = "tab:green"   # colour for the right-axis (counterpart) curve


def _dual_legend(axL, axR, loc="center right", fs=7):
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=fs, loc=loc)


def _twin(ax, x, y, ylabel, label):
    """Add a right-axis counterpart curve (dotted, colour CP)."""
    axr = ax.twinx()
    axr.plot(x, y, ":", color=CP, lw=2.2, label=label)
    axr.set_ylabel(ylabel, color=CP)
    axr.tick_params(axis="y", labelcolor=CP)
    return axr

# ---- Fig 0 : the gate R(xi) ------------------------------------------
xg = np.linspace(0, 5, 400)
fig0, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(xg, R_gate(xg), color="tab:purple")
ax.axvline(1.0, ls="--", color="0.6", lw=1)
ax.axhline(R_gate(1.0), ls=":", color="0.6", lw=1)
ax.annotate(f"R(1) = {R_gate(1.0):.3f}", xy=(1.0, R_gate(1.0)),
            xytext=(1.6, 0.35), arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlabel(r"$\xi = |z|/\delta^{*}$")
ax.set_ylabel(r"$R(\xi)$  (Mindlin slip-annulus fraction)")
ax.set_title("Spin-coupling gate")
ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
fig0.tight_layout(); fig0.savefig(os.path.join(OUTDIR, "fig0_gate.png"))

# ---- Fig 1 : Regime 1/2  (top: translation, bottom: rotation) --------
#      Right axis of each panel = the SAME coupled run's counterpart channel
#      (coupled scenario: ||Vp||=5 mm/s AND |wz|a=5 mm/s).
fig1, ((axa, axb), (axar, axbr)) = plt.subplots(2, 2, figsize=(10.2, 7.2))
axa.plot(t1a * 1e3, xi1a, label="spin-free")
axa.plot(t1b * 1e3, xi1b, "--", label=r"spin-on $|\omega_z|a=5$ mm/s")
axa.axhline(1.0, ls=":", color="0.6")
axa.set_xlabel("t [ms]"); axa.set_ylabel(r"$\xi=z/\delta^{*}$")
axa.set_title("Translation: bristle build-up (zero ICs)"); axa.grid(alpha=0.3)
axa_r = _twin(axa, t1rb * 1e3, xr1b, r"$\xi_{\psi}$ (counterpart)",
              r"$\xi_{\psi}$ | coupled")
_dual_legend(axa, axa_r)

axb.plot(t1a * 1e3, f1a, label="spin-free")
axb.plot(t1b * 1e3, f1b, "--", label="spin-on")
axb.set_xlabel("t [ms]"); axb.set_ylabel(r"$f=F/(\mu_s N)$")
axb.set_title(r"Translation force: jump $\tau/T$ then $f\to\xi$"); axb.grid(alpha=0.3)
axb_r = _twin(axb, t1rb * 1e3, m1b, r"$m$ (counterpart)", r"$m$ | coupled")
_dual_legend(axb, axb_r)

axar.plot(t1ra * 1e3, xr1a, color="tab:red", label="transl-free (pure spin)")
axar.plot(t1rb * 1e3, xr1b, "--", color="tab:brown", label=r"transl-on $\|V_p\|=5$ mm/s")
axar.axhline((3 * np.pi / 16), ls=":", color="0.6")
axar.set_xlabel("t [ms]"); axar.set_ylabel(r"$\xi_{\psi}=z_{\psi}/\delta^{*}_{\psi}$")
axar.set_title(r"Drilling: bristle build-up (ceiling $\frac{3\pi}{16}\frac{g}{\mu_s}$)")
axar.grid(alpha=0.3)
axar_r = _twin(axar, t1rb * 1e3, xi1b, r"$\xi$ (counterpart)", r"$\xi$ | coupled")
_dual_legend(axar, axar_r)

axbr.plot(t1ra * 1e3, m1a, color="tab:red", label="transl-free")
axbr.plot(t1rb * 1e3, m1b, "--", color="tab:brown", label="transl-on")
axbr.set_xlabel("t [ms]"); axbr.set_ylabel(r"$m=M_z/(\mu_s N a)$")
axbr.set_title(r"Drilling torque: $m=\xi_{\psi}+\tau\dot\xi_{\psi}$"); axbr.grid(alpha=0.3)
axbr_r = _twin(axbr, t1rb * 1e3, f1b, r"$f$ (counterpart)", r"$f$ | coupled")
_dual_legend(axbr, axbr_r)
fig1.tight_layout(); fig1.savefig(os.path.join(OUTDIR, "fig1_regime12.png"))

# ---- Fig 2 : Regime 3  (top: translation, bottom: rotation) ----------
#      Right axis = same-drive counterpart channel (Fig-2 counterpart runs).
fig2, ((axc, axd), (axcr, axdr)) = plt.subplots(2, 2, figsize=(10.2, 7.2))
axc.plot(Vp3(t3a) * 1e3, f3a, label="spin-free")
axc.plot(Vp3(t3b) * 1e3, f3b, "--", label=r"spin-on $|\omega_z|a=50$ mm/s")
axc.axhline(1.0 / r_s, ls=":", color="0.6"); axc.text(200, 1/r_s + 0.004, r"$1/r_s$", fontsize=8)
axc.set_xlabel(r"$\|V_p\|$ [mm/s]"); axc.set_ylabel(r"$f=F/(\mu_s N)$")
axc.set_title("Translation: stick$\\to$slip force$-$velocity"); axc.grid(alpha=0.3)
axc_r = _twin(axc, Vp3(t3rc) * 1e3, m3c, r"$m$ (counterpart)", r"$m$ | coupled")
_dual_legend(axc, axc_r, loc="lower right")

axd.plot(t3a, xi3a, label="spin-free"); axd.plot(t3b, xi3b, "--", label="spin-on")
axd.axhline(1.0, ls=":", color="0.6")
axd.set_xlabel("t [s]"); axd.set_ylabel(r"$\xi=z/\delta^{*}$")
axd.set_title("Translation: bristle deflection during ramp"); axd.grid(alpha=0.3)
axd_r = _twin(axd, t3rc, xr3c, r"$\xi_{\psi}$ (counterpart)", r"$\xi_{\psi}$ | coupled")
_dual_legend(axd, axd_r, loc="lower right")

axcr.plot(awz3a * 1e3, m3a, color="tab:red", label="transl-free")
axcr.plot(awz3b * 1e3, m3b, "--", color="tab:brown", label=r"transl-on $\|V_p\|=50$ mm/s")
axcr.axhline((3 * np.pi / 16) / r_s, ls=":", color="0.6")
axcr.text(200, (3*np.pi/16)/r_s + 0.004, r"$\frac{3\pi}{16}\frac{1}{r_s}$", fontsize=8)
axcr.set_xlabel(r"$a\,|\omega_z|$ [mm/s]"); axcr.set_ylabel(r"$m=M_z/(\mu_s N a)$")
axcr.set_title("Drilling: spin$\\to$slip torque$-$spin rate"); axcr.grid(alpha=0.3)
axcr_r = _twin(axcr, awz3d * 1e3, f3d, r"$f$ (counterpart)", r"$f$ | coupled")
_dual_legend(axcr, axcr_r, loc="lower right")

axdr.plot(t3ra, xr3a, color="tab:red", label="transl-free")
axdr.plot(t3rb, xr3b, "--", color="tab:brown", label="transl-on")
axdr.axhline((3 * np.pi / 16), ls=":", color="0.6")
axdr.set_xlabel("t [s]"); axdr.set_ylabel(r"$\xi_{\psi}=z_{\psi}/\delta^{*}_{\psi}$")
axdr.set_title("Drilling: bristle deflection during ramp"); axdr.grid(alpha=0.3)
axdr_r = _twin(axdr, t3d, xi3d, r"$\xi$ (counterpart)", r"$\xi$ | coupled")
_dual_legend(axdr, axdr_r, loc="lower right")
fig2.tight_layout(); fig2.savefig(os.path.join(OUTDIR, "fig2_regime3.png"))

# ---- Fig 3 : Regime 4 recovery + freezing ----------------------------
fig3, (axe, axf) = plt.subplots(1, 2, figsize=(9.4, 3.6))
axe.plot(t4a * 1e3, xi4a, color="tab:green")
axe.axhline(1.0, ls=":", color="0.6")
axe.axvline(200, ls="--", color="0.7", lw=1); axe.text(205, 0.93, "stop", fontsize=8)
axe.set_xlabel("t [ms]"); axe.set_ylabel(r"$\xi=z/\delta^{*}$")
axe.set_title("Regime 4: slip$\\to$stick recovery + freeze"); axe.grid(alpha=0.3)
axf.plot(t4a * 1e3, f4a, color="tab:green")
axf.axvline(200, ls="--", color="0.7", lw=1)
axf.set_xlabel("t [ms]"); axf.set_ylabel(r"$f=F/(\mu_s N)$")
axf.set_title("Residual stick force after stop"); axf.grid(alpha=0.3)
fig3.tight_layout(); fig3.savefig(os.path.join(OUTDIR, "fig3_regime4.png"))

# ---- Fig 4 : Regime 4b/4c  rate gating + residual-spin no-freeze -----
fig4, (axg, axh) = plt.subplots(1, 2, figsize=(9.4, 3.6))
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sweep_curves)))
for (td, tt, xx), c in zip(sweep_curves, cmap):
    axg.plot((tt - td) * 1e3, xx, color=c, label=f"{td*1e3:.0f} ms")
axg.axhline(1.0, ls=":", color="0.6")
axg.set_xlabel("t - t_stop [ms]"); axg.set_ylabel(r"$\xi=z/\delta^{*}$")
axg.set_title("Regime 4b: recovery vs stop time (Stribeck-rate gating)")
axg.legend(title="decel time", fontsize=7, ncol=2); axg.grid(alpha=0.3)
axg.set_xlim(-50, 300)

axh.plot(t4a * 1e3, xi4a, color="tab:green", label=r"$\omega_z=0$ (freezes)")
axh.plot(t4c * 1e3, xi4c, "--", color="tab:red", label=r"residual $|\omega_z|a=20$ mm/s")
axh.axvline(200, ls="--", color="0.7", lw=1); axh.text(205, 0.83, "stop", fontsize=8)
axh.set_xlabel("t [ms]"); axh.set_ylabel(r"$\xi=z/\delta^{*}$")
axh.set_title("Regime 4c: residual spin prevents freeze"); axh.grid(alpha=0.3)
axh_r = _twin(axh, t4cr * 1e3, xr4c, r"$\xi_{\psi}$ (counterpart)", r"$\xi_{\psi}$ | coupled")
_dual_legend(axh, axh_r, loc="lower right")
fig4.tight_layout(); fig4.savefig(os.path.join(OUTDIR, "fig4_regime4_gating.png"))

# ---- Fig 5 : Regime 5  stick-slip-stick-slip (2x2: full + zoomed) -----
fig5, ((axi, axj), (axiz, axjz)) = plt.subplots(2, 2, figsize=(9.8, 7.4))

# (a) time series: xi and f on left axis, |Vp| on right axis (full range)
axi.plot(t5 * 1e3, xi5, color="tab:blue", label=r"$\xi=z/\delta^{*}$")
axi.plot(t5 * 1e3, f5, color="tab:orange", label=r"$f=F/(\mu_s N)$")
axi.axhline(1.0, ls=":", color="0.6")
axi.axhline(1.0 / r_s, ls=":", color="0.5")
axi.set_xlabel("t [ms]"); axi.set_ylabel(r"$\xi,\ f$")
axi.set_title("Regime 5: stick$\\to$slip$\\to$stick$\\to$slip")
axi.grid(alpha=0.3)
axk = axi.twinx()
axk.plot(t5 * 1e3, Vp5arr * 1e3, color="0.55", lw=1.2, ls="-.",
         label=r"$\|V_p\|$ [mm/s]")
axk.set_ylabel(r"$\|V_p\|$ [mm/s]", color="0.4")
axk.tick_params(axis="y", labelcolor="0.4")
lines1, lab1 = axi.get_legend_handles_labels()
lines2, lab2 = axk.get_legend_handles_labels()
axi.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="center right")

# (b) hysteresis loop in the f-|Vp| plane (full range)
sc = axj.scatter(Vp5arr * 1e3, f5, c=t5 * 1e3, cmap="plasma", s=6)
axj.axhline(1.0 / r_s, ls=":", color="0.5"); axj.text(105, 1/r_s + 0.004, r"$1/r_s$", fontsize=8)
axj.set_xlabel(r"$\|V_p\|$ [mm/s]"); axj.set_ylabel(r"$f=F/(\mu_s N)$")
axj.set_title("Force$-$velocity hysteresis loop"); axj.grid(alpha=0.3)
cb = fig5.colorbar(sc, ax=axj, pad=0.02); cb.set_label("t [ms]", fontsize=8)

# (c) zoomed time series (y in [0.8, 1.0], xi & f only)
axiz.plot(t5 * 1e3, xi5, color="tab:blue", label=r"$\xi$")
axiz.plot(t5 * 1e3, f5, color="tab:orange", label=r"$f$")
axiz.axhline(1.0, ls=":", color="0.6"); axiz.axhline(1.0 / r_s, ls=":", color="0.5")
axiz.set_ylim(0.8, 1.0)
axiz.set_xlabel("t [ms]"); axiz.set_ylabel(r"$\xi,\ f$ (zoom)")
axiz.set_title("Zoom: $y\\in[0.8,1.0]$"); axiz.legend(fontsize=8); axiz.grid(alpha=0.3)

# (d) zoomed hysteresis loop (y in [0.8, 1.0])
scz = axjz.scatter(Vp5arr * 1e3, f5, c=t5 * 1e3, cmap="plasma", s=6)
axjz.axhline(1.0 / r_s, ls=":", color="0.5"); axjz.text(105, 1/r_s + 0.003, r"$1/r_s$", fontsize=8)
axjz.set_ylim(0.8, 1.0)
axjz.set_xlabel(r"$\|V_p\|$ [mm/s]"); axjz.set_ylabel(r"$f$ (zoom)")
axjz.set_title("Zoom: $y\\in[0.8,1.0]$"); axjz.grid(alpha=0.3)
cbz = fig5.colorbar(scz, ax=axjz, pad=0.02); cbz.set_label("t [ms]", fontsize=8)
fig5.tight_layout(); fig5.savefig(os.path.join(OUTDIR, "fig5_regime5_cycle.png"))

# ----------------------------------------------------------------------
# console dump (captured into the markdown by hand / reference)
# ----------------------------------------------------------------------
import json
def _fmt(o):
    if isinstance(o, dict):  return {k: _fmt(v) for k, v in o.items()}
    if isinstance(o, list):  return [_fmt(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 6)
    if isinstance(o, (np.integer, int)):    return int(o)
    if isinstance(o, tuple): return [_fmt(v) for v in o]
    return o
print(json.dumps(_fmt(results), indent=2))
print("\nSaved figures to:", OUTDIR)
