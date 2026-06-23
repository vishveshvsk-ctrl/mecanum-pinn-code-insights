# -*- coding: utf-8 -*-
"""
Figure generator for the Mecanum-PINN academic slide deck.
Emits self-contained SVGs (text rendered as vector paths) into ./assets/.
Palette: Academic Navy (navy + gold + steel).
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Wedge, Arc, FancyArrow, Polygon
from matplotlib.lines import Line2D

# ---------- palette ----------
NAVY  = "#10243E"; NAVY2 = "#1B3A5B"; STEEL = "#3E6691"; STEELL = "#7E9BBE"
GOLD  = "#B07D2B"; GOLDL = "#D8AE5C"; INK = "#1C2530"
LIGHT = "#EEF2F7"; PAPER = "#FFFFFF"; GRID = "#C9D3DF"
RED   = "#9B2D30"; GREEN = "#2E6B4F"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "path",     # embed glyphs as paths -> self-contained
    "axes.edgecolor": NAVY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 1.0, "figure.dpi": 100,
})

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

def save(fig, name):
    p = os.path.join(ASSETS, name)
    fig.savefig(p, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)

# platform geometry (base.toml / nd711)
h, l, R, Rd = 0.235, 0.15, 0.05, 0.0355

# =====================================================================
# 1) Roller handoff sawtooth  (12 rollers vs paper's 6)
# =====================================================================
def fig_roller_handoff():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    th = np.linspace(0, 360, 4000)
    ideal12 = np.mod(th + 15, 30) - 15
    ideal6  = np.mod(th + 30, 60) - 30
    K = 60.0; thr = np.deg2rad(th)
    smooth = np.degrees(np.arctan2(K*np.sin(12*thr), K*np.cos(12*thr)+1)/12)
    ax.plot(th, ideal6,  color=STEELL, lw=1.4, ls=(0,(5,3)), label=r"6 rollers / wheel  (paper: $\pm30^\circ$, $60^\circ$ period)")
    ax.plot(th, ideal12, color=NAVY,  lw=1.3, ls=(0,(1,1.4)), label=r"12 rollers / wheel  ideal ($\pm15^\circ$, $30^\circ$)")
    ax.plot(th, smooth,  color=GOLD,  lw=2.6, label=r"12 rollers smooth $C^\infty$ ($\tanh$-peak, $K{=}60$)")
    ax.axhline(15, color=GRID, lw=0.8); ax.axhline(-15, color=GRID, lw=0.8)
    ax.set_xlim(0, 360); ax.set_ylim(-34, 34)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel(r"wheel angle  $\theta$  (deg)")
    ax.set_ylabel(r"contacting-roller angle  $\tilde\varphi$  (deg)")
    ax.set_title(r"Roller contact hand-off: $\tilde\varphi(\theta)=\bigl((\theta+15^\circ)\bmod 30^\circ\bigr)-15^\circ$",
                 color=NAVY, fontsize=11, pad=8)
    # annotate one handoff
    ax.annotate("hand-off\n(jump)", xy=(30, 14.0), xytext=(52, 26),
                fontsize=8.5, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate("", xy=(30, -14.5), xytext=(30, 14.5),
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0, ls=(0,(2,2))))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.46), ncol=1,
              frameon=False, fontsize=8.6, handlelength=2.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    save(fig, "fig_roller_handoff.svg")

# =====================================================================
# 2) O vs X roller configuration (top view)
# =====================================================================
def _draw_platform(ax, deltas, title):
    # body
    ax.add_patch(Rectangle((-h, -l), 2*h, 2*l, fill=True, fc=LIGHT, ec=NAVY, lw=1.6, zorder=1, joinstyle="round"))
    centers = [( h, l), ( h,-l), (-h, l), (-h,-l)]  # FL FR RL RR  (X fwd=right, Y left=up)
    wx, wy = 0.085, 0.052
    for (cx, cy), d in zip(centers, deltas):
        ax.add_patch(Rectangle((cx-wx, cy-wy), 2*wx, 2*wy, fill=True, fc="#DDE5EE", ec=NAVY, lw=1.2, zorder=2))
        ang = np.deg2rad(d)
        dirv = np.array([np.cos(ang), np.sin(ang)])
        # 3 roller stripes across the wheel, perpendicular offset
        perp = np.array([-np.sin(ang), np.cos(ang)])
        for off in (-0.55, 0.0, 0.55):
            c = np.array([cx, cy]) + perp*off*wy*1.4
            p1 = c - dirv*wx*0.95; p2 = c + dirv*wx*0.95
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=GOLD, lw=2.2, solid_capstyle="round", zorder=3)
    # body axes
    ax.annotate("", xy=(h*0.55, 0), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    ax.annotate("", xy=(0, l*0.85), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    ax.text(h*0.58, -0.018, r"$X$ (fwd)", fontsize=8.5, color=NAVY, va="top")
    ax.text(0.012, l*0.86, r"$Y$", fontsize=8.5, color=NAVY)
    ax.text(0.0, -0.012, r"$O$", fontsize=8, color=NAVY, ha="right", va="top")
    ax.set_title(title, color=NAVY, fontsize=10.5, pad=6)
    ax.set_xlim(-h-0.1, h+0.1); ax.set_ylim(-l-0.12, l+0.12)
    ax.set_aspect("equal"); ax.axis("off")

def fig_oconfig():
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9))
    _draw_platform(axs[0], [-45, 45, 45, -45], r"O-configuration   $\delta=(-,+,+,-)\,45^\circ$")
    _draw_platform(axs[1], [ 45,-45,-45, 45], r"X-configuration   $\delta=(+,-,-,+)\,45^\circ$")
    fig.text(0.5, -0.02, "Roller axle orientation (gold) on each wheel; sign pattern of $\\delta_i$ sets the lateral / yaw coupling.",
             ha="center", fontsize=8.4, color=INK)
    save(fig, "fig_oconfig.svg")

# =====================================================================
# 3) Friction circle: F_par / F_perp decomposition
# =====================================================================
def fig_friction_circle():
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    Rc = 1.0
    ax.add_patch(Circle((0,0), Rc, fill=True, fc=LIGHT, ec=NAVY, lw=2.0, zorder=1))
    # available radius mu N
    ax.annotate("", xy=(Rc*np.cos(np.deg2rad(35)), Rc*np.sin(np.deg2rad(35))), xytext=(0,0),
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.0, ls=(0,(3,3))))
    ax.text(0.40, 0.42, r"$\mu N_i$", fontsize=11, color=NAVY, rotation=35)
    # axes e_hat (axle, F_par) horizontal ; n_hat (rolling, F_perp) vertical
    ax.annotate("", xy=(1.28,0), xytext=(-1.28,0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("", xy=(0,1.28), xytext=(0,-1.28), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(1.30, 0.03, r"$\hat e_i$  (axle, $F_\parallel$: motor)", fontsize=8.6, va="bottom")
    ax.text(0.03, 1.30, r"$\hat n_i$  (roll, $F_\perp$: roller drag)", fontsize=8.6)
    # velocity-slaved F_perp consumes radius
    Fp = 0.62
    ax.add_patch(Rectangle((-1.18, -0.012), 2.36, Fp+0.012, fill=True, fc=GOLDL, alpha=0.30, ec="none", zorder=0))
    ax.annotate("", xy=(0, Fp), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3.0))
    ax.text(0.06, Fp*0.6, r"$F_\perp$", color=GOLD, fontsize=11)
    # headroom F_par
    Fpar = np.sqrt(Rc**2 - Fp**2)
    ax.annotate("", xy=(Fpar, Fp), xytext=(0, Fp), arrowprops=dict(arrowstyle="-|>", color=STEEL, lw=3.0))
    ax.text(Fpar*0.5, Fp+0.07, r"$F_\parallel$", color=STEEL, fontsize=11)
    # resultant
    ax.annotate("", xy=(Fpar, Fp), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    ax.plot([Fpar],[Fp], marker="o", color=NAVY, ms=4)
    ax.text(-1.15, 0.30, "velocity-slaved\ndrag band", fontsize=8.0, color=GOLD)
    ax.set_title(r"Per-contact friction circle:  $F_\parallel^2+F_\perp^2\le(\mu N_i)^2$",
                 color=NAVY, fontsize=10.5, pad=8)
    ax.set_xlim(-1.45,1.55); ax.set_ylim(-1.4,1.5); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "fig_friction_circle.svg")

# =====================================================================
# 4) Trackable velocity envelope (Vx,Vy) anisotropic superellipse
# =====================================================================
def fig_velocity_envelope():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    Vxc, Vyc, n = 4.55, 0.63, 1.5
    t = np.linspace(0, 2*np.pi, 800)
    cx = np.sign(np.cos(t))*(np.abs(np.cos(t))**(2.0/n))*Vxc
    cy = np.sign(np.sin(t))*(np.abs(np.sin(t))**(2.0/n))*Vyc
    ax.fill(cx, cy, color=STEEL, alpha=0.16, zorder=0)
    ax.plot(cx, cy, color=NAVY, lw=2.2, zorder=2, label="trackable envelope")
    # caps
    ax.axvline( Vxc, color=GOLD, lw=1.4, ls=(0,(6,3))); ax.axvline(-Vxc, color=GOLD, lw=1.4, ls=(0,(6,3)))
    ax.axhline( Vyc, color=RED,  lw=1.4, ls=(0,(4,3))); ax.axhline(-Vyc, color=RED,  lw=1.4, ls=(0,(4,3)))
    ax.text(Vxc-0.05, 0.18, r"$V_{x,\mathrm{cap}}\approx4.55$"+"\ntorque (soft)", color=GOLD, fontsize=8.4, ha="right")
    ax.text(0.1, Vyc+0.03, r"$V_{y,\mathrm{crit}}\approx0.63$  friction (hard)", color=RED, fontsize=8.4, va="bottom")
    ax.set_xlabel(r"$V_x$  (m/s)"); ax.set_ylabel(r"$V_y$  (m/s)")
    ax.set_xlim(-5.2,5.2); ax.set_ylim(-1.15,1.15)
    ax.set_title(r"Translational envelope ($\dot\psi=0$): anisotropy $V_{x,\mathrm{cap}}/V_{y,\mathrm{crit}}\approx7$",
                 color=NAVY, fontsize=10, pad=6)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    save(fig, "fig_velocity_envelope.svg")

# =====================================================================
# 5) Per-trajectory figures: motion (left) + friction-circle loadout (right)
# =====================================================================
def _circle_base(ax):
    ax.add_patch(Circle((0,0), 1.0, fill=True, fc=LIGHT, ec=NAVY, lw=1.6, zorder=1))
    ax.annotate("", xy=(1.18,0), xytext=(-1.18,0), arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9))
    ax.annotate("", xy=(0,1.18), xytext=(0,-1.18), arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9))
    ax.text(1.16,0.04,r"$F_\parallel$",fontsize=8,color=INK)
    ax.text(0.04,1.16,r"$F_\perp$",fontsize=8,color=INK)
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); ax.axis("off")

def _arrow(ax, x, y, color=NAVY, lw=2.4):
    ax.annotate("", xy=(x,y), xytext=(0,0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

def _traj_fig(name, motion_fn, load_fn, mtitle, ctitle):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.95), gridspec_kw={"width_ratios":[1.25,1]})
    motion_fn(axs[0]); axs[0].set_title(mtitle, color=NAVY, fontsize=10, pad=5)
    _circle_base(axs[1]); load_fn(axs[1]); axs[1].set_title(ctitle, color=NAVY, fontsize=10, pad=5)
    save(fig, name)

def _setm(ax, lim=1.25, eq=True):
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    if eq: ax.set_aspect("equal")
    ax.axis("off")

# octagon
def _m_oct(ax):
    for k in range(8):
        a = k*np.pi/4
        ax.annotate("", xy=(0.95*np.cos(a),0.95*np.sin(a)), xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.0))
    # lateral wiggle glyph on one leg
    a=0; tt=np.linspace(0,1,60); wig=0.12*np.sin(6*np.pi*tt)
    ax.plot(0.95*tt, wig, color=GOLD, lw=1.6)
    _setm(ax)
def _c_oct(ax):
    for k in range(8):
        a=k*np.pi/4; _arrow(ax,0.45*np.cos(a),0.45*np.sin(a),color=STEEL,lw=1.7)
    _arrow(ax,0.0,0.5,color=GOLD,lw=2.2)
    ax.text(0,-1.18,"azimuthal sweep,\nlow utilization",ha="center",fontsize=8,color=INK)

# long_circle
def _m_circ(ax):
    t=np.linspace(0,2*np.pi,200); ax.plot(np.cos(t),np.sin(t),color=NAVY,lw=2.2)
    for a in np.linspace(0,2*np.pi,8,endpoint=False):
        tx,ty=-np.sin(a),np.cos(a)
        ax.annotate("",xy=(np.cos(a)+0.22*tx,np.sin(a)+0.22*ty),xytext=(np.cos(a),np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=1.5))
    _setm(ax)
def _c_circ(ax):
    _arrow(ax,0.66,0.0,color=STEEL,lw=3.0); ax.text(0.2,0.1,r"$F_\parallel$",color=STEEL,fontsize=9)
    _arrow(ax,0.0,0.2,color=GOLD,lw=2.0)
    ax.text(0,-1.18,"forward-rolling,\nsmall $F_\\perp$ (low load)",ha="center",fontsize=8,color=INK)

# spin_creep
def _m_spin(ax):
    ax.add_patch(Circle((0,0),0.07,fc=NAVY,ec="none"))
    ax.annotate("",xy=(0.28,0.0),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=STEEL,lw=1.6))
    ax.add_patch(Arc((0,0),1.7,1.7,theta1=20,theta2=320,color=GOLD,lw=2.6))
    a=np.deg2rad(320); ax.annotate("",xy=(0.85*np.cos(a),0.85*np.sin(a)),
        xytext=(0.85*np.cos(np.deg2rad(330)),0.85*np.sin(np.deg2rad(330))),
        arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=2.6))
    ax.text(0,-1.12,r"high $|\Omega|$, creep $|V|$",ha="center",fontsize=8.5,color=INK)
    _setm(ax)
def _c_spin(ax):
    ax.add_patch(Rectangle((-1.0,-0.02),2.0,0.92,fc=GOLDL,alpha=0.30,ec="none"))
    _arrow(ax,0.0,0.9,color=GOLD,lw=3.2); ax.text(0.06,0.55,r"$F_\perp$",color=GOLD,fontsize=10)
    _arrow(ax,0.32,0.9,color=STEEL,lw=1.6)
    ax.text(0,-1.18,"spin-dominated:\n$F_\\perp$ fills circle ($\\to\\dot\\psi_{\\max}$)",ha="center",fontsize=8,color=INK)

# coupled_vomega
def _m_cv(ax):
    t=np.linspace(0,1.6,200); x=t-0.5; y=0.45*np.sin(1.6*t)
    ax.plot(x*1.0,y+0.0,color=NAVY,lw=2.4)
    ax.annotate("",xy=(x[-1],y[-1]),xytext=(x[-5],y[-5]),arrowprops=dict(arrowstyle="-|>",color=NAVY,lw=2.4))
    ax.text(0,-1.12,r"independent $|V|,\,\Omega$ ramps",ha="center",fontsize=8.5,color=INK)
    _setm(ax)
def _c_cv(ax):
    _arrow(ax,0.6,0.55,color=NAVY,lw=2.8)
    ax.text(0.62,0.5,"(V,Ω) ray",color=NAVY,fontsize=8.5)
    ax.text(0,-1.18,"rides a $(F_\\parallel,F_\\perp)$ ray\n(interior coupling)",ha="center",fontsize=8,color=INK)

# spiral_orbit
def _m_spiral(ax):
    t=np.linspace(0,3.2*2*np.pi,800); r=0.12+t/(3.2*2*np.pi)*0.95
    ax.plot(r*np.cos(t),r*np.sin(t),color=NAVY,lw=2.0)
    ax.annotate("",xy=(r[-1]*np.cos(t[-1]),r[-1]*np.sin(t[-1])),
                xytext=(r[-6]*np.cos(t[-6]),r[-6]*np.sin(t[-6])),
                arrowprops=dict(arrowstyle="-|>",color=NAVY,lw=2.0))
    _setm(ax)
def _c_spiral(ax):
    ax.add_patch(Arc((0,0),2.0,2.0,theta1=8,theta2=82,color=RED,lw=3.0))
    _arrow(ax,0.92*np.cos(np.deg2rad(45)),0.92*np.sin(np.deg2rad(45)),color=NAVY,lw=2.4)
    ax.text(0,-1.18,"sustained near-boundary\n(V-, Ω-, iso-accel marginals)",ha="center",fontsize=8,color=INK)

# multisine
def _m_ms(ax):
    t=np.linspace(0,2*np.pi,1000)
    x=0.5*np.sin(3*t+0.4)+0.45*np.sin(5*t)+0.2*np.sin(8*t+1)
    y=0.5*np.sin(2*t)+0.4*np.sin(7*t+0.7)+0.2*np.sin(11*t)
    ax.plot(x,y,color=NAVY,lw=1.3)
    _setm(ax,lim=1.25)
    ax.text(0,-1.18,"zero-mean harmonic sums",ha="center",fontsize=8.5,color=INK)
def _c_ms(ax):
    rng=np.random.default_rng(3)
    for _ in range(11):
        a=rng.uniform(0,2*np.pi); m=rng.uniform(0.2,0.8)
        _arrow(ax,m*np.cos(a),m*np.sin(a),color=STEEL,lw=1.3)
    ax.text(0,-1.18,"broadband transient load\n(persistent excitation)",ha="center",fontsize=8,color=INK)

# ellipse
def _m_ell(ax):
    t=np.linspace(0,2*np.pi,300); ax.plot(1.05*np.cos(t),0.55*np.sin(t),color=NAVY,lw=2.2)
    # crab heading arrows (fixed)
    for tt in np.linspace(0,2*np.pi,7,endpoint=False):
        x,y=1.05*np.cos(tt),0.55*np.sin(tt)
        ax.annotate("",xy=(x+0.18,y),xytext=(x,y),arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=1.3))
    _setm(ax)
    ax.text(0,-1.12,"world-position tracked\n(tangent / crab heading)",ha="center",fontsize=8,color=INK)
def _c_ell(ax):
    _arrow(ax,0.0,0.78,color=GOLD,lw=3.0); ax.text(0.06,0.5,r"crab $F_\perp$",color=GOLD,fontsize=8.5)
    _arrow(ax,0.7,0.0,color=STEEL,lw=2.4); ax.text(0.2,-0.16,r"tangent $F_\parallel$",color=STEEL,fontsize=8.5)
    ax.text(0,-1.18,"crab probes lateral wall;\ntangent forward-drives",ha="center",fontsize=8,color=INK)

def fig_trajectories():
    _traj_fig("fig_traj_octagon.svg", _m_oct, _c_oct,
              "Octagon: start–cruise–stop legs (8 dirs)", "friction-circle loadout")
    _traj_fig("fig_traj_long_circle.svg", _m_circ, _c_circ,
              "Long circle: heading-aligned orbit", "friction-circle loadout")
    _traj_fig("fig_traj_spin_creep.svg", _m_spin, _c_spin,
              "Spin-creep: yaw pulses + creep", "friction-circle loadout")
    _traj_fig("fig_traj_coupled_vomega.svg", _m_cv, _c_cv,
              "Coupled V–Ω: independent ramps", "friction-circle loadout")
    _traj_fig("fig_traj_spiral_orbit.svg", _m_spiral, _c_spiral,
              "Spiral orbit: heading-locked spiral", "friction-circle loadout")
    _traj_fig("fig_traj_multisine.svg", _m_ms, _c_ms,
              "Multisine: zippered harmonic combs", "friction-circle loadout")
    _traj_fig("fig_traj_ellipse.svg", _m_ell, _c_ell,
              "Ellipse: position-tracked orbit", "friction-circle loadout")

if __name__ == "__main__":
    fig_roller_handoff()
    fig_oconfig()
    fig_friction_circle()
    fig_velocity_envelope()
    fig_trajectories()
    print("ALL FIGURES DONE ->", ASSETS)
