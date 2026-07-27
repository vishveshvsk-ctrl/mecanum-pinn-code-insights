# Evolution of the Friction Force and Bristle State in ODE Simulation

*Scaled (δ\*-normalised) analysis of the modified LuGre law with the ε/ξ-gated
spin coupling. Companion code: `lugre_scaled_gated_regimes.py`; figures
`fig0_gate.png`, `fig1_regime12.png`, `fig2_regime3.png`, `fig3_regime4.png`,
`fig4_regime4_gating.png`, `fig5_regime5_cycle.png` (all in this folder).*

---

## Scaled form of the model

Having defined each term of the modified LuGre law, we now examine how the
internal bristle state and the tangential force evolve through the four
canonical phases of the stick–slip cycle. It is convenient to normalise the
bristle deflection by the break-away deflection and the force by the static
friction capacity, using the two calibration identities already established,

$$
\delta^{*} = \frac{\mu_s}{\sigma_0}, \qquad \tau = \frac{\sigma_1}{\sigma_0},
$$

so that with the scaled state $\xi \equiv z/\delta^{*}$ the per-roller state
equation and force reduce to

$$
\boxed{\;\dot z = V_{\mathrm{slip}} - V_{\mathrm{eff}}\,\frac{z}{\delta^{*}}\,
\frac{\mu_s}{g(V_{\mathrm{eff}})}\;}
\qquad
\boxed{\;f \equiv \frac{F}{\mu_s N_i}
= \frac{z}{\delta^{*}} + \tau\,\frac{\dot z}{\delta^{*}}
= \xi + \tau\,\dot\xi\;}
$$

with the dimensionless Stribeck ratio
$g(s)=\mu_c\!\left[\,1+(r_s-1)\,e^{-(s/v_{\mathrm{str}})^2}\right]$,
$r_s=\mu_s/\mu_c$.

Crucially, the driving speed is **not** the raw translational slip
$V_{\mathrm{slip}}=\lVert V_p\rVert$ but the Adamov *pressure-weighted mean
slip*, in which the spin cross-coupling is gated by the Mindlin slip-annulus
area fraction $R(\xi)$ evaluated at the **same** normalised deflection
$\xi=|z|/\delta^{*}$ that scales the force:

$$
V_{\mathrm{eff}} = \lVert V_p\rVert + R(\xi)\,\frac{8}{3\pi}\,|\omega_z|\,a,
\qquad
R(\xi) = 1-\bigl(1-\xi_{\mathrm{safe}}\bigr)^{2/3},
\qquad
\xi_{\mathrm{safe}} = \frac{\xi}{\sqrt{1+\xi^{2}}} .
$$

The native translation term is unramped; only the spin contribution is
modulated by $R(\xi)$, which maps $[0,\infty)\!\to\![0,1)$ and is $C^{\infty}$
(the $\xi_{\mathrm{safe}}$ saturator removes the derivative singularity of the
bare Mindlin form at $\xi=1$). Because $R$ is a function of $\xi$, the state
equation is *self-gated*: the bristle winding up its own deflection
progressively unlocks the spin contribution to the effective slip.

**Gate values (from simulation).**

| regime | $\xi=z/\delta^{*}$ | $R(\xi)$ | effect on $V_{\mathrm{eff}}$ |
|---|---|---|---|
| deep stick (zero ICs) | 0.00 | **0.000** | spin coupling OFF → $V_{\mathrm{eff}}=\lVert V_p\rVert$ |
| quarter-loaded | 0.25 | 0.169 | spin admitted weakly |
| half-loaded | 0.50 | 0.326 | — |
| at break-away $\delta^{*}$ | 1.00 | **0.559** | ~half the Adamov spin slip active |
| beyond break-away | 2.00 | 0.777 | — |
| gross slip | 5.00 | 0.928 | — |
| gross slip | $\to\infty$ | $\to 1$ | full Adamov coupling, $V_{\mathrm{eff}}\to\lVert V_p\rVert+\tfrac{8}{3\pi}|\omega_z|a$ |

![Spin-coupling gate R(xi)](fig0_gate.png)

All simulations below use the worked-example calibration of the note
($\sigma_0=1.64\times10^{3}\,\mathrm{m^{-1}}$, $\sigma_1=\tau\sigma_0\approx1.64\,\mathrm{s\,m^{-1}}$
with the time constant fixed at $\tau=1.0\,\mathrm{ms}$,
$\mu_c=0.60$, $\mu_s=0.66$, $r_s=1.1$, $v_{\mathrm{str}}=0.01\,\mathrm{m\,s^{-1}}$,
$N=87\,\mathrm{N}$, $a=5\,\mathrm{mm}$ contact radius), for which
$\delta^{*}=0.402\,\mathrm{mm}$ and $\tau=1.0\,\mathrm{ms}$. The Stribeck
velocity $v_{\mathrm{str}}$ is a discretionary (empirical) parameter — the
calibration note lists it as $O(0.01\text{–}0.1\,\mathrm{m\,s^{-1}})$ for PU —
and is set here to the low end, $10\,\mathrm{mm\,s^{-1}}$; it controls how sharply
the friction softens with speed and therefore how pronounced the break-away
overshoot is (see §X.3).

**Velocity domains exercised.** With $v_{\mathrm{str}}=10\,\mathrm{mm\,s^{-1}}$ as
the reference, each regime is driven over the following ranges (spin quoted as
the tangential spin slip $|\omega_z|a$):

| regime | $\lVert V_p\rVert$ | in units of $v_{\mathrm{str}}$ | spin $|\omega_z|a$ |
|---|---|---|---|
| X.1 / X.2 presliding | $5\,\mathrm{mm\,s^{-1}}$ (const) | $0.5\,v_{\mathrm{str}}$ (presliding) | $0$ or $5\,\mathrm{mm\,s^{-1}}$ |
| X.3 stick→slip | $0\to300\,\mathrm{mm\,s^{-1}}$ (ramp) | $0\to30\,v_{\mathrm{str}}$ (crosses $v_{\mathrm{str}}$) | $0$ or $50\,\mathrm{mm\,s^{-1}}$ |
| X.4 slip→stick | $300\to0\,\mathrm{mm\,s^{-1}}$ (decel) | $30\,v_{\mathrm{str}}\to0$ | $0$; residual case $20\,\mathrm{mm\,s^{-1}}$ |

So X.1/X.2 stays in presliding ($0.5\,v_{\mathrm{str}}$); X.3 spans presliding →
Stribeck crossing → gross slip ($30\,v_{\mathrm{str}}$); and X.4 sweeps back down
through $v_{\mathrm{str}}$ to rest.

> **The pure-translation limit is recovered exactly.** When $\omega_z=0$ the
> $R$-term vanishes identically and $V_{\mathrm{eff}}\equiv\lVert V_p\rVert$;
> every result below reduces to the spin-free case. The gate only activates the
> spin correction, and always with the correct microslip behaviour ($R(0)=0$).

### The drilling (spin) channel — $\delta^{*}_{\psi}$ scaling

The contact-patch spin (drilling) degree of freedom carries its own bristle
$z_{\psi}$, and admits the *same* scaling construction as translation. With the
calibration $\sigma_{0,\psi}=\tfrac{2}{3}\sigma_0$ and
$\sigma_{1,\psi}=\tfrac{2}{3}\sigma_1$ (Reissner–Sagoci torsion, §4 of the note),
the rotational break-away deflection and time constant are

$$
\delta^{*}_{\psi} = \frac{\mu_s}{\sigma_{0,\psi}} = \tfrac{3}{2}\,\delta^{*},
\qquad
\tau_{\psi} = \frac{\sigma_{1,\psi}}{\sigma_{0,\psi}} = \tau .
$$

Writing the drilling law of Eq. (17) at the **contact edge** (multiplying the
$\mathrm{rad\,s^{-1}}$ effective slip of Eq. (12) by $a$, so every term is an edge
speed in $\mathrm{m\,s^{-1}}$) gives the exact mirror of the translational pair,
with $\xi_{\psi}\equiv z_{\psi}/\delta^{*}_{\psi}$:

$$
\dot z_{\psi} = a\omega_z - V_{\mathrm{eff},\psi}\,\frac{z_{\psi}}{\delta^{*}_{\psi}}\,
\frac{\mu_s}{g(V_{\mathrm{eff},\psi})},
\qquad
m \equiv \frac{M_z}{\mu_s N_i a} = \xi_{\psi} + \tau\,\dot\xi_{\psi},
$$

$$
V_{\mathrm{eff},\psi} = \underbrace{\tfrac{16}{3\pi}\,(a|\omega_z|)}_{\text{native drilling}}
+ \underbrace{R(\xi_{\psi})\,\cdot 5\,\lVert V_p\rVert}_{\text{$\xi_{\psi}$-gated cross-coupling}} .
$$

Two structural differences from the translational axis are worth flagging — the
"$V_{\mathrm{eff}}$ is slightly different for this axis" caveat made concrete:

1. **The roles swap.** Here **spin is the native (unramped) drive** and
   **translation $\lVert V_p\rVert$ is the $R(\xi_{\psi})$-gated cross-coupling**
   (carrying the factor $5$, not $\tfrac{8}{3\pi}$). Accordingly the spin/spin-free
   comparison of the translational panels becomes a **translation-on / translation-free
   (pure-spin)** comparison in the drilling panels.

2. **The ceiling is lower.** The native drive $a\omega_z$ enters with coefficient
   $1$, but its image in $V_{\mathrm{eff},\psi}$ carries $\tfrac{16}{3\pi}\approx1.70$.
   The steady state is therefore
   $\xi_{\psi,\mathrm{ss}} = \tfrac{3\pi}{16}\,g/\mu_s$, so the drilling bristle
   saturates at a **ceiling of $\tfrac{3\pi}{16}\approx0.589$** (quasi-static pure
   spin) rather than $1$, and at $\tfrac{3\pi}{16}/r_s\approx0.535$ in gross
   sliding. Physically, the drilling contact never mobilises as large a fraction of
   its break-away twist as translation does, because the Adamov mean slip
   over-counts the drilling motion.

All other qualitative behaviour (build-up, saturation, the cross-coupling lowering
the ceiling, freezing at rest) carries over unchanged; the drilling panels in
Figs. 1–2 (bottom rows) make the comparison explicit.

---

## X.1 Initiation from rest: all initial conditions zero (deep presliding)

With $\xi(0)=0$ and a small imposed slip speed ($\lVert V_p\rVert = 5\,\mathrm{mm\,s^{-1}}$,
i.e. $0.5\,v_{\mathrm{str}}$), the relaxation term
$-V_{\mathrm{eff}}\,\xi\,\mu_s/g$ vanishes at $\xi=0$, so initially
$\dot z\approx V_{\mathrm{slip}}$: the bristles deflect at the full relative
velocity and the contact is in **pure stick** — a tangential spring being wound
up. Moreover, because $R(0)=0$, the interface **cannot feel spin while it is
undeflected**: even with spin present ($|\omega_z|a=5\,\mathrm{mm\,s^{-1}}$,
equal to the translational slip) the build-up starts *identically* to the
spin-free case. This is precisely where the bare Adamov coupling would
over-predict, and the gate switches it off.

The force at $t=0^{+}$ is not zero: the micro-damping channel $\tau\dot\xi$
produces an immediate normalised jump

$$
f(0^{+}) = \tau\,\frac{V_{\mathrm{slip}}}{\delta^{*}} = \frac{\tau}{T},
\qquad
T = \frac{g(V_{\mathrm{eff}})}{\mu_s}\,\frac{\delta^{*}}{V_{\mathrm{eff}}}
   \;\approx\; \frac{\delta^{*}}{\lVert V_p\rVert},
$$

i.e. the force overshoots the elastic curve by the ratio of the micro-damping
relaxation time $\tau$ to the presliding fill time $T$. The simulation gives
$f(0^{+})=\tau/T=0.0124$ (fill time $T=78.9\,\mathrm{ms}$) — a $\sim1\%$
overshoot, confirming that $\sigma_1$ is a *stabiliser*, not a force
contributor. Beyond the jump the composition of $f$ migrates from 100 %
viscous ($\tau\dot\xi$ at $t=0^{+}$) to 100 % elastic ($f=\xi$ at steady
state, where $\dot\xi\to0$). This continuous exchange between the damping and
stiffness channels is the ODE-level signature of presliding, and it is what
distinguishes the LuGre contact from a Coulomb element (which is
force-indeterminate at $v\approx0$).

![Regime 1/2 build-up (top: translation, bottom: drilling)](fig1_regime12.png)

*Each panel overlays, on a dotted **right-hand axis** (green), the
**counterpart channel** taken from the same coupled run ($\|V_p\|=5$ mm/s
**and** $|\omega_z|a=5$ mm/s): the drilling bristle $\xi_{\psi}$ and torque
$m$ on the translation panels, and the translation bristle $\xi$ and force
$f$ on the drilling panels. This shows how the two independent LuGre
channels evolve side-by-side under one shared drive.*

---

## X.2 Approach to the break-away deflection δ\*

As integration proceeds, $\xi$ asymptotes to its steady value

$$
\xi_{\mathrm{ss}} = f_{\mathrm{ss}} = \frac{g(V_{\mathrm{eff}})}{\mu_s},
$$

which for $V_{\mathrm{eff}}\ll v_{\mathrm{str}}$ tends to $1$, i.e.
$z\to\delta^{*}$. The scaled steady state is therefore **bounded in a fixed,
calibration-independent interval**

$$
\frac{\mu_c}{\mu_s} = \frac{1}{r_s} \;\le\; \xi_{\mathrm{ss}} \;\le\; 1,
$$

here $[0.909,\,1.000]$. Thus $\xi=1$ (i.e. $z=\delta^{*}$) is *visibly* the
ceiling of the presliding regime, and the static→kinetic drop is simply $\xi$
falling from $1$ to $1/r_s$.

Two features deserve emphasis:

1. **Saturation, not divergence.** The relaxation term grows linearly in
   $\xi$, so $\dot\xi\to0$ smoothly as $\xi\to\xi_{\mathrm{ss}}$; there is no
   discrete "break-away event" — stick and slip are limits of one continuous
   flow. In the spin-free case at $\lVert V_p\rVert=0.5\,v_{\mathrm{str}}$ the
   simulation reaches $\xi_{\mathrm{ss}}=0.980$ and $f_{\mathrm{ss}}=0.980$
   (slightly below $1$ because $5\,\mathrm{mm\,s^{-1}}$ already sits partway up
   the Stribeck shoulder at this $v_{\mathrm{str}}$).

2. **Spin lowers the ceiling via the gate.** As $\xi$ climbs $0\to1$, the gate
   opens $R:0\to0.559$, so with spin present $V_{\mathrm{eff}}$ *grows as the
   bristle winds up*. This is a self-reinforcing feedback: more deflection →
   more effective slip → the Stribeck $g(V_{\mathrm{eff}})$ softens → the target
   $\xi_{\mathrm{ss}}=g/\mu_s$ drops. With $|\omega_z|a=5\,\mathrm{mm\,s^{-1}}$
   the state saturates at only $\xi_{\mathrm{ss}}=0.705$ — well **below** the
   spin-free ceiling — because the gate has admitted enough spin slip to push
   $g$ partway down toward $\mu_c$. The break-away deflection $\delta^{*}$
   remains a useful sanity check: at $\xi=1$ the physical deflection is
    $\sim20\%$ of the contact radius $a=5\,\mathrm{mm}$, consistent with the
   requirement that presliding travel stay a small fraction of the contact
   spot.

**Drilling channel (Fig. 1, bottom row).** Under the analogous drive
($a|\omega_z|=5\,\mathrm{mm\,s^{-1}}$ native spin), the drilling bristle
$\xi_{\psi}$ builds up with the same first-order character but saturates at
$\xi_{\psi,\mathrm{ss}}=0.561$ in the pure-spin case — close to the ceiling
$\tfrac{3\pi}{16}\,g/\mu_s\approx0.577$, well below the translational $\approx1$.
Adding the $R(\xi_{\psi})$-gated translational cross-coupling
($\lVert V_p\rVert=5\,\mathrm{mm\,s^{-1}}$) lowers it further to
$\xi_{\psi,\mathrm{ss}}=0.329$, exactly mirroring how spin lowered the
translational ceiling: more cross-slip → softer $g$ → lower target. The scaled
drilling torque tracks $m=\xi_{\psi}$ at steady state.

---

## X.3 Stick-to-slip transition

The transition is traversed by ramping $\lVert V_p\rVert$ from rest through
the Stribeck velocity ($0\to0.3\,\mathrm{m\,s^{-1}}=0\to30\,v_{\mathrm{str}}$
over $1\,\mathrm{s}$). Three phases appear in sequence: **(i) elastic loading**
($\xi$ climbs toward $1$, $f\approx\xi$); **(ii) break-away** (force peak, if
any); **(iii) frictional weakening** ($\xi$ unwinds toward the kinetic branch as
the relaxation term dominates).

**Spin-free — the overshoot is $v_{\mathrm{str}}$-dependent.** At this narrower
Stribeck ($v_{\mathrm{str}}=10\,\mathrm{mm\,s^{-1}}$) the break-away overshoot is
essentially **suppressed**: the force rises monotonically to the kinetic floor
$f=1/r_s=0.909$ (reached near $\lVert V_p\rVert\approx0.15\,\mathrm{m\,s^{-1}}$)
with $\xi_{\max}=0.909$ — no peak above the floor. The reason is a race between
two rates. Winding the bristle up to the static envelope ($\xi\to1$) requires the
long presliding fill time $T\approx\delta^{*}/\lVert V_p\rVert$, which is large at
the low speeds where $g$ is still near $\mu_s$; but with a small
$v_{\mathrm{str}}$ the friction has already softened to $\mu_c$ by the time the
ramp lifts $\lVert V_p\rVert$ into that range, so $\xi$ never catches its
(collapsing) target. Contrast the wider-Stribeck case
($v_{\mathrm{str}}=50\,\mathrm{mm\,s^{-1}}$), where the same ramp produced a
distinct peak $f=0.965$ at $\approx35\,\mathrm{mm\,s^{-1}}$. **The break-away
overshoot is therefore not intrinsic — its height is set jointly by
$v_{\mathrm{str}}$ and the loading rate**; a slower ramp (or larger
$v_{\mathrm{str}}$) restores it, exactly the rate-dependence quantified for the
reverse transition in §X.4.

This regime is still where the $\tau\dot\xi$ term earns its keep: with
$\tau=\sigma_1/\sigma_0\approx1\,\mathrm{ms}$ the damping contribution opposes
rapid changes of $\xi$ and suppresses the numerically stiff, underdamped
presliding oscillation the undamped ($\sigma_1=0$) model exhibits. Whenever a
peak *is* present (wider $v_{\mathrm{str}}$ or slower loading), the ensuing
negative force–velocity slope is the destabilising mechanism responsible for
stick–slip limit cycles in the coupled platform dynamics.

**Spin-on ($|\omega_z|a=50\,\mathrm{mm\,s^{-1}}$).** Spin drives the force
*below* the translational kinetic floor. Because $R(\xi)$ ramps with the
deflection, the effective slip carries the extra spin term and pushes
$g(V_{\mathrm{eff}})$ further down: the force rises monotonically and saturates
at $f=0.849 < 1/r_s=0.909$. Mechanically, **contact-patch spin helps break the
interface loose** and lowers the sustained tangential force — the same qualitative
effect seen at the wider $v_{\mathrm{str}}$.

**Drilling channel (Fig. 2, bottom row).** Ramping the native spin
($a|\omega_z|:0\to300\,\mathrm{mm\,s^{-1}}$) the drilling torque rises
monotonically to its gross-sliding floor $m=\tfrac{3\pi}{16}/r_s=0.535$ (pure
spin), with no overshoot — the drilling analogue of the suppressed break-away
peak, and again capped well below the translational floor $1/r_s=0.909$ by the
$\tfrac{16}{3\pi}$ over-counting. Turning on the gated translational
cross-coupling ($\lVert V_p\rVert=50\,\mathrm{mm\,s^{-1}}$) pushes the sustained
torque down to $m=0.466$ and the drilling bristle to $\xi_{\psi,\max}=0.466$ —
i.e. translation erodes the drilling capacity just as spin eroded the
translational capacity in the top row.

![Regime 3 stick-to-slip (top: translation, bottom: drilling)](fig2_regime3.png)

*Dotted **right-hand axis** curves are the **counterpart channel under the
same drive**: the drilling $m$/$\xi_{\psi}$ (bottom-row inputs) overlaid on
the translation panels, and the translation $f$/$\xi$ (top-row inputs)
overlaid on the drilling panels. Because the two rows use different drives,
these counterparts come from matched single-coupled runs (the same
$\|V_p\|$ ramp with $|\omega_z|a=50$ mm/s, and the same $a|\omega_z|$ ramp
with $\|V_p\|=50$ mm/s), so the overlay answers "what would the *other*
channel have done in this exact scenario?"*

---

## X.4 Slip-to-stick transition

The reverse transition is simulated by decelerating from steady sliding
($\lVert V_p\rVert=0.3\,\mathrm{m\,s^{-1}}=30\,v_{\mathrm{str}}$,
$\xi(0)=1/r_s=0.909$) to rest in $0.2\,\mathrm{s}$, then holding. The velocity
therefore sweeps down through $v_{\mathrm{str}}$ into presliding and to
zero. Two effects govern the response.

### Stribeck-rate gating of the recovery

As $\lVert V_p\rVert$ falls back **through $v_{\mathrm{str}}$**, the Stribeck
function $g(V_{\mathrm{eff}})$ *rises* from $\mu_c$ toward $\mu_s$, so the
instantaneous steady-state target

$$
\xi_{\mathrm{ss}}(t) = \frac{g\bigl(V_{\mathrm{eff}}(t)\bigr)}{\mu_s}
\;\xrightarrow[\;V_{\mathrm{eff}}\to0\;]{}\; 1 .
$$

The scaled bristle state therefore *grows* during deceleration — the model's
continuous representation of static-friction recovery (junction
re-strengthening). **This recovery is itself Stribeck-gated:** it is driven
entirely by the $v_{\mathrm{str}}$-shaped rise of $g$, and how far $\xi$ climbs
before the contact freezes is set by the race between the deceleration rate and
the presliding fill time $T=g\,\delta^{*}/(\mu_s V_{\mathrm{eff}})\to\infty$ as
$V_{\mathrm{eff}}\to0$. A **slow** stop lets $\xi$ track the rising target and
recover almost fully; a **fast** stop freezes it only slightly above its kinetic
value. The rate sweep makes this explicit (frozen $\xi$, starting from
$\xi=0.909$, target-at-rest $=1$):

| deceleration time | 20 ms | 50 ms | 100 ms | 200 ms | 500 ms | 1 s | 2 s |
|---|---|---|---|---|---|---|---|
| frozen $\xi$ | 0.910 | 0.911 | 0.913 | **0.916** | 0.925 | 0.936 | 0.951 |

so at this narrower $v_{\mathrm{str}}$ even a $100\times$ slower stop recovers only
to $\xi=0.951$ (about halfway from the kinetic value $0.909$ to $\delta^{*}$),
while a fast stop barely recovers at all. Recovery is weaker than at the wider
$v_{\mathrm{str}}=50\,\mathrm{mm\,s^{-1}}$ because the target $\xi_{\mathrm{ss}}=g/\mu_s$
only rises steeply within the last $\sim v_{\mathrm{str}}=10\,\mathrm{mm\,s^{-1}}$
before rest, leaving a shorter window in which to wind the bristle back up. This
rate-dependent static recovery is a genuine prediction of the model, not a fitted
feature — it falls straight out of the $v_{\mathrm{str}}$ gating of $g$.

### Freezing: yes, because ż → 0 — and specifically because relaxation is $V_{\mathrm{eff}}$-gated

Once $V_p=0$ **and** $\omega_z=0$, the effective slip
$V_{\mathrm{eff}}=\lVert V_p\rVert+R(\xi)\tfrac{8}{3\pi}|\omega_z|a$ is exactly
zero, and *both* terms of the state equation vanish:

$$
\dot z = \underbrace{V_{\mathrm{slip}}}_{=\,0}
- \underbrace{V_{\mathrm{eff}}}_{=\,0}\,\frac{z}{\delta^{*}}\,\frac{\mu_s}{g}
\;=\; 0 \quad\text{for any } z .
$$

The simulation confirms $\dot z_{\text{end}} = 0$ to machine precision. The
essential point is that the relaxation (decay) term is **multiplicatively gated
by $V_{\mathrm{eff}}$**, not merely by the driving speed: it is *not* enough that
the input $V_{\mathrm{slip}}$ has stopped — the bristle would still creep back if
the decay term were active. Because that term carries $V_{\mathrm{eff}}$ as a
factor, the deflection is *held* rather than relaxed, and the interface retains a
locked-in elastic force $f=\xi=0.916$ (i.e.
$F=\mu_s N\cdot0.916=52.6\,\mathrm{N}$) with no relative motion. This is the
mechanism by which the roller sustains a tangential preload at rest (holding the
platform against a slope or residual actuation torque). The stored force is
released only when subsequent motion re-activates the state equation; a reversal
first unwinds $\xi$ through zero, reproducing presliding hysteresis on the return
branch.

### Freezing requires *both* $V_p\to0$ and $\omega_z\to0$

Because the decay term is gated by $V_{\mathrm{eff}}$ — which contains the spin
slip — a roller that stops **translating** but retains a residual **drilling
spin** does *not* freeze. With $V_p=0$ but $|\omega_z|a=20\,\mathrm{mm\,s^{-1}}$
the effective slip stays positive, the decay term remains active with no
compensating drive, and the stored deflection *bleeds off*: the simulation shows
$\xi$ collapsing from $0.92$ toward $\xi\to0$ (reaching $0.054$ and still
falling), i.e. the tangential preload is eroded by the spin. The erosion is
self-limiting — as $\xi\to0$ the gate $R(\xi)\to0$, so $V_{\mathrm{eff}}\to0$ and
the decay slows — but the qualitative message stands: **contact-patch spin
prevents a stopped roller from holding a static tangential load.** This is a
directly testable, mechanically meaningful consequence of the $\xi$-gated
$V_{\mathrm{eff}}$.

![Regime 4 recovery + freezing](fig3_regime4.png)

*Fig. 3 is intrinsically spin-free (the stopped-roller freeze scenario), so
no spin-coupled counterpart exists to overlay.*

![Regime 4 rate gating and residual spin](fig4_regime4_gating.png)

*Left: the decel-rate sweep is also spin-free, so the right axis is omitted.
Right: the dotted **right-hand axis** (green) overlays the **drilling
bristle $\xi_{\psi}$** from the *same* residual-spin coupled run
($\|V_p\|$ stop **with** $|\omega_z|a=20$ mm/s). It tracks the translation
bristle $\xi$ closely, confirming that under sustained spin *both* channels
keep relaxing instead of freezing.*

---

## X.5 Repeated stick–slip–stick–slip cycling

Regimes X.3 and X.4 examined the two transitions in isolation, each poised near
the break-away limit. In practice the contact does not linger on that limit: once
it breaks loose it runs **deep into gross sliding** (where $f$ collapses to the
kinetic floor and the bristle sits far below $\delta^{*}$), then decelerates back
into a **stick dwell**, and the cycle repeats. To exercise this we drive a
periodic, unidirectional velocity profile — a stick dwell, an acceleration to
$\lVert V_p\rVert_{\max}=150\,\mathrm{mm\,s^{-1}}=15\,v_{\mathrm{str}}$, a
symmetric deceleration, and a second stick dwell — repeated for two cycles
($P=0.5\,\mathrm{s}$, spin-free). The velocity therefore spans the full range
$0\to15\,v_{\mathrm{str}}\to0$ each cycle, i.e. from frozen stick through
break-away into well-developed gross sliding and back.

The response traces a **closed hysteresis loop** in the $f$–$\lVert V_p\rVert$
plane — the mechanical signature of a stick–slip limit cycle:

1. **First stick dwell (fresh contact).** Starting from $\xi(0)=0$, during the
   initial dwell there is no stored deflection, so $f\approx0$. This is the only
   "memoryless" stick in the run — every later dwell inherits a wound-up bristle.

2. **Break-away.** On acceleration $\xi$ winds up and $f$ rises to a modest peak
   $f_{\max}=0.926$ (slightly above the kinetic floor $1/r_s=0.909$): at this
   $v_{\mathrm{str}}$ and loading rate the overshoot is small, consistent with
   §X.3.

3. **Gross sliding.** As $\lVert V_p\rVert$ climbs past a few $v_{\mathrm{str}}$
   the Stribeck term saturates ($g\to\mu_c$) and the operating point drops onto
   the kinetic floor $f\to1/r_s=0.909$; the bristle unwinds to
   $\xi\approx0.88\text{–}0.91$, transiently dipping slightly below the floor
   during the fastest sliding before settling. **This is the key distinction
   from X.3/X.4** — the contact is no longer at the break-away limit but well
   inside gross sliding, storing the minimum bristle strain.

4. **Fall back to stick.** On deceleration $g$ recovers up the Stribeck shoulder,
   $\xi$ climbs back (static recovery, §X.4), and once $\lVert V_p\rVert=0$ the
   state **freezes** at $\xi=0.920$, holding a locked-in preload $f=0.920$.

5. **Re-break and repeat.** The next acceleration breaks the contact loose again
   from this frozen state, and the loop closes. Because each dwell freezes at a
   finite $\xi$ while gross sliding always returns to the same kinetic floor, the
   up-going and down-going branches do not coincide: the enclosed area of the
   $f$–$\lVert V_p\rVert$ loop is the energy dissipated per cycle, and it is
   traversed identically on the second cycle — a stable limit cycle rather than a
   transient.

The distinguishing feature of this regime is thus the **excursion**: the state
does not sit at $\xi\approx1$ (break-away) but sweeps between the frozen stick
value ($\xi\approx0.92$) and the gross-sliding floor ($\xi\approx0.88$–$0.91$),
and the force between a locked-in preload and the kinetic level. It is this
repeated loop — not any single transition — that a controller must contend with
when a Mecanum roller chatters between grip and slip. Because these excursions
are confined to a narrow band, the bottom row of Fig. 5 replots both panels with
the vertical axis restricted to $[0.8,1.0]$, where the break-away peak, the
gross-sliding floor, and the per-cycle recovery of the frozen stick value are
clearly resolved.

![Regime 5 stick-slip-stick-slip cycle and hysteresis loop (top: full range; bottom: zoom $y\in[0.8,1.0]$)](fig5_regime5_cycle.png)

---

## Summary: the asymmetry and the role of the gate

The asymmetry between §X.3 and §X.4 is the essence of the model:
**stick→slip destroys stored bristle strain** via the velocity-driven
relaxation term, whereas **slip→stick preserves and even augments it**, because
the relaxation term is gated by $V_{\mathrm{eff}}$ (which $\to0$ at rest). This
one-sided memory gives the model its hysteretic, dissipative character and
distinguishes it from any static $\mu(v)$ map. Chained together over repeated
cycles (§X.5) this asymmetry produces a **stable stick–slip limit cycle**: a
closed $f$–$\lVert V_p\rVert$ hysteresis loop whose enclosed area is the energy
dissipated per cycle, with the state sweeping between a frozen stick preload and
the gross-sliding kinetic floor rather than resting at the break-away limit.

The $\varepsilon/\xi$-gate adds a second, orthogonal asymmetry: the spin
cross-coupling is **absent in deep presliding** ($R(0)=0$, so microslip is
uncontaminated by the Adamov gross-sliding estimate) and **fully active in
gross sliding** ($R\to1$). Expressed in the scaled state $\xi=z/\delta^{*}$,
the gate is a natural state-feedback — the same variable that measures how
close the contact is to break-away also sets how much spin slip enters the
effective velocity.

| quantity | spin-free | spin-on | note |
|---|---|---|---|
| $\xi_{\mathrm{ss}}$, presliding (5 mm/s $=0.5\,v_{\mathrm{str}}$) | 0.980 | 0.705 | gate softens $g$ |
| $f$ break-away peak (ramp) | 0.909 (no overshoot) | 0.849 (no peak) | overshoot suppressed at small $v_{\mathrm{str}}$ |
| $f$ steady slip | 0.909 $(=1/r_s)$ | 0.849 | spin lowers sustained force |
| $\xi$ frozen after stop (200 ms) | 0.916 | 0.916 | spin self-extinguishes at rest |
| $\xi$ frozen, fast stop (20 ms) | 0.910 | — | Stribeck-rate-gated recovery |
| $\xi$ frozen, slow stop (2 s) | 0.951 | — | recovers ~halfway to $\delta^{*}$ |
| $\xi$ after stop, residual spin | — | $\to 0$ | preload eroded by drilling spin |
| $f(0^{+})$ damping jump | $\tau/T=0.012$ | $0.012$ | unaffected ($R(0)=0$) |
| cyclic loop (§X.5): dwell / peak / floor | $0.920$ / $0.926$ / $0.909$ | — | stable stick–slip limit cycle |
| drilling $\xi_{\psi,\mathrm{ss}}$ (presliding) | $0.561$ (pure spin) | $0.329$ (transl-on) | ceiling $\tfrac{3\pi}{16}\tfrac{g}{\mu_s}$, not $1$ |
| drilling $m$ gross floor (ramp) | $0.535$ (pure spin) | $0.466$ (transl-on) | $\tfrac{3\pi}{16}/r_s$; transl. cross-coupling erodes it |

*(Numbers are for $v_{\mathrm{str}}=10\,\mathrm{mm\,s^{-1}}$. The break-away
overshoot and the strength of static recovery both scale with $v_{\mathrm{str}}$;
for reference, at $v_{\mathrm{str}}=50\,\mathrm{mm\,s^{-1}}$ the spin-free peak was
$0.965$ and the $200\,\mathrm{ms}$-stop recovery reached $\xi=0.971$.)*

---

### Reproducibility

Run `python lugre_scaled_gated_regimes.py` in this folder (requires
`numpy`, `scipy`, `matplotlib`). It regenerates the six figures and prints the
JSON block of scalar results quoted above. All parameters are set at the top of
the file — including the discretionary Stribeck velocity `v_str = 0.01` (m/s);
set the spin inputs `wzON`, `wz3` to zero to recover the exact pure-translation
limit.
