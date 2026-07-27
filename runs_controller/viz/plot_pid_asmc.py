import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    import csv
    d = {}
    with open(p) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        for c in cols: d[c] = []
        for row in r:
            for c in cols: d[c].append(float(row[c]))
    return {c: np.array(v) for c, v in d.items()}

ae = load("asmc_ellipse.csv"); pe = load("pid_ellipse.csv")
ao = load("asmc_octagon.csv"); po = load("pid_octagon.csv")

def pos_err_cm(d):
    return np.hypot(d["Xo"]-d["xo_ref"], d["Yo"]-d["yo_ref"])*100

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (0,0) Ellipse world path
ax[0,0].plot(ae["xo_ref"], ae["yo_ref"], 'k--', lw=2, label="reference", zorder=1)
ax[0,0].plot(ae["Xo"], ae["Yo"], '-', color="#1f77b4", lw=1.6, label="ASMC", zorder=3)
ax[0,0].plot(pe["Xo"], pe["Yo"], '-', color="#d62728", lw=1.6, label="PID", zorder=2)
ax[0,0].set_title("Ellipse (pose tracking) — world path")
ax[0,0].set_xlabel("X (m)"); ax[0,0].set_ylabel("Y (m)"); ax[0,0].axis("equal")
ax[0,0].legend(); ax[0,0].grid(alpha=0.3)

# (0,1) Ellipse position error vs time (log)
ax[0,1].semilogy(ae["t"], np.maximum(pos_err_cm(ae),1e-4), color="#1f77b4", lw=1.4,
                 label=f"ASMC (final {pos_err_cm(ae)[-1]:.3f} cm)")
ax[0,1].semilogy(pe["t"], np.maximum(pos_err_cm(pe),1e-4), color="#d62728", lw=1.4,
                 label=f"PID (final {pos_err_cm(pe)[-1]:.2f} cm)")
ax[0,1].axhline(1.0, color="gray", ls=":", label="1 cm target")
ax[0,1].set_title("Ellipse — position error |pos - ref|")
ax[0,1].set_xlabel("t (s)"); ax[0,1].set_ylabel("error (cm, log)")
ax[0,1].legend(); ax[0,1].grid(alpha=0.3, which="both")

# (1,0) Octagon Vx vs time
ax[1,0].plot(ao["t"], ao["Vx_des"], 'k--', lw=1.8, label="desired")
ax[1,0].plot(ao["t"], ao["Vx"], color="#1f77b4", lw=1.2, label="ASMC")
ax[1,0].plot(po["t"], po["Vx"], color="#d62728", lw=1.2, label="PID")
ax[1,0].set_title("Octagon (velocity tracking) — Vx")
ax[1,0].set_xlabel("t (s)"); ax[1,0].set_ylabel("Vx (m/s)")
ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

# (1,1) Octagon Vy vs time
ax[1,1].plot(ao["t"], ao["Vy_des"], 'k--', lw=1.8, label="desired")
ax[1,1].plot(ao["t"], ao["Vy"], color="#1f77b4", lw=1.2, label="ASMC")
ax[1,1].plot(po["t"], po["Vy"], color="#d62728", lw=1.2, label="PID")
ax[1,1].set_title("Octagon (velocity tracking) — Vy")
ax[1,1].set_xlabel("t (s)"); ax[1,1].set_ylabel("Vy (m/s)")
ax[1,1].legend(); ax[1,1].grid(alpha=0.3)

fig.suptitle("ASMC vs PID — tuned gains, clean (oracle) state feed", fontsize=14, y=0.995)
fig.tight_layout()
fig.savefig("asmc_vs_pid.png", dpi=130, bbox_inches="tight")
print("saved asmc_vs_pid.png")

# quick numeric summary
for name, o in [("ASMC", ao), ("PID", po)]:
    exx = np.sqrt(np.mean((o["Vx"]-o["Vx_des"])**2))*1e3
    eyy = np.sqrt(np.mean((o["Vy"]-o["Vy_des"])**2))*1e3
    print(f"{name} octagon: Vx_rms={exx:.1f} mm/s  Vy_rms={eyy:.1f} mm/s")
