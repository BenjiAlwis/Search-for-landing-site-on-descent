"""Post-flight plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PHASE_COLORS = {"SEARCH": "#8c8c8c", "ORBIT": "#2a78c4", "FINAL": "#d9822b", "FLARE": "#c23b3b"}


def plot_flight(result, runway, path):
    L = result.log
    t, pos = L["t"], L["pos"]
    h = -pos[:, 2]
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(2, 3, 1)
    rc = np.vstack([runway.corners_ne(), runway.corners_ne()[:1]])
    ax.fill(rc[:, 1], rc[:, 0], color="k", alpha=0.7, label="runway")
    for ph, colr in PHASE_COLORS.items():
        m = L["phase"] == ph
        if m.any():
            ax.plot(pos[m, 1], pos[m, 0], ".", ms=1.5, color=colr, label=ph)
    ests = [e for e in L["rwy_est"] if e is not None]
    if ests:
        c = np.array([e[0] for e in ests])
        ax.plot(c[:, 1], c[:, 0], "m+", ms=4, alpha=0.4, label="runway estimate")
    ax.set_xlabel("East [m]"), ax.set_ylabel("North [m]"), ax.set_aspect("equal"), ax.legend(fontsize=7)
    ax.set_title("Ground track")

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.plot(pos[:, 1], pos[:, 0], h, lw=0.8)
    ax.plot(rc[:, 1], rc[:, 0], 0 * rc[:, 0], "k", lw=2)
    ax.set_xlabel("E"), ax.set_ylabel("N"), ax.set_zlabel("h"), ax.set_title("Trajectory")

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, h, label="altitude [m]")
    ax.plot(t, L["airspeed"], label="airspeed [m/s]")
    ax.set_xlabel("t [s]"), ax.legend(), ax.grid(alpha=0.3), ax.set_title("Altitude / airspeed")

    ax = fig.add_subplot(2, 3, 4)
    for i, lab in enumerate(("roll", "pitch")):
        ax.plot(t, np.degrees(L["euler"][:, i]), label=lab)
    ax.plot(t, np.degrees(L["alpha"]), label="alpha")
    ax.set_xlabel("t [s]"), ax.set_ylabel("deg"), ax.legend(), ax.grid(alpha=0.3), ax.set_title("Attitude")

    ax = fig.add_subplot(2, 3, 5)
    for i, lab in enumerate(("elevator", "aileron", "rudder", "flap")):
        ax.plot(t, np.degrees(L["surf"][:, i]), label=lab, lw=0.8)
    ax.set_xlabel("t [s]"), ax.set_ylabel("deg"), ax.legend(), ax.grid(alpha=0.3), ax.set_title("Control surfaces")

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, -L["vel_ned"][:, 2], label="climb rate [m/s]")
    ax.plot(t, np.linalg.norm(L["nav_pos"][:, :2] - pos[:, :2], axis=1), label="nav pos error [m]")
    ax.set_xlabel("t [s]"), ax.legend(), ax.grid(alpha=0.3), ax.set_title("Vertical speed / nav error")

    td = result.touchdown
    if td:
        fig.suptitle(f"Touchdown: {'ON' if td['on_runway'] else 'OFF'} runway, "
                     f"{td['dist_from_threshold_m']:.1f} m past threshold, {td['cross_track_m']:.1f} m cross-track, "
                     f"sink {td['sink_rate_mps']:.2f} m/s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
