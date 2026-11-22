#!/usr/bin/env python3
"""
SQE_Bloch_visualisation_animated.py

Run one SQE-state optimisation and produce:

  1. Static summary panel — control pulse, infidelity evolution, entropy trajectory.
  2. Static Bloch sphere — SQE trajectory in the {|01>, |10>} subspace.
  3. Configurable animation — selected panels synced in time.

Animation options
-----------------
Set these in the settings block:

    ANIMATION_PANELS = ["pulse", "infidelity", "entropy", "bloch"]
    ANIMATION_PULSE_MODE = "both"
    BLOCH_ROTATE = True

Panel options:
    "pulse", "infidelity", "entropy", "bloch"

Pulse mode options:
    "infidelity"    -> infidelity-optimised pulse only
    "entanglement"  -> entanglement-constrained TEM pulse only
    "both"          -> show both pulses/trajectories simultaneously

For slides, a clean recommended setting is:

    ANIMATION_PANELS = ["entropy", "bloch"]
    ANIMATION_PULSE_MODE = "both"
    BLOCH_ROTATE = True
    SAVE_ANIMATION_MP4 = True
    SAVE_ANIMATION_GIF = False

Output
------
    OUTDIR/sqe_state_summary_<mode>.png
    OUTDIR/sqe_state_bloch_sphere_<mode>.png
    OUTDIR/sqe_state_animation_<mode>_<panels>_<fixed/rotating>.mp4/.gif
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.special import xlogy

# pulse_vqe_entanglement is not pip-installed; add the repo root to sys.path.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_state import sqe_optimisation, single_qubit_excitation
from pulse_vqe_entanglement.entanglement_measures import entropy_bipartite_single


# =============================================================================
# User settings
# =============================================================================

OUTDIR = (
    Path(__file__).resolve().parent
)
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_ENTANGLEMENT_OPTIMISATION = True

THETA = 0.05 * np.pi  # SQE gate angle, radians
T_NS = 500.0         # total pulse duration, ns
N_TIME_STEPS = 30
N_DISCRETISATION = 5

J_MIN = 0.0
J_MAX = 10.0
J_INIT_MEAN = 0.0
J_INIT_SCALE = 1e0
J_INIT_SEED = 1111

MAXITER_BASELINE = 200

S_MAX = 0.01
EPS_INFIDELITY = 1e-6
RHO = 0.0
DUAL_LR = 1e12
N_OUTER = 3
INNER_MAXITER = 200
FD_EPS = 1e-6

GTOL = 1e-12

STATE_ENT_MEASURE = "von_neumann"

# Default: Bell-like state in the {|01>, |10>} subspace.
# For pure |01>, use: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
INITIAL_STATE = np.array(
    [0.0, 1.0, 0.0, 0.0] / np.sqrt(1.0),
    dtype=np.complex128,
)

ZEEMAN_SPLITTINGS = np.array([1000.0, 1008.0])  # MHz

# Bloch sphere heatmap parameters.
SURFACE_ALPHA = 0.32
SURFACE_BLEND = 0.12


# =============================================================================
# Animation settings
# =============================================================================

MAKE_ANIMATION = True

ANIMATION_PANELS = ["pulse", "infidelity", "entropy", "bloch"]
# Options: "pulse", "infidelity", "entropy", "bloch"

ANIMATION_PULSE_MODE = "both"
# Options: "infidelity", "entanglement", "both"

BLOCH_ROTATE = False
BLOCH_AZIM_START = 45
BLOCH_AZIM_END = 405
BLOCH_ELEV = 22

SAVE_ANIMATION_MP4 = True
SAVE_ANIMATION_GIF = False
ANIMATION_FPS = 15
ANIMATION_DPI_MP4 = 180
ANIMATION_DPI_GIF = 140


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "lines.linewidth": 1.5,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COLOR_BASELINE = "#2ca02c"
COLOR_ENT = "#d62728"
COLOR_TARGET = "black"


# =============================================================================
# Helpers
# =============================================================================

def make_config() -> SimulationConfig:
    """Build SimulationConfig from module-level settings."""
    return SimulationConfig(
        zeeman_splittings=ZEEMAN_SPLITTINGS,
        initial_state=INITIAL_STATE,
        theta_target=THETA,
        n_time_steps=N_TIME_STEPS,
        n_discretisation=N_DISCRETISATION,
        T=T_NS,
        max_drive_strength=0.0,
        J_max=J_MAX,
        J_min=J_MIN,
        J_init_scale=J_INIT_SCALE,
        J_init_mean=J_INIT_MEAN,
        J_init_seed=J_INIT_SEED,
        feromagnetic=False,
        gtol=GTOL,
        maxiter=MAXITER_BASELINE,
        entanglement_optimisation=RUN_ENTANGLEMENT_OPTIMISATION,
        state_ent_measure=STATE_ENT_MEASURE,
        S_max=S_MAX,
        eps_infidelity=EPS_INFIDELITY,
        rho=RHO,
        dual_lr=DUAL_LR,
        n_outer=N_OUTER,
        inner_maxiter=INNER_MAXITER,
        fd_eps=FD_EPS,
    )


def rescale_J(x: np.ndarray) -> np.ndarray:
    """Map normalised pulse values in [-1, 1] back to physical J in MHz."""
    return 0.5 * (J_MAX - J_MIN) * np.asarray(x) + 0.5 * (J_MAX + J_MIN)


def as_time_major_traj(traj: np.ndarray) -> np.ndarray:
    """Return trajectory in (time, state_dim) order regardless of input orientation."""
    traj = np.asarray(traj, dtype=np.complex128)
    if traj.ndim != 2:
        raise ValueError(f"Expected 2D trajectory, got shape {traj.shape}")
    if traj.shape[0] == 4:
        return traj.T
    if traj.shape[1] == 4:
        return traj
    raise ValueError(f"Cannot interpret trajectory shape {traj.shape}")


def entropy_traj_2q(traj: np.ndarray) -> np.ndarray:
    """Von Neumann entropy of qubit 0 at each time step along the trajectory."""
    traj_tm = as_time_major_traj(traj)
    return np.array([
        entropy_bipartite_single(psi, subsystem=(0,), n_qubits=2)
        for psi in traj_tm
    ])


def mean_entropy(S: np.ndarray) -> float:
    return float(np.mean(np.asarray(S, float)))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary base-2 entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
    p = np.clip(np.asarray(p, float), 1e-15, 1.0 - 1e-15)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)) / np.log(2.0)


def projected_bloch_coords(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the SQE state trajectory onto the Bloch sphere in the {|01>, |10>} subspace.

    Coordinates:
        x = 2 Re(a* b)
        y = 2 Im(a* b)
        z = |a|^2 - |b|^2

    where a and b are the |01> and |10> amplitudes, normalised to that subspace.
    """
    traj_tm = as_time_major_traj(traj)
    a = traj_tm[:, 1]  # |01> amplitude
    b = traj_tm[:, 2]  # |10> amplitude

    norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
    norm = np.where(norm < 1e-14, 1.0, norm)

    a = a / norm
    b = b / norm

    x = 2.0 * np.real(np.conj(a) * b)
    y = 2.0 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2

    return x, y, z


def get_baseline_and_ent_results(results):
    """Extract baseline and TEM results from sqe_optimisation output."""
    if RUN_ENTANGLEMENT_OPTIMISATION:
        if results.baseline is None:
            raise RuntimeError("Expected results.baseline from sqe_optimisation.")
        return results.baseline, results
    return results, None


def mode_suffix() -> str:
    return "entanglement_constrained" if RUN_ENTANGLEMENT_OPTIMISATION else "infidelity_only"


def hide_3d_axes(ax) -> None:
    """Remove default Matplotlib 3D axis decorations for a clean Bloch sphere."""
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_alpha(0.0)


def add_clean_bloch_axes(ax, length: float = 1.28) -> None:
    """Optional: draw custom x/y/z arrows and basis labels for the Bloch sphere."""
    arrow_kw = dict(arrow_length_ratio=0.08, linewidth=1.2, color="black", alpha=0.75)
    ax.quiver(0, 0, 0, length, 0, 0, **arrow_kw)
    ax.quiver(0, 0, 0, 0, length, 0, **arrow_kw)
    ax.quiver(0, 0, 0, 0, 0, length, **arrow_kw)
    ax.text(length + 0.06, 0, 0, r"$x$", ha="center", va="center")
    ax.text(0, length + 0.06, 0, r"$y$", ha="center", va="center")
    ax.text(0, 0, length + 0.06, r"$|01\rangle$", ha="center", va="center")
    ax.text(0, 0, -length - 0.08, r"$|10\rangle$", ha="center", va="center")


def make_bloch_surface(ax):
    """Draw entropy-coloured Bloch sphere surface and return colour mappable."""
    theta_mesh = np.linspace(0, np.pi, 100)
    phi_mesh = np.linspace(0, 2 * np.pi, 200)
    TH, PH = np.meshgrid(theta_mesh, phi_mesh)

    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)

    # TH is the Bloch-sphere polar angle, so p01 = cos^2(TH/2).
    p01 = np.cos(TH / 2) ** 2
    S = binary_entropy(p01)

    cmap = plt.colormaps["viridis"]
    norm = plt.Normalize(0.0, 1.0)
    facecolors = cmap(norm(S))

    # Blend slightly toward white to make trajectories more visible.
    facecolors[..., :3] = (1.0 - SURFACE_BLEND) * facecolors[..., :3] + SURFACE_BLEND
    facecolors[..., 3] = SURFACE_ALPHA

    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=0,
    )

    return ScalarMappable(norm=norm, cmap=cmap), S


def select_animation_results(res_base, res_ent, pulse_mode: str):
    """Select one or two trajectories for animation."""
    pulse_mode = pulse_mode.lower().strip()

    if pulse_mode in {"infidelity", "baseline", "infidelity_only", "i-only"}:
        return [
            {
                "res": res_base,
                "label": "Infidelity optimised",
                "color": COLOR_BASELINE,
                "tag": "infidelity",
            }
        ]

    if pulse_mode in {"entanglement", "tem", "entropy", "entanglement_constrained"}:
        if res_ent is None:
            raise ValueError(
                "Entanglement mode requested, but res_ent is None. "
                "Set RUN_ENTANGLEMENT_OPTIMISATION=True or use pulse_mode='infidelity'."
            )
        return [
            {
                "res": res_ent,
                "label": "Entanglement constrained",
                "color": COLOR_ENT,
                "tag": "entanglement",
            }
        ]

    if pulse_mode == "both":
        if res_ent is None:
            raise ValueError(
                "Both mode requested, but res_ent is None. "
                "Set RUN_ENTANGLEMENT_OPTIMISATION=True."
            )
        return [
            {
                "res": res_base,
                "label": "Infidelity optimised",
                "color": COLOR_BASELINE,
                "tag": "infidelity",
            },
            {
                "res": res_ent,
                "label": "Entanglement constrained",
                "color": COLOR_ENT,
                "tag": "entanglement",
            },
        ]

    raise ValueError("Use pulse_mode='infidelity', 'entanglement', or 'both'.")


# =============================================================================
# Run optimisation
# =============================================================================

def run_optimisation():
    """Run sqe_optimisation and split output into baseline/TEM results."""
    config = make_config()
    results = sqe_optimisation(config)
    return get_baseline_and_ent_results(results)


# =============================================================================
# Static summary plots
# =============================================================================

def plot_summary(res_base, res_ent) -> None:
    """Three-row summary: control pulse, infidelity evolution, entropy trajectory."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 6))

    t_seg = np.arange(N_TIME_STEPS + 1) * T_NS / N_TIME_STEPS
    n_fine = N_TIME_STEPS * N_DISCRETISATION
    t_fine = np.arange(n_fine + 1) * T_NS / n_fine

    # Control pulse.
    xb = np.concatenate([res_base.x_final, [res_base.x_final[-1]]])
    axes[0].step(
        t_seg,
        rescale_J(xb),
        where="post",
        label="Infidelity optimised",
        color=COLOR_BASELINE,
    )

    if res_ent is not None:
        xe = np.concatenate([res_ent.x_final, [res_ent.x_final[-1]]])
        axes[0].step(
            t_seg,
            rescale_J(xe),
            where="post",
            label="Entanglement constrained",
            color=COLOR_ENT,
        )

    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel(r"$J$ (MHz)")
    axes[0].set_title("(a) Control pulse")
    axes[0].legend()

    # Infidelity.
    axes[1].plot(
        t_fine,
        np.maximum(res_base.val_final, 1e-16),
        label="Infidelity optimised",
        color=COLOR_BASELINE,
    )

    if res_ent is not None:
        axes[1].plot(
            t_fine,
            np.maximum(res_ent.val_final, 1e-16),
            label="Entanglement constrained",
            color=COLOR_ENT,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Infidelity")
    axes[1].set_title("(b) Infidelity evolution")
    axes[1].legend()

    # Entropy.
    S_base = entropy_traj_2q(res_base.final_traj)
    axes[2].plot(
        t_fine,
        S_base,
        label="Infidelity optimised",
        color=COLOR_BASELINE,
    )

    if res_ent is not None:
        S_ent = entropy_traj_2q(res_ent.final_traj)
        axes[2].plot(
            t_fine,
            S_ent,
            label="Entanglement constrained",
            color=COLOR_ENT,
        )

    target_state = single_qubit_excitation(THETA) @ INITIAL_STATE
    S_target = entropy_bipartite_single(target_state, subsystem=(0,), n_qubits=2)
    axes[2].hlines(
        S_target,
        0,
        T_NS,
        colors=COLOR_TARGET,
        linestyles="dashed",
        label="Target",
    )

    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel(r"$S$ (bits)")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("(c) Entanglement entropy trajectory")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / f"sqe_state_summary_{mode_suffix()}.png", dpi=300)
    plt.show()


# =============================================================================
# Static Bloch sphere
# =============================================================================

def plot_bloch_sphere(res_base, res_ent) -> None:
    """Plot SQE trajectory in the {|01>, |10>} Bloch sphere."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    mappable, _ = make_bloch_surface(ax)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label(r"$S$ (bits)")

    r_offset = 1.025

    xb, yb, zb = projected_bloch_coords(res_base.final_traj)
    ax.plot(
        r_offset * xb,
        r_offset * yb,
        r_offset * zb,
        color=COLOR_BASELINE,
        linewidth=2.2,
        label="Infidelity optimised",
        zorder=10,
    )
    ax.scatter(
        [r_offset * xb[0]],
        [r_offset * yb[0]],
        [r_offset * zb[0]],
        color=COLOR_BASELINE,
        s=45,
        marker="o",
        zorder=12,
    )
    ax.scatter(
        [r_offset * xb[-1]],
        [r_offset * yb[-1]],
        [r_offset * zb[-1]],
        color=COLOR_BASELINE,
        s=70,
        marker="*",
        zorder=12,
    )

    if res_ent is not None:
        xe, ye, ze = projected_bloch_coords(res_ent.final_traj)
        ax.plot(
            r_offset * xe,
            r_offset * ye,
            r_offset * ze,
            color=COLOR_ENT,
            linewidth=2.2,
            label="Entanglement constrained",
            zorder=11,
        )
        ax.scatter(
            [r_offset * xe[0]],
            [r_offset * ye[0]],
            [r_offset * ze[0]],
            color=COLOR_ENT,
            s=45,
            marker="o",
            zorder=13,
        )
        ax.scatter(
            [r_offset * xe[-1]],
            [r_offset * ye[-1]],
            [r_offset * ze[-1]],
            color=COLOR_ENT,
            s=70,
            marker="*",
            zorder=13,
        )

    # Clean sphere: no x/y/z axes.
    hide_3d_axes(ax)

    lim = 1.00
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=BLOCH_ELEV, azim=BLOCH_AZIM_START)
    ax.set_title(r"(d) Trajectory in $\{|01\rangle, |10\rangle\}$")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTDIR / f"sqe_state_bloch_sphere_{mode_suffix()}.png", dpi=300)
    plt.show()


# =============================================================================
# Configurable animation
# =============================================================================

def animate_trajectory(
    res_base,
    res_ent=None,
    panels=None,
    pulse_mode="both",
    save_mp4=True,
    save_gif=False,
    fps=15,
) -> None:
    """Animate selected panels for one or both pulse modes."""
    if panels is None:
        panels = ["pulse", "infidelity", "entropy", "bloch"]

    panels = [p.lower().strip() for p in panels]
    valid_panels = {"pulse", "infidelity", "entropy", "bloch"}

    for p in panels:
        if p not in valid_panels:
            raise ValueError(
                f"Unknown animation panel {p!r}. "
                "Use any of: 'pulse', 'infidelity', 'entropy', 'bloch'."
            )

    if len(panels) == 0:
        raise ValueError("At least one animation panel must be selected.")

    curves = select_animation_results(res_base, res_ent, pulse_mode)

    n_fine = N_TIME_STEPS * N_DISCRETISATION
    t_fine = np.arange(n_fine + 1) * T_NS / n_fine
    t_seg = np.arange(N_TIME_STEPS + 1) * T_NS / N_TIME_STEPS

    for c in curves:
        c["S"] = entropy_traj_2q(c["res"].final_traj)
        c["I"] = np.maximum(c["res"].val_final, 1e-16)
        c["x"], c["y"], c["z"] = projected_bloch_coords(c["res"].final_traj)
        c["J_step"] = np.concatenate([c["res"].x_final, [c["res"].x_final[-1]]])
        c["J_phys"] = rescale_J(c["J_step"])

    has_bloch = "bloch" in panels
    non_bloch_panels = [p for p in panels if p != "bloch"]

    if has_bloch and len(non_bloch_panels) > 0:
        fig = plt.figure(figsize=(17, 8))

        gs = fig.add_gridspec(
            len(non_bloch_panels),
            2,
            width_ratios=[1.05, 1.55],
            hspace=0.35,
)

        axes = {}
        for row, p in enumerate(non_bloch_panels):
            axes[p] = fig.add_subplot(gs[row, 0])

        axes["bloch"] = fig.add_subplot(gs[:, 1], projection="3d")
        axes["bloch"].set_position([0.52, 0.08, 0.43, 0.82])

    elif has_bloch:
        fig = plt.figure(figsize=(7, 7))
        axes = {"bloch": fig.add_subplot(111, projection="3d")}

    else:
        fig, ax_arr = plt.subplots(
            len(non_bloch_panels),
            1,
            figsize=(8.5, 2.5 * len(non_bloch_panels)),
            squeeze=False,
        )
        axes = {p: ax_arr[i, 0] for i, p in enumerate(non_bloch_panels)}

    artists = {c["tag"]: {} for c in curves}

    if "pulse" in panels:
        ax = axes["pulse"]

        for c in curves:
            ax.step(
                t_seg,
                c["J_phys"],
                where="post",
                color=c["color"],
                alpha=0.25,
            )

            line, = ax.step(
                [],
                [],
                where="post",
                color=c["color"],
                lw=2.3,
                label=c["label"],
            )

            marker, = ax.plot([], [], "o", color=c["color"], ms=6)

            artists[c["tag"]]["pulse_line"] = line
            artists[c["tag"]]["pulse_marker"] = marker

        ax.set_xlim(0, T_NS)
        ax.set_ylabel(r"$J$ (MHz)")
        ax.set_title("(a) Control pulse")

    if "infidelity" in panels:
        ax = axes["infidelity"]

        for c in curves:
            ax.plot(t_fine, c["I"], color=c["color"], alpha=0.25)
            line, = ax.plot([], [], color=c["color"], lw=2.3, label=c["label"])
            marker, = ax.plot([], [], "o", color=c["color"], ms=6)
            artists[c["tag"]]["inf_line"] = line
            artists[c["tag"]]["inf_marker"] = marker

        ax.set_xlim(0, T_NS)
        ax.set_yscale("log")
        ax.set_ylabel("Infidelity")
        ax.set_title("(b) Infidelity evolution")

    if "entropy" in panels:
        ax = axes["entropy"]

        for c in curves:
            ax.plot(t_fine, c["S"], color=c["color"], alpha=0.25)
            line, = ax.plot([], [], color=c["color"], lw=2.3, label=c["label"])
            marker, = ax.plot([], [], "o", color=c["color"], ms=6)
            artists[c["tag"]]["ent_line"] = line
            artists[c["tag"]]["ent_marker"] = marker

        ax.set_xlim(0, T_NS)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(r"$S$ (bits)")
        ax.set_title("(c) Entanglement entropy")

    if "bloch" in panels:
        ax = axes["bloch"]

        make_bloch_surface(ax)

        r_offset = 1.025

        # Start and target states
        psi_start = INITIAL_STATE
        psi_target = single_qubit_excitation(THETA) @ INITIAL_STATE

        x0, y0, z0 = projected_bloch_coords(psi_start.reshape(1, 4))
        xT, yT, zT = projected_bloch_coords(psi_target.reshape(1, 4))

        ax.scatter(
            [r_offset * x0[0]],
            [r_offset * y0[0]],
            [r_offset * z0[0]],
            color="#d62728",
            edgecolors="white",
            linewidths=1.5,
            s=140,
            zorder=30,
        )

        ax.scatter(
            [r_offset * xT[0]],
            [r_offset * yT[0]],
            [r_offset * zT[0]],
            color="#d62728",
            edgecolors="white",
            linewidths=1.5,
            s=140,
            zorder=30,
        )

        for c in curves:
            line, = ax.plot(
                [],
                [],
                [],
                color=c["color"],
                lw=2.6,
                label=c["label"],
            )

            dot = ax.scatter([], [], [], color=c["color"], s=55)

            artists[c["tag"]]["bloch_line"] = line
            artists[c["tag"]]["bloch_dot"] = dot

        hide_3d_axes(ax)

        lim = 1.00
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=BLOCH_ELEV, azim=BLOCH_AZIM_START)
        ax.set_title("(d) Trajectory", pad=2)
        ax.legend(loc="upper left")

    pulse_mode_clean = pulse_mode.lower().strip()
    if pulse_mode_clean == "both":
        title_label = "Infidelity optimised vs entanglement constrained"
    elif pulse_mode_clean in {"infidelity", "baseline", "infidelity_only", "i-only"}:
        title_label = "Infidelity optimised"
    else:
        title_label = "Entanglement constrained"

    title_text = fig.suptitle(
        "",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        i = frame
        returned_artists = []

        title_text.set_text(
            rf"t = {t_fine[i]:.1f} ns, {title_label}: "
            rf"$\theta={THETA / np.pi:.2f}\pi$, $T={T_NS:.0f}$ ns"
        )
        returned_artists.append(title_text)

        if "bloch" in panels and BLOCH_ROTATE:
            frac = i / max(1, len(t_fine) - 1)
            azim = BLOCH_AZIM_START + frac * (BLOCH_AZIM_END - BLOCH_AZIM_START)
            axes["bloch"].view_init(elev=BLOCH_ELEV, azim=azim)

        for c in curves:
            tag = c["tag"]

            if "pulse" in panels:
                seg_i = min(i // N_DISCRETISATION, N_TIME_STEPS - 1)
                J_now = rescale_J([c["res"].x_final[seg_i]])[0]

                pulse_t = np.concatenate([t_seg[:seg_i + 1], [t_fine[i]]])
                pulse_J = np.concatenate([c["J_phys"][:seg_i + 1], [J_now]])

                artists[tag]["pulse_line"].set_data(pulse_t, pulse_J)
                artists[tag]["pulse_marker"].set_data([t_fine[i]], [J_now])

                returned_artists.extend([
                    artists[tag]["pulse_line"],
                    artists[tag]["pulse_marker"],
                ])

            if "infidelity" in panels:
                artists[tag]["inf_line"].set_data(t_fine[:i + 1], c["I"][:i + 1])
                artists[tag]["inf_marker"].set_data([t_fine[i]], [c["I"][i]])
                returned_artists.extend([
                    artists[tag]["inf_line"],
                    artists[tag]["inf_marker"],
                ])

            if "entropy" in panels:
                artists[tag]["ent_line"].set_data(t_fine[:i + 1], c["S"][:i + 1])
                artists[tag]["ent_marker"].set_data([t_fine[i]], [c["S"][i]])
                returned_artists.extend([
                    artists[tag]["ent_line"],
                    artists[tag]["ent_marker"],
                ])

            if "bloch" in panels:
                r_offset = 1.025

                artists[tag]["bloch_line"].set_data(
                    r_offset * c["x"][:i + 1],
                    r_offset * c["y"][:i + 1],
                )
                artists[tag]["bloch_line"].set_3d_properties(
                    r_offset * c["z"][:i + 1],
                )

                artists[tag]["bloch_dot"]._offsets3d = (
                    np.array([r_offset * c["x"][i]]),
                    np.array([r_offset * c["y"][i]]),
                    np.array([r_offset * c["z"][i]]),
                )

                returned_artists.extend([
                    artists[tag]["bloch_line"],
                    artists[tag]["bloch_dot"],
                ])

        return tuple(returned_artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t_fine),
        interval=1000 / fps,
        blit=False,
    )

    panel_tag = "_".join(panels)
    mode_tag = pulse_mode_clean
    rotate_tag = "rotating" if BLOCH_ROTATE and "bloch" in panels else "fixed"
    base_name = f"sqe_state_animation_{mode_tag}_{panel_tag}_{rotate_tag}"

    if save_mp4:
        mp4_path = OUTDIR / f"{base_name}.mp4"
        anim.save(
            mp4_path,
            writer=FFMpegWriter(fps=fps, bitrate=2400),
            dpi=ANIMATION_DPI_MP4,
        )
        print(f"Saved MP4 animation to {mp4_path}")

    if save_gif:
        gif_path = OUTDIR / f"{base_name}.gif"
        anim.save(
            gif_path,
            writer=PillowWriter(fps=fps),
            dpi=ANIMATION_DPI_GIF,
        )
        print(f"Saved GIF animation to {gif_path}")

    plt.show()


# =============================================================================
# Print summary
# =============================================================================

def print_summary(res_base, res_ent) -> None:
    """Print a concise numerical summary of the optimisation results."""
    S_base = entropy_traj_2q(res_base.final_traj)

    print("\n===== SQE-state trajectory visualisation =====")
    print(f"RUN_ENTANGLEMENT_OPT  = {RUN_ENTANGLEMENT_OPTIMISATION}")
    print(f"theta/pi              = {THETA / np.pi:.4f}")
    print(f"T                     = {T_NS:.2f} ns")
    print(f"n_time_steps          = {N_TIME_STEPS}")
    print(f"n_discretisation      = {N_DISCRETISATION}")
    print(f"J_init_seed           = {J_INIT_SEED}")
    print(f"S_MAX                 = {S_MAX:.3e}")
    print(f"EPS_INFIDELITY        = {EPS_INFIDELITY:.3e}")
    print(f"N_OUTER               = {N_OUTER}")
    print(f"DUAL_LR               = {DUAL_LR:.3e}")
    print()
    print(f"I-only final infid    = {float(res_base.val_final[-1]):.6e}")
    print(f"I-only mean S         = {mean_entropy(S_base):.6e}")
    print(f"I-only peak S         = {float(np.max(S_base)):.6e}")

    if res_ent is not None:
        S_ent = entropy_traj_2q(res_ent.final_traj)
        print()
        print(f"S-constr final infid  = {float(res_ent.val_final[-1]):.6e}")
        print(f"S-constr mean S       = {mean_entropy(S_ent):.6e}")
        print(f"mean S reduction      = {mean_entropy(S_base) - mean_entropy(S_ent):.6e}")
        print(f"S-constr peak S       = {float(np.max(S_ent)):.6e}")
        if getattr(res_ent, "lam_final", None) is not None:
            print(f"lambda final          = {float(res_ent.lam_final):.6e}")

    print("==============================================\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    res_base, res_ent = run_optimisation()

    print_summary(res_base, res_ent)
    plot_summary(res_base, res_ent)
    plot_bloch_sphere(res_base, res_ent)

    if MAKE_ANIMATION:
        animate_trajectory(
            res_base,
            res_ent,
            panels=ANIMATION_PANELS,
            pulse_mode=ANIMATION_PULSE_MODE,
            save_mp4=SAVE_ANIMATION_MP4,
            save_gif=SAVE_ANIMATION_GIF,
            fps=ANIMATION_FPS,
        )


if __name__ == "__main__":
    main()
