#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_state import sqe_optimisation, single_qubit_excitation
from pulse_vqe_entanglement.entanglement_measures import entropy_bipartite_single


# =============================================================================
# User settings
# =============================================================================

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/trajectory_visualisation/sqe_state/"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

RUN_ENTANGLEMENT_OPTIMISATION = True

THETA = 0.05 * np.pi
T_NS = 300.0
N_TIME_STEPS = 50
N_DISCRETISATION = 1

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
N_OUTER = 5
INNER_MAXITER = 200
FD_EPS = 1e-6

GTOL = 1e-12

STATE_ENT_MEASURE = "von_neumann"

INITIAL_STATE = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
ZEEMAN_SPLITTINGS = np.array([1000.0, 1008.0])

# Bloch sphere heatmap intensity.
SURFACE_ALPHA = 0.32
SURFACE_BLEND = 0.12


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

def make_config():
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


def rescale_J(x):
    return 0.5 * (J_MAX - J_MIN) * np.asarray(x) + 0.5 * (J_MAX + J_MIN)


def as_time_major_traj(traj):
    traj = np.asarray(traj, dtype=np.complex128)

    if traj.ndim != 2:
        raise ValueError(f"Expected 2D trajectory, got shape {traj.shape}")

    if traj.shape[0] == 4:
        return traj.T

    if traj.shape[1] == 4:
        return traj

    raise ValueError(f"Cannot interpret trajectory shape {traj.shape}")


def entropy_traj_2q(traj):
    traj_tm = as_time_major_traj(traj)
    return np.array([
        entropy_bipartite_single(psi, subsystem=(0,), n_qubits=2)
        for psi in traj_tm
    ])


def mean_entropy(S):
    return float(np.mean(np.asarray(S, float)))


def binary_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1.0 - 1e-15)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p)) / np.log(2.0)


def projected_bloch_coords(traj):
    traj_tm = as_time_major_traj(traj)

    a = traj_tm[:, 1]
    b = traj_tm[:, 2]

    norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
    norm = np.where(norm < 1e-14, 1.0, norm)

    a = a / norm
    b = b / norm

    x = 2.0 * np.real(np.conj(a) * b)
    y = 2.0 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2

    return x, y, z


def get_baseline_and_ent_results(results):
    if RUN_ENTANGLEMENT_OPTIMISATION:
        if results.baseline is None:
            raise RuntimeError("Expected results.baseline from sqe_optimisation.")
        return results.baseline, results

    return results, None


def mode_suffix():
    return "entanglement_constrained" if RUN_ENTANGLEMENT_OPTIMISATION else "infidelity_only"


def hide_3d_axes(ax):
    ax.set_axis_off()
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_alpha(0.0)


def add_clean_bloch_axes(ax, length=1.28):
    arrow_kw = dict(
        arrow_length_ratio=0.08,
        linewidth=1.2,
        color="black",
        alpha=0.75,
    )

    ax.quiver(0, 0, 0, length, 0, 0, **arrow_kw)
    ax.quiver(0, 0, 0, 0, length, 0, **arrow_kw)
    ax.quiver(0, 0, 0, 0, 0, length, **arrow_kw)

    ax.text(length + 0.06, 0, 0, r"$x$", ha="center", va="center")
    ax.text(0, length + 0.06, 0, r"$y$", ha="center", va="center")
    ax.text(0, 0, length + 0.06, r"$|01\rangle$", ha="center", va="center")

    ax.text(0, 0, -length - 0.08, r"$|10\rangle$", ha="center", va="center")

    #ax.text(length / np.sqrt(2) + 0.08, 0, length / np.sqrt(2),
    #        r"$\frac{|01\rangle+|10\rangle}{\sqrt{2}}$",
    #        ha="left", va="center")

    #ax.text(0, length / np.sqrt(2) + 0.08, length / np.sqrt(2),
    #        r"$\frac{|01\rangle+i|10\rangle}{\sqrt{2}}$",
    #        ha="left", va="center")


# =============================================================================
# Run optimisation
# =============================================================================

def run_optimisation():
    config = make_config()
    results = sqe_optimisation(config)
    return get_baseline_and_ent_results(results)


# =============================================================================
# Summary plots
# =============================================================================

def plot_summary(res_base, res_ent):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))

    t_seg = np.arange(N_TIME_STEPS + 1) * T_NS / N_TIME_STEPS
    n_fine = N_TIME_STEPS * N_DISCRETISATION
    t_fine = np.arange(n_fine + 1) * T_NS / n_fine

    xb = np.concatenate([res_base.x_final, [res_base.x_final[-1]]])

    axes[0].step(t_seg, rescale_J(xb), where="post",
                 label="Infidelity optimised", color=COLOR_BASELINE)

    if res_ent is not None:
        xe = np.concatenate([res_ent.x_final, [res_ent.x_final[-1]]])
        axes[0].step(t_seg, rescale_J(xe), where="post",
                     label="Entanglement constrained", color=COLOR_ENT)

    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel(r"$J$ (MHz)")
    axes[0].set_title("Control pulse")
    axes[0].legend()

    axes[1].plot(t_fine, np.maximum(res_base.val_final, 1e-16),
                 label="Infidelity optimised", color=COLOR_BASELINE)

    if res_ent is not None:
        axes[1].plot(t_fine, np.maximum(res_ent.val_final, 1e-16),
                     label="Entanglement constrained", color=COLOR_ENT)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Infidelity")
    axes[1].set_title("Infidelity evolution")
    axes[1].legend()

    S_base = entropy_traj_2q(res_base.final_traj)
    axes[2].plot(t_fine, S_base, label="Infidelity optimised", color=COLOR_BASELINE)

    if res_ent is not None:
        S_ent = entropy_traj_2q(res_ent.final_traj)
        axes[2].plot(t_fine, S_ent, label="Entanglement constrained", color=COLOR_ENT)

    target_state = single_qubit_excitation(THETA) @ INITIAL_STATE
    S_target = entropy_bipartite_single(target_state, subsystem=(0,), n_qubits=2)

    axes[2].hlines(S_target, 0, T_NS, colors=COLOR_TARGET,
                   linestyles="dashed", label="Target")

    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel(r"$S$ (bits)")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Entanglement entropy trajectory")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUTDIR / f"sqe_state_summary_{mode_suffix()}.png", dpi=300)
    plt.show()


# =============================================================================
# Bloch sphere
# =============================================================================

def plot_bloch_sphere(res_base, res_ent):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)

    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)

    p01 = np.cos(TH / 2) ** 2
    S = binary_entropy(p01)

    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    facecolors = cmap(norm(S))

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

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(S)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label(r"$S$ (bits)")

    xb, yb, zb = projected_bloch_coords(res_base.final_traj)
    r_offset = 1.025

    ax.plot(r_offset * xb, r_offset * yb, r_offset * zb,
            color=COLOR_BASELINE, linewidth=2.2,
            label="Infidelity optimised", zorder=10)

    ax.scatter([r_offset * xb[0]], [r_offset * yb[0]], [r_offset * zb[0]],
               color=COLOR_BASELINE, s=45, marker="o", zorder=12)
    ax.scatter([r_offset * xb[-1]], [r_offset * yb[-1]], [r_offset * zb[-1]],
               color=COLOR_BASELINE, s=70, marker="*", zorder=12)

    if res_ent is not None:
        xe, ye, ze = projected_bloch_coords(res_ent.final_traj)

        ax.plot(r_offset * xe, r_offset * ye, r_offset * ze,
                color=COLOR_ENT, linewidth=2.2,
                label="Entanglement constrained", zorder=11)

        ax.scatter([r_offset * xe[0]], [r_offset * ye[0]], [r_offset * ze[0]],
                   color=COLOR_ENT, s=45, marker="o", zorder=13)
        ax.scatter([r_offset * xe[-1]], [r_offset * ye[-1]], [r_offset * ze[-1]],
                   color=COLOR_ENT, s=70, marker="*", zorder=13)

    add_clean_bloch_axes(ax)
    hide_3d_axes(ax)

    lim = 1.42
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=45)

    ax.set_title(r"Projected SQE trajectory in $\{|01\rangle, |10\rangle\}$")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTDIR / f"sqe_state_bloch_sphere_{mode_suffix()}.png", dpi=300)
    plt.show()


def print_summary(res_base, res_ent):
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


def main():
    res_base, res_ent = run_optimisation()

    print_summary(res_base, res_ent)
    plot_summary(res_base, res_ent)
    plot_bloch_sphere(res_base, res_ent)


if __name__ == "__main__":
    main()