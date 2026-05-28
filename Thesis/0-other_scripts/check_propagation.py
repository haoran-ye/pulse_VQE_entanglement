#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


DATA = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE/"
    "sqe_gate_positiveJ_rowprop_pulses.npz"
)

INFID_THRESHOLD = 1e-5


def fraction(n, d):
    return float(n) / float(d) if d else np.nan


def selected_good_count(
    *,
    accepted,
    selected_slot,
    final_infids,
    required_slot=0,
    threshold=INFID_THRESHOLD,
):
    count = 0

    n_theta, n_T = accepted.shape

    for i in range(n_theta):
        for j in range(n_T):
            if not accepted[i, j]:
                continue

            k = int(selected_slot[i, j])
            if k != required_slot:
                continue

            infid = final_infids[i, j, k]

            if np.isfinite(infid) and infid <= threshold:
                count += 1

    return count


def make_propagation_status(*, attempted, accepted, selected_slot):
    """
    Status values:
        0 = not attempted
        1 = attempted but rejected
        2 = accepted
        3 = accepted and selected
    """
    status = np.zeros(accepted.shape, dtype=int)

    status[attempted] = 1
    status[accepted] = 2
    status[accepted & (selected_slot == 0)] = 3

    return status


def plot_status_map(ax, theta_over_pi, durations_ns, status, title):
    cmap = ListedColormap([
        "white",       # 0 not attempted
        "lightgray",   # 1 attempted rejected
        "tab:orange",  # 2 accepted
        "tab:green",   # 3 accepted selected
    ])

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.pcolormesh(
        theta_over_pi,
        durations_ns,
        status.T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.set_title(title)
    ax.set_xlabel(r"Excitation strength $\theta/\pi$")
    ax.set_ylabel(r"Pulse duration $T$ / ns")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_box_aspect(1)

    return im


def main():

    data = np.load(DATA)

    thetas = data["thetas"]
    durations_ns = data["durations_ns"]
    theta_over_pi = thetas / np.pi

    baseline_prop_used = data["baseline_used_propagated_init"]
    ent_prop_used = data["ent_used_propagated_init"]

    baseline_prop_accepted = data["baseline_propagation_accepted"]
    ent_prop_accepted = data["ent_propagation_accepted"]

    baseline_final_infids = data["baseline_final_infids"]
    ent_final_infids = data["ent_final_infids"]

    baseline_selected_slot = data["baseline_selected_slot"]
    ent_selected_slot = data["ent_selected_slot"]

    n_theta = len(thetas)
    n_T = len(durations_ns)

    total_points = n_theta * n_T

    baseline_attempted = baseline_prop_used[:, :, 0]
    ent_attempted = ent_prop_used[:, :, 0]

    baseline_prop_attempted = int(np.sum(baseline_attempted))
    baseline_prop_successful = int(np.sum(baseline_prop_accepted))
    baseline_prop_successful_and_good = selected_good_count(
        accepted=baseline_prop_accepted,
        selected_slot=baseline_selected_slot,
        final_infids=baseline_final_infids,
        required_slot=0,
    )

    ent_prop_attempted = int(np.sum(ent_attempted))
    ent_prop_successful = int(np.sum(ent_prop_accepted))
    ent_prop_successful_and_good = selected_good_count(
        accepted=ent_prop_accepted,
        selected_slot=ent_selected_slot,
        final_infids=ent_final_infids,
        required_slot=0,
    )

    print("=" * 72)
    print("ROW-PROPAGATION SUMMARY")
    print("=" * 72)

    print(f"Data file                  : {DATA}")
    print(f"Total theta values         : {n_theta}")
    print(f"Total duration values      : {n_T}")
    print(f"Total grid points          : {total_points}")

    print()

    print("-" * 72)
    print("BASELINE")
    print("-" * 72)

    print(f"Propagation attempted      : {baseline_prop_attempted}")
    print(f"Propagation accepted       : {baseline_prop_successful}")
    print(f"Accepted + selected + good : {baseline_prop_successful_and_good}")
    print(f"Attempt fraction           : {fraction(baseline_prop_attempted, total_points):.3f}")
    print(f"Acceptance fraction        : {fraction(baseline_prop_successful, total_points):.3f}")
    print(f"Accepted / attempted       : {fraction(baseline_prop_successful, baseline_prop_attempted):.3f}")

    print()

    print("-" * 72)
    print("TEM")
    print("-" * 72)

    print(f"Propagation attempted      : {ent_prop_attempted}")
    print(f"Propagation accepted       : {ent_prop_successful}")
    print(f"Accepted + selected + good : {ent_prop_successful_and_good}")
    print(f"Attempt fraction           : {fraction(ent_prop_attempted, total_points):.3f}")
    print(f"Acceptance fraction        : {fraction(ent_prop_successful, total_points):.3f}")
    print(f"Accepted / attempted       : {fraction(ent_prop_successful, ent_prop_attempted):.3f}")

    print("=" * 72)

    baseline_status = make_propagation_status(
        attempted=baseline_attempted,
        accepted=baseline_prop_accepted,
        selected_slot=baseline_selected_slot,
    )

    ent_status = make_propagation_status(
        attempted=ent_attempted,
        accepted=ent_prop_accepted,
        selected_slot=ent_selected_slot,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    im0 = plot_status_map(
        axes[0],
        theta_over_pi,
        durations_ns,
        baseline_status,
        "Baseline propagation",
    )

    im1 = plot_status_map(
        axes[1],
        theta_over_pi,
        durations_ns,
        ent_status,
        "TEM propagation",
    )

    cbar = fig.colorbar(
        im1,
        ax=axes,
        ticks=[0, 1, 2, 3],
        pad=0.02,
        fraction=0.046,
    )

    cbar.ax.set_yticklabels([
        "not attempted",
        "attempted rejected",
        "accepted",
        "accepted selected",
    ])

    plt.show()


if __name__ == "__main__":
    main()