#!/usr/bin/env python3
"""
plot_noisy_pulse_database_met.py

Generic noisy-MET plotting script for raw pulse databases.

Works for:
    - sqe_state
    - sqe_gate
    - dqe_state
    - dqe_gate

This script:
    1. Loads a raw pulse database.
    2. Selects pulses locally at each (theta, T):
         baseline: lowest clean infidelity at that exact grid point.
         TEM: lowest entanglement among successful clean runs at that exact grid point;
              if no successful TEM run exists, lowest clean infidelity at that grid point.
    3. Evolves the selected pulse under one fixed T1/T2 Lindblad noise model.
    4. Plots:
         - baseline noisy infidelity map
         - TEM noisy infidelity map
         - improvement ratio map
    5. Saves figures only.

No duration overriding.
No output .npz.
No re-optimisation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm

from qugradlab.systems.semiconducting.esr import SpinChain


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import NS_TO_US
from pulse_vqe_entanglement.sqe_state import single_qubit_excitation
from pulse_vqe_entanglement.dqe_state import dqe_unitary, basis_state


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE/"
    "sqe_gate_mixedJ_pulses.npz"
)

# Output goes to the same directory as this plotting script.
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

PLOT_PREFIX = INPUT_FILE.stem

N_WORKERS = 9
CHECKPOINT_EVERY_PRINT = 50

T1_US = 1.0e6
T2_US = 1.0e3

INFID_FLOOR = 1e-16
CLEAN_INF_THRESHOLD = 1e-5
MET_THRESHOLD = 1e-5

# Entanglement metric used for TEM local selection.
# "avg" matches your met_ent average-entanglement story.
# "max" can be used if needed.
ENTANGLEMENT_STAT = "avg"
# ENTANGLEMENT_STAT = "max"

N_DISCRETISATION = 1


# =============================================================================
# Operators
# =============================================================================

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Same convention as your previous noisy scripts.
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)


def op_on_qubit(op, q, n):
    out = np.array([[1.0]], dtype=np.complex128)
    for i in range(n):
        out = np.kron(out, op if i == q else I2)
    return out


def z_ops(n):
    return [op_on_qubit(Z, q, n) for q in range(n)]


# =============================================================================
# Database helpers
# =============================================================================

def scalar_str(x):
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def get_array(data, *names):
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of these arrays found: {names}")


def infer_system_name(data, input_file):
    if "system_name" in data.files:
        return scalar_str(data["system_name"])

    name = input_file.stem.lower()
    if "sqe" in name and "gate" in name:
        return "sqe_gate"
    if "sqe" in name and "state" in name:
        return "sqe_state"
    if "dqe" in name and "gate" in name:
        return "dqe_gate"
    if "dqe" in name and "state" in name:
        return "dqe_state"

    raise ValueError(
        "Could not infer system_name. Store system_name in the database or "
        "use a filename containing sqe/dqe and state/gate."
    )


def infer_excitation_and_target(system_name):
    s = system_name.lower()

    if "sqe" in s:
        excitation = "sqe"
    elif "dqe" in s:
        excitation = "dqe"
    else:
        raise ValueError(f"Could not infer excitation from system_name={system_name!r}")

    if "state" in s:
        target_kind = "state"
    elif "gate" in s:
        target_kind = "gate"
    else:
        raise ValueError(f"Could not infer target kind from system_name={system_name!r}")

    return excitation, target_kind


def infer_n_spins(excitation, zeeman):
    if excitation == "sqe":
        return 2
    if excitation == "dqe":
        return 4
    return len(zeeman)


def infer_n_J_channels(excitation, data, pulse_width, max_n_time_steps):
    if "n_J_channels" in data.files:
        return int(data["n_J_channels"])

    if max_n_time_steps > 0 and pulse_width % max_n_time_steps == 0:
        n_channels = pulse_width // max_n_time_steps
        if n_channels >= 1:
            return int(n_channels)

    if excitation == "sqe":
        return 1
    if excitation == "dqe":
        return 3

    raise ValueError("Could not infer n_J_channels.")


def infer_initial_state(data, excitation, target_kind, n_spins):
    if target_kind == "state":
        if "initial_state" in data.files:
            return np.asarray(data["initial_state"], dtype=np.complex128)

        if excitation == "dqe":
            return basis_state(4, "0011")

        raise KeyError("State target requires initial_state in database.")

    # Dummy initial state for channel evolution.
    dim = 2 ** n_spins
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def get_entanglement_array(data, label, target_kind):
    if target_kind == "state":
        if label == "baseline":
            if ENTANGLEMENT_STAT == "avg":
                return get_array(data, "baseline_s_avg", "baseline_s_max")
            return get_array(data, "baseline_s_max", "baseline_s_avg")
        if label == "ent":
            if ENTANGLEMENT_STAT == "avg":
                return get_array(data, "ent_s_avg", "ent_s_max")
            return get_array(data, "ent_s_max", "ent_s_avg")

    if target_kind == "gate":
        if label == "baseline":
            if ENTANGLEMENT_STAT == "avg":
                return get_array(data, "baseline_ent_avg", "baseline_ent_max")
            return get_array(data, "baseline_ent_max", "baseline_ent_avg")
        if label == "ent":
            if ENTANGLEMENT_STAT == "avg":
                return get_array(data, "ent_ent_avg", "ent_ent_max")
            return get_array(data, "ent_ent_max", "ent_ent_avg")

    raise ValueError((label, target_kind))


# =============================================================================
# Local selection only
# =============================================================================

def select_baseline_local(data):
    """
    Baseline selection at each exact (theta, T):
        choose lowest clean infidelity among all baseline inits.
    """
    infids = get_array(data, "baseline_final_infids")

    n_theta, n_T, _ = infids.shape
    seed_idx = np.full((n_theta, n_T), -1, dtype=int)

    for i in range(n_theta):
        for j in range(n_T):
            finite = np.isfinite(infids[i, j])
            if not np.any(finite):
                continue

            idx = np.where(finite)[0]
            seed_idx[i, j] = idx[np.nanargmin(infids[i, j, finite])]

    return seed_idx


def select_ent_local(data, target_kind, threshold):
    """
    TEM selection at each exact (theta, T):
        if any successful clean TEM run exists:
            choose lowest-entanglement successful run.
        otherwise:
            choose lowest clean infidelity run.

    No duration overriding.
    """
    infids = get_array(data, "ent_final_infids")
    ent_values = get_entanglement_array(data, "ent", target_kind)

    n_theta, n_T, _ = infids.shape
    seed_idx = np.full((n_theta, n_T), -1, dtype=int)
    selected_successful = np.full((n_theta, n_T), False, dtype=bool)

    for i in range(n_theta):
        for j in range(n_T):
            finite = np.isfinite(infids[i, j]) & np.isfinite(ent_values[i, j])
            successful = finite & (infids[i, j] <= threshold)

            if np.any(successful):
                idx = np.where(successful)[0]
                seed_idx[i, j] = idx[np.nanargmin(ent_values[i, j, successful])]
                selected_successful[i, j] = True
            else:
                finite_I = np.isfinite(infids[i, j])
                if not np.any(finite_I):
                    continue
                idx = np.where(finite_I)[0]
                seed_idx[i, j] = idx[np.nanargmin(infids[i, j, finite_I])]
                selected_successful[i, j] = False

    return seed_idx, selected_successful


def gather_selected_scalar_local(arr, seed_idx):
    out = np.full(seed_idx.shape, np.nan)
    for i in range(seed_idx.shape[0]):
        for j in range(seed_idx.shape[1]):
            k = seed_idx[i, j]
            if k >= 0:
                out[i, j] = arr[i, j, k]
    return out


def get_pulse_local(pulse_array, i, j, seed, n_steps, n_J_channels):
    if seed < 0:
        return None
    width = int(n_steps) * int(n_J_channels)
    return np.asarray(pulse_array[i, j, seed, :width], dtype=float)


# =============================================================================
# Targets
# =============================================================================

def free_rotating_unitary(zeeman, T_ns, n_spins):
    deltaB = zeeman - np.mean(zeeman)
    H_rot = sum(deltaB[q] * z_ops(n_spins)[q] for q in range(n_spins))
    return expm(-1j * np.pi * H_rot * T_ns * NS_TO_US)


def target_unitary(excitation, theta, T_ns, zeeman):
    n_spins = infer_n_spins(excitation, zeeman)

    if excitation == "sqe":
        U_exc = single_qubit_excitation(theta)
    elif excitation == "dqe":
        U_exc = dqe_unitary(theta)
    else:
        raise ValueError(excitation)

    return free_rotating_unitary(zeeman, T_ns, n_spins) @ U_exc


def target_rho(excitation, theta, T_ns, zeeman, initial_state):
    psi = target_unitary(excitation, theta, T_ns, zeeman) @ initial_state
    return np.outer(psi, psi.conj())


def target_superoperator(excitation, theta, T_ns, zeeman):
    U = target_unitary(excitation, theta, T_ns, zeeman)
    return np.kron(U, U.conj())


# =============================================================================
# Noise and system construction
# =============================================================================

def lindblad_ops_T1_T2(n_spins, T1_us, T2_us):
    gamma_1 = 1.0 / T1_us
    gamma_phi = max(0.0, 1.0 / T2_us - 0.5 / T1_us)

    ops = []

    for q in range(n_spins):
        ops.append(np.sqrt(gamma_1 / 4.0) * op_on_qubit(SIGMA_MINUS, q, n_spins))

    if gamma_phi > 0.0:
        for q in range(n_spins):
            ops.append(np.sqrt(gamma_phi / 4.0) * op_on_qubit(Z, q, n_spins))

    return np.asarray(ops, dtype=np.complex128)


def drive_frequencies_for_pulse_form(excitation, zeeman):
    """
    Matches your previous noisy scripts:
        SQE: pass ZEEMAN.
        DQE: pass zeros.
    """
    if excitation == "sqe":
        return np.asarray(zeeman, dtype=np.float64)
    if excitation == "dqe":
        return np.zeros_like(zeeman, dtype=np.float64)
    raise ValueError(excitation)


def make_noisy_driven_system(
    *,
    excitation,
    target_kind,
    T_ns,
    n_steps,
    n_discretisation,
    n_spins,
    n_J_channels,
    zeeman,
    initial_state,
    J_min,
    J_max,
    L0,
):
    deltaB = zeeman - np.mean(zeeman)

    sys_obj = SpinChain(
        spins=n_spins,
        zeeman_splittings=deltaB,
        max_drive_strength=0.0,
        J_max=J_max,
        J_min=J_min,
        feromagnetic=False,
        use_graph=True,
    ).add_lindblad_terms(L0=L0)

    n_fine = int(n_steps) * int(n_discretisation)
    dt_fine_us = T_ns / n_fine * NS_TO_US

    drive_freqs = drive_frequencies_for_pulse_form(excitation, zeeman)

    rho0 = np.outer(initial_state, initial_state.conj())

    def pulse_function(x):
        x = tf.cast(tf.convert_to_tensor(x), tf.complex128)
        x = tf.reshape(x, (n_steps, n_J_channels))
        x = tf.repeat(x, repeats=n_discretisation, axis=0)

        return (
            tf.zeros((n_fine, n_spins), dtype=tf.complex128),
            tf.cast(tf.convert_to_tensor(drive_freqs), tf.complex128),
            x,
            tf.cast(tf.convert_to_tensor(rho0), tf.complex128),
            tf.cast(dt_fine_us * 2 * np.pi, tf.complex128),
        )

    return sys_obj.pulse_form(pulse_function)


# =============================================================================
# Infidelity
# =============================================================================

def state_infidelity(rho, rho_target):
    return max(1.0 - float(np.real(np.trace(rho @ rho_target))), INFID_FLOOR)


def gate_infidelity(channel, target_channel):
    """
    Channel/superoperator average infidelity.

    Here d_super = channel.shape[0].
    This is the same convention you used for gate-channel comparison in the
    notebook-style code.
    """
    d_super = channel.shape[0]
    val = 1.0 - (
        np.abs(np.trace(target_channel.conj().T @ channel)) ** 2 + d_super
    ) / (d_super * (d_super + 1))
    return max(float(np.real(val)), INFID_FLOOR)


def noisy_infidelity_worker(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    i = task["i"]
    j = task["j"]
    label = task["label"]

    try:
        excitation = task["excitation"]
        target_kind = task["target_kind"]
        theta = task["theta"]
        T_ns = task["T_ns"]
        n_steps = task["n_steps"]
        n_discretisation = task["n_discretisation"]
        n_spins = task["n_spins"]
        n_J_channels = task["n_J_channels"]
        zeeman = task["zeeman"]
        initial_state = task["initial_state"]
        J_min = task["J_min"]
        J_max = task["J_max"]
        pulse = task["pulse"]

        L0 = lindblad_ops_T1_T2(n_spins, task["T1_us"], task["T2_us"])

        driven = make_noisy_driven_system(
            excitation=excitation,
            target_kind=target_kind,
            T_ns=T_ns,
            n_steps=n_steps,
            n_discretisation=n_discretisation,
            n_spins=n_spins,
            n_J_channels=n_J_channels,
            zeeman=zeeman,
            initial_state=initial_state,
            J_min=J_min,
            J_max=J_max,
            L0=L0,
        )

        if target_kind == "state":
            rho = driven.propagate(pulse)
            rho_t = target_rho(excitation, theta, T_ns, zeeman, initial_state)
            noisy_I = state_infidelity(rho, rho_t)

        elif target_kind == "gate":
            channel = driven.get_evolution(pulse)
            target_channel = target_superoperator(excitation, theta, T_ns, zeeman)
            noisy_I = gate_infidelity(channel, target_channel)

        else:
            raise ValueError(target_kind)

        return dict(
            i=i,
            j=j,
            label=label,
            failed=False,
            error="",
            noisy_infid=noisy_I,
        )

    except Exception as exc:
        return dict(
            i=i,
            j=j,
            label=label,
            failed=True,
            error=repr(exc),
            noisy_infid=np.nan,
        )


# =============================================================================
# Plotting
# =============================================================================

def add_met_contour(ax, theta_over_pi, durations_ns, logI_T_theta):
    level = np.log10(MET_THRESHOLD)

    if not np.any(np.isfinite(logI_T_theta)):
        return None

    zmin = np.nanmin(logI_T_theta)
    zmax = np.nanmax(logI_T_theta)

    if not (zmin <= level <= zmax):
        return None

    cs = ax.contour(
        theta_over_pi,
        durations_ns,
        logI_T_theta,
        levels=[level],
        colors="white",
        linestyles="--",
        linewidths=1.8,
    )
    return cs


def plot_noisy_maps(
    *,
    thetas,
    durations_ns,
    base_noisy_infids,
    ent_noisy_infids,
    system_name,
):
    theta_over_pi = thetas / np.pi

    base = np.maximum(base_noisy_infids, INFID_FLOOR)
    ent = np.maximum(ent_noisy_infids, INFID_FLOOR)

    log_base = np.log10(base).T
    log_ent = np.log10(ent).T

    ratio = base / ent
    log_ratio = np.log10(ratio).T

    # Same colour scale for the first two infidelity maps only.
    infid_vmin = np.nanmin([np.nanmin(log_base), np.nanmin(log_ent)])
    infid_vmax = np.nanmax([np.nanmax(log_base), np.nanmax(log_ent)])

    # Independent colour scale for the improvement-ratio map.
    ratio_vmin = np.nanmin(log_ratio)
    ratio_vmax = np.nanmax(log_ratio)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # First two panels: noisy infidelity, shared scale and shared colourbar
    # -------------------------------------------------------------------------

    im0 = axes[0].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_base,
        shading="auto",
        cmap="inferno",
        vmin=infid_vmin,
        vmax=infid_vmax,
    )
    cs0 = add_met_contour(axes[0], theta_over_pi, durations_ns, log_base)

    im1 = axes[1].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_ent,
        shading="auto",
        cmap="inferno",
        vmin=infid_vmin,
        vmax=infid_vmax,
    )
    cs1 = add_met_contour(axes[1], theta_over_pi, durations_ns, log_ent)

    axes[0].set_title("Baseline pulse")
    axes[1].set_title("TEM pulse")

    axes[0].set_ylabel("Duration / ns")
    axes[0].set_xlabel(r"$\theta/\pi$")
    axes[1].set_xlabel(r"$\theta/\pi$")

    cbar_inf = fig.colorbar(
        im1,
        ax=axes[:2],
        pad=0.02,
        fraction=0.046,
    )
    cbar_inf.set_label(r"$\log_{10}(\mathrm{noisy\ final\ infidelity})$")

    contour_handle = None
    if cs0 is not None:
        contour_handle = cs0.legend_elements()[0][0]
    elif cs1 is not None:
        contour_handle = cs1.legend_elements()[0][0]

    if contour_handle is not None:
        axes[1].legend(
            [contour_handle],
            [rf"$I={MET_THRESHOLD:.0e}$"],
            loc="upper right",
            frameon=True,
        )

    # -------------------------------------------------------------------------
    # Third panel: improvement ratio, independent scale and colourbar
    # -------------------------------------------------------------------------

    im2 = axes[2].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_ratio,
        shading="auto",
        cmap="inferno",
        vmin=ratio_vmin,
        vmax=ratio_vmax,
    )

    axes[2].set_title(r"$\log_{10}(I_{\rm base}/I_{\rm TEM})$")
    axes[2].set_xlabel(r"$\theta/\pi$")

    cbar_ratio = fig.colorbar(
        im2,
        ax=axes[2],
        pad=0.02,
        fraction=0.046,
    )
    cbar_ratio.set_label(r"$\log_{10}(I_{\rm base}/I_{\rm TEM})$")

    fig.suptitle(
        rf"{system_name} noisy MET, $T_1={T1_US:.2e}\,\mu s$, "
        rf"$T_2={T2_US:.2e}\,\mu s$"
    )

    path = OUTDIR / f"{PLOT_PREFIX}_noisy_met.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

# =============================================================================
# Main
# =============================================================================

def main():
    raw = np.load(INPUT_FILE, allow_pickle=True)

    system_name = infer_system_name(raw, INPUT_FILE)
    excitation, target_kind = infer_excitation_and_target(system_name)

    thetas = get_array(raw, "thetas", "excitation_strengths")
    durations_ns = get_array(raw, "durations_ns", "pulse_durations")

    zeeman = np.asarray(get_array(raw, "zeeman"), dtype=np.float64)
    n_spins = infer_n_spins(excitation, zeeman)

    initial_state = infer_initial_state(raw, excitation, target_kind, n_spins)

    J_min = float(get_array(raw, "J_min"))
    J_max = float(get_array(raw, "J_max"))

    max_n_time_steps = int(get_array(raw, "max_n_time_steps"))
    n_steps_used = get_array(raw, "n_time_steps_used", "n_steps_used").astype(int)

    threshold = (
        float(raw["infidelity_threshold"])
        if "infidelity_threshold" in raw.files
        else CLEAN_INF_THRESHOLD
    )

    baseline_x_final = get_array(raw, "baseline_x_final")
    ent_x_final = get_array(raw, "ent_x_final")

    pulse_width = baseline_x_final.shape[-1]
    n_J_channels = infer_n_J_channels(
        excitation,
        raw,
        pulse_width,
        max_n_time_steps,
    )

    baseline_seed = select_baseline_local(raw)
    ent_seed, ent_selected_successful = select_ent_local(
        raw,
        target_kind,
        threshold,
    )

    baseline_final_infids = get_array(raw, "baseline_final_infids")
    ent_final_infids = get_array(raw, "ent_final_infids")

    base_clean_infids = gather_selected_scalar_local(
        baseline_final_infids,
        baseline_seed,
    )
    ent_clean_infids = gather_selected_scalar_local(
        ent_final_infids,
        ent_seed,
    )

    n_theta = len(thetas)
    n_T = len(durations_ns)

    base_noisy_infids = np.full((n_theta, n_T), np.nan)
    ent_noisy_infids = np.full((n_theta, n_T), np.nan)

    failed_base = np.full((n_theta, n_T), False, dtype=bool)
    failed_ent = np.full((n_theta, n_T), False, dtype=bool)
    error_base = np.full((n_theta, n_T), "", dtype=object)
    error_ent = np.full((n_theta, n_T), "", dtype=object)

    tasks = []

    for i, theta in enumerate(thetas):
        for j, T_ns in enumerate(durations_ns):
            n_steps = int(n_steps_used[i, j])

            b_seed = baseline_seed[i, j]
            if b_seed >= 0:
                b_pulse = get_pulse_local(
                    baseline_x_final,
                    i=i,
                    j=j,
                    seed=b_seed,
                    n_steps=n_steps,
                    n_J_channels=n_J_channels,
                )

                tasks.append(dict(
                    i=i,
                    j=j,
                    label="base",
                    excitation=excitation,
                    target_kind=target_kind,
                    theta=float(theta),
                    T_ns=float(T_ns),
                    n_steps=n_steps,
                    n_discretisation=N_DISCRETISATION,
                    n_spins=n_spins,
                    n_J_channels=n_J_channels,
                    zeeman=zeeman,
                    initial_state=initial_state,
                    J_min=J_min,
                    J_max=J_max,
                    T1_us=T1_US,
                    T2_us=T2_US,
                    pulse=b_pulse,
                ))

            e_seed = ent_seed[i, j]
            if e_seed >= 0:
                e_pulse = get_pulse_local(
                    ent_x_final,
                    i=i,
                    j=j,
                    seed=e_seed,
                    n_steps=n_steps,
                    n_J_channels=n_J_channels,
                )

                tasks.append(dict(
                    i=i,
                    j=j,
                    label="ent",
                    excitation=excitation,
                    target_kind=target_kind,
                    theta=float(theta),
                    T_ns=float(T_ns),
                    n_steps=n_steps,
                    n_discretisation=N_DISCRETISATION,
                    n_spins=n_spins,
                    n_J_channels=n_J_channels,
                    zeeman=zeeman,
                    initial_state=initial_state,
                    J_min=J_min,
                    J_max=J_max,
                    T1_us=T1_US,
                    T2_us=T2_US,
                    pulse=e_pulse,
                ))

    print("=" * 72)
    print("Noisy pulse-database MET plot")
    print(f"INPUT_FILE = {INPUT_FILE}")
    print(f"OUTDIR = {OUTDIR}")
    print(f"system_name = {system_name}")
    print(f"excitation = {excitation}")
    print(f"target_kind = {target_kind}")
    print(f"n_spins = {n_spins}")
    print(f"n_J_channels = {n_J_channels}")
    print(f"T1 = {T1_US:.6e} us")
    print(f"T2 = {T2_US:.6e} us")
    print("TEM selection = local lowest-entanglement successful run,")
    print("                otherwise local lowest clean-infidelity run")
    print(f"Total noisy evolutions = {len(tasks)}")
    print("=" * 72)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(noisy_infidelity_worker, task) for task in tasks]

        with tqdm(total=len(futures), desc="noisy evolutions", unit="pulse") as pbar:
            for n_done, future in enumerate(as_completed(futures), start=1):
                result = future.result()

                i = result["i"]
                j = result["j"]
                label = result["label"]

                if label == "base":
                    base_noisy_infids[i, j] = result["noisy_infid"]
                    failed_base[i, j] = result["failed"]
                    error_base[i, j] = result["error"]
                elif label == "ent":
                    ent_noisy_infids[i, j] = result["noisy_infid"]
                    failed_ent[i, j] = result["failed"]
                    error_ent[i, j] = result["error"]
                else:
                    raise ValueError(label)

                if result["failed"]:
                    pbar.write(
                        f"FAILED {label} theta_idx={i}, T_idx={j}: {result['error']}"
                    )

                if CHECKPOINT_EVERY_PRINT and n_done % CHECKPOINT_EVERY_PRINT == 0:
                    done_base = np.sum(np.isfinite(base_noisy_infids))
                    done_ent = np.sum(np.isfinite(ent_noisy_infids))
                    pbar.write(
                        f"progress: base={done_base}/{n_theta*n_T}, "
                        f"ent={done_ent}/{n_theta*n_T}"
                    )

                pbar.update(1)

    plot_noisy_maps(
        thetas=thetas,
        durations_ns=durations_ns,
        base_noisy_infids=base_noisy_infids,
        ent_noisy_infids=ent_noisy_infids,
        system_name=system_name,
    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Best clean baseline infidelity = {np.nanmin(base_clean_infids):.6e}")
    print(f"Best clean TEM infidelity      = {np.nanmin(ent_clean_infids):.6e}")
    print(f"Best noisy baseline infidelity = {np.nanmin(base_noisy_infids):.6e}")
    print(f"Best noisy TEM infidelity      = {np.nanmin(ent_noisy_infids):.6e}")
    print(f"Max noisy improvement ratio    = {np.nanmax(base_noisy_infids / ent_noisy_infids):.6e}")
    print(f"Failed baseline evolutions     = {np.sum(failed_base)}")
    print(f"Failed TEM evolutions          = {np.sum(failed_ent)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
