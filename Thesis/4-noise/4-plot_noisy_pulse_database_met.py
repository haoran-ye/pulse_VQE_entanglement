#!/usr/bin/env python3
"""
4-plot_noisy_pulse_database_met.py

Evaluate baseline and TEM pulses under Lindblad noise and plot the noisy
infidelity maps.

Steps
-----
  1. Load a raw pulse database (.npz).
  2. Infer the system type (SQE/DQE, state/gate).
  3. Select the best pulse at each (theta, T) for both baseline and TEM.
  4. Simulate each selected pulse under T1/T2 noise using qugradlab.
  5. Plot a three-panel map (baseline | TEM | improvement ratio).

Physics note — gate_infidelity
------------------------------
  The process fidelity between superoperators S_target and S_expt is:
      F_avg = (Re[Tr(S_target† S_expt)] + d) / (d(d + 1))
  where d is the PHYSICAL dimension (not the superoperator dimension d²).
  The gate infidelity is 1 − F_avg.  (P4 fix)

  Reference: Nielsen, Phys. Lett. A 303 (2002) 249–252.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm

from qugradlab.systems.semiconducting.esr import SpinChain

# B2 fix: pulse_vqe_entanglement is pip-installed; no sys.path hack needed.
# Update: sys.path needed for this

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pulse_vqe_entanglement.dataclasses import NS_TO_US
from pulse_vqe_entanglement.sqe_state import single_qubit_excitation
from pulse_vqe_entanglement.dqe_state import dqe_unitary, basis_state


# =============================================================================
# User settings
# =============================================================================

# B1 fix: INPUT_FILE resolved relative to this script.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
INPUT_FILE = PULSES_DATABASE / "sqe_state_initstate_bell_positiveJ_pulses.npz"

OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

PLOT_PREFIX = INPUT_FILE.stem

N_WORKERS = 9
CHECKPOINT_EVERY_PRINT = 50

T1_US = 1.0e6  # T1 relaxation time (µs)
T2_US = 1.0e3  # T2 dephasing time (µs)

INFID_FLOOR = 1e-16
CLEAN_INF_THRESHOLD = 1e-5
MET_THRESHOLD = 1e-20

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

# B7 fix: renamed to SIGMA_MINUS to match the physical role (lowering/de-excitation
# jump operator for T1 relaxation).  Same matrix [[0,1],[0,0]] as before.
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)


def op_on_qubit(op: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed a single-qubit operator *op* on qubit *q* in an *n*-qubit space."""
    out = np.array([[1.0]], dtype=np.complex128)
    for i in range(n):
        out = np.kron(out, op if i == q else I2)
    return out


def z_ops(n: int) -> list[np.ndarray]:
    """List of single-qubit Z operators on each qubit in an n-qubit system."""
    return [op_on_qubit(Z, q, n) for q in range(n)]


# =============================================================================
# Database helpers
# =============================================================================

def scalar_str(x) -> str:
    arr = np.asarray(x)
    return str(arr.item()) if arr.shape == () else str(x)


def get_array(data, *names: str) -> np.ndarray:
    """Return the first of *names* found in *data.files*."""
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of these arrays found: {names}")


def infer_system_name(data, input_file: Path) -> str:
    """Read system_name from the database or infer it from the filename."""
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
        "Could not infer system_name.  Store system_name in the database or "
        "use a filename containing sqe/dqe and state/gate."
    )


def infer_excitation_and_target(system_name: str) -> tuple[str, str]:
    """Parse system_name into (excitation, target_kind)."""
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


def infer_n_spins(excitation: str, zeeman: np.ndarray) -> int:
    if excitation == "sqe":
        return 2
    if excitation == "dqe":
        return 4
    return len(zeeman)


def infer_n_J_channels(excitation: str, data, pulse_width: int, max_n_time_steps: int) -> int:
    """Read n_J_channels from database, infer from pulse width, or use defaults."""
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


def infer_initial_state(data, excitation: str, target_kind: str, n_spins: int) -> np.ndarray:
    """Return the initial state for state targets, or |0…0⟩ for gate targets."""
    if target_kind == "state":
        if "initial_state" in data.files:
            return np.asarray(data["initial_state"], dtype=np.complex128)
        if excitation == "dqe":
            return basis_state(4, "0011")
        raise KeyError("State target requires initial_state in database.")
    dim = 2 ** n_spins
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def get_entanglement_array(data, label: str, target_kind: str) -> np.ndarray:
    """Return the appropriate entanglement array (state or gate, avg or max)."""
    if target_kind == "state":
        if label == "baseline":
            return (get_array(data, "baseline_s_avg", "baseline_s_max")
                    if ENTANGLEMENT_STAT == "avg"
                    else get_array(data, "baseline_s_max", "baseline_s_avg"))
        if label == "ent":
            return (get_array(data, "ent_s_avg", "ent_s_max")
                    if ENTANGLEMENT_STAT == "avg"
                    else get_array(data, "ent_s_max", "ent_s_avg"))

    if target_kind == "gate":
        if label == "baseline":
            return (get_array(data, "baseline_ent_avg", "baseline_ent_max")
                    if ENTANGLEMENT_STAT == "avg"
                    else get_array(data, "baseline_ent_max", "baseline_ent_avg"))
        if label == "ent":
            return (get_array(data, "ent_ent_avg", "ent_ent_max")
                    if ENTANGLEMENT_STAT == "avg"
                    else get_array(data, "ent_ent_max", "ent_ent_avg"))

    raise ValueError((label, target_kind))


# =============================================================================
# Local selection only
# =============================================================================

def select_baseline_local(data) -> np.ndarray:
    """
    For each (theta, T), return the init index with the lowest finite infidelity.

    Returns index array with -1 where no finite infidelity exists.
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


def select_ent_local(data, target_kind: str, threshold: float):
    """
    For each (theta, T), select the TEM init: lowest-entanglement successful run,
    or lowest-infidelity run if none succeed.

    Returns
    -------
    seed_idx : (n_theta, n_T), -1 where no finite value found.
    selected_successful : (n_theta, n_T) bool.
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

    return seed_idx, selected_successful


def gather_selected_scalar_local(arr: np.ndarray, seed_idx: np.ndarray) -> np.ndarray:
    """Gather the scalar value at the selected init index for each (theta, T)."""
    out = np.full(seed_idx.shape, np.nan)
    for i in range(seed_idx.shape[0]):
        for j in range(seed_idx.shape[1]):
            k = seed_idx[i, j]
            if k >= 0:
                out[i, j] = arr[i, j, k]
    return out


def get_pulse_local(
    pulse_array: np.ndarray, i: int, j: int, seed: int, n_steps: int, n_J_channels: int
) -> np.ndarray | None:
    """Extract the active portion of a stored pulse vector at (i, j, seed)."""
    if seed < 0:
        return None
    width = int(n_steps) * int(n_J_channels)
    return np.asarray(pulse_array[i, j, seed, :width], dtype=float)


# =============================================================================
# Targets
# =============================================================================

def free_rotating_unitary(zeeman: np.ndarray, T_ns: float, n_spins: int) -> np.ndarray:
    """Free-rotation unitary in the rotating frame for duration T_ns."""
    deltaB = zeeman - np.mean(zeeman)
    H_rot = sum(deltaB[q] * z_ops(n_spins)[q] for q in range(n_spins))
    return expm(-1j * np.pi * H_rot * T_ns * NS_TO_US)


def target_unitary(excitation: str, theta: float, T_ns: float, zeeman: np.ndarray) -> np.ndarray:
    """Full target unitary including free rotation in the rotating frame."""
    n_spins = infer_n_spins(excitation, zeeman)
    if excitation == "sqe":
        U_exc = single_qubit_excitation(theta)
    elif excitation == "dqe":
        U_exc = dqe_unitary(theta)
    else:
        raise ValueError(excitation)
    return free_rotating_unitary(zeeman, T_ns, n_spins) @ U_exc


def target_rho(excitation: str, theta: float, T_ns: float, zeeman: np.ndarray,
               initial_state: np.ndarray) -> np.ndarray:
    """Target density matrix for state fidelity calculations."""
    psi = target_unitary(excitation, theta, T_ns, zeeman) @ initial_state
    return np.outer(psi, psi.conj())


def target_superoperator(excitation: str, theta: float, T_ns: float,
                         zeeman: np.ndarray) -> np.ndarray:
    """Target superoperator S_U = U ⊗ U* for gate fidelity calculations."""
    U = target_unitary(excitation, theta, T_ns, zeeman)
    return np.kron(U, U.conj())


# =============================================================================
# Noise and system construction
# =============================================================================

def lindblad_ops_T1_T2(n_spins: int, T1_us: float, T2_us: float) -> np.ndarray:
    """
    Lindblad jump operators for T1 relaxation and T2 pure dephasing.

    T2 dephasing rate: γ_φ = 1/T2 - 1/(2T1)  (pure dephasing contribution).
    """
    gamma_1 = 1.0 / T1_us
    gamma_phi = max(0.0, 1.0 / T2_us - 0.5 / T1_us)

    ops = []
    for q in range(n_spins):
        # T1 relaxation: σ- jump operator on each qubit.
        ops.append(np.sqrt(gamma_1 / 4.0) * op_on_qubit(SIGMA_MINUS, q, n_spins))
    if gamma_phi > 0.0:
        for q in range(n_spins):
            # Pure dephasing: Z jump operator on each qubit.
            ops.append(np.sqrt(gamma_phi / 4.0) * op_on_qubit(Z, q, n_spins))

    return np.asarray(ops, dtype=np.complex128)


def drive_frequencies_for_pulse_form(excitation: str, zeeman: np.ndarray) -> np.ndarray:
    """Drive frequencies: Zeeman splittings for SQE, zero for DQE."""
    if excitation == "sqe":
        return np.asarray(zeeman, dtype=np.float64)
    if excitation == "dqe":
        return np.zeros_like(zeeman, dtype=np.float64)
    raise ValueError(excitation)


def make_noisy_driven_system(
    *,
    excitation: str,
    target_kind: str,
    T_ns: float,
    n_steps: int,
    n_discretisation: int,
    n_spins: int,
    n_J_channels: int,
    zeeman: np.ndarray,
    initial_state: np.ndarray,
    J_min: float,
    J_max: float,
    L0: np.ndarray,
):
    """Build a qugradlab driven Lindblad system for one (theta, T) grid point."""
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
# Infidelity metrics
# =============================================================================

def state_infidelity(rho: np.ndarray, rho_target: np.ndarray) -> float:
    """1 - Tr(ρ ρ_target) for pure target state, clipped to INFID_FLOOR."""
    return max(1.0 - float(np.real(np.trace(rho @ rho_target))), INFID_FLOOR)


def gate_infidelity(channel: np.ndarray, target_channel: np.ndarray) -> float:
    """
    Average gate infidelity between superoperators *channel* and *target_channel*.

    P4 fix: correct formula uses the PHYSICAL dimension d, not the superoperator
    dimension d² = channel.shape[0].

        F_avg = (Re[Tr(S_target† S_expt)] + d) / (d(d + 1))
        I = max(1 - F_avg, INFID_FLOOR)

    Reference: Nielsen (2002), Phys. Lett. A 303, 249–252.
    """
    d_super = channel.shape[0]           # superoperator dim = d² (e.g. 16 for 2 qubits)
    d = int(np.round(np.sqrt(d_super)))  # physical Hilbert-space dimension (e.g. 4)
    trace_val = float(np.real(np.trace(target_channel.conj().T @ channel)))
    f_avg = (trace_val + d) / (d * (d + 1))
    return max(float(1.0 - f_avg), INFID_FLOOR)


# =============================================================================
# Worker (runs in a child process)
# =============================================================================

def noisy_infidelity_worker(task: dict) -> dict:
    """
    Simulate one pulse under Lindblad noise and return the noisy infidelity.

    Thread control is set here (not at import time) because each worker is
    a fresh child process and the env vars must be set before TF initialises.
    """
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

    i, j, label = task["i"], task["j"], task["label"]

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
            excitation=excitation, target_kind=target_kind,
            T_ns=T_ns, n_steps=n_steps, n_discretisation=n_discretisation,
            n_spins=n_spins, n_J_channels=n_J_channels,
            zeeman=zeeman, initial_state=initial_state,
            J_min=J_min, J_max=J_max, L0=L0,
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

        return dict(i=i, j=j, label=label, failed=False, error="", noisy_infid=noisy_I)

    except Exception as exc:
        return dict(i=i, j=j, label=label, failed=True, error=repr(exc), noisy_infid=np.nan)


# =============================================================================
# Plotting
# =============================================================================

def add_met_contour(ax, theta_over_pi: np.ndarray, durations_ns: np.ndarray,
                    logI_T_theta: np.ndarray):
    """Draw the MET contour (log₁₀(I) = log₁₀(MET_THRESHOLD)) on *ax*."""
    level = np.log10(MET_THRESHOLD)
    if not np.any(np.isfinite(logI_T_theta)):
        return None
    zmin = np.nanmin(logI_T_theta)
    zmax = np.nanmax(logI_T_theta)
    if not (zmin <= level <= zmax):
        return None
    cs = ax.contour(
        theta_over_pi, durations_ns, logI_T_theta,
        levels=[level], colors="white", linestyles="--", linewidths=1.8,
    )
    return cs


# =============================================================================
# Plot appearance
# =============================================================================

FIGSIZE = (14.5, 4.8)
DPI = 300

TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 12
CBAR_LABEL_FONTSIZE = 12
CBAR_TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11

GRID_ALPHA = 0.25


def plot_noisy_maps(
    *,
    thetas: np.ndarray,
    durations_ns: np.ndarray,
    base_noisy_infids: np.ndarray,
    ent_noisy_infids: np.ndarray,
    system_name: str,
) -> None:
    """Three-panel figure: baseline | TEM | improvement ratio."""
    theta_over_pi = thetas / np.pi

    base = np.maximum(base_noisy_infids, INFID_FLOOR)
    ent = np.maximum(ent_noisy_infids, INFID_FLOOR)

    log_base = np.log10(base).T
    log_ent = np.log10(ent).T
    log_ratio = np.log10(base / ent).T

    fig = plt.figure(figsize=FIGSIZE)

    # Same manual geometry as the working TEM plots.
    ax_positions = [
        [0.07, 0.15, 0.255, 0.72],   # panel (a)
        [0.350, 0.15, 0.255, 0.72],  # panel (b)
        [0.700, 0.15, 0.255, 0.72],  # panel (c)
    ]

    cbar_width = 0.012
    cbar_pad = 0.010

    axes = [fig.add_axes(pos) for pos in ax_positions]

    im0 = axes[0].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_base,
        shading="auto",
        cmap="inferno",
        vmin=-8,
        vmax=0,
    )

    im1 = axes[1].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_ent,
        shading="auto",
        cmap="inferno",
        vmin=-8,
        vmax=0,
    )

    im2 = axes[2].pcolormesh(
        theta_over_pi,
        durations_ns,
        log_ratio,
        shading="auto",
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
    )

    cs0 = add_met_contour(axes[0], theta_over_pi, durations_ns, log_base)
    cs1 = add_met_contour(axes[1], theta_over_pi, durations_ns, log_ent)

    axes[0].set_title("(a) Baseline", fontsize=TITLE_FONTSIZE)
    axes[1].set_title("(b) TEM", fontsize=TITLE_FONTSIZE)
    axes[2].set_title(
        r"(c) $\log_{10}(I_{\rm base}/I_{\rm TEM})$",
        fontsize=TITLE_FONTSIZE,
    )

    for ax in axes:
        ax.set_xlabel(
            r"Excitation strength $\theta/\pi$",
            fontsize=LABEL_FONTSIZE,
        )
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=GRID_ALPHA, linestyle="--")

    axes[0].set_ylabel(
        r"Pulse duration $T$ / ns",
        fontsize=LABEL_FONTSIZE,
    )

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # Colourbars exactly as tall as the plot panels.
    cbar_items = [
        (
            axes[1],
            im1,
            r"$\log_{10}(\mathrm{noisy\ final\ infidelity})$",
        ),
        (
            axes[2],
            im2,
            r"$\log_{10}(I_{\rm base}/I_{\rm TEM})$",
        ),
    ]

    for ax, im, label in cbar_items:
        pos = ax.get_position()
        cax = fig.add_axes([
            pos.x1 + cbar_pad,
            pos.y0,
            cbar_width,
            pos.height,
        ])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(label, fontsize=CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

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
            fontsize=LEGEND_FONTSIZE,
        )

    path = OUTDIR / f"{PLOT_PREFIX}_noisy_met.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
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
    n_J_channels = infer_n_J_channels(excitation, raw, pulse_width, max_n_time_steps)

    baseline_seed = select_baseline_local(raw)
    ent_seed, ent_selected_successful = select_ent_local(raw, target_kind, threshold)

    baseline_final_infids = get_array(raw, "baseline_final_infids")
    ent_final_infids = get_array(raw, "ent_final_infids")
    base_clean_infids = gather_selected_scalar_local(baseline_final_infids, baseline_seed)
    ent_clean_infids = gather_selected_scalar_local(ent_final_infids, ent_seed)

    n_theta = len(thetas)
    n_T = len(durations_ns)

    base_noisy_infids = np.full((n_theta, n_T), np.nan)
    ent_noisy_infids = np.full((n_theta, n_T), np.nan)
    failed_base = np.full((n_theta, n_T), False, dtype=bool)
    failed_ent = np.full((n_theta, n_T), False, dtype=bool)
    error_base = np.full((n_theta, n_T), "", dtype=object)
    error_ent = np.full((n_theta, n_T), "", dtype=object)

    # Build task list: one task per (theta, T) per pulse family.
    tasks = []
    for i, theta in enumerate(thetas):
        for j, T_ns in enumerate(durations_ns):
            n_steps = int(n_steps_used[i, j])
            common = dict(
                excitation=excitation, target_kind=target_kind,
                theta=float(theta), T_ns=float(T_ns),
                n_steps=n_steps, n_discretisation=N_DISCRETISATION,
                n_spins=n_spins, n_J_channels=n_J_channels,
                zeeman=zeeman, initial_state=initial_state,
                J_min=J_min, J_max=J_max, T1_us=T1_US, T2_us=T2_US,
            )

            b_seed = baseline_seed[i, j]
            if b_seed >= 0:
                b_pulse = get_pulse_local(baseline_x_final, i, j, b_seed,
                                          n_steps, n_J_channels)
                tasks.append(dict(i=i, j=j, label="base", pulse=b_pulse, **common))

            e_seed = ent_seed[i, j]
            if e_seed >= 0:
                e_pulse = get_pulse_local(ent_x_final, i, j, e_seed,
                                          n_steps, n_J_channels)
                tasks.append(dict(i=i, j=j, label="ent", pulse=e_pulse, **common))

    print("=" * 72)
    print("Noisy pulse-database MET plot")
    print(f"INPUT_FILE   = {INPUT_FILE}")
    print(f"OUTDIR       = {OUTDIR}")
    print(f"system_name  = {system_name}")
    print(f"excitation   = {excitation}")
    print(f"target_kind  = {target_kind}")
    print(f"n_spins      = {n_spins}")
    print(f"n_J_channels = {n_J_channels}")
    print(f"T1           = {T1_US:.6e} µs")
    print(f"T2           = {T2_US:.6e} µs")
    print("TEM selection = local lowest-entanglement successful run,")
    print("                otherwise local lowest clean-infidelity run")
    print(f"Total noisy evolutions = {len(tasks)}")
    print("=" * 72)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(noisy_infidelity_worker, task) for task in tasks]

        with tqdm(total=len(futures), desc="noisy evolutions", unit="pulse") as pbar:
            for n_done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                i, j, label = result["i"], result["j"], result["label"]

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
        thetas=thetas, durations_ns=durations_ns,
        base_noisy_infids=base_noisy_infids, ent_noisy_infids=ent_noisy_infids,
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
    print(f"Max noisy improvement ratio    = "
          f"{np.nanmax(base_noisy_infids / ent_noisy_infids):.6e}")
    print(f"Failed baseline evolutions     = {np.sum(failed_base)}")
    print(f"Failed TEM evolutions          = {np.sum(failed_ent)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
