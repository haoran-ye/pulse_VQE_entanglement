#!/usr/bin/env python3
"""
plot_noisy_cross_section.py

Read a raw SQE-state pulse database, evaluate selected baseline/TEM pulses under
T1/T2 Lindblad noise for one theta cross-section, and plot noisy final
infidelity versus pulse duration.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pulse_vqe_entanglement.dataclasses import NS_TO_US
from pulse_vqe_entanglement.sqe_state import single_qubit_excitation


# =============================================================================
# User settings
# =============================================================================

T_MIN_NS = 95.0
T_MAX_NS = 1005.0

PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
INPUT_FILE = PULSES_DATABASE / "sqe_state_initstate_01_positiveJ_pulses.npz"

OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

THETA_OVER_PI_TARGET = 0.01

N_WORKERS = 9
N_DISCRETISATION = 1

T1_US = 1.0e6
T2_US = 1.0e3

INFID_FLOOR = 1e-16
CLEAN_INF_THRESHOLD = 1e-5
ENTANGLEMENT_STAT = "avg"

SAVE_FIG = True
OUTFIG = OUTDIR / f"{INPUT_FILE.stem}_noisy_cross_section_theta_{THETA_OVER_PI_TARGET:.3f}.png"

COLOR_BASELINE = "#1f4e79"   # Cambridge-ish blue
COLOR_TEM = "#e67e22"   # strong orange


# =============================================================================
# Operators
# =============================================================================

I2 = np.eye(2, dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)


def op_on_qubit(op: np.ndarray, q: int, n: int) -> np.ndarray:
    out = np.array([[1.0]], dtype=np.complex128)
    for i in range(n):
        out = np.kron(out, op if i == q else I2)
    return out


def z_ops(n: int) -> list[np.ndarray]:
    return [op_on_qubit(Z, q, n) for q in range(n)]


# =============================================================================
# Database helpers
# =============================================================================

def get_array(data, *names: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of these arrays found: {names}")


def scalar_str(x) -> str:
    arr = np.asarray(x)
    return str(arr.item()) if arr.shape == () else str(x)


def infer_system_name(data, input_file: Path) -> str:
    if "system_name" in data.files:
        return scalar_str(data["system_name"])

    name = input_file.stem.lower()
    if "sqe" in name and "state" in name:
        return "sqe_state"

    raise ValueError("This script currently expects an SQE-state pulse database.")


def get_entanglement_array(data, label: str) -> np.ndarray:
    if label == "ent":
        return (
            get_array(data, "ent_s_avg", "ent_s_max")
            if ENTANGLEMENT_STAT == "avg"
            else get_array(data, "ent_s_max", "ent_s_avg")
        )
    raise ValueError(label)


def select_baseline_local(data) -> np.ndarray:
    infids = get_array(data, "baseline_final_infids")
    n_theta, n_T, _ = infids.shape
    seed_idx = np.full((n_theta, n_T), -1, dtype=int)

    for i in range(n_theta):
        for j in range(n_T):
            finite = np.isfinite(infids[i, j])
            if np.any(finite):
                idx = np.where(finite)[0]
                seed_idx[i, j] = idx[np.nanargmin(infids[i, j, finite])]

    return seed_idx


def select_ent_local(data, threshold: float):
    infids = get_array(data, "ent_final_infids")
    ent_values = get_entanglement_array(data, "ent")

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
                if np.any(finite_I):
                    idx = np.where(finite_I)[0]
                    seed_idx[i, j] = idx[np.nanargmin(infids[i, j, finite_I])]

    return seed_idx, selected_successful


def get_pulse_local(
    pulse_array: np.ndarray,
    i: int,
    j: int,
    seed: int,
    n_steps: int,
    n_J_channels: int,
) -> np.ndarray | None:
    if seed < 0:
        return None
    width = int(n_steps) * int(n_J_channels)
    return np.asarray(pulse_array[i, j, seed, :width], dtype=float)


# =============================================================================
# Physics
# =============================================================================

def free_rotating_unitary(zeeman: np.ndarray, T_ns: float, n_spins: int) -> np.ndarray:
    deltaB = zeeman - np.mean(zeeman)
    H_rot = sum(deltaB[q] * z_ops(n_spins)[q] for q in range(n_spins))
    return expm(-1j * np.pi * H_rot * T_ns * NS_TO_US)


def target_rho(theta: float, T_ns: float, zeeman: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    U_exc = single_qubit_excitation(theta)
    U = free_rotating_unitary(zeeman, T_ns, n_spins=2) @ U_exc
    psi = U @ initial_state
    return np.outer(psi, psi.conj())


def lindblad_ops_T1_T2(n_spins: int, T1_us: float, T2_us: float) -> np.ndarray:
    gamma_1 = 1.0 / T1_us
    gamma_phi = max(0.0, 1.0 / T2_us - 0.5 / T1_us)

    ops = []
    for q in range(n_spins):
        ops.append(np.sqrt(gamma_1 / 4.0) * op_on_qubit(SIGMA_MINUS, q, n_spins))

    if gamma_phi > 0.0:
        for q in range(n_spins):
            ops.append(np.sqrt(gamma_phi / 4.0) * op_on_qubit(Z, q, n_spins))

    return np.asarray(ops, dtype=np.complex128)


def make_noisy_driven_system(
    *,
    T_ns: float,
    n_steps: int,
    n_discretisation: int,
    zeeman: np.ndarray,
    initial_state: np.ndarray,
    J_min: float,
    J_max: float,
    L0: np.ndarray,
):
    n_spins = 2
    n_J_channels = 1

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
    drive_freqs = np.asarray(zeeman, dtype=np.float64)
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


def state_infidelity(rho: np.ndarray, rho_target: np.ndarray) -> float:
    return max(1.0 - float(np.real(np.trace(rho @ rho_target))), INFID_FLOOR)


# =============================================================================
# Worker
# =============================================================================

def noisy_worker(task: dict) -> dict:
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

    try:
        L0 = lindblad_ops_T1_T2(2, task["T1_us"], task["T2_us"])

        driven = make_noisy_driven_system(
            T_ns=task["T_ns"],
            n_steps=task["n_steps"],
            n_discretisation=task["n_discretisation"],
            zeeman=task["zeeman"],
            initial_state=task["initial_state"],
            J_min=task["J_min"],
            J_max=task["J_max"],
            L0=L0,
        )

        rho = driven.propagate(task["pulse"])
        rho_t = target_rho(
            task["theta"],
            task["T_ns"],
            task["zeeman"],
            task["initial_state"],
        )

        noisy_I = state_infidelity(rho, rho_t)

        return dict(
            j=task["j"],
            label=task["label"],
            failed=False,
            noisy_infid=noisy_I,
            error="",
        )

    except Exception as exc:
        return dict(
            j=task["j"],
            label=task["label"],
            failed=True,
            noisy_infid=np.nan,
            error=repr(exc),
        )


# =============================================================================
# Plot
# =============================================================================

def plot_cross_section(
    *,
    theta_selected: float,
    durations_ns: np.ndarray,
    base_noisy: np.ndarray,
    ent_noisy: np.ndarray,
) -> None:
    base = np.maximum(base_noisy, INFID_FLOOR)
    ent = np.maximum(ent_noisy, INFID_FLOOR)
    ratio = base / ent

    finite = (
        np.isfinite(base)
        & np.isfinite(ent)
        & (durations_ns >= T_MIN_NS)
        & (durations_ns <= T_MAX_NS)
    )
    T = durations_ns[finite]
    base = base[finite]
    ent = ent[finite]
    ratio = ratio[finite]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)

    ax.plot(
        T,
        base,
        color=COLOR_BASELINE,
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        label="Baseline",
    )

    ax.plot(
        T,
        ent,
        color=COLOR_TEM,
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        label="TEM",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"Pulse duration $T$ / ns", fontsize=12)
    ax.set_ylabel("Noisy final infidelity", fontsize=12)

    ax.set_title(
        rf"Final infidelity with noise at $\theta = {theta_selected / np.pi:.3f} \pi$",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.28)
    ax.tick_params(axis="both", labelsize=11)

    lines1, labels1 = ax.get_legend_handles_labels()

    #ax.legend(
    #    lines1,
    #    labels1,
    #    frameon=True,
    #    fontsize=10,
    #    loc="best",
    #)

    fig.tight_layout()

    if SAVE_FIG:
        fig.savefig(OUTFIG, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {OUTFIG}")

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    raw = np.load(INPUT_FILE, allow_pickle=True)

    system_name = infer_system_name(raw, INPUT_FILE)
    if system_name != "sqe_state":
        raise ValueError(f"Expected sqe_state database, got {system_name}")

    thetas = get_array(raw, "thetas", "excitation_strengths")
    theta_over_pi = thetas / np.pi
    durations_ns = get_array(raw, "durations_ns", "pulse_durations")

    i_theta = int(np.argmin(np.abs(theta_over_pi - THETA_OVER_PI_TARGET)))
    theta_selected = float(thetas[i_theta])

    zeeman = np.asarray(get_array(raw, "zeeman"), dtype=np.float64)
    initial_state = np.asarray(get_array(raw, "initial_state"), dtype=np.complex128)

    J_min = float(get_array(raw, "J_min"))
    J_max = float(get_array(raw, "J_max"))

    n_steps_used = get_array(raw, "n_time_steps_used", "n_steps_used").astype(int)

    threshold = (
        float(raw["infidelity_threshold"])
        if "infidelity_threshold" in raw.files
        else CLEAN_INF_THRESHOLD
    )

    baseline_x_final = get_array(raw, "baseline_x_final")
    ent_x_final = get_array(raw, "ent_x_final")

    baseline_seed = select_baseline_local(raw)
    ent_seed, _ = select_ent_local(raw, threshold)

    n_T = len(durations_ns)

    base_noisy = np.full(n_T, np.nan)
    ent_noisy = np.full(n_T, np.nan)

    tasks = []

    for j, T_ns in enumerate(durations_ns):
        n_steps = int(n_steps_used[i_theta, j])

        common = dict(
            j=j,
            theta=theta_selected,
            T_ns=float(T_ns),
            n_steps=n_steps,
            n_discretisation=N_DISCRETISATION,
            zeeman=zeeman,
            initial_state=initial_state,
            J_min=J_min,
            J_max=J_max,
            T1_us=T1_US,
            T2_us=T2_US,
        )

        b_seed = int(baseline_seed[i_theta, j])
        if b_seed >= 0:
            b_pulse = get_pulse_local(
                baseline_x_final,
                i_theta,
                j,
                b_seed,
                n_steps,
                1,
            )
            tasks.append(dict(label="base", pulse=b_pulse, **common))

        e_seed = int(ent_seed[i_theta, j])
        if e_seed >= 0:
            e_pulse = get_pulse_local(
                ent_x_final,
                i_theta,
                j,
                e_seed,
                n_steps,
                1,
            )
            tasks.append(dict(label="ent", pulse=e_pulse, **common))

    print("=" * 72)
    print("Noisy infidelity cross-section")
    print(f"INPUT_FILE     = {INPUT_FILE}")
    print(f"theta target   = {THETA_OVER_PI_TARGET:.6f} pi")
    print(f"theta selected = {theta_selected / np.pi:.6f} pi")
    print(f"N durations    = {n_T}")
    print(f"N noisy tasks  = {len(tasks)}")
    print(f"T1             = {T1_US:.3e} us")
    print(f"T2             = {T2_US:.3e} us")
    print("=" * 72)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(noisy_worker, task) for task in tasks]

        with tqdm(total=len(futures), desc="noisy cross-section", unit="pulse") as pbar:
            for future in as_completed(futures):
                result = future.result()
                j = result["j"]

                if result["label"] == "base":
                    base_noisy[j] = result["noisy_infid"]
                elif result["label"] == "ent":
                    ent_noisy[j] = result["noisy_infid"]

                if result["failed"]:
                    pbar.write(
                        f"FAILED {result['label']} T_idx={j}: {result['error']}"
                    )

                pbar.update(1)

    plot_cross_section(
        theta_selected=theta_selected,
        durations_ns=durations_ns,
        base_noisy=base_noisy,
        ent_noisy=ent_noisy,
    )

    finite = np.isfinite(base_noisy) & np.isfinite(ent_noisy)

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Best baseline noisy infidelity = {np.nanmin(base_noisy):.6e}")
    print(f"Best TEM noisy infidelity      = {np.nanmin(ent_noisy):.6e}")
    print(f"Max improvement ratio          = {np.nanmax(base_noisy[finite] / ent_noisy[finite]):.6e}")
    print("=" * 72)


if __name__ == "__main__":
    main()