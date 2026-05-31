#!/usr/bin/env python3
"""
4A-noisy_case_study.py

Single-point noise case study for one (theta, T) grid point.

Loads one baseline and one TEM pulse from a database, then sweeps noise
rates across three noise models (dephasing, depolarising, relaxation) and
plots the noise-induced infidelity and baseline/TEM improvement ratio.

Usage
-----
    Set TARGET_THETA, TARGET_T_NS, and DATABASE below, then run as a script.

Output
------
    DATABASE.parent/noise_tests_selected_sqe_state/<stem>_noise_test.png
"""

from pathlib import Path
import os

# Force single-threaded execution — this script is not parallelised, and
# single-threaded TF is significantly faster for small state-vector simulations.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path as _Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qugradlab.systems.semiconducting.esr import SpinChain

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# pulse_vqe_entanglement is not pip-installed; add the repo root to sys.path.
_ROOT = _Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig


# =========================
# User settings
# =========================

# B1 fix: DATABASE resolved relative to this script.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
DATABASE = PULSES_DATABASE / "sqe_state_initstate_01_positiveJ_pulses.npz"

OUTDIR = DATABASE.parent / "noise_tests_selected_sqe_state"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_THETA = 0.01 * np.pi  # gate angle to select from database
TARGET_T_NS = 500.0           # pulse duration to select from database

NOISE_RATES = np.logspace(-18, 5, 21)  # noise rate γ scan (µs⁻¹)

SAVE_FIG = OUTDIR / (
    f"sqe_state_theta_{TARGET_THETA / np.pi:.4f}pi_T_{TARGET_T_NS:.0f}ns_noise_test.png"
)

INFIDELITY_FLOOR = 1e-14

NS_TO_US = 1e-3  # nanoseconds → microseconds conversion factor


# =========================
# Operators
# =========================

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# B7 fix: renamed from SPLUS to SIGMA_MINUS for consistency with the rest of the
# noise scripts.  Matrix [[0,1],[0,0]] is the lowering (de-excitation) jump operator
# in this spin convention.
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)

Z1 = np.kron(Z, I2)
Z2 = np.kron(I2, Z)


def convert_dtype(x):
    return tf.cast(tf.convert_to_tensor(x), dtype=tf.complex128)


def single_qubit_excitation(theta: float) -> np.ndarray:
    """SQE gate U(θ): rotation in the {|01⟩, |10⟩} subspace by angle θ."""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )


def dm_infidelity(rho: np.ndarray, target_psi: np.ndarray) -> float:
    """State infidelity: 1 - ⟨ψ|ρ|ψ⟩ for pure target state |ψ⟩."""
    target_rho = np.outer(target_psi, target_psi.conj())
    return float(np.real(1.0 - np.trace(rho @ target_rho)))


# =========================
# Lindblad operators
# =========================

def local_dephasing_ops_2q(gamma: float) -> np.ndarray:
    """Z dephasing operators on each qubit, prefactor √(γ/4)."""
    pref = np.sqrt(gamma / 4.0)
    return np.array(
        [pref * np.kron(Z, I2), pref * np.kron(I2, Z)],
        dtype=np.complex128,
    )


def local_depolarising_ops_2q(gamma: float) -> np.ndarray:
    """Full X, Y, Z depolarising operators on each qubit."""
    pref = np.sqrt(gamma / 4.0)
    return np.array(
        [
            pref * np.kron(X, I2), pref * np.kron(Y, I2), pref * np.kron(Z, I2),
            pref * np.kron(I2, X), pref * np.kron(I2, Y), pref * np.kron(I2, Z),
        ],
        dtype=np.complex128,
    )


def local_relaxation_ops_2q(gamma: float) -> np.ndarray:
    """σ- relaxation (T1 decay) operators on each qubit."""
    pref = np.sqrt(gamma / 4.0)
    return np.array(
        [pref * np.kron(SIGMA_MINUS, I2), pref * np.kron(I2, SIGMA_MINUS)],
        dtype=np.complex128,
    )


# =========================
# Database loading
# =========================

def get_first_existing(data, names: list[str]) -> tuple[str, np.ndarray]:
    """Return (key, value) for the first key in *names* found in *data.files*."""
    for name in names:
        if name in data.files:
            return name, data[name]
    raise KeyError(
        f"None of these keys found: {names}\nAvailable keys:\n{data.files}"
    )


def pick_index(values: np.ndarray, target: float) -> int:
    """Return the index of the closest value to *target* in *values*."""
    return int(np.nanargmin(np.abs(np.asarray(values, dtype=float) - target)))


def squeeze_pulse(pulse: np.ndarray) -> np.ndarray:
    """Ensure *pulse* is a 1D float array; raise if shape is unexpected."""
    pulse = np.asarray(pulse, dtype=float)
    pulse = np.squeeze(pulse)
    if pulse.ndim == 2 and pulse.shape[-1] == 1:
        pulse = pulse[:, 0]
    if pulse.ndim != 1:
        raise ValueError(f"Expected 1D SQE pulse after squeezing, got shape {pulse.shape}")
    return pulse


def extract_pulse_array(
    data, candidates: list[str], i_theta: int, i_T: int,
    infid_candidates: list[str] | None = None
) -> tuple[np.ndarray, str]:
    """
    Extract the pulse at (i_theta, i_T) from the first available candidate key.

    For 4D arrays (theta, T, init, pulse_len), selects the init with the lowest
    finite infidelity (from *infid_candidates*) or defaults to init 0.
    """
    key, arr = get_first_existing(data, candidates)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        pulse = arr[i_theta, i_T]

    elif arr.ndim == 4:
        block = arr[i_theta, i_T]
        if infid_candidates is not None:
            try:
                _, infids = get_first_existing(data, infid_candidates)
                run_infids = np.asarray(infids)[i_theta, i_T]
                i_run = int(np.nanargmin(run_infids))
                pulse = block[i_run]
            except Exception:
                pulse = block[0]
        else:
            pulse = block[0]

    elif arr.ndim == 5:
        block = arr[i_theta, i_T]
        pulse = block[0]

    else:
        raise ValueError(f"Unsupported pulse array shape for key {key}: {arr.shape}")

    return squeeze_pulse(pulse), key


def load_selected_pulses(database: Path):
    """
    Load baseline and TEM pulses closest to (TARGET_THETA, TARGET_T_NS).

    Returns
    -------
    theta : float    — actual selected gate angle (rad)
    T_ns  : float    — actual selected duration (ns)
    baseline_pulse   : 1D float array
    tem_pulse        : 1D float array
    """
    data = np.load(database, allow_pickle=True)

    _, thetas = get_first_existing(
        data, ["thetas", "theta_targets", "theta_values", "theta_grid"],
    )
    _, durations = get_first_existing(
        data, ["durations_ns", "Ts", "T_values", "duration_values", "T_grid"],
    )

    i_theta = pick_index(thetas, TARGET_THETA)
    i_T = pick_index(durations, TARGET_T_NS)

    theta = float(np.asarray(thetas)[i_theta])
    T_ns = float(np.asarray(durations)[i_T])

    baseline_pulse, baseline_key = extract_pulse_array(
        data,
        candidates=[
            "baseline_x_final", "baseline_xf", "best_baseline_x_final",
            "best_baseline_xf", "baseline_pulses", "x_baseline", "xf_baseline",
        ],
        i_theta=i_theta, i_T=i_T,
        infid_candidates=[
            "baseline_final_infids", "baseline_infidelities", "best_baseline_infid",
        ],
    )

    tem_pulse, tem_key = extract_pulse_array(
        data,
        candidates=[
            "ent_x_final", "ent_xf", "best_ent_x_final", "best_ent_xf",
            "tem_x_final", "tem_xf", "ent_pulses", "tem_pulses",
            "x_ent", "xf_ent",
        ],
        i_theta=i_theta, i_T=i_T,
        infid_candidates=[
            "ent_final_infids", "tem_final_infids", "ent_infidelities", "best_ent_infid",
        ],
    )

    print(f"Loaded database: {database}")
    print(f"Selected theta  = {theta / np.pi:.6f} π")
    print(f"Selected T      = {T_ns:.3f} ns")
    print(f"Baseline pulse key: {baseline_key}, shape = {baseline_pulse.shape}")
    print(f"TEM pulse key:      {tem_key}, shape = {tem_pulse.shape}")

    return theta, T_ns, baseline_pulse, tem_pulse


# =========================
# System setup
# =========================

def build_target_state(config: SimulationConfig) -> np.ndarray:
    """Target state = U_free(T) · U_SQE(θ) · ψ₀ in the rotating frame."""
    B = np.asarray(config.zeeman_splittings, dtype=float)
    deltaB = B - np.mean(B)
    U_rot = expm(
        -1j * np.pi * (deltaB[0] * Z1 + deltaB[1] * Z2) * config.T * NS_TO_US
    )
    return U_rot @ single_qubit_excitation(config.theta_target) @ config.initial_state


def build_clean_system(config: SimulationConfig):
    """Build a noise-free SpinChain system from the given config."""
    B = np.asarray(config.zeeman_splittings, dtype=float)
    deltaB = B - np.mean(B)
    return SpinChain(
        spins=2,
        zeeman_splittings=deltaB,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )


def make_lindblad_driven_system(sys, config: SimulationConfig, n_steps: int):
    """Attach a pulse function to a Lindblad system for propagation."""
    n_disc = int(config.n_discretisation)
    n_fine = n_steps * n_disc
    dt_fine_us = (config.T / n_fine) * NS_TO_US

    def lindblad_pulse_function(x):
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        psi0 = np.asarray(config.initial_state, dtype=np.complex128)
        rho0 = np.outer(psi0, psi0.conj())
        return (
            convert_dtype(np.zeros((n_fine, 2), dtype=np.complex128)),
            convert_dtype(config.zeeman_splittings),
            convert_dtype(x_fine),
            convert_dtype(rho0),
            convert_dtype(dt_fine_us * 2 * np.pi),
        )

    return sys.pulse_form(lindblad_pulse_function)


def run_noise_scan(base_sys, config: SimulationConfig, target_state: np.ndarray,
                   pulse: np.ndarray, noise_op_fn) -> tuple[np.ndarray, float]:
    """
    Scan NOISE_RATES for a single pulse and noise model.

    First evaluates at zero noise to get the noiseless floor, then subtracts
    it from each noisy evaluation to isolate the noise-induced infidelity.

    Returns
    -------
    infids : (len(NOISE_RATES),) noise-induced infidelity at each rate.
    clean_floor : float, noiseless infidelity (optimisation residual).
    """
    # Noiseless floor: optimiser leaves a small residual infidelity.
    clean_sys = base_sys.add_lindblad_terms(L0=noise_op_fn(0.0))
    clean_driven = make_lindblad_driven_system(clean_sys, config, len(pulse))
    rho_clean = clean_driven.propagate(pulse)
    clean_floor = dm_infidelity(rho_clean, target_state)

    infids = []
    for gamma in NOISE_RATES:
        noisy_sys = base_sys.add_lindblad_terms(L0=noise_op_fn(float(gamma)))
        noisy_driven = make_lindblad_driven_system(noisy_sys, config, len(pulse))
        rho = noisy_driven.propagate(pulse)
        I_total = dm_infidelity(rho, target_state)
        # Subtract noiseless floor to isolate noise-induced component.
        infids.append(max(I_total - clean_floor, INFIDELITY_FLOOR))

    return np.asarray(infids), clean_floor


# =========================
# Plotting
# =========================

def plot_results(results: dict) -> None:
    """3×2 grid: infidelity and improvement ratio for each noise model."""
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True, constrained_layout=True)

    noise_order = [
        ("dephasing", "dephasing"),
        ("depolarising", "depolarising"),
        ("relaxation", "relaxation"),
    ]

    baseline_colour = "tab:green"
    tem_colour = "tab:red"
    ratio_colour = "tab:blue"

    for row, (key, title) in enumerate(noise_order):
        baseline = results[key]["baseline"]
        tem = results[key]["tem"]
        ratio = baseline / tem

        ax_inf = axes[row, 0]
        ax_ratio = axes[row, 1]

        # --- Infidelity ---
        ax_inf.plot(NOISE_RATES, baseline, marker="o", lw=2, ms=4,
                    color=baseline_colour, label="baseline")
        ax_inf.plot(NOISE_RATES, tem, marker="s", lw=2, ms=4,
                    color=tem_colour, label="TEM pulse")
        ax_inf.set_xscale("log")
        ax_inf.set_yscale("log")
        ax_inf.set_ylabel("Final infidelity", fontsize=10)
        ax_inf.set_title(f"{title} noise", fontsize=11)
        ax_inf.grid(True, which="both", alpha=0.25)
        ax_inf.legend(frameon=True, fontsize=8)
        ax_inf.tick_params(axis="both", labelsize=8)

        # --- Improvement ratio ---
        ax_ratio.plot(NOISE_RATES, ratio, marker="o", lw=2, ms=4, color=ratio_colour)
        ax_ratio.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5)
        ax_ratio.set_xscale("log")
        ax_ratio.set_ylabel(r"$I_{\rm base}/I_{\rm TEM}$", fontsize=10)
        ax_ratio.set_title(f"{title} improvement ratio", fontsize=11)
        ax_ratio.grid(True, which="both", alpha=0.25)
        ax_ratio.tick_params(axis="both", labelsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Noise rate $\gamma\ /\ \mu s^{-1}$", fontsize=10)

    fig.savefig(SAVE_FIG, dpi=300)
    print(f"Saved plot to:\n{SAVE_FIG}")
    plt.show()


# =========================
# Main
# =========================

def main() -> None:
    theta, T_ns, baseline_pulse, tem_pulse = load_selected_pulses(DATABASE)

    if len(baseline_pulse) != len(tem_pulse):
        raise ValueError(
            f"baseline and TEM pulses have different lengths: "
            f"{len(baseline_pulse)} vs {len(tem_pulse)}"
        )

    config = SimulationConfig(
        zeeman_splittings=np.array([1000.0, 1008.0]),
        initial_state=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),  # |01⟩
        theta_target=theta,
        n_time_steps=len(baseline_pulse),
        n_discretisation=1,
        T=T_ns,
        max_drive_strength=0.0,
        J_max=10.0,
        J_min=0.0,
        J_init_scale=1.0,
        J_init_mean=0.0,
        feromagnetic=False,
        gtol=1e-12,
        maxiter=500,
        entanglement_optimisation=False,
        state_ent_measure="von_neumann",
        noise_type="none",
        noise_rate=0.0,
    )

    target_state = build_target_state(config)
    base_sys = build_clean_system(config)

    noise_models = {
        "dephasing": local_dephasing_ops_2q,
        "depolarising": local_depolarising_ops_2q,
        "relaxation": local_relaxation_ops_2q,
    }

    results = {}
    for name, op_fn in noise_models.items():
        print(f"\nRunning {name} noise scan...")

        baseline_infids, baseline_floor = run_noise_scan(
            base_sys, config, target_state, baseline_pulse, op_fn,
        )
        tem_infids, tem_floor = run_noise_scan(
            base_sys, config, target_state, tem_pulse, op_fn,
        )

        print(f"  baseline noiseless floor = {baseline_floor:.3e}")
        print(f"  TEM noiseless floor      = {tem_floor:.3e}")

        results[name] = {"baseline": baseline_infids, "tem": tem_infids}

    plot_results(results)


if __name__ == "__main__":
    main()
