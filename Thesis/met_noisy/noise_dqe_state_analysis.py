#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import expm

from qugradlab.systems.semiconducting.esr import SpinChain

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import NS_TO_US
from pulse_vqe_entanglement.dqe_state import dqe_unitary, basis_state


# ============================================================
# Settings
# ============================================================

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_noisy/dqe_state_noise_characterisation/"
)

INPUT_DATA = OUTDIR / "dqe_state_noise_characterisation_results.npz"
OUTPUT_DATA = OUTDIR / "dqe_state_noise_characterisation_lindblad_analysis.npz"

THETA = 0.01 * np.pi
T_NS = 500.0

N_TIME_STEPS = 50
N_DISCRETISATION = 1

J_MIN = 0.0
J_MAX = 10.0

ZEEMAN = np.array([1000.0, 1008.0, 1016.0, 1024.0])
INITIAL_STATE = basis_state(4, "0011")

NOISE_RATES = np.logspace(-15, 5, 30)
T1_VALUES_US = np.logspace(0, 6, 31)
T2_VALUES_US = np.logspace(0, 6, 31)

INFID_FLOOR = 1e-18


# ============================================================
# Operators
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)


def op_on_qubit(op, q, n=4):
    out = np.array([[1.0]], dtype=np.complex128)
    for i in range(n):
        out = np.kron(out, op if i == q else I2)
    return out


Z_OPS = [op_on_qubit(Z, q) for q in range(4)]


# ============================================================
# Noise models
# ============================================================

def lindblad_ops(kind, rate):
    ops = []

    for q in range(4):
        if kind == "depolarising":
            ops += [op_on_qubit(X, q), op_on_qubit(Y, q), op_on_qubit(Z, q)]
        elif kind == "dephasing":
            ops += [op_on_qubit(Z, q)]
        elif kind == "relaxation":
            ops += [op_on_qubit(SIGMA_MINUS, q)]
        else:
            raise ValueError(kind)

    return np.sqrt(rate / 4.0) * np.asarray(ops, dtype=np.complex128)


def T1_T2_lindblad_ops(T1_us, T2_us):
    gamma_1 = 1.0 / T1_us
    gamma_phi = max(0.0, 1.0 / T2_us - 0.5 / T1_us)

    ops = list(lindblad_ops("relaxation", gamma_1))

    if gamma_phi > 0.0:
        ops += list(lindblad_ops("dephasing", gamma_phi))

    return np.array(ops, dtype=np.complex128)


# ============================================================
# Target and noisy evolution
# ============================================================

def make_target_rho():
    deltaB = ZEEMAN - np.mean(ZEEMAN)
    H_rot = sum(deltaB[q] * Z_OPS[q] for q in range(4))

    U_rot = expm(-1j * np.pi * H_rot * T_NS * NS_TO_US)
    psi_target = U_rot @ dqe_unitary(THETA) @ INITIAL_STATE

    return np.outer(psi_target, psi_target.conj())


def make_noisy_system(L0):
    deltaB = ZEEMAN - np.mean(ZEEMAN)

    sys = SpinChain(
        spins=4,
        zeeman_splittings=deltaB,
        max_drive_strength=0.0,
        J_max=J_MAX,
        J_min=J_MIN,
        feromagnetic=False,
        use_graph=True,
    ).add_lindblad_terms(L0=L0)

    n_fine = N_TIME_STEPS * N_DISCRETISATION
    dt_fine_us = T_NS / n_fine * NS_TO_US
    rho0 = np.outer(INITIAL_STATE, INITIAL_STATE.conj())

    def pulse_function(x):
        x = tf.cast(tf.convert_to_tensor(x), tf.complex128)
        x = tf.reshape(x, (N_TIME_STEPS, 3))
        x = tf.repeat(x, repeats=N_DISCRETISATION, axis=0)

        return (
            tf.zeros((n_fine, 4), dtype=tf.complex128),
            tf.zeros(4, dtype=tf.complex128),
            x,
            tf.cast(tf.convert_to_tensor(rho0), tf.complex128),
            tf.cast(dt_fine_us * 2 * np.pi, tf.complex128),
        )

    return sys.pulse_form(pulse_function)


def infidelity_from_rho(rho, target_rho):
    return max(1.0 - float(np.real(np.trace(rho @ target_rho))), INFID_FLOOR)


# ============================================================
# Plotting
# ============================================================

def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def plot_rate_scan(kind, rates, I_base, I_ent):
    I_base = np.asarray(I_base)
    I_ent = np.asarray(I_ent)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(rates, I_base, "o-", label="baseline")
    plt.plot(rates, I_ent, "s-", label="entanglement-suppressed")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Noise rate $\gamma$ / $\mu s^{-1}$")
    plt.ylabel("Final infidelity")
    plt.title(f"DQE state: {kind} noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    savefig(f"dqe_state_{kind}_infidelity_vs_rate_analysis.png")

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(rates, I_base / I_ent, "o-")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Noise rate $\gamma$ / $\mu s^{-1}$")
    plt.ylabel(r"$I_{\mathrm{base}}/I_{\mathrm{ent}}$")
    plt.title(f"DQE state: {kind} improvement ratio")
    plt.grid(True, which="both", alpha=0.35)
    savefig(f"dqe_state_{kind}_ratio_vs_rate_analysis.png")


def plot_T1_T2_heatmap(ratio):
    T1, T2 = np.meshgrid(T1_VALUES_US, T2_VALUES_US, indexing="ij")
    log_ratio = np.ma.masked_invalid(np.log10(ratio))

    plt.figure(figsize=(6.8, 5.4))
    im = plt.pcolormesh(
        T1,
        T2,
        log_ratio,
        shading="auto",
        cmap="inferno",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$T_1$ / $\mu s$")
    plt.ylabel(r"$T_2$ / $\mu s$")
    plt.title(r"DQE state: $\log_{10}(I_{\rm base}/I_{\rm ent})$")
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\log_{10}(I_{\rm base}/I_{\rm ent})$")
    savefig("dqe_state_T1_T2_improvement_heatmap_analysis.png")


def plot_T1_T2_infidelity_maps(I_base_map, I_ent_map):
    T1, T2 = np.meshgrid(T1_VALUES_US, T2_VALUES_US, indexing="ij")

    for label, arr in [
        ("baseline", I_base_map),
        ("entanglement_suppressed", I_ent_map),
    ]:
        plt.figure(figsize=(6.8, 5.4))
        im = plt.pcolormesh(
            T1,
            T2,
            np.ma.masked_invalid(np.log10(arr)),
            shading="auto",
            cmap="inferno",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$T_1$ / $\mu s$")
        plt.ylabel(r"$T_2$ / $\mu s$")
        plt.title(f"DQE state: {label} " + r"$\log_{10}(I)$")
        cbar = plt.colorbar(im)
        cbar.set_label(r"$\log_{10}(I)$")
        savefig(f"dqe_state_T1_T2_{label}_infidelity_analysis.png")


# ============================================================
# Main
# ============================================================

def main():
    data = np.load(INPUT_DATA)

    x_base = np.asarray(data["x_base"], dtype=float)
    x_ent = np.asarray(data["x_ent"], dtype=float)

    print("\nLoaded pulses")
    print("=" * 72)
    print(f"INPUT_DATA = {INPUT_DATA}")
    print(f"x_base shape = {x_base.shape}")
    print(f"x_ent  shape = {x_ent.shape}")
    print(f"base noiseless I = {float(data['base_noiseless_infid']):.12e}")
    print(f"ent  noiseless I = {float(data['ent_noiseless_infid']):.12e}")
    print(f"base S_sum_max   = {float(data['base_s_sum_max']):.12e}")
    print(f"ent  S_sum_max   = {float(data['ent_s_sum_max']):.12e}")
    print("=" * 72 + "\n")

    target_rho = make_target_rho()

    results = {
        "theta": THETA,
        "T_ns": T_NS,
        "noise_rates": NOISE_RATES,
        "T1_values_us": T1_VALUES_US,
        "T2_values_us": T2_VALUES_US,
        "x_base": x_base,
        "x_ent": x_ent,
        "base_noiseless_infid": float(data["base_noiseless_infid"]),
        "ent_noiseless_infid": float(data["ent_noiseless_infid"]),
        "base_s_sum_max": float(data["base_s_sum_max"]),
        "ent_s_sum_max": float(data["ent_s_sum_max"]),
    }

    for kind in ["depolarising", "dephasing", "relaxation"]:
        print(f"Running {kind} noise scan...")

        I_base = []
        I_ent = []

        for gamma in NOISE_RATES:
            driven = make_noisy_system(lindblad_ops(kind, gamma))

            rho_base = driven.propagate(x_base)
            rho_ent = driven.propagate(x_ent)

            ib = infidelity_from_rho(rho_base, target_rho)
            ie = infidelity_from_rho(rho_ent, target_rho)

            I_base.append(ib)
            I_ent.append(ie)

            print(
                f"{kind:13s} gamma={gamma:.3e}, "
                f"I_base={ib:.3e}, I_ent={ie:.3e}, ratio={ib / ie:.3e}"
            )

        I_base = np.asarray(I_base)
        I_ent = np.asarray(I_ent)

        results[f"{kind}_base_infids"] = I_base
        results[f"{kind}_ent_infids"] = I_ent
        results[f"{kind}_ratio"] = I_base / I_ent

        plot_rate_scan(kind, NOISE_RATES, I_base, I_ent)

    print("Running T1/T2 heatmap...")

    I_base_map = np.full((len(T1_VALUES_US), len(T2_VALUES_US)), np.nan)
    I_ent_map = np.full_like(I_base_map, np.nan)

    for i, T1 in enumerate(T1_VALUES_US):
        for j, T2 in enumerate(T2_VALUES_US):
            L0 = T1_T2_lindblad_ops(T1, T2)

            if L0 is None:
                continue

            driven = make_noisy_system(L0)

            rho_base = driven.propagate(x_base)
            rho_ent = driven.propagate(x_ent)

            I_base_map[i, j] = infidelity_from_rho(rho_base, target_rho)
            I_ent_map[i, j] = infidelity_from_rho(rho_ent, target_rho)

        print(f"T1 row {i + 1}/{len(T1_VALUES_US)} done")

    ratio = I_base_map / I_ent_map

    results["T1_T2_base_infids"] = I_base_map
    results["T1_T2_ent_infids"] = I_ent_map
    results["T1_T2_ratio"] = ratio

    plot_T1_T2_heatmap(ratio)
    plot_T1_T2_infidelity_maps(I_base_map, I_ent_map)

    np.savez_compressed(OUTPUT_DATA, **results)

    print(f"\nSaved analysis data to:\n{OUTPUT_DATA}")
    print("Done.")


if __name__ == "__main__":
    main()