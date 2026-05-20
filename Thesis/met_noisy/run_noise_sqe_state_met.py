#!/usr/bin/env python3

from pathlib import Path
import sys
import os
import numpy as np
import tensorflow as tf
from scipy.linalg import expm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from qugradlab.systems.semiconducting.esr import SpinChain

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig, NS_TO_US
from pulse_vqe_entanglement.sqe_state import sqe_optimisation, single_qubit_excitation
from pulse_vqe_entanglement.entanglement_measures import STATE_MEASURES


# ============================================================
# Settings
# ============================================================

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_noisy/sqe_state_noisy_met/"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_OUT = OUTDIR / "sqe_state_noisy_met_scan.npz"

N_WORKERS = 9

ZEEMAN = np.array([1000.0, 1008.0])
INITIAL_STATE = np.array([0.0, 1.0, 1.0, 0.0]/np.sqrt(2), dtype=np.complex128)

THETAS = np.linspace(0.0 * np.pi, 0.5 * np.pi, 40)
DURATIONS_NS = np.arange(1.0, 1200.0, 30)

N_INITS = 4

J_MIN = 0.0
J_MAX = 10.0
J_INIT_SCALE = 1e-1
J_INIT_MEANS = [0.0, -1.0 + 1e-3]

MAX_N_TIME_STEPS = 50
MIN_SEGMENT_DURATION_NS = 1.0

ENTANGLEMENT_CAP = 0.01
CLEAN_INF_THRESHOLD = 1e-5

T1_US = 1.0e6
T2_US = 1.0e3

INFID_FLOOR = 1e-16


# ============================================================
# Operators
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)

Z1 = np.kron(Z, I2)
Z2 = np.kron(I2, Z)


# ============================================================
# Helpers
# ============================================================

def n_time_steps_for_duration(T_ns):
    return max(
        1,
        min(MAX_N_TIME_STEPS, int(np.floor(T_ns / MIN_SEGMENT_DURATION_NS))),
    )


def j_init_mean_for_init(init_index):
    return J_INIT_MEANS[0] if init_index < N_INITS // 2 else J_INIT_MEANS[1]


def local_relaxation_ops(gamma_1):
    return np.sqrt(gamma_1 / 4.0) * np.array(
        [np.kron(SIGMA_MINUS, I2), np.kron(I2, SIGMA_MINUS)],
        dtype=np.complex128,
    )


def local_dephasing_ops(gamma_phi):
    return np.sqrt(gamma_phi / 4.0) * np.array(
        [np.kron(Z, I2), np.kron(I2, Z)],
        dtype=np.complex128,
    )


def T1_T2_lindblad_ops(T1_us, T2_us):
    gamma_1 = 1.0 / T1_us
    gamma_phi = max(0.0, 1.0 / T2_us - 0.5 / T1_us)

    ops = list(local_relaxation_ops(gamma_1))
    if gamma_phi > 0.0:
        ops += list(local_dephasing_ops(gamma_phi))

    return np.asarray(ops, dtype=np.complex128)


def make_config(theta, T_ns, n_steps, seed, j_init_mean, entanglement_optimisation):
    return SimulationConfig(
        zeeman_splittings=ZEEMAN,
        initial_state=INITIAL_STATE,
        theta_target=theta,
        n_time_steps=n_steps,
        n_discretisation=1,
        T=T_ns,
        max_drive_strength=0.0,
        J_max=J_MAX,
        J_min=J_MIN,
        J_init_scale=J_INIT_SCALE,
        J_init_mean=j_init_mean,
        feromagnetic=False,
        gtol=1e-12,
        maxiter=500,
        entanglement_optimisation=entanglement_optimisation,
        state_ent_measure="von_neumann",
        J_init_seed=seed,
        S_max=ENTANGLEMENT_CAP,
        eps_infidelity=1e-12,
        rho=0.0,
        dual_lr=1e10,
        n_outer=5,
        inner_maxiter=500,
        fd_eps=1e-6,
    )


def result_infidelity(res):
    return max(float(np.real(res.val_final[-1])), INFID_FLOOR)


def result_s_max(res):
    _, ent_traj_fn = STATE_MEASURES[res.config.state_ent_measure]
    return float(np.max(ent_traj_fn(np.asarray(res.final_traj))))


def make_target_rho(theta, T_ns):
    deltaB = ZEEMAN - np.mean(ZEEMAN)

    U_rot = expm(
        -1j
        * np.pi
        * (deltaB[0] * Z1 + deltaB[1] * Z2)
        * T_ns
        * NS_TO_US
    )

    psi_target = U_rot @ single_qubit_excitation(theta) @ INITIAL_STATE
    return np.outer(psi_target, psi_target.conj())


def make_noisy_system(T_ns, n_steps, L0):
    deltaB = ZEEMAN - np.mean(ZEEMAN)

    sys = SpinChain(
        spins=2,
        zeeman_splittings=deltaB,
        max_drive_strength=0.0,
        J_max=J_MAX,
        J_min=J_MIN,
        feromagnetic=False,
        use_graph=True,
    ).add_lindblad_terms(L0=L0)

    dt_us = T_ns / n_steps * NS_TO_US
    rho0 = np.outer(INITIAL_STATE, INITIAL_STATE.conj())

    def pulse_function(x):
        x = tf.cast(tf.convert_to_tensor(x), tf.complex128)
        x = tf.reshape(x, (n_steps, 1))

        return (
            tf.zeros((n_steps, 2), dtype=tf.complex128),
            tf.cast(tf.convert_to_tensor(ZEEMAN), tf.complex128),
            x,
            tf.cast(tf.convert_to_tensor(rho0), tf.complex128),
            tf.cast(dt_us * 2 * np.pi, tf.complex128),
        )

    return sys.pulse_form(pulse_function)


def noisy_infidelity(x, theta, T_ns, n_steps, L0):
    driven = make_noisy_system(T_ns, n_steps, L0)
    rho = driven.propagate(x)
    target_rho = make_target_rho(theta, T_ns)
    return max(1.0 - float(np.real(np.trace(rho @ target_rho))), INFID_FLOOR)


def select_best(candidates, entanglement_optimisation):
    if not entanglement_optimisation:
        return min(candidates, key=lambda c: c["I"])

    feasible = [c for c in candidates if c["I"] <= CLEAN_INF_THRESHOLD]
    if feasible:
        return min(feasible, key=lambda c: (c["S"], c["I"]))

    return min(candidates, key=lambda c: (c["I"], c["S"]))


# ============================================================
# Worker function
# ============================================================

def run_single_point(task):
    i, j, theta, T_ns = task

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

    n_steps = n_time_steps_for_duration(T_ns)
    L0 = T1_T2_lindblad_ops(T1_US, T2_US)

    output = {
        "i": i,
        "j": j,
        "theta": theta,
        "T_ns": T_ns,
        "n_steps": n_steps,
        "failed": False,
        "error": "",
    }

    try:
        point_results = {}

        for ent_flag, label in [(False, "base"), (True, "ent")]:
            candidates = []

            for init_index in range(N_INITS):
                seed = init_index
                j_mean = j_init_mean_for_init(init_index)

                config = make_config(
                    theta=theta,
                    T_ns=T_ns,
                    n_steps=n_steps,
                    seed=seed,
                    j_init_mean=j_mean,
                    entanglement_optimisation=ent_flag,
                )

                res = sqe_optimisation(config)

                I = result_infidelity(res)
                S = result_s_max(res)

                candidates.append(
                    {
                        "seed": seed,
                        "j_init_mean": j_mean,
                        "x": np.asarray(res.x_final, dtype=float),
                        "I": I,
                        "S": S,
                        "success": bool(res.optimization_success),
                        "nit": int(res.n_iterations),
                    }
                )

            best = select_best(candidates, ent_flag)

            noisy_I = noisy_infidelity(
                best["x"],
                theta=theta,
                T_ns=T_ns,
                n_steps=n_steps,
                L0=L0,
            )

            point_results[label] = {
                **best,
                "noisy_I": noisy_I,
            }

        output.update(point_results)

    except Exception as exc:
        output["failed"] = True
        output["error"] = repr(exc)

    return output


# ============================================================
# Saving
# ============================================================

def initialise_arrays():
    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)

    return {
        "base_noisy_infids": np.full((n_theta, n_T), np.nan),
        "ent_noisy_infids": np.full((n_theta, n_T), np.nan),
        "base_clean_infids": np.full((n_theta, n_T), np.nan),
        "ent_clean_infids": np.full((n_theta, n_T), np.nan),
        "base_s_max": np.full((n_theta, n_T), np.nan),
        "ent_s_max": np.full((n_theta, n_T), np.nan),
        "base_seed": np.full((n_theta, n_T), -1, dtype=int),
        "ent_seed": np.full((n_theta, n_T), -1, dtype=int),
        "base_j_init_mean": np.full((n_theta, n_T), np.nan),
        "ent_j_init_mean": np.full((n_theta, n_T), np.nan),
        "base_success": np.full((n_theta, n_T), False, dtype=bool),
        "ent_success": np.full((n_theta, n_T), False, dtype=bool),
        "base_nit": np.full((n_theta, n_T), -1, dtype=int),
        "ent_nit": np.full((n_theta, n_T), -1, dtype=int),
        "n_steps_used": np.full((n_theta, n_T), -1, dtype=int),
        "x_base": np.full((n_theta, n_T, MAX_N_TIME_STEPS), np.nan),
        "x_ent": np.full((n_theta, n_T, MAX_N_TIME_STEPS), np.nan),
        "failed": np.full((n_theta, n_T), False, dtype=bool),
        "error_messages": np.full((n_theta, n_T), "", dtype=object),
    }


def update_arrays(arrays, result):
    i = result["i"]
    j = result["j"]
    n_steps = result["n_steps"]

    arrays["n_steps_used"][i, j] = n_steps
    arrays["failed"][i, j] = result["failed"]
    arrays["error_messages"][i, j] = result["error"]

    if result["failed"]:
        return

    base = result["base"]
    ent = result["ent"]

    arrays["base_clean_infids"][i, j] = base["I"]
    arrays["ent_clean_infids"][i, j] = ent["I"]

    arrays["base_noisy_infids"][i, j] = base["noisy_I"]
    arrays["ent_noisy_infids"][i, j] = ent["noisy_I"]

    arrays["base_s_max"][i, j] = base["S"]
    arrays["ent_s_max"][i, j] = ent["S"]

    arrays["base_seed"][i, j] = base["seed"]
    arrays["ent_seed"][i, j] = ent["seed"]

    arrays["base_j_init_mean"][i, j] = base["j_init_mean"]
    arrays["ent_j_init_mean"][i, j] = ent["j_init_mean"]

    arrays["base_success"][i, j] = base["success"]
    arrays["ent_success"][i, j] = ent["success"]

    arrays["base_nit"][i, j] = base["nit"]
    arrays["ent_nit"][i, j] = ent["nit"]

    arrays["x_base"][i, j, :n_steps] = base["x"]
    arrays["x_ent"][i, j, :n_steps] = ent["x"]


def save_npz(arrays):
    np.savez_compressed(
        DATA_OUT,
        thetas=THETAS,
        durations_ns=DURATIONS_NS,
        T1_us=T1_US,
        T2_us=T2_US,
        max_n_time_steps=MAX_N_TIME_STEPS,
        min_segment_duration_ns=MIN_SEGMENT_DURATION_NS,
        entanglement_cap=ENTANGLEMENT_CAP,
        clean_inf_threshold=CLEAN_INF_THRESHOLD,
        zeeman=ZEEMAN,
        initial_state=INITIAL_STATE,
        J_min=J_MIN,
        J_max=J_MAX,
        J_init_scale=J_INIT_SCALE,
        J_init_means=np.asarray(J_INIT_MEANS),
        n_inits=N_INITS,
        n_workers=N_WORKERS,
        **arrays,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Parallel SQE state noisy MET scan")
    print(f"N_WORKERS = {N_WORKERS}")
    print(f"T1 = {T1_US:.3e} us")
    print(f"T2 = {T2_US:.3e} us")
    print(f"N_INITS = {N_INITS}")
    print(f"First half J_init_mean = {J_INIT_MEANS[0]}")
    print(f"Second half J_init_mean = {J_INIT_MEANS[1]}")
    print(f"Output = {DATA_OUT}")
    print("=" * 72)

    arrays = initialise_arrays()

    tasks = [
        (i, j, float(theta), float(T_ns))
        for i, theta in enumerate(THETAS)
        for j, T_ns in enumerate(DURATIONS_NS)
    ]

    total = len(tasks)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_single_point, task) for task in tasks]

        with tqdm(total=total, desc="SQE noisy MET", unit="point") as pbar:
            for future in as_completed(futures):
                result = future.result()
                update_arrays(arrays, result)

                i = result["i"]
                j = result["j"]

                if result["failed"]:
                    pbar.write(
                        f"FAILED theta_idx={i}, T_idx={j}: {result['error']}"
                    )
                else:
                    pbar.write(
                        f"Done theta_idx={i}, T_idx={j}, "
                        f"I_base={result['base']['noisy_I']:.3e}, "
                        f"I_ent={result['ent']['noisy_I']:.3e}, "
                        f"ratio={result['base']['noisy_I'] / result['ent']['noisy_I']:.3e}"
                    )

                save_npz(arrays)
                pbar.update(1)

    save_npz(arrays)
    print(f"\nAll jobs finished. Saved results to {DATA_OUT}")


if __name__ == "__main__":
    main()