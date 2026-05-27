"""
Entanglement measures for quantum states and unitaries.

Three sections
--------------
State measures    : functions of a 2-qubit pure-state vector or trajectory.
Bipartite entropy : general n-qubit bipartite von Neumann entropy, used by DQE.
Gate measures     : functions of a 4×4 unitary matrix.

Lookup registries
-----------------
STATE_MEASURES : dict[str, (single_fn, traj_fn)]
    single_fn(state: ndarray(4,))       -> float
    traj_fn(traj:  ndarray(4, n_t))     -> ndarray(n_t,)

GATE_MEASURES : dict[str, fn]
    fn(U: ndarray(4, 4))                -> float

STATE_MEASURE_LABELS, GATE_MEASURE_LABELS : dict[str, str]
    Human-readable y-axis labels for plots.

GATE_MEASURE_DETERMINISTIC : dict[str, bool]
    True for measures that are cheap to evaluate at plot-time (no MC).

Selecting measures
------------------
Set ``config.state_ent_measure`` or ``config.gate_ent_measure`` to any key
in the corresponding registry. Any registered measure can be optimised or
constrained without changing any other code.

Set ``config.extra_ent_measures`` to a list of additional measure names to
have them plotted on separate axes alongside the main result.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# Private helpers
# =============================================================================

def _haar_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random pure state in C^dim."""
    z = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return z / np.linalg.norm(z)


def _product_state(rng: np.random.Generator) -> np.ndarray:
    """Random 2-qubit product state |ψ_A⟩ ⊗ |ψ_B⟩."""
    return np.kron(_haar_state(2, rng), _haar_state(2, rng))


# =============================================================================
# State entanglement measures
# =============================================================================

def von_neumann_entropy_single(state: np.ndarray) -> float:
    """
    Von Neumann entropy S(ρ_A) in bits for a 2-qubit pure state.

    Computes rho_A = Tr_B(|ψ⟩⟨ψ|) then S = -Tr(ρ_A log₂ ρ_A).
    """
    rho_A = state.reshape(2, 2) @ state.reshape(2, 2).conj().T
    eig = np.clip(np.linalg.eigvalsh(rho_A), 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def von_neumann_entropy_traj(traj: np.ndarray) -> np.ndarray:
    """
    Von Neumann entropy S(ρ_A) in bits along a 2-qubit state trajectory.

    Parameters
    ----------
    traj : ndarray, shape (4, n_t)
        State at each time step.

    Returns
    -------
    ndarray, shape (n_t,)
    """
    psi = np.asarray(traj).T.reshape(-1, 2, 2)          # (n_t, 2, 2)
    rho_A = psi @ np.conj(psi).transpose(0, 2, 1)        # (n_t, 2, 2)
    eig = np.clip(np.linalg.eigvalsh(rho_A), 1e-15, 1.0)
    return -np.sum(eig * np.log2(eig), axis=1)


def linear_entropy_single(state: np.ndarray) -> float:
    """
    Linear entropy S_L = 1 - Tr(ρ_A²) for a 2-qubit pure state.

    Ranges from 0 (product state) to 1/2 (maximally entangled).
    Cheaper to compute than von Neumann entropy (no eigendecomposition).
    """
    rho_A = state.reshape(2, 2) @ state.reshape(2, 2).conj().T
    return float(np.real(1.0 - np.trace(rho_A @ rho_A)))


def linear_entropy_traj(traj: np.ndarray) -> np.ndarray:
    """
    Linear entropy S_L = 1 - Tr(ρ_A²) along a 2-qubit state trajectory.

    Parameters
    ----------
    traj : ndarray, shape (4, n_t)

    Returns
    -------
    ndarray, shape (n_t,)
    """
    psi = np.asarray(traj).T.reshape(-1, 2, 2)           # (n_t, 2, 2)
    rho_A = psi @ np.conj(psi).transpose(0, 2, 1)         # (n_t, 2, 2)
    tr_rho2 = np.einsum('tij,tji->t', rho_A, rho_A).real
    return 1.0 - tr_rho2


# =============================================================================
# N-qubit bipartite entropy (general, used by DQE)
# =============================================================================

def _reshape_state_for_partition(state: np.ndarray,
                                  subsystem: tuple[int, ...],
                                  n_qubits: int) -> np.ndarray:
    """
    Reshape a pure state into (dim_A, dim_B) for Schmidt decomposition.

    Reorders qubit axes so subsystem A comes first, then traces over B.
    Convention: big-endian, index = int(q0 q1 ... q_{n-1}, 2).
    """
    psi = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)
    A   = tuple(subsystem)
    B   = tuple(q for q in range(n_qubits) if q not in A)
    psi = np.transpose(psi, A + B)
    return psi.reshape(2 ** len(A), 2 ** len(B))


def entropy_bipartite_single(state: np.ndarray, *,
                              subsystem=(0, 1), n_qubits=4) -> float:
    """
    Von Neumann entropy S(A) for bipartition A|B of a pure state (in bits).

    Parameters
    ----------
    state : ndarray, shape (2^n_qubits,)
    subsystem : tuple[int, ...]
        Qubit indices forming subsystem A.
    n_qubits : int
        Total number of qubits.
    """
    psi_AB = _reshape_state_for_partition(state, tuple(subsystem), n_qubits)
    rho_A = psi_AB @ psi_AB.conj().T
    eig = np.clip(np.linalg.eigvalsh(rho_A), 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def entropy_bipartite_traj(traj: np.ndarray, *,
                            subsystem=(0, 1), n_qubits=4) -> np.ndarray:
    """
    Von Neumann entropy S(A) along a state trajectory.

    Parameters
    ----------
    traj : ndarray, shape (2^n_qubits, n_t)
    subsystem : tuple[int, ...]
    n_qubits : int

    Returns
    -------
    ndarray, shape (n_t,)
    """
    traj = np.asarray(traj)
    return np.array([
        entropy_bipartite_single(traj[:, t], subsystem=subsystem, n_qubits=n_qubits)
        for t in range(traj.shape[1])
    ])


# All distinct bipartite partitions of 4 qubits
ALL_BIPARTITIONS_4Q: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (2,), (3,),      # four single-qubit cuts
    (0, 1), (0, 2), (0, 3),      # three two-qubit cuts
)


def entropy_sum_single(state: np.ndarray, *, n_qubits=4) -> float:
    """
    Sum of von Neumann entropies over all 7 distinct bipartitions of 4 qubits.

    Used as the DQE entanglement cost function.
    """
    return sum(
        entropy_bipartite_single(state, subsystem=p, n_qubits=n_qubits)
        for p in ALL_BIPARTITIONS_4Q
    )


def entropy_sum_traj(traj: np.ndarray, *, n_qubits=4) -> np.ndarray:
    """
    Sum of bipartite entropies over all 7 distinct bipartitions along a trajectory.

    Parameters
    ----------
    traj : ndarray, shape (2^n_qubits, n_t)

    Returns
    -------
    ndarray, shape (n_t,)
    """
    return sum(
        entropy_bipartite_traj(traj, subsystem=p, n_qubits=n_qubits)
        for p in ALL_BIPARTITIONS_4Q
    )


# =============================================================================
# Gate entanglement measures — 2-qubit (4×4 unitaries)
# =============================================================================

def operator_entanglement(U: np.ndarray) -> float:
    """
    Shannon entropy of the squared operator-Schmidt coefficients (deterministic).

    Treats U as a state on (A_out ⊗ A_in) | (B_out ⊗ B_in), computes the
    operator Schmidt decomposition via SVD, and returns -Σ p_i log₂(p_i)
    where p_i = σ_i² are the normalised squared singular values.

    This is the default gate entanglement measure.
    """
    T = (U / 2.0).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    s = np.linalg.svd(T, compute_uv=False)
    p = s**2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def average_entangling_power(U: np.ndarray, n_samples: int = 20,
                             seed: int | None = None) -> float:
    """
    MC estimate of average output entanglement over random product inputs.

    Computes mean_{|ψ⟩} S(U|ψ⟩) where |ψ⟩ is a Haar-random product state
    and S is the von Neumann entropy of qubit A.
    """
    rng = np.random.default_rng(seed)
    return float(np.mean([
        von_neumann_entropy_single(U @ _product_state(rng)) for _ in range(n_samples)
    ]))

"""
def average_entangling_power(
    U: np.ndarray,
) -> float:

    state1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    state2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)

    return float((von_neumann_entropy_single(U @ state1) + von_neumann_entropy_single(U @ state2)) / 2.0)
"""


def max_output_entanglement(U: np.ndarray, n_trials: int = 10,
                            seed: int | None = None) -> float:
    """
    MC estimate of maximum output entanglement over random product inputs.

    Approximates max_{|ψ⟩} S(U|ψ⟩) over product states via random search.
    """
    rng = np.random.default_rng(seed)
    return float(max(
        von_neumann_entropy_single(U @ _product_state(rng)) for _ in range(n_trials)
    ))


def assisted_entangling_power(U: np.ndarray, n_trials: int = 10,
                              seed: int | None = None) -> float:
    """
    MC estimate of maximum entanglement gain over arbitrary pure inputs.

    Approximates max_{|ψ⟩} [S(U|ψ⟩) - S(|ψ⟩)] over all 2-qubit pure states
    via random search. Does not include ancilla assistance.
    """
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(n_trials):
        psi = _haar_state(4, rng)
        best = max(best,
                   von_neumann_entropy_single(U @ psi) - von_neumann_entropy_single(psi))
    return float(best)


# Default alias used when no config measure is specified
gate_entanglement = operator_entanglement


# =============================================================================
# Gate entanglement measures — n-qubit (general, used by DQE gate)
# =============================================================================

def operator_entanglement_bipartite(U: np.ndarray, *,
                                     subsystem=(0, 1), n_qubits=4) -> float:
    """
    Operator entanglement for an n-qubit gate over bipartition subsystem|rest.

    Reshapes U as a bilinear map (A_out ⊗ B_out) ← (A_in ⊗ B_in), computes
    the operator Schmidt decomposition via SVD, and returns the Shannon entropy
    of the squared singular values.  Normalised so that Σ p_i = 1 for unitary U.

    Convention matches operator_entanglement for 2-qubit gates: divide by
    sqrt(2^n_qubits) before SVD.
    """
    n = n_qubits
    A = tuple(subsystem)
    B = tuple(q for q in range(n) if q not in A)
    # Reshape U (2^n × 2^n): axes 0..n-1 = output qubits, axes n..2n-1 = input qubits
    U_r = U.reshape((2,) * (2 * n))
    # Permute to (A_out, A_in, B_out, B_in), then reshape to (dim_A², dim_B²)
    perm = tuple(A) + tuple(a + n for a in A) + tuple(B) + tuple(b + n for b in B)
    T = U_r.transpose(perm).reshape(4 ** len(A), 4 ** len(B))
    T = T / np.sqrt(2 ** n)
    s = np.linalg.svd(T, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def operator_entanglement_sum_4q(U: np.ndarray) -> float:
    """
    Sum of operator entanglement over all 7 distinct bipartitions of a 4-qubit gate.

    Analogous to entropy_sum_single for states: sums operator_entanglement_bipartite
    over ALL_BIPARTITIONS_4Q.  Used as the default DQE gate entanglement cost.
    """
    return sum(
        operator_entanglement_bipartite(U, subsystem=p, n_qubits=4)
        for p in ALL_BIPARTITIONS_4Q
    )


# =============================================================================
# Lookup registries
# =============================================================================

STATE_MEASURES: dict[str, tuple] = {
    "von_neumann": (von_neumann_entropy_single, von_neumann_entropy_traj),
    "linear":      (linear_entropy_single,       linear_entropy_traj),
}
"""Registry: name -> (single_fn, traj_fn) for state entanglement measures."""

STATE_MEASURE_LABELS: dict[str, str] = {
    "von_neumann": "Von Neumann Entropy S (bits)",
    "linear":      r"Linear Entropy $S_L = 1 - \mathrm{Tr}(\rho_A^2)$",
}
"""Y-axis labels for state entanglement plots."""

STATE_MEASURE_TITLES: dict[str, str] = {
    "von_neumann": "Von Neumann Entropy Evolution",
    "linear":      "Linear Entropy Evolution",
}
"""Panel titles for state entanglement plots."""

GATE_MEASURES: dict[str, object] = {
    # 2-qubit (4×4 unitary)
    "operator_entanglement":        operator_entanglement,
    "average_entangling_power":     average_entangling_power,
    "max_output_entanglement":      max_output_entanglement,
    "assisted_entangling_power":    assisted_entangling_power,
    # 4-qubit (16×16 unitary)
    "operator_entanglement_sum_4q": operator_entanglement_sum_4q,
}
"""Registry: name -> fn(U) -> float for gate entanglement measures."""

# Names of measures valid for 4-qubit (16×16) gates
GATE_MEASURES_4Q: frozenset[str] = frozenset({"operator_entanglement_sum_4q"})

GATE_MEASURE_LABELS: dict[str, str] = {
    "operator_entanglement":        "Operator Entanglement (bits)",
    "average_entangling_power":     "Avg. Entangling Power (bits)",
    "max_output_entanglement":      "Max Output Entanglement (bits)",
    "assisted_entangling_power":    "Assisted Entangling Power (bits)",
    "operator_entanglement_sum_4q": r"$\Sigma_P$ Op. Entanglement (bits)",
}
"""Y-axis labels for gate entanglement plots."""

GATE_MEASURE_TITLES: dict[str, str] = {
    "operator_entanglement":        "Operator Entanglement Evolution",
    "average_entangling_power":     "Average Entangling Power Evolution",
    "max_output_entanglement":      "Max Output Entanglement Evolution",
    "assisted_entangling_power":    "Assisted Entangling Power Evolution",
    "operator_entanglement_sum_4q": "Total Operator Entanglement (sum over 7 bipartitions)",
}
"""Panel titles for gate entanglement plots."""

GATE_MEASURE_DETERMINISTIC: dict[str, bool] = {
    "operator_entanglement":        True,
    "average_entangling_power":     False,
    "max_output_entanglement":      False,
    "assisted_entangling_power":    False,
    "operator_entanglement_sum_4q": True,
}
"""True for measures cheap enough to evaluate a target line at plot time."""
