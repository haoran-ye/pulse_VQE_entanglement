# DQE state adjoint gradient

[`dqe_state_entropy_gradient.py`](dqe_state_entropy_gradient.py) supplies the
analytic entropy-cost gradient used by
[`dqe_state_adjoint.py`](dqe_state_adjoint.py). The cost and fidelity constraint
are those of [`dqe_state.py`](dqe_state.py).

## Four-spin step

There are three exchange controls per coarse time step. A control row
`x[j, 0:3]` is held for `n_discretisation` fine steps. The physical exchange
couplings are

\[
J_{j,c}=\tfrac12\{J_{\min}+J_{\max}+(J_{\max}-J_{\min})x_{j,c}\},
\qquad \partial J_{j,c}/\partial x_{j,c}=(J_{\max}-J_{\min})/2.
\]

The simulator's physical spin 0 is the **rightmost** factor of a state-vector
tensor product. Its Pauli `Z` is `diag(-1, 1)`. Accordingly, if `Z` below
denotes the conventional `diag(1, -1)`, the drift matrix is

\[
H_0=-\tfrac12\sum_{i=0}^3 (B_i-\bar B)Z_{3-i},
\qquad
H_c=\tfrac14(X_{3-c}X_{2-c}+Y_{3-c}Y_{2-c}+Z_{3-c}Z_{2-c}).
\]

Here subscripts on Pauli matrices identify tensor axes from the left. With
`s = -1` for `feromagnetic=True` and `+1` otherwise, the fine step is the
**ordered** Trotter product

\[
U_k=E_{2,k}E_{1,k}E_{0,k}D,\qquad
D=e^{-i\Delta t H_0},\quad E_{c,k}=e^{-i\Delta t s J_{j(k),c}H_c},
\]

where `j(k) = k // n_discretisation` and
`Δt = float32(2π T_ns × 10⁻³ / n_fine)` matches the simulator's timestep
rounding. The matrices `H_c` for adjacent bonds do not commute. Differentiating
the actual ordered product gives, for example,

\[
\partial_{x_{j,1}}U_k
=E_{2,k}(-i\Delta t\,s\,\partial J_{j,1}/\partial x_{j,1}\,H_1)
E_{1,k}E_{0,k}D
\]

when `j = j(k)`, and zero otherwise. The implementation computes all three
derivatives with prefix and suffix products.

## Entropy and backward pass

For each of the seven partitions
`(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3)`, reshape the pure state into
an amplitude matrix `M_P` with the partition's tensor axes first and set
`ρ_P = M_P M_P†`. The entropy sum is `S(ψ) = Σ_P -Tr(ρ_P log₂ρ_P)`.
Using `dS = 2 Re(q† dψ)`, each partition contributes

\[
q_P=\operatorname{unpermute}\!\left[
-\frac{\log\rho_P+I}{\ln 2}M_P\right].
\]

Eigenvalues are floored at `10⁻¹⁵` in both the logarithm and entropy, as in
the finite-difference entropy measure. This regularizes product states; the
gradient is exact away from that floor. The cost includes both the initial
and final states:

\[
C=\frac1{N+1}\sum_{k=0}^N v_k^2,\qquad
v_k=[S(\psi_k)-S_{\max}]_+,\qquad
q_k=\frac{2v_k}{N+1}\sum_P q_{P,k}.
\]

The backward costate and pulse gradient are

\[
p_N=q_N,\quad p_k=q_k+U_k^\dagger p_{k+1},\qquad
\frac{\partial C}{\partial x_{j,c}}
=2\operatorname{Re}\sum_{k:j(k)=j}
p_{k+1}^\dagger(\partial_{x_{j,c}}U_k)\psi_k.
\]

The optimiser adds QuGrad's fidelity gradient to this entropy gradient in the
same augmented Lagrangian used by the finite-difference DQE optimiser. The
notebook [`dqe_state_adjoint.ipynb`](notebooks/dqe_state_adjoint.ipynb)
compares the complete trajectory, cost, and gradient against SpinChain and
central differences for positive and signed ferromagnetic exchange.
