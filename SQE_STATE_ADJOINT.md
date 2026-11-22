# Adjoint gradient of transient SQE entanglement entropy

This note derives the gradient implemented in
[`sqe_state_entropy_gradient.py`](sqe_state_entropy_gradient.py) and used by
[`sqe_state_adjoint.py`](sqe_state_adjoint.py). The
calculation concerns a pure state of two spin qubits driven by one piecewise
constant exchange coupling. The implementation supports von Neumann entropy
and linear entropy.

## 1. Pulse and propagation conventions

Let there be $m$ coarse pulse parameters $x_j\in[-1,1]$, with each one
held for $d$ fine steps. There are $N=md$ fine steps and $N+1$ states,
including the initial state. The physical exchange coupling in MHz is

$$
J_j=\frac{J_{\min}+J_{\max}+(J_{\max}-J_{\min})x_j}{2},
\qquad
\frac{\partial J_j}{\partial x_j}=\frac{J_{\max}-J_{\min}}{2}.
$$

The optimizer removes the mean Zeeman splitting. Write

$$
\delta B_i=B_i-\frac{B_0+B_1}{2},\qquad
H_0=\frac{\delta B_0}{2}(Z\otimes I)
   +\frac{\delta B_1}{2}(I\otimes Z),\qquad
H_{\rm ex}=\frac{X\otimes X+Y\otimes Y+Z\otimes Z}{4}.
$$

These matrices use $Z=\operatorname{diag}(1,-1)$ and the state-vector basis
$(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$. The resulting matrices
agree with the two-spin `SpinChain` Hamiltonians after its site and Pauli
conventions are accounted for. Set $s=-1$ for `feromagnetic=True` and
$s=+1$ otherwise.

The dimensionless timestep passed to PySTE is

$$
\Delta t=2\pi\frac{T_{\rm ns}\,10^{-3}}{N}.
$$

The QuGrad pulse path rounds this timestep through `float32` before PySTE
uses it. The adjoint implementation performs the same rounding so its
trajectory matches the simulator at floating-point precision.

PySTE uses a first-order Suzuki–Trotter step. For fine step $k$, let
$j(k)=\lfloor k/d\rfloor$. Then

$$
\psi_{k+1}=U_k\psi_k,\qquad
U_k=\exp[-i\Delta t\,sJ_{j(k)}H_{\rm ex}]
    \exp[-i\Delta t H_0].
$$

It is essential to differentiate this ordered product: replacing it with
$\exp[-i\Delta t(H_0+sJH_{\rm ex})]$ would describe a different numerical
propagator. As $H_{\rm ex}$ commutes with its own exponential,

$$
\boxed{\displaystyle
\frac{\partial U_k}{\partial x_j}
=\mathbf{1}_{j=j(k)}
\left[-i\Delta t\,s\frac{J_{\max}-J_{\min}}{2}H_{\rm ex}\right]U_k.}
\tag{1}
$$

## 2. Entropy derivative with respect to the state

Reshape $\psi_k\in\mathbb C^4$ into a $2\times2$ amplitude matrix
$M_k$, with the first qubit indexing rows. The reduced state is
$\rho_k=M_kM_k^\dagger$. For any real function $S$ of the state, define
its complex state gradient $q_k^S$ by the differential convention

$$
dS=2\operatorname{Re}\big[(q_k^S)^\dagger d\psi_k\big].
\tag{2}
$$

### Von Neumann entropy

For entropy measured in bits,

$$
S_{\rm vN}(\rho)=-\operatorname{Tr}(\rho\log_2\rho).
$$

At full rank, matrix differentiation gives

$$
dS_{\rm vN}
=-\frac{1}{\ln2}\operatorname{Tr}[(\ln\rho+I)d\rho],
\qquad
d\rho=dM\,M^\dagger+M\,dM^\dagger.
$$

Combining the two conjugate terms and comparing with (2) yields

$$
\boxed{\displaystyle
q^S_{\rm vN}
=\operatorname{vec}\!\left[
-\frac{\ln\rho+I}{\ln2}\,M
\right].}
\tag{3}
$$

Here $\operatorname{vec}$ flattens rows in the same order as NumPy's
default reshape of the state vector.

In code, $\rho=V\operatorname{diag}(\lambda_a)V^\dagger$, and each
$\lambda_a$ is floored at $10^{-15}$ when evaluating both the entropy
and logarithm. Equation (3) is exact away from the eigenvalue floor; the
floor regularizes the logarithm at product states.

### Linear entropy

For $S_L=1-\operatorname{Tr}(\rho^2)$,

$$
dS_L=-2\operatorname{Tr}(\rho\,d\rho),\qquad
\boxed{\displaystyle q^S_L=\operatorname{vec}(-2\rho M).}
\tag{4}
$$

## 3. Transient entropy cost

The optimizer minimizes the mean squared violation of an entropy cap:

$$
C(x)=\frac{1}{N+1}\sum_{k=0}^{N}v_k^2,
\qquad
v_k=[S(\psi_k)-S_{\max}]_+
=\max(S(\psi_k)-S_{\max},0).
\tag{5}
$$

Both $\psi_0$ and $\psi_N$ are included, matching the simulator’s
`propagate_all` trajectory. The derivative of the squared hinge is zero on
the inactive side and at the cap. Thus the direct state gradient of the
*total cost contribution* at time $k$ is

$$
\boxed{\displaystyle
q_k=\frac{2v_k}{N+1}\,q_k^S,\qquad
dC=2\operatorname{Re}\sum_{k=0}^{N}q_k^\dagger d\psi_k.}
\tag{6}
$$

## 4. Backward adjoint and pulse gradient

A perturbation of the trajectory satisfies

$$
d\psi_{k+1}=U_k\,d\psi_k+(dU_k)\psi_k.
$$

Because the initial state is fixed, $d\psi_0=0$. Define a costate that
collects the effect of all future entropy terms:

$$
\boxed{\displaystyle
p_N=q_N,\qquad p_k=q_k+U_k^\dagger p_{k+1}
\quad(k=N-1,\ldots,0).}
\tag{7}
$$

Substituting the perturbed state recurrence into (6), then working backward,
gives one contribution per fine step:

$$
dC=2\operatorname{Re}\sum_{k=0}^{N-1}
p_{k+1}^\dagger(dU_k)\psi_k.
$$

All $d$ fine steps belonging to coarse parameter $x_j$ contribute to
the same gradient component. Using equation (1),

$$
\boxed{\displaystyle
\frac{\partial C}{\partial x_j}
=2\operatorname{Re}
\sum_{k=jd}^{(j+1)d-1}
p_{k+1}^\dagger
\left[-i\Delta t\,s\frac{J_{\max}-J_{\min}}{2}
H_{\rm ex}U_k\right]\psi_k.}
\tag{8}
$$

The forward states, entropy gradients, and backward costates give every
component of $\nabla C$ in one forward and one backward traversal.

## 5. Fidelity constraint and Lagrangian

Let $\psi_{\rm target}$ be the target state in the rotating frame and
$O=I-|\psi_{\rm target}\rangle\langle\psi_{\rm target}|$. The final-state
infidelity and the constrained problem are

$$
I(x)=\langle\psi_N|O|\psi_N\rangle,
\qquad
\min_{-1\le x_j\le1} C(x)
\quad\text{subject to}\quad I(x)-\varepsilon\le0.
\tag{9}
$$

The augmented Lagrangian used for each inner pulse optimization is

$$
\mathcal L_\rho(x,\lambda)
=C(x)+\lambda c(x)+\frac{\rho}{2}c(x)^2,
\qquad c(x)=I(x)-\varepsilon,
\qquad
\nabla_x\mathcal L_\rho
=\nabla C+[\lambda+\rho c(x)]\nabla I.
\tag{10}
$$

Here $\rho$ in (10) is the nonnegative penalty coefficient, distinct from
the reduced density matrix used in the entropy formulas.

The inner problem minimizes (10) with L-BFGS-B and pulse bounds. After
each inner solve, the nonnegative multiplier is updated by

$$
\lambda_{r+1}
=\max\{0,\lambda_r+\alpha c(x_{r+1})\},
\qquad \lambda_0=1.
\tag{11}
$$

The fidelity-only baseline is computed separately for comparison. As in
the earlier finite-difference augmented-Lagrangian implementation, the
entropy stage starts from the original pulse $x_0$. Equation (8) supplies
$\nabla C$; QuGrad supplies $I$ and $\nabla I$. At a constrained optimum,
the KKT conditions include $c\le0$, $\lambda\ge0$, and $\lambda c=0$,
together with stationarity subject to the pulse bounds. The inner solver
uses a relative function-change tolerance of $10^{-14}$ because its
default tolerance can stop while the entropy cost is still falling.

## 6. Cost and numerical check

Central differences require two complete simulator trajectories per coarse
parameter, or $2m$ propagations for one gradient. The adjoint uses one
forward trajectory and one backward traversal, plus one small exchange
spectral phase calculation per coarse parameter. For fixed two-qubit matrix size,
its work grows linearly with the number of fine steps.

In the notebook's 12-parameter, two-fine-steps-per-parameter check:

| Comparison | Maximum absolute difference |
| --- | ---: |
| Adjoint and PySTE trajectories | $1.247\times10^{-14}$ |
| Entropy costs | $1.177\times10^{-14}$ |
| Adjoint and central-difference gradients | $6.344\times10^{-10}$ |

The measured gradient time was 0.001 s for the adjoint and 0.007 s for
central differences (7.6× faster in that run). Timings depend on the machine
and pulse length. The notebook reproduces this check before running the
constrained optimizer.

With the same 50-parameter pulse and augmented-Lagrangian settings as the
earlier finite-difference notebook, a complete local run gave:

| Method | Final infidelity | Entropy cost | Maximum entropy | Time |
| --- | ---: | ---: | ---: | ---: |
| Adjoint | $8.179\times10^{-12}$ | $2.548\times10^{-7}$ | 0.011273 bits | 0.450 s |
| Central differences | $9.043\times10^{-12}$ | $5.413\times10^{-7}$ | 0.012535 bits | 2.254 s |

Both runs met the $10^{-8}$ infidelity limit. Their final pulses need not
coincide because the constrained objective has multiple local solutions and
their gradient and stopping behavior differ.
