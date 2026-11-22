# pulse_VQE_entanglement
Extension for QuGrad and QuGradLab for entanglement optimisation.

The state optimisers with analytic transient-entropy gradients are
[`sqe_state_adjoint.py`](sqe_state_adjoint.py) and
[`dqe_state_adjoint.py`](dqe_state_adjoint.py). Their gradient implementations
are [`sqe_state_entropy_gradient.py`](sqe_state_entropy_gradient.py) and
[`dqe_state_entropy_gradient.py`](dqe_state_entropy_gradient.py). Matching
notebooks are in [`notebooks/`](notebooks/), with derivations in
[`SQE_STATE_ADJOINT.md`](SQE_STATE_ADJOINT.md) and
[`DQE_STATE_ADJOINT.md`](DQE_STATE_ADJOINT.md).
