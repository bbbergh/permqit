"""Superactivation channel utilities.

The channel used to demonstrate non-asymptotic superactivation is a flagged mixture of the
identity and the independent X-Z Pauli channel of ``pauli_channel``:

    N(rho) = q |0><0|_Z (x) id(rho)_B + (1-q) |1><1|_Z (x) P_p(rho)_B,

where P_p applies X with probability p and Z with probability 1/2, and the classical flag Z is
delivered to the receiver. Both branches have zero quantum capacity, yet the mixture does not.

The flag is classical, so for n uses the flag string reveals how many uses k took the noisy
branch, and the n-use channel decomposes into the n+1 sectors

    N_k = P_p^{(x)k} (x) id^{(x)(n-k)},   with weight C(n,k) (1-q)^k q^(n-k),

each of which is decoded separately. The seesaw functions take the two branches as a list and
build the sectors internally, which is what ``superactivation_choi`` returns.
"""
import numpy as np

from examples.channels.pauli_channel import independent_xz_choi

__all__ = [
    "DEFAULT_P",
    "identity_choi",
    "superactivation_choi",
    "superactivation_flagged_choi",
]

DEFAULT_P = 1.0 / (1.0 + 2.0**0.5)
"""Value of p at which the two branches are both antidegradable / PPT, and the mixture is not."""


def identity_choi() -> np.ndarray:
    """Return the unnormalized Choi matrix of the qubit identity channel."""
    J = np.zeros((4, 4), dtype=complex)
    J[0, 0] = J[0, 3] = J[3, 0] = J[3, 3] = 1.0
    return J


def superactivation_choi(p: float = DEFAULT_P) -> list[np.ndarray]:
    """Return the two branches [P_p, id] of the superactivation channel.

    This is the form expected by ``compute_tensor_product_fidelity_seesaw``, which pairs the
    branches with the mixing probabilities and builds the n+1 flag sectors itself.

    Args:
        p: Probability of an X error in the noisy branch (default: DEFAULT_P).
    """
    return [independent_xz_choi(p), identity_choi()]


def superactivation_flagged_choi(p: float = DEFAULT_P, q: float = 0.5) -> np.ndarray:
    """Return the unnormalized Choi matrix of one use of the flagged channel A -> Z (x) B.

    The output is ordered (Z, B) with Z = 0 the identity branch, giving a 2x4 = 8 dimensional
    output and an 16x16 Choi matrix. Provided for inspection; the seesaw uses the branch list
    from ``superactivation_choi`` instead.

    Args:
        p: Probability of an X error in the noisy branch.
        q: Probability of the identity branch.
    """
    branches = [(q, identity_choi()), (1 - q, independent_xz_choi(p))]
    J = np.zeros((16, 16), dtype=complex)
    for flag, (weight, branch) in enumerate(branches):
        block = weight * branch.reshape(2, 2, 2, 2)  # (A, B, A', B')
        for a in range(2):
            for b in range(2):
                for a2 in range(2):
                    for b2 in range(2):
                        row = a * 8 + flag * 2 + b
                        col = a2 * 8 + flag * 2 + b2
                        J[row, col] = block[a, b, a2, b2]
    return J


if __name__ == "__main__":
    branches = superactivation_choi()
    print(f"superactivation channel at p = {DEFAULT_P:.6f}")
    for name, J in zip(("P_p", "id"), branches):
        print(f"  {name}: Tr_B J = I is {np.allclose(np.einsum('ibjb->ij', J.reshape(2, 2, 2, 2)), np.eye(2))}")
    J = superactivation_flagged_choi()
    print(f"  flagged Choi is {J.shape[0]}x{J.shape[1]}, "
          f"Tr_ZB J = I is {np.allclose(np.einsum('ibjb->ij', J.reshape(2, 8, 2, 8)), np.eye(2))}")
