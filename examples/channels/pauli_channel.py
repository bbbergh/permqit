"""Pauli channel utilities.

A mixed Pauli channel applies I, X, Y, Z with probabilities (p_I, p_X, p_Y, p_Z):

    P(rho) = p_I rho + p_X X rho X + p_Y Y rho Y + p_Z Z rho Z.

The independent X-Z channel applies X with probability q_X and Z with probability q_Z,
independently of each other, so its Pauli weights factorize:

    (p_I, p_X, p_Y, p_Z) = ((1-q_X)(1-q_Z), q_X(1-q_Z), q_X q_Z, (1-q_X)q_Z).

Equivalently, a Pauli channel is of this form iff p_I p_Y = p_X p_Z, which leaves it determined
by (p_X, p_Y) alone: q_X = p_X + p_Y and q_Z = p_Y / (p_X + p_Y).  This is the parametrization
used by ``independent_xz_choi``.

For p_X = p_Y the Z part dephases completely (q_Z = 1/2) and the channel reduces to the binary
symmetric channel with crossover q_X; that is the noisy branch of the superactivation channel,
see ``superactivation_channel``.
"""
import numpy as np

from examples.channels._choi_decomposition import decompose_choi_tensor_product, print_decomposition_stats

__all__ = [
    "pauli_choi",
    "independent_xz_choi",
    "block_decompose_pauli_tensor_power",
]

_PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def pauli_choi(p_I: float, p_X: float, p_Y: float, p_Z: float) -> np.ndarray:
    """Return the unnormalized Choi matrix of the mixed Pauli channel (p_I, p_X, p_Y, p_Z).

    Uses the convention Tr_B(J) = I_in, so that J = sum_ij |i><j| (x) P(|i><j|).

    Args:
        p_I, p_X, p_Y, p_Z: Probabilities of the four Pauli operators; must sum to 1.

    Returns:
        4x4 complex128 Choi matrix with row/col ordering (A*2 + B).
    """
    probabilities = np.array([p_I, p_X, p_Y, p_Z], dtype=float)
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(f"Pauli probabilities must sum to 1, got {probabilities.sum()}")

    J = np.zeros((4, 4), dtype=complex)
    for prob, sigma in zip(probabilities, _PAULI):
        vec = sigma.T.reshape(-1)  # vec(sigma) in the (in, out) ordering used for Choi matrices
        J += prob * np.outer(vec, vec.conj())
    return J


def independent_xz_choi(p_X: float, p_Y: float) -> np.ndarray:
    """Return the unnormalized Choi matrix of the independent X-Z Pauli channel.

    The channel is fixed by its X and Y weights: independence of the X and Z flips means
    p_I p_Y = p_X p_Z, so the flip probabilities are q_X = p_X + p_Y and q_Z = p_Y / q_X, and
    the remaining weights follow as p_I = (1-q_X)(1-q_Z) and p_Z = (1-q_X) q_Z.

    Args:
        p_X: Weight of the X Pauli.
        p_Y: Weight of the Y Pauli.  Equal weights give q_Z = 1/2, complete dephasing.
    """
    q_X = p_X + p_Y
    if q_X <= 0:
        raise ValueError("p_X + p_Y must be positive; with no X or Y weight the independent "
                         "X-Z channel is not determined by (p_X, p_Y).")
    q_Z = p_Y / q_X
    return pauli_choi((1 - q_X) * (1 - q_Z), p_X, p_Y, (1 - q_X) * q_Z)


def block_decompose_pauli_tensor_power(n: int, p_X: float, p_Y: float):
    """Decompose the normalized Choi matrix of the independent X-Z channel to the n-th tensor
    power into its components between the irrep blocks of End^(S_n)(C^2)."""
    return decompose_choi_tensor_product(independent_xz_choi(p_X, p_Y) / 2, d_in=2, d_out=2, n=n)


if __name__ == "__main__":
    n, p = 5, 1.0 / (1.0 + 2.0**0.5)
    blocks = block_decompose_pauli_tensor_power(n=n, p_X=p / 2, p_Y=p / 2)
    print(f"Normalized Choi parts of the independent X-Z channel for {n=}, p_X = p_Y = {p / 2}:")
    print_decomposition_stats(blocks, d_in=2, d_out=2)
