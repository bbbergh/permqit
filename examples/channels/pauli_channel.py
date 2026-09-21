"""Pauli channel utilities.

A mixed Pauli channel applies I, X, Y, Z with probabilities (p_I, p_X, p_Y, p_Z):

    P(rho) = p_I rho + p_X X rho X + p_Y Y rho Y + p_Z Z rho Z.

The independent X-Z channel applies X with probability p_x and Z with probability p_z,
independently, i.e. (p_I, p_X, p_Y, p_Z) = ((1-p_x)(1-p_z), p_x(1-p_z), p_x p_z, (1-p_x)p_z).

For p_z = 1/2 the Z part dephases completely and the channel reduces to the binary symmetric
channel with crossover p_x; this is the noisy branch of the superactivation channel, see
``superactivation_channel``.
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


def independent_xz_choi(p_x: float, p_z: float = 0.5) -> np.ndarray:
    """Return the unnormalized Choi matrix of the independent X-Z Pauli channel.

    X is applied with probability p_x and Z with probability p_z, independently of each other.

    Args:
        p_x: Probability of an X error.
        p_z: Probability of a Z error (default 1/2, i.e. complete dephasing).
    """
    return pauli_choi((1 - p_x) * (1 - p_z), p_x * (1 - p_z), p_x * p_z, (1 - p_x) * p_z)


def block_decompose_pauli_tensor_power(n: int, p_x: float, p_z: float = 0.5):
    """Decompose the normalized Choi matrix of the independent X-Z channel to the n-th tensor
    power into its components between the irrep blocks of End^(S_n)(C^2)."""
    return decompose_choi_tensor_product(independent_xz_choi(p_x, p_z) / 2, d_in=2, d_out=2, n=n)


if __name__ == "__main__":
    n, p_x = 5, 1.0 / (1.0 + 2.0**0.5)
    blocks = block_decompose_pauli_tensor_power(n=n, p_x=p_x)
    print(f"Normalized Choi parts of the independent X-Z channel for {n=}, {p_x=}:")
    print_decomposition_stats(blocks, d_in=2, d_out=2)
