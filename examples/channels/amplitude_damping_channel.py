"""Amplitude Damping Channel (ADC) utilities.

The qubit amplitude damping channel (N=0 bath) with damping parameter γ ∈ [0, 1]:

    A_γ(ρ) = K_0 ρ K_0† + K_1 ρ K_1†

    K_0 = [[1,       0      ],    K_1 = [[0,    sqrt(γ)],
            [0, sqrt(1-γ)  ]]             [0,    0      ]]

For γ=0: identity channel. For γ=1: maps everything to |0><0|.

Choi matrix (standard convention, Tr_B(J) = I_in):

    J = [[1,         0, 0, sqrt(1-γ)],
         [0,         0, 0, 0        ],
         [0,         0, γ, 0        ],
         [sqrt(1-γ), 0, 0, 1-γ     ]]

Row/col ordering: (R=0,B=0)=0, (R=0,B=1)=1, (R=1,B=0)=2, (R=1,B=1)=3.
"""
import numpy as np


__all__ = [
    "amplitude_damping_choi",
]

from examples.channels._choi_decomposition import decompose_choi_tensor_product, print_decomposition_stats


def amplitude_damping_choi(gamma: float) -> np.ndarray:
    """Return the unnormalized Choi matrix of the amplitude damping channel.

    Uses the convention Tr_B(J) = I_in (consistent with J_identity = Σ_{ij} |i><j|⊗N(|i><j|)).

    At γ=0 this equals depolarizing_choi(p=0), the identity channel Choi.

    Args:
        gamma: Damping parameter γ ∈ [0, 1]. γ=0: identity. γ=1: maps to |0><0|.

    Returns:
        4×4 complex128 Choi matrix with row/col ordering (R*2+B).
    """
    J = np.zeros((4, 4), dtype=complex)
    J[0, 0] = 1.0
    J[0, 3] = np.sqrt(1.0 - gamma)
    J[3, 0] = np.sqrt(1.0 - gamma)
    J[2, 2] = gamma
    J[3, 3] = 1.0 - gamma
    return J


def block_decompose_amplitude_damping_tensor_power(n: int, gamma: float):
    """Decompose the normalized Choi matrix of N_gamma^{⊗n} (n tensor copies of the amplitude damping channel) into its
    components between the irrep blocks of End^(S_n)(C^2), using block_decompose_choi_matrix.

    Args:
        n: Number of tensor copies.
        gamma: Damping parameter γ ∈ [0, 1].
    """
    return decompose_choi_tensor_product(amplitude_damping_choi(gamma)/2, d_in=2, d_out=2, n=n)



if __name__ == "__main__":
    d = 2
    print_decomposition_stats(
        block_decompose_amplitude_damping_tensor_power(n=8, gamma=0.4), d_in=d, d_out=d
    )
