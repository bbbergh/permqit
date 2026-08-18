"""
Depolarizing channel utilities.

The qubit depolarizing channel is:
    N_p(ρ) = (1 - p) ρ + p · I/2

Its (normalized) Choi matrix equals the isotropic state with mixing parameter (1-p):
    N_choi_normalized = isotropic_state(1 - p, d=2, fidelity=False)

The seesaw expects the UNNORMALIZED Choi (Tr_out = d_in · I_in):
    N_choi = d · isotropic_state(1 - p, d=2, fidelity=False)
"""
from typing import cast

import numpy as np

from examples.channels._print_decomposition import print_decomposition_stats

from channels._choi_decomposition import decompose_choi_tensor_product
from examples.states.isotropic import isotropic_state
from permqit.algebra import EndSnOrbitBasis
from permqit.representation.partial_traces import PartialTraceRelations, block_decompose_choi_matrix

__all__ = [
    "depolarizing_choi",
    "block_decompose_depolarizing_tensor_power",
]

def depolarizing_choi(p: float, d: int = 2) -> np.ndarray:
    """Return the unnormalized Choi matrix of the d-dimensional depolarizing channel.

    The depolarizing channel is:
        N_p(ρ) = (1 - p) ρ + p · I/d

    The UNNORMALIZED Choi matrix satisfies Tr_out(N_choi) = d · I_in and equals:
        N_choi = d · isotropic_state(1 - p, d, fidelity=False)
             = d · [(1-p) · |Φ+><Φ+| + p · I/d²]

    This is the convention expected by the seesaw optimization functions.

    Args:
        p: Depolarizing parameter, p ∈ [0, 1].
           p=0: identity channel; p=1: fully depolarizing (N(ρ) = I/d).
        d: Qudit dimension (default: 2 for qubits).

    Returns:
        Unnormalized Choi matrix, shape (d², d²), dtype complex128.
    """
    return d * isotropic_state(1.0 - p, d, fidelity=False)


def block_decompose_depolarizing_tensor_power(n: int, p: float, d: int = 2):
    """Decompose the normalized Choi matrix of N_p^{⊗n} (n tensor copies of the depolarizing channel) into its
    components between the irrep blocks of End^(S_n)(C^d), using block_decompose_choi_matrix.

    Args:
        n: Number of tensor copies.
        p: Depolarizing parameter, p ∈ [0, 1].
        d: Qudit dimension (default: 2 for qubits).

    Returns:
        {(label_A, label_B): choi_block}, as returned by block_decompose_choi_matrix.
    """
    return decompose_choi_tensor_product(depolarizing_choi(p, d=d)/d, d_in=d, d_out=d, n=n)


if __name__ == "__main__":
    d = 2
    print_decomposition_stats(
        block_decompose_depolarizing_tensor_power(n=5, p=0.4, d=d), d_in=d, d_out=d
    )
