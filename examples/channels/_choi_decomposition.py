import numpy as np

from permqit.algebra import EndSnOrbitBasis
from permqit.representation import SSYT
from permqit.representation.partial_traces import DecomposedChoiBlocks, PartialTraceRelations, block_decompose_choi_matrix
from permqit.utilities import is_psd
from permqit.utilities.numpy_utils import ArrayAPICompatible


def decompose_choi_tensor_product(single_copy_choi_matrix: ArrayAPICompatible, d_in: int, d_out: int, n: int) -> DecomposedChoiBlocks:
    """Decompose a Choi matrix into its components between the irrep blocks of End^(S_n)(C^d), using block_decompose_choi_matrix.

    Args:
        single_copy_choi_matrix: Choi matrix of a single copy of the channel, shape (d_in * d_out, d_in * d_out).
        d_in: Input dimension of the channel.
        d_out: Output dimension of the channel.
        n: Number of tensor copies.
    """
    basisA = EndSnOrbitBasis(n, d_in)
    basisB = EndSnOrbitBasis(n, d_out)
    basisAB = EndSnOrbitBasis(n, d_in * d_out)
    relations = PartialTraceRelations(basisA, basisB, basisAB)

    choi_coeffs = basisAB.coefficients_for_tensor_product(single_copy_choi_matrix)
    return block_decompose_choi_matrix(relations, choi_coeffs)


def print_decomposition_stats(blocks: DecomposedChoiBlocks, d_in: int, d_out: int):
    for (label_A, label_B), choi_block in sorted(blocks.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        m_A = SSYT.count(label_A[0], d_in)
        m_B = SSYT.count(label_B[0], d_out)
        assert choi_block.shape == (m_A * m_B, m_A * m_B)

        is_cp = is_psd(choi_block)
        partial_trace_out = np.einsum("aoAO->aA", choi_block.reshape(m_A, m_B, m_A, m_B))
        is_trace_non_increasing = is_psd(np.eye(m_A) - partial_trace_out)

        print(
            f"Block for: A={label_A[0]!s:<10} B={label_B[0]!s:<10} sub-channel-dim={m_A}->{m_B:<3} "
            f"CP={is_cp!s:<5} trace-non-increasing={is_trace_non_increasing}: "
        )
        block_str = np.array2string(np.real_if_close(choi_block), precision=4, suppress_small=True, max_line_width=1000, threshold=1e4)
        print(block_str)
