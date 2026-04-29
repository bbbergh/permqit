from array_api_compat import array_namespace
import sparse

import math

from ...algebra import MatrixTensorProductBasis, MatrixDirectSumBasis
from ...utilities.numpy_utils import multi_vector_kron
import picos as pc
from typing import Sequence, cast
from ...algebra.endomorphism_basis import EndSnOrbitBasis, EndSnOrbitBasisSubset, EndSnIrrepBasis
from ...representation.isomorphism import (
    EndSnAlgebraIsomorphism,
    tensor_product_block_diagonalization_basis,
    EndSnBlockDiagonalization,
    tensor_product_block_diagonalization,
)


def get_masked_variable(
    bases: Sequence[EndSnOrbitBasis | EndSnOrbitBasisSubset], name: str, hermitian: bool = True
) -> tuple[list[pc.expressions.ComplexAffineExpression], MatrixDirectSumBasis]:
    """
    Creates a list or picos Hermitian variables that live in the block-diagonalization of the tensor product of the given bases
    Parameters
    ----------
    bases
    name
    hermitian

    Returns
    -------
    [blocks, block_diagonal_basis]: 1. blocks: A list of picos expressions corresponding to the different blocks
    2. A single MatrixDirectSumBasis whose elements correspond to the individual blocks.
    """
    isos = [EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(b.n, b.d)) for b in bases]
    diagonal_basis = tensor_product_block_diagonalization_basis(isos)

    if not any(isinstance(b, EndSnOrbitBasisSubset) for b in bases):
        # If nothing is a subset just create a bunch of matrix variables for each block
        return [
            (pc.HermitianVariable if hermitian else pc.ComplexVariable)(f"{name}_block_{i}", (size, size))
            for i, size in enumerate(diagonal_basis.block_sizes)
        ], diagonal_basis
    else:
        # The idea here is to create a list of transition matrices [(num_params) -> block_matrix] which we then dot with parameter vector
        if not hermitian:
            variable = pc.ComplexVariable(name, math.prod([b.size() for b in bases]))

            # This is a matrix (num_params, total_basis_size) that expands our subset into the full basis
            expansion_matrix = multi_vector_kron(
                array_namespace(sparse),
                [
                    sparse.eye(basis.size())
                    if isinstance(basis, EndSnOrbitBasis)
                    else basis.index_mapping.as_transition_matrix().to_pydata_sparse()
                    for basis in bases
                ],
            ).transpose(1, 0)
        else:
            # If we only want hermitian matrices, we need an additional step
            hermitian_embeddings = [b.hermitian_subset_embedding for b in bases]
            num_params = math.prod(s.to_scipy_sparse().shape[-1] for s in hermitian_embeddings) # Careful here: basis_from.size() is not accurate for hermitian_embeddings
            variable = pc.ComplexVariable(name, num_params)

            expansion_matrix = multi_vector_kron(array_namespace(sparse), hermitian_embeddings)

        block_expansion_matrices = [
            sparse.to_scipy(m.transpose(1, 0)) for m in tensor_product_block_diagonalization(expansion_matrix, isos)
        ]  # These are of the form [(block_size**2, num_params) for block]

        multiplicities = [
            math.prod(
                cast(EndSnIrrepBasis, b).partition.count_standard_tableaux()
                for b in cast(MatrixTensorProductBasis, basis_block).bases
            )
            for basis_block in diagonal_basis.bases
        ]

        block_variables = [(pc.Constant(b * mult) * variable).reshaped(size, size) for b, mult, size in zip(block_expansion_matrices, multiplicities, diagonal_basis.block_sizes)]
        if hermitian:
            block_variables = [v.hermitianized for v in block_variables]
        return block_variables, diagonal_basis
