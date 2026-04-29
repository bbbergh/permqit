import math
from collections import defaultdict
from ...algebra import EndSnOrbitBasisSubset, MatrixDirectSumBasis, MatrixTensorProductBasis, EndSnIrrepBasis
import numpy as np
import picos as pc
from typing import List, Sequence, cast, Collection
from ...representation.partition import Partition
from ...representation.isomorphism import EndSnOrbitBasis
from .masked import get_masked_variable


def choi_partial_trace_constraints(
    blocks: Sequence[pc.expressions.ComplexAffineExpression],
    block_diagonal_basis: MatrixDirectSumBasis,
    problem: pc.Problem,
    input_system_indices: Collection[int] = (0,)
):
    """
    Given a block-diagonal decomposition (corresponding to block_diagonal_basis) of a picos variable,
    adds the partial trace constraints to make this variable a choi matrix of a CPTP channel.
    It is assumed that every block has itself a tensor product structure with the same number of tensor product factors
    for every block, and the tensor product factors at ``input_system_indices`` are considered input systems, everything else are output systems.

    The constraints turn out to work out in such a way that this is equivalent to constraining this to be a choi matix of a CPU map,
    but with input and output systems exchanged.
    """

    aggregated: dict[tuple[Partition, ...], pc.expressions.ComplexAffineExpression] = defaultdict(lambda: pc.Constant(0))
    assert all(isinstance(b, MatrixTensorProductBasis) for b in block_diagonal_basis.bases)
    output_system_indices = set(range(len(block_diagonal_basis.bases[0].bases))) - set(input_system_indices)  # ty:ignore[unresolved-attribute]

    for block, basis_block in zip(blocks, block_diagonal_basis.bases):
        # Aggregate traced out systems by partition corresponding to the remaining systems
        basis_block = cast(MatrixTensorProductBasis, basis_block)
        traced = pc.partial_trace(block, subsystems=output_system_indices, dimensions=[b.dimension for b in basis_block.bases])  # type: ignore[arg-type]
        partitions = tuple(cast(EndSnIrrepBasis, basis_block.bases[input_idx]).partition for input_idx in input_system_indices)

        # This is the multiplicity with which these parts of the block-diagonal decomposition are present in a unitary block-diagonalization
        aggregated[partitions] += traced * math.prod(p.count_standard_tableaux() for p in partitions)

    """
    To see why this aggregation structure is the right thing to do, consider the following examples, where
    R = (dxd)^1, S = (d_Sxd_S)^n

    For Schrodinger picture (trace-preserving): Tr_S(C_{RS}) = I_R
        => Σ_λ Tr_S(C^λ_{RS}) = I_R (sum over blocks)

    For Heisenberg picture (unital): Tr_R(C_{RS}) = I_S
        => Tr_R(C^λ_{RS}) = I_{m_λ} for each block

    """

    for partitions, expr in aggregated.items():
        problem.add_constraint(expr == np.eye(expr.shape[0]))


def choi_variable(
            problem: pc.Problem,
            bases: Sequence[EndSnOrbitBasis|EndSnOrbitBasisSubset],
            input_system_indices: Collection[int] = (0,),
    ) -> List[pc.expressions.ComplexAffineExpression]:
    """
    Given bases [B_i] and input_system_indices = [i_1, ..., i_k], creates a picos variable with constraints to be a valid choi matrix
    of a channel B_i_1 ⊗ ... ⊗ B_i_k -> B_j_1 ⊗ ...
    where the indices j_1, ..., on the rhs are all indices not present in input_system_indices.
    Parameters
    ----------
    problem The picos problem to add the constraints to

    Returns
    -------

    """
    blocks, basis = get_masked_variable(bases, name="C_RS")

    for block in blocks:
        problem.add_constraint(block >> 0)

    # 2. Apply Trace-Preserving / Unitary constraints based on Picture
    choi_partial_trace_constraints(blocks, basis, problem, input_system_indices)

    return blocks


