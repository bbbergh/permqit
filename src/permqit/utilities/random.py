import math
from typing import Collection, cast

import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm

from ..algebra import EndSnIrrepBasis, MatrixTensorProductBasis, EndSnOrbitBasis
from ..representation.isomorphism import tensor_product_inverse_block_diagonalization, tensor_product_block_diagonalization_basis, EndSnAlgebraIsomorphism, EndSnBlockDiagonalization
from ..utilities.numpy_utils import ArrayAPICompatible
from .general_functions import choi_representation, partial_trace


def random_density_matrix(d: int) -> np.ndarray:
    """
    Generate a random density matrix of dimension d.
    Uses the Ginibre ensemble: rho = X @ X^dagger / Tr(X @ X^dagger).
    """
    X = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def random_state_vector(dims) -> np.ndarray:
    """
    Generate a random pure state vector.
    dims can be an int or tuple of ints.
    """
    if isinstance(dims, int):
        d = dims
    else:
        d = np.prod(dims)
    psi = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_channel(d_in, d_out, TP: bool = True, U: bool = False):
    """Return the Choi matrix of a random channel d_in -> d_out.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
        symmetry: Symmetry constraint passed to random_state ('none', etc.).
        TP: If True, normalize to be trace-preserving (Tr_out = I_in).
        U: If True, also normalize to be unital (Tr_in = I_out).

    Returns:
        Unnormalized Choi matrix of shape (d_in*d_out, d_in*d_out), dtype complex128.
    """
    if TP and U:
        raise ValueError("Sampling from the intersection of TP and U channels not supported yet.")
        # Doing this is non-trivial, already classically, see e.g. https://arxiv.org/pdf/0711.3345 and
        # also https://arxiv.org/pdf/0709.0824 for a quantum version

    # This code is incorrect if TP and U are set at the same time (the second constraint destroys the first one)
    C_AB = random_density_matrix(d_in*d_out)
    if TP:
        C_A = partial_trace(C_AB, (d_in, d_out), 1)
        C_A_inv_sq = np.kron(inv(sqrtm(C_A)), np.eye(d_out))
        C_AB = C_A_inv_sq @ C_AB @ C_A_inv_sq
    if U:
        C_B = partial_trace(C_AB, (d_in, d_out), 0)
        C_B_inv_sq = np.kron(np.eye(d_in), inv(sqrtm(C_B)))
        C_AB = C_B_inv_sq @ C_AB @ C_B_inv_sq
    return C_AB


def random_isometry(d_in, d_out, TP: bool = True, U: bool = False):
    if d_out == 0:
        return np.asarray([])
    if d_out == 1:
        return np.eye(d_in)
    # assert TP, "Isometries are always trace preserving" # This currently breaks some code
    if U:
        # TODO: This is not correct. Isometries make sense also in non-product dimensions, but even if this is more complicated,
        #  this should raise instead of just not returning an isometry.
        # Homomorphic decoder seed: D†(x) = u(x ⊗ I_E)u* with d_E = d_out // d_in.
        # Requires d_in | d_out; fall back to general unital channel otherwise.
        if d_out % d_in != 0:
            return random_channel(d_in, d_out, TP=False, U=True)
        d_E = d_out // d_in
        X = (np.random.randn(d_out, d_out) + 1j * np.random.randn(d_out, d_out)) / np.sqrt(2)
        u, _ = np.linalg.qr(X)
        # Choi block (r,r'): U_r @ U_{r'}† where U_r = u[:, r*d_E:(r+1)*d_E]
        choi = np.zeros((d_in * d_out, d_in * d_out), dtype=np.complex128)
        for r in range(d_in):
            U_r = u[:, r * d_E:(r + 1) * d_E]
            for rp in range(d_in):
                U_rp = u[:, rp * d_E:(rp + 1) * d_E]
                choi[r * d_out:(r + 1) * d_out, rp * d_out:(rp + 1) * d_out] = U_r @ U_rp.conj().T
        return choi
    else:
        X = (np.random.randn(d_out, d_in) + 1j * np.random.randn(d_out, d_in)) / np.sqrt(2)
        Q, _ = np.linalg.qr(X)
        V = Q[:, :d_in]
        return choi_representation([V], d_in).astype(np.complex128)


def random_permutation_invariant_channel(
    bases: list[EndSnOrbitBasis],
    input_system_indices: Collection[int] = (0,),
    *,
    isometry=False,
    seed: int|np.random.Generator|None=None,
    TP: bool = True,
    U: bool = False,
    xp=np,
) -> ArrayAPICompatible:
    """
    Returns a random channel End^{S_μ_(i_1)}(A_(i_1)^μ_(i_1)) ⊗ ...  -> End^{S_μ_(j_1)}(A_(j_1)^μ_(j_1)) ⊗ ...,
    where bases = [End^{S_μ_1}(A_1^μ_1) ⊗ ... ⊗ End^{S_μ_n}(A_n^μ_n)], and the (i_1, ...) are the indices specified in input_system_indices, and the
    (j_1, ...) are all the indices in range(n) not specified by input_system_indices.
    Note that this is *not* a permutation covariant channel, the permutations act separately on each system specified in the tensor product, and in particular separately on input and output systems.

    This is constructed by taking mixtures of random channels onto the different blocks of the block-diagonalization of End^{Sn}(A^n)
    and appropriately renormalising them so that the channel (seen as a channel into A^n) becomes trace-preserving.

    The key point is that every block-diagonal element of End^{Sn}(A^n) corresponds to an element in L(A^n) by first repeating each block a certain number of times,
    and then performing a change of basis (of A^n, and which is hence just a unitary operation). Thus, the partial trace constraint can be satisfied by dividing each block
    by their multiplicity in L(A^n), which is given by the dimension of the irrep corresponding to that block (which can be calculated by the number of standard tableaux for this partition,
    which is given by the hook-length formula).

    Parameters
    ----------
    bases [End^{S_μ_1}(A_1^μ_1) ⊗ ... ⊗ End^{S_μ_n}(A_n^μ_n)]
    input_system_indices (i_1, ...)
    isometry Whether the returned random channel should be sampled from the space of isametries rather than all possible channels.
    seed Seed used to initialize the random number generators
    TP Whether the channel should be trace preserving
    U Whether the channel should be unital
    xp ArrayAPI compatible module whose array type will be returned.

    Returns
    -------
    The flat coefficients of the Choi matrix of the channel in TensorProductBasis(*bases), where bases are exactly in the order as provided.
    """
    if TP and U:
        raise ValueError("Sampling from the intersection of TP and U channels not supported yet.")
        # See the comments inside the ``random_channel`` function on why this is non-trivial, if we implement it there,
        # also adding the permutation invariant case is not hard.

    input_system_indices = list(i % len(bases) for i in input_system_indices) # Need this to have stable iteration order
    output_system_indices = [i for i in range(len(bases)) if i not in input_system_indices]

    # TODO: random_channel/random_isometry used below should take a np.random.rng instead of needing this
    if seed is not None:
        if isinstance(seed, np.random.Generator):
            np.random.seed(
                seed.integers(low=0, high=np.iinfo(np.int_).max, size=(1,), dtype=np.int_)
            )
        else:
            np.random.seed(seed)


    isos = [
        EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(b.n, b.d)) for b in bases
    ]  # This does not recompute due to caching.WeakRefMemoize

    # For every block in a block-decomposition of the input space, we construct a channel going to each block of the output
    # space and then weigh them with a probability distribution to satisfy the TP constraints
    # To get the coefficient ordering right, it is easiest to iterate over the joint block-decomposition, but we still need
    # to make sure that the same probability distribution gets applied to all joint blocks corresponding to the same input

    input_isos = [iso for idx, iso in enumerate(isos) if idx in input_system_indices]
    output_isos = [iso for idx, iso in enumerate(isos) if idx not in input_system_indices]

    joint_block_basis = tensor_product_block_diagonalization_basis(isos)
    input_block_basis = tensor_product_block_diagonalization_basis(input_isos)
    output_block_basis = tensor_product_block_diagonalization_basis(output_isos)
    num_output_blocks = output_block_basis.number_of_blocks
    num_input_blocks = input_block_basis.number_of_blocks

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # This allows us to figure out which part of the probability distribution belongs to a particular block
    input_block_ordering = {
        tuple(cast(EndSnIrrepBasis, tp).partition for tp in cast(MatrixTensorProductBasis, b).bases): i
        for i, b in enumerate(input_block_basis.bases)
    }
    output_block_ordering = {
        tuple(cast(EndSnIrrepBasis, tp).partition for tp in cast(MatrixTensorProductBasis, b).bases): i
        for i, b in enumerate(output_block_basis.bases)
    }

    # For the TP/U constraints we need to multiply individual random TP/U maps on the blocks by certain numbers
    # to satisfy the constraint for the channel as a whole. For the TP case these normalize to 1 when summing over
    # output_blocks, for the U case when summing over input blocks.
    # Here we also include the multiplicity of every block. This is needed since the block-diagonalization is not
    # unitary. The block-diagonalization could be made unitary if in the final block-decomposition every block
    # occurs a certain number of times (as identical copies). This multiplicity we need to multiply with to get the
    # right partial-trace constraint.

    if TP:
        prob_matrix = rng.dirichlet(np.ones(num_output_blocks), (num_input_blocks,))

        # For the TP constraint we trace over output systems of the choi matrix
        for output_partitions, idx in output_block_ordering.items():
            prob_matrix[:, idx] /= math.prod(p.count_standard_tableaux() for p in output_partitions)
    elif U:
        prob_matrix = rng.dirichlet(np.ones(num_input_blocks), (num_output_blocks,)).T

        # For the U constraint we trace over input systems of the choi matrix
        for input_partitions, idx in input_block_ordering.items():
            prob_matrix[idx, :] /= math.prod(p.count_standard_tableaux() for p in input_partitions)
    else:
        prob_matrix = np.ones((num_input_blocks, num_output_blocks))

    coeff_arrays = []
    for block in joint_block_basis.bases:
        block = cast(MatrixTensorProductBasis, block)
        if block.size() == 0:
            continue

        input_bases = [b for idx, b in enumerate(block.bases) if idx in input_system_indices]
        output_bases = [b for idx, b in enumerate(block.bases) if idx not in input_system_indices]
        input_dimension = math.prod(b.dimension for b in input_bases)
        output_dimension = math.prod(b.dimension for b in output_bases)

        # These create choi matrices of shape (input_dimension, output_dimension, input_dimension, output_dimension)
        if isometry:
            local_channel = random_isometry(input_dimension, output_dimension, TP=TP, U=U)
        else:
            local_channel = random_channel(input_dimension, output_dimension, TP=TP, U=U)

        # These are integer partitions corresponding to irreps of S_n associated with the blocks
        input_partitions = tuple(cast(EndSnIrrepBasis, tp).partition for idx, tp in enumerate(block.bases) if idx in input_system_indices)
        output_partitions = tuple(cast(EndSnIrrepBasis, tp).partition for idx, tp in enumerate(block.bases) if idx not in input_system_indices)

        # Multiply by the probability distribution element and divide by the multiplicity of the block in the output space
        local_channel *= prob_matrix[input_block_ordering[input_partitions]][output_block_ordering[output_partitions]]

        split_shape = tuple(i.dimension for i in input_bases) + tuple(o.dimension for o in output_bases)
        index_arrangement = input_system_indices + output_system_indices
        reshaped = local_channel.reshape(split_shape + split_shape).transpose(index_arrangement + [i + len(bases) for i in index_arrangement]).reshape(block.dimension, block.dimension)
        coeff_arrays.append(xp.asarray(reshaped))

    # This takes block matrices
    return tensor_product_inverse_block_diagonalization(coeff_arrays, isos)


def random_channel_with_permutation_invariant_output(
    d_R, iso_A: EndSnAlgebraIsomorphism, *, isometry=False, seed=None, TP: bool = True, U: bool = False, xp=np
):
    """Returns the Choi matrix of a random permutation-invariant channel R -> A^n.
    This is constructed by taking mixtures of random channels onto the different blocks of the block-diagonalization of End^{Sn}(A^n)
    and appropriately renormalising them so that the channel (seen as a channel into A^n) becomes trace-preserving.

    :return: Coefficients of the Choi Matrix of the channel R -> A^n in the TensorProductBasis([MatrixStandardBasis(d_R), EndSnOrbitBasis(n, d_A)])
    """
    return random_permutation_invariant_channel([EndSnOrbitBasis(1, d_R), iso_A.basis_from], isometry=isometry, seed=seed, TP=TP, U=U, xp=xp)
