from array_api_compat import array_namespace
import itertools

import math
from typing import cast, Sequence, Optional

import numpy as np
import scipy.linalg
import sparse

from . import isomorphism_gijswijt
from ..utilities.caching import WeakRefMemoize
from .isomorphism_kappa import calculate_f
from .orbits import PairOrbit
from .partition import Partition
from ..algebra.basis import TensorProductBasis, MatrixStandardBasis, MatrixTensorProductBasis
from ..algebra.endomorphism_direct_sum_basis import MatrixDirectSumBasis
from ..algebra.endomorphism_basis import EndSnBlockDiagonalBasis, EndSnOrbitBasis, EndSnIrrepBasis
from ..algebra.linear_map import TransitionMatrix, GivenTransitionMatrix, StorageFormat, CoefficientData, Identity
from .young_tableau import SSYT
from ..utilities import caching, backend
from ..utilities.timing import ExpensiveComputation


class EndSnTensorProductIsomorphism(TransitionMatrix):
    """
    Implements the embedding End^(S_n)(V_1^n) ⊗ ... ⊗ End^(S_n)(V_k^n) ⊆ End^(S_n)((V_1 ⊗ ... ⊗ V_k)^n)
    This is not actually an isomorphism, but just a homomorphism (it is an isomorphism onto it's image).
    The basis on the left is a tensor product of two EndSnOrbitBases, and the basis on the right is a single EndSnOrbitBasis on the tensor product space.

    Be careful that the ordering of tensor product factors is different when (numerically) evaluating the basis elements.
    The LHS returns matrices of system order V_1^1...V_1^n ... V_k^1... V_k^n, while the RHS returns
    V_1^1...V_k^1 ... V_1^n ... V_k^n.
    This only matters when explicitly constructing the matrices (such as for tests).
    """

    basis_from: TensorProductBasis # A tensor product of multiple EndSnOrbitBases
    basis_to: EndSnOrbitBasis # A single EndSnOrbitBasis on the tensor product space

    def __init__(self, *bases: EndSnOrbitBasis):
        self.basis_from = TensorProductBasis(bases=bases)
        assert len(set(b.n for b in bases)) == 1, f"The n in EndSnTensorProductIsomorphism must be the same for all bases, but got {bases}"
        self.basis_to = EndSnOrbitBasis(n=bases[0].n, d=math.prod(basis.d for basis in bases))

    def __str__(self):
        return f"{type(self).__name__}(n={self.basis_to.n}, {'x'.join(str(cast(EndSnOrbitBasis, basis).d) for basis in self.basis_from.bases)} -> {self.basis_to.d})"

    def _calculate_transition_matrix(self):
        # noinspection PyAbstractClass
        matrix = sparse.DOK((self.basis_to.size(), self.basis_from.size()), dtype=np.int_)
        bases = cast(Sequence[EndSnOrbitBasis], self.basis_from.bases)
        for i, o_big in enumerate(self.basis_to.iterate_labels()): # Iterate over all the orbits in End^(S_n)((V_1 ⊗ ... ⊗ V_k)^n)
            # It is not hard to see that this orbit has overlap with only one specific tensor product of orbits, which can be calculated from the count matrix.
            # This is the orbit where the counts 'match up', i.e. we have the right count of each symbol.
            # Note that this is not one-to-one, each of these tensor products of orbits has overlap with multiple orbits in End^(S_n)((V_1 ⊗ ... ⊗ V_k)^n)

            # To get the counts of each individual tensor product factor we transpose the count matrix from axis order
            # ((i_1...i_k), (j_1, ..., j_k)), to ((i_1 j_1), ..., (i_k j_k)). Then, summing over all but one axis, gives the count distribution
            # of a pair (i_k j_k), which we reshape into a count matrix of an orbit in End^(S_n)(V_k^n).
            k = len(self.basis_from.bases)
            transposed = (o_big.count_matrix
                          .reshape([b.d for b in bases] *2) # reshape to (i_1,...i_k, j_1, ..., j_k)
                          .transpose(*(j for i in range(k) for j in (i, i+k))) # transpose to (i_1 j_1, ..., i_k j_k)
                          .reshape([b.d**2 for b in bases])) # reshape to ((i_1, j_1), ..., (i_k, j_k))
            sums = [transposed.sum(axis=tuple(range(i)) + tuple(range(i+1, k))) for i in range(k)]  # Sum over all but one axis to get the counts for each factor

            matrix[i, self.basis_from.label_to_index(tuple(PairOrbit(sum.reshape((b.d, b.d))) for sum, b in zip(sums, bases)))] = 1

        return matrix.asformat('gcxs')



class BaseEndSnBlockDiagonalization(TransitionMatrix, metaclass=WeakRefMemoize):
    """
    Abstract base class for block diagonalization isomorphisms of the space of permutation invariant matrices.
    Implements the block diagonalization of the matrix group End^(S_n)(V^n), i.e. the isomorphism that maps the Endomorphism space into the direct sum of blocks on the irreps.
    Takes coefficients of permutation invariant matrix in the EndSnOrbit basis, and returns the corresponding coefficients in the EndSnBlockDiagonalBasis.
    See below for different implementations.
    """
    basis_from: EndSnOrbitBasis
    basis_to:  EndSnBlockDiagonalBasis

    default_cache_formats = {StorageFormat.PYDATA_SPARSE, StorageFormat.GPU_SPARSE}

    def __init__(self, n, d):
        self.n = n
        self.d = d

        self.basis_from = EndSnOrbitBasis(n, d)
        self.basis_to = EndSnBlockDiagonalBasis(n, d)
        self._dense_matrix_cache = None

    @classmethod
    def __intercept_new__(cls, n, d):
        if n <= 1:
            return TrivialAlgebraIsomorphism(d=d, n = n)
        return None


    def get_block_transition_matrix(self, partition: Partition) -> np.ndarray:
        """
        Returns the transition matrix for the block corresponding to the given partition, i.e.
        returns an array of shape (self.basis_from.size(), m_λ, m_λ), where m_λ is the block size corresponding to the given partition.
        The isomorphism then maps the basis element i to block result[i] (which is an m_λ x m_λ matrix).
        """
        mat = self.coefficient_transition_matrix(StorageFormat.PYDATA_SPARSE)
        partition_idx = self.basis_to.partitions.index(partition)
        idx_in_basis = self.basis_to.basis_indices_start[partition_idx] # This is the start index of the coefficients corresponding to the block we want
        block_size = self.basis_to.basis_sizes[partition_idx]

        block_dense = np.asarray(mat[idx_in_basis : idx_in_basis + block_size, :].todense())  # ty:ignore[not-subscriptable]

        return block_dense.reshape(
            (self.basis_to.block_sizes[partition_idx], self.basis_to.block_sizes[partition_idx], self.basis_from.size())
        )

    def __str__(self):
        return f"{type(self).__name__}({self.n=}, {self.d=})"



class EndSnBlockDiagonalizationKappa(BaseEndSnBlockDiagonalization):
    """Uses the kappa based construction from DOI:10.1007/s10623-016-0216-5"""

    def _calculate_transition_matrix(self):
        size = self.basis_to.size()
        assert size == self.basis_from.size()
        # noinspection PyAbstractClass
        matrix = sparse.DOK((size, size), dtype=np.int_)

        # Iterate over all pairs of SSYT
        for loop in enumerate(self.basis_to.iterate_labels()):
            idx_to, (t1, t2) = loop  # Needed for stupid type checker reasons
            assert t1.partition == t2.partition

            poly = calculate_f(t1.partition, t1, t2, self.d) # Calculate the polynomial
            for monomial, coeff in poly.coeffs.items():
                orbit = PairOrbit.from_monomial(monomial, self.d)
                idx_from = self.basis_from.label_to_index(orbit)
                matrix[idx_to, idx_from] = coeff

        return matrix.asformat('gcxs')

class EndSnBlockDiagonalizationGijswijt(BaseEndSnBlockDiagonalization):
    """Uses the construction from https://arxiv.org/abs/0910.4515v1

    The polynomial coefficients are computed with floating-point arithmetic (divisions by
    1/(k+1) in tableaux_polynomial) and must be rounded to the nearest integer before
    storage.  Without rounding, a coefficient such as 59.9999... truncates to 59 in the
    integer sparse matrix, breaking the adjoint-preservation property for n >= 7 (d=2).
    With round() the result is exact and identical to EndSnBlockDiagonalizationKappa.
    """

    def _calculate_transition_matrix(self):
        size = self.basis_to.size()
        assert size == self.basis_from.size()
        # noinspection PyAbstractClass
        matrix = sparse.DOK((size, size), dtype=np.int_)

        # Iterate over all pairs of SSYT
        for idx_to, (t1, t2) in enumerate(self.basis_to.iterate_labels()):
            assert t1.partition == t2.partition

            poly = isomorphism_gijswijt.BlockDiagonalization.tableaux_polynomial(t2, t1)  # Calculate the polynomial
            for monomial, coeff in poly.coeffs.items():
                orbit = PairOrbit.from_monomial(monomial, self.d)
                idx_from = self.basis_from.label_to_index(orbit)
                # round() is required: tableaux_polynomial uses floating-point 1/(k+1)
                # factors that can produce e.g. 59.9999... instead of 60 for n>=7.
                matrix[idx_to, idx_from] = round(coeff)

        return matrix.asformat('gcxs')


EndSnBlockDiagonalization = EndSnBlockDiagonalizationGijswijt  # faster than Kappa

class EndSnAlgebraIsomorphism(BaseEndSnBlockDiagonalization):
    """
    The previous block-diagonalizations are not actually Algebra Isomorphisms, i.e. they don't preserve matrix products,
    this is because the basis-vectors indexed by SSYTs are not orthonormal. To turn these constructions into algebra isomorphisms
    we have to sandwich with a square root of the gram matrix, which is what this implementation does.
    This takes a block diagonalization, calculates the gram matrix by computing the image of the identity matrix, and then
    returns a new isomorphism that is a proper *-(matrix)-algebra isomorphism.
    """

    block_diagonalization: BaseEndSnBlockDiagonalization


    def __init__(self, block_diagonalization: BaseEndSnBlockDiagonalization):
        self.block_diagonalization = block_diagonalization
        super().__init__(block_diagonalization.n, block_diagonalization.d)

    @classmethod
    def __intercept_new__(cls, block_diagonalization: BaseEndSnBlockDiagonalization):  # ty:ignore[invalid-method-override]
        if isinstance(block_diagonalization, EndSnAlgebraIsomorphism):
            return block_diagonalization
        return None

    def _calculate_transition_matrix(self):
        # Will use GPU if available
        xp = backend.xp

        new_transition_elements = []

        identity_coeffs = xp.asarray(self.basis_from.coefficients_of_identity())
        
        for idx, part in enumerate(self.basis_to.partitions):
            transition = self.block_diagonalization.get_block_transition_matrix(part)
            if transition.shape[0] == 0: continue
            transition = backend.to_xp(transition)
            
            gram_matrix = xp.dot(transition, identity_coeffs) # Same as transition @ identity_coeffs
            inv = xp.linalg.inv(gram_matrix)
            cholesky = xp.linalg.cholesky(inv)

            # Einsum: cholesky^T @ transition @ cholesky
            # transition is (m_λ, m_λ, basis_from.size())
            # cholesky is (m_λ, m_λ)
            # We want: adjusted[i, m, a] = sum_jk cholesky[j,i] * transition[j, k, a] * cholesky[k, m]
            adjusted = xp.einsum('ji,jka,km->ima', cholesky, transition, cholesky)

            # Convert back to CPU numpy array for sparse conversion (if we ever were on GPU)
            adjusted = backend.to_cpu(adjusted)

            new_transition_elements.append(sparse.GCXS.from_numpy(adjusted.reshape(-1, self.basis_from.size())))

        return sparse.concatenate(new_transition_elements)

    @caching.cache_noargs
    def inverse(self) -> TransitionMatrix:
        mat = self.coefficient_transition_matrix(StorageFormat.PYDATA_SPARSE)  # type: sparse.SparseArray
        with ExpensiveComputation(f"Calculating inverse of {str(self)}"):
            mat = scipy.sparse.linalg.inv(mat.asformat('coo').tocsc())
            sp = sparse.COO.from_scipy_sparse(mat).asformat('gcxs')
            return GivenTransitionMatrix(self.basis_to, self.basis_from, sp, cache_formats=self.default_cache_formats)


class TrivialAlgebraIsomorphism(EndSnAlgebraIsomorphism):
    """
    This is a shortcut for the n = 1 case, which is trivial, but still useful to have to keep the code compact and consistent when reference systems are involved
    """
    
    def __init__(self, d, *, n=1):
        BaseEndSnBlockDiagonalization.__init__(self, n=n, d=d)

    # Needed to override the previous definitions
    @classmethod
    def __intercept_new__(cls, *args, **kwargs):
        return None
        
    def _calculate_transition_matrix(self):
        return scipy.sparse.eye_array(self.basis_from.size())

    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional[StorageFormat] = None,
    ) -> T:
        return x

    @caching.cache_noargs
    def inverse(self) -> TransitionMatrix:
        return Identity(self.basis_to, self.basis_from)

def tensor_product_block_diagonalization_basis(block_diagonalizations: Sequence[BaseEndSnBlockDiagonalization]) -> MatrixDirectSumBasis[tuple[tuple[SSYT, SSYT], ...]]:
    """
    Returns the basis of the block-diagonalization of a tensor product of permutation invariant matrix spaces, given the block diagonalizations of each factor.
    The resulting basis is a direct sum of tensor products of the blocks of the individual block diagonalizations.
    """
    return MatrixDirectSumBasis(tuple(MatrixTensorProductBasis(tuple(blocks)) for blocks in itertools.product(*(bd.basis_to.bases for bd in block_diagonalizations))))

def tensor_product_block_diagonalization[T: CoefficientData](
        orbit_basis_coefficients: T,
        block_diagonalizations: Sequence[BaseEndSnBlockDiagonalization],
) -> list[T]:
    """
    Takes a flat array of coefficients corresponding to a tensor product of permutation invariant matrices, and returns the corresponding block-diagonal matrix, where a block-diagonalization is applied to each tensor factor,
    and subsequentely multiplied out, so there is a single sum over blocks remaining. Returns this single direct sum as a list of blocks.
    Parameters
    ----------
    orbit_basis_coefficients: Coefficient array, whose last dimension specifies an element of End^(S_(μ_1))(A_1^(μ_1)) ⊗ End^(S_(μ_2)}(A_2^(μ_2)) ⊗ ... by being its coefficients in the TensorProductBasis(EndSnOrbitBasis(μ_1, d_A_1), ..., EndSnOrbitBasis(μ_k, d_A_k))
    block_diagonalizations: A list of block diagonalizations: [EndSnBlockDiagonalization(μ_1, d_A_1), EndSnBlockDiagonalization(μ_2, d_A_2), ...]

    Returns
    -------
    A block-diagonalization of the input element's specified axis, i.e. an element of \bigoplus_{i_1}^{t_1} ... \bigoplus_{i_n}^{t_n} ℂ^{(m_i_1 * ... m_i_n) x (m_i_1 * ... m_i_n)} returned as a flat list of matrices, with respective shape (*other_axes, m_i_1 * ... m_i_n, m_i_1 * ... m_i_n),
    i.e. the list is of size t_1*...*t_n.
    """
    orig_shape = orbit_basis_coefficients.shape
    assert orig_shape[-1] == math.prod(bd.basis_from.size() for bd in block_diagonalizations)
    other_shape = orig_shape[:-1]
    num_other_indices = len(other_shape)
    orbit_basis_coefficients = orbit_basis_coefficients.reshape(other_shape + tuple(bd.basis_from.size() for bd in block_diagonalizations))
    
    block_diagonal_coefficients = orbit_basis_coefficients
    for i, bd in enumerate(block_diagonalizations):
        block_diagonal_coefficients =  bd.apply_to_coefficient_vector(block_diagonal_coefficients, axis=num_other_indices + i)
        
    # Now we have to extract the actual blocks from the block-diagonal coefficients
    block_diagonal_basis = tensor_product_block_diagonalization_basis(block_diagonalizations)
    block_splits = itertools.product(*([slice(start, start+size) for start, size in zip(bd.basis_to.basis_indices_start, bd.basis_to.basis_sizes)] for bd in block_diagonalizations))
    split = [block_diagonal_coefficients[...,*slices] for slices in block_splits]
    reshaped = [spl.reshape(other_shape + tuple(b.dimension for b in cast(MatrixTensorProductBasis, blockb).bases for _ in range(2))) for spl, blockb in zip(split, block_diagonal_basis.bases)]
    num_factors = len(block_diagonalizations)
    permuted_rows_first = [spl.transpose(tuple(range(num_other_indices)) + tuple(range(num_other_indices, num_other_indices + 2*num_factors, 2)) + tuple(range(num_other_indices + 1, num_other_indices + 2* num_factors, 2))) for spl in reshaped]
    return [spl.reshape(other_shape + (block_size, block_size)) for spl, block_size in zip(permuted_rows_first, block_diagonal_basis.block_sizes)]


def tensor_product_inverse_block_diagonalization[T: CoefficientData](
        block_matrices: list[T],
        block_diagonalizations: Sequence[EndSnAlgebraIsomorphism],
) -> T:
    """
    The inverse of ``tensor_product_block_diagonalization``, takes a list of blocks and returns the coefficients in the TensorProductBasis(EndSnOrbitBasis(μ_1, d_A_1), ..., EndSnOrbitBasis(μ_k, d_A_k)) as a flat coefficient vector.
    """
    block_diagonal_basis = tensor_product_block_diagonalization_basis(block_diagonalizations)
    assert len(set(b.shape[:-2] for b in block_matrices)) == 1, "block matrices have incompatible shapes"
    other_shape = block_matrices[0].shape[:-2]
    num_other_indices = len(other_shape)
    assert all(b.shape[-2:] == (block_size, block_size) for b, block_size in zip(block_matrices, block_diagonal_basis.block_sizes)), "block matrices have incompatible shapes given block_diagonalizations"
    reshaped = [spl.reshape(other_shape + (tuple(b.dimension for b in cast(MatrixTensorProductBasis, blockb).bases)*2)) for spl, blockb in zip(block_matrices, block_diagonal_basis.bases)]
    permuted_by_systems = [
        spl.transpose(
            other_shape
            + tuple(t + o for t in range(len(block_diagonalizations)) for o in range(0, 2*len(block_diagonalizations), len(block_diagonalizations)))
        ).reshape(other_shape + tuple(b.size() for b in cast(MatrixTensorProductBasis, blockb).bases))
        for spl, blockb in zip(reshaped, block_diagonal_basis.bases)]

    xp = array_namespace(block_matrices[0])
    # TODO: find a way to determine the smallest necessary dtype here
    joined = xp.empty(other_shape + tuple(bd.basis_to.size() for bd in block_diagonalizations), dtype=xp.complex128)
    block_splits = itertools.product(*([slice(start, start+size) for start, size in zip(bd.basis_to.basis_indices_start, bd.basis_to.basis_sizes)] for bd in block_diagonalizations))
    for slices, permuted in zip(block_splits, permuted_by_systems):
        joined[...,*slices] = permuted

    orbit_basis_coefficients = joined
    for i, bd in enumerate(block_diagonalizations):
        orbit_basis_coefficients = bd.inverse().apply_to_coefficient_vector(orbit_basis_coefficients, axis=num_other_indices + i)
    return orbit_basis_coefficients.reshape(other_shape + (-1,))

