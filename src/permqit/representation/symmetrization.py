from functools import cached_property
from typing import Sequence, cast

import numpy as np
import scipy.sparse
import sparse

from .combinatorics import multinomial_coeff
from ..algebra import EndSnOrbitBasis, MatrixTensorProductBasis, MatrixStandardBasis, EndSnOrbitBasisSubset
from .orbits import PairOrbit
from ..algebra.basis_subset import MatrixBasisSubset, MatrixStandardBasisSubset, OrbitFitsMaskPredicate
from ..algebra.linear_map import ScatterIndexMapping, MatrixCache, TransitionMatrix, \
    GivenTransitionMatrix, GivenScatterIndexMapping
from ..utilities import caching
from ..utilities.numpy_utils import sum_combinations, product_combinations


class SymmetrizationRelations(metaclass=caching.WeakRefMemoize):
    """
    Every element ρ ∈ End^{S_n}(M^n) naturally also lives in ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i), where μ is a partition of n.
    Conversely, every element of ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i) can be symmetrized (i.e. averaged over the group S_n) to yield again
    an element of End^{S_n}(M^n). This class computes and stores the relation between basis elements of these two spaces. In particular, basis elements of
    ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i), i.e. every tensor product of EndSnOrbitBasis elements, correspond to a single element of
    EndSnOrbitBasis in End^{S_n}(M^n), (which is obtained by summing the count matrices of the individual basis elements), and obtains a prefactor
    when symmetrizing which is related to the size of the original and final orbits.

    As a linear map, this class implements the ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i) -> End^{S_n}(M^n) mapping that takes every basis vector to its
    symmetrization -- this can be accessed via SymmetrizationRelations.symmetrization --, as well as the canonical embedding
    End^{S_n}(M^n) -> ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i) that takes every basis vector to a linear combination of all basis vectors
    in ⊕_{i = 1}^t End^{S_μ_i}(M^μ_i) that correspond to it -- this can be accessed via SymmetrizationRelations.embedding.
    """

    symmetric_basis: EndSnOrbitBasis|EndSnOrbitBasisSubset
    split_basis: MatrixTensorProductBasis[PairOrbit]
    partition: Sequence[int]

    def __init__(self, single_copy_basis: MatrixStandardBasis|MatrixStandardBasisSubset, partition: Sequence[int]):
        self.partition = partition
        n = sum(partition)
        if isinstance(single_copy_basis, MatrixBasisSubset):
            assert isinstance(single_copy_basis, MatrixStandardBasisSubset)
            predicate = OrbitFitsMaskPredicate(single_copy_basis.mask)
            self.symmetric_basis = EndSnOrbitBasisSubset(EndSnOrbitBasis(n, single_copy_basis.subset_of.d), predicate)
            self.split_basis = MatrixTensorProductBasis(bases=tuple(EndSnOrbitBasisSubset(EndSnOrbitBasis(k, single_copy_basis.subset_of.d), predicate) for k in partition if k))
        else:
            self.symmetric_basis = EndSnOrbitBasis(n, single_copy_basis.d)
            self.split_basis = MatrixTensorProductBasis(bases=tuple(EndSnOrbitBasis(k, single_copy_basis.d) for k in partition if k))

    @classmethod
    def __process_init_args_for_cache_key__(cls, **kwargs):
        if "partition" in kwargs:
            kwargs["partition"] = tuple(kwargs["partition"])
        return kwargs


    @cached_property
    def index_mapping(self) -> ScatterIndexMapping:
        """Returns the index mapping that maps each basis element in \bigotimes_{i = 1}^t End^{S_μ_i}(M^μ_i)
        to the corresponding symmetrized basis element in End^{S_n}(M^n)"""
        summed_count_matrices = sum_combinations([cast(EndSnOrbitBasis|EndSnOrbitBasisSubset, b).all_count_matrices for b in  self.split_basis.bases]).reshape(-1, self.symmetric_basis.d, self.symmetric_basis.d)
        arr = np.fromiter((self.symmetric_basis.count_matrix_to_index(m) for m in summed_count_matrices), dtype=np.int_, count=self.split_basis.size())
        return GivenScatterIndexMapping(MatrixCache(arr), self.split_basis, self.symmetric_basis)

    @cached_property
    def symmetrization_multiplicities(self) -> MatrixCache:
        # TODO: reimplement this by deriving this from index_mapping.as_transition_matrix() instead of calculating all products of norm coefficients

        # The multiplicity can be shown to be the product of the multinomial coefficients of the combinations that make up summedB,
        # i.e. if the count matrix K = summedB[idx] can be written as K[a,b] = Σ_{i = 1}^t K_i[a,b], then
        # the corresponding multiplicity is prod(multinomial_coeff(K_i[a,b] for i in range(t)) for (a,b) in range(d^2))
        # This is actually very annoying to calculate in one go via numpy broadcasting because we don't have a good numpy ufunc for (integer) multinomial coefficients. It can be done with scipy.comb, but this is fp and also somewhat slow.
        # Instead, we use that the multiplicity can also be computed from the quotient of the orbit sizes of the individual and total orbits.

        return MatrixCache.as_cache(
            (multinomial_coeff(self.partition) *
            product_combinations(
                [cast(EndSnOrbitBasis|EndSnOrbitBasisSubset, b).norm_coefficients for b in self.split_basis.bases]
            ).ravel()) / self.symmetric_basis.norm_coefficients[self.index_mapping.index_mapping().as_numpy()]
        )

    @cached_property
    def embedding(self) -> TransitionMatrix:
        """Represents the embedding End^{S_n}(M^n) -> \bigotimes_{i = 1}^t End^{S_μ_i}(M^μ_i) that takes each basis element of End^{S_n}(M^n) to the corresponding sum of basis elements in \bigotimes_{i = 1}^t End^{S_μ_i}(M^μ_i)"""

        # The transpose does the correct thing of giving us sums of all the different terms that have the same symmetrized orbit
        return GivenTransitionMatrix(self.symmetric_basis, self.split_basis, self.index_mapping.as_transition_matrix().to_scipy_sparse().transpose())

    @cached_property
    def symmetrization(self) -> TransitionMatrix:
        """Represents the mapping from \bigotimes_{i = 1}^t End^{S_μ_i}(M^μ_i) to End^{S_n}(M^n) that takes each basis element to its symmetrization"""
        mat = scipy.sparse.diags(self.symmetrization_multiplicities.as_numpy()) @ self.index_mapping.as_transition_matrix().to_scipy_sparse()
        return GivenTransitionMatrix(self.split_basis, self.symmetric_basis, mat)

    def __str__(self):
        return f"{type(self).__name__}({self.split_basis!s} <-> {self.symmetric_basis!s})"