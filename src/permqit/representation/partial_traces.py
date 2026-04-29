import abc
import dataclasses
import itertools
from typing import cast, ClassVar
import numpy as np
from array_api_compat import array_namespace

from .symmetrization import SymmetrizationRelations
from ..algebra.endomorphism_direct_sum_basis import EndSnBlockOrbitBasisSubset, EndSnSingleBlockOrbitBasis
from ..algebra.endomorphism_basis import EndSnOrbitBasisSubset
from ..algebra.linear_map import MatrixCache, CoefficientData, StorageFormat, GivenGatherIndexMapping
from ..utilities import backend
from ..utilities.backend import USE_GPU
from ..utilities.numpy_utils import sum_combinations, product_combinations, take_groups, ArrayAPICompatible, multi_vector_kron
from ..utilities.timing import ExpensiveComputation
from .combinatorics import (
    multinomial_coeff,
    get_multinomial_coeff_func_xp,
)


from ..algebra import EndSnOrbitBasis, EndSnBlockOrbitBasis, MatrixStandardBasis, Basis
from ..utilities.caching import WeakRefMemoize


class BasePartialTraceRelations(metaclass=WeakRefMemoize):
    """
    Baseclass for partial trace relations which allow the following computations:
    Take some basisA on A and basisB on B as well as a joint basis on AB.
    This class computes Tr_A[(C_A ⊗ I_B)^T_A D_AB] for all basis elements C_A in basisA and D_AB in joint_basis,
    where I_B is the identity operator on the B system of appropriate dimension.
    Stores this in an efficient way so that one can easily compute linear combinations of this for coefficients of basisA and
    joint_basis.

    For the particular instances we are interested in, it is always true that for every basis element D_AB of the joint basis, there exists a single
    element C_A such that Tr_A[(C_A ⊗ I_B)^T_A D_AB] is non-zero, and in fact there is also a single basis element C_B such that
    Tr_A[(C_A ⊗ I_B)^T_A D_AB] = c C_B for some number c (which depends on D_AB). These classes efficently compute the relations between the (indices of)
    the elements D_AB, C_A and C_B as well as the coefficient c.

    """

    basisA: Basis
    basisB: Basis
    basisAB: Basis

    @property
    def joint_basis(self) -> Basis:
        """Alias for basisAB for backwards compatibility."""
        return self.basisAB

    A_index_from_joint: MatrixCache
    """A_index_from_joint[i] = j is the unique index j of C_A in basisA such that Tr_A[(C_A ⊗ I_B)^T_A D_AB] is nonzero, where D_AB is at index i in basisAB """
    B_index_from_joint: MatrixCache  # by default np.ndarray
    """B_index_from_joint[i] = j is the unique index j of C_B in basisB such that Tr_B[(I_A ⊗ C_B)^T_A D_AB] is nonzero, where D_AB is at index i in basisAB """
    joint_indices_from_A: MatrixCache  # by default sparse.GCXS
    """ joint_indices_from_A[i,j] = 1 if and only if Tr_B[(I_A ⊗ C_B)^T_A D_AB] = c C_A for some non-zero number c, and C_A is at index i in basisA 
    and D_AB is at index j in basisAB (also C_B is at index B_index_from_joint[j] in basisB). All other matrix entries are zero
    """
    joint_indices_from_B: MatrixCache  # by default sparse.GCXS
    """ joint_indices_from_B[i,j] = 1 if and only if Tr_A[(C_A ⊗ I_B)^T_A D_AB] = c C_B for some non-zero number c, and C_B is at index i in basisB
    and D_AB is at index j in basisAB (also C_A is at index A_index_from_joint[j] in basisA). All other matrix entries are zero
    """
    trace_coefficientsA: MatrixCache  # by default np.ndarray
    """
    trace_coefficientsA[i] = c, where c is the non-zero number in the relation Tr_A[(C_A ⊗ I_B)^T_A D_AB] = c C_B, where D_AB is at index i in basisAB
    (and C_A is at index A_index_from_joint[i] in basisA, C_B is at index B_index_from_joint[i] in basisB).
    """
    trace_coefficientsB: MatrixCache  # by default np.ndarray
    """
    trace_coefficientsB[i] = c, where c is the non-zero number in the relation Tr_B[(I_A ⊗ C_B)^T_A D_AB] = c C_A, where D_AB is at index i in basisAB
    (and C_A is at index A_index_from_joint[i] in basisA, C_B is at index B_index_from_joint[i] in basisB).
    """

    def ensure_calculated(self):
        if not hasattr(self, "A_index_from_joint"):
            with ExpensiveComputation(f"Calculating {str(self)}"):
                self._calculate_matrices()

                # The reverse embeddings are not really embeddings but the transpose of the original embedding as a matrix.
                self.joint_indices_from_A = MatrixCache(
                    GivenGatherIndexMapping(self.A_index_from_joint, self.basisA, self.basisAB)
                    .as_transition_matrix()
                    .to_scipy_sparse()
                    .transpose()
                )
                self.joint_indices_from_B = MatrixCache(
                    GivenGatherIndexMapping(self.B_index_from_joint, self.basisB, self.basisAB)
                    .as_transition_matrix()
                    .to_scipy_sparse()
                    .transpose()
                )

                # These are the dense matrices
                for mat in [self.A_index_from_joint, self.B_index_from_joint, self.trace_coefficientsA, self.trace_coefficientsB]:
                    mat.set_cache_formats(
                        *([StorageFormat.GPU, StorageFormat.NUMPY] if backend.USE_GPU else [StorageFormat.NUMPY])
                    )

                # These are the sparse matrices
                for mat in [self.joint_indices_from_A, self.joint_indices_from_B]:
                    mat.set_cache_formats(
                        *(
                            [StorageFormat.GPU_SPARSE, StorageFormat.SCIPY_SPARSE]
                            if backend.USE_GPU
                            else [StorageFormat.SCIPY_SPARSE]
                        )
                    )

    @abc.abstractmethod
    def _calculate_matrices(self):
        pass

    @dataclasses.dataclass
    class StoredData:
        A_index_from_joint: CoefficientData
        B_index_from_joint: CoefficientData
        joint_indices_from_A: CoefficientData
        joint_indices_from_B: CoefficientData
        trace_coefficientsA: CoefficientData
        trace_coefficientsB: CoefficientData

        def __getitem__(self, item: str):
            # For legacy purposes only, will be removed eventually
            if item == "embeddingA_gpu":
                return self.A_index_from_joint
            elif item == "embeddingB_gpu":
                return self.B_index_from_joint
            elif item == "reverse_embeddingA_gpu":
                return self.joint_indices_from_A
            elif item == "reverse_embeddingB_gpu":
                return self.joint_indices_from_B
            elif item == "trace_coefficientsA_gpu":
                return self.trace_coefficientsA
            elif item == "trace_coefficientsB_gpu":
                return self.trace_coefficientsB

    def _for_coefficient_format(self, *x: CoefficientData):
        coeff_fmts = set(StorageFormat.detect_format(xx) for xx in x)
        assert len(coeff_fmts) == 1, f"Coefficients must be in the same format, found {coeff_fmts}"
        return self.data_in_format(coeff_fmts.pop())

    def data_in_format(self, coeff_fmt: StorageFormat):
        self.ensure_calculated()
        if coeff_fmt is StorageFormat.GPU:
            return self.StoredData(
                A_index_from_joint=self.A_index_from_joint.get(StorageFormat.GPU),
                B_index_from_joint=self.B_index_from_joint.get(StorageFormat.GPU),
                joint_indices_from_A=self.joint_indices_from_A.get(StorageFormat.GPU_SPARSE),
                joint_indices_from_B=self.joint_indices_from_B.get(StorageFormat.GPU_SPARSE),
                trace_coefficientsA=self.trace_coefficientsA.get(StorageFormat.GPU),
                trace_coefficientsB=self.trace_coefficientsB.get(StorageFormat.GPU),
            )
        elif coeff_fmt is StorageFormat.NUMPY:
            return self.StoredData(
                A_index_from_joint=self.A_index_from_joint.get(StorageFormat.NUMPY),
                B_index_from_joint=self.B_index_from_joint.get(StorageFormat.NUMPY),
                joint_indices_from_A=self.joint_indices_from_A.get(StorageFormat.SCIPY_SPARSE),
                joint_indices_from_B=self.joint_indices_from_B.get(StorageFormat.SCIPY_SPARSE),
                trace_coefficientsA=self.trace_coefficientsA.get(StorageFormat.NUMPY),
                trace_coefficientsB=self.trace_coefficientsB.get(StorageFormat.NUMPY),
            )
        else:
            raise ValueError(f"Unsupported coefficient format: {coeff_fmt}")

    def apply_traceA_to_coefficient_vectors[T: CoefficientData](self, coefficientsA: T, coefficients_joint: T) -> T:
        """
        Given coefficient vectors v for self.basisA and z for self.joint_basis, compute the coefficients y of the partial trace result
        in basisB, i.e. Σ_ij Tr_A((v_i C_i^A) ⊗ I_B)^T_A (z_j C_j^{AB}) = Σ_k y_k C_k^B
        :param coefficientsA: Coefficient vector for basisA. This can also be an array of shape (basisA.size(), ...), in which case the calculation is performed simultaneously on each row vector.
        :param coefficients_joint: Coefficient vector for joint_basis
        :return: Coefficient vector for basisB (o an array of shape (basisB.size(), ...) if coefficientsA was a higher dimensional array)
        """
        assert coefficients_joint.ndim == 1, "coefficients_joint must be one-dimensional"
        assert coefficients_joint.shape[0] == self.basisAB.size(), (
            f"coefficients_joint must be of shape ({self.basisAB.size()},), found {coefficients_joint.shape}"
        )

        data = self._for_coefficient_format(coefficients_joint, coefficientsA)
        xp = array_namespace(coefficients_joint)

        embedded_A = coefficientsA.take(data.A_index_from_joint, axis=0)
        return data.joint_indices_from_B @ xp.einsum(
            "i,i,i...->i...", data.trace_coefficientsA, coefficients_joint, embedded_A
        )

    def apply_traceB_to_coefficient_vectors[T: CoefficientData](self, coefficientsB: T, coefficients_joint: T) -> T:
        """
        Given coefficient vectors v for self.basisB and z for self.joint_basis, compute the coefficients y of the partial trace result
        in basisa, i.e. Σ_ij Tr_A(I_A ⊗ (v_i C_i^B))^T_B (z_j C_j^{AB}) = Σ_k y_k C_k^A
        :param coefficientsB: Coefficient vector for basisB. This can also be an array of shape (basisB.size(), ...), in which case the calculation is performed simultaneously on each row vector.
        :param coefficients_joint: Coefficient vector for joint_basis
        :return: Coefficient vector for basisA (or an array of shape (basisA.size(), ...) if coefficientsB was a higher dimensional array)
        """
        assert coefficients_joint.ndim == 1, "coefficients_joint must be one-dimensional"
        assert coefficients_joint.shape[0] == self.basisAB.size(), (
            f"coefficients_joint must be of shape ({self.basisAB.size()},), found {coefficients_joint.shape}"
        )
        data = self._for_coefficient_format(coefficients_joint, coefficientsB)
        xp = array_namespace(coefficients_joint)

        embedded_B = coefficientsB.take(data.B_index_from_joint, axis=0)
        return data.joint_indices_from_A @ xp.einsum(
            "i,i,i...->i...", data.trace_coefficientsB, coefficients_joint, embedded_B
        )

    def get_basis_index_mapping(self, joint_idx: int) -> tuple[int, int]:
        """
        Helper methods for tests that given an index i of the joint_basis gets the corresponding index pair (j, k), which is the
        only index pair such that Tr_A[C_j^A^T ⊗ C_k^B C_i^{AB}] != 0 (and equivalently Tr_B[C_j^A ⊗ C_k^B^T C_i^{AB}] != 0).
        :return:
        """
        self.ensure_calculated()
        return self.A_index_from_joint.as_numpy()[joint_idx], self.B_index_from_joint.as_numpy()[joint_idx]

    def __str__(self):
        return f"{type(self).__name__}(A: {self.basisA!s}, B: {self.basisB!s}, AB: {self.basisAB!s})"


OrbitOrSubset = EndSnOrbitBasis | EndSnOrbitBasisSubset


class PartialTraceRelations(BasePartialTraceRelations):
    """
    Given basisA = EndSnOrbitBasis(n, d_A) and basisB = EndSnOrbitBasis(n, d_B), take joint_basis = EndSnOrbitBasis(n, d_A*d_B).
    This class computes Tr_A^n[(C_A ⊗ I_B)^T_A D_AB] for all basis elements C_A in basisA and D_AB in joint_basis,
    where I_B is the identity operator on the B system of appropriate dimension.
    Stores this in an efficient way so that one can easily compute linear combinations of this for coefficients of basisA and
    joint_basis.

    The key point of the calculation is that there is exactly one combination of basis elements C_A, C_B and D_AB
    for which Tr_A^n[(C_A ⊗ I_B)^T_A D_AB] = c C_B, and this is the only non-zero combination.
    We calculate this combination, store it in sparse matrices, as well as the numerical prefactor c.
    In the same go we also calculate the same for Tr_B^n[(I_A ⊗ C_B)^T_A D_AB]

    Performance
    -----------
    Construction is done in three phases to keep cost low for large n (e.g. n=8, joint size ~320k):
    1. Collect count matrices by iterating weak_compositions(n, d_AB²) and reshaping — no PairOrbit
       allocation, same canonical order as the joint basis.
    2. Batched reshape/transpose/sum on GPU (if permqit_USE_GPU) or CPU; single upload and a few
       downloads when using GPU.
    3. Indexing via count_matrix_to_index (no PairOrbit per row), reverse embeddings built as COO
       from (embedding, arange(N)), and trace coefficients from vectorized multinomial_coeff_fast.
    """

    basisA: OrbitOrSubset
    basisB: OrbitOrSubset
    basisAB: OrbitOrSubset

    CHUNK_SIZE: ClassVar[int] = 2_000_000
    """Computations are run in chunks of this size to avoid materializing the full (N, d_AB, d_AB) array for large N"""

    def __init__(self, basisA: OrbitOrSubset, basisB: OrbitOrSubset, basisAB: OrbitOrSubset):
        assert basisA.n == basisB.n == basisAB.n, (
            f"All bases must have the same n, got A: {basisA.n}, B: {basisB.n}, AB: {basisAB.n}"
        )
        assert basisA.d * basisB.d == basisAB.d, (
            f"Basis dimensions must satisfy dA*dB = dAB, got A: {basisA.d}, B: {basisB.d}, AB: {basisAB.d}"
        )

        self.basisA = basisA
        self.basisB = basisB
        self.basisAB = basisAB

    def _calculate_matrices(self):
        # Performance: Phase 1 = weak_compositions only (no PairOrbit). Phase 2 = batched GPU/CPU
        # reshape/transpose/sum. Phase 3 = count_matrix_to_index + COO reverse + vectorized multinomial.
        # For large N (e.g. n=13, d_AB=4 -> N~37e6), process in chunks to avoid materializing
        # the full (N, d_AB, d_AB) array before creating the partial trace.
        joint_basis = self.basisAB
        bases = [self.basisA, self.basisB]
        d_A, d_B = bases[0].d, bases[1].d
        d_AB = joint_basis.d
        N = joint_basis.size()

        # These are always computed on cpu
        A_index_from_joint = np.empty((N,), dtype=np.int_)
        B_index_from_joint = np.empty((N,), dtype=np.int_)

        xp = backend.xp

        # TODO: benchmark how much the gpu path actually helps here

        trace_coefficientsA = xp.empty((N,), dtype=xp.int_)
        trace_coefficientsB = xp.empty((N,), dtype=xp.int_)

        # Chunked path: never hold full (N, d_AB, d_AB). Process weak_compositions in chunks.

        multinomial_coeff_func = get_multinomial_coeff_func_xp()

        chunk_start = 0
        for chunk in itertools.batched(joint_basis.iterate_count_matrices(), self.CHUNK_SIZE):
            chunk_len = len(chunk)
            count_stack = np.stack(chunk, axis=0).reshape(chunk_len, d_A, d_B, d_A, d_B)
            transposed = np.transpose(count_stack, (0, 1, 3, 2, 4)).reshape(chunk_len, d_A**2, d_B**2)
            transposed_xp = xp.asarray(transposed)
            sums_0_chunk = backend.to_cpu(xp.sum(transposed_xp, axis=2))
            sums_1_chunk = backend.to_cpu(xp.sum(transposed_xp, axis=1))
            for j in range(chunk_len):
                A_index_from_joint[chunk_start + j] = self.basisA.count_matrix_to_index(sums_0_chunk[j].reshape((d_A, d_A)))
                B_index_from_joint[chunk_start + j] = self.basisB.count_matrix_to_index(sums_1_chunk[j].reshape((d_B, d_B)))
            row_mult = multinomial_coeff_func(transposed_xp, axis=2)
            trace_coefficientsB[chunk_start : chunk_start + chunk_len] = xp.rint(xp.prod(row_mult, axis=1)).astype(
                xp.int_
            )
            col_mult = multinomial_coeff_func(transposed_xp.transpose(0, 2, 1), axis=2)
            trace_coefficientsA[chunk_start : chunk_start + chunk_len] = xp.rint(xp.prod(col_mult, axis=1)).astype(
                xp.int_
            )
            chunk_start += chunk_len

        self.A_index_from_joint = MatrixCache(A_index_from_joint)
        self.B_index_from_joint = MatrixCache(B_index_from_joint)
        self.trace_coefficientsA = MatrixCache(trace_coefficientsA)
        self.trace_coefficientsB = MatrixCache(trace_coefficientsB)

    def __str__(self):
        return f"{type(self).__name__}(n={self.basisA.n}, {self.basisA.d}x{self.basisB.d})"


type SingleBlockOrbitOrSubset = EndSnSingleBlockOrbitBasis


class SingleBlockPartialTraceRelations(BasePartialTraceRelations):
    """
    As shown e.g. in http://arxiv.org/abs/0910.4515, Section 4, every element of End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n) is uniquely associated with an
    element of ⊕_μ ⊗_{i = 1}^t End^{S_(μ_i)}((ℂ^{p_i x p_i})^{μ_i}), where μ ∈ ℕ_n^{× t} is a t-tuple with entries in 0, ..., n where
    the i-th entry counts the number of occurrences of block i in the tensor product.
    In fact, these two Algebras are *-isomorphic. Assume now that ⊕_i^t_{AB}(ℂ^{p_i x p_i}) is actually bipartite, so of the form
    [⊕_a^t_A ℂ^{d_a x d_a}] ⊗ [⊕_b^t_B ℂ^{e_b x e_b}]. This then also means that every i = 0, ..., t_AB is uniquely associated to a pair of indices i ≅ (a,b),
    where a ∈ {0, ..., t_A}, b ∈ {0, ..., t_B}. For a μ = μ_{AB} ∈ ℕ_n^{× t_{AB}}, which counts the number of occurrences of each i = (a,b), let
    μ_A ∈ ℕ_n^{t_A} (respectively μ_B ∈ ℕ_n^{t_B}) correspond to the number of occurrences of the 'marginal indices' a (or b respectively),
    i.e. (μ_A)_a = Σ_{i, s.t. i = (a,b) for some b} (μ_{AB})_i, and similarly for μ_{B}_b.
    We then also have the association between End^{S_n}(⊕_a^t_A(ℂ^{d_a x d_a})^n) ≅ ⊕_{μ_A} ⊗_{a = 1}^t_A End^{S_((μ_A)_a)}((ℂ^{d_a x d_a})^{μ_A_a})
    and similarly for the system B.

    Given a fixed block μ_{AB} with corresponding marginals μ_A and μ_B, this class allows the calculation of Tr_A^n[(C_A ⊗ I_B)^T_A D_AB], where D_AB is an element of
    End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n) -- associated to an element in ⊗_{i = 1}^t End^{S_(μ_{AB}_i)}((ℂ^{p_i x p_i})^{μ_i}) for the given μ --
    and C_A is the unique element of End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n for which this is non-zero, which happens to be an element associated to
    ⊗_{a = 1}^t End^{S_(μ_{A}_a)}((ℂ^{d_a x d_a})^{μ_A_a}).
    """

    basisA: SingleBlockOrbitOrSubset
    basisB: SingleBlockOrbitOrSubset
    basisAB: SingleBlockOrbitOrSubset

    def __init__(
        self,
        basisA: SingleBlockOrbitOrSubset,
        basisB: SingleBlockOrbitOrSubset,
        basisAB: SingleBlockOrbitOrSubset,
    ):
        """
        In general here, there need not be a 1-1 correspondences between the multiplicities of blocks, for example we can have that
        the AB basis has muptiplicities [(0,0): 1, (0,1): 1], which will then mean that the A basis has multiplicities [(0): 2], and the B basis has
        [(0): 1, (1): 1].
        """
        # TODO: add an assertion that at every 'multiplicity point' the block dimensions add up
        self.basisA, self.basisB, self.basisAB = basisA, basisB, basisAB

    def __str__(self):
        return f"{type(self).__name__}(A: {self.basisA!s}, B: {self.basisB!s}, AB: {self.basisAB!s})"

    def _split_marginal_tensor_product(
        self, basisAB: SingleBlockOrbitOrSubset, basisA: SingleBlockOrbitOrSubset
    ) -> tuple[list[EndSnOrbitBasis | EndSnOrbitBasisSubset], list[SymmetrizationRelations | None]]:
        """
        'Splits' the End^{S_μ_i} parts of a marginal basis into smaller parts corresponding to each tensor product factor of basisAB.
        This is necessary since the AB system will have blocks where only the B-part of the block is different, and these multiple blocks
        will correspond to a single A block.
        For example, if basisAB has multiplicities [(0,0): 2, (0,1): 1] (that means basisAB is the canonical basis of End^{S_2}((M_0)_A ⊗ (M_0)_B) ⊗ End^{S_1}((M_0)_A ⊗ (M_1)_B)),
        then we split basisA = End^{S_3}(((M_0)_A)^{⊗3}) into End^{S_2}(M_0_A^{⊗2}) ⊗ End^{S_1}(M_0_A^{⊗1})
        While the parameter is called basisA, this code works for both basisA and basisB.

        :return: A tuple with elements:
        1. A list of all the individual tensor product components after the split, i.e.
            [OrbitBasis of: End^{S_2}(M_0_A^{⊗2}), OrbitBasis of:End^{S_1}(M_0_A^{⊗1})]
        2. A list of all the symmetrization relations that correspond to the split, i.e. SymmetrizationRelations(M_0, (2,1))
        """

        AB_mult_cumsum = np.cumsum(basisAB.multiplicities)
        A_mult_cumsum = np.cumsum(basisA.multiplicities)

        A_split_tensor_product_elements = []
        A_symmetrization_relations = []
        prev_index = 0

        # Iterate through all different parts of the A marginal tensor product decomposition, i.e. iterate through a in
        # ⊗_{a = 1}^t End^{S_(μ_{A}_a)}((ℂ^{d_a x d_a})^{μ_A_a})
        for basisA_orig_block, accumulated_mu, mu in zip(basisA.original_blocks, A_mult_cumsum, basisA.multiplicities):
            # Now for each block, find all the tensor product elements in the AB tensor product decomposition that correspond to it, we do this
            # by just counting the number n of 'systems' in the block and taking the respective systems from the AB decomposition
            joint_idx = prev_index + np.searchsorted(AB_mult_cumsum[prev_index:], accumulated_mu, "left") + 1
            # This is the index in the tensor product decomposition
            # We have to be careful here as we allow for μ_i = 0, i.e. blocks which appear zero times. This means that the cumsum can have repeated entries.
            # Hence we cannot take np.searchsorted(..., 'right'), but have to do 'left' and then add 1

            # TODO: Add some shortcuts to deal with the mu = 1 case specially (since this is just the basis itself)
            if mu > 0:
                rel = SymmetrizationRelations(basisA_orig_block, basisAB.multiplicities[prev_index:joint_idx])
                A_symmetrization_relations.append(rel)
                A_split_tensor_product_elements.extend(rel.split_basis.bases)
            else:
                A_symmetrization_relations.append(None)

            prev_index = joint_idx

        return A_split_tensor_product_elements, A_symmetrization_relations

    def _per_factor_marginal_split_indices(
        self,
        individual_partial_trace_relation_index_mappings: list[np.ndarray],
        symmetrization_relations: list[SymmetrizationRelations | None],
    ):
        return [
            # arr is of form split_A_index[AB_index], sym_rel.index_mapping is of form symmetrized_A_index[split_A_index]
            sum_combinations(
                [
                    arr * split_index_multiplier
                    for arr, split_index_multiplier in zip(grp, sym_rel.split_basis.basis_indices_multiplier)
                ]
            ).ravel()
            if sym_rel
            else None
            for grp, sym_rel in zip(
                take_groups(
                    individual_partial_trace_relation_index_mappings,
                    [len(sr.split_basis.bases) if sr else 0 for sr in symmetrization_relations],
                ),
                symmetrization_relations,
            )
        ]

    def _per_factor_split_indices_to_symmetrized_indices(
        self, split_indices: list[np.ndarray | None], symmetrization_relations: list[SymmetrizationRelations | None]
    ):
        return [
            sym_rel.index_mapping.index_mapping().as_numpy()[indices] if sym_rel else None
            for sym_rel, indices in zip(symmetrization_relations, split_indices)
        ]

    def _symmetrized_indices_to_full_basis_indices(
        self, per_factor_symmetrized_indices: list[np.ndarray | None], marginal_basis: SingleBlockOrbitOrSubset
    ):
        assert len(per_factor_symmetrized_indices) == len(marginal_basis.bases), (
            f"{{{len(per_factor_symmetrized_indices)} != {len(marginal_basis.bases)}}}"
        )

        return sum_combinations(
            (
                local_idxs * index_multiplier
                for local_idxs, index_multiplier in zip(
                    per_factor_symmetrized_indices, marginal_basis.basis_indices_multiplier
                )
                if local_idxs is not None
            ),
            length=len(per_factor_symmetrized_indices),
        ).ravel()

    def _marginal_indices_from_joint(
        self,
        split_indices: list[np.ndarray | None],
        symmetrization_relations: list[SymmetrizationRelations | None],
        full_marginal_basis: EndSnSingleBlockOrbitBasis,
    ):
        symmetrized_indices = self._per_factor_split_indices_to_symmetrized_indices(
            split_indices, symmetrization_relations
        )
        return self._symmetrized_indices_to_full_basis_indices(symmetrized_indices, full_marginal_basis)

    def _per_factor_marginal_indices_from_joint(
        self,
        individual_partial_trace_relation_index_mappings: list[np.ndarray],
        symmetrization_relations: list[SymmetrizationRelations | None],
        full_basis: EndSnSingleBlockOrbitBasis,
    ) -> list[np.ndarray | None]:
        """
        Combines the individual partial trace relation index mappings on each tensor factor into a big index mapping, and then takes care
        of the necessary resymmetrization for any parts of the marginal tensor product decomposition that was previously split (as given by the passed
        symmetrization relations).

        :params:
        individual_partial_trace_relation_index_mappings: How the individual parts of the marginal tensor product decomposition are embedded into the corresponding part of the joint
        tensor product decomposition.
        symmetrization_relations: the symmetrization relations according to which individual parts of the tensor product decomposition will be resymmetrized
        full_basis: The full marginal basis (without any splitting)
        :return: The index mapping as an array basisAIndex[basisABIndex] (if full_basis is basisA)
        """
        assert len(symmetrization_relations) == len(full_basis.bases), (
            f"{{{len(symmetrization_relations)} != {len(full_basis.bases)}}}"
        )

        return [
            # arr is of form split_A_index[AB_index], sym_rel.index_mapping is of form symmetrized_A_index[split_A_index]
            sym_rel.index_mapping.index_mapping().as_numpy()[
                sum_combinations(
                    [
                        arr * split_index_multiplier
                        for arr, split_index_multiplier in zip(grp, sym_rel.split_basis.basis_indices_multiplier)
                    ]
                ).ravel()
            ]
            if sym_rel
            else None
            for grp, sym_rel in zip(
                take_groups(
                    individual_partial_trace_relation_index_mappings,
                    [len(sr.split_basis.bases) if sr else 0 for sr in symmetrization_relations],
                ),
                symmetrization_relations,
            )
        ]

    def _trace_coefficients(
        self,
        individual_partial_trace_coefficients: list[np.ndarray],
        other_marginal_symmetrization_relations: list[SymmetrizationRelations | None],
        other_marginal_split_indices_from_joint: list[np.ndarray | None],
    ) -> np.ndarray:
        """
        Returns the prefactor after the partial trace (see above). Note that the way this is affected by symmetrization depends on the splitting of the
        *target* basis (i.e. if we are calculating Tr_(A^n)[(C_A^T ⊗ id_B)(D_AB)] = p C_B, then this returns the prefactors p
        (for each D_AB) and the resymmetrization we have to perform depends on what exactly the C_B is.
        """
        assert len(other_marginal_symmetrization_relations) == len(other_marginal_split_indices_from_joint), (
            f"{len(other_marginal_symmetrization_relations)} != {len(other_marginal_split_indices_from_joint)}"
        )

        # TODO: shortcut here on no splitting
        return multi_vector_kron(
            np,  # might want to be backend.xp
            *[
                multi_vector_kron(
                    np,  # might want to be backend.xp
                    *grp,
                ).ravel()  # This part comes from the individual partial trace relations (for each tensor product factor)
                * sym_rel.symmetrization_multiplicities.as_numpy()[indices_from_joint]
                # Indices from joint here are in the symmetric_basis, whereas symmetrization_multiplicities are in the split_basis
                # And this part adds the multiplicities from the resymmetrization
                for grp, sym_rel, indices_from_joint in zip(
                    take_groups(
                        individual_partial_trace_coefficients,
                        [len(sr.split_basis.bases) if sr else 0 for sr in other_marginal_symmetrization_relations],
                    ),
                    other_marginal_symmetrization_relations,
                    other_marginal_split_indices_from_joint,
                )
                if sym_rel
            ],
        )

    def _calculate_matrices(self):
        # We first 'split' the End^{S_μ_i} parts of both basisA and basisB into smaller parts corresponding to each tensor product factor of basisAB,
        # then later on, we will deal with the combinatorical factors that come up in the resymmetrization of reverting this split.
        # I.e. if basisAB has multiplicities [(0,0): 2, (0,1): 1], then we split basisA = End^{S_3}(M_A^{⊗3}) into End^{S_2}(M_A^{⊗2}) ⊗ End^{S_1}(M_A^{⊗1})

        A_split_tensor_product_elements, A_symmetrization_relations = self._split_marginal_tensor_product(
            self.basisAB, self.basisA
        )
        B_split_tensor_product_elements, B_symmetrization_relations = self._split_marginal_tensor_product(
            self.basisAB, self.basisB
        )

        # Filter out the n = 0 terms here (since they are also not part of the split_tensor_product_elements)
        non_zero_AB = [b for b in self.basisAB.bases if b.n > 0]

        assert len(A_split_tensor_product_elements) == len(B_split_tensor_product_elements) == len(non_zero_AB), (
            f"{{{len(A_split_tensor_product_elements)}, {len(B_split_tensor_product_elements)}, {len(non_zero_AB)}}}"
        )

        # These are the individual partial trace relations for each of the split parts we just created. The whole point of splitting was that we now
        # have exactly one marginal factor for each joint factor.

        relations = [
            PartialTraceRelations(b_A, b_B, b_AB)
            for b_A, b_B, b_AB in zip(
                A_split_tensor_product_elements,
                B_split_tensor_product_elements,
                non_zero_AB,
            )
        ]
        for rel in relations:
            rel.ensure_calculated()

        A_split_index_from_joint = self._per_factor_marginal_split_indices(
            [ptr.A_index_from_joint.as_numpy() for ptr in relations], A_symmetrization_relations
        )
        B_split_index_from_joint = self._per_factor_marginal_split_indices(
            [ptr.B_index_from_joint.as_numpy() for ptr in relations], B_symmetrization_relations
        )

        A_index_from_joint = self._marginal_indices_from_joint(
            A_split_index_from_joint, A_symmetrization_relations, self.basisA
        )
        B_index_from_joint = self._marginal_indices_from_joint(
            B_split_index_from_joint, B_symmetrization_relations, self.basisB
        )

        trace_coefficientsA = self._trace_coefficients(
            [ptr.trace_coefficientsA.as_numpy() for ptr in relations],
            B_symmetrization_relations,
            B_split_index_from_joint,
        )
        trace_coefficientsB = self._trace_coefficients(
            [ptr.trace_coefficientsB.as_numpy() for ptr in relations],
            A_symmetrization_relations,
            A_split_index_from_joint,
        )

        self.A_index_from_joint = MatrixCache(A_index_from_joint)
        self.B_index_from_joint = MatrixCache(B_index_from_joint)
        self.trace_coefficientsA = MatrixCache(trace_coefficientsA)
        self.trace_coefficientsB = MatrixCache(trace_coefficientsB)


BlockOrbitOrSubset = EndSnBlockOrbitBasis | EndSnBlockOrbitBasisSubset


class BlockPartialTraceRelations(BasePartialTraceRelations):
    basisA: BlockOrbitOrSubset
    basisB: BlockOrbitOrSubset
    basisAB: BlockOrbitOrSubset

    def __init__(
        self,
        basisA: BlockOrbitOrSubset,
        basisB: BlockOrbitOrSubset | EndSnOrbitBasis,
        basisAB: BlockOrbitOrSubset | None = None,
    ):
        if isinstance(basisB, EndSnOrbitBasis):
            basisB = EndSnBlockOrbitBasis(basisB.n, MatrixStandardBasis(basisB.d))

        self.basisA, self.basisB = basisA, basisB
        if basisAB:
            self.basisAB = basisAB
        else:
            assert all(isinstance(b, MatrixStandardBasis) for b in basisA.original_blocks + basisB.original_blocks), (
                "Automatic creation of AB basis based on A and B basis not yet implemented for subsets"
            )
            assert basisA.n == basisB.n, f"{basisA.n} != {basisB.n}"
            self.basisAB = EndSnBlockOrbitBasis(
                basisA.n,
                *(MatrixStandardBasis(bA.d * bB.d) for bA in basisA.original_blocks for bB in basisB.original_blocks),
            )

    def _calculate_matrices(self):
        xp = np
        fmt = StorageFormat.NUMPY

        num_blocks_A = len(self.basisA.original_blocks)
        num_blocks_B = len(self.basisB.original_blocks)
        assert len(self.basisAB.original_blocks) == num_blocks_A * num_blocks_B

        # This is an index representation of the mapping D_AB -> C_A (i.e. if A_index_from_joint[i] = j, then the ith element in basisAB gets mapped to the jth element in basisA)
        A_index_from_joint = xp.empty((self.basisAB.size(),), dtype=xp.int_)
        # Same for the mapping D_AB -> C_B
        B_index_from_joint = xp.empty((self.basisAB.size(),), dtype=xp.int_)

        # This stores the prefactor for Tr_A[(C_A ⊗ I_B) D_AB]
        trace_coefficientsA = xp.empty((self.basisAB.size(),), dtype=xp.int_)
        # This stores the prefactor for Tr_B[(I_B ⊗ C_B) D_AB]
        trace_coefficientsB = xp.empty((self.basisAB.size(),), dtype=xp.int_)

        # Used to ensure relations are kept alive to be reused for the duration of this function
        _relations_cache: set[SingleBlockPartialTraceRelations] = set()

        for blockAB_index, partition in enumerate(self.basisAB.compositions_for_blocks):
            bipartite_partition = np.asarray(partition).reshape(num_blocks_A, num_blocks_B)
            partitionA = np.sum(bipartite_partition, axis=1)
            partitionB = np.sum(bipartite_partition, axis=0)

            blockA_index = self.basisA.composition_to_subbasis_index(partitionA)
            blockB_index = self.basisB.composition_to_subbasis_index(partitionB)

            rel = SingleBlockPartialTraceRelations(
                self.basisA.bases[blockA_index], self.basisB.bases[blockB_index], self.basisAB.bases[blockAB_index]
            )
            rel.ensure_calculated()
            _relations_cache.add(rel)

            AB_offset = self.basisAB.basis_indices_start[blockAB_index]
            i, j = AB_offset, AB_offset + self.basisAB.bases[blockAB_index].size()

            A_index_from_joint[i:j] = rel.A_index_from_joint.get(fmt) + self.basisA.basis_indices_start[blockA_index]
            B_index_from_joint[i:j] = rel.B_index_from_joint.get(fmt) + self.basisB.basis_indices_start[blockB_index]

            trace_coefficientsA[i:j] = rel.trace_coefficientsA.get(fmt)
            trace_coefficientsB[i:j] = rel.trace_coefficientsB.get(fmt)

        self.A_index_from_joint = MatrixCache(A_index_from_joint)
        self.B_index_from_joint = MatrixCache(B_index_from_joint)
        self.trace_coefficientsA = MatrixCache(trace_coefficientsA)
        self.trace_coefficientsB = MatrixCache(trace_coefficientsB)



def right_compose_permutation_invariant_with_covariant_channel(
    perm_cov_channel: ArrayAPICompatible,
    perm_cov_channel_partial_trace_relations: BasePartialTraceRelations,
    other_channel: ArrayAPICompatible,
    other_channel_output_basis: Basis,
) -> ArrayAPICompatible:
    """
    Computes the channel composition N ∘ M, where M is a permutation-covariant channel M:A^n -> B^n, specified through different blocks, and N is another channel
    N: B^n -> R whose output is invariant under permutations of its input.
    :param perm_cov_channel: coefficients of the choi matrix of the permutation covariant channel, specified in ``perm_cov_channel_partial_trace_relations.basisAB``
    :param other_channel: coefficients of the channel N in MatrixTensorProductBasis(perm_cov_channel_partial_trace_relations.basisB, other_channel_output_basis)
    :returns: coefficients of the choi matrix of N ∘ M in the MatrixTensorProductBasis(perm_cov_channel_partial_trace_relations.basisA, other_channel_output_basis)
    """

    assert perm_cov_channel.shape == (perm_cov_channel_partial_trace_relations.basisAB.size(),)
    return perm_cov_channel_partial_trace_relations.apply_traceB_to_coefficient_vectors(
        other_channel.reshape(perm_cov_channel_partial_trace_relations.basisB.size(), other_channel_output_basis.size()), perm_cov_channel
    ).reshape(-1) # ravel() not available in GCXS which is included in ArrayAPICompatible

def left_compose_permutation_invariant_with_covariant_channel(
    perm_cov_channel: ArrayAPICompatible,
    perm_cov_channel_partial_trace_relations: BasePartialTraceRelations,
    other_channel: ArrayAPICompatible,
    other_channel_input_basis: Basis,
):
    """
    Computes the channel composition M ∘ N, where M is a permutation-covariant channel M:A^n -> B^n, specified through different blocks, and N is another channel
    N: R -> B^n whose output is invariant under permutations of its input.
    :param perm_cov_channel: coefficients of the choi matrix of the permutation covariant channel, specified in ``perm_cov_channel_partial_trace_relations.basisAB``
    :param other_channel: coefficients of the channel N in MatrixTensorProductBasis(other_channel_input_basis, perm_cov_channel_partial_trace_relations.basisA)
    :returns: coefficients of the choi matrix of M ∘ N in the MatrixTensorProductBasis(other_channel_input_basis, perm_cov_channel_partial_trace_relations.basisB)
    """

    assert perm_cov_channel.shape == (perm_cov_channel_partial_trace_relations.basisAB.size(),)
    return perm_cov_channel_partial_trace_relations.apply_traceB_to_coefficient_vectors(
        other_channel.reshape(other_channel_input_basis.size(), perm_cov_channel_partial_trace_relations.basisB.size()).transpose(1, 0), perm_cov_channel
    ).transpose(1, 0).reshape(-1) # ravel() not available in GCXS which is included in ArrayAPICompatible

