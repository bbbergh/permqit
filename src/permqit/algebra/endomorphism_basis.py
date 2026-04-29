from __future__ import annotations

import scipy
from permqit.algebra.linear_map import TransitionMatrix, GivenTransitionMatrix
from ..algebra.basis_subset import MatrixEntryMask, OrbitFitsMaskPredicate
from permqit.utilities.numpy_utils import contract_at_axis, ArrayAPICompatible
from array_api_compat import array_namespace

import dataclasses
import itertools
import types
from functools import cached_property
from typing import Iterable, Tuple, Sequence, Any

import numpy as np
import scipy.sparse as sp
import sparse
from sparse import SparseArray

from .basis import MatrixBasis, Basis, MatrixStandardBasis, DenseOrSparse, dtypeT
from .basis_subset import MatrixBasisSubset
from .matrix import BlockDiagonalMatrix
from ..representation.young_tableau import SSYT, Partition
from ..representation.orbits import PairOrbit
from ..representation.combinatorics import weak_composition_to_index, weak_composition_from_index, weak_compositions
from ..utilities import caching
from ..utilities.timing import ExpensiveComputation, MaybeExpensiveComputation
from ..utilities.deprecated import deprecated


__all__ = [
    "EndSnOrbitBasis",
    "EndSnIrrepBasis",
    "EndSnBlockDiagonalBasis",
    "MatrixDirectSumBasis",
    "EndSnOrbitBasisSubset",
]


@dataclasses.dataclass(unsafe_hash=True)
class EndSnOrbitBasis(MatrixBasis[PairOrbit]):
    """Canonical basis for the vector space End^(S_n)(V^n) where the basis vectors are the indicator matrices corresponding to PairOrbits.
    The labels are the PairOrbits.
    """

    n: int
    d: int
    """The dimension of V (i.e. the single-copy Hilbert space on which the matrices act)"""

    xp: types.ModuleType = dataclasses.field(default=sparse, kw_only=True, repr=False)
    dtype: dtypeT = dataclasses.field(default=np.int_, kw_only=True, repr=False)

    @caching.cache_noargs
    def all_vectors(self) -> sparse.SparseArray:
        return self.xp.stack(list(self.iterate_vectors())).asformat("gcxs")

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> sparse.GCXS:
        return contract_at_axis(self.all_vectors(), coeffs, 0, coeff_axis, xp=self.xp)

    def iterate_vectors(self) -> Iterable[sparse.GCXS]:
        for o in PairOrbit.generate_all(self.n, self.d):
            yield o.indicator_matrix._get_matrix(self.xp, self.dtype)

    def size(self) -> int:
        return PairOrbit.count(self.n, self.d)

    def iterate_labels(self) -> Iterable[PairOrbit]:
        return PairOrbit.generate_all(self.n, self.d)

    def iterate_count_matrices(self) -> Iterable[np.ndarray]:
        return PairOrbit.generate_all_count_matrices(self.n, self.d)

    def label_to_vector(self, label: PairOrbit) -> sparse.GCXS:
        return label.indicator_matrix.matrix

    def count_matrix_to_index(self, count_matrix: np.ndarray) -> int:
        return weak_composition_to_index(self.n, self.d**2, count_matrix.ravel())

    def label_to_index(self, label: PairOrbit) -> int:
        return self.count_matrix_to_index(label.count_matrix)

    def label_at_index(self, idx: int) -> PairOrbit:
        return PairOrbit(self.count_matrix_at_index(idx), self.n, self.d)

    def count_matrix_at_index(self, idx: int) -> np.ndarray:
        comp = weak_composition_from_index(self.n, self.d**2, idx)
        return comp.reshape(self.d, self.d)

    def vector_at_index(self, idx: int) -> sparse.GCXS:
        return self.label_to_vector(self.label_at_index(idx))

    def coefficients_for_tensor_product(self, matrix: np.ndarray) -> np.ndarray:
        """For a given dxd matrix A, returns the coefficients of the matrix A^n (i.e. the n-fold tensor product of A with itself) in this basis."""
        assert matrix.shape == (self.d, self.d), f"matrix must be {self.d}x{self.d}, got {matrix.shape}"
        return np.fromiter(
            (np.prod(matrix**orbit.count_matrix) for orbit in self.iterate_labels()),
            dtype=matrix.dtype,
            count=self.size(),
        )

    @caching.cache_noargs
    def coefficients_of_identity(self) -> np.ndarray:
        """Returns the coefficients of the identity matrix"""
        # TODO: optimize this by explicitly constructing the diagonal count_matrices
        return self.coefficients_for_tensor_product(np.eye(self.d))

    @caching.cache_noargs
    def transpose_map(self):
        """Return the :class:`~permqit.algebra.endomorphism_basis_maps.TransposeIndexMapping`
        that permutes coefficient vectors to apply the matrix transpose."""
        from .endomorphism_basis_maps import TransposeIndexMapping  # local import – avoids circular deps

        return TransposeIndexMapping(self)

    def transpose(self, coeffs: np.ndarray, axis: int = -1) -> np.ndarray:
        return self.transpose_map().apply_to_coefficient_vector(coeffs, axis=axis)

    @deprecated("Use the functionality of transpose_map() (with buil_in multiformat caching) instead of accessing the numpy array directly")
    def transpose_index_lookup(self):
        return self.transpose_map().index_mapping().as_numpy()

    @property
    def dimension(self) -> int:
        return self.d**self.n

    @cached_property
    def all_count_matrices(self) -> np.ndarray:
        return np.stack(list(o.count_matrix for o in self.iterate_labels()))

    @cached_property
    def norm_coefficients(self) -> np.ndarray:
        with ExpensiveComputation(f"Generating HS norms of indicator matrices for {str(self)}"):
            return np.fromiter(
                (m.indicator_matrix.HS_norm for m in self.iterate_labels()), dtype=np.int_, count=self.size()
            )

    @caching.cache
    def partial_transpose_index_lookup(self, system: str = "B") -> np.ndarray:
        """
        Returns an array mapping orbit indices to their partial transpose orbit indices.

        For orbit i, partial_transpose_index_lookup()[i] = j means that the partial transpose
        of orbit i is orbit j. This can be used to transform coefficients:
            sigma_TB_coeffs = sigma_coeffs[partial_transpose_index_lookup()]

        :param system: 'A' or 'B' - which subsystem to transpose
        :return: Array of shape (m,) where m is the number of orbits
        """
        with MaybeExpensiveComputation(f"Constructing partial_transpose_index_lookup for {str(self)}"):
            lookup = (self.label_to_index(orbit.partial_transpose(system)) for orbit in self)
            return np.fromiter(lookup, dtype=int, count=self.size())

    def partial_transpose(self, coeffs: np.ndarray, system: str = "B", axis: int = -1) -> np.ndarray:
        """
        Returns coefficients of the partial transpose of a matrix given its orbit basis coefficients.

        :param coeffs: Coefficients in the orbit basis
        :param system: 'A' or 'B' - which subsystem to transpose
        :param axis: Axis along which the coefficients are stored
        :return: Coefficients of the partial transpose in the orbit basis
        """
        return coeffs.take(self.partial_transpose_index_lookup(system), axis=axis)

    @caching.cache
    def partial_transpose_sparse_matrix(self, system: str = "B") -> sp.csr_matrix:
        """
        Returns a sparse permutation matrix P such that P @ sigma_orbit = sigma_TB_orbit.

        This is O(m) in memory and O(m) for matrix-vector multiplication.

        :param system: 'A' or 'B' - which subsystem to transpose
        :return: Sparse permutation matrix of shape (m, m) where m is the number of orbits
        """
        pt_indices = self.partial_transpose_index_lookup(system)
        m = len(pt_indices)
        # P[j, i] = 1 means orbit i maps to orbit j
        return sp.csr_matrix((np.ones(m, dtype=np.float64), (pt_indices, np.arange(m))), shape=(m, m))

    @cached_property
    def trace_coefficients(self) -> np.ndarray:
        with ExpensiveComputation(f"Generating trace coefficients for {str(self)}"):
            return np.fromiter(
                (m.indicator_matrix.trace for m in self.iterate_labels()), dtype=np.int_, count=self.size()
            )

    def trace(self, coeffs: np.ndarray, axis: int = -1) -> np.ndarray:
        xp = array_namespace(coeffs)
        return contract_at_axis(self.trace_coefficients, coeffs, 0, axis, xp=xp)

    @cached_property
    def partial_trace_map(self) -> sp.csr_matrix:
        """
        Returns the sparse matrix P implementing the partial trace Tr_{2...n}(·).

        P maps a vector of coefficients in the n-copy Sn-orbit basis to a vector
        representing the d×d single-copy matrix (flattened in row-major order).

        The output uses the standard matrix basis {|i><j|} for i,j in [0, d-1],
        so the result is a d² dimensional vector where index i*d + j corresponds to |i><j|.

        Shape: (d², m) where m = self.size() is the number of orbits.

        Usage:
            rho_AB_vec = P @ rho_n_orbit_coeffs
            rho_AB = rho_AB_vec.reshape((d, d))
        """
        with MaybeExpensiveComputation(f"Constructing partial_trace_map for {str(self)}"):
            m = self.size()
            d = self.d

            rows_P = []
            cols_P = []
            data_P = []

            for j, orbit in enumerate(self.iterate_labels()):
                # Use the existing partial_trace_to_single method on PairOrbit
                pt_matrix = orbit.partial_trace_to_single()

                # pt_matrix is d×d, flatten to d² in row-major order
                for i in range(d):
                    for k in range(d):
                        val = pt_matrix[i, k]
                        if val != 0:
                            row_idx = i * d + k  # Flattened index for |i><k|
                            rows_P.append(row_idx)
                            cols_P.append(j)
                            data_P.append(val)

            return sp.csr_matrix((data_P, (rows_P, cols_P)), shape=(d * d, m))

    def norm(self, coeffs: np.ndarray, axis=-1):
        """
        Returns the coefficients corresponding to the HS-norm of the matrix which is the linear combination of the given coefficients.
        :param coeffs:
        :param axis: Axis which takes the coefficients. All other axes are ignored.
        :return: Array of shape similar to coeffs where axis='axis' is replaced with the HS-norm of the corresponding matrix
        """
        return contract_at_axis(self.norm_coefficients, coeffs, 0, axis, xp=self.xp)

    @cached_property
    def hermitian_subset_embedding(self) -> TransitionMatrix:
        """Returns a TransitionMatrix that embeds the subset of this basis corresponding to hermitian matrices into the full basis.
        In particular matrix-multiplying this transition matrix with any real coefficient vector will give a hermitian matrix
        Since there is no good way to capture this in our framework, the basis_from attribute of the returned transitionmatrix is inaccurate.
        """
        upper_triangular_indices = MatrixEntryMask(
            index_pairs=tuple(zip(*np.triu_indices(self.d))), shape=(self.d, self.d)
        )
        upper_triangular_subset = EndSnOrbitBasisSubset(self, OrbitFitsMaskPredicate(upper_triangular_indices))

        identity_part = upper_triangular_subset.index_mapping.as_transition_matrix().to_scipy_sparse()
        transposed_part = (
            self.transpose_map()
            .concatenate(upper_triangular_subset.index_mapping)
            .as_transition_matrix()
            .to_scipy_sparse()
        )
        real_part = identity_part + transposed_part  # TODO: maybe remove double counting the diagonal here
        imaginary_part = 1j * identity_part + -1j * transposed_part

        # Giving upper_triangular_subset as basis_from here is wrong, but this is also a bit awkward, because we really now want to view this as a real vector space
        # ynd we have nothing to capture this notion of real coefficient only basis
        return GivenTransitionMatrix(
            upper_triangular_subset, self, scipy.sparse.vstack([real_part, imaginary_part], format="csr")
        )


@dataclasses.dataclass(unsafe_hash=True)
class EndSnIrrepBasis(MatrixBasis[tuple[SSYT, SSYT]]):
    """The canonical (i.e. only one non-zero entry in the matrix) basis for one m_λxm_λ block corresponding to one irreducible representation of End^(S_n)(V^n).
    We label the basis elements [0, ..., m_Λ - 1] by their semistandard young tableaux. This gives the same matrices as MatrixStandardBasis, but
    the labels of the matrix entries are SSYTs instead of just indices.
    """

    partition: Partition
    single_dimension: int
    dtype: dtypeT = dataclasses.field(default=np.int_, kw_only=True, repr=False)

    def __post_init__(self):
        self.m = SSYT.count(self.partition, self.single_dimension)
        self._matrix_standard_basis = MatrixStandardBasis(self.m, xp=self.xp, dtype=self.dtype)

    def iterate_labels(self):
        return itertools.product(SSYT.generate_all(self.partition, self.single_dimension), repeat=2)

    def label_to_vector(self, label: Tuple[SSYT, SSYT]):
        standard_indices = tuple(SSYT.index_of(s, self.single_dimension) for s in label)
        return self._matrix_standard_basis.label_to_vector(standard_indices)

    def label_to_index(self, label: Tuple[SSYT, SSYT]):
        standard_indices = tuple(SSYT.index_of(s, self.single_dimension) for s in label)
        return self._matrix_standard_basis.label_to_index(standard_indices)

    def size(self) -> int:
        return self.m**2

    def label_at_index(self, idx: int) -> tuple[SSYT, SSYT]:
        standard_indices = self._matrix_standard_basis.label_at_index(idx)
        return tuple(SSYT.tableau_at_index(self.partition, self.single_dimension, s) for s in standard_indices)  # type: ignore

    def vector_at_index(self, idx: int) -> np.ndarray:
        return self._matrix_standard_basis.vector_at_index(idx)

    def all_vectors(self) -> np.ndarray:
        return self._matrix_standard_basis.all_vectors()

    def iterate_vectors(self) -> Iterable[np.ndarray]:
        return self._matrix_standard_basis.iterate_vectors()

    def linear_combination(self, coeffs: np.ndarray, *, coeff_axis=-1) -> DenseOrSparse:
        return self._matrix_standard_basis.linear_combination(coeffs, coeff_axis=coeff_axis)

    def __str__(self):
        return f"EndSnIrrepBasis(d={self.single_dimension}, partition={self.partition})"

    def __repr__(self):
        return str(self)

    def transpose(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        return self._matrix_standard_basis.transpose(coeffs, axis=axis)

    def trace(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        return self._matrix_standard_basis.trace(coeffs, axis=axis)

    @property
    def dimension(self) -> int:
        return self.m


@dataclasses.dataclass(unsafe_hash=True)
class MatrixDirectSumBasis[LabelT](MatrixBasis[LabelT]):
    """Represents the direct sum of a list of bases"""

    bases: tuple[MatrixBasis[LabelT], ...]

    def __post_init__(self):
        assert len(set(b.xp for b in self.bases)) == 1, f"All bases must have the same xp, but got {self.bases}"
        self.xp = self.bases[0].xp

    @cached_property
    def basis_sizes(self) -> list[int]:
        """Returns the size of each basis in the direct sum."""
        return [b.size() for b in self.bases]

    @cached_property
    def block_sizes(self) -> tuple[int, ...]:
        """Returns the dimension (i.e. the side-length of the matrix) of each block in the direct sum."""
        return tuple(b.dimension for b in self.bases)

    @cached_property
    def basis_indices_start(self) -> np.ndarray:
        """Returns the start index of each sub-basis in the total sequence of basis vectors."""
        return np.cumsum([0] + self.basis_sizes[:-1])

    @property
    def number_of_blocks(self) -> int:
        return len(self.bases)

    def size(self) -> int:
        return sum(b.size() for b in self.bases)

    def iterate_labels(self) -> Iterable[LabelT]:
        for basis in self.bases:
            yield from basis.iterate_labels()

    def iterate_vectors(self) -> Iterable[BlockDiagonalMatrix]:
        for basis_idx, basis in enumerate(self.bases):
            for v in basis.iterate_vectors():
                yield self._process_vector(v, basis, basis_idx)

    def all_vectors(self) -> np.ndarray:
        return self.xp.stack(list(v.to_full_matrix() for v in self.iterate_vectors()))

    def _process_vector(self, vector: np.ndarray, basis: Basis, basis_index: int) -> BlockDiagonalMatrix:
        i = basis_index
        num_blocks = len(self.block_sizes)
        return BlockDiagonalMatrix(
            (np.array(0),) * i + (vector,) + (np.array(0),) * (num_blocks - i - 1), tuple(self.block_sizes)
        )

    def _label_to_subbasis_index(self, label: LabelT) -> int:
        raise NotImplementedError()

    def label_to_vector(self, label: LabelT) -> BlockDiagonalMatrix:
        idx = self._label_to_subbasis_index(label)
        basis = self.bases[idx]
        return self._process_vector(basis.label_to_vector(label), basis, idx)

    def label_to_index(self, label: LabelT) -> int:
        basis_idx = self._label_to_subbasis_index(label)
        basis = self.bases[basis_idx]
        return basis.label_to_index(label) + self.basis_indices_start[basis_idx]  # type: ignore

    def label_at_index(self, idx: int) -> LabelT:
        basis_idx = np.searchsorted(self.basis_indices_start, idx, side="right") - 1
        basis = self.bases[basis_idx]
        return basis.label_at_index(idx - self.basis_indices_start[basis_idx])

    def vector_at_index(self, idx: int) -> BlockDiagonalMatrix:
        basis_idx = np.searchsorted(self.basis_indices_start, idx, side="right") - 1
        basis = self.bases[basis_idx]
        return self._process_vector(basis.vector_at_index(idx - self.basis_indices_start[basis_idx]), basis, basis_idx)  # type: ignore

    def split_coefficients(self, coeffs: np.ndarray, coeff_axis=-1):
        xp = array_namespace(coeffs)
        return xp.split(coeffs, self.basis_indices_start[1:], axis=coeff_axis)

    def subcoefficients_with_bases(self, coeffs: np.ndarray, coeff_axis=-1) -> Iterable[tuple[np.ndarray, Basis]]:
        return zip(self.split_coefficients(coeffs, coeff_axis=coeff_axis), self.bases)

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> BlockDiagonalMatrix:
        assert len(coeffs.shape) == 1, (
            f"Linear combination of block diagonal matrices currently only supported for one-dimensional arrays"
        )
        block_coeffs = self.split_coefficients(coeffs, coeff_axis=coeff_axis)
        block_matrices = [basis.linear_combination(coeffs) for basis, coeffs in zip(self.bases, block_coeffs)]
        return BlockDiagonalMatrix(block_matrices, tuple(self.block_sizes))

    def transpose(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        block_coeffs = self.split_coefficients(coeffs, coeff_axis=axis)
        transposed_blocks = [basis.transpose(coeffs, axis=axis) for basis, coeffs in zip(self.bases, block_coeffs)]
        return np.concatenate(transposed_blocks, axis=axis)

    def trace(self, coeffs: np.ndarray, axis=-1) -> np.ndarray | float:
        block_coeffs = self.split_coefficients(coeffs, coeff_axis=axis)
        traced_blocks = [basis.trace(coeffs, axis=axis) for basis, coeffs in zip(self.bases, block_coeffs)]
        return sum(traced_blocks)

    @cached_property
    def partial_trace_map(self) -> sp.csr_matrix:
        """
        Block-diagonal partial trace map.

        If each sub-basis B_i provides a partial_trace_map
            P_i : C^{m_i} -> C^{d_i},

        then the direct-sum basis implements
            P = ⊕_i P_i.
        """
        blocks = []

        for basis in self.bases:
            if not hasattr(basis, "partial_trace_map"):
                raise NotImplementedError(f"Basis {basis} does not implement partial_trace_map")
            blocks.append(basis.partial_trace_map)

        return sp.block_diag(blocks, format="csr")

    @property
    def dimension(self) -> int:
        return sum(self.block_sizes)


class EndSnBlockDiagonalBasis(MatrixDirectSumBasis[tuple[SSYT, SSYT]]):
    bases: tuple[EndSnIrrepBasis, ...]

    def __init__(self, n: int, d: int, *, xp=np, dtype=np.int_):
        self.n, self.d, self.xp, self.dtype = n, d, xp, dtype
        self.partitions = list(Partition.generate_all(n, max_length=d))
        bases = tuple(EndSnIrrepBasis(partition, d, xp=xp, dtype=dtype) for partition in self.partitions)
        super().__init__(bases=bases)

    def _label_to_subbasis_index(self, label: tuple[SSYT, SSYT]):
        t1, t2 = label
        assert t1.partition == t2.partition
        return self.partitions.index(t1.partition)

    @caching.cache
    def partial_transpose_operator(self, iso, system: str = "B") -> np.ndarray:
        """
        Returns the linear operator matrix T that implements the Partial Transpose
        in the flattened block basis.

        Since PT in the block basis mixes irreps, this returns a matrix T such that:
            sigma_pt_block_coeffs = T @ sigma_block_coeffs

        The formula is: T = T_iso @ P_orbit @ T_iso^{-1}
        where T_iso is the isomorphism from orbit to block basis,
        and P_orbit is the permutation matrix for PT in orbit basis.

        Args:
            iso: The EndSnAlgebraIsomorphism instance linking Orbit and Block bases.
            system: 'A' or 'B', the subsystem to transpose.
        """
        with ExpensiveComputation(f"Constructing Block Basis Partial Transpose Operator"):
            orbit_basis = iso.basis_from
            pt_indices_orbit = orbit_basis.partial_transpose_index_lookup(system)
            m_total = self.size()

            # Get the isomorphism matrices
            T_iso = iso.to_scipy_sparse().toarray()
            T_iso_inv = iso.inverse().to_scipy_sparse().toarray()

            # Build the permutation matrix P_orbit where P @ sigma = sigma[pt_indices]
            # i.e., P[i, pt_indices[i]] = 1
            P_orbit = np.zeros((m_total, m_total), dtype=np.complex128)
            for i in range(m_total):
                P_orbit[i, pt_indices_orbit[i]] = 1

            # T_ppt = T_iso @ P_orbit @ T_iso_inv
            T_matrix = T_iso @ P_orbit @ T_iso_inv

            return T_matrix

    @caching.cache
    def partial_trace_operator(self, iso) -> np.ndarray:
        """
        Returns the linear operator P that implements the Partial Trace
        in the flattened block basis.

        Maps:
            sigma_block_coeffs  ->  vec( Tr_{2..n}(sigma) )

        Shape:
            (d^2, m_total)
        """
        with ExpensiveComputation("Constructing Block Basis Partial Trace Operator"):
            orbit_basis = iso.basis_from

            # Orbit-basis partial trace: (d^2, m_orbit)
            P_orbit = orbit_basis.partial_trace_map

            # Isomorphism matrices
            T_iso_inv = iso.inverse().to_scipy_sparse()

            # Block-basis partial trace
            # P_block = P_orbit @ T_iso^{-1}
            P_block = P_orbit @ T_iso_inv

            return P_block.toarray()


@dataclasses.dataclass(unsafe_hash=True)
class EndSnOrbitBasisSubset(MatrixBasisSubset[PairOrbit]):
    subset_of: EndSnOrbitBasis

    @cached_property
    def valid_indices(self) -> np.ndarray:
        """Override with fast direct enumeration when the predicate is an OrbitFitsMaskPredicate.

        The generic BasisSubset.valid_indices iterates all C(n+d²-1, d²-1) orbits via
        BasisSubsetIndices.  When the predicate is an OrbitFitsMaskPredicate with |S|
        supported entries, enumerate only the C(n+|S|-1, |S|-1) valid orbits directly,
        which is orders of magnitude faster for large n and small support.
        """
        if self.from_valid_indices is not None:
            return self.from_valid_indices
        predicate = getattr(self.predicate, '__func__', self.predicate)
        if isinstance(predicate, OrbitFitsMaskPredicate):
            n, d = self.subset_of.n, self.subset_of.d
            support = sorted(predicate.mask.index_pairs)
            k = len(support)
            indices = []
            for comp in weak_compositions(n, k):
                count_matrix = np.zeros((d, d), dtype=np.int_)
                for (i, j), c in zip(support, comp):
                    count_matrix[i, j] = c
                indices.append(weak_composition_to_index(n, d**2, count_matrix.ravel()))
            return np.sort(np.asarray(indices, dtype=np.intp))
        return self.index_mapping.index_mapping().as_numpy()

    def coefficients_for_tensor_product(self, matrix: np.ndarray) -> np.ndarray:
        """For a given dxd matrix A, returns the coefficients of the matrix A^n (i.e. the n-fold tensor product of A with itself) in this basis.
        Does not validate that the given matrix only has valid entries, only valid entries from matrix will be considered.
        """
        assert matrix.shape == (self.subset_of.d, self.subset_of.d)
        return np.fromiter(
            (np.prod(matrix**orbit.count_matrix) for orbit in self.iterate_labels()),
            dtype=matrix.dtype,
            count=self.size(),
        )

    def count_matrix_to_index(self, count_matrix: np.ndarray) -> int:
        full_idx = self.subset_of.count_matrix_to_index(count_matrix)
        masked_idx = int(self.inverse_index_mapping.index_mapping()[full_idx])
        if masked_idx == -1:
            raise KeyError(f"Count Matrix {count_matrix} is not present in this masked basis.")
        return masked_idx

    @cached_property
    def all_count_matrices(self) -> np.ndarray:
        return np.stack(list(self.iterate_count_matrices()))

    def iterate_count_matrices(self) -> Iterable[np.ndarray]:
        full = self.subset_of
        for full_idx in self.valid_indices:
            yield full.count_matrix_at_index(int(full_idx))

    @cached_property
    def norm_coefficients(self) -> np.ndarray:
        with ExpensiveComputation(f"Generating HS norms of indicator matrices for {str(self)}"):
            return np.fromiter(
                (m.indicator_matrix.HS_norm for m in self.iterate_labels()), dtype=np.int_, count=self.size()
            )

    @property
    def d(self):
        return self.subset_of.d

    @property
    def n(self):
        return self.subset_of.n

    def intersect_with_valid_indices(self, other_valid_indices: Sequence[int] | np.ndarray):
        return type(self)(
            subset_of=self.subset_of,
            from_valid_indices=np.sort(np.asarray(set(self.valid_indices).intersection(set(other_valid_indices)))),
        )

    @cached_property
    def symmetric_subset(self) -> EndSnOrbitBasisSubset:
        """Returns an additional subset of this subset which only includes count_matrices whose transpose is also in the set.
        This means that the subset this function returns is sufficient to capture all symmetric/hermitian matrices which live on the original subset.
        """
        valid_transpose_indices = self.subset_of.transpose_index_lookup()[self.valid_indices]
        return self.intersect_with_valid_indices(valid_transpose_indices)

    @cached_property
    def hermitian_subset_embedding(self) -> TransitionMatrix:
        """Returns a TransitionMatrix that embeds the subset of this subset corresponding to hermitian matrices into the full basis.
        In particular matrix-multiplying this transition matrix with any real coefficient vector will give a hermitian matrix.
        Since there is no good way to capture this in our framework, the basis_from attribute of the returned transitionmatrix is inaccurate.
        """
        strictly_upper_triangular_indices = MatrixEntryMask(
            index_pairs=tuple(zip(*np.triu_indices(self.d, k=1))), shape=(self.d, self.d)
        )
        strictly_upper_triangular_subset = EndSnOrbitBasisSubset(
            self.subset_of, OrbitFitsMaskPredicate(strictly_upper_triangular_indices)
        )
        intersection_subset = self.intersect_with_valid_indices(strictly_upper_triangular_subset.valid_indices)

        diagonal_subset = self.intersect_with_valid_indices(
            EndSnOrbitBasisSubset(
                self.subset_of,
                OrbitFitsMaskPredicate(MatrixEntryMask((self.d, self.d), tuple((i, i) for i in range(self.d)))),
            ).valid_indices
        )

        diagonal_part = diagonal_subset.index_mapping.as_transition_matrix().to_scipy_sparse()
        identity_part = intersection_subset.index_mapping.as_transition_matrix().to_scipy_sparse()
        transposed_part = self.transpose(
            intersection_subset.index_mapping.as_transition_matrix().to_scipy_sparse(), axis=0
        )
        real_part = identity_part + transposed_part
        imaginary_part = 1j * identity_part + -1j * transposed_part

        # Giving self as basis_from here is wrong, but this is also a bit awkward, because we really now want to view this as a real vector space
        # ynd we have nothing to capture this notion of real coefficient only basis
        return GivenTransitionMatrix(self, self, scipy.sparse.vstack([diagonal_part, real_part, imaginary_part], format="csr"))

    # TODO: benchmark and optimize iterate_labels()
