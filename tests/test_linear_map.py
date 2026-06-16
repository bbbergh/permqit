"""Tests for algebra.linear_map: InjectiveIndexMapping, SurjectiveIndexMapping, and TransitionMatrix.

Covered:
- InjectiveIndexMapping.apply_to_coefficient_vector – scatter across all backends / axes.
- InjectiveIndexMapping.as_transition_matrix() – matrix correctness and apply equivalence.
- InjectiveIndexMapping.inverse() – returns InjectiveScatterMappingInverse; round-trips; matrix
  is transpose of forward for permutations.
- SurjectiveIndexMapping.apply_to_coefficient_vector – gather across all backends / axes,
  including repeated source indices (true projection case).
- SurjectiveIndexMapping.as_transition_matrix() – shape, content, caching.
- SurjectiveIndexMapping.inverse() – returns SurjectiveMappingInverse; round-trips;
  matrix is transpose of forward for permutations.
- TransitionMatrix format conversions – to_numpy(), to_scipy_sparse(), and
  to_pydata_sparse() must all describe the same linear map; MatrixCache lazy-converts
  and (optionally) retains the result.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import sparse
import pytest

from permqit.algebra.basis import VectorStandardBasis
from permqit.algebra.linear_map import (
    GivenTransitionMatrix,
    ScatterIndexMapping,
    MatrixCache,
    StorageFormat,
    GatherIndexMapping,
    TransitionMatrix, GivenGatherIndexMapping,
)


# ---------------------------------------------------------------------------
# Helpers / concrete test doubles
# ---------------------------------------------------------------------------

class FixedIndexMapping(ScatterIndexMapping):
    """Concrete IndexMapping whose permutation is given at construction time.

    Parameters
    ----------
    perm : array-like of int
        ``perm[i]`` is the *output* index that input index ``i`` maps to.
        In other words ``output[perm[i]] = input[i]``.
    n_from : int
        Size of the source basis.
    n_to : int
        Size of the target basis (>= max(perm)+1).
    """

    def __init__(self, perm: list[int], n_from: int, n_to: int):
        self.basis_from = VectorStandardBasis(n_from)
        self.basis_to = VectorStandardBasis(n_to)
        self._perm = np.asarray(perm, dtype=int)

    def _calculate_index_mapping(self) -> np.ndarray:
        return self._perm



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity_perm(n: int) -> FixedIndexMapping:
    """Identity permutation on n elements."""
    return FixedIndexMapping(list(range(n)), n, n)


def _reversal_perm(n: int) -> FixedIndexMapping:
    """Reversal permutation: maps index i → (n-1-i)."""
    return FixedIndexMapping(list(range(n - 1, -1, -1)), n, n)


def _cycle_perm(n: int) -> FixedIndexMapping:
    """Cyclic shift: maps index i → (i+1) % n."""
    return FixedIndexMapping([(i + 1) % n for i in range(n)], n, n)


def _embed_perm(n_from: int, n_to: int) -> FixedIndexMapping:
    """Embedding: maps index i → i (n_from < n_to, so not surjective)."""
    return FixedIndexMapping(list(range(n_from)), n_from, n_to)


# ---------------------------------------------------------------------------
# Tests: IndexMapping.as_transition_matrix()
# ---------------------------------------------------------------------------

class TestIndexMappingAsTransitionMatrix:
    """Verify that as_transition_matrix() produces a GivenTransitionMatrix that
    is mathematically equivalent to the original IndexMapping."""


    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_basis_attributes_preserved(self, n):
        iso = _reversal_perm(n)
        tm = iso.as_transition_matrix()
        assert tm.basis_from is iso.basis_from
        assert tm.basis_to is iso.basis_to

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_apply_matches_index_mapping(self, n):
        """Applying the transition matrix reproduces the same output as the
        IndexMapping.apply_to_coefficient_vector()."""
        iso = _reversal_perm(n)
        tm = iso.as_transition_matrix()
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            expected = iso.apply_to_coefficient_vector(x)
            actual = tm.apply_to_coefficient_vector(x)
            np.testing.assert_allclose(actual.todense(), expected.todense(), atol=1e-12)

    def test_identity_gives_identity_matrix(self):
        n = 4
        iso = _identity_perm(n)
        tm = iso.as_transition_matrix()
        mat = tm.to_numpy()
        np.testing.assert_array_equal(mat, np.eye(n, dtype=int))

    def test_cycle_matrix_structure(self):
        """For cyclic shift by 1, the transition matrix has exactly one 1 per row/column."""
        n = 5
        iso = _cycle_perm(n)
        tm = iso.as_transition_matrix()
        mat = tm.to_numpy()
        assert mat.shape == (n, n)
        np.testing.assert_array_equal(mat.sum(axis=0), np.ones(n))
        np.testing.assert_array_equal(mat.sum(axis=1), np.ones(n))

    def test_embed_shape(self):
        """Non-square (embedding) case: matrix shape is (n_to, n_from)."""
        n_from, n_to = 3, 6
        iso = _embed_perm(n_from, n_to)
        tm = iso.as_transition_matrix()
        mat = tm.to_numpy()
        assert mat.shape == (n_to, n_from)

    @pytest.mark.parametrize("n", [3, 5])
    def test_result_is_cached(self, n):
        """Calling as_transition_matrix() twice returns the same object."""
        iso = _reversal_perm(n)
        assert iso.as_transition_matrix() is iso.as_transition_matrix()


# ---------------------------------------------------------------------------
# Tests: IndexMapping.inverse()
# ---------------------------------------------------------------------------

class TestIndexMappingInverse:
    """Verify that inverse() returns the mathematical inverse of the IndexMapping."""


    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_has_swapped_bases(self, n):
        iso = _reversal_perm(n)
        inv = iso.inverse()
        assert inv.basis_from is iso.basis_to
        assert inv.basis_to is iso.basis_from

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_round_trip_coefficient_vector(self, n):
        """iso followed by its inverse should recover the original coefficients."""
        iso = _cycle_perm(n)
        inv = iso.inverse()
        rng = np.random.default_rng(1)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            y = iso.apply_to_coefficient_vector(x)
            recovered = inv.apply_to_coefficient_vector(y)
            np.testing.assert_allclose(recovered.todense(), x.todense(), atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_round_trip_coefficient_vector(self, n):
        """Applying the inverse first, then the forward map, also recovers the input."""
        iso = _reversal_perm(n)
        inv = iso.inverse()
        rng = np.random.default_rng(2)
        for _ in range(5):
            # Apply inverse first (input lives in basis_to)
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            w = inv.apply_to_coefficient_vector(x)
            recovered = iso.apply_to_coefficient_vector(w)
            np.testing.assert_allclose(recovered.todense(), x.todense(), atol=1e-12)

    def test_identity_inverse_is_identity(self):
        n = 4
        iso = _identity_perm(n)
        inv = iso.inverse()
        mat = inv.as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(mat, np.eye(n, dtype=int))

    @pytest.mark.parametrize("n", [3, 5])
    def test_inverse_cached(self, n):
        iso = _reversal_perm(n)
        assert iso.inverse() is iso.inverse()

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_matrix_is_transpose_of_forward(self, n):
        """For a permutation matrix P, P^{-1} = P^T."""
        iso = _cycle_perm(n)
        fwd = iso.as_transition_matrix().to_numpy()
        inv_mat = iso.inverse().as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(inv_mat, fwd.T)


# ---------------------------------------------------------------------------
# Tests: TransitionMatrix format conversions
# ---------------------------------------------------------------------------

class TestTransitionMatrixFormatConversions:
    """Verify that coefficient_transition_matrix() returns numerically identical
    content in all supported CPU formats, and that MatrixCache behaves correctly."""

    def _make_random_dense_tm(self, n: int, seed: int = 0) -> GivenTransitionMatrix:
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((n, n))
        b = VectorStandardBasis(n)
        return GivenTransitionMatrix(b, b, mat)

    def _make_sparse_tm(self, n: int) -> GivenTransitionMatrix:
        """Transition matrix with only a few non-zeros."""
        perm_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            perm_mat[i, (i + 1) % n] = 1.0
        sp_mat = sp.csr_matrix(perm_mat)
        b = VectorStandardBasis(n)
        return GivenTransitionMatrix(b, b, sp_mat)

    # -- numpy primary format ----------------------------------------------

    @pytest.mark.parametrize("n", [4, 6])
    def test_numpy_to_scipy_sparse(self, n):
        tm = self._make_random_dense_tm(n)
        dense = tm.to_numpy()
        csr = tm.to_scipy_sparse()
        assert sp.issparse(csr)
        np.testing.assert_allclose(csr.toarray(), dense, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6])
    def test_numpy_to_pydata_sparse(self, n):
        tm = self._make_random_dense_tm(n)
        dense = tm.to_numpy()
        ps = tm.to_pydata_sparse()
        assert isinstance(ps, sparse.SparseArray)
        np.testing.assert_allclose(ps.todense(), dense, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6])
    def test_all_cpu_formats_agree(self, n):
        tm = self._make_random_dense_tm(n)
        dense = tm.to_numpy()
        csr_arr = tm.to_scipy_sparse().toarray()
        pydata_arr = tm.to_pydata_sparse().todense()
        np.testing.assert_allclose(csr_arr, dense, atol=1e-12)
        np.testing.assert_allclose(pydata_arr, dense, atol=1e-12)

    # -- scipy sparse primary format ---------------------------------------

    @pytest.mark.parametrize("n", [4, 6])
    def test_scipy_primary_to_numpy(self, n):
        tm = self._make_sparse_tm(n)
        sparse_mat = tm.to_scipy_sparse()
        dense = tm.to_numpy()
        np.testing.assert_allclose(dense, sparse_mat.toarray(), atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6])
    def test_scipy_primary_to_pydata(self, n):
        tm = self._make_sparse_tm(n)
        csr = tm.to_scipy_sparse()
        pydata = tm.to_pydata_sparse()
        np.testing.assert_allclose(pydata.todense(), csr.toarray(), atol=1e-12)

    # -- pydata/sparse primary format -------------------------------------

    @pytest.mark.parametrize("n", [4, 6])
    def test_pydata_primary_to_numpy(self, n):
        b = VectorStandardBasis(n)
        eye = sparse.eye(n, format="gcxs")
        tm = GivenTransitionMatrix(b, b, eye)
        dense = tm.to_numpy()
        np.testing.assert_allclose(dense, np.eye(n), atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6])
    def test_pydata_primary_to_scipy(self, n):
        b = VectorStandardBasis(n)
        eye = sparse.eye(n, format="gcxs")
        tm = GivenTransitionMatrix(b, b, eye)
        csr = tm.to_scipy_sparse()
        assert sp.issparse(csr)
        np.testing.assert_allclose(csr.toarray(), np.eye(n), atol=1e-12)

    # -- MatrixCache caching behaviour ------------------------------------

    def test_primary_format_detected_numpy(self):
        tm = self._make_random_dense_tm(4)
        tm.to_numpy()  # trigger computation
        assert tm.matrix_cache is not None
        assert tm.matrix_cache.primary_format == StorageFormat.NUMPY

    def test_primary_format_detected_scipy(self):
        tm = self._make_sparse_tm(4)
        tm.to_scipy_sparse()  # trigger computation
        assert tm.matrix_cache is not None
        assert tm.matrix_cache.primary_format == StorageFormat.SCIPY_SPARSE

    def test_derived_format_cached_after_first_request(self):
        tm = self._make_random_dense_tm(4)
        tm.to_numpy()
        # Ask for scipy format and enable caching
        tm.matrix_cache.set_cache_formats(StorageFormat.NUMPY, StorageFormat.SCIPY_SPARSE)
        tm.to_scipy_sparse()
        assert StorageFormat.SCIPY_SPARSE in tm.matrix_cache.cached_formats

    def test_derived_format_not_cached_when_not_in_cache_formats(self):
        tm = self._make_random_dense_tm(4)
        tm.to_numpy()  # primary = NUMPY, cache_formats = {NUMPY}
        # SCIPY_SPARSE is not in cache_formats by default
        _ = tm.to_scipy_sparse()
        assert StorageFormat.SCIPY_SPARSE not in tm.matrix_cache.cached_formats

    def test_evict_non_primary_format(self):
        tm = self._make_random_dense_tm(4)
        cache = MatrixCache(
            np.eye(4),
            cache_formats={StorageFormat.NUMPY, StorageFormat.SCIPY_SPARSE},
        )
        cache.get(StorageFormat.SCIPY_SPARSE)
        assert StorageFormat.SCIPY_SPARSE in cache.cached_formats
        cache.evict(StorageFormat.SCIPY_SPARSE)
        assert StorageFormat.SCIPY_SPARSE not in cache.cached_formats

    def test_cannot_evict_primary_format(self):
        cache = MatrixCache(np.eye(4))
        with pytest.raises(ValueError, match="primary"):
            cache.evict(StorageFormat.NUMPY)

    def test_precompute_caches_formats(self):
        cache = MatrixCache(np.eye(4))
        cache.precompute(StorageFormat.SCIPY_SPARSE, StorageFormat.PYDATA_SPARSE)
        assert StorageFormat.SCIPY_SPARSE in cache.cached_formats
        assert StorageFormat.PYDATA_SPARSE in cache.cached_formats

    # -- apply_to_coefficient_vector correctness --------------------------

    @pytest.mark.parametrize("n", [4, 6])
    def test_apply_sparse_input_matches_numpy_matmul(self, n):
        """TransitionMatrix.apply_to_coefficient_vector with a pydata/sparse input
        must give the same result as dense matrix-vector multiplication."""
        tm = self._make_random_dense_tm(n)
        mat = tm.to_numpy()
        rng = np.random.default_rng(42)
        x_dense = rng.standard_normal(n)
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = tm.apply_to_coefficient_vector(x_sparse)
        expected = mat @ x_dense
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: IndexMapping.apply_to_coefficient_vector – non-zero axis
# ---------------------------------------------------------------------------

class TestIndexMappingApplyNonZeroAxis:
    """Verify apply_to_coefficient_vector works for axis != 0 across all
    backends (numpy, pydata/sparse, scipy.sparse) and edge cases."""

    # ------------------------------------------------------------------
    # Reference helper
    # ------------------------------------------------------------------
    @staticmethod
    def _reference(x: np.ndarray, perm: np.ndarray|MatrixCache, axis: int, n_to: int) -> np.ndarray:
        """Brute-force reference: scatter x along `axis` according to perm."""
        if isinstance(perm, MatrixCache):
            perm = perm.as_numpy()

        out_shape = list(x.shape)
        out_shape[axis] = n_to
        out = np.zeros(out_shape, dtype=x.dtype)
        idx_src: list[slice|int] = [slice(None)] * x.ndim
        idx_dst: list[slice|int] = [slice(None)] * x.ndim
        for i, j in enumerate(perm):
            idx_src[axis] = i
            idx_dst[axis] = j
            out[tuple(idx_dst)] = x[tuple(idx_src)]
        return out

    # ------------------------------------------------------------------
    # numpy – axis 1 on a 2D array
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_numpy_axis1_matches_reference(self, n):
        iso = _reversal_perm(n)
        rng = np.random.default_rng(10)
        x = rng.standard_normal((4, n))       # shape (rows, n)
        result = iso.apply_to_coefficient_vector(x, axis=1)
        expected = self._reference(x, iso.index_mapping(), axis=1, n_to=n)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (4, n)

    # ------------------------------------------------------------------
    # numpy – axis 0 still correct (regression check)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_numpy_axis0_unchanged(self, n):
        iso = _cycle_perm(n)
        rng = np.random.default_rng(11)
        x = rng.standard_normal(n)
        result = iso.apply_to_coefficient_vector(x, axis=0)
        expected = self._reference(x, iso.index_mapping(), axis=0, n_to=n)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # numpy – negative axis
    # ------------------------------------------------------------------
    def test_numpy_negative_axis(self):
        n = 4
        iso = _reversal_perm(n)
        rng = np.random.default_rng(12)
        x = rng.standard_normal((3, n))
        result_neg = iso.apply_to_coefficient_vector(x, axis=-1)
        result_pos = iso.apply_to_coefficient_vector(x, axis=1)
        np.testing.assert_allclose(result_neg, result_pos, atol=1e-12)

    # ------------------------------------------------------------------
    # numpy – 3D tensor, axis 2
    # ------------------------------------------------------------------
    def test_numpy_3d_axis2(self):
        n = 4
        iso = _cycle_perm(n)
        rng = np.random.default_rng(13)
        x = rng.standard_normal((2, 3, n))
        result = iso.apply_to_coefficient_vector(x, axis=2)
        expected = self._reference(x, iso.index_mapping(), axis=2, n_to=n)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (2, 3, n)

    # ------------------------------------------------------------------
    # numpy – embedding (n_from < n_to), axis 1
    # ------------------------------------------------------------------
    def test_numpy_embed_axis1(self):
        n_from, n_to = 3, 6
        iso = _embed_perm(n_from, n_to)
        rng = np.random.default_rng(14)
        x = rng.standard_normal((4, n_from))
        result = iso.apply_to_coefficient_vector(x, axis=1)
        expected = self._reference(x, iso.index_mapping(), axis=1, n_to=n_to)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (4, n_to)

    # ------------------------------------------------------------------
    # pydata/sparse – axis 1 on a 2D sparse array
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_pydata_sparse_axis1(self, n):
        iso = _reversal_perm(n)
        rng = np.random.default_rng(20)
        x_dense = rng.standard_normal((4, n))
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = iso.apply_to_coefficient_vector(x_sparse, axis=1)
        expected = self._reference(x_dense, iso.index_mapping(), axis=1, n_to=n)
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # pydata/sparse – 1D (axis 0)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_pydata_sparse_axis0(self, n):
        iso = _cycle_perm(n)
        rng = np.random.default_rng(21)
        x_dense = rng.standard_normal(n)
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = iso.apply_to_coefficient_vector(x_sparse, axis=0)
        expected = self._reference(x_dense, iso.index_mapping(), axis=0, n_to=n)
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # scipy.sparse – axis 0 (row reorder)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_scipy_sparse_axis0(self, n):
        iso = _reversal_perm(n)
        rng = np.random.default_rng(30)
        x_dense = rng.standard_normal((n, n))
        x_sp = sp.csr_matrix(x_dense)
        result = iso.apply_to_coefficient_vector(x_sp, axis=0)
        expected = self._reference(x_dense, iso.index_mapping(), axis=0, n_to=n)
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # scipy.sparse – axis 1 (column reorder)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [3, 5])
    def test_scipy_sparse_axis1(self, n):
        iso = _reversal_perm(n)
        rng = np.random.default_rng(31)
        x_dense = rng.standard_normal((n, n))
        x_sp = sp.csr_matrix(x_dense)
        result = iso.apply_to_coefficient_vector(x_sp, axis=1)
        expected = self._reference(x_dense, iso.index_mapping(), axis=1, n_to=n)
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # All three backends agree for axis 1
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("n", [4, 6])
    def test_all_backends_agree_axis1(self, n):
        iso = _cycle_perm(n)
        rng = np.random.default_rng(40)
        x_dense = rng.standard_normal((n, n))

        r_numpy = iso.apply_to_coefficient_vector(x_dense, axis=1)
        r_sparse = iso.apply_to_coefficient_vector(
            sparse.COO.from_numpy(x_dense), axis=1
        ).todense()
        r_scipy = iso.apply_to_coefficient_vector(
            sp.csr_matrix(x_dense), axis=1
        ).toarray()

        np.testing.assert_allclose(r_sparse, r_numpy, atol=1e-12)
        np.testing.assert_allclose(r_scipy, r_numpy, atol=1e-12)


# ---------------------------------------------------------------------------
# Helpers for SurjectiveIndexMapping tests
# ---------------------------------------------------------------------------

def FixedGatherIndexMapping(perm: list[int], n_from: int, n_to: int):
    return GivenGatherIndexMapping(MatrixCache(np.asarray(perm)), VectorStandardBasis(n_from), VectorStandardBasis(n_to))


def _surj_identity(n: int) -> GatherIndexMapping:
    """Identity gather on n elements (n_from == n_to == n)."""
    return FixedGatherIndexMapping(list(range(n)), n, n)


def _surj_reversal(n: int) -> GatherIndexMapping:
    """Reversal gather: output[i] = input[n-1-i]."""
    return FixedGatherIndexMapping(list(range(n - 1, -1, -1)), n, n)


def _surj_cycle(n: int) -> GatherIndexMapping:
    """Cyclic gather: output[i] = input[(i+1) % n]."""
    return FixedGatherIndexMapping([(i + 1) % n for i in range(n)], n, n)


def _surj_project(n_from: int, n_to: int) -> GatherIndexMapping:
    """Projection: output[i] = input[i % n_from]  (n_to >= n_from, repeats wrap)."""
    return FixedGatherIndexMapping([i % n_from for i in range(n_to)], n_from, n_to)


def _surj_constant(n_from: int, n_to: int, src: int = 0) -> GatherIndexMapping:
    """All outputs read from the same source index ``src``."""
    return FixedGatherIndexMapping([src] * n_to, n_from, n_to)


# ---------------------------------------------------------------------------
# Tests: SurjectiveIndexMapping.apply_to_coefficient_vector
# ---------------------------------------------------------------------------

class TestSurjectiveApply:
    """apply_to_coefficient_vector for SurjectiveIndexMapping: gather semantics."""

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------
    @staticmethod
    def _reference(x: np.ndarray, perm: np.ndarray|MatrixCache, axis: int) -> np.ndarray:
        """Dense reference gather: out[i] = x[perm[i]] along `axis`."""
        if isinstance(perm, MatrixCache):
            perm = perm.as_numpy()
        return np.take(x, perm, axis=axis)

    # ------------------------------------------------------------------
    # numpy – permutation cases
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_numpy_identity(self, n):
        surj = _surj_identity(n)
        rng = np.random.default_rng(100)
        x = rng.standard_normal(n)
        result = surj.apply_to_coefficient_vector(x)
        np.testing.assert_allclose(result, x, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_numpy_reversal(self, n):
        surj = _surj_reversal(n)
        rng = np.random.default_rng(101)
        x = rng.standard_normal(n)
        result = surj.apply_to_coefficient_vector(x)
        expected = self._reference(x, surj.index_mapping(), axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_numpy_cycle(self, n):
        surj = _surj_cycle(n)
        rng = np.random.default_rng(102)
        x = rng.standard_normal(n)
        result = surj.apply_to_coefficient_vector(x)
        expected = self._reference(x, surj.index_mapping(), axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # numpy – true projection (repeated source indices)
    # ------------------------------------------------------------------

    def test_numpy_projection_repeats(self):
        """output[i] = input[i % n_from]: the same source value appears twice."""
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        rng = np.random.default_rng(103)
        x = rng.standard_normal(n_from)
        result = surj.apply_to_coefficient_vector(x)
        expected = self._reference(x, surj.index_mapping(), axis=0)
        assert result.shape == (n_to,)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_numpy_constant_mapping(self):
        """All outputs equal input[0]."""
        n_from, n_to = 4, 5
        surj = _surj_constant(n_from, n_to, src=0)
        rng = np.random.default_rng(104)
        x = rng.standard_normal(n_from)
        result = surj.apply_to_coefficient_vector(x)
        np.testing.assert_allclose(result, np.full(n_to, x[0]), atol=1e-12)

    # ------------------------------------------------------------------
    # numpy – non-zero axis
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [3, 5])
    def test_numpy_axis1(self, n):
        surj = _surj_reversal(n)
        rng = np.random.default_rng(110)
        x = rng.standard_normal((4, n))
        result = surj.apply_to_coefficient_vector(x, axis=1)
        expected = self._reference(x, surj.index_mapping(), axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (4, n)

    def test_numpy_negative_axis(self):
        n = 4
        surj = _surj_reversal(n)
        rng = np.random.default_rng(111)
        x = rng.standard_normal((3, n))
        result_neg = surj.apply_to_coefficient_vector(x, axis=-1)
        result_pos = surj.apply_to_coefficient_vector(x, axis=1)
        np.testing.assert_allclose(result_neg, result_pos, atol=1e-12)

    def test_numpy_3d_axis2(self):
        n = 4
        surj = _surj_cycle(n)
        rng = np.random.default_rng(112)
        x = rng.standard_normal((2, 3, n))
        result = surj.apply_to_coefficient_vector(x, axis=2)
        expected = self._reference(x, surj.index_mapping(), axis=2)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (2, 3, n)

    def test_numpy_projection_axis1(self):
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        rng = np.random.default_rng(113)
        x = rng.standard_normal((4, n_from))
        result = surj.apply_to_coefficient_vector(x, axis=1)
        expected = self._reference(x, surj.index_mapping(), axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        assert result.shape == (4, n_to)

    # ------------------------------------------------------------------
    # pydata/sparse – axis 0 (1-D)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [3, 5])
    def test_pydata_sparse_axis0(self, n):
        surj = _surj_cycle(n)
        rng = np.random.default_rng(120)
        x_dense = rng.standard_normal(n)
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = surj.apply_to_coefficient_vector(x_sparse, axis=0)
        expected = self._reference(x_dense, surj.index_mapping(), axis=0)
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # pydata/sparse – axis 1 (2-D)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [3, 5])
    def test_pydata_sparse_axis1(self, n):
        surj = _surj_reversal(n)
        rng = np.random.default_rng(121)
        x_dense = rng.standard_normal((4, n))
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = surj.apply_to_coefficient_vector(x_sparse, axis=1)
        expected = self._reference(x_dense, surj.index_mapping(), axis=1)
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)

    def test_pydata_sparse_projection_repeats(self):
        """Repeated source indices in pydata/sparse: the same stored value is
        emitted at multiple output coordinates."""
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        rng = np.random.default_rng(122)
        x_dense = rng.standard_normal(n_from)
        x_sparse = sparse.COO.from_numpy(x_dense)
        result = surj.apply_to_coefficient_vector(x_sparse)
        expected = self._reference(x_dense, surj.index_mapping(), axis=0)
        assert result.shape == (n_to,)
        np.testing.assert_allclose(result.todense(), expected, atol=1e-12)

    def test_pydata_sparse_all_zero_input(self):
        """All-zero sparse input should produce all-zero output (empty COO)."""
        n = 4
        surj = _surj_cycle(n)
        x_sparse = sparse.COO(np.zeros(n))
        result = surj.apply_to_coefficient_vector(x_sparse)
        assert result.shape == (n,)
        np.testing.assert_allclose(result.todense(), np.zeros(n), atol=1e-12)

    # ------------------------------------------------------------------
    # scipy.sparse – axis 0 and axis 1
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [3, 5])
    def test_scipy_sparse_axis0(self, n):
        surj = _surj_reversal(n)
        rng = np.random.default_rng(130)
        x_dense = rng.standard_normal((n, n))
        x_sp = sp.csr_matrix(x_dense)
        result = surj.apply_to_coefficient_vector(x_sp, axis=0)
        expected = self._reference(x_dense, surj.index_mapping(), axis=0)
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5])
    def test_scipy_sparse_axis1(self, n):
        surj = _surj_reversal(n)
        rng = np.random.default_rng(131)
        x_dense = rng.standard_normal((n, n))
        x_sp = sp.csr_matrix(x_dense)
        result = surj.apply_to_coefficient_vector(x_sp, axis=1)
        expected = self._reference(x_dense, surj.index_mapping(), axis=1)
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)


    def test_scipy_sparse_projection_repeats(self):
        """scipy.sparse with repeated source rows (true surjection)."""
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        rng = np.random.default_rng(132)
        x_dense = rng.standard_normal((n_from, 4))
        x_sp = sp.csr_matrix(x_dense)
        result = surj.apply_to_coefficient_vector(x_sp, axis=0)
        expected = self._reference(x_dense, surj.index_mapping(), axis=0)
        assert result.shape == (n_to, 4)
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-12)

    # ------------------------------------------------------------------
    # All backends agree
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [4, 6])
    def test_all_backends_agree_axis0(self, n):
        surj = _surj_cycle(n)
        rng = np.random.default_rng(140)
        x_dense = rng.standard_normal(n)

        r_numpy = surj.apply_to_coefficient_vector(x_dense)
        r_sparse = surj.apply_to_coefficient_vector(
            sparse.COO.from_numpy(x_dense)
        ).todense()
        np.testing.assert_allclose(r_sparse, r_numpy, atol=1e-12)

    @pytest.mark.parametrize("n", [4, 6])
    def test_all_backends_agree_axis1(self, n):
        surj = _surj_cycle(n)
        rng = np.random.default_rng(141)
        x_dense = rng.standard_normal((n, n))

        r_numpy = surj.apply_to_coefficient_vector(x_dense, axis=1)
        r_sparse = surj.apply_to_coefficient_vector(
            sparse.COO.from_numpy(x_dense), axis=1
        ).todense()
        r_scipy = surj.apply_to_coefficient_vector(
            sp.csr_matrix(x_dense), axis=1
        ).toarray()
        np.testing.assert_allclose(r_sparse, r_numpy, atol=1e-12)
        np.testing.assert_allclose(r_scipy, r_numpy, atol=1e-12)

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_output_shape_n_from_lt_n_to(self):
        """n_from < n_to: output is larger than input."""
        n_from, n_to = 3, 7
        surj = _surj_project(n_from, n_to)
        x = np.ones(n_from)
        result = surj.apply_to_coefficient_vector(x)
        assert result.shape == (n_to,)

    def test_output_shape_n_from_gt_n_to(self):
        """n_from > n_to: output is smaller (sub-selection)."""
        perm = [2, 0, 4]        # select 3 elements from a 5-element source
        n_from, n_to = 5, 3
        surj = FixedGatherIndexMapping(perm, n_from, n_to)
        x = np.arange(n_from, dtype=float)
        result = surj.apply_to_coefficient_vector(x)
        np.testing.assert_allclose(result, np.array([2.0, 0.0, 4.0]), atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: SurjectiveIndexMapping.as_transition_matrix()
# ---------------------------------------------------------------------------

class TestSurjectiveAsTransitionMatrix:
    """Verify as_transition_matrix() for SurjectiveIndexMapping."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_returns_given_transition_matrix(self, n):
        surj = _surj_reversal(n)
        tm = surj.as_transition_matrix()
        assert isinstance(tm, GivenTransitionMatrix)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_basis_attributes_preserved(self, n):
        surj = _surj_cycle(n)
        tm = surj.as_transition_matrix()
        assert tm.basis_from is surj.basis_from
        assert tm.basis_to is surj.basis_to

    def test_identity_gives_identity_matrix(self):
        n = 4
        surj = _surj_identity(n)
        mat = surj.as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(mat, np.eye(n, dtype=int))

    @pytest.mark.parametrize("n", [3, 5])
    def test_apply_matches_direct(self, n):
        """Applying the transition matrix must equal apply_to_coefficient_vector."""
        surj = _surj_cycle(n)
        tm = surj.as_transition_matrix()
        rng = np.random.default_rng(200)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            expected = surj.apply_to_coefficient_vector(x)
            actual = tm.apply_to_coefficient_vector(x)
            np.testing.assert_allclose(actual.todense(), expected.todense(), atol=1e-12)

    def test_shape_n_from_lt_n_to(self):
        """Matrix shape is (basis_to.size(), basis_from.size())."""
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        mat = surj.as_transition_matrix().to_numpy()
        assert mat.shape == (n_to, n_from)

    def test_shape_n_from_gt_n_to(self):
        perm = [2, 0, 4]
        n_from, n_to = 5, 3
        surj = FixedGatherIndexMapping(perm, n_from, n_to)
        mat = surj.as_transition_matrix().to_numpy()
        assert mat.shape == (n_to, n_from)

    def test_projection_matrix_content(self):
        """For a projection perm=[0,1,2,0,1,2] the matrix should have exactly
        one 1 per row and two 1s per column (for the repeated columns)."""
        n_from, n_to = 3, 6
        surj = _surj_project(n_from, n_to)
        mat = surj.as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(mat.sum(axis=1), np.ones(n_to))  # one 1 per row
        np.testing.assert_array_equal(mat.sum(axis=0), 2 * np.ones(n_from))  # two per col

    @pytest.mark.parametrize("n", [3, 5])
    def test_result_is_cached(self, n):
        surj = _surj_reversal(n)
        assert surj.as_transition_matrix() is surj.as_transition_matrix()

    @pytest.mark.parametrize("n", [3, 5])
    def test_index_mapping_cached(self, n):
        """index_mapping() is computed once and reused."""
        surj = _surj_cycle(n)
        perm1 = surj.index_mapping()
        perm2 = surj.index_mapping()
        assert perm1 is perm2


# ---------------------------------------------------------------------------
# Tests: SurjectiveIndexMapping.inverse() / SurjectiveMappingInverse
# ---------------------------------------------------------------------------

class TestSurjectiveMappingInverse:
    """Verify inverse() and SurjectiveMappingInverse for permutation cases."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_has_swapped_bases(self, n):
        surj = _surj_reversal(n)
        inv = surj.inverse()
        assert inv.basis_from is surj.basis_to
        assert inv.basis_to is surj.basis_from

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_of_inverse_is_original(self, n):
        surj = _surj_cycle(n)
        assert surj.inverse().inverse() is surj

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_round_trip_surj_then_inv(self, n):
        """Apply surjective gather then its inverse (injective scatter) recovers x."""
        surj = _surj_cycle(n)
        inv = surj.inverse()
        rng = np.random.default_rng(300)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            y = surj.apply_to_coefficient_vector(x)
            recovered = inv.apply_to_coefficient_vector(y)
            np.testing.assert_allclose(recovered.todense(), x.todense(), atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_round_trip_inv_then_surj(self, n):
        """Apply inverse (injective scatter) first, then the surjective gather."""
        surj = _surj_reversal(n)
        inv = surj.inverse()
        rng = np.random.default_rng(301)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            y = inv.apply_to_coefficient_vector(x)
            recovered = surj.apply_to_coefficient_vector(y)
            np.testing.assert_allclose(recovered.todense(), x.todense(), atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_matrix_is_transpose_of_forward(self, n):
        """For a permutation matrix P, P^{-1} = P^T."""
        surj = _surj_cycle(n)
        fwd = surj.as_transition_matrix().to_numpy()
        inv_mat = surj.inverse().as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(inv_mat, fwd.T)

    def test_identity_inverse_is_identity(self):
        n = 4
        surj = _surj_identity(n)
        inv_mat = surj.inverse().as_transition_matrix().to_numpy()
        np.testing.assert_array_equal(inv_mat, np.eye(n, dtype=int))

    @pytest.mark.parametrize("n", [3, 5])
    def test_inverse_cached(self, n):
        surj = _surj_reversal(n)
        assert surj.inverse() is surj.inverse()


# ---------------------------------------------------------------------------
# Tests: InjectiveScatterMappingInverse extras
# ---------------------------------------------------------------------------

class TestInjectiveMappingInverseExtras:
    """Additional tests for InjectiveScatterMappingInverse not covered elsewhere."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_inverse_of_inverse_is_original(self, n):
        """InjectiveScatterMappingInverse.inverse() returns the original InjectiveIndexMapping."""
        iso = _cycle_perm(n)
        assert iso.inverse().inverse() is iso

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_as_transition_matrix_apply_matches_direct(self, n):
        """Transition matrix of the inverse gives same results as apply_to_coefficient_vector."""
        iso = _reversal_perm(n)
        inv = iso.inverse()
        tm = inv.as_transition_matrix()
        rng = np.random.default_rng(400)
        for _ in range(5):
            x = sparse.COO.from_numpy(rng.standard_normal(n))
            expected = inv.apply_to_coefficient_vector(x)
            actual = tm.apply_to_coefficient_vector(x)
            np.testing.assert_allclose(actual.todense(), expected.todense(), atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5])
    def test_all_backends_agree_for_inverse(self, n):
        """numpy, pydata/sparse, and scipy.sparse paths of InjectiveScatterMappingInverse agree."""
        iso = _cycle_perm(n)
        inv = iso.inverse()
        rng = np.random.default_rng(401)
        x_dense = rng.standard_normal(n)

        r_numpy = inv.apply_to_coefficient_vector(x_dense)
        r_sparse = inv.apply_to_coefficient_vector(
            sparse.COO.from_numpy(x_dense)
        ).todense()
        # scipy.sparse: use a column vector (n, 1) so the gather axis is 0.
        x_col = sp.csr_matrix(x_dense.reshape(-1, 1))
        r_scipy = inv.apply_to_coefficient_vector(x_col, axis=0).toarray().ravel()
        np.testing.assert_allclose(r_sparse, r_numpy, atol=1e-12)
        np.testing.assert_allclose(r_scipy, r_numpy, atol=1e-12)
