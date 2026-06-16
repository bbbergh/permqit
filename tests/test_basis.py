import unittest
import numpy as np
from permqit.algebra.basis import (
    VectorStandardBasis,
    TensorProductBasis,
    StandardTensorProductBasis,
    MatrixStandardBasis,
    MatrixTensorProductBasis,
)


class TestVectorStandardBasis(unittest.TestCase):
    """Test suite for VectorStandardBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dim = 4
        self.basis = VectorStandardBasis(self.dim)

    def test_size(self):
        """Test that size returns the correct dimension."""
        self.assertEqual(self.basis.size(), self.dim)

    def test_len(self):
        """Test that __len__ returns the correct dimension."""
        self.assertEqual(len(self.basis), self.dim)

    def test_all_vectors(self):
        """Test that all_vectors returns the identity matrix."""
        expected = np.eye(self.dim)
        np.testing.assert_array_equal(self.basis.all_vectors(), expected)

    def test_iterate_labels(self):
        """Test that iterate_labels returns correct labels."""
        labels = list(self.basis.iterate_labels())
        expected = list(range(self.dim))
        self.assertEqual(labels, expected)

    def test_iterate_vectors(self):
        """Test that iterate_vectors yields correct vectors."""
        vectors = list(self.basis.iterate_vectors())
        expected = np.eye(self.dim)
        for i, vec in enumerate(vectors):
            np.testing.assert_array_equal(vec, expected[i])

    def test_label_to_vector(self):
        """Test label_to_vector returns correct basis vector."""
        for i in range(self.dim):
            vec = self.basis.label_to_vector(i)
            expected = np.zeros(self.dim)
            expected[i] = 1
            np.testing.assert_array_equal(vec, expected)

    def test_label_to_index(self):
        """Test label_to_index returns the same index."""
        for i in range(self.dim):
            self.assertEqual(self.basis.label_to_index(i), i)

    def test_label_at_index(self):
        """Test label_at_index returns correct label."""
        for i in range(self.dim):
            self.assertEqual(self.basis.label_at_index(i), i)

    def test_vector_at_index(self):
        """Test vector_at_index returns correct vector."""
        for i in range(self.dim):
            vec = self.basis.vector_at_index(i)
            expected = np.zeros(self.dim)
            expected[i] = 1
            np.testing.assert_array_equal(vec, expected)

    def test_getitem(self):
        """Test __getitem__ returns correct label."""
        for i in range(self.dim):
            label = self.basis[i]
            self.assertEqual(label, i)

    def test_iter(self):
        """Test __iter__ returns labels."""
        labels = list(self.basis)
        expected = list(range(self.dim))
        self.assertEqual(labels, expected)

    def test_linear_combination(self):
        """Test linear_combination returns the coefficients."""
        coeffs = np.array([1, 2, 3, 4])
        result = self.basis.linear_combination(coeffs)
        np.testing.assert_array_equal(result, coeffs)

    def test_linear_combination_batch(self):
        """Test linear_combination with batched coefficients."""
        coeffs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        result = self.basis.linear_combination(coeffs)
        np.testing.assert_array_equal(result, coeffs)

    def test_dtype(self):
        """Test that custom dtype is respected."""
        basis_float = VectorStandardBasis(3, dtype=np.float64)
        vec = basis_float.label_to_vector(0)
        self.assertEqual(vec.dtype, np.float64)


class TestTensorProductBasis(unittest.TestCase):
    """Test suite for TensorProductBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        self.basis1 = VectorStandardBasis(2)
        self.basis2 = VectorStandardBasis(3)
        self.tensor_basis = TensorProductBasis([self.basis1, self.basis2])

    def test_size(self):
        """Test that size returns the product of component sizes."""
        expected_size = 2 * 3
        self.assertEqual(self.tensor_basis.size(), expected_size)

    def test_n_property(self):
        """Test that n property returns number of bases."""
        self.assertEqual(self.tensor_basis.n, 2)

    def test_iterate_labels(self):
        """Test that iterate_labels returns all combinations."""
        labels = list(self.tensor_basis.iterate_labels())
        expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        self.assertEqual(labels, expected)

    def test_label_to_vector(self):
        """Test label_to_vector returns correct Kronecker product."""
        label = (1, 2)
        vec = self.tensor_basis.label_to_vector(label)

        # Expected: e_1 ⊗ e_2
        e1 = np.array([0, 1])
        e2 = np.array([0, 0, 1])
        expected = np.kron(e1, e2)

        np.testing.assert_array_equal(vec, expected)

    def test_label_to_index(self):
        """Test label_to_index returns correct index.

        TensorProductBasis uses CANONICAL LEXICOGRAPHIC ordering (row-major).
        For bases of size 2 and 3 with basis_indices_multiplier [3, 1]:
        The indexing is: (i, j) -> i*3 + j*1

        This matches the standard lexicographic ordering:
        (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> indices 0,1,2,3,4,5

        This matches numpy.unravel_index and itertools.product.
        """
        test_cases = [
            ((0, 0), 0),
            ((0, 1), 1),
            ((0, 2), 2),
            ((1, 0), 3),
            ((1, 1), 4),
            ((1, 2), 5),
        ]
        for label, expected_idx in test_cases:
            self.assertEqual(self.tensor_basis.label_to_index(label), expected_idx)

    def test_label_at_index(self):
        """Test label_at_index returns correct label (round-trip test)."""
        for idx in range(self.tensor_basis.size()):
            label = self.tensor_basis.label_at_index(idx)
            # Check round-trip: label_at_index and label_to_index are inverses
            self.assertEqual(self.tensor_basis.label_to_index(label), idx)

    def test_vector_at_index(self):
        """Test vector_at_index returns correct vector."""
        for idx in range(self.tensor_basis.size()):
            vec = self.tensor_basis.vector_at_index(idx)
            label = self.tensor_basis.label_at_index(idx)
            expected = self.tensor_basis.label_to_vector(label)
            np.testing.assert_array_equal(vec, expected)

    def test_three_factor_tensor(self):
        """Test tensor product of three bases."""
        basis1 = VectorStandardBasis(2)
        basis2 = VectorStandardBasis(2)
        basis3 = VectorStandardBasis(2)
        tensor_basis = TensorProductBasis([basis1, basis2, basis3])

        self.assertEqual(tensor_basis.size(), 8)
        self.assertEqual(tensor_basis.n, 3)

        # Test a specific vector
        label = (1, 0, 1)
        vec = tensor_basis.label_to_vector(label)
        e1 = np.array([0, 1])
        e0 = np.array([1, 0])
        expected = np.kron(np.kron(e1, e0), e1)
        np.testing.assert_array_equal(vec, expected)

    def test_empty_bases_assertion(self):
        """Test that empty bases list raises assertion error."""
        with self.assertRaises(AssertionError):
            TensorProductBasis([])

    def test_canonical_ordering(self):
        """Test and verify that TensorProductBasis uses CANONICAL ordering.

        TensorProductBasis uses standard lexicographic (row-major) ordering,
        matching numpy.unravel_index and itertools.product.

        For a 2x3 tensor product (basis sizes [2, 3]):
        - Canonical lexicographic order:
          (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> indices 0,1,2,3,4,5
        - TensorProductBasis order:
          (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> indices 0,1,2,3,4,5

        They match! ✓
        """
        # The canonical lexicographic ordering
        canonical_ordering = [
            (0, 0),  # index 0
            (0, 1),  # index 1
            (0, 2),  # index 2
            (1, 0),  # index 3
            (1, 1),  # index 4
            (1, 2),  # index 5
        ]

        for idx, expected_label in enumerate(canonical_ordering):
            label = self.tensor_basis.label_at_index(idx)
            self.assertEqual(label, expected_label,
                           f"Index {idx} maps to {label}, expected {expected_label}")

        # Verify it matches numpy.unravel_index
        for idx in range(6):
            label = self.tensor_basis.label_at_index(idx)
            numpy_label = np.unravel_index(idx, (2, 3))
            self.assertEqual(label, numpy_label,
                           f"Index {idx}: TensorProductBasis gives {label}, "
                           f"numpy.unravel_index gives {numpy_label}")

        # Verify it matches itertools.product order
        import itertools
        for idx, expected_label in enumerate(itertools.product(range(2), range(3))):
            label = self.tensor_basis.label_at_index(idx)
            self.assertEqual(label, expected_label,
                           f"Index {idx}: TensorProductBasis gives {label}, "
                           f"itertools.product gives {expected_label}")


class TestStandardTensorProductBasis(unittest.TestCase):
    """Test suite for StandardTensorProductBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        self.single_dim = 2
        self.n = 3
        self.basis = StandardTensorProductBasis(self.single_dim, self.n)

    def test_size(self):
        """Test that size returns single_dimension^n."""
        expected_size = self.single_dim ** self.n
        self.assertEqual(self.basis.size(), expected_size)

    def test_dim_property(self):
        """Test that dim property equals single_dimension^n."""
        self.assertEqual(self.basis.dim, self.single_dim ** self.n)

    def test_from_n_indices(self):
        """Test from_n_indices creates correct tensor product."""
        indices = (0, 1, 0)
        vec = self.basis.from_n_indices(indices)

        # Should have exactly one 1 and rest zeros
        self.assertEqual(np.sum(vec), 1)
        self.assertEqual(vec.shape, (self.single_dim ** self.n,))

    def test_iterate_labels(self):
        """Test that iterate_labels returns all tuples."""
        labels = list(self.basis.iterate_labels())
        self.assertEqual(len(labels), self.single_dim ** self.n)

        # Check that all labels are tuples of length n
        for label in labels:
            self.assertEqual(len(label), self.n)
            for idx in label:
                self.assertIn(idx, range(self.single_dim))

    def test_label_to_vector(self):
        """Test label_to_vector returns correct vector."""
        label = (1, 0, 1)
        vec = self.basis.label_to_vector(label)

        # Vector should have exactly one 1
        self.assertEqual(np.sum(vec), 1)
        self.assertEqual(vec.shape, (self.single_dim ** self.n,))

    def test_label_to_index(self):
        """Test label_to_index for standard tensor product."""
        # For single_dim=2, n=3: binary indexing
        test_cases = [
            ((0, 0, 0), 0),
            ((0, 0, 1), 1),
            ((0, 1, 0), 2),
            ((0, 1, 1), 3),
            ((1, 0, 0), 4),
            ((1, 0, 1), 5),
            ((1, 1, 0), 6),
            ((1, 1, 1), 7),
        ]
        for label, expected_idx in test_cases:
            self.assertEqual(self.basis.label_to_index(label), expected_idx)

    def test_label_at_index(self):
        """Test label_at_index returns correct label."""
        for idx in range(self.basis.size()):
            label = self.basis.label_at_index(idx)
            # Check round-trip
            self.assertEqual(self.basis.label_to_index(label), idx)

    def test_vector_at_index(self):
        """Test vector_at_index returns correct vector."""
        for idx in range(self.basis.size()):
            vec = self.basis.vector_at_index(idx)
            # Should be a standard basis vector
            self.assertEqual(np.sum(vec), 1)
            self.assertEqual(vec[idx], 1)

    def test_linear_combination(self):
        """Test linear_combination returns coefficients."""
        coeffs = np.arange(self.basis.size())
        result = self.basis.linear_combination(coeffs)
        np.testing.assert_array_equal(result, coeffs)

    def test_different_dimensions(self):
        """Test with different single dimension."""
        basis_3x2 = StandardTensorProductBasis(3, 2)
        self.assertEqual(basis_3x2.size(), 9)
        self.assertEqual(basis_3x2.single_dimension, 3)
        self.assertEqual(basis_3x2.n, 2)

    def test_from_n_indices_assertion(self):
        """Test from_n_indices with wrong number of indices."""
        with self.assertRaises(AssertionError):
            self.basis.from_n_indices((0, 1))  # Should be 3 indices, not 2


class TestMatrixStandardBasis(unittest.TestCase):
    """Test suite for MatrixStandardBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        self.d = 3
        self.basis = MatrixStandardBasis(self.d)

    def test_size(self):
        """Test that size returns d^2."""
        self.assertEqual(self.basis.size(), self.d ** 2)

    def test_all_vectors(self):
        """Test that all_vectors returns correct shape."""
        vectors = self.basis.all_vectors()
        self.assertEqual(vectors.shape, (self.d ** 2, self.d, self.d))

        # Each should be a matrix with exactly one 1
        for vec in vectors:
            self.assertEqual(np.sum(vec), 1)

    def test_label_to_vector(self):
        """Test label_to_vector returns correct matrix."""
        for i in range(self.d):
            for j in range(self.d):
                mat = self.basis.label_to_vector((i, j))

                # Should be all zeros except at position (i, j)
                self.assertEqual(mat.shape, (self.d, self.d))
                self.assertEqual(np.sum(mat), 1)
                self.assertEqual(mat[i, j], 1)

    def test_iterate_labels(self):
        """Test that iterate_labels returns all (i,j) pairs."""
        labels = list(self.basis.iterate_labels())
        self.assertEqual(len(labels), self.d ** 2)

        # Check all combinations are present
        expected = [(i, j) for i in range(self.d) for j in range(self.d)]
        self.assertEqual(labels, expected)

    def test_label_to_index(self):
        """Test label_to_index for matrix basis."""
        # For d=3, should be row-major indexing
        test_cases = [
            ((0, 0), 0),
            ((0, 1), 1),
            ((0, 2), 2),
            ((1, 0), 3),
            ((1, 1), 4),
            ((1, 2), 5),
            ((2, 0), 6),
            ((2, 1), 7),
            ((2, 2), 8),
        ]
        for label, expected_idx in test_cases:
            self.assertEqual(self.basis.label_to_index(label), expected_idx)

    def test_label_at_index(self):
        """Test label_at_index returns correct label."""
        for idx in range(self.basis.size()):
            label = self.basis.label_at_index(idx)
            # Check round-trip
            self.assertEqual(self.basis.label_to_index(label), idx)

    def test_vector_at_index(self):
        """Test vector_at_index returns correct matrix."""
        for idx in range(self.basis.size()):
            mat = self.basis.vector_at_index(idx)
            label = self.basis.label_at_index(idx)
            expected = self.basis.label_to_vector(label)
            np.testing.assert_array_equal(mat, expected)

    def test_linear_combination(self):
        """Test linear_combination reshapes coefficients correctly."""
        coeffs = np.arange(self.d ** 2)
        result = self.basis.linear_combination(coeffs)

        # Should reshape to (d, d) matrix
        self.assertEqual(result.shape, (self.d, self.d))
        np.testing.assert_array_equal(result.flatten(), coeffs)

    def test_linear_combination_batch(self):
        """Test linear_combination with batched coefficients."""
        coeffs = np.arange(2 * self.d ** 2).reshape(2, self.d ** 2)
        result = self.basis.linear_combination(coeffs)

        # Should reshape to (2, d, d)
        self.assertEqual(result.shape, (2, self.d, self.d))

    def test_linear_combination_wrong_axis(self):
        """Test linear_combination raises error for non-default axis."""
        coeffs = np.arange(self.d ** 2)
        with self.assertRaises(ValueError):
            self.basis.linear_combination(coeffs, coeff_axis=0)

    def test_linear_combination_wrong_size(self):
        """Test linear_combination asserts on wrong coefficient size."""
        coeffs = np.arange(self.d)  # Wrong size
        with self.assertRaises(AssertionError):
            self.basis.linear_combination(coeffs)

    def test_iterate_vectors(self):
        """Test iterate_vectors yields correct matrices."""
        vectors = list(self.basis.iterate_vectors())
        self.assertEqual(len(vectors), self.d ** 2)

        for vec in vectors:
            self.assertEqual(vec.shape, (self.d, self.d))
            self.assertEqual(np.sum(vec), 1)

    def test_getitem(self):
        """Test __getitem__ returns correct label."""
        for idx in range(min(3, self.basis.size())):  # Test first few
            label = self.basis[idx]
            expected_label = self.basis.label_at_index(idx)
            self.assertEqual(label, expected_label)

    def test_2x2_matrices(self):
        """Test with 2x2 matrices."""
        basis_2x2 = MatrixStandardBasis(2)
        self.assertEqual(basis_2x2.size(), 4)

        # Test all four basis matrices
        expected_matrices = [
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 0], [0, 1]],
        ]

        for idx, expected in enumerate(expected_matrices):
            mat = basis_2x2.vector_at_index(idx)
            np.testing.assert_array_equal(mat, expected)


class TestBasisConsistency(unittest.TestCase):
    """Test consistency between different basis implementations."""

    def test_roundtrip_label_index(self):
        """Test round-trip conversion between labels and indices."""
        bases = [
            VectorStandardBasis(5),
            StandardTensorProductBasis(2, 3),
            MatrixStandardBasis(3),
        ]

        for basis in bases:
            for idx in range(basis.size()):
                label = basis.label_at_index(idx)
                idx_back = basis.label_to_index(label)
                self.assertEqual(idx, idx_back)

    def test_vector_consistency(self):
        """Test that different methods of getting vectors are consistent."""
        bases = [
            VectorStandardBasis(4),
            StandardTensorProductBasis(2, 2),
            MatrixStandardBasis(2),
        ]

        for basis in bases:
            for idx in range(min(4, basis.size())):  # Test first few
                # Two ways to get the same vector
                vec1 = basis.vector_at_index(idx)
                label = basis.label_at_index(idx)
                vec2 = basis.label_to_vector(label)

                np.testing.assert_array_equal(vec1, vec2)

                # __getitem__ now returns labels, not vectors
                self.assertEqual(basis[idx], label)

    def test_all_vectors_matches_iteration(self):
        """Test that all_vectors matches iterate_vectors."""
        bases = [
            VectorStandardBasis(3),
            MatrixStandardBasis(2),
        ]

        for basis in bases:
            all_vecs = basis.all_vectors()
            iter_vecs = list(basis.iterate_vectors())

            self.assertEqual(len(iter_vecs), basis.size())
            for i, vec in enumerate(iter_vecs):
                np.testing.assert_array_equal(all_vecs[i], vec)


class TestMatrixStandardBasisTranspose(unittest.TestCase):
    """Test suite for MatrixStandardBasis.transpose() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.d = 3
        self.basis = MatrixStandardBasis(self.d)

    def test_transpose_1d_coeffs_mathematical_correctness(self):
        """Test that transpose gives coefficients of the transposed matrix (1D case)."""
        # Create a random matrix as coefficients
        coeffs = np.random.randn(self.d ** 2)

        # Get the matrix from coefficients
        matrix = self.basis.linear_combination(coeffs)

        # Transpose the matrix directly
        transposed_matrix = matrix.T

        # Get the transposed coefficients
        transposed_coeffs = self.basis.transpose(coeffs)

        # Reconstruct the matrix from transposed coefficients
        reconstructed = self.basis.linear_combination(transposed_coeffs)

        # The reconstructed matrix should equal the transposed matrix
        np.testing.assert_array_almost_equal(reconstructed, transposed_matrix)

    def test_transpose_1d_specific_matrix(self):
        """Test transpose with a specific matrix for verification."""
        # Create a simple 2x2 matrix [[1,2],[3,4]]
        basis_2x2 = MatrixStandardBasis(2)
        # Coefficients: index 0 -> (0,0), index 1 -> (0,1), index 2 -> (1,0), index 3 -> (1,1)
        # For matrix [[1,2],[3,4]], coeffs = [1, 2, 3, 4]
        coeffs = np.array([1.0, 2.0, 3.0, 4.0])

        # After transpose, matrix becomes [[1,3],[2,4]]
        # So coeffs should become [1, 3, 2, 4]
        expected_transposed_coeffs = np.array([1.0, 3.0, 2.0, 4.0])

        transposed_coeffs = basis_2x2.transpose(coeffs)
        np.testing.assert_array_almost_equal(transposed_coeffs, expected_transposed_coeffs)

    def test_transpose_2d_coeffs_batch(self):
        """Test transpose with 2D coefficient array (batched matrices)."""
        batch_size = 5
        coeffs = np.random.randn(batch_size, self.d ** 2)

        transposed_coeffs = self.basis.transpose(coeffs, axis=-1)

        # Verify each batch element independently
        for i in range(batch_size):
            matrix = self.basis.linear_combination(coeffs[i])
            expected_transposed = matrix.T
            reconstructed = self.basis.linear_combination(transposed_coeffs[i])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_3d_coeffs_different_axes(self):
        """Test transpose with 3D coefficient array on different axes."""
        shape = (4, self.d ** 2, 3)
        coeffs = np.random.randn(*shape)

        # Transpose along axis=1 (where the coefficients are)
        transposed_coeffs = self.basis.transpose(coeffs, axis=1)

        # Verify the result by checking a sample
        for i in range(shape[0]):
            for k in range(shape[2]):
                matrix = self.basis.linear_combination(coeffs[i, :, k])
                expected_transposed = matrix.T
                reconstructed = self.basis.linear_combination(transposed_coeffs[i, :, k])
                np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_axis_0(self):
        """Test transpose with coefficients on axis 0."""
        shape = (self.d ** 2, 5)
        coeffs = np.random.randn(*shape)

        transposed_coeffs = self.basis.transpose(coeffs, axis=0)

        # Verify
        for j in range(shape[1]):
            matrix = self.basis.linear_combination(coeffs[:, j])
            expected_transposed = matrix.T
            reconstructed = self.basis.linear_combination(transposed_coeffs[:, j])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_negative_axis(self):
        """Test that negative axis indexing works correctly."""
        shape = (2, self.d ** 2)
        coeffs = np.random.randn(*shape)

        # axis=-1 should be the same as axis=1 for 2D array
        transposed_1 = self.basis.transpose(coeffs, axis=-1)
        transposed_2 = self.basis.transpose(coeffs, axis=1)

        np.testing.assert_array_equal(transposed_1, transposed_2)

    def test_transpose_involutive(self):
        """Test that transpose is an involution (transposing twice gives original)."""
        coeffs = np.random.randn(self.d ** 2)

        double_transposed = self.basis.transpose(self.basis.transpose(coeffs))

        np.testing.assert_array_almost_equal(double_transposed, coeffs)

    def test_transpose_preserves_shape(self):
        """Test that transpose preserves the coefficient array shape."""
        for shape in [(self.d ** 2,), (3, self.d ** 2), (2, 3, self.d ** 2)]:
            coeffs = np.random.randn(*shape)
            transposed = self.basis.transpose(coeffs)
            self.assertEqual(transposed.shape, coeffs.shape)

    def test_transpose_identity_matrix(self):
        """Test that transposing the identity matrix gives the identity."""
        # Identity matrix coefficients (1 on diagonal, 0 elsewhere)
        coeffs = np.zeros(self.d ** 2)
        for i in range(self.d):
            idx = self.basis.label_to_index((i, i))
            coeffs[idx] = 1.0

        transposed = self.basis.transpose(coeffs)

        # Identity is symmetric, so transpose should give same coefficients
        np.testing.assert_array_almost_equal(transposed, coeffs)

    def test_transpose_symmetric_matrix(self):
        """Test that transposing a symmetric matrix gives the same coefficients."""
        # Create a symmetric matrix
        coeffs = np.zeros(self.d ** 2)
        for i in range(self.d):
            for j in range(self.d):
                idx = self.basis.label_to_index((i, j))
                coeffs[idx] = i + j  # Symmetric: A[i,j] = A[j,i]

        transposed = self.basis.transpose(coeffs)

        np.testing.assert_array_almost_equal(transposed, coeffs)


class TestMatrixTensorProductBasisTranspose(unittest.TestCase):
    """Test suite for MatrixTensorProductBasis.transpose() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.basis1 = MatrixStandardBasis(2)
        self.basis2 = MatrixStandardBasis(3)
        self.tensor_basis = MatrixTensorProductBasis((self.basis1, self.basis2))

    def test_transpose_1d_coeffs(self):
        """Test transpose with 1D coefficients."""
        # For tensor product of matrices A ⊗ B, transpose gives A^T ⊗ B^T
        coeffs = np.random.randn(self.tensor_basis.size())

        transposed_coeffs = self.tensor_basis.transpose(coeffs)

        # Verify by constructing the matrices explicitly
        matrix = self.tensor_basis.linear_combination(coeffs)
        expected_transposed = matrix.T
        reconstructed = self.tensor_basis.linear_combination(transposed_coeffs)

        np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_2d_coeffs_batch(self):
        """Test transpose with 2D batched coefficients."""
        batch_size = 4
        coeffs = np.random.randn(batch_size, self.tensor_basis.size())

        transposed_coeffs = self.tensor_basis.transpose(coeffs, axis=-1)

        for i in range(batch_size):
            matrix = self.tensor_basis.linear_combination(coeffs[i])
            expected_transposed = matrix.T
            reconstructed = self.tensor_basis.linear_combination(transposed_coeffs[i])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_involutive(self):
        """Test that transpose is an involution."""
        coeffs = np.random.randn(self.tensor_basis.size())

        double_transposed = self.tensor_basis.transpose(self.tensor_basis.transpose(coeffs))

        np.testing.assert_array_almost_equal(double_transposed, coeffs)

    def test_transpose_single_factor(self):
        """Test transpose with a single MatrixStandardBasis factor."""
        single_basis = MatrixTensorProductBasis([MatrixStandardBasis(3)])
        coeffs = np.random.randn(single_basis.size())

        transposed = single_basis.transpose(coeffs)

        # Should be equivalent to the underlying basis transpose
        expected = MatrixStandardBasis(3).transpose(coeffs)
        np.testing.assert_array_almost_equal(transposed, expected)

    def test_transpose_three_factors(self):
        """Test transpose with three MatrixStandardBasis factors."""
        basis = MatrixTensorProductBasis([
            MatrixStandardBasis(2),
            MatrixStandardBasis(2),
            MatrixStandardBasis(2),
        ])
        coeffs = np.random.randn(basis.size())

        transposed = basis.transpose(coeffs)

        # Verify by construction
        matrix = basis.linear_combination(coeffs)
        expected_transposed = matrix.T
        reconstructed = basis.linear_combination(transposed)

        np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_different_axis(self):
        """Test transpose on different axes in multi-dimensional array."""
        shape = (3, self.tensor_basis.size(), 2)
        coeffs = np.random.randn(*shape)

        transposed = self.tensor_basis.transpose(coeffs, axis=1)

        # Verify a sample
        matrix = self.tensor_basis.linear_combination(coeffs[1, :, 0])
        expected_transposed = matrix.T
        reconstructed = self.tensor_basis.linear_combination(transposed[1, :, 0])

        np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_preserves_shape(self):
        """Test that transpose preserves coefficient array shape."""
        for shape in [(self.tensor_basis.size(),), (5, self.tensor_basis.size()), (2, 3, self.tensor_basis.size())]:
            coeffs = np.random.randn(*shape)
            transposed = self.tensor_basis.transpose(coeffs)
            self.assertEqual(transposed.shape, coeffs.shape)


if __name__ == '__main__':
    unittest.main()

