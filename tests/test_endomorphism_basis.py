import unittest
import numpy as np
from permqit.algebra.endomorphism_basis import (
    EndSnIrrepBasis,
    MatrixDirectSumBasis,
    EndSnBlockDiagonalBasis, EndSnOrbitBasis,
)
from permqit.algebra.basis import MatrixStandardBasis
from permqit.algebra.matrix import BlockDiagonalMatrix
from permqit.representation.young_tableau import SSYT, Partition


class TestEndSnIrrepBasis(unittest.TestCase):
    """Test suite for EndSnIrrepBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use partition [2] (a single row of 2 boxes) with d=2
        self.partition = Partition([2])
        self.d = 2
        self.basis = EndSnIrrepBasis(self.partition, self.d)

    def test_initialization(self):
        """Test that basis initializes correctly."""
        self.assertEqual(self.basis.partition, self.partition)
        self.assertEqual(self.basis.single_dimension, self.d)
        self.assertIsInstance(self.basis._matrix_standard_basis, MatrixStandardBasis)

    def test_size(self):
        """Test that size returns d^2."""
        # For partition [2] with d=2, there are SSYT.count([2], 2) SSYTs
        num_ssyts = SSYT.count(self.partition, self.d)
        expected_size = num_ssyts ** 2
        self.assertEqual(self.basis.size(), expected_size)

    def test_size_matches_matrix_standard_basis(self):
        """Test that size matches the underlying MatrixStandardBasis."""
        self.assertEqual(self.basis.size(), self.basis._matrix_standard_basis.size())

    def test_iterate_labels_count(self):
        """Test that iterate_labels produces correct number of labels."""
        labels = list(self.basis.iterate_labels())
        self.assertEqual(len(labels), self.basis.size())

    def test_iterate_labels_structure(self):
        """Test that labels are tuples of (SSYT, SSYT)."""
        labels = list(self.basis.iterate_labels())
        for label in labels:
            self.assertIsInstance(label, tuple)
            self.assertEqual(len(label), 2)
            self.assertIsInstance(label[0], SSYT)
            self.assertIsInstance(label[1], SSYT)

    def test_iterate_labels_same_partition(self):
        """Test that both SSYTs in each label have the same partition."""
        labels = list(self.basis.iterate_labels())
        for ssyt1, ssyt2 in labels:
            self.assertEqual(ssyt1.partition, self.partition)
            self.assertEqual(ssyt2.partition, self.partition)

    def test_label_to_vector_shape(self):
        """Test that label_to_vector returns correct shape."""
        labels = list(self.basis.iterate_labels())
        if labels:
            vec = self.basis.label_to_vector(labels[0])
            expected_shape = (self.basis.m, self.basis.m)
            self.assertEqual(vec.shape, expected_shape)

    def test_label_to_vector_is_standard_basis_matrix(self):
        """Test that label_to_vector returns a matrix with exactly one 1."""
        labels = list(self.basis.iterate_labels())
        for label in labels:
            vec = self.basis.label_to_vector(label)
            # Should be a standard basis matrix (exactly one 1, rest 0s)
            self.assertEqual(np.sum(vec), 1)
            self.assertEqual(np.sum(vec == 1), 1)

    def test_roundtrip_label_index(self):
        """Test round-trip conversion between labels and indices."""
        labels = list(self.basis.iterate_labels())
        for label in labels:
            idx = self.basis.label_to_index(label)
            label_back = self.basis.label_at_index(idx)
            self.assertEqual(label, label_back)

    def test_roundtrip_index_label(self):
        """Test round-trip conversion from index to label and back."""
        for idx in range(self.basis.size()):
            label = self.basis.label_at_index(idx)
            idx_back = self.basis.label_to_index(label)
            self.assertEqual(idx, idx_back)

    def test_vector_at_index_matches_label_to_vector(self):
        """Test that vector_at_index matches label_to_vector."""
        for idx in range(min(4, self.basis.size())):  # Test first few
            vec1 = self.basis.vector_at_index(idx)
            label = self.basis.label_at_index(idx)
            vec2 = self.basis.label_to_vector(label)
            np.testing.assert_array_equal(vec1, vec2)

    def test_all_vectors_shape(self):
        """Test that all_vectors returns correct shape."""
        all_vecs = self.basis.all_vectors()
        expected_shape = (self.basis.size(), self.basis.m, self.basis.m)
        self.assertEqual(all_vecs.shape, expected_shape)

    def test_all_vectors_matches_iterate(self):
        """Test that all_vectors matches iterate_vectors."""
        all_vecs = self.basis.all_vectors()
        iter_vecs = list(self.basis.iterate_vectors())
        self.assertEqual(len(iter_vecs), self.basis.size())
        for i, vec in enumerate(iter_vecs):
            np.testing.assert_array_equal(all_vecs[i], vec)

    def test_linear_combination_identity(self):
        """Test linear combination with identity coefficients."""
        # Coefficient vector with 1 at position i should give the i-th basis vector
        for idx in range(min(3, self.basis.size())):
            coeffs = np.zeros(self.basis.size())
            coeffs[idx] = 1
            result = self.basis.linear_combination(coeffs)
            expected = self.basis.vector_at_index(idx)
            np.testing.assert_array_equal(result, expected)

    def test_linear_combination_sum(self):
        """Test linear combination sums correctly."""
        # Linear combination of all basis vectors with coeff 1 should sum them
        coeffs = np.ones(self.basis.size())
        result = self.basis.linear_combination(coeffs)
        # Each position should have value equal to number of basis vectors that have 1 there
        # For standard basis matrices, this is just the identity matrix
        expected = np.sum(self.basis.all_vectors(), axis=0)
        np.testing.assert_array_equal(result, expected)


class TestEndSnBlockDiagonalBasis(unittest.TestCase):
    """Test suite for EndSnBlockDiagonalBasis class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use n=2, d=2
        self.n = 2
        self.d = 2
        self.basis = EndSnBlockDiagonalBasis(self.n, self.d)

    def test_initialization(self):
        """Test that basis initializes correctly."""
        self.assertEqual(self.basis.n, self.n)
        self.assertEqual(self.basis.d, self.d)
        self.assertIsInstance(self.basis.bases, tuple)
        self.assertTrue(all(isinstance(b, EndSnIrrepBasis) for b in self.basis.bases))

    def test_partitions_match_n(self):
        """Test that partitions are generated for the correct n."""
        partitions = list(Partition.generate_all(self.n))
        self.assertEqual(len(self.basis.partitions), len(partitions))
        for p1, p2 in zip(self.basis.partitions, partitions):
            self.assertEqual(p1, p2)

    def test_number_of_bases(self):
        """Test that number of bases equals number of partitions."""
        num_partitions = len(list(Partition.generate_all(self.n)))
        self.assertEqual(len(self.basis.bases), num_partitions)

    def test_block_sizes_correct(self):
        """Test that block_sizes match the multiplicities of irreps."""
        for i, (partition, basis) in enumerate(zip(self.basis.partitions, self.basis.bases)):
            expected_size = SSYT.count(partition, self.d)
            self.assertEqual(self.basis.block_sizes[i], expected_size)
            self.assertEqual(basis.m, expected_size)

    def test_size_is_sum_of_squared_blocks(self):
        """Test that total size equals sum of m_λ^2."""
        expected_size = sum(m**2 for m in self.basis.block_sizes)
        self.assertEqual(self.basis.size(), expected_size)

    def test_iterate_labels_count(self):
        """Test that iterate_labels produces correct number of labels."""
        labels = list(self.basis.iterate_labels())
        self.assertEqual(len(labels), self.basis.size())

    def test_iterate_vectors_count(self):
        """Test that iterate_vectors produces correct number of vectors."""
        vectors = list(self.basis.iterate_vectors())
        self.assertEqual(len(vectors), self.basis.size())

    def test_iterate_vectors_are_block_diagonal(self):
        """Test that all vectors are BlockDiagonalMatrix instances."""
        for vec in self.basis.iterate_vectors():
            self.assertIsInstance(vec, BlockDiagonalMatrix)

    def test_vector_block_structure(self):
        """Test that vectors have correct block structure."""
        for vec in self.basis.iterate_vectors():
            self.assertEqual(len(vec.blocks), len(self.basis.block_sizes))
            self.assertEqual(vec.block_sizes, tuple(self.basis.block_sizes))

    def test_vector_has_one_nonzero_block(self):
        """Test that each basis vector has exactly one non-zero block."""
        for vec in self.basis.iterate_vectors():
            # Count non-zero blocks
            non_zero_blocks = sum(1 for block in vec.blocks if np.any(block != 0))
            self.assertEqual(non_zero_blocks, 1,
                           msg=f"Expected exactly 1 non-zero block, got {non_zero_blocks}")

    def test_roundtrip_label_index(self):
        """Test round-trip conversion between labels and indices."""
        # Test first few labels
        labels = list(self.basis.iterate_labels())[:5]
        for label in labels:
            idx = self.basis.label_to_index(label)
            label_back = self.basis.label_at_index(idx)
            self.assertEqual(label, label_back)

    def test_roundtrip_index_label(self):
        """Test round-trip conversion from index to label and back."""
        # Test first few indices
        for idx in range(min(5, self.basis.size())):
            label = self.basis.label_at_index(idx)
            idx_back = self.basis.label_to_index(label)
            self.assertEqual(idx, idx_back)

    def test_label_to_subbasis_correct_partition(self):
        """Test that _label_to_subbasis returns basis with matching partition."""
        labels = list(self.basis.iterate_labels())[:3]
        for label in labels:
            ssyt1, ssyt2 = label
            subbasis = self.basis.bases[self.basis._label_to_subbasis_index(label)]
            self.assertEqual(subbasis.partition, ssyt1.partition)
            self.assertEqual(subbasis.partition, ssyt2.partition)

    def test_linear_combination_splits_correctly(self):
        """Test that linear_combination splits coefficients correctly."""
        # Create coefficient vector
        coeffs = np.arange(self.basis.size())
        result = self.basis.linear_combination(coeffs)

        # Result should be a BlockDiagonalMatrix
        self.assertIsInstance(result, BlockDiagonalMatrix)
        self.assertEqual(len(result.blocks), len(self.basis.block_sizes))

    def test_linear_combination_block_sizes(self):
        """Test that linear combination preserves block sizes."""
        coeffs = np.ones(self.basis.size())
        result = self.basis.linear_combination(coeffs)
        self.assertEqual(result.block_sizes, tuple(self.basis.block_sizes))

    def test_linear_combination_identity_coefficient(self):
        """Test linear combination with single 1 coefficient."""
        # Test a few indices
        for test_idx in range(min(3, self.basis.size())):
            coeffs = np.zeros(self.basis.size())
            coeffs[test_idx] = 1
            result = self.basis.linear_combination(coeffs)

            # Result should match the basis vector at that index
            expected = self.basis.vector_at_index(test_idx)

            # Compare the full matrices
            result_full = result.to_full_matrix()
            expected_full = expected.to_full_matrix()
            np.testing.assert_array_equal(result_full, expected_full)

    def test_all_vectors_shape(self):
        """Test that all_vectors returns correctly shaped array."""
        all_vecs = self.basis.all_vectors()
        total_dim = sum(self.basis.block_sizes)
        expected_shape = (self.basis.size(), total_dim, total_dim)
        self.assertEqual(all_vecs.shape, expected_shape)

    def test_all_vectors_are_full_matrices(self):
        """Test that all_vectors converts BlockDiagonalMatrix to full matrices."""
        all_vecs = self.basis.all_vectors()
        # Each vector should be a 2D matrix (not BlockDiagonalMatrix)
        for vec in all_vecs:
            self.assertEqual(len(vec.shape), 2)

    def test_basis_indices_start_correct(self):
        """Test that basis_indices_start is computed correctly."""
        expected = np.cumsum([0] + self.basis.basis_sizes[:-1])
        np.testing.assert_array_equal(self.basis.basis_indices_start, expected)

    def test_index_ranges_non_overlapping(self):
        """Test that index ranges for different bases don't overlap."""
        # Check that indices are partitioned correctly
        indices_by_basis = [[] for _ in self.basis.bases]

        for idx in range(self.basis.size()):
            label = self.basis.label_at_index(idx)
            basis_idx = self.basis.partitions.index(label[0].partition)
            indices_by_basis[basis_idx].append(idx)

        # Check each basis gets the expected number of indices
        for basis_idx, (basis, indices) in enumerate(zip(self.basis.bases, indices_by_basis)):
            self.assertEqual(len(indices), basis.size(),
                           f"Basis {basis_idx} should have {basis.size()} indices, got {len(indices)}")

        # Check indices are contiguous
        for indices in indices_by_basis:
            if indices:
                sorted_indices = sorted(indices)
                self.assertEqual(sorted_indices, list(range(sorted_indices[0], sorted_indices[-1] + 1)))


class TestMatrixDirectSumBasisConsistency(unittest.TestCase):
    """Test consistency properties of MatrixDirectSumBasis."""

    def setUp(self):
        """Set up test fixtures."""
        self.basis = EndSnBlockDiagonalBasis(2, 2)

    def test_size_matches_iteration(self):
        """Test that size() matches the number of iterated labels."""
        labels = list(self.basis.iterate_labels())
        self.assertEqual(len(labels), self.basis.size())

    def test_all_indices_covered(self):
        """Test that all indices from 0 to size-1 are covered."""
        all_labels = list(self.basis.iterate_labels())
        all_indices = [self.basis.label_to_index(label) for label in all_labels]

        # Should have exactly size() indices
        self.assertEqual(len(all_indices), self.basis.size())

        # Should cover 0 to size-1 without duplicates
        self.assertEqual(sorted(all_indices), list(range(self.basis.size())))

    def test_labels_unique(self):
        """Test that all labels are unique."""
        all_labels = list(self.basis.iterate_labels())
        # Can't use set directly with SSYT, so compare lengths
        label_indices = [self.basis.label_to_index(label) for label in all_labels]
        self.assertEqual(len(set(label_indices)), len(all_labels))


class TestEndSnBlockDiagonalBasisSmallCases(unittest.TestCase):
    """Test EndSnBlockDiagonalBasis with small parameter values."""

    def test_n1_d2(self):
        """Test with n=1, d=2."""
        basis = EndSnBlockDiagonalBasis(n=1, d=2)
        # For n=1, there's only one partition: [1]
        self.assertEqual(len(basis.partitions), 1)
        self.assertEqual(basis.partitions[0], Partition([1]))

        # SSYT.count([1], 2) should be 2 (tableaux [0] and [1])
        expected_size = 2
        self.assertEqual(basis.block_sizes[0], expected_size)

        # Total size should be 2^2 = 4
        self.assertEqual(basis.size(), expected_size ** 2)

    def test_n2_d2(self):
        """Test with n=2, d=2."""
        basis = EndSnBlockDiagonalBasis(n=2, d=2)
        # For n=2, there are two partitions: [2] and [1,1]
        self.assertEqual(len(basis.partitions), 2)

        # Verify size is sum of squared block sizes
        total_size = sum(m**2 for m in basis.block_sizes)
        self.assertEqual(basis.size(), total_size)

    def test_roundtrip_all_labels(self):
        """Test round-trip for all labels in small case."""
        basis = EndSnBlockDiagonalBasis(n=2, d=2)
        for label in basis.iterate_labels():
            idx = basis.label_to_index(label)
            label_back = basis.label_at_index(idx)
            self.assertEqual(label, label_back)


class TestEndSnOrbitBasis(unittest.TestCase):
    """Test suite for EndSnOrbitBasis class."""
    def test_matrix_tensor_product(self):
        """Test that matrix tensor product is correct."""
        for n in [2,3,4]:
            for d in [2,3]:
                mat = np.random.randn(d, d)
                basis = EndSnOrbitBasis(n, d)
                coeffs = basis.coefficients_for_tensor_product(mat)
                self.assertEqual(coeffs.shape, (basis.size(),))

                explicit = mat
                for i in range(1, n):
                    explicit = np.kron(explicit, mat)
                np.testing.assert_array_almost_equal(basis.linear_combination(coeffs), explicit)


class TestEndSnOrbitBasisTranspose(unittest.TestCase):
    """Test suite for EndSnOrbitBasis.transpose() method."""

    def test_transpose_1d_coeffs_mathematical_correctness(self):
        """Test that transpose gives coefficients of the transposed matrix (1D case)."""
        for n in [2, 3]:
            for d in [2, 3]:
                basis = EndSnOrbitBasis(n, d)
                coeffs = np.random.randn(basis.size())

                # Get the matrix from coefficients
                matrix = basis.linear_combination(coeffs)

                # Transpose the matrix directly
                transposed_matrix = matrix.T

                # Get the transposed coefficients
                transposed_coeffs = basis.transpose(coeffs)

                # Reconstruct the matrix from transposed coefficients
                reconstructed = basis.linear_combination(transposed_coeffs)

                # The reconstructed matrix should equal the transposed matrix
                np.testing.assert_array_almost_equal(reconstructed, transposed_matrix)

    def test_transpose_2d_coeffs_batch(self):
        """Test transpose with 2D coefficient array (batched matrices)."""
        basis = EndSnOrbitBasis(n=2, d=3)
        batch_size = 5
        coeffs = np.random.randn(batch_size, basis.size())

        transposed_coeffs = basis.transpose(coeffs, axis=-1)

        # Verify each batch element independently
        for i in range(batch_size):
            matrix = basis.linear_combination(coeffs[i])
            expected_transposed = matrix.T
            reconstructed = basis.linear_combination(transposed_coeffs[i])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_3d_coeffs_different_axes(self):
        """Test transpose with 3D coefficient array on different axes."""
        basis = EndSnOrbitBasis(n=2, d=2)
        shape = (3, basis.size(), 2)
        coeffs = np.random.randn(*shape)

        # Transpose along axis=1 (where the coefficients are)
        transposed_coeffs = basis.transpose(coeffs, axis=1)

        # Verify the result by checking samples
        for i in range(shape[0]):
            for k in range(shape[2]):
                matrix = basis.linear_combination(coeffs[i, :, k])
                expected_transposed = matrix.T
                reconstructed = basis.linear_combination(transposed_coeffs[i, :, k])
                np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_axis_0(self):
        """Test transpose with coefficients on axis 0."""
        basis = EndSnOrbitBasis(n=2, d=2)
        shape = (basis.size(), 4)
        coeffs = np.random.randn(*shape)

        transposed_coeffs = basis.transpose(coeffs, axis=0)

        # Verify
        for j in range(shape[1]):
            matrix = basis.linear_combination(coeffs[:, j])
            expected_transposed = matrix.T
            reconstructed = basis.linear_combination(transposed_coeffs[:, j])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_negative_axis(self):
        """Test that negative axis indexing works correctly."""
        basis = EndSnOrbitBasis(n=2, d=2)
        shape = (3, basis.size())
        coeffs = np.random.randn(*shape)

        # axis=-1 should be the same as axis=1 for 2D array
        transposed_1 = basis.transpose(coeffs, axis=-1)
        transposed_2 = basis.transpose(coeffs, axis=1)

        np.testing.assert_array_equal(transposed_1, transposed_2)

    def test_transpose_involutive(self):
        """Test that transpose is an involution (transposing twice gives original)."""
        for n in [2, 3]:
            for d in [2, 3]:
                basis = EndSnOrbitBasis(n, d)
                coeffs = np.random.randn(basis.size())

                double_transposed = basis.transpose(basis.transpose(coeffs))

                np.testing.assert_array_almost_equal(double_transposed, coeffs)

    def test_transpose_preserves_shape(self):
        """Test that transpose preserves the coefficient array shape."""
        basis = EndSnOrbitBasis(n=2, d=2)
        for shape in [(basis.size(),), (3, basis.size()), (2, 3, basis.size())]:
            coeffs = np.random.randn(*shape)
            transposed = basis.transpose(coeffs)
            self.assertEqual(transposed.shape, coeffs.shape)

    def test_transpose_identity_matrix(self):
        """Test that transposing the identity matrix gives the identity."""
        for n in [2, 3]:
            for d in [2, 3]:
                basis = EndSnOrbitBasis(n, d)
                # Get the identity matrix coefficients
                identity_coeffs = basis.coefficients_of_identity()

                transposed = basis.transpose(identity_coeffs)

                # Identity is symmetric, so transpose should give same coefficients
                np.testing.assert_array_almost_equal(transposed, identity_coeffs)

    def test_transpose_symmetric_matrix(self):
        """Test that transposing a symmetric matrix gives the same coefficients."""
        for n in [2, 3]:
            for d in [2, 3]:
                basis = EndSnOrbitBasis(n, d)

                # Create a symmetric matrix: A = B + B^T for some random B
                rand_coeffs = np.random.randn(basis.size())
                symmetric_coeffs = rand_coeffs + basis.transpose(rand_coeffs)

                transposed = basis.transpose(symmetric_coeffs)

                np.testing.assert_array_almost_equal(transposed, symmetric_coeffs)

    def test_transpose_index_lookup_consistency(self):
        """Test that transpose_index_lookup is consistent with direct orbit transpose."""
        for n in [2, 3]:
            for d in [2, 3]:
                basis = EndSnOrbitBasis(n, d)
                lookup = basis.transpose_index_lookup()

                # Verify each entry matches the expected transposed orbit index
                for idx, orbit in enumerate(basis.iterate_labels()):
                    transposed_orbit = orbit.transpose()
                    expected_idx = basis.label_to_index(transposed_orbit)
                    self.assertEqual(lookup[idx], expected_idx)


class TestEndSnIrrepBasisTranspose(unittest.TestCase):
    """Test suite for EndSnIrrepBasis.transpose() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.partition = Partition([2])
        self.d = 2
        self.basis = EndSnIrrepBasis(self.partition, self.d)

    def test_transpose_1d_coeffs_mathematical_correctness(self):
        """Test that transpose gives coefficients of the transposed matrix (1D case)."""
        coeffs = np.random.randn(self.basis.size())

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

    def test_transpose_2d_coeffs_batch(self):
        """Test transpose with 2D coefficient array (batched matrices)."""
        batch_size = 4
        coeffs = np.random.randn(batch_size, self.basis.size())

        transposed_coeffs = self.basis.transpose(coeffs, axis=-1)

        # Verify each batch element independently
        for i in range(batch_size):
            matrix = self.basis.linear_combination(coeffs[i])
            expected_transposed = matrix.T
            reconstructed = self.basis.linear_combination(transposed_coeffs[i])
            np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_3d_coeffs_different_axes(self):
        """Test transpose with 3D coefficient array on different axes."""
        shape = (3, self.basis.size(), 2)
        coeffs = np.random.randn(*shape)

        # Transpose along axis=1 (where the coefficients are)
        transposed_coeffs = self.basis.transpose(coeffs, axis=1)

        # Verify the result by checking samples
        for i in range(shape[0]):
            for k in range(shape[2]):
                matrix = self.basis.linear_combination(coeffs[i, :, k])
                expected_transposed = matrix.T
                reconstructed = self.basis.linear_combination(transposed_coeffs[i, :, k])
                np.testing.assert_array_almost_equal(reconstructed, expected_transposed)

    def test_transpose_axis_0(self):
        """Test transpose with coefficients on axis 0."""
        shape = (self.basis.size(), 5)
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
        shape = (2, self.basis.size())
        coeffs = np.random.randn(*shape)

        # axis=-1 should be the same as axis=1 for 2D array
        transposed_1 = self.basis.transpose(coeffs, axis=-1)
        transposed_2 = self.basis.transpose(coeffs, axis=1)

        np.testing.assert_array_equal(transposed_1, transposed_2)

    def test_transpose_involutive(self):
        """Test that transpose is an involution (transposing twice gives original)."""
        coeffs = np.random.randn(self.basis.size())

        double_transposed = self.basis.transpose(self.basis.transpose(coeffs))

        np.testing.assert_array_almost_equal(double_transposed, coeffs)

    def test_transpose_preserves_shape(self):
        """Test that transpose preserves the coefficient array shape."""
        for shape in [(self.basis.size(),), (3, self.basis.size()), (2, 3, self.basis.size())]:
            coeffs = np.random.randn(*shape)
            transposed = self.basis.transpose(coeffs)
            self.assertEqual(transposed.shape, coeffs.shape)

    def test_transpose_identity_matrix(self):
        """Test that transposing the identity matrix gives the identity."""
        # Create identity matrix coefficients
        coeffs = np.zeros(self.basis.size())
        for i in range(self.basis.m):
            # Identity has 1 on diagonal: (i, i) entry
            label = self.basis.label_at_index(i * self.basis.m + i)
            idx = self.basis.label_to_index(label)
            coeffs[idx] = 1.0

        transposed = self.basis.transpose(coeffs)

        # Identity is symmetric, so transpose should give same coefficients
        np.testing.assert_array_almost_equal(transposed, coeffs)

    def test_transpose_symmetric_matrix(self):
        """Test that transposing a symmetric matrix gives the same coefficients."""
        # Create a symmetric matrix: A = B + B^T for some random B
        rand_coeffs = np.random.randn(self.basis.size())
        symmetric_coeffs = rand_coeffs + self.basis.transpose(rand_coeffs)

        transposed = self.basis.transpose(symmetric_coeffs)

        np.testing.assert_array_almost_equal(transposed, symmetric_coeffs)

    def test_transpose_different_partitions(self):
        """Test transpose for different partition shapes."""
        partitions_and_d = [
            (Partition([2]), 2),
            (Partition([1, 1]), 2),
            (Partition([3]), 3),
            (Partition([2, 1]), 3),
            (Partition([1, 1, 1]), 3),
        ]

        for partition, d in partitions_and_d:
            basis = EndSnIrrepBasis(partition, d)
            if basis.size() == 0:
                continue

            coeffs = np.random.randn(basis.size())

            # Verify mathematical correctness
            matrix = basis.linear_combination(coeffs)
            transposed_matrix = matrix.T
            transposed_coeffs = basis.transpose(coeffs)
            reconstructed = basis.linear_combination(transposed_coeffs)

            np.testing.assert_array_almost_equal(
                reconstructed, transposed_matrix,
                err_msg=f"Failed for partition {partition} with d={d}"
            )


class TestEndSnBlockDiagonalBasisTranspose(unittest.TestCase):
    """Test suite for EndSnBlockDiagonalBasis.transpose() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.n = 2
        self.d = 2
        self.basis = EndSnBlockDiagonalBasis(self.n, self.d)

    def test_transpose_1d_coeffs_mathematical_correctness(self):
        """Test that transpose gives coefficients of the transposed block diagonal matrix (1D case)."""
        coeffs = np.random.randn(self.basis.size())

        # Get the block diagonal matrix from coefficients
        block_matrix = self.basis.linear_combination(coeffs)

        # Transpose each block of the block diagonal matrix
        # For block diagonal matrices, transpose is done block-wise
        transposed_blocks = [block.T for block in block_matrix.blocks]

        # Get the transposed coefficients
        transposed_coeffs = self.basis.transpose(coeffs)

        # Reconstruct the block diagonal matrix from transposed coefficients
        reconstructed = self.basis.linear_combination(transposed_coeffs)

        # Compare block by block
        for i, (reconstructed_block, expected_block) in enumerate(zip(reconstructed.blocks, transposed_blocks)):
            np.testing.assert_array_almost_equal(
                reconstructed_block, expected_block,
                err_msg=f"Block {i} mismatch"
            )

    def test_transpose_2d_coeffs_batch(self):
        """Test transpose with 2D coefficient array (batched)."""
        batch_size = 4
        coeffs = np.random.randn(batch_size, self.basis.size())

        transposed_coeffs = self.basis.transpose(coeffs, axis=-1)

        # Verify each batch element independently
        for b in range(batch_size):
            block_matrix = self.basis.linear_combination(coeffs[b])
            transposed_blocks = [block.T for block in block_matrix.blocks]

            reconstructed = self.basis.linear_combination(transposed_coeffs[b])

            for i, (rec_block, exp_block) in enumerate(zip(reconstructed.blocks, transposed_blocks)):
                np.testing.assert_array_almost_equal(
                    rec_block, exp_block,
                    err_msg=f"Batch {b}, Block {i} mismatch"
                )

    def test_transpose_3d_coeffs_different_axes(self):
        """Test transpose with 3D coefficient array on different axes."""
        shape = (2, self.basis.size(), 3)
        coeffs = np.random.randn(*shape)

        # Transpose along axis=1 (where the coefficients are)
        transposed_coeffs = self.basis.transpose(coeffs, axis=1)

        # Verify samples
        for i in range(shape[0]):
            for k in range(shape[2]):
                block_matrix = self.basis.linear_combination(coeffs[i, :, k])
                transposed_blocks = [block.T for block in block_matrix.blocks]

                reconstructed = self.basis.linear_combination(transposed_coeffs[i, :, k])

                for j, (rec_block, exp_block) in enumerate(zip(reconstructed.blocks, transposed_blocks)):
                    np.testing.assert_array_almost_equal(rec_block, exp_block)

    def test_transpose_axis_0(self):
        """Test transpose with coefficients on axis 0."""
        shape = (self.basis.size(), 3)
        coeffs = np.random.randn(*shape)

        transposed_coeffs = self.basis.transpose(coeffs, axis=0)

        # Verify
        for j in range(shape[1]):
            block_matrix = self.basis.linear_combination(coeffs[:, j])
            transposed_blocks = [block.T for block in block_matrix.blocks]

            reconstructed = self.basis.linear_combination(transposed_coeffs[:, j])

            for i, (rec_block, exp_block) in enumerate(zip(reconstructed.blocks, transposed_blocks)):
                np.testing.assert_array_almost_equal(rec_block, exp_block)

    def test_transpose_negative_axis(self):
        """Test that negative axis indexing works correctly."""
        shape = (2, self.basis.size())
        coeffs = np.random.randn(*shape)

        # axis=-1 should be the same as axis=1 for 2D array
        transposed_1 = self.basis.transpose(coeffs, axis=-1)
        transposed_2 = self.basis.transpose(coeffs, axis=1)

        np.testing.assert_array_equal(transposed_1, transposed_2)

    def test_transpose_involutive(self):
        """Test that transpose is an involution (transposing twice gives original)."""
        coeffs = np.random.randn(self.basis.size())

        double_transposed = self.basis.transpose(self.basis.transpose(coeffs))

        np.testing.assert_array_almost_equal(double_transposed, coeffs)

    def test_transpose_preserves_shape(self):
        """Test that transpose preserves the coefficient array shape."""
        for shape in [(self.basis.size(),), (3, self.basis.size()), (2, 3, self.basis.size())]:
            coeffs = np.random.randn(*shape)
            transposed = self.basis.transpose(coeffs)
            self.assertEqual(transposed.shape, coeffs.shape)

    def test_transpose_symmetric_blocks(self):
        """Test that transposing symmetric block matrices gives the same coefficients."""
        # Create a symmetric block matrix: for each block, make it symmetric
        coeffs = np.random.randn(self.basis.size())
        symmetric_coeffs = coeffs + self.basis.transpose(coeffs)

        transposed = self.basis.transpose(symmetric_coeffs)

        np.testing.assert_array_almost_equal(transposed, symmetric_coeffs)

    def test_transpose_larger_case(self):
        """Test transpose with larger n and d values."""
        for n, d in [(3, 2), (2, 3)]:
            basis = EndSnBlockDiagonalBasis(n, d)
            coeffs = np.random.randn(basis.size())

            block_matrix = basis.linear_combination(coeffs)
            transposed_blocks = [block.T for block in block_matrix.blocks]

            transposed_coeffs = basis.transpose(coeffs)
            reconstructed = basis.linear_combination(transposed_coeffs)

            for i, (rec_block, exp_block) in enumerate(zip(reconstructed.blocks, transposed_blocks)):
                np.testing.assert_array_almost_equal(
                    rec_block, exp_block,
                    err_msg=f"Failed for n={n}, d={d}, block {i}"
                )

    def test_transpose_consistent_with_subbases(self):
        """Test that transpose of block diagonal is consistent with transposing individual blocks."""
        coeffs = np.random.randn(self.basis.size())

        # Split coefficients by block
        block_coeffs = np.split(coeffs, self.basis.basis_indices_start[1:])

        # Transpose each block's coefficients using subbasis transpose
        transposed_block_coeffs = [
            subbasis.transpose(bc)
            for subbasis, bc in zip(self.basis.bases, block_coeffs)
        ]

        # Concatenate
        expected_transposed = np.concatenate(transposed_block_coeffs)

        # Compare with direct transpose
        actual_transposed = self.basis.transpose(coeffs)

        np.testing.assert_array_almost_equal(actual_transposed, expected_transposed)


if __name__ == '__main__':
    unittest.main()

