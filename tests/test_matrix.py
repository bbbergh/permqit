import unittest
import numpy as np
from permqit.algebra.matrix import BlockDiagonalMatrix


class TestBlockDiagonalMatrixInit(unittest.TestCase):
    """Test BlockDiagonalMatrix initialization and validation."""

    def test_init_valid_scalar_blocks(self):
        """Test initialization with scalar blocks."""
        blocks = [np.array(2), np.array(3)]
        sizes = (2, 3)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        self.assertEqual(bdm.block_sizes, sizes)
        self.assertEqual(len(bdm.blocks), 2)

    def test_init_valid_1d_blocks(self):
        """Test initialization with 1D (diagonal) blocks."""
        blocks = [np.array([1, 2]), np.array([3, 4, 5])]
        sizes = (2, 3)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        self.assertEqual(bdm.block_sizes, sizes)

    def test_init_valid_2d_blocks(self):
        """Test initialization with 2D square matrix blocks."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        self.assertEqual(bdm.block_sizes, sizes)

    def test_init_mixed_blocks(self):
        """Test initialization with mixed block types."""
        blocks = [
            np.array(2),  # scalar
            np.array([1, 2]),  # 1D diagonal
            np.array([[1, 2], [3, 4]])  # 2D matrix
        ]
        sizes = (3, 2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        self.assertEqual(bdm.block_sizes, sizes)

    def test_init_wrong_number_of_blocks(self):
        """Test that mismatched number of blocks and sizes raises error."""
        blocks = [np.array(2), np.array(3)]
        sizes = (2, 3, 4)  # Wrong number
        with self.assertRaises(ValueError):
            BlockDiagonalMatrix(blocks, sizes)

    def test_init_wrong_1d_block_size(self):
        """Test that 1D block with wrong size raises error."""
        blocks = [np.array([1, 2, 3])]  # Size 3
        sizes = (2,)  # Expected size 2
        with self.assertRaises(ValueError):
            BlockDiagonalMatrix(blocks, sizes)

    def test_init_wrong_2d_block_size(self):
        """Test that 2D block with wrong size raises error."""
        blocks = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]  # 3x3
        sizes = (2,)  # Expected size 2
        with self.assertRaises(ValueError):
            BlockDiagonalMatrix(blocks, sizes)

    def test_init_non_square_2d_block(self):
        """Test that non-square 2D block raises error."""
        blocks = [np.array([[1, 2, 3], [4, 5, 6]])]  # 2x3, not square
        sizes = (2,)
        with self.assertRaises(ValueError):
            BlockDiagonalMatrix(blocks, sizes)

    def test_skip_validation(self):
        """Test that validation can be skipped."""
        blocks = [np.array([1, 2, 3])]  # Wrong size
        sizes = (2,)
        # Should not raise when validate=False
        bdm = BlockDiagonalMatrix(blocks, sizes, validate=False)
        self.assertEqual(bdm.block_sizes, sizes)


class TestBlockDiagonalMatrixProperties(unittest.TestCase):
    """Test properties of BlockDiagonalMatrix."""

    def test_total_size(self):
        """Test total_size property."""
        blocks = [np.array(2), np.array(3), np.array(4)]
        sizes = (2, 3, 4)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        self.assertEqual(bdm.total_size, 9)

    def test_to_full_matrix_scalar_blocks(self):
        """Test conversion to full matrix with scalar blocks."""
        blocks = [np.array(2), np.array(3)]
        sizes = (2, 3)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        full = bdm.to_full_matrix()

        expected = np.array([
            [2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 3]
        ])
        np.testing.assert_array_equal(full, expected)

    def test_to_full_matrix_1d_blocks(self):
        """Test conversion to full matrix with 1D diagonal blocks."""
        blocks = [np.array([1, 2]), np.array([3, 4, 5])]
        sizes = (2, 3)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        full = bdm.to_full_matrix()

        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5]
        ])
        np.testing.assert_array_equal(full, expected)

    def test_to_full_matrix_2d_blocks(self):
        """Test conversion to full matrix with 2D blocks."""
        blocks = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5]])
        ]
        sizes = (2, 1)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        full = bdm.to_full_matrix()

        expected = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 5]
        ])
        np.testing.assert_array_equal(full, expected)

    def test_transpose(self):
        """Test transpose property."""
        blocks = [
            np.array([[1, 2], [3, 4]]),
            np.array([5, 6])  # 1D stays same
        ]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        bdm_t = bdm.T

        expected_blocks = [
            np.array([[1, 3], [2, 4]]),
            np.array([5, 6])
        ]

        for b1, b2 in zip(bdm_t.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_conjugate(self):
        """Test conjugate method."""
        blocks = [
            np.array([[1+2j, 3-1j], [4j, 5]]),
            np.array([2+3j, 1-1j])
        ]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)
        bdm_conj = bdm.conjugate()

        expected_blocks = [
            np.array([[1-2j, 3+1j], [-4j, 5]]),
            np.array([2-3j, 1+1j])
        ]

        for b1, b2 in zip(bdm_conj.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)


class TestBlockDiagonalMatrixArithmetic(unittest.TestCase):
    """Test arithmetic operations on BlockDiagonalMatrix."""

    def test_add_block_diagonal(self):
        """Test addition of two block diagonal matrices."""
        blocks1 = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        blocks2 = [np.array([[2, 1], [1, 2]]), np.array([1, 2])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)
        result = bdm1 + bdm2

        expected_blocks = [np.array([[3, 3], [4, 6]]), np.array([6, 8])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_add_mismatched_sizes(self):
        """Test that adding matrices with different block sizes raises error."""
        bdm1 = BlockDiagonalMatrix([np.array(1)], (2,))
        bdm2 = BlockDiagonalMatrix([np.array(1)], (3,))

        with self.assertRaises(AssertionError):
            bdm1 + bdm2

    def test_subtract_block_diagonal(self):
        """Test subtraction of two block diagonal matrices."""
        blocks1 = [np.array([[5, 2], [3, 4]]), np.array([10, 6])]
        blocks2 = [np.array([[2, 1], [1, 2]]), np.array([5, 2])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)
        result = bdm1 - bdm2

        expected_blocks = [np.array([[3, 1], [2, 2]]), np.array([5, 4])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_multiply_by_scalar(self):
        """Test multiplication by scalar."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = bdm * 3
        expected_blocks = [np.array([[3, 6], [9, 12]]), np.array([15, 18])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_right_multiply_by_scalar(self):
        """Test right multiplication by scalar (scalar * matrix)."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = 3 * bdm
        expected_blocks = [np.array([[3, 6], [9, 12]]), np.array([15, 18])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_divide_by_scalar(self):
        """Test division by scalar."""
        blocks = [np.array([[6, 12], [18, 24]]), np.array([15, 18])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = bdm / 3
        expected_blocks = [np.array([[2, 4], [6, 8]]), np.array([5, 6])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_floor_divide_by_scalar(self):
        """Test floor division by scalar."""
        blocks = [np.array([[7, 13], [19, 25]]), np.array([16, 19])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = bdm // 3
        expected_blocks = [np.array([[2, 4], [6, 8]]), np.array([5, 6])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_negation(self):
        """Test negation operator."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = -bdm
        expected_blocks = [np.array([[-1, -2], [-3, -4]]), np.array([-5, -6])]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_right_add_with_scalar(self):
        """Test right addition (scalar + matrix)."""
        blocks = [np.array(2)]
        sizes = (2,)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        # This should apply this as a multiple of identity
        result = 5 + bdm
        expected = np.array([[7, 0], [0, 7]])
        np.testing.assert_array_equal(result.to_full_matrix(), expected)

    def test_right_subtract(self):
        """Test right subtraction (scalar - matrix)."""
        blocks = [np.array(2)]
        sizes = (2,)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        # This should apply this as a multiple of identity
        result = 10 - bdm
        expected = np.array([[8, 0], [0, 8]])
        np.testing.assert_array_equal(result.to_full_matrix(), expected)


class TestBlockDiagonalMatrixMultiplication(unittest.TestCase):
    """Test matrix multiplication operations."""

    def test_matmul_block_diagonal_matrices(self):
        """Test multiplication of two block diagonal matrices."""
        blocks1 = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        blocks2 = [np.array([[2, 0], [1, 2]]), np.array([3, 2])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)
        result = bdm1 @ bdm2

        # Block 0: [[1,2],[3,4]] @ [[2,0],[1,2]] = [[4,4],[10,8]]
        # Block 1: [2,3] * [3,2] = [6,6] (elementwise for diagonal)
        expected_blocks = [
            np.array([[4, 4], [10, 8]]),
            np.array([6, 6])
        ]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_matmul_with_vector(self):
        """Test multiplication of block diagonal matrix with vector."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        vec = np.array([1, 2, 3, 4])
        result = bdm @ vec

        # Block 0: [[1,2],[3,4]] @ [1,2] = [5, 11]
        # Block 1: [2,3] * [3,4] = [6, 12]
        expected = np.array([5, 11, 6, 12])

        np.testing.assert_array_equal(result, expected)

    def test_matmul_with_2d_array(self):
        """Test multiplication with 2D array (matrix with multiple columns)."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2])]
        sizes = (2, 1)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = bdm @ arr

        # Block 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10], [15,22]]
        # Block 1: [2] * [5, 6] = [10, 12]
        expected = np.array([[7, 10], [15, 22], [10, 12]])

        np.testing.assert_array_equal(result, expected)

    def test_rmatmul_with_vector(self):
        """Test left multiplication with vector (vector @ matrix)."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        vec = np.array([1, 2, 3, 4])
        result = vec @ bdm

        # Block 0: [1,2] @ [[1,2],[3,4]] = [7, 10]
        # Block 1: [3,4] * [2,3] = [6, 12]
        expected = np.array([7, 10, 6, 12])

        np.testing.assert_array_equal(result, expected)

    def test_rmatmul_with_2d_array(self):
        """Test left multiplication with 2D array."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2])]
        sizes = (2, 1)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = arr @ bdm

        # Block 0: [[1,2],[4,5]] @ [[1,2],[3,4]] = [[7,10], [19,28]]
        # Block 1: [[3],[6]] * [2] = [[6], [12]]
        expected = np.array([[7, 10, 6], [19, 28, 12]])

        np.testing.assert_array_equal(result, expected)

    def test_matmul_with_scalar_blocks(self):
        """Test multiplication with scalar blocks (represented as scalars times identity)."""
        blocks1 = [np.array(2), np.array(3)]
        blocks2 = [np.array(4), np.array(5)]
        sizes = (2, 3)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)
        result = bdm1 @ bdm2

        # Scalar blocks multiply: 2*4=8, 3*5=15
        expected_blocks = [np.array(8), np.array(15)]

        for b1, b2 in zip(result.blocks, expected_blocks):
            np.testing.assert_array_equal(b1, b2)

    def test_matmul_mismatched_sizes(self):
        """Test that matmul with mismatched block sizes raises error."""
        bdm1 = BlockDiagonalMatrix([np.array(1)], (2,))
        bdm2 = BlockDiagonalMatrix([np.array(1)], (3,))

        with self.assertRaises(AssertionError):
            bdm1 @ bdm2

    def test_matmul_wrong_vector_size(self):
        """Test that matmul with wrong vector size raises error."""
        bdm = BlockDiagonalMatrix([np.array(1)], (2,))
        vec = np.array([1, 2, 3])  # Size 3, expected 2

        with self.assertRaises(AssertionError):
            bdm @ vec


class TestBlockDiagonalMatrixEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_single_block(self):
        """Test matrix with single block."""
        blocks = [np.array([[1, 2], [3, 4]])]
        sizes = (2,)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        full = bdm.to_full_matrix()
        np.testing.assert_array_equal(full, blocks[0])

    def test_large_number_of_blocks(self):
        """Test matrix with many small blocks."""
        n_blocks = 10
        blocks = [np.array(i) for i in range(1, n_blocks + 1)]
        sizes = tuple([1] * n_blocks)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        self.assertEqual(len(bdm.blocks), n_blocks)
        self.assertEqual(bdm.total_size, n_blocks)

    def test_zero_blocks(self):
        """Test blocks containing zeros."""
        blocks = [np.array([[0, 0], [0, 0]]), np.array([0, 0])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        full = bdm.to_full_matrix()
        expected = np.zeros((4, 4))
        np.testing.assert_array_equal(full, expected)

    def test_identity_blocks(self):
        """Test identity matrix as block diagonal."""
        blocks = [np.array([[1, 0], [0, 1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
        sizes = (2, 3)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        full = bdm.to_full_matrix()
        expected = np.eye(5)
        np.testing.assert_array_equal(full, expected)

    def test_multiply_by_zero(self):
        """Test multiplication by zero."""
        blocks = [np.array([[1, 2], [3, 4]])]
        sizes = (2,)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = bdm * 0
        np.testing.assert_array_equal(result.blocks[0], np.zeros((2, 2)))

    def test_multiply_by_one(self):
        """Test multiplication by one (identity)."""
        blocks = [np.array([[1, 2], [3, 4]])]
        sizes = (2,)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result = bdm * 1
        np.testing.assert_array_equal(result.blocks[0], blocks[0])


class TestBlockDiagonalMatrixEquality(unittest.TestCase):
    """Test equality and hashing."""

    def test_equality_same_matrices(self):
        """Test that identical matrices are equal."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks, sizes)
        bdm2 = BlockDiagonalMatrix(blocks, sizes)

        self.assertTrue(bdm1 == bdm2)

    def test_equality_different_blocks(self):
        """Test that matrices with different blocks are not equal."""
        blocks1 = [np.array([[1, 2], [3, 4]])]
        blocks2 = [np.array([[1, 2], [3, 5]])]
        sizes = (2,)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)

        self.assertFalse(bdm1 == bdm2)

    def test_equality_different_sizes(self):
        """Test that matrices with different block sizes are not equal."""
        bdm1 = BlockDiagonalMatrix([np.array(1)], (2,))
        bdm2 = BlockDiagonalMatrix([np.array(1)], (3,))

        self.assertFalse(bdm1 == bdm2)

    def test_hash_same_matrices(self):
        """Test that identical matrices have same hash."""
        blocks = [np.array([[1, 2], [3, 4]])]
        sizes = (2,)

        bdm1 = BlockDiagonalMatrix(blocks, sizes)
        bdm2 = BlockDiagonalMatrix(blocks, sizes)

        self.assertEqual(hash(bdm1), hash(bdm2))

    def test_hash_usable_in_set(self):
        """Test that BlockDiagonalMatrix can be used in a set."""
        blocks = [np.array([[1, 2], [3, 4]])]
        sizes = (2,)

        bdm1 = BlockDiagonalMatrix(blocks, sizes)
        bdm2 = BlockDiagonalMatrix(blocks, sizes)

        s = {bdm1, bdm2}
        self.assertEqual(len(s), 1)  # Should be treated as same element


class TestBlockDiagonalMatrixStringRepresentation(unittest.TestCase):
    """Test string representations."""

    def test_str(self):
        """Test __str__ method."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        s = str(bdm)
        self.assertIn("⊕", s)  # Should contain direct sum symbol



class TestBlockDiagonalMatrixConsistency(unittest.TestCase):
    """Test consistency between block operations and full matrix operations."""

    def test_matmul_consistency_with_full_matrix(self):
        """Test that block matmul gives same result as full matrix matmul."""
        blocks1 = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        blocks2 = [np.array([[2, 1], [1, 2]]), np.array([4, 5])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)

        # Block multiplication
        result_block = bdm1 @ bdm2
        result_block_full = result_block.to_full_matrix()

        # Full matrix multiplication
        full1 = bdm1.to_full_matrix()
        full2 = bdm2.to_full_matrix()
        result_full = full1 @ full2

        np.testing.assert_array_almost_equal(result_block_full, result_full)

    def test_addition_consistency_with_full_matrix(self):
        """Test that block addition gives same result as full matrix addition."""
        blocks1 = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        blocks2 = [np.array([[5, 6], [7, 8]]), np.array([1, 1])]
        sizes = (2, 2)

        bdm1 = BlockDiagonalMatrix(blocks1, sizes)
        bdm2 = BlockDiagonalMatrix(blocks2, sizes)

        result_block = (bdm1 + bdm2).to_full_matrix()
        result_full = bdm1.to_full_matrix() + bdm2.to_full_matrix()

        np.testing.assert_array_almost_equal(result_block, result_full)

    def test_scalar_multiplication_consistency(self):
        """Test that scalar multiplication is consistent."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        scalar = 5
        result_block = (bdm * scalar).to_full_matrix()
        result_full = bdm.to_full_matrix() * scalar

        np.testing.assert_array_almost_equal(result_block, result_full)

    def test_transpose_consistency(self):
        """Test that transpose is consistent with full matrix transpose."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        result_block = bdm.T.to_full_matrix()
        result_full = bdm.to_full_matrix().T

        np.testing.assert_array_almost_equal(result_block, result_full)

    def test_vector_multiplication_consistency(self):
        """Test that vector multiplication is consistent."""
        blocks = [np.array([[1, 2], [3, 4]]), np.array([2, 3])]
        sizes = (2, 2)
        bdm = BlockDiagonalMatrix(blocks, sizes)

        vec = np.array([1, 2, 3, 4])
        result_block = bdm @ vec
        result_full = bdm.to_full_matrix() @ vec

        np.testing.assert_array_almost_equal(result_block, result_full)


if __name__ == '__main__':
    unittest.main()

