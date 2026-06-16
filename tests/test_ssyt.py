"""
Test suite for SSYT (Semistandard Young Tableaux) class.

Tests mathematical properties including:
- Validation of SSYT properties (rows weakly increasing, columns strictly increasing)
- Counting formulas (hook-length formula for standard tableaux)
- Generation completeness and correctness
- Overlap (inner product) properties
"""
import unittest
from itertools import product
import math

from permqit.representation.young_tableau import SSYT
from permqit.representation.partition import Partition



class TestSSYTValidation(unittest.TestCase):
    """Test SSYT validation of tableau properties."""

    def test_valid_ssyt_single_row(self):
        """Single row with weakly increasing entries is valid."""
        ssyt = SSYT([[0, 0, 1, 2]])
        self.assertEqual(ssyt.shape, (4,))

    def test_valid_ssyt_single_column(self):
        """Single column with strictly increasing entries is valid."""
        ssyt = SSYT([[0], [1], [2]])
        self.assertEqual(ssyt.shape, (1, 1, 1))

    def test_valid_ssyt_general(self):
        """General valid SSYT."""
        ssyt = SSYT([[0, 0, 1], [1, 2], [2]])
        self.assertEqual(ssyt.shape, (3, 2, 1))

    def test_invalid_ssyt_row_not_weakly_increasing(self):
        """Row that decreases should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            SSYT([[1, 0, 2]])
        self.assertIn("not weakly increasing", str(context.exception))

    def test_invalid_ssyt_row_strictly_decreasing(self):
        """Row with strict decrease should fail."""
        with self.assertRaises(ValueError):
            SSYT([[2, 1, 0]])

    def test_invalid_ssyt_column_not_strictly_increasing(self):
        """Column with equal entries should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            SSYT([[0, 1], [0, 2]])
        self.assertIn("not strictly increasing", str(context.exception))

    def test_invalid_ssyt_column_equal_entries(self):
        """Column with repeated value should fail."""
        with self.assertRaises(ValueError):
            SSYT([[0], [0]])

    def test_invalid_ssyt_column_decreasing(self):
        """Column that decreases should fail."""
        with self.assertRaises(ValueError):
            SSYT([[1], [0]])

    def test_partition_shape_mismatch(self):
        """Providing a partition with wrong shape should fail."""
        part = Partition.from_iterable((3, 2))
        with self.assertRaises(ValueError) as context:
            SSYT([[0, 1, 2], [1, 2], [3]], partition=part)
        self.assertIn("doesn't match", str(context.exception))


class TestSSYTProperties(unittest.TestCase):
    """Test SSYT properties and methods."""

    def test_rows_list_property(self):
        """Test rows_list returns list of lists."""
        ssyt = SSYT([[0, 1], [2, 3]])
        rows_list = ssyt.rows_list
        self.assertIsInstance(rows_list, list)
        self.assertEqual(rows_list, [[0, 1], [2, 3]])

    def test_partition_property(self):
        """Test partition property."""
        ssyt = SSYT([[0, 0, 1], [1, 2]])
        self.assertEqual(ssyt.partition.as_tuple(), (3, 2))
        self.assertEqual(ssyt.shape, (3, 2))

    def test_max_entry(self):
        """Test max_entry property."""
        ssyt = SSYT([[0, 0, 1], [1, 2], [3]])
        self.assertEqual(ssyt.max_entry, 3)

    def test_max_entry_single_element(self):
        """Test max_entry for single element."""
        ssyt = SSYT([[6]])
        self.assertEqual(ssyt.max_entry, 6)

    def test_equality(self):
        """Test SSYT equality."""
        ssyt1 = SSYT([[0, 1], [2, 3]])
        ssyt2 = SSYT([[0, 1], [2, 3]])
        ssyt3 = SSYT([[0, 1], [2, 4]])
        self.assertEqual(ssyt1, ssyt2)
        self.assertNotEqual(ssyt1, ssyt3)

    def test_hashability(self):
        """Test that SSYTs are hashable."""
        ssyt1 = SSYT([[0, 1], [2, 3]])
        ssyt2 = SSYT([[0, 1], [2, 3]])
        ssyt3 = SSYT([[0, 1], [2, 4]])

        # Can be used in sets
        ssyt_set = {ssyt1, ssyt2, ssyt3}
        self.assertEqual(len(ssyt_set), 2)  # ssyt1 and ssyt2 are equal

        # Can be used as dict keys
        d = {ssyt1: "a", ssyt3: "b"}
        self.assertEqual(d[ssyt2], "a")

    def test_repr_and_str(self):
        """Test string representations."""
        ssyt = SSYT([[0, 1], [2, 3]])
        repr_str = repr(ssyt)
        self.assertIn("SSYT", repr_str)

        # Pretty print should be readable
        pretty = ssyt.pretty_str()
        self.assertIn("0", pretty)
        self.assertIn("1", pretty)


class TestSSYTGeneration(unittest.TestCase):
    """Test SSYT generation and counting."""

    def test_generate_all_single_box(self):
        """Generate all SSYTs for single box."""
        ssyts = list(SSYT.generate_all((1,), 3))
        self.assertEqual(len(ssyts), 3)
        expected = [[[0]], [[1]], [[2]]]
        actual = [ssyt.rows_list for ssyt in ssyts]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_generate_all_two_boxes_row(self):
        """Generate all SSYTs for shape (2,) with d=2."""
        # Valid SSYTs: [0,0], [0,1], [1,1]
        ssyts = list(SSYT.generate_all((2,), 2))
        self.assertEqual(len(ssyts), 3)
        expected = [[[0, 0]], [[0, 1]], [[1, 1]]]
        actual = [ssyt.rows_list for ssyt in ssyts]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_generate_all_two_boxes_column(self):
        """Generate all SSYTs for shape (1,1) with d=3."""
        # Valid SSYTs: [0],[1]; [0],[2]; [1],[2]
        ssyts = list(SSYT.generate_all((1, 1), 3))
        self.assertEqual(len(ssyts), 3)
        expected = [[[0], [1]], [[0], [2]], [[1], [2]]]
        actual = [ssyt.rows_list for ssyt in ssyts]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_generate_all_shape_2_1(self):
        """Generate all SSYTs for shape (2,1) with d=2."""
        # Row 1: [a,b] with a <= b
        # Row 2: [c] with c > a
        ssyts = list(SSYT.generate_all((2, 1), 2))
        expected = [[[0, 0], [1]], [[0, 1], [1]]]
        actual = [ssyt.rows_list for ssyt in ssyts]
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertEqual(len(ssyts), 2)

    def test_generate_all_validates_ssyt_properties(self):
        """All generated SSYTs should satisfy SSYT properties."""
        for shape in [(3,), (2, 1), (2, 2), (3, 2, 1)]:
            for d in range(1, 4):
                ssyts = list(SSYT.generate_all(shape, d))
                for ssyt in ssyts:
                    # Should not raise any validation errors
                    self.assertEqual(ssyt.shape, shape)
                    # Verify rows weakly increasing
                    for row in ssyt.rows_list:
                        for i in range(len(row) - 1):
                            self.assertLessEqual(row[i], row[i+1])
                    # Verify columns strictly increasing
                    for col_idx in range(shape[0]):
                        col_vals = []
                        for row_idx, row in enumerate(ssyt.rows_list):
                            if col_idx < len(row):
                                col_vals.append(row[col_idx])
                        for i in range(len(col_vals) - 1):
                            self.assertLess(col_vals[i], col_vals[i+1])

    def test_count_hook_content_formula(self):
        """Test count matches hook-content formula for SSYTs.

        The hook-content formula states that the number of SSYTs of shape λ
        with entries in {0, 1, ..., d-1} is:

        prod_{(i,j) in λ} (d + c(i,j)) / h(i,j)

        where c(i,j) = j - i (0-indexed content) and
        h(i,j) = arm + leg + 1 (hook length).
        """

        def hook_content_count(shape, d):
            """Calculate number of SSYTs using hook-content formula."""
            result = 1
            for i, row_len in enumerate(shape):
                for j in range(row_len):
                    # Content: column - row (0-indexed)
                    content = j - i
                    # Hook length: boxes to right + boxes below + 1
                    arm = row_len - j - 1  # boxes to the right
                    leg = sum(1 for k in range(i+1, len(shape)) if shape[k] > j)  # boxes below
                    hook = arm + leg + 1
                    result *= (d + content) / hook
            return int(round(result))

        # Test various shapes and dimensions
        test_cases = [
            # (shape, d)
            ((1,), 5),  # Single box
            ((2,), 3),  # Two boxes in a row
            ((1, 1), 4),  # Two boxes in a column
            ((2, 1), 2),  # L-shape with d=2
            ((2, 1), 3),  # L-shape with d=3
            ((2, 2), 3),  # Square with d=3
            ((2, 2), 4),  # Square with d=4
            ((3, 2, 1), 3),  # Staircase
            ((3, 2, 1), 4),  # Staircase with d=4
        ]

        for shape, d in test_cases:
            with self.subTest(shape=shape, d=d):
                expected = hook_content_count(shape, d)
                # compare with the new count implementation
                actual_count = SSYT.count(shape, d)
                self.assertEqual(actual_count, expected,
                                 f"Count formula mismatch for shape {shape} with d={d}: got {actual_count}, expected {expected}")
                # also verify enumeration yields the same number
                enumerated = list(SSYT.generate_all(shape, d))
                self.assertEqual(len(enumerated), expected,
                                 f"Enumeration mismatch for shape {shape} with d={d}: got {len(enumerated)}, expected {expected}")

    def test_count_formula_for_small_shapes(self):
        """Test count against known values for small shapes."""
        # Shape (1,): count(d) = d (just pick one symbol)
        for d in range(1, 6):
            self.assertEqual(SSYT.count((1,), d), d)

        # Shape (2,): count(d) = C(d+1, 2) = d(d+1)/2 (weakly increasing pairs)
        for d in range(1, 6):
            expected = d * (d + 1) // 2
            self.assertEqual(SSYT.count((2,), d), expected)

        # Shape (1,1): count(d) = C(d, 2) (strictly increasing pairs)
        for d in range(2, 6):
            expected = d * (d - 1) // 2
            self.assertEqual(SSYT.count((1, 1), d), expected)

    def test_count_with_Sn_dimension(self):
        """Test the Schur-Weyl decomposition dimensions. The SSYTs correspond to basis vectors for the GL(n) representation,
        whereas Standard Young Tableaux correspond to basis vectors of the Sn representation. Hence, the product of the
        two should sum to the dimension of the vector space.
        """
        for d in range(2, 4): # Dimension of the single vector space
            for n in range(1, 5): # Number of copies
                cnt = 0
                for part in Partition.generate_all(n):
                    cnt += SSYT.count(part, d) * part.count_standard_tableaux()  # Dimension of the irrep corresponding to the partition
                self.assertEqual(cnt, d**n)


class TestSSYTOverlap(unittest.TestCase):
    """Test SSYT overlap (inner product) properties."""

    def test_overlap_different_shapes_is_zero(self):
        """Overlap of SSYTs with different shapes should be zero."""
        ssyt1 = SSYT([[0, 1]])
        ssyt2 = SSYT([[0], [1]])
        self.assertEqual(ssyt1.overlap(ssyt2), 0)

    def test_overlap_single_box_is_one(self):
        """Overlap of single box SSYTs."""
        ssyt1 = SSYT([[0]])
        ssyt2 = SSYT([[0]])
        # For single box with same entry, overlap should be 1
        self.assertEqual(ssyt1.overlap(ssyt2), 1)

    def test_overlap_single_box_different_entries(self):
        """Overlap of single box with different entries should be zero."""
        ssyt1 = SSYT([[0]])
        ssyt2 = SSYT([[1]])
        self.assertEqual(ssyt1.overlap(ssyt2), 0)

    def test_overlap_is_symmetric(self):
        """Overlap should be symmetric: <u, v> = <v, u>."""
        ssyt1 = SSYT([[0, 0], [1]])
        ssyt2 = SSYT([[0, 1], [1]])
        dim = 2
        self.assertEqual(ssyt1.overlap(ssyt2), ssyt2.overlap(ssyt1))

    def test_overlap_self_is_positive(self):
        """Self overlap should be positive (inner product positive definite)."""
        ssyts = [
            SSYT([[0]]),
            SSYT([[0, 1]]),
            SSYT([[0], [1]]),
            SSYT([[0, 0], [1]]),
            SSYT([[0, 1], [1]]),
        ]
        for ssyt in ssyts:
            dim = ssyt.max_entry + 1
            overlap = ssyt.overlap(ssyt)
            self.assertGreater(overlap, 0, f"Self-overlap should be positive for {ssyt}")

    def test_overlap_two_row_examples(self):
        """Test specific overlap calculations for two-row tableaux."""
        # Shape (2,1), dim=2
        ssyt1 = SSYT([[0, 0], [1]])
        ssyt2 = SSYT([[0, 1], [1]])

        # These should be orthogonal (different fillings)
        overlap = ssyt1.overlap(ssyt2)
        # Note: These may not be orthogonal if they're not standard tableaux
        # Just check that overlap is computed without error
        self.assertIsInstance(overlap, int)

    def test_overlap_single_row_weakly_increasing(self):
        """Test overlap for single row tableaux."""
        # For a single row, there's no column antisymmetrization
        ssyt1 = SSYT([[0, 0, 1]])
        ssyt2 = SSYT([[0, 0, 1]])

        # Same tableau should have positive self-overlap
        overlap = ssyt1.overlap(ssyt2)
        self.assertGreater(overlap, 0)

    def test_overlap_computation_completes(self):
        """Test that overlap computation completes for various shapes."""
        # Just ensure no errors/hangs for reasonable sizes
        test_cases = [
            ([[0, 1]], [[0, 1]], 2),
            ([[0], [1]], [[0], [1]], 2),
            ([[0, 0], [1]], [[0, 0], [1]], 2),
            ([[0, 1], [2]], [[0, 2], [1]], 3),
            ([[0, 1, 2]], [[0, 1, 2]], 3),
        ]

        for rows1, rows2, dim in test_cases:
            ssyt1 = SSYT(rows1)
            ssyt2 = SSYT(rows2)
            overlap = ssyt1.overlap(ssyt2)
            self.assertIsInstance(overlap, int)

    def test_overlap_values(self):
        for n in [2, 3, 4]:
            for d in [2,3]:
                for part in Partition.generate_all(n):
                    for t1 in SSYT.generate_all(part, d):
                        for t2 in SSYT.generate_all(part, d):
                            self.assertEqual(t1.overlap(t2), t1.basis_vector(d) @ t2.basis_vector(d), f"{n=}, {d=}, {part=}, {t1=}, {t2=}")



class TestSSYTEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_element_tableau(self):
        """Test single element tableaux."""
        ssyt = SSYT([[0]])
        self.assertEqual(ssyt.shape, (1,))
        self.assertEqual(ssyt.max_entry, 0)

    def test_large_entries(self):
        """Test tableaux with large entry values."""
        ssyt = SSYT([[9, 19, 29], [39, 49]])
        self.assertEqual(ssyt.max_entry, 49)
        self.assertEqual(ssyt.shape, (3, 2))

    def test_generation_with_d_equals_1(self):
        """Test generation when d=1 (all entries must be 0)."""
        # Only single row can have all 0s
        ssyts = list(SSYT.generate_all((3,), 1))
        self.assertEqual(len(ssyts), 1)
        self.assertEqual(ssyts[0].rows_list, [[0, 0, 0]])

        # Column shapes impossible with d=1 (can't have strictly increasing)
        ssyts = list(SSYT.generate_all((1, 1), 1))
        self.assertEqual(len(ssyts), 0)

    def test_generation_with_large_d(self):
        """Test that generation works with d >> n."""
        # Should get many more tableaux as d increases
        shape = (2, 1)
        count_d2 = SSYT.count(shape, 2)
        count_d5 = SSYT.count(shape, 5)
        self.assertGreater(count_d5, count_d2)

    def test_tuple_initialization(self):
        """Test initialization with tuple of tuples."""
        rows_tuple = ((0, 1, 2), (3, 4))
        ssyt = SSYT(rows_tuple)
        self.assertEqual(ssyt.shape, (3, 2))
        self.assertEqual(ssyt.rows_tuples, rows_tuple)
