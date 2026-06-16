import unittest
import numpy as np

from permqit.representation.partition import Partition


class TestPartition(unittest.TestCase):
    """Test suite for the Partition class."""

    def setUp(self):
        """Clear the Partition cache before each test to ensure independence."""
        Partition._cache.clear()

    def test_from_tuple_basic(self):
        """Test creating a partition from a tuple."""
        p = Partition.from_iterable((3, 2, 1))
        self.assertEqual(p.as_tuple(), (3, 2, 1))
        self.assertEqual(p.height, 3)
        self.assertEqual(p.width, 3)
        self.assertEqual(p.n, 6)

    def test_single_row_partition(self):
        """Test a partition with a single row."""
        p = Partition.from_iterable((5,))
        self.assertEqual(p.as_tuple(), (5,))
        self.assertEqual(p.height, 1)
        self.assertEqual(p.width, 5)
        self.assertEqual(p.n, 5)

    def test_single_column_partition(self):
        """Test a partition that forms a single column."""
        p = Partition.from_iterable((1, 1, 1, 1))
        self.assertEqual(p.as_tuple(), (1, 1, 1, 1))
        self.assertEqual(p.height, 4)
        self.assertEqual(p.width, 1)
        self.assertEqual(p.n, 4)

    def test_square_partition(self):
        """Test a square partition."""
        p = Partition.from_iterable((3, 3, 3))
        self.assertEqual(p.as_tuple(), (3, 3, 3))
        self.assertEqual(p.height, 3)
        self.assertEqual(p.width, 3)
        self.assertEqual(p.n, 9)

    def test_valid_empty_partition(self):
        """Test that an empty partition is valid."""
        p = Partition.from_iterable(())
        self.assertEqual(p.height, 0)
        self.assertEqual(p.width, 0)
        self.assertEqual(p.n, 0)

    def test_invalid_zero_in_partition(self):
        """Test that a partition with zero raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Partition.from_iterable((3, 0, 1))
        self.assertIn("positive", str(context.exception))

    def test_invalid_negative_in_partition(self):
        """Test that a partition with negative number raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Partition.from_iterable((3, 2, -1))
        self.assertIn("positive", str(context.exception))

    def test_invalid_not_nonincreasing(self):
        """Test that a non-nonincreasing sequence raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Partition.from_iterable((2, 3, 1))
        self.assertIn("nonincreasing", str(context.exception))

    def test_positions_ordering(self):
        """Test that positions are in lexicographic order by (j, then i)."""
        p = Partition.from_iterable((3, 2))
        # Expected positions (0-based): row 0 (j=0): (0,0), (1,0), (2,0)
        #                              row 1 (j=1): (0,1), (1,1)
        expected = np.array([
            (0, 0), (1, 0), (2, 0),  # row 0
            (0, 1), (1, 1)            # row 1
        ], dtype=np.int32)
        np.testing.assert_array_equal(p.positions, expected)

    def test_pos_to_label_mapping(self):
        """Test position to label mapping."""
        p = Partition.from_iterable((2, 1))
        # Positions (0-based): (0,0), (1,0), (0,1)
        # Labels:               0,     1,     2
        self.assertEqual(p.pos_to_label[(0, 0)], 0)
        self.assertEqual(p.pos_to_label[(1, 0)], 1)
        self.assertEqual(p.pos_to_label[(0, 1)], 2)

    def test_label_to_pos_mapping(self):
        """Test label to position mapping."""
        p = Partition.from_iterable((2, 1))
        self.assertEqual(p.label_to_pos[0], (0, 0))
        self.assertEqual(p.label_to_pos[1], (1, 0))
        self.assertEqual(p.label_to_pos[2], (0, 1))

    def test_pos_label_bidirectional(self):
        """Test that pos_to_label and label_to_pos are inverses."""
        p = Partition.from_iterable((3, 2, 1))
        for label in range(p.n):
            pos = p.label_to_pos[label]
            recovered_label = p.pos_to_label[pos]
            self.assertEqual(recovered_label, label)

    def test_rows_positions(self):
        """Test that rows_positions correctly identifies row positions."""
        p = Partition.from_iterable((3, 2))
        self.assertEqual(len(p.rows_positions), 2)

        # Row 0 (j=0): (0,0), (1,0), (2,0)
        expected_row1 = np.array([(0, 0), (1, 0), (2, 0)], dtype=np.int32)
        np.testing.assert_array_equal(p.rows_positions[0], expected_row1)

        # Row 1 (j=1): (0,1), (1,1)
        expected_row2 = np.array([(0, 1), (1, 1)], dtype=np.int32)
        np.testing.assert_array_equal(p.rows_positions[1], expected_row2)

    def test_rows_labels(self):
        """Test that rows_labels correctly identifies row labels."""
        p = Partition.from_iterable((3, 2))
        self.assertEqual(len(p.rows_labels), 2)

        # Row 0: labels 0, 1, 2
        expected_row1 = np.array([0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(p.rows_labels[0], expected_row1)

        # Row 1: labels 3, 4
        expected_row2 = np.array([3, 4], dtype=np.int32)
        np.testing.assert_array_equal(p.rows_labels[1], expected_row2)

    def test_cols_positions(self):
        """Test that cols_positions correctly identifies column positions."""
        p = Partition.from_iterable((3, 2))
        self.assertEqual(len(p.cols_positions), 3)

        # Col 0 (i=0): (0,0), (0,1)
        expected_col1 = np.array([(0, 0), (0, 1)], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_positions[0], expected_col1)

        # Col 1 (i=1): (1,0), (1,1)
        expected_col2 = np.array([(1, 0), (1, 1)], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_positions[1], expected_col2)

        # Col 2 (i=2): (2,0)
        expected_col3 = np.array([(2, 0)], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_positions[2], expected_col3)

    def test_cols_labels(self):
        """Test that cols_labels correctly identifies column labels."""
        p = Partition.from_iterable((3, 2))
        self.assertEqual(len(p.cols_labels), 3)

        # Col 0: labels 0, 3
        expected_col1 = np.array([0, 3], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_labels[0], expected_col1)

        # Col 1: labels 1, 4
        expected_col2 = np.array([1, 4], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_labels[1], expected_col2)

        # Col 2: label 2
        expected_col3 = np.array([2], dtype=np.int32)
        np.testing.assert_array_equal(p.cols_labels[2], expected_col3)

    def test_label_to_row_index(self):
        """Test that label_to_row_index correctly maps labels to row indices."""
        p = Partition.from_iterable((3, 2))
        # Row 0 (index 0): labels 0, 1, 2
        self.assertEqual(p.label_to_row_index[0], 0)
        self.assertEqual(p.label_to_row_index[1], 0)
        self.assertEqual(p.label_to_row_index[2], 0)
        # Row 1 (index 1): labels 3, 4
        self.assertEqual(p.label_to_row_index[3], 1)
        self.assertEqual(p.label_to_row_index[4], 1)

    def test_caching_same_tuple(self):
        """Test that creating the same partition returns the cached instance."""
        p1 = Partition.from_iterable((3, 2, 1))
        p2 = Partition.from_iterable((3, 2, 1))
        self.assertIs(p1, p2)  # Same object reference

    def test_equality(self):
        """Test equality comparison between partitions."""
        Partition._cache.clear()  # Clear to create different instances
        p1 = Partition.from_iterable((3, 2, 1))
        Partition._cache.clear()
        p2 = Partition.from_iterable((3, 2, 1))
        p3 = Partition.from_iterable((3, 2, 2))

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)

    def test_equality_with_non_partition(self):
        """Test that equality with non-Partition returns False."""
        p = Partition.from_iterable((3, 2, 1))
        self.assertNotEqual(p, (3, 2, 1))
        self.assertNotEqual(p, [3, 2, 1])
        self.assertNotEqual(p, "Partition")

    def test_hash(self):
        """Test that partitions can be hashed and used in sets/dicts."""
        p1 = Partition.from_iterable((3, 2, 1))
        p2 = Partition.from_iterable((3, 2, 1))
        p3 = Partition.from_iterable((2, 2))

        # Same partition should have same hash
        self.assertEqual(hash(p1), hash(p2))

        # Can use in set
        partition_set = {p1, p2, p3}
        self.assertEqual(len(partition_set), 2)  # p1 and p2 are the same

        # Can use as dict keys
        partition_dict = {p1: "first", p3: "second"}
        self.assertEqual(partition_dict[p2], "first")

    def test_repr(self):
        """Test string representation of partition."""
        p = Partition.from_iterable((3, 2, 1))
        self.assertEqual(repr(p), "Partition(3, 2, 1)")

    def test_large_partition(self):
        """Test with a larger partition to ensure scalability."""
        p = Partition.from_iterable((5, 4, 3, 2, 1))
        self.assertEqual(p.n, 15)
        self.assertEqual(p.height, 5)
        self.assertEqual(p.width, 5)
        self.assertEqual(len(p.positions), 15)
        self.assertEqual(len(p.rows_positions), 5)
        self.assertEqual(len(p.cols_positions), 5)

    def test_cached_property_computed_once(self):
        """Test that cached properties are computed only once."""
        p = Partition.from_iterable((3, 2))

        # Access positions multiple times
        pos1 = p.positions
        pos2 = p.positions

        # Should be the same object (cached)
        self.assertIs(pos1, pos2)


if __name__ == '__main__':
    unittest.main()
