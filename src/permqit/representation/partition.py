"""
Partition class for representing Young diagrams.

Classes:
- Partition: Represents a partition with memoized Young diagram properties
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Iterator, Iterable, Collection, Sequence
from functools import cached_property
import numpy as np

class Partition:
    """Represents a partition λ with memoized Young diagram properties.

    A partition λ = (λ1, λ2, ..., λ_d) in nonincreasing order.
    Properties are computed lazily and cached.

    Use class methods `from_numbers` or `from_iterable` to create instances, which check
    a global cache to avoid recomputing properties for the same partition.
    """

    _cache: Dict[Tuple[int, ...], Partition] = {}

    def __init__(self, lam: Sequence[int]):
        """Private constructor. Use from_numbers() or from_iterable() instead."""
        lam = tuple(lam)
        if any(L <= 0 for L in lam):
            raise ValueError(f"λ must be a positive partition, got {lam}")
        if any(lam[k] < lam[k+1] for k in range(len(lam)-1)):
            raise ValueError(f"λ must be nonincreasing, got {lam}")
        self._lam = lam

    def print_young_diagram(self):
        """Prints the Young diagram (in english notation) corresponding to this partition."""
        for row_length in self._lam:
            print("☐" * row_length)

    def count_standard_tableaux(self):
        """Calculates the number of standard Young tableaux corresponding to this partition via the hook length formula.
        This corresponds to the dimension of the irreducible representation indexed by this partition.
        """
        n = self.n
        hook_product = 1
        for j in range(self.height):
            for i in range(self._lam[j]):
                # Calculate hook length for cell (i,j)
                right = self._lam[j] - i - 1
                below = sum(1 for jj in range(j + 1, self.height) if self._lam[jj] > i)
                hook_product *= 1 + right + below
        return math.factorial(n) // hook_product

    @cached_property
    def canonical_tableau(self) -> SYT:
        """
        Returns the canonical standard Young Tableau corresponding to this partition.
        The canonical tableau has contents 0, ..., d-1, by filling each rows left to right, i.e.
        for example
        1 2 3
        4 5
        6
        :return:
        """
        rows = []
        label = 0
        for row_length in self._lam:
            rows.append(label + np.arange(row_length))
            label += row_length
        return SYT(rows, partition=self)


    @cached_property
    def constant_tableau(self):
        """Returns a tableau where the ith row is filled just with the number i. This is used extensively in
        https://arxiv.org/abs/0910.4515
        """
        rows = [(i,)*l for i, l in enumerate(self._lam)]
        return SSYT(rows, partition=self)


    @classmethod
    def from_numbers(cls, *numbers: int):
        """Create a partition from a list of numbers."""
        return cls.from_iterable(tuple(numbers))

    @classmethod
    def normalize(cls, lam: Partition|Iterable[int]) -> Partition:
        """Normalize a list or tuple to a partition."""
        if isinstance(lam, Partition):
            return lam
        return cls.from_iterable(tuple(lam))

    @classmethod
    def from_iterable(cls, lam: Iterable[int]) -> Partition:
        """Create or retrieve a Partition from a tuple."""
        lam = tuple(lam)
        if lam not in cls._cache:
            cls._cache[lam] = cls(lam)
        return cls._cache[lam]

    def as_tuple(self) -> Tuple[int, ...]:
        """Return the partition as a tuple."""
        return self._lam

    @property
    def height(self) -> int:
        """Return the height (number of rows) of the partition."""
        return len(self._lam)

    @property
    def width(self) -> int:
        """Return the width (maximum row length) of the partition."""
        try:
            return self._lam[0]
        except IndexError:
            return 0

    @property
    def n(self) -> int:
        """Return the total of the partition, i.e. the number that is partitioned."""
        return sum(self._lam)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the parts of the partition."""
        return iter(self._lam)

    def __getitem__(self, item):
        return self._lam[item]

    def __len__(self):
        return self.height

    @cached_property
    def positions(self) -> np.ndarray:
        """Return positions (i,j) in lexicographic order by (j, then i).

        Shape: (n, 2) where n is the size of the partition.
        French notation: j = 0..d-1 (row index), i = 0..λ_j-1 (column index).
        """
        positions = []
        for j in range(self.height):
            for i in range(self._lam[j]):
                positions.append((i, j))
        return np.array(positions, dtype=np.int_)

    @cached_property
    def pos_to_label(self) -> Dict[Tuple[int, int], int]:
        """Map position (i,j) to label in {1..n}."""
        return {tuple(pos): k for k, pos in enumerate(self.positions)}

    @cached_property
    def label_to_pos(self) -> Dict[int, Tuple[int, int]]:
        """Map label in {1..n} to position (i,j)."""
        return {k: tuple(pos) for k, pos in enumerate(self.positions)}

    @cached_property
    def rows_positions(self) -> List[np.ndarray]:
        """List of row positions as numpy arrays."""
        rows = []
        for j0 in range(self.height):
            row = np.array([(i, j0) for i in range(self._lam[j0])], dtype=np.int32)
            rows.append(row)
        return rows

    @property
    def rows_labels(self) -> Collection[np.ndarray]:
        """List of row labels as numpy arrays."""
        return self.canonical_tableau.rows

    @cached_property
    def cols_positions(self) -> List[np.ndarray]:
        """List of column positions as numpy arrays."""
        cols = []
        for i0 in range(self.width):
            col = np.array([(i0, j) for j in range(self.height) if self._lam[j] > i0], dtype=np.int32)
            cols.append(col)
        return cols

    @cached_property
    def cols_labels(self) -> List[np.ndarray]:
        """List of column labels as numpy arrays."""
        cols = []
        for i0 in range(self.width):
            col_pos = self.cols_positions[i0]
            labels = np.array([self.pos_to_label[(int(p[0]), int(p[1]))] for p in col_pos], dtype=np.int32)
            cols.append(labels)
        return cols

    @cached_property
    def label_to_row_index(self) -> Dict[int, int]:
        """Map each label to its row index (0-based)."""
        return {lab: j for lab, (i, j) in self.label_to_pos.items()}

    def __repr__(self) -> str:
        return f"Partition{self._lam}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Partition):
            return False
        return self._lam == other._lam

    def __hash__(self) -> int:
        return hash(self._lam)

    @classmethod
    def generate_all(cls, n: int, max_length=None, *, _max_value=None) -> Iterable[Partition]:
        """Return all partitions of n."""
        if n == 0:
            yield cls(())
            return
        if max_length is None:
            max_length = n
        if max_length < 1:
            return
        if _max_value is None:
            _max_value = n
        for k in range(min(n - 1, _max_value), 0, -1):
            for p in cls.generate_all(n-k, max_length=max_length-1, _max_value=k):
                yield cls((k,) + p.as_tuple())
        if n <= _max_value:
            yield cls((n,))


from .young_tableau import YoungTableau, SSYT, SYT
