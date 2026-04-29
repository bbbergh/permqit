"""
Young tableau classes and combinatorial utilities.

Classes:
- YoungTableau: Base class for Young tableaux with row/column access and
  combinatorial operations.
- SYT: Standard Young tableau (strictly increasing rows and columns,
  entries 0..n-1 each appearing once).
- SSYT: Semistandard Young tableau (weakly increasing rows, strictly
  increasing columns, entries in {0..d-1}).
- SSYTCache: Lazy cache of all SSYTs for a given shape and alphabet size.
- Gram_matrix: Function computing the combinatorial Gram matrix for the SSYT
  basis of a given shape.
"""
from __future__ import annotations

from collections import Counter
from functools import cached_property, lru_cache

from typing import List, Tuple, Dict, Iterator, Optional, Union, Iterable, Collection, Sequence, cast
import math
import numpy as np
from fractions import Fraction

from .combinatorics import multinomial_coeff
from .partition import Partition

import sympy.combinatorics as comb

from ..utilities.timing import MaybeExpensiveComputation

class YoungTableau:
    """
    Represents a Young tableau, i.e. a Young diagram with boxes filled in by integers.
    Tableaux are stored in english notation, i.e. with lenghts of rows decreasing.
    """
    def __init__(self, rows: Iterable[Iterable[int]], partition: Optional[Partition] = None):
        """Initialize a YT from rows.

        Args:
            rows: Nested iterable of integers.
            partition: Optional Partition object. If not provided, inferred from shape.
        """
        # Store rows internally as a tuple of numpy arrays (dtype=int)
        self._rows = tuple(np.asarray(row, dtype=int) for row in rows)

        # shape derived from internal rows
        shape = tuple(int(len(row)) for row in self._rows)
        if partition is None:
            self._partition = Partition.from_iterable(shape)
        else:
            if partition.as_tuple() != shape:
                raise ValueError(f"Partition shape {partition.as_tuple()} doesn't match tableau shape {shape}")
            self._partition = partition

        # Validate YT properties
        self._validate()

    @classmethod
    def from_flat(cls, flattened_rows: Sequence[int], partition: Partition):
        """Construct a YoungTableau from a flat row-major sequence.

        Args:
            flattened_rows: Row-major sequence of tableau entries.
            partition: Partition giving the row lengths.

        Returns:
            A new YoungTableau instance.
        """
        rows = []
        index = 0
        for length in partition:
            row = flattened_rows[index:index+length]
            rows.append(row)
            index += length
        return cls(rows, partition)

    def _validate(self):
        """Validate that the tableau satisfies basic properties."""
        pass

    def pretty_str(self) -> str:
        """Return a human-readable string representation."""
        if not self._rows:
            return ""
        w = max(len(str(int(x))) for row in self._rows for x in row)
        return "\n".join(" ".join(f"{int(x):>{w}}" for x in row) for row in self._rows)

    def pretty_print(self):
        """Print a human-readable representation of the tableau."""
        print(self.pretty_str())

    @property
    def rows(self) -> tuple[np.ndarray, ...]:
        """Return the tableau rows as a tuple of numpy integer arrays."""
        return self._rows

    @property
    def english(self):
        """
        Access tableau entries or rows in English notation.
        Use via SSYT.english[i] (ith row), or SSYT.english[i][j] (ith row, jth column).
        """
        return self._rows

    def from_canonical_index(self, k):
        """Return the tableau entry at row-major position k.

        Positions are 0-based and follow the canonical filling order of the
        partition (left-to-right, top-to-bottom in English notation).

        Args:
            k: Zero-based position index.

        Returns:
            The integer entry at position k.
        """
        row_index = 0
        while k > 0:
            k -= self._partition[row_index]
            row_index += 1
        return self._rows[row_index-1][k + self._partition[row_index-1] - 1]

    def symbol_count(self):
        """Return a Counter mapping each entry value to its total multiplicity."""
        return Counter(self.flat_iter())

    def symbol_count_per_row(self):
        """Return a list of Counters, one per row, with per-row entry multiplicities."""
        return [Counter(r) for r in self._rows]

    def symbol_count_per_column(self):
        """Return a list of Counters, one per column, with per-column entry multiplicities."""
        return [Counter(c) for c in self.columns]

    class _FrenchAccessor:
        def __init__(self, yt):
            self.yt = yt

        def __getitem__(self, i):
            return self.yt._rows[self.yt.height - i - 1]

    @cached_property
    def french(self):
        """
        Access tableau entries in French notation (row (reversed) first, then column).
        Use via YT.french[i] (self.height - i - 1 th row), or YT.french[i][j] (self.height - i - 1th row, jth column).
        """
        return self._FrenchAccessor(self)

    class _ColumnAccessor:
        class _Column:
            def __init__(self, yt: YoungTableau, column: int):
                self.yt, self.column = yt, column

            def __getitem__(self, i):
                return self.yt._rows[i][self.column]

            def __iter__(self):
                for row in self.yt._rows:
                    if self.column >= len(row):
                        break
                    yield row[self.column]

            def __str__(self):
                return str(list(self))
            def __repr__(self):
                return f"Column({str(self)}"

            def __len__(self):
                return sum(1 for l in self.yt.partition if l > self.column)

        def __init__(self, yt: YoungTableau):
            self.yt = yt

        def __getitem__(self, i):
            return self._Column(self.yt, i)

        def __iter__(self):
            for i in range(self.yt.partition.width):
                yield self[i]

        def __len__(self):
            return self.yt.partition.width


    @cached_property
    def columns(self):
        """Column-major accessor for tableau entries.

        ``YT.columns[i]`` returns the i-th column; ``YT.columns[i][j]`` returns
        the j-th entry (j-th row) of the i-th column.
        """
        return self._ColumnAccessor(self)

    @cached_property
    def rows_tuples(self) -> Tuple[Tuple[int, ...], ...]:
        """Return rows as hashable tuple of tuples (keeps public API stable)."""
        return tuple(tuple(int(x) for x in row) for row in self._rows)

    @property
    def rows_list(self) -> List[List[int]]:
        """Return rows as a list of lists of Python ints. Prefer rows_tuples for new code."""
        return [ [int(x) for x in row.tolist()] for row in self._rows ]

    @property
    def partition(self) -> Partition:
        """Return the underlying partition."""
        return self._partition

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tableau."""
        return self._partition.as_tuple()

    def __iter__(self):
        return iter(self._rows)

    def flat_iter(self):
        """Yield all entries in row-major order."""
        return (it for row in self._rows for it in row)

    @cached_property
    def flat(self) -> np.ndarray:
        """Return the flattened representation of the tableau."""
        if self._rows:
            return np.concatenate(self._rows)
        return np.array([], dtype=np.int_)

    @cached_property
    def max_entry(self) -> int:
        """Return the maximum entry in the tableau."""
        if not self._rows:
            return 0
        return max(int(row.max()) for row in self._rows)

    def __repr__(self) -> str:
        return f"YT({', '.join(str(list(int(n) for n in r)) for r in self.rows)})"

    def __str__(self) -> str:
        return self.pretty_str()

    def __eq__(self, other) -> bool:
        if not isinstance(other, YoungTableau):
            return False
        if len(self._rows) != len(other._rows):
            return False
        return all(np.array_equal(self_row, other_row) for self_row, other_row in zip(self._rows, other._rows))

    def __hash__(self) -> int:
        # numpy arrays are not hashable; build hash from tuple-of-tuples representation
        return hash(self.rows_tuples)

    def permute(self, permutation: comb.perm_groups.Permutation):
        """Return a new tableau obtained by permuting the flattened positions.

        The permutation acts on box positions (not on entry values), so
        permuting a uniformly-filled tableau has no effect.

        Args:
            permutation: A sympy Permutation of size n (= total number of boxes).

        Returns:
            A new YoungTableau with the same partition and permuted entries.
        """
        assert permutation.size == self.partition.n, f"Permutation size of {permutation} does not match partition size of {self.partition}"
        return YoungTableau.from_flat(permutation(self.flat), self.partition)

    def basis_vector(self, single_system_dimension: int) -> np.ndarray:
        """Return the Schur-Weyl basis vector in V^(tensor n) corresponding to this tableau.

        Args:
            single_system_dimension: Dimension of the single-site Hilbert space V.

        Returns:
            Integer numpy array of length single_system_dimension ** n representing
            the basis vector in the standard tensor-product basis of V^(tensor n).
            Returns the zero vector if the partition height exceeds single_system_dimension.
        """
        from ..algebra import basis
        canonical = self.partition.canonical_tableau
        row_perms = canonical.value_row_stabilizer
        column_perms = canonical.value_column_stabilizer
        flattened = self.flat
        vector = np.zeros(single_system_dimension ** self.partition.n, dtype=np.int_)

        if self.partition.height > single_system_dimension:
            return vector # Vector is zero if the partition height is too big

        # TODO: maybe optimize this by taking the quotient group with the permutation group that keeps the tableaux invariant
        seen_rows = set()

        basis_set = basis.StandardTensorProductBasis(single_system_dimension, self.partition.n)
        for row_perm in row_perms.elements:  # type: comb.Permutation
            # We want to count every row-equivalent tableau only once
            row_perm_labels = tuple(row_perm(flattened))
            if row_perm_labels in seen_rows:
                continue
            seen_rows.add(row_perm_labels)
            for column_perm in column_perms.elements:  # type: comb.Permutation
                labels = np.asarray(column_perm(row_perm_labels))
                vector += column_perm.signature() * basis_set.label_to_vector(labels)
        return vector


    def overlap(self, other: SSYT) -> int:
        """Compute the combinatorial inner product <u_self, u_other>.

        Args:
            other: Another SSYT with the same shape

        Returns:
            Integer overlap value
        """
        if self.partition != other.partition or self.symbol_count() != other.symbol_count():
            return 0

        total = 0

        for self_perm in self.row_equivalent_permutations:
            permuted_self = self.permute(self_perm)
            for other_perm in other.row_equivalent_permutations:
                permuted_other = other.permute(other_perm)
                if (symbol_count_per_column := permuted_self.symbol_count_per_column()) == permuted_other.symbol_count_per_column():
                    if any(any(i >= 2 for i in count.values()) for count in symbol_count_per_column):
                        continue # Repetitions lead to zero because of antisymmetry
                    total += math.prod(comb.Permutation.from_sequence(self_c).signature()*
                                       (~comb.Permutation.from_sequence(other_c)).signature()*multinomial_coeff(count.values())
                                       for self_c, other_c, count in zip(permuted_self.columns, permuted_other.columns, symbol_count_per_column))
        return total

    @cached_property
    def row_equivalent_permutations(self) -> list[comb.Permutation]:
        """Return position permutations that produce row-equivalent tableaux.

        Each permutation in the returned list yields a distinct tableau with the
        same multiset of entries per row. The list is not closed under composition
        (it is a coset transversal, not a group).
        """

        # .coset_transversal() returns a list of right-cosets g such that Hg (when g goes over the list) spans the entire group
        # We want left cosets, which we can obtain by inverting every element, which is the ~ (i.e. __invert__) operation
        return [~g for g in self.partition.canonical_tableau.value_row_stabilizer.coset_transversal(self.stabilizing_row_permutations)]

    def row_equivalent_tableaux(self) -> list[YoungTableau]:
        """Return all distinct tableaux reachable by intra-row entry permutations."""
        return [self.from_flat(p(self.flat), self.partition) for p in self.row_equivalent_permutations]

    @cached_property
    def stabilizing_row_permutations(self) -> comb.PermutationGroup:
        """Return the subgroup of S_n that leaves the tableau exactly unchanged.

        These are position permutations that swap identical entries within rows,
        keeping every row's sequence exactly the same.
        """
        row_start_indices = np.cumsum([0] + list(self.partition))
        # These are just all the permutations that permute the positions of identical elements in each row
        by_row = [
            self._combine_permutation_groups( # Combine the permutations of each individual symbol
                *(self._permutation_group_from_elements(*(row_start_index + i for i, value in enumerate(row) if value == symbol)) # These are the indices of the symbol 'symbol'
                  for symbol, count in symbol_count.items() if count > 1) # No point permuting if there is only a single occurance
            )
            for row, row_start_index, symbol_count in zip(self._rows, row_start_indices, self.symbol_count_per_row())]

        return self._combine_permutation_groups(*by_row) # Combine the permutations of the individual rows


    def _permutation_group_from_elements(self, *els: int) -> comb.PermutationGroup:
        """Return the subgroup of S_n that permutes only the specified positions.

        Args:
            *els: Distinct position indices in {0, ..., n-1}.

        Returns:
            PermutationGroup acting as S_{|els|} on the given positions and as
            the identity on all others.
        """
        max_num = self.partition.n
        if len(els) == 1:
            return comb.PermutationGroup(comb.Permutation([els], size=max_num))
        # The symmetric group on any elements is generated by the cycle spanning the whole list and a single transposition
        return comb.PermutationGroup(comb.Permutation([els], size=max_num), comb.Permutation([[els[0], els[1]]], size=max_num))

    def _combine_permutation_groups(self, *groups: comb.PermutationGroup):
        """Return the permutation group generated by the union of generators of all given groups."""
        if not groups:
            return comb.PermutationGroup(comb.Permutation([], size=self.partition.n))
        return comb.PermutationGroup(*(generator for group in groups for generator in group))

class SYT(YoungTableau):
    """Represents a standard Young tableau (SYT) with associated operations.

    An SYT is a filling of a Young diagram with positive integers such that rows and column are increasing, and such that contains all elements [0, ..., n-1] exactly once.
    This in particular implies that the top-left corner has to be 0.
    """

    @cached_property
    def value_row_stabilizer(self) -> comb.perm_groups.PermutationGroup:
        """Return the group of value permutations that keep each row's value set unchanged.

        For example, for the tableau::

            2 3
            1

        this is the group generated by the transposition (2 3).
        """
        groups = [self._permutation_group_from_elements(*r) for r in self._rows]
        return comb.PermutationGroup(*(generator for group in groups for generator in group))

    @cached_property
    def value_column_stabilizer(self) -> comb.perm_groups.PermutationGroup:
        """Return the group of value permutations that keep each column's value set unchanged.

        For example, for the tableau::

            2 3
            1

        this is the group generated by the transposition (1 2).
        """

        groups = [self._permutation_group_from_elements(*r) for r in self.columns]
        return comb.PermutationGroup(*(generator for group in groups for generator in group))


class SSYT(YoungTableau):
    """Represents a semistandard Young tableau (SSYT) with associated operations.

    An SSYT is a filling of a Young diagram with positive integers such that:
    - Rows are weakly increasing (left to right)
    - Columns are strictly increasing (top to bottom in English notation)

    Unlike the Combinatorics convention, we fill with values starting from zero, i.e. a semistandard tableau will have entries
    0, ..., d-1.
    """

    def _validate(self):
        """Validate that the tableau satisfies SSYT properties."""
        # Check rows are weakly increasing
        super()._validate()
        for i, row in enumerate(self._rows):
            # numpy arrays support indexing to produce scalars
            if any(int(row[j]) > int(row[j+1]) for j in range(len(row)-1)):
                raise ValueError(f"Row {i} is not weakly increasing: {row.tolist()}")

        # Check columns are strictly increasing
        for col_idx in range(self._partition.width):
            col_vals = []
            for row_idx, row in enumerate(self._rows):
                if col_idx < len(row):
                    col_vals.append(int(row[col_idx]))
            if any(col_vals[i] >= col_vals[i+1] for i in range(len(col_vals)-1)):
                raise ValueError(f"Column {col_idx} is not strictly increasing: {col_vals}")

    def __repr__(self):
        return f"SS{super().__repr__()}"

    #
    # Since we iterate over SSYTs a lot, let's cache them.
    #
    _caches = {}

    @classmethod
    def generate_all(cls, partition: Union[Partition, List[int], Tuple[int, ...]], d: int) -> Iterable[SSYT]:
        """Return all SSYTs of given shape with alphabet {0, ..., d-1}."""
        return cls._get_cache(Partition.normalize(partition), d).generate_all()

    @classmethod
    def count(cls, partition: Union[Partition, List[int], Tuple[int, ...]], d: int):
        """Return the number of SSYTs of given shape with alphabet {0, ..., d-1}."""
        return cls._get_cache(Partition.normalize(partition), d).count()

    @classmethod
    def index_of(cls, ssyt: SSYT, d: int):
        """Return the canonical index of ssyt in the ordered SSYT list for its shape."""
        return cls._get_cache(ssyt.partition, d).index_of(ssyt)

    @classmethod
    def tableau_at_index(cls, partition: Partition, d: int, index: int) -> SSYT:
        """Return the SSYT at position index in the canonical ordered SSYT list."""
        return cls._get_cache(partition, d).tableau_at_index(index)

    @classmethod
    def _get_cache(cls, partition: Partition, d: int) -> SSYTCache:
        """Return (or create) the SSYTCache for the given partition and alphabet size."""
        if (partition, d) not in cls._caches:
            cls._caches[(partition, d)] = SSYTCache(partition, d)
        return cls._caches[(partition, d)]

class SSYTCache:
    """Cache for SSYTs of given shape and alphabet size."""
    cache: list[SSYT]
    lookup: dict[SSYT, int]

    def __init__(self, partition: Partition, d: int):
        self.partition = partition
        self.d = d
        self.populated = False

    def generate_all(self):
        """Return the list of all SSYTs, populating the cache on first call."""
        if not self.populated:
            self.populate()
        return self.cache

    def populate(self):
        """Enumerate all SSYTs and build the index lookup table."""
        with MaybeExpensiveComputation(f"Generating all SSYTs of shape {self.partition} and alphabet size {self.d}"):
            self.cache = []
            self.lookup = {}
            for i, ssyt in enumerate(self.iterate_all()):
                self.cache.append(ssyt)
                self.lookup[ssyt] = i
            self.populated = True

    def count(self):
        """Return the number of SSYTs of this shape and alphabet size."""
        if self.populated:
            return len(self.cache)
        else:
            return self.count_formula()

    def index_of(self, ssyt: SSYT):
        """Return the cache index of a given SSYT, populating the cache if needed."""
        if not self.populated:
            self.populate()
        return self.lookup[ssyt]

    def tableau_at_index(self, index: int) -> SSYT:
        """Return the SSYT at the given cache index, populating the cache if needed."""
        if not self.populated:
            self.populate()
        return self.cache[index]

    def iterate_all(self) -> Iterator[SSYT]:
        """Generate all SSYTs of given shape with entries in {0..d-1}.

        Yields:
            SSYT objects
        """
        lam = self.partition

        # Inlined semistandard_tableaux generator (rows weakly increasing, cols strictly increasing)
        T: List[List[Optional[int]]] = [[None] * L for L in lam]
        fill_order = [(i, j) for i, L in enumerate(lam) for j in range(L)]

        def allowed_range(i: int, j: int) -> range:
            left = T[i][j - 1] if j - 1 >= 0 else None
            above = T[i - 1][j] if i - 1 >= 0 and j < len(T[i - 1]) else None
            lo = 0  # zero-based entries
            if left is not None:
                lo = max(lo, int(left))
            if above is not None:
                lo = max(lo, int(above) + 1)
            return range(lo, self.d)

        def backtrack(k: int):
            if k == len(fill_order):
                # convert to concrete int rows and wrap into SSYT
                rows = [[int(val) for val in row if val is not None] for row in T]
                yield SSYT(rows, lam)
                return
            i, j = fill_order[k]
            for v in allowed_range(i, j):
                T[i][j] = v
                yield from backtrack(k + 1)
            T[i][j] = None

        yield from backtrack(0)


    def count_formula(self) -> int:
        """Count SSYTs of given shape with entries in {1..d} using the hook-content formula.

        Uses exact rational arithmetic (Fraction) to avoid floating point rounding.
        """
        part = self.partition
        d = self.d
        res = Fraction(1, 1)
        # iterate rows j (1-based) and columns i (1-based)
        for j in range(1, part.height + 1):
            row_len = part[j - 1]
            for i in range(1, row_len + 1):
                # content = column_index - row_index (0-based equivalent)
                content = i - j
                # arm: boxes to the right in the same row
                arm = row_len - i
                # leg: boxes below in same column
                leg = sum(1 for k in range(j + 1, part.height + 1) if part[k - 1] >= i)
                hook = arm + leg + 1
                res *= Fraction(d + content, hook)
        # result should be integer
        return int(res)


def Gram_matrix(lam: Partition, dim: int):
    """Return the Gram matrix for the SSYT basis of shape `lam` with alphabet size `dim`.

    The matrix G has entries G[i,j] = <u_{τ_i}, u_{τ_j}> computed combinatorially.
    """
    tableaux = list(SSYT.generate_all(lam, dim))
    m = len(tableaux)
    G = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        G[i, i] = tableaux[i].overlap(tableaux[i])
        for j in range(i+1, m):
            val = tableaux[i].overlap(tableaux[j])
            G[i, j] = G[j, i] = val
    return G
