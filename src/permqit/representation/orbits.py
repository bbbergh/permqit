"""
Orbit-related utilities and indexing helpers.
"""
from __future__ import annotations

from functools import cached_property
from typing import List, Tuple, Sequence, Iterable
import numpy as np
import itertools
import scipy
import sparse
from sparse import SparseArray
from sympy.combinatorics import SymmetricGroup

from .partition import Partition
from .combinatorics import multinomial_coeff, weak_compositions
from .young_tableau import SSYT, YoungTableau


class PairOrbit:
    """
    Represents an orbit of pairs of indices (i,j) under the joint action of the symmetric group Sn.
    Each orbit corresponds to a basis matrix of the algebra End^(Sn)(V^n). The orbits can be represented by a count matrix
    which counts how many times each pair (i,j) appears in the two n-strings.
    """
    count_matrix: np.ndarray # A matrix whose (i,j)th entry counts the number of times the pair (i,j) appears in the (pair of) n-strings. This is obviously invariant under Sn, and hence constant on the group orbit
    n: int # the n in Sn we consider. The sum of all elements of self.count_matrix will be equal to this
    d: int # The dimension of the single vector space, i.e. the number of pairs (i,j) is given by d^2. The shape of _count_matrix is d x d

    def __init__(self, count_matrix: np.ndarray, n: int=None, d: int=None):
        if n is not None:
            assert np.sum(count_matrix) == n
        else:
            n = np.sum(count_matrix)
        if d is None:
            d = count_matrix.shape[0]
        assert count_matrix.shape == (d, d)
        self.count_matrix = count_matrix
        self.n = n
        self.d = d

    @classmethod
    def from_two_sequences(cls, first_string: Sequence[int], second_string: Sequence[int], single_dimension: int) -> PairOrbit:
        """Constructs the orbit corresponding to all joint permutations of the two given sequences of numbers.
        I.e. PairOrbit.from_two_sequences([0,1,3], [1,2,2], 4) constructs the orbit where the pairs (0,1), (1,2) and (3,2) each appear once.
        The single dimension must be larger or equal than the largest element of the sequences.
        """
        n = len(first_string)
        assert n == len(second_string), f"Lengths of {first_string} and {second_string} do not match"
        d = single_dimension
        count_matrix = np.zeros((d,d), dtype=np.int_)
        for i,j in zip(first_string, second_string):
            count_matrix[i,j] += 1
        return cls(count_matrix, n, d)

    @classmethod
    def from_two_strings(cls, first_string: str, second_string: str, single_dimension: int) -> PairOrbit:
        """Constructs the orbit corresponding to all joint permutations of the two given sequences, which are given as strings.
        I.e. PairOrbit.from_two_strings('013', '122', 4) constructs the orbit where the pairs (0,1), (1,2) and (3,2) each appear once.
        The single dimension must be larger or equal than the largest element of the sequences.
        """
        return cls.from_two_sequences(tuple(int(k) for k in first_string), tuple(int(k) for k in second_string), single_dimension)

    def get_representative(self) -> np.ndarray:
        """Returns two representative strings for this orbit as a numpy array of shape (n, 2) and with entries being elements of [0, ..., d-1]"""
        seq = []
        it = np.nditer(self.count_matrix, flags=['multi_index'])
        for cnt in it:
            seq.extend((it.multi_index,)*int(cnt))
        return np.asarray(seq)

    def get_representative_strings(self) -> tuple[str, str]:
        return tuple("".join(str(c) for c in s) for s in self.get_representative().T) # type: ignore

    @cached_property
    def indicator_matrix(self) -> "OrbitMatrix":
        """
        Returns an object corresponding to a matrix in End^(S_n)(V^n). The set of all of these matrices (corresponding to all possible PairOrbits) form
        a basis of End^(S_n)(V^n). The matrix has the property that it has a 1 at index (i,j) if (i,j) lies in the orbit.
        This will be the case if (i,j) can be constructed by applying the same permutation to both parts of self.get_representative().
        Returns a wrapper object which exposes some properties which can be computed efficiently even without constructing the matrix explicitly.
        This wrapper object also allows to explicitly construct the matrix (exponentially large in n) if desired.
        :return:
        """
        return OrbitMatrix(self)

    @classmethod
    def count(cls, n: int, d: int):
        """Returns the number of distinct PairOrbits for given n and d. This is equal to the dimension of End^(S_n)(V^n)."""
        return scipy.special.comb(n + d**2 - 1, n, exact=True)

    @classmethod
    def generate_all(cls, n: int, d: int):
        """Generates all distinct PairOrbits for given n and d."""

        # This is really just all ways of assigning n values to d**2 possibilities in the count matrix
        for arr in weak_compositions(n, d**2): # This iterates through all possibilities of choosing d^2 indices n times
            yield cls(arr.reshape((d,d)), n, d)

    @classmethod
    def generate_all_count_matrices(cls, n: int, d: int) -> Iterable[np.ndarray]:
        """Same as generate_all, but instead of returning PairOrbits just returns the corresponding count matrices as numpy arrays"""
        for arr in weak_compositions(n, d**2): # This iterates through all possibilities of choosing d^2 indices n times
            yield arr.reshape((d,d))

    @classmethod
    def index_of(cls, orbit: PairOrbit) -> int:
        """Returns the index of the given orbit in the list of all PairOrbits for n and d."""
        raise NotImplementedError()

    def __repr__(self):
        return f"PairOrbit(\n{self.count_matrix})"

    def to_monomial(self) -> tuple[tuple[int, int], ...]:
        """
        Returns the monomial representation of this orbit, i.e. a product of terms (i,j)**count[i,j]
        where count[i,j] is the number of times the pair (i,j) appears in the orbit. The indices i,j are zero-based.
        :return:
        """
        seq = []
        it = np.nditer(self.count_matrix, flags=['multi_index'])
        for cnt in it:
            seq.extend((it.multi_index,)*int(cnt))
        # nditer is not guaranteed to run in ascending order, for example if the array we passed into PairOrbit was (...).T or (...)[::-1]
        return tuple(sorted(tuple(seq)))

    @classmethod
    def from_monomial(cls, monomial: tuple[tuple[int, int], ...], single_dimension:int) -> PairOrbit:
        """Returns the PairOrbit corresponding to the given monomial representation."""
        if not monomial:
            return cls.from_two_sequences([], [], single_dimension)
        seq = np.asarray(monomial).T
        return cls.from_two_sequences(seq[0], seq[1], single_dimension)

    def __eq__(self, other):
        return np.array_equal(self.count_matrix, other.count_matrix)

    def __hash__(self):
        return hash(self.count_matrix.data.tobytes())

    def get_associated_partition(self):
        """Returns a partition associated to this orbit."""

        return Partition(tuple(sorted(tuple(n for n in self.count_matrix.sum(axis=0) if n > 0))))

    def get_associated_tableau(self) -> YoungTableau | tuple[tuple[int, ...], ...] :
        """Returns a tableau associated to this orbit. This in general will not have decreasing lengths of rows, in which case it is returned as a nested tuple of ints."""
        nested = [sum(((j,) * self.count_matrix[j, i] for j in range(self.d)), ()) for i in range(self.d)]
        tableau = tuple(row for row in nested if row)
        lengths = [len(row) for row in tableau]
        if sorted(lengths) != lengths:
            return tableau
        return YoungTableau(tableau)

    @property
    def T(self):
        return self.transpose()

    def transpose(self) -> PairOrbit:
        """
        Returns the transpose of this orbit. This is the orbit whose indicator matrix is the transpose of this orbit's indicator matrix.
        Equivalently, this corresponds to just transposing the count matrix.
        """
        return PairOrbit(self.count_matrix.T, self.n, self.d)
    
    @property
    def TB(self):
        return self.partial_transpose()
    
    def partial_transpose(self, system: str = 'B') -> PairOrbit:
        """
        Returns the partial transpose of this orbit. This is the orbit whose indicator matrix is the partial transpose of this orbit's indicator matrix.

        The count matrix is d x d where d = dA * dB. Index i decomposes as i = iA * dB + iB.
        Reshaping to (dA, dB, dA, dB) gives C[iA, iB, jA, jB].

        :param system: 'A' for partial transpose on A (swaps iA <-> jA),
                       'B' for partial transpose on B (swaps iB <-> jB)
        :return: PairOrbit with the partial-transposed count matrix
        """
        dA = int(np.sqrt(self.d))
        dB = dA  # Not implemented for non-square systems yet

        # Reshape: C[i, j] -> C[iA, iB, jA, jB]
        reshaped = self.count_matrix.reshape(dA, dB, dA, dB)

        if system == 'A':
            # Partial transpose on A swaps iA <-> jA (indices 0 and 2)
            transposed = reshaped.transpose(2, 1, 0, 3)
        elif system == 'B':
            # Partial transpose on B swaps iB <-> jB (indices 1 and 3)
            transposed = reshaped.transpose(0, 3, 2, 1)
        else:
            raise ValueError(f"system must be 'A' or 'B', got {system!r}")

        # Reshape back to (d, d)
        new_count_matrix = transposed.reshape(self.d, self.d)

        return PairOrbit(new_count_matrix, self.n, self.d)
    

    def partial_trace_to_single(self) -> np.ndarray:
        """
        Compute Tr_{2...n}(C_M) where C_M is the indicator matrix of this orbit.
        
        Returns a d×d matrix representing the partial trace over subsystems 2...n,
        keeping only the first subsystem.
        
        Key insights:
        - If m_off > 1: Multiple mismatches → trace kills everything → return 0
        - If m_off = 0: Fully diagonal orbit → all elements survive → return (N/n)*M
        - If m_off = 1: One mismatch at (a,b) → only N/n elements survive, 
        all contributing to position (a,b) → return (N/n) ONLY at (a,b), zero elsewhere
        
        The diagonal entries of M in case 3 represent the "tail" matches that get traced out!
        
        Returns:
            np.ndarray: d×d matrix, the reduced density matrix on subsystem 1
        """
        d = self.d
        n = self.n
        
        if n == 0:
            return np.zeros((d, d), dtype=float)
        
        N = multinomial_coeff(self.count_matrix.ravel())
        
        # Count off-diagonal transitions
        off_diag_mask = ~np.eye(d, dtype=bool)
        m_off = int(np.sum(self.count_matrix[off_diag_mask]))
        
        if m_off > 1:
            # Too many mismatches - trace kills everything
            return np.zeros((d, d), dtype=float)
        
        if m_off == 0:
            # Fully diagonal: standard formula
            return (N / n) * self.count_matrix
        
        # m_off == 1: Find the single off-diagonal entry
        result = np.zeros((d, d), dtype=float)
        off_indices = np.argwhere(off_diag_mask & (self.count_matrix > 0))
        
        if len(off_indices) == 1:
            a, b = off_indices[0]
            result[a, b] = N / n
        
        return result
    
    
class OrbitMatrix:
    """
    To each Orbit we can associate a matrix in End^(S_n)(V^n). The set of all of these matrices corresponding to all the PairOrbits form
    a basis of End^(S_n)(V^n). The matrix has the property that it has a 1 at index (i,j) if (i,j) lies in the orbit.
    This will be the case if (i,j) can be constructed by applying the same permutation to both parts of PairOrbit.get_representative().
    The matrix is of exponential size, but many properties of it can be computed in polynomial time.
    """

    def __init__(self, orbit: PairOrbit):
        self.orbit = orbit

    @cached_property
    def matrix(self) -> sparse.GCXS:
        """
        This constructs the matrix (of exponential size in n) explicitly.
        :return: a d**n x d**n dimensional matrix
        """
        return self._get_matrix()

    def _get_matrix(self, xp=sparse, dtype=np.int_) -> sparse.GCXS:
        orbit = self.orbit
        if orbit.n == 0:
            return xp.ones((), dtype=dtype)
        mat = xp.DOK(shape=(orbit.d,) * 2 * orbit.n, dtype=dtype) # All n rows first, then all n columns
        idx = orbit.get_representative().transpose()
        for perm in SymmetricGroup(orbit.n).elements:
            idx0, idx1 = perm(idx[0]), perm(idx[1])
            mat[tuple(idx0) + tuple(idx1)] = 1
        return xp.reshape(mat, (orbit.d ** orbit.n, orbit.d ** orbit.n)).asformat('gcxs')

    @cached_property
    def trace(self):
        """Returns the trace of this orbit matrix. Can be calculated in polynomial time without actually constructing the matrix."""
        if self.orbit.count_matrix.trace() != self.orbit.n: # If not all entries lie on the diagonal of the count matrix, none lie on the diagonal of the indicator matrix
            return 0
        return multinomial_coeff(self.orbit.count_matrix.diagonal())

    @cached_property
    def HS_norm(self):
        """Returns the Hilbert Schmidt norm (i.e. Tr(A^dagger A) ) of this orbit matrix. Can be calculated in polynomial time without actually constructing the matrix."""
        return multinomial_coeff(self.orbit.count_matrix.ravel())
    
    @cached_property
    def partial_trace_single(self) -> np.ndarray:
        """
        Returns Tr_{2...n}(indicator_matrix) as a d×d matrix.
        Can be calculated in polynomial time without constructing the full matrix.
        This is useful for computing marginals in k-extendibility SDPs.
        """
        return self.orbit.partial_trace_to_single()


    def __repr__(self):
        return f"OrbitMatrix(\n{self.orbit.count_matrix})"

    def __eq__(self, other):
        return self.orbit == other.orbit

    def __hash__(self):
        return hash((type(self), self.orbit))  # Make this different from PairOrbit.__hash__
