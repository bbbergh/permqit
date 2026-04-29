import itertools
import math
import operator
from functools import lru_cache

import sympy.combinatorics

from .orbits import PairOrbit
from .partition import Partition
from ..algebra.polynomial import Polynomial
from .young_tableau import SSYT


class BlockDiagonalization:
    """
    Represents (properties of) the algebra isomorphism that maps the Endomorphism space into the direct sum of
    matrices on the irreps. Based on the construction in https://arxiv.org/abs/0910.4515v1
    """

    @classmethod
    def tableaux_polynomial(cls, t1: SSYT, t2: SSYT):
        r"""
        Computes the polynomial \ Sum_{r} x^r <C_r e_t1, e_t2>
        where the sum goes over all PairOrbits, and x^r is PairOrbit.to_monomial(), and C_r is PairOrbit.indicator_matrix.matrix
        Since the vectors e_t1, e_t2 span the individual blocks of the irrep decomposition, this polynomial allows one to calculate the
        algebra isomorphism that block-diagonalizes the endomorphism space.

        This follows Theorem 7 of https://arxiv.org/abs/0910.4515v1
        :param t1: A semi-standard young tableau
        :param t2: Another semi-standard young tableau
        :return: The polynomial as specified above
        """
        partition = t1.partition
        assert partition == t2.partition, f"SSYTs must have the same partition, {t1.partition} != {t2.partition}"
        dimension = max(t1.max_entry, t2.max_entry) + 1 # Effective dimension is the maximum entry of the tableaux (+ 1 because the entries are zero-based)
        poly = cls.polynomial(partition) # This is P_λ, the polynomial corresponding to the partition
        partition_tableau = partition.constant_tableau
        orbit_t1 = PairOrbit.from_two_sequences(t1.flat, partition_tableau.flat, dimension)
        orbit_t2 = PairOrbit.from_two_sequences(t2.flat, partition_tableau.flat, dimension)

        for j in range(dimension - 2, -1, -1):  # j decresce: d-2, d-3, ..., 0
            for i in range(dimension - 1, j, -1):  # i decresce: d-1, d-2, ..., j+1
                for k1 in range(orbit_t1.count_matrix[i,j]):
                    _sum = 0
                    for s in range(dimension):
                        _sum += Polynomial.monomial((s, i)) * poly.derivative((s, j)) * (1/(k1 + 1))
                    poly = _sum
                for k2 in range(orbit_t2.count_matrix[i,j]):
                    _sum = 0
                    for s in range(dimension):
                        _sum += Polynomial.monomial((i, s)) * poly.derivative((j, s)) * (1/(k2 + 1))
                    poly = _sum

        return poly

    @classmethod
    def polynomial(cls, partition: Partition) -> Polynomial[tuple[int, int], int]:
        """Returns a polynomial associated to this partition (and thus irreducible representation of Sn)
        This corresponds to (21) of https://arxiv.org/abs/0910.4515v1"""
        # Calculates the differences between the values in the partition
        if partition.height == 0:
            return Polynomial.one()
        differences = [operator.sub(*pair) for pair in itertools.pairwise(partition)] + [partition[-1]]
        prod = Polynomial.one()
        for k in range(partition.height):
            prod *= cls._polynomial_part(k)**differences[k]
        return prod

    @classmethod
    @lru_cache(None)
    def _polynomial_part(cls, k: int) -> Polynomial[tuple[int, int], int]:
        """Returns the Q_k as defined in equation (20) in https://arxiv.org/abs/0910.4515v1
        but with k zero-based instead of one-based."""
        return math.factorial(k+1) * cls._symbolic_determinant(k)

    @staticmethod
    def _symbolic_determinant(k: int):
        """Returns the symbolic determinant of the matrix
        (x_(i,j))_(i,j), where i,j ∈ [0, ..., k-1]
        as a polynomial
        """
        ret = Polynomial.zero()
        # Calculate the determinant by Leibniz's formula
        for perm in sympy.combinatorics.SymmetricGroup(k+1).elements: # type: sympy.combinatorics.Permutation
            monomial_indices = zip(range(k+1), perm(range(k+1)))
            ret += perm.signature() * Polynomial.monomial(*monomial_indices)

        return ret



