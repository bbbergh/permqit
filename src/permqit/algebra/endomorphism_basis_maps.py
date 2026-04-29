"""IndexMapping subclasses for EndomorphismBasis operations.

This module contains LinearMap / IndexMapping implementations that express
structural operations on EndSnOrbitBasis coefficient vectors as pure index
permutations, so they compose naturally with the rest of the LinearMap
infrastructure.

Classes
-------
TransposeIndexMapping
    IndexMapping that reorders an EndSnOrbitBasis coefficient vector to give
    the coefficients of the transposed operator.
"""

from __future__ import annotations

import numpy as np

from .linear_map import ScatterIndexMapping
from ..utilities.timing import ExpensiveComputation
from ..utilities.caching import WeakRefMemoize

__all__ = ["TransposeIndexMapping"]


class TransposeIndexMapping(ScatterIndexMapping, metaclass=WeakRefMemoize):
    """IndexMapping that permutes EndSnOrbitBasis coefficients to apply transpose.

    ``perm[i] = j`` means that basis element *i* (in ``basis``) maps to
    basis element *j* after transposition, i.e.
    ``transposed_coeffs[j] = original_coeffs[i]``.

    Parameters
    ----------
    basis : EndSnOrbitBasis
        The orbit basis on which the transpose acts.
    """

    def __init__(self, basis) -> None:
        # basis_from and basis_to are the same object: transpose is an
        # automorphism of the basis (as a vector space).
        self.basis_from = basis
        self.basis_to = basis
        self._basis = basis

    def _calculate_index_mapping(self) -> np.ndarray:
        with ExpensiveComputation(
            f"Constructing TransposeIndexMapping for {self._basis}"
        ):
            lookup = (
                self._basis.label_to_index(r.transpose())
                for r in self._basis
            )
            return np.fromiter(lookup, dtype=np.intp, count=self._basis.size())

    def __str__(self) -> str:
        return f"TransposeIndexMapping({self._basis})"

    def __repr__(self) -> str:
        return str(self)
