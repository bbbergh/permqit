import dataclasses
import types
import typing
from functools import cached_property
from typing import Callable, Iterable, Self, Tuple, Collection

import numpy as np

from .basis import Basis, MatrixBasis, MatrixStandardBasis, DenseOrSparse
from .linear_map import ScatterIndexMapping, GivenScatterIndexMapping, CoefficientData
from ..representation.orbits import PairOrbit
from ..utilities import caching
from ..utilities.numpy_utils import contract_at_axis

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(unsafe_hash=True)
class BasisSubset[LabelT](Basis[LabelT]):
    """
    Represents a basis given by all elements from ``BasisSubset.subset_of``
    that meet the given ``BasisSubset.predicate``. This obviously spans a subspace of the span of ``BasisSubset.subset_of``.
    The predicate is a function that takes a label and returns True if it should be included in the subset, and False otherwise.

    In order to make use of the caching ane reusing features of the library, the predicate provided should be hashable.
    """

    subset_of: Basis[LabelT]
    predicate: Callable[[LabelT], bool]|None = None
    from_valid_indices: np.ndarray|None = None

    def __post_init__(self):
        # Need to wrap the predicate in a staticmethod, since otherwise it would be treated as an instance method and
        # receive 'self' as the first argument, which is not what we want.
        if self.predicate is not None:
            assert self.from_valid_indices is None, "Cannot provide both a predicate and from_valid_indices."
            self.predicate = staticmethod(self.__dict__['predicate'])
        elif self.from_valid_indices is not None:
            assert self.predicate is None, "Cannot provide both a predicate and from_valid_indices."
        else:
            raise ValueError("Must provide a predicate or from_valid_indices.")

    @property
    def predicate_instance(self) -> typing.Any:
        return self.predicate.__func__  # ty:ignore[unresolved-attribute]
        
    @property
    def valid_indices(self) -> np.ndarray:
        """Returns a numpy array of indices in the full basis."""
        if self.from_valid_indices is not None:
            return self.from_valid_indices
        return self.index_mapping.index_mapping().as_numpy()

    @cached_property
    def index_mapping(self) -> ScatterIndexMapping:
        """Returns the mapping of indices in the subset basis to indices in the full basis."""
        if self.from_valid_indices is not None:
            return GivenScatterIndexMapping(self.from_valid_indices, self, self.subset_of)
        return BasisSubsetIndices(self)

    @cached_property
    def inverse_index_mapping(self):
        """Returns the inverse mapping, mapping indices in the full basis to indices in the subset."""
        return self.index_mapping.inverse()

    def size(self) -> int:
        return len(self.valid_indices)

    def iterate_labels(self) -> Iterable[LabelT]:
        full = self.subset_of
        for full_idx in self.valid_indices:
            yield full.label_at_index(int(full_idx))

    def iterate_vectors(self) -> Iterable[DenseOrSparse]:
        full = self.subset_of
        for full_idx in self.valid_indices:
            yield full.vector_at_index(int(full_idx))

    def label_to_vector(self, label: LabelT) -> DenseOrSparse:
        return self.subset_of.label_to_vector(label)

    def label_to_index(self, label: LabelT) -> int:
        full_idx = self.subset_of.label_to_index(label)
        masked_idx = int(self.inverse_index_mapping.index_mapping()[full_idx])
        if masked_idx == -1:
            raise KeyError(f"LabelT {label} is not present in this masked basis.")
        return masked_idx

    def label_at_index(self, idx: int) -> LabelT:
        return self.subset_of.label_at_index(int(self.valid_indices[idx]))

    def vector_at_index(self, idx: int) -> DenseOrSparse:
        return self.subset_of.vector_at_index(int(self.valid_indices[idx]))

    @caching.cache_noargs
    def all_vectors(self) -> np.ndarray:
        return self.xp.stack(list(self.iterate_vectors())).asformat('gcxs')

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> np.ndarray:
        return contract_at_axis(self.all_vectors(), coeffs,0, coeff_axis, xp=self.xp)

    def __str__(self):
        return f"{type(self).__name__}(subset_of={self.subset_of!s}, predicate={self.predicate.__func__!s})"  # ty:ignore[unresolved-attribute]

@dataclasses.dataclass(unsafe_hash=True)
class MatrixBasisSubset[LabelT](BasisSubset[LabelT], MatrixBasis[LabelT]):
    """
    A version of BasisSubset that also inherits from MatrixBasis, and implements the matrix-specific methods.
    """
    subset_of: MatrixBasis[LabelT]

    def transpose[T: CoefficientData](self, coeffs: T, axis: int = -1) -> T:
        return self.inverse_index_mapping.apply_to_coefficient_vector(self.subset_of.transpose(
            self.index_mapping.apply_to_coefficient_vector(coeffs, axis=axis),
            axis=axis,
        ), axis=axis)

    def trace(self, coeffs: np.ndarray, axis: int = -1) -> np.ndarray|float:
        return self.subset_of.trace(self.index_mapping.apply_to_coefficient_vector(coeffs, axis=axis), axis=axis)

    @property
    def dimension(self) -> int:
        return self.subset_of.dimension

class MatrixStandardBasisSubset(MatrixBasisSubset[tuple[int, int]]):
    subset_of: MatrixStandardBasis

    @cached_property
    def mask(self) -> 'MatrixEntryMask':
        return MatrixEntryMask((self.subset_of.d, self.subset_of.d), [self.subset_of.label_at_index(i) for i in self.valid_indices])


class BasisSubsetIndices(ScatterIndexMapping, metaclass=caching.WeakRefMemoize):
    basis_from: BasisSubset
    basis_to: Basis

    def __init__(self, subset: BasisSubset):
        self.basis_from = subset
        self.basis_to = subset.subset_of

    def _calculate_index_mapping(self) -> np.ndarray:
        valid = [idx for idx, label in enumerate(self.basis_to.iterate_labels()) if self.basis_from.predicate(label)]
        return np.asarray(valid)


class MatrixEntryMask:
    """
    Represents a sparsity mask for matrices (or tensors) reflecting a given symmetry.

    Attributes:
        shape: The NumPy shape of the matrix/tensor.
        index_pairs: The allowed indices that are permitted to be non-zero.
    """
    index_pairs: frozenset[tuple[int, ...]]
    shape: tuple[int, ...]
    name: str|None = None # Used for pretty printing

    def __init__(self, shape: Tuple[int, ...], index_pairs: Collection[Tuple[int, ...]], * , name: str|None = None):
        self.name = name
        self.shape = shape
        index_array = np.asarray(list(index_pairs))
        self.index_pairs = frozenset(index_pairs)

        self._valid_mask = np.zeros(shape, dtype=bool)
        if self.index_pairs:
            # Handle edge case where index_pairs might be a set or list
            indices = tuple(index_array.T) # Need tuple here to get the indexing right
            self._valid_mask[indices] = True

    def is_valid_matrix(self, mat: np.ndarray, tol: float = 0) -> bool:
        """
        Check if a given matrix is valid under this mask (i.e. all elements outside
        the mask are zero).
        """
        # Automatically reshape if sizes match
        if mat.shape != self.shape:
            if mat.size == self._valid_mask.size:
                mat = mat.reshape(self.shape)
            else:
                raise ValueError(f"Matrix size {mat.size} does not match expected size {self._valid_mask.size}")

        invalid_region = mat[~self._valid_mask]
        if invalid_region.size == 0:
            return True
        return float(np.max(np.abs(invalid_region))) <= tol

    def __hash__(self):
        return hash(self.shape) + hash(self.index_pairs)

    def __eq__(self, other):
        if not isinstance(other, MatrixEntryMask):
            return NotImplemented
        return self.shape == other.shape and self.index_pairs == other.index_pairs

    def __str__(self):
        if self.name is not None:
            return self.name
        return str(self.index_pairs)

    def __repr__(self):
        if self.name is not None:
            return self.name
        return f"{type(self).__name__}({self!s})"


@dataclasses.dataclass(frozen=True)
class OrbitFitsMaskPredicate:
    mask: MatrixEntryMask

    def __call__(self, orbit: PairOrbit):
        return self.mask.is_valid_matrix(orbit.count_matrix)


@dataclasses.dataclass(frozen=True)
class IndexIsValidPredicate:
    mask: MatrixEntryMask

    def __call__(self, index: tuple[int, ...]):
        return self.mask._valid_mask[index]
