from __future__ import annotations
from permqit.utilities.numpy_utils import contract_at_axis, ArrayAPICompatible

import abc
import dataclasses
import itertools
import math
import types
import typing
from functools import cached_property
from typing import Iterable, Sequence
import numpy as np
import sparse

from ..utilities.caching import WeakRefMemoize
from ..utilities.numpy_utils import multi_vector_kron

__all__ = ["Basis", "VectorStandardBasis", "TensorProductBasis", "StandardTensorProductBasis", "MatrixBasis", "MatrixStandardBasis", "MatrixTensorProductBasis"]

type DenseOrSparse = np.ndarray | sparse.GCXS
type dtypeT = np.dtype | type[np.number]

@dataclasses.dataclass(unsafe_hash=True)
class Basis[LabelT](abc.ABC, metaclass=WeakRefMemoize):
    """An abstract base class for bases of vector spaces.
    """

    xp: types.ModuleType = dataclasses.field(default=np, kw_only=True, repr=False)

    @abc.abstractmethod
    def size(self) -> int:
        """Returns the size of the basis, i.e. the number of basis vectors"""
        ...

    def __len__(self):
        return self.size()

    def __getitem__(self, i: int) -> LabelT:
        """Returns the ith label."""
        return self.label_at_index(i)

    def __iter__(self):
        return iter(self.iterate_labels())

    @abc.abstractmethod
    def iterate_labels(self) -> Iterable[LabelT]:
        """Returns an iterable of labels for this basis. Labels can have different forms/shapes depending on the basis."""
        ...

    @abc.abstractmethod
    def iterate_vectors(self) -> Iterable[DenseOrSparse]:
        ...

    def all_vectors(self) -> DenseOrSparse:
        """Returns all the basis vectors as a numpy array, where the first dimension enumerates the different basis vectors."""
        return self.xp.stack(list(self.iterate_vectors()))

    @abc.abstractmethod
    def label_to_vector(self, label: LabelT) -> DenseOrSparse:
        """Returns the basis vector corresponding to the label"""
        ...

    @abc.abstractmethod
    def label_to_index(self, label: LabelT) -> int:
        """Returns the index of the basis vector corresponding to the label"""
        ...

    @abc.abstractmethod
    def label_at_index(self, idx: int) -> LabelT:
        """Returns the label of the basis vector at the given index"""
        ...

    @abc.abstractmethod
    def vector_at_index(self, idx: int) -> DenseOrSparse:
        """Returns the basis vector at the given index"""
        ...

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> DenseOrSparse|typing.Any:
        """Returns the linear combination of the basis vectors with coefficients coeffs.
        Coeffs can be an array of dimension larger than one, in which case this operation is applied to coeff_axis.
        """
        return contract_at_axis(self.all_vectors(), coeffs, matrix_axis=0, vector_axis=coeff_axis, xp=self.xp)

@dataclasses.dataclass(unsafe_hash=True)
class VectorStandardBasis(Basis[int]):
    """Represents the standard basis of a 'dim'-dimensional vector space, i.e. the vectors (e_i)_j = δ_(ij).
    Vectors are labelled by their numeric index i ∈ [0, ..., dim-1]
     """
    dim: int
    dtype: dtypeT = dataclasses.field(default=np.int_, kw_only=True, repr=False)

    def all_vectors(self) -> np.ndarray:
        return self.xp.eye(self.dim, dtype=self.dtype)

    def size(self) -> int:
        return self.dim

    def iterate_labels(self) -> Iterable[int]:
        return np.arange(self.dim)

    def iterate_vectors(self) -> Iterable[np.ndarray]:
        return self.all_vectors()

    def label_to_vector(self, label: int) -> np.ndarray:
        v = self.xp.zeros(self.dim, dtype=self.dtype)
        v[label] = 1
        return v

    def label_to_index(self, label: int) -> int:
        return label

    def label_at_index(self, idx: int) -> int:
        return idx

    def vector_at_index(self, idx: int) -> np.ndarray:
        return self.label_to_vector(idx)

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> DenseOrSparse:
        return coeffs


@dataclasses.dataclass(unsafe_hash=True)
class TensorProductBasis[LabelT](Basis[tuple[LabelT, ...]]):
    """
    Represents the tensor product basis of a sequence of bases.
    Basis vectors are the self.xp.kron product of the vectors of the bases.
    Labels are tuples of labels corresponding to the labels of the tensor components.
    Ordering is lexicographic (row-major), i.e. the rightmost index varies fastest.
    """
    bases: tuple[Basis[LabelT], ...]

    @cached_property
    def basis_indices_multiplier(self) -> np.ndarray:
        """
        Returns the cumulative product of the sizes of the bases. This is so that
        np.dot(local_indices, basis_indices_multiplier) gives the (flat) index of the basis vector.
        """
        # Lexicographic (row-major) ordering: rightmost index varies fastest
        # This matches numpy.unravel_index and itertools.product conventions
        return np.concatenate([np.cumprod(self.sizes[::-1][:-1])[::-1], [1]])

    @cached_property
    def non_trivial_basis_indices_multiplier(self) -> np.ndarray:
        """
        Similar to self.basis_indices_multiplier, but ignoring any parts that have dimension 1 (as for those there is only a single valid index: 0).
        """
        return np.concatenate([np.cumprod(self.sizes[self.sizes > 0][::-1][:-1])[::-1], [1]])


    def __post_init__(self):
        assert len(self.bases) > 0, (
            "TensorProductBasis must have at least one basis, but has none."
        )
        # Lexicographic (row-major) ordering: rightmost index varies fastest
        # This matches numpy.unravel_index and itertools.product conventions
        self.sizes = np.asarray([b.size() for b in self.bases])

        self.xp = sparse if any(b.xp == sparse for b in self.bases) else np

    @property
    def n(self):
        return len(self.bases)

    def size(self) -> int:
        return math.prod(b.size() for b in self.bases)

    def iterate_labels(self) -> Iterable[tuple[LabelT, ...]]:
        return itertools.product(*(b.iterate_labels() for b in self.bases))

    def multi_vector_kron(self, *vectors: DenseOrSparse) -> np.ndarray:
        return multi_vector_kron(self.xp, *vectors)

    def iterate_vectors(self) -> Iterable[np.ndarray]:
        # Very inefficient implementation but probably not needed anyway
        for vectors in itertools.product(*(b.iterate_vectors() for b in self.bases)):
            yield self.multi_vector_kron(*vectors)

    def label_to_vector(self, label: tuple[LabelT, ...]) -> np.ndarray:
        vectors = [b.label_to_vector(l) for b, l in zip(self.bases, label)]
        return self.multi_vector_kron(*vectors)

    def label_to_index(self, label: tuple[LabelT, ...]) -> int:
        local_indices = [b.label_to_index(l) for b, l in zip(self.bases, label)]
        return np.dot(local_indices, self.basis_indices_multiplier)

    def label_at_index(self, idx: int) -> tuple[LabelT, ...]:
        local_indices = (idx // self.basis_indices_multiplier) % self.sizes
        return tuple(b.label_at_index(int(i)) for b, i in zip(self.bases, local_indices))

    def vector_at_index(self, idx: int) -> np.ndarray:
        local_indices = (idx // self.basis_indices_multiplier) % self.sizes
        return self.multi_vector_kron(*(b.vector_at_index(int(i)) for b, i in zip(self.bases, local_indices)))

    def permute_tensor_factors(self, coeffs: np.ndarray, permutation: Sequence[int], *, axis=-1):
        axis = axis % coeffs.ndim
        origshape = coeffs.shape
        assert len(permutation) == len(self.bases)
        coeffs = coeffs.reshape(coeffs.shape[:axis] + self.sizes + coeffs.shape[axis+1:])
        coeffs = coeffs.transpose(*[*range(axis), *(axis + p for p in permutation), *(range(axis + len(self.bases), coeffs.ndim))])
        return coeffs.reshape(origshape)




@dataclasses.dataclass(unsafe_hash=True, init=False)
class StandardTensorProductBasis(VectorStandardBasis):
    """Represents the standard basis of 'n' copies of a 'single_dimensional'-dimensional vector space, made up
    of all possible tensor products of n 'VectorStandardBasis' vectors.
    Basis vectors are labelled by a tuple of labels corresponding to the labels of their tensor components.
    This is an optimization of TensorProductBasis for the case where all bases are n identical copies of VectorStandardBasis.
    """
    single_dimension: int
    n: int
    def __init__(self, single_dimension: int, n: int, / , xp=np, dtype=np.int_):
        self.single_dimension, self.n = single_dimension, n
        super().__init__(single_dimension**n, xp=xp, dtype=dtype)

    def from_n_indices(self, indices: tuple[int, ...]) -> np.ndarray:
        """Returns the tensor product basis vector e_{i_1} ⊗ e_{i_2} ⊗ ... ⊗ e_{i_n}, where indices = [i_1, i_2, ..., i_n]."""
        assert len(indices) == self.n
        arr = self.xp.zeros((self.single_dimension,) * self.n, dtype=self.dtype)
        arr[*indices] = 1
        return arr.ravel()

    def iterate_labels(self) -> Iterable[tuple[int, ...]]:  # ty:ignore[invalid-method-override]
        for i in range(self.single_dimension ** self.n):
            yield np.unravel_index(i, (self.single_dimension,) * self.n)

    def label_to_vector(self, label: tuple[int, ...]):  # ty:ignore[invalid-method-override]
        """Returns the tensor product basis vector e_{i_1} ⊗ e_{i_2} ⊗ ... ⊗ e_{i_n}, where label = [i_1, i_2, ..., i_n]."""
        return self.from_n_indices(label)

    def label_to_index(self, label: tuple[int, ...]):  # ty:ignore[invalid-method-override]
        """Returns the index of the basis vector corresponding to the label"""
        return np.ravel_multi_index(label, (self.single_dimension,) * self.n)

    def label_at_index(self, idx: int) -> tuple[int, ...]:  # ty:ignore[invalid-method-override]
        return tuple(np.unravel_index(idx, (self.single_dimension,) * self.n)) # type: ignore

    def vector_at_index(self, idx: int) -> np.ndarray:
        """Override to convert index to label tuple before calling label_to_vector."""
        label = self.label_at_index(idx)
        return self.label_to_vector(label)


class MatrixBasis[LabelT](Basis[LabelT]):
    """ Abstract base class for bases where the 'vectors', i.e., the elements of the vector space, are matrices"""
    @abc.abstractmethod
    def transpose[T: ArrayAPICompatible](self, coeffs: T, axis=-1) -> T:
        """Returns the coefficients corresponding to the transpose of the matrix which is the linear combination of the given coefficients.
        :param axis: Axis which takes the coefficients to transpose. All other axes are ignored.
        """
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """
        Returns the dimension of the vector space the matrix acts on, i.e. the side length of the matrix.
        :return:
        """
        pass

    @abc.abstractmethod
    def trace[T: ArrayAPICompatible](self, coeffs: T, axis=-1) -> T|float:
        """Calculates the trace of the matrix given by the linear combination of the basis vectors with coefficients coeffs.
        If axis is given, calculates the partial trace, where the given axis of the array coeffs contains the coeffs with respect to this basis.
        """
        pass


@dataclasses.dataclass(unsafe_hash=True)
class MatrixStandardBasis(MatrixBasis[tuple[int, int]]):
    """Represents the standard basis of the (vector space of) dxd matrices, i.e. (e_(ij))_(kl) = δ_(ik) δ_(jl).
    Basis elements are labeled by their index (i,j), represented as a tuple with entries i,j ∈ [0, ..., d-1]
    """

    d: int
    dtype: dtypeT = dataclasses.field(default=np.int_, kw_only=True, repr=False)

    def all_vectors(self) -> np.ndarray:
        return self.xp.eye(self.d**2, dtype=self.dtype).reshape((self.d**2, self.d, self.d))

    def label_to_vector(self, label: tuple[int, int]):
        i,j = label
        mat = self.xp.zeros((self.d, self.d), dtype=self.dtype)
        mat[i, j] = 1
        return mat

    def iterate_labels(self):
        for i in range(self.d ** 2):
            yield np.unravel_index(i, (self.d,) * 2)

    def label_to_index(self, label: tuple[int, int]):
        return np.ravel_multi_index(label, (self.d,) * 2)

    def linear_combination(self, coeffs: DenseOrSparse, *, coeff_axis=-1) -> DenseOrSparse:
        if coeff_axis != -1:
            raise ValueError("Linear combination of MatrixStandardBasis currently only supported for coeff_axis=-1")
        assert coeffs.shape[-1] == self.d**2, f"Last dimension of coeffs must be equal to d**2, got {coeffs.shape}"
        return coeffs.reshape(coeffs.shape[:-1] + (self.d, self.d))

    def size(self) -> int:
        return self.d**2

    def iterate_vectors(self) -> Iterable[np.ndarray]:
        return self.all_vectors()

    def label_at_index(self, idx: int) -> tuple[int, int]:
        return np.unravel_index(idx, (self.d,) * 2) # type: ignore

    def vector_at_index(self, idx: int) -> np.ndarray:
        return self.label_to_vector(self.label_at_index(idx))

    def transpose(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        # Essentially equivalent to coeffs.reshape((self.d, self.d)).T.ravel()
        axis = axis % coeffs.ndim
        origshape = coeffs.shape
        assert coeffs.shape[axis] == self.d**2, f"Chosen axis size of coeffs must be equal to d**2={self.d**2}, got shape={coeffs.shape}, with {axis=}"
        coeffs = coeffs.reshape(coeffs.shape[:axis] + (self.d, self.d) + coeffs.shape[axis+1:])
        return np.swapaxes(coeffs, axis, axis+1).reshape(origshape)

    @property
    def dimension(self) -> int:
        return self.d

    def trace(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        axis = axis % coeffs.ndim
        coeffs.reshape(coeffs[:axis] + (self.d, self.d) + coeffs[axis+1:])
        return coeffs.trace(axis1=axis, axis2=axis+1)


@dataclasses.dataclass(unsafe_hash=True)
class MatrixTensorProductBasis[LabelT](TensorProductBasis[LabelT], MatrixBasis[tuple[LabelT, ...]]):
    """A specialization of TensorProductBasis to the case where all individual bases are MatrixBases, implements the .transpose method."""
    bases: tuple[MatrixBasis[LabelT], ...]

    def transpose(self, coeffs: np.ndarray, axis=-1) -> np.ndarray:
        axis = axis % coeffs.ndim
        origshape = coeffs.shape
        assert coeffs.shape[axis] == self.size(), f"Chosen axis size of coeffs must be equal to self.size()={self.size}, got shape={coeffs.shape}, with {axis=}"
        split = coeffs.reshape(coeffs.shape[:axis] + tuple(b.size() for b in self.bases) + coeffs.shape[axis+1:])
        for i, basis in zip(range(axis, axis+self.n), self.bases):
            split = basis.transpose(split, axis=i)
        return split.reshape(origshape)
    
    def trace[T: ArrayAPICompatible](self, coeffs: T, axis=-1) -> T|float:
        axis = axis % coeffs.ndim
        traced = coeffs.reshape(coeffs[:axis] + tuple(b.size() for b in self.bases) + coeffs[axis+1:])
        for basis in self.bases:
            traced = basis.trace(traced, axis=axis) # After trace the axis is removed, so we don't have to increment  # ty:ignore[invalid-argument-type]
        return traced
    
    def partial_trace(self, coeffs: np.ndarray, *system_indices: int, coeff_axis=-1):
        axis = coeff_axis
        axis = axis % coeffs.ndim
        traced = coeffs.reshape(coeffs[:axis] + tuple(b.size() for b in self.bases) + coeffs[axis + 1 :])
        for already_done, idx in enumerate(sorted(system_indices)):
            basis = self.bases[idx]
            traced = basis.trace(
                traced, axis=axis + idx - already_done
            )  # After trace the axis is removed
        return traced # ty:ignore[invalid-argument-type]

    @property
    def dimension(self) -> int:
        return math.prod(b.dimension for b in self.bases)

    def __str__(self):
        return " ⊗ ".join(str(b) for b in self.bases)
