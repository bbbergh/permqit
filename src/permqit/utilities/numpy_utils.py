import math
import typing
from typing import overload, Iterable, Sequence, Generator, Any

import numpy as np


# TODO: scipy.sparse is not really array API compatible. Investigate where we really need that and then replace with a different type for this
if typing.TYPE_CHECKING:
    import sparse
    import scipy.sparse
    try:
        import cupy as cp
        import cupyx
    except ImportError:
        type ArrayAPICompatible = np.ndarray | sparse.GCXS
        type DenseArray = np.ndarray
    else:
        type ArrayAPICompatible = np.ndarray | sparse.GCXS  | cp.ndarray | cupyx.scipy.sparse.csr_matrix
        type DenseArray = np.ndarray | cp.ndarray
else:
    type ArrayAPICompatible = typing.Any
    type DenseArray = typing.Any

def take_groups[T: Sequence|np.ndarray](arr: T, group_sizes: Iterable[int]) -> Generator[T]:
    """
    Takes groups of elements from an array, where the size of each group is given by group_sizes.
    """
    idx = 0
    for b in group_sizes:
        yield arr[idx:idx+b]
        idx += b


def contract_at_axis[T: ArrayAPICompatible](matrix: T, ndvector: T, matrix_axis: int = 0, vector_axis: int = -1, *, xp=np):
    """
    If matrix is an array of shape (*x, i, *y), (where is is at index 'matrix_axis') and ndvector is an array of shape (*a, i, *b), where i is at index 'vector_axis',
    returns an array of shape (*a, *x, *y, *b), where the matrix is contracted with vector at the respective indices. 
    Parameters
    ----------
    matrix
    ndvector
    vector_axis
    xp

    Returns
    -------

    """
    target_num_axes = len(matrix.shape) - 1
    vector_axis = vector_axis % ndvector.ndim

    return xp.moveaxis(
        xp.tensordot(ndvector, matrix, ([vector_axis], [matrix_axis])),  # ty:ignore[no-matching-overload]
        tuple(range(-target_num_axes, 0)),
        tuple(range(vector_axis, vector_axis + target_num_axes)),
    )


@overload
def sum_combinations(individual_arrays: Iterable[np.ndarray], *, length: int) -> np.ndarray: ...
@overload
def sum_combinations(individual_arrays: Sequence[np.ndarray], *, length: None = None) -> np.ndarray: ...

def sum_combinations(individual_arrays: Iterable[np.ndarray], *, length: int | None = None) -> np.ndarray:
    """
    Given a sequence of matrices A_1, ..., A_t, calculate all possible combinations of sums of these matrices.
    If each A_i is an array of shape (n, *s), returns an array B of shape ((n,)*t, *s) such that
    B[i_1, ..., i_t, :] = sum_{r = 1 to t} A_r[i_r, :]
    :param individual_arrays: the sequence A_1, ..., A_t
    :param length: the length of the sequence, must be provided if individual_arrays is an Iterable, otherwise len(individual_arrays) is used
    :return: The resulting numpy array of shape ((n,)*t, *s)
    """
    return apply_to_combinations(individual_arrays, sum, length=length)

@overload
def product_combinations(individual_arrays: Iterable[np.ndarray], *, length: int) -> np.ndarray: ...
@overload
def product_combinations(individual_arrays: Sequence[np.ndarray], *, length: None = None) -> np.ndarray: ...

def product_combinations(individual_arrays: Iterable[np.ndarray], *, length: int | None = None) -> np.ndarray:
    return apply_to_combinations(individual_arrays, math.prod, length=length)


@overload
def apply_to_combinations(individual_arrays: Iterable[np.ndarray], func=sum, *, length: int) -> np.ndarray: ...
@overload
def apply_to_combinations(individual_arrays: Sequence[np.ndarray], func=sum, *, length: None = None) -> np.ndarray: ...

def apply_to_combinations(individual_arrays: Iterable[np.ndarray], func=sum, *, length: int | None = None) -> np.ndarray:
    """
    Given a sequence of matrices A_1, ..., A_t, calculate all possible combinations of operations on a selection of t of these matrices.
    Asssuming each A_i is an array of shape (n, *s), returns an array B of shape ((n,)*t, *s) such that
    B[i_1, ..., i_t, :] = func_{r = 1 to t} A_r[i_r, :]
    :param individual_arrays: the sequence A_1, ..., A_t
    :param length: the length of the sequence, must be provided if individual_arrays is an Iterable, otherwise len(individual_arrays) is used
    :return: The resulting numpy array of shape ((n,)*t, *s)
    """
    t = length if length is not None else len(individual_arrays)
    return func(v.reshape([1] * i + [v.shape[0]] + (t - i - 1) * [1] + list(v.shape[1:])) for i, v in  # type: ignore
                enumerate(individual_arrays))


def multi_vector_kron(xp, *vectors: Any) -> np.ndarray:
    if not vectors:
        return xp.asarray([])
    ret = vectors[0]
    for v in vectors[1:]:
        ret = xp.kron(ret, v)
    return ret
