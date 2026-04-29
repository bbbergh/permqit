"""Linear maps that map coefficient vectors in one basis to coefficient vectors in another (or possibly the same) basis.

This module contains the abstract base classes for maps that transform
coefficient vectors from one basis to another, plus the infrastructure
for multi-format matrix storage.

Classes
-------
StorageFormat
    Enum of supported storage back-ends.
MatrixCache
    Holds one matrix in multiple formats; converts on demand or lazily caches.
LinearMap
    Abstract base; any map between two bases.
IndexMapping
    Isomorphisms that just remap indices 1-to-1.
TransitionMatrix
    Linear maps backed by a (sparse) transition matrix.
GivenTransitionMatrix
    Concrete TransitionMatrix whose matrix is supplied at construction time.
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Literal, Optional, Self, Set, overload

import numpy as np
import scipy
import scipy.sparse as sp
import sparse
from array_api_compat import array_namespace
from sparse import SparseArray

from ..utilities.numpy_utils import contract_at_axis, ArrayAPICompatible
from ..utilities import caching
from ..utilities.timing import ExpensiveComputation
from .basis import Basis

__all__ = [
    "StorageFormat",
    "MatrixCache",
    "LinearMap",
    "ScatterIndexMapping",
    "GatherIndexMapping",
    "ScatterMappingInverse",
    "GatherMappingInverse",
    "TransitionMatrix",
    "GivenTransitionMatrix",
]

# Numerical data can be in any of the supported storage formats:
#   NUMPY        → np.ndarray
#   SCIPY_SPARSE → sp.spmatrix  (csr_matrix, csc_matrix, …)
#   PYDATA_SPARSE → sparse.SparseArray  (GCXS, COO, DOK, …)
#   GPU / GPU_SPARSE → cupy.ndarray / cupyx.scipy.sparse matrix
type CoefficientData = ArrayAPICompatible


# ---------------------------------------------------------------------------
# Storage format registry
# ---------------------------------------------------------------------------


class StorageFormat(enum.Enum):
    """Supported matrix storage back-ends.

    Values
    ------
    NUMPY
        Dense ``numpy.ndarray`` on CPU.
    SCIPY_SPARSE
        ``scipy.sparse.csr_matrix`` on CPU.
    PYDATA_SPARSE
        ``sparse.GCXS`` (pydata/sparse) on CPU.
    GPU
        Dense ``cupy.ndarray`` on GPU (requires CuPy).
    GPU_SPARSE
        ``cupyx.scipy.sparse.csr_matrix`` on GPU (requires CuPy).
    """

    NUMPY = "numpy"
    SCIPY_SPARSE = "scipy_sparse"
    PYDATA_SPARSE = "pydata_sparse"
    GPU = "gpu"
    GPU_SPARSE = "gpu_sparse"

    def is_gpu(self) -> bool:
        return self in (StorageFormat.GPU, StorageFormat.GPU_SPARSE)
    
    @staticmethod
    def detect_format(mat) -> StorageFormat:
        type_name = type(mat).__module__ + "." + type(mat).__qualname__

        if "cupy" in type_name or "cupyx" in type_name:
            if sp.issparse(mat) or "sparse" in type_name:
                return StorageFormat.GPU_SPARSE
            return StorageFormat.GPU

        if isinstance(mat, np.ndarray):
            return StorageFormat.NUMPY

        if sp.issparse(mat):
            return StorageFormat.SCIPY_SPARSE

        # pydata/sparse: SparseArray or any sparse.* subclass
        if isinstance(mat, SparseArray):
            return StorageFormat.PYDATA_SPARSE

        raise TypeError(
            f"Cannot detect StorageFormat for type {type(mat)!r}. "
            "Pass a numpy array, scipy sparse matrix, pydata/sparse array, "
            "or a CuPy array/sparse matrix."
        )


class MatrixCache:
    """Multi-format cache for a single logical matrix.

    The matrix is stored internally as a dictionary mapping
    :class:`StorageFormat` → concrete matrix object.  At most one format is
    computed eagerly (the *primary* format, set at construction); all others
    are derived lazily on the first request.

    Whether a derived format is **retained** after conversion is controlled
    by ``cache_formats``.  If a requested format is not in ``cache_formats``
    the conversion result is returned but *not* stored – it will be
    recomputed on the next request.

    Parameters
    ----------
    matrix : array-like
        The initial matrix in any supported format.  Its format is detected
        automatically and used as the primary format.
    cache_formats : set of StorageFormat, optional
        Formats that should be retained after conversion.  Defaults to
        ``{primary_format}`` (only the primary format is cached).
        Pass ``set(StorageFormat)`` to cache everything eagerly.
    """

    @classmethod
    def as_cache(cls, matrix: CoefficientData | MatrixCache, cache_formats: Set[StorageFormat] | None = None):
        if isinstance(matrix, MatrixCache):
            return matrix
        return cls(matrix, cache_formats)

    def __init__(
        self,
        matrix,
        cache_formats: Set[StorageFormat] | None = None,
    ):
        primary_fmt = StorageFormat.detect_format(matrix)
        self._cache: dict[StorageFormat, object] = {primary_fmt: matrix}
        self._primary: StorageFormat = primary_fmt
        self._cache_formats: Set[StorageFormat] = cache_formats if cache_formats is not None else {primary_fmt}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def primary_format(self) -> StorageFormat:
        """The format in which the matrix was first stored."""
        return self._primary

    @property
    def cached_formats(self) -> set[StorageFormat]:
        """The set of formats currently held in cache."""
        return set(self._cache.keys())

    def get(self, fmt: StorageFormat|None=None) -> CoefficientData:
        """Return the matrix in the requested format.

        If not already cached, converts from the primary (or any available)
        format.  The result is stored only if ``fmt`` is in
        ``self.cache_formats``.
        """
        if fmt is None:
            fmt = self._primary
        if fmt in self._cache:
            return self._cache[fmt]  # ty:ignore[invalid-return-type]

        if fmt not in self._cache_formats:
            # Warn if that format is not cached.
            print(
                f"{type(self).__name__}.get(): "
                f"format {fmt.value!r} is (and will not be) cached; "
                "converting matrix to this format now (repeatedly doing this may be expensive).",
                # UserWarning,
                # stacklevel=1,
            )

        result = self._convert_to(fmt)
        if fmt in self._cache_formats:
            self._cache[fmt] = result
        return result

    def set_cache_formats(self, *formats: StorageFormat) -> Self:
        """Update which formats are retained after conversion."""
        new_formats = set(formats)
        if self._cache_formats == new_formats:
            return self
        self._cache_formats = new_formats
        assert self.cached_formats, "set_cache_formats cannot be empty"
        if self._primary not in self._cache_formats:  # Make sure we precompute at least one if we evict the primary
            self.precompute(formats[0])
            self._primary = formats[0]
        # Evict formats that are no longer wanted
        to_evict = [f for f in self._cache if f not in self._cache_formats]
        for f in to_evict:
            del self._cache[f]

        return self

    def evict(self, fmt: StorageFormat) -> None:
        """Remove a cached format (the primary format cannot be evicted)."""
        if fmt == self._primary:
            raise ValueError("Cannot evict the primary format.")
        self._cache.pop(fmt, None)

    def precompute(self, *formats: StorageFormat) -> None:
        """Eagerly convert and cache the requested formats."""
        for fmt in formats:
            if fmt not in self._cache:
                result = self._convert_to(fmt)
                self._cache[fmt] = result
                self._cache_formats.add(fmt)

    # ------------------------------------------------------------------
    # Convenience accessors (mirror the old API surface)
    # ------------------------------------------------------------------

    def as_numpy(self) -> np.ndarray:
        return self.get(StorageFormat.NUMPY)  # ty:ignore[invalid-return-type]

    def as_scipy_sparse(self) -> sp.csr_matrix:
        return self.get(StorageFormat.SCIPY_SPARSE)  # ty:ignore[invalid-return-type]

    def as_pydata_sparse(self) -> SparseArray:
        return self.get(StorageFormat.PYDATA_SPARSE)  # ty:ignore[invalid-return-type]

    def as_gpu(self):
        """Return a dense CuPy array (requires CuPy)."""
        return self.get(StorageFormat.GPU)

    def as_gpu_sparse(self):
        """Return a cupyx.scipy.sparse CSR matrix (requires CuPy)."""
        return self.get(StorageFormat.GPU_SPARSE)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Format auto-detection helper
    # ------------------------------------------------------------------

    def _best_format_for(self, x) -> StorageFormat:
        """Choose the optimal :class:`StorageFormat` for a matrix-vector product
        given the type of the coefficient vector ``x``.

        * CuPy dense/sparse arrays → :attr:`StorageFormat.GPU_SPARSE`
        * SciPy sparse matrices    → :attr:`StorageFormat.SCIPY_SPARSE`
        * pydata/sparse arrays     → :attr:`StorageFormat.PYDATA_SPARSE`
        * Everything else          → :attr:`StorageFormat.PYDATA_SPARSE`
          (pydata/sparse tensordot is the cheapest CPU path)
        """
        type_name = type(x).__module__ + "." + type(x).__qualname__
        if "cupy" in type_name or "cupyx" in type_name:
            return StorageFormat.GPU_SPARSE
        if sp.issparse(x):
            return StorageFormat.SCIPY_SPARSE
        if isinstance(x, SparseArray):
            return StorageFormat.PYDATA_SPARSE
        return self.first_format_available(StorageFormat.SCIPY_SPARSE, StorageFormat.PYDATA_SPARSE, StorageFormat.NUMPY)

    def first_format_available(self, *formats: StorageFormat) -> StorageFormat:
        """Return the first format in ``formats`` that is available."""
        for fmt in formats:
            if fmt in self._cache or fmt in self._cache_formats:
                return fmt
        return StorageFormat.NUMPY



    # ------------------------------------------------------------------
    # Conversion dispatch
    # ------------------------------------------------------------------

    def _convert_to(self, fmt: StorageFormat):
        """Convert from any available cached format to ``fmt``."""
        # Pick a convenient source; prefer the primary
        src_fmt = self._primary
        src = self._cache[src_fmt]
        return self._do_convert(src, src_fmt, fmt)

    def _do_convert(self, src, src_fmt: StorageFormat, dst_fmt: StorageFormat):
        if src_fmt == dst_fmt:
            return src

        # ---- CPU conversions ----------------------------------------
        if src_fmt == StorageFormat.NUMPY:
            if dst_fmt == StorageFormat.SCIPY_SPARSE:
                return sp.csr_matrix(src)
            if dst_fmt == StorageFormat.PYDATA_SPARSE:
                return sparse.GCXS.from_numpy(src)
            if dst_fmt == StorageFormat.GPU:
                return self._to_cupy_dense(src)
            if dst_fmt == StorageFormat.GPU_SPARSE:
                return self._to_cupy_sparse(sp.csr_matrix(src))

        if src_fmt == StorageFormat.SCIPY_SPARSE:
            if dst_fmt == StorageFormat.NUMPY:
                return np.asarray(src.todense())
            if dst_fmt == StorageFormat.PYDATA_SPARSE:
                return sparse.GCXS.from_scipy_sparse(src)
            if dst_fmt == StorageFormat.GPU:
                return self._to_cupy_dense(np.asarray(src.todense()))
            if dst_fmt == StorageFormat.GPU_SPARSE:
                return self._to_cupy_sparse(src)

        if src_fmt == StorageFormat.PYDATA_SPARSE:
            if dst_fmt == StorageFormat.NUMPY:
                return src.todense()
            if dst_fmt == StorageFormat.SCIPY_SPARSE:
                return src.to_scipy_sparse().tocsr()
            if dst_fmt == StorageFormat.GPU:
                return self._to_cupy_dense(src.todense())
            if dst_fmt == StorageFormat.GPU_SPARSE:
                return self._to_cupy_sparse(src.to_scipy_sparse().tocsr())

        # ---- GPU → CPU conversions ----------------------------------
        if src_fmt == StorageFormat.GPU:
            import cupy as cp  # ty:ignore[unresolved-import]

            cpu = cp.asnumpy(src)
            if dst_fmt == StorageFormat.NUMPY:
                return cpu
            if dst_fmt == StorageFormat.SCIPY_SPARSE:
                return sp.csr_matrix(cpu)
            if dst_fmt == StorageFormat.PYDATA_SPARSE:
                return sparse.GCXS.from_numpy(cpu)
            if dst_fmt == StorageFormat.GPU_SPARSE:
                return self._to_cupy_sparse(sp.csr_matrix(cpu))

        if src_fmt == StorageFormat.GPU_SPARSE:
            import cupy as cp  # ty:ignore[unresolved-import]

            cpu_sparse = src.get()  # cupyx → scipy.sparse
            if dst_fmt == StorageFormat.NUMPY:
                return np.asarray(cpu_sparse.todense())
            if dst_fmt == StorageFormat.SCIPY_SPARSE:
                return cpu_sparse.tocsr()
            if dst_fmt == StorageFormat.PYDATA_SPARSE:
                return sparse.GCXS.from_scipy_sparse(cpu_sparse.tocsr())
            if dst_fmt == StorageFormat.GPU:
                return self._to_cupy_dense(np.asarray(cpu_sparse.todense()))

        raise NotImplementedError(f"No conversion path from {src_fmt!r} to {dst_fmt!r}.")

    # ------------------------------------------------------------------
    # CuPy helpers (imported lazily so CuPy is an optional dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_supported_gpu_sparse_dtype(dtype: np.dtype) -> bool:
        """Check if a dtype is supported on GPU (CuPy).

        Only bool, float32, float64, complex64, and complex128 are supported.
        """
        supported_dtypes = {
            np.dtype(bool),
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.complex64),
            np.dtype(np.complex128),
        }
        return np.dtype(dtype) in supported_dtypes

    @staticmethod
    def _convert_to_supported_gpu_sparse_dtype(arr: np.ndarray) -> np.ndarray:
        """Convert array to a GPU-supported dtype if needed.

        Conversion rules:
        - bool, float32, float64, complex64, complex128 → no change
        - other int types → float64
        - other float types → float64
        - other complex types → complex128
        - other dtypes → float64 (fallback)
        """
        if MatrixCache._is_supported_gpu_sparse_dtype(arr.dtype):
            return arr

        # Determine the conversion target based on the array's dtype kind
        dtype_kind = arr.dtype.kind
        if dtype_kind in ("i", "u"):  # signed/unsigned integer
            return arr.astype(np.float64)
        elif dtype_kind == "f":  # floating point
            return arr.astype(np.float64)
        elif dtype_kind == "c":  # complex
            return arr.astype(np.complex128)
        else:
            # Fallback for other kinds (object, string, etc.)
            return arr.astype(np.float64)

    @staticmethod
    def _to_cupy_dense(cpu_array: np.ndarray):
        """Convert a numpy array to a CuPy dense array, ensuring GPU-supported dtype.

        If the array has an unsupported dtype, it is converted to a supported dtype
        before being moved to GPU.
        """
        try:
            import cupy as cp  # ty:ignore[unresolved-import]
        except ImportError as e:
            raise ImportError("CuPy is required for GPU storage formats.") from e

        return cp.asarray(cpu_array)

    @staticmethod
    def _to_cupy_sparse(scipy_csr: sp.csr_matrix):
        """Convert a scipy sparse matrix to a CuPy sparse matrix, ensuring GPU-supported dtype.

        If the matrix has an unsupported dtype, it is converted to a supported dtype
        before being moved to GPU.
        """
        try:
            import cupyx.scipy.sparse as csp  # ty:ignore[unresolved-import]
        except ImportError as e:
            raise ImportError("CuPy/cupyx is required for GPU_SPARSE storage.") from e

        # Ensure dtype is cusparse-compatible
        if not MatrixCache._is_supported_gpu_sparse_dtype(scipy_csr.dtype):
            scipy_csr = scipy_csr.astype(MatrixCache._convert_to_supported_gpu_sparse_dtype(scipy_csr.data).dtype)

        return csp.csr_matrix(scipy_csr)

    def __repr__(self) -> str:
        cached = ", ".join(f.value for f in self._cache)
        return (
            f"MatrixCache(primary={self._primary.value}, "
            f"cached=[{cached}], "
            f"will_cache=[{', '.join(f.value for f in self._cache_formats)}])"
        )


# ---------------------------------------------------------------------------
# Linear map base classes
# ---------------------------------------------------------------------------


class LinearMap(abc.ABC):
    """Abstract base class for maps that transform coefficient vectors from one basis to another."""

    basis_from: Basis
    basis_to: Basis

    @abc.abstractmethod
    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional["StorageFormat"] = None,
    ) -> T:
        """Apply the map to a coefficient vector ``x``.

        An element with coefficients ``x`` in ``self.basis_from`` is mapped to
        an element in ``self.basis_to`` whose coefficients are returned.

        :param axis: Axis along which the coefficients live.
        :param format: Preferred :class:`StorageFormat` for any internal matrix
            used during the computation.  When ``None`` (the default) the
            format is auto-detected from the type of ``x``.  A
            :class:`UserWarning` is emitted if the chosen format is not already
            cached and must therefore be recomputed.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.apply_to_coefficient_vector(*args, **kwargs)


class ScatterIndexMapping(LinearMap):
    """Abstract base class for maps that map basis indices to each other, and the index mapping is stored in 'scatter' format,
    i.e. we store an array of shape (basis_from.size(),) where the i-th entry is the index in basis_to that basis_from[i] maps to."""

    _index_mapping: MatrixCache | None = None
    """Represents the index mapping as a numpy array of shape (basis_from.size(),) where the i-th entry is the index in basis_to that basis_from[i] maps to. """

    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional[StorageFormat] = None,
    ) -> T:
        n_to = self.basis_to.size()

        # Normalise negative axes.
        ndim = len(x.shape)
        axis = axis % ndim

        # Build out_shape: size along `axis` becomes n_to.
        out_shape = list(x.shape)
        out_shape[axis] = n_to
        out_shape = tuple(out_shape)

        # --- numpy dense path -------------------------------------------
        if isinstance(x, np.ndarray):
            perm = self.index_mapping().as_numpy()
            x_moved = np.moveaxis(x, axis, 0)
            out = np.zeros((n_to,) + x_moved.shape[1:], dtype=x.dtype)
            out[perm] = x_moved
            return np.moveaxis(out, 0, axis)  # ty:ignore[invalid-return-type]

        # --- pydata/sparse path: remap COO coordinates -------------------
        if isinstance(x, SparseArray):
            perm = self.index_mapping().as_numpy()
            coo = sparse.COO(x)
            new_coords = coo.coords.copy()
            new_coords[axis] = perm[coo.coords[axis]]
            return sparse.COO(new_coords, coo.data, shape=out_shape).asformat('gcxs')

        # --- scipy.sparse path (always 2-D) ------------------------------
        if sp.issparse(x):
            perm = self.index_mapping().as_numpy()
            if axis not in (0, 1):
                raise ValueError(f"scipy.sparse matrices are 2-D; axis must be 0 or 1, got {axis}.")
            coo = x.tocoo()  # ty:ignore[unresolved-attribute]
            new_row = perm[coo.row] if axis == 0 else coo.row
            new_col = perm[coo.col] if axis == 1 else coo.col
            return sp.coo_matrix(
                (coo.data, (new_row, new_col)),
                shape=out_shape,
            ).tocsr()  # ty:ignore[invalid-return-type]

        # --- fallback: delegate to __array_namespace__ (e.g. CuPy) ------
        fmt = StorageFormat.detect_format(x)
        perm = self.index_mapping().get(StorageFormat.NUMPY if fmt.is_gpu() else StorageFormat.GPU)
        xp = array_namespace(x)
        x_moved = xp.moveaxis(x, axis, 0)
        out = xp.zeros((n_to,) + x_moved.shape[1:], dtype=x.dtype)  # ty:ignore[unresolved-attribute]
        out[perm] = x_moved
        return xp.moveaxis(out, 0, axis)

    def index_mapping(self) -> MatrixCache:
        if self._index_mapping is None:
            with ExpensiveComputation(f"Calculation of {str(self)} index mapping"):
                self._index_mapping = MatrixCache.as_cache(self._calculate_index_mapping())
        return self._index_mapping

    @abc.abstractmethod
    def _calculate_index_mapping(self) -> MatrixCache | CoefficientData:
        pass

    def inverse(self) -> GatherIndexMapping:
        return ScatterMappingInverse(self)

    @caching.cache_noargs
    def as_transition_matrix(self) -> "GivenTransitionMatrix":
        """
        matrix = sparse.COO(
            np.stack([self.index_mapping().as_numpy(), np.arange(self.basis_from.size())]),
            np.ones(self.basis_to.size()),
            shape=(self.basis_to.size(), self.basis_from.size())
        ).asformat("gcxs")
        """
        matrix = scipy.sparse.coo_matrix((
                np.ones(self.basis_from.size()), # The entries
                [self.index_mapping().as_numpy(), np.arange(self.basis_from.size())] # The indices
            ), shape=(self.basis_to.size(), self.basis_from.size())
        ).tocsr()

        return GivenTransitionMatrix(self.basis_from, self.basis_to, matrix)

    def __str__(self):
        return f"{type(self).__name__}({self.basis_from!s} ↪ {self.basis_to!s})"

    def concatenate(self, *others: ScatterIndexMapping) -> ScatterIndexMapping:
        return ConcatenatedScatterIndexMapping(self, *others)

class ConcatenatedScatterIndexMapping(ScatterIndexMapping):
    def __init__(self, *maps: ScatterIndexMapping):
        self.maps = maps
        assert maps, "ConcatenatedScatterIndexMapping must have at least one map."

    def _calculate_index_mapping(self) -> MatrixCache | CoefficientData:
        map = self.maps[0].index_mapping().get()
        for newmap in self.maps[1:]:
            map = newmap.index_mapping().get()[map]
        return map

class GivenScatterIndexMapping(ScatterIndexMapping):
    """Implements a SurjectiveIndexMapping with a given index matrix"""

    def __init__(self, index_mapping: MatrixCache|CoefficientData, basis_from: Basis, basis_to: Basis):
        index_mapping = MatrixCache.as_cache(index_mapping)
        assert len(index_mapping.get()) == basis_from.size(), (  # ty:ignore[invalid-argument-type]
            f"Index mapping must have length {basis_from.size()}, "
            f"but has length {len(index_mapping.get())}."  # ty:ignore[invalid-argument-type]
        )
        self._index_mapping = index_mapping
        self.basis_from = basis_from
        self.basis_to = basis_to

    def _calculate_index_mapping(self):
        raise RuntimeError("This should never be called on GivenScatterIndexMapping.")



class GatherIndexMapping(LinearMap):
    """Abstract base class for map that maps basis indices to each other, and the index mapping is stored in 'gather' format,
    i.e. we store an array of shape (basis_to.size(),) where the i-th entry is the index in basis_from that maps to basis_to[i]. """

    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional[StorageFormat] = None,
    ) -> T:

        # --- scipy.sparse path (always 2-D) ------------------------------
        if sp.issparse(x):
            perm = self.index_mapping().as_numpy()  # shape (basis_to.size(),): out[i] = x[perm[i]]

            csr = x.tocsr()  # ty:ignore[unresolved-attribute]
            if axis == 0:
                return csr[perm, :]
            else:  # axis == 1
                return csr[:, perm].tocsr()

        # --- array api offers .take method -------------------------------------------
        perm = self.index_mapping().get(StorageFormat.GPU if StorageFormat.detect_format(x).is_gpu() else StorageFormat.NUMPY)
        return array_namespace(x).take(x, perm, axis=axis)

    _index_mapping: MatrixCache | None = None
    """Represents the index mapping as an array of shape (basis_to.size(),) where the i-th entry is the index in basis_from that maps to basis_to[i]. """

    def index_mapping(self) -> MatrixCache:
        """Represents the index mapping as an array of shape (basis_to.size(),) where the i-th entry is the index in basis_from that maps to basis_to[i]."""
        if self._index_mapping is None:
            with ExpensiveComputation(f"Calculation of {str(self)} index mapping"):
                self._index_mapping = MatrixCache.as_cache(self._calculate_index_mapping())
        return self._index_mapping

    @abc.abstractmethod
    def _calculate_index_mapping(self) -> np.ndarray | MatrixCache:
        pass

    def inverse(self) -> ScatterIndexMapping:
        return GatherMappingInverse(self)

    @caching.cache_noargs
    def as_transition_matrix(self) -> "GivenTransitionMatrix":
        matrix = scipy.sparse.coo_matrix((
                np.ones(self.basis_to.size()), # The entries
                [np.arange(self.basis_to.size()), self.index_mapping().as_numpy()] # The indices
            ), shape=(self.basis_to.size(), self.basis_from.size())
        ).tocsr()
        return GivenTransitionMatrix(self.basis_from, self.basis_to, matrix)


class GivenGatherIndexMapping(GatherIndexMapping):
    """Implements a SurjectiveIndexMapping with a given index matrix"""

    def __init__(self, index_mapping: MatrixCache, basis_from: Basis, basis_to: Basis):
        assert len(index_mapping.get()) == basis_to.size(), (  # ty:ignore[invalid-argument-type]
            f"Index mapping must have length {basis_to.size()}, "
            f"but has length {len(index_mapping.get())}."  # ty:ignore[invalid-argument-type]
        )
        self._index_mapping = index_mapping
        self.basis_from = basis_from
        self.basis_to = basis_to

    def _calculate_index_mapping(self):
        raise RuntimeError("This should never be called on GivenGatherIndexMapping.")


class ScatterMappingInverse(GatherIndexMapping, metaclass=caching.WeakRefMemoize):
    """
    Represents the inverse of an injective scatter mapping on its image. I.e. it takes the parts of the array that lie in the image of the original injective mapping
    and reverts the mapping on these. If the given scatter mapping is not injective, the highest index element of the preimage is used for every input.
    """

    def __init__(self, mapping: ScatterIndexMapping):
        self.mapping = mapping

    def inverse(self) -> "ScatterIndexMapping":
        return self.mapping

    @property
    def basis_from(self) -> Basis:
        return self.mapping.basis_to

    @property
    def basis_to(self) -> Basis:
        return self.mapping.basis_from

    def _calculate_index_mapping(self) -> MatrixCache:
        # The forward injective map scatters: out[perm[i]] = x[i].
        # The inverse is a gather: out2[i] = y[perm[i]], so the surjective
        # index mapping for the inverse is just the original forward perm.
        return self.mapping.index_mapping()


class GatherMappingInverse(ScatterIndexMapping, metaclass=caching.WeakRefMemoize):
    """
    Represents an inverse of a gather mapping that is also injective (gather mappings are surjective by definition).
    If the given gather mapping is not injective, the highest index element of the preimage is used for every input.
    """

    def __init__(self, mapping: GatherIndexMapping):
        self.mapping = mapping

    def inverse(self) -> "GatherIndexMapping":
        return self.mapping

    @property
    def basis_from(self) -> Basis:
        return self.mapping.basis_to

    @property
    def basis_to(self) -> Basis:
        return self.mapping.basis_from

    def _calculate_index_mapping(self) -> MatrixCache:
        # The forward surjective map gathers: out[i] = x[perm[i]].
        # The inverse is an injective scatter: out2[perm[i]] = y[i],
        # so the injective index mapping for the inverse is perm itself.
        return self.mapping.index_mapping()


class TransitionMatrix(LinearMap):
    """Abstract base class for maps backed by a precomputed transition matrix.

    The matrix is stored inside a :class:`MatrixCache`, which allows it to be
    held simultaneously in multiple storage formats (numpy, scipy_sparse,
    pydata_sparse, gpu, gpu_sparse) and converted between them on demand.

    Subclasses must implement :meth:`_calculate_transition_matrix`, which
    should return the matrix in whatever format is cheapest to compute.  The
    returned matrix is wrapped in a ``MatrixCache`` automatically.

    Parameters
    ----------
    default_cache_formats : set of StorageFormat, optional
        Override which formats are retained between calls.  Can also be
        changed at runtime via ``self.matrix_cache.set_cache_formats(...)``.
    """

    _matrix_cache: MatrixCache | None = None

    # Subclasses can override to precompute / persist additional formats.
    default_cache_formats: Set[StorageFormat] | None = None

    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional[StorageFormat] = None,
    ) -> T:
        # Determine the best matrix format for this computation.
        best_fmt = format if format is not None else self.matrix_cache._best_format_for(x)
        mat = self.coefficient_transition_matrix(best_fmt)

        if best_fmt in [StorageFormat.SCIPY_SPARSE, StorageFormat.GPU_SPARSE]:
            # These ones don't have tensordot
            xp = array_namespace(x)
            x_reshaped = xp.moveaxis(x, axis, 0)
            x_2d = x_reshaped.reshape(x_reshaped.shape[0], -1)
            result_2d = mat @ x_2d
            result_reshaped = result_2d.reshape((result_2d.shape[0],) + x_reshaped.shape[1:])
            return xp.moveaxis(result_reshaped, 0, axis)

        xp = array_namespace(mat)
        return contract_at_axis(mat, x, 1, axis, xp=xp)

    # ------------------------------------------------------------------
    # Matrix access
    # ------------------------------------------------------------------

    @overload
    def coefficient_transition_matrix(self, fmt: Literal[StorageFormat.NUMPY]) -> np.ndarray: ...
    @overload
    def coefficient_transition_matrix(self, fmt: Literal[StorageFormat.SCIPY_SPARSE]) -> sp.csr_matrix: ...
    @overload
    def coefficient_transition_matrix(self, fmt: Literal[StorageFormat.PYDATA_SPARSE]) -> SparseArray: ...
    @overload
    def coefficient_transition_matrix(self, fmt: Literal[StorageFormat.GPU]) -> Any: ...
    @overload
    def coefficient_transition_matrix(self, fmt: Literal[StorageFormat.GPU_SPARSE]) -> Any: ...
    @overload
    def coefficient_transition_matrix(
        self, fmt: StorageFormat=StorageFormat.PYDATA_SPARSE
    ) -> np.ndarray | sp.csr_matrix | SparseArray | Any: ...

    def coefficient_transition_matrix(self, fmt: StorageFormat = StorageFormat.PYDATA_SPARSE):
        """Return the transition matrix in the requested format.

        The first call triggers computation via :meth:`_calculate_transition_matrix`.
        Subsequent calls for already-cached formats are O(1).

        :param fmt: Desired :class:`StorageFormat`.  Defaults to
            ``PYDATA_SPARSE`` (the cheapest format for tensordot via pydata/sparse).
        """
        return self.matrix_cache.get(fmt)

    @property
    def matrix_cache(self) -> MatrixCache:
        """Direct access to the underlying :class:`MatrixCache` (``None`` if not yet computed)."""
        if self._matrix_cache is None:
            with ExpensiveComputation(f"Calculation of {str(self)} transition matrix"):
                raw = self._calculate_transition_matrix()
            self._matrix_cache = MatrixCache(raw, cache_formats=self.default_cache_formats)
        return self._matrix_cache

    @abc.abstractmethod
    def _calculate_transition_matrix(self):
        """Compute the transition matrix from basis_from to basis_to.

        Must return a matrix of shape ``(basis_to.size(), basis_from.size())``
        in any supported format (numpy, scipy sparse, pydata/sparse, CuPy …).
        The result is wrapped in a :class:`MatrixCache` automatically.
        """
        pass

    # ------------------------------------------------------------------
    # Convenience format accessors (mirrors old API + extends it)
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return transition matrix as a dense numpy array."""
        return self.coefficient_transition_matrix(StorageFormat.NUMPY)

    def to_scipy_sparse(self) -> sp.csr_matrix:
        """Return transition matrix as a scipy CSR matrix."""
        return self.coefficient_transition_matrix(StorageFormat.SCIPY_SPARSE)

    def to_pydata_sparse(self) -> SparseArray:
        """Return transition matrix as a pydata/sparse GCXS array."""
        return self.coefficient_transition_matrix(StorageFormat.PYDATA_SPARSE)

    def to_gpu(self):
        """Return transition matrix as a dense CuPy array (requires CuPy)."""
        return self.coefficient_transition_matrix(StorageFormat.GPU)

    def to_gpu_sparse(self):
        """Return transition matrix as a cupyx.scipy.sparse CSR matrix (requires CuPy)."""
        return self.coefficient_transition_matrix(StorageFormat.GPU_SPARSE)

class Identity(TransitionMatrix, metaclass=caching.WeakRefMemoize):
    def __init__(self, basis_from: Basis, basis_to: Basis):
        self.basis_from = basis_from
        self.basis_to = basis_to
        assert self.basis_from.size() == self.basis_to.size()

    def apply_to_coefficient_vector[T: CoefficientData](
        self,
        x: T,
        *,
        axis: int = 0,
        format: Optional[StorageFormat] = None,
    ) -> T:
        return x

    def _calculate_transition_matrix(self):
        return scipy.sparse.eye_array(self.basis_from.size())

    def inverse(self):
        return Identity(self.basis_to, self.basis_from)

    def __str__(self):
        return f"Identity({self.basis_from!s} -> {self.basis_to!s})"

class GivenTransitionMatrix(TransitionMatrix):
    """A :class:`TransitionMatrix` whose matrix is provided directly at construction."""

    def __init__(
        self,
        basis_from: Basis,
        basis_to: Basis,
        transition_matrix,
        cache_formats: Set[StorageFormat] | None = None,
    ):
        self.basis_from = basis_from
        self.basis_to = basis_to
        self._matrix_cache = MatrixCache.as_cache(transition_matrix, cache_formats=cache_formats)

    def _calculate_transition_matrix(self):
        raise RuntimeError("This should never be called on GivenTransitionMatrix.")
