import numbers
from typing import Sequence, Self

import numpy as np
import scipy.linalg

__all__ = ["BlockDiagonalMatrix"]

import sparse


class BlockDiagonalMatrix:
    """
    Efficiently stores a block-diagonal matrix as a list of blocks, and implements basic operations on it.
    Storage:
    If block_sizes = (λ_1, ..., λ_k), then blocks is a list of numpy arrays, which are each either scalars (thought of as this scalar times the identity matrix), 1D entries (thought of as diagonal) or of size λ_i x λ_i,
    """
    blocks: list[np.ndarray]
    block_sizes: tuple[int, ...]

    def __init__(self, blocks: Sequence[np.ndarray]|np.ndarray, block_sizes:tuple[int, ...], *, validate=True):
        self.block_sizes = block_sizes
        self.blocks = list(blocks)
        if validate:
            self._validate()

    def _validate(self):
        if len(self.blocks) != len(self.block_sizes):
            raise ValueError(f"Number of blocks {len(self.blocks)} must match number of block sizes {len(self.block_sizes)}")

        for block, size in zip(self.blocks, self.block_sizes):
            match block.shape:
                case ():
                    continue
                case (n, ):
                    if n != size:
                        raise ValueError(f"1D block in BlockDiagonalMatrix {block} must have size {size}")
                case (d1, d2):
                    if not (d1 == d2 == size):
                        raise ValueError(f"2D block in BlockDiagonalMatrix {block} must have size {size}x{size}")
                case _:
                    raise ValueError(f"Block {block} of shape {block.shape} not supported")

    @property
    def total_size(self):
        return sum(self.block_sizes)

    def to_full_matrix(self) -> np.ndarray:
        return scipy.linalg.block_diag(*(self._block_to_matrix(block, size) for block, size in zip(self.blocks, self.block_sizes)))

    def _block_to_matrix(self, block: np.ndarray, size: int) -> np.ndarray:
        match block.shape:
            case ():
                return block * np.eye(size)
            case (n,):
                return np.diag(block)
            case (d1, d2):
                if isinstance(block, sparse.SparseArray):
                    return block.todense()
                return block
            case _:
                raise ValueError(f"Block {block} of shape {block.shape} not supported")

    __array_ufunc__ = None  # Prevent numpy from calling ufuncs on this object directly

    def __add__(self, other):
        if isinstance(other, BlockDiagonalMatrix):
            assert self.block_sizes == other.block_sizes, f"Cannot add BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {other.block_sizes}"
            return BlockDiagonalMatrix([sb + ob for sb, ob in zip(self.blocks, other.blocks)], self.block_sizes, validate=False)
        if isinstance(other, numbers.Number):
            return BlockDiagonalMatrix([b + other for b in self.blocks], self.block_sizes, validate=False)
        return self.to_full_matrix() + other

    def __sub__(self, other):
        if isinstance(other, BlockDiagonalMatrix):
            assert self.block_sizes == other.block_sizes, f"Cannot subtract BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {other.block_sizes}"
            return BlockDiagonalMatrix([sb - ob for sb, ob in zip(self.blocks, other.blocks)], self.block_sizes, validate=False)
        return self.to_full_matrix() + (-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return BlockDiagonalMatrix([other * b for b in self.blocks], self.block_sizes, validate=False)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return BlockDiagonalMatrix([b / other for b in self.blocks], self.block_sizes, validate=False)
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, numbers.Number):
            return BlockDiagonalMatrix([b // other for b in self.blocks], self.block_sizes, validate=False)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return BlockDiagonalMatrix([-b for b in self.blocks], self.block_sizes, validate=False)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other: Self|np.ndarray[tuple[int, ...], np.dtype]):
        """Right matrix multiplication. Currently, does not support broadcasting to more than 2-dimensional arrays."""
        if isinstance(other, BlockDiagonalMatrix):
            assert self.block_sizes == other.block_sizes, f"Cannot multiply BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {other.block_sizes}"
            return BlockDiagonalMatrix([self._right_multiply_block(sb, ob) for sb, ob in zip(self.blocks, other.blocks)], self.block_sizes, validate=False)
        if isinstance(other, np.ndarray):
            assert other.shape[0] == self.total_size, f"Cannot multiply BlockDiagonalMatrix of total size {self.total_size} with ndarray of shape {other.shape} "
            return np.concatenate([self._right_multiply_block(block, other[offset:offset + size]) for block, size, offset in zip(self.blocks, self.block_sizes, np.cumsum((0,) + self.block_sizes[:-1]))])
        return self.to_full_matrix() @ other

    def __rmatmul__(self, other: Self|np.ndarray[tuple[int, ...], np.dtype]):
        """Left matrix multiplication. Currently, does not support broadcasting to more than 2-dimensional arrays."""
        if isinstance(other, BlockDiagonalMatrix):
            assert self.block_sizes == other.block_sizes, f"Cannot multiply BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {other.block_sizes}"
            return BlockDiagonalMatrix([self._left_multiply_block(sb, ob) for sb, ob in zip(self.blocks, other.blocks)], self.block_sizes, validate=False)
        if isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                # Treat 1D array as row vector, result should be 1D
                result = other[np.newaxis, :] @ self
                return result.squeeze()
            assert other.shape[1] == self.total_size, f"Cannot multiply BlockDiagonalMatrix of total size {self.total_size} with ndarray of shape {other.shape} "
            return np.concatenate([self._left_multiply_block(block, other[:, offset:offset + size]) for block, size, offset in zip(self.blocks, self.block_sizes, np.cumsum((0,) + self.block_sizes[:-1]))], axis=1)
        return other @ self.to_full_matrix()

    def _right_multiply_block(self, block, other):
        """Returns block @ other"""
        match block.shape:
            case ():
                return block * other # This is scalar multiplication
            case (n, ):
                return block * other # This is broadcasted elementwise multiplication which works for diagonal matrices
            case (d1, d2):
                return block @ other
            case _:
                raise ValueError(f"Block {block} of shape {block.shape} not supported")

    def _left_multiply_block(self, block, other):
        """Returns other @ block"""
        match block.shape:
            case ():
                return block * other # This is scalar multiplication
            case (n, ):
                return other * block # This is broadcasted elementwise multiplication which works for diagonal matrices
            case (d1, d2):
                return other @ block
            case _:
                raise ValueError(f"Block {block} of shape {block.shape} not supported")

    def __pow__(self, power, modulo=None):
        if isinstance(power, BlockDiagonalMatrix) and (modulo is None or isinstance(modulo, BlockDiagonalMatrix)):
            assert self.block_sizes == power.block_sizes, f"Cannot raise BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {power.block_sizes}"
            if modulo is not None:
                assert self.block_sizes == modulo.block_sizes, f"Cannot raise BlockDiagonalMatrices with different block sizes: {self.block_sizes} and {modulo.block_sizes}"
                return BlockDiagonalMatrix([pow(*(self._block_to_matrix(k, s) for k in (b,p,m))) for b,p,m,s in zip(self.blocks, power.blocks, modulo.blocks, self.block_sizes)], self.block_sizes, validate=False)
            return BlockDiagonalMatrix([pow(self._block_to_matrix(b, s), self._block_to_matrix(p, s)) for b,p,s in zip(self.blocks, power.blocks, self.block_sizes)], self.block_sizes, validate=False)
        return pow(self.to_full_matrix(), power, modulo)


    @property
    def T(self):
        """Returns the transpose of this matrix."""
        # Only transpose the blocks which are not already diagonal
        return BlockDiagonalMatrix([b.T if len(b.shape) == 2 else b for b in self.blocks], self.block_sizes, validate=False)

    def conjugate(self):
        return BlockDiagonalMatrix([b.conjugate() for b in self.blocks], self.block_sizes, validate=False)

    def __str__(self):
        return " ⊕ \n".join(str(b) if b.shape == (size, size) else f'{str(b)}<{size}x{size}>'for b, size in zip(self.blocks, self.block_sizes))

    def __repr__(self):
        return f"BlockDiagonalMatrix(\n{str(self)})"

    def __hash__(self):
        return hash((self.block_sizes, tuple(b.data.tobytes() for b in self.blocks)))

    def __eq__(self, other):
        return (self.block_sizes == other.block_sizes) and (all(np.all(self_b == other_b) for self_b, other_b in zip(self.blocks, other.blocks)))

    def trace_with_multiplicities(self, multiplicities: np.ndarray|None = None):
        if multiplicities is None:
            multiplicities = np.ones(len(self.block_sizes))
        assert multiplicities.shape == (len(self.block_sizes),)
        ret = 0
        for block, size, mult in zip(self.blocks, self.block_sizes, multiplicities):
            match block.shape:
                case ():
                    ret += block * size * mult # If the block is a scalar times the identity
                case (n, ):
                    ret += np.sum(block) * mult # If the block is a diagonal
                case (d1, d2):
                    ret += np.trace(block) * mult

        return ret

    @classmethod
    def asnumpy(cls, arr: 'BlockDiagonalMatrix|np.ndarray'):
        if isinstance(arr, BlockDiagonalMatrix):
            return arr.to_full_matrix()
        return arr



