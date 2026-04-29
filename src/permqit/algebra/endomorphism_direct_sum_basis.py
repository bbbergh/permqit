"""
The naming of this file is terrible, but I can't think of something better.
This file is about bases for Matrix-Algebras which are strict subalgebras of the full matrix algebra,
and hence can be written as a direct sum of full matrix algebras.
"""
import dataclasses
import math
from functools import cached_property
from typing import cast, Sequence, Self

import numpy as np

from .basis import MatrixBasis, Basis, MatrixTensorProductBasis
from .basis_subset import BasisSubset, MatrixBasisSubset, MatrixStandardBasisSubset, OrbitFitsMaskPredicate, IndexIsValidPredicate
from .matrix import BlockDiagonalMatrix
from .linear_map import ScatterIndexMapping, MatrixCache
from ..representation.combinatorics import weak_compositions
from ..representation.orbits import PairOrbit
from .endomorphism_basis import (
    MatrixDirectSumBasis,
    MatrixStandardBasis,
    EndSnOrbitBasis,
    EndSnBlockDiagonalBasis,
    EndSnOrbitBasisSubset,
)
from ..utilities import caching

__all__ = ["EndSnBlockOrbitBasis", "EmbeddingIntoFullOrbit"]

type OrbitBlockLike = EndSnSingleBlockOrbitBasis|EndSnOrbitBasis|EndSnOrbitBasisSubset|MatrixStandardBasis|MatrixStandardBasisSubset|MatrixTensorProductBasis

@dataclasses.dataclass(init=False)
class EndSnSingleBlockOrbitBasis(MatrixTensorProductBasis[PairOrbit]):
    """
    Represents a single block in the *-algebra of the RHS of the *-algebra isomorphism End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n) ≅ ⊕_μ ⊗_{i = 1}^t End^{S_(μ_i)}((ℂ^{p_i x p_i})^{μ_i})
    Supports each single-copy block being sparse (i.e. a masked subset of ℂ^{p_i x p_i}).
    """

    bases: tuple[EndSnOrbitBasis | EndSnOrbitBasisSubset, ...]
    original_blocks: tuple[MatrixStandardBasis | MatrixStandardBasisSubset, ...]
    multiplicities: Sequence[int]

    def __init__(
        self,
        single_copy_blocks: tuple[MatrixStandardBasis | MatrixStandardBasisSubset, ...],
        multiplicities: Sequence[int],
    ):
        from ..algebra.basis_subset import OrbitFitsMaskPredicate
        for block in single_copy_blocks:
            assert isinstance(block, MatrixStandardBasis|MatrixStandardBasisSubset), f"Expected MatrixStandardBasis or MatrixStandardBasisSubset, got {block}"

        self.original_blocks = single_copy_blocks
        self.multiplicities = multiplicities
        masks = [b.mask if isinstance(b, MatrixStandardBasisSubset) else None for b in single_copy_blocks]

        # need to convert to int to avoid np.int_, which doesn't memoize correctly since it's different from the integer
        full_bases = [EndSnOrbitBasis(int(mi), b.dimension) for mi, b in zip(multiplicities, single_copy_blocks)]
        bases = tuple(
            EndSnOrbitBasisSubset(orbitb, OrbitFitsMaskPredicate(mask))  # ty:ignore[invalid-argument-type]
            if isinstance(b, MatrixStandardBasisSubset)
            else orbitb
            for orbitb, b, mask in zip(full_bases, single_copy_blocks, masks)
        )
        super().__init__(bases=bases)

    @classmethod
    def of(cls, b: OrbitBlockLike) -> "EndSnSingleBlockOrbitBasis":
        if isinstance(b, EndSnSingleBlockOrbitBasis):
            return b
        if isinstance(b, MatrixStandardBasis|MatrixStandardBasisSubset|EndSnOrbitBasis|EndSnOrbitBasisSubset):
            bp, mult = cls._args_for_part(b)
            return EndSnSingleBlockOrbitBasis((bp,), (mult,))
        elif isinstance(b, MatrixTensorProductBasis):
            single_copy_bases, multiplicities = zip(*(cls._args_for_part(bp) for bp in b.bases))
            return EndSnSingleBlockOrbitBasis(single_copy_bases, multiplicities)
        raise ValueError(f"Unsupported basis: {b}")

    @classmethod
    def _args_for_part(cls, b: OrbitBlockLike) -> tuple[MatrixStandardBasis|MatrixStandardBasisSubset, int]:
        if isinstance(b, MatrixStandardBasis|MatrixStandardBasisSubset):
            return b, 1
        if isinstance(b, EndSnOrbitBasis):
            return MatrixStandardBasis(b.d), b.n
        if isinstance(b, EndSnOrbitBasisSubset):
            assert isinstance(b.predicate_instance, OrbitFitsMaskPredicate), "Auto-conversion only supported for EndSnBlockOrbitBasisSubset with OrbitFitsMaskPredicate"
            return MatrixStandardBasisSubset(MatrixStandardBasis(b.dimension), IndexIsValidPredicate(b.predicate_instance.mask)), b.n
        raise ValueError(f"Unsupported basis: {b}")


    @cached_property
    def subset_of(self) -> Self:
        if any(isinstance(b, EndSnOrbitBasisSubset) for b in self.bases):
            return type(self)(tuple(b.subset_of if isinstance(b, MatrixStandardBasisSubset) else b for b in self.original_blocks), self.multiplicities)
        return self

    def __str__(self):
        return f"EndSnSingleBlockOrbitBasis({' ⨂ '.join(str(b) + f'^{n}' for b,n in zip(self.original_blocks, self.multiplicities))})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__() # This hashes based on self.bases which is sufficient

    @classmethod
    def __process_init_args_for_cache_key__(cls, **kwargs):
        if "multiplicities" in kwargs:
            kwargs["multiplicities"] = tuple(kwargs["multiplicities"])
        return kwargs

class _BaseEndSnBlockOrbitBasis(MatrixDirectSumBasis[tuple[PairOrbit, ...]]):
    n: int  # The n in S_n
    t: int  # The number of blocks
    bases: tuple[EndSnSingleBlockOrbitBasis, ...]  # This are the bases within MatrixDirectSumBasis

    compositions_for_blocks: list[
        np.ndarray
    ]  # The compositions of n into t parts, in the same order as the bases. We need this to be able to map from labels to the correct subbasis index.
    original_blocks: tuple[
        MatrixStandardBasis | MatrixBasisSubset, ...
    ]  # The different blocks in the direct sum ⊕_i^t(ℂ^{p_i x p_i})^n (see below for how this is used)

    def _label_to_subbasis_index(self, label: tuple[PairOrbit, ...]) -> int:
        return self.composition_to_subbasis_index(np.fromiter((o.n for o in label), dtype=np.int_))

    def composition_to_subbasis_index(self, composition: np.ndarray) -> int:
        for i, c in enumerate(self.compositions_for_blocks):
            if np.array_equal(c, composition):
                return i
        raise ValueError(f"Composition {composition} not found in {self.compositions_for_blocks}")

    def coefficients_for_tensor_product(self, matrix: BlockDiagonalMatrix) -> np.ndarray:
        """For a given matrix A, given as a BlockDiagonalMatrix, returns the coefficients of the matrix A^n (i.e. the n-fold tensor product of A with itself) in this basis."""
        assert matrix.block_sizes == tuple(b.dimension for b in self.original_blocks)

        def coeff_from_label(label: tuple[PairOrbit, ...]):
            blocked_power = matrix ** BlockDiagonalMatrix(tuple(l.count_matrix for l in label), matrix.block_sizes)
            return math.prod(np.prod(b) for b in blocked_power.blocks)

        return np.fromiter(
            (coeff_from_label(label) for label in self.iterate_labels()),
            dtype=matrix.blocks[0].dtype,
            count=self.size(),
        )

    @caching.cache_noargs
    def coefficients_of_identity(self) -> np.ndarray:
        """Returns the coefficients of the identity matrix"""
        # TODO: optimize this by calling [b.coefficients_of_identity() for b in self.bases]
        return self.coefficients_for_tensor_product(
            BlockDiagonalMatrix(
                np.ones(len(self.original_blocks), dtype=int), tuple(b.dimension for b in self.original_blocks)
            )
        )

    def __str__(self):
        return f"{type(self).__name__}({self.n}, {' ⨁ '.join(str(b) for b in self.original_blocks)})"



class EndSnBlockOrbitBasis(_BaseEndSnBlockOrbitBasis):
    """
    Represents a somewhat canonical Basis of End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n) as a direct sum (over μ ∈ {all (weak) compositions of n into t parts})
    ⊕_μ ⊗_{i = 1}^t End^{S_(μ_i)}((ℂ^{p_i x p_i})^{μ_i})
    This follows the notation in http://arxiv.org/abs/0910.4515, Section 4
    """

    def __init__(self, n: int, *blocks: MatrixStandardBasis):
        """
        blocks represents the different
        """
        self.n = n
        self.t = len(blocks)
        self.compositions_for_blocks = list(weak_compositions(n, self.t))
        self.original_blocks = blocks
        bases = tuple(EndSnSingleBlockOrbitBasis(blocks, mu) for mu in self.compositions_for_blocks)
        super().__init__(bases=bases)


class EndSnBlockOrbitBasisSubset(_BaseEndSnBlockOrbitBasis):
    """
    Similar to EndSnBlockOrbitBasis, but now the blocks are subsets of MatrixStandardBasis.
    This is useful for example for LDUI, where the blocks are the subsets of the standard basis corresponding to the allowed matrix entries.
    """

    # Implements part of the BasisSubset interface
    subset_of: EndSnBlockOrbitBasis

    def __init__(self, n: int, *blocks: MatrixStandardBasisSubset|MatrixStandardBasis):
        self.subset_of = EndSnBlockOrbitBasis(n, *[b.subset_of if isinstance(b, MatrixStandardBasisSubset) else b for b in blocks])

        self.n = n
        self.t = len(blocks)
        self.compositions_for_blocks = list(weak_compositions(n, self.t))
        self.original_blocks = blocks

        bases = tuple(EndSnSingleBlockOrbitBasis(blocks, mu) for mu in self.compositions_for_blocks)
        super().__init__(bases=bases)

    @cached_property
    def index_mapping(self):
        return EndSnBlockOrbitBasisSubsetIndices(self)

    @property
    def valid_indices(self) -> np.ndarray:
        return self.index_mapping.index_mapping().as_numpy()

    @property
    def split_local_valid_indices(self) -> list[np.ndarray]:
        # TODO: instead of splitting, just return the index mapping of the parts
        """Split all the valid indices into parts for each sub-basis and then subtract the offset of the corresponding sub-basis."""
        return [
            a - b
            for a, b in zip(
                np.split(self.valid_indices, self.basis_indices_start[1:]), self.subset_of.basis_indices_start
            )
        ]


class EndSnBlockOrbitBasisSubsetIndices(ScatterIndexMapping, metaclass=caching.WeakRefMemoize):
    basis_from: EndSnBlockOrbitBasisSubset
    basis_to: EndSnBlockOrbitBasis

    def __init__(self, subset: EndSnBlockOrbitBasisSubset):
        self.basis_from = subset
        self.basis_to = subset.subset_of

    def _calculate_index_mapping(self) -> MatrixCache:
        raveled = []

        # Elements of the direct sum decomposition
        for b, b_to, offset in zip(
            self.basis_from.bases, self.basis_to.bases, self.basis_to.basis_indices_start
        ):
            # Elements of the tensor product
            part_indices = [cast(EndSnOrbitBasisSubset, tp_part).valid_indices for tp_part in b.bases]
            # Use np.add.outer to broadcast all the combinations and multiply each index by the corresponding multiplier (which is the size of the basis of the other tensor factors)
            # to get the correct indices in the full tensor product basis. Finally, add the offset to get the correct indices in the full direct sum basis.
            indices = (
                offset
                + np.add.outer(
                    *[mult * part_idx for mult, part_idx in zip(b.basis_indices_multiplier, part_indices)]
                ).ravel()
            )
            raveled.append(indices)
        return MatrixCache(np.concatenate(raveled))


class EmbeddingIntoFullOrbit(ScatterIndexMapping):
    """
    Embeds the EndSnBlockOrbitBasis (i.e. End^{S_n}(⊕_i^t(ℂ^{p_i x p_i})^n)), or any subset thereof,
     into the EndSnOrbitBasis of the full (no longer necessarily block-diagonal) matrices,
    i.e. into End^{S_n}(ℂ^{pxp}) where p = Σ_i p_i. This is mostly intended for testing purposes.
    """

    basis_from: EndSnBlockOrbitBasis | EndSnBlockOrbitBasisSubset
    basis_to: EndSnOrbitBasis

    def __init__(self, _from: EndSnBlockOrbitBasis | EndSnBlockOrbitBasisSubset):
        self.n = _from.n
        self.original_blocks = _from.original_blocks
        self.basis_from = _from
        self.basis_to = EndSnOrbitBasis(self.n, sum(b.dimension for b in self.original_blocks))

    def _calculate_index_mapping(self):
        # noinspection PyAbstractClass
        matrix = np.empty((self.basis_from.size(),), dtype=np.int_)

        block_sizes = tuple(b.dimension for b in self.original_blocks)

        for idx, label in enumerate(self.basis_from.iterate_labels()):
            full_orbit = PairOrbit(BlockDiagonalMatrix([l.count_matrix for l in label], block_sizes).to_full_matrix())
            matrix[idx] = self.basis_to.label_to_index(full_orbit)
        return matrix

    def __str__(self):
        return f"EmbeddingIntoFullOrbit(n={self.n}, {self.original_blocks})"
