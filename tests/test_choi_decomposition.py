import unittest
from typing import Any, cast

import numpy as np
import pytest

from permqit.algebra import EndSnOrbitBasis
from permqit.algebra.basis import MatrixStandardBasis, MatrixTensorProductBasis
from permqit.algebra.endomorphism_basis import EndSnIrrepBasis
from permqit.algebra.endomorphism_direct_sum_basis import EndSnBlockOrbitBasis
from permqit.representation.isomorphism import (
    OrbitBasisLike,
    block_orbit_full_block_diagonalization_basis,
    block_orbit_block_diagonalization,
    block_orbit_inverse_block_diagonalization,
)
from permqit.representation.partial_traces import (
    BasePartialTraceRelations,
    BlockPartialTraceRelations,
    PartialTraceRelations,
    block_decompose_choi_matrix,
)


def _choi_apply(J: Any, m_A: int, m_B: int, X: Any) -> np.ndarray:
    """Applies a Choi matrix J (shape (m_A*m_B, m_A*m_B), convention row/col=(input,output)) to X (shape (m_A,m_A))."""
    return np.einsum("aoAO,aA->oO", J.reshape(m_A, m_B, m_A, m_B), X)


def _assert_block_decomposition_consistent(relations: BasePartialTraceRelations):
    """
    Checks block_decompose_choi_matrix against relations.apply_traceA_to_coefficient_vectors (already
    covered elsewhere, e.g. test_isomorphism.TestPartialTraceRelations) without ever materializing the
    full dense Choi matrix: applies a random operator X (in orbit basis A) to a random Choi matrix
    both directly, and by reassembling the per-block Choi matrices returned by
    block_decompose_choi_matrix, and checks the results agree.
    """
    relations.ensure_calculated()
    rng = np.random.default_rng(0)
    basisA = cast(OrbitBasisLike, relations.basisA)
    basisB = cast(OrbitBasisLike, relations.basisB)

    choi_coeffs = rng.standard_normal(relations.basisAB.size()) + 1j * rng.standard_normal(relations.basisAB.size())
    X_orbit = rng.standard_normal(basisA.size()) + 1j * rng.standard_normal(basisA.size())

    Y_orbit_expected = relations.apply_traceA_to_coefficient_vectors(X_orbit, choi_coeffs)

    result = block_decompose_choi_matrix(relations, choi_coeffs)

    block_basis_A = block_orbit_full_block_diagonalization_basis(basisA)
    block_basis_B = block_orbit_full_block_diagonalization_basis(basisB)
    X_blocks = block_orbit_block_diagonalization(X_orbit, basisA)
    labels_A = [
        tuple(cast(EndSnIrrepBasis, irrep).partition for irrep in cast(MatrixTensorProductBasis, sub).bases)
        for sub in block_basis_A.bases
    ]
    labels_B = [
        tuple(cast(EndSnIrrepBasis, irrep).partition for irrep in cast(MatrixTensorProductBasis, sub).bases)
        for sub in block_basis_B.bases
    ]

    Y_blocks = []
    for sub_B, label_B in zip(block_basis_B.bases, labels_B):
        m_B = sub_B.dimension
        acc = np.zeros((m_B, m_B), dtype=complex)
        for label_A, X_block in zip(labels_A, X_blocks):
            m_A = X_block.shape[-1]
            if m_A == 0 or m_B == 0:
                continue
            choi_block = result[(label_A, label_B)]
            assert choi_block.shape == (m_A * m_B, m_A * m_B)
            acc += _choi_apply(choi_block, m_A, m_B, X_block)
        Y_blocks.append(acc)

    Y_orbit_actual = block_orbit_inverse_block_diagonalization(Y_blocks, basisB)
    np.testing.assert_allclose(Y_orbit_actual, Y_orbit_expected, atol=1e-8)

    # Every returned block must be labelled by a pair the caller can independently reconstruct, and
    # zero-dimensional blocks (partitions with too many rows) must be omitted.
    for (label_A, label_B), block in result.items():
        assert label_A in labels_A
        assert label_B in labels_B
        assert 0 not in block.shape


class TestBlockDecomposeChoiMatrixSimple(unittest.TestCase):
    """basisA/basisB are both plain EndSnOrbitBasis (PartialTraceRelations)."""

    def test_various_sizes(self):
        for n, d_A, d_B in [(2, 2, 2), (2, 2, 3), (3, 2, 2), (3, 2, 3)]:
            relations = PartialTraceRelations(EndSnOrbitBasis(n, d_A), EndSnOrbitBasis(n, d_B), EndSnOrbitBasis(n, d_A * d_B))
            _assert_block_decomposition_consistent(relations)


class TestBlockDecomposeChoiMatrixComposite(unittest.TestCase):
    """One side is an EndSnBlockOrbitBasis (multiple types), the other a plain EndSnOrbitBasis."""

    def test_composite_input(self):
        basisA = EndSnBlockOrbitBasis(3, MatrixStandardBasis(2), MatrixStandardBasis(2))
        basisB = EndSnOrbitBasis(3, 3)
        _assert_block_decomposition_consistent(BlockPartialTraceRelations(basisA, basisB))

    def test_composite_output(self):
        # BlockPartialTraceRelations only auto-wraps a plain EndSnOrbitBasis for its second (basisB)
        # argument, so basisA must be wrapped explicitly here.
        basisA = EndSnBlockOrbitBasis(2, MatrixStandardBasis(3))
        basisB = EndSnBlockOrbitBasis(2, MatrixStandardBasis(2), MatrixStandardBasis(3))
        _assert_block_decomposition_consistent(BlockPartialTraceRelations(basisA, basisB))

    def test_composite_input_distinct_types(self):
        basisA = EndSnBlockOrbitBasis(3, MatrixStandardBasis(2), MatrixStandardBasis(3))
        basisB = EndSnOrbitBasis(3, 2)
        _assert_block_decomposition_consistent(BlockPartialTraceRelations(basisA, basisB))

    @pytest.mark.xfail(
        strict=True,
        reason="Known pre-existing bug: SingleBlockPartialTraceRelations mismatches per-type splits "
        "when both basisA and basisB are composite (t>1) simultaneously for n>=2 -- see the TODO on "
        "SingleBlockPartialTraceRelations in partial_traces.py. Not introduced by, or fixable within, "
        "block_decompose_choi_matrix.",
    )
    def test_composite_both_sides_known_bug(self):
        basisA = EndSnBlockOrbitBasis(2, MatrixStandardBasis(2), MatrixStandardBasis(4))
        basisB = EndSnBlockOrbitBasis(2, MatrixStandardBasis(3), MatrixStandardBasis(5))
        relations = BlockPartialTraceRelations(basisA, basisB)
        relations.ensure_calculated()
