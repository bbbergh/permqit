import unittest

import numpy as np

from permqit.algebra import EndSnOrbitBasis
from permqit.algebra.basis import MatrixStandardBasis
from permqit.algebra.basis_subset import IndexIsValidPredicate, MatrixEntryMask, MatrixStandardBasisSubset
from permqit.algebra.endomorphism_direct_sum_basis import (
    EmbeddingIntoFullOrbit,
    EndSnBlockOrbitBasis,
    EndSnSingleBlockOrbitBasis,
)
from permqit.algebra.linear_map import StorageFormat
from permqit.algebra.matrix import BlockDiagonalMatrix
from permqit.representation.combinatorics import weak_compositions
from permqit.representation.orbits import PairOrbit
from permqit.representation.partial_traces import (
    BlockPartialTraceRelations,
    PartialTraceRelations,
    SingleBlockPartialTraceRelations,
)


class TestPartialTraceRelations(unittest.TestCase):
    def test_with_random_coefficients(self):
        for n in [2,3]:
            for d in [2, 3]:
                d2 = 2
                rel = PartialTraceRelations(EndSnOrbitBasis(n, d), EndSnOrbitBasis(n, d2), EndSnOrbitBasis(n, d*d2))
                vec = np.random.randn(rel.basisA.size()) # type: np.ndarray
                joint_vec = np.random.randn(rel.basisAB.size()) # type: np.ndarray

                # The joint basis returns systems in order (A_1B_1) ... (A_n B_n)
                # To take partial traces we need to group the A and B systems together, i.e. order as (A_1...A_n)(B_1...B_n)

                explicit_joint = (rel.basisAB.linear_combination(joint_vec).reshape((d, d2)*n * 2)
                               .transpose(tuple(range(0, 2*n-1, 2)) + tuple(range(1, 2*n, 2))
                                          + tuple(range(2*n, 4*n-1, 2)) + tuple(range(2*n + 1, 4*n, 2)))
                               .reshape((d**n, d2**n)*2))
                explicit_A = rel.basisA.linear_combination(vec).T

                partial_tr = np.einsum("ij,jmin->mn", explicit_A, explicit_joint)

                np.testing.assert_almost_equal(
                    rel.basisB.linear_combination(rel.apply_traceA_to_coefficient_vectors(vec, joint_vec)),
                    partial_tr, err_msg=f"{n=}, {d=}, {d2=}")

                vec_B = np.random.randn(rel.basisB.size()) # type: np.ndarray
                explicit_B = rel.basisB.linear_combination(vec_B).T
                partial_tr_B = np.einsum("ij,mjni->mn", explicit_B, explicit_joint)
                np.testing.assert_almost_equal(
                    rel.basisA.linear_combination(rel.apply_traceB_to_coefficient_vectors(vec_B, joint_vec)),
                    partial_tr_B, err_msg=f"{n=}, {d=}, {d2=}"
                )

    def test_individual(self):
        for n in [2, 3]:
            for d in [2]:
                d2 = 2
                rel = PartialTraceRelations(EndSnOrbitBasis(n, d), EndSnOrbitBasis(n, d2), EndSnOrbitBasis(n, d*d2))
                rel.ensure_calculated()
                for joint_idx in range(rel.basisAB.size()):
                    idxA, idxB = rel.get_basis_index_mapping(joint_idx)
                    joint = ((rel.basisAB.vector_at_index(joint_idx).reshape((d, d2)*n * 2)
                               .transpose(tuple(range(0, 2*n-1, 2)) + tuple(range(1, 2*n, 2))
                                          + tuple(range(2*n, 4*n-1, 2)) + tuple(range(2*n + 1, 4*n, 2)))
                               .reshape((d**n, d2**n)*2))).todense()
                    explicit_A = rel.basisA.vector_at_index(idxA).T.todense()
                    partial_tr = np.einsum("ij,jmin->mn", explicit_A, joint)
                    np.testing.assert_almost_equal(rel.basisB.vector_at_index(idxB).todense()*rel.trace_coefficientsA.get(StorageFormat.NUMPY)[joint_idx], partial_tr, err_msg=f"{n=}, {d=}, {d2=}, {joint_idx=}, {idxA=}, {idxB=}")

                    explicit_B = rel.basisB.vector_at_index(idxB).T.todense()
                    partial_tr_B = np.einsum("ij,mjni->mn", explicit_B, joint)
                    np.testing.assert_almost_equal(rel.basisA.vector_at_index(idxA).todense()*rel.trace_coefficientsB.get(StorageFormat.NUMPY)[joint_idx], partial_tr_B, err_msg=f"{n=}, {d=}, {d2=}, {joint_idx=}, {idxA=}, {idxB=}")


class TestBlockPartialTraceRelations(unittest.TestCase):
    """
    Checks ``BlockPartialTraceRelations`` against an independent reference: both the joint operator and the
    marginals are embedded into plain (non-block) ``EndSnOrbitBasis``\\ es via ``EmbeddingIntoFullOrbit``, and
    the partial trace is taken there with the well-tested ``PartialTraceRelations`` (see ``TestEmbedding``).

    The one subtlety is the ordering of the single-copy AB space: ``BlockPartialTraceRelations`` lays it out
    block-diagonally as ⊕_{a,b} ℂ^{d_a·e_b}, whereas the reference needs the tensor layout
    (⊕_a ℂ^{d_a}) ⊗ (⊕_b ℂ^{e_b}). The two have the same dimension (Σ_{a,b} d_a·e_b = (Σ_a d_a)(Σ_b e_b)) and
    differ by the fixed single-copy permutation built in ``_block_to_tensor_layout_permutation``.
    """

    @staticmethod
    def _block_to_tensor_layout_permutation(A_dims, B_dims) -> np.ndarray:
        """permutation[block_diagonal_index] = tensor_product_index for the single-copy AB space."""
        D_B = sum(B_dims)
        offsetsA = np.concatenate([[0], np.cumsum(A_dims)[:-1]])
        offsetsB = np.concatenate([[0], np.cumsum(B_dims)[:-1]])
        permutation = np.empty(sum(A_dims) * D_B, dtype=np.int_)
        position = 0
        for a, d_a in enumerate(A_dims):
            for b, e_b in enumerate(B_dims):
                for i in range(d_a):
                    for j in range(e_b):
                        permutation[position] = (offsetsA[a] + i) * D_B + offsetsB[b] + j
                        position += 1
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(permutation)))
        return permutation

    @staticmethod
    def _embed(block_basis, coefficients: np.ndarray, permutation: np.ndarray | None = None):
        """
        Scatters coefficients given in ``block_basis`` into the plain ``EndSnOrbitBasis`` that block_basis embeds
        into, optionally relabelling the single-copy space by ``permutation`` first.

        :return: a tuple (full orbit basis, coefficient vector in that basis)
        """
        embedding = EmbeddingIntoFullOrbit(block_basis)
        full_basis = embedding.basis_to

        if permutation is None:
            indices = embedding.index_mapping().as_numpy()
        else:
            # Conjugating each copy by the permutation matrix P maps the matrix unit E_{r,c} to E_{P(r),P(c)},
            # so the count matrix K of an orbit becomes K[inverse, inverse].
            inverse = np.argsort(permutation)
            block_sizes = tuple(b.dimension for b in block_basis.original_blocks)
            indices = np.fromiter(
                (
                    full_basis.label_to_index(
                        PairOrbit(
                            BlockDiagonalMatrix([part.count_matrix for part in label], block_sizes).to_full_matrix()[
                                np.ix_(inverse, inverse)
                            ]
                        )
                    )
                    for label in block_basis.iterate_labels()
                ),
                dtype=np.int_,
                count=block_basis.size(),
            )

        assert len(set(indices.tolist())) == len(indices), "the embedding into the full orbit basis must be injective"
        full_coefficients = np.zeros(full_basis.size(), dtype=complex)
        full_coefficients[indices] = coefficients
        return full_basis, full_coefficients, indices

    def _assert_matches_full_orbit_reference(self, A_dims, B_dims, n, seed=0):
        rng = np.random.default_rng(seed)
        basisA = EndSnBlockOrbitBasis(n, *[MatrixStandardBasis(d) for d in A_dims])
        basisB = EndSnBlockOrbitBasis(n, *[MatrixStandardBasis(d) for d in B_dims])
        relations = BlockPartialTraceRelations(basisA, basisB)
        relations.ensure_calculated()

        def random_vector(size):
            return rng.standard_normal(size) + 1j * rng.standard_normal(size)

        vA = random_vector(basisA.size())
        vB = random_vector(basisB.size())
        vAB = random_vector(relations.basisAB.size())

        full_A, wA, indices_A = self._embed(basisA, vA)
        full_B, wB, indices_B = self._embed(basisB, vB)
        full_AB, wAB, _ = self._embed(
            relations.basisAB, vAB, self._block_to_tensor_layout_permutation(A_dims, B_dims)
        )
        reference = PartialTraceRelations(full_A, full_B, full_AB)

        for apply_trace, block_input, full_input, embedded_indices, full_output_size in [
            (
                "apply_traceA_to_coefficient_vectors",
                vA,
                wA,
                indices_B,
                full_B.size(),
            ),
            (
                "apply_traceB_to_coefficient_vectors",
                vB,
                wB,
                indices_A,
                full_A.size(),
            ),
        ]:
            expected = getattr(reference, apply_trace)(full_input, wAB)
            actual = np.zeros(full_output_size, dtype=complex)
            actual[embedded_indices] = getattr(relations, apply_trace)(block_input, vAB)

            # Tracing out one side of a block-diagonal joint operator stays block diagonal, so the reference
            # result must live entirely inside the image of the embedding.
            outside = np.delete(expected, embedded_indices)
            np.testing.assert_allclose(
                outside, 0, atol=1e-9, err_msg=f"{apply_trace} leaked outside the blocks for {A_dims=} {B_dims=} {n=}"
            )
            np.testing.assert_allclose(
                actual, expected, atol=1e-9, err_msg=f"{apply_trace} mismatch for {A_dims=} {B_dims=} {n=}"
            )

    def test_no_composite_side(self):
        self._assert_matches_full_orbit_reference((2,), (3,), 2)

    def test_composite_input_only(self):
        self._assert_matches_full_orbit_reference((2, 2), (3,), 2)
        self._assert_matches_full_orbit_reference((2, 3), (2,), 2)

    def test_composite_output_only(self):
        self._assert_matches_full_orbit_reference((3,), (2, 3), 2)

    def test_composite_both_sides_equal_dimensions(self):
        # Regression test: with equal block dimensions on each side the joint types (0,1) and (1,0) used to be
        # paired with swapped B-parts, which passed every dimension check and silently produced wrong indices.
        self._assert_matches_full_orbit_reference((1, 1), (2, 2), 2)
        self._assert_matches_full_orbit_reference((2, 2), (2, 2), 2)

    def test_composite_both_sides_distinct_dimensions(self):
        # Regression test: with distinct block dimensions the same mispairing instead made the construction fail
        # the dA*dB == dAB assertion in PartialTraceRelations.
        self._assert_matches_full_orbit_reference((1, 2), (1, 3), 2)
        self._assert_matches_full_orbit_reference((2, 2), (2, 3), 2)

    def test_composite_both_sides_three_copies(self):
        self._assert_matches_full_orbit_reference((1, 2), (1, 2), 3)

    def test_three_types_on_both_sides(self):
        self._assert_matches_full_orbit_reference((1, 1, 2), (1, 2, 1), 2)


class TestSingleBlockPartialTraceRelationsWithBasisSubsets(unittest.TestCase):
    def _assert_all_sectors_construct(self, n: int, m: int, d_A: int = 2, d_B: int = 2):
        d_AB = d_A * d_B

        # Type 0 sparse (a diagonally supported single-copy Choi), the remaining types dense
        diagonal_support = [(a, a) for a in range(d_AB)]
        ab_single_bases = tuple(
            MatrixStandardBasisSubset(
                MatrixStandardBasis(d_AB), IndexIsValidPredicate(MatrixEntryMask((d_AB, d_AB), diagonal_support))
            )
            if i == 0
            else MatrixStandardBasis(d_AB)
            for i in range(m)
        )

        block_basisA = EndSnSingleBlockOrbitBasis((MatrixStandardBasis(d_A),), [n])
        for composition in weak_compositions(n, m):
            block_basisB = EndSnSingleBlockOrbitBasis(
                tuple(MatrixStandardBasis(d_B) for _ in range(m)), list(composition)
            )
            block_basisAB = EndSnSingleBlockOrbitBasis(ab_single_bases, list(composition))

            relations = SingleBlockPartialTraceRelations(block_basisA, block_basisB, block_basisAB)
            relations.ensure_calculated()

            self.assertEqual(relations.A_index_from_joint.as_numpy().shape, (block_basisAB.size(),))
            self.assertEqual(relations.B_index_from_joint.as_numpy().shape, (block_basisAB.size(),))
            self.assertLess(relations.A_index_from_joint.as_numpy().max(initial=0), block_basisA.size())
            self.assertLess(relations.B_index_from_joint.as_numpy().max(initial=0), block_basisB.size())

    def test_two_types(self):
        self._assert_all_sectors_construct(n=2, m=2)
        self._assert_all_sectors_construct(n=3, m=2)

    def test_three_types(self):
        self._assert_all_sectors_construct(n=2, m=3)
