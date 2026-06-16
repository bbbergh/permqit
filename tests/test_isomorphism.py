import itertools
import unittest

import numpy as np
import pytest

from permqit.algebra import EndSnOrbitBasis, EndSnBlockDiagonalBasis
from permqit.algebra.linear_map import StorageFormat
from permqit.representation.isomorphism import EndSnBlockDiagonalizationKappa, \
    EndSnTensorProductIsomorphism, EndSnAlgebraIsomorphism, EndSnBlockDiagonalization
from permqit.representation.partial_traces import PartialTraceRelations


class TestEndSnBlockDiagonalization(unittest.TestCase):
    def test_kappa(self):
        for n in [2,3, 4]:
            for d in [2, 3]:
                orbits = EndSnOrbitBasis(n, d)
                vec = np.random.randn(orbits.size())
                iso = EndSnBlockDiagonalizationKappa(n, d)
                full_matrix = orbits.linear_combination(vec)

                coeffs_ssyt = iso.apply_to_coefficient_vector(vec)

                ssyt_pairs = EndSnBlockDiagonalBasis(n, d)
                for i, (t1, t2) in enumerate(ssyt_pairs.iterate_labels()):
                    np.testing.assert_almost_equal(coeffs_ssyt[i], t1.basis_vector(d) @ full_matrix @ t2.basis_vector(d))

    def test_positivity_preserving(self):
        for n in [2,3,4]:
            for d in [2, 3]:
                orbits = EndSnOrbitBasis(n, d)
                vec = np.random.randn(orbits.size())
                vec = vec + vec[orbits.transpose_index_lookup()]
                iso = EndSnBlockDiagonalization(n, d)
                full_eigenvalues = np.linalg.eigvals(orbits.linear_combination(vec))
                min_ev = - np.min(full_eigenvalues)

                blocked = iso.basis_to.linear_combination(iso.apply_to_coefficient_vector(vec + orbits.coefficients_of_identity() * min_ev))

                eigenvalues = np.linalg.eigvals(blocked.to_full_matrix())

                np.testing.assert_array_less(-1e-7, eigenvalues)

    def test_individual_blocks(self):
        for n in [2,3,4]:
            for d in [2, 3]:
                orbits = EndSnOrbitBasis(n, d)
                vec = np.random.randn(orbits.size())
                iso = EndSnBlockDiagonalization(n, d)
                blocked = iso.basis_to.linear_combination(iso.apply_to_coefficient_vector(vec))
                for block, partition in zip(blocked.blocks, iso.basis_to.partitions):
                    np.testing.assert_almost_equal(iso.get_block_transition_matrix(partition) @ vec, block)

class TestEndSnAlgebraIsomorphism(unittest.TestCase):
    def test_identity_and_algebra_property(self):
        for n in [2, 3, 4]:
            for d in [2, 3]:
                iso = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d))
                np.testing.assert_almost_equal(iso.basis_to.linear_combination(
                    iso.apply_to_coefficient_vector(iso.basis_from.coefficients_of_identity())).to_full_matrix(),
                                               np.eye(int(np.sum(iso.basis_to.block_sizes))))

                A = np.random.randn(d, d) # type: np.ndarray
                B = np.random.randn(d, d) # type: np.ndarray
                vecA = iso.basis_from.coefficients_for_tensor_product(A)
                vecB = iso.basis_from.coefficients_for_tensor_product(B)
                vecAB = iso.basis_from.coefficients_for_tensor_product(A @ B)

                np.testing.assert_almost_equal(iso.basis_to.linear_combination(iso.apply_to_coefficient_vector(vecAB)).to_full_matrix(),
                                               (iso.basis_to.linear_combination(iso.apply_to_coefficient_vector(vecA)) @ iso.basis_to.linear_combination(
                                                   iso.apply_to_coefficient_vector(vecB))).to_full_matrix())

    def test_inverse(self):
        for n in [2,3,4]:
            for d in [2,3]:
                iso = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d))
                inviso = iso.inverse()
                vec = np.random.randn(iso.basis_from.size())
                np.testing.assert_almost_equal(inviso.apply_to_coefficient_vector(iso.apply_to_coefficient_vector(vec)), vec)



class TestEndSnTensorProductIsomorphism(unittest.TestCase):
    @pytest.mark.slow
    def test_with_full_matrices(self):
        for n in [2,3]:
            for d in [2, 3]:
                d2 = 2
                iso = EndSnTensorProductIsomorphism(EndSnOrbitBasis(n, d), EndSnOrbitBasis(n, d2))
                vec = np.random.randn(iso.basis_from.size())
                full_matrix = iso.basis_from.linear_combination(vec)
                # If we call the two systems A and B, then this returns system order A_1...A_nB_1...B_n
                # The matrix after the isomorphism however is returned as A_1B_1...A_nB_n
                # So lets permute
                full_matrix = (full_matrix
                               .reshape((d,)*n + (d2,)*n + (d,)*n + (d2,)*n)
                               .transpose(tuple(j for i in range(n) for j in (i, i + n)) + tuple(j for i in range(n) for j in (i + 2*n, i + n + 2*n)))
                               .reshape((d**n * d2**n,)*2))


                coeffs_after = iso.apply_to_coefficient_vector(vec)
                full_matrix_after = iso.basis_to.linear_combination(coeffs_after)
                np.testing.assert_array_equal(full_matrix, full_matrix_after, f"{n=}, {d=}, {d2=}")


class TestEmbedding(unittest.TestCase):

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
