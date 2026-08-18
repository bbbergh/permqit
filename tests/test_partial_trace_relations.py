import unittest

import numpy as np

from permqit.algebra import EndSnOrbitBasis
from permqit.algebra.linear_map import StorageFormat
from permqit.representation.partial_traces import PartialTraceRelations


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
