import unittest

import numpy as np
import pytest

from permqit.representation import Partition, SSYT
from permqit.representation.isomorphism import (
    EndSnBlockDiagonalizationGijswijt,
    EndSnBlockDiagonalizationKappa,
)
from permqit.representation.isomorphism_gijswijt import BlockDiagonalization
from permqit.representation.isomorphism_kappa import calculate_f
from permqit.representation.orbits import PairOrbit
from permqit.algebra.endomorphism_basis import EndSnOrbitBasis


class TestBlockDiagonalizationGijswijt(unittest.TestCase):
    def test_simple_inner_product(self):
        """Tests (21) of https://arxiv.org/abs/0910.4515v1, by comparing the inner products computed through the combinatorical construction with an explicit calculation"""
        for n in [2,3,4]:
            for d in [2,3]:
                for part in Partition.generate_all(n):
                    poly = BlockDiagonalization.polynomial(part)
                    vector = part.constant_tableau.basis_vector(d)
                    for orbit in PairOrbit.generate_all(n, d):
                        self.assertEqual(vector.T @ orbit.indicator_matrix.matrix @ vector, poly.coeffs.get(orbit.to_monomial(), 0), f"{n=}, {d=}, {poly=}, {orbit=}, {part=}")

    @pytest.mark.slow
    def test_general_inner_product(self):
        """Tests Theorem 7 of https://arxiv.org/abs/0910.4515v1, by comparing the inner products computed through the combinatorical construction with an explicit calculation"""
        for n in [2,3]:
            for d in [3]:
                orbits = list(PairOrbit.generate_all(n, d))
                for part in Partition.generate_all(n):
                    for t1 in SSYT.generate_all(part, d):
                        for t2 in SSYT.generate_all(part, d):
                            poly = BlockDiagonalization.tableaux_polynomial(t1, t2)
                            for orbit in orbits:
                                self.assertEqual(t2.basis_vector(d).T @ orbit.indicator_matrix.matrix @ t1.basis_vector(d), poly.coeffs.get(orbit.to_monomial(), 0),
                                                 f"{n=}, {d=}, {t1=}, {t2=}, {part=}, {orbit=}, {poly=}")


    @pytest.mark.slow
    def test_tableaux_orbits(self):
        """
        Test that the orbits of a tableau with respect to the constant_tableau of the partition can be alternatively calculated
        by eq (3.23) in https://arxiv.org/pdf/2005.02945
        :return:
        """
        for n in [2,3,4]:
            for d in [2,3]:
                for part in Partition.generate_all(n):
                    for t in SSYT.generate_all(part, d):
                        from_tableau = PairOrbit.from_two_sequences(t.flat, part.constant_tableau.flat, d)
                        cnt = np.zeros((d,d), dtype=np.int_)
                        for j, row in enumerate(t.rows):
                            for s in range(d):
                                cnt[s, j] = sum(1 for el in row if el == s)

                        self.assertEqual(from_tableau, PairOrbit(cnt, n, d))


class TestAdjointPreservation(unittest.TestCase):
    """Regression test for the Gijswijt n=7 floating-point bug.

    The Gijswijt tableaux_polynomial uses floating-point 1/(k+1) factors.
    For n=7 one coefficient computes as 59.9999... instead of 60, which (before
    the round() fix) truncated to 59 in the integer sparse matrix, breaking the
    adjoint-preservation property phi(X^T) = phi(X)^T.
    """

    @staticmethod
    def _adjoint_error(iso):
        """Return max |T[:,P_from] - P_to @ T| where P_from/P_to are the adjoint permutations."""
        T = np.asarray(iso.coefficient_transition_matrix().todense()).astype(float)
        basis_from, basis_to = iso.basis_from, iso.basis_to
        adj_from = np.array([
            basis_from.label_to_index(PairOrbit(o.count_matrix.T))
            for o in basis_from.iterate_labels()
        ])
        labels = list(basis_to.iterate_labels())
        adj_to = np.array([basis_to.label_to_index((t2, t1)) for t1, t2 in labels])
        return float(np.max(np.abs(T[:, adj_from] - T[adj_to, :])))

    def test_gijswijt_adjoint_n7(self):
        """Gijswijt must preserve the adjoint for n=7 (regression test for the round() fix)."""
        err = self._adjoint_error(EndSnBlockDiagonalizationGijswijt(7, 2))
        self.assertEqual(err, 0.0, msg=f"Gijswijt n=7: adjoint error {err} != 0")

    def test_gijswijt_equals_kappa(self):
        """After the round() fix, Gijswijt and Kappa must produce identical matrices."""
        for n in [3, 4, 5, 6, 7]:
            T_gij = np.asarray(EndSnBlockDiagonalizationGijswijt(n, 2).coefficient_transition_matrix().todense())
            T_kap = np.asarray(EndSnBlockDiagonalizationKappa(n, 2).coefficient_transition_matrix().todense())
            self.assertTrue(np.array_equal(T_gij, T_kap), msg=f"Gijswijt != Kappa for n={n}")


class TestBlockDiagonalizationKappa(unittest.TestCase):
    @pytest.mark.slow
    def test_general_inner_product(self):
        """Tests calculate_f(), which implements the formula of Appendix 2 of DOI 10.1007/s10623-016-0216-5.
        Test this by comparing the inner products computed through the combinatorical construction with an explicit calculation"""
        for n in [2,3]:
            for d in [2, 3]:
                orbits = list(PairOrbit.generate_all(n, d))
                for part in Partition.generate_all(n):
                    for t1 in SSYT.generate_all(part, d):
                        t1vec = t1.basis_vector(d)
                        for t2 in SSYT.generate_all(part, d):
                            t2vec = t2.basis_vector(d)
                            poly = calculate_f(part, t1, t2, d)
                            for orbit in orbits:
                                self.assertEqual(t1vec.T @ orbit.indicator_matrix.matrix @ t2vec, poly.coeffs.get(orbit.to_monomial(), 0),
                                                 f"{n=}, {d=}, {t1=}, {t2=}, {part=}, {orbit=}, {poly=}")