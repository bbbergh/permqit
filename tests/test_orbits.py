import unittest

import numpy as np
import sparse

from permqit.representation.orbits import PairOrbit


class TestOrbits(unittest.TestCase):

    def test_transpose(self):
        """
        Test that the transpose of an orbit is the same as the transpose of the indicator matrix
        :return:
        """
        for n in [2, 3, 4]:
            for d in [2, 3]:
                for orbit in PairOrbit.generate_all(n, d):
                    np.testing.assert_array_equal(orbit.indicator_matrix.matrix.T.todense(), orbit.T.indicator_matrix.matrix.todense())

    def test_matrix_properties(self):
        """Test that the computation of trace and HS norm via combinatorics is equivalent to taking the trace of the explicit matrix"""
        for n in [2,3,4]:
            for d in [2,3]:
                for orbit in PairOrbit.generate_all(n, d):
                    self.assertEqual(orbit.indicator_matrix.trace, orbit.indicator_matrix.matrix.todense().trace())
                    mat = orbit.indicator_matrix.matrix
                    self.assertEqual(orbit.indicator_matrix.HS_norm, np.trace((mat.T @ mat).todense()))


    def test_count(self):
        """Test that PairOrbit.generate_all generates the expected number of orbits"""
        for n in [2,3,4]:
            for d in [2,3]:
                self.assertEqual(len(list(PairOrbit.generate_all(n, d))), PairOrbit.count(n, d), msg=f"{n=}, {d=}")


    def test_to_from_monomial(self):
        """Test that PairOrbit.from_monomial generates the expected orbits"""
        for n in [2,3,4]:
            for d in [2,3]:
                for o in PairOrbit.generate_all(n, d):
                    self.assertEqual(o, PairOrbit.from_monomial(o.to_monomial(), d), msg=f"{n=}, {d=}, {o=}")

    def test_indicator_matrix(self):
        """
        Test that all the indicator matrices together cover the whole space
        :return:
        """
        for n in [2, 3, 4]:
            for d in [2, 3]:
                summed: np.ndarray = sum(o.indicator_matrix.matrix.todense() for o in PairOrbit.generate_all(n, d)) # type: ignore
                self.assertEqual(summed[summed <= 0].size, 0)



