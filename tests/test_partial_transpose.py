"""
Test that the partial_transpose method on PairOrbit correctly computes
the partial transpose at the orbit level.
"""
import numpy as np
import pytest
from permqit.utilities.general_functions import partial_transpose as pt

from permqit.representation.orbits import PairOrbit


@pytest.mark.parametrize("system", ['A', 'B'])
def test_partial_transpose_n1(system):
    """
    Test with n=1. For n=1, the indicator matrix equals the count matrix.
    """
    dA, dB = 2, 2
    d = dA * dB
    n = 1

    orbits = list(PairOrbit.generate_all(n, d))

    dims = [dA, dB]
    subsystem_indices = [1] if system == 'A' else [2]

    errors = []
    for i, orbit in enumerate(orbits):
        indicator = np.array(orbit.indicator_matrix.matrix.todense())
        assert np.array_equal(indicator, orbit.count_matrix)

        pt_indicator_direct = pt(indicator, subsystem_indices, dims)
        pt_orbit = orbit.partial_transpose(system=system)
        pt_indicator_orbit = np.array(pt_orbit.indicator_matrix.matrix.todense())

        error = np.max(np.abs(pt_indicator_direct - pt_indicator_orbit))
        errors.append(error)

    max_error = max(errors)
    assert max_error < 1e-10, f"n=1, PT on {system}: max_error={max_error}"


@pytest.mark.parametrize("system", ['A', 'B'])
def test_partial_transpose_n2(system):
    """
    Test with n=2. The indicator matrix is d^n x d^n.
    """
    dA, dB = 2, 2
    d = dA * dB
    n = 2

    orbits = list(PairOrbit.generate_all(n, d))

    dims = [dA, dB] * n
    if system == 'A':
        subsystem_indices = [1, 3]
    else:
        subsystem_indices = [2, 4]

    errors = []
    for i, orbit in enumerate(orbits):
        indicator = np.array(orbit.indicator_matrix.matrix.todense())
        pt_indicator_direct = pt(indicator, subsystem_indices, dims)

        pt_orbit = orbit.partial_transpose(system=system)
        pt_indicator_orbit = np.array(pt_orbit.indicator_matrix.matrix.todense())

        error = np.max(np.abs(pt_indicator_direct - pt_indicator_orbit))
        errors.append(error)

    max_error = max(errors)
    assert max_error < 1e-10, f"n=2, PT on {system}: max_error={max_error}"


@pytest.mark.parametrize("system", ['A', 'B'])
def test_partial_transpose_n3(system):
    """
    Test with n=3. The indicator matrix is d^n x d^n = 64 x 64.
    """
    dA, dB = 2, 2
    d = dA * dB
    n = 3

    orbits = list(PairOrbit.generate_all(n, d))

    dims = [dA, dB] * n
    if system == 'A':
        subsystem_indices = list(range(1, 2*n + 1, 2))
    else:
        subsystem_indices = list(range(2, 2*n + 1, 2))

    errors = []
    for i, orbit in enumerate(orbits):
        indicator = np.array(orbit.indicator_matrix.matrix.todense())
        pt_indicator_direct = pt(indicator, subsystem_indices, dims)

        pt_orbit = orbit.partial_transpose(system=system)
        pt_indicator_orbit = np.array(pt_orbit.indicator_matrix.matrix.todense())

        error = np.max(np.abs(pt_indicator_direct - pt_indicator_orbit))
        errors.append(error)

    max_error = max(errors)
    assert max_error < 1e-10, f"n=3, PT on {system}: max_error={max_error}"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("system", ['A', 'B'])
def test_partial_transpose_index_lookup(n, system):
    """
    Test that partial_transpose_index_lookup correctly maps orbit coefficients.
    """
    from permqit.algebra.endomorphism_basis import EndSnOrbitBasis

    dA, dB = 2, 2
    d = dA * dB

    basis = EndSnOrbitBasis(n, d)
    m = basis.size()

    pt_indices = basis.partial_transpose_index_lookup(system)

    np.random.seed(42)
    coeffs = np.random.randn(m)

    sigma = np.zeros((d**n, d**n), dtype=np.float64)
    for i, orbit in enumerate(basis.iterate_labels()):
        indicator = np.array(orbit.indicator_matrix.matrix.todense())
        sigma += coeffs[i] * indicator

    dims = [dA, dB] * n
    if system == 'A':
        subsystem_indices = list(range(1, 2*n + 1, 2))
    else:
        subsystem_indices = list(range(2, 2*n + 1, 2))

    sigma_pt_direct = pt(sigma, subsystem_indices, dims)

    coeffs_pt = coeffs[pt_indices]
    sigma_pt_via_lookup = np.zeros((d**n, d**n), dtype=np.float64)
    for i, orbit in enumerate(basis.iterate_labels()):
        indicator = np.array(orbit.indicator_matrix.matrix.todense())
        sigma_pt_via_lookup += coeffs_pt[i] * indicator

    error = np.max(np.abs(sigma_pt_direct - sigma_pt_via_lookup))
    assert error < 1e-10, f"n={n}, system={system}: max_error={error}"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("system", ['A', 'B'])
def test_partial_transpose_is_involution(n, system):
    """
    Test that partial transpose is an involution: PT(PT(sigma)) = sigma.
    """
    from permqit.algebra.endomorphism_basis import EndSnOrbitBasis

    dA, dB = 2, 2
    d = dA * dB

    basis = EndSnOrbitBasis(n, d)
    m = basis.size()

    pt_indices = basis.partial_transpose_index_lookup(system)
    double_pt_indices = pt_indices[pt_indices]
    expected = np.arange(m)

    assert np.array_equal(double_pt_indices, expected), \
        f"n={n}, system={system}: PT is not an involution"


if __name__ == "__main__":
    # Run all tests manually
    for system in ['A', 'B']:
        test_partial_transpose_n1(system)
        test_partial_transpose_n2(system)
        test_partial_transpose_n3(system)

    for n in [1, 2, 3]:
        for system in ['A', 'B']:
            test_partial_transpose_index_lookup(n, system)

    for n in [1, 2, 3]:
        for system in ['A', 'B']:
            test_partial_transpose_is_involution(n, system)

    print("\n=== ALL TESTS PASSED ===")
