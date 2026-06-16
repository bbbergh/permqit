"""
Tests for the seesaw optimization methods.

The refactored ``permqit`` exposes two independent seesaw implementations that
should agree on the achievable channel fidelity:

* ``permqit.SDP.seesaw`` - solves each half-step as an SDP (singlet fraction).
* ``permqit.power_method.seesaw`` - uses the symmetric power method.

Both take the single-copy Choi matrix of a permutation-invariant channel.  We
drive them with the depolarizing channel from the examples package, for which
``p=0`` is the identity channel (fidelity 1) regardless of ``n``.
"""
import numpy as np
import pytest

from examples.channels.depolarizing_channel import depolarizing_choi

from permqit.SDP.seesaw import (
    compute_tensor_product_fidelity_seesaw as sdp_seesaw,
    compute_tensor_product_fidelity_without_symmetry_seesaw as sdp_seesaw_brute_force,
)
from permqit.power_method.seesaw import (
    compute_tensor_product_fidelity_seesaw as power_seesaw,
)


def _power(n, d_R, N, d_A, d_B, **kwargs):
    """Run the power-method seesaw with test-friendly defaults."""
    return power_seesaw(
        n, d_R, N, d_A, d_B,
        print_iterations=False,
        verbose=False,
        return_optimizers=False,
        timing_analysis=False,
        **kwargs,
    ).get_value()


class TestSeesawSDP:
    """Sanity checks on the SDP-based seesaw."""

    def test_identity_channel_is_perfect(self):
        """p=0 is the identity channel; entanglement fidelity must be 1."""
        N = depolarizing_choi(0.0, 2)
        fidelity = sdp_seesaw(1, 2, N, 2, 2, timing_analysis=False).get_value()
        assert fidelity == pytest.approx(1.0, abs=1e-6)

    def test_fidelity_in_unit_interval(self):
        """A noisy channel yields a fidelity strictly inside [0, 1]."""
        N = depolarizing_choi(0.1, 2)
        fidelity = sdp_seesaw(1, 2, N, 2, 2, timing_analysis=False).get_value()
        assert 0.0 <= fidelity <= 1.0
        # Known reproducible value for the n=1 depolarizing channel (p=0.1, seed=42).
        assert fidelity == pytest.approx(0.925, abs=1e-4)


class TestSeesawCrossCheck:
    """The SDP and power-method implementations must agree."""

    def test_agreement_n1(self):
        N = depolarizing_choi(0.1, 2)
        fid_sdp = sdp_seesaw(1, 2, N, 2, 2, timing_analysis=False).get_value()
        fid_power = _power(1, 2, N, 2, 2)
        assert fid_sdp == pytest.approx(fid_power, abs=1e-3)

    @pytest.mark.slow
    def test_agreement_n2(self):
        N = depolarizing_choi(0.1, 2)
        fid_sdp = sdp_seesaw(2, 2, N, 2, 2, timing_analysis=False).get_value()
        fid_power = _power(2, 2, N, 2, 2)
        assert fid_sdp == pytest.approx(fid_power, abs=1e-3)


class TestSeesawBruteForce:
    """The brute-force path builds the full tensor-product channel and runs the
    seesaw with n=1 (no symmetry reduction)."""

    def test_runs_and_is_valid_n2(self):
        """Regression: the function used to crash calling ``.todense()`` on the
        dense ndarray returned by ``linear_combination``.  It must now run and
        return a fidelity in [0, 1].  Note: at n>=2 this value need NOT equal the
        symmetric result, since the two methods optimize over different ansätze."""
        N = depolarizing_choi(0.1, 2)
        fidelity = sdp_seesaw_brute_force(2, 2, N, 2, 2, timing_analysis=False).get_value()
        assert 0.0 <= fidelity <= 1.0

    def test_matches_symmetric_n1(self):
        """At n=1 there is no symmetry to exploit, so brute-force and symmetric
        must agree."""
        N = depolarizing_choi(0.1, 2)
        fid_sym = sdp_seesaw(1, 2, N, 2, 2, timing_analysis=False).get_value()
        fid_brute = sdp_seesaw_brute_force(1, 2, N, 2, 2, timing_analysis=False).get_value()
        assert fid_sym == pytest.approx(fid_brute, abs=1e-6)
