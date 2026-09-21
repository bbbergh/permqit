"""Tests for the fully-isometric encoder ansatz (``isometry=True``).

A permutation-invariant encoder E : R -> A^n has Choi matrix J = (+)_lambda B^lambda (x) I_{f_lambda},
so rank J = sum_lambda f_lambda rank(B^lambda).  E is an isometry iff J is rank one, which forces a
single lambda with f_lambda = 1 and rank B^lambda = 1; for n > d_A the only such partition is (n).
Hence "isometric encoder" == "isometry into Sym^n(A)".
"""
import numpy as np
import pytest

from permqit.power_method import power_iteration
from permqit.power_method.seesaw import compute_tensor_product_fidelity_seesaw
from permqit.representation.isomorphism import (
    EndSnAlgebraIsomorphism,
    EndSnBlockDiagonalization,
    TrivialAlgebraIsomorphism,
    tensor_product_block_diagonalization,
)
from permqit.SDP.seesaw_utils import random_perm_inv_encoder
from permqit.utilities.random import (
    isometry_into_symmetric_block,
    random_channel_with_permutation_invariant_output,
    random_symmetric_isometric_channel,
    symmetric_block_index,
)

D_R = 2


def _iso(n, d=2):
    return EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d))


def _choi_profile(c_E, iso_A, d_R=D_R):
    """(total Choi rank, weight on lambda=(n), total weight off that block)."""
    blocks = tensor_product_block_diagonalization(np.asarray(c_E), [TrivialAlgebraIsomorphism(d_R), iso_A])
    n = iso_A.basis_from.n
    rank, w_sym, w_off = 0, 0.0, 0.0
    for block, partition in zip(blocks, iso_A.basis_to.partitions):
        if block.size == 0:
            continue
        f = partition.count_standard_tableaux()
        eigs = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
        rank += f * int(np.sum(eigs > 1e-9 * max(1.0, eigs.max())))
        weight = f * float(np.real(np.trace(block))) / d_R
        if tuple(partition) == (n,):
            w_sym = weight
        else:
            w_off += abs(weight)
    return rank, w_sym, w_off


class TestIsometricEncoderSampling:
    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_is_a_genuine_isometry(self, n):
        iso_A = _iso(n)
        c_E = random_perm_inv_encoder(iso_A, D_R, isometry=True, seed=n)
        rank, w_sym, w_off = _choi_profile(c_E, iso_A)
        assert rank == 1, "the Choi matrix of an isometry must have rank one"
        assert w_sym == pytest.approx(1.0, abs=1e-10)
        assert w_off < 1e-10

    @pytest.mark.parametrize("n", [3, 5])
    def test_blockwise_is_not_an_isometry_and_matches_legacy(self, n):
        """``isometry="blockwise"`` must reproduce the historical ``isometry=True`` exactly."""
        iso_A = _iso(n)
        new = np.asarray(random_perm_inv_encoder(iso_A, D_R, isometry="blockwise", seed=7))
        legacy = np.asarray(
            random_channel_with_permutation_invariant_output(D_R, iso_A, isometry=True, seed=7)
        )
        np.testing.assert_allclose(new, legacy, atol=1e-12)
        rank, _, _ = _choi_profile(new, iso_A)
        assert rank > 1, "the blockwise mixture is not an isometry"

    @pytest.mark.parametrize("n", [3, 4])
    def test_trace_preserving(self, n):
        iso_A = _iso(n)
        c_E = np.asarray(random_perm_inv_encoder(iso_A, D_R, isometry=True, seed=n))
        blocks = tensor_product_block_diagonalization(c_E, [TrivialAlgebraIsomorphism(D_R), iso_A])
        total = np.zeros((D_R, D_R), dtype=complex)
        for block, partition in zip(blocks, iso_A.basis_to.partitions):
            if block.size == 0:
                continue
            m = block.shape[0] // D_R
            total += partition.count_standard_tableaux() * np.einsum(
                "isjs->ij", block.reshape(D_R, m, D_R, m)
            )
        np.testing.assert_allclose(total, np.eye(D_R), atol=1e-10)

    def test_rejects_non_isometric_V(self):
        iso_A = _iso(4)
        m = iso_A.basis_to.block_sizes[symmetric_block_index(iso_A)]
        with pytest.raises(ValueError, match="not an isometry"):
            isometry_into_symmetric_block(np.zeros((m, D_R)), D_R, iso_A)

    def test_rejects_too_small_symmetric_subspace(self):
        # dim Sym^1(C^2) = 2 >= d_R = 2 is fine; d_R = 3 is not
        with pytest.raises(ValueError, match="No isometry"):
            random_symmetric_isometric_channel(3, _iso(1), seed=0)


class TestIsometricEncoderStep:
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_monotone_and_stays_isometric(self, n):
        """The polar ascent never decreases the objective and preserves the isometry exactly."""
        iso_A = _iso(n)
        # M must be PSD (it is the Choi matrix of a CP map in every real use); the Choi matrix of
        # a random channel is a convenient stand-in in exactly the right coefficient layout.
        c_M = np.asarray(random_perm_inv_encoder(iso_A, D_R, isometry=False, seed=1000 + n))
        c_E = random_perm_inv_encoder(iso_A, D_R, isometry=True, seed=n)
        previous = -np.inf
        for _ in range(4):
            result = power_iteration.isometric_preparation_coefficient(
                np.asarray(c_M), np.asarray(c_E), D_R, [iso_A], power_max_iterations=200
            )
            c_E = result.assert_get_first_optimizer()
            assert result.get_value() >= previous - 1e-12
            previous = result.get_value()
            rank, w_sym, w_off = _choi_profile(c_E, iso_A)
            assert rank == 1 and w_off < 1e-10 and w_sym == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("n", [3, 4])
    def test_agrees_with_generic_power_step_on_the_symmetric_block(self, n):
        """Restricted to lambda=(n), the Schroedinger power step *is* the polar ascent."""
        iso_A = _iso(n)
        c_M = np.asarray(random_perm_inv_encoder(iso_A, D_R, isometry=False, seed=100 + n))
        c_E = np.asarray(random_perm_inv_encoder(iso_A, D_R, isometry=True, seed=n))
        iso_res = power_iteration.isometric_preparation_coefficient(
            c_M, c_E, D_R, [iso_A], power_max_iterations=500, power_tolerance=1e-13
        )
        gen_res = power_iteration.preparation_coefficient(
            c_M, c_E, D_R, [iso_A], power_max_iterations=500, power_tolerance=1e-13
        )
        # the generic step, seeded with an isometry, must stay isometric and reach the same value
        rank, _, w_off = _choi_profile(gen_res.assert_get_first_optimizer(), iso_A)
        assert rank == 1 and w_off < 1e-9
        assert iso_res.get_value() == pytest.approx(gen_res.get_value(), abs=1e-7)


class TestSkipZeroMBlocks:
    def test_equivalent_to_not_skipping(self):
        """Blocks with M = 0 must end at I_R (x) I_m / d_R either way."""
        n = 4
        iso_B = _iso(n)
        bases = [TrivialAlgebraIsomorphism(D_R).basis_to, iso_B.basis_to]
        rng = np.random.default_rng(0)
        M_blocks, C_blocks = [], []
        for i, m in enumerate(iso_B.basis_to.block_sizes):
            dim = D_R * m
            X = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
            M = X @ X.conj().T
            M /= np.linalg.norm(M) * 50
            M_blocks.append(np.zeros((dim, dim), dtype=complex) if i % 2 else M)
            Y = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
            C_blocks.append(Y @ Y.conj().T)

        # start from *valid* (unital) decoder blocks -- otherwise the very first step lowers the
        # (meaningless) objective, the iteration reverts, and nothing is comparable
        C_blocks = power_iteration._normalize_blocks(
            C_blocks, list(iso_B.basis_to.block_sizes), [1] * len(C_blocks), D_R, "h"
        )

        f_skip, C_skip, _, _ = power_iteration.power_iteration(
            bases, M_blocks, C_blocks, picture="h", max_iterations=60, skip_zero_M_blocks=True
        )
        f_full, C_full, _, _ = power_iteration.power_iteration(
            bases, M_blocks, C_blocks, picture="h", max_iterations=60, skip_zero_M_blocks=False
        )
        assert f_skip == pytest.approx(f_full, abs=1e-9)
        for Cs, Cf in zip(C_skip, C_full):
            np.testing.assert_allclose(np.asarray(Cs), np.asarray(Cf), atol=1e-8)

    def test_rejects_schroedinger_picture(self):
        iso_A = _iso(3)
        bases = [TrivialAlgebraIsomorphism(D_R).basis_to, iso_A.basis_to]
        blocks = [np.eye(D_R * m, dtype=complex) for m in iso_A.basis_to.block_sizes]
        with pytest.raises(ValueError, match="Heisenberg"):
            power_iteration.power_iteration(
                bases, blocks, blocks, picture="s", skip_zero_M_blocks=True
            )


def _choi_depolarizing(p):
    """Choi matrix (unnormalized) of (1-p) id + p (I/2) Tr on a qubit."""
    J = np.zeros((4, 4), dtype=complex)
    J[0, 0] = J[3, 3] = 1 - p / 2
    J[0, 3] = J[3, 0] = 1 - p
    J[1, 1] = J[2, 2] = p / 2
    return J


class TestIsometricSeesawMatchesUnconstrained:
    @pytest.mark.parametrize("n", [2, 3])
    def test_depolarizing(self, n):
        J = _choi_depolarizing(0.15)
        kwargs = dict(
            n=n, d_R=D_R, N=J, d_A=2, d_B=2, repetitions=4, iterations=200,
            seesaw_accuracy=1e-11, print_iterations=False, return_optimizers=True,
        )
        iso = compute_tensor_product_fidelity_seesaw(isometry=True, **kwargs)
        free = compute_tensor_product_fidelity_seesaw(isometry="blockwise", **kwargs)
        assert iso.get_value() >= free.get_value() - 1e-6
        iso_A = _iso(n)
        rank, _, w_off = _choi_profile(iso.get_optimizers()[0], iso_A)
        assert rank == 1 and w_off < 1e-9
