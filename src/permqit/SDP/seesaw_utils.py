from __future__ import annotations

from typing import List

import numpy as np

from ..algebra import EndSnOrbitBasis, CoefficientData, EndSnOrbitBasisSubset, Basis
from ..representation.isomorphism import EndSnAlgebraIsomorphism
from ..representation.partial_traces import PartialTraceRelations, BasePartialTraceRelations
from ..utilities import backend
from ..utilities.backend import xp
from ..utilities.random import random_channel_with_permutation_invariant_output, random_permutation_invariant_channel


def random_perm_inv_encoder(iso_A: EndSnAlgebraIsomorphism, d_R: int, isometry=False, seed=None):
    """
    Returns the Choi matrix coefficients of a random permutation-invariant encoder.

    This is constructed by taking mixtures of random channels onto the different
    blocks of the block-diagonalization of End^{Sn}(V^n) and appropriately
    renormalising them.

    Args:
        ctx: FullContext containing simulation parameters
        isometry: Whether to use isometric encoding
        seed: Random seed for reproducibility

    Returns:
        Coefficients of the Choi Matrix in the orbit basis
    """
    return random_channel_with_permutation_invariant_output(d_R, iso_A, isometry=isometry, seed=seed, TP = True, U = False, xp=backend.xp)


def random_perm_inv_decoder(isos: list[EndSnAlgebraIsomorphism], d_R: int, isometry: bool = False, seed=None):
    """Return Choi coefficients of a random decoder for S_{k_0} × … × S_{k_{m-1}}.

    Generalises ``gpu_random_perm_inv_decoder`` (1 iso) and
    ``gpu_random_perm_inv_decoder_subgroup_k`` (2 isos) to m factors.

    The result is in **SR format** ``(∏_i m_orb_i * d_R²,)``.

    Args:
        isos: List of EndSnAlgebraIsomorphism, one per subgroup factor.
        d_R: Reference system dimension.
        isometry: Whether to use isometric channels.
        seed: Random seed.
    """
    return random_permutation_invariant_channel([iso.basis_from for iso in isos] + [EndSnOrbitBasis(1, d_R)], (-1,), isometry=isometry, seed=seed)


def get_coefficient_AlicePOV(c_N: CoefficientData, c_D: CoefficientData, basisA: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisB: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisAB: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisR: Basis):
    return get_coefficient_AlicePOV_from_relation(c_N, c_D, basisB, basisR, PartialTraceRelations(basisA, basisB, basisAB))


def get_coefficient_AlicePOV_from_relation(c_N: CoefficientData, c_D: CoefficientData, c_D_basis_in: Basis, c_D_basis_out: Basis, relations: BasePartialTraceRelations):
    return relations.apply_traceB_to_coefficient_vectors(c_D.reshape((c_D_basis_in.size(), c_D_basis_out.size())), c_N)


def get_coefficient_BobPOV_from_relation(c_E: CoefficientData, c_E_basis_in: Basis, c_E_basis_out: Basis, c_N: CoefficientData, relations: BasePartialTraceRelations):
    return relations.apply_traceA_to_coefficient_vectors(c_E.reshape((c_E_basis_in.size(), c_E_basis_out.size())).transpose(1, 0), c_N).transpose(1, 0).ravel()


def get_coefficient_BobPOV(c_E: CoefficientData, c_N: CoefficientData, basisA: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisB: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisAB: "EndSnOrbitBasis|EndSnOrbitBasisSubset", basisR: Basis):
    return get_coefficient_BobPOV_from_relation(c_E, basisR, basisA, c_N, PartialTraceRelations(basisA, basisB, basisAB))




# ──────────────────────────────────────────────────────────────────────────────
# Adjoint / transpose conversions
# ──────────────────────────────────────────────────────────────────────────────


def get_coefficient_adjoint_RS(X, d_R, basis: 'EndSnOrbitBasis'):
    """Adjoint of a coefficient vector in RS ordering (d_R² × m_S).

    Transposes the d_R indices and applies the orbit-transpose permutation.

    Args:
        X: Flat coefficient array, shape (d_R² * m_S,)
        d_R: Reference system dimension
        m_S: Orbit basis size
        basis: EndSnOrbitBasis providing the transpose index lookup

    Returns:
        Adjoint coefficients, same shape and device as X
    """
    m_S = basis.size()
    T = X.reshape(d_R, d_R, m_S)
    T_perm = xp.transpose(T, (2, 1, 0))
    idx = basis.transpose_index_lookup()
    T_perm = xp.take(T_perm, xp.asarray(idx), axis=0)
    return T_perm.ravel()


def get_coefficient_adjoint_SR(X, d_R, basis: 'EndSnOrbitBasis'):
    """Adjoint of a coefficient vector in SR ordering (m_S × d_R²).

    Transposes the d_R indices and applies the orbit-transpose permutation.

    Args:
        X: Flat coefficient array, shape (m_S * d_R²,)
        d_R: Reference system dimension
        m_S: Orbit basis size
        basis: EndSnOrbitBasis providing the transpose index lookup

    Returns:
        Adjoint coefficients, same shape and device as X
    """
    m_S = basis.size()
    T = X.reshape(m_S, d_R, d_R)
    T_perm = xp.transpose(T, (2, 1, 0))
    idx = basis.transpose_index_lookup()
    T_perm = xp.take(T_perm, xp.asarray(idx), axis=-1)
    return T_perm.ravel()


def get_coefficient_adjoint_general_RS(
    X,
    d_R: int,
    bases: List['EndSnOrbitBasis'],
) -> np.ndarray:
    """Adjoint of a coefficient vector in RS ordering for m orbit-basis factors.

    Generalises ``gpu_get_coefficient_adjoint_RS`` (1 factor) and
    ``gpu_get_coefficient_adjoint_subgroup_RS`` (2 factors) to an arbitrary
    number of factors.

    Applies the d_R index transposition and the combined orbit-transpose
    permutation for the S_{k_0} × … × S_{k_{m-1}} case.

    Args:
        X: Flat coefficient array in RS format, shape (d_R² * ∏_i m_i,)
        d_R: Reference system dimension.
        bases: List of m orbit bases, one per subgroup factor.

    Returns:
        Adjoint coefficients in SR format (∏_i m_i * d_R²,), same device as X.
    """
    m_sizes = [b.size() for b in bases]
    m_tot = 1
    for s in m_sizes:
        m_tot *= s

    X_arr = X
    T = X_arr.reshape(d_R, d_R, m_tot)
    T_perm = xp.transpose(T, (2, 1, 0))  # (m_tot, d_R, d_R) — SR layout with transposed R

    # Build combined orbit-transpose permutation over all factors.
    idx_arrays = [np.asarray(b.transpose_index_lookup()) for b in bases]
    grid = np.arange(m_tot, dtype=np.int64).reshape(m_sizes)
    for axis, t_idx in enumerate(idx_arrays):
        grid = np.take(grid, t_idx, axis=axis)
    t_to_tt = grid.ravel()

    return T_perm[xp.asarray(t_to_tt), :, :].ravel()


def get_coefficient_adjoint_general_SR(
    X,
    d_R: int,
    bases: List['EndSnOrbitBasis'],
) -> np.ndarray:
    """Adjoint of a coefficient vector in SR ordering for m orbit-basis factors.

    Generalises ``gpu_get_coefficient_adjoint_SR`` (1 factor) and
    ``gpu_get_coefficient_adjoint_subgroup_SR`` (2 factors) to an arbitrary
    number of factors.

    Args:
        X: Flat coefficient array in SR format, shape (∏_i m_i * d_R²,)
        d_R: Reference system dimension.
        bases: List of m orbit bases, one per subgroup factor.

    Returns:
        Adjoint coefficients in RS format (d_R² * ∏_i m_i,), same device as X.
    """
    m_sizes = [b.size() for b in bases]
    m_tot = 1
    for s in m_sizes:
        m_tot *= s

    X_arr = X
    T = X_arr.reshape(m_tot, d_R, d_R)
    T_perm = xp.transpose(T, (2, 1, 0))  # (d_R, d_R, m_tot) — RS layout with transposed R

    # Build combined orbit-transpose permutation over all factors.
    idx_arrays = [np.asarray(b.transpose_index_lookup()) for b in bases]
    grid = np.arange(m_tot, dtype=np.int64).reshape(m_sizes)
    for axis, t_idx in enumerate(idx_arrays):
        grid = np.take(grid, t_idx, axis=axis)
    t_to_tt = grid.ravel()

    return T_perm[:, :, xp.asarray(t_to_tt)].ravel()
