"""
GPU-accelerated power iteration method for seesaw optimization.

This module implements a GPU-accelerated power iteration algorithm for optimizing
quantum encoder and decoder channels in the seesaw method. The power method replaces
expensive SDP solves with iterative matrix operations that can be efficiently
computed on GPUs using CuPy.

Algorithm Overview:
------------------
The power method optimizes encoder/decoder by iteratively applying:
    C_new = M @ C @ M
followed by normalization to enforce physical constraints:
- Trace-preserving (Schrödinger picture / encoder): Tr_S(E) = I_R
- Unitality (Heisenberg picture / decoder): Tr_R(D) = I_S
"""
from __future__ import annotations

import time
import warnings
from typing import List, Optional, Sequence, TYPE_CHECKING, Literal

import numpy as np

from ..algebra import EndSnBlockDiagonalBasis
from ..representation.isomorphism import (
    EndSnAlgebraIsomorphism,
    TrivialAlgebraIsomorphism,
    tensor_product_block_diagonalization,
    tensor_product_inverse_block_diagonalization,
)
from ..utilities.backend import xp, USE_GPU
from ..utilities.timing import MaybeExpensiveComputation


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_NUM_ITERATIONS_POWER = 5000
DEFAULT_POWER_ACCURACY = 1e-8
NOISE_TOLERANCE = 1e-7


def _normalize_blocks(
    C_blocks: List, 
    block_sizes_m: List[int], 
    weights: List[int], 
    d_R: int,
    picture: Literal['h', 's'],
) -> List:
    """Unified Choi-block normalization.

    picture='h' (Heisenberg / decoder): per-block unitality Tr_R(D^λ) = I_{m_λ}.
        D_new = (I_R ⊗ X^{-1/2}) D (I_R ⊗ X^{-1/2}), X = Tr_R(D).
        Null-space fill ensures Tr_R(D_new) = I when X is rank-deficient.

    picture='s' (Schrödinger / encoder): coupled trace-preserving Σ_λ w_λ Tr_S(E^λ) = I_R.
        First pass: X = Σ_λ w_λ Tr_S(E^λ).
        E_new = (X^{-1/2} ⊗ I_S) E (X^{-1/2} ⊗ I_S).
        Null-space fill distributes residual weight uniformly.

    Zero-sized blocks pass through unchanged.
    """
    if picture == 'h':
        C_normalized = []
        for C, m, _w in zip(C_blocks, block_sizes_m, weights):
            if m == 0:
                C_normalized.append(C)
                continue
            C_reshaped = C.reshape(d_R, m, d_R, m)
            C_S = xp.einsum('iris->rs', C_reshaped, optimize='optimal')
            C_S_inv_sqrt = matrix_inverse_sqrt(C_S)
            C_new = xp.einsum('pr,irjs,sq->ipjq', C_S_inv_sqrt, C_reshaped, C_S_inv_sqrt, optimize='optimal')
            
            #Check rank deficiency of C_new and apply null-space fill if needed
            P = C_S_inv_sqrt @ C_S @ C_S_inv_sqrt
            if float(xp.max(xp.abs(P - xp.eye(m, dtype=P.dtype)))) > NOISE_TOLERANCE:
                Q = xp.eye(m, dtype=C_new.dtype) - P
                I_R = xp.eye(d_R, dtype=C_new.dtype)
                C_new = C_new + (1.0 / d_R) * xp.einsum('ij,rs->irjs', I_R, Q)
            
            C_normalized.append(C_new.reshape(d_R * m, d_R * m))
        return C_normalized
    if picture == 's':
        # First pass: accumulate X = Σ_λ w_λ Tr_S(E^λ)
        T = xp.zeros((d_R, d_R), dtype=C_blocks[0].dtype)
        for C, m, w in zip(C_blocks, block_sizes_m, weights):
            if m == 0:
                continue
            T += w * xp.einsum('irjr->ij', C.reshape(d_R, m, d_R, m), optimize='optimal')

        T_inv_sqrt = matrix_inverse_sqrt(T)
        P_R = T_inv_sqrt @ T @ T_inv_sqrt
        #check rank deficiency of T and apply null-space fill if needed
        Q_R = None
        total_wm = None
        if float(xp.max(xp.abs(P_R - xp.eye(d_R, dtype=P_R.dtype)))) > NOISE_TOLERANCE:
            Q_R = xp.eye(d_R, dtype=P_R.dtype) - P_R
            total_wm = float(sum(w * m for w, m in zip(weights, block_sizes_m) if m > 0))

        # Second pass: apply (X^{-1/2} ⊗ I_S) E (X^{-1/2} ⊗ I_S)
        C_normalized = []
        for C, m, w in zip(C_blocks, block_sizes_m, weights):
            if m == 0:
                C_normalized.append(C)
                continue
            C_reshaped = C.reshape(d_R, m, d_R, m)
            C_new = xp.einsum('ip,pkql,qj->ikjl', T_inv_sqrt, C_reshaped, T_inv_sqrt, optimize='optimal')
            if Q_R is not None:
                I_S = xp.eye(m, dtype=C_new.dtype)
                C_new = C_new + (1.0 / total_wm) * xp.einsum('ij,rs->irjs', Q_R, I_S)  # ty:ignore[unsupported-operator]
            C_normalized.append(C_new.reshape(d_R * m, d_R * m))
        return C_normalized
    else:
        raise ValueError(f"Invalid picture '{picture}'; expected 'h' or 's'.")


def _compute_fidelity(
    M_blocks: List, 
    C_blocks: List, 
    weights: List[int], 
    d_R: int
) -> float:
    """Fidelity F = (1/d_R²) Σ_λ w_λ Tr(M^λ @ C^λ).  Zero-sized blocks skipped."""
    fidelity = 0.0
    for M, C, w in zip(M_blocks, C_blocks, weights):
        if M.size == 0:
            continue
        tr = xp.trace(M @ C)
        tr_val = float(xp.real(tr).item()) if (USE_GPU and hasattr(tr, 'item')) else float(xp.real(tr))
        fidelity += w * tr_val
    result = fidelity / d_R ** 2
    if result > 1.0 + NOISE_TOLERANCE:
        warnings.warn(
            f"_compute_fidelity: fidelity {result:.6f} > 1 (before clip); "
            "possible scaling or normalization bug."
        )
    return max(0.0, min(1.0, result))


# ─────────────────────────────────────────────────────────────────────────────
# New unified power iteration
# ─────────────────────────────────────────────────────────────────────────────

def power_iteration(
    bases: List[EndSnBlockDiagonalBasis],
    M_blocks: List[np.ndarray],
    C_blocks_init: List[np.ndarray],
    picture: Literal['h', 's'] = 'h',
    max_iterations: int = DEFAULT_NUM_ITERATIONS_POWER,
    tolerance: float = DEFAULT_POWER_ACCURACY,
    verbose: bool = False,
):
    """Unified power iteration on pre-computed block matrices.

    Handles both full S_n symmetry (one basis) and subgroup S_k × S_{n-k}
    (two or more bases) via the same code path.  Block sizes and weights are
    derived from the bases; the caller provides pre-converted block matrices.

    Args:
        bases: List of EndSnBlockDiagonalBasis objects.  **bases[0] must be the
               reference system R** (typically TrivialAlgebraIsomorphism(d_R).basis_to).
               d_R is derived as bases[0].block_sizes[0].  Remaining entries are
               the S_n symmetry factors (e.g. iso_B.basis_to, iso_B_k.basis_to, …).
        M_blocks: Pre-computed M block matrices, one per block combination.
                  Typically built via tensor_product_block_diagonalization with
                  [TrivialAlgebraIsomorphism(d_R)] + isos.
        C_blocks_init: Initial C block matrices, same structure as M_blocks.
        picture: 'h' for Heisenberg (decoder, unitality), 's' for Schrödinger
                 (encoder, trace-preserving).
        max_iterations: Maximum number of power iterations.
        tolerance: Convergence tolerance (absolute and relative).
        verbose: Print fidelity at each iteration.

    Returns:
        (final_fidelity, final_C_blocks, num_iterations, time_elapsed)
    """
    # d_R is the single block size of the trivial R basis (bases[0]).
    
    # Block info is computed over ALL bases (including R), which naturally gives
    # block_sizes_m = d_R * m_λ and weights = 1 * f_λ for each block.
    # We then strip the d_R factor so that normalization functions receive m_λ.

    #For each block combination (one partition per basis), m = product of m_λ_i
    #and w = product of f_λ_i (SYT counts).  Ordering matches
    #tensor_product_block_diagonalization (first basis = outermost loop).

    d_R = bases[0].block_sizes[0]
    block_sizes_full = [1]
    weights = [1]
    for b in bases:
        new_block_sizes = []
        new_weights = []
        for m_prev, w_prev in zip(block_sizes_full, weights):
            for m_lam, partition in zip(b.block_sizes, b.partitions):
                new_block_sizes.append(m_prev * m_lam)
                new_weights.append(w_prev * partition.count_standard_tableaux())
        block_sizes_full = new_block_sizes
        weights = new_weights

    block_sizes_m = [s // d_R for s in block_sizes_full]

    M_blocks = [hermitianize(xp.asarray(M)) for M in M_blocks]
    C_blocks = [hermitianize(xp.asarray(C)) for C in C_blocks_init]

    prev_fidelity = _compute_fidelity(M_blocks, C_blocks, weights, d_R)
    start_time = time.time()
    num_iter = 0
    new_fidelity = prev_fidelity

    if verbose:
        print(f"Initial fidelity: {float(prev_fidelity)}")

    for iteration in range(max_iterations):
        C_blocks_prev = C_blocks
        C_new_blocks = [M @ C @ M for M, C in zip(M_blocks, C_blocks)]
        C_new_blocks = _normalize_blocks(C_new_blocks, block_sizes_m, weights, d_R, picture)

        # Enforce Hermiticity + PSD projection per block
        C_blocks = []
        for C in C_new_blocks:
            C = hermitianize(C)
            eigs, vecs = xp.linalg.eigh(C)
            C_blocks.append(
                (vecs * xp.maximum(xp.real(eigs), 0.0).astype(C.dtype)[None, :]) @ vecs.conj().T
            )

        new_fidelity = _compute_fidelity(M_blocks, C_blocks, weights, d_R)
        num_iter = iteration + 1

        if verbose:
            print(f"Power Iteration  {num_iter}  Fidelity:  {float(new_fidelity)}")

        if new_fidelity < prev_fidelity - tolerance:
            if verbose:
                print(f"  Fidelity decreased ({prev_fidelity:.8f} -> {new_fidelity:.8f}), reverting")
            C_blocks = C_blocks_prev
            new_fidelity = prev_fidelity
            break

        abs_diff = abs(new_fidelity - prev_fidelity)
        rel_diff = abs_diff / (abs(prev_fidelity) + 1e-12)
        if abs_diff < tolerance or rel_diff < tolerance * 100:
            break

        prev_fidelity = new_fidelity

    return new_fidelity, C_blocks, num_iter, time.time() - start_time


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient optimization functions
# ─────────────────────────────────────────────────────────────────────────────

def recovery_coefficient(
    c_M,
    c_D_adj_init,
    d_R: int,
    isos: Sequence[EndSnAlgebraIsomorphism],
    verbose: bool = False,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    use_warmstart: bool = True,
):
    """Optimize decoder coefficients using power iteration (Heisenberg picture).

    Handles both full S_n symmetry (isos = [iso_B]) and subgroup S_k × S_{n-k}
    (isos = [iso_B_k, iso_B_nmk]) via a unified code path.

    Args:
        c_M: Flat orbit-basis coefficient vector of M.
        c_D_adj_init: Initial decoder (adjoint) coefficient vector (orbit basis).
        d_R: Reference system dimension.
        isos: Sequence of algebra isomorphisms (one per symmetry factor).
        verbose: Print per-iteration fidelities.
        power_max_iterations: Maximum power-method iterations.
        power_tolerance: Convergence tolerance.
        use_warmstart: Ignored (kept for API compatibility).

    Returns:
        SDPResult(fidelity, time=elapsed, optimizers=c_D_adj)
    """
    from ..utilities.sdp_result import SDPResult

    trivial_iso = TrivialAlgebraIsomorphism(d_R)
    iso_list = [trivial_iso] + list(isos)
    M_blocks = tensor_product_block_diagonalization(c_M, iso_list)
    C_blocks_init = tensor_product_block_diagonalization(c_D_adj_init, iso_list)
    bases = [trivial_iso.basis_to] + [iso.basis_to for iso in isos]

    max_iter = power_max_iterations if power_max_iterations is not None else DEFAULT_NUM_ITERATIONS_POWER
    tol = power_tolerance if power_tolerance is not None else DEFAULT_POWER_ACCURACY

    with MaybeExpensiveComputation("Power iteration (decoder)"):
        fidelity, C_blocks_final, num_iter, elapsed = power_iteration(
            bases, M_blocks, C_blocks_init,
            picture='h', max_iterations=max_iter, tolerance=tol, verbose=verbose,
        )

    c_D_adj = tensor_product_inverse_block_diagonalization(C_blocks_final, iso_list).ravel()
    return SDPResult(fidelity, time=elapsed, optimizers=c_D_adj)


def preparation_coefficient(
    c_M,
    c_E_init,
    d_R: int,
    isos: Sequence[EndSnAlgebraIsomorphism],
    verbose: bool = False,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    use_warmstart: bool = True,
):
    """Optimize encoder coefficients using power iteration (Schrödinger picture).

    Handles both full S_n symmetry (isos = [iso_A]) and subgroup S_k × S_{n-k}
    (isos = [iso_A_k, iso_A_nmk]) via a unified code path.

    Args:
        c_M: Flat orbit-basis coefficient vector of M.
        c_E_init: Initial encoder coefficient vector (orbit basis).
        d_R: Reference system dimension.
        isos: Sequence of algebra isomorphisms (one per symmetry factor).
        verbose: Print per-iteration fidelities.
        power_max_iterations: Maximum power-method iterations.
        power_tolerance: Convergence tolerance.
        use_warmstart: Ignored (kept for API compatibility).

    Returns:
        SDPResult(fidelity, time=elapsed, optimizers=c_E)
    """
    from ..utilities.sdp_result import SDPResult

    trivial_iso = TrivialAlgebraIsomorphism(d_R)
    iso_list = [trivial_iso] + list(isos)
    M_blocks = tensor_product_block_diagonalization(c_M, iso_list)
    C_blocks_init = tensor_product_block_diagonalization(c_E_init, iso_list)
    bases = [trivial_iso.basis_to] + [iso.basis_to for iso in isos]

    max_iter = power_max_iterations if power_max_iterations is not None else DEFAULT_NUM_ITERATIONS_POWER
    tol = power_tolerance if power_tolerance is not None else DEFAULT_POWER_ACCURACY

    with MaybeExpensiveComputation("Power iteration (encoder)"):
        fidelity, C_blocks_final, num_iter, elapsed = power_iteration(
            bases, M_blocks, C_blocks_init,
            picture='s', max_iterations=max_iter, tolerance=tol, verbose=verbose,
        )

    c_E = tensor_product_inverse_block_diagonalization(C_blocks_final, iso_list).ravel()
    return SDPResult(fidelity, time=elapsed, optimizers=c_E)


__all__ = [
    'power_iteration',
    'recovery_coefficient',
    'preparation_coefficient',
]


def hermitianize(X):
    """Return (X + X†) / 2, enforcing exact Hermiticity.

    Used to correct numerical drift after matrix multiplications that should
    preserve Hermitianness but accumulate floating-point asymmetry.

    Args:
        X: Square matrix (xp array — GPU if USE_GPU, CPU otherwise).

    Returns:
        Hermitian matrix (X + X†) / 2, same shape and device as X.
    """
    return 0.5 * (X + X.conj().T)


def matrix_inverse_sqrt(A, eps=1e-12):
    """
    Compute the pseudoinverse square root of a Hermitian PSD matrix.

    Uses float64 for eigendecomposition to avoid precision loss (important for
    large n and ill-conditioned C_S). Result is cast back to input dtype.

    Eigenvalues below ``eps`` (including negative numerical noise) are treated
    as belonging to the null space: their inverse-sqrt is set to **zero** so
    the normalization projects into the supported subspace rather than blowing
    up.  This is essential for the Heisenberg/Schrödinger normalization when
    ``Tr_R(C)`` or ``Tr_S(C)`` is rank-deficient (common after the power step
    concentrates the iterate on the dominant eigenvector).

    Args:
        A: Hermitian matrix (xp array - GPU if USE_GPU, CPU otherwise)
        eps: Threshold below which eigenvalues are considered zero (default 1e-12).

    Returns:
        A^{-1/2} (pseudoinverse square root, xp array)
    """
    A = xp.asarray(A)
    out_dtype = A.dtype
    # Use float64 for eigh/inv_sqrt to avoid precision loss and underflow
    A_f64 = A.astype(xp.complex128) if xp.issubdtype(A.dtype, xp.complexfloating) else A.astype(xp.float64)
    A_f64 = hermitianize(A_f64)

    eigvals, eigvecs = xp.linalg.eigh(A_f64)
    eigvals_real = eigvals.real
    # Pseudoinverse: eigenvalues below eps are in the null space → inv_sqrt = 0
    # Clamp before sqrt to avoid divide-by-zero / invalid-value RuntimeWarnings
    safe_eigvals = xp.maximum(eigvals_real, eps)
    inv_sqrt_vals = xp.where(
        eigvals_real > eps,
        1.0 / xp.sqrt(safe_eigvals.astype(eigvecs.dtype)),
        xp.zeros_like(eigvals_real, dtype=eigvecs.dtype),
    )
    result = (eigvecs * inv_sqrt_vals) @ eigvecs.conj().T
    return result.astype(out_dtype)
