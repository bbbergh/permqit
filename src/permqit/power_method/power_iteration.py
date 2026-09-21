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
from ..utilities import backend
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
    xp = backend.xp
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
        tr = backend.xp.trace(M @ C)
        tr_val = float(backend.xp.real(tr).item()) if (backend.USE_GPU and hasattr(tr, 'item')) else float(backend.xp.real(tr))
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
    skip_zero_M_blocks: bool = False,
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
        skip_zero_M_blocks: Heisenberg picture only.  Blocks whose M^lambda vanishes contribute
            nothing to the fidelity and are left completely unconstrained by the optimization; the
            iteration drives them to the canonical unital point I_R (x) I_m / d_R (M @ C @ M = 0,
            followed by the null-space fill in ``_normalize_blocks``).  Setting this fixes them
            there directly and skips all work on them, which is exactly equivalent and is a large
            saving when the encoder is supported on the symmetric block only -- then the channel's
            selection rule forces most output blocks to vanish (e.g. 240 of 330 at n=17 for the
            flagged Pauli channel).

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

    M_blocks = [hermitianize(backend.xp.asarray(M)) for M in M_blocks]
    C_blocks = [hermitianize(backend.xp.asarray(C)) for C in C_blocks_init]

    if skip_zero_M_blocks and picture != 'h':
        raise ValueError("skip_zero_M_blocks is only valid in the Heisenberg ('h') picture, "
                         "where the normalization decouples block by block.")
    if skip_zero_M_blocks:
        active = [bool(M.size) and float(backend.xp.max(backend.xp.abs(M))) > NOISE_TOLERANCE
                  for M in M_blocks]
        for i, (act, m) in enumerate(zip(active, block_sizes_m)):
            if not act and m > 0:
                # the unique point the full iteration converges to when M^lambda = 0
                C_blocks[i] = backend.xp.eye(d_R * m, dtype=C_blocks[i].dtype) / d_R
        idx = [i for i, act in enumerate(active) if act]
    else:
        idx = list(range(len(M_blocks)))

    prev_fidelity = _compute_fidelity(M_blocks, C_blocks, weights, d_R)
    start_time = time.time()
    num_iter = 0
    new_fidelity = prev_fidelity

    if verbose:
        print(f"Initial fidelity: {float(prev_fidelity)}")

    for iteration in range(max_iterations):
        C_blocks_prev = C_blocks
        C_new_blocks = [M_blocks[i] @ C_blocks[i] @ M_blocks[i] for i in idx]
        C_new_blocks = _normalize_blocks(
            C_new_blocks, [block_sizes_m[i] for i in idx], [weights[i] for i in idx], d_R, picture
        )

        # Enforce Hermiticity + PSD projection per block
        C_blocks = list(C_blocks_prev)
        for i, C in zip(idx, C_new_blocks):
            C = hermitianize(C)
            eigs, vecs = _eigh_psd_project(C)
            if eigs is None:
                continue  # keep the previous iterate for this block; see _eigh_psd_project
            C_blocks[i] = (
                vecs * backend.xp.maximum(backend.xp.real(eigs), 0.0).astype(C.dtype)[None, :]
            ) @ vecs.conj().T

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
    skip_zero_M_blocks: bool = False,
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
            skip_zero_M_blocks=skip_zero_M_blocks,
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


def isometric_preparation_coefficient(
    c_M,
    c_E_init,
    d_R: int,
    isos: Sequence[EndSnAlgebraIsomorphism],
    verbose: bool = False,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    use_warmstart: bool = True,
):
    """Encoder half-step restricted to *genuine* isometries V : C^d_R -> Sym^n(A).

    A permutation-invariant encoder is an isometry iff its Choi matrix is rank one, which (for
    n > d_A) forces it into the single multiplicity-free block lambda = (n).  Writing
    |v> = sum_i |i>_R (x) V|i>, the half-step is therefore

        maximise  <v| M^(n) |v>   subject to   V^dagger V = I_{d_R},

    a convex quadratic on the Stiefel manifold whenever M^(n) >= 0, which holds in every real use
    since M is the Choi matrix of a CP map.  We solve it by the polar ("Procrustes") ascent
    K <- polar(dF/dKbar), which increases the objective monotonically because a convex function
    dominates its linearisation and the linearised problem is solved exactly by the polar factor.
    (Monotonicity, like that of the generic power step, relies on M >= 0; it is not checked here
    because doing so on every call would be needlessly expensive.)

    This is *the same map* as the generic Schroedinger-picture power step restricted to the
    lambda = (n) block: there M @ C @ M followed by the trace-preserving normalization
    (T^{-1/2} (x) I) . (T^{-1/2} (x) I) is exactly G -> G (G^dagger G)^{-1/2} = polar(G).  Using the
    SVD-based polar factor directly is numerically better conditioned and cannot trigger the
    null-space fill, which is what lets the isometric path converge to machine precision.

    Args:
        c_M: Flat orbit-basis coefficient vector of M (the adjoint-side AlicePOV operator).
        c_E_init: Initial encoder coefficients; only its lambda = (n) block is used as a warm start.
        d_R: Reference system dimension.
        isos: A single-element sequence [iso_A].
        verbose: Print per-iteration fidelities.
        power_max_iterations: Maximum ascent steps.
        power_tolerance: Stop when the objective improves by less than this.
        use_warmstart: Ignored (kept for API compatibility).

    Returns:
        SDPResult(fidelity, time=elapsed, optimizers=c_E), with c_E the Choi coefficients of an
        exact isometry.
    """
    import time as _time

    from ..utilities.sdp_result import SDPResult
    from ..utilities.random import symmetric_block_index, isometry_into_symmetric_block

    if len(isos) != 1:
        raise ValueError(
            f"The isometric encoder step needs exactly one symmetry factor, got {len(isos)}."
        )
    iso_A = isos[0]
    trivial_iso = TrivialAlgebraIsomorphism(d_R)
    iso_list = [trivial_iso, iso_A]
    sym_idx = symmetric_block_index(iso_A)
    m_sym = iso_A.basis_to.block_sizes[sym_idx]

    M = hermitianize(backend.xp.asarray(tensor_product_block_diagonalization(c_M, iso_list)[sym_idx]))

    C_init = tensor_product_block_diagonalization(c_E_init, iso_list)[sym_idx]
    K = _polar(backend.to_cpu(_leading_isometry(C_init, d_R, m_sym)))

    max_iter = power_max_iterations if power_max_iterations is not None else DEFAULT_NUM_ITERATIONS_POWER
    tol = power_tolerance if power_tolerance is not None else DEFAULT_POWER_ACCURACY

    M_cpu = backend.to_cpu(M)
    start = _time.time()
    previous = -np.inf
    objective = previous
    with MaybeExpensiveComputation("Polar ascent (isometric encoder)"):
        for iteration in range(max_iter):
            v = K.T.ravel()
            Mv = M_cpu @ v
            objective = float(np.real(np.vdot(v, Mv)))
            if verbose:
                print(f"Polar ascent {iteration}  Fidelity: {objective / d_R ** 2}")
            if objective - previous < tol:
                break
            previous = objective
            K = _polar(Mv.reshape(d_R, m_sym).T)
    elapsed = _time.time() - start

    v = K.T.ravel()
    fidelity = float(np.real(np.vdot(v, M_cpu @ v))) / d_R ** 2
    c_E = isometry_into_symmetric_block(K, d_R, iso_A, xp=backend.xp)
    return SDPResult(max(0.0, min(1.0, fidelity)), time=elapsed, optimizers=c_E)


def _polar(G):
    """Polar factor of G: the isometry maximising Re Tr(G^dagger K) over K^dagger K = I."""
    U, _, Vh = np.linalg.svd(np.asarray(G), full_matrices=False)
    return U @ Vh


def _leading_isometry(C, d_R: int, m_sym: int):
    """Extract a warm-start isometry from an arbitrary (possibly higher-rank) block C."""
    C = backend.to_cpu(backend.xp.asarray(C))
    C = 0.5 * (C + C.conj().T)
    eigs, vecs = np.linalg.eigh(C)
    v = vecs[:, int(np.argmax(np.real(eigs)))]
    return v.reshape(d_R, m_sym).T


__all__ = [
    'power_iteration',
    'recovery_coefficient',
    'preparation_coefficient',
    'isometric_preparation_coefficient',
]


def _eigh_psd_project(C):
    """``eigh`` for the per-block PSD projection, tolerant of LAPACK non-convergence.

    LAPACK occasionally fails to converge on a badly scaled block (the normalization steps divide
    by pseudo-inverse square roots, which can leave a block spanning many orders of magnitude).
    Rescaling to unit max-norm almost always fixes it; if it still fails we return ``(None, None)``
    and the caller keeps the previous iterate for that block, which costs one iteration of
    progress instead of aborting a multi-hour run.

    Returns ``(eigenvalues, eigenvectors)``, or ``(None, None)`` if the decomposition failed.
    """
    try:
        return backend.xp.linalg.eigh(C)
    except np.linalg.LinAlgError:
        pass
    try:
        scale = float(backend.xp.max(backend.xp.abs(C)))
        if scale > 0:
            eigs, vecs = backend.xp.linalg.eigh(C / scale)
            return eigs * scale, vecs
    except np.linalg.LinAlgError:
        pass
    warnings.warn(
        "power_iteration: eigh failed to converge on a block even after rescaling; "
        "keeping the previous iterate for it."
    )
    return None, None


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
    A = backend.xp.asarray(A)
    out_dtype = A.dtype
    # Use float64 for eigh/inv_sqrt to avoid precision loss and underflow
    A_f64 = A.astype(backend.xp.complex128) if backend.xp.issubdtype(A.dtype, backend.xp.complexfloating) else A.astype(backend.xp.float64)
    A_f64 = hermitianize(A_f64)

    eigvals, eigvecs = backend.xp.linalg.eigh(A_f64)
    eigvals_real = eigvals.real
    # Pseudoinverse: eigenvalues below eps are in the null space → inv_sqrt = 0
    # Clamp before sqrt to avoid divide-by-zero / invalid-value RuntimeWarnings
    safe_eigvals = backend.xp.maximum(eigvals_real, eps)
    inv_sqrt_vals = backend.xp.where(
        eigvals_real > eps,
        1.0 / backend.xp.sqrt(safe_eigvals.astype(eigvecs.dtype)),
        backend.xp.zeros_like(eigvals_real, dtype=eigvecs.dtype),
    )
    result = (eigvecs * inv_sqrt_vals) @ eigvecs.conj().T
    return result.astype(out_dtype)
