"""
Unified seesaw optimization for permutation-invariant quantum channels.

Two cases are supported:

1. **Unflagged channel** (``N^{⊗n}``): the input system A^n and output system B^n
   both carry full S_n symmetry.  A single encoder E (A^n → R) and single decoder
   D (B^n → R) are optimized.

2. **Flagged-output channel** (``⊕_k N_k``, e.g. superactivation): the output B^n
   is flagged by a composition (k_0, …, k_{m-1}) with k_i ≥ 0 and Σ k_i = n.
   Sector (k_0, …, k_{m-1}) applies N_i^{⊗k_i} to copies of type i.  The encoder
   E (A^n → R) is shared; one decoder D_{k_0,…} per sector is optimized using
   S_{k_0} × … × S_{k_{m-1}} subgroup symmetry.  The encoder step uses
   aggregation + Hermitian resymmetrization.

   Sparsity is exploited automatically from the support of each single-copy Choi
   matrix: if N_i has |supp(N_i)| < d_AB² nonzero entries, only
   C(k_i + |supp(N_i)| - 1, k_i) orbit coefficients are nonzero for type i.

The input (encoder) side always carries full S_n symmetry.  All representation
details (isomorphisms, orbit bases, partial trace relations, sector weights) are
derived internally.

Primary entry point: ``compute_tensor_product_fidelity_seesaw_block``.
Low-level entry point: ``compute_fidelity_seesaw`` (for custom channel structures).

Per-sector decoders (inside ``compute_fidelity_seesaw``) are passed as a tuple of
``EndSnAlgebraIsomorphism`` objects, one per channel type (length m).  A 1-tuple is
treated as full S_n symmetry; longer tuples use the subgroup product.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..algebra import EndSnOrbitBasis
from ..algebra.basis import MatrixStandardBasis
from ..algebra.basis_subset import MatrixEntryMask, MatrixStandardBasisSubset, IndexIsValidPredicate
from ..algebra.endomorphism_direct_sum_basis import EndSnSingleBlockOrbitBasis
from ..representation.isomorphism import EndSnAlgebraIsomorphism
from ..representation.partial_traces import BasePartialTraceRelations
from ..representation.combinatorics import multinomial_coeff, weak_compositions
from ..utilities import backend
from ..utilities.backend import xp
from ..utilities.timing import MaybeExpensiveComputation
from ..utilities.sdp_result import SDPResult
from . import power_iteration
from ..SDP.seesaw_utils import (
    random_perm_inv_encoder,
    random_perm_inv_decoder,
    get_coefficient_AlicePOV_from_relation,
    get_coefficient_BobPOV_from_relation,
    get_coefficient_adjoint_SR,
    get_coefficient_adjoint_general_RS,
    get_coefficient_adjoint_general_SR,
)

DEFAULT_SEESAW_ACCURACY = 1e-7
DEFAULT_SEESAW_REPETITIONS = 50
DEFAULT_SEESAW_ITERATIONS = 500
DEFAULT_ISOMETRY = True
DEFAULT_SEED = 18

# Each sector's decoder symmetry is described by a tuple of EndSnAlgebraIsomorphism,
# one per channel type (length m).  A 1-tuple means full S_n; longer = subgroup product.
IsoTuple = Tuple[EndSnAlgebraIsomorphism, ...]


# ---------------------------------------------------------------------------
# Per-sector decoder helpers
# ---------------------------------------------------------------------------


def _init_decoder(iso_tuple: IsoTuple, d_R: int, isometry: bool, seed: int) -> NDArray:
    """Return a random initial decoder in SR format for one sector."""
    return random_perm_inv_decoder(list(iso_tuple), d_R, isometry=isometry, seed=seed)


def _optimize_decoder(
    c_M_k: NDArray,
    c_D_k: NDArray,
    d_R: int,
    iso_tuple: IsoTuple,
    power_max_iter: int,
    power_tol: float,
):
    """Optimize decoder for one sector via power iteration.

    Args:
        c_M_k: BobPOV M matrix in RS format ``(d_R² * m_B,)``.
        c_D_k: Current decoder in SR format.
        d_R: Reference dimension.
        iso_tuple: Tuple of m EndSnAlgebraIsomorphism, one per channel type.
        power_max_iter: Max power-method iterations.
        power_tol: Power-method convergence tolerance.

    Returns:
        (F_D_k, c_D_k_new, elapsed_time)
    """
    bases = [iso.basis_from for iso in iso_tuple]
    c_D_adj_init = get_coefficient_adjoint_general_SR(c_D_k, d_R, bases)
    result = power_iteration.recovery_coefficient(
        c_M_k,
        c_D_adj_init,
        d_R,
        isos=list(iso_tuple),
        power_max_iterations=power_max_iter,
        power_tolerance=power_tol,
    )
    c_D_adj = result.get_optimizers()
    if isinstance(c_D_adj, (list, tuple)):
        c_D_adj = c_D_adj[0]
    c_D_k_new = get_coefficient_adjoint_general_RS(c_D_adj, d_R, bases)
    return result.get_value(), c_D_k_new, result.get_time()


# ---------------------------------------------------------------------------
# Main seesaw function
# ---------------------------------------------------------------------------


def compute_fidelity_seesaw(
    c_N_sectors: List[NDArray],
    basisB_sectors: List,
    partial_trace_sectors: List[BasePartialTraceRelations],
    iso_A: EndSnAlgebraIsomorphism,
    output_sector_isos: List[IsoTuple],
    d_R: int,
    *,
    q: Union[float, List[float]] = 0.5,
    sector_alice_weights: Optional[List[float]] = None,
    sector_fidelity_weights: Optional[List[float]] = None,
    repetitions: int = DEFAULT_SEESAW_REPETITIONS,
    iterations: int = DEFAULT_SEESAW_ITERATIONS,
    seesaw_accuracy: float = DEFAULT_SEESAW_ACCURACY,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    isometry: bool = DEFAULT_ISOMETRY,
    print_iterations: bool = False,
    verbose: bool = False,
    return_optimizers: bool = True,
    timing_analysis: bool = True,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Optional[str] = None,
    checkpoint_threshold: float = 0.75,
) -> SDPResult:
    """Low-level seesaw over pre-built orbit-basis channel representations.

    Handles unflagged (n_sectors=1) and flagged-output (n_sectors>1) channels
    via the same code path.  The input A^n always carries full S_n symmetry
    (single encoder); each output sector has its own decoder.

    Prefer ``compute_tensor_product_fidelity_seesaw_block`` unless you need
    custom bases or partial-trace relations.

    Parameters
    ----------
    c_N_sectors:
        Channel Choi coefficients in orbit basis, one array per output sector.
    basisB_sectors:
        Decoder-input orbit basis for each sector.
    partial_trace_sectors:
        ``BasePartialTraceRelations`` objects, one per sector.
    iso_A:
        Algebra isomorphism for the encoder's output system (A^n).
    output_sector_isos:
        Per-sector decoder descriptor — a tuple of ``EndSnAlgebraIsomorphism``
        objects, one per channel type.  A 1-tuple means full S_n symmetry; a
        longer tuple means the subgroup product S_{k_0} × … × S_{k_{m-1}}.
    d_R:
        Reference system dimension.
    q:
        Mixing probability (float for 2-type channels, list for m-type).
        Used only to compute default sector weights.
    sector_alice_weights:
        Weights for Alice POV aggregation ``M_avg = Σ_k w_k M_enc_k``.
    sector_fidelity_weights:
        Weights for fidelity averaging ``F_D_avg = Σ_k p_k F_D_k``.
    repetitions:
        Number of random restarts.
    iterations:
        Maximum seesaw iterations per restart.
    seesaw_accuracy:
        Stop seesaw when ``|F_E - F_D_avg| < seesaw_accuracy``.
    power_max_iterations:
        Maximum power-method iterations per seesaw step.
    power_tolerance:
        Power-method convergence tolerance.
    isometry:
        Initialize encoder as isometry if True.
    print_iterations:
        Print fidelity estimates after each seesaw iteration.
    verbose:
        Print detailed progress messages.
    return_optimizers:
        Return best encoder and decoder list in ``SDPResult``.
    timing_analysis:
        Return total power-method time in ``SDPResult``.
    seed:
        Base random seed; seed for repetition ``rep`` is ``seed + rep * seed``.
    checkpoint_path:
        Save best encoder/decoders as .npz when fidelity >= ``checkpoint_threshold``.
    checkpoint_threshold:
        Minimum fidelity to trigger a checkpoint save.

    Returns
    -------
    SDPResult
        ``.get_value()`` → best fidelity (float).
        ``.get_optimizers()`` → ``(c_E, [c_D_k])`` when ``return_optimizers=True``.
        ``.get_time()`` → total power-method time when ``timing_analysis=True``.
    """
    n_sectors = len(c_N_sectors)
    if n_sectors == 0:
        raise ValueError("c_N_sectors must have at least one element.")
    if len(basisB_sectors) != n_sectors:
        raise ValueError(f"basisB_sectors length {len(basisB_sectors)} != {n_sectors}")
    if len(partial_trace_sectors) != n_sectors:
        raise ValueError(f"partial_trace_sectors length {len(partial_trace_sectors)} != {n_sectors}")
    if len(output_sector_isos) != n_sectors:
        raise ValueError(f"output_sector_isos length {len(output_sector_isos)} != {n_sectors}")

    # ---------- Derive internal bases ----------
    basisA_orbit = iso_A.basis_from  # EndSnOrbitBasis(n, d_A)
    d_A = basisA_orbit.d
    n = basisA_orbit.n
    basisA_single = EndSnSingleBlockOrbitBasis((MatrixStandardBasis(d_A),), [n])
    basisR = MatrixStandardBasis(d_R)

    # ---------- Sector weights ----------
    if sector_alice_weights is None:
        sector_alice_weights = [1.0] if n_sectors == 1 else None
    if sector_fidelity_weights is None:
        sector_fidelity_weights = [1.0] if n_sectors == 1 else None
    # If still None (n_sectors > 1 and no explicit weights), we require the caller to provide them.
    # compute_tensor_product_fidelity_seesaw_block always passes them explicitly.
    if sector_alice_weights is None or sector_fidelity_weights is None:
        raise ValueError(
            "sector_alice_weights and sector_fidelity_weights must be provided "
            "for multi-sector channels.  Use compute_tensor_product_fidelity_seesaw_block "
            "for automatic weight computation."
        )

    if len(sector_alice_weights) != n_sectors:
        raise ValueError("sector_alice_weights length must match n_sectors.")
    if len(sector_fidelity_weights) != n_sectors:
        raise ValueError("sector_fidelity_weights length must match n_sectors.")

    # ---------- Hermitian symmetrization precomputation ----------
    t_A_xp = xp.asarray(basisA_orbit.transpose_index_lookup())
    t_R_xp = xp.asarray(np.arange(d_R**2).reshape(d_R, d_R).T.ravel())

    # ---------- Power-method hyperparameters ----------
    power_max_iter = (
        power_max_iterations if power_max_iterations is not None else power_iteration.DEFAULT_NUM_ITERATIONS_POWER
    )
    power_tol = power_tolerance if power_tolerance is not None else power_iteration.DEFAULT_POWER_ACCURACY

    # ---------- Main seesaw loop ----------
    total_time = 0.0
    fidelities: List[float] = []
    best_encoders: List = []
    best_decoders: List = []

    for rep in range(repetitions):
        rep_seed = seed + rep * seed
        if verbose:
            print(f"\n[Repetition {rep + 1}/{repetitions}] Initializing encoder and decoders...")

        with MaybeExpensiveComputation("Initializing encoder"):
            c_E = random_perm_inv_encoder(iso_A, d_R, isometry=isometry, seed=rep_seed)
        c_D_list = [_init_decoder(iso_t, d_R, isometry, rep_seed) for iso_t in output_sector_isos]

        F_E: float = 0.0
        F_D_avg: float = 0.0

        for i in range(iterations):
            # ------ Decoder step ------
            c_M_dec = [
                get_coefficient_BobPOV_from_relation(c_E, basisR, basisA_single, c_N_k, ptr)
                for c_N_k, ptr in zip(c_N_sectors, partial_trace_sectors)
            ]

            F_D_sectors: List[float] = []
            new_c_D_list = []
            t_dec = 0.0
            with MaybeExpensiveComputation(f"Optimizing {n_sectors} decoder(s)"):
                for c_M_k, c_D_k, iso_t in zip(c_M_dec, c_D_list, output_sector_isos):
                    F_D_k, c_D_k_new, t_k = _optimize_decoder(c_M_k, c_D_k, d_R, iso_t, power_max_iter, power_tol)
                    F_D_sectors.append(F_D_k)
                    new_c_D_list.append(c_D_k_new)
                    t_dec += t_k or 0.0
            c_D_list = new_c_D_list

            F_D_avg = float(sum(w * F for w, F in zip(sector_fidelity_weights, F_D_sectors)))

            # ------ Encoder step ------
            c_M_enc = [
                get_coefficient_AlicePOV_from_relation(c_N_k, c_D_k, basisB_k, basisR, ptr)
                for c_N_k, c_D_k, basisB_k, ptr in zip(c_N_sectors, c_D_list, basisB_sectors, partial_trace_sectors)
            ]

            c_M_avg = sum(  # type: ignore[assignment]
                w * c for w, c in zip(sector_alice_weights, c_M_enc)
            )

            # Hermitian symmetrization: M_avg = (M_avg + M_avg†) / 2.
            c_M_2d = c_M_avg.reshape(basisA_orbit.size(), basisR.size())
            c_M_dag = c_M_2d[t_A_xp][:, t_R_xp].conj()
            c_M_avg = (0.5 * (c_M_2d + c_M_dag)).ravel()

            with MaybeExpensiveComputation("Optimizing encoder"):
                _c_M_adj = get_coefficient_adjoint_SR(c_M_avg, d_R, iso_A.basis_from)
                _enc_result = power_iteration.preparation_coefficient(
                    _c_M_adj,
                    c_E,
                    d_R,
                    [iso_A],
                    power_max_iterations=power_max_iter,
                    power_tolerance=power_tol,
                )
                F_E = _enc_result.get_value()
                _enc_opt = _enc_result.get_optimizers()
                c_E = _enc_opt[0] if isinstance(_enc_opt, (list, tuple)) else _enc_opt
                t_enc = _enc_result.get_time()

            total_time += t_dec + (t_enc or 0.0)

            if print_iterations:
                print(
                    f"Iteration {i}: F_D_avg={F_D_avg:.6f}, F_E={F_E:.6f}"
                    + (f", gap={F_D_avg - F_E:.2e}" if i > 0 else "")
                )
            if verbose and n_sectors > 1:
                dec_str = ", ".join(f"F_D[{k}]={F:.6f}" for k, F in enumerate(F_D_sectors))
                print(f"  {dec_str}  →  F_D_avg={F_D_avg:.6f}  F_E={F_E:.6f}")

            fidelities.append(F_E)
            best_encoders.append(c_E)
            best_decoders.append(list(c_D_list))

            if abs(F_E - F_D_avg) < seesaw_accuracy:
                break

    # ---------- Select global best ----------
    fidelities_xp = xp.asarray(fidelities)
    idx_best = int(xp.argmax(fidelities_xp))
    max_fidelity = float(fidelities_xp[idx_best])
    max_encoder = best_encoders[idx_best]
    max_decoder = best_decoders[idx_best]

    # ---------- Optional checkpoint ----------
    if checkpoint_path and max_fidelity >= checkpoint_threshold:
        from ..utilities.backend import to_cpu

        save_dict: dict = {
            "encoder": np.asarray(to_cpu(max_encoder)),
            "fidelity": np.float64(max_fidelity),
            "n": np.int64(n),
        }
        for k_idx, c_D_k in enumerate(max_decoder):
            save_dict[f"decoder_{k_idx}"] = np.asarray(to_cpu(c_D_k))
        np.savez(checkpoint_path, **save_dict)
        print(f"Checkpoint saved: F={max_fidelity:.6f} → {checkpoint_path}")

    optim = (max_encoder, max_decoder) if return_optimizers else None
    time_val = total_time if timing_analysis else None
    return SDPResult(max_fidelity, time=time_val, optimizers=optim)


# ---------------------------------------------------------------------------
# User-facing entry point
# ---------------------------------------------------------------------------


def compute_tensor_product_fidelity_seesaw(
    n: int,
    d_R: int,
    N: Union[np.ndarray, List[np.ndarray]],
    d_A: int,
    d_B: int,
    *,
    q: Union[float, List[float]] = 0.5,
    repetitions: int = DEFAULT_SEESAW_REPETITIONS,
    iterations: int = DEFAULT_SEESAW_ITERATIONS,
    seesaw_accuracy: float = DEFAULT_SEESAW_ACCURACY,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    isometry: bool = DEFAULT_ISOMETRY,
    print_iterations: bool = False,
    verbose: bool = False,
    return_optimizers: bool = True,
    timing_analysis: bool = True,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Optional[str] = None,
    checkpoint_threshold: float = 0.75,
) -> SDPResult:
    """Compute entanglement fidelity for a permutation-invariant tensor-product channel.

    This is the primary entry point for the seesaw method using the power method.
    It accepts the channel as a physical Choi matrix (unflagged case) or a list of
    m single-copy Choi matrices (flagged case), and handles all internal
    representation details automatically.

    Parameters
    ----------
    n:
        Number of channel copies.
    d_R:
        Reference system dimension.
    N:
        Channel specification — one of two forms:

        * ``np.ndarray`` of shape ``(d_A*d_B, d_A*d_B)`` — single-copy Choi matrix
          of an **unflagged** channel.  The n-copy channel is ``N^{⊗n}`` with full
          S_n symmetry on both A^n and B^n.

        * ``[N_0, N_1, …, N_{m-1}]`` — a list of m single-copy Choi matrices for a
          **flagged** channel with m types.  Sector (k_0, …, k_{m-1}) applies
          ``N_i^{⊗k_i}`` to k_i copies of type i (Σ k_i = n).  The sector weight is
          ``C(n; k_0,…,k_{m-1}) · q_0^{k_0} · … · q_{m-1}^{k_{m-1}}``.

          **Automatic sparsity**: if ``N_i`` has fewer than ``d_A²·d_B²`` nonzero
          entries, the orbit basis for that type is automatically restricted to the
          nonzero support, giving a speedup for large n.

    d_A:
        Single-copy input dimension.
    d_B:
        Single-copy output dimension.
    q:
        Mixing probability / probabilities.

        * For m=2: a single float ``q`` maps to ``[1-q, q]``.
        * For m≥3: a list of m floats summing to 1; defaults to equal mixing
          ``[1/m, …, 1/m]`` if a single float is passed.

        Ignored for the unflagged case.
    repetitions, iterations, seesaw_accuracy, power_max_iterations,
    power_tolerance, isometry, print_iterations, verbose, return_optimizers,
    timing_analysis, seed, checkpoint_path, checkpoint_threshold:
        Standard seesaw hyperparameters (see ``compute_fidelity_seesaw``).

    Returns
    -------
    SDPResult
        ``.get_value()`` → best entanglement fidelity (float).
        ``.get_optimizers()`` → ``(c_E, [c_D_k])`` when ``return_optimizers=True``.
        ``.get_time()`` → total power-method time when ``timing_analysis=True``.

    Examples
    --------
    Unflagged depolarizing channel:

    >>> result = compute_tensor_product_fidelity_seesaw(
    ...     n=4, d_R=2, N=J_depolarizing, d_A=2, d_B=2,
    ... )

    Flagged (superactivation-style) channel with 2 types:

    >>> result = compute_tensor_product_fidelity_seesaw(
    ...     n=4, d_R=2, N=[J_erasure, J_dephasing], d_A=2, d_B=2, q=0.5,
    ... )

    Flagged channel with 3 types (equal mixing):

    >>> result = compute_tensor_product_fidelity_seesaw(
    ...     n=4, d_R=2, N=[J_A, J_B, J_C], d_A=2, d_B=2,
    ... )
    """
    _seesaw_kwargs = dict(
        repetitions=repetitions,
        iterations=iterations,
        seesaw_accuracy=seesaw_accuracy,
        power_max_iterations=power_max_iterations,
        power_tolerance=power_tolerance,
        isometry=isometry,
        print_iterations=print_iterations,
        verbose=verbose,
        return_optimizers=return_optimizers,
        timing_analysis=timing_analysis,
        seed=seed,
        checkpoint_path=checkpoint_path,
        checkpoint_threshold=checkpoint_threshold,
    )

    # ==================================================================
    # CASE 1 — unflagged: single-copy Choi N, channel = N^{⊗n}
    # Redirect to the flagged path (m=1) to enable automatic sparsity
    # detection via EndSnSingleBlockOrbitBasis.
    # ==================================================================
    if isinstance(N, np.ndarray):
        return compute_tensor_product_fidelity_seesaw(n, d_R, [N], d_A, d_B, q=q, **_seesaw_kwargs)

    # ==================================================================
    # CASE 2 — flagged: list of m Choi matrices
    # ==================================================================
    if not isinstance(N, (list, tuple)) or len(N) == 0:
        raise ValueError(
            "N must be either a single np.ndarray (unflagged) or a non-empty "
            "list of single-copy Choi matrices (flagged)."
        )

    N_list = [np.asarray(Ni) for Ni in N]
    m = len(N_list)

    from ..algebra import EndSnOrbitBasis
    from ..representation.partial_traces import SingleBlockPartialTraceRelations
    from ..representation.isomorphism import EndSnBlockDiagonalization

    d_AB = d_A * d_B

    # ---- Parse mixing probabilities ----
    if isinstance(q, (int, float)):
        if m == 2:
            q_vec = [1.0 - float(q), float(q)]
        else:
            q_vec = [1.0 / m] * m
    else:
        q_vec = [float(qi) for qi in q]
    if len(q_vec) != m:
        raise ValueError(f"q must have length m={m}, got {len(q_vec)}.")

    # ---- Per-type support masks for automatic sparsity ----
    tol = 1e-12
    ab_single_bases = []
    for Ni in N_list:
        support_pairs = [(a, b) for a in range(d_AB) for b in range(d_AB) if abs(Ni[a, b]) > tol]
        if len(support_pairs) < d_AB * d_AB:
            mask = MatrixEntryMask((d_AB, d_AB), support_pairs)
            ab_single_bases.append(MatrixStandardBasisSubset(MatrixStandardBasis(d_AB), IndexIsValidPredicate(mask)))
            if verbose:
                print(
                    f"  [Sparsity] Type {N_list.index(Ni)}: "
                    f"|support|={len(support_pairs)} < {d_AB**2}; "
                    f"using sparse AB orbit basis."
                )
        else:
            ab_single_bases.append(MatrixStandardBasis(d_AB))

    # ---- Shared encoder iso ----
    iso_A = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d_A))

    # ---- Input (A) single-block basis, shared across all sectors ----
    block_basisA = EndSnSingleBlockOrbitBasis((MatrixStandardBasis(d_A),), [n])

    # ---- Pre-build iso_B for each possible copy count (0..n) ----
    iso_B_for_k = {k: EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(k, d_B)) for k in range(n + 1)}

    # ---- Per-type orbit-basis coefficient vectors for each copy count ----
    # _get_type_coeffs(i, k) returns the orbit-basis coefficient vector for N_i^{⊗k},
    # restricted to the support of N_i when ab_single_bases[i] is sparse.
    # This ensures the size matches block_basisAB (used in SingleBlockPartialTraceRelations).
    _coeffs_cache: dict = {}

    def _get_type_coeffs(i: int, k: int) -> np.ndarray:
        if (i, k) not in _coeffs_cache:
            if k == 0:
                _coeffs_cache[(i, k)] = np.array([1.0 + 0j])
            else:
                # Use the single-block AB basis for type i to get the orbit basis.
                # When ab_single_bases[i] is sparse, this is an EndSnOrbitBasisSubset
                # restricted to the support of N_i — its size is C(k+|support_i|-1, k).
                single_block = EndSnSingleBlockOrbitBasis((ab_single_bases[i],), [k])
                orbit_basis = single_block.bases[0]  # EndSnOrbitBasis or EndSnOrbitBasisSubset
                _coeffs_cache[(i, k)] = np.array(
                    orbit_basis.coefficients_for_tensor_product(N_list[i]),
                    dtype=np.complex128,
                )
        return _coeffs_cache[(i, k)]

    # ---- Build per-sector data ----
    c_N_sectors: List[np.ndarray] = []
    basisB_sectors = []
    partial_trace_sectors = []
    output_sector_isos: List[IsoTuple] = []
    sector_alice_weights = []
    sector_fidelity_weights = []

    for composition in weak_compositions(n, m):
        # ---- Channel coefficients for this sector ----
        # c_N_sector = ⊗_i c_{N_i}^{⊗k_i}  (outer product of per-type coeff arrays)
        factor_coeffs = [_get_type_coeffs(i, k_i) for i, k_i in enumerate(composition)]
        c_sector = factor_coeffs[0]
        for f in factor_coeffs[1:]:
            c_sector = np.outer(c_sector, f).ravel()
        c_N_sectors.append(backend.xp.asarray(c_sector, dtype=backend.xp.complex128))

        # ---- B basis for this sector (decoder input) ----
        block_basisB = EndSnSingleBlockOrbitBasis(
            tuple(MatrixStandardBasis(d_B) for _ in range(m)),
            list(composition),
        )
        basisB_sectors.append(block_basisB)

        # ---- AB basis for this sector (sparse per type) ----
        block_basisAB = EndSnSingleBlockOrbitBasis(
            tuple(ab_single_bases[i] for i in range(m)),
            list(composition),
        )
        partial_trace_sectors.append(SingleBlockPartialTraceRelations(block_basisA, block_basisB, block_basisAB))

        # ---- Decoder iso tuple ----
        output_sector_isos.append(tuple(iso_B_for_k[k_i] for k_i in composition))

        # ---- Sector weights ----
        alice_w = 1.0
        for qi, ki in zip(q_vec, composition):
            alice_w *= qi**ki
        sector_alice_weights.append(alice_w)
        sector_fidelity_weights.append(float(multinomial_coeff(composition)) * alice_w)

    return compute_fidelity_seesaw(
        c_N_sectors=c_N_sectors,
        basisB_sectors=basisB_sectors,
        partial_trace_sectors=partial_trace_sectors,
        iso_A=iso_A,
        output_sector_isos=output_sector_isos,
        d_R=d_R,
        sector_alice_weights=sector_alice_weights,
        sector_fidelity_weights=sector_fidelity_weights,
        **_seesaw_kwargs,
    )


def compute_tensor_product_fidelity_without_symmetry_seesaw(
    n: int,
    d_R: int,
    N: np.ndarray,
    d_A: int,
    d_B: int,
    *,
    repetitions: int = DEFAULT_SEESAW_REPETITIONS,
    iterations: int = DEFAULT_SEESAW_ITERATIONS,
    seesaw_accuracy: float = DEFAULT_SEESAW_ACCURACY,
    power_max_iterations: Optional[int] = None,
    power_tolerance: Optional[float] = None,
    isometry: bool = DEFAULT_ISOMETRY,
    print_iterations: bool = False,
    verbose: bool = False,
    return_optimizers: bool = True,
    timing_analysis: bool = True,
    seed: int = DEFAULT_SEED,
    checkpoint_path: Optional[str] = None,
    checkpoint_threshold: float = 0.75,
) -> SDPResult:
    basisAB = EndSnOrbitBasis(n, d_A * d_B)

    # Construct the full dense tensor product channel and then feed it into symmetric seesaw with n = 1.
    channel = basisAB.linear_combination(basisAB.coefficients_for_tensor_product(N)).todense()
    channel = backend.to_gpu(channel)
    return compute_tensor_product_fidelity_seesaw(
        1,
        d_R,
        channel,
        d_A**n,
        d_B**n,
        repetitions=repetitions,
        iterations=iterations,
        seesaw_accuracy=seesaw_accuracy,
        power_max_iterations=power_max_iterations,
        power_tolerance=power_tolerance,
        isometry=isometry,
        print_iterations=print_iterations,
        verbose=verbose,
        return_optimizers=return_optimizers,
        timing_analysis=timing_analysis,
        seed=seed,
        checkpoint_path=checkpoint_path,
        checkpoint_threshold=checkpoint_threshold,
    )
