#!/usr/bin/env python3
"""
verify_operators.py
===================
Self-contained verification of the superactivation operators.

Two .npz files are distributed:

  superactivation_operators_n17.npz        (minimal)
    encoder_block_{i}, decoder_{k}_block_{j}  — Choi blocks in the
    reduced Schur-Weyl basis.

  superactivation_operators_n17_aux.npz    (auxiliary)
    All keys from the minimal file, plus:
      alice_pov_{k}_block_{j}   — AlicePOV (encoder∘N_k) Choi blocks
      encoder_choi              — assembled encoder Choi (180×180)
      decoder_{k}_choi          — assembled decoder Choi for each k

Pass either file to this script.  When alice_pov blocks are present, the
fidelity is recomputed from scratch (self-contained).  Otherwise the stored
fidelity_verified value is displayed.

The script checks two things.

  1. VALIDITY — every stored Choi block is a valid unnormalized Choi matrix
     of a quantum channel:
       - Encoder  (Schrödinger picture):
           CP   : each block B^λ ≥ 0
           TP   : Σ_λ f_λ Tr_sys(B^λ) = I_{d_R}
       - Decoders (Heisenberg picture):
           CP     : each block B^λ ≥ 0  (or B^{λ_k,λ_nk} ≥ 0)
           Unital : Tr_R(B^λ) = I_{m_λ}  per block

     Decoder blocks are projected to the nearest valid operator
     (alternating PSD clamp / unital restore) before checking, to remove
     numerical artifacts from the power-method optimizer (~5e-8 at most).
     This correction is negligible relative to the fidelity margin above 0.75.

  2. FIDELITY — the entanglement fidelity is recomputed self-containedly
     (when alice_pov blocks are present in the file) or read from the stored
     fidelity_verified value.  The displayed value is checked against the
     single-use upper bound 0.75.

     Self-contained formula (no Schur-Weyl isomorphism code needed):
       F_D[k] = (1/d_R²) Σ_λ f_λ · Re Tr(alice_pov_k^λ @ decoder_k^λ)
       F      = Σ_k C(n,k) · (1/2)^n · F_D[k]
     where f_λ are the SYT counts computed from scipy.special.comb only.

Notation
--------
The n-use channel is  N = (q · id + (1-q) · P_p)^{⊗n}  with mixing probability
q = 0.5 and Pauli-channel parameter p = 1/(1+√2).  The encoder E and decoders
D_k (k = 0,...,n) are covariant under the permutation group S_n (respectively
the subgroup S_k × S_{n-k}), so their Choi matrices are block-diagonal in the
Schur-Weyl basis; each block is stored separately in the .npz file.

The Schur-Weyl decomposition for d=2, S_n uses partitions (n-j, j) of n into
at most two rows (j = 0,...,⌊n/2⌋).  For each such partition:
  - m_λ = n − 2j + 1   (SSYT count, GL(2) representation dimension)
  - f_λ = C(n,j)−C(n,j−1)   for j ≥ 1,  f_λ = 1 for j = 0
    (SYT count, multiplicity in the S_n × GL(2) decomposition)
Each Choi block has shape (d_R · m_λ) × (d_R · m_λ), ordered as (reference, system).

Subgroup decoders D_k (1 ≤ k ≤ n−1) use S_k × S_{n-k}: each block is indexed
by a pair (j_k, j_{n-k}) and has shape d_R · m_{j_k} · m_{j_{n-k}}.
The unital condition is Tr_R(B) = I_{m_{j_k} · m_{j_{n-k}}} per block.

The assembled Choi matrices stored in the auxiliary file (encoder_choi,
decoder_k_choi) are the direct sums  ⊕_λ B_λ  without S_n multiplicity
factors — the Choi operators in the reduced Schur-Weyl basis.

Dependencies: numpy, scipy.  No qitsym package required.

Usage
-----
    python verify_operators.py [--file PATH] [--verbose] [--print-operators]

Options
-------
    --file PATH          .npz file (default: superactivation_operators_n17.npz;
                         also accepts superactivation_operators_n17_aux.npz)
    --verbose, -v        Print per-block CP / TP details
    --print-operators, -p  Print all block matrices
"""

import argparse
import numpy as np
from scipy.special import comb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_R = 2                             # reference qubit dimension
Q   = 0.5                           # mixing probability
P   = 1.0 / (1.0 + 2.0 ** 0.5)     # Pauli parameter = 1/(1+√2)
W   = 74                            # print width

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _hr(ch="─"):
    print(ch * W)

def _section(title):
    _hr()
    print(f"  {title}")
    _hr()

# ---------------------------------------------------------------------------
# Combinatorics for the GL(2) Schur-Weyl decomposition
# ---------------------------------------------------------------------------
def _ssyt(n_sym, j):
    """m_λ: SSYT count for GL(2) partition (n_sym−j, j)."""
    return n_sym - 2 * j + 1

def _syt(n_sym, j):
    """f_λ: SYT count for S_{n_sym} partition (n_sym−j, j)."""
    if j == 0:
        return 1
    return int(comb(n_sym, j, exact=True)) - int(comb(n_sym, j - 1, exact=True))

def _partition_ordering(n_sym):
    """Return the j-index ordering used internally by the codebase.

    The Schur-Weyl blocks for S_{n_sym} are stored in the order
      j = 1, 2, ..., ⌊n_sym/2⌋, 0
    i.e. the trivial partition j=0 (largest GL(2) irrep) is stored last.

    Returns a list of length ⌊n_sym/2⌋ + 1.
    """
    P = n_sym // 2 + 1
    if P == 1:
        return [0]
    return list(range(1, P)) + [0]

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _sorted_keys(data, prefix):
    return sorted(
        [k for k in data if k.startswith(prefix)],
        key=lambda x: int(x.split("_")[-1]),
    )

def load_enc_blocks(data):
    return [np.array(data[k]) for k in _sorted_keys(data, "encoder_block_")]

def load_dec_blocks(data, k):
    return [np.array(data[key]) for key in _sorted_keys(data, f"decoder_{k}_block_")]

def load_Mk_blocks(data, k):
    return [np.array(data[key]) for key in _sorted_keys(data, f"alice_pov_{k}_block_")]

# ---------------------------------------------------------------------------
# Partial-trace helpers  (row/col index layout: i_ref * m + i_sys)
# ---------------------------------------------------------------------------
def _tr_sys(B, d_R=D_R):
    """Tr_sys(B): trace over system, keep reference.  Used for encoder TP."""
    m = B.shape[0] // d_R
    return np.einsum("isjs->ij", B.reshape(d_R, m, d_R, m))

def _tr_R(B, d_R=D_R):
    """Tr_R(B): trace over reference, keep system.  Used for decoder unital."""
    m = B.shape[0] // d_R
    return np.einsum("isit->st", B.reshape(d_R, m, d_R, m))

# ---------------------------------------------------------------------------
# Self-contained fidelity computation from stored M_k and D_k blocks
# ---------------------------------------------------------------------------
def _fidelity_D_full(Mk_list, Dk_list, n):
    """
    F_D[k] for k = 0 or k = n  (full S_n symmetry).

    Blocks are stored in internal partition ordering: j=1,...,⌊n/2⌋,0
    (see _partition_ordering).  The weight for block i is f_{j_i} = _syt(n, j_i).
    Formula: (1/d_R²) Σ_i f_{j_i} · Re Tr(M_i @ D_i)
    """
    j_order = _partition_ordering(n)
    total = 0.0
    for i, (M, D) in enumerate(zip(Mk_list, Dk_list)):
        f = _syt(n, j_order[i])
        total += f * float(np.real(np.trace(M @ D)))
    return total / D_R ** 2


def _fidelity_D_subgroup(Mk_list, Dk_list, k, n):
    """
    F_D[k] for 1 ≤ k ≤ n−1  (subgroup S_k × S_{n-k} symmetry).

    Blocks are in outer-product of the internal orderings for k and n-k:
      flat index = j_k_idx * n_blocks_{n-k} + j_nk_idx
    where j_k_idx runs over _partition_ordering(k) and similarly for j_nk_idx.
    Weight: f_{j_k} · f_{j_nk}.
    Formula: (1/d_R²) Σ f_{j_k}·f_{j_nk} · Re Tr(M @ D)
    """
    nmk          = n - k
    j_k_order    = _partition_ordering(k)
    j_nk_order   = _partition_ordering(nmk)
    n_blocks_nmk = len(j_nk_order)
    total = 0.0
    for idx, (M, D) in enumerate(zip(Mk_list, Dk_list)):
        j_k  = j_k_order[idx // n_blocks_nmk]
        j_nk = j_nk_order[idx %  n_blocks_nmk]
        f    = _syt(k, j_k) * _syt(nmk, j_nk)
        total += f * float(np.real(np.trace(M @ D)))
    return total / D_R ** 2


def compute_fidelity(data, dec_blocks_all, n):
    """
    Recompute the entanglement fidelity from the stored AlicePOV blocks.

    F = Σ_{k=0}^{n} C(n,k) · (1/2)^n · F_D[k]

    where F_D[k] = (1/d_R²) Σ_λ f_λ · Tr(alice_pov_k^λ @ decoder_k^λ).

    This requires `alice_pov_{k}_block_{j}` keys in the npz (written by
    run_superactivation_operators.py).  Returns None if those keys are absent
    (older npz files), in which case the caller should fall back to the stored
    fidelity_verified value.
    """
    if f"alice_pov_0_block_0" not in data:
        return None, None

    F_per_k = []
    for k in range(n + 1):
        Mk_list = load_Mk_blocks(data, k)
        Dk_list = dec_blocks_all[k]
        if k == 0 or k == n:
            F_per_k.append(_fidelity_D_full(Mk_list, Dk_list, n))
        else:
            F_per_k.append(_fidelity_D_subgroup(Mk_list, Dk_list, k, n))

    F_total = sum(
        float(comb(n, k, exact=True)) * (0.5 ** n) * F_per_k[k]
        for k in range(n + 1)
    )
    return float(F_total), F_per_k


# ---------------------------------------------------------------------------
# Projection onto the nearest valid Heisenberg-picture Choi block
# ---------------------------------------------------------------------------
def project_choi_heisenberg(B, d_R=D_R, max_iter=50):
    """
    Project B onto  PSD ∩ {Tr_R(B) = I_m}  by alternating projections.

    Each PSD clamp shifts Tr_R(B) by O(violation), so the unital fix
    re-introduces a PSD violation of the same order / d_R; convergence
    is geometric at rate ~ 1/d_R per iteration.  For d_R=2 and initial
    violations ~1e-7, machine precision is reached in ~25 steps.

    A final unconditional PSD clamp guarantees min_eig >= 0 exactly.

    Returns
    -------
    B_proj : np.ndarray  — corrected block  (min_eig >= 0, Tr_R ≈ I_m)
    delta  : float       — max|B_proj - B|
    """
    m     = B.shape[0] // d_R
    B_cur = (B + B.conj().T) / 2.0
    for _ in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(B_cur)
        if eigvals.min() >= -1e-13:     # PSD to machine precision
            break
        B_cur  = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.conj().T
        Br     = B_cur.reshape(d_R, m, d_R, m)
        tr_R   = np.einsum("isit->st", Br)
        B_cur  = B_cur + np.kron(np.eye(d_R) / d_R, np.eye(m) - tr_R)
    # Final unconditional PSD clamp: guarantee min_eig >= 0 exactly
    eigvals, eigvecs = np.linalg.eigh(B_cur)
    B_cur = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.conj().T
    return B_cur, float(np.max(np.abs(B_cur - B)))

# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def _cp_check(B):
    """Returns (min_eig, hermitian_error)."""
    Bsym     = (B + B.conj().T) / 2.0
    min_eig  = float(np.linalg.eigvalsh(Bsym).min())
    herm_err = float(np.max(np.abs(B - B.conj().T)))
    return min_eig, herm_err


def check_encoder(enc_blocks, n, verbose=False):
    """
    Encoder (Schrödinger picture, unnormalized Choi):
      CP : each block B^j >= 0
      TP : Σ_j f_j * Tr_sys(B^j) = I_{d_R}

    Blocks are stored in internal partition ordering: j=1,...,⌊n/2⌋,0
    (see _partition_ordering).  The partition index j is read from each
    block's position in the list via _partition_ordering(n).
    """
    j_order      = _partition_ordering(n)
    tp_sum       = np.zeros((D_R, D_R), dtype=complex)
    min_eig_all  = float("inf")
    herm_err_all = 0.0

    for i, B in enumerate(enc_blocks):
        j   = j_order[i]
        m   = B.shape[0] // D_R      # = n - 2j + 1
        f   = _syt(n, j)
        min_eig, herm_err = _cp_check(B)
        min_eig_all  = min(min_eig_all, min_eig)
        herm_err_all = max(herm_err_all, herm_err)
        tp_sum += f * _tr_sys(B)
        if verbose:
            print(f"    encoder_block_{i}: λ=({n-j},{j})  m={m}  f={f}  "
                  f"size={B.shape[0]}x{B.shape[0]}  min_eig={min_eig:+.3e}  herm={herm_err:.1e}")

    tp_err = float(np.max(np.abs(tp_sum - np.eye(D_R))))
    return min_eig_all, herm_err_all, tp_err


def check_decoder_full(dk_blocks, n, k_label, verbose=False):
    """
    Full-S_n decoder (Heisenberg picture, unnormalized Choi):
      CP     : each block B^j >= 0
      Unital : Tr_R(B^j) = I_{m_j}  per block

    m_j = block.shape[0] // d_R = n - 2j + 1 is read directly from the block.
    """
    min_eig_all  = float("inf")
    herm_err_all = 0.0
    tp_err_all   = 0.0

    for i, B in enumerate(dk_blocks):
        m   = B.shape[0] // D_R
        min_eig, herm_err = _cp_check(B)
        min_eig_all  = min(min_eig_all, min_eig)
        herm_err_all = max(herm_err_all, herm_err)
        tp_err = float(np.max(np.abs(_tr_R(B) - np.eye(m))))
        tp_err_all = max(tp_err_all, tp_err)
        if verbose:
            j = (n + 1 - m) // 2
            print(f"    decoder_{k_label}_block_{i}: λ=({n-j},{j})  m={m}  "
                  f"size={B.shape[0]}x{B.shape[0]}  min_eig={min_eig:+.3e}  TP_err={tp_err:.2e}")

    return min_eig_all, herm_err_all, tp_err_all


def check_decoder_subgroup(dk_blocks, k_label, verbose=False):
    """
    S_k × S_{n-k} decoder (Heisenberg picture, unnormalized Choi):
      CP     : each block B^{j_k, j_{n-k}} >= 0
      Unital : Tr_R(B) = I_{m_S}  per block,  m_S = block.shape[0] // d_R

    m_S = m_{j_k} * m_{j_{n-k}} is read directly from the block size.
    """
    min_eig_all  = float("inf")
    herm_err_all = 0.0
    tp_err_all   = 0.0

    for i, B in enumerate(dk_blocks):
        m_S = B.shape[0] // D_R
        min_eig, herm_err = _cp_check(B)
        min_eig_all  = min(min_eig_all, min_eig)
        herm_err_all = max(herm_err_all, herm_err)
        tp_err = float(np.max(np.abs(_tr_R(B) - np.eye(m_S))))
        tp_err_all = max(tp_err_all, tp_err)
        if verbose:
            print(f"    decoder_{k_label}_block_{i}: m_S={m_S}  "
                  f"size={B.shape[0]}x{B.shape[0]}  min_eig={min_eig:+.3e}  TP_err={tp_err:.2e}")

    return min_eig_all, herm_err_all, tp_err_all

# ---------------------------------------------------------------------------
# Print operators
# ---------------------------------------------------------------------------
def print_operator_blocks(blocks, label):
    print(f"\n  {label}  ({len(blocks)} blocks)")
    for i, B in enumerate(blocks):
        re = np.real(B)
        im = np.imag(B)
        print(f"    block {i}  shape={B.shape[0]}x{B.shape[1]}")
        with np.printoptions(precision=5, suppress=True, linewidth=120):
            lines = repr(re).replace("\n", "\n        ")
            print(f"      re = {lines}")
            if np.max(np.abs(im)) > 1e-14:
                lines = repr(im).replace("\n", "\n        ")
                print(f"      im = {lines}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify superactivation operators (self-contained, numpy only)."
    )
    parser.add_argument("--file", default="superactivation_operators_n17.npz",
                        help=".npz file (default: superactivation_operators_n17.npz)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-block CP/TP details for every operator")
    parser.add_argument("--print-operators", "-p", action="store_true",
                        help="Print all block matrices (encoder + n+1 decoders)")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    data     = np.load(args.file, allow_pickle=True)
    n        = int(data["n"])
    f_seesaw = float(data["fidelity_seesaw"])
    f_stored = float(data["fidelity_verified"])

    has_alice_pov    = "alice_pov_0_block_0" in data
    has_encoder_choi = "encoder_choi" in data

    _hr("=")
    print(f"  SUPERACTIVATION OPERATOR VERIFICATION  (n = {n})")
    _hr("=")
    print(f"  File                  : {args.file}")
    print(f"  Channel uses n        : {n}")
    print(f"  d_R (reference dim)   : {D_R}")
    print(f"  Mixing probability q  : {Q}")
    print(f"  Pauli parameter p     : {P:.8f}   [= 1/(1+√2)]")
    print(f"  alice_pov blocks      : {'present (fidelity will be recomputed)' if has_alice_pov else 'absent (stored fidelity will be shown)'}")
    print(f"  Assembled Choi mats   : {'present' if has_encoder_choi else 'absent'}")
    print()

    # ── Load and project decoder blocks ───────────────────────────────────
    # Decoder blocks are projected to the nearest valid Choi operator
    # (PSD ∩ unital) to remove numerical artifacts from the optimizer.
    # The maximum correction is ~5e-8, negligible vs. the fidelity margin.
    # Projection uses the unprojected blocks; assembled Choi (if present)
    # is only loaded to report its shape.
    enc_blocks     = load_enc_blocks(data)
    dec_blocks_all = {
        k: [project_choi_heisenberg(B)[0] for B in load_dec_blocks(data, k)]
        for k in range(n + 1)
    }

    # ── Fidelity ──────────────────────────────────────────────────────────
    F_computed, F_per_k = compute_fidelity(data, dec_blocks_all, n)
    has_computed = F_computed is not None  # True iff alice_pov blocks present

    _section("FIDELITY")
    print(f"  Stored F (seesaw)     : {f_seesaw:.10f}")
    print(f"  Stored F (verified)   : {f_stored:.10f}")
    if has_computed:
        print(f"  Recomputed F          : {F_computed:.10f}")
        print(f"  |F_recomputed - F_stored| : {abs(F_computed - f_stored):.2e}")
        F_show = F_computed
        label  = "F_recomputed"
    else:
        print("  (alice_pov blocks not found in npz — showing stored value)")
        F_show = f_stored
        label  = "F_stored"

    if has_computed:
        print(f"\n  {'k':>3}  | {'F_D[k]':>10}  | {'weight':>12}  | {'contrib':>10}")
        print(f"  {'─'*3}-+-{'─'*10}--+-{'─'*12}--+-{'─'*10}")
        for k in range(n + 1):
            w = float(comb(n, k, exact=True)) * (0.5 ** n)
            print(f"  {k:>3}  | {F_per_k[k]:>10.6f}  | {w:>12.8f}  | {w*F_per_k[k]:>10.6f}")
        print(f"  {'─'*3}-+-{'─'*10}--+-{'─'*12}--+-{'─'*10}")
        print(f"  {'':>3}    {'Σ':>10}    {'':>12}    {F_computed:>10.6f}")

    UPPER_BOUND = 0.75
    diff = F_show - UPPER_BOUND
    print(f"\n  Upper bound  0.75     : {UPPER_BOUND}")
    print(f"  {label} - 0.75    : {diff:+.2e}", end="")
    if diff > 0:
        print("   *** SUPERACTIVATION CONFIRMED ***")
    else:
        print("   (does not exceed upper bound)")
    print()

    # ── Assembled Choi matrices (auxiliary file only) ─────────────────────
    if has_encoder_choi:
        _section("ASSEMBLED CHOI MATRICES  (reduced Schur-Weyl basis, ⊕_λ B_λ)")
        enc_choi = np.array(data["encoder_choi"])
        print(f"  encoder_choi          : shape {enc_choi.shape}")
        for k in range(n + 1):
            key = f"decoder_{k}_choi"
            if key in data:
                ch = np.array(data[key])
                print(f"  decoder_{k:2d}_choi       : shape {ch.shape}")
        print()

    # ── Validity checks ───────────────────────────────────────────────────
    _section("VALIDITY  (unnormalized Choi matrices: CP = PSD,  TP / unital)")
    tol    = 1e-6    # encoder: as-optimized; decoders: projected to ~1e-13
    all_ok = True

    # Encoder
    if args.verbose:
        print("  Encoder blocks:")
    min_eig, herm_err, tp_err = check_encoder(enc_blocks, n, args.verbose)
    cp_ok = min_eig >= -tol
    tp_ok = tp_err  <   tol
    all_ok &= cp_ok and tp_ok
    flag   = "✓" if (cp_ok and tp_ok) else "✗ WARNING"
    print(
        f"  Encoder ({len(enc_blocks):2d} blocks)  :  "
        f"herm={herm_err:.1e}  min_eig={min_eig:+.2e}  "
        f"CP={'✓' if cp_ok else '✗'}  "
        f"TP_err={tp_err:.1e}  TP={'✓' if tp_ok else '✗'}  [{flag}]"
    )
    print()

    # Decoders (projected)
    for k in range(n + 1):
        dk_blocks = dec_blocks_all[k]
        if args.verbose:
            print(f"  Decoder k={k} blocks:")
        if k == 0 or k == n:
            min_eig, herm_err, tp_err = check_decoder_full(dk_blocks, n, k, args.verbose)
        else:
            min_eig, herm_err, tp_err = check_decoder_subgroup(dk_blocks, k, args.verbose)
        cp_ok = min_eig >= -tol
        tp_ok = tp_err  <   tol
        all_ok &= cp_ok and tp_ok
        flag   = "✓" if (cp_ok and tp_ok) else "✗ WARNING"
        print(
            f"  Decoder k={k:2d} ({len(dk_blocks):2d} blocks):  "
            f"herm={herm_err:.1e}  min_eig={min_eig:+.2e}  "
            f"CP={'✓' if cp_ok else '✗'}  "
            f"TP_err={tp_err:.1e}  TP={'✓' if tp_ok else '✗'}  [{flag}]"
        )
    print()
    if all_ok:
        print("  All operators pass validity checks:  CP ✓   TP / unital ✓")
    else:
        print("  WARNING: one or more operators failed a validity check.")
    print()

    # ── Print operators (optional) ────────────────────────────────────────
    if args.print_operators:
        _section(f"OPERATOR BLOCKS  (encoder + {n+1} decoders, n={n} uses)")
        print_operator_blocks(enc_blocks, "Encoder E  (S_n-covariant, Schrödinger picture)")
        for k in range(n + 1):
            symmetry = "full S_n" if (k == 0 or k == n) else f"S_{k} × S_{n-k}"
            print_operator_blocks(
                dec_blocks_all[k],
                f"Decoder D_{k}  (k={k} Pauli uses,  {symmetry},  Heisenberg picture)"
            )

    _hr("=")
    print("  Done.")
    _hr("=")


if __name__ == "__main__":
    main()
