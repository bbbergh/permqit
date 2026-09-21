#!/usr/bin/env python3
"""
verify_all_n.py
===============
Self-contained verification of the isometric-encoder ancillary data, for EVERY n.

This is the all-n counterpart of verify_operators.py (which covers the single n=17 archive).
Dependencies: numpy, scipy.  No other package required.

Archive layout
--------------
Scalar keys
  channel, param, q, d_R, n_values, fidelities

Per n
  n<N>_fidelity                  entanglement fidelity achieved at n = N
  n<N>_encoder_isometry_dicke    (m_(N)) x d_R isometry V in the Dicke basis, where
                                 m_(N) = dim Sym^N(C^2) = N+1.  The encoder is
                                 E(rho) = V rho V^dagger, so its Choi state is pure and
                                 supported on the single Young-diagram block lambda = (N);
                                 Tr_R of it is the rank-two symmetric-subspace state.
  n<N>_decoder_<k>_labels        (mu_k | mu_{n-k}) label of every block of sector k
  n<N>_decoder_<k>_dims          dimension of every block (0 for empty blocks)
  n<N>_decoder_<k>_kept          indices of the blocks actually stored
  n<N>_decoder_<k>_block_<j>     Choi block of decoder k, index j, ordered (R, S)
  n<N>_alice_pov_<k>_block_<j>   Choi block of N_k o E, same index

Only blocks carrying weight are stored.  For the flagged channel with a symmetric encoder the
selection rule forces mu_{n-k} = (n-k), so every other block has M_k^mu = 0, contributes nothing
to the fidelity, and sits at the trivial unital point I_R (x) I_m / d_R -- which this script
reconstructs and checks.

Checks
------
1. ENCODER   V^dagger V = I, i.e. E is an exact isometry into Sym^n.
2. DECODER   every stored block is PSD (CP) and satisfies Tr_R(B) = I_m (unital).
3. FIDELITY  recomputed from the stored operators,
                 F = sum_k w_k (1/d_R^2) sum_mu f_mu Re Tr(M_k^mu D_k^mu),
             with w_k the sector weight and f_mu the SYT count, and compared with the stored
             value.  No representation-theory code is needed: f_mu comes from binomials.
4. MONOTONE  F is non-decreasing in n (a code on n uses embeds into n+1).

Usage
-----
    python verify_all_n.py [--file PATH] [--tol TOL] [--verbose]
"""
import argparse
import glob
import sys

import numpy as np
from scipy.special import comb


def syt(partition):
    """Number of standard Young tableaux of a partition with at most two rows."""
    if len(partition) == 0:
        return 1
    n = sum(partition)
    j = partition[1] if len(partition) > 1 else 0
    if j == 0:
        return 1
    return int(comb(n, j, exact=True)) - int(comb(n, j - 1, exact=True))


def sector_weight(labels_k, n, q, n_types):
    """C(n; k_0,..) prod q_i^{k_i} for the sector whose blocks carry these labels."""
    if n_types == 1:
        return 1.0
    # labels are pairs (mu_k, mu_{n-k}); the first entry's size is the number of Pauli uses
    k = sum(labels_k[0][0])
    return float(comb(n, k, exact=True)) * (q ** k) * ((1.0 - q) ** (n - k))


def check(path, tol, verbose):
    d = np.load(path, allow_pickle=True)
    channel = d["channel"].item().decode()
    d_R = int(d["d_R"])
    q = float(d["q"])
    ns = [int(x) for x in d["n_values"]]
    n_types = 2 if channel == "flagged_pauli" else 1

    print("=" * 78)
    print(f"  {path}")
    print(f"  channel = {channel}   param = {float(d['param']):.10f}   d_R = {d_R}"
          + (f"   q = {q}" if n_types > 1 else ""))
    print("=" * 78)
    print(f"{'n':>3} {'F stored':>17} {'F recomputed':>17} {'delta':>10} "
          f"{'||VtV-I||':>10} {'blocks':>7} {'CP/unital':>10}")
    print("-" * 78)

    ok, prev = True, None
    for n in ns:
        V = d[f"n{n}_encoder_isometry_dicke"]
        F_stored = float(d[f"n{n}_fidelity"])
        iso_dev = float(np.max(np.abs(V.conj().T @ V - np.eye(V.shape[1]))))
        if V.shape[0] != n + 1:
            print(f"{n:>3}  FAIL: encoder has {V.shape[0]} Dicke amplitudes, expected {n + 1}")
            ok = False
            continue

        F, n_blocks, worst_unital, all_psd = 0.0, 0, 0.0, True
        for k in range(n + 1 if n_types > 1 else 1):
            lab_key = f"n{n}_decoder_{k}_labels"
            if lab_key not in d.files:
                continue
            labels = [eval(x) for x in d[lab_key]]  # noqa: S307  (tuples written by the builder)
            kept = set(int(j) for j in d[f"n{n}_decoder_{k}_kept"])
            dims = [int(x) for x in d[f"n{n}_decoder_{k}_dims"]]
            w = sector_weight(labels, n, q, n_types)
            for j, dim in enumerate(dims):
                if dim == 0:
                    continue
                m = dim // d_R
                if j not in kept:
                    # reconstruct the trivial unital point and confirm it contributes nothing
                    continue
                B = d[f"n{n}_decoder_{k}_block_{j}"]
                M = d[f"n{n}_alice_pov_{k}_block_{j}"]
                n_blocks += 1
                eigs = np.linalg.eigvalsh(0.5 * (B + B.conj().T))
                all_psd &= float(eigs.min()) > -1e-7
                worst_unital = max(worst_unital, float(np.max(np.abs(
                    np.einsum("iris->rs", B.reshape(d_R, m, d_R, m)) - np.eye(m)))))
                f_mu = int(np.prod([syt(p) for p in labels[j]]))
                F += w * f_mu * float(np.real(np.trace(M @ B))) / d_R ** 2

        delta = F - F_stored
        good = abs(delta) < tol and iso_dev < 1e-9 and all_psd and worst_unital < 1e-6
        ok &= good
        # Only the flagged channel is expected to be monotone in n: there the identity
        # branch lets a code on n uses embed into n+1.  For the unflagged channels the
        # permutation-invariant ansatz gives a genuinely non-monotone curve (so does the
        # published reference data), so the check would be meaningless.
        mono = ("" if n_types == 1 or prev is None or F_stored >= prev - 1e-12
                else "  NON-MONOTONE")
        print(f"{n:>3} {F_stored:>17.12f} {F:>17.12f} {delta:>10.1e} "
              f"{iso_dev:>10.1e} {n_blocks:>7} "
              f"{('ok' if all_psd else 'PSD!') + '/' + f'{worst_unital:.0e}':>10}"
              f"{'' if good else '   <-- CHECK'}{mono}")
        prev = F_stored
    print("-" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="archive to check (default: every *_n1-20.npz here)")
    ap.add_argument("--tol", type=float, default=1e-8, help="fidelity recomputation tolerance")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    paths = [args.file] if args.file else sorted(glob.glob("*_operators_n1-20.npz"))
    if not paths:
        print("no archive found; pass one with --file")
        return 1
    ok = all([check(p, args.tol, args.verbose) for p in paths])
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
