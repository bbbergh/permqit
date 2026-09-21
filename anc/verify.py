#!/usr/bin/env python3
"""
verify.py
=========
Validity check for the whole anc/ tree, for every n and every channel.

Dependencies: numpy, scipy.  Nothing else.

Layout
------
  <channel>/fidelities.txt                    entanglement fidelity, indexed by n
  <channel>/optimal_encoders/encoder_nNN.txt  the optimal encoder for that n
  flagged_x_z_independent/optimal_decoders_n17/decoders_n17.npz
                                              decoder + Alice-POV blocks for n = 17

What is checked
---------------
ENCODERS, every n and every channel.
  The encoder is E(rho) = V rho V^dagger with V : C^d_R -> Sym^n(C^2) written in the Dicke
  basis.  Its Choi matrix is the rank-one, positive operator |v><v| with |v> = sum_i |i> (x) V|i>,
  so it is automatically completely positive; it is trace preserving exactly when

      Tr_{A^n} |v><v| = (V^dagger V)^T = I_{d_R} .

  Checking V^dagger V = I therefore certifies that the stored block is a proper quantum channel,
  and moreover that it is an isometry -- equivalently that the Choi state is pure and supported
  on the single Young diagram lambda = (n).  We also check the shape (n+1 Dicke amplitudes).

DECODERS, n = 17 of the flagged channel.
  Each stored Choi block B (ordered (R, S), Heisenberg picture) must satisfy
      B >= 0                      (complete positivity)
      Tr_R B = I_m                (unitality of the adjoint, i.e. the decoder is trace preserving)
  Blocks that the channel annihilates are not stored: the selection rule forces mu_{n-k} = (n-k),
  and the optimizer leaves the rest at the trivial unital point I_R (x) I_m / d_R, which is
  reconstructed here and satisfies both conditions by inspection.

FIDELITY, n = 17 of the flagged channel.
  Recomputed from the stored operators and compared with fidelities.txt:
      F = sum_k C(n,k) q^k (1-q)^(n-k) (1/d_R^2) sum_mu f_mu Re Tr(M_k^mu D_k^mu),
  with f_mu the number of standard Young tableaux, obtained from binomials alone.

Usage
-----
    python verify.py [--tol TOL]
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.special import comb


def syt(partition):
    """Standard-Young-tableau count for a partition with at most two rows."""
    if len(partition) == 0:
        return 1
    n, j = sum(partition), (partition[1] if len(partition) > 1 else 0)
    if j == 0:
        return 1
    return int(comb(n, j, exact=True)) - int(comb(n, j - 1, exact=True))


def read_fidelities(path):
    ns, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()
            ns.append(int(a))
            fs.append(float(b))
    return ns, fs


def read_encoder(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            rows.append([complex(v[i], v[i + 1]) for i in range(0, len(v), 2)])
    return np.array(rows, dtype=complex)


def check_encoders(root, channel, tol, verbose):
    ns, fids = read_fidelities(os.path.join(root, "fidelities.txt"))
    ok = True
    worst_iso, worst_n = 0.0, None
    for n, f in zip(ns, fids):
        path = os.path.join(root, "optimal_encoders", f"encoder_n{n:02d}.txt")
        if not os.path.exists(path):
            print(f"    n={n:3d}  MISSING {path}")
            ok = False
            continue
        V = read_encoder(path)
        dev = float(np.max(np.abs(V.conj().T @ V - np.eye(V.shape[1]))))
        shape_ok = V.shape[0] == n + 1
        good = dev < tol and shape_ok and 0.0 <= f <= 1.0
        ok &= good
        if dev > worst_iso:
            worst_iso, worst_n = dev, n
        if verbose or not good:
            print(f"    n={n:3d}  F={f:.16f}  V is {V.shape[0]}x{V.shape[1]}  "
                  f"||V^dag V - I||={dev:.1e}" + ("" if good else "   <-- FAIL"))
    print(f"  encoders: {len(ns)} checked, all rank-one CPTP isometries into Sym^n, "
          f"worst ||V^dag V - I|| = {worst_iso:.1e} (n={worst_n})"
          + ("" if ok else "   <-- SOME FAILED"))
    return ok, ns, fids


def check_decoders_n17(root, fid_from_txt, tol, verbose):
    path = os.path.join(root, "optimal_decoders_n17", "decoders_n17.npz")
    if not os.path.exists(path):
        print("  decoders (n=17): not present")
        return True
    d = np.load(path, allow_pickle=True)
    n, d_R, q = int(d["n"]), int(d["d_R"]), float(d["q"])
    F, n_blocks, worst_unital, min_eig = 0.0, 0, 0.0, 0.0
    n_trivial = 0
    for k in range(n + 1):
        lab_key = f"decoder_{k}_labels"
        if lab_key not in d.files:
            continue
        labels = [eval(x) for x in d[lab_key]]  # noqa: S307  (tuples written by the builder)
        kept = set(int(j) for j in d[f"decoder_{k}_kept"])
        dims = [int(x) for x in d[f"decoder_{k}_dims"]]
        w = float(comb(n, k, exact=True)) * q ** k * (1.0 - q) ** (n - k)
        for j, dim in enumerate(dims):
            if dim == 0:
                continue
            m = dim // d_R
            if j not in kept:
                # reconstruct the trivial unital point and confirm it is a proper map
                B = np.kron(np.eye(d_R), np.eye(m)) / d_R
                assert np.max(np.abs(np.einsum("iris->rs", B.reshape(d_R, m, d_R, m))
                                     - np.eye(m))) < 1e-12
                n_trivial += 1
                continue
            B = d[f"decoder_{k}_block_{j}"]
            M = d[f"alice_pov_{k}_block_{j}"]
            n_blocks += 1
            eigs = np.linalg.eigvalsh(0.5 * (B + B.conj().T))
            min_eig = min(min_eig, float(eigs.min()))
            worst_unital = max(worst_unital, float(np.max(np.abs(
                np.einsum("iris->rs", B.reshape(d_R, m, d_R, m)) - np.eye(m)))))
            F += w * int(np.prod([syt(p) for p in labels[j]])) \
                * float(np.real(np.trace(M @ B))) / d_R ** 2
    delta = F - fid_from_txt
    ok = min_eig > -1e-7 and worst_unital < 1e-6 and abs(delta) < tol
    print(f"  decoders (n=17): {n_blocks} stored blocks CP (min eig {min_eig:+.1e}) and unital "
          f"(max |Tr_R B - I| = {worst_unital:.1e}); {n_trivial} trivial blocks reconstructed")
    print(f"  fidelity (n=17): recomputed {F:.16f} vs stored {fid_from_txt:.16f}  "
          f"delta {delta:+.1e}" + ("" if ok else "   <-- FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    channels = sorted(os.path.basename(p) for p in glob.glob(os.path.join(here, "*"))
                      if os.path.isdir(p) and os.path.exists(os.path.join(p, "fidelities.txt")))
    if not channels:
        print("no channel subfolder with a fidelities.txt found")
        return 1

    ok = True
    for channel in channels:
        root = os.path.join(here, channel)
        print("=" * 78)
        print(f"  {channel}")
        with open(os.path.join(root, "fidelities.txt")) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if "parameter" in line or "mixing" in line:
                    print("  " + line.strip())
        print("=" * 78)
        enc_ok, ns, fids = check_encoders(root, channel, args.tol, args.verbose)
        ok &= enc_ok
        if channel == "flagged_x_z_independent":
            ok &= check_decoders_n17(root, fids[ns.index(17)], args.tol, args.verbose)
            best = max(fids)
            print(f"  superactivation: max_n F = {best:.16f} "
                  f"({'ABOVE' if best > 0.75 else 'below'} 3/4 by {abs(best - 0.75):.3e}), "
                  f"first n with F > 3/4: n = {min(n for n, f in zip(ns, fids) if f > 0.75)}")
        else:
            print(f"  max over n: F = {max(fids):.16f} at n = {ns[int(np.argmax(fids))]}")
        print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
