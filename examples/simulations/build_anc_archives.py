"""Build the ancillary-data archives distributed with the papers.

Reads the seesaw outputs under results/ and writes, for every n, a self-contained archive with
  * the optimal encoder, as the (dim Sym^n) x d_R matrix of the isometry V in the Dicke basis,
  * the decoder Choi blocks,
  * the Alice-POV blocks M_k^mu (Choi of N_k o E) needed to recompute the fidelity,
  * the fidelity itself.

Only blocks that carry weight are stored.  For the flagged channel with a symmetric-subspace
encoder the channel's selection rule forces mu_{n-k} = (n-k), so most output blocks have
M_k^mu = 0; those contribute nothing to the fidelity and the power iteration leaves the decoder
at the trivial unital point I_R (x) I_m / d_R, which the verifier reconstructs.  Storing only
the rest shrinks the archive by roughly an order of magnitude.

    uv run python examples/simulations/build_anc_archives.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")
from examples.channels.amplitude_damping_channel import amplitude_damping_choi  # noqa: E402
from examples.channels.depolarizing_channel import depolarizing_choi  # noqa: E402
from examples.simulations.run_flagged_pauli_operators import (  # noqa: E402
    CHOI_IDENTITY, DEFAULT_P, choi_pauli,
)
from permqit.algebra.basis import MatrixStandardBasis  # noqa: E402
from permqit.algebra.basis_subset import (  # noqa: E402
    IndexIsValidPredicate, MatrixEntryMask, MatrixStandardBasisSubset,
)
from permqit.algebra.endomorphism_direct_sum_basis import EndSnSingleBlockOrbitBasis  # noqa: E402
from permqit.representation.combinatorics import multinomial_coeff, weak_compositions  # noqa: E402
from permqit.representation.isomorphism import (  # noqa: E402
    EndSnAlgebraIsomorphism, EndSnBlockDiagonalization, TrivialAlgebraIsomorphism,
    tensor_product_block_diagonalization,
)
from permqit.representation.partial_traces import SingleBlockPartialTraceRelations  # noqa: E402
from permqit.SDP.seesaw_utils import get_coefficient_BobPOV_from_relation  # noqa: E402
from permqit.utilities.backend import to_cpu  # noqa: E402
from permqit.utilities.random import isometry_into_symmetric_block  # noqa: E402

D_R = 2
CONTRIBUTION_TOL = 1e-12   # drop a block only if it moves F by less than this


def _ab_bases(N_list, d_AB):
    out = []
    for Ni in N_list:
        sup = [(a, b) for a in range(d_AB) for b in range(d_AB) if abs(Ni[a, b]) > 1e-12]
        out.append(
            MatrixStandardBasisSubset(MatrixStandardBasis(d_AB),
                                      IndexIsValidPredicate(MatrixEntryMask((d_AB, d_AB), sup)))
            if len(sup) < d_AB * d_AB else MatrixStandardBasis(d_AB))
    return out


def alice_pov_blocks(n, V, N_list, q_vec):
    """Recompute the M_k^mu blocks (Choi of N_k o E) from the stored encoder isometry."""
    m = len(N_list)
    iso_A = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, 2))
    iso_B = {k: EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(k, 2)) for k in range(n + 1)}
    c_E = isometry_into_symmetric_block(V, D_R, iso_A)
    bA = EndSnSingleBlockOrbitBasis((MatrixStandardBasis(2),), [n])
    ab = _ab_bases(N_list, 4)
    basisR = MatrixStandardBasis(D_R)

    coeff_cache: dict = {}

    def type_coeffs(i, k):
        if (i, k) not in coeff_cache:
            if k == 0:
                coeff_cache[(i, k)] = np.array([1.0 + 0j])
            else:
                sb = EndSnSingleBlockOrbitBasis((ab[i],), [k])
                coeff_cache[(i, k)] = np.array(
                    sb.bases[0].coefficients_for_tensor_product(N_list[i]), dtype=np.complex128)
        return coeff_cache[(i, k)]

    out, weights = [], []
    for comp in weak_compositions(n, m):
        c = type_coeffs(0, comp[0])
        for i in range(1, m):
            c = np.outer(c, type_coeffs(i, comp[i])).ravel()
        bB = EndSnSingleBlockOrbitBasis(tuple(MatrixStandardBasis(2) for _ in range(m)), list(comp))
        bAB = EndSnSingleBlockOrbitBasis(tuple(ab), list(comp))
        rel = SingleBlockPartialTraceRelations(bA, bB, bAB)
        c_M = get_coefficient_BobPOV_from_relation(c_E, basisR, bA, c, rel)
        isos = [TrivialAlgebraIsomorphism(D_R)] + [iso_B[k] for k in comp]
        out.append([np.asarray(to_cpu(B)) for B in
                    tensor_product_block_diagonalization(np.asarray(c_M), isos)])
        w = float(multinomial_coeff(comp))
        for qi, ki in zip(q_vec, comp):
            w *= qi ** ki
        weights.append(w)
    return out, weights


def syt(partition):
    from permqit.representation.partition import Partition
    return Partition(partition).count_standard_tableaux() if partition else 1


def build(channel, src, out_path, param, N_list, q_vec, n_types):
    src_npz = np.load(src, allow_pickle=True)
    ns = sorted(int(k[1:].split("_")[0]) for k in src_npz.files if k.endswith("_fidelity"))
    payload = {
        "channel": np.bytes_(channel.encode()), "param": np.float64(param),
        "q": np.float64(q_vec[1] if len(q_vec) > 1 else 1.0), "d_R": np.int64(D_R),
        "n_values": np.array(ns, dtype=np.int64),
        "fidelities": np.array([float(src_npz[f"n{n}_fidelity"]) for n in ns]),
    }
    for n in ns:
        V = src_npz[f"n{n}_encoder_isometry_dicke"]
        F_stored = float(src_npz[f"n{n}_fidelity"])
        payload[f"n{n}_fidelity"] = np.float64(F_stored)
        payload[f"n{n}_encoder_isometry_dicke"] = V

        M_sectors, sector_w = alice_pov_blocks(n, V, N_list, q_vec)
        F_check = 0.0
        for k, (Ms, w) in enumerate(zip(M_sectors, sector_w)):
            labels = [eval(x) for x in src_npz[f"n{n}_decoder_{k}_labels"]]  # noqa: S307
            kept = []
            for j, M in enumerate(Ms):
                if M.size == 0:
                    continue
                Dblk = src_npz[f"n{n}_decoder_{k}_block_{j}"]
                f_lam = int(np.prod([syt(p) for p in labels[j]]))
                contrib = w * f_lam * float(np.real(np.trace(M @ Dblk))) / D_R ** 2
                # Drop a block only when it genuinely cannot matter.  Thresholding on |M| instead
                # silently loses blocks that are small but numerous (depolarizing at large n).
                if abs(contrib) < CONTRIBUTION_TOL:
                    continue
                F_check += contrib
                payload[f"n{n}_decoder_{k}_block_{j}"] = Dblk
                payload[f"n{n}_alice_pov_{k}_block_{j}"] = M
                kept.append(j)
            payload[f"n{n}_decoder_{k}_labels"] = src_npz[f"n{n}_decoder_{k}_labels"]
            payload[f"n{n}_decoder_{k}_kept"] = np.array(kept, dtype=np.int64)
            payload[f"n{n}_decoder_{k}_dims"] = np.array(
                [Ms[j].shape[0] if Ms[j].size else 0 for j in range(len(Ms))], dtype=np.int64)
        print(f"  n={n:3d}  F stored {F_stored:.12f}  recomputed {F_check:.12f}"
              f"   delta {F_check - F_stored:+.2e}", flush=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_path)
    print(f"-> {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)\n")


def main():
    os.makedirs("anc", exist_ok=True)
    os.makedirs("anc_2", exist_ok=True)
    only = set(sys.argv[1:])
    if only and "flagged" not in only:
        pass
    else:
        print("flagged Pauli (superactivation):")
        build("flagged_pauli", "results/flagged_pauli_isometric.npz",
              "anc/superactivation_operators_n1-20.npz", DEFAULT_P,
              [choi_pauli(DEFAULT_P), CHOI_IDENTITY], [0.5, 0.5], 2)
    if not only or "depol" in only:
        print("depolarizing:")
        build("depolarizing", "results/depolarizing_isometric.npz",
              "anc_2/depolarizing_operators_n1-20.npz", 0.15102040816326533,
              [depolarizing_choi(0.15102040816326533)], [1.0], 1)
    if not only or "ad" in only:
        print("amplitude damping:")
        build("amplitude_damping", "results/amplitude_damping_isometric.npz",
              "anc_2/amplitude_damping_operators_n1-20.npz", 0.5102040816326531,
              [amplitude_damping_choi(0.5102040816326531)], [1.0], 1)


if __name__ == "__main__":
    main()
