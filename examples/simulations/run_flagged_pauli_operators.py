"""Symmetric seesaw for the **flagged Pauli channel**, with a fully-isometric encoder.

Channel (Proposition "equivalent channel" of the superactivation paper): each use of
N : M_2 -> M_2 (x) M_2 is a flagged mixture

    N(rho) = q |0><0|_Z (x) id(rho)_B  +  (1-q) |1><1|_Z (x) P_p(rho)_B ,

with P_p the Pauli channel ((1-p)/2, p/2, p/2, (1-p)/2) -- i.e. the binary symmetric channel
with crossover p -- and the classical flag Z delivered to the receiver.  For n uses the flag
string reveals the number k of "Pauli" uses, so the n-use channel splits into n+1 sectors
N_k = P_p^{(x)k} (x) id^{(x)(n-k)}, each with its own decoder D_k.

The encoder is constrained to a genuine isometry V : C^d_R -> Sym^n(A) (``isometry=True``); see
``permqit.utilities.random.random_symmetric_isometric_channel``.  Output, following the
convention of Agarwal et al. [arXiv:2605.09138], stores the encoder as the (n+1) x d_R matrix of
V in the Dicke basis, alongside the Choi blocks of every decoder.

Usage
-----
    uv run python examples/simulations/run_flagged_pauli_operators.py --n 17 \
        --seeds 18 42 137 --iterations 2000 --out results/flagged_pauli_n17.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from permqit.power_method.seesaw import compute_tensor_product_fidelity_seesaw
from permqit.representation.combinatorics import weak_compositions
from permqit.representation.isomorphism import (
    EndSnAlgebraIsomorphism,
    EndSnBlockDiagonalization,
    TrivialAlgebraIsomorphism,
    tensor_product_block_diagonalization,
)
from permqit.utilities.backend import to_cpu
from permqit.utilities.random import symmetric_isometry_from_choi_coefficients

DEFAULT_P = 1.0 / (1.0 + 2.0**0.5)


def choi_pauli(p: float) -> np.ndarray:
    """Choi matrix of the BSC(p) = Pauli channel ((1-p)/2, p/2, p/2, (1-p)/2)."""
    return np.array(
        [[1 - p, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0], [0, 0, 0, 1 - p]], dtype=np.complex128
    )


CHOI_IDENTITY = np.array(
    [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.complex128
)


def decoder_blocks(c_D_k, d_R: int, iso_tuple):
    """Choi blocks of one sector's decoder, in (R, S) order, indexed by (mu_k, mu_{n-k})."""
    from permqit.SDP.seesaw_utils import get_coefficient_adjoint_general_SR

    bases = [iso.basis_from for iso in iso_tuple]
    c_D_adj = get_coefficient_adjoint_general_SR(c_D_k, d_R, bases)
    iso_list = [TrivialAlgebraIsomorphism(d_R)] + list(iso_tuple)
    blocks = tensor_product_block_diagonalization(c_D_adj, iso_list)
    labels = [
        tuple(tuple(p) for p in parts)
        for parts in _label_product([list(iso.basis_to.partitions) for iso in iso_tuple])
    ]
    return labels, [np.asarray(to_cpu(b)) for b in blocks]


def _label_product(partition_lists):
    out = [()]
    for parts in partition_lists:
        out = [prev + (p,) for prev in out for p in parts]
    return out


def load_checkpoint(path, n_sectors):
    """Return (F, c_E, [c_D_k], iterations_done) from a checkpoint, or None."""
    import os

    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    if "state_encoder" not in d.files:
        return None
    c_D = [d[f"state_decoder_{k}"] for k in range(n_sectors)]
    return float(d["fidelity"]), d["state_encoder"], c_D, int(d.get("iterations_done", 0))


def run(n, d_R, p, q, seeds, iterations, accuracy, out_path, power_tolerance, verbose,
        chunk=0, resume=False, warm_encoder=None):
    J = [choi_pauli(p), CHOI_IDENTITY]
    iso_A = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, 2))
    iso_B = {k: EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(k, 2)) for k in range(n + 1)}
    sector_isos = [tuple(iso_B[k] for k in comp) for comp in weak_compositions(n, 2)]

    n_sectors = len(sector_isos)
    best_F, best, done = -1.0, None, 0

    if resume:
        state = load_checkpoint(out_path, n_sectors)
        if state is not None:
            best_F, c_E, c_D, done = state
            best = (c_E, c_D)
            print(f"[resume] {out_path}: F = {best_F:.14f} after {done} iterations", flush=True)

    warm_c_E = None
    if warm_encoder is not None and best is None:
        from permqit.utilities.random import isometry_into_symmetric_block
        V = np.load(warm_encoder)
        warm_c_E = isometry_into_symmetric_block(V, d_R, iso_A)
        print(f"[warm] starting from {warm_encoder} (V shape {V.shape})", flush=True)

    # ``chunk`` splits a seed's iteration budget into short runs, each warm-started from the
    # previous one and checkpointed on the way out.  Mathematically identical to one long run
    # (the seesaw is memoryless between iterations), but an interruption costs only one chunk.
    for i, seed in enumerate(seeds):
        remaining = iterations if not (resume and i == 0) else max(0, iterations - done)
        while remaining > 0:
            this = min(chunk, remaining) if chunk > 0 else remaining
            t0 = time.perf_counter()
            result = compute_tensor_product_fidelity_seesaw(
                n=n, d_R=d_R, N=J, d_A=2, d_B=2, q=q,
                repetitions=1, iterations=this, seesaw_accuracy=accuracy,
                power_tolerance=power_tolerance, isometry=True,
                print_iterations=verbose, return_optimizers=True, seed=seed,
                initial_encoder=(best[0] if best is not None else warm_c_E),
                initial_decoders=None if best is None else best[1],
            )
            F, elapsed = result.get_value(), time.perf_counter() - t0
            done += this
            remaining -= this
            improved = F > best_F
            if improved or best is None:
                best_F, best = F, result.get_optimizers()
            save(out_path, n, d_R, p, q, best_F, best, iso_A, sector_isos, seed, done)
            print(f"[seed {seed}] ({i + 1}/{len(seeds)})  iters={done:5d}  F = {F:.14f}"
                  f"   ({elapsed:.0f} s)  -> {out_path}", flush=True)
        done = 0
        best = None if len(seeds) > 1 and i + 1 < len(seeds) else best  # fresh seed next round
    return best_F


def save(path, n, d_R, p, q, F, optimizers, iso_A, sector_isos, seed, iterations_done=0):
    c_E, c_D_list = optimizers
    V = symmetric_isometry_from_choi_coefficients(c_E, d_R, iso_A)  # raises if not an isometry
    payload = {
        "n": np.int64(n), "d_R": np.int64(d_R), "p": np.float64(p), "q": np.float64(q),
        "fidelity": np.float64(F), "seed": np.int64(seed),
        "iterations_done": np.int64(iterations_done),
        # raw optimizers, so a run can be resumed exactly where it stopped
        "state_encoder": np.asarray(to_cpu(c_E)),
        "encoder_isometry_dicke": V,  # (n+1) x d_R, columns = V|i> in the Dicke basis
        "channel": np.bytes_(b"flagged-Pauli"),
    }
    for k, (c_D_k, iso_tuple) in enumerate(zip(c_D_list, sector_isos)):
        payload[f"state_decoder_{k}"] = np.asarray(to_cpu(c_D_k))
        labels, blocks = decoder_blocks(c_D_k, d_R, iso_tuple)
        payload[f"decoder_{k}_labels"] = np.array([str(lab) for lab in labels])
        for j, B in enumerate(blocks):
            payload[f"decoder_{k}_block_{j}"] = B
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    import os
    os.replace(tmp, path)   # atomic: a kill mid-write can never corrupt the checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d-R", type=int, default=2)
    ap.add_argument("--p", type=float, default=DEFAULT_P)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[18])
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--accuracy", type=float, default=1e-13)
    ap.add_argument("--power-tolerance", type=float, default=1e-12)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--chunk", type=int, default=0,
                    help="checkpoint every CHUNK iterations (0 = only at the end of a seed)")
    ap.add_argument("--resume", action="store_true", help="continue from an existing --out file")
    ap.add_argument("--warm-encoder", type=str, default=None,
                    help="start from an isometry stored as an (n+1) x d_R .npy of Dicke "
                         "amplitudes (e.g. one lifted from the n-1 solution)")
    args = ap.parse_args()

    print(f"flagged Pauli channel: n={args.n}, p={args.p:.6f}, q={args.q}, d_R={args.d_R}")
    print(f"seeds={args.seeds}  iterations={args.iterations}  accuracy={args.accuracy:g}", flush=True)
    F = run(args.n, args.d_R, args.p, args.q, args.seeds, args.iterations, args.accuracy,
            args.out, args.power_tolerance, args.verbose, chunk=args.chunk, resume=args.resume,
            warm_encoder=args.warm_encoder)
    print(f"\nbest fidelity = {F:.14f}   ({'>' if F > 0.75 else '<='} 0.75)")


if __name__ == "__main__":
    main()
