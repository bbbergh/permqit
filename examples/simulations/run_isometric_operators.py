"""Symmetric seesaw with a **fully-isometric encoder**, for every channel of interest.

Sweeps over n and stores, for each n, the optimal encoder and decoders together with the
entanglement fidelity they achieve.  The encoder is constrained to a genuine isometry
V : C^d_R -> Sym^n(A) (``isometry=True``), so it is fully described by the (dim Sym^n) x d_R
matrix of V in the Dicke basis -- the same description used by Agarwal et al.
[arXiv:2605.09138], whose rank-two symmetric states are exactly Tr_R of this encoder's Choi state.

Channels
--------
``depolarizing``        N_p(rho) = (1-p) rho + p I/2
``amplitude_damping``   A_gamma
``flagged_pauli``       q |0><0|_Z (x) id + (1-q) |1><1|_Z (x) P_p, the superactivation channel;
                        the flag splits n uses into n+1 sectors N_k = P_p^{(x)k} (x) id^{(x)(n-k)},
                        each with its own decoder D_k.

Usage
-----
    uv run python examples/simulations/run_isometric_operators.py \
        --channel depolarizing --param 0.151020 --n-values 1 2 3 4 5 6 \
        --seeds 18 42 137 --out results/depolarizing_isometric.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from examples.channels.amplitude_damping_channel import amplitude_damping_choi  # noqa: E402
from examples.channels.depolarizing_channel import depolarizing_choi  # noqa: E402
from examples.simulations.run_flagged_pauli_operators import (  # noqa: E402
    CHOI_IDENTITY, DEFAULT_P, choi_pauli, decoder_blocks,
)
from permqit.power_method.seesaw import compute_tensor_product_fidelity_seesaw  # noqa: E402
from permqit.representation.combinatorics import weak_compositions  # noqa: E402
from permqit.representation.isomorphism import (  # noqa: E402
    EndSnAlgebraIsomorphism, EndSnBlockDiagonalization,
)
from permqit.utilities.random import symmetric_isometry_from_choi_coefficients  # noqa: E402


def _atomic_savez(path, payload):
    """Write the archive atomically, so an interrupted run can never truncate it."""
    ns = sorted(int(k[1:].split("_")[0]) for k in payload if k.endswith("_fidelity"))
    payload = dict(payload)
    payload["n_values"] = np.array(ns, dtype=np.int64)
    payload["fidelities"] = np.array([float(payload[f"n{n}_fidelity"]) for n in ns])
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def build_channel(name: str, param: float):
    """Returns (N, q, n_types) in the form ``compute_tensor_product_fidelity_seesaw`` expects."""
    if name == "depolarizing":
        return depolarizing_choi(param), 0.5, 1
    if name == "amplitude_damping":
        return amplitude_damping_choi(param), 0.5, 1
    if name == "flagged_pauli":
        return [choi_pauli(param), CHOI_IDENTITY], 0.5, 2
    raise ValueError(f"unknown channel {name!r}")


def sector_isos(n: int, n_types: int):
    iso_B = {k: EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(k, 2)) for k in range(n + 1)}
    return [tuple(iso_B[k] for k in comp) for comp in weak_compositions(n, n_types)]


def run_single_n(n, d_R, N, q, n_types, seeds, iterations, accuracy, power_tolerance, verbose):
    best_F, best_opt, best_seed = -1.0, None, None
    for seed in seeds:
        result = compute_tensor_product_fidelity_seesaw(
            n=n, d_R=d_R, N=N, d_A=2, d_B=2, q=q, repetitions=1, iterations=iterations,
            seesaw_accuracy=accuracy, power_tolerance=power_tolerance, isometry=True,
            print_iterations=verbose, return_optimizers=True, seed=seed,
        )
        if result.get_value() > best_F:
            best_F, best_opt, best_seed = result.get_value(), result.get_optimizers(), seed
    return best_F, best_opt, best_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True,
                    choices=["depolarizing", "amplitude_damping", "flagged_pauli"])
    ap.add_argument("--param", type=float, default=None, help="p or gamma (default: paper value)")
    ap.add_argument("--n-values", type=int, nargs="+", required=True)
    ap.add_argument("--d-R", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[18, 42, 137])
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--accuracy", type=float, default=1e-13)
    ap.add_argument("--power-tolerance", type=float, default=1e-12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep-best", action="store_true",
                    help="merge into an existing --out file, keeping whichever run did better "
                         "at each n (so adding seeds can only improve the stored curve)")
    args = ap.parse_args()

    param = args.param if args.param is not None else (DEFAULT_P if args.channel == "flagged_pauli" else 0.1)
    N, q, n_types = build_channel(args.channel, param)
    d_R = args.d_R

    print(f"{args.channel}: param={param:.8f}  d_R={d_R}  n={args.n_values}")
    print(f"seeds={args.seeds}  iterations={args.iterations}  accuracy={args.accuracy:g}", flush=True)

    payload = {
        "channel": np.bytes_(args.channel.encode()), "param": np.float64(param),
        "d_R": np.int64(d_R), "q": np.float64(q),
        "n_values": np.array(args.n_values, dtype=np.int64),
    }
    previous = {}
    if args.keep_best and os.path.exists(args.out):
        old_npz = np.load(args.out, allow_pickle=True)
        for key in old_npz.files:
            previous[key] = old_npz[key]
        # Carry EVERY previously stored n across, not just the ones being recomputed now -- a
        # top-up run covers a handful of n, and dropping the rest would silently delete them.
        carried = sorted(int(k[1:].split("_")[0]) for k in previous if k.endswith("_fidelity"))
        for key, value in previous.items():
            if key.startswith("n") and key[1:2].isdigit():
                payload[key] = value
        payload["n_values"] = np.array(
            sorted(set(carried) | set(args.n_values)), dtype=np.int64
        )
        print(f"  [keep-best] merging into {args.out}; carrying n = {carried}", flush=True)

    fidelities = []
    for n in args.n_values:
        t0 = time.perf_counter()
        F, opt, seed = run_single_n(n, d_R, N, q, n_types, args.seeds, args.iterations,
                                    args.accuracy, args.power_tolerance, args.verbose)
        prev_F = float(previous.get(f"n{n}_fidelity", -np.inf))
        if prev_F > F:
            # the stored run did better at this n; carry it over untouched
            for key in list(previous):
                if key.startswith(f"n{n}_"):
                    payload[key] = previous[key]
            fidelities.append(prev_F)
            print(f"  n={n:3d}  F = {prev_F:.14f}   (kept previous, this run gave {F:.14f})",
                  flush=True)
            _atomic_savez(args.out, payload)
            continue
        c_E, c_D_list = opt
        iso_A = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, 2))
        V = symmetric_isometry_from_choi_coefficients(c_E, d_R, iso_A)  # raises if not isometric
        payload[f"n{n}_fidelity"] = np.float64(F)
        payload[f"n{n}_seed"] = np.int64(seed)
        payload[f"n{n}_encoder_isometry_dicke"] = V
        for k, (c_D_k, iso_tuple) in enumerate(zip(c_D_list, sector_isos(n, n_types))):
            labels, blocks = decoder_blocks(c_D_k, d_R, iso_tuple)
            payload[f"n{n}_decoder_{k}_labels"] = np.array([str(lab) for lab in labels])
            for j, B in enumerate(blocks):
                payload[f"n{n}_decoder_{k}_block_{j}"] = B
        fidelities.append(F)
        _atomic_savez(args.out, payload)
        print(f"  n={n:3d}  F = {F:.14f}   (seed {seed}, {time.perf_counter()-t0:.0f} s)  -> {args.out}",
              flush=True)
    print("\ndone.")


if __name__ == "__main__":
    main()
