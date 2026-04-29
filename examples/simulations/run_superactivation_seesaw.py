"""
Run the superactivation-channel seesaw using the symmetric power-iteration method.

The channel is a 50/50 mixture of Pauli-dephasing (N_0) and identity (N_1), both
qubit-to-qubit (d_A=d_B=2). Sector k has k copies of N_0 and n-k copies of N_1.
"""
import time
import numpy as np

from permqit.power_method.seesaw import (
    compute_tensor_product_fidelity_seesaw,
    DEFAULT_SEESAW_REPETITIONS,
    DEFAULT_SEESAW_ITERATIONS,
)

DEFAULT_P = 1.0 / (1.0 + 2.0 ** 0.5)


def _choi_pauli(p: float) -> np.ndarray:
    return np.array([
        [1 - p, 0, 0, 0],
        [0,     p, 0, 0],
        [0,     0, p, 0],
        [0,     0, 0, 1 - p],
    ], dtype=np.complex128)


_CHOI_IDENTITY = np.array([
    [1.0, 0, 0, 1.0],
    [0,   0, 0, 0  ],
    [0,   0, 0, 0  ],
    [1.0, 0, 0, 1.0],
], dtype=np.complex128)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Superactivation-channel seesaw (power method)")
    parser.add_argument("--n", type=int, default=3, help="Number of channel uses (default: 3). Ignored if --batch is set.")
    parser.add_argument("--p", type=float, default=DEFAULT_P,
                        help=f"Pauli-dephasing parameter (default: {DEFAULT_P:.4f})")
    parser.add_argument("--batch", type=int, nargs="+", metavar="N",
                        help="Run for several n values and print summary (e.g. --batch 13 14 15)")
    parser.add_argument("--runs-per-n", type=int, default=3,
                        help="For batch: number of full runs per n (default: 3)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_SEESAW_ITERATIONS,
                        help=f"Seesaw iterations (default: {DEFAULT_SEESAW_ITERATIONS})")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_SEESAW_REPETITIONS,
                        help=f"Seesaw repetitions (default: {DEFAULT_SEESAW_REPETITIONS})")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    J_pauli    = _choi_pauli(args.p)
    J_identity = _CHOI_IDENTITY

    def _run(n):
        return compute_tensor_product_fidelity_seesaw(
            n=n, d_R=2,
            N=[J_pauli, J_identity],
            d_A=2, d_B=2,
            q=0.5,
            repetitions=args.repetitions,
            iterations=args.iterations,
            print_iterations=not args.quiet,
            verbose=args.verbose,
            return_optimizers=False,
        )

    if args.batch is not None:
        n_list = sorted(set(args.batch))
        runs_per_n = max(1, args.runs_per_n)

        summary = []
        for n in n_list:
            fidelities_this_n = []
            for run in range(runs_per_n):
                if args.verbose:
                    print(f"\n[Batch] n={n} run {run+1}/{runs_per_n} ...", flush=True)
                result = _run(n)
                fidelities_this_n.append(result.get_value())
            top2 = sorted(fidelities_this_n, reverse=True)[:2]
            summary.append((n, top2[0], top2[1] if len(top2) > 1 else top2[0]))

        print(f"\n{'='*70}")
        print("BATCH SUMMARY (top-2 fidelities per n)")
        print(f"{'='*70}")
        print(f"  n   |  1st max fidelity  |  2nd max fidelity")
        print(f"{'-'*70}")
        for n, f1, f2 in summary:
            print(f"  {n:2}  |  {f1:.8f}       |  {f2:.8f}")
        print(f"{'='*70}")
        return

    print(f"{'='*70}")
    print(f"SUPERACTIVATION SEESAW: n={args.n}, p={args.p:.4f}")
    print(f"  Iterations: {args.iterations}   Repetitions: {args.repetitions}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = _run(args.n)
    elapsed = time.perf_counter() - t0

    fidelity = result.get_value()
    print(f"\n{'='*70}")
    print(f"Final fidelity : {fidelity:.8f}")
    print(f"Wall-clock time: {elapsed:.3f} s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
