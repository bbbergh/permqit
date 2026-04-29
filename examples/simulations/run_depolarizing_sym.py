"""
Depolarizing channel seesaw simulation for p ∈ [0, 0.2].

Runs both seesaws with CQ-seesaw-level accuracy settings:
  - Symmetric seesaw:     n = 1, ..., n_sym  (default 15)
  - Non-symmetric seesaw: n = 1, ..., n_nonsym (default 7)
  - p range: [0, 0.2], 50 values (configurable)

Accuracy settings:
  - seesaw_iterations   = 500
  - seesaw_accuracy     = 1e-7
  - power_max_iterations = 5000
  - power_tolerance      = 1e-6
  - seeds               = [18, 42, 137]  (3 independent restarts)

Non-symmetric seesaw uses trivial_init=True to guarantee the seesaw always
finds at least the single-copy strategy F_1 = 1 - 3p/4.

Analytical benchmarks:
  - n=1:  F_1(p) = 1 - 3p/4
  - n=5 five-qubit code (first-order):
      F_5(p) = (1-3p/4)^5 + (15p/4)(1-3p/4)^4

Usage
-----
Run with defaults (n_sym=15, n_nonsym=7, p∈[0,0.2], 50 p values)::

    python -m permqit.simulations.run_depolarizing_sym --save results_small_p.npz

Custom parameters::

    python -m permqit.simulations.run_depolarizing_sym --n-sym 10 --n-nonsym 5 --n-p 25 --save results.npz

Plot results::

    python -m permqit.plotting.plot_depolarizing results_small_p.npz
    python -m permqit.plotting.plot_depolarizing results_small_p.npz --curves
"""
import time
import argparse
import numpy as np

from examples.channels.depolarizing_channel import depolarizing_choi
from permqit.power_method.seesaw import compute_tensor_product_fidelity_seesaw, compute_tensor_product_fidelity_without_symmetry_seesaw
from examples.simulations.defaults import (
    SYM_SEESAW_ITERATIONS,
    SYM_SEESAW_ACCURACY,
    SYM_POWER_MAX_ITERATIONS,
    SYM_POWER_TOLERANCE,
    SYM_SEEDS,
    NONSYM_REPS_PER_SEED,
)

NONSYM_MAX_N_WARN = 7  # warn if n_nonsym > this (slow on CPU)

# ── Analytical benchmarks ─────────────────────────────────────────────────────

def fidelity_n1_analytical(p: float) -> float:
    """Optimal single-copy entanglement fidelity: F_1 = 1 - 3p/4."""
    return 1.0 - 0.75 * p


def fidelity_n5_five_qubit_code(p: float) -> float:
    """[[5,1,3]] first-order: F_5(p) = (1-3p/4)^5 + (15p/4)(1-3p/4)^4."""
    q = 1.0 - 0.75 * p
    return q ** 5 + (15.0 * p / 4.0) * q ** 4


# ── Seesaw runners ────────────────────────────────────────────────────────────

def _run_sym(n, N_choi, seeds, iterations, accuracy,
             power_max_iterations=None, power_tolerance=None):
    """Run symmetric seesaw once per seed; sparsity is handled automatically."""
    return [
        compute_tensor_product_fidelity_seesaw(
            n=n, d_R=2, N=N_choi, d_A=2, d_B=2,
            repetitions=1,
            iterations=iterations,
            seesaw_accuracy=accuracy,
            power_max_iterations=power_max_iterations,
            power_tolerance=power_tolerance,
            return_optimizers=False,
            seed=seed,
        ).get_value()
        for seed in seeds
    ]


def _run_nonsym(N_choi, n, seeds, iterations, power_max_iterations,
                power_tolerance, seesaw_accuracy, verbose):
    """Run non-symmetric seesaw with NONSYM_REPS_PER_SEED restarts per seed.

    First seed also prepends the trivial isometry init (|i⟩→|i,0,...,0⟩) to
    guarantee the single-copy strategy F_1=1-3p/4 is always found.
    Returns the best fidelity across all seeds+repetitions as a single value,
    plus a per-seed-best list for printing.
    """
    per_seed_best = []
    for idx, seed in enumerate(seeds):
        result = compute_tensor_product_fidelity_without_symmetry_seesaw(
            N=N_choi,
            n=n,
            d_R=2, d_A=2, d_B=2,
            repetitions=NONSYM_REPS_PER_SEED,
            iterations=iterations,
            seesaw_accuracy=seesaw_accuracy,
            power_max_iterations=power_max_iterations,
            power_tolerance=power_tolerance,
            verbose=verbose,
            seed=seed,
        )
        per_seed_best.append(result.get_value())
    return per_seed_best


# ── Main simulation ───────────────────────────────────────────────────────────

def run_depolarizing_sym(
    n_sym: int = 10,
    n_nonsym: int = 5,
    n_p: int = 20,
    p_min: float = 0.0,
    p_max: float = 0.2,
    seesaw_iterations: int = SYM_SEESAW_ITERATIONS,
    seesaw_accuracy: float = SYM_SEESAW_ACCURACY,
    power_max_iterations: int = SYM_POWER_MAX_ITERATIONS,
    power_tolerance: float = SYM_POWER_TOLERANCE,
    seeds=None,
    verbose: bool = False,
    save_path: str|None = None,
):
    """Run the depolarizing channel simulation on p ∈ [p_min, p_max].

    Args:
        n_sym: Max n for symmetric seesaw (n=1,...,n_sym).
        n_nonsym: Max n for non-symmetric seesaw (n=1,...,n_nonsym).
        n_p: Number of p values in [p_min, p_max].
        p_min: Lower bound for depolarizing parameter.
        p_max: Upper bound for depolarizing parameter.
        seesaw_iterations: Max seesaw iterations per restart.
        seesaw_accuracy: Seesaw convergence threshold.
        power_max_iterations: Max power iterations per seesaw step.
        power_tolerance: Power method convergence threshold.
        seeds: List of integer seeds for independent restarts.
        verbose: Print per-iteration progress.
        save_path: Save results to this .npz file.

    Returns:
        dict with keys:
            p_values:    (n_p,) p values
            fid_sym:     (n_sym, n_p) symmetric seesaw best fidelities
            fid_nonsym:  (n_nonsym, n_p) non-symmetric seesaw best fidelities
            fid_n1:      (n_p,) analytical n=1 fidelities
            fid_n5_code: (n_p,) five-qubit code fidelities
    """
    if seeds is None:
        seeds = SYM_SEEDS

    if n_nonsym > NONSYM_MAX_N_WARN:
        print(f"WARNING: n_nonsym={n_nonsym} > {NONSYM_MAX_N_WARN}. The non-symmetric seesaw "
              f"works in a {2**n_nonsym}-dimensional space — expect very long runtimes on CPU.")
        print(f"         Consider reducing --n-nonsym to ≤ {NONSYM_MAX_N_WARN}.")

    p_values = np.linspace(p_min, p_max, n_p)

    fid_sym    = np.full((n_sym,    n_p), np.nan)
    fid_nonsym = np.full((n_nonsym, n_p), np.nan)
    fid_n1  = np.array([fidelity_n1_analytical(p) for p in p_values])
    fid_n5  = np.array([fidelity_n5_five_qubit_code(p) for p in p_values])

    print(f"{'='*70}")
    print(f"DEPOLARIZING CHANNEL — p ∈ [{p_min}, {p_max}]")
    print(f"{'='*70}")
    print(f"  Symmetric seesaw:     n = 1, ..., {n_sym}")
    print(f"  Non-symmetric seesaw: n = 1, ..., {n_nonsym}  (trivial_init on first seed)")
    print(f"  p values:             {n_p}")
    print(f"  Seeds:                {seeds}")
    print(f"  Seesaw iterations:    {seesaw_iterations}  (default: {SYM_SEESAW_ITERATIONS})")
    print(f"  Seesaw accuracy:      {seesaw_accuracy:.0e}  (default: {SYM_SEESAW_ACCURACY:.0e})")
    print(f"  Power max iter:       {power_max_iterations}")
    print(f"  Power tolerance:      {power_tolerance:.0e}")
    print(f"{'='*70}")


    total_t0 = time.perf_counter()

    for p_idx, p in enumerate(p_values):
        print(f"\n{'='*70}")
        print(f"p = {p:.5f}  [{p_idx+1}/{n_p}]"
              f"   F_1={fidelity_n1_analytical(p):.6f}"
              f"   F_5(5qb)={fidelity_n5_five_qubit_code(p):.6f}")
        print(f"{'='*70}")

        N_choi = depolarizing_choi(p, d=2)

        # ── Symmetric seesaw n=1,...,n_sym ────────────────────────────────
        print("  [Symmetric seesaw]")
        best_sym_so_far = -1.0
        for n in range(1, n_sym + 1):
            t0 = time.perf_counter()
            # depolarizing_channel_with_support: uses precomputed orbit bases (per n)
            # and evaluates coefficients as alpha^g0 * beta^g1 * gamma^g2 —
            # O(C(n+5,5)) vectorized ops, vs O(C(n+15,15)) for the naive method.
            fs = _run_sym(n, N_choi, seeds, seesaw_iterations, seesaw_accuracy,
                          power_max_iterations=power_max_iterations,
                          power_tolerance=power_tolerance)
            best_f = max(fs)
            elapsed = time.perf_counter() - t0
            fid_sym[n - 1, p_idx] = best_f
            best_sym_so_far = max(best_sym_so_far, best_f)
            seed_str = "  ".join(f"{f:.6f}" for f in fs)
            print(f"    n={n:2d}: [{seed_str}]  best={best_f:.6f}  (max_n={best_sym_so_far:.6f})"
                  f"  ({elapsed:.2f}s)")

        # ── Non-symmetric seesaw n=1,...,n_nonsym ─────────────────────────
        print("  [Non-symmetric seesaw]")
        best_nonsym_so_far = -1.0
        for n in range(1, n_nonsym + 1):
            t0 = time.perf_counter()
            fs = _run_nonsym(N_choi, n, seeds, seesaw_iterations,
                             power_max_iterations, power_tolerance, seesaw_accuracy, verbose)
            best_f = max(fs)
            elapsed = time.perf_counter() - t0
            fid_nonsym[n - 1, p_idx] = best_f
            best_nonsym_so_far = max(best_nonsym_so_far, best_f)
            seed_str = "  ".join(f"{f:.6f}" for f in fs)
            print(f"    n={n:2d}: [{seed_str}]  best={best_f:.6f}  (max_n={best_nonsym_so_far:.6f})  ({elapsed:.2f}s)")

        # ── Summary ───────────────────────────────────────────────────────
        best_overall_sym = np.nanmax(fid_sym[:, p_idx])
        best_overall_nonsym = np.nanmax(fid_nonsym[:, p_idx])
        print(f"\n  Summary p={p:.5f}:")
        print(f"    best sym    (n=1..{n_sym}):    {best_overall_sym:.6f}")
        print(f"    best nonsym (n=1..{n_nonsym}): {best_overall_nonsym:.6f}")
        print(f"    F_1 analytical:              {fidelity_n1_analytical(p):.6f}")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Total simulation time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    results = dict(
        p_values=p_values,
        fid_sym=fid_sym,
        fid_nonsym=fid_nonsym,
        fid_n1=fid_n1,
        fid_n5_code=fid_n5,
    )

    if save_path is not None:
        np.savez(save_path, **results)  # ty:ignore[invalid-argument-type]
        print(f"\nResults saved to {save_path}")
        print(f"\nTo plot:")
        print(f"  python -m permqit.plotting.plot_depolarizing {save_path}")
        print(f"  python -m permqit.plotting.plot_depolarizing {save_path} --curves")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Depolarizing channel seesaw on p ∈ [p_min, p_max] with CQ-level accuracy. "
            "Symmetric seesaw for n=1,...,n_sym; non-symmetric for n=1,...,n_nonsym."
        )
    )
    parser.add_argument("--n-sym", type=int, default=20,
                        help="Max n for symmetric seesaw (default: 15)")
    parser.add_argument("--n-nonsym", type=int, default=5,
                        help=f"Max n for non-symmetric seesaw (default: 7; warn if > {NONSYM_MAX_N_WARN})")
    parser.add_argument("--n-p", type=int, default=50,
                        help="Number of p values (default: 50)")
    parser.add_argument("--p-min", type=float, default=0.0,
                        help="Min depolarizing parameter (default: 0.0)")
    parser.add_argument("--p-max", type=float, default=0.2,
                        help="Max depolarizing parameter (default: 0.2)")
    parser.add_argument("--seesaw-iterations", type=int, default=SYM_SEESAW_ITERATIONS,
                        help=f"Seesaw iterations per restart (default: {SYM_SEESAW_ITERATIONS})")
    parser.add_argument("--seesaw-accuracy", type=float, default=SYM_SEESAW_ACCURACY,
                        help=f"Seesaw convergence threshold (default: {SYM_SEESAW_ACCURACY:.0e})")
    parser.add_argument("--power-iterations", type=int, default=SYM_POWER_MAX_ITERATIONS,
                        help=f"Max power method iterations (default: {SYM_POWER_MAX_ITERATIONS})")
    parser.add_argument("--power-tol", type=float, default=SYM_POWER_TOLERANCE,
                        help=f"Power method tolerance (default: {SYM_POWER_TOLERANCE:.0e})")
    parser.add_argument("--seeds", type=int, nargs="+", default=SYM_SEEDS,
                        help=f"Random seeds (default: {SYM_SEEDS})")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save results (.npz)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed per-iteration progress")
    args = parser.parse_args()

    run_depolarizing_sym(
        n_sym=args.n_sym,
        n_nonsym=args.n_nonsym,
        n_p=args.n_p,
        p_min=args.p_min,
        p_max=args.p_max,
        seesaw_iterations=args.seesaw_iterations,
        seesaw_accuracy=args.seesaw_accuracy,
        power_max_iterations=args.power_iterations,
        power_tolerance=args.power_tol,
        seeds=args.seeds,
        verbose=args.verbose,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
