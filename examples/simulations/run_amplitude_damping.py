"""Amplitude Damping Channel (ADC) seesaw simulation for γ ∈ [0, 1].

Runs both seesaws:
  - Symmetric seesaw:     n = 1, ..., n_sym  (default 20); plot takes max over n
  - Non-symmetric seesaw: n = 5 only (fixed)

Analytical benchmarks:
  - Uncoded (n=1 trivial):  F_e(A_γ) = ((1 + sqrt(1-γ)) / 2)²
  - Leung 4-qubit [[4,1]] code (Leung et al. 1997):
        F_e = 1/2 + s/(2√2) + γ - sγ/(2√2) - 15γ²/4 + 7γ³/2 - γ⁴
        where s = sqrt(1 + (1-γ)^4)

Usage
-----
Run with defaults (n_sym=20, 50 γ values)::

    python -m permqit.simulations.run_amplitude_damping --save results_adc.npz

Custom parameters::

    python -m permqit.simulations.run_amplitude_damping --n-sym 10 --n-gamma 25 --save results.npz

Plot results::

    python -m permqit.plotting.plot_amplitude_damping results_adc.npz
"""
import time
import argparse
import numpy as np

from examples.channels.amplitude_damping_channel import amplitude_damping_choi, ADC_SUPPORT_GROUPED
from permqit.power_method.seesaw import compute_tensor_product_fidelity_seesaw, compute_tensor_product_fidelity_without_symmetry_seesaw
from examples.simulations.defaults import (
    SYM_SEESAW_ITERATIONS,
    SYM_SEESAW_ACCURACY,
    SYM_POWER_MAX_ITERATIONS,
    SYM_POWER_TOLERANCE,
    SYM_SEEDS,
    NONSYM_REPS_PER_SEED,
)

NONSYM_N = 5  # non-symmetric seesaw fixed at n=5


# ── Analytical benchmarks ─────────────────────────────────────────────────────

def fidelity_uncoded_adc(gamma: float) -> float:
    """Entanglement fidelity of A_γ with trivial (n=1) encoding.

    F_e = ((1 + sqrt(1-γ)) / 2)²
    """
    return ((1.0 + np.sqrt(1.0 - gamma)) / 2.0) ** 2


def fidelity_leung_4qubit(gamma: float) -> float:
    """Entanglement fidelity of Leung et al. [[4,1]] code for the ADC.

    Reference: Leung, Nielsen, Chuang, Yamamoto (1997).

    F_e = 1/2 + s/(2√2) + γ - sγ/(2√2) - 15γ²/4 + 7γ³/2 - γ⁴
    where s = sqrt(1 + (1-γ)^4).
    """
    s = np.sqrt(1.0 + (1.0 - gamma) ** 4)
    sqrt2 = np.sqrt(2.0)
    return (
        0.5
        + s / (2.0 * sqrt2)
        + gamma
        - s * gamma / (2.0 * sqrt2)
        - 15.0 * gamma ** 2 / 4.0
        + 7.0 * gamma ** 3 / 2.0
        - gamma ** 4
    )


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


def _run_nonsym(N_choi, seeds, iterations, power_max_iterations,
                power_tolerance, seesaw_accuracy, verbose,
                trivial_init_first_seed=True, reps_per_seed=None,
                isometric_seed=False):
    """Run non-symmetric seesaw for n=NONSYM_N with restarts per seed.

    Args:
        trivial_init_first_seed: If True, first seed uses trivial_init=True as a
            lower-bound guarantee. For large γ this should be False — the trivial
            isometry (|ψ>→|ψ,0,...>) collapses to all-zeros output under heavy
            damping, creating a local minimum the seesaw cannot escape.
        reps_per_seed: Repetitions per seed (default: NONSYM_REPS_PER_SEED).
        isometric_seed: If True, enable ISOMETRIC_SEED mode — extra trials with
            random isometry encoders and decoder warm-start disabled, so that
            different starting decoders produce genuinely different trajectories.
    """
    if reps_per_seed is None:
        reps_per_seed = NONSYM_REPS_PER_SEED
    per_seed_best = []
    for idx, seed in enumerate(seeds):
        result = compute_tensor_product_fidelity_without_symmetry_seesaw(
            n=NONSYM_N,
            d_R=2,
            N=N_choi,
            d_A=2, d_B=2,
            repetitions=reps_per_seed,
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

def run_amplitude_damping(
    n_sym: int = 20,
    n_gamma: int = 50,
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
    seesaw_iterations: int = SYM_SEESAW_ITERATIONS,
    seesaw_accuracy: float = SYM_SEESAW_ACCURACY,
    power_max_iterations: int = SYM_POWER_MAX_ITERATIONS,
    power_tolerance: float = SYM_POWER_TOLERANCE,
    seeds=None,
    verbose: bool = False,
    save_path: str|None = None,
):
    """Run the amplitude damping channel simulation on γ ∈ [gamma_min, gamma_max].

    Non-symmetric seesaw runs for n=5 only (NONSYM_N=5 is fixed).

    Args:
        n_sym: Max n for symmetric seesaw (n=1,...,n_sym).
        n_gamma: Number of γ values in [gamma_min, gamma_max].
        gamma_min: Lower bound for damping parameter.
        gamma_max: Upper bound for damping parameter.
        seesaw_iterations: Max seesaw iterations per restart.
        seesaw_accuracy: Seesaw convergence threshold.
        power_max_iterations: Max power iterations per seesaw step.
        power_tolerance: Power method convergence threshold.
        seeds: List of integer seeds for independent restarts.
        verbose: Print per-iteration progress.
        save_path: Save results to this .npz file.

    Returns:
        dict with keys:
            gamma_values:  (n_gamma,) γ values
            fid_sym:       (n_sym, n_gamma) symmetric seesaw best fidelities
            fid_nonsym:    (n_gamma,) non-symmetric seesaw best fidelities (n=5 only)
            fid_uncoded:   (n_gamma,) uncoded analytical fidelities
            fid_leung4:    (n_gamma,) Leung 4-qubit code fidelities
    """
    if seeds is None:
        seeds = SYM_SEEDS

    gamma_values = np.linspace(gamma_min, gamma_max, n_gamma)

    fid_sym    = np.full((n_sym,   n_gamma), np.nan)
    fid_nonsym = np.full((n_gamma,),         np.nan)
    fid_uncoded = np.array([fidelity_uncoded_adc(g) for g in gamma_values])
    fid_leung4  = np.array([fidelity_leung_4qubit(g) for g in gamma_values])

    print(f"{'='*70}")
    print(f"AMPLITUDE DAMPING CHANNEL — γ ∈ [{gamma_min}, {gamma_max}]")
    print(f"{'='*70}")
    print(f"  Symmetric seesaw:     n = 1, ..., {n_sym}")
    print(f"  Non-symmetric seesaw: n = {NONSYM_N} (fixed)")
    print(f"  γ values:             {n_gamma}")
    print(f"  Seeds:                {seeds}")
    print(f"  Seesaw iterations:    {seesaw_iterations}  (default: {SYM_SEESAW_ITERATIONS})")
    print(f"  Seesaw accuracy:      {seesaw_accuracy:.0e}  (default: {SYM_SEESAW_ACCURACY:.0e})")
    print(f"  Power max iter:       {power_max_iterations}")
    print(f"  Power tolerance:      {power_tolerance:.0e}")
    print(f"{'='*70}")


    total_t0 = time.perf_counter()

    for g_idx, gamma in enumerate(gamma_values):
        f_uncoded = fidelity_uncoded_adc(gamma)
        f_leung   = fidelity_leung_4qubit(gamma)
        print(f"\n{'='*70}")
        print(f"γ = {gamma:.5f}  [{g_idx+1}/{n_gamma}]"
              f"   F_uncoded={f_uncoded:.6f}"
              f"   F_Leung4={f_leung:.6f}")
        print(f"{'='*70}")

        N_choi = amplitude_damping_choi(gamma)

        # ── Symmetric seesaw n=1,...,n_sym ────────────────────────────────
        print("  [Symmetric seesaw]")
        best_sym_so_far = -1.0
        for n in range(1, n_sym + 1):
            t0 = time.perf_counter()
            fs = _run_sym(n, N_choi, seeds, seesaw_iterations, seesaw_accuracy,
                          power_max_iterations=power_max_iterations,
                          power_tolerance=power_tolerance)
            best_f = max(fs)
            elapsed = time.perf_counter() - t0
            fid_sym[n - 1, g_idx] = best_f
            best_sym_so_far = max(best_sym_so_far, best_f)
            seed_str = "  ".join(f"{f:.6f}" for f in fs)
            print(f"    n={n:2d}: [{seed_str}]  best={best_f:.6f}  (max_n={best_sym_so_far:.6f})"
                  f"  ({elapsed:.2f}s)")

        # ── Non-symmetric seesaw n=5 ───────────────────────────────────────
        print(f"  [Non-symmetric seesaw  n={NONSYM_N}]")
        t0 = time.perf_counter()
        fs = _run_nonsym(N_choi, seeds, seesaw_iterations,
                         power_max_iterations, power_tolerance, seesaw_accuracy, verbose)
        best_f = max(fs)
        elapsed = time.perf_counter() - t0
        fid_nonsym[g_idx] = best_f
        seed_str = "  ".join(f"{f:.6f}" for f in fs)
        print(f"    n={NONSYM_N}: [{seed_str}]  best={best_f:.6f}  ({elapsed:.2f}s)")

        # ── Summary ───────────────────────────────────────────────────────
        best_overall_sym = np.nanmax(fid_sym[:, g_idx])
        print(f"\n  Summary γ={gamma:.5f}:")
        print(f"    best sym    (n=1..{n_sym}): {best_overall_sym:.6f}")
        print(f"    best nonsym (n={NONSYM_N}):       {fid_nonsym[g_idx]:.6f}")
        print(f"    F_uncoded:                {f_uncoded:.6f}")
        print(f"    F_Leung4:                 {f_leung:.6f}")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'='*70}")
    print(f"Total simulation time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    results = dict(
        gamma_values=gamma_values,
        fid_sym=fid_sym,
        fid_nonsym=fid_nonsym,
        fid_uncoded=fid_uncoded,
        fid_leung4=fid_leung4,
    )

    if save_path is not None:
        np.savez(save_path, **results)  # ty:ignore[invalid-argument-type]
        print(f"\nResults saved to {save_path}")
        print(f"\nTo plot:")
        print(f"  python -m permqit.plotting.plot_amplitude_damping {save_path}")

    return results


# ── Targeted re-run ──────────────────────────────────────────────────────────

# Extra seeds for re-runs (extends SYM_SEEDS without repeating)
_EXTRA_SEEDS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                3000, 3001, 3002, 3003, 3004]


def rerun_nonsym_selected(
    load_path: str,
    gamma_indices=None,
    last_n: int|None = None,
    seesaw_iterations: int = SYM_SEESAW_ITERATIONS,
    seesaw_accuracy: float = SYM_SEESAW_ACCURACY,
    power_max_iterations: int = SYM_POWER_MAX_ITERATIONS,
    power_tolerance: float = SYM_POWER_TOLERANCE,
    seeds=None,
    reps_per_seed: int = 5,
    verbose: bool = False,
    save_path: str|None = None,
):
    """Re-run the non-symmetric seesaw for selected γ indices and update stored results.

    For large γ, the default trivial_init=True traps the seesaw in the uncoded
    local minimum (the trivial isometry |ψ>→|ψ,0,...> collapses to all-zeros
    output under heavy damping, giving F_e ≈ uncoded). This re-run disables
    trivial_init for all seeds so random initialization can explore better codes.

    Uses more seeds and repetitions than the default run to compensate.

    Args:
        load_path: Path to existing .npz results file.
        gamma_indices: Explicit list of γ indices to re-run. Mutually exclusive
            with last_n.
        last_n: Re-run the last N γ values (alternative to gamma_indices).
        seeds: Seeds to use (default: SYM_SEEDS + _EXTRA_SEEDS = 30 seeds).
        reps_per_seed: Seesaw random restarts per seed (default: 5, vs 3 default).
        save_path: Where to save updated results (defaults to load_path).

    Returns:
        Updated results dict with fid_nonsym patched at the selected indices.
    """
    loaded = np.load(load_path)
    results = {k: loaded[k].copy() for k in loaded.files}

    gamma_values = results["gamma_values"]
    n_gamma = len(gamma_values)

    if gamma_indices is None and last_n is None:
        raise ValueError("Provide gamma_indices or last_n.")
    if gamma_indices is not None and last_n is not None:
        raise ValueError("Provide gamma_indices OR last_n, not both.")
    if last_n is not None:
        gamma_indices = list(range(n_gamma - last_n, n_gamma))

    if seeds is None:
        seeds = SYM_SEEDS + _EXTRA_SEEDS  # 30 seeds total

    print(f"{'='*70}")
    print(f"RE-RUN: Non-symmetric seesaw (n={NONSYM_N}) — selected γ values")
    print(f"{'='*70}")
    print(f"  γ indices to re-run:  {gamma_indices}")
    print(f"  γ values:             {[float(gamma_values[i]) for i in gamma_indices or []]}")
    print(f"  Seeds:                {len(seeds)} seeds (trivial_init=False for all)")
    print(f"  Reps per seed:        {reps_per_seed}")
    print(f"  Seesaw iterations:    {seesaw_iterations}")
    print(f"  Power tolerance:      {power_tolerance:.0e}")
    print(f"{'='*70}\n")

    total_t0 = time.perf_counter()

    for g_idx in gamma_indices or []:
        gamma = float(gamma_values[g_idx])
        f_uncoded = fidelity_uncoded_adc(gamma)
        f_leung   = fidelity_leung_4qubit(gamma)
        old_val   = float(results["fid_nonsym"][g_idx])

        print(f"γ = {gamma:.5f}  [idx={g_idx}]"
              f"   F_uncoded={f_uncoded:.6f}"
              f"   F_Leung4={f_leung:.6f}"
              f"   old={old_val:.6f}")

        N_choi = amplitude_damping_choi(gamma)
        t0 = time.perf_counter()

        # trivial_init_first_seed=False: all seeds start from random initialization.
        # Near γ=1, trivial_init produces the |ψ>→|ψ,0,...> isometry which collapses
        # to all-zeros under heavy damping — the seesaw can't escape this minimum.
        # isometric_seed=True: add extra ISOMETRIC_SEED trials (random isometry
        # encoder + decoder warm-start disabled) to escape fixed points where the
        # warm-start collapses all seeds to the same local minimum.
        fs = _run_nonsym(
            N_choi, seeds, seesaw_iterations,
            power_max_iterations, power_tolerance, seesaw_accuracy,
            verbose,
            trivial_init_first_seed=False,
            reps_per_seed=reps_per_seed,
            isometric_seed=True,
        )
        best_f = max(fs)
        elapsed = time.perf_counter() - t0

        # Only update if we found something strictly better
        if best_f > old_val:
            results["fid_nonsym"][g_idx] = best_f
            marker = f"  ↑ improved {old_val:.6f} → {best_f:.6f}"
        else:
            marker = f"  (no improvement over {old_val:.6f})"

        seed_str = "  ".join(f"{f:.6f}" for f in fs)
        print(f"  [{seed_str}]")
        print(f"  best={best_f:.6f}{marker}  ({elapsed:.2f}s)\n")

    total_elapsed = time.perf_counter() - total_t0
    print(f"{'='*70}")
    print(f"Re-run total time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    if save_path is None:
        save_path = load_path
    np.savez(save_path, **results)
    print(f"Results saved to {save_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Amplitude damping channel seesaw on γ ∈ [gamma_min, gamma_max]. "
            f"Symmetric seesaw for n=1,...,n_sym; non-symmetric for n={NONSYM_N} only.\n\n"
            "Re-run mode (--rerun): loads existing .npz and re-runs non-symmetric seesaw\n"
            "for selected γ indices with more seeds and no trivial_init."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Full run args ──────────────────────────────────────────────────────────
    parser.add_argument("--n-sym", type=int, default=20,
                        help="Max n for symmetric seesaw (default: 20)")
    parser.add_argument("--n-gamma", type=int, default=50,
                        help="Number of γ values (default: 50)")
    parser.add_argument("--gamma-min", type=float, default=0.0,
                        help="Min damping parameter (default: 0.0)")
    parser.add_argument("--gamma-max", type=float, default=1.0,
                        help="Max damping parameter (default: 1.0)")
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
    # ── Re-run args ────────────────────────────────────────────────────────────
    parser.add_argument("--rerun", type=str, default=None, metavar="RESULTS_FILE",
                        help="Load existing .npz and re-run non-symmetric seesaw for "
                             "selected γ indices (use with --rerun-last or --rerun-indices)")
    parser.add_argument("--rerun-last", type=int, default=None, metavar="N",
                        help="Re-run last N γ values (use with --rerun)")
    parser.add_argument("--rerun-indices", type=int, nargs="+", default=None,
                        help="Explicit γ indices to re-run (use with --rerun)")
    parser.add_argument("--rerun-reps", type=int, default=5,
                        help="Repetitions per seed in re-run mode (default: 5)")
    args = parser.parse_args()

    if args.rerun is not None:
        # ── Re-run mode ───────────────────────────────────────────────────────
        if args.rerun_last is None and args.rerun_indices is None:
            parser.error("--rerun requires --rerun-last N or --rerun-indices i1 i2 ...")
        rerun_nonsym_selected(
            load_path=args.rerun,
            gamma_indices=args.rerun_indices,
            last_n=args.rerun_last,
            seesaw_iterations=args.seesaw_iterations,
            seesaw_accuracy=args.seesaw_accuracy,
            power_max_iterations=args.power_iterations,
            power_tolerance=args.power_tol,
            reps_per_seed=args.rerun_reps,
            verbose=args.verbose,
            save_path=args.save if args.save else args.rerun,
        )
    else:
        # ── Full run mode ─────────────────────────────────────────────────────
        run_amplitude_damping(
            n_sym=args.n_sym,
            n_gamma=args.n_gamma,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
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
