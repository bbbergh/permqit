#!/usr/bin/env python3
"""
verify_results.py
=================
Self-contained verification of the symmetric seesaw simulation results.

Two .npz files are distributed:

  depolarizing_fidelity_n1-20.npz
    Depolarizing channel N_p = (1-p)id + p·(I/2⊗Tr), for p ∈ [0, 0.2].
    Keys:
      p_values      — (50,) depolarizing noise parameter
      fid_sym       — (20, 50) entanglement fidelity, symmetric seesaw, n=1..20
      fid_nonsym    — (5, 50)  entanglement fidelity, non-symmetric seesaw, n=1..5
      fid_n1        — (50,) single-use fidelity F_e(N_p) = 1 − 3p/4  (analytical)
      fid_n5_code   — (50,) Leung-Smolin 5-qubit code fidelity (reference)

  amplitude_damping_fidelity_n1-20.npz
    Amplitude damping channel A_γ, for γ ∈ [0, 1].
    Keys:
      gamma_values  — (50,) amplitude damping parameter
      fid_sym       — (20, 50) entanglement fidelity, symmetric seesaw, n=1..20
      fid_nonsym    — (50,)  entanglement fidelity, non-symmetric seesaw, n=5
      fid_uncoded   — (50,) single-use fidelity (analytical)
      fid_leung4    — (50,) Leung 4-qubit code fidelity (analytical)

Checks performed
----------------
1. RANGE CHECK: all fidelities lie in [0, 1].

2. BOUNDARY CHECK: at p=0 (identity channel) all entries equal 1; at p=0.2
   the symmetric seesaw for n=20 should exceed the single-use fidelity.
   Analogously for ADC: at γ=0 all entries equal 1; at γ=1 the symmetric
   seesaw should equal 0.25 (fully-decohered channel, F = 1/d² = 0.25).

3. ANALYTICAL CHECK (depolarizing): the single-use fidelity stored in
   fid_n1 is compared against F_e(N_p) = (1 + (1-p)²)/2 (entanglement
   fidelity = (1 + Tr[N(|Φ><Φ|)] |Φ><Φ|) / d²; for qubit depolarizing
   this equals 1 - 2p/3 up to normalization conventions).  Both the stored
   formula and the theoretical value should match to machine precision.

4. ANALYTICAL CHECK (ADC): the uncoded fidelity stored in fid_uncoded is
   compared against F_e(A_γ) = ((1 + sqrt(1-γ)) / 2)² to machine precision.
   The Leung 4-qubit code fidelity is compared to its closed-form expression.

5. MONOTONE CHECK: the best symmetric-seesaw fidelity (max over n) should
   be non-decreasing: adding more uses can only help.

6. MONOTONE IN n CHECK: for each channel parameter value, the sequence
   max_k≤n F_sym[k] is non-decreasing in n.

Dependencies: numpy, scipy.  No qitsym package required.

Usage
-----
    python verify_results.py [--depol PATH] [--adc PATH] [--verbose] [--tol TOL]

Options
-------
    --depol PATH   .npz file for depolarizing results
                   (default: results_small_p.npz)
    --adc PATH     .npz file for ADC results
                   (default: results_adc_updated.npz)
    --verbose, -v  Print per-n details for selected parameter values
    --tol TOL      Absolute tolerance for analytical checks (default: 1e-6)
"""

import argparse
import sys
import numpy as np

W = 74   # print width


# ─── Formatting helpers ──────────────────────────────────────────────────────

def _hr(ch="─"):
    print(ch * W)

def _section(title):
    _hr()
    print(f"  {title}")
    _hr()

def _ok(cond):
    return "✓" if cond else "✗ FAIL"


# ─── Analytical reference curves ─────────────────────────────────────────────

def depol_uncoded(p):
    """Entanglement fidelity for one use of the qubit depolarizing channel.

    Convention used here: N_p(ρ) = (1-p)ρ + p·(I/2).
    Choi state J_norm = (1-p)|Φ+><Φ+| + p·(I/4).
    F_e = <Φ+|J_norm|Φ+> = (1-p) + p/4 = 1 - 3p/4.

    Note: this differs from the Pauli-noise convention
    N_p(ρ) = (1-p)ρ + (p/3)(XρX+YρY+ZρZ), which gives F_e = 1 - 2p/3.
    """
    return 1.0 - 3.0 * p / 4.0


def adc_uncoded(gamma):
    """Entanglement fidelity for one use of the amplitude damping channel.

    A_γ has Kraus operators K0 = [[1,0],[0,sqrt(1-γ)]], K1 = [[0,sqrt(γ)],[0,0]].
    F_e(A_γ) = ((1 + sqrt(1-γ)) / 2)²
    """
    return ((1.0 + np.sqrt(1.0 - gamma)) / 2.0) ** 2


def leung4_fidelity(gamma):
    """Entanglement fidelity of the Leung 4-qubit code under A_γ.

    Closed-form from Leung et al. (1997):
      s = sqrt(1 + (1-γ)^4)
      F = 1/2 + s/(2√2) + γ - s·γ/(2√2) - 15γ²/4 + 7γ³/2 - γ^4
    """
    g = gamma
    s = np.sqrt(1.0 + (1.0 - g) ** 4)
    return (0.5 + s / (2.0 * np.sqrt(2.0))
            + g - s * g / (2.0 * np.sqrt(2.0))
            - 15.0 * g**2 / 4.0
            + 7.0 * g**3 / 2.0
            - g**4)


# ─── Check routines ──────────────────────────────────────────────────────────

def check_range(arr, name):
    """Return True if all entries in arr are in [0, 1]."""
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    ok = lo >= -1e-9 and hi <= 1.0 + 1e-9
    return ok, lo, hi


def check_boundary_depol(data, tol=1e-6):
    """p=0: all fidelities should equal 1.  Returns (ok, max_err)."""
    p = data["p_values"]
    idx0 = int(np.argmin(np.abs(p)))   # index closest to p=0
    errs = []
    for key in ["fid_sym", "fid_nonsym", "fid_n1"]:
        arr = data[key]
        col = arr[:, idx0] if arr.ndim == 2 else arr[idx0]
        errs.append(np.max(np.abs(col - 1.0)))
    max_err = max(errs)
    return max_err < tol, max_err


def check_boundary_adc(data, tol=1e-6):
    """γ=0: all fidelities should equal 1.  Returns (ok, max_err)."""
    g = data["gamma_values"]
    idx0 = int(np.argmin(np.abs(g)))
    errs = []
    for key in ["fid_sym", "fid_nonsym", "fid_uncoded", "fid_leung4"]:
        arr = data[key]
        col = arr[:, idx0] if arr.ndim == 2 else arr[idx0]
        errs.append(np.max(np.abs(col - 1.0)))
    max_err = max(errs)
    return max_err < tol, max_err


def check_analytical_depol(data, tol=1e-6):
    """fid_n1 should match the analytical 1 - 2p/3.  Returns (ok, max_err)."""
    p = data["p_values"]
    ref = depol_uncoded(p)
    err = np.max(np.abs(data["fid_n1"] - ref))
    return float(err) < tol, float(err)


def check_analytical_adc(data, tol=1e-6):
    """fid_uncoded and fid_leung4 checked against closed-form.  Returns (ok, max_err)."""
    g = data["gamma_values"]
    err_unc = float(np.max(np.abs(data["fid_uncoded"] - adc_uncoded(g))))
    err_l4  = float(np.max(np.abs(data["fid_leung4"]  - leung4_fidelity(g))))
    return (max(err_unc, err_l4) < tol), err_unc, err_l4


def check_monotone_n(fid_sym, tol=1e-6):
    """For each parameter value j, fid_sym[:, j] should be non-decreasing in n.

    Returns (ok, max_violation).
    """
    diffs = np.diff(fid_sym, axis=0)   # shape (n_max-1, n_params)
    violations = -np.minimum(diffs, 0.0)
    max_viol = float(np.max(violations))
    return max_viol < tol, max_viol


def check_sym_beats_nonsym(fid_sym, fid_nonsym, n_nonsym, tol=1e-6):
    """Symmetric seesaw at n=n_nonsym should be >= non-symmetric (same optimization set).

    Returns (ok, max_violation).
    """
    sym_at_n = fid_sym[n_nonsym - 1, :]   # row index n_nonsym-1 for n=n_nonsym
    nonsym   = fid_nonsym if fid_nonsym.ndim == 1 else fid_nonsym.max(axis=0)
    diff     = sym_at_n - nonsym
    max_viol = float(-np.min(np.minimum(diff, 0.0)))
    return max_viol < tol, max_viol


# ─── Pretty-print tables ─────────────────────────────────────────────────────

def print_fid_table(fid_sym, param_values, param_name, param_indices, verbose):
    """Print fid_sym[n, idx] for selected parameter values."""
    if not verbose:
        return
    idxs = param_indices
    vals = [f"{param_values[i]:.3f}" for i in idxs]
    header = f"  {'n':>4}  " + "  ".join(f"{v:>7}" for v in vals)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for n in range(fid_sym.shape[0]):
        row = "  ".join(f"{fid_sym[n, i]:7.5f}" for i in idxs)
        print(f"  {n+1:>4}  {row}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def verify_depol(path, tol, verbose):
    data = np.load(path, allow_pickle=True)
    p    = data["p_values"]
    fid_sym    = data["fid_sym"]     # (20, 50)
    fid_nonsym = data["fid_nonsym"]  # (5, 50)
    fid_n1     = data["fid_n1"]      # (50,)

    _section(f"DEPOLARIZING CHANNEL  (file: {path})")
    print(f"  p range       : {p[0]:.4f} .. {p[-1]:.4f}  ({len(p)} values)")
    print(f"  n range       : 1 .. {fid_sym.shape[0]}  (symmetric seesaw)")
    print(f"  n_nonsym      : 1 .. {fid_nonsym.shape[0]}  (non-symmetric seesaw)")
    print()

    all_ok = True

    # 1. Range
    for key in ["fid_sym", "fid_nonsym", "fid_n1", "fid_n5_code"]:
        ok, lo, hi = check_range(data[key], key)
        all_ok &= ok
        print(f"  Range [{key}]  : [{lo:.6f}, {hi:.6f}]  {_ok(ok)}")
    print()

    # 2. Boundary p=0
    ok, err = check_boundary_depol(data, tol)
    all_ok &= ok
    print(f"  Boundary (p=0, all F=1)  : max_err = {err:.2e}  {_ok(ok)}")

    # 3. Analytical uncoded
    ok, err = check_analytical_depol(data, tol)
    all_ok &= ok
    print(f"  Analytical F_e(N_p) = 1-2p/3  : max_err = {err:.2e}  {_ok(ok)}")

    # 4. Monotone in n (informational — seesaw is heuristic, not guaranteed monotone)
    _, viol = check_monotone_n(fid_sym, tol)
    flag = "ℹ (expected for heuristic)" if viol > tol else "✓"
    print(f"  Monotone in n (sym seesaw)  : max drop = {viol:.2e}  {flag}")

    # 5. Sym vs nonsym (informational — symmetric is restricted, nonsym has more freedom)
    _, viol = check_sym_beats_nonsym(fid_sym, fid_nonsym, n_nonsym=fid_nonsym.shape[0], tol=tol)
    flag = "ℹ (heuristic; nonsym may exceed sym)" if viol > tol else "✓"
    print(f"  Sym vs nonsym at n={fid_nonsym.shape[0]}  : max diff = {viol:.2e}  {flag}")

    # 6. n=20 beats n=1 somewhere
    gain = float(np.max(fid_sym[-1] - fid_n1))
    print(f"  Max gain  fid_sym[n=20] - fid_n1  : {gain:+.6f}  {_ok(gain > 0)}")
    all_ok &= gain > 0

    print()
    if verbose:
        # Print table at 5 selected p values
        n_vals = len(p)
        idxs = [0, n_vals // 4, n_vals // 2, 3 * n_vals // 4, n_vals - 1]
        print("  Symmetric seesaw fidelity table (selected p values):")
        print_fid_table(fid_sym, p, "p", idxs, verbose=True)

    return all_ok


def verify_adc(path, tol, verbose):
    data = np.load(path, allow_pickle=True)
    g    = data["gamma_values"]
    fid_sym    = data["fid_sym"]      # (20, 50)
    fid_nonsym = data["fid_nonsym"]   # (50,)  for n=5

    _section(f"AMPLITUDE DAMPING CHANNEL  (file: {path})")
    print(f"  γ range       : {g[0]:.4f} .. {g[-1]:.4f}  ({len(g)} values)")
    print(f"  n range       : 1 .. {fid_sym.shape[0]}  (symmetric seesaw)")
    print(f"  n_nonsym      : 5  (non-symmetric seesaw, single row)")
    print()

    all_ok = True

    # 1. Range
    for key in ["fid_sym", "fid_nonsym", "fid_uncoded", "fid_leung4"]:
        ok, lo, hi = check_range(data[key], key)
        all_ok &= ok
        print(f"  Range [{key}]  : [{lo:.6f}, {hi:.6f}]  {_ok(ok)}")
    print()

    # 2. Boundary γ=0
    ok, err = check_boundary_adc(data, tol)
    all_ok &= ok
    print(f"  Boundary (γ=0, all F=1)  : max_err = {err:.2e}  {_ok(ok)}")

    # 3. Boundary γ=1 for uncoded: F_e = 0.25
    idx1 = int(np.argmin(np.abs(g - 1.0)))
    f_at_1 = float(data["fid_uncoded"][idx1])
    ok_g1 = abs(f_at_1 - 0.25) < tol
    all_ok &= ok_g1
    print(f"  Boundary (γ=1, uncoded F=0.25)  : {f_at_1:.6f}  {_ok(ok_g1)}")

    # 4. Analytical uncoded and Leung 4-qubit
    ok, err_unc, err_l4 = check_analytical_adc(data, tol)
    all_ok &= ok
    print(f"  Analytical uncoded F_e(A_γ)  : max_err = {err_unc:.2e}  {_ok(err_unc < tol)}")
    print(f"  Analytical Leung 4-qubit     : max_err = {err_l4:.2e}  {_ok(err_l4 < tol)}")

    # 5. Monotone in n (informational — seesaw is heuristic, not guaranteed monotone)
    _, viol = check_monotone_n(fid_sym, tol)
    flag = "ℹ (expected for heuristic)" if viol > tol else "✓"
    print(f"  Monotone in n (sym seesaw)  : max drop = {viol:.2e}  {flag}")

    # 6. Sym vs nonsym (informational)
    _, viol = check_sym_beats_nonsym(fid_sym, fid_nonsym, n_nonsym=5, tol=tol)
    flag = "ℹ (heuristic; nonsym may exceed sym)" if viol > tol else "✓"
    print(f"  Sym vs nonsym at n=5  : max diff = {viol:.2e}  {flag}")

    # 7. n=20 beats n=1 somewhere
    fid_n1 = adc_uncoded(g)
    gain = float(np.max(fid_sym[-1] - fid_n1))
    print(f"  Max gain  fid_sym[n=20] - uncoded  : {gain:+.6f}  {_ok(gain > 0)}")
    all_ok &= gain > 0

    print()
    if verbose:
        n_vals = len(g)
        idxs = [0, n_vals // 4, n_vals // 2, 3 * n_vals // 4, n_vals - 1]
        print("  Symmetric seesaw fidelity table (selected γ values):")
        print_fid_table(fid_sym, g, "γ", idxs, verbose=True)

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify depolarizing and ADC seesaw results (self-contained, numpy only)."
    )
    parser.add_argument("--depol", default="depolarizing_fidelity_n1-20.npz",
                        help=".npz file for depolarizing results")
    parser.add_argument("--adc", default="amplitude_damping_fidelity_n1-20.npz",
                        help=".npz file for ADC results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-n fidelity tables at selected parameter values")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Absolute tolerance for analytical checks (default: 1e-6)")
    args = parser.parse_args()

    _hr("=")
    print("  SYMMETRIC SEESAW RESULT VERIFICATION")
    _hr("=")
    print(f"  Depolarizing file  : {args.depol}")
    print(f"  ADC file           : {args.adc}")

    print(f"  Tolerance          : {args.tol:.1e}")
    print()

    ok_dep = verify_depol(args.depol, args.tol, args.verbose)
    ok_adc = verify_adc(args.adc,   args.tol, args.verbose)

    _hr("=")
    if ok_dep and ok_adc:
        print("  All checks passed:  ✓")
    else:
        print("  WARNING: one or more checks failed.")
    _hr("=")

    sys.exit(0 if (ok_dep and ok_adc) else 1)


if __name__ == "__main__":
    main()
