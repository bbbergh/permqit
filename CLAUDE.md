# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`permqit` is the code package for the paper *Bergh, Parentin: Permutation Invariant Optimization Problems in
Quantum Information Theory: A Framework for Channel Fidelity and Beyond* (arXiv:2604.27040). It computes
high-dimensional permutation-invariant optimization problems from quantum information theory (partial traces,
(partial) channel applications, channel fidelity) by working entirely inside the permutation-invariant subspace,
without ever constructing the exponentially large matrices on the full tensor product space `V^n`.

The API is not yet stable — call signatures and package structure may still change.

## Commands

Dependency management and running code goes through [`uv`](https://docs.astral.sh/uv/); there is no separate lint/build step beyond ruff/ty.

```bash
uv sync                              # install dependencies into local .venv
uv sync --extra cuda12               # + GPU acceleration (CUDA 12, requires local CUDA)
uv sync --extra cuda12standalone     # + full CUDA 12 toolchain installed into the venv

uv run pytest                        # run tests (slow/veryslow tests are skipped by default, see below)
uv run pytest tests/test_matrix.py   # run a single test file
uv run pytest tests/test_matrix.py::TestClass::test_name -v   # run a single test

uv run pytest -m slow                # also run tests marked slow (~10s+)
uv run pytest -m "slow or veryslow"  # also run tests marked veryslow (~1min+)

uv run ruff check .                  # lint (line-length 120, configured in pyproject.toml)
uv run ty check                      # type check
```

Notes on tests:
- `tests/conftest.py` sets `NUMBA_DISABLE_JIT=1` to avoid a long startup time for tests, and forces `slow`/`veryslow`-marked tests to be skipped unless explicitly selected with `-m`.
- Requires Python >= 3.12.

## GPU backend

`permqit.utilities.backend` auto-detects CuPy at import time and exposes `xp` (either `numpy` or `cupy`),
`USE_GPU`, `to_cpu`/`asnumpy`, and `to_gpu`/`asxp`/`to_xp`. Set the environment variable `PERMQIT_USE_GPU=false`
*before* importing the package to force CPU-only, or call `backend.reinitialize_backend()` after changing the env
var at runtime (e.g. in a notebook/Colab). GPU acceleration is currently used by `power_method/`.

## Architecture

The package (`src/permqit/`) is organized in four layers, roughly bottom-up:

- **`algebra/`** — generic linear-algebra scaffolding, independent of the representation theory. Key abstractions:
  - `Basis` (in `basis.py`) — abstract base for labelled vector-space bases (`VectorStandardBasis`,
    `MatrixStandardBasis`, `TensorProductBasis`, `MatrixTensorProductBasis`, ...). Every basis knows how to
    enumerate/index its labels and vectors and compute linear combinations.
  - `LinearMap` / `TransitionMatrix` (in `linear_map.py`) — represents a linear map between two `Basis` instances
    as a matrix that converts coefficient vectors from one basis into another; supports multiple sparse/dense/GPU
    `StorageFormat`s and caches computed matrices (`MatrixCache`). Most "expensive object" classes in the codebase
    (isomorphisms, partial trace relations, etc.) subclass `TransitionMatrix` and only need to implement
    `_calculate_transition_matrix()`.
  - `endomorphism_basis.py` / `endomorphism_direct_sum_basis.py` — bases for `End^(S_n)(V^n)`: the orbit basis
    (`EndSnOrbitBasis`), the block-diagonal basis (`EndSnBlockDiagonalBasis`), and per-irrep bases.

- **`representation/`** — the representation-theoretic constructions from Section III of the paper: partitions,
  Young tableaux/SSYT, orbits, and — most importantly — `isomorphism.py`, which implements the block-diagonalizing
  *-isomorphism `EndSnAlgebraIsomorphism`/`BaseEndSnBlockDiagonalization` mapping the permutation-invariant
  subspace `End^(S_n)(V^n)` onto a direct sum of irrep blocks (there are two alternate implementations,
  `isomorphism_gijswijt.py` and `isomorphism_kappa.py`, selectable per use case). `partial_traces.py` builds the
  transition matrices for taking partial traces and applying channels entirely within this invariant subspace,
  avoiding construction of the full `V^n`-sized matrices.

- **`SDP/`** — the symmetric seesaw method for channel fidelity via SDP solvers (Section IV of the paper), built
  on `picos`/`cvxpy`/`qics`. `variables/` defines SDP variable wrappers (Choi matrices, masked variables).

- **`power_method/`** — a power-method implementation for channel fidelity (Section IV of the paper), GPU
  accelerated via the `utilities.backend` module when CuPy/CUDA is available.

- **`utilities/`** — cross-cutting helpers: caching (see below), the numpy/GPU backend, numpy helper functions
  (`numpy_utils.py`), `quantum_info.py` for quantum-info-specific helpers, `sdp_result.py`, `timing.py`
  (`ExpensiveComputation` for logging long-running constructions), and `testing.py` (shared assertion helpers used
  across the test suite, e.g. `assert_is_valid_choi`, `assert_is_valid_choi_via_isomorphism`).

### Caching framework

Many expensive objects (the *-isomorphisms, channel link-product relations, etc.) are memoized via
`utilities.caching.WeakRefMemoize`, a metaclass that caches instances by `(class, init-args)` in a
`weakref.WeakValueDictionary`. As long as an instance is referenced somewhere else, constructing another instance
with the same parameters returns the cached one instead of recomputing it; once nothing else references it, it is
garbage collected rather than held forever. Classes can override `__intercept_new__` to redirect construction
entirely (e.g. `BaseEndSnBlockDiagonalization` returns a `TrivialAlgebraIsomorphism` when `n <= 1`) and
`__process_init_args_for_cache_key__` to customize the cache key derived from constructor args. Construction of
these expensive objects is normally logged to stdout. Separately, `utilities.caching.cache`/`cache_noargs` are
lightweight per-instance memoizing decorators for methods (cache is cleared when the instance is garbage
collected).

### Examples

`examples/` contains runnable code reproducing the paper's numerical results:
- `examples/simulations/` — end-to-end scripts (non-asymptotic superactivation with a pre/post-processed
  Smith-Yard channel pair, channel fidelity of the qubit amplitude damping and depolarizing channels).
- `examples/channels/` and `examples/states/` — reusable channel/state definitions used by the simulations.
