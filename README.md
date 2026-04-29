# A Toolbox for Permutation Invariant Optimization Problems in Quantum Information Theory
This is the code repository for the paper: Bergh, Parentin: Permutation Invariant Optimization Problems in Quantum Information Theory: A Framework for Channel Fidelity and Beyond. arXiv:&lt;will be added once published>

It provides a python package with tools for efficiently computing values of high-dimensional permutation-invariant optimization problems as they appear in quantum information theory. In particular, it allows phrasing many
typical operations used in quantum information theory (partial traces, (partial) channel applications) wholly within the permutation-invariant subspace and without ever constructing exponentially large matrices.

**Note that this package is still in its early stages. While the code is fully functional we currently don't guarantee any form of API stability, in particular we might still refactor the structure of the package and change the call signature of certain functions.**
Once the API has matured we will release a published version on pypi. 

## Installation
This package requires Python version >= 3.12. Clone the repository and make sure you have [`uv`](https://docs.astral.sh/uv/) installed. Then run `uv sync`, this will install all dependencies into a local venv. If you would like GPU acceleration you need a CUDA compatible GPU. If you have CUDA installed locally, you can install with the necessary `cupy` dependencies by running `uv sync --extra cuda12` or `uv sync --extra cuda13` depending on your CUDA version. If you don't already have CUDA installed you can run `uv sync --extra cuda12standalone` or `uv sync --extra cuda13standalone` to install the full CUDA toolchain into the python venv.
Unfortunately, on MacOS Intel (x86-64) Python 3.14 is not supported due to some dependencies not being available.

## Repository Overview
TBA

## Examples
TBA

## Caching Framework
Many expensive sections of the code (in particular creating the *-isomorphisms and channel link product relations) use a caching system based on python [weak references](https://docs.python.org/3/library/weakref.html). That means that if any of these classes have been created and are still referenced somewhere, any subsequent creation with the same parameters will reuse the previous instance and not create a new one. This is implemented in ``utilities.caching.WeakRefMemoize``. The functions of this package will hold such references as long as such objects are used within the function (and thus avoid recomputation within the function), but do not create eternal objects by themselves to not clog up the available memory. If you would like any of these objects to be cached for a longer period of time, just create and store them yourself. Creation of such expensive objects is usually logged to stdout.
