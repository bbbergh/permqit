"""
Backend selection for power method (NumPy vs CuPy).
If you want to disable the GPU even if you have one, set os.environ.set("PERMQIT_USE_GPU", "false") before importing the package.
"""

import os
import typing
import numpy as np

# in code switch (sometimes easier than using environment variables)
DISABLE_GPU = False

USE_GPU = False
xp = np  # type: ignore
cupyx = None
cp = None

# These are just for type annotations
if typing.TYPE_CHECKING:
    try:
        import cupy as cp
        cupyx = cp.cupyx
    except ImportError:
        cp = np
    type ndarray = np.ndarray | cp.ndarray
else:
    type ndarray = np.ndarray

cp_or_np_ndarray = ndarray
"""Just to make type annotations more explicit"""

def _add_cuda_12_bins_to_path():
    """cuda 12 installed on Windows via ``pip install cuda-toolkit`` will install its binaries into a splayed layout.
    This first of all means that a warning is printed that no CUDA_PATH was found (since there is no unique CUDA_PATH),
    and secondly the DLLs will not be found.

    This function adds all the dlls to the PATH
    """
    import importlib.metadata as meta
    import sys
    if sys.platform != 'win32':
        return
    if not meta.version('cuda-toolkit').startswith('12'):
        return

    import site, pathlib
    base_paths = [s for s in site.getsitepackages() if 'lib' in s.lower()]
    if len(base_paths) > 1:
        print(f"Found more than one site-packages path, this is not supported yet: {base_paths}")
        return
    base_path = pathlib.Path(base_paths[0])
    print("CUDA 12 install via pip detected. This will print a warning about CUDA_PATH which you can ignore. Manually added the dll directories.")
    paths = set(str(base_path / f.parent) for s in meta.packages_distributions()['nvidia'] for f in meta.files(s) if '/bin/' in str(f))
    import os
    for path in paths:
        os.add_dll_directory(path)


def reinitialize_backend() -> None:
    """Re-read PERMQIT_USE_GPU and set xp/USE_GPU. Call after setting env (e.g. in Colab)."""
    global USE_GPU, xp, cupyx, cp
    env_use_gpu = os.environ.get("PERMQIT_USE_GPU", "").lower() in ("true", "1", "yes")
    env_disable_gpu = os.environ.get("PERMQIT_USE_GPU", "").lower() in ("false", "0", "no")
    USE_GPU = False
    xp = np
    if env_disable_gpu or DISABLE_GPU:
        return
    try:
        import cupy as _cp
        cp = _cp
        _test = cp.array([1.0])
        _ = _test.device
        xp = cp
        USE_GPU = True

        _add_cuda_12_bins_to_path()
        # cupyx may live as ``cupy.cupyx`` (older builds) or as a standalone
        # ``cupyx`` top-level package (newer builds).
        try:
            import cupyx as _cupyx  # type: ignore[import-untyped]
            cupyx = _cupyx
        except ImportError:
            cupyx = getattr(cp, "cupyx", None)  # type: ignore[union-attr]
            assert cupyx, "Could not import cupyx, please check your cuda/cupy installation"

        if env_use_gpu:
            print("✓ Using CuPy (GPU) backend (PERMQIT_USE_GPU=true)")
        else:
            print("✓ Using CuPy (GPU) backend (auto-detected)")
    except Exception as e:
        USE_GPU = False
        xp = np
        if env_use_gpu:
            print(f"⚠ PERMQIT_USE_GPU=true but GPU not available: {e}")
            print("⚠ Falling back to NumPy (CPU) backend")
        else:
            print("⚠ Using NumPy (CPU) backend - GPU not available")


def to_cpu(arr: cp_or_np_ndarray) -> ndarray:
    """Move array to CPU (numpy array)."""
    if USE_GPU:
        return cp.asnumpy(arr)
    return np.asarray(arr)

asnumpy = to_cpu

def to_gpu(arr: cp_or_np_ndarray) -> cp_or_np_ndarray:
    """Move array to GPU if using cupy, otherwise return as-is."""
    if not USE_GPU:
        return arr
    return xp.asarray(arr)

to_xp = to_gpu
asxp = to_xp

# Initialize on module import
reinitialize_backend()
