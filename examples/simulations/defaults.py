"""Shared default parameters for symmetric seesaw simulation scripts.

These constants govern the number of random seeds, seesaw iterations, and
power method convergence criteria used by run_depolarizing_sym.py,
run_amplitude_damping.py, and similar production simulation scripts.
"""
from permqit.power_method.seesaw import (
    DEFAULT_SEESAW_ITERATIONS as SYM_SEESAW_ITERATIONS,
    DEFAULT_SEESAW_ACCURACY   as SYM_SEESAW_ACCURACY,
) # noqa

SYM_POWER_MAX_ITERATIONS = 5000
SYM_POWER_TOLERANCE      = 1e-9

# 15 independent random seeds for symmetric seesaw sweeps.
SYM_SEEDS = [18, 42, 137, 256, 777, 213, 23, 435, 564, 575, 34, 657, 890, 901, 1234]

# Non-symmetric seesaw: random restarts per seed call.
NONSYM_REPS_PER_SEED = 3
