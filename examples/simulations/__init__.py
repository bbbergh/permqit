"""
Simulation entry points for seesaw optimization and asymptotic bounds.

Main scripts (run with ``python -m permqit.simulations.<name>``):

- ``run_depolarizing_sym``      — depolarizing channel, symmetric seesaw
- ``run_amplitude_damping``     — amplitude damping channel
- ``run_seesaw_cq_power``       — CQ seesaw (GPU power method)
- ``run_superactivation_operators`` — save optimal operators for n=17
- ``generate_aux_operators``    — generate auxiliary npz from minimal npz
- ``run_second_order``          — second-order asymptotic bounds (no seesaw)

Development / diagnostic scripts live in ``simulations/diagnostics/``.
"""

# Import Colab-specific function if available
try:
    from .run_seesaw_cq_power_colab import run_cq_seesaw_gpu
except ImportError:
    pass
