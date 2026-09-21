"""Run the superactivation seesaw.

The superactivation channel is the flagged mixture of the identity and the independent X-Z
Pauli channel (see ``examples.channels.superactivation_channel``), so this is a thin wrapper
around ``run_flagged_pauli`` with the parameters fixed to the superactivation point.

    uv run python -m examples.simulations.run_superactivation --n 17 --seeds 18 42 137 \
        --iterations 4000 --chunk 250 --resume --out results/superactivation_n17.npz

The earlier power-method driver for this channel is kept in ``run_superactivation_seesaw``.
"""
import sys

from examples.simulations.run_flagged_pauli import main as run_flagged_pauli_main

if __name__ == "__main__":
    sys.exit(run_flagged_pauli_main())
