"""
Context/dataclass definitions consolidating parameters and precomputed representation-theoretic data.
Extracted from sections: Constants, Parameters of the simulation, Representation Theoretic Parameters,
Isomorphism Parameters. Functions that previously relied on global variables now accept these
context objects explicitly, improving purity and testability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

@dataclass
class SDPResult:
    value: float
    time: Optional[float] = None
    optimizers: Optional[Union[Any, Sequence[Any]]] = None

    # Accessors
    def get_value(self) -> float:
        """Return the value of the measurement."""
        return self.value

    def get_time(self) -> Optional[float]:
        """Return the runtime or time associated with this measurement, if available."""
        return self.time

    def get_optimizers(self) -> Optional[Sequence[Any]]:
        """Return the optimizers as a sequence. Wrap single optimizer in a list if needed."""
        if self.optimizers is None:
            return None
        if isinstance(self.optimizers, Sequence) and not isinstance(self.optimizers, (str, bytes)):
            return self.optimizers
        return [self.optimizers]

    def get_first_optimizer(self) -> Optional[Any]:
        """Return the first optimizer if multiple exist, else return single optimizer."""
        optimizers = self.get_optimizers()
        if optimizers:
            return optimizers[0]
        return None

    def assert_get_first_optimizer(self) -> Any:
        opt = self.get_first_optimizer()
        if opt is None:
            raise ValueError("No optimizer present")
        return opt

    def has_optimizers(self) -> bool:
        """Return True if there is at least one optimizer stored."""
        return self.optimizers is not None

    # String representation for convenience
    def __repr__(self) -> str:
        opt_count = len(self.get_optimizers()) if self.optimizers else 0  # ty:ignore[invalid-argument-type]
        return f"<SDPResult value={self.value}, time={self.time}, optimizers={opt_count}>"
    
    
