from typing import Union, Tuple
import time
import cvxpy as cp
import picos as pc
from picos.modeling.problem import SolutionFailure


def solve_sdp(
    prob: Union[cp.Problem, pc.Problem],
    solver: str = 'SCS',
    sdp_accuracy: float = 1e-6,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Solve an SDP problem with CVXPY or PICOS using a unified interface.

    Args:
        prob: cvxpy.Problem or picos.Problem
        solver: Solver name (e.g. 'SCS', 'MOSEK', 'CVXOPT')
        sdp_accuracy: Desired accuracy tolerance
        verbose: Whether to show solver output

    Returns:
        (optimal_value, solve_time)
    """
    start_time = time.time()

    # ------------------
    # CVXPY
    # ------------------
    if isinstance(prob, cp.Problem):
        solver = solver.upper()

        kwargs = {"verbose": verbose}

        if solver == "SCS":
            kwargs["eps"] = sdp_accuracy

        elif solver == "ECOS":
            kwargs["abstol"] = sdp_accuracy
            kwargs["reltol"] = sdp_accuracy
            kwargs["feastol"] = sdp_accuracy

        elif solver == "CVXOPT":
            kwargs["abstol"] = sdp_accuracy
            kwargs["reltol"] = sdp_accuracy
            kwargs["feastol"] = sdp_accuracy

        # MOSEK: CVXPY handles tolerances internally
        prob.solve(solver=solver, **kwargs)

        value = prob.value

    # ------------------
    # PICOS
    # ------------------
    elif isinstance(prob, pc.Problem):
        solver = solver.lower()

        # Pre-flight check: fall back if requested solver is not available
        available = [s.lower() for s in pc.available_solvers()]
        if solver not in available:
            fallbacks = [s for s in ("mosek", "qics", "cvxopt", "ecos") if s in available]
            if fallbacks:
                solver = fallbacks[0]
            else:
                raise RuntimeError(f"No supported solver available. Found: {sorted(available)}")

        def _solve_with(solver_name: str):
            kwargs = {
                "verbosity": 1 if verbose else 0,
            }

            if solver_name == "scs":
                kwargs["eps"] = sdp_accuracy

            elif solver_name == "cvxopt":
                kwargs["abstol"] = sdp_accuracy
                kwargs["reltol"] = sdp_accuracy
                kwargs["feastol"] = sdp_accuracy

            try:
                prob.solve(solver=solver_name, **kwargs)
            except SolutionFailure:
                # Retry allowing incomplete primals.
                try:
                    prob.solve(solver=solver_name, primals=None, **kwargs)
                except LookupError:
                    prob.solve(solver=solver_name, primals=None, verbosity=kwargs["verbosity"])
            except LookupError:
                # Fall back to minimal options if solver rejects kwargs.
                prob.solve(solver=solver_name, verbosity=kwargs["verbosity"])
            return prob.value

        try:
            value = _solve_with(solver)
        except Exception as exc:
            original_exc = exc
            available = [s.lower() for s in pc.available_solvers()]
            fallbacks = [s for s in ("mosek", "qics", "cvxopt", "ecos") if s in available]
            value = None
            for fallback in fallbacks:
                if fallback == solver:
                    continue
                try:
                    value = _solve_with(fallback)
                    break
                except Exception:
                    value = None
            if value is None:
                raise original_exc

    else:
        raise TypeError(
            f"Unsupported problem type: {type(prob)}. "
            "Expected cvxpy.Problem or picos.Problem."
        )

    solve_time = time.time() - start_time
    return value, solve_time