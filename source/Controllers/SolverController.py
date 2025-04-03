import Data.data_Types as _Types

from typing import Callable

from Solvers import PySAT, AStar, Backtracking  #, BruteForce

SolverFunction = Callable[[_Types.Level], _Types.Matrix | None]

# Define the solvers
solvers: dict[str, SolverFunction] = {
    "PySAT": PySAT.Solve,
    "AStar": AStar.Solve,
    "Backtracking": Backtracking.Solve,
    # "BruteForce": BruteForce.Solve,
}


def Solve(level: _Types.Level, solver: str = "Backtracking") -> _Types.Matrix | None:
    """Solve the level.

    ### Parameters:
    - level: The level to solve.
    - solver: The solver to use. Possible values are
        + "PySAT"
        + "AStar"
        + "Backtracking"
        + "BruteForce".

    ### Returns:
    - The solution matrix or `None` if no solution is found.
    """
    if solver not in solvers:
        raise ValueError(f"Solver '{solver}' is not available.")

    return solvers[solver](level)


__all__ = ["Solve"]
