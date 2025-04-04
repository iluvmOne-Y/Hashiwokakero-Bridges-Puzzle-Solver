import Data.data_Types as _Types

from typing import Callable

from Solvers import PySAT, AStar , BruteForce ,  Backtracking

import time 
SolverFunction = Callable[[_Types.Level], _Types.Matrix | None]

# Define the solvers
solvers: dict[str, SolverFunction] = {
    "PySAT": PySAT.Solve,
    "AStar": AStar.Solve,
     "Backtracking": Backtracking.Solve,
    "BruteForce": BruteForce.Solve,
}


def Solve(level: _Types.Level, solver: str = "") -> _Types.Matrix | None:
    """Solve the level."""
    if solver not in solvers:
        raise ValueError(f"Solver '{solver}' is not available.")
    
    start_time = time.time()

    solution = solvers[solver](level)
    
    solving_time = time.time() - start_time

    level.solving_time = solving_time
    print(f"Solving time: {solving_time:.4f} seconds")
    # If we have a solution, ensure it's in the correct format for saving
    if solution:
        # Create a proper Matrix instance from the solution
        formatted_grid = _Types.Matrix()
        for row in solution:
            # Convert each row to a list of single characters
            formatted_row = []
            for char in row:
                formatted_row.append(char)
            formatted_grid.append(formatted_row)
        return formatted_grid
    
    return None

__all__ = ["Solve"]
