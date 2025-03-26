import Data.data_Types as _Types

import Controllers.FileController as FileController
import Controllers.SolverController as SolverController

if __name__ == "__main__":
    # Parse the level from the input file
    level: _Types.Level = FileController.ParseLevelFromFile(1)

    # Generate the edges of the level
    if level:
        solution: _Types.Matrix = SolverController.Solve(level)
    else:
        solution = None

    # Write the solution to the output file
    if solution:
        level.grid = solution
        FileController.WriteSolutionToFile(level)

        print("Solution found and written to the output file.")

    else:
        print("No solution found.")
