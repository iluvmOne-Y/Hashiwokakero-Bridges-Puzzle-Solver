import Data.data_Types as _Types

from Utilities import DefineBridgeVariables, GenerateEdges, GenerateCNF


def Solve(level: _Types.Level) -> _Types.Matrix | None:
    """Solver for the level using PySAT library.

    ### Parameters:
    - level: The level to solve.

    ### Returns:
    - The solution matrix or `None` if no solution is found.
    """
    # Generate the edges of the level and the CNF formula
    edges = GenerateEdges(level)
    cnf = GenerateCNF(level, edges)

    # Solve the CNF formula
    solver: _Types.Solver = _Types.Solver("glucose3", cnf)

    if solver.solve():
        model = solver.getModel()
        solutionMatrix: _Types.Matrix = _Types.Matrix.__copy__(level.grid)

        # Get the single and double bridge variables
        singleBridgeVars, doubleBridgeVars, _ = DefineBridgeVariables(edges)

        for edge in edges:
            singleBridgeVar = singleBridgeVars[edge]
            doubleBridgeVar = doubleBridgeVars[edge]

            # Check if the edge is a single bridge
            if singleBridgeVar in model:
                startingPosition, endingPosition = (
                    edge.startingPosition,
                    edge.endingPosition,
                )

                # Add the single bridge to the solution matrix horizontally or vertically
                if startingPosition.y == endingPosition.y:
                    for col in range(startingPosition.x + 1, endingPosition.x):
                        solutionMatrix[startingPosition.y][col] = "-"

                else:
                    for row in range(startingPosition.y + 1, endingPosition.y):
                        solutionMatrix[row][startingPosition.x] = "|"

            # Check if the edge is a double bridge
            elif doubleBridgeVar in model:
                startingPosition, endingPosition = (
                    edge.startingPosition,
                    edge.endingPosition,
                )

                # Add the double bridge to the solution matrix horizontally or vertically
                if startingPosition.y == endingPosition.y:
                    for col in range(startingPosition.x + 1, endingPosition.x):
                        solutionMatrix[startingPosition.y][col] = "="

                else:
                    for row in range(startingPosition.y + 1, endingPosition.y):
                        solutionMatrix[row][startingPosition.x] = "$"

        return solutionMatrix

    # No solution found
    else:
        return None


__all__ = ["Solve"]
