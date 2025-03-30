import Data.data_Types as _Types


def GenerateEdges(level: _Types.Level) -> list[_Types.Edge]:
    """Generate the edges of the level.

    ### Parameters:
    - level: The level to generate the edges for.

    ### Returns:
    - The edges of the level.
    """

    # Initialize the edge id
    edgeId = 1
    edges: list[_Types.Edge] = []

    # Get the level grid and islands
    grid: _Types.Matrix = level.grid
    gridRows, gridCols = level.gridSize
    islands: dict[_Types.Position, int] = level.islands

    # Iterate through the islands to generate the edges
    for island, _ in islands.items():
        # Get the row and column of the island
        row, col = island.y, island.x

        # Check for horizontal edges
        for currentCol in range(col + 1, gridCols):
            # Continue if current cell is not an island
            if grid[row][currentCol] == "0":
                continue

            # Add the edge to the edges list and increment the edge id
            edges.append(
                _Types.Edge(
                    edgeId, _Types.Position(row, col), _Types.Position(row, currentCol)
                )
            )
            edgeId += 1

            break

        # Check for vertical edges
        for currentRow in range(row + 1, gridRows):
            # Continue if current cell is not an island
            if grid[currentRow][col] == "0":
                continue

            # Add the edge to the edges list and increment the edge id
            edges.append(
                _Types.Edge(
                    edgeId, _Types.Position(row, col), _Types.Position(currentRow, col)
                )
            )
            edgeId += 1

            # Move on to the next row
            break

    return edges


__all__ = ["GenerateEdges"]
