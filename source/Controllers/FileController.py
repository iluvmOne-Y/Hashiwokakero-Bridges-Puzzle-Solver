import Data.data_Types as _Types

from os.path import dirname, abspath


def ParseLevelFromFile(levelNumber: int) -> _Types.Level:
    """Parse a matrix from the input file and return a `Level` object.

    ### Parameters:
    - levelNumber: The number of the level to parse

    ### Returns:
    - The level instance with the parsed data
    """
    # Initialize the level
    level: _Types.Level = _Types.Level()
    level.number = levelNumber

    # Get the input file path
    inputFilePath = (
        dirname(dirname(abspath(__file__)))
        + "/Inputs/input-"
        + (str(levelNumber) if levelNumber > 9 else "0" + str(levelNumber))
        + ".txt"
    )

    try:
        # Open input file and read the level
        with open(inputFilePath, "r") as inputFile:
            # Initialize the row and column count
            rowCount: int = 0
            columnCount: int = 0

            # Iterate and get single row each iteration from the file
            for row in inputFile.read().splitlines():
                # Remove the trailing whitespaces on of the row
                row = row.strip()

                currentRow: list[str] = []
                currentColumnIndex: int = -1

                # Check each cell in the row
                for cell in row:
                    # Skip the cell if it is a comma or a space
                    if cell in [",", " "]:
                        continue

                    # Raise an error if the cell is not a number
                    if cell < "0" or cell > "9":
                        raise ValueError("The cell value must be a number.")

                    # Increment the column index and update the current row
                    currentRow.append(cell)
                    currentColumnIndex += 1

                    # Also skip if the current position is an empty space
                    if cell == "0":
                        continue

                    # Add the island to the islands dictionary
                    level.islands.update(
                        {_Types.Position(rowCount, currentColumnIndex): int(cell)}
                    )

                # Append the current row to the grid
                level.grid.append(currentRow)

                # Update the row and column count
                rowCount += 1
                columnCount = currentColumnIndex + 1

            # Update the grid size
            level.gridSize = _Types.Position(rowCount, columnCount)

    except FileNotFoundError:
        print(f"Error: The file {inputFilePath} was not found.")
        return None
    except IOError:
        print(f"Error: An I/O error occurred while opening the file {inputFilePath}.")
        return None

    return level


def WriteSolutionToFile(level: _Types.Level) -> None:
    """Write the solution matrix to the output file."""
    # Get the output file path
    outputFilePath = (
        dirname(dirname(abspath(__file__)))
        + "/Outputs/output-"
        + (str(level.number) if level.number > 9 else "0" + str(level.number))
        + ".txt"
    )

    try:
        # Open the output file
        with open(outputFilePath, "w") as f:
            # Format the grid for output
            formatted_rows = []
            for row in level.grid:
                # Format each character in the row as a quoted string
                quoted_chars = [f'"{char}"' for char in row]
                # Join with commas and spaces
                formatted_row = "[ " + " , ".join(quoted_chars) + " ]"
                formatted_rows.append(formatted_row)
            formatted_grid = "\n".join(formatted_rows)
            
            # Write the formatted solution to the file
            f.write(formatted_grid)

    except FileNotFoundError:
        print(f"Error: The file {outputFilePath} was not found.")
    except IOError:
        print(f"Error: An I/O error occurred while opening the file {outputFilePath}.")


__all__ = ["ParseLevelFromFile", "WriteSolutionToFile"]
