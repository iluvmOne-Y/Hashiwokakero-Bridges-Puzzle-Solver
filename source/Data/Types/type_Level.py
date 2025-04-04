from __future__ import annotations 

from copy import deepcopy


class Position:
    """The class that represents a position in the 2D space.

    ### Attributes
    - x: The x coordinate of the position.
    - y: The y coordinate of the position.
    """

    def __init__(
        self,
        value: Position | tuple[int, int] | int = None,
        x: int = None,
    ) -> None:
        """Initialize the position.

        ### Parameters
        - value: The value of the position. Could be a Position object, a tuple of two integers. Or an integer if x is provided.
        - x: The x coordinate of the position.
        """
        if x is not None:
            # Check if the x is an integer
            assert isinstance(value, int) and isinstance(
                x, int
            ), "The x and y coordinates must be integers."

            self.y = value
            self.x = x

        elif value is not None:
            if isinstance(value, Position):
                self.y = value.y
                self.x = value.x

            elif isinstance(value, tuple) and len(value) == 2:
                # Check if the tuple is a tuple of two integers
                assert isinstance(value[0], int) and isinstance(
                    value[1], int
                ), "The tuple must contain two integers."

                self.y = value[0]
                self.x = value[1]

            else:
                raise TypeError(
                    "The value must be an integer, a tuple of two integers, or a Position object."
                )

        else:
            # Set the default values for the position
            self.y = 0
            self.x = 0

    def __getitem__(self, index: int) -> int:
        """Return the x or y coordinate of the position.

        ### Parameters
        - index: The index of the coordinate.
        """
        if index == 0:
            return self.y

        elif index == 1:
            return self.x

        else:
            raise IndexError("The index must be 0 or 1.")

    def __eq__(self, other: Position | tuple[int, int]) -> bool:
        """Return the equality of the two positions.

        ### Parameters
        - other: The other position to be compared.
        """
        if isinstance(other, tuple) and len(other) == 2:
            assert isinstance(other[0], int) and isinstance(
                other[1], int
            ), "The tuple must contain two integers."

            return self.x == other[1] and self.y == other[0]

        elif isinstance(other, Position):
            return self.x == other.x and self.y == other.y

        else:
            # Return False if the other object is not a supported type
            return False

    def __str__(self) -> str:
        """Return the string representation of the position."""
        return f"({self.x}, {self.y})"

    def __tuple__(self) -> tuple[int, int]:
        """Return the tuple representation of the position."""
        return tuple[self.y, self.x]

    def __hash__(self) -> int:
        """Return the hash value of the position."""
        return hash((self.y, self.x))


class Matrix(list[list[str]]):
    """The class that represents a 2D matrix.

    ### Attributes
    - size: The size of the matrix.
    """

    def __init__(
        self, size: Position | tuple[int, int] | int = None, columns: int = None
    ) -> None:
        """Initialize the matrix.

        ### Parameters
        - size: The size of the matrix. Could be the number of rows if number of columns is provided.
        - columns: The number of columns in the matrix.
        """
        # Initialize the matrix with an empty size if not provided
        if columns is not None:
            assert isinstance(size, int) and isinstance(
                columns, int
            ), "The row and column numbers must be integers."

            size = Position(size, columns)
            super().__init__()

        # Initialize the matrix with zeros
        elif size is not None:
            if isinstance(size, tuple) and len(size) == 2:
                # Check if the tuple is a tuple of two integers
                assert isinstance(size[0], int) and isinstance(
                    size[1], int
                ), "The tuple must contain two integers."

                super().__init__(["0"] * size[1] for _ in range(size[0]))
                self.size = Position(size[0], size[1])

            elif isinstance(size, Position):
                super().__init__(["0"] * size.x for _ in range(size.y))
                self.size = size

            else:
                raise TypeError(
                    "The size must be a Position or a tuple of two integers."
                )

        else:
            # Initialize the matrix with an empty size
            super().__init__()
            self.size = Position()

    def __getitem__(self, key: Position | tuple[int, int] | int) -> str | list[str]:
        """Return the value of the matrix at the given position.

        ### Parameters
        - key: The position of the value to be returned.
        """
        if isinstance(key, Position):
            # Check if the key is in the range of the matrix
            assert (
                key.x < self.size.x and key.y < self.size.y
            ), "Attempting to index out of range!"

            return super().__getitem__(key.y)[key.x]

        elif isinstance(key, tuple) and len(key) == 2:
            # Check if the tuple is a tuple of two integers
            assert isinstance(key[0], int) and isinstance(
                key[1], int
            ), "The tuple must contain two integers."

            # Check if the key is in the range of the matrix
            assert (
                key[1] < self.size.x and key[0] < self.size.y
            ), "Attempting to index out of range!"

            return super().__getitem__(key[0])[key[1]]

        elif isinstance(key, int):
            # Check if the key is in the range of the matrix
            assert key < len(
                self
            ), f"Attempting to index out of range! Key: {key}, Length: {len(self)}"

            return super().__getitem__(key)

        else:
            raise TypeError("Attempting to index Matrix object with invalid key!")

    def __setitem__(self, key: Position | tuple[int, int] | int, value: str):
        """Set the value of the matrix at the given position.

        ### Parameters
        - key: The position of the value to be set.
        - value: The value to be set.
        """
        if isinstance(key, Position):
            # Check if the key is in the range of the matrix
            assert (
                key.x < self.size.x and key.y < self.size.y
            ), "Attempting to index out of range!"

            # Check if the value is a single character
            assert len(value) == 1, "The value must be a single character."

            row = list(super().__getitem__(key.y))
            row[key.x] = value

        elif isinstance(key, tuple) and len(key) == 2:
            # Check if the tuple is a tuple of two integers
            assert isinstance(key[0], int) and isinstance(
                key[1], int
            ), "The tuple must contain two integers."

            # Check if the key is in the range of the matrix
            assert (
                key[1] < self.size.x and key[0] < self.size.y
            ), "Attempting to index out of range!"

            # Check if the value is a single character
            assert len(value) == 1, "The value must be a single character."

            row = list(super().__getitem__(key[0]))
            row[key[1]] = value

        elif isinstance(key, int):
            # Check if the key is in the range of the matrix
            assert key < len(self), "Attempting to index out of range!"

            if isinstance(value, list):
                # Check if the value is a list of strings
                assert all(
                    isinstance(x, str) and len(x) == 1 for x in value
                ), "The list must contain only strings of length 1."

                # Check if the value is as long as matrix's row
                if len(value) < self.size.x:
                    value += "0" * (self.size.x - len(value))
                elif len(value) > self.size.x:
                    value = value[: self.size.x]

                # Update the row with the new value
                super().__setitem__(key, value)

            else:
                raise TypeError("The value must be a list containing string.")

        else:
            raise TypeError("Attempting to index Matrix object with invalid key!")

    def append(self, row: list[str]):
        """Append a row to the matrix.

        ### Parameters
        - row: The row to be appended.
        """
        # Fill the matrix's rows with zeros if its size is shorter than new row
        if len(row) > self.size.x:
            for i in range(len(self)):
                self[i] += "0" * (len(row) - self.size.x)

            self.size.x = len(row)

        # Fill the row with zeros if it is shorter than the matrix
        elif len(row) < self.size.x:
            row += "0" * (self.size.x - len(row))

        # Check if the row is a list of strings of length 1
        assert isinstance(row, list), "The row must be a list of strings."
        assert all(
            isinstance(x, str) and len(x) == 1 for x in row
        ), "The list must contain only strings of length 1."

        # Append the row to the matrix
        super().append(row)
        # Update the size of the matrix
        self.size.y += 1

    def resize(
        self, size: Position | tuple[int, int] | int = None, columns: int = None
    ):
        """Resize the matrix to the new size.

        ### Parameters
        - size: The new size of the matrix. Could be the number of rows if number of columns is provided.
        - columns: The number of columns in the matrix.
        """
        # Initialize the new size
        oldSize = self.size
        newSize: Position = None

        if columns is not None:
            assert isinstance(size, int) and isinstance(
                columns, int
            ), "The row and column numbers must be integers."

            newSize = Position(size, columns)

        elif size is not None:
            # Get the new size from the provided value
            if isinstance(size, tuple) and len(size) == 2:
                # Check if the tuple is a tuple of two integers
                assert isinstance(size[0], int) and isinstance(
                    size[1], int
                ), "The tuple must contain two integers."

                newSize = Position(size[0], size[1])

            elif isinstance(size, Position):
                newSize = size

            else:
                raise TypeError(
                    "The size must be a Position or a tuple of two integers."
                )

        else:
            raise ValueError("The new size must be provided.")

        # Update the size of the matrix
        self.size = Position(
            newSize.y - oldSize.y if newSize.y > oldSize.y else newSize.y,
            newSize.x,
        )

        # Resize the matrix to the new size

        # Remove the rows if the new size is smaller
        if newSize.y < oldSize.y:
            del self[newSize.y : oldSize.y]

        # Add new rows if the new size is larger
        elif newSize.y > oldSize.y:
            for i in range(newSize.y - oldSize.y):
                self.append(["0"] * newSize.x)

        # Resize the rows to the new size
        if newSize.x < oldSize.x:
            for i in range(newSize.y):
                self[i] = self[i][: newSize.x]

        # Fill the rows with zeros if the new size is larger
        elif newSize.x > oldSize.x:
            for i in range(self.size.y):
                self[i] += "0" * (newSize.x - oldSize.x)

    def __copy__(self) -> Matrix:
        """Return a deep copy of the matrix."""
        return deepcopy(self)

    def __str__(self) -> str:
        """Return the string representation of the matrix."""
        formattedRows: list[str] = []

        for row in self:
            formattedRow = "[ " + " , ".join(f'"{cell}"' for cell in row) + " ]"
            formattedRows.append(formattedRow)

        return "\n".join(formattedRows)

    def __list__(self) -> list[str]:
        """Return the list representation of the matrix."""
        return self


class Edge:
    """The class that represents an edge of Hashiwokakero puzzle.

    ### Attributes
    - id: The unique identifier of the edge.
    - startingPosition: The position of the starting cell.
    - endingPosition: The position of the ending cell.
    """

    def __init__(
        self,
        id: int = None,
        position1: Position | int = None,
        position2: Position | int = None,
        y: int = None,
        x: int = None,
    ):
        """Initialize the edge.

        ### Parameters
        - id: The unique identifier of the variable.
        - position1: The position of the starting cell. Could be the y coordinate if ending cell's coordinates are provided.
        - position2: The position of the ending cell. Could be the x coordinate if starting cell's coordinates are provided.
        - y1: The y coordinate of the ending cell.
        - x1: The x coordinate of the ending cell.
        """
        if x is not None:
            # Check if the x and y coordinates are integers
            assert isinstance(position1, int) and isinstance(
                position2, int
            ), "The x and y coordinates of the starting cell must be integers."

            assert isinstance(x, int) and isinstance(
                y, int
            ), "The x and y coordinates of the ending cell must be integers."

            self.startingPosition = Position(position2, position1)
            self.endingPosition = Position(y, x)

            assert isinstance(id, int), "The id must be an integer."
            self.id = id

        elif position2 is not None:
            assert isinstance(position1, Position) and isinstance(
                position2, Position
            ), "The positions must be Position objects."

            self.startingPosition = position1
            self.endingPosition = position2

            assert isinstance(id, int), "The id must be an integer."
            self.id = id

        else:
            raise ValueError("The positions of the edge must be provided.")

    def __getitem__(self, index: int) -> int:
        """Return the value of the edge at the given index.

        ### Parameters
        - index: The index of the coordinate.
        """
        assert (
            index >= 0 and index < 4
        ), "The index must be a non-negative integer and less than 4."

        if index < 2:
            return self.startingPosition[index]

        else:
            return self.endingPosition[index - 2]

    def __str__(self) -> str:
        """Return the string representation of the edge."""
        return f"({self.startingPosition.x}, {self.startingPosition.y}, {self.endingPosition.x}, {self.endingPosition.y})"

    def __iter__(self):
        """Return an iterator over the edge."""
        return iter(
            (
                self.startingPosition.y,
                self.startingPosition.x,
                self.endingPosition.y,
                self.endingPosition.x,
            )
        )


class Level:
    """The class that represents the level of the game.

    ### Attributes
    - number: The order number of the level.
    - grid: The grid of the level.
    - gridSize: The size of the grid.
    - islands: The dictionary of islands in the grid.
    """

    def __init__(self):
        """Initialize the level."""
        # Set the level order number
        self.number: int = 0

        # Initialize the attributes
        self.grid: Matrix = Matrix()
        self.gridSize: Position = Position()

        self.islands: dict[Position, int] = {}


__all__ = ["Position", "Matrix", "Edge", "Level"]
