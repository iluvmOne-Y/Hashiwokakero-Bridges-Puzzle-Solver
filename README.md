# Hashiwokakero Puzzle Solver

This project implements a solver for the Hashiwokakero puzzle using a SAT
solver. The program reads a grid from a text file, generates edges, defines
bridge variables, generates CNF clauses, and solves the puzzle. The solution is
then printed in a visual format.

## Project Structure

```
Project2
├── src
│   ├── main.py          # Main logic for solving the Hashiwokakero puzzle
│   ├── solver.py        # Solver implementation
│   └── utils
│       ├── __init__.py  # Package initializer for utility functions
│       └── helpers.py   # Helper functions for the solver
├── data
│   └── 1.txt            # Input data for the Hashiwokakero puzzle
└── README.md            # Documentation for the project
```

## Requirements

-   Python 3.x
-   pysat library

## Setup

1. Clone the repository or download the project files.
2. Install the required libraries using pip:

    ```
    pip install pysat
    ```

## Running the Program

To run the Hashiwokakero puzzle solver, execute the following command in your
terminal:

```
python -O source/main.py
```

Make sure that the input file `data/1.txt` is present and formatted correctly as
a comma-separated grid.

## Input Format

The input file should contain a grid where:

-   `0` indicates no island.
-   Numbers `1-8` indicate islands with the corresponding bridge count.

Example of `data/1.txt`:

```
0, 1, 0, 2
1, 0, 2, 0
0, 2, 0, 1
2, 0, 1, 0
```

## Output

The program will print the solution in a visual format, indicating the bridges
between islands.

## License

This project is licensed under the MIT License.
