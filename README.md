

This project implements a solver for the Hashiwokakero (Bridges) puzzle, a logic puzzle where the goal is to connect islands (numbered circles) with bridges following specific rules.

## About the Puzzle

In Hashiwokakero:
- Islands are represented by circles with numbers
- Each number indicates how many bridges must connect to that island
- Bridges can only run horizontally or vertically
- Bridges cannot cross other bridges
- All islands must be connected in a single network

## Project Structure

```
Hashiwokakero/
├── script/             # Scripts for running the application
├── source/             # Main source code
│   ├── Controllers/    # Controllers for file and solver operations
│   ├── Data/           # Data types and structures
│   ├── GUI/            # PyQt5-based graphical user interface
│   ├── Inputs/         # Input puzzles
│   ├── Outputs/        # Solved puzzle outputs
│   ├── Solvers/        # Different solving algorithms
│   └── Utilities/      # Helper functions
├── main.py             # Command-line entry point
├── gui_main.py         # GUI entry point
└── benchmark.py        # Performance benchmarking tool
```

## Features

- **Multiple Solving Algorithms**:
  - **PySAT**: Uses SAT solver with CNF formulas
  - **A\***: Implements A* search with custom heuristics
  - **Backtracking**: Standard backtracking approach
  - **BruteForce**: Simple exhaustive search

- **Graphical User Interface**:
  - Load puzzles from files
  - Choose solving algorithm
  - Visualize the solution process
  - Save solutions

## Input Map Sizes

The project includes several predefined puzzle inputs with varying sizes and difficulty levels:

| Input File | Grid Size | Difficulty |
|------------|-----------|------------|
| input-01   | 7×7       | Easy       |
| input-02   | 7×7       | Hard       |
| input-03   | 9×9       | Easy       |
| input-04   | 9×9       | Hard       |
| input-05   | 11×11     | Easy       |
| input-06   | 11×11     | Hard       |
| input-07   | 13×13     | -          |
| input-08   | 15×15     | -          |
| input-09   | 18×18     | -          |
| input-10   | 20×20     | -          |

Larger grid sizes and higher difficulty levels generally require more computational resources to solve and serve as excellent benchmarks for comparing different solving algorithms.

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install python-sat PyQt5 tabulate
   ```

## Usage

### Graphical Interface

```
python source/gui_main.py
```

The GUI allows you to:
1. Load a puzzle using the "Load Puzzle" button
2. Select a solving algorithm from the dropdown
3. Click "Solve" to find a solution
4. View the solution visually
5. Save the solution to a file

### Benchmark Tool

```
python source/benchmark.py
```

Runs all solvers on all available levels and produces a performance comparison table.

## Solution Representation

In the solution:
- `-` represents a single horizontal bridge
- `=` represents a double horizontal bridge
- `|` represents a single vertical bridge
- `$` represents a double vertical bridge
- Numbers represent the islands

## License

This project is available for educational and academic purposes.

## Requirements

- Python 3.x
- python-sat
- PyQt5
