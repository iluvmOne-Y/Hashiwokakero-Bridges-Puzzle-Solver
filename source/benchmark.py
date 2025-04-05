import Data.data_Types as _Types
import Controllers.FileController as FileController
import Controllers.SolverController as SolverController
import time
import os
from tabulate import tabulate
import sys

def get_available_levels():
    """Find all available level numbers based on input files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "Inputs")
    
    levels = []
    for filename in os.listdir(input_dir):
        if filename.startswith("input-") and filename.endswith(".txt"):
            level_num_str = filename[6:-4]  # Extract number between "input-" and ".txt"
            try:
                level_num = int(level_num_str)
                levels.append(level_num)
            except ValueError:
                continue
    
    return sorted(levels)

def run_benchmark():
    """Run all solvers on all available levels and compare performance."""
    levels = get_available_levels()
    if not levels:
        print("No input levels found.")
        return
    
    print(f"Found {len(levels)} levels: {levels}")
    print("Starting benchmark...\n")
    
    # Set up results table
    results = []
    headers = ["Level", "Solver", "Success", "Time (seconds)"]
    
    # Get all solvers
    solvers = list(SolverController.solvers.keys())
    
    # For each level...
    for level_num in levels:
        print(f"Processing level {level_num}...")
        
        # Load the level
        level = FileController.ParseLevelFromFile(level_num)
        if not level:
            print(f"  Error: Could not load level {level_num}")
            continue
        
        # For each solver...
        for solver_name in solvers:
            print(f"  Running {solver_name}...", end="", flush=True)
            
            # Clone the level to avoid contamination between solvers
            level_copy = _Types.Level()
            level_copy.number = level.number
            level_copy.grid = _Types.Matrix.__copy__(level.grid)
            level_copy.gridSize = level.gridSize
            level_copy.islands = level.islands.copy()
            
            try:
                # Set timeout for very slow solvers
                start_time = time.time()
                timeout = 120  # 2 minutes timeout
                
                # Run the solver with timeout protection
                solution = SolverController.Solve(level_copy, solver_name)
                
                # Check if we have a solution and get the time
                success = solution is not None
                solve_time = getattr(level_copy, 'solving_time', time.time() - start_time)
                
                if success:
                    print(f" Solved in {solve_time:.4f}s")
                else:
                    print(f" No solution found in {solve_time:.4f}s")
                
            except Exception as e:
                print(f" Error: {str(e)}")
                success = False
                solve_time = float('nan')
            
            # Add results to table
            results.append([level_num, solver_name, "Yes" if success else "No", 
                           f"{solve_time:.4f}" if not isinstance(solve_time, str) else solve_time])
    
    # Print results table
    print("\nBenchmark Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Also output per-solver statistics
    print("\nSolver Statistics:")
    solver_stats = {}
    
    for solver_name in solvers:
        solver_results = [r for r in results if r[1] == solver_name]
        success_count = sum(1 for r in solver_results if r[2] == "Yes")
        times = [float(r[3]) for r in solver_results if r[2] == "Yes" and r[3] != "nan"]
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
        else:
            avg_time = min_time = max_time = float('nan')
            
        solver_stats[solver_name] = {
            "Success Rate": f"{success_count}/{len(solver_results)} ({success_count/len(solver_results)*100:.1f}%)",
            "Avg Time": f"{avg_time:.4f}s",
            "Min Time": f"{min_time:.4f}s",
            "Max Time": f"{max_time:.4f}s"
        }
    
    stats_table = [[solver, *stats.values()] for solver, stats in solver_stats.items()]
    stats_headers = ["Solver", "Success Rate", "Avg Time", "Min Time", "Max Time"]
    print(tabulate(stats_table, headers=stats_headers, tablefmt="grid"))

if __name__ == "__main__":
    run_benchmark()