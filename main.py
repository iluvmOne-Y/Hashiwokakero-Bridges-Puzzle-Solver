from pysat.solvers import Glucose3
from pysat.card import CardEnc
import itertools

# Example grid for Hashiwokakero (0 means no island, numbers 1-8 indicate island with bridge count)
# Try a different grid that is known to be solvable for testing:
def read_grid_from_file(file_path):
    """Read a Hashiwokakero grid from a text file."""
    grid = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split by commas and convert to integers
            row = [int(x.strip()) for x in line.split(',')]
            grid.append(row)
    return grid

# Step 1: Generate possible bridge edges (E)
def generate_edges(grid):
    rows, cols = len(grid), len(grid[0])
    edges = {}  # (i1, j1, i2, j2): edge_id
    edge_id = 1
    
    # Horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            if grid[i][j] != 0:  # Start island
                for k in range(j + 1, cols):
                    if grid[i][k] != 0:  # End island
                        edges[(i, j, i, k)] = edge_id
                        edge_id += 1
                        break
    
    # Vertical edges
    for j in range(cols):
        for i in range(rows - 1):
            if grid[i][j] != 0:  # Start island
                for k in range(i + 1, rows):
                    if grid[k][j] != 0:  # End island
                        edges[(i, j, k, j)] = edge_id
                        edge_id += 1
                        break
    
    return edges

# Step 2: Define variables for bridge states (p_e and q_e)
def define_bridge_variables(edges):
    p_vars = {}  # p_e: single bridge
    q_vars = {}  # q_e: double bridge
    var_counter = 1
    
    for edge in edges:
        p_vars[edge] = var_counter  # Single bridge variable
        var_counter += 1
        q_vars[edge] = var_counter  # Double bridge variable
        var_counter += 1
    
    return p_vars, q_vars, var_counter

# Step 3: Generate CNF clauses
def generate_cnf(grid, edges, p_vars, q_vars, var_counter):
    clauses = []
    rows, cols = len(grid), len(grid[0])
    
    # Constraint 1: A bridge cannot be both single and double (¬p_e ∨ ¬q_e)
    for edge in edges:
        p, q = p_vars[edge], q_vars[edge]
        clauses.append([-p, -q])
    
    # Constraint 2: Island bridge count constraints
    island_edges = {}  # Map each island to its connected edges
    islands = []  # List of island positions
    
    for edge in edges:
        i1, j1, i2, j2 = edge
        v1, v2 = (i1, j1), (i2, j2)
        island_edges.setdefault(v1, []).append(edge)
        island_edges.setdefault(v2, []).append(edge)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:  # Island exists
                islands.append((i, j))
                n = grid[i][j]  # Required bridge count
                ev = island_edges.get((i, j), [])
                
                print(f"Island ({i},{j}) value={n} has {len(ev)} possible edges")
                
                # Create variables that represent the contribution of each edge to this island
                vars_list = []
                for e in ev:
                    # Single bridge contributes 1
                    p = p_vars[e]
                    vars_list.append(p)
                    
                    # Double bridge contributes 2 (add the variable twice)
                    q = q_vars[e]
                    vars_list.append(q)
                    vars_list.append(q)
                
                # Now add the cardinality constraint: exactly n bridges
                if vars_list:
                    # Use AtMostK and AtLeastK to create exactly-K
                    at_least = CardEnc.atleast(lits=vars_list, bound=n, encoding=1, top_id=var_counter)
                    var_counter = at_least.nv + 1
                    clauses.extend(at_least.clauses)
                    
                    at_most = CardEnc.atmost(lits=vars_list, bound=n, encoding=1, top_id=var_counter)
                    var_counter = at_most.nv + 1
                    clauses.extend(at_most.clauses)
    

    
    # Constraint 3: No crossing bridges
    for e1, id1 in edges.items():
        for e2, id2 in edges.items():
            if id1 < id2:  # Avoid duplicates
                i1, j1, i2, j2 = e1
                p1, q1 = p_vars[e1], q_vars[e1]
                i3, j3, i4, j4 = e2
                p2, q2 = p_vars[e2], q_vars[e2]
                
                # Check if bridges would cross
                if i1 == i2 and j3 == j4:  # e1 is horizontal, e2 is vertical
                    if min(j1, j2) < j3 < max(j1, j2) and min(i3, i4) < i1 < max(i3, i4):
                        clauses.extend([
                            [-p1, -p2], [-p1, -q2],
                            [-q1, -p2], [-q1, -q2]
                        ])
                elif j1 == j2 and i3 == i4:  # e1 is vertical, e2 is horizontal
                    if min(i1, i2) < i3 < max(i1, i2) and min(j3, j4) < j1 < max(j3, j4):
                        clauses.extend([
                            [-p1, -p2], [-p1, -q2],
                            [-q1, -p2], [-q1, -q2]
                        ])
    
        # Add this after the no crossing bridges constraints
    
    # Constraint 4: All islands must be connected
    # We'll use a simpler approach - make sure at least |islands|-1 bridges exist
    bridge_vars = []
    for edge in edges:
        # A bridge exists on this edge if either p or q is true
        bridge_exists = var_counter
        var_counter += 1
        p, q = p_vars[edge], q_vars[edge]
        
        # p → bridge_exists, q → bridge_exists
        clauses.append([-p, bridge_exists])
        clauses.append([-q, bridge_exists])
        
        # ¬bridge_exists → ¬p ∧ ¬q
        clauses.append([-bridge_exists, p, q])
        
        bridge_vars.append(bridge_exists)
    
    # Need at least |islands|-1 bridges to connect all islands
    if len(islands) > 1:
        min_bridges = len(islands) - 1
        atleast = CardEnc.atleast(lits=bridge_vars, bound=min_bridges, encoding=1, top_id=var_counter)
        var_counter = atleast.nv + 1
        clauses.extend(atleast.clauses)
    
    return clauses, var_counter
# Step 4: Solve with SAT solver and interpret result
def solve_hashiwokakero(grid):
    edges = generate_edges(grid)
    p_vars, q_vars, var_counter = define_bridge_variables(edges)
    clauses, var_counter = generate_cnf(grid, edges, p_vars, q_vars, var_counter)
    
    solver = Glucose3()
    for clause in clauses:
        solver.add_clause(clause)
    
    if solver.solve():
        model = solver.get_model()
        # Debug output
        print(f"\nDebug - Model: {model[:20]}...")  # Show first 20 variables to avoid clutter
        
        # Further debugging to help understand what bridges are being selected
        print("\nSelected bridges:")
        
        solution = {}
        for edge in edges:
            p, q = p_vars[edge], q_vars[edge]
            # A positive literal in the model means the variable is true
            p_val = p in model
            q_val = q in model
            
            if p_val or q_val:
                bridges = 1 if p_val else 0
                bridges += 2 if q_val else 0
                i1, j1, i2, j2 = edge
                print(f"Edge ({i1},{j1}) to ({i2},{j2}): {'single' if p_val else ''}{' double' if q_val else ''} bridge")
                solution[edge] = bridges
                
        return solution
    else:
        print("\nDebug - No satisfying assignment found")
        return None
# Step 5: Print the solution with improved visualization
def print_solution(grid, solution):
    """Print the solution in the required format."""
    rows, cols = len(grid), len(grid[0])
    
    # Create a visualization grid with the correct symbols
    visual_grid = [["0" if grid[i][j] == 0 else str(grid[i][j]) for j in range(cols)] for i in range(rows)]
    
    # Track vertical bridges for potential crossings
    vertical_bridges = {}
    
    # First mark all vertical bridges
    for (i1, j1, i2, j2), bridges in solution.items():
        if i1 != i2:  # Vertical bridge
            for i in range(i1 + 1, i2):
                vertical_bridges[(i, j1)] = bridges
    
    # Then place all bridges in the grid
    for (i1, j1, i2, j2), bridges in solution.items():
        if i1 == i2:  # Horizontal bridge
            for j in range(j1 + 1, j2):
                # Check if there's a vertical bridge crossing
                if (i1, j) in vertical_bridges:
                    visual_grid[i1][j] = "$"
                else:
                    visual_grid[i1][j] = "=" if bridges == 2 else "−"
        else:  # Vertical bridge
            for i in range(i1 + 1, i2):
                # Only place if not already marked as crossing
                if visual_grid[i][j1] != "$":
                    visual_grid[i][j1] = "$" if bridges == 2 else "|"
    
    # Print the grid in the requested format
    for row in visual_grid:
        print("[ " + " , ".join(f'"{x}"' for x in row) + " ]")

# Print edge information to debug
def debug_edges(grid):
    edges = generate_edges(grid)
    print("Generated edges:")
    for edge, edge_id in edges.items():
        i1, j1, i2, j2 = edge
        print(f"Edge {edge_id}: ({i1},{j1}) to ({i2},{j2})")
    return edges

# Add this after the existing grid


# Modify the main execution to test both grids
# Add this to the end of your file
def solve_from_file(file_path):
    """Solve a Hashiwokakero puzzle from a file."""
    grid = read_grid_from_file(file_path)
    solution = solve_hashiwokakero(grid)
    
    if solution:
        print("\nSolution found:")
        print_solution(grid, solution)
    else:
        print("No solution exists.")

# Example usage
if __name__ == "__main__":
    solve_from_file("1.txt")
    
    