import Data.data_Types as _Types
from Utilities import GenerateEdges
import heapq
from typing import Dict, Set, List, Tuple, Optional
import copy

# Helper function for canonical bridge representation (no changes needed here)
def _canonical_bridge(pos1: _Types.Position, pos2: _Types.Position, count: int) -> Tuple[_Types.Position, _Types.Position, int]:
    """Returns the bridge tuple with positions sorted lexicographically (y, then x)."""
    if (pos1.y, pos1.x) < (pos2.y, pos2.x):
        return (pos1, pos2, count)
    else:
        return (pos2, pos1, count)

class PuzzleState:
    """Represents a state in the Hashiwokakero puzzle solving process."""

    def __init__(self, grid: _Types.Matrix, islands: Dict[_Types.Position, int],
                 remaining_bridges: Dict[_Types.Position, int],
                 bridges_placed: Set[Tuple[_Types.Position, _Types.Position, int]], # Store canonical bridges
                 cost: int = 0):
        """Initialize a puzzle state."""
        self.grid = grid
        self.islands = islands
        self.remaining_bridges = remaining_bridges
        # Ensure bridges_placed contains only canonical tuples if not guaranteed by caller
        # For performance, assume the caller provides canonical tuples
        self.bridges_placed = bridges_placed # Set of canonical (pos1, pos2, count) where pos1 < pos2
        self.cost = cost
        self.h_score = self._calculate_heuristic()
        self.f_score = self.cost + self.h_score
        self._hash = None
        # _bridge_lookup keys are (pos1, pos2) tuples where pos1 < pos2
        self._bridge_lookup = { (p1, p2): count for p1, p2, count in self.bridges_placed }

    def _calculate_heuristic(self) -> int:
        """Calculate heuristic value: sum of remaining bridges."""
        return sum(self.remaining_bridges.values())


    def _find_connected_islands(self) -> Set[_Types.Position]:
        """Find all connected islands using DFS based on placed bridges."""
        if not self.islands:
            return set()

        num_islands = len(self.islands)
        if num_islands == 0:
            return set()

        start_node = next(iter(self.islands.keys())) # Default start
        if not self.bridges_placed:
             # If no bridges, connectivithy only met if 0 or 1 island exists
             return set(self.islands.keys()) if num_islands <= 1 else {start_node}

        # Build adjacency list only from islands involved in bridges
        adj = {pos: [] for pos in self.islands}
        nodes_in_graph = set()
        for p1, p2, _ in self.bridges_placed:
             adj[p1].append(p2)
             adj[p2].append(p1)
             nodes_in_graph.add(p1)
             nodes_in_graph.add(p2)
             start_node = p1 # Start DFS from a node known to be in the graph

        if not nodes_in_graph: # Should not happen if bridges_placed is not empty
             return set(self.islands.keys()) if num_islands <= 1 else {start_node}

        visited = set()
        stack = [start_node]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # Use .get() for safety, though adj should contain all bridged islands
            for neighbor in adj.get(current, []):
                 if neighbor not in visited:
                     stack.append(neighbor)

        return visited

    def is_goal(self) -> bool:
        """Check if this state is a goal state."""
        # 1. All islands must have exactly 0 remaining bridges required.
        if any(rem != 0 for rem in self.remaining_bridges.values()):
             return False

        # 2. All islands must form a single connected component.
        num_islands = len(self.islands)
        if num_islands <= 1:
            return True # 0 or 1 island is trivially connected and meets goal if rem==0

        connected_islands = self._find_connected_islands()
        # The connected component must contain *all* islands present in the puzzle
        return len(connected_islands) == num_islands

    # --- THIS METHOD IS CORRECTED ---
    def get_current_bridge_count(self, pos1: _Types.Position, pos2: _Types.Position) -> int:
        """Get the number of bridges currently between pos1 and pos2."""
        # Create the canonical key (positions sorted lexicographically)
        # This key matches how _bridge_lookup is populated.
        if (pos1.y, pos1.x) < (pos2.y, pos2.x):
            canonical_key = (pos1, pos2)
        else:
            canonical_key = (pos2, pos1)
        # Lookup using the canonical key
        return self._bridge_lookup.get(canonical_key, 0)

    def __lt__(self, other):
        """Comparison for priority queue (lower f_score is better)."""
        if not isinstance(other, PuzzleState):
            return NotImplemented # Important for heapq
        if self.f_score == other.f_score:
            return self.h_score < other.h_score
        return self.f_score < other.f_score

    # --- THIS METHOD IS CORRECTED ---
    def __hash__(self):
        """Hash function based on the canonical set of placed bridges."""
        if self._hash is None:
            # Sort the canonical bridge tuples themselves for a fully deterministic hash.
            # The key ensures sorting works even if Position objects aren't directly comparable.
            def sort_key(bridge_tuple):
                pos1, pos2, count = bridge_tuple
                # Sort based on pos1 (y,x), then pos2 (y,x), then count
                return ((pos1.y, pos1.x), (pos2.y, pos2.x), count)

            # Convert set to list, sort using the key, then convert to tuple for hashing
            bridge_list = sorted(list(self.bridges_placed), key=sort_key)
            self._hash = hash(tuple(bridge_list))
        return self._hash

    def __eq__(self, other):
        """Equality check based on placed bridges."""
        if not isinstance(other, PuzzleState):
            return False
        # Compare the sets directly. Relies on __hash__ being correct for set elements.
        # Also relies on bridges_placed containing canonical tuples.
        return self.bridges_placed == other.bridges_placed

# --- Helper Functions --- (no changes needed)

def can_place_bridge(grid: _Types.Matrix, islands: Dict[_Types.Position, int], start: _Types.Position, end: _Types.Position) -> bool:
    """Check if a bridge path is clear between start and end."""
    # Ensure islands are in same row or column, and not the same island
    if start == end or (start.x != end.x and start.y != end.y):
        return False

    rows, cols = len(grid), len(grid[0])

    if start.y == end.y:  # Horizontal bridge
        y = start.y
        min_x, max_x = min(start.x, end.x), max(start.x, end.x)
        for x in range(min_x + 1, max_x):
            # Basic boundary check (should be redundant if edges are generated correctly)
            if not (0 <= x < cols): return False
            cell = grid[y][x]
            pos = _Types.Position(y,x)
            # Check for obstruction: another island or a crossing bridge
            if pos in islands or cell in ('|', '$'):
                return False
            # Check for unexpected characters (e.g., island numbers, though islands check handles this)
            if cell not in ('0', '-', '='):
                 # This case might catch errors or invalid grid states
                 # print(f"Warning: Unexpected char '{cell}' at {pos} during horizontal check")
                 return False # Treat unexpected chars as obstructions

    else:  # Vertical bridge (start.x == end.x)
        x = start.x
        min_y, max_y = min(start.y, end.y), max(start.y, end.y)
        for y in range(min_y + 1, max_y):
            # Basic boundary check
            if not (0 <= y < rows): return False
            cell = grid[y][x]
            pos = _Types.Position(y,x)
            # Check for obstruction: another island or a crossing bridge
            if pos in islands or cell in ('-', '='):
                return False
            # Check for unexpected characters
            if cell not in ('0', '|', '$'):
                 # print(f"Warning: Unexpected char '{cell}' at {pos} during vertical check")
                 return False

    return True


def place_bridge(grid: _Types.Matrix, start: _Types.Position, end: _Types.Position, is_double: bool) -> _Types.Matrix:
    """Place/update a bridge between two islands in the grid. Assumes path is clear."""
    # Create a mutable copy (list of lists)
    new_grid_list = [list(row) for row in grid]

    if start.y == end.y:  # Horizontal bridge
        bridge_symbol = "=" if is_double else "-"
        min_x, max_x = min(start.x, end.x), max(start.x, end.x)
        for col in range(min_x + 1, max_x):
            new_grid_list[start.y][col] = bridge_symbol
    else:  # Vertical bridge
        bridge_symbol = "$" if is_double else "|"
        min_y, max_y = min(start.y, end.y), max(start.y, end.y)
        for row in range(min_y + 1, max_y):
            new_grid_list[row][start.x] = bridge_symbol

    # Convert back to the original matrix type (e.g., tuple of strings) if necessary
    # Adapt this based on the actual type of _Types.Matrix
    if isinstance(grid, tuple) and all(isinstance(r, str) for r in grid):
        return tuple("".join(row) for row in new_grid_list)
    elif isinstance(grid, list) and all(isinstance(r, list) for r in grid):
         return new_grid_list # Return list of lists if that's the type
    else:
         # Default or fallback: Assume tuple of strings is desired
         return tuple("".join(row) for row in new_grid_list)


# --- A* Solver ---

def Solve(level: _Types.Level) -> Optional[_Types.Matrix]:
    """Solver for the level using A* algorithm."""
    if not level.islands:
        print("Warning: Level has no islands.")
        # Return the grid if it's considered solved, or None/error
        return level.grid if not any(c != '0' for row in level.grid for c in row) else None

    edges = GenerateEdges(level)

    initial_grid = level.grid
    islands = level.islands # Keep original island requirements
    remaining_bridges = level.islands.copy()

    initial_state = PuzzleState(
        grid=initial_grid,
        islands=islands,
        remaining_bridges=remaining_bridges,
        bridges_placed=set(),
        cost=0
    )

    open_set = [initial_state]
    closed_set = set() # Stores hashes of visited states

    max_iter = 100000 # Safety break for potentially huge search spaces
    count = 0

    while open_set:
        count += 1
        if count > max_iter:
             print(f"Error: Solver exceeded maximum iterations ({max_iter})")
             return None # Avoid infinite loops

        current_state = heapq.heappop(open_set)

        state_hash = hash(current_state)
        if state_hash in closed_set:
            continue

        # Goal check after potentially skipping duplicate
        if current_state.is_goal():
            print(f"Solution Found! Cost: {current_state.cost}, States Explored: {len(closed_set)}")
            return current_state.grid

        closed_set.add(state_hash)

        # --- Generate Successor States ---
        for edge in edges:
            p1, p2 = edge.startingPosition, edge.endingPosition

            # --- THIS PART IS CORRECTED ---
            # Use the state's method to find the current number of bridges
            current_num = current_state.get_current_bridge_count(p1, p2)

            # (Removed the redundant/incorrect lookup logic that was here)

            # --- Try adding ONE more bridge layer ---
            if current_num < 2: # Can add a 1st or 2nd bridge
                # Check capacity BEFORE checking path (quick check)
                if current_state.remaining_bridges[p1] >= 1 and \
                   current_state.remaining_bridges[p2] >= 1:

                    # Check if path is clear (more expensive check)
                    if can_place_bridge(current_state.grid, current_state.islands, p1, p2):

                        next_num = current_num + 1
                        is_double = (next_num == 2)

                        new_grid = place_bridge(current_state.grid, p1, p2, is_double)

                        new_remaining = current_state.remaining_bridges.copy()
                        new_remaining[p1] -= 1
                        new_remaining[p2] -= 1

                        new_bridges_set = current_state.bridges_placed.copy()
                        # Create the canonical tuple for the *new* bridge state
                        new_canon_bridge = _canonical_bridge(p1, p2, next_num)

                        # If we added the 2nd bridge, remove the 1st bridge entry
                        if current_num == 1:
                            old_bridge = _canonical_bridge(p1, p2, 1)
                            # Use discard to avoid error if somehow missing (shouldn't happen)
                            new_bridges_set.discard(old_bridge)

                        # Add the new bridge entry (either 1st or 2nd)
                        new_bridges_set.add(new_canon_bridge)

                        new_cost = current_state.cost + 1 # Cost = number of bridge layers added

                        new_state = PuzzleState(
                            grid=new_grid,
                            islands=current_state.islands,
                            remaining_bridges=new_remaining,
                            bridges_placed=new_bridges_set,
                            cost=new_cost
                        )

                        # Add to open set only if it hasn't been closed
                        # The hash check handles states already fully explored.
                        # We might add a state variation that's already in open_set
                        # but with a higher cost - heapq handles this naturally.
                        if hash(new_state) not in closed_set:
                            heapq.heappush(open_set, new_state)


    print(f"No solution found after exploring {len(closed_set)} states.")
    return None


__all__ = ["Solve"]