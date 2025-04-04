import Data.data_Types as _Types
from Utilities import GenerateEdges  # Fixed function name
import heapq
from typing import Dict, Set, List, Tuple, Optional
import copy, sys


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

def Solve(level: _Types.Level) -> Optional[_Types.Matrix]:
    """
    Solve the Hashiwokakero puzzle using an optimized DFS approach (non-recursive),
    incorporating Hashi solving techniques.
    """
    if not level.islands:
        return level.grid if not any(c != '0' for row in level.grid for c in row) else None

    edges = GenerateEdges(level)
    initial_grid = level.grid
    islands = level.islands  # Keep original island requirements
    remaining_bridges = level.islands.copy()

    initial_state = PuzzleState(
        grid=initial_grid,
        islands=islands,
        remaining_bridges=remaining_bridges,
        bridges_placed=set()
    )

    stack = [initial_state]
    visited = set()

    while stack:
        state = stack.pop()

        if state.is_goal():
            return state.grid

        if state in visited:
            continue
        visited.add(state)

        # Sort islands by increasing remaining bridges to prioritize solving constrained islands
        sorted_islands = sorted(state.islands.keys(), key=lambda pos: state.remaining_bridges[pos])
        
        for pos1 in sorted_islands:
            if state.remaining_bridges[pos1] == 0:
                continue  # Skip fully satisfied islands

            # Get possible connections
            valid_neighbors = [pos2 for pos2 in sorted_islands if pos1 != pos2 and state.remaining_bridges[pos2] > 0 and can_place_bridge(state.grid, state.islands, pos1, pos2)]
            
            if not valid_neighbors:
                continue
            
            # Just Enough Neighbor Technique
            if len(valid_neighbors) == state.remaining_bridges[pos1]:
                for pos2 in valid_neighbors:
                    max_bridges = min(2, state.remaining_bridges[pos1], state.remaining_bridges[pos2])
                    new_grid = place_bridge(state.grid, pos1, pos2, max_bridges == 2)
                    new_remaining_bridges = state.remaining_bridges.copy()
                    new_remaining_bridges[pos1] -= max_bridges
                    new_remaining_bridges[pos2] -= max_bridges
                    new_bridges_placed = state.bridges_placed.copy()
                    new_bridges_placed.add(_canonical_bridge(pos1, pos2, max_bridges))

                    stack.append(PuzzleState(
                        grid=new_grid,
                        islands=state.islands,
                        remaining_bridges=new_remaining_bridges,
                        bridges_placed=new_bridges_placed,
                    ))
                continue

            # One Unsolved Neighbor Technique
            if len(valid_neighbors) == 1:
                pos2 = valid_neighbors[0]
                max_bridges = min(2, state.remaining_bridges[pos1], state.remaining_bridges[pos2])
                new_grid = place_bridge(state.grid, pos1, pos2, max_bridges == 2)
                new_remaining_bridges = state.remaining_bridges.copy()
                new_remaining_bridges[pos1] -= max_bridges
                new_remaining_bridges[pos2] -= max_bridges
                new_bridges_placed = state.bridges_placed.copy()
                new_bridges_placed.add(_canonical_bridge(pos1, pos2, max_bridges))

                stack.append(PuzzleState(
                    grid=new_grid,
                    islands=state.islands,
                    remaining_bridges=new_remaining_bridges,
                    bridges_placed=new_bridges_placed,
                ))
                continue

            # Few Neighbors Technique - Prioritize islands with fewer options
            for pos2 in sorted(valid_neighbors, key=lambda p: len([n for n in sorted_islands if can_place_bridge(state.grid, state.islands, p, n)])):
                max_bridges = min(2, state.remaining_bridges[pos1], state.remaining_bridges[pos2])
                for count in range(1, max_bridges + 1):
                    new_grid = place_bridge(state.grid, pos1, pos2, count == 2)
                    new_remaining_bridges = state.remaining_bridges.copy()
                    new_remaining_bridges[pos1] -= count
                    new_remaining_bridges[pos2] -= count
                    new_bridges_placed = state.bridges_placed.copy()
                    new_bridges_placed.add(_canonical_bridge(pos1, pos2, count))

                    stack.append(PuzzleState(
                        grid=new_grid,
                        islands=state.islands,
                        remaining_bridges=new_remaining_bridges,
                        bridges_placed=new_bridges_placed,
                    ))
    
    return None



__all__ = ["Solve"]
