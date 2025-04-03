import Data.data_Types as _Types
from Utilities import GenerateEdges
from collections import deque
from typing import Dict, Set, List, Tuple, Optional
from copy import deepcopy

class PuzzleState:
    """Represents a state in the Hashiwokakero puzzle solving process."""
    def __init__(self, 
                 grid: _Types.Matrix,
                 islands: Dict[_Types.Position, int],
                 remaining_bridges: Dict[_Types.Position, int],
                 bridges_placed: Dict[Tuple[_Types.Position, _Types.Position], int],
                 edge_index: int = 0):
        self.grid = grid
        self.islands = islands
        self.remaining_bridges = remaining_bridges
        self.bridges_placed = bridges_placed
        self.edge_index = edge_index

    def is_goal(self) -> bool:
        """Check if this state is a goal state."""
        # All islands must have exactly 0 remaining bridges
        if any(bridges != 0 for bridges in self.remaining_bridges.values()):
            return False

        # All islands must be connected
        return self._check_connectivity()

    def _check_connectivity(self) -> bool:
        """Check if all islands are connected."""
        if not self.islands:
            return True

        visited = set()
        stack = [next(iter(self.islands.keys()))]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for (start, end), count in self.bridges_placed.items():
                    if count > 0:
                        if start == current and end not in visited:
                            stack.append(end)
                        elif end == current and start not in visited:
                            stack.append(start)

        return len(visited) == len(self.islands)

    def can_place_bridge(self, start: _Types.Position, end: _Types.Position, count: int) -> bool:
        """Check if a bridge can be placed between two islands."""
        if self.remaining_bridges[start] < count or self.remaining_bridges[end] < count:
            return False

        # Check if the path is clear
        if start.x == end.x:  # Vertical bridge
            y_min, y_max = sorted([start.y, end.y])
            for y in range(y_min + 1, y_max):
                if self.grid[y][start.x] not in ('0', '|', '$'):
                    return False
        elif start.y == end.y:  # Horizontal bridge
            x_min, x_max = sorted([start.x, end.x])
            for x in range(x_min + 1, x_max):
                if self.grid[start.y][x] not in ('0', '-', '='):
                    return False
        else:
            return False

        return True

    def place_bridge(self, start: _Types.Position, end: _Types.Position, count: int):
        """Place a bridge between two islands."""
        new_state = deepcopy(self)

        # Update the grid
        if start.x == end.x:  # Vertical bridge
            bridge_char = '|' if count == 1 else '$'
            y_min, y_max = sorted([start.y, end.y])
            for y in range(y_min + 1, y_max):
                new_state.grid[y][start.x] = bridge_char
        elif start.y == end.y:  # Horizontal bridge
            bridge_char = '-' if count == 1 else '='
            x_min, x_max = sorted([start.x, end.x])
            for x in range(x_min + 1, x_max):
                new_state.grid[start.y][x] = bridge_char

        # Update the islands and bridges
        new_state.remaining_bridges[start] -= count
        new_state.remaining_bridges[end] -= count
        new_state.bridges_placed[(start, end)] = new_state.bridges_placed.get((start, end), 0) + count

        return new_state

def Solve(level: _Types.Level) -> Optional[_Types.Matrix]:
    """Solve the Hashiwokakero puzzle using DFS."""
    edges = GenerateEdges(level)
    initial_state = PuzzleState(
        grid=level.grid,
        islands=level.islands,
        remaining_bridges=level.islands.copy(),
        bridges_placed={},
        edge_index=0,
    )

    stack = deque([initial_state])
    counter = 0

    while stack:
        current_state = stack.pop()
        counter += 1

        # Check if the current state is a solution
        if current_state.is_goal():
            print(f"Solution found after exploring {counter} states!")
            return current_state.grid

        # If all edges have been processed, skip
        if current_state.edge_index >= len(edges):
            continue

        # Get the current edge
        edge = edges[current_state.edge_index]
        start, end = edge.startingPosition, edge.endingPosition

        # Try placing 2 bridges
        if current_state.can_place_bridge(start, end, 2):
            new_state = current_state.place_bridge(start, end, 2)
            new_state.edge_index += 1
            stack.append(new_state)

        # Try placing 1 bridge
        if current_state.can_place_bridge(start, end, 1):
            new_state = current_state.place_bridge(start, end, 1)
            new_state.edge_index += 1
            stack.append(new_state)

        # Try skipping this edge
        new_state = deepcopy(current_state)
        new_state.edge_index += 1
        stack.append(new_state)

    print(f"No solution found after exploring {counter} states.")
    return None

__all__ = ["Solve"]