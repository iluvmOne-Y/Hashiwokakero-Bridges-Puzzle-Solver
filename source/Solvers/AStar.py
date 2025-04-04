
import Data.data_Types as _Types
from Utilities import GenerateEdges # Keep import if needed elsewhere
import heapq
import sys
from typing import Dict, Set, List, Tuple, Optional, Any
import copy
import time


sys.setrecursionlimit(10000) # Keep if complex states cause deep checks

# --- Constants ---
BRIDGE_H_SINGLE = '-'
BRIDGE_H_DOUBLE = '='
BRIDGE_V_SINGLE = '|'
BRIDGE_V_DOUBLE = '$' # Standard symbol
OBSTACLES_H = {BRIDGE_V_SINGLE, BRIDGE_V_DOUBLE}
OBSTACLES_V = {BRIDGE_H_SINGLE, BRIDGE_H_DOUBLE}
EMPTY_CELL = '0'
BRIDGE_SYMBOLS = {BRIDGE_H_SINGLE, BRIDGE_H_DOUBLE, BRIDGE_V_SINGLE, BRIDGE_V_DOUBLE}


# --- Helper Functions ---

def _canonical_bridge(pos1: _Types.Position, pos2: _Types.Position, count: int) -> Tuple[_Types.Position, _Types.Position, int]:
    """Returns the bridge tuple with positions sorted lexicographically (y, then x)."""
    if (pos1.y, pos1.x) < (pos2.y, pos2.x):
        return (pos1, pos2, count)
    else:
        return (pos2, pos1, count)

def can_place_bridge(grid: _Types.Matrix, islands: Dict[_Types.Position, int], start: _Types.Position, end: _Types.Position) -> bool:
    """Check if a bridge path is clear between start and end."""
    # Ensure islands are in same row or column, and not the same island
    if start == end or (start.x != end.x and start.y != end.y):
        return False

    rows, cols = len(grid), len(grid[0])

    if start.y == end.y:  # Horizontal path
        y = start.y
        min_x, max_x = min(start.x, end.x), max(start.x, end.x)
        for x in range(min_x + 1, max_x):
            # Boundary check might be needed if grid access isn't guaranteed safe
            # if not (0 <= x < cols): return False # Should be safe if logic is correct
            pos = _Types.Position(y, x)
            cell = grid[y][x]
            # Check for obstruction: another island OR a crossing (vertical) bridge
            if pos in islands or cell in OBSTACLES_H:
                return False
            # Allow existing horizontal bridge segments of potentially lower thickness
            if cell not in (EMPTY_CELL, BRIDGE_H_SINGLE, BRIDGE_H_DOUBLE):
                # print(f"Warning: Unexpected char '{cell}' at {pos} during horizontal check")
                return False # Treat unexpected non-bridge chars as obstructions

    else:  # Vertical path (start.x == end.x)
        x = start.x
        min_y, max_y = min(start.y, end.y), max(start.y, end.y)
        for y in range(min_y + 1, max_y):
            # Boundary check
            # if not (0 <= y < rows): return False # Should be safe
            pos = _Types.Position(y, x)
            cell = grid[y][x]
            # Check for obstruction: another island OR a crossing (horizontal) bridge
            if pos in islands or cell in OBSTACLES_V:
                return False
            # Allow existing vertical bridge segments of potentially lower thickness
            if cell not in (EMPTY_CELL, BRIDGE_V_SINGLE, BRIDGE_V_DOUBLE):
                # print(f"Warning: Unexpected char '{cell}' at {pos} during vertical check")
                return False

    return True


def place_bridge_on_grid(grid: _Types.Matrix, start: _Types.Position, end: _Types.Position, thickness: int) -> _Types.Matrix:
    """Place/update a bridge on the grid. Assumes path is clear."""
    # Create a mutable copy (list of lists) for modification
    new_grid_list = [list(row) for row in grid]

    if start.y == end.y:  # Horizontal bridge
        bridge_symbol = BRIDGE_H_DOUBLE if thickness == 2 else BRIDGE_H_SINGLE
        min_x, max_x = min(start.x, end.x), max(start.x, end.x)
        for col in range(min_x + 1, max_x):
            # Overwrite is correct here as we manage state externally
            new_grid_list[start.y][col] = bridge_symbol
    else:  # Vertical bridge
        bridge_symbol = BRIDGE_V_DOUBLE if thickness == 2 else BRIDGE_V_SINGLE
        min_y, max_y = min(start.y, end.y), max(start.y, end.y)
        for row in range(min_y + 1, max_y):
            # Overwrite is correct
            new_grid_list[row][start.x] = bridge_symbol

    # Convert back to the expected immutable matrix type (tuple of strings)
    # Adjust if _Types.Matrix is different (e.g., list of lists)
    return tuple("".join(row) for row in new_grid_list)


class PuzzleState:
    """Represents a state in the Hashiwokakero puzzle solving process."""

    def __init__(self, grid: _Types.Matrix, islands: Dict[_Types.Position, int],
                 remaining_bridges: Dict[_Types.Position, int],
                 bridges_placed: Set[Tuple[_Types.Position, _Types.Position, int]], # Expects canonical bridges
                 cost: int = 0):
        """Initialize a puzzle state."""
        self.grid = grid # Current grid reflecting placed bridges
        self.islands = islands # Static island info (original values)
        self.remaining_bridges = remaining_bridges # Current remaining counts
        self.bridges_placed = bridges_placed # Set of canonical (pos1, pos2, count) where pos1 < pos2
        self.cost = cost # Path cost (number of bridge segments placed, maybe?) or steps

        # --- Caching ---
        self._hash = None
        # _bridge_lookup keys are canonical (pos1, pos2) tuples where pos1 < pos2
        self._bridge_lookup = { (p1, p2): count for p1, p2, count in self.bridges_placed }
        # Cache for adjacency list and connectivity info to avoid recomputing
        self._adj_list: Optional[Dict[_Types.Position, List[_Types.Position]]] = None
        self._cached_components: Optional[int] = None

        # --- A* Scores ---
        # Calculate heuristic AFTER setting up caches it might use
        self.h_score = self._calculate_heuristic()
        self.f_score = self.cost + self.h_score


    def _get_adjacency_list(self) -> Dict[_Types.Position, List[_Types.Position]]:
        """Builds or returns cached adjacency list based on bridges_placed."""
        if self._adj_list is None:
            adj = {pos: [] for pos in self.islands}
            for p1, p2, _ in self.bridges_placed:
                adj[p1].append(p2)
                adj[p2].append(p1)
            self._adj_list = adj
        return self._adj_list

    def _count_connected_components(self) -> int:
        """Efficiently count connected components using BFS/DFS and caching."""
        if self._cached_components is not None:
            return self._cached_components

        num_islands = len(self.islands)
        if num_islands <= 1:
            self._cached_components = num_islands
            return num_islands

        adj = self._get_adjacency_list()
        visited = set()
        components = 0
        queue = [] # Using BFS

        for island_pos in self.islands:
            if island_pos not in visited:
                components += 1
                queue.append(island_pos)
                visited.add(island_pos)

                while queue:
                    current = queue.pop(0)
                    # Use the adjacency list for neighbors
                    for neighbor in adj.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

        self._cached_components = components
        return components

    def _calculate_heuristic(self) -> int:
        """Calculate heuristic. Less reliance on expensive checks."""
        # Heuristic 1: Sum of remaining bridges (divided by 2 as each bridge counts for two islands)
        # This is admissible.
        remaining_sum = sum(self.remaining_bridges.values())
        if remaining_sum < 0 : return float('inf') # Should not happen, but indicates invalid state

        h = remaining_sum / 2.0

        # Heuristic 2: Connectivity Penalty (Admissible)
        # Add a penalty for each component more than 1. Each extra component needs at least one bridge to connect.
        num_components = self._count_connected_components()
        if num_components > 1:
            # Adding (num_components - 1) is admissible because at least that many
            # bridges are needed to connect the components.
            h += (num_components - 1)

        # Heuristic 3: Penalty for impossible situations (makes heuristic non-admissible but useful)
        # Check for islands that *cannot* be satisfied. This helps prune bad states early.
        # Keep this check relatively lightweight. Avoid can_place_bridge here.
        for pos, remaining in self.remaining_bridges.items():
            if remaining <= 0: continue

            max_possible_from_neighbors = 0
            potential_neighbors = 0
            for other_pos, other_rem in self.remaining_bridges.items():
                if pos == other_pos or other_rem <= 0: continue

                # Check alignment only
                if pos.x == other_pos.x or pos.y == other_pos.y:
                    potential_neighbors += 1
                    current_bridges = self.get_current_bridge_count(pos, other_pos)
                    can_add = 2 - current_bridges
                    if can_add > 0:
                        # Max bridges this neighbor *could* contribute
                        add_contribution = min(can_add, other_rem)
                        max_possible_from_neighbors += add_contribution

            # If remaining is greater than max possible, this state is impossible
            if remaining > max_possible_from_neighbors:
                return float('inf') # Prune this state immediately

            # If an island needs bridges but has no potential neighbors left
            # (This check is partly covered above, but explicit check is clearer)
            if remaining > 0 and potential_neighbors == 0:
                 return float('inf')


        # Heuristic 4: Critical islands (Islands needing 1 or 2 bridges) - Tie breaking
        # Adding a small value can help prioritize states closer to completion.
        # Be careful not to make the heuristic inadmissible if optimality matters.
        # h += sum(0.1 for rem in self.remaining_bridges.values() if rem == 1)

        return int(h) # Return integer heuristic


    def prioritize_islands(self) -> List[Tuple[_Types.Position, int]]:
        """Prioritize islands for exploration based on constraints."""
        # Simpler, faster prioritization without can_place_bridge
        island_priorities = []
        for pos, rem in self.remaining_bridges.items():
            if rem <= 0: continue

            max_potential_bridges = 0
            viable_neighbor_slots = 0
            num_viable_neighbors = 0

            for other_pos, other_rem in self.remaining_bridges.items():
                 if pos == other_pos or other_rem <= 0: continue
                 if pos.x == other_pos.x or pos.y == other_pos.y:
                      current_num = self.get_current_bridge_count(pos, other_pos)
                      potential_add = 2 - current_num
                      if potential_add > 0:
                          max_contribution = min(other_rem, potential_add)
                          if max_contribution > 0:
                              # Check path *only if* heuristics aren't pruning enough
                              # if not can_place_bridge(self.grid, self.islands, pos, other_pos):
                              #      continue # Skip if path blocked
                              max_potential_bridges += max_contribution
                              viable_neighbor_slots += potential_add
                              num_viable_neighbors += 1


            if rem > max_potential_bridges: continue # Skip impossible islands found earlier
            if rem > 0 and num_viable_neighbors == 0: continue # Skip islands with no options

            # Priority Score Calculation:
            # Higher priority for:
            # 1. Islands that *must* connect to all possible remaining slots (rem == max_potential_bridges)
            # 2. Islands with only one neighbor option (num_viable_neighbors == 1)
            # 3. Islands needing fewer bridges (lower rem)
            # 4. Islands with fewer connection *slots* available (lower viable_neighbor_slots)

            is_forced = (rem == max_potential_bridges and rem > 0)
            is_single_option = (num_viable_neighbors == 1 and rem > 0)

            # Score: Higher is better priority. Tuple comparison works well.
            priority_score = (is_forced, is_single_option, -rem, -viable_neighbor_slots)

            island_priorities.append((priority_score, pos, rem))

        # Sort by priority score (descending)
        island_priorities.sort(key=lambda x: x[0], reverse=True)

        # Limit branching factor
        limit = min(8, max(3, len(self.islands) // 5)) # Dynamic limit
        return [(item[1], item[2]) for item in island_priorities[:limit]]


    def is_goal(self) -> bool:
        """Check if this state is a goal state."""
        # 1. All islands must have exactly 0 remaining bridges required.
        if any(rem != 0 for rem in self.remaining_bridges.values()):
             return False

        # 2. All islands must form a single connected component.
        num_islands = len(self.islands)
        if num_islands <= 1:
            return True # 0 or 1 island is trivially connected

        # Use the efficient component count
        return self._count_connected_components() == 1

    def get_current_bridge_count(self, pos1: _Types.Position, pos2: _Types.Position) -> int:
        """Get the number of bridges currently between pos1 and pos2."""
        # Create the canonical key
        key = (min(pos1, pos2, key=lambda p: (p.y, p.x)),
               max(pos1, pos2, key=lambda p: (p.y, p.x)))
        return self._bridge_lookup.get(key, 0)

    def __lt__(self, other: Any):
        """Comparison for priority queue (lower f_score is better)."""
        if not isinstance(other, PuzzleState):
            return NotImplemented
        # Primary sort by f_score, secondary by h_score (lower is better)
        if self.f_score != other.f_score:
            return self.f_score < other.f_score
        return self.h_score < other.h_score
        # Further tie-breaking could use cost (prefer shallower nodes)
        # if self.h_score != other.h_score:
        #     return self.h_score < other.h_score
        # return self.cost < other.cost


    def __hash__(self):
        """Hash function based on the canonical set of placed bridges."""
        if self._hash is None:
            # Hash based on the frozenset of canonical bridge tuples.
            # Sorting isn't strictly necessary if using frozenset, but doesn't hurt.
            # Using the frozenset directly might be slightly faster.
            self._hash = hash(frozenset(self.bridges_placed))
        return self._hash

    def __eq__(self, other: Any):
        """Equality check based on placed bridges."""
        if not isinstance(other, PuzzleState):
            return False
        # Optimization: check hash first if hashes are cached and reliable
        if self._hash is not None and other._hash is not None and self._hash != other._hash:
             return False
        # Compare the sets of bridges directly.
        return self.bridges_placed == other.bridges_placed

# --- Deterministic Preprocessing ---

def apply_deterministic_techniques(level: _Types.Level) -> Tuple[_Types.Matrix, Dict[_Types.Position, int], Set[Tuple[_Types.Position, _Types.Position, int]]]:
    """
    Apply deterministic solving techniques as preprocessing.
    Returns the updated grid, updated remaining counts, and the set of bridges placed.
    """
    # Clone grid (mutable) and island counts
    current_grid_list = [list(row) for row in level.grid]
    current_remaining = level.islands.copy()
    placed_bridges: Set[Tuple[_Types.Position, _Types.Position, int]] = set()
    # Use a lookup for faster bridge checks during this phase
    bridge_lookup: Dict[Tuple[_Types.Position, _Types.Position], int] = {}

    rows, cols = len(current_grid_list), len(current_grid_list[0])

    def get_current_bridge_count_local(p1, p2):
         key = (min(p1, p2, key=lambda p: (p.y, p.x)), max(p1, p2, key=lambda p: (p.y, p.x)))
         return bridge_lookup.get(key, 0)

    def place_bridge_local(p1, p2, thickness):
        # Update grid (list of lists version)
        if p1.y == p2.y: # Horizontal
            sym = BRIDGE_H_DOUBLE if thickness == 2 else BRIDGE_H_SINGLE
            for x in range(min(p1.x, p2.x) + 1, max(p1.x, p2.x)):
                current_grid_list[p1.y][x] = sym
        else: # Vertical
            sym = BRIDGE_V_DOUBLE if thickness == 2 else BRIDGE_V_SINGLE
            for y in range(min(p1.y, p2.y) + 1, max(p1.y, p2.y)):
                 current_grid_list[y][p1.x] = sym

        # Update remaining counts
        current_remaining[p1] -= thickness
        current_remaining[p2] -= thickness

        # Update placed bridges set and lookup
        canon_bridge = _canonical_bridge(p1, p2, thickness)
        key = (canon_bridge[0], canon_bridge[1])

        # Remove old entry if upgrading
        if thickness == 2 and get_current_bridge_count_local(p1, p2) == 1:
             old_bridge = _canonical_bridge(p1, p2, 1)
             placed_bridges.discard(old_bridge)

        placed_bridges.add(canon_bridge)
        bridge_lookup[key] = thickness


    changed = True
    iterations = 0
    max_deterministic_iter = len(level.islands) * 5 # Safety break

    # Convert grid to tuple of strings temporarily for can_place_bridge calls
    def get_immutable_grid():
        return tuple("".join(row) for row in current_grid_list)

    try:
        while changed and iterations < max_deterministic_iter:
            changed = False
            iterations += 1
            immutable_grid = get_immutable_grid() # Snapshot for checks in this iteration

            island_list = list(current_remaining.keys()) # Iterate over fixed list

            for pos in island_list:
                remaining = current_remaining[pos]
                if remaining <= 0: continue

                # Find valid neighbors and max possible bridges based on *current* state
                valid_neighbors: List[Tuple[_Types.Position, int]] = [] # Store neighbor and max bridges possible TO it
                total_max_possible_bridges = 0

                for other_pos in island_list:
                    if pos == other_pos: continue
                    other_rem = current_remaining[other_pos]
                    if other_rem <= 0: continue

                    if pos.x == other_pos.x or pos.y == other_pos.y:
                        current_num = get_current_bridge_count_local(pos, other_pos)
                        can_add = 2 - current_num
                        if can_add <= 0: continue

                        # Check path on the *current snapshot* of the grid
                        if can_place_bridge(immutable_grid, level.islands, pos, other_pos):
                            max_add_to_neighbor = min(can_add, remaining, other_rem)
                            if max_add_to_neighbor > 0:
                                valid_neighbors.append((other_pos, max_add_to_neighbor))
                                total_max_possible_bridges += max_add_to_neighbor

                # --- Apply Techniques ---

                # Technique 1: Impossible state check
                if remaining > total_max_possible_bridges:
                     print(f"Error: Deterministic preprocessing found impossible state for island {pos} (needs {remaining}, max possible {total_max_possible_bridges}).")
                     # Return original state or raise error? Let's return original for now.
                     return level.grid, level.islands.copy(), set()

                # Technique 2: Forced moves (remaining == total_max_possible)
                if remaining > 0 and remaining == total_max_possible_bridges:
                    # Must add max possible bridges to all valid neighbors
                    action_taken = False
                    for other_pos, max_add_to_neighbor in valid_neighbors:
                         if max_add_to_neighbor > 0 and get_current_bridge_count_local(pos, other_pos) < 2 :
                              # Determine thickness needed (1 or 2)
                              current_num = get_current_bridge_count_local(pos, other_pos)
                              needed_thickness = max_add_to_neighbor
                              
                              # If adding 1 makes it 2 bridges total, place double bridge
                              final_thickness = 2 if (current_num + needed_thickness) == 2 else needed_thickness
                              if current_num + needed_thickness > 2 : # should not happen
                                   print("Logic error in forced moves")
                                   continue

                              # We place the final number of bridges (1 or 2)
                              place_bridge_local(pos, other_pos, final_thickness)
                              action_taken = True
                              # print(f"Deterministic (Forced): Added {final_thickness} bridge(s) {pos}<->{other_pos}")


                    if action_taken:
                         changed = True
                         # Since we modified counts, recalculate for next island
                         immutable_grid = get_immutable_grid() # Update snapshot

                # Technique 3: Specific value islands (1, 2, 7, 8) in corners/edges (Simplified version)
                # Example: Island with value 1 and only one neighbor -> place 1 bridge
                # Example: Island with value 2 and only one neighbor -> place 2 bridges (if neighbor allows)
                elif len(valid_neighbors) == 1 and remaining > 0:
                     other_pos, max_add_to_neighbor = valid_neighbors[0]
                     other_rem = current_remaining[other_pos]
                     current_num = get_current_bridge_count_local(pos, other_pos)

                     if current_num < 2:
                          # If island needs 1 or 2, and neighbor allows, place it
                          if remaining == 1 and other_rem >= 1:
                                place_bridge_local(pos, other_pos, 1)
                                # print(f"Deterministic (Single N): Added 1 bridge {pos}<->{other_pos}")
                                changed = True
                                immutable_grid = get_immutable_grid()
                          elif remaining == 2 and other_rem >= 2:
                                place_bridge_local(pos, other_pos, 2)
                                # print(f"Deterministic (Single N): Added 2 bridges {pos}<->{other_pos}")
                                changed = True
                                immutable_grid = get_immutable_grid()

                # Add other deterministic rules here if needed (e.g., 3/4 on edge, 5/6 on edge, 7/8)

            if changed: continue # Restart loop if changes were made

        if iterations >= max_deterministic_iter:
             print("Warning: Deterministic preprocessing hit iteration limit.")

        final_immutable_grid = get_immutable_grid()
        return final_immutable_grid, current_remaining, placed_bridges

    except Exception as e:
        print(f"Error during deterministic preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to original state
        return level.grid, level.islands.copy(), set()


# --- A* Solver ---

def Solve(level: _Types.Level) -> Optional[_Types.Matrix]:
    """Enhanced A* solver using Hashiwokakero solving techniques."""
    start_time = time.time()
    try:
        if not level.islands:
            print("Warning: Level has no islands.")
            return level.grid if not any(c != EMPTY_CELL for row in level.grid for c in row) else None

        print("Applying deterministic preprocessing...")
        processed_grid, processed_remaining, processed_bridges = apply_deterministic_techniques(level)
        print(f"Preprocessing placed {len(processed_bridges)} bridges.")

        # Check if preprocessing already solved it
        is_solved_pre = True
        if any(rem != 0 for rem in processed_remaining.values()):
            is_solved_pre = False
        else:
            # Check connectivity if counts are zero
            temp_state_for_goal_check = PuzzleState(processed_grid, level.islands, processed_remaining, processed_bridges)
            if not temp_state_for_goal_check.is_goal():
                is_solved_pre = False

        if is_solved_pre:
             print(f"Puzzle solved during preprocessing. Time: {time.time() - start_time:.2f}s")
             return processed_grid


        print("Starting A* search...")
        initial_state = PuzzleState(
            grid=processed_grid,
            islands=level.islands, # Use original island values here
            remaining_bridges=processed_remaining,
            bridges_placed=processed_bridges,
            cost=0 # Initial cost is 0
        )

        # Check if initial state is already invalid
        if initial_state.h_score == float('inf'):
             print("Error: Initial state after preprocessing is invalid.")
             return None

        open_set = [initial_state] # Priority queue (min-heap)
        closed_set: Set[int] = set() # Store hashes of visited states

        # Limits and Counters
        island_count = len(level.islands)
        max_iter = min(500000, island_count * 1000000 + 50000) # Increased limit
        max_states_explored = min(300000, island_count * 500000 + 30000) # Limit explored states
        count = 0
        log_interval = 10000

        while open_set:
            count += 1
            if count > max_iter:
                print(f"Error: Exceeded maximum iterations ({max_iter})")
                break
            if len(closed_set) > max_states_explored:
                 print(f"Error: Exceeded maximum explored states ({max_states_explored})")
                 break

            # Progress report
            if count % log_interval == 0:
                elapsed = time.time() - start_time
                best_f = open_set[0].f_score if open_set else -1
                print(f"Iter: {count}, Explored: {len(closed_set)}, Open: {len(open_set)}, Best F: {best_f:.2f}, Time: {elapsed:.2f}s")

            current_state = heapq.heappop(open_set)

            # Goal Check
            if current_state.is_goal():
                elapsed = time.time() - start_time
                print(f"\nSolution Found! Cost: {current_state.cost}, Iterations: {count}, States Explored: {len(closed_set)}, Time: {elapsed:.2f}s")
                return current_state.grid
                
            

            # Closed Set Check (using hash)
            state_hash = hash(current_state)
            if state_hash in closed_set:
                continue
            closed_set.add(state_hash)

            # --- Successor Generation ---
            prioritized_islands = current_state.prioritize_islands()

            # If no islands can be prioritized (e.g., all remaining > 0 but impossible), stop early?
            # The heuristic should return inf for such states, preventing them from being explored.

            for island_pos, remaining in prioritized_islands:
                 # Double check if island still needs bridges (can change if processed differently)
                if current_state.remaining_bridges[island_pos] <= 0:
                     continue

                # Find potential neighbors for this island
                for other_pos in current_state.islands:
                    # Basic checks: not self, neighbor needs bridges, aligned
                    if (island_pos == other_pos or
                        current_state.remaining_bridges[other_pos] <= 0 or
                        (island_pos.x != other_pos.x and island_pos.y != other_pos.y)):
                        continue

                    # Check max bridge count constraint (max 2)
                    current_num = current_state.get_current_bridge_count(island_pos, other_pos)
                    if current_num >= 2:
                        continue # Cannot add more bridges here

                    # --- Try adding ONE bridge ---
                    # A* explores by adding one bridge at a time. The state tracks the total.
                    bridges_to_add = 1
                    next_total_num = current_num + bridges_to_add # Total bridges between them in the *next* state

                    # Check if path is clear (dynamic check required for crossing existing bridges)
                    if not can_place_bridge(current_state.grid, current_state.islands, island_pos, other_pos):
                         continue # Path blocked by an existing bridge or island

                    # --- If path clear, create the successor state ---
                    new_grid = place_bridge_on_grid(current_state.grid, island_pos, other_pos, next_total_num) # Place final thickness

                    new_remaining = current_state.remaining_bridges.copy()
                    new_remaining[island_pos] -= bridges_to_add
                    new_remaining[other_pos] -= bridges_to_add

                    # Check if counts went negative (should be prevented by initial checks, but safeguard)
                    if new_remaining[island_pos] < 0 or new_remaining[other_pos] < 0:
                         continue # Invalid move resulted in negative remaining count

                    new_bridges_set = current_state.bridges_placed.copy()
                    new_canon_bridge = _canonical_bridge(island_pos, other_pos, next_total_num)

                    # Manage bridge set: remove old if upgrading (current_num=1), add new
                    if current_num == 1: # Upgrading 1 -> 2
                        old_bridge = _canonical_bridge(island_pos, other_pos, 1)
                        new_bridges_set.discard(old_bridge) # Remove the single bridge entry
                    # Add the new entry (single or double)
                    new_bridges_set.add(new_canon_bridge)

                    # Create and evaluate the new state
                    new_state = PuzzleState(
                        grid=new_grid,
                        islands=current_state.islands,
                        remaining_bridges=new_remaining,
                        bridges_placed=new_bridges_set,
                        cost=current_state.cost + 1 # Increment cost per step/bridge addition
                    )

                    # Check heuristic for immediate pruning
                    if new_state.h_score == float('inf'):
                         continue # Don't add impossible states to open set

                    # Add to open set if not already visited (check hash)
                    new_state_hash = hash(new_state)
                    if new_state_hash not in closed_set:
                        # Check if a state with the same bridge configuration but potentially higher cost
                        # is already in the open set. A* handles this implicitly with the priority queue.
                        heapq.heappush(open_set, new_state)
                    # else: # Debugging
                    #     print(f"State {new_state_hash} already in closed set.")


        print(f"No solution found after {count} iterations and exploring {len(closed_set)} states.")
        return None

    except Exception as e:
        print(f"\n--- An error occurred during solving ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
        return None
    finally:
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")
        

__all__ = ["Solve"]

