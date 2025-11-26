import json
from typing import List, Optional, Dict
from copy import deepcopy
from collections import deque


class QuoridorGame:
    """
    Represents Quoridor game state with helper methods.
    
    Key methods for your solver:
    - game.get_my_position()           # Your pawn position [row, col]
    - game.get_opponent_position()     # Opponent pawn position [row, col]
    - game.get_walls()                 # List of walls [[r, c, 'H'/'V'], ...]
    - game.get_remaining_walls()       # Your remaining wall count
    - game.get_valid_moves()           # All valid moves
    - game.get_valid_pawn_moves()      # Only pawn moves
    - game.get_valid_wall_moves()      # Only wall placements
    - game.simulate_move(move)         # Simulate move for search
    - game.is_terminal()               # Check if game over
    - game.print_board()               # Debug visualization
    """

    def __init__(self, state: str, status: str, current_player: str, my_player: str):
        self.state_str = state
        self.status = status
        self.current_player = current_player
        self.my_player = my_player
        self._state = None
        self._valid_moves = None

    @property
    def state(self) -> Dict:
        """Get state dictionary."""
        if self._state is None:
            self._state = json.loads(self.state_str)
        return self._state

    def get_my_position(self) -> List[int]:
        """Get your pawn position [row, col]."""
        return self.state['pawns'][self.my_player]

    def get_opponent_position(self) -> List[int]:
        """Get opponent pawn position [row, col]."""
        opp = '2' if self.my_player == '1' else '1'
        return self.state['pawns'][opp]

    def get_walls(self) -> List[List]:
        """Get list of walls [[row, col, 'H'/'V'], ...]."""
        return self.state['walls']

    def get_remaining_walls(self, player: Optional[str] = None) -> int:
        """Get remaining wall count for player (defaults to you)."""
        if player is None:
            player = self.my_player
        return self.state['remaining_walls'][player]

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.status == 'complete'

    def is_waiting(self) -> bool:
        """Check if waiting for opponent."""
        return self.status == 'waiting'

    def get_winner(self) -> Optional[str]:
        """Get winner ('1', '2', None if ongoing)."""
        if not self.is_terminal():
            return None
        return self.current_player

    def get_opponent(self, player: str) -> str:
        """Get opponent's identifier."""
        return '2' if player == '1' else '1'

    def _is_wall_blocking(self, walls, from_pos, to_pos) -> bool:
        """Check if a wall blocks movement."""
        r1, c1 = from_pos
        r2, c2 = to_pos

        def has_wall(r, c, orient):
            return [r, c, orient] in walls

        # Moving up
        if r2 < r1 and c1 == c2:
            return has_wall(r2, c1, 'H') or has_wall(r2, c1 - 1, 'H')
        # Moving down
        elif r2 > r1 and c1 == c2:
            return has_wall(r1, c1, 'H') or has_wall(r1, c1 - 1, 'H')
        # Moving left
        elif c2 < c1 and r1 == r2:
            return has_wall(r1, c2, 'V') or has_wall(r1 - 1, c2, 'V')
        # Moving right
        elif c2 > c1 and r1 == r2:
            return has_wall(r1, c1, 'V') or has_wall(r1 - 1, c1, 'V')

        return False

    def _build_wall_lookup(self, walls) -> set:
        """Fast lookup set for wall membership checks."""
        return {(w[0], w[1], w[2]) for w in walls}

    def _is_wall_blocking_lookup(self, wall_lookup: set, from_pos, to_pos) -> bool:
        """Set-based variant of wall blocking check used by fast BFS routines."""
        r1, c1 = from_pos
        r2, c2 = to_pos

        def has_wall(r, c, orient):
            return (r, c, orient) in wall_lookup

        if r2 < r1 and c1 == c2:
            return has_wall(r2, c1, 'H') or has_wall(r2, c1 - 1, 'H')
        elif r2 > r1 and c1 == c2:
            return has_wall(r1, c1, 'H') or has_wall(r1, c1 - 1, 'H')
        elif c2 < c1 and r1 == r2:
            return has_wall(r1, c2, 'V') or has_wall(r1 - 1, c2, 'V')
        elif c2 > c1 and r1 == r2:
            return has_wall(r1, c1, 'V') or has_wall(r1 - 1, c1, 'V')

        return False

    def _neighbors_without_walls(self, pos, wall_lookup: set):
        """Yield reachable neighbors from pos given a wall lookup."""
        r, c = pos
        for new_r, new_c in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= new_r < 9 and 0 <= new_c < 9:
                if not self._is_wall_blocking_lookup(wall_lookup, (r, c), (new_r, new_c)):
                    yield (new_r, new_c)

    def _path_exists_to_goal(self, player: str, wall_lookup: set) -> bool:
        """Check reachability to the goal row with early-exit BFS."""
        start = tuple(self.state['pawns'][player])
        goal_row = 8 if player == '1' else 0
        queue = deque([start])
        visited = {start}

        while queue:
            pos = queue.popleft()
            if pos[0] == goal_row:
                return True
            for neighbor in self._neighbors_without_walls(pos, wall_lookup):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def _shortest_path_cells(self, player: str, wall_lookup: set) -> List[tuple]:
        """Return one shortest path as a list of cells for heuristic targeting."""
        start = tuple(self.state['pawns'][player])
        goal_row = 8 if player == '1' else 0

        parents = {start: None}
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos[0] == goal_row:
                path = []
                curr = pos
                while curr is not None:
                    path.append(curr)
                    curr = parents[curr]
                return list(reversed(path))

            for neighbor in self._neighbors_without_walls(pos, wall_lookup):
                if neighbor not in parents:
                    parents[neighbor] = pos
                    queue.append(neighbor)

        return []

    def _wall_anchor_distance(self, wall, cell: List[int]) -> int:
        """Manhattan distance from a wall's anchor to a board cell."""
        r, c, _ = wall
        return abs(r - cell[0]) + abs(c - cell[1])

    def _wall_near_point(self, wall, cell: List[int], radius: Optional[int]) -> bool:
        """Check if wall anchor lies within radius of a given cell."""
        if radius is None:
            return True
        return self._wall_anchor_distance(wall, cell) <= radius

    def _wall_near_path(self, wall, path_cells: List[tuple], radius: Optional[int]) -> bool:
        """Check if wall is near any cell on a path."""
        if radius is None:
            return True
        if not path_cells:
            return False
        return any(self._wall_anchor_distance(wall, cell) <= radius for cell in path_cells)

    def _wall_extends_existing(self, wall, wall_lookup: set) -> bool:
        """Light heuristic: prefer walls that extend an existing barrier."""
        r, c, orient = wall
        if orient == 'H':
            return (r, c - 1, 'H') in wall_lookup or (r, c + 1, 'H') in wall_lookup
        return (r - 1, c, 'V') in wall_lookup or (r + 1, c, 'V') in wall_lookup

    def _all_wall_positions(self):
        """Iterate over every legal wall coordinate (without validation)."""
        for r in range(9):
            for c in range(8):
                yield (r, c, 'H')
        for r in range(8):
            for c in range(9):
                yield (r, c, 'V')

    def _generate_wall_candidates(self, player: str, strategy: str, wall_lookup: set,
                                  limit: Optional[int]) -> List[List]:
        """
        Produce a pruned, ordered list of wall candidates using heuristics.

        Strategies:
            - 'path_focus' (default): favor walls near opponent path/pawn and existing barriers.
            - 'proximity': larger net; near either pawn or either shortest path.
            - 'exhaustive': consider every slot (still runs path checks later).
        """
        configs = {
            'path_focus': {'path_radius': 3, 'pawn_radius': 3, 'candidate_cap': 70},
            'proximity': {'path_radius': 4, 'pawn_radius': 4, 'candidate_cap': 110},
            'exhaustive': {'path_radius': None, 'pawn_radius': None, 'candidate_cap': None}
        }
        cfg = configs.get(strategy, configs['path_focus'])

        # Inflate candidate cap a bit when a limit is requested so we still find enough valid walls.
        candidate_cap = cfg['candidate_cap']
        if limit is not None:
            if candidate_cap is None:
                candidate_cap = limit * 3
            else:
                candidate_cap = max(candidate_cap, limit * 2)

        opponent = self.get_opponent(player)
        my_pos = self.state['pawns'][player]
        opp_pos = self.state['pawns'][opponent]

        opp_path = self._shortest_path_cells(opponent, wall_lookup)
        my_path = self._shortest_path_cells(player, wall_lookup)

        scored_candidates = []
        for wall in self._all_wall_positions():
            extends = self._wall_extends_existing(wall, wall_lookup)
            near_opp_path = self._wall_near_path(wall, opp_path, cfg['path_radius'])
            near_my_path = self._wall_near_path(wall, my_path, cfg['path_radius'])
            near_opp_pawn = self._wall_near_point(wall, opp_pos, cfg['pawn_radius'])
            near_my_pawn = self._wall_near_point(wall, my_pos, cfg['pawn_radius'])

            if strategy == 'exhaustive':
                keep = True
            elif strategy == 'proximity':
                keep = near_opp_path or near_my_path or near_opp_pawn or near_my_pawn or extends
            else:  # path_focus
                keep = near_opp_path or (near_opp_pawn and (near_opp_path or near_my_path)) or (extends and (near_opp_pawn or near_opp_path))

            if not keep:
                continue

            score = 0.0
            if near_opp_path:
                score += 3.0
            if near_opp_pawn:
                score += 1.8
            if extends:
                score += 0.8
            if near_my_pawn:
                score += 0.5
            if near_my_path:
                score += 0.2

            score -= self._wall_anchor_distance(wall, opp_pos) * 0.05
            scored_candidates.append((score, [wall[0], wall[1], wall[2]]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        if candidate_cap is not None:
            scored_candidates = scored_candidates[:candidate_cap]

        return [wall for _, wall in scored_candidates]

    def _wall_preserves_paths(self, wall, player: str, wall_lookup: set) -> bool:
        """Check that adding wall keeps at least one path for both players."""
        new_lookup = set(wall_lookup)
        new_lookup.add((wall[0], wall[1], wall[2]))
        opponent = self.get_opponent(player)

        if not self._path_exists_to_goal(opponent, new_lookup):
            return False
        return self._path_exists_to_goal(player, new_lookup)





    def get_valid_pawn_moves(self, player: Optional[str] = None) -> List[List]:
        """Get valid pawn moves for player."""
        if player is None:
            player = self.current_player

        state = self.state
        my_pos = state['pawns'][player]
        opp_pos = state['pawns'][self.get_opponent(player)]
        walls = state['walls']

        r, c = my_pos
        moves = []

        # Basic adjacent moves
        adjacent = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        for new_r, new_c in adjacent:
            if 0 <= new_r < 9 and 0 <= new_c < 9:
                if [new_r, new_c] == opp_pos:
                    # Try to jump over opponent
                    dr = new_r - r
                    dc = new_c - c
                    jump_r = new_r + dr
                    jump_c = new_c + dc

                    if 0 <= jump_r < 9 and 0 <= jump_c < 9:
                        if not self._is_wall_blocking(walls, opp_pos, [jump_r, jump_c]):
                            moves.append(['M', jump_r, jump_c])
                    else:
                        # Diagonal jumps when can't jump straight
                        if dr != 0:  # Moving vertically, try left/right
                            for side_dc in [-1, 1]:
                                side_c = new_c + side_dc
                                if 0 <= side_c < 9:
                                    if not self._is_wall_blocking(walls, opp_pos, [new_r, side_c]):
                                        moves.append(['M', new_r, side_c])
                        else:  # Moving horizontally, try up/down
                            for side_dr in [-1, 1]:
                                side_r = new_r + side_dr
                                if 0 <= side_r < 9:
                                    if not self._is_wall_blocking(walls, opp_pos, [side_r, new_c]):
                                        moves.append(['M', side_r, new_c])
                else:
                    # Normal move
                    if not self._is_wall_blocking(walls, my_pos, [new_r, new_c]):
                        moves.append(['M', new_r, new_c])

        return moves

    def get_valid_wall_moves(self, player: Optional[str] = None, limit: int = None,
                             strategy: str = "path_focus") -> List[List]:
        """
        Get valid wall placements for player using heuristics and path safety checks.
        
        Args:
            player: Player to get walls for (defaults to current player)
            limit: Maximum number of wall moves to return after validation
            strategy: Wall generation strategy ('path_focus', 'proximity', 'exhaustive')
        """
        if player is None:
            player = self.current_player

        if self.get_remaining_walls(player) == 0:
            return []

        walls = self.get_walls()
        wall_lookup = self._build_wall_lookup(walls)
        candidates = self._generate_wall_candidates(player, strategy, wall_lookup, limit)

        moves = []
        for wall in candidates:
            if not self._is_valid_wall_simple(wall, walls):
                continue
            if not self._wall_preserves_paths(wall, player, wall_lookup):
                continue
            moves.append(['W', wall[0], wall[1], wall[2]])
            if limit is not None and len(moves) >= limit:
                break

        # If a tight heuristic returns nothing, retry with exhaustive once.
        if not moves and strategy != 'exhaustive':
            return self.get_valid_wall_moves(player=player, limit=limit, strategy='exhaustive')

        return moves

    def _is_valid_wall_simple(self, wall, existing_walls) -> bool:
        """Simplified wall validation (doesn't check pathfinding)."""
        r, c, orient = wall

        # Check if wall already exists
        if wall in existing_walls:
            return False

        # Check for overlaps and crossings (simplified)
        if orient == 'H':
            if [r, c + 1, 'H'] in existing_walls or [r, c - 1, 'H'] in existing_walls:
                return False
            if [r - 1, c, 'V'] in existing_walls or [r, c, 'V'] in existing_walls:
                return False
            if [r - 1, c + 1, 'V'] in existing_walls or [r, c + 1, 'V'] in existing_walls:
                return False
        else:  # 'V'
            if [r + 1, c, 'V'] in existing_walls or [r - 1, c, 'V'] in existing_walls:
                return False
            if [r, c - 1, 'H'] in existing_walls or [r, c, 'H'] in existing_walls:
                return False
            if [r + 1, c - 1, 'H'] in existing_walls or [r + 1, c, 'H'] in existing_walls:
                return False

        # NOTE: This doesn't check if wall blocks all paths!
        # The server will reject invalid walls, but checking here is expensive
        return True

    def get_valid_moves(self, player: Optional[str] = None, limit_walls: int = 20) -> List[List]:
        """
        Get all valid moves.
        
        Args:
            player: Player to get moves for
            limit_walls: Limit wall checks for performance (set to None for all)
        """
        if self._valid_moves is None or player != self.current_player:
            pawn_moves = self.get_valid_pawn_moves(player)
            wall_moves = self.get_valid_wall_moves(player, limit=limit_walls)
            self._valid_moves = pawn_moves + wall_moves
        return self._valid_moves

    def simulate_move(self, move: List) -> Dict:
        """
        Simulate a move and return new state.
        Does NOT contact server or modify original state.
        Essential for minimax/alpha-beta, but this can be expensive because of deepcopy.
        """
        new_state = deepcopy(self.state)
        player = self.current_player

        if move[0] == 'M':
            # Pawn move
            new_state['pawns'][player] = [move[1], move[2]]
        elif move[0] == 'W':
            # Wall placement
            new_state['walls'].append([move[1], move[2], move[3]])
            new_state['remaining_walls'][player] -= 1

        return new_state

    def shortest_path_length(self, player: Optional[str] = None, state: Optional[Dict] = None) -> int:
        """
        Calculate shortest path length to goal using BFS.
        Useful for evaluation functions!
        
        Returns:
            int: Number of moves to goal, or 999 if no path
        """
        if player is None:
            player = self.my_player
        if state is None:
            state = self.state

        start = tuple(state['pawns'][player])
        goal_row = 8 if player == '1' else 0
        walls = state['walls']

        visited = {start: 0}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            dist = visited[(r, c)]

            if r == goal_row:
                return dist

            for new_r, new_c in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= new_r < 9 and 0 <= new_c < 9:
                    if (new_r, new_c) not in visited:
                        if not self._is_wall_blocking(walls, [r, c], [new_r, new_c]):
                            visited[(new_r, new_c)] = dist + 1
                            queue.append((new_r, new_c))

        return 999  # No path found

    def print_board(self):
        """Print nice board visualization."""
        state = self.state
        pawns = state['pawns']
        walls = state['walls']

        print("\n" + "=" * 37)
        print(f"Player 1: {pawns['1']}  Player 2: {pawns['2']}")
        print(f"Walls remaining - You: {self.get_remaining_walls(self.my_player)}, " +
              f"Opponent: {self.get_remaining_walls(self.get_opponent(self.my_player))}")
        print("=" * 37)

        # Create visual board
        for r in range(9):
            # Print row with cells
            row_str = ""
            for c in range(9):
                if pawns['1'] == [r, c]:
                    row_str += "1"
                elif pawns['2'] == [r, c]:
                    row_str += "2"
                else:
                    row_str += "·"

                # Check for vertical wall to the right
                if c < 8:
                    if [r, c, 'V'] in walls or (r > 0 and [r - 1, c, 'V'] in walls):
                        row_str += " ║ "
                    else:
                        row_str += "   "

            print(row_str)

            # Print horizontal walls
            if r < 8:
                wall_str = ""
                for c in range(9):
                    if [r, c, 'H'] in walls or (c > 0 and [r, c - 1, 'H'] in walls):
                        wall_str += "═"
                    else:
                        wall_str += " "

                    if c < 8:
                        wall_str += "   "

                print(wall_str)

        print("=" * 37 + "\n")
