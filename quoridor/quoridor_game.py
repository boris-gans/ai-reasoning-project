import json
import random
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

    def get_valid_wall_moves(self, player: Optional[str] = None, limit: int = None) -> List[List]:
        """
        Get valid wall placements for player.
        
        Args:
            player: Player to get walls for (defaults to current player)
            limit: Maximum number of wall moves to return (for efficiency)
        """
        if player is None:
            player = self.current_player

        if self.get_remaining_walls(player) == 0:
            return []

        # NOTE: Checking all wall placements is expensive!
        # For performance, you may want to limit the number checked
        # or implement strategic wall placement heuristics
        
        moves = []
        walls = self.get_walls()

        # This is computationally expensive - consider limiting in your implementation
        # Try horizontal walls (simplified - only checks some positions)
        if limit:
            # Sample some random positions instead of checking all
            positions = [(r, c) for r in range(9) for c in range(8)]
            random.shuffle(positions)
            positions = positions[:limit]
            
            for r, c in positions:
                if self._is_valid_wall_simple([r, c, 'H'], walls):
                    moves.append(['W', r, c, 'H'])
            
            positions = [(r, c) for r in range(8) for c in range(9)]
            random.shuffle(positions)
            positions = positions[:limit]
            
            for r, c in positions:
                if self._is_valid_wall_simple([r, c, 'V'], walls):
                    moves.append(['W', r, c, 'V'])
        else:
            # Check all positions (slow!)
            for r in range(9):
                for c in range(8):
                    if self._is_valid_wall_simple([r, c, 'H'], walls):
                        moves.append(['W', r, c, 'H'])

            for r in range(8):
                for c in range(9):
                    if self._is_valid_wall_simple([r, c, 'V'], walls):
                        moves.append(['W', r, c, 'V'])

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