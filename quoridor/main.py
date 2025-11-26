import argparse
import requests
import json
import time
import random
from typing import List, Optional, Tuple, Any, Dict
from copy import deepcopy
from collections import deque

from quoridor.game_client import GameClient
from quoridor.quoridor_game import QuoridorGame

# Wall generation tuning: options are 'path_focus', 'proximity', 'exhaustive'
WALL_STRATEGY = "path_focus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quoridor tests or full game.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full game flow instead of the partial/local test.",
    )
    parser.add_argument(
        "--selfplay",
        type=int,
        default=0,
        help="Run N local self-play games (my_agent vs my_agent).",
    )
    parser.add_argument(
        "--play-agent",
        choices=["1", "2"],
        nargs="?",
        const="1",
        help="Play locally vs the agent; choose your player number (default: 1).",
    )
    return parser.parse_args()


def play_game(
    solver,
    base_url: str,
    token: str,
    game_type: str,
    game_class,
    multiplayer: bool = False,
    match_id: Optional[str] = None,
    num_games: int = 1,
    debug: bool = False,
    verbose: bool = True
) -> Tuple:
    client = GameClient(base_url, token, debug=debug)
    session_turn_times: List[float] = []

    if match_id is None:
        if verbose:
            print(f"🎮 Creating new match: {num_games} x {game_type}")
        match_id = client.create_match(game_type, num_games, multiplayer)
        if verbose:
            print(f"   Match ID: {match_id}")

    if verbose:
        print(f"🔗 Joining match {match_id}...")
    match = client.join_match(match_id)
    player = match['player']
    num_games = match.get('num-games', num_games)
    if verbose:
        print(f"   You are player: {player}")

    game_state = client.get_game_state(match_id, 0)
    if game_state['status'] == 'waiting':
        if verbose:
            print("⏳ Waiting for opponent to join...")
        while game_state['status'] == 'waiting':
            time.sleep(2)
            game_state = client.get_game_state(match_id, 0)

    all_results = []
    wins = 0
    losses = 0
    draws = 0

    while True:
        match_state = client.get_match_state(match_id)
        if match_state['status'] != 'in_progress':
            break
        game_num = match_state['current-game-index']

        if verbose:
            print(f"\n{'='*50}")
            print(f"🎮 GAME {game_num + 1}/{num_games}")
            print(f"{'='*50}\n")

        # Get initial game state and check player assignment
        game_state = client.get_game_state(match_id, game_num)
        
        # Update player sign if it changed (randomized per game)
        if 'my-player' in game_state and game_state['my-player']:
            new_player = game_state['my-player']
            if new_player != player and verbose and game_num > 0:
                print(f"ℹ️  Player assignment changed: You are now Player {new_player}\n")
            player = new_player
        
        game = game_class(game_state['state'], game_state['status'], game_state['player'], player)

        game_turn_times: List[float] = []
        move_count = 0
        while game_state['status'] != 'complete':
            game_state = client.get_game_state(match_id, game_num)
            player = game_state['my-player']
            if 'winner' in game_state:
                break

            game = game_class(game_state['state'], game_state['status'], game_state['player'], player)
            
            if game.is_terminal():
                break

            if verbose:
                game.print_board()

            if game.current_player == player:
                if verbose:
                    print(f"🤔 Your turn (Player {player})...")

                try:
                    start_time = time.perf_counter()
                    move = solver(game)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    game_turn_times.append(elapsed_ms)
                    session_turn_times.append(elapsed_ms)

                    if verbose:
                        if hasattr(move, '__iter__') and not isinstance(move, str):
                            if len(move) > 0 and move[0] == 'M':
                                print(f"   Moving pawn to ({move[1]}, {move[2]})")
                            elif len(move) > 0 and move[0] == 'W':
                                print(f"   Placing {move[3]} wall at ({move[1]}, {move[2]})")
                            else:
                                print(f"   Move: {move}")
                        else:
                            print(f"   Move: {move}")
                        print(f"⏱️ Solver time: {elapsed_ms:.1f} ms")

                    client.make_move(match_id, player, move)
                    move_count += 1

                except Exception as e:
                    print(f"❌ Error in solver: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append(('error', None))
                    break
            else:
                if verbose:
                    print(f"⏳ Waiting for opponent (Player {game.current_player})...")
                time.sleep(2)

        # game is already terminal from the loop above
        if verbose:
            game.print_board()
            print("=" * 40)
            if game_turn_times:
                avg_ms = sum(game_turn_times) / len(game_turn_times)
                print(f"⏱️ Turn timing this game: avg {avg_ms:.1f} ms, max {max(game_turn_times):.1f} ms over {len(game_turn_times)} turns")

        winner = game_state['winner']
        if winner == '-':
            if verbose:
                print("🤝 Game ended in a DRAW!")
            result = 'draw'
            draws += 1
        elif winner == player:
            if verbose:
                print("🎉 You WON! Congratulations!")
            result = 'win'
            wins += 1
        else:
            if verbose:
                print("😞 You LOST. Better luck next time!")
            result = 'loss'
            losses += 1

        all_results.append((result, player, winner))

        if verbose and num_games > 1:
            print(f"\n📊 Current Record: {wins}W - {losses}L - {draws}D")
            print(f"   Games Remaining: {num_games - game_num - 1}\n")

    if session_turn_times and verbose:
        avg_ms = sum(session_turn_times) / len(session_turn_times)
        print(f"⏱️ Overall solver timing: avg {avg_ms:.1f} ms, max {max(session_turn_times):.1f} ms across {len(session_turn_times)} turns")

    # Return results
    stats = {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': wins / num_games if num_games > 0 else 0,
        'player': player,
        'match_id': match_id
    }
    if session_turn_times:
        stats['avg_turn_ms'] = sum(session_turn_times) / len(session_turn_times)
        stats['max_turn_ms'] = max(session_turn_times)

    return stats, all_results

def manual_player_solver(game: QuoridorGame) -> List:

    """
    Interactive manual player - YOU choose the moves!
    Perfect for testing the game and understanding the rules.
    """
    game.print_board()
    
    pawn_moves = game.get_valid_pawn_moves()
    wall_moves = game.get_valid_wall_moves(limit=30)
    
    print(f"\n🎮 YOUR TURN (Player {game.my_player})!")
    print("\nValid pawn moves:")
    for i, move in enumerate(pawn_moves):
        print(f"  {i}: Move to ({move[1]}, {move[2]})")
    
    if wall_moves and game.get_remaining_walls() > 0:
        print(f"\nOr place a wall (you have {game.get_remaining_walls()} left):")
        print("  Enter: W row col H/V (e.g., 'W 3 4 H')")
    
    while True:
        try:
            choice = input("\nEnter your choice (number for pawn move, or 'W r c H/V' for wall): ").strip()
            
            if choice.lower() == 'q':
                raise KeyboardInterrupt()
            
            if choice.upper().startswith('W'):
                parts = choice.split()
                if len(parts) == 4:
                    move = ['W', int(parts[1]), int(parts[2]), parts[3].upper()]
                    return move
                else:
                    print("❌ Invalid format! Use: W row col H/V")
            else:
                idx = int(choice)
                if 0 <= idx < len(pawn_moves):
                    return pawn_moves[idx]
                else:
                    print(f"❌ Invalid index! Choose 0-{len(pawn_moves)-1}")
        
        except ValueError:
            print("❌ Invalid input! Enter a number or 'W row col H/V'")
        except KeyboardInterrupt:
            print("\n👋 Thanks for playing!")
            raise

def my_agent(game: QuoridorGame) -> List:
    """
    Your AI implementation.
    
    Args:
        game: QuoridorGame object with helper methods
    
    Returns:
        List: Move in format ['M', row, col] or ['W', row, col, 'H'/'V']
    """
    
    # Basic identifiers
    my_player = game.my_player
    opponent = game.get_opponent(my_player)
    root_state = game.state

    # Search depth: keep shallow when walls are abundant, look a bit deeper in simplified endgames
    search_depth = 2
    total_walls = root_state['remaining_walls']['1'] + root_state['remaining_walls']['2']
    if total_walls <= 4:
        search_depth = 3

    def opponent_of(player: str) -> str:
        return '2' if player == '1' else '1'

    def apply_move(state: Dict, player: str, move: List) -> Dict:
        """Lightweight move application on a state dictionary."""
        new_state = deepcopy(state)
        if move[0] == 'M':
            new_state['pawns'][player] = [move[1], move[2]]
        else:
            new_state['walls'].append([move[1], move[2], move[3]])
            new_state['remaining_walls'][player] -= 1
        return new_state

    def evaluate_state(state: Dict) -> float:
        """Heuristic evaluation using shortest path lengths and wall advantage."""
        my_dist = game.shortest_path_length(my_player, state)
        opp_dist = game.shortest_path_length(opponent, state)

        if my_dist == 0:
            return 1e6
        if opp_dist == 0:
            return -1e6

        wall_diff = state['remaining_walls'][my_player] - state['remaining_walls'][opponent]
        wall_penalty = -2 if state['remaining_walls'][my_player] == 0 and state['remaining_walls'][opponent] > 0 else 0

        # Favor being ahead in the race (distance difference), keep some value on spare walls
        return (opp_dist - my_dist) * 12 + wall_diff * 1.5 + wall_penalty

    def should_sprint(state: Dict, player: str) -> bool:
        """
        Decide if we should ignore walls and just run (e.g., opponent has no walls,
        or we are ahead enough with few walls left).
        """
        my_walls = state['remaining_walls'][player]
        opp_walls = state['remaining_walls'][opponent_of(player)]
        if my_walls == 0 and opp_walls > 0:
            return True
        if opp_walls == 0:
            return True

        my_dist = game.shortest_path_length(player, state)
        opp_dist = game.shortest_path_length(opponent_of(player), state)

        if my_dist + 1 < opp_dist and opp_walls <= 1:
            return True
        return False

    def candidate_wall_moves(state: Dict, player: str) -> List[Tuple[float, List]]:
        """
        Score wall moves by how much they hurt the opponent without over-harming us.
        Returns a list of (score, move) pairs.
        """
        if state['remaining_walls'][player] == 0:
            return []

        temp_game = QuoridorGame(json.dumps(state), 'playing', player, player)
        raw_walls = temp_game.get_valid_wall_moves(player, limit=None, strategy=WALL_STRATEGY)
        if not raw_walls:
            return []

        opp = opponent_of(player)
        my_pos = state['pawns'][player]
        opp_pos = state['pawns'][opp]
        existing = {(w[0], w[1], w[2]) for w in state['walls']}
        base_my = game.shortest_path_length(player, state)
        base_opp = game.shortest_path_length(opp, state)

        scored = []
        for move in raw_walls:
            _, r, c, orient = move
            near_opp = abs(r - opp_pos[0]) + abs(c - opp_pos[1]) <= 3
            near_me = abs(r - my_pos[0]) + abs(c - my_pos[1]) <= 3

            # Encourage extending existing barriers
            extends = False
            if orient == 'H':
                extends = ((r, c - 1, 'H') in existing or (r, c + 1, 'H') in existing)
            else:
                extends = ((r - 1, c, 'V') in existing or (r + 1, c, 'V') in existing)

            # Opening guideline: avoid random walls very early
            if len(state['walls']) <= 1 and state['remaining_walls'][player] >= 8:
                if not (near_opp or extends):
                    continue

            if not (near_opp or near_me or extends):
                continue

            next_state = apply_move(state, player, move)
            new_opp = game.shortest_path_length(opp, next_state)
            new_my = game.shortest_path_length(player, next_state)

            opp_gain = new_opp - base_opp
            my_cost = max(0, new_my - base_my)

            # Demand the wall to add real tax to the opponent and not hurt us too much
            if opp_gain < 1:
                continue

            score = opp_gain - 0.8 * my_cost
            if near_opp:
                score += 0.4
            if extends:
                score += 0.25

            # If we are already well ahead, only keep walls that add a big detour
            if (base_opp - base_my) >= 2 and opp_gain < 2:
                continue

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:8]  # keep top candidates to control branching

    def generate_moves(state: Dict, player: str) -> List[List]:
        """Generate ordered moves for alpha-beta search."""
        temp_game = QuoridorGame(json.dumps(state), 'playing', player, player)
        pawn_moves = temp_game.get_valid_pawn_moves(player)

        # Order pawn moves by how much they shorten our path
        base_dist = game.shortest_path_length(player, state)
        pawn_scored = []
        for mv in pawn_moves:
            next_state = apply_move(state, player, mv)
            new_dist = game.shortest_path_length(player, next_state)
            pawn_scored.append((base_dist - new_dist, mv))
        pawn_scored.sort(key=lambda x: x[0], reverse=True)

        # Toggle race mode when walls are ineffective
        if should_sprint(state, player):
            return [mv for _, mv in pawn_scored]

        wall_scored = candidate_wall_moves(state, player)
        ordered = [mv for _, mv in pawn_scored] + [mv for _, mv in wall_scored]
        return ordered

    def minimax(state: Dict, player: str, depth: int, alpha: float, beta: float) -> float:
        my_turn = player == my_player
        my_dist = game.shortest_path_length(my_player, state)
        opp_dist = game.shortest_path_length(opponent, state)

        # Quick terminal checks
        if my_dist == 0:
            return 1e6
        if opp_dist == 0:
            return -1e6
        if depth == 0:
            return evaluate_state(state)

        moves = generate_moves(state, player)
        if not moves:
            return evaluate_state(state)

        if my_turn:
            value = -float('inf')
            for mv in moves:
                child_state = apply_move(state, player, mv)
                value = max(value, minimax(child_state, opponent_of(player), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for mv in moves:
                child_state = apply_move(state, player, mv)
                value = min(value, minimax(child_state, opponent_of(player), depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # Root search
    best_score = -float('inf')
    best_move = None
    root_moves = generate_moves(root_state, my_player)

    # If we somehow have no generated moves, fall back to any legal move
    if not root_moves:
        fallback = game.get_valid_pawn_moves() + game.get_valid_wall_moves(limit=10, strategy=WALL_STRATEGY)
        return random.choice(fallback)

    for move in root_moves:
        next_state = apply_move(root_state, my_player, move)
        score = minimax(next_state, opponent, search_depth - 1, -float('inf'), float('inf'))
        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move is not None else random.choice(root_moves)

def test_partial(game_state: Optional[Dict] = None) -> Dict:
    """
    Run a single local move and return the updated game state for reuse.
    """
    if game_state is None:
        # Initialize a test state
        game_state = {
            'pawns': {'1': [2, 4], '2': [6, 4]},
            'walls': [[3, 3, 'H'], [4, 5, 'V']],
            'remaining_walls': {'1': 8, '2': 9},
            'current_player': '1',
        }

    current_player = game_state.get('current_player', '1')
    state_payload = {
        'pawns': game_state['pawns'],
        'walls': game_state['walls'],
        'remaining_walls': game_state['remaining_walls'],
    }

    test_game = QuoridorGame(json.dumps(state_payload), 'playing', current_player, current_player)

    print("Test board:")
    test_game.print_board()

    print(f"Valid pawn moves: {test_game.get_valid_pawn_moves()}")
    print(f"My path length: {test_game.shortest_path_length('1')}")
    print(f"Opponent path length: {test_game.shortest_path_length('2')}")

    start_time = time.perf_counter()
    move = my_agent(test_game)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"\nYour solver chose: {move} (took {elapsed_ms:.1f} ms)")
    
    # Apply the move locally and prepare the next state
    updated_state = test_game.simulate_move(move)
    next_player = test_game.get_opponent(current_player)
    next_game_state = {
        'pawns': updated_state['pawns'],
        'walls': updated_state['walls'],
        'remaining_walls': updated_state['remaining_walls'],
        'current_player': next_player,
    }

    # Show the board after the move
    updated_game = QuoridorGame(json.dumps(updated_state), 'playing', next_player, next_player)
    print("\nBoard after move:")
    updated_game.print_board()

    return next_game_state

def self_play(num_games: int = 1, max_moves: int = 200, verbose: bool = True) -> List[Tuple[str, int]]:
    """
    Run my_agent vs itself locally (no server). Useful for quick behavioral smoke tests.
    """
    results = []

    for game_idx in range(num_games):
        state = {
            'pawns': {'1': [0, 4], '2': [8, 4]},
            'walls': [],
            'remaining_walls': {'1': 10, '2': 10},
            'current_player': '1',
        }
        winner = None

        if verbose:
            print(f"\n=== SELF-PLAY GAME {game_idx + 1}/{num_games} ===")
        turn_times: List[float] = []

        for ply in range(max_moves):
            current_player = state['current_player']

            state_payload = {
                'pawns': state['pawns'],
                'walls': state['walls'],
                'remaining_walls': state['remaining_walls'],
            }
            game_obj = QuoridorGame(json.dumps(state_payload), 'playing', current_player, current_player)

            start_time = time.perf_counter()
            move = my_agent(game_obj)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            turn_times.append(elapsed_ms)
            next_state = game_obj.simulate_move(move)

            # Check for goal reach
            if current_player == '1' and next_state['pawns']['1'][0] == 8:
                winner = '1'
            elif current_player == '2' and next_state['pawns']['2'][0] == 0:
                winner = '2'

            if verbose:
                print(f"Player {current_player} plays {move} ({elapsed_ms:.1f} ms)")

            if winner:
                if verbose:
                    end_game = QuoridorGame(json.dumps(next_state), 'complete', current_player, current_player)
                    end_game.print_board()
                    print(f"Winner: Player {winner}\n")
                break

            # Switch player and continue
            state = next_state
            state['current_player'] = '2' if current_player == '1' else '1'

        if not winner:
            winner = '-'
            if verbose:
                print("Reached max moves; declaring draw.\n")

        if turn_times and verbose:
            avg_ms = sum(turn_times) / len(turn_times)
            print(f"⏱️ Timing: avg {avg_ms:.1f} ms, max {max(turn_times):.1f} ms over {len(turn_times)} turns")

        results.append((winner, game_idx))

    return results

def human_vs_agent(human_player: str = "1", max_moves: int = 200, verbose: bool = True):
    """
    Play locally against my_agent. human_player is "1" or "2".
    """
    agent_player = '2' if human_player == '1' else '1'
    state = {
        'pawns': {'1': [0, 4], '2': [8, 4]},
        'walls': [],
        'remaining_walls': {'1': 10, '2': 10},
        'current_player': '1',
    }
    winner = None
    agent_turn_times: List[float] = []

    if verbose:
        print(f"\n=== HUMAN (Player {human_player}) vs AGENT (Player {agent_player}) ===")

    for ply in range(max_moves):
        current_player = state['current_player']

        state_payload = {
            'pawns': state['pawns'],
            'walls': state['walls'],
            'remaining_walls': state['remaining_walls'],
        }

        game_obj = QuoridorGame(json.dumps(state_payload), 'playing', current_player, current_player)
        if current_player == human_player:
            move = manual_player_solver(game_obj)
            elapsed_ms = None
        else:
            start_time = time.perf_counter()
            move = my_agent(game_obj)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            agent_turn_times.append(elapsed_ms)

        next_state = game_obj.simulate_move(move)

        # Check for win
        if current_player == '1' and next_state['pawns']['1'][0] == 8:
            winner = '1'
        elif current_player == '2' and next_state['pawns']['2'][0] == 0:
            winner = '2'

        if verbose:
            if elapsed_ms is not None:
                print(f"Player {current_player} plays {move} ({elapsed_ms:.1f} ms)")
            else:
                print(f"Player {current_player} plays {move}")

        if winner:
            if verbose:
                end_game = QuoridorGame(json.dumps(next_state), 'complete', current_player, human_player)
                end_game.print_board()
                print(f"Winner: Player {winner}\n")
            break

        # Switch turn
        state = next_state
        state['current_player'] = '2' if current_player == '1' else '1'

    if not winner:
        print("Reached max moves; draw.")

    if agent_turn_times and verbose:
        avg_ms = sum(agent_turn_times) / len(agent_turn_times)
        print(f"⏱️ Agent timing: avg {avg_ms:.1f} ms, max {max(agent_turn_times):.1f} ms over {len(agent_turn_times)} turns")

def test_full():
    STUDENT_TOKEN = 'BORIS-GANS'  # e.g., 'JOHN-DOE'
    SOLVER = my_agent
    MULTIPLAYER = False
    MATCH_ID = None
    NUM_GAMES = 1
    BASE_URL = 'https://ie-aireasoning-gr4r5bl6tq-ew.a.run.app'  # Your Cloud Run URL

    result = play_game(
        solver=SOLVER,
        base_url=BASE_URL,
        token=STUDENT_TOKEN,
        game_type='quoridor',
        game_class=QuoridorGame,
        multiplayer=MULTIPLAYER,
        match_id=MATCH_ID,
        num_games=NUM_GAMES,
        debug=False,
        verbose=True
    )

    stats, all_results = result
    print("\n📊 Summary:")
    print(f"   Record: {stats['wins']}W - {stats['losses']}L - {stats['draws']}D")
    print(f"   Win Rate: {stats['win_rate']*100:.1f}%")

if __name__ == "__main__":
    args = parse_args()
    if args.play_agent:
        human_vs_agent(human_player=args.play_agent, verbose=True)
    elif args.selfplay > 0:
        self_play(num_games=args.selfplay, verbose=True)
    elif args.full:
        test_full()
    else:
        state = None
        while True:
            state = test_partial(state)

            while True:
                user_input = input("Press Enter for next move or type 'stop' to exit: ").strip().lower()
                if user_input == "":
                    break  # Continue to the next move
                if user_input == "stop":
                    print("Stopping partial test.")
                    raise SystemExit
                print("Please press Enter to continue or type 'stop' to end.")
