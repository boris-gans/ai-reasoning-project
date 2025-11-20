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


BASE_URL = 'https://ie-aireasoning-gr4r5bl6tq-ew.a.run.app'  # Your Cloud Run URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quoridor tests or full game.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full game flow instead of the partial/local test.",
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
                    move = solver(game)

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
    
    # Get basic info
    my_player = game.my_player
    my_pos = game.get_my_position()
    opp_pos = game.get_opponent_position()
    
    # ============================================================
    # TODO: IMPLEMENT YOUR ALGORITHM HERE!
    # ============================================================
    
    # Example: Random valid move (replace with your algorithm!)
    moves = game.get_valid_pawn_moves() + game.get_valid_wall_moves(limit=10)
        
    # Just pick a random pawn move
    return random.choice(moves)
    
    # ============================================================
    # Ideas to implement:
    # 1. Use shortest_path_length() to evaluate positions
    # 2. Alpha-beta minimax with evaluation heuristic (e.g. my_path_length - opponent_path_length)
    # 4. Strategic wall placement to maximize opponent's path
    # ============================================================

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

    move = my_agent(test_game)
    print(f"\nYour solver chose: {move}")

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

def test_full():
    STUDENT_TOKEN = 'BORIS-GANS'  # e.g., 'JOHN-DOE'
    SOLVER = my_agent
    MULTIPLAYER = False
    MATCH_ID = None
    NUM_GAMES = 1

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
    if args.full:
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
