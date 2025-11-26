import requests
import json
import time
import random
from typing import List, Optional, Tuple, Any, Dict
from copy import deepcopy
from collections import deque

class GameClient:
    def __init__(self, base_url: str, token: str, debug: bool = False):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.debug = debug

    def _make_request(self, endpoint: str, params: dict, max_retries: int = 10) -> dict:
        params['TOKEN'] = self.token
        url = f'{self.base_url}{endpoint}'

        for attempt in range(max_retries):
            try:
                if self.debug:
                    print(f"[DEBUG] Request: {endpoint}")
                    print(f"[DEBUG] Params: {params}")

                response = requests.get(url, params=params, timeout=30)

                if self.debug:
                    print(f"[DEBUG] Response [{response.status_code}]: {response.text[:200]}")

                if response.status_code == 200:
                    if response.text:
                        try:
                            return response.json()
                        except (json.JSONDecodeError, ValueError) as e:
                            if self.debug:
                                print(f"[DEBUG] Non-JSON response: {response.text[:100]}")
                            return {}
                    return {}
                else:
                    print(f"⚠️  HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"⚠️  Request timeout (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Request error: {e} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"⚠️  Unexpected error: {type(e).__name__}: {e} (attempt {attempt + 1}/{max_retries})")

            if attempt < max_retries - 1:
                time.sleep(1)

        raise Exception(f"Failed to connect to {endpoint} after {max_retries} attempts")

    def create_match(self, game_type: str, num_games: int, multiplayer: bool = False) -> str:
        response = self._make_request('/new-match', {
            'game-type': game_type,
            'num-games': str(num_games),
            'multi-player': 'True' if multiplayer else 'False'
        })
        
        if 'match-id' not in response:
            print(f"❌ Server response missing 'match-id'. Response: {response}")
            raise KeyError(f"Server response missing 'match-id'. Got: {response}")
        
        return response['match-id']

    def join_match(self, match_id: str) -> dict:
        response = self._make_request('/join-match', {
            'match-id': match_id
        })
        return response

    def get_game_state(self, match_id: str, game_index: int) -> dict:
        return self._make_request('/game-state-in-match', {
            'match-id': match_id,
            'game-index': str(game_index)
        })

    def get_match_state(self, match_id: str) -> dict:
        return self._make_request('/match-state', {
            'match-id': match_id
        })

    def make_move(self, match_id: str, player: str, move: Any) -> bool:
        move_str = move if isinstance(move, str) else json.dumps(move)
        
        self._make_request('/make-move-in-match', {
            'match-id': match_id,
            'player': player,
            'move': move_str
        })
        return True
