# AI Reasoning Project

## Quoridor
- **Goal:** Race your pawn to the far side while using walls to tax the opponent’s path without blocking all routes.
- **Core idea:** Shallow alpha-beta (depth 2, depth 3 late) with strong move ordering. Evaluation is dominated by shortest-path race (BFS), with wall inventory as a secondary asset and sprint mode when walls no longer matter.
- **Wall logic:** Two-stage pruning: the game class proposes walls near opponent paths/barriers and rejects illegal blocks; the agent then keeps only walls that add real detours while avoiding self-harm and trimming to the top few candidates.
- **Behavior:** Runs when ahead or when walls are ineffective; otherwise balances path-shortening pawn moves with targeted walls that extend barriers or sit on the opponent’s route.

## Ghosts
- **Goal:** Exit a good ghost or capture all opposing good ghosts while protecting your own.
- **Setup:** Aggressive placement—good ghosts on the flanks to sprint for exits; evil ghosts in the center as bait.
- **Evaluation:** Material-aware with big bonuses/penalties for wiping or losing good ghosts, distance-to-exit terms for both sides, and a threat penalty when good ghosts are adjacent to foes.
- **Search:** Heuristic scoring with shallow lookahead (1–2 ply); critical moves involving good ghosts search slightly deeper. Immediate winning exits are taken before any search.
- **Behavior:** Pushes safe exits, hunts opponent good ghosts when opportunities appear, and avoids exposing its own good pieces to clustered enemies.

## UNO
- **Goal:** Shed cards fastest while disrupting opponents, especially those close to going out.
- **Core idea:** Hybrid of card-counting, opponent hand-size awareness, and color/hand-shaping heuristics. An aggression factor scales plays based on whether the agent is behind or ahead.
- **Scoring highlights:** Penalizes colors likely held by low-card opponents, prefers skips/reverses/draws when someone is short, shapes the hand toward fewer colors, and picks wild colors via a discard/hand mix plus edge-case logic when about to call UNO.
- **Behavior:** Plays more punitive specials when opponents are low, conserves power cards when ahead early, and accelerates with wilds and high-value numbers in late-game races.