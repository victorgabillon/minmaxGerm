# expert_game_lab

Small exact experiments for the finite-horizon prediction-with-expert-advice minimax game.

State convention:

- A state is an integer vector of expert cumulative gains.
- States are canonicalized by sorting in descending order and subtracting the minimum coordinate.
- Terminal payoff is the leading coordinate, equivalently `max(x)`.

Core Bellman step:

```
V_t(x) = min_p max_a [V_{t-1}(canon(x + a)) - p . a]
```

The one-step minimax problem is solved by `scipy.optimize.linprog(method="highs")`.

Typical workflow:

```
pytest
python -m expert_game_lab.experiments compare --k 3 --T 20
python -m expert_game_lab.experiments greedy --k 5 --T 20 --policy packet_frontier
python -m expert_game_lab.experiments commute --k 5 --T 20 --policy packet_frontier
```
