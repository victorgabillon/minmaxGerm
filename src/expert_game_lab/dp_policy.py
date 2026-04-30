from __future__ import annotations

import numpy as np

from .state import all_states, canon


def evaluate_balanced_policy(k: int, T: int, policy_fn) -> list[dict[tuple[int, ...], float]]:
    if T < 0:
        raise ValueError("T must be nonnegative")

    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = []

    terminal_states = state_layers[T]
    values.append({state: float(max(state, default=0)) for state in terminal_states})

    for horizon in range(1, T + 1):
        prev = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        domain = state_layers[T - horizon]
        for state in domain:
            policy = policy_fn(state)
            expected_value = 0.0
            expected_action = np.zeros(k)
            for probability, action in policy:
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                expected_value += probability * prev[next_state]
                expected_action += probability * np.asarray(action, dtype=float)
            current[state] = expected_value - float(expected_action.max(initial=0.0))
        values.append(current)
    return values
