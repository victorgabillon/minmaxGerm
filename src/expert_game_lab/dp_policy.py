from __future__ import annotations

from collections import defaultdict

import numpy as np

from .state import all_states, canon


def expected_action(policy: list[tuple[float, tuple[int, ...]]], k: int) -> np.ndarray:
    expectation = np.zeros(k)
    for probability, action in policy:
        expectation += probability * np.asarray(action, dtype=float)
    return expectation


def _next_state_value(
    state: tuple[int, ...],
    action: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
) -> float:
    raw_next = tuple(state[index] + action[index] for index in range(len(state)))
    return float(min(raw_next, default=0)) + continuation[canon(raw_next)]


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
            expected_action_vector = expected_action(policy, k)
            for probability, action in policy:
                expected_value += probability * _next_state_value(state, action, prev)
            current[state] = expected_value - float(expected_action_vector.max(initial=0.0))
        values.append(current)
    return values


def state_occupancy(k: int, T: int, policy_fn) -> list[dict[tuple[int, ...], float]]:
    if T < 0:
        raise ValueError("T must be nonnegative")

    zero = tuple(0 for _ in range(k))
    occupancy: list[dict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0

    for time in range(T):
        for state, state_probability in occupancy[time].items():
            for action_probability, action in policy_fn(state):
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += state_probability * action_probability
    return [dict(layer) for layer in occupancy]
