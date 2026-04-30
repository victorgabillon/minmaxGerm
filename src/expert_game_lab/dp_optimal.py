from __future__ import annotations

from .actions import all_actions
from .lp_game import solve_minimax_step
from .state import all_states, canon


def _next_state_value(
    state: tuple[int, ...],
    action: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
) -> float:
    raw_next = tuple(state[index] + action[index] for index in range(len(state)))
    return float(min(raw_next, default=0)) + continuation[canon(raw_next)]


def optimal_values(
    k: int,
    T: int,
    actions: list[tuple[int, ...]] | None = None,
) -> list[dict[tuple[int, ...], float]]:
    if T < 0:
        raise ValueError("T must be nonnegative")

    action_list = actions if actions is not None else all_actions(k)
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = []

    terminal_states = state_layers[T]
    values.append({state: float(max(state, default=0)) for state in terminal_states})

    for horizon in range(1, T + 1):
        prev = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        domain = state_layers[T - horizon]
        for state in domain:
            q_by_action = {
                action: _next_state_value(state, action, prev)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
        values.append(current)
    return values
