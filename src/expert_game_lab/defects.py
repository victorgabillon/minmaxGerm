from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .actions import all_actions
from .lp_game import solve_minimax_step
from .state import canon


def _lookup(values_or_fn, state: tuple[int, ...]) -> float:
    if isinstance(values_or_fn, Mapping):
        return float(values_or_fn[state])
    return float(values_or_fn(state))


def _distribution_after_policy(
    x: tuple[int, ...],
    policy_fn,
    delayed_action: tuple[int, ...] | None = None,
) -> dict[tuple[int, ...], float]:
    distribution: dict[tuple[int, ...], float] = defaultdict(float)
    for probability, action in policy_fn(x):
        state_after_policy = canon(tuple(x[index] + action[index] for index in range(len(x))))
        if delayed_action is None:
            distribution[state_after_policy] += probability
            continue
        delayed_state = canon(
            tuple(state_after_policy[index] + delayed_action[index] for index in range(len(x)))
        )
        distribution[delayed_state] += probability
    return dict(distribution)


def greedy_defect(
    k: int,
    x: tuple[int, ...],
    V_prev_policy,
    policy_fn,
    actions: list[tuple[int, ...]] | None = None,
) -> float:
    action_list = actions if actions is not None else all_actions(k)
    q_by_action = {
        action: _lookup(
            V_prev_policy,
            canon(tuple(x[index] + action[index] for index in range(k))),
        )
        for action in action_list
    }
    solution = solve_minimax_step(q_by_action, k)
    if not solution.success:
        raise RuntimeError(f"LP failed at state {x}: {solution.message}")

    policy_step = 0.0
    expected_coordinate = [0.0] * k
    for probability, action in policy_fn(x):
        policy_step += probability * q_by_action[action]
        for index, bit in enumerate(action):
            expected_coordinate[index] += probability * bit
    policy_step -= max(expected_coordinate, default=0.0)
    return solution.value - policy_step


def commutation_defect(
    k: int,
    x: tuple[int, ...],
    first_action: tuple[int, ...],
    policy_fn,
) -> tuple[float, tuple[int, ...]]:
    first_state = canon(tuple(x[index] + first_action[index] for index in range(k)))
    after_first_then_policy = _distribution_after_policy(first_state, policy_fn)

    best_tv = float("inf")
    best_action = all_actions(k)[0]
    for delayed_action in all_actions(k):
        after_policy_then_delayed = _distribution_after_policy(x, policy_fn, delayed_action)
        support = set(after_first_then_policy) | set(after_policy_then_delayed)
        tv = 0.5 * sum(
            abs(after_first_then_policy.get(state, 0.0) - after_policy_then_delayed.get(state, 0.0))
            for state in support
        )
        if tv < best_tv:
            best_tv = tv
            best_action = delayed_action
    return best_tv, best_action
