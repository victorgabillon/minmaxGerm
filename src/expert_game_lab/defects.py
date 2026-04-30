from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .actions import all_actions
from .lp_game import solve_minimax_step
from .state import canon


@dataclass(frozen=True)
class MixedCommutationSolution:
    best_tv: float
    weights_by_action: dict[tuple[int, ...], float]
    success: bool
    message: str


def _continuation_value(
    x: tuple[int, ...],
    action: tuple[int, ...],
    values_or_fn,
) -> float:
    raw_next = tuple(x[index] + action[index] for index in range(len(x)))
    return float(min(raw_next, default=0)) + _lookup(values_or_fn, canon(raw_next))


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
        action: _continuation_value(x, action, V_prev_policy)
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


def commutation_defect_mixed(
    k: int,
    x: tuple[int, ...],
    first_action: tuple[int, ...],
    policy_fn,
) -> MixedCommutationSolution:
    actions = all_actions(k)
    first_state = canon(tuple(x[index] + first_action[index] for index in range(k)))
    target = _distribution_after_policy(first_state, policy_fn)
    delayed_distributions = {
        action: _distribution_after_policy(x, policy_fn, action)
        for action in actions
    }

    support = sorted(
        set(target).union(*(distribution.keys() for distribution in delayed_distributions.values()))
    )
    state_index = {state: index for index, state in enumerate(support)}
    action_count = len(actions)
    residual_count = len(support)
    variable_count = action_count + residual_count

    objective = np.zeros(variable_count)
    objective[action_count:] = 0.5

    a_ub = np.zeros((2 * residual_count, variable_count))
    b_ub = np.zeros(2 * residual_count)
    for row, state in enumerate(support):
        for action_index, action in enumerate(actions):
            probability = delayed_distributions[action].get(state, 0.0)
            a_ub[2 * row, action_index] = -probability
            a_ub[2 * row + 1, action_index] = probability
        residual_index = action_count + state_index[state]
        a_ub[2 * row, residual_index] = -1.0
        a_ub[2 * row + 1, residual_index] = -1.0
        target_probability = target.get(state, 0.0)
        b_ub[2 * row] = -target_probability
        b_ub[2 * row + 1] = target_probability

    a_eq = np.zeros((1, variable_count))
    a_eq[0, :action_count] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * variable_count

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return MixedCommutationSolution(
            best_tv=float("nan"),
            weights_by_action={},
            success=False,
            message=result.message,
        )

    weights = {
        action: float(result.x[index])
        for index, action in enumerate(actions)
        if result.x[index] > 1e-9
    }
    return MixedCommutationSolution(
        best_tv=float(result.fun),
        weights_by_action=weights,
        success=True,
        message=result.message,
    )
