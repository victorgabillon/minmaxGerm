from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class StepSolution:
    value: float
    p: tuple[float, ...]
    z: float
    success: bool
    message: str


def solve_minimax_step(
    q_by_action: dict[tuple[int, ...], float],
    k: int,
) -> StepSolution:
    if not q_by_action:
        raise ValueError("q_by_action must be non-empty")

    variables = k + 1
    c = np.zeros(variables)
    c[-1] = 1.0

    actions = list(q_by_action)
    a_ub = np.zeros((len(actions), variables))
    b_ub = np.zeros(len(actions))
    for row, action in enumerate(actions):
        a_ub[row, :k] = -np.asarray(action, dtype=float)
        a_ub[row, -1] = -1.0
        b_ub[row] = -float(q_by_action[action])

    a_eq = np.zeros((1, variables))
    a_eq[0, :k] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * k + [(None, None)]

    result = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return StepSolution(
            value=float("nan"),
            p=tuple(float("nan") for _ in range(k)),
            z=float("nan"),
            success=False,
            message=result.message,
        )

    p = tuple(float(value) for value in result.x[:k])
    z = float(result.x[-1])
    return StepSolution(value=z, p=p, z=z, success=True, message=result.message)
