import pytest

from expert_game_lab.defects import commutation_defect, commutation_defect_mixed
from expert_game_lab.policies import packet_frontier_policy, packet_minimal_frontier_policy


@pytest.mark.parametrize(
    ("state", "first_action", "policy_fn"),
    [
        ((0, 0, 0), (1, 0, 0), packet_frontier_policy),
        ((2, 2, 1, 0), (0, 1, 0, 0), packet_minimal_frontier_policy),
        ((3, 3, 3, 0), (0, 0, 1, 0), packet_minimal_frontier_policy),
    ],
)
def test_mixed_commutation_is_no_worse_than_deterministic(
    state: tuple[int, ...],
    first_action: tuple[int, ...],
    policy_fn,
) -> None:
    deterministic_tv, _ = commutation_defect(len(state), state, first_action, policy_fn)
    mixed = commutation_defect_mixed(len(state), state, first_action, policy_fn)
    assert mixed.success
    assert mixed.best_tv >= 0.0
    assert mixed.best_tv <= deterministic_tv + 1e-9


@pytest.mark.parametrize(
    ("state", "first_action"),
    [
        ((0, 0, 0), (1, 0, 0)),
        ((2, 2, 1, 0), (0, 1, 0, 0)),
    ],
)
def test_mixed_commutation_weights_sum_to_one(
    state: tuple[int, ...],
    first_action: tuple[int, ...],
) -> None:
    solution = commutation_defect_mixed(
        len(state),
        state,
        first_action,
        packet_minimal_frontier_policy,
    )
    assert solution.success
    assert sum(solution.weights_by_action.values()) == pytest.approx(1.0)
    assert solution.best_tv >= 0.0