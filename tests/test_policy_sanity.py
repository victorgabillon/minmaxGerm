import pytest

from expert_game_lab.dp_optimal import optimal_values
from expert_game_lab.dp_policy import evaluate_balanced_policy, expected_action
from expert_game_lab.policies import comb_policy, packet_frontier_policy, packet_minimal_frontier_policy


POLICIES = [
    ("comb", comb_policy),
    ("packet_frontier", packet_frontier_policy),
    ("packet_minimal_frontier", packet_minimal_frontier_policy),
]


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(("_name", "policy_fn"), POLICIES)
def test_policy_value_satisfies_basic_bounds(k: int, T: int, _name: str, policy_fn) -> None:
    zero = tuple(0 for _ in range(k))
    v_star = optimal_values(k, T)[T][zero]
    v_policy = evaluate_balanced_policy(k, T, policy_fn)[T][zero]
    gap = v_star - v_policy

    assert v_policy >= -1e-9
    assert v_policy <= v_star + 1e-9
    assert gap / (T ** 0.5) <= v_star / (T ** 0.5) + 1e-9


@pytest.mark.parametrize(("_name", "policy_fn"), POLICIES)
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_balanced_policy_has_value_one_half_at_horizon_one(k: int, _name: str, policy_fn) -> None:
    zero = tuple(0 for _ in range(k))
    v_policy = evaluate_balanced_policy(k, 1, policy_fn)[1][zero]
    assert v_policy == pytest.approx(0.5)


@pytest.mark.parametrize(("_name", "policy_fn"), POLICIES)
@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0),
        (2, 2, 1),
        (3, 3, 3, 0),
        (5, 5, 5, 5, 0),
    ],
)
def test_expected_action_is_coordinate_balanced(_name: str, policy_fn, state: tuple[int, ...]) -> None:
    policy = policy_fn(state)
    expectation = expected_action(policy, len(state))
    for value in expectation:
        assert value == pytest.approx(0.5)