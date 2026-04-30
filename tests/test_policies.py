import pytest

from expert_game_lab.policies import packet_frontier_policy, packet_minimal_frontier_policy


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0),
        (3, 3, 1, 0),
        (4, 2, 2, 2, 0),
        (5, 4, 4, 1, 0),
    ],
)
def test_packet_frontier_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_frontier_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0),
        (3, 3, 1, 0),
        (4, 2, 2, 2, 0),
        (5, 4, 4, 1, 0),
    ],
)
def test_packet_frontier_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_frontier_policy(state)
    k = len(state)
    expected_bits = [0.0] * k
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0),
        (3, 3, 1, 0),
        (5, 5, 5, 5, 0),
    ],
)
def test_packet_minimal_frontier_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_minimal_frontier_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0),
        (3, 3, 1, 0),
        (5, 5, 5, 5, 0),
    ],
)
def test_packet_minimal_frontier_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_minimal_frontier_policy(state)
    k = len(state)
    expected_bits = [0.0] * k
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


def test_packet_minimal_frontier_policy_support_contains_singleton_top_splits() -> None:
    policy = packet_minimal_frontier_policy((5, 5, 5, 5, 0))
    support = {action for _, action in policy}
    expected = {
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 1, 1, 1, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 1, 0, 1),
    }
    assert expected <= support