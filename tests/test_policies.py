import pytest

from expert_game_lab.policies import (
    packet_balanced_partition_policy,
    packet_frontier_policy,
    packet_minimal_frontier_policy,
    packet_regime5_policy,
)


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


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0),
    ],
)
def test_packet_balanced_partition_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_balanced_partition_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0),
    ],
)
def test_packet_balanced_partition_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_balanced_partition_policy(state)
    k = len(state)
    expected_bits = [0.0] * k
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


def test_packet_balanced_partition_policy_origin_support_uses_two_vs_three() -> None:
    policy = packet_balanced_partition_policy((0, 0, 0, 0, 0))
    support = {action for _, action in policy}
    assert all(sum(action) in {2, 3} for action in support)
    assert (0, 0, 0, 1, 1) in support


def test_packet_balanced_partition_policy_pairs_two_chasers_against_leader_state() -> None:
    policy = packet_balanced_partition_policy((1, 0, 0, 0, 0))
    support = {action for _, action in policy}
    assert (0, 0, 0, 1, 1) in support


def test_packet_balanced_partition_policy_uses_one_top_plus_tail_split() -> None:
    policy = packet_balanced_partition_policy((1, 1, 1, 1, 0))
    support = {action for _, action in policy}
    expected = {
        (1, 0, 0, 0, 1),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 0, 1),
        (0, 0, 0, 1, 1),
        (0, 1, 1, 1, 0),
        (1, 0, 1, 1, 0),
        (1, 1, 0, 1, 0),
        (1, 1, 1, 0, 0),
    }
    assert expected <= support


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0),
        (2, 2, 2, 0, 0),
    ],
)
def test_packet_regime5_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_regime5_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (1, 1, 1, 1, 0),
        (2, 2, 2, 0, 0),
    ],
)
def test_packet_regime5_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_regime5_policy(state)
    k = len(state)
    expected_bits = [0.0] * k
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


def test_packet_regime5_policy_origin_support_uses_two_vs_three() -> None:
    policy = packet_regime5_policy((0, 0, 0, 0, 0))
    support = {action for _, action in policy}
    assert all(sum(action) in {2, 3} for action in support)
    assert (0, 0, 0, 1, 1) in support


def test_packet_regime5_policy_pairs_two_chasers_against_leader_state() -> None:
    policy = packet_regime5_policy((1, 0, 0, 0, 0))
    support = {action for _, action in policy}
    assert (0, 0, 0, 1, 1) in support


def test_packet_regime5_policy_uses_one_top_plus_tail_split() -> None:
    policy = packet_regime5_policy((1, 1, 1, 1, 0))
    support = {action for _, action in policy}
    assert (1, 0, 0, 0, 1) in support


def test_packet_regime5_policy_breaks_size_three_packet_on_three_two_states() -> None:
    policy = packet_regime5_policy((2, 2, 2, 0, 0))
    support = {action for _, action in policy}
    expected_singletons = {
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
    }
    expected_complements = {
        (0, 1, 1, 1, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 0, 1, 1),
    }
    assert expected_singletons & support
    assert expected_complements & support