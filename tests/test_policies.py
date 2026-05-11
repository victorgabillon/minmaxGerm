import pytest

from expert_game_lab.policies import (
    packet_balanced_partition_policy,
    packet_frontier_policy,
    packet_minimal_frontier_policy,
    packet_regime5_policy,
    packet_regime5b_policy,
    packet_regime5c_policy,
    top_prefix_chase_length_policy,
    top_prefix_gap_sum_long_policy,
    top_prefix_gap_sum_short_policy,
    top_prefix_longest_policy,
    top_prefix_shortest_policy,
    top_prefix_three_regime_policy,
    top_prefix_three_regime_v2_policy,
    top_prefix_three_regime_v3_policy,
    top_prefix_three_regime_v4_policy,
    top_prefix_three_regime_v5_policy,
    top_prefix_three_regime_v6_policy,
    top_prefix_three_regime_v7_policy,
    top_prefix_tie_mimic_policy,
    twin_comb3_policy,
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
    ("policy_fn", "expected_support"),
    [
        (top_prefix_shortest_policy, {(1, 0, 0, 0, 0), (0, 1, 1, 1, 1)}),
        (top_prefix_longest_policy, {(1, 0, 1, 0, 1), (0, 1, 0, 1, 0)}),
        (top_prefix_chase_length_policy, {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
    ],
)
def test_top_prefix_fixed_length_policies_use_expected_support(
    policy_fn,
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = policy_fn((0, 0, 0, 0, 0))

    assert {action for _, action in policy} == expected_support
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_top_prefix_gap_sum_policies_differ_only_on_tie_breaks() -> None:
    short_policy = top_prefix_gap_sum_short_policy((0, 0, 0, 0))
    long_policy = top_prefix_gap_sum_long_policy((0, 0, 0, 0))

    assert {action for _, action in short_policy} == {
        (1, 0, 0, 0),
        (0, 1, 1, 1),
    }
    assert {action for _, action in long_policy} == {
        (1, 0, 1, 0),
        (0, 1, 0, 1),
    }


@pytest.mark.parametrize("state", [(0, 0, 0), (2, 1, 1)])
def test_twin_comb3_policy_uses_leader_against_tail_support(state: tuple[int, ...]) -> None:
    policy = twin_comb3_policy(state)

    assert {action for _, action in policy} == {(1, 0, 0), (0, 1, 1)}
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    expected_bits = [0.0, 0.0, 0.0]
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    assert expected_bits == pytest.approx([0.5, 0.5, 0.5])


def test_top_prefix_three_regime_policy_probabilities_sum_to_one_and_is_balanced() -> None:
    policy = top_prefix_three_regime_policy((3, 1, 1, 0, 0))

    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    for index in range(5):
        assert sum(probability * action[index] for probability, action in policy) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((0, 0, 0, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)}),
        ((0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((3, 1, 1, 0, 0), {(1, 0, 0, 0, 0), (0, 1, 1, 1, 1)}),
        ((2, 1, 1, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)}),
        ((1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_policy_uses_expected_support(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_policy(state)

    assert {action for _, action in policy} == expected_support


def test_top_prefix_three_regime_v2_policy_probabilities_sum_to_one_and_is_balanced() -> None:
    policy = top_prefix_three_regime_v2_policy((3, 1, 1, 0, 0))

    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    for index in range(5):
        assert sum(probability * action[index] for probability, action in policy) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((0, 0, 0, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)}),
        ((0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((3, 1, 1, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((4, 2, 2, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((3, 2, 2, 1, 1, 0), {(1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1)}),
        ((3, 2, 2, 1, 1, 0, 0), {(1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 1, 1, 1)}),
        ((2, 1, 1, 1, 1, 1, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((3, 0, 0, 0, 0), {(1, 0, 0, 0, 0), (0, 1, 1, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_v2_policy_uses_expected_support(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v2_policy(state)

    assert {action for _, action in policy} == expected_support


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((0, 0, 0, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)}),
        ((0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((3, 1, 1, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((4, 2, 2, 0, 0), {(1, 0, 1, 0, 0), (0, 1, 0, 1, 1)}),
        ((3, 2, 2, 1, 1, 0), {(1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 1, 1)}),
        ((3, 2, 2, 1, 1, 1, 0), {(1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 1, 1, 1)}),
        ((1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 2, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((3, 0, 0, 0, 0), {(1, 0, 0, 0, 0), (0, 1, 1, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_v3_policy_uses_expected_support(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v3_policy(state)

    assert {action for _, action in policy} == expected_support


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((2, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 1, 1, 1)}),
        ((1, 1, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 2, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 1, 1, 1, 1, 1, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_v4_policy_refines_large_tail_frontier(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v4_policy(state)

    assert {action for _, action in policy} == expected_support


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((0, 0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1, 0, 1)}),
        ((2, 2, 2, 1, 1, 1, 1, 0), {(1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 1, 1)}),
        ((2, 1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 1, 1, 1)}),
        ((1, 1, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 2, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 1, 1, 1, 1, 1, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_v5_policy_caps_broad_nonflat_prefix(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v5_policy(state)

    assert {action for _, action in policy} == expected_support


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((1, 1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1, 0, 1)}),
        ((2, 2, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 1, 1)}),
        ((2, 1, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 0, 0, 0, 0), (0, 1, 0, 1, 1, 1, 1, 1)}),
        ((0, 0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1, 0, 1)}),
        ((1, 1, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
        ((2, 1, 1, 1, 1, 1, 0), {(1, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 1, 1)}),
    ],
)
def test_top_prefix_three_regime_v6_policy_refines_global_and_broad_local_mass(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v6_policy(state)

    assert {action for _, action in policy} == expected_support


@pytest.mark.parametrize(
    ("state", "expected_support"),
    [
        ((1, 1, 1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 1, 1), (0, 1, 0, 1, 0, 1, 0, 0, 0)}),
        ((1, 1, 1, 1, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 1, 1), (0, 1, 0, 1, 0, 1, 0, 0, 0)}),
        ((2, 2, 2, 1, 1, 1, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 1, 1), (0, 1, 0, 1, 0, 1, 0, 0, 0)}),
        ((2, 1, 1, 1, 1, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 1, 1), (0, 1, 0, 1, 0, 1, 0, 0, 0)}),
        ((0, 0, 0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 0, 1, 0, 1, 0)}),
        ((0, 0, 0, 0, 0, 0, 0, 0), {(1, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1, 0, 1)}),
    ],
)
def test_top_prefix_three_regime_v7_policy_adds_near_global_scale(
    state: tuple[int, ...],
    expected_support: set[tuple[int, ...]],
) -> None:
    policy = top_prefix_three_regime_v7_policy(state)

    assert {action for _, action in policy} == expected_support


def test_top_prefix_tie_mimic_policy_uses_diagnostic_lengths() -> None:
    assert {action for _, action in top_prefix_tie_mimic_policy((0, 0, 0, 0, 0))} == {
        (1, 0, 0, 0, 0),
        (0, 1, 1, 1, 1),
    }
    assert {action for _, action in top_prefix_tie_mimic_policy((0, 0, 0, 0, 0, 0))} == {
        (1, 0, 1, 0, 0, 0),
        (0, 1, 0, 1, 1, 1),
    }


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


@pytest.mark.parametrize(
    "state",
    [
        (2, 1, 1, 0, 0),
        (1, 1, 1, 1, 0),
    ],
)
def test_packet_regime5b_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_regime5b_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (2, 1, 1, 0, 0),
        (1, 1, 1, 1, 0),
    ],
)
def test_packet_regime5b_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_regime5b_policy(state)
    expected_bits = [0.0] * len(state)
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


def test_packet_regime5b_policy_uses_local_chase_rule_at_near_collision_frontier() -> None:
    policy = packet_regime5b_policy((2, 1, 1, 0, 0))
    support = {action for _, action in policy}
    assert support == {
        (1, 0, 1, 0, 0),
        (1, 1, 0, 0, 0),
        (0, 0, 1, 1, 1),
        (0, 1, 0, 1, 1),
    }


def test_packet_regime5b_policy_falls_back_to_packet_regime5_outside_special_rule() -> None:
    assert packet_regime5b_policy((1, 1, 1, 1, 0)) == packet_regime5_policy((1, 1, 1, 1, 0))


@pytest.mark.parametrize(
    "state",
    [
        (1, 1, 0, 0, 0),
        (1, 1, 1, 0, 0),
        (2, 1, 1, 0, 0),
    ],
)
def test_packet_regime5c_policy_probabilities_sum_to_one(state: tuple[int, ...]) -> None:
    policy = packet_regime5c_policy(state)
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "state",
    [
        (1, 1, 0, 0, 0),
        (1, 1, 1, 0, 0),
        (2, 1, 1, 0, 0),
    ],
)
def test_packet_regime5c_policy_is_coordinate_balanced(state: tuple[int, ...]) -> None:
    policy = packet_regime5c_policy(state)
    expected_bits = [0.0] * len(state)
    for probability, action in policy:
        for index, bit in enumerate(action):
            expected_bits[index] += probability * bit
    for value in expected_bits:
        assert value == pytest.approx(0.5)


@pytest.mark.parametrize("state", [(1, 1, 0, 0, 0), (1, 1, 1, 0, 0)])
def test_packet_regime5c_policy_uses_chase_support_on_two_packet_near_collision_states(
    state: tuple[int, ...],
) -> None:
    policy = packet_regime5c_policy(state)
    support = {action for _, action in policy}
    assert support == {
        (1, 0, 1, 0, 0),
        (0, 1, 0, 1, 1),
    }


def test_packet_regime5c_policy_matches_packet_regime5b_on_three_packet_near_collision_state() -> None:
    assert packet_regime5c_policy((2, 1, 1, 0, 0)) == packet_regime5b_policy((2, 1, 1, 0, 0))
