import pytest

from expert_game_lab.dp_policy import state_occupancy
from expert_game_lab.experiments import (
    _edge_signature,
    _one_run_edge_action_library,
    _prefix_one_run_action_library,
    _prefix_plus_tail_anchor_action_library,
    _local_edge_action_library,
    filter_weighted_greedy_contributions,
    library_oracle,
    occupation_weighted_greedy_defects,
    summarize_weighted_greedy_by_packet,
    summarize_weighted_greedy_by_regime,
)
from expert_game_lab.policies import comb_policy, packet_minimal_frontier_policy


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ((1, 0, 1, 0, 0), (1, 1, 1, 0)),
        ((1, 0, 1, 0, 1), (1, 1, 1, 1)),
    ],
)
def test_edge_signature(action: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert _edge_signature(action) == expected


def test_local_edge_action_library_contains_expected_motifs() -> None:
    library5 = set(_local_edge_action_library(5))
    library6 = set(_local_edge_action_library(6))

    assert (1, 0, 1, 0, 0) in library5
    assert (1, 0, 1, 0, 1) in library5
    assert (1, 0, 1, 0, 0, 1) in library6


def test_strict_action_libraries_have_expected_membership() -> None:
    one_run6 = set(_one_run_edge_action_library(6))
    prefix6 = set(_prefix_one_run_action_library(6))
    prefix_tail6 = set(_prefix_plus_tail_anchor_action_library(6))

    assert (1, 0, 1, 0, 0, 0) in one_run6
    assert (1, 0, 1, 0, 0, 1) not in one_run6
    assert (1, 0, 1, 0, 0, 0) in prefix6
    assert (0, 0, 1, 0, 1, 0) not in prefix6
    assert (1, 0, 1, 0, 0, 1) in prefix_tail6
    assert (1, 0, 1, 0, 1, 0) not in prefix_tail6


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_state_occupancy_layers_sum_to_one(k: int, T: int, policy_fn) -> None:
    occupancy = state_occupancy(k, T, policy_fn)
    for layer in occupancy:
        assert sum(layer.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_defect_is_nonnegative(k: int, T: int, policy_fn) -> None:
    total_defect, _ = occupation_weighted_greedy_defects(k, T, policy_fn)
    assert total_defect >= -1e-9


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 3, comb_policy), (3, 4, packet_minimal_frontier_policy)])
def test_weighted_greedy_runs_on_tiny_instances(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    assert total_defect >= -1e-9
    assert contributions


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 3, comb_policy), (3, 4, packet_minimal_frontier_policy)])
def test_policy_edge_signature_probabilities_sum_to_one(k: int, T: int, policy_fn) -> None:
    _, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    for item in contributions:
        assert sum(probability for probability, _ in item.policy_edge_signatures) == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_packet_summaries_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    summaries = summarize_weighted_greedy_by_packet(contributions, n=10)

    assert sum(summary.total_contribution for summary in summaries) == pytest.approx(total_defect)
    assert sum(summary.occupancy_mass for summary in summaries) == pytest.approx(float(T))
    assert summaries


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_regime_summaries_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    summaries = summarize_weighted_greedy_by_regime(contributions, n=10)

    assert sum(summary.total_contribution for summary in summaries) == pytest.approx(total_defect)
    assert sum(summary.occupancy_mass for summary in summaries) == pytest.approx(float(T))
    assert summaries


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_signature_groupings_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    best_signature_totals: dict[tuple[int, ...], float] = {}
    policy_signature_totals: dict[tuple[tuple[float, tuple[int, ...]], ...], float] = {}
    for item in contributions:
        best_signature_totals[item.best_edge_signature] = (
            best_signature_totals.get(item.best_edge_signature, 0.0) + item.contribution
        )
        policy_signature_totals[item.policy_edge_signatures] = (
            policy_signature_totals.get(item.policy_edge_signatures, 0.0) + item.contribution
        )

    assert sum(best_signature_totals.values()) == pytest.approx(total_defect)
    assert sum(policy_signature_totals.values()) == pytest.approx(total_defect)


def test_all_action_library_oracle_has_zero_restriction_loss() -> None:
    _, _, total_loss, contributions = library_oracle(3, 4, "all", comb_policy)

    assert total_loss == pytest.approx(0.0)
    assert sum(item.loss for item in contributions) == pytest.approx(0.0)


def test_weighted_greedy_filter_matches_requested_regime() -> None:
    _, contributions = occupation_weighted_greedy_defects(5, 5, comb_policy)
    filtered = filter_weighted_greedy_contributions(
        contributions,
        packet_type_filter=(1, 2, 2),
        packet_gaps_filter=(1, 1),
    )

    assert filtered
    for item in filtered:
        assert item.packet_type == (1, 2, 2)
        values = []
        for value in item.state:
            if not values or value != values[-1]:
                values.append(value)
        gaps = tuple(values[index] - values[index + 1] for index in range(len(values) - 1))
        assert gaps == (1, 1)
