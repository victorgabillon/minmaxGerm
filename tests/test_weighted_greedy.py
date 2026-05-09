import pytest

from expert_game_lab.dp_policy import state_occupancy
from expert_game_lab.experiments import (
    _block_gap_stats,
    _edge_signature,
    _edge_run_intervals,
    _gap_vector,
    _is_one_run,
    _one_run_edge_action_library,
    _prefix_one_run_action_library,
    _prefix_plus_tail_anchor_action_library,
    _local_edge_action_library,
    _top_prefix_length_from_policy,
    evaluate_top_prefix_oracle,
    filter_weighted_greedy_contributions,
    library_oracle,
    one_run_tie_analysis,
    occupation_weighted_greedy_defects,
    summarize_weighted_greedy_by_packet,
    summarize_weighted_greedy_by_regime,
    print_top_prefix_length_regimes,
    print_top_prefix_next_state_debt,
    print_top_prefix_policy_vs_oracle_labels,
    print_top_prefix_restricted_optimal,
    print_policy_occupancy_diff,
    print_top_prefix_candidate_values,
    print_top_prefix_scale_rows,
    print_top_prefix_valid_length_structure,
    top_prefix_oracle_labels,
    top_prefix_tie_analysis,
    weighted_top_prefix_oracle_labels,
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


def test_edge_run_intervals_and_one_run_predicate() -> None:
    assert _edge_run_intervals((0, 1, 1, 0, 1)) == [(1, 2), (4, 4)]
    assert _edge_run_intervals((1, 1, 1, 0)) == [(0, 2)]
    assert _is_one_run((1, 1, 1, 0))
    assert not _is_one_run((0, 1, 1, 0, 1))


def test_block_gap_stats_for_one_run_signature() -> None:
    stats = _block_gap_stats((5, 4, 4, 2, 0), (1, 1, 0, 0))

    assert _gap_vector((5, 4, 4, 2, 0)) == (1, 0, 2, 2)
    assert stats is not None
    assert stats.interval == (0, 1)
    assert stats.sum_gap == 1
    assert stats.max_gap == 1
    assert stats.length == 2
    assert _block_gap_stats((5, 4, 4, 2, 0), (1, 0, 1, 0)) is None


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


def test_one_run_tie_analysis_finds_optimal_one_run_rows() -> None:
    rows, total_occupancy, total_value, one_run_occupancy, one_run_value = one_run_tie_analysis(
        3,
        4,
        comb_policy,
    )

    assert rows
    assert total_occupancy > 0.0
    assert total_value > 0.0
    assert 0.0 < one_run_occupancy <= total_occupancy
    assert 0.0 < one_run_value <= total_value
    assert all(row.optimal_one_run_count >= 1 for row in rows)


def test_top_prefix_tie_analysis_finds_optimal_lengths() -> None:
    rows, total_occupancy, total_value, prefix_occupancy, prefix_value = top_prefix_tie_analysis(
        3,
        4,
        comb_policy,
    )

    assert rows
    assert total_occupancy > 0.0
    assert total_value > 0.0
    assert 0.0 < prefix_occupancy <= total_occupancy
    assert 0.0 < prefix_value <= total_value
    for row in rows:
        assert row.optimal_lengths
        assert row.min_optimal_length == min(row.optimal_lengths)
        assert row.max_optimal_length == max(row.optimal_lengths)


def test_top_prefix_oracle_eval_runs_for_all_selectors() -> None:
    for selector in ("min_valid", "max_valid", "median_valid", "chase_preferred"):
        result = evaluate_top_prefix_oracle(3, 4, selector)

        assert result.selector == selector
        assert result.value <= result.optimal_value + 1e-9
        assert result.gap >= -1e-9
        assert result.selected_length_counts


def test_top_prefix_oracle_labels_runs() -> None:
    rows = top_prefix_oracle_labels(3, 4, "min_valid")

    assert rows
    assert all(row.valid_lengths for row in rows)
    assert all(row.selected_length in row.valid_lengths for row in rows)


def test_weighted_top_prefix_oracle_labels_runs_for_all_selectors() -> None:
    for selector in ("min_valid", "max_valid", "median_valid", "chase_preferred"):
        rows = weighted_top_prefix_oracle_labels(3, 4, selector)

        assert rows
        assert sum(row.occupancy_probability for row in rows) == pytest.approx(4.0)
        assert all(row.valid_lengths for row in rows)
        assert all(row.selected_length in row.valid_lengths for row in rows)


def test_top_prefix_length_from_policy_infers_support_length() -> None:
    policy = [(0.5, (1, 0, 1, 0, 0)), (0.5, (0, 1, 0, 1, 1))]

    assert _top_prefix_length_from_policy(policy) == 3


def test_top_prefix_length_regimes_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_length_regimes(3, 4, "min_valid", n=2)

    captured = capsys.readouterr()
    assert "Selected L=1" in captured.out


def test_top_prefix_policy_vs_oracle_labels_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_policy_vs_oracle_labels(3, 4, "comb", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "invalid occupancy" in captured.out


def test_top_prefix_valid_length_structure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_valid_length_structure(3, 4, "min_valid", occupancy_policy_name="comb", n=2)

    captured = capsys.readouterr()
    assert "Valid parity histogram by value" in captured.out


def test_top_prefix_scale_rows_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_scale_rows(3, 4, "min_valid", occupancy_policy_name="comb", n=2)

    captured = capsys.readouterr()
    assert "near-global L" in captured.out


def test_policy_occupancy_diff_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_policy_occupancy_diff(3, 4, "comb", "top_prefix_shortest", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "total positive occupancy shift" in captured.out


def test_top_prefix_next_state_debt_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_next_state_debt(3, 4, "comb", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "Candidate L cleanup debt" in captured.out


def test_top_prefix_candidate_values_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_candidate_values(3, 4, "comb", length_set="all", n=2)

    captured = capsys.readouterr()
    assert "total weighted regret" in captured.out


def test_top_prefix_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_restricted_optimal(3, 4, "all", n=2)

    captured = capsys.readouterr()
    assert "V_restricted" in captured.out


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
