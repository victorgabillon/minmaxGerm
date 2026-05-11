import pytest

from expert_game_lab.dp_policy import state_occupancy
from expert_game_lab.experiments import (
    _block_gap_stats,
    _edge_signature,
    _edge_run_intervals,
    _gap_vector,
    _is_one_run,
    _edge_run_restricted_candidates,
    _exchangeable_actions_for_orbit_key,
    _one_run_edge_action_library,
    _one_run_restricted_candidates,
    _lift_canonical_action_to_named_distribution,
    _lift_canonical_vector_to_named,
    _parse_cases,
    _prefix_one_run_action_library,
    _prefix_plus_tail_anchor_action_library,
    _terminal_named_winner_probabilities,
    _local_edge_action_library,
    _top_prefix_length_from_policy,
    evaluate_top_prefix_oracle,
    filter_weighted_greedy_contributions,
    k3_motif_sweep,
    k9_motif_library_sweep,
    library_oracle,
    probability_matching_inspect_rows,
    probability_matching_named_inspect_rows,
    probability_matching_named_residual_aggregates,
    library_lp_restricted_optimal,
    one_run_tie_analysis,
    one_run_restricted_optimal,
    edge_run_restricted_optimal,
    occupation_weighted_greedy_defects,
    print_one_run_restricted_optimal,
    print_edge_run_restricted_optimal,
    print_library_lp_restricted_optimal,
    print_library_lp_dual_inspect,
    print_library_lp_dual_orbit_completion,
    print_library_lp_dual_orbits,
    print_k3_motif_sweep,
    print_k9_motif_library_sweep,
    print_probability_matching_inspect,
    print_probability_matching_dual_face_repair,
    print_probability_matching_named_inspect,
    print_probability_matching_named_residuals,
    probability_matching_dual_face_repair_rows,
    explicit_policy_benchmark_rows,
    print_explicit_policy_benchmark,
    print_strategy_class_relative_benchmark,
    print_strategy_class_benchmark,
    strategy_class_relative_benchmark_rows,
    strategy_class_benchmark_rows,
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
    top_prefix_restricted_optimal,
    top_prefix_tie_analysis,
    two_run_orbit_mixture_v1_policy,
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


def test_one_run_restricted_candidates_include_prefix_and_centered_intervals() -> None:
    candidates = _one_run_restricted_candidates(5)
    by_interval = {candidate.intervals[0]: candidate for candidate in candidates}

    assert (0, 2) in by_interval
    assert by_interval[(0, 2)].edge_signature == (1, 1, 1, 0)
    assert (1, 2) in by_interval
    assert by_interval[(1, 2)].edge_signature == (0, 1, 1, 0)


def test_two_run_restricted_candidates_include_one_run_and_two_run_patterns() -> None:
    candidates = _edge_run_restricted_candidates(9, 2)
    signatures = {candidate.edge_signature for candidate in candidates}

    assert (1, 1, 1, 0, 0, 0, 0, 0) in signatures
    assert (1, 0, 1, 1, 0, 0, 0, 0) in signatures


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


def test_one_run_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_one_run_restricted_optimal(3, 4, n=2)

    captured = capsys.readouterr()
    assert "library size" in captured.out


def test_two_run_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_edge_run_restricted_optimal(3, 4, 2, n=2)

    captured = capsys.readouterr()
    assert "library size" in captured.out


def test_one_run_restricted_value_dominates_top_prefix_restricted() -> None:
    one_run_result = one_run_restricted_optimal(3, 4)
    top_prefix_result = top_prefix_restricted_optimal(3, 4, "all")

    assert one_run_result.value >= top_prefix_result.value - 1e-9


def test_two_run_restricted_value_dominates_one_run_restricted() -> None:
    two_run_result = edge_run_restricted_optimal(3, 4, 2)
    one_run_result = one_run_restricted_optimal(3, 4)

    assert two_run_result.value >= one_run_result.value - 1e-9


def test_library_lp_all_matches_optimal_on_small_instance() -> None:
    result = library_lp_restricted_optimal(3, 4, "all")

    assert result.gap == pytest.approx(0.0)


def test_library_lp_monotone_for_run_libraries() -> None:
    one_run_result = library_lp_restricted_optimal(3, 4, "one_run")
    two_run_result = library_lp_restricted_optimal(3, 4, "two_run")

    assert two_run_result.value >= one_run_result.value - 1e-9


def test_library_lp_one_run_dominates_deterministic_one_run_pair() -> None:
    lp_result = library_lp_restricted_optimal(3, 4, "one_run")
    deterministic_result = one_run_restricted_optimal(3, 4)

    assert lp_result.value >= deterministic_result.value - 1e-9


def test_library_lp_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_restricted_optimal(3, 4, "one_run", n=2)

    captured = capsys.readouterr()
    assert "Library LP restricted optimal" in captured.out


def test_library_lp_dual_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_inspect(3, 4, "two_run", support_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual inspect" in captured.out
    assert "adversary dual support" in captured.out


def test_library_lp_dual_orbits_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_orbits(3, 4, "two_run", orbit_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual support orbits" in captured.out
    assert "support orbits" in captured.out


def test_library_lp_dual_orbit_completion_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_orbit_completion(3, 4, "two_run", orbit_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual orbit completion" in captured.out
    assert "library_compatible" in captured.out


def test_k3_motif_sweep_runs() -> None:
    rows = k3_motif_sweep((4, 5))

    assert rows
    assert {row.library_name for row in rows} >= {
        "comb",
        "twin_comb3",
        "tail_singleton",
        "comb_twin_comb3",
        "all_three",
    }
    assert all(row.lp_value <= row.optimal_value + 1e-9 for row in rows)


def test_k3_motif_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k3_motif_sweep((4,))

    captured = capsys.readouterr()
    assert "k=3 motif sweep" in captured.out
    assert "comb_twin_comb3" in captured.out


def test_k9_motif_library_sweep_runs() -> None:
    rows = k9_motif_library_sweep((1,))

    assert rows
    assert {row.family_name for row in rows} >= {"top_prefix_all", "one_run", "two_run"}


def test_k9_motif_library_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k9_motif_library_sweep((1,), n=2)

    captured = capsys.readouterr()
    assert "k=9 motif-library sweep" in captured.out
    assert "top_prefix_all" in captured.out


def test_probability_matching_inspect_rows_have_winner_probabilities() -> None:
    _, rows = probability_matching_inspect_rows(3, 4, "all_three")

    assert rows
    for row in rows:
        assert sum(row.winner_probabilities) == pytest.approx(1.0)
        assert row.l1_error >= 0.0
        assert row.linf_error >= 0.0


def test_probability_matching_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_inspect(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Probability matching inspect" in captured.out
    assert "weighted L1 error" in captured.out


def test_lift_canonical_vector_to_named_averages_ties() -> None:
    lifted = _lift_canonical_vector_to_named((5, 3, 3, 1), (0.6, 0.2, 0.1, 0.1))

    assert lifted == pytest.approx((0.6, 0.15, 0.15, 0.1))


def test_exchangeable_action_lift_preserves_mass_and_packet_counts() -> None:
    lifted = _lift_canonical_action_to_named_distribution((5, 3, 3, 1), (1, 1, 0, 0))

    assert sum(probability for probability, _ in lifted) == pytest.approx(1.0)
    assert dict((action, probability) for probability, action in lifted) == {
        (1, 0, 1, 0): pytest.approx(0.5),
        (1, 1, 0, 0): pytest.approx(0.5),
    }
    for _, action in lifted:
        assert (action[0], action[1] + action[2], action[3]) == (1, 1, 0)


def test_terminal_named_winner_probabilities_sum_to_one() -> None:
    winners = _terminal_named_winner_probabilities((2, 3, 3, 1))

    assert winners == pytest.approx((0.0, 0.5, 0.5, 0.0))
    assert sum(winners) == pytest.approx(1.0)


def test_probability_matching_named_inspect_rows_run_and_keep_named_root() -> None:
    _, rows = probability_matching_named_inspect_rows(3, 4, "all_three")

    assert rows
    for row in rows:
        assert sum(row.winner_probabilities) == pytest.approx(1.0)
        assert row.l1_error >= 0.0
        assert row.linf_error >= 0.0

    root_row = next(row for row in rows if row.time == 0 and row.named_state == (0, 0, 0))
    assert root_row.winner_probabilities != pytest.approx((1.0, 0.0, 0.0))
    assert root_row.winner_probabilities == pytest.approx((1.0 / 3.0,) * 3)


def test_probability_matching_named_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_named_inspect(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Named probability matching inspect" in captured.out
    assert "weighted L1 error" in captured.out


def test_probability_matching_named_residuals_are_zero_for_k3_all_three() -> None:
    rows, canonical_aggregates, regime_aggregates = probability_matching_named_residual_aggregates(
        3,
        4,
        "all_three",
    )

    assert rows
    assert canonical_aggregates
    assert regime_aggregates
    assert sum(row.weighted_l1_error for row in rows) == pytest.approx(0.0)
    assert sum(row.weighted_linf_error for row in rows) == pytest.approx(0.0)
    assert sum(item.weighted_linf_error for item in canonical_aggregates) == pytest.approx(0.0)


def test_probability_matching_named_residuals_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_named_residuals(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Named probability matching residuals" in captured.out
    assert "Top canonical-state residual aggregates" in captured.out


def test_probability_matching_dual_face_repair_keeps_k3_zero_residual() -> None:
    _, repair_rows = probability_matching_dual_face_repair_rows(3, 4, "all_three", n=5)

    assert repair_rows
    assert all(row.success for row in repair_rows)
    assert sum(row.original_weighted_l1_error for row in repair_rows) == pytest.approx(0.0)
    assert sum(row.repaired_weighted_l1_error for row in repair_rows) == pytest.approx(0.0, abs=1e-7)
    assert sum(row.repaired_weighted_linf_error for row in repair_rows) <= (
        sum(row.original_weighted_linf_error for row in repair_rows) + 1e-7
    )


def test_probability_matching_dual_face_repair_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_dual_face_repair(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Probability matching dual-face repair" in captured.out
    assert "Top repaired rows" in captured.out


def test_parse_cases() -> None:
    assert _parse_cases("3:20, 9:7") == ((3, 20), (9, 7))


def test_strategy_class_benchmark_rows_run_for_tiny_case() -> None:
    rows, skipped = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
    )

    assert not skipped
    assert {row.library_name for row in rows} == {"top_prefix_all", "one_run", "two_run"}
    assert all(row.library_size > 0 for row in rows)
    assert all(row.value <= row.optimal_value + 1e-9 for row in rows)


def test_strategy_class_benchmark_no_active_counts_preserves_values() -> None:
    rows_with_counts, skipped_with_counts = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run"),
    )
    rows_without_counts, skipped_without_counts = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run"),
        collect_active_counts=False,
    )

    assert not skipped_with_counts
    assert not skipped_without_counts
    by_library_with_counts = {row.library_name: row for row in rows_with_counts}
    by_library_without_counts = {row.library_name: row for row in rows_without_counts}
    assert by_library_with_counts.keys() == by_library_without_counts.keys()
    for library_name, row in by_library_with_counts.items():
        cheap_row = by_library_without_counts[library_name]
        assert cheap_row.value == pytest.approx(row.value)
        assert cheap_row.gap == pytest.approx(row.gap)
        assert cheap_row.active_action_count == -1
        assert cheap_row.active_edge_signature_count == -1


def test_strategy_class_relative_benchmark_matches_exact_values() -> None:
    exact_rows, exact_skipped = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        collect_active_counts=False,
    )
    relative_rows, relative_skipped = strategy_class_relative_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        reference_library_name="two_run",
    )

    assert not exact_skipped
    assert not relative_skipped
    exact_by_library = {row.library_name: row for row in exact_rows}
    relative_by_library = {row.library_name: row for row in relative_rows}
    reference_value = exact_by_library["two_run"].value
    assert relative_by_library["two_run"].gap_to_reference == pytest.approx(0.0)
    for library_name, row in relative_by_library.items():
        assert row.value == pytest.approx(exact_by_library[library_name].value)
        assert row.reference_value == pytest.approx(reference_value)
        assert row.gap_to_reference == pytest.approx(reference_value - row.value)


def test_strategy_class_relative_benchmark_reference_can_be_implicit() -> None:
    rows, skipped = strategy_class_relative_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all",),
        reference_library_name="two_run",
    )

    assert not skipped
    assert {row.library_name for row in rows} == {"top_prefix_all"}
    assert rows[0].reference_library_name == "two_run"


def test_strategy_class_benchmark_all_library_is_valid_beyond_k3() -> None:
    rows, skipped = strategy_class_benchmark_rows(((4, 3),), ("all", "all_three"))

    assert any(row.library_name == "all" for row in rows)
    assert any("all_three is only valid for k=3" in message for message in skipped)


def test_strategy_class_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_strategy_class_benchmark(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        include_probability_matching=True,
        probability_matching_max_k=3,
        probability_matching_max_T=4,
        n=2,
    )

    captured = capsys.readouterr()
    assert "Strategy class benchmark" in captured.out
    assert "gap/sqrt(T)" in captured.out
    assert "pm_avg_linf" in captured.out


def test_strategy_class_relative_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_strategy_class_relative_benchmark(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        reference_library_name="two_run",
        n=2,
    )

    captured = capsys.readouterr()
    assert "Strategy class relative benchmark" in captured.out
    assert "gap_to_reference/sqrt(T)" in captured.out


def test_exchangeable_actions_for_orbit_key_respects_packet_counts() -> None:
    policy = _exchangeable_actions_for_orbit_key(
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (2, 1),
        max_runs=2,
    )

    assert policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    for _, action in policy:
        assert sum(action[:4]) == 2
        assert sum(action[4:]) == 1
        assert sum(action) == 3
        assert action in {candidate for _, candidate in policy}


def test_two_run_orbit_mixture_v1_policy_returns_distribution() -> None:
    policy = two_run_orbit_mixture_v1_policy((0,) * 9)

    assert policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    assert all(len(action) == 9 for _, action in policy)


def test_explicit_policy_benchmark_rows_run_without_reference_lp() -> None:
    rows, skipped = explicit_policy_benchmark_rows(
        ((9, 2),),
        ("two_run_orbit_mixture_v1",),
        include_reference=False,
    )

    assert not skipped
    assert len(rows) == 1
    assert rows[0].policy_name == "two_run_orbit_mixture_v1"
    assert rows[0].reference_value is None


def test_explicit_policy_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_explicit_policy_benchmark(
        ((9, 2),),
        ("two_run_orbit_mixture_v1",),
        include_reference=False,
    )

    captured = capsys.readouterr()
    assert "Explicit policy benchmark" in captured.out
    assert "two_run_orbit_mixture_v1" in captured.out


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
