from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np
from scipy.optimize import linprog

from .actions import all_actions, comb_action, complement, fixed_rank_action
from .defects import commutation_defect, commutation_defect_mixed, greedy_defect
from .dp_optimal import optimal_values
from .dp_policy import evaluate_balanced_policy, state_occupancy
from .lp_game import solve_adversary_dual, solve_minimax_step
from .policies import (
    comb_policy,
    fixed_rank_policy,
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
from .state import all_states, canon, packet_type


def _next_state_value(
    state: tuple[int, ...],
    action: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
) -> float:
    raw_next = tuple(state[index] + action[index] for index in range(len(state)))
    return float(min(raw_next, default=0)) + continuation[canon(raw_next)]


@dataclass(frozen=True)
class WeightedGreedyContribution:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    occupancy_probability: float
    local_defect: float
    contribution: float
    best_action: tuple[int, ...]
    best_edge_signature: tuple[int, ...]
    best_action_value: float
    policy_support: tuple[tuple[float, tuple[int, ...]], ...]
    policy_edge_signatures: tuple[tuple[float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class PacketWeightedGreedySummary:
    packet_type: tuple[int, ...]
    total_contribution: float
    occupancy_mass: float
    average_local_defect: float
    top_best_actions: tuple[tuple[str, float], ...]
    top_policy_supports: tuple[tuple[str, float], ...]
    top_best_edge_signatures: tuple[tuple[str, float], ...]
    top_policy_edge_signatures: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RegimeWeightedGreedySummary:
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    total_contribution: float
    occupancy_mass: float
    average_local_defect: float
    top_best_actions: tuple[tuple[str, float], ...]
    top_policy_supports: tuple[tuple[str, float], ...]
    top_best_edge_signatures: tuple[tuple[str, float], ...]
    top_policy_edge_signatures: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class LibraryOracleContribution:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    unrestricted_action: tuple[int, ...]
    unrestricted_score: float
    library_action: tuple[int, ...]
    library_score: float
    loss: float


@dataclass(frozen=True)
class LibraryOracleRegimeSummary:
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    total_loss: float
    occupancy_mass: float
    unrestricted_value: float
    library_value: float
    top_unrestricted_actions: tuple[tuple[str, float], ...]
    top_library_actions: tuple[tuple[str, float], ...]
    top_unrestricted_edge_signatures: tuple[tuple[str, float], ...]
    top_library_edge_signatures: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class BlockGapStats:
    interval: tuple[int, int]
    sum_gap: int
    max_gap: int
    length: int
    start: int
    end: int


@dataclass(frozen=True)
class OneRunOracleAnalysisRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    unrestricted_action: tuple[int, ...]
    unrestricted_score: float
    edge_signature: tuple[int, ...]
    block_stats: BlockGapStats
    is_min_sum_gap: bool
    is_min_max_gap: bool
    is_top_prefix: bool
    is_longest_min_sum_gap: bool


@dataclass(frozen=True)
class OneRunTieAnalysisRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    unrestricted_best_score: float
    optimal_one_run_count: int
    exists_min_sum_gap: bool
    exists_min_max_gap: bool
    exists_top_prefix: bool
    exists_prefix_plus_tail_anchor: bool
    smallest_sum_signature: tuple[int, ...]
    smallest_sum_stats: BlockGapStats
    topmost_signature: tuple[int, ...]
    topmost_stats: BlockGapStats
    longest_signature: tuple[int, ...]
    longest_stats: BlockGapStats


@dataclass(frozen=True)
class TopPrefixTieAnalysisRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    unrestricted_best_score: float
    optimal_lengths: tuple[int, ...]
    min_optimal_length: int
    max_optimal_length: int


@dataclass(frozen=True)
class TopPrefixOracleEvalResult:
    selector: str
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    selected_length_counts: tuple[tuple[int, int], ...]
    fallback_count: int


@dataclass(frozen=True)
class TopPrefixOracleLabelRow:
    horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    valid_lengths: tuple[int, ...]
    selected_length: int
    used_fallback: bool


@dataclass(frozen=True)
class WeightedTopPrefixOracleLabelRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    unrestricted_best_score: float
    valid_lengths: tuple[int, ...]
    selected_length: int
    used_fallback: bool

    @property
    def value_weight(self) -> float:
        return self.occupancy_probability * self.unrestricted_best_score


@dataclass(frozen=True)
class PolicyOccupancyDiffRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_a: float
    occupancy_b: float
    diff: float
    unrestricted_best_score: float
    value_diff_weight: float
    valid_lengths: tuple[int, ...]
    selected_length: int
    policy_a_length: int
    policy_b_length: int
    used_fallback: bool


@dataclass(frozen=True)
class TopPrefixNextStateDebtRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    current_valid_lengths: tuple[int, ...]
    current_selected_length: int
    policy_length: int
    candidate_length: int
    action_probability: float
    next_state: tuple[int, ...]
    next_packet_type: tuple[int, ...]
    next_packet_gaps: tuple[int, ...]
    next_adjacent_gaps: tuple[int, ...]
    next_valid_lengths: tuple[int, ...]
    next_selected_length: int
    next_policy_length: int
    next_policy_length_valid: bool
    next_unrestricted_best_score: float
    debt_weight: float
    used_fallback: bool
    next_used_fallback: bool


@dataclass(frozen=True)
class TopPrefixCandidateValueRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    policy_length: int
    policy_value: float
    best_length: int
    best_value: float
    regret: float
    candidate_scores: tuple[tuple[int, float], ...]

    @property
    def weighted_regret(self) -> float:
        return self.occupancy_probability * self.regret


@dataclass(frozen=True)
class TopPrefixRestrictedOptimalResult:
    length_set_name: str
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    selected_length_counts: tuple[tuple[int, int], ...]
    occupancy_selected_length_weights: tuple[tuple[int, float], ...]
    top_regime_selected_lengths: tuple[
        tuple[tuple[int, ...], tuple[int, ...], float, tuple[tuple[int, float], ...]],
        ...
    ]


@dataclass(frozen=True)
class EdgeRunRestrictedCandidate:
    intervals: tuple[tuple[int, int], ...]
    edge_signature: tuple[int, ...]
    action: tuple[int, ...]


@dataclass(frozen=True)
class EdgeRunRestrictedOptimalResult:
    max_runs: int
    library_size: int
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    selected_signature_counts: tuple[tuple[str, int], ...]
    occupancy_selected_signature_weights: tuple[tuple[str, float], ...]
    top_regime_selected_signatures: tuple[
        tuple[tuple[int, ...], tuple[int, ...], float, tuple[tuple[str, float], ...]],
        ...
    ]


@dataclass(frozen=True)
class LibraryLPRestrictedOptimalResult:
    library_name: str
    library_size: int
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    active_action_counts: tuple[tuple[str, int], ...]
    active_edge_signature_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class K3MotifSweepRow:
    T: int
    library_name: str
    library_size: int
    deterministic_value: float | None
    lp_value: float
    optimal_value: float

    @property
    def deterministic_gap(self) -> float | None:
        if self.deterministic_value is None:
            return None
        return self.optimal_value - self.deterministic_value

    @property
    def lp_gap(self) -> float:
        return self.optimal_value - self.lp_value


@dataclass(frozen=True)
class K9MotifLibrarySweepRow:
    T: int
    family_name: str
    library_size: int
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    top_active_edge_signatures: tuple[tuple[str, int], ...]
    top_active_orbit_keys: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class StrategyClassBenchmarkRow:
    k: int
    T: int
    library_name: str
    library_size: int
    value: float
    optimal_value: float
    gap: float
    normalized_gap: float
    exact: bool
    active_action_count: int
    active_edge_signature_count: int
    top_active_edge_signatures: tuple[tuple[str, int], ...]
    probability_matching_weighted_l1: float | None = None
    probability_matching_weighted_linf: float | None = None
    probability_matching_avg_l1: float | None = None
    probability_matching_avg_linf: float | None = None
    probability_matching_rows: int | None = None


@dataclass(frozen=True)
class StrategyClassRelativeBenchmarkRow:
    k: int
    T: int
    library_name: str
    library_size: int
    reference_library_name: str
    reference_value: float
    value: float
    gap_to_reference: float
    normalized_gap_to_reference: float
    active_action_count: int
    active_edge_signature_count: int
    top_active_edge_signatures: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExplicitPolicyBenchmarkRow:
    k: int
    T: int
    policy_name: str
    value: float
    reference_library_name: str | None
    reference_value: float | None
    gap_to_reference: float | None
    normalized_gap_to_reference: float | None


@dataclass(frozen=True)
class _StrategyClassBenchmarkCaseCache:
    k: int
    T: int
    state_layers: tuple[tuple[tuple[int, ...], ...], ...]
    optimal_layers: list[dict[tuple[int, ...], float]]
    optimal_value: float
    library_actions_by_name: dict[str, tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class ProbabilityMatchingInspectRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    learner_distribution: tuple[float, ...]
    winner_probabilities: tuple[float, ...]
    l1_error: float
    linf_error: float

    @property
    def weighted_l1_error(self) -> float:
        return self.occupancy_probability * self.l1_error

    @property
    def weighted_linf_error(self) -> float:
        return self.occupancy_probability * self.linf_error


@dataclass(frozen=True)
class NamedProbabilityMatchingInspectRow:
    time: int
    remaining_horizon: int
    named_state: tuple[int, ...]
    canonical_state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    lifted_learner_distribution: tuple[float, ...]
    winner_probabilities: tuple[float, ...]
    l1_error: float
    linf_error: float

    @property
    def weighted_l1_error(self) -> float:
        return self.occupancy_probability * self.l1_error

    @property
    def weighted_linf_error(self) -> float:
        return self.occupancy_probability * self.linf_error


@dataclass(frozen=True)
class NamedProbabilityMatchingResidualAggregate:
    key: tuple[object, ...]
    named_row_count: int
    total_occupancy: float
    weighted_l1_error: float
    weighted_linf_error: float
    avg_l1_error: float
    avg_linf_error: float
    max_linf_error: float
    representative_row: NamedProbabilityMatchingInspectRow


@dataclass(frozen=True)
class DualFaceRepairRow:
    original_row: NamedProbabilityMatchingInspectRow
    repaired_winner_probabilities: tuple[float, ...]
    repaired_l1_error: float
    repaired_linf_error: float
    active_candidate_count: int
    support: tuple[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
    success: bool
    message: str

    @property
    def original_weighted_l1_error(self) -> float:
        return self.original_row.weighted_l1_error

    @property
    def original_weighted_linf_error(self) -> float:
        return self.original_row.weighted_linf_error

    @property
    def repaired_weighted_l1_error(self) -> float:
        return self.original_row.occupancy_probability * self.repaired_l1_error

    @property
    def repaired_weighted_linf_error(self) -> float:
        return self.original_row.occupancy_probability * self.repaired_linf_error


@dataclass(frozen=True)
class LibraryLPDualInspectRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    value: float
    learner_distribution: tuple[float, ...]
    dual_value: float
    expected_action: tuple[float, ...]
    max_expected_action: float
    support: tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True)
class DualSupportOrbitSummary:
    orbit_key: tuple[int, ...]
    packet_sizes: tuple[int, ...]
    orbit_size: int
    support_count: int
    total_weight: float
    representative_action: tuple[int, ...]
    representative_edge_signature: tuple[int, ...]
    support_weights: tuple[float, ...]
    expected_action_contribution: tuple[float, ...]
    packet_expected_averages: tuple[float, ...]
    successor_packet_type_weights: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class TwoRunReplayTemplateRow:
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    support_action_count: int
    support_edge_signature_weights: tuple[tuple[str, float], ...]
    support_orbit_key_weights: tuple[tuple[tuple[int, ...], float], ...]
    successor_packet_type_weights: tuple[tuple[tuple[int, ...], float], ...]
    full_orbit_compatible: bool
    sparse_inside_orbit: bool
    run_profile: str
    template_signature: tuple[
        tuple[tuple[tuple[int, ...], str], ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[tuple[int, ...], str], ...],
    ]
    support: tuple[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True)
class CoarseTemplateAggregate:
    signature: object
    row_count: int
    total_occupancy: float
    horizons: tuple[int, ...]
    representative_state: tuple[int, ...]
    representative_packet_regimes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    representative_support: tuple[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]


def _format_action(action: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in action)


def _edge_signature(action: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        1 if action[index] != action[index + 1] else 0
        for index in range(len(action) - 1)
    )


def _format_edge_signature(sig: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in sig)


def _action_from_edge_signature(sig: tuple[int, ...], first_bit: int = 1) -> tuple[int, ...]:
    bits = [first_bit]
    for edge in sig:
        bits.append(bits[-1] if edge == 0 else 1 - bits[-1])
    return tuple(bits)


def _edge_signature_run_count(sig: tuple[int, ...]) -> int:
    runs = 0
    in_run = False
    for edge in sig:
        if edge == 1 and not in_run:
            runs += 1
            in_run = True
        elif edge == 0:
            in_run = False
    return runs


def _edge_run_intervals(sig: tuple[int, ...]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, edge in enumerate(sig):
        if edge == 1 and start is None:
            start = index
        elif edge == 0 and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(sig) - 1))
    return intervals


def _is_one_run(sig: tuple[int, ...]) -> bool:
    return len(_edge_run_intervals(sig)) == 1


def _gap_vector(state: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(state[index] - state[index + 1] for index in range(len(state) - 1))


def _all_edge_blocks(edge_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, end)
        for start in range(edge_count)
        for end in range(start, edge_count)
    )


def _block_gap_stats_from_interval(
    gaps: tuple[int, ...],
    interval: tuple[int, int],
) -> BlockGapStats:
    start, end = interval
    block_gaps = gaps[start : end + 1]
    return BlockGapStats(
        interval=interval,
        sum_gap=sum(block_gaps),
        max_gap=max(block_gaps),
        length=end - start + 1,
        start=start,
        end=end,
    )


def _block_gap_stats(
    state: tuple[int, ...],
    sig: tuple[int, ...],
) -> BlockGapStats | None:
    intervals = _edge_run_intervals(sig)
    if len(intervals) != 1:
        return None
    return _block_gap_stats_from_interval(_gap_vector(state), intervals[0])


def _prefix_plus_tail_anchor_signatures(k: int) -> set[tuple[int, ...]]:
    return {
        _edge_signature(action)
        for action in _prefix_plus_tail_anchor_action_library(k)
    }


def _top_prefix_signature(k: int, length: int) -> tuple[int, ...]:
    if length < 1 or length > max(k - 1, 0):
        raise ValueError("top-prefix length must lie in [1, k - 1]")
    return tuple(1 if index < length else 0 for index in range(k - 1))


def _top_prefix_action(k: int, length: int) -> tuple[int, ...]:
    return _action_from_edge_signature(_top_prefix_signature(k, length))


def _balanced_top_prefix_policy(k: int, length: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
    action = _top_prefix_action(k, length)
    return ((0.5, action), (0.5, complement(action)))


def _local_edge_action_library(k: int) -> tuple[tuple[int, ...], ...]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return ((),)

    actions: set[tuple[int, ...]] = set()
    for action in (comb_action(k),):
        actions.add(action)
        actions.add(complement(action))

    for edge_bits in range(1, 2 ** max(k - 1, 0)):
        sig = tuple((edge_bits >> index) & 1 for index in reversed(range(k - 1)))
        if _edge_signature_run_count(sig) > 2:
            continue
        action = _action_from_edge_signature(sig)
        actions.add(action)
        actions.add(complement(action))

    return tuple(sorted(actions, key=_format_action))


def _actions_from_edge_signatures(
    signatures: set[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    actions: set[tuple[int, ...]] = set()
    for sig in signatures:
        if not any(sig):
            continue
        action = _action_from_edge_signature(sig)
        actions.add(action)
        actions.add(complement(action))
    return tuple(sorted(actions, key=_format_action))


def _one_run_edge_action_library(k: int) -> tuple[tuple[int, ...], ...]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    signatures: set[tuple[int, ...]] = set()
    for edge_bits in range(1, 2 ** max(k - 1, 0)):
        sig = tuple((edge_bits >> index) & 1 for index in reversed(range(k - 1)))
        if _edge_signature_run_count(sig) <= 1:
            signatures.add(sig)
    return _actions_from_edge_signatures(signatures)


def _edge_run_restricted_candidates(k: int, max_runs: int) -> tuple[EdgeRunRestrictedCandidate, ...]:
    if k < 2:
        return ()
    candidates: list[EdgeRunRestrictedCandidate] = []
    for edge_bits in range(1, 2 ** (k - 1)):
        signature = tuple((edge_bits >> index) & 1 for index in reversed(range(k - 1)))
        intervals = tuple(_edge_run_intervals(signature))
        if not intervals or len(intervals) > max_runs:
            continue
        candidates.append(
            EdgeRunRestrictedCandidate(
                intervals=intervals,
                edge_signature=signature,
                action=_action_from_edge_signature(signature),
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: (candidate.intervals, candidate.edge_signature)))


def _one_run_restricted_candidates(k: int) -> tuple[EdgeRunRestrictedCandidate, ...]:
    return _edge_run_restricted_candidates(k, 1)


def _prefix_one_run_action_library(k: int) -> tuple[tuple[int, ...], ...]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    edge_count = max(k - 1, 0)
    signatures = {
        tuple(1 if index < run_length else 0 for index in range(edge_count))
        for run_length in range(1, edge_count + 1)
    }
    return _actions_from_edge_signatures(signatures)


def _prefix_plus_tail_anchor_action_library(k: int) -> tuple[tuple[int, ...], ...]:
    if k < 0:
        raise ValueError("k must be nonnegative")
    edge_count = max(k - 1, 0)
    signatures: set[tuple[int, ...]] = set()

    for prefix_length in range(1, edge_count):
        signatures.add(tuple(1 if index < prefix_length else 0 for index in range(edge_count)))

    for prefix_length in range(1, edge_count):
        for tail_length in range(1, edge_count - prefix_length):
            gap_length = edge_count - prefix_length - tail_length
            if gap_length < 1:
                continue
            signatures.add((1,) * prefix_length + (0,) * gap_length + (1,) * tail_length)

    return _actions_from_edge_signatures(signatures)


def _best_fixed_top_action_library(k: int, T: int, top_fixed_size: int) -> tuple[tuple[int, ...], ...]:
    if top_fixed_size <= 0:
        raise ValueError("top_fixed_size must be positive")

    zero = tuple(0 for _ in range(k))
    rows: list[tuple[float, tuple[int, ...]]] = []
    for size in range(1, k):
        for subset in combinations(range(1, k + 1), size):
            if 1 not in subset:
                continue
            action = fixed_rank_action(k, set(subset))
            policy_fn = fixed_rank_policy(set(subset))
            values = evaluate_balanced_policy(k, T, policy_fn)
            rows.append((values[T][zero], action))

    rows.sort(key=lambda row: (-row[0], _format_action(row[1])))
    actions: set[tuple[int, ...]] = set()
    for _, action in rows[:top_fixed_size]:
        actions.add(action)
        actions.add(complement(action))
    return tuple(sorted(actions, key=_format_action))


def _action_library(
    k: int,
    library_name: str,
    T: int | None = None,
    top_fixed_size: int = 20,
) -> tuple[tuple[int, ...], ...]:
    if library_name == "top_prefix_scale":
        actions: set[tuple[int, ...]] = set()
        for length in _candidate_scale_lengths(k):
            action = _top_prefix_action(k, length)
            actions.add(action)
            actions.add(complement(action))
        return tuple(sorted(actions, key=_format_action))
    if library_name == "top_prefix_all":
        actions = set(_prefix_one_run_action_library(k))
        return tuple(sorted(actions, key=_format_action))
    if library_name == "one_run":
        return _one_run_edge_action_library(k)
    if library_name == "two_run":
        actions: set[tuple[int, ...]] = set()
        for candidate in _edge_run_restricted_candidates(k, 2):
            actions.add(candidate.action)
            actions.add(complement(candidate.action))
        return tuple(sorted(actions, key=_format_action))
    if library_name == "one_run_edges":
        return _one_run_edge_action_library(k)
    if library_name == "prefix_one_run":
        return _prefix_one_run_action_library(k)
    if library_name == "prefix_plus_tail_anchor":
        return _prefix_plus_tail_anchor_action_library(k)
    if library_name == "local_edges":
        return _local_edge_action_library(k)
    if library_name == "all":
        return tuple(all_actions(k))
    if library_name == "best_fixed_top":
        if T is None:
            raise ValueError("T is required for best_fixed_top")
        return _best_fixed_top_action_library(k, T, top_fixed_size)
    raise ValueError(f"unknown library: {library_name}")


def _format_policy(policy: tuple[tuple[float, tuple[int, ...]], ...], max_items: int = 8) -> str:
    visible = [
        f"{_format_action(action)}:{probability:.3f}"
        for probability, action in policy
        if probability >= 1e-6
    ]
    if len(visible) <= max_items:
        return ", ".join(visible)
    return ", ".join(visible[:max_items]) + ", ..."


def _policy_edge_signatures(
    policy: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for probability, action in policy:
        totals[_edge_signature(action)] += probability
    return tuple(
        (probability, sig)
        for sig, probability in sorted(totals.items())
        if probability > 1e-12
    )


def _format_policy_edge_signatures(
    policy_edge_signatures: tuple[tuple[float, tuple[int, ...]], ...],
    max_items: int = 8,
) -> str:
    visible = [
        f"{_format_edge_signature(sig)}:{probability:.3f}"
        for probability, sig in policy_edge_signatures
        if probability >= 1e-6
    ]
    if len(visible) <= max_items:
        return ", ".join(visible)
    return ", ".join(visible[:max_items]) + ", ..."


def _packet_values(state: tuple[int, ...]) -> tuple[int, ...]:
    values: list[int] = []
    for value in state:
        if not values or value != values[-1]:
            values.append(value)
    return tuple(values)


def _packet_gaps(state: tuple[int, ...]) -> tuple[int, ...]:
    values = _packet_values(state)
    return tuple(values[index] - values[index + 1] for index in range(len(values) - 1))


def _packet_index_groups(state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not state:
        return ()
    groups: list[tuple[int, ...]] = []
    current: list[int] = [0]
    for index in range(1, len(state)):
        if state[index] == state[index - 1]:
            current.append(index)
            continue
        groups.append(tuple(current))
        current = [index]
    groups.append(tuple(current))
    return tuple(groups)


def _merge_policy_terms(
    terms: list[tuple[float, tuple[int, ...]]],
    tolerance: float = 1e-15,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for probability, action in terms:
        if probability > tolerance:
            totals[action] += probability
    total_probability = sum(totals.values())
    if total_probability <= 0:
        raise ValueError("policy has no positive-probability actions")
    return tuple(
        (probability / total_probability, action)
        for action, probability in sorted(totals.items(), key=lambda item: _format_action(item[0]))
        if probability > tolerance
    )


def _policy_from_weighted_actions(
    weighted_actions: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _merge_policy_terms([(float(weight), action) for weight, action in weighted_actions])


def _exchangeable_actions_for_orbit_key(
    state: tuple[int, ...],
    orbit_key: tuple[int, ...],
    max_runs: int | None = None,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    packet_groups = _packet_index_groups(canonical_state)
    if len(orbit_key) != len(packet_groups):
        raise ValueError(
            f"orbit key length {len(orbit_key)} does not match packet count {len(packet_groups)}"
        )

    packet_choices: list[tuple[tuple[int, ...], ...]] = []
    for group, ones_count in zip(packet_groups, orbit_key):
        if ones_count < 0 or ones_count > len(group):
            raise ValueError(f"orbit key {orbit_key} is incompatible with packet sizes {packet_type(canonical_state)}")
        packet_choices.append(tuple(combinations(group, ones_count)))

    actions: list[tuple[float, tuple[int, ...]]] = []
    all_choices = tuple(product(*packet_choices))
    if not all_choices:
        return ()
    base_probability = 1.0 / len(all_choices)
    for choice in all_choices:
        one_indices = {index for subset in choice for index in subset}
        action = tuple(1 if index in one_indices else 0 for index in range(len(canonical_state)))
        if max_runs is not None and _edge_signature_run_count(_edge_signature(action)) > max_runs:
            continue
        actions.append((base_probability, action))
    if not actions:
        return ()
    return _merge_policy_terms(actions)


def _exchangeable_orbit_mixture_policy(
    state: tuple[int, ...],
    weighted_orbit_keys: tuple[tuple[float, tuple[int, ...]], ...],
    max_runs: int | None = 2,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    terms: list[tuple[float, tuple[int, ...]]] = []
    for orbit_weight, orbit_key in weighted_orbit_keys:
        orbit_policy = _exchangeable_actions_for_orbit_key(state, orbit_key, max_runs=max_runs)
        if not orbit_policy and max_runs is not None:
            orbit_policy = _exchangeable_actions_for_orbit_key(state, orbit_key, max_runs=None)
        if not orbit_policy:
            raise ValueError(f"orbit key {orbit_key} has no compatible actions at state {state}")
        for action_probability, action in orbit_policy:
            terms.append((orbit_weight * action_probability, action))
    return _merge_policy_terms(terms)


def two_run_orbit_mixture_v1_policy(
    state: tuple[int, ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    k = len(canonical_state)
    if k != 9:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))

    ptype = packet_type(canonical_state)
    weighted_orbits_by_packet_type: dict[
        tuple[int, ...],
        tuple[tuple[float, tuple[int, ...]], ...],
    ] = {
        (9,): ((1.0, (4,)),),
        (4, 5): ((11.0 / 17.0, (2, 1)), (6.0 / 17.0, (1, 4))),
        (5, 4): ((4.0 / 5.0, (2, 2)), (1.0 / 5.0, (2, 0))),
        (2, 3, 4): ((6.0 / 7.0, (1, 1, 2)), (1.0 / 7.0, (0, 3, 0))),
        (2, 5, 2): (
            (5.0 / 11.0, (1, 2, 0)),
            (10.0 / 33.0, (1, 2, 1)),
            (5.0 / 33.0, (1, 2, 2)),
            (1.0 / 11.0, (0, 5, 0)),
        ),
        (1, 2, 4, 2): (
            (1.0 / 3.0, (0, 2, 0, 0)),
            (2.0 / 9.0, (1, 0, 2, 0)),
            (2.0 / 9.0, (1, 0, 2, 1)),
            (2.0 / 9.0, (0, 1, 4, 0)),
        ),
    }
    weighted_orbits = weighted_orbits_by_packet_type.get(ptype)
    if weighted_orbits is None:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))
    return _exchangeable_orbit_mixture_policy(canonical_state, weighted_orbits, max_runs=2)


@lru_cache(maxsize=1)
def _two_run_dual_support_replay_table_k9_T7(
    support_tolerance: float = 1e-8,
) -> dict[tuple[int, tuple[int, ...]], tuple[tuple[float, tuple[int, ...]], ...]]:
    _, rows = library_lp_dual_inspect_rows(9, 7, "two_run", support_tolerance=support_tolerance)
    table: dict[tuple[int, tuple[int, ...]], tuple[tuple[float, tuple[int, ...]], ...]] = {}
    for row in rows:
        table[(row.remaining_horizon, row.state)] = _policy_from_weighted_actions(
            tuple((weight, action) for weight, action, _ in row.support)
        )
    return table


def two_run_dual_support_replay_k9_T7_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    if len(canonical_state) != 9 or remaining_horizon < 1 or remaining_horizon > 7:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))
    table = _two_run_dual_support_replay_table_k9_T7()
    replay_policy = table.get((remaining_horizon, canonical_state))
    if replay_policy is None:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))
    return replay_policy


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    return tuple(int(piece.strip()) for piece in stripped.split(",") if piece.strip())


def _summarize_weighted_group(
    items: list[WeightedGreedyContribution],
    n: int,
) -> tuple[
    float,
    float,
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
]:
    total_contribution = sum(item.contribution for item in items)
    occupancy_mass = sum(item.occupancy_probability for item in items)
    weighted_defect_sum = sum(item.occupancy_probability * item.local_defect for item in items)
    average_local_defect = weighted_defect_sum / occupancy_mass if occupancy_mass > 0 else 0.0

    best_action_totals: dict[str, float] = defaultdict(float)
    policy_support_totals: dict[str, float] = defaultdict(float)
    best_edge_signature_totals: dict[str, float] = defaultdict(float)
    policy_edge_signature_totals: dict[str, float] = defaultdict(float)
    for item in items:
        best_action_totals[_format_action(item.best_action)] += item.contribution
        policy_support_totals[_format_policy(item.policy_support)] += item.contribution
        best_edge_signature_totals[_format_edge_signature(item.best_edge_signature)] += item.contribution
        policy_edge_signature_totals[
            _format_policy_edge_signatures(item.policy_edge_signatures)
        ] += item.contribution

    top_best_actions = tuple(
        sorted(best_action_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
    )
    top_policy_supports = tuple(
        sorted(policy_support_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
    )
    top_best_edge_signatures = tuple(
        sorted(best_edge_signature_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
    )
    top_policy_edge_signatures = tuple(
        sorted(policy_edge_signature_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
    )
    return (
        total_contribution,
        average_local_defect,
        top_best_actions,
        top_policy_supports,
        top_best_edge_signatures,
        top_policy_edge_signatures,
    )


def summarize_weighted_greedy_by_packet(
    contributions: list[WeightedGreedyContribution],
    n: int = 10,
) -> list[PacketWeightedGreedySummary]:
    grouped: dict[tuple[int, ...], list[WeightedGreedyContribution]] = defaultdict(list)
    for item in contributions:
        grouped[item.packet_type].append(item)

    summaries: list[PacketWeightedGreedySummary] = []
    for ptype, items in grouped.items():
        occupancy_mass = sum(item.occupancy_probability for item in items)
        (
            total_contribution,
            average_local_defect,
            top_best_actions,
            top_policy_supports,
            top_best_edge_signatures,
            top_policy_edge_signatures,
        ) = _summarize_weighted_group(items, n)
        summaries.append(
            PacketWeightedGreedySummary(
                packet_type=ptype,
                total_contribution=total_contribution,
                occupancy_mass=occupancy_mass,
                average_local_defect=average_local_defect,
                top_best_actions=top_best_actions,
                top_policy_supports=top_policy_supports,
                top_best_edge_signatures=top_best_edge_signatures,
                top_policy_edge_signatures=top_policy_edge_signatures,
            )
        )

    summaries.sort(key=lambda item: item.total_contribution, reverse=True)
    return summaries


def summarize_weighted_greedy_by_regime(
    contributions: list[WeightedGreedyContribution],
    n: int = 10,
) -> list[RegimeWeightedGreedySummary]:
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[WeightedGreedyContribution]] = defaultdict(list)
    for item in contributions:
        grouped[(item.packet_type, _packet_gaps(item.state))].append(item)

    summaries: list[RegimeWeightedGreedySummary] = []
    for (ptype, gaps), items in grouped.items():
        occupancy_mass = sum(item.occupancy_probability for item in items)
        (
            total_contribution,
            average_local_defect,
            top_best_actions,
            top_policy_supports,
            top_best_edge_signatures,
            top_policy_edge_signatures,
        ) = _summarize_weighted_group(items, n)
        summaries.append(
            RegimeWeightedGreedySummary(
                packet_type=ptype,
                packet_gaps=gaps,
                total_contribution=total_contribution,
                occupancy_mass=occupancy_mass,
                average_local_defect=average_local_defect,
                top_best_actions=top_best_actions,
                top_policy_supports=top_policy_supports,
                top_best_edge_signatures=top_best_edge_signatures,
                top_policy_edge_signatures=top_policy_edge_signatures,
            )
        )

    summaries.sort(key=lambda item: item.total_contribution, reverse=True)
    return summaries


def filter_weighted_greedy_contributions(
    contributions: list[WeightedGreedyContribution],
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
) -> list[WeightedGreedyContribution]:
    filtered: list[WeightedGreedyContribution] = []
    for item in contributions:
        if packet_type_filter is not None and item.packet_type != packet_type_filter:
            continue
        if packet_gaps_filter is not None and _packet_gaps(item.state) != packet_gaps_filter:
            continue
        filtered.append(item)
    return filtered


def _top_weighted_actions(
    contributions: list[LibraryOracleContribution],
    action_getter,
    weight_getter,
    n: int,
) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = defaultdict(float)
    for item in contributions:
        totals[_format_action(action_getter(item))] += weight_getter(item)
    return tuple(sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n])


def _top_weighted_edge_signatures(
    contributions: list[LibraryOracleContribution],
    action_getter,
    weight_getter,
    n: int,
) -> tuple[tuple[str, float], ...]:
    totals: dict[str, float] = defaultdict(float)
    for item in contributions:
        totals[_format_edge_signature(_edge_signature(action_getter(item)))] += weight_getter(item)
    return tuple(sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n])


def library_oracle(
    k: int,
    T: int,
    library_name: str,
    occupancy_policy_fn,
    top_fixed_size: int = 20,
) -> tuple[float, float, float, list[LibraryOracleContribution]]:
    occupancy = state_occupancy(k, T, occupancy_policy_fn)
    optimal = optimal_values(k, T)
    action_list = all_actions(k)
    library_actions = _action_library(k, library_name, T=T, top_fixed_size=top_fixed_size)

    total_unrestricted_value = 0.0
    total_library_value = 0.0
    contributions: list[LibraryOracleContribution] = []

    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")

            def score(action: tuple[int, ...]) -> float:
                return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

            unrestricted_action = max(action_list, key=score)
            library_action = max(library_actions, key=score)
            unrestricted_score = score(unrestricted_action)
            library_score = score(library_action)
            loss = max(0.0, unrestricted_score - library_score)
            weighted_unrestricted_score = occupancy_probability * unrestricted_score
            weighted_library_score = occupancy_probability * library_score

            total_unrestricted_value += weighted_unrestricted_score
            total_library_value += weighted_library_score
            contributions.append(
                LibraryOracleContribution(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    occupancy_probability=occupancy_probability,
                    unrestricted_action=unrestricted_action,
                    unrestricted_score=unrestricted_score,
                    library_action=library_action,
                    library_score=library_score,
                    loss=occupancy_probability * loss,
                )
            )

    total_loss = total_unrestricted_value - total_library_value
    contributions.sort(key=lambda item: item.loss, reverse=True)
    return total_unrestricted_value, total_library_value, total_loss, contributions


def summarize_library_oracle_by_regime(
    contributions: list[LibraryOracleContribution],
    n: int = 10,
) -> list[LibraryOracleRegimeSummary]:
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[LibraryOracleContribution]] = defaultdict(list)
    for item in contributions:
        grouped[(item.packet_type, item.packet_gaps)].append(item)

    summaries: list[LibraryOracleRegimeSummary] = []
    for (ptype, gaps), items in grouped.items():
        occupancy_mass = sum(item.occupancy_probability for item in items)
        unrestricted_value = sum(
            item.occupancy_probability * item.unrestricted_score
            for item in items
        )
        library_value = sum(
            item.occupancy_probability * item.library_score
            for item in items
        )
        total_loss = sum(item.loss for item in items)
        summaries.append(
            LibraryOracleRegimeSummary(
                packet_type=ptype,
                packet_gaps=gaps,
                total_loss=total_loss,
                occupancy_mass=occupancy_mass,
                unrestricted_value=unrestricted_value,
                library_value=library_value,
                top_unrestricted_actions=_top_weighted_actions(
                    items,
                    lambda item: item.unrestricted_action,
                    lambda item: item.occupancy_probability * item.unrestricted_score,
                    n,
                ),
                top_library_actions=_top_weighted_actions(
                    items,
                    lambda item: item.library_action,
                    lambda item: item.occupancy_probability * item.library_score,
                    n,
                ),
                top_unrestricted_edge_signatures=_top_weighted_edge_signatures(
                    items,
                    lambda item: item.unrestricted_action,
                    lambda item: item.occupancy_probability * item.unrestricted_score,
                    n,
                ),
                top_library_edge_signatures=_top_weighted_edge_signatures(
                    items,
                    lambda item: item.library_action,
                    lambda item: item.occupancy_probability * item.library_score,
                    n,
                ),
            )
        )

    summaries.sort(key=lambda item: item.total_loss, reverse=True)
    return summaries


def one_run_oracle_analysis(
    k: int,
    T: int,
    occupancy_policy_fn,
) -> tuple[list[OneRunOracleAnalysisRow], float, float, float, float]:
    occupancy = state_occupancy(k, T, occupancy_policy_fn)
    optimal = optimal_values(k, T)
    action_list = all_actions(k)
    rows: list[OneRunOracleAnalysisRow] = []
    total_occupancy_weight = 0.0
    total_value_weight = 0.0
    one_run_occupancy_weight = 0.0
    one_run_value_weight = 0.0

    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")

            def score(action: tuple[int, ...]) -> float:
                return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

            unrestricted_action = max(action_list, key=score)
            unrestricted_score = score(unrestricted_action)
            value_weight = occupancy_probability * unrestricted_score
            total_occupancy_weight += occupancy_probability
            total_value_weight += value_weight

            sig = _edge_signature(unrestricted_action)
            stats = _block_gap_stats(state, sig)
            if stats is None:
                continue

            gaps = _gap_vector(state)
            all_stats = tuple(
                _block_gap_stats_from_interval(gaps, interval)
                for interval in _all_edge_blocks(len(gaps))
            )
            min_sum_gap = min(item.sum_gap for item in all_stats)
            min_max_gap = min(item.max_gap for item in all_stats)
            longest_min_sum_length = max(
                item.length for item in all_stats if item.sum_gap == min_sum_gap
            )
            one_run_occupancy_weight += occupancy_probability
            one_run_value_weight += value_weight
            rows.append(
                OneRunOracleAnalysisRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    occupancy_probability=occupancy_probability,
                    unrestricted_action=unrestricted_action,
                    unrestricted_score=unrestricted_score,
                    edge_signature=sig,
                    block_stats=stats,
                    is_min_sum_gap=stats.sum_gap == min_sum_gap,
                    is_min_max_gap=stats.max_gap == min_max_gap,
                    is_top_prefix=stats.start == 0,
                    is_longest_min_sum_gap=(
                        stats.sum_gap == min_sum_gap
                        and stats.length == longest_min_sum_length
                    ),
                )
            )

    return (
        rows,
        total_occupancy_weight,
        total_value_weight,
        one_run_occupancy_weight,
        one_run_value_weight,
    )


def one_run_tie_analysis(
    k: int,
    T: int,
    occupancy_policy_fn,
    tolerance: float = 1e-9,
) -> tuple[list[OneRunTieAnalysisRow], float, float, float, float]:
    occupancy = state_occupancy(k, T, occupancy_policy_fn)
    optimal = optimal_values(k, T)
    action_list = all_actions(k)
    one_run_actions = _one_run_edge_action_library(k)
    prefix_plus_tail_signatures = _prefix_plus_tail_anchor_signatures(k)
    rows: list[OneRunTieAnalysisRow] = []
    total_occupancy_weight = 0.0
    total_value_weight = 0.0
    optimal_one_run_occupancy_weight = 0.0
    optimal_one_run_value_weight = 0.0

    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")

            def score(action: tuple[int, ...]) -> float:
                return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

            unrestricted_best_score = max(score(action) for action in action_list)
            value_weight = occupancy_probability * unrestricted_best_score
            total_occupancy_weight += occupancy_probability
            total_value_weight += value_weight

            optimal_one_run_items: list[tuple[tuple[int, ...], BlockGapStats]] = []
            for action in one_run_actions:
                if score(action) < unrestricted_best_score - tolerance:
                    continue
                sig = _edge_signature(action)
                stats = _block_gap_stats(state, sig)
                if stats is not None:
                    optimal_one_run_items.append((sig, stats))

            if not optimal_one_run_items:
                continue

            gaps = _gap_vector(state)
            all_stats = tuple(
                _block_gap_stats_from_interval(gaps, interval)
                for interval in _all_edge_blocks(len(gaps))
            )
            min_sum_gap = min(item.sum_gap for item in all_stats)
            min_max_gap = min(item.max_gap for item in all_stats)

            smallest_sum_sig, smallest_sum_stats = min(
                optimal_one_run_items,
                key=lambda item: (
                    item[1].sum_gap,
                    item[1].start,
                    -item[1].length,
                    _format_edge_signature(item[0]),
                ),
            )
            topmost_sig, topmost_stats = min(
                optimal_one_run_items,
                key=lambda item: (
                    item[1].start,
                    item[1].sum_gap,
                    -item[1].length,
                    _format_edge_signature(item[0]),
                ),
            )
            longest_sig, longest_stats = min(
                optimal_one_run_items,
                key=lambda item: (
                    -item[1].length,
                    item[1].sum_gap,
                    item[1].start,
                    _format_edge_signature(item[0]),
                ),
            )

            optimal_one_run_occupancy_weight += occupancy_probability
            optimal_one_run_value_weight += value_weight
            rows.append(
                OneRunTieAnalysisRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    occupancy_probability=occupancy_probability,
                    unrestricted_best_score=unrestricted_best_score,
                    optimal_one_run_count=len(optimal_one_run_items),
                    exists_min_sum_gap=any(
                        stats.sum_gap == min_sum_gap
                        for _, stats in optimal_one_run_items
                    ),
                    exists_min_max_gap=any(
                        stats.max_gap == min_max_gap
                        for _, stats in optimal_one_run_items
                    ),
                    exists_top_prefix=any(
                        stats.start == 0
                        for _, stats in optimal_one_run_items
                    ),
                    exists_prefix_plus_tail_anchor=any(
                        sig in prefix_plus_tail_signatures
                        for sig, _ in optimal_one_run_items
                    ),
                    smallest_sum_signature=smallest_sum_sig,
                    smallest_sum_stats=smallest_sum_stats,
                    topmost_signature=topmost_sig,
                    topmost_stats=topmost_stats,
                    longest_signature=longest_sig,
                    longest_stats=longest_stats,
                )
            )

    return (
        rows,
        total_occupancy_weight,
        total_value_weight,
        optimal_one_run_occupancy_weight,
        optimal_one_run_value_weight,
    )


def top_prefix_tie_analysis(
    k: int,
    T: int,
    occupancy_policy_fn,
    tolerance: float = 1e-9,
) -> tuple[list[TopPrefixTieAnalysisRow], float, float, float, float]:
    occupancy = state_occupancy(k, T, occupancy_policy_fn)
    optimal = optimal_values(k, T)
    action_list = all_actions(k)
    rows: list[TopPrefixTieAnalysisRow] = []
    total_occupancy_weight = 0.0
    total_value_weight = 0.0
    top_prefix_occupancy_weight = 0.0
    top_prefix_value_weight = 0.0

    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")

            def score(action: tuple[int, ...]) -> float:
                return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

            unrestricted_best_score = max(score(action) for action in action_list)
            value_weight = occupancy_probability * unrestricted_best_score
            total_occupancy_weight += occupancy_probability
            total_value_weight += value_weight

            optimal_lengths = []
            for length in range(1, k):
                action = _top_prefix_action(k, length)
                if max(score(action), score(complement(action))) >= unrestricted_best_score - tolerance:
                    optimal_lengths.append(length)
            optimal_lengths = tuple(optimal_lengths)
            if not optimal_lengths:
                continue

            top_prefix_occupancy_weight += occupancy_probability
            top_prefix_value_weight += value_weight
            rows.append(
                TopPrefixTieAnalysisRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_probability=occupancy_probability,
                    unrestricted_best_score=unrestricted_best_score,
                    optimal_lengths=optimal_lengths,
                    min_optimal_length=min(optimal_lengths),
                    max_optimal_length=max(optimal_lengths),
                )
            )

    return (
        rows,
        total_occupancy_weight,
        total_value_weight,
        top_prefix_occupancy_weight,
        top_prefix_value_weight,
    )


def _optimal_top_prefix_lengths(
    k: int,
    state: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
    tolerance: float,
) -> tuple[tuple[int, ...], bool]:
    if k < 2:
        return (), False

    action_list = all_actions(k)
    q_by_action = {
        action: _next_state_value(state, action, continuation)
        for action in action_list
    }
    solution = solve_minimax_step(q_by_action, k)
    if not solution.success:
        raise RuntimeError(f"LP failed at state {state}: {solution.message}")

    def score(action: tuple[int, ...]) -> float:
        return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

    unrestricted_best_score = max(score(action) for action in action_list)
    length_scores = tuple(
        (
            length,
            max(score(_top_prefix_action(k, length)), score(complement(_top_prefix_action(k, length)))),
        )
        for length in range(1, k)
    )
    valid_lengths = tuple(
        length
        for length, length_score in length_scores
        if length_score >= unrestricted_best_score - tolerance
    )
    if valid_lengths:
        return valid_lengths, False

    best_top_prefix_score = max(length_score for _, length_score in length_scores)
    fallback_lengths = tuple(
        length
        for length, length_score in length_scores
        if length_score >= best_top_prefix_score - tolerance
    )
    return fallback_lengths, True


def _unrestricted_best_score(
    k: int,
    state: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
) -> float:
    action_list = all_actions(k)
    q_by_action = {
        action: _next_state_value(state, action, continuation)
        for action in action_list
    }
    solution = solve_minimax_step(q_by_action, k)
    if not solution.success:
        raise RuntimeError(f"LP failed at state {state}: {solution.message}")

    def score(action: tuple[int, ...]) -> float:
        return q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k))

    return max(score(action) for action in action_list)


def _select_top_prefix_length(lengths: tuple[int, ...], selector: str) -> int:
    if not lengths:
        raise ValueError("cannot select from an empty length set")
    ordered = tuple(sorted(lengths))
    if selector == "min_valid":
        return ordered[0]
    if selector == "max_valid":
        return ordered[-1]
    if selector == "median_valid":
        return ordered[(len(ordered) - 1) // 2]
    if selector == "chase_preferred":
        target = 3
        return min(ordered, key=lambda length: (abs(length - target), length))
    raise ValueError(f"unknown top-prefix oracle selector: {selector}")


def evaluate_top_prefix_oracle(
    k: int,
    T: int,
    selector: str,
    tolerance: float = 1e-9,
) -> TopPrefixOracleEvalResult:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 2:
        raise ValueError("k must be at least 2 for top-prefix oracle evaluation")

    zero = tuple(0 for _ in range(k))
    optimal = optimal_values(k, T)
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in state_layers[T]}
    ]
    selected_length_counts: dict[int, int] = defaultdict(int)
    fallback_count = 0

    for horizon in range(1, T + 1):
        policy_continuation = values[horizon - 1]
        optimal_continuation = optimal[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            valid_lengths, used_fallback = _optimal_top_prefix_lengths(
                k,
                state,
                optimal_continuation,
                tolerance,
            )
            length = _select_top_prefix_length(valid_lengths, selector)
            selected_length_counts[length] += 1
            if used_fallback:
                fallback_count += 1

            expected_value = 0.0
            for probability, action in _balanced_top_prefix_policy(k, length):
                expected_value += probability * _next_state_value(state, action, policy_continuation)
            current[state] = expected_value - 0.5
        values.append(current)

    value = values[T][zero]
    optimal_value = optimal[T][zero]
    gap = optimal_value - value
    normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
    return TopPrefixOracleEvalResult(
        selector=selector,
        value=value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=normalized_gap,
        selected_length_counts=tuple(sorted(selected_length_counts.items())),
        fallback_count=fallback_count,
    )


def top_prefix_oracle_labels(
    k: int,
    T: int,
    selector: str,
    tolerance: float = 1e-9,
) -> list[TopPrefixOracleLabelRow]:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 2:
        raise ValueError("k must be at least 2 for top-prefix oracle labels")

    optimal = optimal_values(k, T)
    rows: list[TopPrefixOracleLabelRow] = []
    for horizon in range(1, T + 1):
        continuation = optimal[horizon - 1]
        for state in all_states(k, T - horizon):
            valid_lengths, used_fallback = _optimal_top_prefix_lengths(
                k,
                state,
                continuation,
                tolerance,
            )
            selected_length = _select_top_prefix_length(valid_lengths, selector)
            rows.append(
                TopPrefixOracleLabelRow(
                    horizon=horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    valid_lengths=valid_lengths,
                    selected_length=selected_length,
                    used_fallback=used_fallback,
                )
            )
    return rows


def _top_prefix_oracle_occupancy(
    k: int,
    T: int,
    selector: str,
    optimal: list[dict[tuple[int, ...], float]],
    tolerance: float,
) -> list[dict[tuple[int, ...], float]]:
    zero = tuple(0 for _ in range(k))
    occupancy: list[dict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    for time in range(T):
        continuation = optimal[T - time - 1]
        for state, probability in occupancy[time].items():
            valid_lengths, _ = _optimal_top_prefix_lengths(k, state, continuation, tolerance)
            length = _select_top_prefix_length(valid_lengths, selector)
            for action_probability, action in _balanced_top_prefix_policy(k, length):
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * action_probability
    return [dict(layer) for layer in occupancy]


def weighted_top_prefix_oracle_labels(
    k: int,
    T: int,
    selector: str,
    occupancy_policy_name: str | None = "oracle",
    tolerance: float = 1e-9,
) -> list[WeightedTopPrefixOracleLabelRow]:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 2:
        raise ValueError("k must be at least 2 for weighted top-prefix oracle labels")

    optimal = optimal_values(k, T)
    if occupancy_policy_name is None or occupancy_policy_name == "oracle":
        occupancy = _top_prefix_oracle_occupancy(k, T, selector, optimal, tolerance)
    else:
        policies = _policy_registry(k)
        if occupancy_policy_name not in policies:
            raise ValueError(f"unknown occupancy policy: {occupancy_policy_name}")
        occupancy = state_occupancy(k, T, policies[occupancy_policy_name])

    rows: list[WeightedTopPrefixOracleLabelRow] = []
    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            valid_lengths, used_fallback = _optimal_top_prefix_lengths(
                k,
                state,
                continuation,
                tolerance,
            )
            selected_length = _select_top_prefix_length(valid_lengths, selector)
            rows.append(
                WeightedTopPrefixOracleLabelRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_probability=occupancy_probability,
                    unrestricted_best_score=_unrestricted_best_score(k, state, continuation),
                    valid_lengths=valid_lengths,
                    selected_length=selected_length,
                    used_fallback=used_fallback,
                )
            )
    return rows


def _policy_registry(k: int) -> dict[str, object]:
    policies: dict[str, object] = {
        "comb": comb_policy,
        "packet_balanced_partition": packet_balanced_partition_policy,
        "packet_frontier": packet_frontier_policy,
        "packet_minimal_frontier": packet_minimal_frontier_policy,
        "top_prefix_chase_length": top_prefix_chase_length_policy,
        "top_prefix_gap_sum_long": top_prefix_gap_sum_long_policy,
        "top_prefix_gap_sum_short": top_prefix_gap_sum_short_policy,
        "top_prefix_longest": top_prefix_longest_policy,
        "top_prefix_shortest": top_prefix_shortest_policy,
        "top_prefix_three_regime": top_prefix_three_regime_policy,
        "top_prefix_three_regime_v2": top_prefix_three_regime_v2_policy,
        "top_prefix_three_regime_v3": top_prefix_three_regime_v3_policy,
        "top_prefix_three_regime_v4": top_prefix_three_regime_v4_policy,
        "top_prefix_three_regime_v5": top_prefix_three_regime_v5_policy,
        "top_prefix_three_regime_v6": top_prefix_three_regime_v6_policy,
        "top_prefix_three_regime_v7": top_prefix_three_regime_v7_policy,
        "top_prefix_tie_mimic": top_prefix_tie_mimic_policy,
        "two_run_orbit_mixture_v1": two_run_orbit_mixture_v1_policy,
    }
    if k == 5:
        policies["chase5"] = fixed_rank_policy({1, 3})
        policies["packet_regime5"] = packet_regime5_policy
        policies["packet_regime5b"] = packet_regime5b_policy
        policies["packet_regime5c"] = packet_regime5c_policy
    for size in range(1, k):
        for subset in combinations(range(1, k + 1), size):
            if 1 not in subset:
                continue
            name = "fixed_" + "".join(str(rank) for rank in subset)
            policies[name] = fixed_rank_policy(set(subset))
    return policies


def compare_values(k: int, T: int) -> None:
    zero = tuple(0 for _ in range(k))
    optimal = optimal_values(k, T)
    optimal_zero = optimal[T][zero]
    print(f"Exact minimax value at zero: {optimal_zero:.6f}")
    print("policy                V_star     V_policy        gap    gap/sqrt(T)   V_policy/sqrt(T)")
    for name, policy_fn in sorted(_policy_registry(k).items()):
        values = evaluate_balanced_policy(k, T, policy_fn)
        policy_zero = values[T][zero]
        gap = optimal_zero - policy_zero
        normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
        normalized_policy = policy_zero / (T ** 0.5 if T > 0 else 1.0)
        print(
            f"{name:20s} {optimal_zero:10.6f} {policy_zero:12.6f} {gap:10.6f}"
            f" {normalized_gap:12.6f} {normalized_policy:16.6f}"
        )


def print_top_greedy_defects(k: int, T: int, policy_name: str, n: int = 20) -> None:
    policies = _policy_registry(k)
    policy_fn = policies[policy_name]
    policy_values = evaluate_balanced_policy(k, T, policy_fn)
    action_list = all_actions(k)
    rows = []
    for state in all_states(k, max(T - 1, 0)):
        remaining = T - max(state, default=0)
        if remaining <= 0:
            continue
        prev = policy_values[remaining - 1]
        defect = greedy_defect(k, state, prev, policy_fn, actions=action_list)
        q_by_action = {
            action: _next_state_value(state, action, prev)
            for action in action_list
        }
        solution = solve_minimax_step(q_by_action, k)
        best_action = max(
            action_list,
            key=lambda action: q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k)),
        )
        rows.append((defect, state, packet_type(state), best_action))

    rows.sort(reverse=True)
    print(f"Top greedy defects for {policy_name}, k={k}, T={T}")
    print()
    print("state               packet type      defect      best action")
    for defect, state, ptype, best_action in rows[:n]:
        action_bits = "".join(str(bit) for bit in best_action)
        print(f"{str(state):18s} {str(ptype):14s} {defect:10.6f}   {action_bits}")


def print_top_commutation_defects(k: int, T: int, policy_name: str, n: int = 20) -> None:
    policies = _policy_registry(k)
    policy_fn = policies[policy_name]
    rows = []
    for state in all_states(k, max(T - 2, 0)):
        for action in all_actions(k, include_trivial=False):
            defect, best_action = commutation_defect(k, state, action, policy_fn)
            rows.append((defect, state, action, best_action))

    rows.sort(reverse=True)
    print(f"Top commutation defects for {policy_name}, k={k}, T={T}")
    print()
    print("state               first      defect      best delayed")
    for defect, state, action, best_action in rows[:n]:
        first_bits = "".join(str(bit) for bit in action)
        delayed_bits = "".join(str(bit) for bit in best_action)
        print(f"{str(state):18s} {first_bits:8s} {defect:10.6f}   {delayed_bits}")


def print_top_mixed_commutation_defects(k: int, T: int, policy_name: str, n: int = 20) -> None:
    policies = _policy_registry(k)
    policy_fn = policies[policy_name]
    rows = []
    for state in all_states(k, max(T - 2, 0)):
        for action in all_actions(k, include_trivial=False):
            solution = commutation_defect_mixed(k, state, action, policy_fn)
            if not solution.success:
                raise RuntimeError(
                    f"Mixed commutation LP failed at state {state}, action {action}: {solution.message}"
                )
            rows.append((solution.best_tv, state, action, solution.weights_by_action))

    rows.sort(reverse=True)
    print(f"Top mixed commutation defects for {policy_name}, k={k}, T={T}")
    print()
    print("state               first      defect      best delayed mix")
    for defect, state, action, weights in rows[:n]:
        first_bits = "".join(str(bit) for bit in action)
        mixture = ", ".join(
            f"{''.join(str(bit) for bit in delayed_action)}:{weight:.3f}"
            for delayed_action, weight in sorted(weights.items())
        )
        print(f"{str(state):18s} {first_bits:8s} {defect:10.6f}   {mixture}")


def occupation_weighted_greedy_defects(k: int, T: int, policy_fn) -> tuple[float, list[WeightedGreedyContribution]]:
    occupancy = state_occupancy(k, T, policy_fn)
    policy_values = evaluate_balanced_policy(k, T, policy_fn)
    contributions: list[WeightedGreedyContribution] = []
    action_list = all_actions(k)

    for time in range(T):
        remaining_horizon = T - time
        continuation = policy_values[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in action_list
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            best_action = max(
                action_list,
                key=lambda action: q_by_action[action] - sum(solution.p[index] * action[index] for index in range(k)),
            )
            best_action_value = q_by_action[best_action] - sum(
                solution.p[index] * best_action[index] for index in range(k)
            )
            local_defect = greedy_defect(k, state, continuation, policy_fn)
            policy_support = tuple(policy_fn(state))
            policy_edge_signatures = _policy_edge_signatures(policy_support)
            contributions.append(
                WeightedGreedyContribution(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    occupancy_probability=occupancy_probability,
                    local_defect=local_defect,
                    contribution=occupancy_probability * local_defect,
                    best_action=best_action,
                    best_edge_signature=_edge_signature(best_action),
                    best_action_value=best_action_value,
                    policy_support=policy_support,
                    policy_edge_signatures=policy_edge_signatures,
                )
            )

    total_weighted_defect = sum(item.contribution for item in contributions)
    contributions.sort(key=lambda item: item.contribution, reverse=True)
    return total_weighted_defect, contributions


def print_occupation_weighted_greedy_defects(k: int, T: int, policy_name: str, n: int = 20) -> None:
    policies = _policy_registry(k)
    total_weighted_defect, contributions = occupation_weighted_greedy_defects(k, T, policies[policy_name])
    normalized_total = total_weighted_defect / (T ** 0.5 if T > 0 else 1.0)

    print(f"Occupation-weighted greedy defects for {policy_name}, k={k}, T={T}")
    print()
    print(f"total weighted defect: {total_weighted_defect:.6f}")
    print(f"total weighted defect / sqrt(T): {normalized_total:.6f}")
    print()
    print(
        "time  remaining  state               packet type      occupancy    defect"
        "    contribution  best   edge   value     policy edges      policy support"
    )
    for item in contributions[:n]:
        print(
            f"{item.time:4d} {item.remaining_horizon:10d} {str(item.state):18s}"
            f" {str(item.packet_type):14s} {item.occupancy_probability:10.6f}"
            f" {item.local_defect:9.6f} {item.contribution:13.6f}"
            f"  {_format_action(item.best_action):5s}"
            f"  {_format_edge_signature(item.best_edge_signature):5s}"
            f"  {item.best_action_value:8.6f}"
            f"  {_format_policy_edge_signatures(item.policy_edge_signatures):16s}"
            f"  {_format_policy(item.policy_support)}"
        )


def print_weighted_greedy_by_packet(k: int, T: int, policy_name: str, n: int = 10) -> None:
    policies = _policy_registry(k)
    total_weighted_defect, contributions = occupation_weighted_greedy_defects(k, T, policies[policy_name])
    normalized_total = total_weighted_defect / (T ** 0.5 if T > 0 else 1.0)
    summaries = summarize_weighted_greedy_by_packet(contributions, n=n)

    print(f"Occupation-weighted greedy defects by packet type for {policy_name}, k={k}, T={T}")
    print()
    print(f"total weighted defect: {total_weighted_defect:.6f}")
    print(f"total weighted defect / sqrt(T): {normalized_total:.6f}")
    print()
    for summary in summaries:
        normalized_contribution = summary.total_contribution / (T ** 0.5 if T > 0 else 1.0)
        print(f"packet type: {summary.packet_type}")
        print(f"  total contribution: {summary.total_contribution:.6f}")
        print(f"  total contribution / sqrt(T): {normalized_contribution:.6f}")
        print(f"  occupancy mass: {summary.occupancy_mass:.6f}")
        print(f"  avg local defect (occupancy-weighted): {summary.average_local_defect:.6f}")
        print("  top best actions by contribution:")
        for action, contribution in summary.top_best_actions:
            print(f"    {action}: {contribution:.6f}")
        print("  top policy supports by contribution:")
        for support, contribution in summary.top_policy_supports:
            print(f"    {support}: {contribution:.6f}")
        print("  top best edge signatures by contribution:")
        for signature, contribution in summary.top_best_edge_signatures:
            print(f"    {signature}: {contribution:.6f}")
        print("  top policy edge signatures by contribution:")
        for signatures, contribution in summary.top_policy_edge_signatures:
            print(f"    {signatures}: {contribution:.6f}")
        print()


def print_weighted_greedy_by_regime(k: int, T: int, policy_name: str, n: int = 10) -> None:
    policies = _policy_registry(k)
    total_weighted_defect, contributions = occupation_weighted_greedy_defects(k, T, policies[policy_name])
    normalized_total = total_weighted_defect / (T ** 0.5 if T > 0 else 1.0)
    summaries = summarize_weighted_greedy_by_regime(contributions, n=n)

    print(f"Occupation-weighted greedy defects by regime for {policy_name}, k={k}, T={T}")
    print()
    print(f"total weighted defect: {total_weighted_defect:.6f}")
    print(f"total weighted defect / sqrt(T): {normalized_total:.6f}")
    print()
    for summary in summaries:
        normalized_contribution = summary.total_contribution / (T ** 0.5 if T > 0 else 1.0)
        print(f"packet type: {summary.packet_type}")
        print(f"packet gaps: {summary.packet_gaps}")
        print(f"  total contribution: {summary.total_contribution:.6f}")
        print(f"  total contribution / sqrt(T): {normalized_contribution:.6f}")
        print(f"  occupancy mass: {summary.occupancy_mass:.6f}")
        print(f"  avg local defect (occupancy-weighted): {summary.average_local_defect:.6f}")
        print("  top best actions by contribution:")
        for action, contribution in summary.top_best_actions:
            print(f"    {action}: {contribution:.6f}")
        print("  top policy supports by contribution:")
        for support, contribution in summary.top_policy_supports:
            print(f"    {support}: {contribution:.6f}")
        print("  top best edge signatures by contribution:")
        for signature, contribution in summary.top_best_edge_signatures:
            print(f"    {signature}: {contribution:.6f}")
        print("  top policy edge signatures by contribution:")
        for signatures, contribution in summary.top_policy_edge_signatures:
            print(f"    {signatures}: {contribution:.6f}")
        print()


def print_weighted_greedy_filter(
    k: int,
    T: int,
    policy_name: str,
    packet_type_filter: tuple[int, ...] | None,
    packet_gaps_filter: tuple[int, ...] | None,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    total_weighted_defect, contributions = occupation_weighted_greedy_defects(k, T, policies[policy_name])
    filtered = filter_weighted_greedy_contributions(
        contributions,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
    )
    filtered_total = sum(item.contribution for item in filtered)

    print(f"Filtered occupation-weighted greedy defects for {policy_name}, k={k}, T={T}")
    print()
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"matching rows: {len(filtered)}")
    print(f"filtered total contribution: {filtered_total:.6f}")
    print(f"overall total weighted defect: {total_weighted_defect:.6f}")
    print()
    print(
        "time  remaining  state               occupancy    defect    contribution"
        "  best   edge   policy edges      policy support"
    )
    for item in filtered[:n]:
        print(
            f"{item.time:4d} {item.remaining_horizon:10d} {str(item.state):18s}"
            f" {item.occupancy_probability:10.6f} {item.local_defect:9.6f}"
            f" {item.contribution:13.6f}  {_format_action(item.best_action):5s}"
            f"  {_format_edge_signature(item.best_edge_signature):5s}"
            f"  {_format_policy_edge_signatures(item.policy_edge_signatures):16s}"
            f"  {_format_policy(item.policy_support)}"
        )


def print_best_fixed(k: int, T: int, n: int = 20) -> None:
    zero = tuple(0 for _ in range(k))
    optimal = optimal_values(k, T)
    optimal_zero = optimal[T][zero]
    rows = []

    for size in range(1, k):
        for subset in combinations(range(1, k + 1), size):
            if 1 not in subset:
                continue
            action = fixed_rank_action(k, set(subset))
            policy_fn = fixed_rank_policy(set(subset))
            values = evaluate_balanced_policy(k, T, policy_fn)
            policy_zero = values[T][zero]
            gap = optimal_zero - policy_zero
            normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
            rows.append((policy_zero, action, gap, normalized_gap))

    rows.sort(key=lambda row: (-row[0], _format_action(row[1])))

    print(f"Best fixed-rank policies, k={k}, T={T}")
    print()
    print(f"Exact minimax value at zero: {optimal_zero:.6f}")
    print()
    print("rank  action   edge    value        gap    gap/sqrt(T)")
    for rank, (value, action, gap, normalized_gap) in enumerate(rows[:n], start=1):
        print(
            f"{rank:4d}  {_format_action(action):7s}"
            f"  {_format_edge_signature(_edge_signature(action)):7s}"
            f"  {value:10.6f} {gap:10.6f} {normalized_gap:12.6f}"
        )


def print_library_oracle(
    k: int,
    T: int,
    library_name: str,
    occupancy_policy_name: str,
    top_fixed_size: int = 20,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    if occupancy_policy_name not in policies:
        raise ValueError(f"unknown occupancy policy: {occupancy_policy_name}")

    library_actions = _action_library(k, library_name, T=T, top_fixed_size=top_fixed_size)
    total_unrestricted_value, total_library_value, total_loss, contributions = library_oracle(
        k,
        T,
        library_name,
        policies[occupancy_policy_name],
        top_fixed_size=top_fixed_size,
    )
    normalized_loss = total_loss / (T ** 0.5 if T > 0 else 1.0)
    summaries = summarize_library_oracle_by_regime(contributions, n=n)

    print(f"Library oracle for {library_name}, k={k}, T={T}")
    print()
    print(f"occupancy policy: {occupancy_policy_name}")
    print(f"library size: {len(library_actions)}")
    if library_name == "best_fixed_top":
        print(f"top fixed size: {top_fixed_size}")
    print(f"total unrestricted greedy value: {total_unrestricted_value:.6f}")
    print(f"total library-restricted greedy value: {total_library_value:.6f}")
    print(f"loss from restricting to library: {total_loss:.6f}")
    print(f"loss from restricting to library / sqrt(T): {normalized_loss:.6f}")
    print()
    print("Top regimes where library fails:")
    for summary in summaries[:n]:
        if summary.total_loss <= 1e-12:
            continue
        print(f"packet type: {summary.packet_type}")
        print(f"packet gaps: {summary.packet_gaps}")
        print(f"  total loss: {summary.total_loss:.6f}")
        print(f"  occupancy mass: {summary.occupancy_mass:.6f}")
        print(f"  unrestricted value: {summary.unrestricted_value:.6f}")
        print(f"  library value: {summary.library_value:.6f}")
        print("  top unrestricted actions by value:")
        for action, value in summary.top_unrestricted_actions:
            print(f"    {action}: {value:.6f}")
        print("  top library actions by value:")
        for action, value in summary.top_library_actions:
            print(f"    {action}: {value:.6f}")
        print("  top unrestricted edge signatures by value:")
        for signature, value in summary.top_unrestricted_edge_signatures:
            print(f"    {signature}: {value:.6f}")
        print("  top library edge signatures by value:")
        for signature, value in summary.top_library_edge_signatures:
            print(f"    {signature}: {value:.6f}")
        print()

    print("Top selected library actions overall:")
    top_library_actions = _top_weighted_actions(
        contributions,
        lambda item: item.library_action,
        lambda item: item.occupancy_probability * item.library_score,
        n,
    )
    for action, value in top_library_actions:
        print(f"  {action} edge={_format_edge_signature(_edge_signature(tuple(int(bit) for bit in action)))} value={value:.6f}")


def print_one_run_oracle_analysis(
    k: int,
    T: int,
    occupancy_policy_name: str,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    if occupancy_policy_name not in policies:
        raise ValueError(f"unknown occupancy policy: {occupancy_policy_name}")

    (
        rows,
        total_occupancy_weight,
        total_value_weight,
        one_run_occupancy_weight,
        one_run_value_weight,
    ) = one_run_oracle_analysis(k, T, policies[occupancy_policy_name])

    def occupancy_weight(row: OneRunOracleAnalysisRow) -> float:
        return row.occupancy_probability

    def value_weight(row: OneRunOracleAnalysisRow) -> float:
        return row.occupancy_probability * row.unrestricted_score

    def weighted_rate(predicate, weight_getter) -> float:
        denominator = sum(weight_getter(row) for row in rows)
        if denominator <= 0:
            return 0.0
        return sum(weight_getter(row) for row in rows if predicate(row)) / denominator

    def print_rate(label: str, predicate) -> None:
        print(
            f"{label:28s}"
            f" occupancy={100.0 * weighted_rate(predicate, occupancy_weight):8.3f}%"
            f" value={100.0 * weighted_rate(predicate, value_weight):8.3f}%"
        )

    interval_totals: dict[tuple[int, int], float] = defaultdict(float)
    length_totals: dict[int, float] = defaultdict(float)
    signature_totals: dict[str, float] = defaultdict(float)
    regime_non_min_sum: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for row in rows:
        weight = value_weight(row)
        interval_totals[row.block_stats.interval] += weight
        length_totals[row.block_stats.length] += weight
        signature_totals[_format_edge_signature(row.edge_signature)] += weight
        if not row.is_min_sum_gap:
            regime_non_min_sum[(row.packet_type, row.packet_gaps)] += weight

    print(f"One-run oracle analysis, k={k}, T={T}")
    print()
    print(f"occupancy policy: {occupancy_policy_name}")
    print(f"visited rows: {sum(len(layer) for layer in state_occupancy(k, T, policies[occupancy_policy_name])[:-1])}")
    print(f"one-run rows: {len(rows)}")
    print(f"one-run occupancy coverage: {one_run_occupancy_weight / total_occupancy_weight if total_occupancy_weight > 0 else 0.0:.6f}")
    print(f"one-run value coverage: {one_run_value_weight / total_value_weight if total_value_weight > 0 else 0.0:.6f}")
    print()
    print("Selected block tests among one-run unrestricted best actions:")
    print_rate("min sum gap", lambda row: row.is_min_sum_gap)
    print_rate("min max gap", lambda row: row.is_min_max_gap)
    print_rate("top prefix", lambda row: row.is_top_prefix)
    print_rate("longest min sum gap", lambda row: row.is_longest_min_sum_gap)
    print()
    print("Top selected intervals by value weight:")
    for interval, weight in sorted(interval_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {interval}: {weight:.6f}")
    print()
    print("Top selected block lengths by value weight:")
    for length, weight in sorted(length_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {length}: {weight:.6f}")
    print()
    print("Top selected edge signatures by value weight:")
    for signature, weight in sorted(signature_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {signature}: {weight:.6f}")
    print()
    print("Top regimes where selected block is not min-sum by value weight:")
    for (ptype, gaps), weight in sorted(regime_non_min_sum.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  packet type={ptype} gaps={gaps}: {weight:.6f}")


def print_one_run_tie_analysis(
    k: int,
    T: int,
    occupancy_policy_name: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    if occupancy_policy_name not in policies:
        raise ValueError(f"unknown occupancy policy: {occupancy_policy_name}")

    (
        rows,
        total_occupancy_weight,
        total_value_weight,
        optimal_one_run_occupancy_weight,
        optimal_one_run_value_weight,
    ) = one_run_tie_analysis(k, T, policies[occupancy_policy_name], tolerance=tolerance)

    def occupancy_weight(row: OneRunTieAnalysisRow) -> float:
        return row.occupancy_probability

    def value_weight(row: OneRunTieAnalysisRow) -> float:
        return row.occupancy_probability * row.unrestricted_best_score

    def weighted_rate(predicate, weight_getter) -> float:
        denominator = sum(weight_getter(row) for row in rows)
        if denominator <= 0:
            return 0.0
        return sum(weight_getter(row) for row in rows if predicate(row)) / denominator

    def print_rate(label: str, predicate) -> None:
        print(
            f"{label:34s}"
            f" occupancy={100.0 * weighted_rate(predicate, occupancy_weight):8.3f}%"
            f" value={100.0 * weighted_rate(predicate, value_weight):8.3f}%"
        )

    def print_canonical_distribution(
        title: str,
        signature_getter,
        stats_getter,
    ) -> None:
        interval_totals: dict[tuple[int, int], float] = defaultdict(float)
        signature_totals: dict[str, float] = defaultdict(float)
        length_totals: dict[int, float] = defaultdict(float)
        for row in rows:
            weight = value_weight(row)
            stats = stats_getter(row)
            interval_totals[stats.interval] += weight
            signature_totals[_format_edge_signature(signature_getter(row))] += weight
            length_totals[stats.length] += weight

        print(title)
        print("  intervals:")
        for interval, weight in sorted(interval_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
            print(f"    {interval}: {weight:.6f}")
        print("  signatures:")
        for signature, weight in sorted(signature_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
            print(f"    {signature}: {weight:.6f}")
        print("  lengths:")
        for length, weight in sorted(length_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
            print(f"    {length}: {weight:.6f}")
        print()

    visited_rows = sum(len(layer) for layer in state_occupancy(k, T, policies[occupancy_policy_name])[:-1])
    print(f"One-run tie analysis, k={k}, T={T}")
    print()
    print(f"occupancy policy: {occupancy_policy_name}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"visited rows: {visited_rows}")
    print(f"rows with optimal one-run action: {len(rows)}")
    print(
        "optimal one-run occupancy coverage:"
        f" {optimal_one_run_occupancy_weight / total_occupancy_weight if total_occupancy_weight > 0 else 0.0:.6f}"
    )
    print(
        "optimal one-run value coverage:"
        f" {optimal_one_run_value_weight / total_value_weight if total_value_weight > 0 else 0.0:.6f}"
    )
    print()
    print("Existence tests among states with optimal one-run actions:")
    print_rate("exists min-sum-gap block", lambda row: row.exists_min_sum_gap)
    print_rate("exists min-max-gap block", lambda row: row.exists_min_max_gap)
    print_rate("exists top-prefix block", lambda row: row.exists_top_prefix)
    print_rate(
        "exists prefix-plus-tail-anchor",
        lambda row: row.exists_prefix_plus_tail_anchor,
    )
    print()
    print_canonical_distribution(
        "Canonical representative: smallest sum_gap",
        lambda row: row.smallest_sum_signature,
        lambda row: row.smallest_sum_stats,
    )
    print_canonical_distribution(
        "Canonical representative: topmost start",
        lambda row: row.topmost_signature,
        lambda row: row.topmost_stats,
    )
    print_canonical_distribution(
        "Canonical representative: longest block",
        lambda row: row.longest_signature,
        lambda row: row.longest_stats,
    )


def print_top_prefix_tie_analysis(
    k: int,
    T: int,
    occupancy_policy_name: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    if occupancy_policy_name not in policies:
        raise ValueError(f"unknown occupancy policy: {occupancy_policy_name}")

    (
        rows,
        total_occupancy_weight,
        total_value_weight,
        top_prefix_occupancy_weight,
        top_prefix_value_weight,
    ) = top_prefix_tie_analysis(k, T, policies[occupancy_policy_name], tolerance=tolerance)

    def value_weight(row: TopPrefixTieAnalysisRow) -> float:
        return row.occupancy_probability * row.unrestricted_best_score

    length_set_totals: dict[tuple[int, ...], float] = defaultdict(float)
    min_length_totals: dict[int, float] = defaultdict(float)
    max_length_totals: dict[int, float] = defaultdict(float)
    multiple_regime_totals: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    regime_length_set_totals: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        dict[tuple[int, ...], float],
    ] = defaultdict(lambda: defaultdict(float))
    raw_rows = sorted(rows, key=value_weight, reverse=True)
    for row in rows:
        weight = value_weight(row)
        length_set_totals[row.optimal_lengths] += weight
        min_length_totals[row.min_optimal_length] += weight
        max_length_totals[row.max_optimal_length] += weight
        regime_key = (row.packet_type, row.packet_gaps)
        regime_length_set_totals[regime_key][row.optimal_lengths] += weight
        if len(row.optimal_lengths) > 1:
            multiple_regime_totals[regime_key] += weight

    visited_rows = sum(len(layer) for layer in state_occupancy(k, T, policies[occupancy_policy_name])[:-1])
    print(f"Top-prefix tie analysis, k={k}, T={T}")
    print()
    print(f"occupancy policy: {occupancy_policy_name}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"visited rows: {visited_rows}")
    print(f"rows with optimal top-prefix action: {len(rows)}")
    print(
        "optimal top-prefix occupancy coverage:"
        f" {top_prefix_occupancy_weight / total_occupancy_weight if total_occupancy_weight > 0 else 0.0:.6f}"
    )
    print(
        "optimal top-prefix value coverage:"
        f" {top_prefix_value_weight / total_value_weight if total_value_weight > 0 else 0.0:.6f}"
    )
    print()

    print("Top optimal L sets by value weight:")
    for lengths, weight in sorted(length_set_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {lengths}: {weight:.6f}")
    print()
    print("Top min optimal L by value weight:")
    for length, weight in sorted(min_length_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {length}: {weight:.6f}")
    print()
    print("Top max optimal L by value weight:")
    for length, weight in sorted(max_length_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {length}: {weight:.6f}")
    print()
    print("Top regimes where multiple L are optimal by value weight:")
    for (ptype, gaps), weight in sorted(multiple_regime_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        length_sets = regime_length_set_totals[(ptype, gaps)]
        rendered_sets = ", ".join(
            f"{lengths}:{set_weight:.6f}"
            for lengths, set_weight in sorted(length_sets.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        )
        print(f"  packet type={ptype} gaps={gaps}: {weight:.6f} [{rendered_sets}]")
    print()
    print("Raw top rows by value weight:")
    for row in raw_rows[:n]:
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
            f" L={row.optimal_lengths}"
            f" weight={value_weight(row):.6f}"
        )


def print_top_prefix_oracle_eval(
    k: int,
    T: int,
    selector: str,
    tolerance: float = 1e-9,
) -> None:
    result = evaluate_top_prefix_oracle(k, T, selector, tolerance=tolerance)
    print(f"Top-prefix oracle eval, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"V_star: {result.optimal_value:.6f}")
    print(f"V_policy: {result.value:.6f}")
    print(f"gap: {result.gap:.6f}")
    print(f"gap/sqrt(T): {result.normalized_gap:.6f}")
    print(f"fallback states: {result.fallback_count}")
    print()
    print("Selected L counts across DP states:")
    for length, count in result.selected_length_counts:
        print(f"  {length}: {count}")


def print_top_prefix_oracle_labels(
    k: int,
    T: int,
    selector: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = top_prefix_oracle_labels(k, T, selector, tolerance=tolerance)

    regime_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[TopPrefixOracleLabelRow]] = defaultdict(list)
    selected_totals: dict[int, int] = defaultdict(int)
    valid_set_totals: dict[tuple[int, ...], int] = defaultdict(int)
    fallback_count = 0
    for row in rows:
        regime_rows[(row.packet_type, row.packet_gaps)].append(row)
        selected_totals[row.selected_length] += 1
        valid_set_totals[row.valid_lengths] += 1
        if row.used_fallback:
            fallback_count += 1

    print(f"Top-prefix oracle labels, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print(f"fallback rows: {fallback_count}")
    print()
    print("Selected L histogram:")
    for length, count in sorted(selected_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {length}: {count}")
    print()
    print("Valid L set histogram:")
    for lengths, count in sorted(valid_set_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        print(f"  {lengths}: {count}")
    print()
    print("Top regimes by DP-state count:")
    for (ptype, gaps), items in sorted(regime_rows.items(), key=lambda pair: (-len(pair[1]), pair[0]))[:n]:
        selected_hist: dict[int, int] = defaultdict(int)
        valid_hist: dict[tuple[int, ...], int] = defaultdict(int)
        regime_fallback_count = 0
        for item in items:
            selected_hist[item.selected_length] += 1
            valid_hist[item.valid_lengths] += 1
            if item.used_fallback:
                regime_fallback_count += 1
        rendered_selected = ", ".join(
            f"{length}:{count}"
            for length, count in sorted(selected_hist.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        )
        rendered_valid = ", ".join(
            f"{lengths}:{count}"
            for lengths, count in sorted(valid_hist.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        )
        print(
            f"  packet type={ptype} gaps={gaps} count={len(items)}"
            f" fallback={regime_fallback_count}"
            f" selected=[{rendered_selected}]"
            f" valid=[{rendered_valid}]"
        )
    print()
    print("Raw rows with largest horizons:")
    for row in sorted(rows, key=lambda item: (-item.horizon, item.state))[:n]:
        marker = " fallback" if row.used_fallback else ""
        print(
            f"  horizon={row.horizon:2d} state={row.state}"
            f" ptype={row.packet_type} packet_gaps={row.packet_gaps}"
            f" adjacent_gaps={row.adjacent_gaps}"
            f" valid={row.valid_lengths} selected={row.selected_length}{marker}"
        )


def _top_weighted_histogram(
    rows,
    key_getter,
    weight_getter,
    n: int,
) -> tuple[tuple[object, float], ...]:
    totals: dict[object, float] = defaultdict(float)
    for row in rows:
        totals[key_getter(row)] += weight_getter(row)
    return tuple(sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n])


def _top_prefix_length_from_policy(policy: tuple[tuple[float, tuple[int, ...]], ...] | list[tuple[float, tuple[int, ...]]]) -> int:
    signatures = {
        _edge_signature(action)
        for probability, action in policy
        if probability > 1e-12
    }
    if len(signatures) != 1:
        raise ValueError(f"policy support is not a single edge signature: {signatures}")
    (signature,) = tuple(signatures)
    if not any(signature):
        return 0
    intervals = _edge_run_intervals(signature)
    if intervals != [(0, sum(signature) - 1)]:
        raise ValueError(f"policy edge signature is not top-prefix: {_format_edge_signature(signature)}")
    return sum(signature)


def print_top_prefix_policy_vs_oracle_labels(
    k: int,
    T: int,
    policy_name: str,
    selector: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    policies = _policy_registry(k)
    if policy_name not in policies:
        raise ValueError(f"unknown policy: {policy_name}")
    policy_fn = policies[policy_name]
    rows = weighted_top_prefix_oracle_labels(
        k,
        T,
        selector,
        occupancy_policy_name=policy_name,
        tolerance=tolerance,
    )

    row_policy_lengths: list[tuple[WeightedTopPrefixOracleLabelRow, int]] = [
        (row, _top_prefix_length_from_policy(policy_fn(row.state)))
        for row in rows
    ]
    invalid_items = [
        (row, policy_length)
        for row, policy_length in row_policy_lengths
        if policy_length not in row.valid_lengths
    ]
    mismatch_items = [
        (row, policy_length)
        for row, policy_length in row_policy_lengths
        if policy_length != row.selected_length
    ]

    invalid_occupancy = sum(row.occupancy_probability for row, _ in invalid_items)
    invalid_value = sum(row.value_weight for row, _ in invalid_items)
    mismatch_occupancy = sum(row.occupancy_probability for row, _ in mismatch_items)
    mismatch_value = sum(row.value_weight for row, _ in mismatch_items)

    print(f"Top-prefix policy vs oracle labels, k={k}, T={T}")
    print()
    print(f"policy: {policy_name}")
    print(f"selector: {selector}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print(f"invalid occupancy: {invalid_occupancy:.6f}")
    print(f"invalid value weight: {invalid_value:.6f}")
    print(f"mismatch occupancy: {mismatch_occupancy:.6f}")
    print(f"mismatch value weight: {mismatch_value:.6f}")
    print()

    grouped_invalid: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[WeightedTopPrefixOracleLabelRow, int]]] = defaultdict(list)
    for row, policy_length in invalid_items:
        grouped_invalid[(row.packet_type, row.packet_gaps)].append((row, policy_length))

    print("Top invalid regimes by value weight:")
    for (ptype, gaps), items in sorted(
        grouped_invalid.items(),
        key=lambda pair: (-sum(row.value_weight for row, _ in pair[1]), pair[0]),
    )[:n]:
        occupancy_mass = sum(row.occupancy_probability for row, _ in items)
        value_weight = sum(row.value_weight for row, _ in items)
        policy_hist = _top_weighted_histogram(
            items,
            lambda item: item[1],
            lambda item: item[0].value_weight,
            5,
        )
        valid_hist = _top_weighted_histogram(
            items,
            lambda item: item[0].valid_lengths,
            lambda item: item[0].value_weight,
            5,
        )
        adjacent_examples = tuple(
            key
            for key, _ in _top_weighted_histogram(
                items,
                lambda item: item[0].adjacent_gaps,
                lambda item: item[0].value_weight,
                3,
            )
        )
        rendered_policy = ", ".join(f"{length}:{weight:.6f}" for length, weight in policy_hist)
        rendered_valid = ", ".join(f"{lengths}:{weight:.6f}" for lengths, weight in valid_hist)
        print(
            f"  packet type={ptype} gaps={gaps}"
            f" occupancy={occupancy_mass:.6f}"
            f" value={value_weight:.6f}"
            f" adjacent_examples={adjacent_examples}"
            f" policy_L=[{rendered_policy}]"
            f" oracle_valid=[{rendered_valid}]"
        )
    print()
    print("Raw top invalid rows:")
    for row, policy_length in sorted(invalid_items, key=lambda item: item[0].value_weight, reverse=True)[:n]:
        marker = " fallback" if row.used_fallback else ""
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
            f" prob={row.occupancy_probability:.6f}"
            f" value_weight={row.value_weight:.6f}"
            f" policy_L={policy_length}"
            f" selected={row.selected_length}"
            f" valid={row.valid_lengths}{marker}"
        )


def policy_occupancy_diff(
    k: int,
    T: int,
    policy_a_name: str,
    policy_b_name: str,
    selector: str,
    tolerance: float = 1e-9,
) -> list[PolicyOccupancyDiffRow]:
    policies = _policy_registry(k)
    if policy_a_name not in policies:
        raise ValueError(f"unknown policy A: {policy_a_name}")
    if policy_b_name not in policies:
        raise ValueError(f"unknown policy B: {policy_b_name}")

    policy_a = policies[policy_a_name]
    policy_b = policies[policy_b_name]
    occupancy_a = state_occupancy(k, T, policy_a)
    occupancy_b = state_occupancy(k, T, policy_b)
    optimal = optimal_values(k, T)
    rows: list[PolicyOccupancyDiffRow] = []

    for time in range(T):
        remaining_horizon = T - time
        continuation = optimal[remaining_horizon - 1]
        states = set(occupancy_a[time]) | set(occupancy_b[time])
        for state in states:
            probability_a = occupancy_a[time].get(state, 0.0)
            probability_b = occupancy_b[time].get(state, 0.0)
            diff = probability_b - probability_a
            if abs(diff) <= 1e-15:
                continue
            unrestricted_best_score = _unrestricted_best_score(k, state, continuation)
            valid_lengths, used_fallback = _optimal_top_prefix_lengths(
                k,
                state,
                continuation,
                tolerance,
            )
            rows.append(
                PolicyOccupancyDiffRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_a=probability_a,
                    occupancy_b=probability_b,
                    diff=diff,
                    unrestricted_best_score=unrestricted_best_score,
                    value_diff_weight=diff * unrestricted_best_score,
                    valid_lengths=valid_lengths,
                    selected_length=_select_top_prefix_length(valid_lengths, selector),
                    policy_a_length=_top_prefix_length_from_policy(policy_a(state)),
                    policy_b_length=_top_prefix_length_from_policy(policy_b(state)),
                    used_fallback=used_fallback,
                )
            )
    return rows


def print_policy_occupancy_diff(
    k: int,
    T: int,
    policy_a_name: str,
    policy_b_name: str,
    selector: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = policy_occupancy_diff(
        k,
        T,
        policy_a_name,
        policy_b_name,
        selector,
        tolerance=tolerance,
    )
    positive_shift = sum(row.diff for row in rows if row.diff > 0.0)
    negative_shift = sum(row.diff for row in rows if row.diff < 0.0)

    print(f"Policy occupancy diff, k={k}, T={T}")
    print()
    print(f"policy A: {policy_a_name}")
    print(f"policy B: {policy_b_name}")
    print(f"selector: {selector}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print(f"total positive occupancy shift A->B: {positive_shift:.6f}")
    print(f"total negative occupancy shift A->B: {negative_shift:.6f}")
    print()

    def print_rows(title: str, selected_rows: list[PolicyOccupancyDiffRow]) -> None:
        print(title)
        for row in selected_rows[:n]:
            marker = " fallback" if row.used_fallback else ""
            print(
                f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
                f" state={row.state} ptype={row.packet_type}"
                f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
                f" occ_a={row.occupancy_a:.6f} occ_b={row.occupancy_b:.6f}"
                f" diff={row.diff:.6f} value_diff={row.value_diff_weight:.6f}"
                f" valid={row.valid_lengths} selected={row.selected_length}"
                f" A_L={row.policy_a_length} B_L={row.policy_b_length}{marker}"
            )
        print()

    gained = sorted(
        (row for row in rows if row.diff > 0.0),
        key=lambda row: (-row.value_diff_weight, row.time, row.state),
    )
    lost = sorted(
        (row for row in rows if row.diff < 0.0),
        key=lambda row: (row.value_diff_weight, row.time, row.state),
    )
    print_rows("Top states gained by policy B by value-diff weight:", gained)
    print_rows("Top states lost by policy B by value-diff weight:", lost)


def _candidate_scale_lengths(k: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                length
                for length in (1, 3, 5, k - 3, k - 1)
                if 1 <= length <= k - 1
            }
        )
    )


def _top_prefix_candidate_lengths(k: int, length_set: str) -> tuple[int, ...]:
    if length_set == "scale":
        return _candidate_scale_lengths(k)
    if length_set == "all":
        return tuple(range(1, k))
    raise ValueError(f"unknown top-prefix length set: {length_set}")


def top_prefix_next_state_debt(
    k: int,
    T: int,
    policy_name: str,
    selector: str,
    tolerance: float = 1e-9,
) -> list[TopPrefixNextStateDebtRow]:
    policies = _policy_registry(k)
    if policy_name not in policies:
        raise ValueError(f"unknown policy: {policy_name}")
    policy = policies[policy_name]
    occupancy = state_occupancy(k, T, policy)
    optimal = optimal_values(k, T)
    candidate_lengths = _candidate_scale_lengths(k)
    rows: list[TopPrefixNextStateDebtRow] = []

    for time in range(T):
        remaining_horizon = T - time
        if remaining_horizon <= 1:
            continue
        continuation = optimal[remaining_horizon - 1]
        next_continuation = optimal[remaining_horizon - 2] if remaining_horizon >= 2 else optimal[0]
        for state, occupancy_probability in occupancy[time].items():
            current_valid_lengths, used_fallback = _optimal_top_prefix_lengths(
                k,
                state,
                continuation,
                tolerance,
            )
            current_selected_length = _select_top_prefix_length(current_valid_lengths, selector)
            policy_length = _top_prefix_length_from_policy(policy(state))
            for candidate_length in candidate_lengths:
                for action_probability, action in _balanced_top_prefix_policy(k, candidate_length):
                    next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                    next_valid_lengths, next_used_fallback = _optimal_top_prefix_lengths(
                        k,
                        next_state,
                        next_continuation,
                        tolerance,
                    )
                    next_policy_length = _top_prefix_length_from_policy(policy(next_state))
                    next_unrestricted_best_score = _unrestricted_best_score(k, next_state, next_continuation)
                    next_policy_length_valid = next_policy_length in next_valid_lengths
                    transition_weight = occupancy_probability * action_probability
                    rows.append(
                        TopPrefixNextStateDebtRow(
                            time=time,
                            remaining_horizon=remaining_horizon,
                            state=state,
                            packet_type=packet_type(state),
                            packet_gaps=_packet_gaps(state),
                            adjacent_gaps=_gap_vector(state),
                            occupancy_probability=occupancy_probability,
                            current_valid_lengths=current_valid_lengths,
                            current_selected_length=current_selected_length,
                            policy_length=policy_length,
                            candidate_length=candidate_length,
                            action_probability=action_probability,
                            next_state=next_state,
                            next_packet_type=packet_type(next_state),
                            next_packet_gaps=_packet_gaps(next_state),
                            next_adjacent_gaps=_gap_vector(next_state),
                            next_valid_lengths=next_valid_lengths,
                            next_selected_length=_select_top_prefix_length(next_valid_lengths, selector),
                            next_policy_length=next_policy_length,
                            next_policy_length_valid=next_policy_length_valid,
                            next_unrestricted_best_score=next_unrestricted_best_score,
                            debt_weight=0.0 if next_policy_length_valid else transition_weight * next_unrestricted_best_score,
                            used_fallback=used_fallback,
                            next_used_fallback=next_used_fallback,
                        )
                    )
    return rows


def print_top_prefix_next_state_debt(
    k: int,
    T: int,
    policy_name: str,
    selector: str,
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = top_prefix_next_state_debt(
        k,
        T,
        policy_name,
        selector,
        tolerance=tolerance,
    )

    print(f"Top-prefix next-state debt, k={k}, T={T}")
    print()
    print(f"policy: {policy_name}")
    print(f"selector: {selector}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print()

    print("Candidate L cleanup debt by value:")
    candidate_rows: dict[int, list[TopPrefixNextStateDebtRow]] = defaultdict(list)
    for row in rows:
        candidate_rows[row.candidate_length].append(row)
    for length, items in sorted(candidate_rows.items()):
        total_transition_mass = sum(row.occupancy_probability * row.action_probability for row in items)
        invalid_mass = sum(
            row.occupancy_probability * row.action_probability
            for row in items
            if not row.next_policy_length_valid
        )
        debt = sum(row.debt_weight for row in items)
        print(
            f"  L={length}: debt={debt:.6f}"
            f" invalid_mass={invalid_mass:.6f}"
            f" transition_mass={total_transition_mass:.6f}"
        )
    print()

    grouped: dict[
        tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]],
        list[TopPrefixNextStateDebtRow],
    ] = defaultdict(list)
    for row in rows:
        if row.debt_weight > 0.0:
            grouped[
                (
                    row.packet_type,
                    row.packet_gaps,
                    row.candidate_length,
                    row.next_packet_type,
                )
            ].append(row)

    print("Top debt regimes by current/candidate/next packet:")
    for (ptype, gaps, length, next_ptype), items in sorted(
        grouped.items(),
        key=lambda pair: (-sum(row.debt_weight for row in pair[1]), pair[0]),
    )[:n]:
        debt = sum(row.debt_weight for row in items)
        transition_mass = sum(row.occupancy_probability * row.action_probability for row in items)
        next_valid_hist = _top_weighted_histogram(
            items,
            lambda row: row.next_valid_lengths,
            lambda row: row.debt_weight,
            3,
        )
        next_policy_hist = _top_weighted_histogram(
            items,
            lambda row: row.next_policy_length,
            lambda row: row.debt_weight,
            3,
        )
        rendered_valid = ", ".join(f"{lengths}:{weight:.6f}" for lengths, weight in next_valid_hist)
        rendered_policy = ", ".join(f"{length_value}:{weight:.6f}" for length_value, weight in next_policy_hist)
        print(
            f"  packet type={ptype} gaps={gaps} candidate_L={length}"
            f" next_type={next_ptype} mass={transition_mass:.6f}"
            f" debt={debt:.6f}"
            f" next_policy_L=[{rendered_policy}]"
            f" next_valid=[{rendered_valid}]"
        )
    print()

    print("Raw top debt rows:")
    for row in sorted(rows, key=lambda item: item.debt_weight, reverse=True)[:n]:
        if row.debt_weight <= 0.0:
            break
        marker = " fallback" if row.next_used_fallback else ""
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} policy_L={row.policy_length}"
            f" current_valid={row.current_valid_lengths}"
            f" candidate_L={row.candidate_length}"
            f" next={row.next_state} next_ptype={row.next_packet_type}"
            f" next_valid={row.next_valid_lengths}"
            f" next_policy_L={row.next_policy_length}"
            f" debt={row.debt_weight:.6f}{marker}"
        )


def _candidate_top_prefix_policy_value(
    k: int,
    state: tuple[int, ...],
    length: int,
    continuation: dict[tuple[int, ...], float],
) -> float:
    expected_value = 0.0
    for probability, action in _balanced_top_prefix_policy(k, length):
        expected_value += probability * _next_state_value(state, action, continuation)
    return expected_value - 0.5


def top_prefix_candidate_values(
    k: int,
    T: int,
    policy_name: str,
    length_set: str = "scale",
) -> list[TopPrefixCandidateValueRow]:
    policies = _policy_registry(k)
    if policy_name not in policies:
        raise ValueError(f"unknown policy: {policy_name}")
    policy = policies[policy_name]
    occupancy = state_occupancy(k, T, policy)
    policy_values = evaluate_balanced_policy(k, T, policy)
    candidate_lengths = _top_prefix_candidate_lengths(k, length_set)
    rows: list[TopPrefixCandidateValueRow] = []

    for time in range(T):
        remaining_horizon = T - time
        continuation = policy_values[remaining_horizon - 1]
        for state, occupancy_probability in occupancy[time].items():
            policy_length = _top_prefix_length_from_policy(policy(state))
            candidate_scores = tuple(
                (
                    length,
                    _candidate_top_prefix_policy_value(k, state, length, continuation),
                )
                for length in candidate_lengths
            )
            best_length, best_value = max(
                candidate_scores,
                key=lambda item: (item[1], -item[0]),
            )
            policy_value = _candidate_top_prefix_policy_value(k, state, policy_length, continuation)
            rows.append(
                TopPrefixCandidateValueRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_probability=occupancy_probability,
                    policy_length=policy_length,
                    policy_value=policy_value,
                    best_length=best_length,
                    best_value=best_value,
                    regret=max(best_value - policy_value, 0.0),
                    candidate_scores=candidate_scores,
                )
            )
    return rows


def print_top_prefix_candidate_values(
    k: int,
    T: int,
    policy_name: str,
    length_set: str = "scale",
    n: int = 20,
) -> None:
    rows = top_prefix_candidate_values(k, T, policy_name, length_set=length_set)

    print(f"Top-prefix candidate value, k={k}, T={T}")
    print()
    print(f"policy: {policy_name}")
    print(f"length set: {length_set}")
    print(f"rows: {len(rows)}")
    print(f"total weighted regret: {sum(row.weighted_regret for row in rows):.6f}")
    print()

    print("Best candidate L histogram by occupancy:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.best_length, lambda row: row.occupancy_probability, n):
        print(f"  {length}: {weight:.6f}")
    print()

    print("Best candidate L histogram by weighted value opportunity:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.best_length, lambda row: row.weighted_regret, n):
        print(f"  {length}: {weight:.6f}")
    print()

    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[TopPrefixCandidateValueRow]] = defaultdict(list)
    for row in rows:
        if row.regret > 1e-12:
            grouped[(row.packet_type, row.packet_gaps)].append(row)

    print("Top regimes where policy L is not best:")
    for (ptype, gaps), items in sorted(
        grouped.items(),
        key=lambda pair: (-sum(row.weighted_regret for row in pair[1]), pair[0]),
    )[:n]:
        regret = sum(row.weighted_regret for row in items)
        occupancy_mass = sum(row.occupancy_probability for row in items)
        policy_hist = _top_weighted_histogram(
            items,
            lambda row: row.policy_length,
            lambda row: row.weighted_regret,
            3,
        )
        best_hist = _top_weighted_histogram(
            items,
            lambda row: row.best_length,
            lambda row: row.weighted_regret,
            3,
        )
        rendered_policy = ", ".join(f"{length}:{weight:.6f}" for length, weight in policy_hist)
        rendered_best = ", ".join(f"{length}:{weight:.6f}" for length, weight in best_hist)
        print(
            f"  packet type={ptype} gaps={gaps}"
            f" occupancy={occupancy_mass:.6f}"
            f" weighted_regret={regret:.6f}"
            f" policy_L=[{rendered_policy}]"
            f" best_L=[{rendered_best}]"
        )
    print()

    print("Raw top regret rows:")
    for row in sorted(rows, key=lambda item: item.weighted_regret, reverse=True)[:n]:
        if row.weighted_regret <= 1e-12:
            break
        scores = ", ".join(
            f"{length}:{score:.6f}"
            for length, score in sorted(row.candidate_scores)
        )
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
            f" prob={row.occupancy_probability:.6f}"
            f" policy_L={row.policy_length} policy_value={row.policy_value:.6f}"
            f" best_L={row.best_length} best_value={row.best_value:.6f}"
            f" weighted_regret={row.weighted_regret:.6f}"
            f" scores=[{scores}]"
        )


def top_prefix_restricted_optimal(
    k: int,
    T: int,
    length_set: str,
    n: int = 20,
) -> TopPrefixRestrictedOptimalResult:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 2:
        raise ValueError("k must be at least 2 for top-prefix restricted optimal")

    candidate_lengths = _top_prefix_candidate_lengths(k, length_set)
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    terminal_states = state_layers[T]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in terminal_states}
    ]
    argmax_lengths: list[dict[tuple[int, ...], int]] = [{}]
    selected_length_counts: dict[int, int] = defaultdict(int)

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        current_argmax: dict[tuple[int, ...], int] = {}
        for state in state_layers[T - horizon]:
            length_values = tuple(
                (
                    length,
                    _candidate_top_prefix_policy_value(k, state, length, previous),
                )
                for length in candidate_lengths
            )
            best_length, best_value = max(length_values, key=lambda item: (item[1], -item[0]))
            current[state] = best_value
            current_argmax[state] = best_length
            selected_length_counts[best_length] += 1
        values.append(current)
        argmax_lengths.append(current_argmax)

    zero = tuple(0 for _ in range(k))
    restricted_value = values[T][zero]
    optimal_value = optimal_values(k, T)[T][zero]
    gap = optimal_value - restricted_value
    normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)

    occupancy: list[dict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    occupancy_selected_weights: dict[int, float] = defaultdict(float)
    regime_selected_weights: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        dict[int, float],
    ] = defaultdict(lambda: defaultdict(float))
    regime_weights: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            length = argmax_lengths[remaining_horizon][state]
            occupancy_selected_weights[length] += probability
            regime_key = (packet_type(state), _packet_gaps(state))
            regime_selected_weights[regime_key][length] += probability
            regime_weights[regime_key] += probability
            for action_probability, action in _balanced_top_prefix_policy(k, length):
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * action_probability

    top_regime_selected_lengths = []
    for regime_key, weight in sorted(regime_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        selected_hist = tuple(
            sorted(
                regime_selected_weights[regime_key].items(),
                key=lambda pair: (-pair[1], pair[0]),
            )
        )
        top_regime_selected_lengths.append((*regime_key, weight, selected_hist))

    return TopPrefixRestrictedOptimalResult(
        length_set_name=length_set,
        value=restricted_value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=normalized_gap,
        selected_length_counts=tuple(sorted(selected_length_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        occupancy_selected_length_weights=tuple(
            sorted(occupancy_selected_weights.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        top_regime_selected_lengths=tuple(top_regime_selected_lengths),
    )


def print_top_prefix_restricted_optimal(
    k: int,
    T: int,
    length_set: str,
    n: int = 20,
) -> None:
    result = top_prefix_restricted_optimal(k, T, length_set, n=n)

    print(f"Top-prefix restricted optimal, k={k}, T={T}")
    print()
    print(f"length set: {result.length_set_name}")
    print(f"V_star: {result.optimal_value:.6f}")
    print(f"V_restricted: {result.value:.6f}")
    print(f"gap: {result.gap:.6f}")
    print(f"gap/sqrt(T): {result.normalized_gap:.6f}")
    print()

    print("Selected L histogram over DP states:")
    for length, count in result.selected_length_counts[:n]:
        print(f"  {length}: {count}")
    print()

    print("Selected L histogram by restricted-policy occupancy:")
    for length, weight in result.occupancy_selected_length_weights[:n]:
        print(f"  {length}: {weight:.6f}")
    print()

    print("Top regimes by restricted-policy occupancy:")
    for ptype, gaps, weight, selected_hist in result.top_regime_selected_lengths[:n]:
        rendered = ", ".join(
            f"{length}:{length_weight:.6f}"
            for length, length_weight in selected_hist[:5]
        )
        print(f"  packet type={ptype} gaps={gaps} occupancy={weight:.6f} selected=[{rendered}]")


def _balanced_action_pair_value(
    state: tuple[int, ...],
    action: tuple[int, ...],
    continuation: dict[tuple[int, ...], float],
) -> float:
    return (
        0.5 * _next_state_value(state, action, continuation)
        + 0.5 * _next_state_value(state, complement(action), continuation)
        - 0.5
    )


def edge_run_restricted_optimal(
    k: int,
    T: int,
    max_runs: int,
    n: int = 20,
) -> EdgeRunRestrictedOptimalResult:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 2:
        raise ValueError("k must be at least 2 for edge-run restricted optimal")
    if max_runs < 1:
        raise ValueError("max_runs must be positive")

    candidates = _edge_run_restricted_candidates(k, max_runs)
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    terminal_states = state_layers[T]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in terminal_states}
    ]
    argmax_candidates: list[dict[tuple[int, ...], EdgeRunRestrictedCandidate]] = [{}]
    selected_signature_counts: dict[str, int] = defaultdict(int)

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        current_argmax: dict[tuple[int, ...], EdgeRunRestrictedCandidate] = {}
        for state in state_layers[T - horizon]:
            candidate_values = tuple(
                (
                    candidate,
                    _balanced_action_pair_value(state, candidate.action, previous),
                )
                for candidate in candidates
            )
            best_candidate, best_value = max(
                candidate_values,
                key=lambda item: (item[1], _format_edge_signature(item[0].edge_signature)),
            )
            current[state] = best_value
            current_argmax[state] = best_candidate
            selected_signature_counts[_format_edge_signature(best_candidate.edge_signature)] += 1
        values.append(current)
        argmax_candidates.append(current_argmax)

    zero = tuple(0 for _ in range(k))
    restricted_value = values[T][zero]
    optimal_value = optimal_values(k, T)[T][zero]
    gap = optimal_value - restricted_value
    normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)

    occupancy: list[dict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    occupancy_selected_weights: dict[str, float] = defaultdict(float)
    regime_selected_weights: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        dict[str, float],
    ] = defaultdict(lambda: defaultdict(float))
    regime_weights: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            candidate = argmax_candidates[remaining_horizon][state]
            signature = _format_edge_signature(candidate.edge_signature)
            occupancy_selected_weights[signature] += probability
            regime_key = (packet_type(state), _packet_gaps(state))
            regime_selected_weights[regime_key][signature] += probability
            regime_weights[regime_key] += probability
            for action_probability, action in ((0.5, candidate.action), (0.5, complement(candidate.action))):
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * action_probability

    top_regime_selected_signatures = []
    for regime_key, weight in sorted(regime_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        selected_hist = tuple(
            sorted(
                regime_selected_weights[regime_key].items(),
                key=lambda pair: (-pair[1], pair[0]),
            )
        )
        top_regime_selected_signatures.append((*regime_key, weight, selected_hist))

    return EdgeRunRestrictedOptimalResult(
        max_runs=max_runs,
        library_size=len(candidates),
        value=restricted_value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=normalized_gap,
        selected_signature_counts=tuple(
            sorted(selected_signature_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        occupancy_selected_signature_weights=tuple(
            sorted(occupancy_selected_weights.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
        top_regime_selected_signatures=tuple(top_regime_selected_signatures),
    )


def one_run_restricted_optimal(
    k: int,
    T: int,
    n: int = 20,
) -> EdgeRunRestrictedOptimalResult:
    return edge_run_restricted_optimal(k, T, 1, n=n)


def print_edge_run_restricted_optimal(
    k: int,
    T: int,
    max_runs: int,
    n: int = 20,
) -> None:
    result = edge_run_restricted_optimal(k, T, max_runs, n=n)

    label = "One-run" if max_runs == 1 else f"{max_runs}-run"
    print(f"{label} restricted optimal, k={k}, T={T}")
    print()
    print(f"library size: {result.library_size}")
    print(f"V_star: {result.optimal_value:.6f}")
    print(f"V_restricted: {result.value:.6f}")
    print(f"gap: {result.gap:.6f}")
    print(f"gap/sqrt(T): {result.normalized_gap:.6f}")
    print()

    print("Selected edge-signature histogram over DP states:")
    for signature, count in result.selected_signature_counts[:n]:
        print(f"  {signature}: {count}")
    print()

    print("Selected edge-signature histogram by restricted-policy occupancy:")
    for signature, weight in result.occupancy_selected_signature_weights[:n]:
        print(f"  {signature}: {weight:.6f}")
    print()

    print("Top regimes by restricted-policy occupancy:")
    for ptype, gaps, weight, selected_hist in result.top_regime_selected_signatures[:n]:
        rendered = ", ".join(
            f"{signature}:{signature_weight:.6f}"
            for signature, signature_weight in selected_hist[:5]
        )
        print(f"  packet type={ptype} gaps={gaps} occupancy={weight:.6f} selected=[{rendered}]")


def print_one_run_restricted_optimal(
    k: int,
    T: int,
    n: int = 20,
) -> None:
    print_edge_run_restricted_optimal(k, T, 1, n=n)


def library_lp_restricted_optimal(
    k: int,
    T: int,
    library_name: str,
    active_tolerance: float = 1e-8,
) -> LibraryLPRestrictedOptimalResult:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 1:
        raise ValueError("k must be positive")

    library_actions = _action_library(k, library_name, T=T)
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    terminal_states = state_layers[T]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in terminal_states}
    ]
    active_action_counts: dict[str, int] = defaultdict(int)
    active_edge_signature_counts: dict[str, int] = defaultdict(int)

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, previous)
                for action in library_actions
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
            for action, q_value in q_by_action.items():
                score = q_value - sum(solution.p[index] * action[index] for index in range(k))
                if score >= solution.value - active_tolerance:
                    active_action_counts[_format_action(action)] += 1
                    active_edge_signature_counts[_format_edge_signature(_edge_signature(action))] += 1
        values.append(current)

    zero = tuple(0 for _ in range(k))
    restricted_value = values[T][zero]
    optimal_value = optimal_values(k, T)[T][zero]
    gap = optimal_value - restricted_value
    normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
    return LibraryLPRestrictedOptimalResult(
        library_name=library_name,
        library_size=len(library_actions),
        value=restricted_value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=normalized_gap,
        active_action_counts=tuple(sorted(active_action_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        active_edge_signature_counts=tuple(
            sorted(active_edge_signature_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
    )


def _library_lp_restricted_optimal_for_actions(
    k: int,
    T: int,
    library_name: str,
    library_actions: tuple[tuple[int, ...], ...],
    optimal_value: float,
    active_tolerance: float = 1e-8,
    state_layers: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
    collect_active_counts: bool = True,
    precomputed_values: list[dict[tuple[int, ...], float]] | None = None,
) -> LibraryLPRestrictedOptimalResult:
    if precomputed_values is not None:
        zero = tuple(0 for _ in range(k))
        restricted_value = precomputed_values[T][zero]
        gap = optimal_value - restricted_value
        return LibraryLPRestrictedOptimalResult(
            library_name=library_name,
            library_size=len(library_actions),
            value=restricted_value,
            optimal_value=optimal_value,
            gap=gap,
            normalized_gap=gap / (T ** 0.5 if T > 0 else 1.0),
            active_action_counts=(),
            active_edge_signature_counts=(),
        )

    if state_layers is None:
        state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
    terminal_states = state_layers[T]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in terminal_states}
    ]
    active_action_counts: dict[str, int] = defaultdict(int)
    active_edge_signature_counts: dict[str, int] = defaultdict(int)

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, previous)
                for action in library_actions
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
            if collect_active_counts:
                for action, q_value in q_by_action.items():
                    score = q_value - sum(solution.p[index] * action[index] for index in range(k))
                    if score >= solution.value - active_tolerance:
                        active_action_counts[_format_action(action)] += 1
                        active_edge_signature_counts[_format_edge_signature(_edge_signature(action))] += 1
        values.append(current)

    zero = tuple(0 for _ in range(k))
    restricted_value = values[T][zero]
    gap = optimal_value - restricted_value
    return LibraryLPRestrictedOptimalResult(
        library_name=library_name,
        library_size=len(library_actions),
        value=restricted_value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=gap / (T ** 0.5 if T > 0 else 1.0),
        active_action_counts=tuple(sorted(active_action_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        active_edge_signature_counts=tuple(
            sorted(active_edge_signature_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
    )


def _library_lp_restricted_values_for_actions(
    k: int,
    T: int,
    library_actions: tuple[tuple[int, ...], ...],
    state_layers: tuple[tuple[tuple[int, ...], ...], ...] | None = None,
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = False,
) -> tuple[
    float,
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int], ...],
]:
    if state_layers is None:
        state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
    terminal_states = state_layers[T]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in terminal_states}
    ]
    active_action_counts: dict[str, int] = defaultdict(int)
    active_edge_signature_counts: dict[str, int] = defaultdict(int)

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, previous)
                for action in library_actions
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
            if collect_active_counts:
                for action, q_value in q_by_action.items():
                    score = q_value - sum(solution.p[index] * action[index] for index in range(k))
                    if score >= solution.value - active_tolerance:
                        active_action_counts[_format_action(action)] += 1
                        active_edge_signature_counts[_format_edge_signature(_edge_signature(action))] += 1
        values.append(current)

    zero = tuple(0 for _ in range(k))
    return (
        values[T][zero],
        tuple(sorted(active_action_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        tuple(sorted(active_edge_signature_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
    )


def print_library_lp_restricted_optimal(
    k: int,
    T: int,
    library_name: str,
    n: int = 20,
) -> None:
    result = library_lp_restricted_optimal(k, T, library_name)

    print(f"Library LP restricted optimal, k={k}, T={T}")
    print()
    print(f"library: {result.library_name}")
    print(f"library size: {result.library_size}")
    print(f"V_star: {result.optimal_value:.6f}")
    print(f"V_restricted: {result.value:.6f}")
    print(f"gap: {result.gap:.6f}")
    print(f"gap/sqrt(T): {result.normalized_gap:.6f}")
    print()

    print("Top active actions over DP states:")
    for action, count in result.active_action_counts[:n]:
        print(f"  {action}: {count}")
    print()

    print("Top active edge signatures over DP states:")
    for signature, count in result.active_edge_signature_counts[:n]:
        print(f"  {signature}: {count}")


def _library_lp_value_layers(
    k: int,
    T: int,
    library_actions: tuple[tuple[int, ...], ...],
) -> list[dict[tuple[int, ...], float]]:
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in state_layers[T]}
    ]

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, previous)
                for action in library_actions
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
        values.append(current)

    return values


def library_lp_dual_inspect_rows(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
) -> tuple[tuple[dict[tuple[int, ...], float], ...], list[LibraryLPDualInspectRow]]:
    if T < 0:
        raise ValueError("T must be nonnegative")
    if k < 1:
        raise ValueError("k must be positive")

    library_actions = _action_library(k, library_name, T=T)
    values = _library_lp_value_layers(k, T, library_actions)
    zero = tuple(0 for _ in range(k))
    occupancy: list[defaultdict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    rows: list[LibraryLPDualInspectRow] = []

    for time in range(T):
        remaining_horizon = T - time
        continuation = values[remaining_horizon - 1]
        for state, probability in tuple(occupancy[time].items()):
            q_by_action = {
                action: _next_state_value(state, action, continuation)
                for action in library_actions
            }
            primal = solve_minimax_step(q_by_action, k)
            if not primal.success:
                raise RuntimeError(f"primal LP failed at state {state}: {primal.message}")
            dual = solve_adversary_dual(q_by_action, k)
            if not dual.success:
                raise RuntimeError(f"dual LP failed at state {state}: {dual.message}")

            support = tuple(
                sorted(
                    (
                        (weight, action, _edge_signature(action))
                        for action, weight in dual.weights_by_action
                        if weight >= support_tolerance
                    ),
                    key=lambda item: (-item[0], _format_action(item[1])),
                )
            )
            ptype = packet_type(state)
            gaps = _packet_gaps(state)
            if (
                (packet_type_filter is None or ptype == packet_type_filter)
                and (packet_gaps_filter is None or gaps == packet_gaps_filter)
            ):
                rows.append(
                    LibraryLPDualInspectRow(
                        time=time,
                        remaining_horizon=remaining_horizon,
                        state=state,
                        packet_type=ptype,
                        packet_gaps=gaps,
                        adjacent_gaps=_gap_vector(state),
                        occupancy_probability=probability,
                        value=values[remaining_horizon][state],
                        learner_distribution=primal.p,
                        dual_value=dual.value,
                        expected_action=dual.expected_action,
                        max_expected_action=dual.max_expected_action,
                        support=support,
                    )
                )

            for action, weight in dual.weights_by_action:
                if weight < support_tolerance:
                    continue
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * weight

    return tuple(dict(layer) for layer in occupancy), rows


def print_library_lp_dual_inspect(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    support_n: int = 12,
    n: int = 20,
) -> None:
    library_actions = _action_library(k, library_name, T=T)
    occupancy, rows = library_lp_dual_inspect_rows(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    zero = tuple(0 for _ in range(k))
    values = _library_lp_value_layers(k, T, library_actions)
    restricted_value = values[T][zero]
    optimal_value = optimal_values(k, T)[T][zero]
    gap = optimal_value - restricted_value

    print(f"Library LP dual inspect, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {gap:.6f}")
    print(f"total occupancy mass: {sum(sum(layer.values()) for layer in occupancy[:-1]):.6f}")
    print(f"matching rows: {len(rows)}")
    print()

    regime_weights: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    regime_support_weights: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        dict[str, float],
    ] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        regime_key = (row.packet_type, row.packet_gaps)
        regime_weights[regime_key] += row.occupancy_probability
        for weight, _, signature in row.support:
            regime_support_weights[regime_key][_format_edge_signature(signature)] += (
                row.occupancy_probability * weight
            )

    print("Top regimes by restricted-LP occupancy:")
    for (ptype, gaps), weight in sorted(regime_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        support_hist = sorted(
            regime_support_weights[(ptype, gaps)].items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[:5]
        rendered = ", ".join(f"{signature}:{signature_weight:.6f}" for signature, signature_weight in support_hist)
        print(f"  packet type={ptype} gaps={gaps} occupancy={weight:.6f} support=[{rendered}]")
    print()

    print("Raw top rows by occupancy:")
    for row in sorted(rows, key=lambda item: (-item.occupancy_probability, item.time, item.state))[:n]:
        learner = "(" + ", ".join(f"{value:.3f}" for value in row.learner_distribution) + ")"
        expected = "(" + ", ".join(f"{value:.3f}" for value in row.expected_action) + ")"
        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type} gaps={row.packet_gaps}"
            f" adjacent={row.adjacent_gaps} occ={row.occupancy_probability:.6f}"
            f" value={row.value:.6f} dual={row.dual_value:.6f}"
        )
        print(f"  learner p={learner}")
        print(f"  expected action={expected} max={row.max_expected_action:.6f}")
        print("  adversary dual support:")
        for weight, action, signature in row.support[:support_n]:
            print(
                f"    {_format_action(action)}"
                f" edge={_format_edge_signature(signature)}"
                f" weight={weight:.6f}"
            )
        if len(row.support) > support_n:
            print(f"    ... {len(row.support) - support_n} more")
        print()


def _packet_index_groups(state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not state:
        return ()
    groups: list[list[int]] = [[0]]
    for index in range(1, len(state)):
        if state[index] == state[index - 1]:
            groups[-1].append(index)
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _orbit_key_for_action(
    action: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return tuple(sum(action[index] for index in group) for group in groups)


def _orbit_size(
    orbit_key: tuple[int, ...],
    packet_sizes: tuple[int, ...],
) -> int:
    size = 1
    for packet_size, ones_count in zip(packet_sizes, orbit_key, strict=True):
        size *= comb(packet_size, ones_count)
    return size


def _orbit_representative_action(
    orbit_key: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
    k: int,
) -> tuple[int, ...]:
    bits = [0] * k
    for ones_count, group in zip(orbit_key, groups, strict=True):
        for index in group[:ones_count]:
            bits[index] = 1
    return tuple(bits)


def _summarize_dual_support_orbits(row: LibraryLPDualInspectRow) -> tuple[DualSupportOrbitSummary, ...]:
    groups = _packet_index_groups(row.state)
    packet_sizes = tuple(len(group) for group in groups)
    grouped_support: dict[tuple[int, ...], list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
    for weight, action, _ in row.support:
        grouped_support[_orbit_key_for_action(action, groups)].append((weight, action))

    summaries: list[DualSupportOrbitSummary] = []
    for orbit_key, items in grouped_support.items():
        total_weight = sum(weight for weight, _ in items)
        representative_action = _orbit_representative_action(orbit_key, groups, len(row.state))
        expected_action_contribution = tuple(
            sum(weight * action[index] for weight, action in items)
            for index in range(len(row.state))
        )
        packet_expected_averages = tuple(
            (
                sum(expected_action_contribution[index] for index in group)
                / (len(group) * total_weight)
                if total_weight > 0
                else 0.0
            )
            for group in groups
        )
        successor_weights: dict[tuple[int, ...], float] = defaultdict(float)
        for weight, action in items:
            next_state = canon(tuple(row.state[index] + action[index] for index in range(len(row.state))))
            successor_weights[packet_type(next_state)] += weight
        summaries.append(
            DualSupportOrbitSummary(
                orbit_key=orbit_key,
                packet_sizes=packet_sizes,
                orbit_size=_orbit_size(orbit_key, packet_sizes),
                support_count=len(items),
                total_weight=total_weight,
                representative_action=representative_action,
                representative_edge_signature=_edge_signature(representative_action),
                support_weights=tuple(sorted((weight for weight, _ in items), reverse=True)),
                expected_action_contribution=expected_action_contribution,
                packet_expected_averages=packet_expected_averages,
                successor_packet_type_weights=tuple(
                    sorted(successor_weights.items(), key=lambda pair: (-pair[1], pair[0]))
                ),
            )
        )
    return tuple(sorted(summaries, key=lambda item: (-item.total_weight, item.orbit_key)))


def _format_float_tuple(values: tuple[float, ...], precision: int = 3) -> str:
    return "(" + ", ".join(f"{value:.{precision}f}" for value in values) + ")"


def _format_fractionish(value: float, max_denominator: int = 100) -> str:
    fraction = Fraction(value).limit_denominator(max_denominator)
    if abs(float(fraction) - value) <= 1e-8:
        return f"{fraction.numerator}/{fraction.denominator}" if fraction.denominator != 1 else str(fraction.numerator)
    return f"{value:.6f}"


def _two_run_compatible_orbit_size(
    state: tuple[int, ...],
    orbit_key: tuple[int, ...],
    two_run_actions: tuple[tuple[int, ...], ...],
) -> int:
    groups = _packet_index_groups(state)
    return sum(1 for action in two_run_actions if _orbit_key_for_action(action, groups) == orbit_key)


def _two_run_replay_template_from_dual_row(
    row: LibraryLPDualInspectRow,
    two_run_actions: tuple[tuple[int, ...], ...],
) -> TwoRunReplayTemplateRow:
    groups = _packet_index_groups(row.state)
    edge_signature_weights: dict[str, float] = defaultdict(float)
    orbit_key_weights: dict[tuple[int, ...], float] = defaultdict(float)
    successor_weights: dict[tuple[int, ...], float] = defaultdict(float)
    support_with_orbits: list[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    run_counts: set[int] = set()
    used_counts_by_orbit: dict[tuple[int, ...], int] = defaultdict(int)

    for weight, action, edge_signature in row.support:
        orbit_key = _orbit_key_for_action(action, groups)
        edge_signature_weights[_format_edge_signature(edge_signature)] += weight
        orbit_key_weights[orbit_key] += weight
        next_state = canon(tuple(row.state[index] + action[index] for index in range(len(row.state))))
        successor_weights[packet_type(next_state)] += weight
        support_with_orbits.append((weight, action, edge_signature, orbit_key))
        run_count = _edge_signature_run_count(edge_signature)
        if run_count > 0:
            run_counts.add(run_count)
        used_counts_by_orbit[orbit_key] += 1

    full_orbit_compatible = True
    sparse_inside_orbit = False
    for orbit_key, used_count in used_counts_by_orbit.items():
        compatible_count = _two_run_compatible_orbit_size(row.state, orbit_key, two_run_actions)
        if used_count != compatible_count:
            full_orbit_compatible = False
            sparse_inside_orbit = True

    if run_counts <= {1}:
        run_profile = "one-run only"
    elif run_counts <= {2}:
        run_profile = "two-run only"
    else:
        run_profile = "mixed one/two-run"

    orbit_weights = tuple(sorted(orbit_key_weights.items(), key=lambda item: (-item[1], item[0])))
    edge_weights = tuple(sorted(edge_signature_weights.items(), key=lambda item: (-item[1], item[0])))
    successor_packet_type_weights = tuple(sorted(successor_weights.items(), key=lambda item: (-item[1], item[0])))
    template_signature = (
        tuple((orbit_key, _format_fractionish(weight)) for orbit_key, weight in orbit_weights),
        tuple((edge_signature, _format_fractionish(weight)) for edge_signature, weight in edge_weights),
        tuple((ptype, _format_fractionish(weight)) for ptype, weight in successor_packet_type_weights),
    )

    return TwoRunReplayTemplateRow(
        remaining_horizon=row.remaining_horizon,
        state=row.state,
        packet_type=row.packet_type,
        packet_gaps=row.packet_gaps,
        adjacent_gaps=row.adjacent_gaps,
        occupancy_probability=row.occupancy_probability,
        support_action_count=len(row.support),
        support_edge_signature_weights=edge_weights,
        support_orbit_key_weights=orbit_weights,
        successor_packet_type_weights=successor_packet_type_weights,
        full_orbit_compatible=full_orbit_compatible,
        sparse_inside_orbit=sparse_inside_orbit,
        run_profile=run_profile,
        template_signature=template_signature,
        support=tuple(sorted(support_with_orbits, key=lambda item: (-item[0], _format_action(item[1])))),
    )


def two_run_replay_template_rows(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[TwoRunReplayTemplateRow, ...]:
    if k != 9 or T != 7:
        raise ValueError("two-run replay template report is currently implemented for k=9,T=7")
    two_run_actions = _action_library(k, "two_run", T=T)
    _, dual_rows = library_lp_dual_inspect_rows(k, T, "two_run", support_tolerance=support_tolerance)
    return tuple(_two_run_replay_template_from_dual_row(row, two_run_actions) for row in dual_rows)


def _format_weighted_tuple_items(items: tuple[tuple[object, float], ...], max_items: int = 8) -> str:
    visible = [f"{key}:{_format_fractionish(weight)}" for key, weight in items[:max_items]]
    if len(items) > max_items:
        visible.append("...")
    return ", ".join(visible)


def print_two_run_replay_template_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 40,
) -> None:
    rows = two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)
    regimes = {(row.packet_type, row.packet_gaps) for row in rows}
    templates = {row.template_signature for row in rows}

    print(f"Two-run replay template report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"total replay states: {len(rows)}")
    print(f"total distinct packet regimes: {len(regimes)}")
    print(f"total distinct support templates: {len(templates)}")
    print()

    regime_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
    regime_occupancy: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    horizon_regime_counts: dict[tuple[int, tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
    template_rows: dict[object, list[TwoRunReplayTemplateRow]] = defaultdict(list)
    for row in rows:
        regime_key = (row.packet_type, row.packet_gaps)
        regime_counts[regime_key] += 1
        regime_occupancy[regime_key] += row.occupancy_probability
        horizon_regime_counts[(row.remaining_horizon, row.packet_type, row.packet_gaps)] += 1
        template_rows[row.template_signature].append(row)

    print("Top regimes by occupancy:")
    for (ptype, gaps), occupancy in sorted(regime_occupancy.items(), key=lambda item: (-item[1], item[0]))[:n]:
        print(f"  packet type={ptype} gaps={gaps} rows={regime_counts[(ptype, gaps)]} occupancy={occupancy:.6f}")
    print()

    print("Top horizon/regime rows by count:")
    for (remaining, ptype, gaps), count in sorted(horizon_regime_counts.items(), key=lambda item: (-item[1], item[0]))[:n]:
        print(f"  rem={remaining} packet type={ptype} gaps={gaps} rows={count}")
    print()

    print("Top support templates by row count:")
    top_templates = sorted(template_rows.items(), key=lambda item: (-len(item[1]), item[1][0].remaining_horizon, item[1][0].state))[:n]
    for index, (_, template_group) in enumerate(top_templates, start=1):
        representative = template_group[0]
        horizons = tuple(sorted({row.remaining_horizon for row in template_group}, reverse=True))
        packet_regimes = tuple(sorted({(row.packet_type, row.packet_gaps) for row in template_group}))
        print(f"Template {index}: rows={len(template_group)}")
        print(f"  representative rem={representative.remaining_horizon} state={representative.state}")
        print(f"  horizons={horizons}")
        print(f"  packet regimes={packet_regimes[:8]}{' ...' if len(packet_regimes) > 8 else ''}")
        print(f"  support count={representative.support_action_count}")
        print(f"  run profile={representative.run_profile}")
        print(f"  full orbit compatible={representative.full_orbit_compatible}")
        print(f"  sparse inside orbit={representative.sparse_inside_orbit}")
        print(f"  orbit weights={_format_weighted_tuple_items(representative.support_orbit_key_weights)}")
        print(f"  edge signatures={_format_weighted_tuple_items(representative.support_edge_signature_weights)}")
        print(f"  successors={_format_weighted_tuple_items(representative.successor_packet_type_weights)}")
        print("  support:")
        for weight, action, edge_signature, orbit_key in representative.support[:12]:
            print(
                f"    w={_format_fractionish(weight):>8s}"
                f" action={_format_action(action)}"
                f" edge={_format_edge_signature(edge_signature)}"
                f" orbit={orbit_key}"
            )
        if len(representative.support) > 12:
            print("    ...")
        print()


def _coarse_edge_weight_skeleton(row: TwoRunReplayTemplateRow) -> tuple[tuple[str, str], ...]:
    return tuple(
        (edge_signature, _format_fractionish(weight))
        for edge_signature, weight in sorted(row.support_edge_signature_weights)
    )


def _coarse_action_weight_skeleton(row: TwoRunReplayTemplateRow) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_format_action(action), _format_fractionish(weight))
        for weight, action, _, _ in sorted(row.support, key=lambda item: (_format_action(item[1]), item[0]))
    )


def _coarse_weight_multiset_skeleton(row: TwoRunReplayTemplateRow) -> tuple[int, str, tuple[str, ...]]:
    return (
        row.support_action_count,
        row.run_profile,
        tuple(sorted((_format_fractionish(weight) for weight, *_ in row.support), reverse=True)),
    )


def _coarse_orbit_weight_shape(row: TwoRunReplayTemplateRow) -> tuple[tuple[tuple[str, ...], str], ...]:
    packet_sizes = row.packet_type
    entries: list[tuple[tuple[str, ...], str]] = []
    for orbit_key, weight in row.support_orbit_key_weights:
        exposures = tuple(
            _format_fractionish(ones / packet_size if packet_size else 0.0)
            for ones, packet_size in zip(orbit_key, packet_sizes, strict=True)
        )
        entries.append((exposures, _format_fractionish(weight)))
    return tuple(sorted(entries))


def _coarse_successor_shape(row: TwoRunReplayTemplateRow) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (ptype, _format_fractionish(weight))
        for ptype, weight in sorted(row.successor_packet_type_weights)
    )


def _coarse_template_aggregates(
    rows: tuple[TwoRunReplayTemplateRow, ...],
    signature_fn,
) -> tuple[CoarseTemplateAggregate, ...]:
    grouped: dict[object, list[TwoRunReplayTemplateRow]] = defaultdict(list)
    for row in rows:
        grouped[signature_fn(row)].append(row)

    aggregates: list[CoarseTemplateAggregate] = []
    for signature, group in grouped.items():
        representative = max(group, key=lambda row: row.occupancy_probability)
        packet_regimes = tuple(
            sorted({(row.packet_type, row.packet_gaps) for row in group})[:8]
        )
        aggregates.append(
            CoarseTemplateAggregate(
                signature=signature,
                row_count=len(group),
                total_occupancy=sum(row.occupancy_probability for row in group),
                horizons=tuple(sorted({row.remaining_horizon for row in group}, reverse=True)),
                representative_state=representative.state,
                representative_packet_regimes=packet_regimes,
                representative_support=representative.support,
            )
        )
    return tuple(sorted(aggregates, key=lambda item: (-item.total_occupancy, -item.row_count, str(item.signature))))


def two_run_replay_coarse_template_aggregates(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> dict[str, tuple[CoarseTemplateAggregate, ...]]:
    rows = two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)
    return {
        "edge_weight_skeleton": _coarse_template_aggregates(rows, _coarse_edge_weight_skeleton),
        "action_weight_skeleton": _coarse_template_aggregates(rows, _coarse_action_weight_skeleton),
        "weight_multiset_skeleton": _coarse_template_aggregates(rows, _coarse_weight_multiset_skeleton),
        "orbit_weight_shape": _coarse_template_aggregates(rows, _coarse_orbit_weight_shape),
        "successor_shape": _coarse_template_aggregates(rows, _coarse_successor_shape),
    }


def _format_coarse_signature(signature: object, max_length: int = 240) -> str:
    rendered = str(signature)
    if len(rendered) <= max_length:
        return rendered
    return rendered[: max_length - 4] + " ..."


def print_two_run_replay_coarse_template_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 80,
) -> None:
    rows = two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)
    aggregates_by_family = two_run_replay_coarse_template_aggregates(k, T, support_tolerance=support_tolerance)

    print(f"Two-run replay coarse template report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"total replay states: {len(rows)}")
    print()
    print("Distinct counts by coarse signature family:")
    for family_name, aggregates in aggregates_by_family.items():
        print(f"  {family_name}: {len(aggregates)}")
    print()

    for family_name, aggregates in aggregates_by_family.items():
        print(f"{family_name}: top by occupancy")
        for index, aggregate in enumerate(sorted(aggregates, key=lambda item: (-item.total_occupancy, -item.row_count, str(item.signature)))[:n], start=1):
            print(
                f"  {index}. rows={aggregate.row_count}"
                f" occupancy={aggregate.total_occupancy:.6f}"
                f" horizons={aggregate.horizons}"
            )
            print(f"     signature={_format_coarse_signature(aggregate.signature)}")
            print(f"     representative state={aggregate.representative_state}")
            print(f"     packet regimes={aggregate.representative_packet_regimes}")
            support = ", ".join(
                f"{_format_fractionish(weight)}:{_format_action(action)}"
                for weight, action, _, _ in aggregate.representative_support[:8]
            )
            if len(aggregate.representative_support) > 8:
                support += ", ..."
            print(f"     support={support}")
        print()

        print(f"{family_name}: top by row count")
        for index, aggregate in enumerate(sorted(aggregates, key=lambda item: (-item.row_count, -item.total_occupancy, str(item.signature)))[:n], start=1):
            print(
                f"  {index}. rows={aggregate.row_count}"
                f" occupancy={aggregate.total_occupancy:.6f}"
                f" horizons={aggregate.horizons}"
                f" signature={_format_coarse_signature(aggregate.signature)}"
            )
        print()


def print_library_lp_dual_orbits(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    orbit_n: int = 12,
    n: int = 10,
) -> None:
    _, rows = library_lp_dual_inspect_rows(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    library_actions = _action_library(k, library_name, T=T)
    values = _library_lp_value_layers(k, T, library_actions)
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = values[T][zero]

    print(f"Library LP dual support orbits, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"matching rows: {len(rows)}")
    print()

    for row in sorted(rows, key=lambda item: (-item.occupancy_probability, item.time, item.state))[:n]:
        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f} support_actions={len(row.support)}"
        )
        print(f"  learner p={_format_float_tuple(row.learner_distribution)}")
        print(f"  expected action={_format_float_tuple(row.expected_action)} max={row.max_expected_action:.6f}")
        print("  support orbits:")
        for orbit in _summarize_dual_support_orbits(row)[:orbit_n]:
            weights = _format_float_tuple(orbit.support_weights, precision=6)
            packet_avgs = _format_float_tuple(orbit.packet_expected_averages)
            successors = ", ".join(
                f"{ptype}:{weight:.6f}"
                for ptype, weight in orbit.successor_packet_type_weights[:5]
            )
            print(
                f"    key={orbit.orbit_key} packets={orbit.packet_sizes}"
                f" weight={orbit.total_weight:.6f}"
                f" support={orbit.support_count}/{orbit.orbit_size}"
                f" rep={_format_action(orbit.representative_action)}"
                f" edge={_format_edge_signature(orbit.representative_edge_signature)}"
            )
            print(f"      weights={weights}")
            print(f"      packet exposure avg={packet_avgs}")
            print(f"      successor packet types=[{successors}]")
        print()


def print_library_lp_dual_orbit_completion(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    orbit_n: int = 20,
    n: int = 5,
) -> None:
    library_actions = _action_library(k, library_name, T=T)
    _, rows = library_lp_dual_inspect_rows(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    values = _library_lp_value_layers(k, T, library_actions)
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = values[T][zero]

    print(f"Library LP dual orbit completion, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"matching rows: {len(rows)}")
    print()

    for row in sorted(rows, key=lambda item: (-item.occupancy_probability, item.time, item.state))[:n]:
        groups = _packet_index_groups(row.state)
        packet_sizes = tuple(len(group) for group in groups)
        compatible_by_orbit: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
        for action in library_actions:
            compatible_by_orbit[_orbit_key_for_action(action, groups)].append(action)

        used_by_orbit: dict[tuple[int, ...], list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
        for weight, action, _ in row.support:
            used_by_orbit[_orbit_key_for_action(action, groups)].append((weight, action))

        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
        )
        print("  orbit completion:")
        orbit_summaries = _summarize_dual_support_orbits(row)
        for orbit in orbit_summaries[:orbit_n]:
            compatible_actions = compatible_by_orbit[orbit.orbit_key]
            used_items = used_by_orbit[orbit.orbit_key]
            used_weights = tuple(sorted((weight for weight, _ in used_items), reverse=True))
            min_weight = min(used_weights) if used_weights else 0.0
            max_weight = max(used_weights) if used_weights else 0.0
            uniform_gap = max_weight - min_weight
            expected_uniform = orbit.total_weight / len(used_weights) if used_weights else 0.0
            compatible_signatures = {
                _format_edge_signature(_edge_signature(action))
                for action in compatible_actions
            }
            used_signatures = {
                _format_edge_signature(_edge_signature(action))
                for _, action in used_items
            }
            print(
                f"    key={orbit.orbit_key}"
                f" weight={orbit.total_weight:.6f}"
                f" full={orbit.orbit_size}"
                f" library_compatible={len(compatible_actions)}"
                f" used={len(used_items)}"
                f" used_all_compatible={len(used_items) == len(compatible_actions)}"
            )
            print(
                f"      weights={_format_float_tuple(used_weights, precision=6)}"
                f" expected_uniform={expected_uniform:.6f}"
                f" uniform_gap={uniform_gap:.6f}"
            )
            print(
                f"      signatures used/library="
                f"{len(used_signatures)}/{len(compatible_signatures)}"
            )
        print()


def _balanced_fixed_policy_from_action(action: tuple[int, ...]):
    policy = ((0.5, action), (0.5, complement(action)))

    def policy_fn(_state: tuple[int, ...]) -> tuple[tuple[float, tuple[int, ...]], ...]:
        return policy

    return policy_fn


def _restricted_lp_value_for_actions(
    k: int,
    T: int,
    actions: tuple[tuple[int, ...], ...],
) -> float:
    values = _library_lp_value_layers(k, T, actions)
    return values[T][tuple(0 for _ in range(k))]


def _library_lp_restricted_result_for_actions(
    k: int,
    T: int,
    family_name: str,
    actions: tuple[tuple[int, ...], ...],
    orbit_rows: tuple[LibraryLPDualInspectRow, ...] = (),
    active_tolerance: float = 1e-8,
) -> K9MotifLibrarySweepRow:
    if not actions:
        raise ValueError(f"{family_name} action library must be non-empty")

    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in state_layers[T]}
    ]
    active_edge_signature_counts: dict[str, int] = defaultdict(int)
    active_orbit_key_counts: dict[str, int] = defaultdict(int)
    orbit_groups_by_state = {
        row.state: _packet_index_groups(row.state)
        for row in orbit_rows
    }

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, previous)
                for action in actions
            }
            solution = solve_minimax_step(q_by_action, k)
            if not solution.success:
                raise RuntimeError(f"LP failed at state {state}: {solution.message}")
            current[state] = solution.value
            groups = orbit_groups_by_state.get(state)
            for action, q_value in q_by_action.items():
                score = q_value - sum(solution.p[index] * action[index] for index in range(k))
                if score < solution.value - active_tolerance:
                    continue
                active_edge_signature_counts[_format_edge_signature(_edge_signature(action))] += 1
                if groups is not None:
                    active_orbit_key_counts[str(_orbit_key_for_action(action, groups))] += 1
        values.append(current)

    zero = tuple(0 for _ in range(k))
    value = values[T][zero]
    optimal_value = optimal_values(k, T)[T][zero]
    gap = optimal_value - value
    return K9MotifLibrarySweepRow(
        T=T,
        family_name=family_name,
        library_size=len(actions),
        value=value,
        optimal_value=optimal_value,
        gap=gap,
        normalized_gap=gap / (T ** 0.5 if T > 0 else 1.0),
        top_active_edge_signatures=tuple(
            sorted(active_edge_signature_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
        ),
        top_active_orbit_keys=tuple(
            sorted(active_orbit_key_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
        ),
    )


def _k3_motif_libraries() -> dict[str, tuple[tuple[int, ...], ...]]:
    motifs = {
        "twin_comb3": fixed_rank_action(3, {1}),
        "tail_singleton": fixed_rank_action(3, {1, 2}),
        "comb": fixed_rank_action(3, {1, 3}),
    }

    def actions_for(names: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
        actions: set[tuple[int, ...]] = set()
        for name in names:
            action = motifs[name]
            actions.add(action)
            actions.add(complement(action))
        return tuple(sorted(actions, key=_format_action))

    return {
        "twin_comb3": actions_for(("twin_comb3",)),
        "tail_singleton": actions_for(("tail_singleton",)),
        "comb": actions_for(("comb",)),
        "comb_twin_comb3": actions_for(("comb", "twin_comb3")),
        "comb_tail_singleton": actions_for(("comb", "tail_singleton")),
        "twin_comb3_tail_singleton": actions_for(("twin_comb3", "tail_singleton")),
        "all_three": actions_for(("twin_comb3", "comb", "tail_singleton")),
        "all": tuple(all_actions(3)),
    }


def k3_motif_sweep(T_values: tuple[int, ...]) -> tuple[K3MotifSweepRow, ...]:
    rows: list[K3MotifSweepRow] = []
    libraries = _k3_motif_libraries()
    single_motif_actions = {"twin_comb3", "tail_singleton", "comb"}
    for T in T_values:
        optimal_value = optimal_values(3, T)[T][(0, 0, 0)]
        for library_name, actions in libraries.items():
            deterministic_value: float | None = None
            if library_name in single_motif_actions:
                base_action = max(actions, key=_format_action)
                values = evaluate_balanced_policy(3, T, _balanced_fixed_policy_from_action(base_action))
                deterministic_value = values[T][(0, 0, 0)]
            lp_value = _restricted_lp_value_for_actions(3, T, actions)
            rows.append(
                K3MotifSweepRow(
                    T=T,
                    library_name=library_name,
                    library_size=len(actions),
                    deterministic_value=deterministic_value,
                    lp_value=lp_value,
                    optimal_value=optimal_value,
                )
            )
    return tuple(rows)


def _parse_T_values(text: str | None) -> tuple[int, ...]:
    if text is None or not text.strip():
        return (10, 20, 50, 100)
    values = _parse_int_tuple(text)
    if any(value <= 0 for value in values):
        raise ValueError("T values must be positive")
    return values


def _parse_cases(text: str | None) -> tuple[tuple[int, int], ...]:
    if text is None or not text.strip():
        return ((3, 20), (3, 50), (5, 20), (6, 12), (7, 10), (8, 8), (9, 7))
    cases: list[tuple[int, int]] = []
    for piece in text.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"case must use k:T format: {stripped}")
        k_text, T_text = stripped.split(":", 1)
        k = int(k_text)
        T = int(T_text)
        if k <= 0 or T <= 0:
            raise ValueError("case k and T must be positive")
        cases.append((k, T))
    return tuple(cases)


def _parse_library_names(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip():
        return ("top_prefix_all", "one_run", "two_run", "local_edges", "all")
    return tuple(piece.strip() for piece in text.split(",") if piece.strip())


def _parse_policy_names(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip():
        return ("two_run_orbit_mixture_v1", "top_prefix_three_regime_v6", "top_prefix_three_regime_v7")
    return tuple(piece.strip() for piece in text.split(",") if piece.strip())


def _strategy_class_library_actions(
    k: int,
    T: int,
    library_name: str,
) -> tuple[tuple[int, ...], ...]:
    if library_name != "all" and library_name in _k3_motif_libraries():
        if k != 3:
            raise ValueError(f"{library_name} is only valid for k=3")
        return _k3_motif_libraries()[library_name]
    return _action_library(k, library_name, T=T)


def _strategy_class_benchmark_case_cache(
    k: int,
    T: int,
    library_names: tuple[str, ...],
) -> tuple[_StrategyClassBenchmarkCaseCache, tuple[str, ...]]:
    skipped: list[str] = []
    state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
    optimal_layers = optimal_values(k, T)
    zero = tuple(0 for _ in range(k))
    library_actions_by_name: dict[str, tuple[tuple[int, ...], ...]] = {}
    for library_name in library_names:
        try:
            library_actions_by_name[library_name] = _strategy_class_library_actions(k, T, library_name)
        except ValueError as error:
            skipped.append(f"k={k} T={T} library={library_name}: {error}")
    return (
        _StrategyClassBenchmarkCaseCache(
            k=k,
            T=T,
            state_layers=state_layers,
            optimal_layers=optimal_layers,
            optimal_value=optimal_layers[T][zero],
            library_actions_by_name=library_actions_by_name,
        ),
        tuple(skipped),
    )


def _strategy_class_relative_library_actions(
    k: int,
    T: int,
    library_names: tuple[str, ...],
    reference_library_name: str,
) -> tuple[dict[str, tuple[tuple[int, ...], ...]], tuple[str, ...]]:
    skipped: list[str] = []
    actions_by_name: dict[str, tuple[tuple[int, ...], ...]] = {}
    names = tuple(dict.fromkeys((*library_names, reference_library_name)))
    for library_name in names:
        try:
            actions_by_name[library_name] = _strategy_class_library_actions(k, T, library_name)
        except ValueError as error:
            skipped.append(f"k={k} T={T} library={library_name}: {error}")
    return actions_by_name, tuple(skipped)


def _strategy_class_benchmark_row_for_library(
    cache: _StrategyClassBenchmarkCaseCache,
    library_name: str,
    include_probability_matching: bool = False,
    probability_matching_max_k: int = 9,
    probability_matching_max_T: int = 7,
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = True,
) -> tuple[StrategyClassBenchmarkRow, tuple[str, ...]]:
    k = cache.k
    T = cache.T
    skipped: list[str] = []
    actions = cache.library_actions_by_name[library_name]
    precomputed_values = (
        cache.optimal_layers
        if library_name == "all" and not collect_active_counts
        else None
    )
    result = _library_lp_restricted_optimal_for_actions(
        k,
        T,
        library_name,
        actions,
        cache.optimal_value,
        active_tolerance=active_tolerance,
        state_layers=cache.state_layers,
        collect_active_counts=collect_active_counts,
        precomputed_values=precomputed_values,
    )
    pm_weighted_l1: float | None = None
    pm_weighted_linf: float | None = None
    pm_avg_l1: float | None = None
    pm_avg_linf: float | None = None
    pm_rows: int | None = None
    if (
        include_probability_matching
        and k <= probability_matching_max_k
        and T <= probability_matching_max_T
    ):
        try:
            _, named_rows = probability_matching_named_inspect_rows(k, T, library_name)
        except ValueError as error:
            skipped.append(f"k={k} T={T} library={library_name} probability matching: {error}")
        else:
            pm_rows = len(named_rows)
            total_occupancy = sum(row.occupancy_probability for row in named_rows)
            pm_weighted_l1 = sum(row.weighted_l1_error for row in named_rows)
            pm_weighted_linf = sum(row.weighted_linf_error for row in named_rows)
            pm_avg_l1 = pm_weighted_l1 / total_occupancy if total_occupancy > 0 else 0.0
            pm_avg_linf = pm_weighted_linf / total_occupancy if total_occupancy > 0 else 0.0

    row = StrategyClassBenchmarkRow(
        k=k,
        T=T,
        library_name=library_name,
        library_size=result.library_size,
        value=result.value,
        optimal_value=result.optimal_value,
        gap=result.gap,
        normalized_gap=result.normalized_gap,
        exact=abs(result.gap) <= 1e-8,
        active_action_count=len(result.active_action_counts) if collect_active_counts else -1,
        active_edge_signature_count=len(result.active_edge_signature_counts) if collect_active_counts else -1,
        top_active_edge_signatures=result.active_edge_signature_counts[:10] if collect_active_counts else (),
        probability_matching_weighted_l1=pm_weighted_l1,
        probability_matching_weighted_linf=pm_weighted_linf,
        probability_matching_avg_l1=pm_avg_l1,
        probability_matching_avg_linf=pm_avg_linf,
        probability_matching_rows=pm_rows,
    )
    return row, tuple(skipped)


def strategy_class_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    library_names: tuple[str, ...],
    include_probability_matching: bool = False,
    probability_matching_max_k: int = 9,
    probability_matching_max_T: int = 7,
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = True,
) -> tuple[tuple[StrategyClassBenchmarkRow, ...], tuple[str, ...]]:
    rows: list[StrategyClassBenchmarkRow] = []
    skipped: list[str] = []

    for k, T in cases:
        cache, case_skipped = _strategy_class_benchmark_case_cache(k, T, library_names)
        skipped.extend(case_skipped)
        for library_name in library_names:
            if library_name not in cache.library_actions_by_name:
                continue
            row, row_skipped = _strategy_class_benchmark_row_for_library(
                cache,
                library_name,
                include_probability_matching=include_probability_matching,
                probability_matching_max_k=probability_matching_max_k,
                probability_matching_max_T=probability_matching_max_T,
                active_tolerance=active_tolerance,
                collect_active_counts=collect_active_counts,
            )
            rows.append(row)
            skipped.extend(row_skipped)
    return tuple(rows), tuple(skipped)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _format_optional_int(value: int) -> str:
    if value < 0:
        return "-"
    return str(value)


def strategy_class_relative_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    library_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = False,
) -> tuple[tuple[StrategyClassRelativeBenchmarkRow, ...], tuple[str, ...]]:
    rows: list[StrategyClassRelativeBenchmarkRow] = []
    skipped: list[str] = []

    for k, T in cases:
        state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
        actions_by_name, case_skipped = _strategy_class_relative_library_actions(
            k,
            T,
            library_names,
            reference_library_name,
        )
        skipped.extend(case_skipped)
        if reference_library_name not in actions_by_name:
            continue
        reference_value, _, _ = _library_lp_restricted_values_for_actions(
            k,
            T,
            actions_by_name[reference_library_name],
            state_layers=state_layers,
            active_tolerance=active_tolerance,
            collect_active_counts=False,
        )
        for library_name in library_names:
            if library_name not in actions_by_name:
                continue
            value, active_action_counts, active_edge_signature_counts = _library_lp_restricted_values_for_actions(
                k,
                T,
                actions_by_name[library_name],
                state_layers=state_layers,
                active_tolerance=active_tolerance,
                collect_active_counts=collect_active_counts,
            )
            gap = reference_value - value
            rows.append(
                StrategyClassRelativeBenchmarkRow(
                    k=k,
                    T=T,
                    library_name=library_name,
                    library_size=len(actions_by_name[library_name]),
                    reference_library_name=reference_library_name,
                    reference_value=reference_value,
                    value=value,
                    gap_to_reference=gap,
                    normalized_gap_to_reference=gap / (T ** 0.5 if T > 0 else 1.0),
                    active_action_count=len(active_action_counts) if collect_active_counts else -1,
                    active_edge_signature_count=len(active_edge_signature_counts) if collect_active_counts else -1,
                    top_active_edge_signatures=active_edge_signature_counts[:10] if collect_active_counts else (),
                )
            )
    return tuple(rows), tuple(skipped)


def print_strategy_class_relative_benchmark(
    cases: tuple[tuple[int, int], ...],
    library_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = False,
    n: int = 20,
) -> None:
    print("Strategy class relative benchmark")
    print()
    print(f"cases: {cases}")
    print(f"libraries: {library_names}")
    print(f"reference library: {reference_library_name}")
    print(f"active counts: {collect_active_counts}")
    print()
    print(
        "k  T   library          size  reference       V_reference"
        " V_library   gap_to_reference gap_to_reference/sqrt(T)"
        " active_actions active_edges"
    )

    rows: list[StrategyClassRelativeBenchmarkRow] = []
    skipped: list[str] = []
    for k, T in cases:
        case_start = time.perf_counter()
        print(f"# start case k={k} T={T}", flush=True)
        state_start = time.perf_counter()
        state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
        print(
            f"# built state_layers k={k} T={T}"
            f" layers={len(state_layers)} elapsed={time.perf_counter() - state_start:.3f}s",
            flush=True,
        )
        actions_by_name, case_skipped = _strategy_class_relative_library_actions(
            k,
            T,
            library_names,
            reference_library_name,
        )
        skipped.extend(case_skipped)
        if reference_library_name not in actions_by_name:
            print(f"# missing reference library k={k} T={T} library={reference_library_name}", flush=True)
            continue
        reference_start = time.perf_counter()
        print(f"# start reference k={k} T={T} library={reference_library_name}", flush=True)
        reference_value, _, _ = _library_lp_restricted_values_for_actions(
            k,
            T,
            actions_by_name[reference_library_name],
            state_layers=state_layers,
            active_tolerance=active_tolerance,
            collect_active_counts=False,
        )
        print(
            f"# end reference k={k} T={T} library={reference_library_name}"
            f" elapsed={time.perf_counter() - reference_start:.3f}s",
            flush=True,
        )
        case_rows: list[StrategyClassRelativeBenchmarkRow] = []
        for library_name in library_names:
            if library_name not in actions_by_name:
                continue
            row_start = time.perf_counter()
            print(f"# start row k={k} T={T} library={library_name}", flush=True)
            value, active_action_counts, active_edge_signature_counts = _library_lp_restricted_values_for_actions(
                k,
                T,
                actions_by_name[library_name],
                state_layers=state_layers,
                active_tolerance=active_tolerance,
                collect_active_counts=collect_active_counts,
            )
            print(
                f"# end row k={k} T={T} library={library_name}"
                f" elapsed={time.perf_counter() - row_start:.3f}s",
                flush=True,
            )
            gap = reference_value - value
            row = StrategyClassRelativeBenchmarkRow(
                k=k,
                T=T,
                library_name=library_name,
                library_size=len(actions_by_name[library_name]),
                reference_library_name=reference_library_name,
                reference_value=reference_value,
                value=value,
                gap_to_reference=gap,
                normalized_gap_to_reference=gap / (T ** 0.5 if T > 0 else 1.0),
                active_action_count=len(active_action_counts) if collect_active_counts else -1,
                active_edge_signature_count=len(active_edge_signature_counts) if collect_active_counts else -1,
                top_active_edge_signatures=active_edge_signature_counts[:10] if collect_active_counts else (),
            )
            case_rows.append(row)
        for row in sorted(case_rows, key=lambda item: (item.normalized_gap_to_reference, item.library_name)):
            rows.append(row)
            print(
                f"{row.k:1d} {row.T:3d}  {row.library_name:15s}"
                f" {row.library_size:5d}"
                f" {row.reference_library_name:15s}"
                f" {row.reference_value:11.6f}"
                f" {row.value:10.6f}"
                f" {row.gap_to_reference:16.6f}"
                f" {row.normalized_gap_to_reference:25.6f}"
                f" {_format_optional_int(row.active_action_count):>14s}"
                f" {_format_optional_int(row.active_edge_signature_count):>12s}",
                flush=True,
            )
        print(f"# end case k={k} T={T} elapsed={time.perf_counter() - case_start:.3f}s", flush=True)
    print()

    if skipped:
        print("Skipped libraries:")
        for message in skipped:
            print(f"  {message}")
        print()

    print("Top active edge signatures by case/library:")
    grouped: dict[tuple[int, int], list[StrategyClassRelativeBenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.k, row.T)].append(row)
    for case, case_rows in sorted(grouped.items()):
        print(f"k={case[0]} T={case[1]}")
        for row in sorted(case_rows, key=lambda item: (item.normalized_gap_to_reference, item.library_name))[:n]:
            rendered = ", ".join(
                f"{signature}:{count}"
                for signature, count in row.top_active_edge_signatures[:5]
            )
            if not rendered:
                rendered = "-"
            print(
                f"  {row.library_name:15s}"
                f" gap_to_reference/sqrt(T)={row.normalized_gap_to_reference:.6f}"
                f" active_edges=[{rendered}]"
            )
        print()


def explicit_policy_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    include_reference: bool = False,
) -> tuple[tuple[ExplicitPolicyBenchmarkRow, ...], tuple[str, ...]]:
    rows: list[ExplicitPolicyBenchmarkRow] = []
    skipped: list[str] = []

    for k, T in cases:
        policies = _policy_registry(k)
        reference_value: float | None = None
        if include_reference:
            state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
            try:
                reference_actions = _strategy_class_library_actions(k, T, reference_library_name)
            except ValueError as error:
                skipped.append(f"k={k} T={T} reference={reference_library_name}: {error}")
            else:
                reference_value, _, _ = _library_lp_restricted_values_for_actions(
                    k,
                    T,
                    reference_actions,
                    state_layers=state_layers,
                    collect_active_counts=False,
                )
        for policy_name in policy_names:
            if policy_name not in policies:
                skipped.append(f"k={k} T={T} policy={policy_name}: unknown policy")
                continue
            values = evaluate_balanced_policy(k, T, policies[policy_name])
            zero = tuple(0 for _ in range(k))
            value = values[T][zero]
            gap: float | None = None
            normalized_gap: float | None = None
            if reference_value is not None:
                gap = reference_value - value
                normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
            rows.append(
                ExplicitPolicyBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=value,
                    reference_library_name=reference_library_name if reference_value is not None else None,
                    reference_value=reference_value,
                    gap_to_reference=gap,
                    normalized_gap_to_reference=normalized_gap,
                )
            )
    return tuple(rows), tuple(skipped)


def evaluate_time_dependent_policy(k: int, T: int, policy_fn) -> list[dict[tuple[int, ...], float]]:
    if T < 0:
        raise ValueError("T must be nonnegative")

    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    values: list[dict[tuple[int, ...], float]] = [
        {state: float(max(state, default=0)) for state in state_layers[T]}
    ]

    for horizon in range(1, T + 1):
        previous = values[horizon - 1]
        current: dict[tuple[int, ...], float] = {}
        for state in state_layers[T - horizon]:
            policy = policy_fn(state, horizon)
            expected_value = 0.0
            expected_action_vector = np.zeros(k)
            for probability, action in policy:
                expected_action_vector += probability * np.asarray(action, dtype=float)
                expected_value += probability * _next_state_value(state, action, previous)
            current[state] = expected_value - float(expected_action_vector.max(initial=0.0))
        values.append(current)
    return values


def _time_dependent_policy_registry() -> dict[str, object]:
    return {
        "two_run_dual_support_replay_k9_T7": two_run_dual_support_replay_k9_T7_policy,
    }


def _resolve_time_dependent_policy(k: int, policy_name: str):
    time_policies = _time_dependent_policy_registry()
    if policy_name in time_policies:
        return time_policies[policy_name]
    state_policies = _policy_registry(k)
    if policy_name in state_policies:
        state_policy = state_policies[policy_name]

        def wrapped_policy(state: tuple[int, ...], remaining_horizon: int):
            del remaining_horizon
            return state_policy(state)

        return wrapped_policy
    return None


def explicit_time_policy_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    include_reference: bool = False,
) -> tuple[tuple[ExplicitPolicyBenchmarkRow, ...], tuple[str, ...]]:
    rows: list[ExplicitPolicyBenchmarkRow] = []
    skipped: list[str] = []

    for k, T in cases:
        reference_value: float | None = None
        if include_reference:
            state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
            try:
                reference_actions = _strategy_class_library_actions(k, T, reference_library_name)
            except ValueError as error:
                skipped.append(f"k={k} T={T} reference={reference_library_name}: {error}")
            else:
                reference_value, _, _ = _library_lp_restricted_values_for_actions(
                    k,
                    T,
                    reference_actions,
                    state_layers=state_layers,
                    collect_active_counts=False,
                )
        for policy_name in policy_names:
            policy_fn = _resolve_time_dependent_policy(k, policy_name)
            if policy_fn is None:
                skipped.append(f"k={k} T={T} policy={policy_name}: unknown time-dependent policy")
                continue
            values = evaluate_time_dependent_policy(k, T, policy_fn)
            zero = tuple(0 for _ in range(k))
            value = values[T][zero]
            gap: float | None = None
            normalized_gap: float | None = None
            if reference_value is not None:
                gap = reference_value - value
                normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
            rows.append(
                ExplicitPolicyBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=value,
                    reference_library_name=reference_library_name if reference_value is not None else None,
                    reference_value=reference_value,
                    gap_to_reference=gap,
                    normalized_gap_to_reference=normalized_gap,
                )
            )
    return tuple(rows), tuple(skipped)


def print_explicit_time_policy_benchmark(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    include_reference: bool = False,
) -> None:
    print("Explicit time-dependent policy benchmark")
    print()
    print(f"cases: {cases}")
    print(f"policies: {policy_names}")
    print(f"reference library: {reference_library_name}")
    print(f"include reference: {include_reference}")
    print()
    print(
        "k  T   policy                      V_policy"
        " reference       V_reference gap_to_reference gap_to_reference/sqrt(T)"
    )

    skipped: list[str] = []
    for k, T in cases:
        case_start = time.perf_counter()
        print(f"# start case k={k} T={T}", flush=True)
        reference_value: float | None = None
        if include_reference:
            reference_start = time.perf_counter()
            print(f"# start reference k={k} T={T} library={reference_library_name}", flush=True)
            state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
            try:
                reference_actions = _strategy_class_library_actions(k, T, reference_library_name)
            except ValueError as error:
                skipped.append(f"k={k} T={T} reference={reference_library_name}: {error}")
            else:
                reference_value, _, _ = _library_lp_restricted_values_for_actions(
                    k,
                    T,
                    reference_actions,
                    state_layers=state_layers,
                    collect_active_counts=False,
                )
            print(
                f"# end reference k={k} T={T} library={reference_library_name}"
                f" elapsed={time.perf_counter() - reference_start:.3f}s",
                flush=True,
            )
        case_rows: list[ExplicitPolicyBenchmarkRow] = []
        for policy_name in policy_names:
            policy_fn = _resolve_time_dependent_policy(k, policy_name)
            if policy_fn is None:
                skipped.append(f"k={k} T={T} policy={policy_name}: unknown time-dependent policy")
                continue
            row_start = time.perf_counter()
            print(f"# start row k={k} T={T} policy={policy_name}", flush=True)
            values = evaluate_time_dependent_policy(k, T, policy_fn)
            zero = tuple(0 for _ in range(k))
            value = values[T][zero]
            print(
                f"# end row k={k} T={T} policy={policy_name}"
                f" elapsed={time.perf_counter() - row_start:.3f}s",
                flush=True,
            )
            gap: float | None = None
            normalized_gap: float | None = None
            if reference_value is not None:
                gap = reference_value - value
                normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
            case_rows.append(
                ExplicitPolicyBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=value,
                    reference_library_name=reference_library_name if reference_value is not None else None,
                    reference_value=reference_value,
                    gap_to_reference=gap,
                    normalized_gap_to_reference=normalized_gap,
                )
            )
        for row in sorted(
            case_rows,
            key=lambda item: (
                float("inf") if item.normalized_gap_to_reference is None else item.normalized_gap_to_reference,
                item.policy_name,
            ),
        ):
            print(
                f"{row.k:1d} {row.T:3d}  {row.policy_name:27s}"
                f" {row.value:10.6f}"
                f" {(row.reference_library_name or '-'):15s}"
                f" {_format_optional_float(row.reference_value):>11s}"
                f" {_format_optional_float(row.gap_to_reference):>16s}"
                f" {_format_optional_float(row.normalized_gap_to_reference):>25s}",
                flush=True,
            )
        print(f"# end case k={k} T={T} elapsed={time.perf_counter() - case_start:.3f}s", flush=True)
    print()

    if skipped:
        print("Skipped policies/references:")
        for message in skipped:
            print(f"  {message}")


def print_explicit_policy_benchmark(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
    reference_library_name: str = "two_run",
    include_reference: bool = False,
) -> None:
    print("Explicit policy benchmark")
    print()
    print(f"cases: {cases}")
    print(f"policies: {policy_names}")
    print(f"reference library: {reference_library_name}")
    print(f"include reference: {include_reference}")
    print()
    print(
        "k  T   policy                      V_policy"
        " reference       V_reference gap_to_reference gap_to_reference/sqrt(T)"
    )

    rows: list[ExplicitPolicyBenchmarkRow] = []
    skipped: list[str] = []
    for k, T in cases:
        case_start = time.perf_counter()
        print(f"# start case k={k} T={T}", flush=True)
        policies = _policy_registry(k)
        reference_value: float | None = None
        if include_reference:
            reference_start = time.perf_counter()
            print(f"# start reference k={k} T={T} library={reference_library_name}", flush=True)
            state_layers = tuple(tuple(all_states(k, used)) for used in range(T + 1))
            try:
                reference_actions = _strategy_class_library_actions(k, T, reference_library_name)
            except ValueError as error:
                skipped.append(f"k={k} T={T} reference={reference_library_name}: {error}")
            else:
                reference_value, _, _ = _library_lp_restricted_values_for_actions(
                    k,
                    T,
                    reference_actions,
                    state_layers=state_layers,
                    collect_active_counts=False,
                )
            print(
                f"# end reference k={k} T={T} library={reference_library_name}"
                f" elapsed={time.perf_counter() - reference_start:.3f}s",
                flush=True,
            )
        case_rows: list[ExplicitPolicyBenchmarkRow] = []
        for policy_name in policy_names:
            if policy_name not in policies:
                skipped.append(f"k={k} T={T} policy={policy_name}: unknown policy")
                continue
            row_start = time.perf_counter()
            print(f"# start row k={k} T={T} policy={policy_name}", flush=True)
            values = evaluate_balanced_policy(k, T, policies[policy_name])
            zero = tuple(0 for _ in range(k))
            value = values[T][zero]
            print(
                f"# end row k={k} T={T} policy={policy_name}"
                f" elapsed={time.perf_counter() - row_start:.3f}s",
                flush=True,
            )
            gap: float | None = None
            normalized_gap: float | None = None
            if reference_value is not None:
                gap = reference_value - value
                normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
            case_rows.append(
                ExplicitPolicyBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=value,
                    reference_library_name=reference_library_name if reference_value is not None else None,
                    reference_value=reference_value,
                    gap_to_reference=gap,
                    normalized_gap_to_reference=normalized_gap,
                )
            )
        for row in sorted(
            case_rows,
            key=lambda item: (
                float("inf") if item.normalized_gap_to_reference is None else item.normalized_gap_to_reference,
                item.policy_name,
            ),
        ):
            rows.append(row)
            print(
                f"{row.k:1d} {row.T:3d}  {row.policy_name:27s}"
                f" {row.value:10.6f}"
                f" {(row.reference_library_name or '-'):15s}"
                f" {_format_optional_float(row.reference_value):>11s}"
                f" {_format_optional_float(row.gap_to_reference):>16s}"
                f" {_format_optional_float(row.normalized_gap_to_reference):>25s}",
                flush=True,
            )
        print(f"# end case k={k} T={T} elapsed={time.perf_counter() - case_start:.3f}s", flush=True)
    print()

    if skipped:
        print("Skipped policies/references:")
        for message in skipped:
            print(f"  {message}")


def print_strategy_class_benchmark(
    cases: tuple[tuple[int, int], ...],
    library_names: tuple[str, ...],
    include_probability_matching: bool = False,
    probability_matching_max_k: int = 9,
    probability_matching_max_T: int = 7,
    active_tolerance: float = 1e-8,
    collect_active_counts: bool = True,
    n: int = 20,
) -> None:
    print("Strategy class benchmark")
    print()
    print(f"cases: {cases}")
    print(f"libraries: {library_names}")
    print(f"include probability matching: {include_probability_matching}")
    print(f"probability matching max k/T: {probability_matching_max_k}/{probability_matching_max_T}")
    print(f"active counts: {collect_active_counts}")
    print()
    print(
        "k  T   library          size  V_star      V_restricted"
        " gap       gap/sqrt(T) exact active_actions active_edges"
        " pm_avg_linf pm_weighted_linf"
    )
    rows: list[StrategyClassBenchmarkRow] = []
    skipped: list[str] = []
    for k, T in cases:
        case_start = time.perf_counter()
        print(f"# start case k={k} T={T}", flush=True)
        cache, case_skipped = _strategy_class_benchmark_case_cache(k, T, library_names)
        print(f"# built case k={k} T={T} in {time.perf_counter() - case_start:.3f}s", flush=True)
        skipped.extend(case_skipped)
        case_rows: list[StrategyClassBenchmarkRow] = []
        for library_name in library_names:
            if library_name not in cache.library_actions_by_name:
                continue
            row_start = time.perf_counter()
            print(f"# start row k={k} T={T} library={library_name}", flush=True)
            row, row_skipped = _strategy_class_benchmark_row_for_library(
                cache,
                library_name,
                include_probability_matching=include_probability_matching,
                probability_matching_max_k=probability_matching_max_k,
                probability_matching_max_T=probability_matching_max_T,
                active_tolerance=active_tolerance,
                collect_active_counts=collect_active_counts,
            )
            skipped.extend(row_skipped)
            print(
                f"# end row k={k} T={T} library={library_name}"
                f" elapsed={time.perf_counter() - row_start:.3f}s",
                flush=True,
            )
            case_rows.append(row)
        for row in sorted(case_rows, key=lambda item: (item.normalized_gap, item.library_name)):
            rows.append(row)
            print(
                f"{row.k:1d} {row.T:3d}  {row.library_name:15s}"
                f" {row.library_size:5d}"
                f" {row.optimal_value:10.6f}"
                f" {row.value:12.6f}"
                f" {row.gap:9.6f}"
                f" {row.normalized_gap:12.6f}"
                f" {str(row.exact):5s}"
                f" {_format_optional_int(row.active_action_count):>14s}"
                f" {_format_optional_int(row.active_edge_signature_count):>12s}"
                f" {_format_optional_float(row.probability_matching_avg_linf):>12s}"
                f" {_format_optional_float(row.probability_matching_weighted_linf):>18s}",
                flush=True,
            )
        print(f"# end case k={k} T={T} elapsed={time.perf_counter() - case_start:.3f}s", flush=True)
    print()

    if skipped:
        print("Skipped libraries:")
        for message in skipped:
            print(f"  {message}")
        print()

    print("Top active edge signatures by case/library:")
    grouped: dict[tuple[int, int], list[StrategyClassBenchmarkRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.k, row.T)].append(row)
    for case, case_rows in sorted(grouped.items()):
        print(f"k={case[0]} T={case[1]}")
        for row in sorted(case_rows, key=lambda item: (item.normalized_gap, item.library_name))[:n]:
            rendered = ", ".join(
                f"{signature}:{count}"
                for signature, count in row.top_active_edge_signatures[:5]
            )
            if not rendered:
                rendered = "-"
            print(f"  {row.library_name:15s} gap/sqrt(T)={row.normalized_gap:.6f} active_edges=[{rendered}]")
        print()


def print_k3_motif_sweep(T_values: tuple[int, ...]) -> None:
    rows = k3_motif_sweep(T_values)
    print("k=3 motif sweep")
    print()
    print(f"T values: {T_values}")
    print()
    print(
        "T    library          size  V_star      V_det       det_gap"
        "  det_gap/sqrt(T)  V_lp        lp_gap   lp_gap/sqrt(T)"
    )
    for row in rows:
        scale = row.T ** 0.5
        if row.deterministic_value is None or row.deterministic_gap is None:
            det_value = "    -     "
            det_gap = "    -    "
            det_norm = "      -       "
        else:
            det_value = f"{row.deterministic_value:10.6f}"
            det_gap = f"{row.deterministic_gap:9.6f}"
            det_norm = f"{row.deterministic_gap / scale:15.6f}"
        print(
            f"{row.T:3d}  {row.library_name:15s} {row.library_size:4d}"
            f"  {row.optimal_value:10.6f} {det_value} {det_gap} {det_norm}"
            f"  {row.lp_value:10.6f} {row.lp_gap:8.6f} {row.lp_gap / scale:15.6f}"
        )
    print()

    print("Best LP motif family by T:")
    by_T: dict[int, list[K3MotifSweepRow]] = defaultdict(list)
    for row in rows:
        by_T[row.T].append(row)
    for T in T_values:
        best = min(by_T[T], key=lambda row: (row.lp_gap, row.library_size, row.library_name))
        print(
            f"  T={T}: {best.library_name}"
            f" lp_gap={best.lp_gap:.6f}"
            f" lp_gap/sqrt(T)={best.lp_gap / (T ** 0.5):.6f}"
        )


def _actions_with_edge_signatures(
    actions: tuple[tuple[int, ...], ...],
    signatures: set[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    selected: set[tuple[int, ...]] = set()
    for action in actions:
        if _edge_signature(action) not in signatures:
            continue
        selected.add(action)
        selected.add(complement(action))
    return tuple(sorted(selected, key=_format_action))


def _k9_observed_two_run_data(
    k: int,
    T: int,
    support_tolerance: float,
) -> tuple[tuple[tuple[int, ...], ...], tuple[LibraryLPDualInspectRow, ...]]:
    two_run_actions = _action_library(k, "two_run", T=T)
    _, rows = library_lp_dual_inspect_rows(
        k,
        T,
        "two_run",
        support_tolerance=support_tolerance,
    )
    return two_run_actions, tuple(rows)


def _k9_motif_libraries(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[
    dict[str, tuple[tuple[int, ...], ...]],
    tuple[LibraryLPDualInspectRow, ...],
]:
    two_run_actions, observed_rows = _k9_observed_two_run_data(k, T, support_tolerance)
    observed_used_actions: set[tuple[int, ...]] = set()
    observed_orbit_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    observed_small_orbit_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    observed_balanced_orbit_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for row in observed_rows:
        groups = _packet_index_groups(row.state)
        packet_sizes = tuple(len(group) for group in groups)
        for _, action, _ in row.support:
            observed_used_actions.add(action)
            key = _orbit_key_for_action(action, groups)
            pair = (packet_sizes, key)
            observed_orbit_pairs.add(pair)
            if sum(1 for count in key if count > 0) <= 2:
                observed_small_orbit_pairs.add(pair)
            active_exposures = [
                count / packet_size
                for packet_size, count in zip(packet_sizes, key, strict=True)
                if count > 0
            ]
            if active_exposures and all(abs(exposure - 0.5) <= 0.3 for exposure in active_exposures):
                observed_balanced_orbit_pairs.add(pair)

    def actions_realizing_observed_pairs(
        observed_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> tuple[tuple[int, ...], ...]:
        selected: set[tuple[int, ...]] = set()
        by_packet_sizes = {
            tuple(len(group) for group in _packet_index_groups(row.state)): _packet_index_groups(row.state)
            for row in observed_rows
        }
        for action in two_run_actions:
            for packet_sizes, groups in by_packet_sizes.items():
                if (packet_sizes, _orbit_key_for_action(action, groups)) in observed_pairs:
                    selected.add(action)
                    selected.add(complement(action))
                    break
        return tuple(sorted(selected, key=_format_action))

    active_counts = library_lp_restricted_optimal(k, T, "two_run").active_edge_signature_counts
    libraries: dict[str, tuple[tuple[int, ...], ...]] = {
        "top_prefix_all": _action_library(k, "top_prefix_all", T=T),
        "one_run": _action_library(k, "one_run", T=T),
        "two_run": two_run_actions,
        "origin_orbit_only": tuple(
            sorted(
                {
                    candidate
                    for action in two_run_actions
                    if sum(action) == 4
                    for candidate in (action, complement(action))
                },
                key=_format_action,
            )
        ),
        "packet_count_small": actions_realizing_observed_pairs(observed_small_orbit_pairs),
        "packet_count_balanced": actions_realizing_observed_pairs(observed_balanced_orbit_pairs),
        "observed_orbit_keys": actions_realizing_observed_pairs(observed_orbit_pairs),
        "observed_used_actions": tuple(sorted(observed_used_actions, key=_format_action)),
        "observed_used_actions_plus_complements": tuple(
            sorted(
                {
                    candidate
                    for action in observed_used_actions
                    for candidate in (action, complement(action))
                },
                key=_format_action,
            )
        ),
        "all": tuple(all_actions(k)),
    }
    for top_n in (5, 10, 20, 40):
        signatures = {
            tuple(int(bit) for bit in signature)
            for signature, _ in active_counts[:top_n]
        }
        libraries[f"top_{top_n}_active_edge_signatures"] = _actions_with_edge_signatures(two_run_actions, signatures)
    return libraries, observed_rows


def k9_motif_library_sweep(
    T_values: tuple[int, ...],
    support_tolerance: float = 1e-8,
) -> tuple[K9MotifLibrarySweepRow, ...]:
    rows: list[K9MotifLibrarySweepRow] = []
    for T in T_values:
        libraries, observed_rows = _k9_motif_libraries(9, T, support_tolerance=support_tolerance)
        for family_name, actions in libraries.items():
            if not actions:
                continue
            rows.append(
                _library_lp_restricted_result_for_actions(
                    9,
                    T,
                    family_name,
                    actions,
                    orbit_rows=observed_rows,
                )
            )
    return tuple(rows)


def print_k9_motif_library_sweep(
    T_values: tuple[int, ...],
    support_tolerance: float = 1e-8,
    n: int = 10,
) -> None:
    print("k=9 motif-library sweep")
    print()
    print(f"T values: {T_values}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(
        "family definitions: observed_* families are static libraries derived from"
        " the two_run dual occupancy at the same T; packet_count_* are observed"
        " packet-size/orbit-key pairs realized by two_run actions."
    )
    print()
    print("T  family                            size  V_star      V_restricted      gap   gap/sqrt(T)  exact")
    rows = k9_motif_library_sweep(T_values, support_tolerance=support_tolerance)
    for T in T_values:
        print(f"T={T}")
        for row in sorted(
            (row for row in rows if row.T == T),
            key=lambda item: (item.normalized_gap, item.library_size, item.family_name),
        ):
            print(
                f"{T:1d}  {row.family_name:32s} {row.library_size:5d}"
                f"  {row.optimal_value:10.6f} {row.value:13.6f}"
                f" {row.gap:8.6f} {row.normalized_gap:13.6f}"
                f"  {row.gap <= 1e-8}"
            )
            if row.top_active_edge_signatures:
                rendered_edges = ", ".join(
                    f"{signature}:{count}"
                    for signature, count in row.top_active_edge_signatures[:n]
                )
                print(f"    active edge signatures: {rendered_edges}")
            if row.top_active_orbit_keys:
                rendered_orbits = ", ".join(
                    f"{key}:{count}"
                    for key, count in row.top_active_orbit_keys[:n]
                )
                print(f"    active orbit keys: {rendered_orbits}")
        print()


def _named_experiment_action_library(
    k: int,
    library_name: str,
    T: int,
) -> tuple[tuple[int, ...], ...]:
    if k == 3 and library_name in _k3_motif_libraries():
        return _k3_motif_libraries()[library_name]
    return _action_library(k, library_name, T=T)


def _terminal_winner_probabilities(state: tuple[int, ...]) -> tuple[float, ...]:
    if not state:
        return ()
    maximum = max(state)
    winner_count = sum(1 for value in state if value == maximum)
    return tuple(
        1.0 / winner_count if value == maximum else 0.0
        for value in state
    )


def _winner_probability_layers_for_dual_policy(
    k: int,
    T: int,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
) -> list[dict[tuple[int, ...], tuple[float, ...]]]:
    state_layers = [tuple(all_states(k, used)) for used in range(T + 1)]
    winner_layers: list[dict[tuple[int, ...], tuple[float, ...]]] = [
        {state: _terminal_winner_probabilities(state) for state in state_layers[T]}
    ]

    for horizon in range(1, T + 1):
        continuation_values = value_layers[horizon - 1]
        continuation_winners = winner_layers[horizon - 1]
        current: dict[tuple[int, ...], tuple[float, ...]] = {}
        for state in state_layers[T - horizon]:
            q_by_action = {
                action: _next_state_value(state, action, continuation_values)
                for action in library_actions
            }
            dual = solve_adversary_dual(q_by_action, k)
            if not dual.success:
                raise RuntimeError(f"dual LP failed at state {state}: {dual.message}")
            probabilities = [0.0] * k
            retained_mass = 0.0
            for action, weight in dual.weights_by_action:
                if weight < support_tolerance:
                    continue
                retained_mass += weight
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                next_probabilities = continuation_winners[next_state]
                for index, probability in enumerate(next_probabilities):
                    probabilities[index] += weight * probability
            if retained_mass > 0 and retained_mass < 1.0:
                probabilities = [probability / retained_mass for probability in probabilities]
            current[state] = tuple(probabilities)
        winner_layers.append(current)
    return winner_layers


def probability_matching_inspect_rows(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
) -> tuple[list[dict[tuple[int, ...], float]], list[ProbabilityMatchingInspectRow]]:
    library_actions = _named_experiment_action_library(k, library_name, T)
    value_layers = _library_lp_value_layers(k, T, library_actions)
    winner_layers = _winner_probability_layers_for_dual_policy(
        k,
        T,
        library_actions,
        value_layers,
        support_tolerance,
    )
    zero = tuple(0 for _ in range(k))
    occupancy: list[defaultdict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    rows: list[ProbabilityMatchingInspectRow] = []

    for time in range(T):
        remaining_horizon = T - time
        continuation_values = value_layers[remaining_horizon - 1]
        for state, probability in tuple(occupancy[time].items()):
            q_by_action = {
                action: _next_state_value(state, action, continuation_values)
                for action in library_actions
            }
            primal = solve_minimax_step(q_by_action, k)
            if not primal.success:
                raise RuntimeError(f"primal LP failed at state {state}: {primal.message}")
            dual = solve_adversary_dual(q_by_action, k)
            if not dual.success:
                raise RuntimeError(f"dual LP failed at state {state}: {dual.message}")

            ptype = packet_type(state)
            gaps = _packet_gaps(state)
            winner_probabilities = winner_layers[remaining_horizon][state]
            differences = tuple(
                primal.p[index] - winner_probabilities[index]
                for index in range(k)
            )
            if (
                (packet_type_filter is None or ptype == packet_type_filter)
                and (packet_gaps_filter is None or gaps == packet_gaps_filter)
            ):
                rows.append(
                    ProbabilityMatchingInspectRow(
                        time=time,
                        remaining_horizon=remaining_horizon,
                        state=state,
                        packet_type=ptype,
                        packet_gaps=gaps,
                        occupancy_probability=probability,
                        learner_distribution=primal.p,
                        winner_probabilities=winner_probabilities,
                        l1_error=sum(abs(value) for value in differences),
                        linf_error=max((abs(value) for value in differences), default=0.0),
                    )
                )

            for action, weight in dual.weights_by_action:
                if weight < support_tolerance:
                    continue
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * weight

    return [dict(layer) for layer in occupancy], rows


def print_probability_matching_inspect(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    n: int = 20,
) -> None:
    library_actions = _named_experiment_action_library(k, library_name, T)
    occupancy, rows = probability_matching_inspect_rows(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    values = _library_lp_value_layers(k, T, library_actions)
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = values[T][zero]
    total_occupancy = sum(row.occupancy_probability for row in rows)
    total_weighted_l1 = sum(row.weighted_l1_error for row in rows)
    total_weighted_linf = sum(row.weighted_linf_error for row in rows)

    print(f"Probability matching inspect, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"total occupancy mass: {sum(sum(layer.values()) for layer in occupancy[:-1]):.6f}")
    print(f"matching rows: {len(rows)}")
    print(f"weighted L1 error: {total_weighted_l1:.6f}")
    print(f"weighted Linf error: {total_weighted_linf:.6f}")
    print(f"avg L1 error: {total_weighted_l1 / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print(f"avg Linf error: {total_weighted_linf / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print()

    regime_weights: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    regime_l1: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    regime_linf: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for row in rows:
        key = (row.packet_type, row.packet_gaps)
        regime_weights[key] += row.occupancy_probability
        regime_l1[key] += row.weighted_l1_error
        regime_linf[key] += row.weighted_linf_error

    print("Top regimes by weighted Linf error:")
    for key, weight in sorted(regime_linf.items(), key=lambda pair: (-pair[1], pair[0]))[:n]:
        ptype, gaps = key
        occupancy_mass = regime_weights[key]
        print(
            f"  packet type={ptype} gaps={gaps}"
            f" occupancy={occupancy_mass:.6f}"
            f" weighted_l1={regime_l1[key]:.6f}"
            f" weighted_linf={weight:.6f}"
            f" avg_linf={weight / occupancy_mass if occupancy_mass > 0 else 0.0:.6f}"
        )
    print()

    print("Top rows by occupancy-weighted Linf error:")
    for row in sorted(rows, key=lambda item: (-item.weighted_linf_error, item.time, item.state))[:n]:
        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
            f" l1={row.l1_error:.6f} linf={row.linf_error:.6f}"
            f" weighted_linf={row.weighted_linf_error:.6f}"
        )
        print(f"  learner p={_format_float_tuple(row.learner_distribution)}")
        print(f"  winner probs={_format_float_tuple(row.winner_probabilities)}")


def _named_score_groups(named_state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    by_score: dict[int, list[int]] = defaultdict(list)
    for index, score in enumerate(named_state):
        by_score[score].append(index)
    return tuple(
        tuple(by_score[score])
        for score in sorted(by_score, reverse=True)
    )


def _lift_canonical_vector_to_named(
    named_state: tuple[int, ...],
    canonical_vector: tuple[float, ...],
) -> tuple[float, ...]:
    named_vector = [0.0] * len(named_state)
    offset = 0
    for group in _named_score_groups(named_state):
        group_mass = sum(canonical_vector[offset : offset + len(group)])
        share = group_mass / len(group)
        for named_index in group:
            named_vector[named_index] = share
        offset += len(group)
    return tuple(named_vector)


def _lift_canonical_action_to_named_distribution(
    named_state: tuple[int, ...],
    canonical_action: tuple[int, ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    partials: list[tuple[float, tuple[int, ...]]] = [(1.0, tuple(0 for _ in named_state))]
    offset = 0
    for group in _named_score_groups(named_state):
        packet_action = canonical_action[offset : offset + len(group)]
        ones_count = sum(packet_action)
        choices = tuple(combinations(group, ones_count))
        if not choices:
            choices = ((),)
        next_partials: list[tuple[float, tuple[int, ...]]] = []
        choice_probability = 1.0 / len(choices)
        for probability, bits in partials:
            for choice in choices:
                lifted = list(bits)
                for named_index in choice:
                    lifted[named_index] = 1
                next_partials.append((probability * choice_probability, tuple(lifted)))
        partials = next_partials
        offset += len(group)

    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for probability, action in partials:
        aggregated[action] += probability
    return tuple(
        sorted(
            ((probability, action) for action, probability in aggregated.items()),
            key=lambda item: item[1],
        )
    )


def _lift_dual_to_named_distribution(
    named_state: tuple[int, ...],
    dual_weights: tuple[tuple[tuple[int, ...], float], ...],
    support_tolerance: float,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for canonical_action, dual_weight in dual_weights:
        if dual_weight < support_tolerance:
            continue
        for lift_probability, named_action in _lift_canonical_action_to_named_distribution(
            named_state,
            canonical_action,
        ):
            aggregated[named_action] += dual_weight * lift_probability
    total = sum(aggregated.values())
    if total <= 0:
        raise ValueError("lifted dual distribution has zero mass")
    return tuple(
        sorted(
            ((probability / total, action) for action, probability in aggregated.items()),
            key=lambda item: item[1],
        )
    )


def _terminal_named_winner_probabilities(named_state: tuple[int, ...]) -> tuple[float, ...]:
    maximum = max(named_state)
    winners = tuple(index for index, value in enumerate(named_state) if value == maximum)
    return tuple(
        1.0 / len(winners) if index in winners else 0.0
        for index in range(len(named_state))
    )


def _named_probability_matching_dual_policy(
    named_state: tuple[int, ...],
    remaining_horizon: int,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ],
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ],
) -> tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]]:
    named_key = (named_state, remaining_horizon)
    if named_key in named_policy_memo:
        return named_policy_memo[named_key]

    canonical_state = canon(named_state)
    canonical_key = (canonical_state, remaining_horizon)
    if canonical_key not in canonical_policy_memo:
        continuation_values = value_layers[remaining_horizon - 1]
        q_by_action = {
            action: _next_state_value(canonical_state, action, continuation_values)
            for action in library_actions
        }
        primal = solve_minimax_step(q_by_action, len(named_state))
        if not primal.success:
            raise RuntimeError(f"primal LP failed at named state {named_state}: {primal.message}")
        dual = solve_adversary_dual(q_by_action, len(named_state))
        if not dual.success:
            raise RuntimeError(f"dual LP failed at named state {named_state}: {dual.message}")
        canonical_policy_memo[canonical_key] = (dual.weights_by_action, primal.p)

    dual_weights, canonical_p = canonical_policy_memo[canonical_key]
    result = (
        _lift_dual_to_named_distribution(named_state, dual_weights, support_tolerance),
        canonical_p,
    )
    named_policy_memo[named_key] = result
    return result


def _named_winner_probabilities_from_state(
    named_state: tuple[int, ...],
    remaining_horizon: int,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ],
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ],
    memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]],
) -> tuple[float, ...]:
    key = (named_state, remaining_horizon)
    if key in memo:
        return memo[key]
    if remaining_horizon == 0:
        result = _terminal_named_winner_probabilities(named_state)
        memo[key] = result
        return result

    lifted_dual, _ = _named_probability_matching_dual_policy(
        named_state,
        remaining_horizon,
        library_actions,
        value_layers,
        support_tolerance,
        canonical_policy_memo,
        named_policy_memo,
    )
    probabilities = [0.0] * len(named_state)
    for action_probability, named_action in lifted_dual:
        next_state = tuple(named_state[index] + named_action[index] for index in range(len(named_state)))
        next_probabilities = _named_winner_probabilities_from_state(
            next_state,
            remaining_horizon - 1,
            library_actions,
            value_layers,
            support_tolerance,
            canonical_policy_memo,
            named_policy_memo,
            memo,
        )
        for index, probability in enumerate(next_probabilities):
            probabilities[index] += action_probability * probability
    result = tuple(probabilities)
    memo[key] = result
    return result


def probability_matching_named_inspect_rows(
    k: int,
    T: int,
    library_name: str,
    support_tolerance: float = 1e-8,
) -> tuple[list[dict[tuple[int, ...], float]], list[NamedProbabilityMatchingInspectRow]]:
    library_actions = _named_experiment_action_library(k, library_name, T)
    value_layers = _library_lp_value_layers(k, T, library_actions)
    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ] = {}
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ] = {}
    memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]] = {}
    zero = tuple(0 for _ in range(k))
    occupancy: list[defaultdict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    rows: list[NamedProbabilityMatchingInspectRow] = []

    for time in range(T):
        remaining_horizon = T - time
        for named_state, probability in tuple(occupancy[time].items()):
            lifted_dual, canonical_p = _named_probability_matching_dual_policy(
                named_state,
                remaining_horizon,
                library_actions,
                value_layers,
                support_tolerance,
                canonical_policy_memo,
                named_policy_memo,
            )
            lifted_p = _lift_canonical_vector_to_named(named_state, canonical_p)
            winner_probabilities = _named_winner_probabilities_from_state(
                named_state,
                remaining_horizon,
                library_actions,
                value_layers,
                support_tolerance,
                canonical_policy_memo,
                named_policy_memo,
                memo,
            )
            differences = tuple(
                lifted_p[index] - winner_probabilities[index]
                for index in range(k)
            )
            canonical_state = canon(named_state)
            rows.append(
                NamedProbabilityMatchingInspectRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    named_state=named_state,
                    canonical_state=canonical_state,
                    packet_type=packet_type(canonical_state),
                    packet_gaps=_packet_gaps(canonical_state),
                    occupancy_probability=probability,
                    lifted_learner_distribution=lifted_p,
                    winner_probabilities=winner_probabilities,
                    l1_error=sum(abs(value) for value in differences),
                    linf_error=max((abs(value) for value in differences), default=0.0),
                )
            )
            for action_probability, named_action in lifted_dual:
                next_state = tuple(named_state[index] + named_action[index] for index in range(k))
                occupancy[time + 1][next_state] += probability * action_probability

    return [dict(layer) for layer in occupancy], rows


def print_probability_matching_named_inspect(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    n: int = 20,
) -> None:
    library_actions = _named_experiment_action_library(k, library_name, T)
    occupancy, all_rows = probability_matching_named_inspect_rows(
        k,
        T,
        library_name,
        support_tolerance=support_tolerance,
    )
    rows = [
        row
        for row in all_rows
        if (packet_type_filter is None or row.packet_type == packet_type_filter)
        and (packet_gaps_filter is None or row.packet_gaps == packet_gaps_filter)
    ]
    value_layers = _library_lp_value_layers(k, T, library_actions)
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = value_layers[T][zero]
    total_occupancy = sum(row.occupancy_probability for row in rows)
    total_weighted_l1 = sum(row.weighted_l1_error for row in rows)
    total_weighted_linf = sum(row.weighted_linf_error for row in rows)

    print(f"Named probability matching inspect, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"total named occupancy mass: {sum(sum(layer.values()) for layer in occupancy[:-1]):.6f}")
    print(f"matching rows: {len(rows)}")
    print(f"weighted L1 error: {total_weighted_l1:.6f}")
    print(f"weighted Linf error: {total_weighted_linf:.6f}")
    print(f"avg L1 error: {total_weighted_l1 / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print(f"avg Linf error: {total_weighted_linf / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print()

    print("Top rows by occupancy-weighted Linf error:")
    for row in sorted(rows, key=lambda item: (-item.weighted_linf_error, item.time, item.named_state))[:n]:
        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" named={row.named_state} canon={row.canonical_state}"
            f" ptype={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
            f" l1={row.l1_error:.6f} linf={row.linf_error:.6f}"
            f" weighted_linf={row.weighted_linf_error:.6f}"
        )
        print(f"  lifted learner p={_format_float_tuple(row.lifted_learner_distribution)}")
        print(f"  named winner probs={_format_float_tuple(row.winner_probabilities)}")


def _aggregate_named_probability_matching_residuals(
    rows: list[NamedProbabilityMatchingInspectRow],
    key_fn,
) -> list[NamedProbabilityMatchingResidualAggregate]:
    grouped: dict[tuple[object, ...], list[NamedProbabilityMatchingInspectRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)

    aggregates: list[NamedProbabilityMatchingResidualAggregate] = []
    for key, items in grouped.items():
        total_occupancy = sum(item.occupancy_probability for item in items)
        weighted_l1_error = sum(item.weighted_l1_error for item in items)
        weighted_linf_error = sum(item.weighted_linf_error for item in items)
        representative = max(
            items,
            key=lambda item: (item.weighted_linf_error, item.linf_error, item.occupancy_probability),
        )
        aggregates.append(
            NamedProbabilityMatchingResidualAggregate(
                key=key,
                named_row_count=len(items),
                total_occupancy=total_occupancy,
                weighted_l1_error=weighted_l1_error,
                weighted_linf_error=weighted_linf_error,
                avg_l1_error=weighted_l1_error / total_occupancy if total_occupancy > 0 else 0.0,
                avg_linf_error=weighted_linf_error / total_occupancy if total_occupancy > 0 else 0.0,
                max_linf_error=max(item.linf_error for item in items),
                representative_row=representative,
            )
        )
    return sorted(aggregates, key=lambda item: (-item.weighted_linf_error, item.key))


def probability_matching_named_residual_aggregates(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
) -> tuple[
    list[NamedProbabilityMatchingInspectRow],
    list[NamedProbabilityMatchingResidualAggregate],
    list[NamedProbabilityMatchingResidualAggregate],
]:
    _, all_rows = probability_matching_named_inspect_rows(
        k,
        T,
        library_name,
        support_tolerance=support_tolerance,
    )
    rows = [
        row
        for row in all_rows
        if (packet_type_filter is None or row.packet_type == packet_type_filter)
        and (packet_gaps_filter is None or row.packet_gaps == packet_gaps_filter)
    ]
    canonical_aggregates = _aggregate_named_probability_matching_residuals(
        rows,
        lambda row: (
            row.remaining_horizon,
            row.canonical_state,
            row.packet_type,
            row.packet_gaps,
        ),
    )
    regime_aggregates = _aggregate_named_probability_matching_residuals(
        rows,
        lambda row: (row.packet_type, row.packet_gaps),
    )
    return rows, canonical_aggregates, regime_aggregates


def _named_residual_dual_orbit_summary(
    row: NamedProbabilityMatchingInspectRow,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    orbit_n: int,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    continuation_values = value_layers[row.remaining_horizon - 1]
    q_by_action = {
        action: _next_state_value(row.canonical_state, action, continuation_values)
        for action in library_actions
    }
    dual = solve_adversary_dual(q_by_action, len(row.canonical_state))
    if not dual.success:
        raise RuntimeError(f"dual LP failed at state {row.canonical_state}: {dual.message}")

    groups = _packet_index_groups(row.canonical_state)
    orbit_weights: dict[tuple[int, ...], float] = defaultdict(float)
    for action, weight in dual.weights_by_action:
        if weight < support_tolerance:
            continue
        orbit_weights[_orbit_key_for_action(action, groups)] += weight
    return tuple(
        sorted(orbit_weights.items(), key=lambda item: (-item[1], item[0]))[:orbit_n]
    )


def _format_named_residual_difference(row: NamedProbabilityMatchingInspectRow) -> str:
    return _format_float_tuple(
        tuple(
            row.lifted_learner_distribution[index] - row.winner_probabilities[index]
            for index in range(len(row.named_state))
        )
    )


def _print_named_residual_aggregate(
    aggregate: NamedProbabilityMatchingResidualAggregate,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    orbit_n: int,
) -> None:
    row = aggregate.representative_row
    print(
        f"  key={aggregate.key}"
        f" named_rows={aggregate.named_row_count}"
        f" occ={aggregate.total_occupancy:.6f}"
        f" weighted_l1={aggregate.weighted_l1_error:.6f}"
        f" weighted_linf={aggregate.weighted_linf_error:.6f}"
        f" avg_l1={aggregate.avg_l1_error:.6f}"
        f" avg_linf={aggregate.avg_linf_error:.6f}"
        f" max_linf={aggregate.max_linf_error:.6f}"
    )
    print(
        f"    representative t={row.time:2d} rem={row.remaining_horizon:2d}"
        f" named={row.named_state} canon={row.canonical_state}"
        f" occ={row.occupancy_probability:.6f}"
        f" linf={row.linf_error:.6f}"
        f" weighted_linf={row.weighted_linf_error:.6f}"
    )
    print(f"    lifted learner p={_format_float_tuple(row.lifted_learner_distribution)}")
    print(f"    named winner probs={_format_float_tuple(row.winner_probabilities)}")
    print(f"    p - winner={_format_named_residual_difference(row)}")
    orbit_summary = _named_residual_dual_orbit_summary(
        row,
        library_actions,
        value_layers,
        support_tolerance,
        orbit_n,
    )
    rendered_orbits = ", ".join(
        f"{orbit_key}:{weight:.6f}"
        for orbit_key, weight in orbit_summary
    )
    print(f"    canonical dual orbit weights=[{rendered_orbits}]")


def print_probability_matching_named_residuals(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    n: int = 30,
    orbit_n: int = 8,
) -> None:
    library_actions = _named_experiment_action_library(k, library_name, T)
    value_layers = _library_lp_value_layers(k, T, library_actions)
    rows, canonical_aggregates, regime_aggregates = probability_matching_named_residual_aggregates(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = value_layers[T][zero]
    total_occupancy = sum(row.occupancy_probability for row in rows)
    total_weighted_l1 = sum(row.weighted_l1_error for row in rows)
    total_weighted_linf = sum(row.weighted_linf_error for row in rows)

    print(f"Named probability matching residuals, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"matching rows: {len(rows)}")
    print(f"total occupancy: {total_occupancy:.6f}")
    print(f"weighted L1 error: {total_weighted_l1:.6f}")
    print(f"weighted Linf error: {total_weighted_linf:.6f}")
    print(f"avg L1 error: {total_weighted_l1 / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print(f"avg Linf error: {total_weighted_linf / total_occupancy if total_occupancy > 0 else 0.0:.6f}")
    print()

    print("Top canonical-state residual aggregates:")
    for aggregate in canonical_aggregates[:n]:
        _print_named_residual_aggregate(
            aggregate,
            library_actions,
            value_layers,
            support_tolerance,
            orbit_n,
        )
    print()

    print("Top packet-regime residual aggregates:")
    for aggregate in regime_aggregates[:n]:
        _print_named_residual_aggregate(
            aggregate,
            library_actions,
            value_layers,
            support_tolerance,
            orbit_n,
        )


def _canonical_action_named_successor_winner_probabilities(
    named_state: tuple[int, ...],
    remaining_horizon: int,
    canonical_action: tuple[int, ...],
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ],
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ],
    winner_memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]],
) -> tuple[float, ...]:
    probabilities = [0.0] * len(named_state)
    for lift_probability, named_action in _lift_canonical_action_to_named_distribution(
        named_state,
        canonical_action,
    ):
        next_state = tuple(named_state[index] + named_action[index] for index in range(len(named_state)))
        next_probabilities = _named_winner_probabilities_from_state(
            next_state,
            remaining_horizon - 1,
            library_actions,
            value_layers,
            support_tolerance,
            canonical_policy_memo,
            named_policy_memo,
            winner_memo,
        )
        for index, probability in enumerate(next_probabilities):
            probabilities[index] += lift_probability * probability
    return tuple(probabilities)


def _repair_named_probability_matching_row(
    row: NamedProbabilityMatchingInspectRow,
    library_actions: tuple[tuple[int, ...], ...],
    value_layers: list[dict[tuple[int, ...], float]],
    support_tolerance: float,
    active_tolerance: float,
    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ],
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ],
    winner_memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]],
) -> DualFaceRepairRow:
    k = len(row.named_state)
    continuation_values = value_layers[row.remaining_horizon - 1]
    q_by_action = {
        action: _next_state_value(row.canonical_state, action, continuation_values)
        for action in library_actions
    }
    primal = solve_minimax_step(q_by_action, k)
    if not primal.success:
        raise RuntimeError(f"primal LP failed at state {row.canonical_state}: {primal.message}")

    scores = {
        action: q_value - sum(primal.p[index] * action[index] for index in range(k))
        for action, q_value in q_by_action.items()
    }
    active_actions = tuple(
        action
        for action, score in scores.items()
        if score >= primal.value - active_tolerance
    )
    if not active_actions:
        return DualFaceRepairRow(
            original_row=row,
            repaired_winner_probabilities=row.winner_probabilities,
            repaired_l1_error=row.l1_error,
            repaired_linf_error=row.linf_error,
            active_candidate_count=0,
            support=(),
            success=False,
            message="no active actions",
        )

    successor_winners = tuple(
        _canonical_action_named_successor_winner_probabilities(
            row.named_state,
            row.remaining_horizon,
            action,
            library_actions,
            value_layers,
            support_tolerance,
            canonical_policy_memo,
            named_policy_memo,
            winner_memo,
        )
        for action in active_actions
    )

    action_count = len(active_actions)
    variable_count = action_count + 2 * k
    c = np.zeros(variable_count)
    c[action_count:] = 1.0

    a_eq = np.zeros((1 + k, variable_count))
    b_eq = np.zeros(1 + k)
    a_eq[0, :action_count] = 1.0
    b_eq[0] = 1.0
    for expert in range(k):
        row_index = 1 + expert
        for action_index, winner_probabilities in enumerate(successor_winners):
            a_eq[row_index, action_index] = winner_probabilities[expert]
        a_eq[row_index, action_count + expert] = 1.0
        a_eq[row_index, action_count + k + expert] = -1.0
        b_eq[row_index] = row.lifted_learner_distribution[expert]

    a_ub = np.zeros((k, variable_count))
    b_ub = np.full(k, -primal.value + active_tolerance)
    for expert in range(k):
        for action_index, action in enumerate(active_actions):
            a_ub[expert, action_index] = float(action[expert]) - float(q_by_action[action])

    bounds = [(0.0, None)] * variable_count
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
        return DualFaceRepairRow(
            original_row=row,
            repaired_winner_probabilities=row.winner_probabilities,
            repaired_l1_error=row.l1_error,
            repaired_linf_error=row.linf_error,
            active_candidate_count=action_count,
            support=(),
            success=False,
            message=result.message,
        )

    weights = tuple(float(value) for value in result.x[:action_count])
    repaired = tuple(
        float(
            sum(
                weights[action_index] * successor_winners[action_index][expert]
                for action_index in range(action_count)
            )
        )
        for expert in range(k)
    )
    differences = tuple(
        row.lifted_learner_distribution[index] - repaired[index]
        for index in range(k)
    )
    groups = _packet_index_groups(row.canonical_state)
    support = tuple(
        sorted(
            (
                (weight, action, _edge_signature(action), _orbit_key_for_action(action, groups))
                for action, weight in zip(active_actions, weights, strict=True)
                if weight >= support_tolerance
            ),
            key=lambda item: (-item[0], _format_action(item[1])),
        )
    )
    return DualFaceRepairRow(
        original_row=row,
        repaired_winner_probabilities=repaired,
        repaired_l1_error=sum(abs(value) for value in differences),
        repaired_linf_error=max((abs(value) for value in differences), default=0.0),
        active_candidate_count=action_count,
        support=support,
        success=True,
        message=result.message,
    )


def probability_matching_dual_face_repair_rows(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    active_tolerance: float = 1e-8,
    n: int = 20,
) -> tuple[list[NamedProbabilityMatchingInspectRow], list[DualFaceRepairRow]]:
    library_actions = _named_experiment_action_library(k, library_name, T)
    value_layers = _library_lp_value_layers(k, T, library_actions)
    all_rows, canonical_aggregates, _ = probability_matching_named_residual_aggregates(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
    )
    selected_rows: list[NamedProbabilityMatchingInspectRow] = []
    seen: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    for aggregate in canonical_aggregates:
        row = aggregate.representative_row
        key = (row.remaining_horizon, row.canonical_state, row.named_state)
        if key in seen:
            continue
        seen.add(key)
        selected_rows.append(row)
        if len(selected_rows) >= n:
            break

    canonical_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[int, ...], float], tuple[float, ...]],
    ] = {}
    named_policy_memo: dict[
        tuple[tuple[int, ...], int],
        tuple[tuple[tuple[float, tuple[int, ...]], ...], tuple[float, ...]],
    ] = {}
    winner_memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]] = {}
    repair_rows = [
        _repair_named_probability_matching_row(
            row,
            library_actions,
            value_layers,
            support_tolerance,
            active_tolerance,
            canonical_policy_memo,
            named_policy_memo,
            winner_memo,
        )
        for row in selected_rows
    ]
    return all_rows, repair_rows


def print_probability_matching_dual_face_repair(
    k: int,
    T: int,
    library_name: str,
    packet_type_filter: tuple[int, ...] | None = None,
    packet_gaps_filter: tuple[int, ...] | None = None,
    support_tolerance: float = 1e-8,
    active_tolerance: float = 1e-8,
    n: int = 20,
    support_n: int = 12,
) -> None:
    library_actions = _named_experiment_action_library(k, library_name, T)
    value_layers = _library_lp_value_layers(k, T, library_actions)
    all_rows, repair_rows = probability_matching_dual_face_repair_rows(
        k,
        T,
        library_name,
        packet_type_filter=packet_type_filter,
        packet_gaps_filter=packet_gaps_filter,
        support_tolerance=support_tolerance,
        active_tolerance=active_tolerance,
        n=n,
    )
    zero = tuple(0 for _ in range(k))
    optimal_value = optimal_values(k, T)[T][zero]
    restricted_value = value_layers[T][zero]
    total_original_l1 = sum(row.weighted_l1_error for row in all_rows)
    total_original_linf = sum(row.weighted_linf_error for row in all_rows)
    selected_original_l1 = sum(row.original_weighted_l1_error for row in repair_rows)
    selected_original_linf = sum(row.original_weighted_linf_error for row in repair_rows)
    selected_repaired_l1 = sum(row.repaired_weighted_l1_error for row in repair_rows)
    selected_repaired_linf = sum(row.repaired_weighted_linf_error for row in repair_rows)

    print(f"Probability matching dual-face repair, k={k}, T={T}")
    print()
    print(f"library: {library_name}")
    print(f"library size: {len(library_actions)}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"active tolerance: {active_tolerance:.3g}")
    print(f"packet type filter: {packet_type_filter}")
    print(f"packet gaps filter: {packet_gaps_filter}")
    print(f"V_star: {optimal_value:.6f}")
    print(f"V_restricted: {restricted_value:.6f}")
    print(f"gap: {optimal_value - restricted_value:.6f}")
    print(f"total rows inspected: {len(repair_rows)}")
    print(f"full original weighted L1 error: {total_original_l1:.6f}")
    print(f"full original weighted Linf error: {total_original_linf:.6f}")
    print(f"selected original weighted L1 error: {selected_original_l1:.6f}")
    print(f"selected original weighted Linf error: {selected_original_linf:.6f}")
    print(f"selected repaired weighted L1 error: {selected_repaired_l1:.6f}")
    print(f"selected repaired weighted Linf error: {selected_repaired_linf:.6f}")
    print()

    print("Top repaired rows:")
    for repair in sorted(
        repair_rows,
        key=lambda item: (-item.original_weighted_linf_error, item.original_row.time, item.original_row.named_state),
    ):
        row = repair.original_row
        original_diff = _format_named_residual_difference(row)
        repaired_diff = _format_float_tuple(
            tuple(
                row.lifted_learner_distribution[index] - repair.repaired_winner_probabilities[index]
                for index in range(k)
            )
        )
        print(
            f"t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" named={row.named_state} canon={row.canonical_state}"
            f" ptype={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
            f" active={repair.active_candidate_count}"
            f" success={repair.success}"
        )
        print(f"  original winner probs={_format_float_tuple(row.winner_probabilities)}")
        print(f"  original residual l1={row.l1_error:.6f} linf={row.linf_error:.6f} diff={original_diff}")
        print(f"  repaired winner probs={_format_float_tuple(repair.repaired_winner_probabilities)}")
        print(
            f"  repaired residual l1={repair.repaired_l1_error:.6f}"
            f" linf={repair.repaired_linf_error:.6f}"
            f" diff={repaired_diff}"
        )
        print(f"  repaired message={repair.message}")
        print("  repaired support:")
        for weight, action, signature, orbit_key in repair.support[:support_n]:
            print(
                f"    {_format_action(action)}"
                f" edge={_format_edge_signature(signature)}"
                f" orbit={orbit_key}"
                f" weight={weight:.6f}"
            )
        if len(repair.support) > support_n:
            print(f"    ... {len(repair.support) - support_n} more")
        print()


def _length_parity_category(lengths: tuple[int, ...]) -> str:
    has_odd = any(length % 2 == 1 for length in lengths)
    has_even = any(length % 2 == 0 for length in lengths)
    if has_odd and has_even:
        return "mixed"
    if has_even:
        return "even-only"
    return "odd-only"


def _is_contiguous_lengths(lengths: tuple[int, ...]) -> bool:
    if not lengths:
        return False
    return lengths == tuple(range(min(lengths), max(lengths) + 1))


def _policy_lengths_for_rows(
    k: int,
    rows: list[WeightedTopPrefixOracleLabelRow],
    occupancy_policy_name: str | None,
) -> dict[WeightedTopPrefixOracleLabelRow, int] | None:
    if occupancy_policy_name is None or occupancy_policy_name == "oracle":
        return None
    policies = _policy_registry(k)
    policy = policies.get(occupancy_policy_name)
    if policy is None:
        return None
    try:
        return {
            row: _top_prefix_length_from_policy(policy(row.state))
            for row in rows
        }
    except ValueError:
        return None


def print_top_prefix_valid_length_structure(
    k: int,
    T: int,
    selector: str,
    occupancy_policy_name: str | None = "oracle",
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = weighted_top_prefix_oracle_labels(
        k,
        T,
        selector,
        occupancy_policy_name=occupancy_policy_name,
        tolerance=tolerance,
    )
    policy_lengths = _policy_lengths_for_rows(k, rows, occupancy_policy_name)

    print(f"Top-prefix valid length structure, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"occupancy policy: {occupancy_policy_name or 'oracle'}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print()

    print("Selected L histogram by value:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.selected_length, lambda row: row.value_weight, n):
        print(f"  {length}: {weight:.6f}")
    print()

    print("Valid parity histogram by value:")
    for category, weight in _top_weighted_histogram(rows, lambda row: _length_parity_category(row.valid_lengths), lambda row: row.value_weight, n):
        print(f"  {category}: {weight:.6f}")
    print()

    print("Selected parity histogram by value:")
    for category, weight in _top_weighted_histogram(rows, lambda row: "even" if row.selected_length % 2 == 0 else "odd", lambda row: row.value_weight, n):
        print(f"  {category}: {weight:.6f}")
    print()

    print("Contiguous valid-length sets by value:")
    for contiguous, weight in _top_weighted_histogram(rows, lambda row: _is_contiguous_lengths(row.valid_lengths), lambda row: row.value_weight, n):
        print(f"  {contiguous}: {weight:.6f}")
    print()

    probes = tuple(dict.fromkeys((1, 3, 5, k - 3, k - 2, k - 1)))
    print("Probe membership by value:")
    for probe in probes:
        if 1 <= probe <= k - 1:
            contains_weight = sum(row.value_weight for row in rows if probe in row.valid_lengths)
            selected_weight = sum(row.value_weight for row in rows if row.selected_length == probe)
            print(f"  L={probe}: contains={contains_weight:.6f} selected={selected_weight:.6f}")
    print()

    print("Max valid L distance from k-1 by value:")
    for distance, weight in _top_weighted_histogram(rows, lambda row: (k - 1) - max(row.valid_lengths), lambda row: row.value_weight, n):
        print(f"  distance {distance}: {weight:.6f}")
    print()

    print("Top regimes where selected L is even:")
    even_selected = [row for row in rows if row.selected_length % 2 == 0]
    _print_valid_length_structure_regimes(even_selected, n)
    print()

    print("Top regimes with even-only valid lengths:")
    even_only = [row for row in rows if _length_parity_category(row.valid_lengths) == "even-only"]
    _print_valid_length_structure_regimes(even_only, n)
    print()

    print("Top regimes where valid lengths include even but selected L is odd:")
    mixed_even_unselected = [
        row
        for row in rows
        if any(length % 2 == 0 for length in row.valid_lengths)
        and row.selected_length % 2 == 1
    ]
    _print_valid_length_structure_regimes(mixed_even_unselected, n)
    print()

    if policy_lengths is not None:
        print("Raw top rows where valid lengths contain even but policy L is invalid:")
        rows_with_policy_invalid_even = [
            row
            for row in rows
            if any(length % 2 == 0 for length in row.valid_lengths)
            and policy_lengths[row] not in row.valid_lengths
        ]
        for row in sorted(rows_with_policy_invalid_even, key=lambda item: item.value_weight, reverse=True)[:n]:
            marker = " fallback" if row.used_fallback else ""
            print(
                f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
                f" state={row.state} ptype={row.packet_type}"
                f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
                f" prob={row.occupancy_probability:.6f}"
                f" value_weight={row.value_weight:.6f}"
                f" policy_L={policy_lengths[row]}"
                f" selected={row.selected_length}"
                f" valid={row.valid_lengths}{marker}"
            )
        print()

    print("Raw top rows by value:")
    for row in sorted(rows, key=lambda item: item.value_weight, reverse=True)[:n]:
        marker = " fallback" if row.used_fallback else ""
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
            f" prob={row.occupancy_probability:.6f}"
            f" value_weight={row.value_weight:.6f}"
            f" valid={row.valid_lengths} selected={row.selected_length}{marker}"
        )


def _print_valid_length_structure_regimes(
    rows: list[WeightedTopPrefixOracleLabelRow],
    n: int,
) -> None:
    regime_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[WeightedTopPrefixOracleLabelRow]] = defaultdict(list)
    for row in rows:
        regime_rows[(row.packet_type, row.packet_gaps)].append(row)
    for (ptype, gaps), items in sorted(
        regime_rows.items(),
        key=lambda pair: (-sum(row.value_weight for row in pair[1]), pair[0]),
    )[:n]:
        selected_hist = _top_weighted_histogram(
            items,
            lambda row: row.selected_length,
            lambda row: row.value_weight,
            5,
        )
        valid_hist = _top_weighted_histogram(
            items,
            lambda row: row.valid_lengths,
            lambda row: row.value_weight,
            5,
        )
        adjacent_examples = tuple(
            key
            for key, _ in _top_weighted_histogram(
                items,
                lambda row: row.adjacent_gaps,
                lambda row: row.value_weight,
                3,
            )
        )
        rendered_selected = ", ".join(f"{length}:{weight:.6f}" for length, weight in selected_hist)
        rendered_valid = ", ".join(f"{lengths}:{weight:.6f}" for lengths, weight in valid_hist)
        print(
            f"  packet type={ptype} gaps={gaps}"
            f" occupancy={sum(row.occupancy_probability for row in items):.6f}"
            f" value={sum(row.value_weight for row in items):.6f}"
            f" adjacent_examples={adjacent_examples}"
            f" selected=[{rendered_selected}]"
            f" valid=[{rendered_valid}]"
        )


def print_top_prefix_scale_rows(
    k: int,
    T: int,
    selector: str,
    occupancy_policy_name: str | None = "oracle",
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = weighted_top_prefix_oracle_labels(
        k,
        T,
        selector,
        occupancy_policy_name=occupancy_policy_name,
        tolerance=tolerance,
    )
    near_global_length = max(1, k - 3)
    global_length = k - 1

    print(f"Top-prefix scale rows, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"occupancy policy: {occupancy_policy_name or 'oracle'}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"near-global L: {near_global_length}")
    print(f"global L: {global_length}")
    print(f"rows: {len(rows)}")
    print()

    filters = (
        ("even-only valid lengths", lambda row: _length_parity_category(row.valid_lengths) == "even-only"),
        (
            f"max(valid_lengths) >= k-3 ({near_global_length})",
            lambda row: max(row.valid_lengths) >= near_global_length,
        ),
        (f"k-3 ({near_global_length}) in valid_lengths", lambda row: near_global_length in row.valid_lengths),
        (f"k-1 ({global_length}) in valid_lengths", lambda row: global_length in row.valid_lengths),
    )

    for title, predicate in filters:
        filtered = [row for row in rows if predicate(row)]
        print(title)
        print(f"  occupancy: {sum(row.occupancy_probability for row in filtered):.6f}")
        print(f"  value: {sum(row.value_weight for row in filtered):.6f}")
        print("  top regimes:")
        _print_valid_length_structure_regimes(filtered, n)
        print("  raw rows:")
        for row in sorted(filtered, key=lambda item: item.value_weight, reverse=True)[:n]:
            marker = " fallback" if row.used_fallback else ""
            print(
                f"    t={row.time:2d} rem={row.remaining_horizon:2d}"
                f" state={row.state} ptype={row.packet_type}"
                f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
                f" prob={row.occupancy_probability:.6f}"
                f" value_weight={row.value_weight:.6f}"
                f" valid={row.valid_lengths} selected={row.selected_length}{marker}"
            )
        print()


def print_top_prefix_oracle_labels_weighted(
    k: int,
    T: int,
    selector: str,
    occupancy_policy_name: str | None = "oracle",
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = weighted_top_prefix_oracle_labels(
        k,
        T,
        selector,
        occupancy_policy_name=occupancy_policy_name,
        tolerance=tolerance,
    )

    total_occupancy = sum(row.occupancy_probability for row in rows)
    total_value_weight = sum(row.value_weight for row in rows)
    fallback_occupancy = sum(row.occupancy_probability for row in rows if row.used_fallback)
    fallback_value_weight = sum(row.value_weight for row in rows if row.used_fallback)

    print(f"Weighted top-prefix oracle labels, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"occupancy policy: {occupancy_policy_name or 'oracle'}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print(f"total occupancy: {total_occupancy:.6f}")
    print(f"total value weight: {total_value_weight:.6f}")
    print(f"fallback occupancy mass: {fallback_occupancy:.6f}")
    print(f"fallback value weight: {fallback_value_weight:.6f}")
    print()
    print("Selected L by occupancy:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.selected_length, lambda row: row.occupancy_probability, n):
        print(f"  {length}: {weight:.6f}")
    print()
    print("Selected L by value weight:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.selected_length, lambda row: row.value_weight, n):
        print(f"  {length}: {weight:.6f}")
    print()
    print("Valid L sets by occupancy:")
    for lengths, weight in _top_weighted_histogram(rows, lambda row: row.valid_lengths, lambda row: row.occupancy_probability, n):
        print(f"  {lengths}: {weight:.6f}")
    print()
    print("Valid L sets by value weight:")
    for lengths, weight in _top_weighted_histogram(rows, lambda row: row.valid_lengths, lambda row: row.value_weight, n):
        print(f"  {lengths}: {weight:.6f}")
    print()

    regime_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[WeightedTopPrefixOracleLabelRow]] = defaultdict(list)
    for row in rows:
        regime_rows[(row.packet_type, row.packet_gaps)].append(row)

    def regime_value_weight(items: list[WeightedTopPrefixOracleLabelRow]) -> float:
        return sum(item.value_weight for item in items)

    print("Top regimes by value weight:")
    for (ptype, gaps), items in sorted(regime_rows.items(), key=lambda pair: (-regime_value_weight(pair[1]), pair[0]))[:n]:
        occupancy_mass = sum(item.occupancy_probability for item in items)
        value_weight = regime_value_weight(items)
        selected_hist = _top_weighted_histogram(
            items,
            lambda row: row.selected_length,
            lambda row: row.value_weight,
            5,
        )
        valid_hist = _top_weighted_histogram(
            items,
            lambda row: row.valid_lengths,
            lambda row: row.value_weight,
            5,
        )
        rendered_selected = ", ".join(f"{length}:{weight:.6f}" for length, weight in selected_hist)
        rendered_valid = ", ".join(f"{lengths}:{weight:.6f}" for lengths, weight in valid_hist)
        print(
            f"  packet type={ptype} gaps={gaps}"
            f" occupancy={occupancy_mass:.6f}"
            f" value={value_weight:.6f}"
            f" selected=[{rendered_selected}]"
            f" valid=[{rendered_valid}]"
        )
    print()
    print("Raw top rows by value weight:")
    for row in sorted(rows, key=lambda item: item.value_weight, reverse=True)[:n]:
        marker = " fallback" if row.used_fallback else ""
        print(
            f"  t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} ptype={row.packet_type}"
            f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
            f" prob={row.occupancy_probability:.6f}"
            f" value_weight={row.value_weight:.6f}"
            f" valid={row.valid_lengths} selected={row.selected_length}{marker}"
        )


def print_top_prefix_length_regimes(
    k: int,
    T: int,
    selector: str,
    occupancy_policy_name: str | None = "oracle",
    tolerance: float = 1e-9,
    n: int = 20,
) -> None:
    rows = weighted_top_prefix_oracle_labels(
        k,
        T,
        selector,
        occupancy_policy_name=occupancy_policy_name,
        tolerance=tolerance,
    )

    print(f"Top-prefix selected-L regimes, k={k}, T={T}")
    print()
    print(f"selector: {selector}")
    print(f"occupancy policy: {occupancy_policy_name or 'oracle'}")
    print(f"tolerance: {tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print()
    print("Selected L summary by value weight:")
    for length, weight in _top_weighted_histogram(rows, lambda row: row.selected_length, lambda row: row.value_weight, n):
        occupancy_mass = sum(row.occupancy_probability for row in rows if row.selected_length == length)
        print(f"  L={length}: occupancy={occupancy_mass:.6f} value={weight:.6f}")
    print()
    print("Valid L set -> selected L by value weight:")
    valid_groups: dict[tuple[int, ...], list[WeightedTopPrefixOracleLabelRow]] = defaultdict(list)
    for row in rows:
        valid_groups[row.valid_lengths].append(row)
    for lengths, items in sorted(valid_groups.items(), key=lambda pair: (-sum(row.value_weight for row in pair[1]), pair[0]))[:n]:
        selected_hist = _top_weighted_histogram(
            items,
            lambda row: row.selected_length,
            lambda row: row.value_weight,
            n,
        )
        rendered_selected = ", ".join(f"{length}:{weight:.6f}" for length, weight in selected_hist)
        print(f"  {lengths}: [{rendered_selected}]")

    selected_groups: dict[int, list[WeightedTopPrefixOracleLabelRow]] = defaultdict(list)
    for row in rows:
        selected_groups[row.selected_length].append(row)

    for length, items in sorted(selected_groups.items()):
        occupancy_mass = sum(row.occupancy_probability for row in items)
        value_weight = sum(row.value_weight for row in items)
        print()
        print(f"Selected L={length}")
        print(f"  occupancy mass: {occupancy_mass:.6f}")
        print(f"  value weight: {value_weight:.6f}")

        regime_rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[WeightedTopPrefixOracleLabelRow]] = defaultdict(list)
        for row in items:
            regime_rows[(row.packet_type, row.packet_gaps)].append(row)

        print("  top regimes by value:")
        for (ptype, gaps), regime_items in sorted(
            regime_rows.items(),
            key=lambda pair: (-sum(row.value_weight for row in pair[1]), pair[0]),
        )[:n]:
            regime_occupancy = sum(row.occupancy_probability for row in regime_items)
            regime_value = sum(row.value_weight for row in regime_items)
            valid_hist = _top_weighted_histogram(
                regime_items,
                lambda row: row.valid_lengths,
                lambda row: row.value_weight,
                5,
            )
            adjacent_examples = tuple(
                key
                for key, _ in _top_weighted_histogram(
                    regime_items,
                    lambda row: row.adjacent_gaps,
                    lambda row: row.value_weight,
                    3,
                )
            )
            rendered_valid = ", ".join(f"{lengths}:{weight:.6f}" for lengths, weight in valid_hist)
            print(
                f"    packet type={ptype} gaps={gaps}"
                f" occupancy={regime_occupancy:.6f}"
                f" value={regime_value:.6f}"
                f" adjacent_examples={adjacent_examples}"
                f" valid=[{rendered_valid}]"
            )

        print("  raw top rows:")
        for row in sorted(items, key=lambda item: item.value_weight, reverse=True)[:n]:
            marker = " fallback" if row.used_fallback else ""
            print(
                f"    t={row.time:2d} rem={row.remaining_horizon:2d}"
                f" state={row.state} ptype={row.packet_type}"
                f" packet_gaps={row.packet_gaps} adjacent_gaps={row.adjacent_gaps}"
                f" prob={row.occupancy_probability:.6f}"
                f" value_weight={row.value_weight:.6f}"
                f" valid={row.valid_lengths}{marker}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finite-horizon expert game experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--k", type=int, required=True)
    compare.add_argument("--T", type=int, required=True)

    greedy = subparsers.add_parser("greedy")
    greedy.add_argument("--k", type=int, required=True)
    greedy.add_argument("--T", type=int, required=True)
    greedy.add_argument("--policy", required=True)
    greedy.add_argument("-n", type=int, default=20)

    commute = subparsers.add_parser("commute")
    commute.add_argument("--k", type=int, required=True)
    commute.add_argument("--T", type=int, required=True)
    commute.add_argument("--policy", required=True)
    commute.add_argument("-n", type=int, default=20)

    commute_mixed = subparsers.add_parser("commute-mixed")
    commute_mixed.add_argument("--k", type=int, required=True)
    commute_mixed.add_argument("--T", type=int, required=True)
    commute_mixed.add_argument("--policy", required=True)
    commute_mixed.add_argument("-n", type=int, default=20)

    weighted_greedy = subparsers.add_parser("weighted-greedy")
    weighted_greedy.add_argument("--k", type=int, required=True)
    weighted_greedy.add_argument("--T", type=int, required=True)
    weighted_greedy.add_argument("--policy", required=True)
    weighted_greedy.add_argument("-n", type=int, default=20)

    weighted_greedy_by_packet = subparsers.add_parser("weighted-greedy-by-packet")
    weighted_greedy_by_packet.add_argument("--k", type=int, required=True)
    weighted_greedy_by_packet.add_argument("--T", type=int, required=True)
    weighted_greedy_by_packet.add_argument("--policy", required=True)
    weighted_greedy_by_packet.add_argument("-n", type=int, default=10)

    weighted_greedy_by_regime = subparsers.add_parser("weighted-greedy-by-regime")
    weighted_greedy_by_regime.add_argument("--k", type=int, required=True)
    weighted_greedy_by_regime.add_argument("--T", type=int, required=True)
    weighted_greedy_by_regime.add_argument("--policy", required=True)
    weighted_greedy_by_regime.add_argument("-n", type=int, default=10)

    weighted_greedy_filter = subparsers.add_parser("weighted-greedy-filter")
    weighted_greedy_filter.add_argument("--k", type=int, required=True)
    weighted_greedy_filter.add_argument("--T", type=int, required=True)
    weighted_greedy_filter.add_argument("--policy", required=True)
    weighted_greedy_filter.add_argument("--packet-type", required=True)
    weighted_greedy_filter.add_argument("--packet-gaps", required=True)
    weighted_greedy_filter.add_argument("-n", type=int, default=20)

    best_fixed = subparsers.add_parser("best-fixed")
    best_fixed.add_argument("--k", type=int, required=True)
    best_fixed.add_argument("--T", type=int, required=True)
    best_fixed.add_argument("-n", type=int, default=20)

    library_oracle_parser = subparsers.add_parser("library-oracle")
    library_oracle_parser.add_argument("--k", type=int, required=True)
    library_oracle_parser.add_argument("--T", type=int, required=True)
    library_oracle_parser.add_argument("--library", default="local_edges")
    library_oracle_parser.add_argument("--occupancy-policy", default="comb")
    library_oracle_parser.add_argument("--top-fixed-size", type=int, default=20)
    library_oracle_parser.add_argument("-n", type=int, default=20)

    one_run_analysis = subparsers.add_parser("one-run-oracle-analysis")
    one_run_analysis.add_argument("--k", type=int, required=True)
    one_run_analysis.add_argument("--T", type=int, required=True)
    one_run_analysis.add_argument("--occupancy-policy", default="comb")
    one_run_analysis.add_argument("-n", type=int, default=20)

    one_run_tie_analysis_parser = subparsers.add_parser("one-run-tie-analysis")
    one_run_tie_analysis_parser.add_argument("--k", type=int, required=True)
    one_run_tie_analysis_parser.add_argument("--T", type=int, required=True)
    one_run_tie_analysis_parser.add_argument("--occupancy-policy", default="comb")
    one_run_tie_analysis_parser.add_argument("--tolerance", type=float, default=1e-9)
    one_run_tie_analysis_parser.add_argument("-n", type=int, default=20)

    top_prefix_tie_analysis_parser = subparsers.add_parser("top-prefix-tie-analysis")
    top_prefix_tie_analysis_parser.add_argument("--k", type=int, required=True)
    top_prefix_tie_analysis_parser.add_argument("--T", type=int, required=True)
    top_prefix_tie_analysis_parser.add_argument("--occupancy-policy", default="comb")
    top_prefix_tie_analysis_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_tie_analysis_parser.add_argument("-n", type=int, default=20)

    top_prefix_oracle_eval_parser = subparsers.add_parser("top-prefix-oracle-eval")
    top_prefix_oracle_eval_parser.add_argument("--k", type=int, required=True)
    top_prefix_oracle_eval_parser.add_argument("--T", type=int, required=True)
    top_prefix_oracle_eval_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_oracle_eval_parser.add_argument("--tolerance", type=float, default=1e-9)

    top_prefix_oracle_labels_parser = subparsers.add_parser("top-prefix-oracle-labels")
    top_prefix_oracle_labels_parser.add_argument("--k", type=int, required=True)
    top_prefix_oracle_labels_parser.add_argument("--T", type=int, required=True)
    top_prefix_oracle_labels_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_oracle_labels_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_oracle_labels_parser.add_argument("-n", type=int, default=20)

    top_prefix_oracle_labels_weighted_parser = subparsers.add_parser("top-prefix-oracle-labels-weighted")
    top_prefix_oracle_labels_weighted_parser.add_argument("--k", type=int, required=True)
    top_prefix_oracle_labels_weighted_parser.add_argument("--T", type=int, required=True)
    top_prefix_oracle_labels_weighted_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_oracle_labels_weighted_parser.add_argument("--occupancy-policy", default="oracle")
    top_prefix_oracle_labels_weighted_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_oracle_labels_weighted_parser.add_argument("-n", type=int, default=20)

    top_prefix_length_regimes_parser = subparsers.add_parser("top-prefix-length-regimes")
    top_prefix_length_regimes_parser.add_argument("--k", type=int, required=True)
    top_prefix_length_regimes_parser.add_argument("--T", type=int, required=True)
    top_prefix_length_regimes_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_length_regimes_parser.add_argument("--occupancy-policy", default="oracle")
    top_prefix_length_regimes_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_length_regimes_parser.add_argument("-n", type=int, default=20)

    top_prefix_policy_vs_oracle_labels_parser = subparsers.add_parser("top-prefix-policy-vs-oracle-labels")
    top_prefix_policy_vs_oracle_labels_parser.add_argument("--k", type=int, required=True)
    top_prefix_policy_vs_oracle_labels_parser.add_argument("--T", type=int, required=True)
    top_prefix_policy_vs_oracle_labels_parser.add_argument("--policy", required=True)
    top_prefix_policy_vs_oracle_labels_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        default="min_valid",
    )
    top_prefix_policy_vs_oracle_labels_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_policy_vs_oracle_labels_parser.add_argument("-n", type=int, default=20)

    top_prefix_valid_length_structure_parser = subparsers.add_parser("top-prefix-valid-length-structure")
    top_prefix_valid_length_structure_parser.add_argument("--k", type=int, required=True)
    top_prefix_valid_length_structure_parser.add_argument("--T", type=int, required=True)
    top_prefix_valid_length_structure_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_valid_length_structure_parser.add_argument("--occupancy-policy", default="oracle")
    top_prefix_valid_length_structure_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_valid_length_structure_parser.add_argument("-n", type=int, default=20)

    top_prefix_scale_rows_parser = subparsers.add_parser("top-prefix-scale-rows")
    top_prefix_scale_rows_parser.add_argument("--k", type=int, required=True)
    top_prefix_scale_rows_parser.add_argument("--T", type=int, required=True)
    top_prefix_scale_rows_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        required=True,
    )
    top_prefix_scale_rows_parser.add_argument("--occupancy-policy", default="oracle")
    top_prefix_scale_rows_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_scale_rows_parser.add_argument("-n", type=int, default=20)

    policy_occupancy_diff_parser = subparsers.add_parser("policy-occupancy-diff")
    policy_occupancy_diff_parser.add_argument("--k", type=int, required=True)
    policy_occupancy_diff_parser.add_argument("--T", type=int, required=True)
    policy_occupancy_diff_parser.add_argument("--policy-a", required=True)
    policy_occupancy_diff_parser.add_argument("--policy-b", required=True)
    policy_occupancy_diff_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        default="min_valid",
    )
    policy_occupancy_diff_parser.add_argument("--tolerance", type=float, default=1e-9)
    policy_occupancy_diff_parser.add_argument("-n", type=int, default=20)

    top_prefix_next_state_debt_parser = subparsers.add_parser("top-prefix-next-state-debt")
    top_prefix_next_state_debt_parser.add_argument("--k", type=int, required=True)
    top_prefix_next_state_debt_parser.add_argument("--T", type=int, required=True)
    top_prefix_next_state_debt_parser.add_argument("--policy", required=True)
    top_prefix_next_state_debt_parser.add_argument(
        "--selector",
        choices=("min_valid", "max_valid", "median_valid", "chase_preferred"),
        default="min_valid",
    )
    top_prefix_next_state_debt_parser.add_argument("--tolerance", type=float, default=1e-9)
    top_prefix_next_state_debt_parser.add_argument("-n", type=int, default=20)

    top_prefix_candidate_value_parser = subparsers.add_parser("top-prefix-candidate-value")
    top_prefix_candidate_value_parser.add_argument("--k", type=int, required=True)
    top_prefix_candidate_value_parser.add_argument("--T", type=int, required=True)
    top_prefix_candidate_value_parser.add_argument("--policy", required=True)
    top_prefix_candidate_value_parser.add_argument("--length-set", choices=("scale", "all"), default="scale")
    top_prefix_candidate_value_parser.add_argument("-n", type=int, default=20)

    top_prefix_restricted_optimal_parser = subparsers.add_parser("top-prefix-restricted-optimal")
    top_prefix_restricted_optimal_parser.add_argument("--k", type=int, required=True)
    top_prefix_restricted_optimal_parser.add_argument("--T", type=int, required=True)
    top_prefix_restricted_optimal_parser.add_argument("--length-set", choices=("scale", "all"), default="scale")
    top_prefix_restricted_optimal_parser.add_argument("-n", type=int, default=20)

    one_run_restricted_optimal_parser = subparsers.add_parser("one-run-restricted-optimal")
    one_run_restricted_optimal_parser.add_argument("--k", type=int, required=True)
    one_run_restricted_optimal_parser.add_argument("--T", type=int, required=True)
    one_run_restricted_optimal_parser.add_argument("-n", type=int, default=20)

    two_run_restricted_optimal_parser = subparsers.add_parser("two-run-restricted-optimal")
    two_run_restricted_optimal_parser.add_argument("--k", type=int, required=True)
    two_run_restricted_optimal_parser.add_argument("--T", type=int, required=True)
    two_run_restricted_optimal_parser.add_argument("-n", type=int, default=20)

    library_lp_restricted_optimal_parser = subparsers.add_parser("library-lp-restricted-optimal")
    library_lp_restricted_optimal_parser.add_argument("--k", type=int, required=True)
    library_lp_restricted_optimal_parser.add_argument("--T", type=int, required=True)
    library_lp_restricted_optimal_parser.add_argument("--library", required=True)
    library_lp_restricted_optimal_parser.add_argument("-n", type=int, default=20)

    library_lp_dual_inspect_parser = subparsers.add_parser("library-lp-dual-inspect")
    library_lp_dual_inspect_parser.add_argument("--k", type=int, required=True)
    library_lp_dual_inspect_parser.add_argument("--T", type=int, required=True)
    library_lp_dual_inspect_parser.add_argument("--library", required=True)
    library_lp_dual_inspect_parser.add_argument("--packet-type")
    library_lp_dual_inspect_parser.add_argument("--packet-gaps")
    library_lp_dual_inspect_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    library_lp_dual_inspect_parser.add_argument("--support-n", type=int, default=12)
    library_lp_dual_inspect_parser.add_argument("-n", type=int, default=20)

    library_lp_dual_orbits_parser = subparsers.add_parser("library-lp-dual-orbits")
    library_lp_dual_orbits_parser.add_argument("--k", type=int, required=True)
    library_lp_dual_orbits_parser.add_argument("--T", type=int, required=True)
    library_lp_dual_orbits_parser.add_argument("--library", required=True)
    library_lp_dual_orbits_parser.add_argument("--packet-type")
    library_lp_dual_orbits_parser.add_argument("--packet-gaps")
    library_lp_dual_orbits_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    library_lp_dual_orbits_parser.add_argument("--orbit-n", type=int, default=12)
    library_lp_dual_orbits_parser.add_argument("-n", type=int, default=10)

    library_lp_dual_orbit_completion_parser = subparsers.add_parser("library-lp-dual-orbit-completion")
    library_lp_dual_orbit_completion_parser.add_argument("--k", type=int, required=True)
    library_lp_dual_orbit_completion_parser.add_argument("--T", type=int, required=True)
    library_lp_dual_orbit_completion_parser.add_argument("--library", required=True)
    library_lp_dual_orbit_completion_parser.add_argument("--packet-type")
    library_lp_dual_orbit_completion_parser.add_argument("--packet-gaps")
    library_lp_dual_orbit_completion_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    library_lp_dual_orbit_completion_parser.add_argument("--orbit-n", type=int, default=20)
    library_lp_dual_orbit_completion_parser.add_argument("-n", type=int, default=5)

    k3_motif_sweep_parser = subparsers.add_parser("k3-motif-sweep")
    k3_motif_sweep_parser.add_argument(
        "--T-values",
        default="10,20,50,100",
        help="comma-separated horizons, e.g. 10,20,50,100",
    )

    k9_motif_library_sweep_parser = subparsers.add_parser("k9-motif-library-sweep")
    k9_motif_library_sweep_parser.add_argument(
        "--T-values",
        default="7",
        help="comma-separated horizons, e.g. 7",
    )
    k9_motif_library_sweep_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    k9_motif_library_sweep_parser.add_argument("-n", type=int, default=10)

    probability_matching_inspect_parser = subparsers.add_parser("probability-matching-inspect")
    probability_matching_inspect_parser.add_argument("--k", type=int, required=True)
    probability_matching_inspect_parser.add_argument("--T", type=int, required=True)
    probability_matching_inspect_parser.add_argument("--library", required=True)
    probability_matching_inspect_parser.add_argument("--packet-type")
    probability_matching_inspect_parser.add_argument("--packet-gaps")
    probability_matching_inspect_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    probability_matching_inspect_parser.add_argument("-n", type=int, default=20)

    probability_matching_named_inspect_parser = subparsers.add_parser("probability-matching-named-inspect")
    probability_matching_named_inspect_parser.add_argument("--k", type=int, required=True)
    probability_matching_named_inspect_parser.add_argument("--T", type=int, required=True)
    probability_matching_named_inspect_parser.add_argument("--library", required=True)
    probability_matching_named_inspect_parser.add_argument("--packet-type")
    probability_matching_named_inspect_parser.add_argument("--packet-gaps")
    probability_matching_named_inspect_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    probability_matching_named_inspect_parser.add_argument("-n", type=int, default=20)

    probability_matching_named_residuals_parser = subparsers.add_parser("probability-matching-named-residuals")
    probability_matching_named_residuals_parser.add_argument("--k", type=int, required=True)
    probability_matching_named_residuals_parser.add_argument("--T", type=int, required=True)
    probability_matching_named_residuals_parser.add_argument("--library", required=True)
    probability_matching_named_residuals_parser.add_argument("--packet-type")
    probability_matching_named_residuals_parser.add_argument("--packet-gaps")
    probability_matching_named_residuals_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    probability_matching_named_residuals_parser.add_argument("--orbit-n", type=int, default=8)
    probability_matching_named_residuals_parser.add_argument("-n", type=int, default=30)

    probability_matching_dual_face_repair_parser = subparsers.add_parser("probability-matching-dual-face-repair")
    probability_matching_dual_face_repair_parser.add_argument("--k", type=int, required=True)
    probability_matching_dual_face_repair_parser.add_argument("--T", type=int, required=True)
    probability_matching_dual_face_repair_parser.add_argument("--library", required=True)
    probability_matching_dual_face_repair_parser.add_argument("--packet-type")
    probability_matching_dual_face_repair_parser.add_argument("--packet-gaps")
    probability_matching_dual_face_repair_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    probability_matching_dual_face_repair_parser.add_argument("--active-tolerance", type=float, default=1e-8)
    probability_matching_dual_face_repair_parser.add_argument("--support-n", type=int, default=12)
    probability_matching_dual_face_repair_parser.add_argument("-n", type=int, default=20)

    strategy_class_benchmark_parser = subparsers.add_parser("strategy-class-benchmark")
    strategy_class_benchmark_parser.add_argument(
        "--cases",
        default="3:20,3:50,5:20,6:12,7:10,8:8,9:7",
        help="comma-separated k:T cases, e.g. 3:20,9:7",
    )
    strategy_class_benchmark_parser.add_argument(
        "--libraries",
        default="top_prefix_all,one_run,two_run,local_edges,all",
        help="comma-separated action-library names",
    )
    strategy_class_benchmark_parser.add_argument("--include-probability-matching", action="store_true")
    strategy_class_benchmark_parser.add_argument("--probability-matching-max-k", type=int, default=9)
    strategy_class_benchmark_parser.add_argument("--probability-matching-max-T", type=int, default=7)
    strategy_class_benchmark_parser.add_argument("--active-tolerance", type=float, default=1e-8)
    strategy_class_benchmark_parser.add_argument(
        "--no-active-counts",
        action="store_true",
        help="skip active action/signature counting for faster long-horizon value sweeps",
    )
    strategy_class_benchmark_parser.add_argument("-n", type=int, default=20)

    strategy_class_relative_benchmark_parser = subparsers.add_parser("strategy-class-relative-benchmark")
    strategy_class_relative_benchmark_parser.add_argument(
        "--cases",
        default="5:30,5:40,5:50",
        help="comma-separated k:T cases, e.g. 5:30,6:16",
    )
    strategy_class_relative_benchmark_parser.add_argument(
        "--libraries",
        default="top_prefix_all,one_run,two_run,local_edges",
        help="comma-separated action-library names",
    )
    strategy_class_relative_benchmark_parser.add_argument("--reference-library", default="two_run")
    strategy_class_relative_benchmark_parser.add_argument("--active-tolerance", type=float, default=1e-8)
    strategy_class_relative_benchmark_parser.add_argument(
        "--no-active-counts",
        action="store_true",
        default=True,
        help="skip active action/signature counting for faster long-horizon value sweeps",
    )
    strategy_class_relative_benchmark_parser.add_argument(
        "--active-counts",
        dest="no_active_counts",
        action="store_false",
        help="collect active action/signature counts",
    )
    strategy_class_relative_benchmark_parser.add_argument("-n", type=int, default=20)

    explicit_policy_benchmark_parser = subparsers.add_parser("explicit-policy-benchmark")
    explicit_policy_benchmark_parser.add_argument(
        "--cases",
        default="9:7",
        help="comma-separated k:T cases, e.g. 9:7,5:20",
    )
    explicit_policy_benchmark_parser.add_argument(
        "--policies",
        default="two_run_orbit_mixture_v1,top_prefix_three_regime_v6,top_prefix_three_regime_v7",
        help="comma-separated policy names",
    )
    explicit_policy_benchmark_parser.add_argument("--reference-library", default="two_run")
    explicit_policy_benchmark_parser.add_argument("--include-reference", action="store_true")

    explicit_time_policy_benchmark_parser = subparsers.add_parser("explicit-time-policy-benchmark")
    explicit_time_policy_benchmark_parser.add_argument(
        "--cases",
        default="9:7",
        help="comma-separated k:T cases, e.g. 9:7",
    )
    explicit_time_policy_benchmark_parser.add_argument(
        "--policies",
        default="two_run_dual_support_replay_k9_T7",
        help="comma-separated time-dependent policy names",
    )
    explicit_time_policy_benchmark_parser.add_argument("--reference-library", default="two_run")
    explicit_time_policy_benchmark_parser.add_argument("--include-reference", action="store_true")

    two_run_replay_template_report_parser = subparsers.add_parser("two-run-replay-template-report")
    two_run_replay_template_report_parser.add_argument("--k", type=int, required=True)
    two_run_replay_template_report_parser.add_argument("--T", type=int, required=True)
    two_run_replay_template_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_replay_template_report_parser.add_argument("-n", type=int, default=40)

    two_run_replay_coarse_template_report_parser = subparsers.add_parser("two-run-replay-coarse-template-report")
    two_run_replay_coarse_template_report_parser.add_argument("--k", type=int, required=True)
    two_run_replay_coarse_template_report_parser.add_argument("--T", type=int, required=True)
    two_run_replay_coarse_template_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_replay_coarse_template_report_parser.add_argument("-n", type=int, default=80)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "compare":
        compare_values(args.k, args.T)
        return
    if args.command == "greedy":
        print_top_greedy_defects(args.k, args.T, args.policy, args.n)
        return
    if args.command == "commute":
        print_top_commutation_defects(args.k, args.T, args.policy, args.n)
        return
    if args.command == "commute-mixed":
        print_top_mixed_commutation_defects(args.k, args.T, args.policy, args.n)
        return
    if args.command == "weighted-greedy":
        print_occupation_weighted_greedy_defects(args.k, args.T, args.policy, args.n)
        return
    if args.command == "weighted-greedy-by-packet":
        print_weighted_greedy_by_packet(args.k, args.T, args.policy, args.n)
        return
    if args.command == "weighted-greedy-by-regime":
        print_weighted_greedy_by_regime(args.k, args.T, args.policy, args.n)
        return
    if args.command == "weighted-greedy-filter":
        print_weighted_greedy_filter(
            args.k,
            args.T,
            args.policy,
            _parse_int_tuple(args.packet_type),
            _parse_int_tuple(args.packet_gaps),
            args.n,
        )
        return
    if args.command == "best-fixed":
        print_best_fixed(args.k, args.T, args.n)
        return
    if args.command == "library-oracle":
        print_library_oracle(
            args.k,
            args.T,
            args.library,
            args.occupancy_policy,
            top_fixed_size=args.top_fixed_size,
            n=args.n,
        )
        return
    if args.command == "one-run-oracle-analysis":
        print_one_run_oracle_analysis(
            args.k,
            args.T,
            args.occupancy_policy,
            args.n,
        )
        return
    if args.command == "one-run-tie-analysis":
        print_one_run_tie_analysis(
            args.k,
            args.T,
            args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-tie-analysis":
        print_top_prefix_tie_analysis(
            args.k,
            args.T,
            args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-oracle-eval":
        print_top_prefix_oracle_eval(
            args.k,
            args.T,
            args.selector,
            tolerance=args.tolerance,
        )
        return
    if args.command == "top-prefix-oracle-labels":
        print_top_prefix_oracle_labels(
            args.k,
            args.T,
            args.selector,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-oracle-labels-weighted":
        print_top_prefix_oracle_labels_weighted(
            args.k,
            args.T,
            args.selector,
            occupancy_policy_name=args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-length-regimes":
        print_top_prefix_length_regimes(
            args.k,
            args.T,
            args.selector,
            occupancy_policy_name=args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-policy-vs-oracle-labels":
        print_top_prefix_policy_vs_oracle_labels(
            args.k,
            args.T,
            args.policy,
            args.selector,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-valid-length-structure":
        print_top_prefix_valid_length_structure(
            args.k,
            args.T,
            args.selector,
            occupancy_policy_name=args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-scale-rows":
        print_top_prefix_scale_rows(
            args.k,
            args.T,
            args.selector,
            occupancy_policy_name=args.occupancy_policy,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "policy-occupancy-diff":
        print_policy_occupancy_diff(
            args.k,
            args.T,
            args.policy_a,
            args.policy_b,
            args.selector,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-next-state-debt":
        print_top_prefix_next_state_debt(
            args.k,
            args.T,
            args.policy,
            args.selector,
            tolerance=args.tolerance,
            n=args.n,
        )
        return
    if args.command == "top-prefix-candidate-value":
        print_top_prefix_candidate_values(
            args.k,
            args.T,
            args.policy,
            length_set=args.length_set,
            n=args.n,
        )
        return
    if args.command == "top-prefix-restricted-optimal":
        print_top_prefix_restricted_optimal(
            args.k,
            args.T,
            args.length_set,
            n=args.n,
        )
        return
    if args.command == "one-run-restricted-optimal":
        print_one_run_restricted_optimal(
            args.k,
            args.T,
            n=args.n,
        )
        return
    if args.command == "two-run-restricted-optimal":
        print_edge_run_restricted_optimal(
            args.k,
            args.T,
            2,
            n=args.n,
        )
        return
    if args.command == "library-lp-restricted-optimal":
        print_library_lp_restricted_optimal(
            args.k,
            args.T,
            args.library,
            n=args.n,
        )
        return
    if args.command == "library-lp-dual-inspect":
        print_library_lp_dual_inspect(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            support_n=args.support_n,
            n=args.n,
        )
        return
    if args.command == "library-lp-dual-orbits":
        print_library_lp_dual_orbits(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            orbit_n=args.orbit_n,
            n=args.n,
        )
        return
    if args.command == "library-lp-dual-orbit-completion":
        print_library_lp_dual_orbit_completion(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            orbit_n=args.orbit_n,
            n=args.n,
        )
        return
    if args.command == "k3-motif-sweep":
        print_k3_motif_sweep(_parse_T_values(args.T_values))
        return
    if args.command == "k9-motif-library-sweep":
        print_k9_motif_library_sweep(
            _parse_T_values(args.T_values),
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "probability-matching-inspect":
        print_probability_matching_inspect(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "probability-matching-named-inspect":
        print_probability_matching_named_inspect(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "probability-matching-named-residuals":
        print_probability_matching_named_residuals(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            n=args.n,
            orbit_n=args.orbit_n,
        )
        return
    if args.command == "probability-matching-dual-face-repair":
        print_probability_matching_dual_face_repair(
            args.k,
            args.T,
            args.library,
            packet_type_filter=_parse_int_tuple(args.packet_type) if args.packet_type else None,
            packet_gaps_filter=_parse_int_tuple(args.packet_gaps) if args.packet_gaps else None,
            support_tolerance=args.support_tolerance,
            active_tolerance=args.active_tolerance,
            n=args.n,
            support_n=args.support_n,
        )
        return
    if args.command == "strategy-class-benchmark":
        print_strategy_class_benchmark(
            _parse_cases(args.cases),
            _parse_library_names(args.libraries),
            include_probability_matching=args.include_probability_matching,
            probability_matching_max_k=args.probability_matching_max_k,
            probability_matching_max_T=args.probability_matching_max_T,
            active_tolerance=args.active_tolerance,
            collect_active_counts=not args.no_active_counts,
            n=args.n,
        )
        return
    if args.command == "strategy-class-relative-benchmark":
        print_strategy_class_relative_benchmark(
            _parse_cases(args.cases),
            _parse_library_names(args.libraries),
            reference_library_name=args.reference_library,
            active_tolerance=args.active_tolerance,
            collect_active_counts=not args.no_active_counts,
            n=args.n,
        )
        return
    if args.command == "explicit-policy-benchmark":
        print_explicit_policy_benchmark(
            _parse_cases(args.cases),
            _parse_policy_names(args.policies),
            reference_library_name=args.reference_library,
            include_reference=args.include_reference,
        )
        return
    if args.command == "explicit-time-policy-benchmark":
        print_explicit_time_policy_benchmark(
            _parse_cases(args.cases),
            _parse_policy_names(args.policies),
            reference_library_name=args.reference_library,
            include_reference=args.include_reference,
        )
        return
    if args.command == "two-run-replay-template-report":
        print_two_run_replay_template_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "two-run-replay-coarse-template-report":
        print_two_run_replay_coarse_template_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
