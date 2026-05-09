from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .actions import all_actions, comb_action, complement, fixed_rank_action
from .defects import commutation_defect, commutation_defect_mixed, greedy_defect
from .dp_optimal import optimal_values
from .dp_policy import evaluate_balanced_policy, state_occupancy
from .lp_game import solve_minimax_step
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
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
