from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from itertools import combinations, product
from math import comb
from multiprocessing import get_context

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
from .state import all_states, canon, packet_type, packets


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
class TimePolicyBoundaryLossRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    reference_value: float
    candidate_value: float
    reference_step_value_on_reference_continuation: float
    candidate_step_value_on_reference_continuation: float
    action_loss_on_reference_continuation: float
    downstream_loss_after_candidate_action: float
    total_value_gap: float
    weighted_action_loss: float
    weighted_downstream_loss: float
    weighted_total_gap: float
    reference_support_count: int
    candidate_support_count: int


@dataclass(frozen=True)
class ReplayLayerMaskSweepRow:
    replay_horizons: frozenset[int]
    replay_count: int
    fallback_policy_name: str
    value: float
    reference_value: float
    gap_to_reference: float
    normalized_gap_to_reference: float
    exact: bool


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


@dataclass(frozen=True)
class SkeletonPlacementAggregate:
    family_name: str
    signature: object
    row_count: int
    total_occupancy: float
    horizons: tuple[tuple[int, int], ...]
    representative: TwoRunReplayTemplateRow
    packet_type_histogram: tuple[tuple[tuple[int, ...], int], ...]
    packet_gaps_histogram: tuple[tuple[tuple[int, ...], int], ...]
    adjacent_gaps_histogram: tuple[tuple[tuple[int, ...], int], ...]
    top_states_by_occupancy: tuple[tuple[tuple[int, ...], float], ...]
    label: str


@dataclass(frozen=True)
class RuleMiningActionFeature:
    weight: float
    action: tuple[int, ...]
    edge_signature: tuple[int, ...]
    one_positions: tuple[int, ...]
    ones_count: int
    touches_top: bool
    touches_bottom: bool
    touched_packet_indices: tuple[int, ...]
    orbit_key: tuple[int, ...]
    exposure_vector: tuple[str, ...]
    run_intervals: tuple[tuple[int, int], ...]
    run_boundary_distances: tuple[tuple[int, int], ...]
    run_crosses_boundary: tuple[bool, ...]
    run_span_labels: tuple[str, ...]
    successor_packet_type: tuple[int, ...]
    relative_placement_signature: tuple[
        tuple[tuple[int, int, int], ...],
        tuple[tuple[int, int, bool, str], ...],
    ]


@dataclass(frozen=True)
class RuleMiningRow:
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    support_action_count: int
    weight_multiset: tuple[str, ...]
    run_profile: str
    sparse_inside_orbit: bool
    packet_count: int
    top_packet_size: int
    bottom_packet_size: int
    max_gap: int
    unit_gap_count: int
    large_gap_count: int
    top_packet_is_singleton: bool
    bottom_packet_is_singleton: bool
    all_gaps_are_one: bool
    has_large_middle_packet: bool
    sorted_packet_sizes: tuple[int, ...]
    action_features: tuple[RuleMiningActionFeature, ...]
    touched_packet_set_multiset: tuple[tuple[int, ...], ...]
    exposure_vector_multiset: tuple[tuple[str, ...], ...]
    run_boundary_pattern_multiset: tuple[tuple[tuple[int, int, bool, str], ...], ...]
    successor_packet_type_weights: tuple[tuple[tuple[int, ...], float], ...]
    relative_placement_signature: object
    rule_hint: str


@dataclass(frozen=True)
class SuccessorFlowRow:
    source: RuleMiningRow
    successor_distribution: tuple[tuple[tuple[int, ...], str], ...]
    orbit_exposure_multiset: tuple[tuple[str, ...], ...]
    touched_packet_set_multiset: tuple[tuple[int, ...], ...]
    coarse_signature: object
    rule_successor_signature: object
    rule_weight_successor_signature: object
    packet_regime_successor_signature: object


@dataclass(frozen=True)
class FlowRoleAction:
    weight: float
    action: tuple[int, ...]
    child_state: tuple[int, ...]
    child_packet_type: tuple[int, ...]
    child_packet_gaps: tuple[int, ...]
    role_signature: tuple[str, int, str, str, str, str, str, str]


@dataclass(frozen=True)
class FlowRoleRow:
    source: RuleMiningRow
    exact_successor_distribution: tuple[tuple[tuple[int, ...], str], ...]
    role_distribution: tuple[tuple[str, str], ...]
    actions: tuple[FlowRoleAction, ...]


@dataclass(frozen=True)
class K6OneRunAliveActionRow:
    weight: float
    action: tuple[int, ...]
    edge_signature: tuple[int, ...]
    interval: tuple[int, int] | None
    interval_relative_to_alive: tuple[int, int] | None
    relation_class: str
    successor_packet_type: tuple[int, ...]


@dataclass(frozen=True)
class K6OneRunAlivePatternRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    learner_distribution: tuple[float, ...]
    alive_positions: tuple[int, ...]
    alive_count: int
    alive_span: tuple[int, int] | None
    alive_is_prefix: bool
    alive_packet_type: tuple[int, ...]
    support: tuple[K6OneRunAliveActionRow, ...]


@dataclass(frozen=True)
class K6AliveCandidateBenchmarkRow:
    T: int
    name: str
    value: float
    optimal_value: float
    gap_to_optimal: float
    normalized_gap_to_optimal: float
    one_run_value: float
    gap_to_one_run: float
    normalized_gap_to_one_run: float
    alive_count_histogram: tuple[tuple[int, float], ...]
    pattern_usage_histogram: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class K6OneRunIntervalActionRow:
    weight: float
    action: tuple[int, ...]
    edge_signature: tuple[int, ...]
    interval: tuple[int, int] | None
    relation_class: str
    successor_packet_type: tuple[int, ...]


@dataclass(frozen=True)
class K6OneRunIntervalRuleRow:
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    value: float
    learner_distribution: tuple[float, ...]
    alive_count: int
    support: tuple[K6OneRunIntervalActionRow, ...]


@dataclass(frozen=True)
class K6OneRunCandidateFailureRow:
    policy_name: str
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    adjacent_gaps: tuple[int, ...]
    occupancy_probability: float
    one_run_value: float
    candidate_step_value_on_one_run_continuation: float
    action_loss: float
    weighted_action_loss: float
    candidate_actions: tuple[tuple[float, tuple[int, ...]], ...]
    candidate_patterns: tuple[str, ...]
    lp_interval_distribution: tuple[tuple[tuple[int, int] | None, float], ...]
    lp_dominant_interval: tuple[int, int] | None
    lp_dominant_relation: str


@dataclass(frozen=True)
class K6OneRunPatch:
    name: str
    remaining_horizon: int
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    interval: tuple[int, int]


@dataclass(frozen=True)
class K6PatchLadderBenchmarkRow:
    policy_name: str
    patch_names: tuple[str, ...]
    value: float
    one_run_value: float
    gap_to_one_run: float
    normalized_gap_to_one_run: float
    optimal_value: float
    gap_to_optimal: float
    normalized_gap_to_optimal: float
    top_prefix_value: float
    gap_to_top_prefix: float
    patch_fire_counts: tuple[tuple[str, int, float], ...]
    top_failure_summaries: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class K6GreedyPatchSearchStep:
    step: int
    selected_patch: K6OneRunPatch | None
    improvement: float
    value: float
    one_run_value: float
    normalized_gap_to_one_run: float
    top_prefix_value: float
    gap_to_top_prefix: float
    rejected_candidates: tuple[tuple[K6OneRunPatch, float], ...]
    top_failure_summaries: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MixedIntervalPatch:
    name: str
    remaining_horizon: int
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...]


@dataclass(frozen=True)
class K6GreedyMixedPatchSearchStep:
    step: int
    selected_patch: MixedIntervalPatch | None
    improvement: float
    value: float
    one_run_value: float
    normalized_gap_to_one_run: float
    top_prefix_value: float
    gap_to_top_prefix: float
    rejected_candidates: tuple[tuple[MixedIntervalPatch, float], ...]
    top_failure_summaries: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class K6ExplicitLongHorizonSweepRow:
    T: int
    policy_name: str
    value: float
    normalized_value: float
    top_prefix_value: float
    gap_to_top_prefix: float
    normalized_gap_to_top_prefix: float


@dataclass(frozen=True)
class ReachablePolicyEvaluationResult:
    value: float
    visited_state_count: int
    visited_by_horizon: tuple[tuple[int, int], ...]
    cache_hits: int
    max_support_size: int


@dataclass(frozen=True)
class ExplicitTimePolicyReachableBenchmarkRow:
    k: int
    T: int
    policy_name: str
    value: float
    normalized_value: float
    top_prefix_value: float
    gap_to_top_prefix: float
    normalized_gap_to_top_prefix: float
    visited_state_count: int
    visited_by_horizon: tuple[tuple[int, int], ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class BarycenterPacketSummary:
    status: str
    packet_index: int
    packet_size: int
    exposure: float
    split_rate: float


@dataclass(frozen=True)
class BarycenterActionRow:
    library_name: str
    time: int
    remaining_horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    occupancy_probability: float
    mean_score: float
    packet_summaries: tuple[BarycenterPacketSummary, ...]
    top_exposure: float
    bottom_exposure: float
    p_top1_bottom0: float
    p_top0_bottom1: float
    p_top_equals_bottom: float
    top_bottom_covariance: float
    interval_distribution: tuple[tuple[tuple[int, int] | None, float], ...]


@dataclass(frozen=True)
class Chase5VerificationRow:
    T: int
    remaining_horizon: int
    state: tuple[int, ...]
    action: tuple[int, ...]
    occupancy_probability: float
    residual: float
    winner_probabilities: tuple[float, ...]
    chase_support_residuals: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Chase5VerificationSummary:
    T: int
    value: float
    normalized_value: float
    max_residual: float
    weighted_positive_residual: float
    reachable_state_count: int
    top_rows: tuple[Chase5VerificationRow, ...]


@dataclass(frozen=True)
class Chase5CertificateRow:
    T: int
    remaining_horizon: int
    state: tuple[int, ...]
    occupancy_probability: float
    potential_value: float
    pm_learner: tuple[float, ...]
    pm_max_residual: float
    pm_max_action: tuple[int, ...]
    repaired_learner: tuple[float, ...]
    repaired_alpha: float
    repaired_max_action: tuple[int, ...]
    pm_chase_support_residuals: tuple[tuple[str, float], ...]
    repaired_chase_support_residuals: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Chase5CertificateSummary:
    T: int
    value: float
    normalized_value: float
    max_pm_residual: float
    max_repaired_alpha: float
    weighted_positive_pm_residual: float
    weighted_positive_repaired_alpha: float
    pm_fail_repaired_success_rows: tuple[Chase5CertificateRow, ...]
    repaired_failure_rows: tuple[Chase5CertificateRow, ...]


@dataclass(frozen=True)
class K6BarycentricBenchmarkRow:
    T: int
    policy_name: str
    value: float
    normalized_value: float
    fixed_comb_value: float
    gap_to_fixed_comb: float
    normalized_gap_to_fixed_comb: float
    visited_state_count: int
    elapsed_seconds: float
    trigger_state_count: int
    trigger_occupancy: float
    top_triggered_states: tuple[tuple[int, tuple[int, ...], float], ...]


@dataclass(frozen=True)
class K6BarycentricTriggerSpec:
    name: str
    description: str
    predicate: object


@dataclass(frozen=True)
class K6BarycentricTriggerSweepCandidate:
    name: str
    trigger_name: str
    trigger_description: str
    mixture_name: str
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...]
    predicate: object


@dataclass(frozen=True)
class K6BarycentricTriggerSweepRow:
    candidate_name: str
    trigger_name: str
    trigger_description: str
    mixture_name: str
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...]
    T: int
    value: float
    normalized_value: float
    fixed_comb_value: float
    gap_to_fixed_comb: float
    normalized_gap_to_fixed_comb: float
    trigger_state_count: int
    trigger_occupancy: float | None
    visited_state_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class K5CenteredPotentialComparisonRow:
    state: tuple[int, ...]
    used: int
    potential_name: str
    phi_value: float
    comb_delta: float
    chase_delta: float
    delta_diff: float
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    max_gap: int
    top_gap: float
    bottom_gap: float
    multiple_near_leaders: bool
    bottom_below_mean: bool


@dataclass(frozen=True)
class K5SoftmaxPotentialCertificateRow:
    state: tuple[int, ...]
    used: int
    horizon: int
    tau_name: str
    tau: float
    phi_value: float
    learner: tuple[float, ...]
    max_residual: float
    max_centered_residual: float
    max_action: tuple[int, ...]
    chase_average_residual: float
    comb_average_residual: float
    chase_delta: float
    comb_delta: float
    delta_diff: float
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]


@dataclass(frozen=True)
class PotentialGreedyGrammarCandidate:
    name: str
    support: tuple[tuple[float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class PotentialGreedyIntervalBenchmarkRow:
    k: int
    T: int
    policy_name: str
    value: float
    normalized_value: float
    best_baseline_value: float
    gap_to_best_baseline: float
    normalized_gap_to_best_baseline: float
    visited_state_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RandomPlayoutSpec:
    name: str
    grammar_weights: tuple[tuple[str, float], ...]
    support: tuple[tuple[float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class RandomPlayoutValueRow:
    k: int
    T: int
    playout_name: str
    value: float
    normalized_value: float


@dataclass(frozen=True)
class RandomPlayoutGreedyRow:
    k: int
    T: int
    playout_name: str
    value: float
    normalized_value: float
    best_baseline_value: float
    gap_to_best_baseline: float
    normalized_gap_to_best_baseline: float
    visited_state_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class RandomPlayoutResidualRow:
    k: int
    T: int
    playout_name: str
    remaining_horizon: int
    state: tuple[int, ...]
    max_residual: float
    max_action: tuple[int, ...]
    learner: tuple[float, ...]


@dataclass(frozen=True)
class K5GapTransitionRow:
    state: tuple[int, ...]
    gap_vector: tuple[int, ...]
    packet_type: tuple[int, ...]
    packet_gaps: tuple[int, ...]
    action_name: str
    next_states: tuple[tuple[tuple[int, ...], float], ...]
    next_gap_vectors: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class K5ExchangeabilityRow:
    action_a: str
    action_w: str
    max_tv_error: float
    worst_state: tuple[int, ...]
    worst_gap_vector: tuple[int, ...]
    worst_packet_type: tuple[int, ...]


@dataclass(frozen=True)
class K5ScalarPotentialExchangeabilityRow:
    playout_name: str
    horizon: int
    action_a: str
    action_w: str
    max_abs_error: float
    worst_state: tuple[int, ...]
    worst_gap_vector: tuple[int, ...]
    worst_packet_type: tuple[int, ...]
    aw_value: float
    wa_value: float


@dataclass(frozen=True)
class K5BoundaryCorrectionFitRow:
    playout_name: str
    horizon: int
    feature_family: str
    feature_count: int
    sample_count: int
    base_max_abs_defect: float
    base_avg_abs_defect: float
    corrected_max_abs_defect: float
    corrected_avg_abs_defect: float
    improvement_factor: float
    top_weights: tuple[tuple[str, float], ...]
    top_failures: tuple[
        tuple[float, float, tuple[int, ...], tuple[int, ...], tuple[int, ...], str, str],
        ...,
    ]
    grouped_failures: tuple[tuple[tuple[int, ...], tuple[int, ...], float, int], ...]


@dataclass(frozen=True)
class K5RelaxationCertificateRow:
    potential_name: str
    horizon: int
    state: tuple[int, ...]
    gap_vector: tuple[int, ...]
    packet_type: tuple[int, ...]
    zero_gap_pattern: tuple[int, ...]
    rel_value: float
    natural_learner: tuple[float, ...] | None
    natural_alpha: float | None
    repaired_learner: tuple[float, ...]
    repaired_alpha: float
    repaired_worst_action: tuple[int, ...]


@dataclass(frozen=True)
class K5RelaxationCertificateSummaryRow:
    potential_name: str
    horizon: int
    sample_count: int
    max_repaired_alpha: float
    avg_positive_repaired_alpha: float
    max_natural_alpha: float | None
    avg_positive_natural_alpha: float | None


@dataclass(frozen=True)
class K5RepairedLearnerAnatomyRow:
    horizon: int
    state: tuple[int, ...]
    gap_vector: tuple[int, ...]
    packet_type: tuple[int, ...]
    zero_gap_pattern: tuple[int, ...]
    repaired_alpha: float
    repaired_p: tuple[float, ...]
    natural_p: tuple[float, ...] | None
    p_minus_uniform: tuple[float, ...]
    packet_average_p: tuple[float, ...]
    packet_average_by_packet: tuple[float, ...]
    within_packet_variance: float
    worst_action: tuple[int, ...]
    active_worst_actions: tuple[tuple[int, ...], ...]
    packet_uniform_alpha: float
    best_softmax_c: float
    best_softmax_alpha: float


@dataclass(frozen=True)
class K5PacketLearnerFormulaRow:
    source: K5RepairedLearnerAnatomyRow
    packet_weights: tuple[float, ...]
    rational_packet_weights: tuple[str, ...]
    rationalized_p: tuple[float, ...]
    rationalized_alpha: float
    packet_score_softmax_c: float
    packet_score_softmax_alpha: float
    affine_packet_slope: float
    affine_packet_alpha: float


@dataclass(frozen=True)
class K5ReducedPacketLPRow:
    horizon: int
    state: tuple[int, ...]
    packet_type: tuple[int, ...]
    gap_vector: tuple[int, ...]
    zero_gap_pattern: tuple[int, ...]
    full_alpha: float
    full_packet_weights: tuple[float, ...]
    reduced_alpha: float
    reduced_weights: tuple[float, ...]
    alpha_difference: float
    max_weight_difference: float
    count_vector_count: int
    active_constraints: tuple[tuple[tuple[int, ...], float, float], ...]
    dual_marginals: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class K5ReducedPacketLPCoverageRow:
    section: str
    scope: str
    horizon: int
    state_count: int
    max_alpha: float
    avg_positive_alpha: float
    scaled_max_alpha: float
    worst_state: tuple[int, ...]
    worst_packet_type: tuple[int, ...]
    worst_gap_vector: tuple[int, ...]
    worst_weights: tuple[float, ...]
    worst_active_count_vectors: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class K5WorstFamilyScalingRow:
    horizon: int
    family: str
    state: tuple[int, ...]
    s1: float
    s2: float
    s3: float
    alpha: float
    scaled_alpha: float
    weights: tuple[float, ...]
    active_count_vectors: tuple[tuple[int, ...], ...]
    active_constraints: tuple[tuple[tuple[int, ...], float, float], ...]


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


def _fixed_weight_action_policy(
    weighted_actions: tuple[tuple[float, str], ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _policy_from_weighted_actions(
        tuple((weight, tuple(int(bit) for bit in action_text)) for weight, action_text in weighted_actions)
    )


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


def two_run_skeleton_v3a_late_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    k = len(canonical_state)
    if k != 9:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))

    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    max_gap = max(gaps, default=0)

    if (
        remaining_horizon == 2
        and len(ptype) >= 4
        and (2 in gaps or sum(1 for gap in gaps if gap == 1) >= 3)
    ):
        return _fixed_weight_action_policy(
            (
                (1.0 / 3.0, "001000010"),
                (1.0 / 3.0, "010001000"),
                (1.0 / 3.0, "100000100"),
            )
        )

    if (
        remaining_horizon in {1, 3}
        and len(ptype) >= 3
        and ptype[0] >= 3
        and 1 in gaps
    ):
        return _fixed_weight_action_policy(
            (
                (1.0 / 3.0, "001000010"),
                (1.0 / 6.0, "010001000"),
                (1.0 / 6.0, "010100101"),
                (1.0 / 6.0, "100000100"),
                (1.0 / 6.0, "100101000"),
            )
        )

    if (
        remaining_horizon in {1, 2}
        and len(ptype) >= 3
        and (max_gap >= 2 or ptype[0] == 1 or sum(1 for gap in gaps if gap == 1) >= 2)
    ):
        return _fixed_weight_action_policy(
            (
                (1.0 / 5.0, "000010001"),
                (1.0 / 5.0, "000100000"),
                (1.0 / 5.0, "001000010"),
                (1.0 / 5.0, "010001000"),
                (1.0 / 5.0, "100000100"),
            )
        )

    return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))


def two_run_skeleton_v3b_early_late_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    if len(canonical_state) != 9:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))

    if remaining_horizon == 7 and canonical_state == (0, 0, 0, 0, 0, 0, 0, 0, 0):
        return _fixed_weight_action_policy(
            (
                (2.0 / 9.0, "110100001"),
                (2.0 / 9.0, "001011010"),
                (1.0 / 9.0, "000011110"),
                (1.0 / 9.0, "101000101"),
                (1.0 / 9.0, "010110100"),
                (1.0 / 18.0, "001101010"),
                (1.0 / 18.0, "111001000"),
                (1.0 / 18.0, "100000111"),
                (1.0 / 18.0, "010100101"),
            )
        )

    if remaining_horizon == 6 and canonical_state == (1, 1, 1, 1, 0, 0, 0, 0, 0):
        return _fixed_weight_action_policy(
            (
                (5.0 / 17.0, "100101000"),
                (4.0 / 17.0, "001010111"),
                (3.0 / 17.0, "011010000"),
                (1.0 / 17.0, "010001111"),
                (1.0 / 17.0, "110000001"),
                (1.0 / 17.0, "010100100"),
                (1.0 / 17.0, "010100010"),
                (1.0 / 17.0, "100001111"),
            )
        )

    if remaining_horizon == 5 and canonical_state == (2, 2, 1, 1, 1, 0, 0, 0, 0):
        return _fixed_weight_action_policy(
            (
                (2.0 / 7.0, "101001010"),
                (1.0 / 7.0, "010010101"),
                (1.0 / 7.0, "001110000"),
                (1.0 / 7.0, "010100011"),
                (1.0 / 7.0, "010101100"),
                (1.0 / 7.0, "100010101"),
            )
        )

    if remaining_horizon == 4 and canonical_state == (3, 2, 2, 1, 1, 1, 1, 0, 0):
        return _fixed_weight_action_policy(
            (
                (1.0 / 3.0, "011000000"),
                (2.0 / 9.0, "100010101"),
                (2.0 / 9.0, "100101000"),
                (1.0 / 9.0, "001111100"),
                (1.0 / 9.0, "010111100"),
            )
        )

    return two_run_skeleton_v3a_late_policy(canonical_state, remaining_horizon)


def _time_dependent_fallback_policy(
    fallback_name: str,
):
    if fallback_name == "two_run_dual_support_replay_k9_T7":
        return two_run_dual_support_replay_k9_T7_policy
    if fallback_name == "two_run_skeleton_v3a_late_policy":
        return two_run_skeleton_v3a_late_policy
    if fallback_name == "two_run_skeleton_v3b_early_late_policy":
        return two_run_skeleton_v3b_early_late_policy
    if fallback_name == "k6_alive_trunc_comb_policy":
        return k6_alive_trunc_comb_policy
    if fallback_name == "k6_alive_trunc_comb_twin3_policy":
        return k6_alive_trunc_comb_twin3_policy
    if fallback_name == "k6_alive_full_comb_control_policy":
        return k6_alive_full_comb_control_policy
    if fallback_name == "k6_one_run_rule_v1_policy":
        return k6_one_run_rule_v1_policy
    if fallback_name == "k6_one_run_rule_v1_no_mixed_policy":
        return k6_one_run_rule_v1_no_mixed_policy
    if fallback_name == "k6_one_run_rule_v1_aggressive_shoulder_policy":
        return k6_one_run_rule_v1_aggressive_shoulder_policy

    def fallback(state: tuple[int, ...], remaining_horizon: int):
        del remaining_horizon
        policies = _policy_registry(len(state))
        if fallback_name not in policies:
            raise ValueError(f"unknown fallback policy: {fallback_name}")
        return policies[fallback_name](state)

    return fallback


def make_two_run_replay_cutoff_policy(
    *,
    replay_min_remaining: int | None = None,
    replay_max_remaining: int | None = None,
    fallback_name: str = "top_prefix_three_regime_v6",
):
    fallback_policy = _time_dependent_fallback_policy(fallback_name)

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        use_replay = len(canonical_state) == 9 and 1 <= remaining_horizon <= 7
        if replay_min_remaining is not None and remaining_horizon < replay_min_remaining:
            use_replay = False
        if replay_max_remaining is not None and remaining_horizon > replay_max_remaining:
            use_replay = False
        if use_replay:
            replay_policy = _two_run_dual_support_replay_table_k9_T7().get((remaining_horizon, canonical_state))
            if replay_policy is not None:
                return replay_policy
        return fallback_policy(canonical_state, remaining_horizon)

    return policy


def make_two_run_replay_mask_policy(
    replay_horizons: frozenset[int],
    fallback_policy_name: str,
):
    fallback_policy = _time_dependent_fallback_policy(fallback_policy_name)

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        if len(canonical_state) == 9 and remaining_horizon in replay_horizons:
            replay_policy = _two_run_dual_support_replay_table_k9_T7().get((remaining_horizon, canonical_state))
            if replay_policy is not None:
                return replay_policy
        return fallback_policy(canonical_state, remaining_horizon)

    return policy


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


def _histogram_tuple(values: list[object]) -> tuple[tuple[object, int], ...]:
    counts: dict[object, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _skeleton_label(row: TwoRunReplayTemplateRow) -> str:
    if row.packet_type == (9,):
        return "root-special"
    support = row.support
    touches_top = sum(1 for _, action, _, _ in support if action[0] == 1)
    touches_bottom = sum(1 for _, action, _, _ in support if action[-1] == 1)
    one_counts = {sum(action) for _, action, _, _ in support}
    if touches_top == 0 and touches_bottom > 0:
        return "tail-repair"
    if touches_top == len(support):
        return "single-anchor"
    if len(one_counts) > 1:
        return "split-two-packets"
    if touches_top > 0 and touches_bottom > 0:
        return "paired-shoulder"
    return "central-repair"


def _skeleton_placement_aggregates(
    rows: tuple[TwoRunReplayTemplateRow, ...],
    family_name: str,
    signature_fn,
) -> tuple[SkeletonPlacementAggregate, ...]:
    grouped: dict[object, list[TwoRunReplayTemplateRow]] = defaultdict(list)
    for row in rows:
        grouped[signature_fn(row)].append(row)

    aggregates: list[SkeletonPlacementAggregate] = []
    for signature, group in grouped.items():
        representative = max(group, key=lambda row: row.occupancy_probability)
        state_occupancy: dict[tuple[int, ...], float] = defaultdict(float)
        for row in group:
            state_occupancy[row.state] += row.occupancy_probability
        aggregates.append(
            SkeletonPlacementAggregate(
                family_name=family_name,
                signature=signature,
                row_count=len(group),
                total_occupancy=sum(row.occupancy_probability for row in group),
                horizons=tuple(_histogram_tuple([row.remaining_horizon for row in group])),
                representative=representative,
                packet_type_histogram=tuple(_histogram_tuple([row.packet_type for row in group])),
                packet_gaps_histogram=tuple(_histogram_tuple([row.packet_gaps for row in group])),
                adjacent_gaps_histogram=tuple(_histogram_tuple([row.adjacent_gaps for row in group])),
                top_states_by_occupancy=tuple(
                    sorted(state_occupancy.items(), key=lambda item: (-item[1], item[0]))[:8]
                ),
                label=_skeleton_label(representative),
            )
        )
    return tuple(sorted(aggregates, key=lambda item: (-item.total_occupancy, -item.row_count, str(item.signature))))


def two_run_skeleton_placement_aggregates(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> dict[str, tuple[SkeletonPlacementAggregate, ...]]:
    rows = two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)
    return {
        "action_weight_skeleton": _skeleton_placement_aggregates(rows, "action_weight_skeleton", _coarse_action_weight_skeleton),
        "edge_weight_skeleton": _skeleton_placement_aggregates(rows, "edge_weight_skeleton", _coarse_edge_weight_skeleton),
    }


def _format_histogram(items: tuple[tuple[object, int], ...], n: int = 6) -> str:
    rendered = ", ".join(f"{key}:{count}" for key, count in items[:n])
    if len(items) > n:
        rendered += ", ..."
    return rendered


def _format_action_geometry(
    action: tuple[int, ...],
    weight: float,
    edge_signature: tuple[int, ...],
    orbit_key: tuple[int, ...],
) -> str:
    one_positions = tuple(index for index, bit in enumerate(action) if bit == 1)
    touches: list[str] = []
    if action and action[0] == 1:
        touches.append("top")
    if action and action[-1] == 1:
        touches.append("bottom")
    if any(action[1:-1]):
        touches.append("middle")
    return (
        f"w={_format_fractionish(weight):>8s}"
        f" action={_format_action(action)}"
        f" ones={sum(action)}"
        f" edge={_format_edge_signature(edge_signature)}"
        f" runs={tuple(_edge_run_intervals(edge_signature))}"
        f" one_pos={one_positions}"
        f" touches={tuple(touches)}"
        f" orbit={orbit_key}"
    )


def print_two_run_skeleton_placement_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 80,
) -> None:
    aggregates_by_family = two_run_skeleton_placement_aggregates(k, T, support_tolerance=support_tolerance)
    print(f"Two-run skeleton placement report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print()
    for family_name, aggregates in aggregates_by_family.items():
        print(f"{family_name}: distinct skeletons={len(aggregates)}")
    print()

    for family_name, aggregates in aggregates_by_family.items():
        print(f"{family_name}: top skeletons by occupancy")
        for index, aggregate in enumerate(sorted(aggregates, key=lambda item: (-item.total_occupancy, -item.row_count, str(item.signature)))[:n], start=1):
            representative = aggregate.representative
            print(
                f"  {index}. label={aggregate.label}"
                f" rows={aggregate.row_count}"
                f" occupancy={aggregate.total_occupancy:.6f}"
                f" horizons={aggregate.horizons}"
            )
            print(f"     representative state={representative.state}")
            print(f"     representative packet={representative.packet_type} gaps={representative.packet_gaps}")
            print(f"     packet_type hist={_format_histogram(aggregate.packet_type_histogram)}")
            print(f"     packet_gaps hist={_format_histogram(aggregate.packet_gaps_histogram)}")
            print(f"     adjacent_gaps hist={_format_histogram(aggregate.adjacent_gaps_histogram)}")
            print(f"     top states={aggregate.top_states_by_occupancy}")
            print(f"     signature={_format_coarse_signature(aggregate.signature)}")
            print("     action geometry:")
            for weight, action, edge_signature, orbit_key in representative.support[:12]:
                print(f"       {_format_action_geometry(action, weight, edge_signature, orbit_key)}")
            if len(representative.support) > 12:
                print("       ...")
        print()

        print(f"{family_name}: top skeletons by row count")
        for index, aggregate in enumerate(sorted(aggregates, key=lambda item: (-item.row_count, -item.total_occupancy, str(item.signature)))[:n], start=1):
            print(
                f"  {index}. label={aggregate.label}"
                f" rows={aggregate.row_count}"
                f" occupancy={aggregate.total_occupancy:.6f}"
                f" horizons={aggregate.horizons}"
                f" signature={_format_coarse_signature(aggregate.signature)}"
            )
        print()


def _packet_index_for_position(groups: tuple[tuple[int, ...], ...], position: int) -> int:
    for packet_index, group in enumerate(groups):
        if position in group:
            return packet_index
    raise ValueError(f"position {position} is outside packet groups")


def _packet_boundary_edges(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(group[-1] for group in groups[:-1])


def _distance_to_nearest_boundary(edge_index: int, boundaries: tuple[int, ...]) -> int:
    if not boundaries:
        return 0
    return min(abs(edge_index - boundary) for boundary in boundaries)


def _run_span_label(
    interval: tuple[int, int],
    boundaries: tuple[int, ...],
) -> str:
    crossed = sum(1 for boundary in boundaries if interval[0] <= boundary <= interval[1])
    if crossed == 0:
        return "inside-packet"
    if crossed == 1:
        return "bridges-two-packets"
    return "bridges-many-packets"


def _relative_action_feature(
    row: TwoRunReplayTemplateRow,
    weight: float,
    action: tuple[int, ...],
    edge_signature: tuple[int, ...],
    orbit_key: tuple[int, ...],
) -> RuleMiningActionFeature:
    groups = _packet_index_groups(row.state)
    packet_sizes = tuple(len(group) for group in groups)
    one_positions = tuple(index for index, bit in enumerate(action) if bit == 1)
    touched_packet_indices = tuple(
        sorted({_packet_index_for_position(groups, position) for position in one_positions})
    )
    exposure_vector = tuple(
        _format_fractionish(ones / packet_size if packet_size else 0.0)
        for ones, packet_size in zip(orbit_key, packet_sizes, strict=True)
    )
    boundaries = _packet_boundary_edges(groups)
    run_intervals = tuple(_edge_run_intervals(edge_signature))
    run_boundary_distances = tuple(
        (
            _distance_to_nearest_boundary(interval[0], boundaries),
            _distance_to_nearest_boundary(interval[1], boundaries),
        )
        for interval in run_intervals
    )
    run_crosses_boundary = tuple(
        any(interval[0] <= boundary <= interval[1] for boundary in boundaries)
        for interval in run_intervals
    )
    run_span_labels = tuple(_run_span_label(interval, boundaries) for interval in run_intervals)
    next_state = canon(tuple(row.state[index] + action[index] for index in range(len(row.state))))
    relative_one_positions = tuple(
        sorted(
            (
                packet_index,
                position - groups[packet_index][0],
                groups[packet_index][-1] - position,
            )
            for position in one_positions
            for packet_index in (_packet_index_for_position(groups, position),)
        )
    )
    run_pattern = tuple(
        (
            start_distance,
            end_distance,
            crosses,
            span_label,
        )
        for (start_distance, end_distance), crosses, span_label in zip(
            run_boundary_distances,
            run_crosses_boundary,
            run_span_labels,
            strict=True,
        )
    )
    return RuleMiningActionFeature(
        weight=weight,
        action=action,
        edge_signature=edge_signature,
        one_positions=one_positions,
        ones_count=sum(action),
        touches_top=bool(action and action[0] == 1),
        touches_bottom=bool(action and action[-1] == 1),
        touched_packet_indices=touched_packet_indices,
        orbit_key=orbit_key,
        exposure_vector=exposure_vector,
        run_intervals=run_intervals,
        run_boundary_distances=run_boundary_distances,
        run_crosses_boundary=run_crosses_boundary,
        run_span_labels=run_span_labels,
        successor_packet_type=packet_type(next_state),
        relative_placement_signature=(relative_one_positions, run_pattern),
    )


def _rule_hint_for_row(row: TwoRunReplayTemplateRow) -> str:
    if row.packet_type == (9,):
        return "root-special"
    if row.remaining_horizon == 6 and len(row.packet_type) == 2:
        return "first-split"
    if row.remaining_horizon <= 1:
        if row.packet_type and row.packet_type[0] == 1:
            return "singleton-top repair"
        return "final cleanup"
    if row.remaining_horizon <= 2:
        return "bridge"
    if len(row.packet_type) == 2:
        return "two-packet split"
    if row.packet_type and row.packet_type[0] == 1:
        return "singleton-top repair"
    if row.packet_type and row.packet_type[-1] == 1:
        return "tail repair"
    if any(size >= 4 for size in row.packet_type[1:-1]):
        return "shoulder repair"
    return "bridge"


def _rule_mining_row_from_template(row: TwoRunReplayTemplateRow) -> RuleMiningRow:
    gaps = row.packet_gaps
    middle_sizes = row.packet_type[1:-1]
    action_features = tuple(
        _relative_action_feature(row, weight, action, edge_signature, orbit_key)
        for weight, action, edge_signature, orbit_key in row.support
    )
    touched_packet_set_multiset = tuple(sorted(feature.touched_packet_indices for feature in action_features))
    exposure_vector_multiset = tuple(sorted(feature.exposure_vector for feature in action_features))
    run_boundary_pattern_multiset = tuple(sorted(feature.relative_placement_signature[1] for feature in action_features))
    relative_placement_signature = (
        tuple(
            sorted(
                (
                    _format_fractionish(feature.weight),
                    feature.relative_placement_signature,
                )
                for feature in action_features
            )
        ),
    )
    return RuleMiningRow(
        remaining_horizon=row.remaining_horizon,
        state=row.state,
        packet_type=row.packet_type,
        packet_gaps=row.packet_gaps,
        adjacent_gaps=row.adjacent_gaps,
        occupancy_probability=row.occupancy_probability,
        support_action_count=row.support_action_count,
        weight_multiset=tuple(sorted((_format_fractionish(weight) for weight, *_ in row.support), reverse=True)),
        run_profile=row.run_profile,
        sparse_inside_orbit=row.sparse_inside_orbit,
        packet_count=len(row.packet_type),
        top_packet_size=row.packet_type[0] if row.packet_type else 0,
        bottom_packet_size=row.packet_type[-1] if row.packet_type else 0,
        max_gap=max(gaps, default=0),
        unit_gap_count=sum(1 for gap in gaps if gap == 1),
        large_gap_count=sum(1 for gap in gaps if gap >= 2),
        top_packet_is_singleton=bool(row.packet_type and row.packet_type[0] == 1),
        bottom_packet_is_singleton=bool(row.packet_type and row.packet_type[-1] == 1),
        all_gaps_are_one=bool(gaps) and all(gap == 1 for gap in gaps),
        has_large_middle_packet=any(size >= 4 for size in middle_sizes),
        sorted_packet_sizes=tuple(sorted(row.packet_type, reverse=True)),
        action_features=action_features,
        touched_packet_set_multiset=touched_packet_set_multiset,
        exposure_vector_multiset=exposure_vector_multiset,
        run_boundary_pattern_multiset=run_boundary_pattern_multiset,
        successor_packet_type_weights=row.successor_packet_type_weights,
        relative_placement_signature=relative_placement_signature,
        rule_hint=_rule_hint_for_row(row),
    )


def two_run_replay_rule_mining_rows(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[RuleMiningRow, ...]:
    return tuple(
        _rule_mining_row_from_template(row)
        for row in two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)
    )


def _format_weight_multiset(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(values) + ")"


def _format_rule_action_feature(feature: RuleMiningActionFeature) -> str:
    return (
        f"w={_format_fractionish(feature.weight):>8s}"
        f" action={_format_action(feature.action)}"
        f" ones={feature.ones_count}"
        f" one_pos={feature.one_positions}"
        f" touches_top={feature.touches_top}"
        f" touches_bottom={feature.touches_bottom}"
        f" packets={feature.touched_packet_indices}"
        f" orbit={feature.orbit_key}"
        f" exposure={feature.exposure_vector}"
        f" edge={_format_edge_signature(feature.edge_signature)}"
        f" runs={feature.run_intervals}"
        f" run_dist={feature.run_boundary_distances}"
        f" run_cross={feature.run_crosses_boundary}"
        f" run_span={feature.run_span_labels}"
        f" successor={feature.successor_packet_type}"
    )


def _print_rule_mining_row(row: RuleMiningRow, support_n: int = 10) -> None:
    print(
        f"  rem={row.remaining_horizon}"
        f" state={row.state}"
        f" packet={row.packet_type} gaps={row.packet_gaps}"
        f" adjacent={row.adjacent_gaps}"
        f" occ={row.occupancy_probability:.6f}"
        f" hint={row.rule_hint}"
    )
    print(
        f"    support={row.support_action_count}"
        f" weights={_format_weight_multiset(row.weight_multiset)}"
        f" run_profile={row.run_profile}"
        f" sparse_inside_orbit={row.sparse_inside_orbit}"
    )
    print(
        f"    packets={row.packet_count}"
        f" top={row.top_packet_size}"
        f" bottom={row.bottom_packet_size}"
        f" max_gap={row.max_gap}"
        f" unit_gaps={row.unit_gap_count}"
        f" large_gaps={row.large_gap_count}"
        f" top_singleton={row.top_packet_is_singleton}"
        f" bottom_singleton={row.bottom_packet_is_singleton}"
        f" all_gaps_one={row.all_gaps_are_one}"
        f" large_middle={row.has_large_middle_packet}"
        f" sorted_packets={row.sorted_packet_sizes}"
    )
    print(f"    touched packet sets={row.touched_packet_set_multiset}")
    print(f"    exposure vectors={row.exposure_vector_multiset}")
    successors = ", ".join(
        f"{ptype}:{_format_fractionish(weight)}"
        for ptype, weight in row.successor_packet_type_weights[:8]
    )
    print(f"    successor packet types={successors}")
    print("    support actions:")
    for feature in row.action_features[:support_n]:
        print(f"      {_format_rule_action_feature(feature)}")
    if len(row.action_features) > support_n:
        print("      ...")


def _group_rule_rows(rows: tuple[RuleMiningRow, ...], key_fn) -> tuple[tuple[object, list[RuleMiningRow]], ...]:
    grouped: dict[object, list[RuleMiningRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return tuple(sorted(grouped.items(), key=lambda item: (-sum(row.occupancy_probability for row in item[1]), -len(item[1]), str(item[0]))))


def print_two_run_replay_rule_mining_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 80,
) -> None:
    rows = two_run_replay_rule_mining_rows(k, T, support_tolerance=support_tolerance)
    absolute_action_skeletons = {_coarse_action_weight_skeleton(row) for row in two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)}
    edge_skeletons = {_coarse_edge_weight_skeleton(row) for row in two_run_replay_template_rows(k, T, support_tolerance=support_tolerance)}
    relative_signatures = {row.relative_placement_signature for row in rows}
    weight_multisets = {row.weight_multiset for row in rows}
    packet_relative_combinations = {
        (row.packet_type, row.packet_gaps, row.relative_placement_signature)
        for row in rows
    }

    print(f"Two-run replay rule-mining report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"total replay states: {len(rows)}")
    print(f"distinct absolute action skeletons: {len(absolute_action_skeletons)}")
    print(f"distinct edge skeletons: {len(edge_skeletons)}")
    print(f"distinct relative placement signatures: {len(relative_signatures)}")
    print(f"distinct weight multisets: {len(weight_multisets)}")
    print(f"distinct packet-regime -> relative-placement combinations: {len(packet_relative_combinations)}")
    print()

    print("Top replay states by occupancy:")
    for row in sorted(rows, key=lambda item: (-item.occupancy_probability, item.remaining_horizon, item.state))[:n]:
        _print_rule_mining_row(row)
    print()

    print("Important early/bridge/final rows:")
    important_rows = [
        row
        for row in rows
        if row.remaining_horizon in {1, 2}
        or row.packet_type in {(9,), (4, 5), (2, 3, 4), (1, 2, 4, 2), (1, 7, 1)}
    ]
    for row in sorted(important_rows, key=lambda item: (-item.occupancy_probability, -item.remaining_horizon, item.state))[:n]:
        _print_rule_mining_row(row)
    print()

    print("Groups by weight_multiset + relative placement signature:")
    for index, (signature, group) in enumerate(
        _group_rule_rows(rows, lambda row: (row.weight_multiset, row.relative_placement_signature))[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.occupancy_probability)
        print(
            f"  {index}. rows={len(group)}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.remaining_horizon for row in group])}"
            f" hints={_histogram_tuple([row.rule_hint for row in group])}"
        )
        print(f"     signature={_format_coarse_signature(signature)}")
        print(f"     representative state={representative.state}")
        print(f"     representative packet={representative.packet_type} gaps={representative.packet_gaps}")
        print(f"     weights={_format_weight_multiset(representative.weight_multiset)}")
    print()

    print("Groups by packet_type/gaps -> support family:")
    for index, (regime, group) in enumerate(
        _group_rule_rows(rows, lambda row: (row.packet_type, row.packet_gaps))[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.occupancy_probability)
        distinct_weight_multisets = {row.weight_multiset for row in group}
        distinct_relative_signatures = {row.relative_placement_signature for row in group}
        print(
            f"  {index}. packet={regime[0]} gaps={regime[1]}"
            f" rows={len(group)}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
            f" weight_multisets={len(distinct_weight_multisets)}"
            f" relative_signatures={len(distinct_relative_signatures)}"
            f" hints={_histogram_tuple([row.rule_hint for row in group])}"
        )
        print(f"     representative rem={representative.remaining_horizon} state={representative.state}")
        print(f"     representative weights={_format_weight_multiset(representative.weight_multiset)}")
        print("     representative support:")
        for feature in representative.action_features[:8]:
            print(f"       {_format_rule_action_feature(feature)}")
        if len(representative.action_features) > 8:
            print("       ...")
    print()

    print("Candidate rule hint summary:")
    for hint, group in _group_rule_rows(rows, lambda row: row.rule_hint):
        print(
            f"  {hint}: rows={len(group)}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.remaining_horizon for row in group])}"
        )


def _successor_distribution_signature(
    successor_packet_type_weights: tuple[tuple[tuple[int, ...], float], ...],
) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (packet_type_value, _format_fractionish(weight))
        for packet_type_value, weight in successor_packet_type_weights
    )


def _successor_flow_row_from_rule_row(row: RuleMiningRow) -> SuccessorFlowRow:
    successor_distribution = _successor_distribution_signature(row.successor_packet_type_weights)
    orbit_exposure_multiset = tuple(sorted(feature.exposure_vector for feature in row.action_features))
    touched_packet_set_multiset = tuple(sorted(feature.touched_packet_indices for feature in row.action_features))
    coarse_signature = (
        row.rule_hint,
        row.remaining_horizon,
        row.packet_type,
        row.packet_gaps,
        row.weight_multiset,
        successor_distribution,
        orbit_exposure_multiset,
        touched_packet_set_multiset,
        row.support_action_count,
        row.run_profile,
    )
    rule_successor_signature = (row.rule_hint, successor_distribution)
    rule_weight_successor_signature = (row.rule_hint, row.weight_multiset, successor_distribution)
    packet_regime_successor_signature = (row.packet_type, row.packet_gaps, successor_distribution)
    return SuccessorFlowRow(
        source=row,
        successor_distribution=successor_distribution,
        orbit_exposure_multiset=orbit_exposure_multiset,
        touched_packet_set_multiset=touched_packet_set_multiset,
        coarse_signature=coarse_signature,
        rule_successor_signature=rule_successor_signature,
        rule_weight_successor_signature=rule_weight_successor_signature,
        packet_regime_successor_signature=packet_regime_successor_signature,
    )


def two_run_successor_flow_rows(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[SuccessorFlowRow, ...]:
    return tuple(
        _successor_flow_row_from_rule_row(row)
        for row in two_run_replay_rule_mining_rows(k, T, support_tolerance=support_tolerance)
    )


def _group_successor_flow_rows(
    rows: tuple[SuccessorFlowRow, ...],
    key_fn,
) -> tuple[tuple[object, list[SuccessorFlowRow]], ...]:
    grouped: dict[object, list[SuccessorFlowRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return tuple(
        sorted(
            grouped.items(),
            key=lambda item: (
                -sum(row.source.occupancy_probability for row in item[1]),
                -len(item[1]),
                str(item[0]),
            ),
        )
    )


def _format_successor_distribution(distribution: tuple[tuple[tuple[int, ...], str], ...]) -> str:
    return ", ".join(f"{packet_type_value}:{weight}" for packet_type_value, weight in distribution)


def _print_successor_flow_support(row: SuccessorFlowRow, support_n: int = 8) -> None:
    for feature in row.source.action_features[:support_n]:
        print(
            f"       w={_format_fractionish(feature.weight):>8s}"
            f" action={_format_action(feature.action)}"
            f" orbit={feature.orbit_key}"
            f" exposure={feature.exposure_vector}"
            f" packets={feature.touched_packet_indices}"
            f" successor={feature.successor_packet_type}"
        )
    if len(row.source.action_features) > support_n:
        print("       ...")


def print_two_run_successor_flow_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 100,
) -> None:
    rows = two_run_successor_flow_rows(k, T, support_tolerance=support_tolerance)
    rule_hints = {row.source.rule_hint for row in rows}
    weight_multisets = {row.source.weight_multiset for row in rows}
    successor_flow_signatures = {row.coarse_signature for row in rows}
    rule_successor_signatures = {row.rule_successor_signature for row in rows}
    rule_weight_successor_signatures = {row.rule_weight_successor_signature for row in rows}
    packet_regime_successor_signatures = {row.packet_regime_successor_signature for row in rows}

    print(f"Two-run successor-flow report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print("Summary counts:")
    print(f"  total replay states: {len(rows)}")
    print(f"  distinct rule hints: {len(rule_hints)}")
    print(f"  distinct weight multisets: {len(weight_multisets)}")
    print(f"  distinct successor-flow signatures: {len(successor_flow_signatures)}")
    print(f"  distinct rule_hint + successor_distribution signatures: {len(rule_successor_signatures)}")
    print(
        "  distinct rule_hint + weight_multiset + successor_distribution signatures:"
        f" {len(rule_weight_successor_signatures)}"
    )
    print(
        "  distinct packet_regime + successor_distribution signatures:"
        f" {len(packet_regime_successor_signatures)}"
    )
    print()

    print("Top successor-flow groups by occupancy:")
    for index, (signature, group) in enumerate(
        _group_successor_flow_rows(rows, lambda row: row.coarse_signature)[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.source.occupancy_probability)
        print(
            f"  {index}. rows={len(group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.source.remaining_horizon for row in group])}"
            f" hints={_histogram_tuple([row.source.rule_hint for row in group])}"
        )
        print(f"     signature={_format_coarse_signature(signature)}")
        print(
            f"     representative rem={representative.source.remaining_horizon}"
            f" state={representative.source.state}"
            f" packet={representative.source.packet_type}"
            f" gaps={representative.source.packet_gaps}"
        )
        print(f"     weights={_format_weight_multiset(representative.source.weight_multiset)}")
        print(f"     successor={_format_successor_distribution(representative.successor_distribution)}")
        print("     representative support:")
        _print_successor_flow_support(representative)
    print()

    print("Per rule-hint successor distributions:")
    for hint, hint_group in _group_successor_flow_rows(rows, lambda row: row.source.rule_hint):
        successor_groups = _group_successor_flow_rows(tuple(hint_group), lambda row: row.successor_distribution)
        print(
            f"  {hint}: rows={len(hint_group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in hint_group):.6f}"
            f" distinct_successor_distributions={len(successor_groups)}"
        )
        for successor_distribution, group in successor_groups[: min(n, 10)]:
            representative = max(group, key=lambda item: item.source.occupancy_probability)
            print(
                f"     rows={len(group)}"
                f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
                f" successor={_format_successor_distribution(successor_distribution)}"
                f" representative={representative.source.state}"
                f" weights={_format_weight_multiset(representative.source.weight_multiset)}"
            )
    print()

    print("Early basin flow (occupancy >= 0.1):")
    early_rows = [
        row
        for row in rows
        if row.source.occupancy_probability >= 0.1
    ]
    for row in sorted(early_rows, key=lambda item: (-item.source.remaining_horizon, item.source.state))[:n]:
        print(
            f"  rem={row.source.remaining_horizon}"
            f" state={row.source.state}"
            f" packet={row.source.packet_type} gaps={row.source.packet_gaps}"
            f" hint={row.source.rule_hint}"
            f" occ={row.source.occupancy_probability:.6f}"
            f" successor={_format_successor_distribution(row.successor_distribution)}"
            f" weights={_format_weight_multiset(row.source.weight_multiset)}"
        )
    print()

    print("Late cleanup flow, rem=1 and rem=2 grouped by rule/weights/successor:")
    late_rows = tuple(row for row in rows if row.source.remaining_horizon in {1, 2})
    for index, (signature, group) in enumerate(
        _group_successor_flow_rows(
            late_rows,
            lambda row: (row.source.rule_hint, row.source.weight_multiset, row.successor_distribution),
        )[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.source.occupancy_probability)
        print(
            f"  {index}. rows={len(group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.source.remaining_horizon for row in group])}"
            f" signature={_format_coarse_signature(signature)}"
        )
        print(
            f"     representative state={representative.source.state}"
            f" packet={representative.source.packet_type}"
            f" gaps={representative.source.packet_gaps}"
        )
    print()

    print("Candidate v4 grammar sketch:")
    for hint, hint_group in _group_successor_flow_rows(rows, lambda row: row.source.rule_hint):
        successor_groups = _group_successor_flow_rows(tuple(hint_group), lambda row: row.successor_distribution)
        weight_totals: dict[tuple[str, ...], float] = defaultdict(float)
        for row in hint_group:
            weight_totals[row.source.weight_multiset] += row.source.occupancy_probability
        dominant_weight, dominant_weight_occupancy = max(
            weight_totals.items(),
            key=lambda item: (item[1], item[0]),
        )
        total_occupancy = sum(row.source.occupancy_probability for row in hint_group)
        print(
            f"  {hint}: rows={len(hint_group)}"
            f" occupancy={total_occupancy:.6f}"
            f" successor_distributions={len(successor_groups)}"
            f" dominant_weights={_format_weight_multiset(dominant_weight)}"
            f" dominant_weight_share={dominant_weight_occupancy / total_occupancy if total_occupancy else 0.0:.3f}"
        )
        if len(successor_groups) <= 5:
            for successor_distribution, group in successor_groups:
                print(
                    f"     successor={_format_successor_distribution(successor_distribution)}"
                    f" rows={len(group)}"
                    f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
                )
        else:
            print("     fragmented successor distributions")


def _packet_feature_summary(packet_sizes: tuple[int, ...], gaps: tuple[int, ...]) -> dict[str, object]:
    return {
        "packet_count": len(packet_sizes),
        "top_packet_size": packet_sizes[0] if packet_sizes else 0,
        "bottom_packet_size": packet_sizes[-1] if packet_sizes else 0,
        "top_singleton": bool(packet_sizes and packet_sizes[0] == 1),
        "bottom_singleton": bool(packet_sizes and packet_sizes[-1] == 1),
        "max_gap": max(gaps, default=0),
        "unit_gap_count": sum(1 for gap in gaps if gap == 1),
        "large_gap_count": sum(1 for gap in gaps if gap >= 2),
        "all_gaps_one": bool(gaps) and all(gap == 1 for gap in gaps),
        "large_middle": any(size >= 4 for size in packet_sizes[1:-1]),
    }


def _singleton_change(parent: bool, child: bool) -> str:
    if parent and child:
        return "preserved"
    if parent and not child:
        return "removed"
    if not parent and child:
        return "created"
    return "none"


def _gap_role(parent_gaps: tuple[int, ...], child_gaps: tuple[int, ...]) -> str:
    parent_large = any(gap >= 2 for gap in parent_gaps)
    child_large = any(gap >= 2 for gap in child_gaps)
    if parent_large and not child_large:
        return "closes_large_gap"
    if not parent_large and child_large:
        return "creates_large_gap"
    if parent_gaps and child_gaps and all(gap == 1 for gap in parent_gaps) and all(gap == 1 for gap in child_gaps):
        return "keeps_all_unit_gaps"
    if parent_gaps and all(gap == 1 for gap in parent_gaps) and child_gaps and any(gap != 1 for gap in child_gaps):
        return "fragments_unit_gap_chain"
    return "other"


def _size_flow_role(parent_packets: tuple[int, ...], child_packets: tuple[int, ...]) -> str:
    parent_middle_max = max(parent_packets[1:-1], default=0)
    child_middle_max = max(child_packets[1:-1], default=0)
    if parent_middle_max >= 4 and child_middle_max < parent_middle_max:
        return "splits_large_middle"
    if child_middle_max > parent_middle_max:
        return "grows_middle_packet"
    if child_packets and parent_packets and child_packets[0] > parent_packets[0]:
        return "grows_top_packet"
    if child_packets and parent_packets and child_packets[-1] > parent_packets[-1]:
        return "grows_bottom_packet"
    if child_packets.count(1) > parent_packets.count(1):
        return "creates_new_singletons"
    if len(child_packets) == 2 and len(parent_packets) != 2:
        return "merges_to_two_packet"
    return "other"


def _balance_role(parent_packets: tuple[int, ...], child_packets: tuple[int, ...]) -> str:
    if len(child_packets) > len(parent_packets):
        return "more_fragmented"
    if len(child_packets) < len(parent_packets):
        return "less_fragmented"
    return "same_packet_count"


def _touched_role(feature: RuleMiningActionFeature, packet_count: int) -> str:
    touched = set(feature.touched_packet_indices)
    if len(touched) == 1:
        return "touches_single_packet"
    top = 0 in touched
    bottom = packet_count > 0 and packet_count - 1 in touched
    middle = any(0 < index < packet_count - 1 for index in touched)
    if top and middle and bottom:
        return "touches_all_regions"
    if top and middle:
        return "touches_top_and_middle"
    if middle and bottom:
        return "touches_middle_and_bottom"
    if top:
        return "touches_top"
    if bottom:
        return "touches_bottom"
    if middle:
        return "touches_middle"
    return "touches_none"


def _transition_role_label(
    parent_packets: tuple[int, ...],
    child_packets: tuple[int, ...],
    gap_role: str,
    size_flow_role: str,
    balance_role: str,
) -> str:
    if parent_packets == (9,) and child_packets == (4, 5):
        return "root-to-first-split"
    if balance_role == "more_fragmented" and child_packets and child_packets[0] == 1:
        return "bridge-to-singleton-frontier"
    if size_flow_role == "splits_large_middle":
        return "split-large-middle"
    if balance_role == "less_fragmented":
        return "coarsen-frontier"
    if gap_role == "closes_large_gap":
        return "close-gap"
    if gap_role == "creates_large_gap":
        return "create-gap"
    if balance_role == "more_fragmented":
        return "fragment-frontier"
    return "preserve-shape"


def _flow_role_action_from_feature(row: RuleMiningRow, feature: RuleMiningActionFeature) -> FlowRoleAction:
    child_state = canon(tuple(row.state[index] + feature.action[index] for index in range(len(row.state))))
    child_packets = packet_type(child_state)
    child_gaps = _packet_gaps(child_state)
    parent_features = _packet_feature_summary(row.packet_type, row.packet_gaps)
    child_features = _packet_feature_summary(child_packets, child_gaps)
    packet_count_delta = int(child_features["packet_count"]) - int(parent_features["packet_count"])
    top_singleton_change = _singleton_change(
        bool(parent_features["top_singleton"]),
        bool(child_features["top_singleton"]),
    )
    bottom_singleton_change = _singleton_change(
        bool(parent_features["bottom_singleton"]),
        bool(child_features["bottom_singleton"]),
    )
    gap_role = _gap_role(row.packet_gaps, child_gaps)
    size_flow_role = _size_flow_role(row.packet_type, child_packets)
    balance_role = _balance_role(row.packet_type, child_packets)
    touched_role = _touched_role(feature, len(row.packet_type))
    transition_role_label = _transition_role_label(
        row.packet_type,
        child_packets,
        gap_role,
        size_flow_role,
        balance_role,
    )
    return FlowRoleAction(
        weight=feature.weight,
        action=feature.action,
        child_state=child_state,
        child_packet_type=child_packets,
        child_packet_gaps=child_gaps,
        role_signature=(
            transition_role_label,
            packet_count_delta,
            top_singleton_change,
            bottom_singleton_change,
            gap_role,
            size_flow_role,
            balance_role,
            touched_role,
        ),
    )


def _flow_role_row_from_rule_row(row: RuleMiningRow) -> FlowRoleRow:
    actions = tuple(_flow_role_action_from_feature(row, feature) for feature in row.action_features)
    role_weights: dict[str, float] = defaultdict(float)
    for action in actions:
        role_weights[action.role_signature[0]] += action.weight
    role_distribution = tuple(
        (role, _format_fractionish(weight))
        for role, weight in sorted(role_weights.items(), key=lambda item: (-item[1], item[0]))
    )
    return FlowRoleRow(
        source=row,
        exact_successor_distribution=_successor_distribution_signature(row.successor_packet_type_weights),
        role_distribution=role_distribution,
        actions=actions,
    )


def two_run_flow_role_rows(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[FlowRoleRow, ...]:
    return tuple(
        _flow_role_row_from_rule_row(row)
        for row in two_run_replay_rule_mining_rows(k, T, support_tolerance=support_tolerance)
    )


def _group_flow_role_rows(
    rows: tuple[FlowRoleRow, ...],
    key_fn,
) -> tuple[tuple[object, list[FlowRoleRow]], ...]:
    grouped: dict[object, list[FlowRoleRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return tuple(
        sorted(
            grouped.items(),
            key=lambda item: (
                -sum(row.source.occupancy_probability for row in item[1]),
                -len(item[1]),
                str(item[0]),
            ),
        )
    )


def _format_role_distribution(
    role_distribution: tuple[tuple[str, str], ...],
) -> str:
    return "; ".join(f"{role}:{weight}" for role, weight in role_distribution)


def _print_flow_role_support(row: FlowRoleRow, support_n: int = 8) -> None:
    for action in row.actions[:support_n]:
        print(
            f"       w={_format_fractionish(action.weight):>8s}"
            f" action={_format_action(action.action)}"
            f" child={action.child_packet_type}"
            f" child_gaps={action.child_packet_gaps}"
            f" role={action.role_signature}"
        )
    if len(row.actions) > support_n:
        print("       ...")


def print_two_run_flow_role_report(
    k: int,
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 100,
) -> None:
    rows = two_run_flow_role_rows(k, T, support_tolerance=support_tolerance)
    exact_successor_distributions = {row.exact_successor_distribution for row in rows}
    role_distributions = {row.role_distribution for row in rows}
    rule_role_distributions = {(row.source.rule_hint, row.role_distribution) for row in rows}
    rule_weight_role_distributions = {
        (row.source.rule_hint, row.source.weight_multiset, row.role_distribution)
        for row in rows
    }

    print(f"Two-run flow-role report, k={k}, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print("Summary counts:")
    print(f"  total replay states: {len(rows)}")
    print(f"  distinct exact successor distributions: {len(exact_successor_distributions)}")
    print(f"  distinct role distributions: {len(role_distributions)}")
    print(f"  distinct rule_hint + role_distribution: {len(rule_role_distributions)}")
    print(
        "  distinct rule_hint + weight_multiset + role_distribution:"
        f" {len(rule_weight_role_distributions)}"
    )
    exact_count = len(exact_successor_distributions) or 1
    print(f"  compression ratio role/exact: {len(role_distributions) / exact_count:.3f}")
    print()

    print("Top role distributions by occupancy:")
    for index, (signature, group) in enumerate(
        _group_flow_role_rows(rows, lambda row: row.role_distribution)[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.source.occupancy_probability)
        print(
            f"  {index}. rows={len(group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.source.remaining_horizon for row in group])}"
            f" hints={_histogram_tuple([row.source.rule_hint for row in group])}"
        )
        print(f"     packet regimes={_histogram_tuple([(row.source.packet_type, row.source.packet_gaps) for row in group])[:6]}")
        print(f"     weight multisets={_histogram_tuple([row.source.weight_multiset for row in group])[:6]}")
        print(f"     role_distribution={_format_role_distribution(signature)}")
        print(
            f"     representative rem={representative.source.remaining_horizon}"
            f" state={representative.source.state}"
            f" packet={representative.source.packet_type}"
            f" gaps={representative.source.packet_gaps}"
        )
        _print_flow_role_support(representative)
    print()

    print("Per rule-hint compression:")
    for hint, group in _group_flow_role_rows(rows, lambda row: row.source.rule_hint):
        exact_count_for_hint = len({row.exact_successor_distribution for row in group})
        role_groups = _group_flow_role_rows(tuple(group), lambda row: row.role_distribution)
        weight_totals: dict[tuple[str, ...], float] = defaultdict(float)
        for row in group:
            weight_totals[row.source.weight_multiset] += row.source.occupancy_probability
        total_occupancy = sum(row.source.occupancy_probability for row in group)
        dominant_weights, dominant_weight_occupancy = max(weight_totals.items(), key=lambda item: (item[1], item[0]))
        print(
            f"  {hint}: rows={len(group)}"
            f" occupancy={total_occupancy:.6f}"
            f" exact_successors={exact_count_for_hint}"
            f" role_distributions={len(role_groups)}"
            f" dominant_weights={_format_weight_multiset(dominant_weights)}"
            f" dominant_weight_share={dominant_weight_occupancy / total_occupancy if total_occupancy else 0.0:.3f}"
            f" compressible={len(role_groups) <= 5}"
        )
        for role_distribution, role_group in role_groups[: min(n, 8)]:
            print(
                f"     rows={len(role_group)}"
                f" occupancy={sum(row.source.occupancy_probability for row in role_group):.6f}"
                f" roles={_format_role_distribution(role_distribution)}"
            )
    print()

    print("Early basin role chain (occupancy >= 0.1):")
    for row in sorted(
        (row for row in rows if row.source.occupancy_probability >= 0.1),
        key=lambda item: (-item.source.remaining_horizon, item.source.state),
    )[:n]:
        print(
            f"  rem={row.source.remaining_horizon}"
            f" state={row.source.state}"
            f" packet={row.source.packet_type} gaps={row.source.packet_gaps}"
            f" hint={row.source.rule_hint}"
            f" occ={row.source.occupancy_probability:.6f}"
        )
        print(f"     exact_successor={_format_successor_distribution(row.exact_successor_distribution)}")
        print(f"     role_distribution={_format_role_distribution(row.role_distribution)}")
    print()

    print("Late cleanup role compression, rem=1 and rem=2:")
    late_rows = tuple(row for row in rows if row.source.remaining_horizon in {1, 2})
    for index, (signature, group) in enumerate(
        _group_flow_role_rows(
            late_rows,
            lambda row: (row.source.rule_hint, row.source.weight_multiset, row.role_distribution),
        )[:n],
        start=1,
    ):
        representative = max(group, key=lambda item: item.source.occupancy_probability)
        print(
            f"  {index}. rows={len(group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
            f" horizons={_histogram_tuple([row.source.remaining_horizon for row in group])}"
            f" signature={_format_coarse_signature(signature)}"
        )
        print(
            f"     representative state={representative.source.state}"
            f" packet={representative.source.packet_type}"
            f" gaps={representative.source.packet_gaps}"
        )
    print()

    print("Candidate v4 rule inventory:")
    for hint, group in _group_flow_role_rows(rows, lambda row: row.source.rule_hint):
        role_groups = _group_flow_role_rows(tuple(group), lambda row: row.role_distribution)
        parent_stats = _histogram_tuple(
            [
                (
                    row.source.packet_count,
                    row.source.top_packet_size,
                    row.source.bottom_packet_size,
                    row.source.max_gap,
                    row.source.unit_gap_count,
                    row.source.large_gap_count,
                    row.source.has_large_middle_packet,
                )
                for row in group
            ]
        )
        print(
            f"  {hint}:"
            f" role_distributions={len(role_groups)}"
            f" rows={len(group)}"
            f" occupancy={sum(row.source.occupancy_probability for row in group):.6f}"
            f" ready_for_hand_policy={len(role_groups) <= 5}"
        )
        print(f"     common parent features={parent_stats[:5]}")
        print(f"     top role distributions:")
        for role_distribution, role_group in role_groups[:5]:
            print(
                f"       rows={len(role_group)}"
                f" occupancy={sum(row.source.occupancy_probability for row in role_group):.6f}"
                f" roles={_format_role_distribution(role_distribution)}"
            )


def _alive_positions(state: tuple[int, ...], remaining_horizon: int) -> tuple[int, ...]:
    if not state:
        return ()
    max_score = state[0]
    return tuple(index for index, score in enumerate(state) if max_score - score <= remaining_horizon)


def _alive_prefix_length(
    state: tuple[int, ...],
    remaining_horizon: int,
    death_mode: str = "weak",
) -> int:
    if not state:
        return 0
    if death_mode not in {"weak", "strict"}:
        raise ValueError(f"unknown death mode: {death_mode}")

    max_score = state[0]
    alive_count = 0
    for score in state:
        deficit = max_score - score
        if (death_mode == "weak" and deficit <= remaining_horizon) or (
            death_mode == "strict" and deficit < remaining_horizon
        ):
            alive_count += 1
        else:
            break
    return alive_count


def _lift_alive_pair_to_k(
    k: int,
    alive_count: int,
    base_action: tuple[int, ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    if alive_count != len(base_action):
        raise ValueError("base action length must match alive_count")
    if alive_count > k:
        raise ValueError("alive_count cannot exceed k")
    if alive_count <= 1:
        return ((1.0, tuple(0 for _ in range(k))),)

    action = base_action + tuple(0 for _ in range(k - alive_count))
    alive_complement = tuple(1 - bit for bit in base_action) + tuple(0 for _ in range(k - alive_count))
    return _policy_from_weighted_actions(((0.5, action), (0.5, alive_complement)))


def _balanced_policy_from_one_run_interval(
    k: int,
    interval: tuple[int, int],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    start, end = interval
    if k < 2:
        return ((1.0, tuple(0 for _ in range(k))),)
    if start < 0 or end < start or end >= k - 1:
        raise ValueError(f"invalid one-run interval {interval} for k={k}")
    edge_signature = tuple(1 if start <= edge_index <= end else 0 for edge_index in range(k - 1))
    action = _action_from_edge_signature(edge_signature)
    return _policy_from_weighted_actions(((0.5, action), (0.5, complement(action))))


def _mixed_interval_policy(
    k: int,
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    terms: list[tuple[float, tuple[int, ...]]] = []
    for interval_weight, interval in weighted_intervals:
        for action_probability, action in _balanced_policy_from_one_run_interval(k, interval):
            terms.append((interval_weight * action_probability, action))
    return _policy_from_weighted_actions(tuple(terms))


def _one_run_interval_distribution_for_regime(
    k: int,
    T: int,
    remaining_horizon: int,
    ptype: tuple[int, ...],
    gaps: tuple[int, ...],
    support_tolerance: float = 1e-8,
) -> tuple[tuple[float, tuple[int, int]], ...]:
    totals: dict[tuple[int, int], float] = defaultdict(float)
    for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance):
        if (
            row.remaining_horizon != remaining_horizon
            or row.packet_type != ptype
            or row.packet_gaps != gaps
        ):
            continue
        for support in row.support:
            if support.interval is not None:
                totals[support.interval] += row.occupancy_probability * support.weight
    total = sum(totals.values())
    if total <= 0.0:
        return ()
    return tuple(
        (weight / total, interval)
        for interval, weight in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    )


def _one_run_interval_distributions_by_regime(
    T: int,
    support_tolerance: float = 1e-8,
) -> dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[tuple[float, tuple[int, int]], ...]]:
    totals_by_regime: dict[tuple[int, tuple[int, ...], tuple[int, ...]], dict[tuple[int, int], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance):
        key = (row.remaining_horizon, row.packet_type, row.packet_gaps)
        for support in row.support:
            if support.interval is not None:
                totals_by_regime[key][support.interval] += row.occupancy_probability * support.weight
    distributions: dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[tuple[float, tuple[int, int]], ...]] = {}
    for key, totals in totals_by_regime.items():
        total = sum(totals.values())
        if total <= 0.0:
            continue
        distributions[key] = tuple(
            (weight / total, interval)
            for interval, weight in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        )
    return distributions


def _mixed_one_run_interval_policy(
    k: int,
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _mixed_interval_policy(k, weighted_intervals)


def _effective_alive_pattern(alive_count: int, variant: str) -> tuple[int, ...]:
    patterns_by_variant: dict[str, dict[int, tuple[int, ...]]] = {
        "trunc6_chase5_comb_tail": {
            6: (1, 0, 1, 0, 1, 1),
            5: (1, 0, 1, 0, 0),
            4: (1, 0, 1, 0),
            3: (1, 0, 1),
            2: (1, 0),
        },
        "trunc6_chase5_twin3_tail": {
            6: (1, 0, 1, 0, 1, 1),
            5: (1, 0, 1, 0, 0),
            4: (1, 0, 1, 0),
            3: (1, 0, 0),
            2: (1, 0),
        },
        "fullcomb6_chase5_comb_tail": {
            6: (1, 0, 1, 0, 1, 0),
            5: (1, 0, 1, 0, 0),
            4: (1, 0, 1, 0),
            3: (1, 0, 1),
            2: (1, 0),
        },
    }
    if variant not in patterns_by_variant:
        raise ValueError(f"unknown alive-pattern variant: {variant}")
    if alive_count <= 1:
        return tuple(0 for _ in range(max(alive_count, 0)))
    try:
        return patterns_by_variant[variant][alive_count]
    except KeyError as error:
        raise ValueError(f"unsupported alive_count={alive_count} for variant={variant}") from error


def _k6_alive_policy_metadata(
    state: tuple[int, ...],
    remaining_horizon: int,
    variant: str,
    death_mode: str = "weak",
) -> tuple[int, tuple[int, ...], str]:
    canonical_state = canon(state)
    alive_count = _alive_prefix_length(canonical_state, remaining_horizon, death_mode=death_mode)
    base_action = _effective_alive_pattern(alive_count, variant)
    pattern_label = "dead_all_zero" if alive_count <= 1 else _format_action(base_action)
    return alive_count, base_action, pattern_label


def _k6_alive_candidate_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
    variant: str,
    death_mode: str = "weak",
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    k = len(state)
    if k != 6:
        return top_prefix_three_regime_v6_policy(canon(state))
    alive_count, base_action, _ = _k6_alive_policy_metadata(
        state,
        remaining_horizon,
        variant,
        death_mode=death_mode,
    )
    return _lift_alive_pair_to_k(6, alive_count, base_action)


def k6_alive_trunc_comb_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(state, remaining_horizon, "trunc6_chase5_comb_tail")


def k6_alive_trunc_comb_twin3_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(state, remaining_horizon, "trunc6_chase5_twin3_tail")


def k6_alive_full_comb_control_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(state, remaining_horizon, "fullcomb6_chase5_comb_tail")


def k6_alive_trunc_comb_strict_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(
        state,
        remaining_horizon,
        "trunc6_chase5_comb_tail",
        death_mode="strict",
    )


def k6_alive_trunc_comb_twin3_strict_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(
        state,
        remaining_horizon,
        "trunc6_chase5_twin3_tail",
        death_mode="strict",
    )


def k6_alive_full_comb_control_strict_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_alive_candidate_policy(
        state,
        remaining_horizon,
        "fullcomb6_chase5_comb_tail",
        death_mode="strict",
    )


def _all_gaps_are_one(gaps: tuple[int, ...]) -> bool:
    return bool(gaps) and all(gap == 1 for gap in gaps)


def _k6_one_run_rule_v1_interval_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
    *,
    use_mixed_late: bool,
    aggressive_shoulder: bool,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    if len(canonical_state) != 6:
        return tuple((float(probability), action) for probability, action in top_prefix_three_regime_v6_policy(canonical_state))

    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    alive_count = _alive_prefix_length(canonical_state, remaining_horizon)
    all_unit_gaps = _all_gaps_are_one(gaps)

    def interval_policy(interval: tuple[int, int]) -> tuple[tuple[float, tuple[int, ...]], ...]:
        return _balanced_policy_from_one_run_interval(6, interval)

    if ptype in {(6,), (3, 3), (2, 2, 2), (2, 3, 1)} and (not gaps or all_unit_gaps):
        return interval_policy((0, 4))

    shoulder_packets = {
        (1, 4, 1),
        (1, 2, 2, 1),
        (1, 3, 1, 1),
        (1, 2, 3),
        (1, 3, 2),
    }
    if all_unit_gaps and ptype in shoulder_packets:
        return interval_policy((1, 1))
    if (
        aggressive_shoulder
        and all_unit_gaps
        and len(ptype) >= 3
        and ptype[0] == 1
        and ptype[1] >= 2
    ):
        return interval_policy((1, 1))
    if (
        not aggressive_shoulder
        and all_unit_gaps
        and len(ptype) >= 3
        and ptype[0] == 1
        and ptype[1] >= 2
        and not (len(ptype) >= 3 and ptype[:3] == (1, 1, 1))
    ):
        return interval_policy((1, 1))

    if all_unit_gaps and ptype in {(1, 1, 1, 2, 1), (1, 1, 1, 3)}:
        return interval_policy((0, 2))

    if use_mixed_late and ptype in {(1, 2, 2, 1), (1, 2, 3)} and gaps and max(gaps) >= 2:
        return _mixed_interval_policy(
            6,
            (
                (1.0 / 3.0, (0, 0)),
                (1.0 / 3.0, (0, 4)),
                (1.0 / 3.0, (1, 4)),
            ),
        )

    if alive_count == 1 and remaining_horizon <= 3:
        return _mixed_interval_policy(
            6,
            (
                (1.0 / 3.0, (1, 2)),
                (1.0 / 3.0, (3, 4)),
                (1.0 / 3.0, (1, 4)),
            ),
        )

    if remaining_horizon <= 3 and canonical_state[0] - canonical_state[-1] >= 3:
        return interval_policy((1, 4))

    return interval_policy((0, 4))


def k6_one_run_rule_v1_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_one_run_rule_v1_interval_policy(
        state,
        remaining_horizon,
        use_mixed_late=True,
        aggressive_shoulder=False,
    )


def k6_one_run_rule_v1_no_mixed_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_one_run_rule_v1_interval_policy(
        state,
        remaining_horizon,
        use_mixed_late=False,
        aggressive_shoulder=False,
    )


def k6_one_run_rule_v1_aggressive_shoulder_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_one_run_rule_v1_interval_policy(
        state,
        remaining_horizon,
        use_mixed_late=True,
        aggressive_shoulder=True,
    )


def _k6_one_run_patch_list() -> tuple[K6OneRunPatch, ...]:
    return (
        K6OneRunPatch("p_rem10_141_g11_i11", 10, (1, 4, 1), (1, 1), (1, 1)),
        K6OneRunPatch("p_rem9_1221_g111_i11", 9, (1, 2, 2, 1), (1, 1, 1), (1, 1)),
        K6OneRunPatch("p_rem8_11121_g1111_i02", 8, (1, 1, 1, 2, 1), (1, 1, 1, 1), (0, 2)),
        K6OneRunPatch("p_rem7_1311_g111_i11", 7, (1, 3, 1, 1), (1, 1, 1), (1, 1)),
        K6OneRunPatch("p_rem7_123_g11_i11", 7, (1, 2, 3), (1, 1), (1, 1)),
        K6OneRunPatch("p_rem6_1113_g111_i02", 6, (1, 1, 1, 3), (1, 1, 1), (0, 2)),
    )


def _k6_one_run_patch_by_name() -> dict[str, K6OneRunPatch]:
    return {patch.name: patch for patch in _k6_one_run_patch_list()}


def _matching_k6_patch(
    state: tuple[int, ...],
    remaining_horizon: int,
    patches: tuple[K6OneRunPatch, ...],
) -> K6OneRunPatch | None:
    canonical_state = canon(state)
    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    for patch in patches:
        if (
            patch.remaining_horizon == remaining_horizon
            and patch.packet_type == ptype
            and patch.packet_gaps == gaps
        ):
            return patch
    return None


def make_k6_one_run_patch_policy(
    base_policy_name: str,
    patch_names: tuple[str, ...],
):
    patches_by_name = _k6_one_run_patch_by_name()
    patches = tuple(patches_by_name[name] for name in patch_names)
    return make_k6_one_run_patch_policy_from_patches(base_policy_name, patches)


def make_k6_one_run_patch_policy_from_patches(
    base_policy_name: str,
    patches: tuple[K6OneRunPatch, ...],
):
    base_policy = _time_dependent_fallback_policy(base_policy_name)

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        if len(canonical_state) == 6:
            patch = _matching_k6_patch(canonical_state, remaining_horizon, patches)
            if patch is not None:
                return _balanced_policy_from_one_run_interval(6, patch.interval)
        return base_policy(canonical_state, remaining_horizon)

    return policy


def _k6_patch_ladder_policy_specs() -> tuple[tuple[str, tuple[str, ...]], ...]:
    patch_names = tuple(patch.name for patch in _k6_one_run_patch_list())
    return tuple(
        (f"k6_patch_ladder_{count}" + ("_base" if count == 0 else ""), patch_names[:count])
        for count in range(len(patch_names) + 1)
    )


def k6_greedy_deterministic_patched_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    deterministic_patches = _k6_one_run_patch_list()[:5] + (
        K6OneRunPatch(
            name="g_rem5_1221_g112_i11",
            remaining_horizon=5,
            packet_type=(1, 2, 2, 1),
            packet_gaps=(1, 1, 2),
            interval=(1, 1),
        ),
    )
    policy = make_k6_one_run_patch_policy_from_patches(
        "k6_alive_full_comb_control_policy",
        deterministic_patches,
    )
    return policy(state, remaining_horizon)


def _top_prefix_all_reference_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    del remaining_horizon
    k = len(state)
    if k <= 1:
        return ((1.0, tuple(0 for _ in range(k))),)
    return _balanced_top_prefix_policy(k, k - 1)


def _fixed_balanced_action_policy_from_text(action_text: str):
    action = tuple(int(char) for char in action_text.strip())
    if any(bit not in (0, 1) for bit in action):
        raise ValueError(f"invalid fixed action text: {action_text!r}")

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        del remaining_horizon
        if len(state) != len(action):
            raise ValueError(f"fixed action length {len(action)} does not match state length {len(state)}")
        return _policy_from_weighted_actions(((0.5, action), (0.5, complement(action))))

    return policy


fixed_101010_comb_policy = _fixed_balanced_action_policy_from_text("101010")
fixed_101011_truncated_policy = _fixed_balanced_action_policy_from_text("101011")
fixed_101000_prefix3_policy = _fixed_balanced_action_policy_from_text("101000")
fixed_100000_singleton_policy = _fixed_balanced_action_policy_from_text("100000")
fixed_10101_comb_policy = _fixed_balanced_action_policy_from_text("10101")


def _k6_barycentric_weights(variant: str) -> tuple[tuple[float, tuple[int, int]], ...]:
    if variant == "equal":
        return ((1.0 / 3.0, (0, 0)), (1.0 / 3.0, (0, 4)), (1.0 / 3.0, (1, 4)))
    if variant == "tail_heavy":
        return ((0.25, (0, 0)), (0.25, (0, 4)), (0.5, (1, 4)))
    if variant == "top_tail":
        return ((0.5, (0, 0)), (0.5, (1, 4)))
    raise ValueError(f"unknown k=6 barycentric variant: {variant}")


def _k6_barycentric_trigger(
    state: tuple[int, ...],
    remaining_horizon: int,
    mode: str,
) -> bool:
    canonical_state = canon(state)
    if len(canonical_state) != 6:
        return False
    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    if len(ptype) <= 2 and (not gaps or max(gaps) <= 1):
        return False
    if mode == "late" and remaining_horizon > 6:
        return False
    if mode == "middle" and not (3 <= remaining_horizon <= 8):
        return False
    mean_score = sum(canonical_state) / 6.0
    above_count = sum(1 for score in canonical_state if score > mean_score)
    below_count = sum(1 for score in canonical_state if score < mean_score)
    top_gap = canonical_state[0] - mean_score
    bottom_gap = mean_score - canonical_state[-1]
    max_gap = max(gaps, default=0)
    return (
        above_count >= 1
        and below_count >= 2
        and bottom_gap >= 0.5
        and top_gap >= 0.5
        and (len(ptype) >= 4 or max_gap >= 2)
    )


def _k6_barycentric_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
    mode: str,
    variant: str,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    canonical_state = canon(state)
    if len(canonical_state) != 6:
        return _balanced_policy_from_one_run_interval(len(canonical_state), (0, max(0, len(canonical_state) - 2)))
    if _k6_barycentric_trigger(canonical_state, remaining_horizon, mode):
        return _mixed_one_run_interval_policy(6, _k6_barycentric_weights(variant))
    return _balanced_policy_from_one_run_interval(6, (0, 4))


def k6_barycentric_v1_equal_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_barycentric_policy(state, remaining_horizon, "all", "equal")


def k6_barycentric_v1_late_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_barycentric_policy(state, remaining_horizon, "late", "equal")


def k6_barycentric_v1_middle_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_barycentric_policy(state, remaining_horizon, "middle", "equal")


def k6_barycentric_v1_tail_heavy_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_barycentric_policy(state, remaining_horizon, "all", "tail_heavy")


def k6_barycentric_v1_top_tail_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    return _k6_barycentric_policy(state, remaining_horizon, "all", "top_tail")


def _k6_barycentric_policy_specs() -> dict[str, tuple[str, str]]:
    return {
        "k6_barycentric_v1_equal": ("all", "equal"),
        "k6_barycentric_v1_late": ("late", "equal"),
        "k6_barycentric_v1_middle": ("middle", "equal"),
        "k6_barycentric_v1_tail_heavy": ("all", "tail_heavy"),
        "k6_barycentric_v1_top_tail": ("all", "top_tail"),
    }


def _k6_barycentric_default_policy_names() -> tuple[str, ...]:
    return (
        "fixed_101010_comb",
        "fixed_101011_truncated",
        "fixed_101000_prefix3",
        "k6_barycentric_v1_equal",
        "k6_barycentric_v1_late",
        "k6_barycentric_v1_middle",
        "k6_barycentric_v1_tail_heavy",
        "k6_barycentric_v1_top_tail",
        "k6_alive_full_comb_control_policy",
        "k6_alive_trunc_comb_policy",
    )


def _k6_barycentric_sweep_mixtures() -> dict[str, tuple[tuple[float, tuple[int, int]], ...]]:
    return {
        "equal_top_full_tail": ((1.0 / 3.0, (0, 0)), (1.0 / 3.0, (0, 4)), (1.0 / 3.0, (1, 4))),
        "tail_heavy": ((0.25, (0, 0)), (0.25, (0, 4)), (0.5, (1, 4))),
        "top_tail_only": ((0.5, (0, 0)), (0.5, (1, 4))),
        "conservative_tail": ((0.5, (0, 4)), (0.5, (1, 4))),
    }


def _k6_barycentric_trigger_features(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> dict[str, object]:
    canonical_state = canon(state)
    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    mean_score = sum(canonical_state) / len(canonical_state) if canonical_state else 0.0
    return {
        "state": canonical_state,
        "remaining_horizon": remaining_horizon,
        "packet_type": ptype,
        "packet_gaps": gaps,
        "packet_count": len(ptype),
        "max_gap": max(gaps, default=0),
        "num_gaps_eq_1": sum(1 for gap in gaps if gap == 1),
        "num_gaps_ge_2": sum(1 for gap in gaps if gap >= 2),
        "top_packet_size": ptype[0] if ptype else 0,
        "second_packet_size": ptype[1] if len(ptype) > 1 else 0,
        "bottom_packet_size": ptype[-1] if ptype else 0,
        "mean_score": mean_score,
        "top_gap": canonical_state[0] - mean_score if canonical_state else 0.0,
        "bottom_gap": mean_score - canonical_state[-1] if canonical_state else 0.0,
        "above_count": sum(1 for score in canonical_state if score > mean_score),
        "below_count": sum(1 for score in canonical_state if score < mean_score),
    }


def _k6_barycentric_trigger_sweep_specs() -> tuple[K6BarycentricTriggerSpec, ...]:
    specs: list[K6BarycentricTriggerSpec] = [
        K6BarycentricTriggerSpec(
            "top_singleton",
            "packet_type[0] == 1",
            lambda f: f["top_packet_size"] == 1,
        ),
        K6BarycentricTriggerSpec(
            "top_singleton_second_pair",
            "packet_type[0] == 1 and packet_type[1] == 2",
            lambda f: f["top_packet_size"] == 1 and f["second_packet_size"] == 2,
        ),
        K6BarycentricTriggerSpec(
            "top_singleton_fragmented_tail",
            "packet_type[0] == 1 and packet_count >= 4",
            lambda f: f["top_packet_size"] == 1 and f["packet_count"] >= 4,
        ),
        K6BarycentricTriggerSpec(
            "one_run_correction_shape",
            "packet_type[0] == 1 and packet_type[1] == 2 and packet_count >= 3",
            lambda f: f["top_packet_size"] == 1 and f["second_packet_size"] == 2 and f["packet_count"] >= 3,
        ),
        K6BarycentricTriggerSpec(
            "gap2_top_singleton",
            "packet_type[0] == 1 and max_gap >= 2",
            lambda f: f["top_packet_size"] == 1 and f["max_gap"] >= 2,
        ),
    ]
    for threshold in (0.5, 1.0, 1.5):
        specs.append(
            K6BarycentricTriggerSpec(
                f"barycenter_split_t{threshold:g}",
                f"above_count >= 1 and below_count >= 2 and top_gap,bottom_gap >= {threshold:g}",
                lambda f, threshold=threshold: (
                    f["above_count"] >= 1
                    and f["below_count"] >= 2
                    and f["top_gap"] >= threshold
                    and f["bottom_gap"] >= threshold
                ),
            )
        )
    for packet_threshold in (3, 4, 5):
        specs.append(
            K6BarycentricTriggerSpec(
                f"fragmented_barycenter_p{packet_threshold}",
                f"packet_count >= {packet_threshold} and above_count >= 1 and below_count >= 2",
                lambda f, packet_threshold=packet_threshold: (
                    f["packet_count"] >= packet_threshold and f["above_count"] >= 1 and f["below_count"] >= 2
                ),
            )
        )
    for horizon_threshold in (2, 3, 4, 5, 6, 8):
        specs.append(
            K6BarycentricTriggerSpec(
                f"late_top_singleton_h{horizon_threshold}",
                f"packet_type[0] == 1 and h <= {horizon_threshold}",
                lambda f, horizon_threshold=horizon_threshold: (
                    f["top_packet_size"] == 1 and f["remaining_horizon"] <= horizon_threshold
                ),
            )
        )
    for low, high in ((2, 6), (3, 8), (4, 10), (5, 12)):
        specs.append(
            K6BarycentricTriggerSpec(
                f"middle_top_singleton_h{low}_{high}",
                f"packet_type[0] == 1 and {low} <= h <= {high}",
                lambda f, low=low, high=high: (
                    f["top_packet_size"] == 1 and low <= f["remaining_horizon"] <= high
                ),
            )
        )
    for low, high in ((2, 8), (3, 8), (4, 10)):
        specs.append(
            K6BarycentricTriggerSpec(
                f"one_run_report_like_h{low}_{high}",
                (
                    "packet_type[0] == 1 and packet_type[1] == 2 and "
                    f"(max_gap >= 2 or packet_count >= 4) and {low} <= h <= {high}"
                ),
                lambda f, low=low, high=high: (
                    f["top_packet_size"] == 1
                    and f["second_packet_size"] == 2
                    and (f["max_gap"] >= 2 or f["packet_count"] >= 4)
                    and low <= f["remaining_horizon"] <= high
                ),
            )
        )
    return tuple(specs)


def _k6_barycentric_trigger_sweep_candidates() -> tuple[K6BarycentricTriggerSweepCandidate, ...]:
    candidates: list[K6BarycentricTriggerSweepCandidate] = []
    for spec in _k6_barycentric_trigger_sweep_specs():
        for mixture_name, weighted_intervals in _k6_barycentric_sweep_mixtures().items():
            candidates.append(
                K6BarycentricTriggerSweepCandidate(
                    name=f"{spec.name}__{mixture_name}",
                    trigger_name=spec.name,
                    trigger_description=spec.description,
                    mixture_name=mixture_name,
                    weighted_intervals=weighted_intervals,
                    predicate=spec.predicate,
                )
            )
    return tuple(candidates)


def _k6_barycentric_trigger_fires(
    candidate: K6BarycentricTriggerSweepCandidate,
    state: tuple[int, ...],
    remaining_horizon: int,
) -> bool:
    canonical_state = canon(state)
    if len(canonical_state) != 6:
        return False
    features = _k6_barycentric_trigger_features(canonical_state, remaining_horizon)
    return bool(candidate.predicate(features))


def make_k6_barycentric_trigger_sweep_policy(candidate: K6BarycentricTriggerSweepCandidate):
    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        if len(canonical_state) != 6:
            return _balanced_policy_from_one_run_interval(len(canonical_state), (0, max(0, len(canonical_state) - 2)))
        if _k6_barycentric_trigger_fires(candidate, canonical_state, remaining_horizon):
            return _mixed_one_run_interval_policy(6, candidate.weighted_intervals)
        return _balanced_policy_from_one_run_interval(6, (0, 4))

    return policy


def _centered_state_values(state: tuple[int, ...]) -> tuple[float, ...]:
    if not state:
        return ()
    mean_score = sum(state) / len(state)
    return tuple(score - mean_score for score in state)


def _phi_centered_max(state: tuple[int, ...]) -> float:
    if not state:
        return 0.0
    mean_score = sum(state) / len(state)
    return max(state) - mean_score


def _phi_centered_softmax(state: tuple[int, ...], tau: float) -> float:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    centered = _centered_state_values(state)
    if not centered:
        return 0.0
    max_centered = max(centered)
    return tau * (max_centered / tau + np.log(sum(np.exp((value - max_centered) / tau) for value in centered)))


def _softmax_centered_gradient(state: tuple[int, ...], tau: float) -> tuple[float, ...]:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    centered = _centered_state_values(state)
    if not centered:
        return ()
    max_centered = max(centered)
    weights = [float(np.exp((value - max_centered) / tau)) for value in centered]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _k5_comb_support() -> tuple[tuple[float, tuple[int, ...]], ...]:
    return ((0.5, (1, 0, 1, 0, 1)), (0.5, (0, 1, 0, 1, 0)))


def _k5_chase_support() -> tuple[tuple[float, tuple[int, ...]], ...]:
    return ((0.5, (1, 0, 1, 0, 0)), (0.5, (0, 1, 0, 1, 1)))


def _expected_potential_delta(
    state: tuple[int, ...],
    support: tuple[tuple[float, tuple[int, ...]], ...],
    potential_fn,
) -> float:
    base = potential_fn(state)
    expected = 0.0
    for probability, action in support:
        next_state = canon(tuple(state[index] + action[index] for index in range(len(state))))
        expected += probability * potential_fn(next_state)
    return expected - base


def _k5_centered_potential_specs() -> tuple[tuple[str, object], ...]:
    return (
        ("phi_inf", _phi_centered_max),
        ("phi_tau_0.5", lambda state: _phi_centered_softmax(state, 0.5)),
        ("phi_tau_1.0", lambda state: _phi_centered_softmax(state, 1.0)),
        ("phi_tau_2.0", lambda state: _phi_centered_softmax(state, 2.0)),
    )


def k5_centered_potential_comb_vs_chase_rows(
    max_used: int,
) -> tuple[K5CenteredPotentialComparisonRow, ...]:
    rows: list[K5CenteredPotentialComparisonRow] = []
    for used in range(max_used + 1):
        for state in all_states(5, used):
            canonical_state = canon(state)
            mean_score = sum(canonical_state) / 5.0
            ptype = packet_type(canonical_state)
            gaps = _packet_gaps(canonical_state)
            for potential_name, potential_fn in _k5_centered_potential_specs():
                comb_delta = _expected_potential_delta(canonical_state, _k5_comb_support(), potential_fn)
                chase_delta = _expected_potential_delta(canonical_state, _k5_chase_support(), potential_fn)
                rows.append(
                    K5CenteredPotentialComparisonRow(
                        state=canonical_state,
                        used=used,
                        potential_name=potential_name,
                        phi_value=potential_fn(canonical_state),
                        comb_delta=comb_delta,
                        chase_delta=chase_delta,
                        delta_diff=chase_delta - comb_delta,
                        packet_type=ptype,
                        packet_gaps=gaps,
                        max_gap=max(gaps, default=0),
                        top_gap=canonical_state[0] - mean_score,
                        bottom_gap=mean_score - canonical_state[-1],
                        multiple_near_leaders=sum(1 for value in canonical_state if canonical_state[0] - value <= 1) >= 2,
                        bottom_below_mean=canonical_state[-1] < mean_score,
                    )
                )
    return tuple(rows)


def _k5_softmax_tau_specs(horizon: int) -> tuple[tuple[str, float], ...]:
    fixed = (("tau_0.5", 0.5), ("tau_1.0", 1.0), ("tau_2.0", 2.0))
    sqrt_h = horizon ** 0.5 if horizon > 0 else 1.0
    scaled = (
        ("tau_0.25_sqrt_h", 0.25 * sqrt_h),
        ("tau_0.5_sqrt_h", 0.5 * sqrt_h),
        ("tau_1.0_sqrt_h", sqrt_h),
        ("tau_2.0_sqrt_h", 2.0 * sqrt_h),
    )
    return fixed + scaled


def _softmax_certificate_residual(
    state: tuple[int, ...],
    action: tuple[int, ...],
    tau: float,
    learner: tuple[float, ...],
) -> float:
    next_state = canon(tuple(state[index] + action[index] for index in range(len(state))))
    learner_gain = sum(learner[index] * action[index] for index in range(len(state)))
    return _phi_centered_softmax(next_state, tau) - learner_gain - _phi_centered_softmax(state, tau)


def k5_softmax_potential_certificate_rows(
    max_used: int,
    horizons: tuple[int, ...] = (5, 10, 20, 50),
) -> tuple[K5SoftmaxPotentialCertificateRow, ...]:
    all_gain_actions = all_actions(5)
    rows: list[K5SoftmaxPotentialCertificateRow] = []
    for horizon in horizons:
        for tau_name, tau in _k5_softmax_tau_specs(horizon):
            potential_fn = lambda state, tau=tau: _phi_centered_softmax(state, tau)
            for used in range(max_used + 1):
                for state in all_states(5, used):
                    canonical_state = canon(state)
                    learner = _softmax_centered_gradient(canonical_state, tau)
                    residuals = [
                        (_softmax_certificate_residual(canonical_state, action, tau, learner), action)
                        for action in all_gain_actions
                    ]
                    max_residual, max_action = max(residuals, key=lambda item: item[0])
                    chase_residual = sum(
                        probability * _softmax_certificate_residual(canonical_state, action, tau, learner)
                        for probability, action in _k5_chase_support()
                    )
                    comb_residual = sum(
                        probability * _softmax_certificate_residual(canonical_state, action, tau, learner)
                        for probability, action in _k5_comb_support()
                    )
                    chase_delta = _expected_potential_delta(canonical_state, _k5_chase_support(), potential_fn)
                    comb_delta = _expected_potential_delta(canonical_state, _k5_comb_support(), potential_fn)
                    rows.append(
                        K5SoftmaxPotentialCertificateRow(
                            state=canonical_state,
                            used=used,
                            horizon=horizon,
                            tau_name=tau_name,
                            tau=tau,
                            phi_value=_phi_centered_softmax(canonical_state, tau),
                            learner=learner,
                            max_residual=max_residual,
                            max_centered_residual=max_residual - chase_residual,
                            max_action=max_action,
                            chase_average_residual=chase_residual,
                            comb_average_residual=comb_residual,
                            chase_delta=chase_delta,
                            comb_delta=comb_delta,
                            delta_diff=chase_delta - comb_delta,
                            packet_type=packet_type(canonical_state),
                            packet_gaps=_packet_gaps(canonical_state),
                        )
                    )
    return tuple(rows)


def _balanced_pair_from_action_text(action_text: str) -> tuple[tuple[float, tuple[int, ...]], ...]:
    action = tuple(int(bit) for bit in action_text)
    return _policy_from_weighted_actions(((0.5, action), (0.5, complement(action))))


def _potential_greedy_interval_grammar(k: int) -> tuple[PotentialGreedyGrammarCandidate, ...]:
    if k == 5:
        return (
            PotentialGreedyGrammarCandidate("comb_10101", _balanced_pair_from_action_text("10101")),
            PotentialGreedyGrammarCandidate("chase_10100", _balanced_pair_from_action_text("10100")),
            PotentialGreedyGrammarCandidate("one_run_10010", _balanced_pair_from_action_text("10010")),
            PotentialGreedyGrammarCandidate("one_run_10001", _balanced_pair_from_action_text("10001")),
            PotentialGreedyGrammarCandidate("one_run_10110", _balanced_pair_from_action_text("10110")),
            PotentialGreedyGrammarCandidate("one_run_11000", _balanced_pair_from_action_text("11000")),
        )
    if k == 6:
        return (
            PotentialGreedyGrammarCandidate("interval_0_0", _balanced_policy_from_one_run_interval(6, (0, 0))),
            PotentialGreedyGrammarCandidate("interval_0_2", _balanced_policy_from_one_run_interval(6, (0, 2))),
            PotentialGreedyGrammarCandidate("interval_0_3", _balanced_policy_from_one_run_interval(6, (0, 3))),
            PotentialGreedyGrammarCandidate("interval_0_4", _balanced_policy_from_one_run_interval(6, (0, 4))),
            PotentialGreedyGrammarCandidate("interval_1_1", _balanced_policy_from_one_run_interval(6, (1, 1))),
            PotentialGreedyGrammarCandidate("interval_1_4", _balanced_policy_from_one_run_interval(6, (1, 4))),
            PotentialGreedyGrammarCandidate(
                "top_full_tail",
                _mixed_one_run_interval_policy(
                    6,
                    ((1.0 / 3.0, (0, 0)), (1.0 / 3.0, (0, 4)), (1.0 / 3.0, (1, 4))),
                ),
            ),
            PotentialGreedyGrammarCandidate(
                "tail_heavy",
                _mixed_one_run_interval_policy(6, ((0.25, (0, 0)), (0.25, (0, 4)), (0.5, (1, 4)))),
            ),
        )
    return ()


def _potential_greedy_choice(
    state: tuple[int, ...],
    remaining_horizon: int,
    c: float,
) -> PotentialGreedyGrammarCandidate:
    canonical_state = canon(state)
    grammar = _potential_greedy_interval_grammar(len(canonical_state))
    if not grammar:
        raise ValueError(f"no potential-greedy grammar for k={len(canonical_state)}")
    tau_next = c * (max(remaining_horizon - 1, 1) ** 0.5)

    def score(candidate: PotentialGreedyGrammarCandidate) -> float:
        expected = 0.0
        for probability, action in candidate.support:
            next_state = canon(tuple(canonical_state[index] + action[index] for index in range(len(canonical_state))))
            expected += probability * _phi_centered_softmax(next_state, tau_next)
        return expected

    return max(grammar, key=lambda candidate: (score(candidate), -len(candidate.support), candidate.name))


def potential_greedy_interval_policy(k: int, c: float):
    if k not in (5, 6):
        raise ValueError("potential-greedy interval policy is currently implemented for k=5 and k=6")

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        if len(state) != k:
            raise ValueError(f"potential-greedy policy k={k} received state of length {len(state)}")
        return _potential_greedy_choice(state, remaining_horizon, c).support

    return policy


def _potential_greedy_policy_from_c(c: float):
    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        return _potential_greedy_choice(state, remaining_horizon, c).support

    return policy


def _chase5_policy(
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    del state, remaining_horizon
    return (
        (0.5, (1, 0, 1, 0, 0)),
        (0.5, (0, 1, 0, 1, 1)),
    )


def _interval_from_action(
    action: tuple[int, ...],
) -> tuple[int, int] | None:
    return _one_run_interval_from_edge_signature(_edge_signature(action))


def _format_interval_distribution(
    interval_totals: dict[tuple[int, int] | None, float],
) -> str:
    total = sum(interval_totals.values())
    if total <= 0.0:
        return "-"
    return ", ".join(
        f"{_format_interval(interval)}:{weight / total:.3f}"
        for interval, weight in sorted(interval_totals.items(), key=lambda item: (-item[1], str(item[0])))
    )


def _packet_barycenter_status(packet_value: float, mean_score: float, tolerance: float = 1e-9) -> str:
    if packet_value > mean_score + tolerance:
        return "above_mean"
    if packet_value < mean_score - tolerance:
        return "below_mean"
    return "crosses_mean"


def _support_interval_distribution(
    support: tuple[tuple[float, tuple[int, ...], tuple[int, ...]], ...],
) -> tuple[tuple[tuple[int, int] | None, float], ...]:
    totals: dict[tuple[int, int] | None, float] = defaultdict(float)
    for weight, action, _ in support:
        totals[_interval_from_action(action)] += weight
    return tuple(sorted(totals.items(), key=lambda item: (-item[1], str(item[0]))))


def _barycenter_action_row_from_dual_row(
    library_name: str,
    row: LibraryLPDualInspectRow,
) -> BarycenterActionRow:
    groups = _packet_index_groups(row.state)
    mean_score = sum(row.state) / len(row.state) if row.state else 0.0
    summaries: list[BarycenterPacketSummary] = []
    for packet_index, group in enumerate(groups):
        packet_value = row.state[group[0]]
        exposure = sum(row.expected_action[index] for index in group) / len(group)
        split_rate = 0.0
        for weight, action, _ in row.support:
            bits = {action[index] for index in group}
            if len(bits) > 1:
                split_rate += weight
        summaries.append(
            BarycenterPacketSummary(
                status=_packet_barycenter_status(packet_value, mean_score),
                packet_index=packet_index,
                packet_size=len(group),
                exposure=exposure,
                split_rate=split_rate,
            )
        )
    top_exposure = row.expected_action[0] if row.expected_action else 0.0
    bottom_index = len(row.state) - 1
    bottom_exposure = row.expected_action[bottom_index] if row.expected_action else 0.0
    p_top1_bottom0 = 0.0
    p_top0_bottom1 = 0.0
    p_top_equals_bottom = 0.0
    p_top_bottom = 0.0
    for weight, action, _ in row.support:
        top = action[0] if action else 0
        bottom = action[bottom_index] if action else 0
        if top == 1 and bottom == 0:
            p_top1_bottom0 += weight
        if top == 0 and bottom == 1:
            p_top0_bottom1 += weight
        if top == bottom:
            p_top_equals_bottom += weight
        p_top_bottom += weight * top * bottom
    return BarycenterActionRow(
        library_name=library_name,
        time=row.time,
        remaining_horizon=row.remaining_horizon,
        state=row.state,
        packet_type=row.packet_type,
        packet_gaps=row.packet_gaps,
        occupancy_probability=row.occupancy_probability,
        mean_score=mean_score,
        packet_summaries=tuple(summaries),
        top_exposure=top_exposure,
        bottom_exposure=bottom_exposure,
        p_top1_bottom0=p_top1_bottom0,
        p_top0_bottom1=p_top0_bottom1,
        p_top_equals_bottom=p_top_equals_bottom,
        top_bottom_covariance=p_top_bottom - top_exposure * bottom_exposure,
        interval_distribution=_support_interval_distribution(row.support),
    )


def k6_barycenter_action_rows(
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[BarycenterActionRow, ...]:
    rows: list[BarycenterActionRow] = []
    for library_name in ("top_prefix_all", "one_run"):
        _, dual_rows = library_lp_dual_inspect_rows(6, T, library_name, support_tolerance=support_tolerance)
        rows.extend(_barycenter_action_row_from_dual_row(library_name, row) for row in dual_rows)
    return tuple(rows)


def _subsequence_packet_type(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return ()
    sizes: list[int] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            sizes.append(count)
            current = value
            count = 1
    sizes.append(count)
    return tuple(sizes)


def _alive_span(alive_positions: tuple[int, ...]) -> tuple[int, int] | None:
    if not alive_positions:
        return None
    return (min(alive_positions), max(alive_positions))


def _alive_is_prefix(alive_positions: tuple[int, ...]) -> bool:
    return alive_positions == tuple(range(len(alive_positions)))


def _classify_one_run_alive_relation(
    interval: tuple[int, int] | None,
    alive_span: tuple[int, int] | None,
) -> tuple[str, tuple[int, int] | None]:
    if interval is None:
        return "constant_or_no_run", None
    start, end = interval
    if alive_span is None:
        return "outside_alive", None
    alive_start, alive_end = alive_span
    relative = (start - alive_start, end - alive_start)
    if start == 0:
        return "prefix_truncated_comb_on_full_k", relative
    if start == alive_start and end == alive_end - 1:
        return "alive_full_comb", relative
    if start == alive_start and end <= alive_end - 1:
        return "alive_prefix_truncated_comb", relative
    if alive_start < start <= end < alive_end:
        return "alive_internal_repair", relative
    if start <= alive_end <= end:
        return "crosses_alive_dead_boundary", relative
    if end < alive_start or start > alive_end - 1:
        return "outside_alive", relative
    return "shifted_not_alive_explained", relative


def _classify_k6_one_run_interval_relation(
    interval: tuple[int, int] | None,
    state: tuple[int, ...],
) -> str:
    if interval is None:
        return "constant_or_no_run"
    start, end = interval
    edge_count = len(state) - 1
    boundaries = set(_packet_boundary_edges(_packet_index_groups(state)))
    crosses_boundary = any(start <= boundary <= end for boundary in boundaries)
    if start == 0 and end == edge_count - 1:
        return "full_comb_like"
    if start == 0:
        return "prefix_truncated"
    if end == edge_count - 1:
        return "tail_cleanup"
    if start == end:
        if start in boundaries:
            return "crosses_packet_boundary"
        if (start - 1) in boundaries:
            return "left_shoulder_repair"
        if (start + 1) in boundaries:
            return "right_shoulder_repair"
        return "short_internal_repair"
    if crosses_boundary:
        return "crosses_packet_boundary"
    if end - start <= 1:
        return "short_internal_repair"
    return "other"


def _one_run_interval_from_edge_signature(edge_signature: tuple[int, ...]) -> tuple[int, int] | None:
    intervals = tuple(_edge_run_intervals(edge_signature))
    return intervals[0] if len(intervals) == 1 else None


def _interval_distribution_for_interval_support(
    support: tuple[K6OneRunIntervalActionRow, ...],
) -> tuple[tuple[tuple[int, int] | None, float], ...]:
    totals: dict[tuple[int, int] | None, float] = defaultdict(float)
    for item in support:
        totals[item.interval] += item.weight
    return tuple(sorted(totals.items(), key=lambda pair: (-pair[1], pair[0] is None, pair[0] or (-1, -1))))


def _interval_distribution_for_alive_row(
    row: K6OneRunAlivePatternRow,
) -> tuple[tuple[tuple[int, int] | None, str], ...]:
    weights: dict[tuple[int, int] | None, float] = defaultdict(float)
    for item in row.support:
        weights[item.interval] += item.weight
    return tuple(
        (interval, _format_fractionish(weight))
        for interval, weight in sorted(weights.items(), key=lambda pair: (pair[0] is None, pair[0] or (-1, -1)))
    )


def k6_one_run_alive_pattern_rows(
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[K6OneRunAlivePatternRow, ...]:
    _, dual_rows = library_lp_dual_inspect_rows(6, T, "one_run", support_tolerance=support_tolerance)
    rows: list[K6OneRunAlivePatternRow] = []
    for row in dual_rows:
        alive = _alive_positions(row.state, row.remaining_horizon)
        span = _alive_span(alive)
        alive_values = tuple(row.state[index] for index in alive)
        support_rows: list[K6OneRunAliveActionRow] = []
        for weight, action, edge_signature in row.support:
            intervals = tuple(_edge_run_intervals(edge_signature))
            interval = intervals[0] if len(intervals) == 1 else None
            relation_class, relative_interval = _classify_one_run_alive_relation(interval, span)
            next_state = canon(tuple(row.state[index] + action[index] for index in range(len(row.state))))
            support_rows.append(
                K6OneRunAliveActionRow(
                    weight=weight,
                    action=action,
                    edge_signature=edge_signature,
                    interval=interval,
                    interval_relative_to_alive=relative_interval,
                    relation_class=relation_class,
                    successor_packet_type=packet_type(next_state),
                )
            )
        rows.append(
            K6OneRunAlivePatternRow(
                time=row.time,
                remaining_horizon=row.remaining_horizon,
                state=row.state,
                packet_type=row.packet_type,
                packet_gaps=row.packet_gaps,
                adjacent_gaps=row.adjacent_gaps,
                occupancy_probability=row.occupancy_probability,
                learner_distribution=row.learner_distribution,
                alive_positions=alive,
                alive_count=len(alive),
                alive_span=span,
                alive_is_prefix=_alive_is_prefix(alive),
                alive_packet_type=_subsequence_packet_type(alive_values),
                support=tuple(sorted(support_rows, key=lambda item: (-item.weight, _format_action(item.action)))),
            )
        )
    return tuple(rows)


def _group_alive_rows(rows: tuple[K6OneRunAlivePatternRow, ...], key_fn):
    grouped: dict[object, list[K6OneRunAlivePatternRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return tuple(
        sorted(
            grouped.items(),
            key=lambda item: (
                -sum(row.occupancy_probability for row in item[1]),
                -len(item[1]),
                str(item[0]),
            ),
        )
    )


def _alive_support_relation_totals(
    rows: tuple[K6OneRunAlivePatternRow, ...],
) -> tuple[tuple[str, float, float], ...]:
    dual_totals: dict[str, float] = defaultdict(float)
    occupancy_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for support in row.support:
            dual_totals[support.relation_class] += support.weight
            occupancy_totals[support.relation_class] += row.occupancy_probability * support.weight
    return tuple(
        (relation, dual_totals[relation], occupancy_totals[relation])
        for relation in sorted(dual_totals, key=lambda key: (-occupancy_totals[key], key))
    )


def _format_interval(interval: tuple[int, int] | None) -> str:
    return "-" if interval is None else f"{interval[0]}:{interval[1]}"


def _print_alive_row(row: K6OneRunAlivePatternRow, support_n: int = 12) -> None:
    print(
        f"  t={row.time} rem={row.remaining_horizon}"
        f" state={row.state}"
        f" packet={row.packet_type} gaps={row.packet_gaps}"
        f" occ={row.occupancy_probability:.6f}"
    )
    print(
        f"    alive={row.alive_positions}"
        f" alive_count={row.alive_count}"
        f" alive_span={row.alive_span}"
        f" alive_is_prefix={row.alive_is_prefix}"
        f" alive_packet_type={row.alive_packet_type}"
    )
    print(f"    learner p={_format_float_tuple(row.learner_distribution)}")
    print("    support:")
    for support in row.support[:support_n]:
        print(
            f"      w={_format_fractionish(support.weight):>8s}"
            f" action={_format_action(support.action)}"
            f" edge={_format_edge_signature(support.edge_signature)}"
            f" interval={_format_interval(support.interval)}"
            f" rel={_format_interval(support.interval_relative_to_alive)}"
            f" relation={support.relation_class}"
            f" successor={support.successor_packet_type}"
        )
    if len(row.support) > support_n:
        print("      ...")


def print_k6_one_run_alive_pattern_report(
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 80,
) -> None:
    rows = k6_one_run_alive_pattern_rows(T, support_tolerance=support_tolerance)
    zero = (0, 0, 0, 0, 0, 0)
    optimal = optimal_values(6, T)[T][zero]
    library_values: dict[str, float] = {}
    for library_name in ("top_prefix_all", "one_run", "two_run", "local_edges"):
        actions = _strategy_class_library_actions(6, T, library_name)
        values = _library_lp_value_layers(6, T, actions)
        library_values[library_name] = values[T][zero]

    print(f"k=6 one-run alive-pattern report, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print()
    print("Value summary:")
    print(f"  V_star: {optimal:.6f}")
    for library_name in ("top_prefix_all", "one_run", "two_run", "local_edges"):
        value = library_values[library_name]
        gap = optimal - value
        print(
            f"  {library_name:14s}"
            f" V={value:.6f}"
            f" gap={gap:.6f}"
            f" gap/sqrt(T)={gap / (T ** 0.5 if T > 0 else 1.0):.6f}"
        )
    print()

    print("Overall support mass by relation class:")
    for relation, dual_weight, occupancy_weight in _alive_support_relation_totals(rows):
        print(
            f"  {relation:32s}"
            f" dual_weight={dual_weight:.6f}"
            f" occupancy_weight={occupancy_weight:.6f}"
        )
    print()

    print("Support mass by alive_count:")
    for alive_count, group in _group_alive_rows(rows, lambda row: row.alive_count):
        print(
            f"  alive_count={alive_count}"
            f" rows={len(group)}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
        )
        relation_totals = _alive_support_relation_totals(tuple(group))
        print("    top relation classes:")
        for relation, _, occupancy_weight in relation_totals[:8]:
            print(f"      {relation}: {occupancy_weight:.6f}")
        interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
        regime_totals: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
        for row in group:
            regime_totals[(row.packet_type, row.packet_gaps)] += row.occupancy_probability
            for support in row.support:
                interval_totals[support.interval] += row.occupancy_probability * support.weight
        print("    top edge intervals:")
        for interval, weight in sorted(interval_totals.items(), key=lambda item: (-item[1], str(item[0])))[:8]:
            print(f"      {_format_interval(interval)}: {weight:.6f}")
        print("    top packet regimes:")
        for (ptype, gaps), weight in sorted(regime_totals.items(), key=lambda item: (-item[1], item[0]))[:5]:
            print(f"      packet={ptype} gaps={gaps}: {weight:.6f}")
    print()

    print("Top dual rows by occupancy:")
    for row in sorted(rows, key=lambda item: (-item.occupancy_probability, item.time, item.state))[:n]:
        _print_alive_row(row)
    print()

    print("Compression check:")
    mapping_a = {
        (row.remaining_horizon, row.packet_type, row.packet_gaps, _interval_distribution_for_alive_row(row))
        for row in rows
    }
    mapping_b = {
        (
            row.remaining_horizon,
            row.alive_count,
            row.alive_span,
            row.packet_type,
            _interval_distribution_for_alive_row(row),
        )
        for row in rows
    }
    mapping_c: dict[tuple[int, str, tuple[int, int] | None], float] = defaultdict(float)
    for row in rows:
        for support in row.support:
            mapping_c[(row.alive_count, support.relation_class, support.interval_relative_to_alive)] += (
                row.occupancy_probability * support.weight
            )
    print(f"  distinct (rem, packet_type, gaps) -> interval distributions: {len(mapping_a)}")
    print(f"  distinct (rem, alive_count, alive_span, packet_type) -> interval distributions: {len(mapping_b)}")
    print(f"  distinct (alive_count, relation_class, interval_relative_to_alive): {len(mapping_c)}")
    print("  top alive/relation/relative interval masses:")
    for (alive_count, relation, relative_interval), weight in sorted(mapping_c.items(), key=lambda item: (-item[1], str(item[0])))[:n]:
        print(
            f"    alive_count={alive_count}"
            f" relation={relation}"
            f" rel_interval={_format_interval(relative_interval)}"
            f" mass={weight:.6f}"
        )
    print()

    print("Candidate rule hints:")
    for alive_count, group in _group_alive_rows(rows, lambda row: row.alive_count):
        prefix_occupancy = sum(row.occupancy_probability for row in group if row.alive_is_prefix)
        total_occupancy = sum(row.occupancy_probability for row in group)
        interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
        boundary_gap_totals: dict[int, float] = defaultdict(float)
        cleanup_totals: dict[int, float] = defaultdict(float)
        for row in group:
            if row.alive_span is not None and row.alive_span[1] < len(row.state) - 1:
                boundary_gap_totals[row.state[row.alive_span[1]] - row.state[row.alive_span[1] + 1]] += row.occupancy_probability
            if row.remaining_horizon in {1, 2}:
                cleanup_totals[row.remaining_horizon] += row.occupancy_probability
            for support in row.support:
                interval_totals[support.interval] += row.occupancy_probability * support.weight
        print(
            f"  alive_count={alive_count}"
            f" alive_prefix_share={prefix_occupancy / total_occupancy if total_occupancy else 0.0:.3f}"
            f" rows={len(group)}"
            f" occupancy={total_occupancy:.6f}"
        )
        print("    dominant intervals:")
        for interval, weight in sorted(interval_totals.items(), key=lambda item: (-item[1], str(item[0])))[:6]:
            print(f"      {_format_interval(interval)}: {weight:.6f}")
        if boundary_gap_totals:
            print(f"    alive/dead boundary gaps={tuple(sorted(boundary_gap_totals.items(), key=lambda item: (-item[1], item[0]))[:5])}")
        if cleanup_totals:
            print(f"    cleanup horizons={tuple(sorted(cleanup_totals.items()))}")


def _k6_alive_candidate_policy_specs() -> dict[str, tuple[str, str]]:
    return {
        "k6_alive_trunc_comb_policy": ("trunc6_chase5_comb_tail", "weak"),
        "k6_alive_trunc_comb_twin3_policy": ("trunc6_chase5_twin3_tail", "weak"),
        "k6_alive_full_comb_control_policy": ("fullcomb6_chase5_comb_tail", "weak"),
        "k6_alive_trunc_comb_strict_policy": ("trunc6_chase5_comb_tail", "strict"),
        "k6_alive_trunc_comb_twin3_strict_policy": ("trunc6_chase5_twin3_tail", "strict"),
        "k6_alive_full_comb_control_strict_policy": ("fullcomb6_chase5_comb_tail", "strict"),
    }


def _k6_alive_candidate_histograms(
    T: int,
    variant: str,
    death_mode: str,
) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[str, float], ...]]:
    def policy_fn(state: tuple[int, ...], remaining_horizon: int):
        return _k6_alive_candidate_policy(state, remaining_horizon, variant, death_mode=death_mode)

    occupancy = _time_policy_occupancy(6, T, policy_fn)
    alive_counts: dict[int, float] = defaultdict(float)
    pattern_usage: dict[str, float] = defaultdict(float)
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            alive_count, _, pattern_label = _k6_alive_policy_metadata(
                state,
                remaining_horizon,
                variant,
                death_mode=death_mode,
            )
            alive_counts[alive_count] += probability
            pattern_usage[f"m={alive_count}:{pattern_label}"] += probability

    return (
        tuple(sorted(alive_counts.items(), key=lambda item: (-item[1], item[0]))),
        tuple(sorted(pattern_usage.items(), key=lambda item: (-item[1], item[0]))),
    )


def k6_alive_candidate_benchmark_rows(T: int) -> tuple[K6AliveCandidateBenchmarkRow, ...]:
    if T <= 0:
        raise ValueError("T must be positive")
    zero = (0, 0, 0, 0, 0, 0)
    optimal = optimal_values(6, T)[T][zero]
    one_run_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "one_run"))[T][zero]

    rows: list[K6AliveCandidateBenchmarkRow] = []
    for name, (variant, death_mode) in _k6_alive_candidate_policy_specs().items():
        policy_fn = _resolve_time_dependent_policy(6, name)
        if policy_fn is None:
            raise ValueError(f"unknown k=6 alive candidate policy: {name}")
        values = evaluate_time_dependent_policy(6, T, policy_fn)
        value = values[T][zero]
        gap_to_optimal = optimal - value
        gap_to_one_run = one_run_value - value
        alive_histogram, pattern_histogram = _k6_alive_candidate_histograms(T, variant, death_mode)
        rows.append(
            K6AliveCandidateBenchmarkRow(
                T=T,
                name=name,
                value=value,
                optimal_value=optimal,
                gap_to_optimal=gap_to_optimal,
                normalized_gap_to_optimal=gap_to_optimal / (T ** 0.5),
                one_run_value=one_run_value,
                gap_to_one_run=gap_to_one_run,
                normalized_gap_to_one_run=gap_to_one_run / (T ** 0.5),
                alive_count_histogram=alive_histogram,
                pattern_usage_histogram=pattern_histogram,
            )
        )
    return tuple(rows)


def print_k6_alive_candidate_benchmark(T_values: tuple[int, ...]) -> None:
    print("k=6 alive-prefix candidate benchmark")
    print()
    print(f"T values: {T_values}")
    print()
    for T in T_values:
        zero = (0, 0, 0, 0, 0, 0)
        optimal = optimal_values(6, T)[T][zero]
        reference_values: dict[str, float] = {}
        for library_name in ("top_prefix_all", "one_run", "two_run", "local_edges"):
            actions = _strategy_class_library_actions(6, T, library_name)
            reference_values[library_name] = _library_lp_value_layers(6, T, actions)[T][zero]

        print(f"k=6 T={T}")
        print("Reference values:")
        print(f"  V_star: {optimal:.6f}")
        for library_name in ("top_prefix_all", "one_run", "two_run", "local_edges"):
            value = reference_values[library_name]
            gap = optimal - value
            print(
                f"  {library_name:14s}"
                f" V={value:.6f}"
                f" gap={gap:.6f}"
                f" gap/sqrt(T)={gap / (T ** 0.5):.6f}"
            )
        print()
        print(
            "policy                                      V_policy"
            " gap_to_V_star gap/sqrt(T) gap_to_one_run gap_to_one_run/sqrt(T)"
        )
        for row in sorted(k6_alive_candidate_benchmark_rows(T), key=lambda item: (item.normalized_gap_to_optimal, item.name)):
            print(
                f"{row.name:42s}"
                f" {row.value:10.6f}"
                f" {row.gap_to_optimal:13.6f}"
                f" {row.normalized_gap_to_optimal:11.6f}"
                f" {row.gap_to_one_run:14.6f}"
                f" {row.normalized_gap_to_one_run:22.6f}"
            )
            rendered_alive = ", ".join(f"{alive}:{weight:.6f}" for alive, weight in row.alive_count_histogram)
            rendered_patterns = ", ".join(
                f"{pattern}:{weight:.6f}" for pattern, weight in row.pattern_usage_histogram[:8]
            )
            print(f"  alive_count occupancy: {rendered_alive}")
            print(f"  pattern usage: {rendered_patterns}")
        print()


def k6_one_run_interval_rule_rows(
    T: int,
    support_tolerance: float = 1e-8,
) -> tuple[K6OneRunIntervalRuleRow, ...]:
    _, dual_rows = library_lp_dual_inspect_rows(6, T, "one_run", support_tolerance=support_tolerance)
    rows: list[K6OneRunIntervalRuleRow] = []
    for row in dual_rows:
        support_rows: list[K6OneRunIntervalActionRow] = []
        for weight, action, edge_signature in row.support:
            interval = _one_run_interval_from_edge_signature(edge_signature)
            next_state = canon(tuple(row.state[index] + action[index] for index in range(6)))
            support_rows.append(
                K6OneRunIntervalActionRow(
                    weight=weight,
                    action=action,
                    edge_signature=edge_signature,
                    interval=interval,
                    relation_class=_classify_k6_one_run_interval_relation(interval, row.state),
                    successor_packet_type=packet_type(next_state),
                )
            )
        rows.append(
            K6OneRunIntervalRuleRow(
                time=row.time,
                remaining_horizon=row.remaining_horizon,
                state=row.state,
                packet_type=row.packet_type,
                packet_gaps=row.packet_gaps,
                adjacent_gaps=row.adjacent_gaps,
                occupancy_probability=row.occupancy_probability,
                value=row.value,
                learner_distribution=row.learner_distribution,
                alive_count=_alive_prefix_length(row.state, row.remaining_horizon),
                support=tuple(sorted(support_rows, key=lambda item: (-item.weight, _format_action(item.action)))),
            )
        )
    return tuple(rows)


def _candidate_policy_patterns(policy: tuple[tuple[float, tuple[int, ...]], ...]) -> tuple[str, ...]:
    return tuple(
        f"{_format_action(action)}:{_format_fractionish(probability)}"
        for probability, action in sorted(policy, key=lambda item: _format_action(item[1]))
    )


def k6_one_run_candidate_failure_rows(
    T: int,
    policy_names: tuple[str, ...] = (
        "k6_alive_trunc_comb_policy",
        "k6_alive_full_comb_control_policy",
    ),
    support_tolerance: float = 1e-8,
) -> tuple[K6OneRunCandidateFailureRow, ...]:
    one_run_actions = _strategy_class_library_actions(6, T, "one_run")
    one_run_values = _library_lp_value_layers(6, T, one_run_actions)
    interval_rows_by_key = {
        (row.time, row.state): row
        for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance)
    }
    rows: list[K6OneRunCandidateFailureRow] = []
    for policy_name in policy_names:
        policy_fn = _resolve_time_dependent_policy(6, policy_name)
        if policy_fn is None:
            raise ValueError(f"unknown candidate policy: {policy_name}")
        occupancy = _time_policy_occupancy(6, T, policy_fn)
        for time in range(T):
            remaining_horizon = T - time
            continuation = one_run_values[remaining_horizon - 1]
            for state, probability in occupancy[time].items():
                policy = policy_fn(state, remaining_horizon)
                candidate_step = _time_policy_step_value(6, state, policy, continuation)
                one_run_value = one_run_values[remaining_horizon][state]
                action_loss = one_run_value - candidate_step
                interval_row = interval_rows_by_key.get((time, state))
                if interval_row is None:
                    # The candidate can drift outside the one-run LP occupancy basin; inspect
                    # its local one-run dual anyway without assigning LP occupancy.
                    q_by_action = {
                        action: _next_state_value(state, action, continuation)
                        for action in one_run_actions
                    }
                    dual = solve_adversary_dual(q_by_action, 6)
                    if not dual.success:
                        raise RuntimeError(f"dual LP failed at state {state}: {dual.message}")
                    support = tuple(
                        K6OneRunIntervalActionRow(
                            weight=weight,
                            action=action,
                            edge_signature=_edge_signature(action),
                            interval=_one_run_interval_from_edge_signature(_edge_signature(action)),
                            relation_class=_classify_k6_one_run_interval_relation(
                                _one_run_interval_from_edge_signature(_edge_signature(action)),
                                state,
                            ),
                            successor_packet_type=packet_type(canon(tuple(state[index] + action[index] for index in range(6)))),
                        )
                        for action, weight in dual.weights_by_action
                        if weight >= support_tolerance
                    )
                else:
                    support = interval_row.support
                interval_distribution = _interval_distribution_for_interval_support(support)
                dominant_interval = interval_distribution[0][0] if interval_distribution else None
                dominant_relation = next(
                    (item.relation_class for item in support if item.interval == dominant_interval),
                    "-",
                )
                rows.append(
                    K6OneRunCandidateFailureRow(
                        policy_name=policy_name,
                        time=time,
                        remaining_horizon=remaining_horizon,
                        state=state,
                        packet_type=packet_type(state),
                        packet_gaps=_packet_gaps(state),
                        adjacent_gaps=_gap_vector(state),
                        occupancy_probability=probability,
                        one_run_value=one_run_value,
                        candidate_step_value_on_one_run_continuation=candidate_step,
                        action_loss=action_loss,
                        weighted_action_loss=probability * action_loss,
                        candidate_actions=tuple(policy),
                        candidate_patterns=_candidate_policy_patterns(tuple(policy)),
                        lp_interval_distribution=interval_distribution,
                        lp_dominant_interval=dominant_interval,
                        lp_dominant_relation=dominant_relation,
                    )
                )
    return tuple(rows)


def _k6_one_run_candidate_failure_rows_for_policy(
    T: int,
    policy_name: str,
    policy_fn,
    one_run_actions: tuple[tuple[int, ...], ...],
    one_run_values: tuple[dict[tuple[int, ...], float], ...],
    interval_rows_by_key: dict[tuple[int, tuple[int, ...]], K6OneRunIntervalRuleRow],
    support_tolerance: float = 1e-8,
) -> tuple[K6OneRunCandidateFailureRow, ...]:
    rows: list[K6OneRunCandidateFailureRow] = []
    occupancy = _time_policy_occupancy(6, T, policy_fn)
    for time in range(T):
        remaining_horizon = T - time
        continuation = one_run_values[remaining_horizon - 1]
        for state, probability in occupancy[time].items():
            policy = policy_fn(state, remaining_horizon)
            candidate_step = _time_policy_step_value(6, state, policy, continuation)
            one_run_value = one_run_values[remaining_horizon][state]
            action_loss = one_run_value - candidate_step
            interval_row = interval_rows_by_key.get((time, state))
            if interval_row is None:
                q_by_action = {
                    action: _next_state_value(state, action, continuation)
                    for action in one_run_actions
                }
                dual = solve_adversary_dual(q_by_action, 6)
                if not dual.success:
                    raise RuntimeError(f"dual LP failed at state {state}: {dual.message}")
                support = tuple(
                    K6OneRunIntervalActionRow(
                        weight=weight,
                        action=action,
                        edge_signature=_edge_signature(action),
                        interval=_one_run_interval_from_edge_signature(_edge_signature(action)),
                        relation_class=_classify_k6_one_run_interval_relation(
                            _one_run_interval_from_edge_signature(_edge_signature(action)),
                            state,
                        ),
                        successor_packet_type=packet_type(canon(tuple(state[index] + action[index] for index in range(6)))),
                    )
                    for action, weight in dual.weights_by_action
                    if weight >= support_tolerance
                )
            else:
                support = interval_row.support
            interval_distribution = _interval_distribution_for_interval_support(support)
            dominant_interval = interval_distribution[0][0] if interval_distribution else None
            dominant_relation = next(
                (item.relation_class for item in support if item.interval == dominant_interval),
                "-",
            )
            rows.append(
                K6OneRunCandidateFailureRow(
                    policy_name=policy_name,
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_probability=probability,
                    one_run_value=one_run_value,
                    candidate_step_value_on_one_run_continuation=candidate_step,
                    action_loss=action_loss,
                    weighted_action_loss=probability * action_loss,
                    candidate_actions=tuple(policy),
                    candidate_patterns=_candidate_policy_patterns(tuple(policy)),
                    lp_interval_distribution=interval_distribution,
                    lp_dominant_interval=dominant_interval,
                    lp_dominant_relation=dominant_relation,
                )
            )
    return tuple(rows)


def _k6_patch_name_from_key(
    remaining_horizon: int,
    ptype: tuple[int, ...],
    gaps: tuple[int, ...],
    interval: tuple[int, int],
) -> str:
    packet_part = "".join(str(item) for item in ptype)
    gap_part = "".join(str(item) for item in gaps) or "flat"
    return f"g_rem{remaining_horizon}_{packet_part}_g{gap_part}_i{interval[0]}{interval[1]}"


def _k6_greedy_patch_candidates_from_failures(
    failures: tuple[K6OneRunCandidateFailureRow, ...],
    existing_patches: tuple[K6OneRunPatch, ...],
    candidate_n: int,
) -> tuple[K6OneRunPatch, ...]:
    existing_regimes = {
        (patch.remaining_horizon, patch.packet_type, patch.packet_gaps)
        for patch in existing_patches
    }
    weighted_losses: dict[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, int]], float] = defaultdict(float)
    for failure in failures:
        interval = failure.lp_dominant_interval
        if interval is None or failure.weighted_action_loss <= 1e-12:
            continue
        regime = (failure.remaining_horizon, failure.packet_type, failure.packet_gaps)
        if regime in existing_regimes:
            continue
        weighted_losses[regime + (interval,)] += failure.weighted_action_loss
    patches: list[K6OneRunPatch] = []
    for (remaining_horizon, ptype, gaps, interval), _ in sorted(
        weighted_losses.items(),
        key=lambda item: (-item[1], item[0]),
    )[:candidate_n]:
        patches.append(
            K6OneRunPatch(
                name=_k6_patch_name_from_key(remaining_horizon, ptype, gaps, interval),
                remaining_horizon=remaining_horizon,
                packet_type=ptype,
                packet_gaps=gaps,
                interval=interval,
            )
        )
    return tuple(patches)


def _matching_k6_mixed_patch(
    state: tuple[int, ...],
    remaining_horizon: int,
    patches: tuple[MixedIntervalPatch, ...],
) -> MixedIntervalPatch | None:
    canonical_state = canon(state)
    ptype = packet_type(canonical_state)
    gaps = _packet_gaps(canonical_state)
    for patch in patches:
        if (
            patch.remaining_horizon == remaining_horizon
            and patch.packet_type == ptype
            and patch.packet_gaps == gaps
        ):
            return patch
    return None


def make_k6_one_run_mixed_patch_policy(
    base_policy_name: str,
    deterministic_patches: tuple[K6OneRunPatch, ...],
    mixed_patches: tuple[MixedIntervalPatch, ...],
):
    base_policy = make_k6_one_run_patch_policy_from_patches(base_policy_name, deterministic_patches)

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        if len(canonical_state) == 6:
            patch = _matching_k6_mixed_patch(canonical_state, remaining_horizon, mixed_patches)
            if patch is not None:
                return _mixed_one_run_interval_policy(6, patch.weighted_intervals)
        return base_policy(canonical_state, remaining_horizon)

    return policy


def _k6_mixed_patch_name_from_key(
    remaining_horizon: int,
    ptype: tuple[int, ...],
    gaps: tuple[int, ...],
) -> str:
    packet_part = "".join(str(item) for item in ptype)
    gap_part = "".join(str(item) for item in gaps) or "flat"
    return f"m_rem{remaining_horizon}_{packet_part}_g{gap_part}"


def _k6_mixed_patch_candidates_from_failures(
    T: int,
    failures: tuple[K6OneRunCandidateFailureRow, ...],
    existing_mixed_patches: tuple[MixedIntervalPatch, ...],
    candidate_n: int,
    distributions_by_regime: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        tuple[tuple[float, tuple[int, int]], ...],
    ] | None = None,
    support_tolerance: float = 1e-8,
) -> tuple[MixedIntervalPatch, ...]:
    existing_regimes = {
        (patch.remaining_horizon, patch.packet_type, patch.packet_gaps)
        for patch in existing_mixed_patches
    }
    weighted_losses: dict[tuple[int, tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
    for failure in failures:
        if failure.weighted_action_loss <= 1e-12:
            continue
        regime = (failure.remaining_horizon, failure.packet_type, failure.packet_gaps)
        if regime in existing_regimes:
            continue
        weighted_losses[regime] += failure.weighted_action_loss
    patches: list[MixedIntervalPatch] = []
    for (remaining_horizon, ptype, gaps), _ in sorted(
        weighted_losses.items(),
        key=lambda item: (-item[1], item[0]),
    )[:candidate_n]:
        if distributions_by_regime is None:
            weighted_intervals = _one_run_interval_distribution_for_regime(
                6,
                T,
                remaining_horizon,
                ptype,
                gaps,
                support_tolerance=support_tolerance,
            )
        else:
            weighted_intervals = distributions_by_regime.get((remaining_horizon, ptype, gaps), ())
        if not weighted_intervals:
            continue
        patches.append(
            MixedIntervalPatch(
                name=_k6_mixed_patch_name_from_key(remaining_horizon, ptype, gaps),
                remaining_horizon=remaining_horizon,
                packet_type=ptype,
                packet_gaps=gaps,
                weighted_intervals=weighted_intervals,
            )
        )
    return tuple(patches)


def _k6_failure_summary(
    failures: tuple[K6OneRunCandidateFailureRow, ...],
    n: int = 10,
) -> tuple[tuple[str, float], ...]:
    summary: dict[str, float] = defaultdict(float)
    for failure in failures:
        key = (
            f"rem={failure.remaining_horizon}"
            f" packet={failure.packet_type}"
            f" gaps={failure.packet_gaps}"
            f" lp={_format_interval(failure.lp_dominant_interval)}"
        )
        summary[key] += failure.weighted_action_loss
    return tuple(sorted(summary.items(), key=lambda item: (-item[1], item[0]))[:n])


def k6_one_run_greedy_patch_search_steps(
    T: int,
    max_patches: int = 20,
    candidate_n: int = 50,
    support_tolerance: float = 1e-8,
) -> tuple[K6GreedyPatchSearchStep, ...]:
    zero = (0, 0, 0, 0, 0, 0)
    one_run_actions = _strategy_class_library_actions(6, T, "one_run")
    one_run_values = _library_lp_value_layers(6, T, one_run_actions)
    one_run_value = one_run_values[T][zero]
    top_prefix_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "top_prefix_all"))[T][zero]
    interval_rows_by_key = {
        (row.time, row.state): row
        for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance)
    }

    patches = tuple(_k6_one_run_patch_list()[:5])
    steps: list[K6GreedyPatchSearchStep] = []
    current_policy = make_k6_one_run_patch_policy_from_patches("k6_alive_full_comb_control_policy", patches)
    current_value = evaluate_time_dependent_policy(6, T, current_policy)[T][zero]

    for step in range(max_patches + 1):
        failures = _k6_one_run_candidate_failure_rows_for_policy(
            T,
            "k6_greedy_current",
            current_policy,
            one_run_actions,
            one_run_values,
            interval_rows_by_key,
            support_tolerance=support_tolerance,
        )
        if step == 0:
            steps.append(
                K6GreedyPatchSearchStep(
                    step=step,
                    selected_patch=None,
                    improvement=0.0,
                    value=current_value,
                    one_run_value=one_run_value,
                    normalized_gap_to_one_run=(one_run_value - current_value) / (T ** 0.5),
                    top_prefix_value=top_prefix_value,
                    gap_to_top_prefix=current_value - top_prefix_value,
                    rejected_candidates=(),
                    top_failure_summaries=_k6_failure_summary(failures),
                )
            )
            if max_patches == 0:
                break
            continue
        candidates = _k6_greedy_patch_candidates_from_failures(failures, patches, candidate_n)
        evaluations: list[tuple[K6OneRunPatch, float]] = []
        best_patch: K6OneRunPatch | None = None
        best_value = current_value
        for candidate in candidates:
            candidate_policy = make_k6_one_run_patch_policy_from_patches(
                "k6_alive_full_comb_control_policy",
                patches + (candidate,),
            )
            value = evaluate_time_dependent_policy(6, T, candidate_policy)[T][zero]
            improvement = value - current_value
            evaluations.append((candidate, improvement))
            if improvement > best_value - current_value + 1e-12:
                best_patch = candidate
                best_value = value
        rejected = tuple(sorted(evaluations, key=lambda item: (-item[1], item[0].name))[:10])
        if best_patch is not None and best_value > current_value + 1e-12:
            selected_patch: K6OneRunPatch | None = best_patch
            improvement = best_value - current_value
            next_value = best_value
        else:
            selected_patch = None
            improvement = 0.0
            next_value = current_value
        next_failures = failures
        if selected_patch is None:
            steps.append(
                K6GreedyPatchSearchStep(
                    step=step,
                    selected_patch=None,
                    improvement=0.0,
                    value=current_value,
                    one_run_value=one_run_value,
                    normalized_gap_to_one_run=(one_run_value - current_value) / (T ** 0.5),
                    top_prefix_value=top_prefix_value,
                    gap_to_top_prefix=current_value - top_prefix_value,
                    rejected_candidates=rejected,
                    top_failure_summaries=_k6_failure_summary(next_failures),
                )
            )
            break
        patches = patches + (selected_patch,)
        current_value = next_value
        current_policy = make_k6_one_run_patch_policy_from_patches("k6_alive_full_comb_control_policy", patches)
        next_failures = _k6_one_run_candidate_failure_rows_for_policy(
            T,
            "k6_greedy_current",
            current_policy,
            one_run_actions,
            one_run_values,
            interval_rows_by_key,
            support_tolerance=support_tolerance,
        )
        steps.append(
            K6GreedyPatchSearchStep(
                step=step,
                selected_patch=selected_patch,
                improvement=improvement,
                value=next_value,
                one_run_value=one_run_value,
                normalized_gap_to_one_run=(one_run_value - next_value) / (T ** 0.5),
                top_prefix_value=top_prefix_value,
                gap_to_top_prefix=next_value - top_prefix_value,
                rejected_candidates=rejected,
                top_failure_summaries=_k6_failure_summary(next_failures),
            )
        )
    return tuple(steps)


def print_k6_one_run_greedy_patch_search(
    T: int,
    max_patches: int = 20,
    candidate_n: int = 50,
) -> None:
    steps = k6_one_run_greedy_patch_search_steps(T, max_patches=max_patches, candidate_n=candidate_n)
    zero = (0, 0, 0, 0, 0, 0)
    optimal_value = optimal_values(6, T)[T][zero]
    if not steps:
        print("No greedy patch search steps")
        return
    print(f"k=6 one-run greedy patch search, T={T}")
    print()
    print(f"V_star: {optimal_value:.6f}")
    print(f"one_run: {steps[0].one_run_value:.6f}")
    print(f"top_prefix_all: {steps[0].top_prefix_value:.6f}")
    print(f"max_patches: {max_patches}")
    print(f"candidate_n: {candidate_n}")
    print()
    print("Starting patch set:")
    for patch in _k6_one_run_patch_list()[:5]:
        print(
            f"  {patch.name}: rem={patch.remaining_horizon}"
            f" packet={patch.packet_type} gaps={patch.packet_gaps}"
            f" interval={_format_interval(patch.interval)}"
        )
    print()
    selected: list[K6OneRunPatch] = []
    for step in steps:
        print(
            f"step={step.step}"
            f" V={step.value:.6f}"
            f" improvement={step.improvement:.6f}"
            f" gap_to_one_run={step.one_run_value - step.value:.6f}"
            f" gap_to_one_run/sqrt(T)={step.normalized_gap_to_one_run:.6f}"
            f" gap_vs_top_prefix={step.gap_to_top_prefix:.6f}"
        )
        if step.selected_patch is None:
            print("  selected patch: none")
        else:
            selected.append(step.selected_patch)
            patch = step.selected_patch
            print(
                f"  selected patch: {patch.name}"
                f" rem={patch.remaining_horizon}"
                f" packet={patch.packet_type}"
                f" gaps={patch.packet_gaps}"
                f" interval={_format_interval(patch.interval)}"
            )
        if step.rejected_candidates:
            print("  top candidate evaluations:")
            for patch, improvement in step.rejected_candidates[:8]:
                print(
                    f"    {patch.name}: improvement={improvement:.6f}"
                    f" rem={patch.remaining_horizon}"
                    f" packet={patch.packet_type}"
                    f" gaps={patch.packet_gaps}"
                    f" interval={_format_interval(patch.interval)}"
                )
        print("  top remaining failure summaries:")
        for label, loss in step.top_failure_summaries[:8]:
            print(f"    {label}: weighted_loss={loss:.6f}")
        print()
    print("Final selected greedy patches:")
    if selected:
        for patch in selected:
            print(
                f"  {patch.name}: rem={patch.remaining_horizon}"
                f" packet={patch.packet_type}"
                f" gaps={patch.packet_gaps}"
                f" interval={_format_interval(patch.interval)}"
            )
    else:
        print("  none")


def _format_weighted_intervals(
    weighted_intervals: tuple[tuple[float, tuple[int, int]], ...],
) -> str:
    return ", ".join(
        f"{_format_interval(interval)}={_format_fractionish(weight)}"
        for weight, interval in weighted_intervals
    )


def k6_one_run_greedy_mixed_patch_search_steps(
    T: int,
    max_patches: int = 20,
    candidate_n: int = 50,
    support_tolerance: float = 1e-8,
) -> tuple[K6GreedyMixedPatchSearchStep, ...]:
    zero = (0, 0, 0, 0, 0, 0)
    one_run_actions = _strategy_class_library_actions(6, T, "one_run")
    one_run_values = _library_lp_value_layers(6, T, one_run_actions)
    one_run_value = one_run_values[T][zero]
    top_prefix_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "top_prefix_all"))[T][zero]
    interval_rows_by_key = {
        (row.time, row.state): row
        for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance)
    }
    distributions_by_regime = _one_run_interval_distributions_by_regime(T, support_tolerance=support_tolerance)

    deterministic_patches = _k6_one_run_patch_list()[:5] + (
        K6OneRunPatch(
            name="g_rem5_1221_g112_i11",
            remaining_horizon=5,
            packet_type=(1, 2, 2, 1),
            packet_gaps=(1, 1, 2),
            interval=(1, 1),
        ),
    )
    mixed_patches: tuple[MixedIntervalPatch, ...] = ()
    current_policy = make_k6_one_run_mixed_patch_policy(
        "k6_alive_full_comb_control_policy",
        deterministic_patches,
        mixed_patches,
    )
    current_value = evaluate_time_dependent_policy(6, T, current_policy)[T][zero]
    steps: list[K6GreedyMixedPatchSearchStep] = []

    for step in range(max_patches + 1):
        failures = _k6_one_run_candidate_failure_rows_for_policy(
            T,
            "k6_greedy_mixed_current",
            current_policy,
            one_run_actions,
            one_run_values,
            interval_rows_by_key,
            support_tolerance=support_tolerance,
        )
        if step == 0:
            steps.append(
                K6GreedyMixedPatchSearchStep(
                    step=step,
                    selected_patch=None,
                    improvement=0.0,
                    value=current_value,
                    one_run_value=one_run_value,
                    normalized_gap_to_one_run=(one_run_value - current_value) / (T ** 0.5),
                    top_prefix_value=top_prefix_value,
                    gap_to_top_prefix=current_value - top_prefix_value,
                    rejected_candidates=(),
                    top_failure_summaries=_k6_failure_summary(failures),
                )
            )
            if max_patches == 0:
                break
            continue

        candidates = _k6_mixed_patch_candidates_from_failures(
            T,
            failures,
            mixed_patches,
            candidate_n,
            distributions_by_regime=distributions_by_regime,
            support_tolerance=support_tolerance,
        )
        evaluations: list[tuple[MixedIntervalPatch, float]] = []
        best_patch: MixedIntervalPatch | None = None
        best_value = current_value
        for candidate in candidates:
            candidate_policy = make_k6_one_run_mixed_patch_policy(
                "k6_alive_full_comb_control_policy",
                deterministic_patches,
                mixed_patches + (candidate,),
            )
            value = evaluate_time_dependent_policy(6, T, candidate_policy)[T][zero]
            improvement = value - current_value
            evaluations.append((candidate, improvement))
            if improvement > best_value - current_value + 1e-12:
                best_patch = candidate
                best_value = value
        rejected = tuple(sorted(evaluations, key=lambda item: (-item[1], item[0].name))[:10])
        if best_patch is None or best_value <= current_value + 1e-12:
            steps.append(
                K6GreedyMixedPatchSearchStep(
                    step=step,
                    selected_patch=None,
                    improvement=0.0,
                    value=current_value,
                    one_run_value=one_run_value,
                    normalized_gap_to_one_run=(one_run_value - current_value) / (T ** 0.5),
                    top_prefix_value=top_prefix_value,
                    gap_to_top_prefix=current_value - top_prefix_value,
                    rejected_candidates=rejected,
                    top_failure_summaries=_k6_failure_summary(failures),
                )
            )
            break
        improvement = best_value - current_value
        mixed_patches = mixed_patches + (best_patch,)
        current_value = best_value
        current_policy = make_k6_one_run_mixed_patch_policy(
            "k6_alive_full_comb_control_policy",
            deterministic_patches,
            mixed_patches,
        )
        next_failures = _k6_one_run_candidate_failure_rows_for_policy(
            T,
            "k6_greedy_mixed_current",
            current_policy,
            one_run_actions,
            one_run_values,
            interval_rows_by_key,
            support_tolerance=support_tolerance,
        )
        steps.append(
            K6GreedyMixedPatchSearchStep(
                step=step,
                selected_patch=best_patch,
                improvement=improvement,
                value=current_value,
                one_run_value=one_run_value,
                normalized_gap_to_one_run=(one_run_value - current_value) / (T ** 0.5),
                top_prefix_value=top_prefix_value,
                gap_to_top_prefix=current_value - top_prefix_value,
                rejected_candidates=rejected,
                top_failure_summaries=_k6_failure_summary(next_failures),
            )
        )
    return tuple(steps)


def print_k6_one_run_greedy_mixed_patch_search(
    T: int,
    max_patches: int = 20,
    candidate_n: int = 50,
) -> None:
    steps = k6_one_run_greedy_mixed_patch_search_steps(T, max_patches=max_patches, candidate_n=candidate_n)
    zero = (0, 0, 0, 0, 0, 0)
    optimal_value = optimal_values(6, T)[T][zero]
    if not steps:
        print("No greedy mixed patch search steps")
        return
    print(f"k=6 one-run greedy mixed-patch search, T={T}")
    print()
    print(f"V_star: {optimal_value:.6f}")
    print(f"one_run: {steps[0].one_run_value:.6f}")
    print(f"top_prefix_all: {steps[0].top_prefix_value:.6f}")
    print(f"max_patches: {max_patches}")
    print(f"candidate_n: {candidate_n}")
    print()
    print("Starting deterministic patch set:")
    for patch in _k6_one_run_patch_list()[:5]:
        print(
            f"  {patch.name}: rem={patch.remaining_horizon}"
            f" packet={patch.packet_type} gaps={patch.packet_gaps}"
            f" interval={_format_interval(patch.interval)}"
        )
    print("  g_rem5_1221_g112_i11: rem=5 packet=(1, 2, 2, 1) gaps=(1, 1, 2) interval=1:1")
    print()
    selected: list[MixedIntervalPatch] = []
    for step in steps:
        print(
            f"step={step.step}"
            f" V={step.value:.6f}"
            f" improvement={step.improvement:.6f}"
            f" gap_to_one_run={step.one_run_value - step.value:.6f}"
            f" gap_to_one_run/sqrt(T)={step.normalized_gap_to_one_run:.6f}"
            f" gap_vs_top_prefix={step.gap_to_top_prefix:.6f}"
        )
        if step.selected_patch is None:
            print("  selected mixed patch: none")
        else:
            selected.append(step.selected_patch)
            patch = step.selected_patch
            print(
                f"  selected mixed patch: {patch.name}"
                f" rem={patch.remaining_horizon}"
                f" packet={patch.packet_type}"
                f" gaps={patch.packet_gaps}"
            )
            print(f"    weighted intervals: {_format_weighted_intervals(patch.weighted_intervals)}")
        if step.rejected_candidates:
            print("  top candidate evaluations:")
            for patch, improvement in step.rejected_candidates[:8]:
                print(
                    f"    {patch.name}: improvement={improvement:.6f}"
                    f" rem={patch.remaining_horizon}"
                    f" packet={patch.packet_type}"
                    f" gaps={patch.packet_gaps}"
                    f" weighted intervals={_format_weighted_intervals(patch.weighted_intervals)}"
                )
        print("  top remaining failure summaries:")
        for label, loss in step.top_failure_summaries[:8]:
            print(f"    {label}: weighted_loss={loss:.6f}")
        print()
    print("Final selected mixed patches:")
    if selected:
        for patch in selected:
            print(
                f"  {patch.name}: rem={patch.remaining_horizon}"
                f" packet={patch.packet_type}"
                f" gaps={patch.packet_gaps}"
                f" weighted intervals={_format_weighted_intervals(patch.weighted_intervals)}"
            )
    else:
        print("  none (weighted intervals: none)")


def _default_k6_explicit_long_horizon_policies() -> tuple[str, ...]:
    return (
        "top_prefix_all",
        "k6_alive_full_comb_control_policy",
        "k6_alive_trunc_comb_policy",
        "k6_patch_ladder_5",
        "k6_greedy_deterministic_patched_policy",
        "k6_one_run_rule_v1_policy",
        "k6_one_run_rule_v1_no_mixed_policy",
    )


def k6_explicit_long_horizon_sweep_rows(
    T_values: tuple[int, ...],
    policy_names: tuple[str, ...] | None = None,
) -> tuple[K6ExplicitLongHorizonSweepRow, ...]:
    names = policy_names or _default_k6_explicit_long_horizon_policies()
    zero = (0, 0, 0, 0, 0, 0)
    rows: list[K6ExplicitLongHorizonSweepRow] = []
    for T in T_values:
        top_prefix_policy = _resolve_time_dependent_policy(6, "top_prefix_all")
        if top_prefix_policy is None:
            raise ValueError("top_prefix_all policy is not registered")
        top_prefix_value = evaluate_time_dependent_policy(6, T, top_prefix_policy)[T][zero]
        for policy_name in names:
            policy_fn = _resolve_time_dependent_policy(6, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown k=6 explicit policy: {policy_name}")
            if policy_name == "top_prefix_all":
                value = top_prefix_value
            else:
                value = evaluate_time_dependent_policy(6, T, policy_fn)[T][zero]
            gap = top_prefix_value - value
            rows.append(
                K6ExplicitLongHorizonSweepRow(
                    T=T,
                    policy_name=policy_name,
                    value=value,
                    normalized_value=value / (T ** 0.5),
                    top_prefix_value=top_prefix_value,
                    gap_to_top_prefix=gap,
                    normalized_gap_to_top_prefix=gap / (T ** 0.5),
                )
            )
    return tuple(rows)


def print_k6_explicit_long_horizon_sweep(
    T_values: tuple[int, ...],
    policy_names: tuple[str, ...] | None = None,
) -> None:
    names = policy_names or _default_k6_explicit_long_horizon_policies()
    print("k=6 explicit long-horizon sweep")
    print()
    print(f"T values: {T_values}")
    print(f"policies: {names}")
    print()
    print(
        "T policy                                      V_policy"
        " V_policy/sqrt(T) gap_to_top_prefix gap_to_top_prefix/sqrt(T)"
        ,
        flush=True,
    )
    zero = (0, 0, 0, 0, 0, 0)
    for T in T_values:
        top_prefix_policy = _resolve_time_dependent_policy(6, "top_prefix_all")
        if top_prefix_policy is None:
            raise ValueError("top_prefix_all policy is not registered")
        top_prefix_value = evaluate_time_dependent_policy(6, T, top_prefix_policy)[T][zero]
        for policy_name in names:
            policy_fn = _resolve_time_dependent_policy(6, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown k=6 explicit policy: {policy_name}")
            value = top_prefix_value if policy_name == "top_prefix_all" else evaluate_time_dependent_policy(6, T, policy_fn)[T][zero]
            gap = top_prefix_value - value
            print(
                f"{T:3d}"
                f" {policy_name:42s}"
                f" {value:10.6f}"
                f" {value / (T ** 0.5):16.6f}"
                f" {gap:17.6f}"
                f" {gap / (T ** 0.5):26.6f}",
                flush=True,
            )


def _format_visited_by_horizon(
    visited_by_horizon: tuple[tuple[int, int], ...],
    limit: int = 8,
) -> str:
    rendered = ", ".join(f"{h}:{count}" for h, count in visited_by_horizon[:limit])
    if len(visited_by_horizon) > limit:
        rendered += ", ..."
    return rendered


def explicit_time_policy_reachable_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
) -> tuple[ExplicitTimePolicyReachableBenchmarkRow, ...]:
    rows: list[ExplicitTimePolicyReachableBenchmarkRow] = []
    for k, T in cases:
        top_prefix_policy = _resolve_time_dependent_policy(k, "top_prefix_all")
        if top_prefix_policy is None:
            raise ValueError("top_prefix_all policy is not registered")
        top_prefix_result = evaluate_time_dependent_policy_reachable_root(k, T, top_prefix_policy)
        for policy_name in policy_names:
            policy_fn = _resolve_time_dependent_policy(k, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown explicit time policy: {policy_name}")
            start = time.perf_counter()
            result = (
                top_prefix_result
                if policy_name == "top_prefix_all"
                else evaluate_time_dependent_policy_reachable_root(k, T, policy_fn)
            )
            elapsed = time.perf_counter() - start
            gap = top_prefix_result.value - result.value
            rows.append(
                ExplicitTimePolicyReachableBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=result.value,
                    normalized_value=result.value / (T ** 0.5 if T > 0 else 1.0),
                    top_prefix_value=top_prefix_result.value,
                    gap_to_top_prefix=gap,
                    normalized_gap_to_top_prefix=gap / (T ** 0.5 if T > 0 else 1.0),
                    visited_state_count=result.visited_state_count,
                    visited_by_horizon=result.visited_by_horizon,
                    elapsed_seconds=elapsed,
                )
            )
    return tuple(rows)


def print_explicit_time_policy_reachable_benchmark(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...],
) -> None:
    print("Explicit time-policy reachable benchmark")
    print()
    print(f"cases: {cases}")
    print(f"policies: {policy_names}")
    print()
    print(
        "k T policy                                      V_policy"
        " V_policy/sqrt(T) gap_to_top_prefix gap_to_top_prefix/sqrt(T)"
        " visited_state_count elapsed_seconds visited_by_horizon",
        flush=True,
    )
    for k, T in cases:
        top_prefix_policy = _resolve_time_dependent_policy(k, "top_prefix_all")
        if top_prefix_policy is None:
            raise ValueError("top_prefix_all policy is not registered")
        top_prefix_result = evaluate_time_dependent_policy_reachable_root(k, T, top_prefix_policy)
        for policy_name in policy_names:
            policy_fn = _resolve_time_dependent_policy(k, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown explicit time policy: {policy_name}")
            start = time.perf_counter()
            result = (
                top_prefix_result
                if policy_name == "top_prefix_all"
                else evaluate_time_dependent_policy_reachable_root(k, T, policy_fn)
            )
            elapsed = time.perf_counter() - start
            gap = top_prefix_result.value - result.value
            sqrt_T = T ** 0.5 if T > 0 else 1.0
            print(
                f"{k:1d} {T:3d}"
                f" {policy_name:42s}"
                f" {result.value:10.6f}"
                f" {result.value / sqrt_T:16.6f}"
                f" {gap:17.6f}"
                f" {gap / sqrt_T:26.6f}"
                f" {result.visited_state_count:19d}"
                f" {elapsed:15.3f}"
                f" {_format_visited_by_horizon(result.visited_by_horizon)}",
                flush=True,
            )


def _k6_barycentric_trigger_statistics(
    T: int,
    policy_name: str,
    trigger_n: int = 20,
) -> tuple[int, float, tuple[tuple[int, tuple[int, ...], float], ...]]:
    specs = _k6_barycentric_policy_specs()
    if policy_name not in specs:
        return 0, 0.0, ()
    mode, _ = specs[policy_name]
    policy_fn = _resolve_time_dependent_policy(6, policy_name)
    if policy_fn is None:
        raise ValueError(f"unknown k=6 barycentric policy: {policy_name}")
    occupancy = _time_policy_occupancy(6, T, policy_fn)
    trigger_count = 0
    trigger_occupancy = 0.0
    triggered: list[tuple[int, tuple[int, ...], float]] = []
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            if _k6_barycentric_trigger(state, remaining_horizon, mode):
                trigger_count += 1
                trigger_occupancy += probability
                triggered.append((remaining_horizon, state, probability))
    triggered.sort(key=lambda item: (-item[2], -item[0], item[1]))
    return trigger_count, trigger_occupancy, tuple(triggered[:trigger_n])


def k6_barycentric_candidate_benchmark_rows(
    T_values: tuple[int, ...],
    policy_names: tuple[str, ...] | None = None,
    trigger_n: int = 20,
    trigger_stats_max_T: int = 20,
) -> tuple[K6BarycentricBenchmarkRow, ...]:
    names = policy_names or _k6_barycentric_default_policy_names()
    fixed_comb_policy = _resolve_time_dependent_policy(6, "fixed_101010_comb")
    if fixed_comb_policy is None:
        raise ValueError("fixed_101010_comb policy is not registered")

    rows: list[K6BarycentricBenchmarkRow] = []
    for T in T_values:
        sqrt_T = T ** 0.5 if T > 0 else 1.0
        fixed_comb_start = time.perf_counter()
        fixed_comb_result = evaluate_time_dependent_policy_reachable_root(6, T, fixed_comb_policy)
        fixed_comb_elapsed = time.perf_counter() - fixed_comb_start
        fixed_comb_trigger = (
            _k6_barycentric_trigger_statistics(T, "fixed_101010_comb", trigger_n)
            if T <= trigger_stats_max_T
            else (0, 0.0, ())
        )
        for policy_name in names:
            policy_fn = _resolve_time_dependent_policy(6, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown k=6 explicit policy: {policy_name}")
            if policy_name == "fixed_101010_comb":
                result = fixed_comb_result
                elapsed = fixed_comb_elapsed
            else:
                start = time.perf_counter()
                result = evaluate_time_dependent_policy_reachable_root(6, T, policy_fn)
                elapsed = time.perf_counter() - start
            gap = fixed_comb_result.value - result.value
            trigger_count, trigger_occupancy, top_triggered = (
                fixed_comb_trigger
                if policy_name == "fixed_101010_comb"
                else (
                    _k6_barycentric_trigger_statistics(T, policy_name, trigger_n)
                    if T <= trigger_stats_max_T
                    else (0, 0.0, ())
                )
            )
            rows.append(
                K6BarycentricBenchmarkRow(
                    T=T,
                    policy_name=policy_name,
                    value=result.value,
                    normalized_value=result.value / sqrt_T,
                    fixed_comb_value=fixed_comb_result.value,
                    gap_to_fixed_comb=gap,
                    normalized_gap_to_fixed_comb=gap / sqrt_T,
                    visited_state_count=result.visited_state_count,
                    elapsed_seconds=elapsed,
                    trigger_state_count=trigger_count,
                    trigger_occupancy=trigger_occupancy,
                    top_triggered_states=top_triggered,
                )
            )
    return tuple(rows)


def _format_triggered_states(
    triggered_states: tuple[tuple[int, tuple[int, ...], float], ...],
    limit: int = 8,
) -> str:
    if not triggered_states:
        return "-"
    rendered = ", ".join(
        f"rem={remaining} state={state} occ={probability:.4f}"
        for remaining, state, probability in triggered_states[:limit]
    )
    if len(triggered_states) > limit:
        rendered += ", ..."
    return rendered


def print_k6_barycentric_candidate_benchmark(
    T_values: tuple[int, ...],
    policy_names: tuple[str, ...] | None = None,
    trigger_n: int = 20,
    trigger_stats_max_T: int = 20,
) -> None:
    names = policy_names or _k6_barycentric_default_policy_names()
    print("k=6 barycentric candidate benchmark")
    print()
    print(f"T values: {T_values}")
    print(f"policies: {names}")
    print()
    print("Known T=12 LP references from previous certified reports:")
    print("  adaptive top_prefix_all LP: 2.565918")
    print("  one_run LP:                2.618505")
    print("  two_run/local_edges/full:  2.621657")
    print()
    print(
        "T policy                                      V_policy"
        " V_policy/sqrt(T) gap_to_fixed_comb gap_to_fixed_comb/sqrt(T)"
        " visited_state_count elapsed_seconds",
        flush=True,
    )
    rows: list[K6BarycentricBenchmarkRow] = []
    for T in T_values:
        sqrt_T = T ** 0.5 if T > 0 else 1.0
        fixed_comb_policy = _resolve_time_dependent_policy(6, "fixed_101010_comb")
        if fixed_comb_policy is None:
            raise ValueError("fixed_101010_comb policy is not registered")
        fixed_start = time.perf_counter()
        fixed_result = evaluate_time_dependent_policy_reachable_root(6, T, fixed_comb_policy)
        fixed_elapsed = time.perf_counter() - fixed_start
        for policy_name in names:
            policy_fn = _resolve_time_dependent_policy(6, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown k=6 explicit policy: {policy_name}")
            if policy_name == "fixed_101010_comb":
                result = fixed_result
                elapsed = fixed_elapsed
            else:
                start = time.perf_counter()
                result = evaluate_time_dependent_policy_reachable_root(6, T, policy_fn)
                elapsed = time.perf_counter() - start
            trigger_count, trigger_occupancy, top_triggered = (
                _k6_barycentric_trigger_statistics(T, policy_name, trigger_n)
                if T <= trigger_stats_max_T
                else (0, 0.0, ())
            )
            gap = fixed_result.value - result.value
            row = K6BarycentricBenchmarkRow(
                T=T,
                policy_name=policy_name,
                value=result.value,
                normalized_value=result.value / sqrt_T,
                fixed_comb_value=fixed_result.value,
                gap_to_fixed_comb=gap,
                normalized_gap_to_fixed_comb=gap / sqrt_T,
                visited_state_count=result.visited_state_count,
                elapsed_seconds=elapsed,
                trigger_state_count=trigger_count,
                trigger_occupancy=trigger_occupancy,
                top_triggered_states=top_triggered,
            )
            rows.append(row)
            print(
                f"{row.T:3d}"
                f" {row.policy_name:42s}"
                f" {row.value:10.6f}"
                f" {row.normalized_value:16.6f}"
                f" {row.gap_to_fixed_comb:17.6f}"
                f" {row.normalized_gap_to_fixed_comb:28.6f}"
                f" {row.visited_state_count:19d}"
                f" {row.elapsed_seconds:15.3f}",
                flush=True,
            )

    print()
    print("trigger statistics")
    print(f"(computed only for T <= {trigger_stats_max_T})")
    print(
        "T policy                                      trigger_states"
        " trigger_occupancy interval_mixture top_triggered_states"
    )
    specs = _k6_barycentric_policy_specs()
    for row in rows:
        if row.policy_name not in specs:
            continue
        _, variant = specs[row.policy_name]
        print(
            f"{row.T:3d}"
            f" {row.policy_name:42s}"
            f" {row.trigger_state_count:14d}"
            f" {row.trigger_occupancy:17.6f}"
            f" {_format_weighted_intervals(_k6_barycentric_weights(variant)):35s}"
            f" {_format_triggered_states(row.top_triggered_states)}"
        )


def _k6_barycentric_trigger_sweep_stats(
    T: int,
    candidate: K6BarycentricTriggerSweepCandidate,
    policy_fn,
    trigger_n: int = 20,
) -> tuple[int, float, tuple[tuple[int, tuple[int, ...], float], ...]]:
    occupancy = _time_policy_occupancy(6, T, policy_fn)
    trigger_count = 0
    trigger_occupancy = 0.0
    triggered: list[tuple[int, tuple[int, ...], float]] = []
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            if _k6_barycentric_trigger_fires(candidate, state, remaining_horizon):
                trigger_count += 1
                trigger_occupancy += probability
                triggered.append((remaining_horizon, state, probability))
    triggered.sort(key=lambda item: (-item[2], -item[0], item[1]))
    return trigger_count, trigger_occupancy, tuple(triggered[:trigger_n])


def k6_barycentric_trigger_sweep_rows(
    T_values: tuple[int, ...],
    final_T_values: tuple[int, ...] = (),
    trigger_n: int = 20,
    trigger_stats_max_T: int = 20,
    finalist_count: int = 10,
) -> tuple[K6BarycentricTriggerSweepRow, ...]:
    candidates = _k6_barycentric_trigger_sweep_candidates()
    rows: list[K6BarycentricTriggerSweepRow] = []

    def evaluate_candidate(
        T: int,
        candidate: K6BarycentricTriggerSweepCandidate,
        fixed_comb_value: float,
    ) -> K6BarycentricTriggerSweepRow:
        policy_fn = make_k6_barycentric_trigger_sweep_policy(candidate)
        start = time.perf_counter()
        result = evaluate_time_dependent_policy_reachable_root(6, T, policy_fn)
        elapsed = time.perf_counter() - start
        if T <= trigger_stats_max_T:
            trigger_count, trigger_occupancy, _ = _k6_barycentric_trigger_sweep_stats(
                T,
                candidate,
                policy_fn,
                trigger_n=trigger_n,
            )
        else:
            trigger_count, trigger_occupancy = 0, None
        sqrt_T = T ** 0.5 if T > 0 else 1.0
        gap = fixed_comb_value - result.value
        return K6BarycentricTriggerSweepRow(
            candidate_name=candidate.name,
            trigger_name=candidate.trigger_name,
            trigger_description=candidate.trigger_description,
            mixture_name=candidate.mixture_name,
            weighted_intervals=candidate.weighted_intervals,
            T=T,
            value=result.value,
            normalized_value=result.value / sqrt_T,
            fixed_comb_value=fixed_comb_value,
            gap_to_fixed_comb=gap,
            normalized_gap_to_fixed_comb=gap / sqrt_T,
            trigger_state_count=trigger_count,
            trigger_occupancy=trigger_occupancy,
            visited_state_count=result.visited_state_count,
            elapsed_seconds=elapsed,
        )

    fixed_comb_policy = _resolve_time_dependent_policy(6, "fixed_101010_comb")
    if fixed_comb_policy is None:
        raise ValueError("fixed_101010_comb policy is not registered")
    fixed_values: dict[int, float] = {}
    for T in T_values:
        fixed_values[T] = evaluate_time_dependent_policy_reachable_root(6, T, fixed_comb_policy).value
        for candidate in candidates:
            rows.append(evaluate_candidate(T, candidate, fixed_values[T]))

    if final_T_values:
        score: dict[str, float] = {}
        for candidate in candidates:
            candidate_rows = [row for row in rows if row.candidate_name == candidate.name and row.T in T_values]
            if candidate_rows:
                score[candidate.name] = sum(row.normalized_gap_to_fixed_comb for row in candidate_rows) / len(candidate_rows)
        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        finalists = [
            candidate_by_name[name]
            for name, _ in sorted(score.items(), key=lambda item: item[1])[:finalist_count]
        ]
        for T in final_T_values:
            fixed_values[T] = evaluate_time_dependent_policy_reachable_root(6, T, fixed_comb_policy).value
            for candidate in finalists:
                rows.append(evaluate_candidate(T, candidate, fixed_values[T]))

    return tuple(rows)


def _print_k6_barycentric_sweep_row(row: K6BarycentricTriggerSweepRow) -> None:
    print(
        f"{row.T:3d}"
        f" {row.candidate_name:64s}"
        f" {row.mixture_name:22s}"
        f" {row.value:10.6f}"
        f" {row.normalized_value:16.6f}"
        f" {row.gap_to_fixed_comb:17.6f}"
        f" {row.normalized_gap_to_fixed_comb:28.6f}"
        f" {row.trigger_state_count:14d}"
        f" {_format_optional_float(row.trigger_occupancy):>17s}"
        f" {row.visited_state_count:19d}"
        f" {row.elapsed_seconds:15.3f}",
        flush=True,
    )


def _k6_barycentric_sweep_best_by_average(
    rows: tuple[K6BarycentricTriggerSweepRow, ...],
    T_values: tuple[int, ...],
) -> tuple[tuple[str, float], ...]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.T in T_values:
            grouped[row.candidate_name].append(row.normalized_gap_to_fixed_comb)
    return tuple(
        sorted(
            ((name, sum(values) / len(values)) for name, values in grouped.items() if values),
            key=lambda item: item[1],
        )
    )


def _print_k6_barycentric_trigger_examples(
    candidate: K6BarycentricTriggerSweepCandidate,
    T: int,
    trigger_n: int,
) -> None:
    policy_fn = make_k6_barycentric_trigger_sweep_policy(candidate)
    _, trigger_occupancy, triggered_states = _k6_barycentric_trigger_sweep_stats(
        T,
        candidate,
        policy_fn,
        trigger_n=trigger_n,
    )
    print(
        f"  {candidate.name}: trigger_occupancy={trigger_occupancy:.6f}"
        f" mixture={_format_weighted_intervals(candidate.weighted_intervals)}"
    )
    for remaining_horizon, state, probability in triggered_states:
        features = _k6_barycentric_trigger_features(state, remaining_horizon)
        print(
            f"    h={remaining_horizon}"
            f" state={state}"
            f" packet={features['packet_type']}"
            f" gaps={features['packet_gaps']}"
            f" mean={features['mean_score']:.3f}"
            f" occ={probability:.6f}"
            f" reason={candidate.trigger_description}"
        )


def print_k6_barycentric_trigger_sweep(
    T_values: tuple[int, ...],
    final_T_values: tuple[int, ...] = (100,),
    trigger_n: int = 20,
    trigger_stats_max_T: int = 20,
    finalist_count: int = 10,
) -> None:
    print("k=6 barycentric trigger sweep")
    print()
    print(f"T values: {T_values}")
    print(f"final T values: {final_T_values}")
    print(f"candidate count: {len(_k6_barycentric_trigger_sweep_candidates())}")
    print()
    print(
        "T candidate                                                        mixture"
        " V_policy V_policy/sqrt(T) gap_to_fixed_comb gap_to_fixed_comb/sqrt(T)"
        " trigger_states trigger_occupancy visited_state_count elapsed_seconds",
        flush=True,
    )

    rows = k6_barycentric_trigger_sweep_rows(
        T_values,
        final_T_values=final_T_values,
        trigger_n=trigger_n,
        trigger_stats_max_T=trigger_stats_max_T,
        finalist_count=finalist_count,
    )
    for row in rows:
        _print_k6_barycentric_sweep_row(row)

    print()
    for horizon in T_values + final_T_values:
        horizon_rows = [row for row in rows if row.T == horizon]
        if not horizon_rows:
            continue
        best = sorted(horizon_rows, key=lambda row: row.normalized_gap_to_fixed_comb)[:10]
        print(f"Best by T={horizon}")
        for row in best:
            print(
                f"  {row.candidate_name:64s}"
                f" gap/sqrt={row.normalized_gap_to_fixed_comb:.6f}"
                f" V/sqrt={row.normalized_value:.6f}"
                f" trigger={row.trigger_description}"
                f" mixture={_format_weighted_intervals(row.weighted_intervals)}"
            )

    print()
    print("best average normalized gap")
    average_rows = _k6_barycentric_sweep_best_by_average(rows, T_values)
    for name, average_gap in average_rows[:20]:
        candidate_row = next(row for row in rows if row.candidate_name == name)
        print(
            f"  {name:64s}"
            f" avg_gap/sqrt={average_gap:.6f}"
            f" trigger={candidate_row.trigger_description}"
            f" mixture={_format_weighted_intervals(candidate_row.weighted_intervals)}"
        )

    print()
    print("top triggered states")
    candidate_by_name = {candidate.name: candidate for candidate in _k6_barycentric_trigger_sweep_candidates()}
    example_T = 20 if 20 in T_values else T_values[0]
    for name, _ in average_rows[:5]:
        _print_k6_barycentric_trigger_examples(candidate_by_name[name], example_T, trigger_n=min(trigger_n, 20))


def _print_k5_centered_potential_row(row: K5CenteredPotentialComparisonRow) -> None:
    print(
        f"  used={row.used}"
        f" state={row.state}"
        f" packet={row.packet_type}"
        f" gaps={row.packet_gaps}"
        f" phi={row.phi_value:.6f}"
        f" comb_delta={row.comb_delta:.6f}"
        f" chase_delta={row.chase_delta:.6f}"
        f" diff={row.delta_diff:.6f}"
        f" top_gap={row.top_gap:.3f}"
        f" bottom_gap={row.bottom_gap:.3f}"
    )


def print_k5_centered_potential_comb_vs_chase_report(
    max_used: int,
    n: int = 40,
) -> None:
    rows = k5_centered_potential_comb_vs_chase_rows(max_used)
    print("k=5 centered potential COMB vs Chase report")
    print()
    print(f"max_used: {max_used}")
    print("comb support: 10101 / 01010")
    print("chase support: 10100 / 01011")
    print()

    print("Aggregate average delta_diff = delta_chase - delta_comb")
    for potential_name in sorted({row.potential_name for row in rows}):
        potential_rows = [row for row in rows if row.potential_name == potential_name]
        average = sum(row.delta_diff for row in potential_rows) / len(potential_rows)
        positive = sum(1 for row in potential_rows if row.delta_diff > 1e-12)
        negative = sum(1 for row in potential_rows if row.delta_diff < -1e-12)
        print(
            f"  {potential_name:12s}"
            f" avg_diff={average:.6f}"
            f" chase_better={positive}"
            f" comb_better={negative}"
            f" tied={len(potential_rows) - positive - negative}"
        )

    for potential_name in sorted({row.potential_name for row in rows}):
        potential_rows = [row for row in rows if row.potential_name == potential_name]
        print()
        print(f"Top states where Chase beats COMB for {potential_name}")
        for row in sorted(potential_rows, key=lambda item: item.delta_diff, reverse=True)[:n]:
            _print_k5_centered_potential_row(row)
        print()
        print(f"Top states where COMB beats Chase for {potential_name}")
        for row in sorted(potential_rows, key=lambda item: item.delta_diff)[:n]:
            _print_k5_centered_potential_row(row)

    print()
    print("Group by packet_type/gaps")
    group_totals: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (row.potential_name, row.packet_type, row.packet_gaps)
        group_totals[key]["count"] += 1.0
        group_totals[key]["diff"] += row.delta_diff
    for (potential_name, ptype, gaps), totals in sorted(
        group_totals.items(),
        key=lambda item: (-abs(item[1]["diff"] / item[1]["count"]), item[0][0], item[0][1], item[0][2]),
    )[:n]:
        average = totals["diff"] / totals["count"]
        print(
            f"  {potential_name:12s}"
            f" packet={ptype}"
            f" gaps={gaps}"
            f" count={int(totals['count'])}"
            f" avg_diff={average:.6f}"
        )

    print()
    print("Short interpretation counts")
    for potential_name in sorted({row.potential_name for row in rows}):
        potential_rows = [row for row in rows if row.potential_name == potential_name]
        bottom_below_chase = sum(
            1 for row in potential_rows if row.bottom_below_mean and row.delta_diff > 1e-12
        )
        near_leaders_chase = sum(
            1 for row in potential_rows if row.multiple_near_leaders and row.delta_diff > 1e-12
        )
        print(
            f"  {potential_name:12s}"
            f" bottom_below_mean_and_chase_better={bottom_below_chase}"
            f" multiple_near_leaders_and_chase_better={near_leaders_chase}"
        )


def _print_k5_softmax_certificate_row(row: K5SoftmaxPotentialCertificateRow) -> None:
    print(
        f"  h={row.horizon}"
        f" tau={row.tau_name}({row.tau:.3f})"
        f" used={row.used}"
        f" state={row.state}"
        f" packet={row.packet_type}"
        f" gaps={row.packet_gaps}"
        f" max_resid={row.max_residual:.6f}"
        f" centered_max_resid={row.max_centered_residual:.6f}"
        f" max_action={_format_action(row.max_action)}"
        f" chase_avg={row.chase_average_residual:.6f}"
        f" comb_avg={row.comb_average_residual:.6f}"
        f" delta_diff={row.delta_diff:.6f}"
        f" p={_format_float_tuple(row.learner)}"
    )


def print_k5_softmax_potential_certificate_report(
    max_used: int,
    n: int = 40,
) -> None:
    rows = k5_softmax_potential_certificate_rows(max_used)
    print("k=5 softmax potential certificate report")
    print()
    print(f"max_used: {max_used}")
    print("chase support: 10100 / 01011")
    print("comb support: 10101 / 01010")
    print()

    print("Summary by horizon/tau")
    grouped: dict[tuple[int, str], list[K5SoftmaxPotentialCertificateRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.horizon, row.tau_name)].append(row)
    for (horizon, tau_name), group_rows in sorted(grouped.items()):
        max_residual = max(row.max_residual for row in group_rows)
        max_centered = max(row.max_centered_residual for row in group_rows)
        avg_centered = sum(row.max_centered_residual for row in group_rows) / len(group_rows)
        avg_chase_minus_comb = sum(row.chase_average_residual - row.comb_average_residual for row in group_rows) / len(group_rows)
        avg_delta_diff = sum(row.delta_diff for row in group_rows) / len(group_rows)
        chase_growth_better = sum(1 for row in group_rows if row.delta_diff > 1e-12)
        print(
            f"  h={horizon:2d}"
            f" tau={tau_name:16s}"
            f" max_resid={max_residual:.6f}"
            f" max_centered_resid={max_centered:.6f}"
            f" avg_centered_resid={avg_centered:.6f}"
            f" avg_chase_avg_minus_comb_avg={avg_chase_minus_comb:.6f}"
            f" avg_delta_chase_minus_comb={avg_delta_diff:.6f}"
            f" chase_growth_better={chase_growth_better}/{len(group_rows)}"
        )

    print()
    print("Top residual failures")
    for row in sorted(rows, key=lambda item: item.max_centered_residual, reverse=True)[:n]:
        _print_k5_softmax_certificate_row(row)

    print()
    print("Top Chase-vs-COMB growth advantages")
    for row in sorted(rows, key=lambda item: item.delta_diff, reverse=True)[:n]:
        _print_k5_softmax_certificate_row(row)

    print()
    print("Top COMB-vs-Chase growth advantages")
    for row in sorted(rows, key=lambda item: item.delta_diff)[:n]:
        _print_k5_softmax_certificate_row(row)


def _potential_greedy_default_policy_names() -> tuple[str, ...]:
    return (
        "potential_greedy_c0.25",
        "potential_greedy_c0.5",
        "potential_greedy_c1.0",
        "potential_greedy_c2.0",
        "fixed_10101_comb",
        "chase5",
        "fixed_101010_comb",
        "fixed_101011_truncated",
        "k6_barycentric_v1_middle",
    )


def _potential_greedy_baseline_names(k: int) -> tuple[str, ...]:
    if k == 5:
        return ("fixed_10101_comb", "chase5")
    if k == 6:
        return ("fixed_101010_comb", "fixed_101011_truncated", "k6_barycentric_v1_middle")
    return ()


def _potential_greedy_policy_is_compatible(k: int, policy_name: str) -> bool:
    if policy_name.startswith("potential_greedy_c"):
        return k in (5, 6)
    if k == 5:
        return policy_name in {"fixed_10101_comb", "chase5", "top_prefix_all"}
    if k == 6:
        return policy_name in {
            "fixed_101010_comb",
            "fixed_101011_truncated",
            "fixed_101000_prefix3",
            "k6_barycentric_v1_middle",
            "k6_alive_full_comb_control_policy",
            "k6_alive_trunc_comb_policy",
            "top_prefix_all",
        }
    return False


def potential_greedy_interval_benchmark_rows(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...] | None = None,
) -> tuple[PotentialGreedyIntervalBenchmarkRow, ...]:
    names = policy_names or _potential_greedy_default_policy_names()
    rows: list[PotentialGreedyIntervalBenchmarkRow] = []
    baseline_cache: dict[tuple[int, int, str], ReachablePolicyEvaluationResult] = {}

    for k, T in cases:
        baseline_values: list[float] = []
        for baseline_name in _potential_greedy_baseline_names(k):
            policy_fn = _resolve_time_dependent_policy(k, baseline_name)
            if policy_fn is None:
                continue
            result = evaluate_time_dependent_policy_reachable_root(k, T, policy_fn)
            baseline_cache[(k, T, baseline_name)] = result
            baseline_values.append(result.value)
        if not baseline_values:
            raise ValueError(f"no potential-greedy baselines for k={k}")
        best_baseline_value = max(baseline_values)

        for policy_name in names:
            if not _potential_greedy_policy_is_compatible(k, policy_name):
                continue
            policy_fn = _resolve_time_dependent_policy(k, policy_name)
            if policy_fn is None:
                raise ValueError(f"unknown potential-greedy benchmark policy: {policy_name}")
            start = time.perf_counter()
            result = baseline_cache.get((k, T, policy_name))
            if result is None:
                result = evaluate_time_dependent_policy_reachable_root(k, T, policy_fn)
            elapsed = time.perf_counter() - start
            sqrt_T = T ** 0.5 if T > 0 else 1.0
            gap = best_baseline_value - result.value
            rows.append(
                PotentialGreedyIntervalBenchmarkRow(
                    k=k,
                    T=T,
                    policy_name=policy_name,
                    value=result.value,
                    normalized_value=result.value / sqrt_T,
                    best_baseline_value=best_baseline_value,
                    gap_to_best_baseline=gap,
                    normalized_gap_to_best_baseline=gap / sqrt_T,
                    visited_state_count=result.visited_state_count,
                    elapsed_seconds=elapsed,
                )
            )
    return tuple(rows)


def _potential_greedy_c_from_policy_name(policy_name: str) -> float | None:
    if not policy_name.startswith("potential_greedy_c"):
        return None
    return float(policy_name.removeprefix("potential_greedy_c"))


def _potential_greedy_action_stats(
    k: int,
    T: int,
    policy_name: str,
    n: int = 20,
) -> tuple[tuple[str, float], tuple[tuple[int, tuple[int, ...], float, str], ...]]:
    c = _potential_greedy_c_from_policy_name(policy_name)
    if c is None:
        return (), ()
    policy_fn = _resolve_time_dependent_policy(k, policy_name)
    if policy_fn is None:
        return (), ()
    occupancy = _time_policy_occupancy(k, T, policy_fn)
    totals: dict[str, float] = defaultdict(float)
    interesting: list[tuple[int, tuple[int, ...], float, str]] = []
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            choice = _potential_greedy_choice(state, remaining_horizon, c)
            totals[choice.name] += probability
            if (k == 5 and choice.name == "chase_10100") or (k == 6 and choice.name == "top_full_tail"):
                interesting.append((remaining_horizon, state, probability, choice.name))
    histogram = tuple(sorted(totals.items(), key=lambda item: (-item[1], item[0])))
    interesting.sort(key=lambda item: (-item[2], -item[0], item[1]))
    return histogram, tuple(interesting[:n])


def print_potential_greedy_interval_benchmark(
    cases: tuple[tuple[int, int], ...],
    policy_names: tuple[str, ...] | None = None,
    print_action_stats: bool = False,
    n: int = 20,
) -> None:
    names = policy_names or _potential_greedy_default_policy_names()
    print("potential-greedy interval benchmark")
    print()
    print(f"cases: {cases}")
    print(f"policies: {names}")
    print()
    print(
        "k T policy                         V_policy V_policy/sqrt(T)"
        " gap_to_best_baseline gap_to_best_baseline/sqrt(T)"
        " visited_state_count elapsed_seconds",
        flush=True,
    )
    rows = potential_greedy_interval_benchmark_rows(cases, names)
    for row in rows:
        print(
            f"{row.k:1d} {row.T:3d}"
            f" {row.policy_name:30s}"
            f" {row.value:10.6f}"
            f" {row.normalized_value:16.6f}"
            f" {row.gap_to_best_baseline:20.6f}"
            f" {row.normalized_gap_to_best_baseline:32.6f}"
            f" {row.visited_state_count:19d}"
            f" {row.elapsed_seconds:15.3f}",
            flush=True,
        )

    if not print_action_stats:
        return
    print()
    print("action stats")
    for k, T in cases:
        if T > 50:
            continue
        for policy_name in names:
            if not policy_name.startswith("potential_greedy_c") or not _potential_greedy_policy_is_compatible(k, policy_name):
                continue
            histogram, interesting = _potential_greedy_action_stats(k, T, policy_name, n=n)
            print(f"k={k} T={T} policy={policy_name}")
            print("  grammar occupancy:")
            for action_name, mass in histogram[:n]:
                print(f"    {action_name:20s} {mass:.6f}")
            if interesting:
                print("  top interesting states:")
                for remaining_horizon, state, probability, action_name in interesting:
                    print(
                        f"    h={remaining_horizon}"
                        f" state={state}"
                        f" occ={probability:.6f}"
                        f" choice={action_name}"
                    )


def _random_playout_spec(k: int, playout_name: str) -> RandomPlayoutSpec:
    grammar = {candidate.name: candidate.support for candidate in _potential_greedy_interval_grammar(k)}
    if k == 5:
        weights_by_name: dict[str, tuple[tuple[str, float], ...]] = {
            "Q_comb": (("comb_10101", 1.0),),
            "Q_chase": (("chase_10100", 1.0),),
            "Q_uniform_one_run": tuple((name, 1.0) for name in grammar),
        }
    elif k == 6:
        weights_by_name = {
            "Q_comb": (("interval_0_4", 1.0),),
            "Q_uniform_core_intervals": (
                ("interval_0_4", 1.0),
                ("interval_0_0", 1.0),
                ("interval_1_1", 1.0),
                ("interval_0_2", 1.0),
            ),
            "Q_potential_greedy_empirical_c1_like": (
                ("interval_0_4", 21.932539),
                ("interval_0_0", 12.191217),
                ("interval_1_1", 8.073744),
                ("interval_0_2", 7.802500),
            ),
        }
    else:
        weights_by_name = {}
    if playout_name not in weights_by_name:
        raise ValueError(f"unknown random playout {playout_name!r} for k={k}")
    raw_weights = weights_by_name[playout_name]
    total = sum(weight for _, weight in raw_weights)
    grammar_weights = tuple((name, weight / total) for name, weight in raw_weights)
    terms: list[tuple[float, tuple[int, ...]]] = []
    for grammar_name, grammar_weight in grammar_weights:
        if grammar_name not in grammar:
            raise ValueError(f"grammar action {grammar_name!r} is not available for k={k}")
        for action_probability, action in grammar[grammar_name]:
            terms.append((grammar_weight * action_probability, action))
    return RandomPlayoutSpec(
        name=playout_name,
        grammar_weights=grammar_weights,
        support=_policy_from_weighted_actions(tuple(terms)),
    )


def _random_playout_names_for_k(k: int) -> tuple[str, ...]:
    if k == 5:
        return ("Q_comb", "Q_chase", "Q_uniform_one_run")
    if k == 6:
        return ("Q_comb", "Q_uniform_core_intervals", "Q_potential_greedy_empirical_c1_like")
    return ()


def _parse_random_playout_names(text: str | None) -> tuple[str, ...] | None:
    if text is None or not text.strip() or text.strip() == "all":
        return None
    return tuple(piece.strip() for piece in text.split(",") if piece.strip())


@lru_cache(maxsize=None)
def _random_playout_value_function(k: int, T: int, playout_name: str):
    spec = _random_playout_spec(k, playout_name)

    @lru_cache(maxsize=None)
    def value(state: tuple[int, ...], remaining_horizon: int) -> float:
        canonical_state = canon(state)
        if remaining_horizon == 0:
            return float(max(canonical_state, default=0))
        expected = 0.0
        for probability, action in spec.support:
            raw_next = tuple(canonical_state[index] + action[index] for index in range(k))
            expected += probability * (
                float(min(raw_next, default=0)) + value(canon(raw_next), remaining_horizon - 1)
            )
        return expected - 0.5

    return value


def _random_playout_terminal_winner_probabilities(
    k: int,
    playout_name: str,
    state: tuple[int, ...],
    remaining_horizon: int,
):
    spec = _random_playout_spec(k, playout_name)

    @lru_cache(maxsize=None)
    def winner_probabilities(canonical_state: tuple[int, ...], horizon: int) -> tuple[float, ...]:
        canonical_state = canon(canonical_state)
        if horizon == 0:
            return _terminal_named_winner_probabilities(canonical_state)
        totals = [0.0] * k
        for probability, action in spec.support:
            next_state = canon(tuple(canonical_state[index] + action[index] for index in range(k)))
            child = winner_probabilities(next_state, horizon - 1)
            for index, winner_probability in enumerate(child):
                totals[index] += probability * winner_probability
        return tuple(totals)

    return winner_probabilities(canon(state), remaining_horizon)


def make_random_playout_potential_greedy_policy(k: int, T: int, playout_name: str):
    value = _random_playout_value_function(k, T, playout_name)
    grammar = _potential_greedy_interval_grammar(k)
    if not grammar:
        raise ValueError(f"no potential-greedy grammar for k={k}")

    def policy(
        state: tuple[int, ...],
        remaining_horizon: int,
    ) -> tuple[tuple[float, tuple[int, ...]], ...]:
        canonical_state = canon(state)
        if len(canonical_state) != k:
            raise ValueError(f"random-playout greedy k={k} received state of length {len(canonical_state)}")

        def score(candidate: PotentialGreedyGrammarCandidate) -> float:
            expected = 0.0
            for probability, action in candidate.support:
                raw_next = tuple(canonical_state[index] + action[index] for index in range(k))
                expected += probability * (
                    float(min(raw_next, default=0)) + value(canon(raw_next), remaining_horizon - 1)
                )
            return expected

        return max(grammar, key=lambda candidate: (score(candidate), -len(candidate.support), candidate.name)).support

    return policy


def _random_playout_greedy_choice_name(
    k: int,
    T: int,
    playout_name: str,
    state: tuple[int, ...],
    remaining_horizon: int,
) -> str:
    value = _random_playout_value_function(k, T, playout_name)
    grammar = _potential_greedy_interval_grammar(k)
    canonical_state = canon(state)

    def score(candidate: PotentialGreedyGrammarCandidate) -> float:
        expected = 0.0
        for probability, action in candidate.support:
            raw_next = tuple(canonical_state[index] + action[index] for index in range(k))
            expected += probability * (
                float(min(raw_next, default=0)) + value(canon(raw_next), remaining_horizon - 1)
            )
        return expected

    return max(grammar, key=lambda candidate: (score(candidate), -len(candidate.support), candidate.name)).name


def _random_playout_baseline_value(k: int, T: int) -> float:
    baseline_names = ("fixed_10101_comb", "chase5") if k == 5 else (
        "fixed_101010_comb",
        "fixed_101011_truncated",
        "potential_greedy_c1.0",
    )
    values: list[float] = []
    for policy_name in baseline_names:
        policy_fn = _resolve_time_dependent_policy(k, policy_name)
        if policy_fn is None:
            continue
        values.append(evaluate_time_dependent_policy_reachable_root(k, T, policy_fn).value)
    if not values:
        raise ValueError(f"no random-playout baselines for k={k}")
    return max(values)


def _random_playout_names_for_case(k: int, requested_names: tuple[str, ...] | None) -> tuple[str, ...]:
    available = _random_playout_names_for_k(k)
    if requested_names is None:
        return available
    return tuple(name for name in requested_names if name in available)


def random_playout_value_rows(
    cases: tuple[tuple[int, int], ...],
    playout_names: tuple[str, ...] | None = None,
) -> tuple[RandomPlayoutValueRow, ...]:
    rows: list[RandomPlayoutValueRow] = []
    for k, T in cases:
        zero = tuple(0 for _ in range(k))
        for playout_name in _random_playout_names_for_case(k, playout_names):
            value = _random_playout_value_function(k, T, playout_name)(zero, T)
            sqrt_T = T ** 0.5 if T > 0 else 1.0
            rows.append(RandomPlayoutValueRow(k, T, playout_name, value, value / sqrt_T))
    return tuple(rows)


def random_playout_greedy_rows(
    cases: tuple[tuple[int, int], ...],
    playout_names: tuple[str, ...] | None = None,
) -> tuple[RandomPlayoutGreedyRow, ...]:
    rows: list[RandomPlayoutGreedyRow] = []
    for k, T in cases:
        baseline_value = _random_playout_baseline_value(k, T)
        for playout_name in _random_playout_names_for_case(k, playout_names):
            policy_fn = make_random_playout_potential_greedy_policy(k, T, playout_name)
            start = time.perf_counter()
            result = evaluate_time_dependent_policy_reachable_root(k, T, policy_fn)
            elapsed = time.perf_counter() - start
            sqrt_T = T ** 0.5 if T > 0 else 1.0
            gap = baseline_value - result.value
            rows.append(
                RandomPlayoutGreedyRow(
                    k=k,
                    T=T,
                    playout_name=playout_name,
                    value=result.value,
                    normalized_value=result.value / sqrt_T,
                    best_baseline_value=baseline_value,
                    gap_to_best_baseline=gap,
                    normalized_gap_to_best_baseline=gap / sqrt_T,
                    visited_state_count=result.visited_state_count,
                    elapsed_seconds=elapsed,
                )
            )
    return tuple(rows)


def random_playout_residual_rows(
    cases: tuple[tuple[int, int], ...],
    playout_names: tuple[str, ...] | None = None,
    residual_max_T: int = 12,
    n: int = 20,
) -> tuple[RandomPlayoutResidualRow, ...]:
    rows: list[RandomPlayoutResidualRow] = []
    for k, T in cases:
        if T > residual_max_T:
            continue
        all_gain_actions = all_actions(k)
        for playout_name in _random_playout_names_for_case(k, playout_names):
            value = _random_playout_value_function(k, T, playout_name)
            for remaining_horizon in range(1, T + 1):
                used = T - remaining_horizon
                for state in all_states(k, used):
                    canonical_state = canon(state)
                    learner = _random_playout_terminal_winner_probabilities(k, playout_name, canonical_state, remaining_horizon)
                    residuals = []
                    for action in all_gain_actions:
                        raw_next = tuple(canonical_state[index] + action[index] for index in range(k))
                        residual = (
                            float(min(raw_next, default=0))
                            + value(canon(raw_next), remaining_horizon - 1)
                            - sum(learner[index] * action[index] for index in range(k))
                            - value(canonical_state, remaining_horizon)
                        )
                        residuals.append((residual, action))
                    max_residual, max_action = max(residuals, key=lambda item: item[0])
                    rows.append(
                        RandomPlayoutResidualRow(
                            k=k,
                            T=T,
                            playout_name=playout_name,
                            remaining_horizon=remaining_horizon,
                            state=canonical_state,
                            max_residual=max_residual,
                            max_action=max_action,
                            learner=learner,
                        )
                    )
    return tuple(sorted(rows, key=lambda row: row.max_residual, reverse=True)[:n])


def _random_playout_action_stats(
    k: int,
    T: int,
    playout_name: str,
    n: int = 20,
) -> tuple[tuple[str, float], ...]:
    policy_fn = make_random_playout_potential_greedy_policy(k, T, playout_name)
    occupancy = _time_policy_occupancy(k, T, policy_fn)
    totals: dict[str, float] = defaultdict(float)
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            choice_name = _random_playout_greedy_choice_name(k, T, playout_name, state, remaining_horizon)
            totals[choice_name] += probability
    return tuple(sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:n])


def print_random_playout_potential_benchmark(
    cases: tuple[tuple[int, int], ...],
    playout_names: tuple[str, ...] | None = None,
    mode: str = "all",
    print_action_stats: bool = False,
    residual_max_T: int = 12,
    n: int = 20,
) -> None:
    print("random playout potential benchmark")
    print()
    print(f"cases: {cases}")
    print(f"playouts: {playout_names or 'all'}")
    print(f"mode: {mode}")

    if mode in {"values", "all"}:
        print()
        print("Playout values")
        print("k T playout                              Phi_Q Phi_Q/sqrt(T)")
        for row in random_playout_value_rows(cases, playout_names):
            print(f"{row.k:1d} {row.T:3d} {row.playout_name:34s} {row.value:10.6f} {row.normalized_value:13.6f}")

    if mode in {"greedy", "all"}:
        print()
        print("Potential-greedy values")
        print(
            "k T playout                              V_policy V_policy/sqrt(T)"
            " gap_to_best_baseline gap_to_best_baseline/sqrt(T) visited_states elapsed"
        )
        greedy_rows = random_playout_greedy_rows(cases, playout_names)
        for row in greedy_rows:
            print(
                f"{row.k:1d} {row.T:3d} {row.playout_name:34s}"
                f" {row.value:10.6f}"
                f" {row.normalized_value:16.6f}"
                f" {row.gap_to_best_baseline:20.6f}"
                f" {row.normalized_gap_to_best_baseline:32.6f}"
                f" {row.visited_state_count:14d}"
                f" {row.elapsed_seconds:7.3f}"
            )
        if print_action_stats:
            print()
            print("Action stats")
            for k, T in cases:
                if T > 50:
                    continue
                for playout_name in _random_playout_names_for_case(k, playout_names):
                    print(f"k={k} T={T} playout={playout_name}")
                    for action_name, mass in _random_playout_action_stats(k, T, playout_name, n=n):
                        print(f"  {action_name:24s} {mass:.6f}")

    if mode in {"residuals", "all"}:
        print()
        print("Probability-matching residuals")
        print(f"(computed only for T <= {residual_max_T})")
        for row in random_playout_residual_rows(cases, playout_names, residual_max_T=residual_max_T, n=n):
            print(
                f"  k={row.k} T={row.T}"
                f" playout={row.playout_name}"
                f" h={row.remaining_horizon}"
                f" state={row.state}"
                f" max_residual={row.max_residual:.6f}"
                f" action={_format_action(row.max_action)}"
                f" p={_format_float_tuple(row.learner)}"
            )


def _k5_gap_vector(state: tuple[int, ...]) -> tuple[int, ...]:
    canonical_state = canon(state)
    if len(canonical_state) != 5:
        raise ValueError(f"k=5 gap vector expected length 5, got {len(canonical_state)}")
    return tuple(canonical_state[index] - canonical_state[index + 1] for index in range(4))


def _k5_proof_action_pairs() -> dict[str, tuple[tuple[float, tuple[int, ...]], ...]]:
    return {
        "comb_10101": _balanced_pair_from_action_text("10101"),
        "chase_10100": _balanced_pair_from_action_text("10100"),
        "one_run_11000": _balanced_pair_from_action_text("11000"),
        "one_run_10110": _balanced_pair_from_action_text("10110"),
        "one_run_10010": _balanced_pair_from_action_text("10010"),
        "one_run_10001": _balanced_pair_from_action_text("10001"),
    }


def _next_state_distribution(
    state: tuple[int, ...],
    support: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[tuple[int, ...], float], ...]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for probability, action in support:
        next_state = canon(tuple(state[index] + action[index] for index in range(len(state))))
        totals[next_state] += probability
    return tuple(sorted(totals.items(), key=lambda item: item[0]))


def _gap_distribution(
    state_distribution: tuple[tuple[tuple[int, ...], float], ...],
) -> tuple[tuple[tuple[int, ...], float], ...]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for state, probability in state_distribution:
        totals[_k5_gap_vector(state)] += probability
    return tuple(sorted(totals.items(), key=lambda item: item[0]))


def k5_gap_transition_rows(max_used: int) -> tuple[K5GapTransitionRow, ...]:
    rows: list[K5GapTransitionRow] = []
    action_pairs = _k5_proof_action_pairs()
    for used in range(max_used + 1):
        for state in all_states(5, used):
            canonical_state = canon(state)
            for action_name, support in action_pairs.items():
                next_states = _next_state_distribution(canonical_state, support)
                rows.append(
                    K5GapTransitionRow(
                        state=canonical_state,
                        gap_vector=_k5_gap_vector(canonical_state),
                        packet_type=packet_type(canonical_state),
                        packet_gaps=_packet_gaps(canonical_state),
                        action_name=action_name,
                        next_states=next_states,
                        next_gap_vectors=_gap_distribution(next_states),
                    )
                )
    return tuple(rows)


def _two_step_distribution(
    state: tuple[int, ...],
    first_support: tuple[tuple[float, tuple[int, ...]], ...],
    second_support: tuple[tuple[float, tuple[int, ...]], ...],
) -> dict[tuple[int, ...], float]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for first_probability, first_action in first_support:
        first_state = canon(tuple(state[index] + first_action[index] for index in range(len(state))))
        for second_probability, second_action in second_support:
            second_state = canon(tuple(first_state[index] + second_action[index] for index in range(len(state))))
            totals[second_state] += first_probability * second_probability
    return dict(totals)


def _total_variation_distance(
    left: dict[tuple[int, ...], float],
    right: dict[tuple[int, ...], float],
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def k5_exchangeability_rows(max_used: int) -> tuple[K5ExchangeabilityRow, ...]:
    action_pairs = _k5_proof_action_pairs()
    rows: list[K5ExchangeabilityRow] = []
    for action_a, support_a in action_pairs.items():
        for action_w, support_w in action_pairs.items():
            max_tv = -1.0
            worst_state = (0, 0, 0, 0, 0)
            for used in range(max_used + 1):
                for state in all_states(5, used):
                    canonical_state = canon(state)
                    aw = _two_step_distribution(canonical_state, support_a, support_w)
                    wa = _two_step_distribution(canonical_state, support_w, support_a)
                    tv = _total_variation_distance(aw, wa)
                    if tv > max_tv:
                        max_tv = tv
                        worst_state = canonical_state
            rows.append(
                K5ExchangeabilityRow(
                    action_a=action_a,
                    action_w=action_w,
                    max_tv_error=max_tv,
                    worst_state=worst_state,
                    worst_gap_vector=_k5_gap_vector(worst_state),
                    worst_packet_type=packet_type(worst_state),
                )
            )
    return tuple(rows)


def _expected_two_step_potential_value(
    state: tuple[int, ...],
    first_support: tuple[tuple[float, tuple[int, ...]], ...],
    second_support: tuple[tuple[float, tuple[int, ...]], ...],
    continuation_value,
    continuation_horizon: int,
) -> float:
    expected = 0.0
    for first_probability, first_action in first_support:
        first_state = canon(tuple(state[index] + first_action[index] for index in range(len(state))))
        for second_probability, second_action in second_support:
            raw_second = tuple(first_state[index] + second_action[index] for index in range(len(state)))
            expected += first_probability * second_probability * (
                float(min(raw_second, default=0)) + continuation_value(canon(raw_second), continuation_horizon)
            )
    return expected


def k5_scalar_potential_exchangeability_rows(
    max_used: int,
    h_values: tuple[int, ...] = (5, 10, 20),
    playout_names: tuple[str, ...] = ("Q_chase", "Q_comb"),
) -> tuple[K5ScalarPotentialExchangeabilityRow, ...]:
    action_pairs = _k5_proof_action_pairs()
    rows: list[K5ScalarPotentialExchangeabilityRow] = []
    for playout_name in playout_names:
        for horizon in h_values:
            value_fn = _random_playout_value_function(5, horizon, playout_name)
            continuation_horizon = max(horizon - 2, 0)
            for action_a, support_a in action_pairs.items():
                for action_w, support_w in action_pairs.items():
                    max_abs_error = -1.0
                    worst_state = (0, 0, 0, 0, 0)
                    worst_values = (0.0, 0.0)
                    for used in range(max_used + 1):
                        for state in all_states(5, used):
                            canonical_state = canon(state)
                            aw_value = _expected_two_step_potential_value(
                                canonical_state,
                                support_a,
                                support_w,
                                value_fn,
                                continuation_horizon,
                            )
                            wa_value = _expected_two_step_potential_value(
                                canonical_state,
                                support_w,
                                support_a,
                                value_fn,
                                continuation_horizon,
                            )
                            error = abs(aw_value - wa_value)
                            if error > max_abs_error:
                                max_abs_error = error
                                worst_state = canonical_state
                                worst_values = (aw_value, wa_value)
                    rows.append(
                        K5ScalarPotentialExchangeabilityRow(
                            playout_name=playout_name,
                            horizon=horizon,
                            action_a=action_a,
                            action_w=action_w,
                            max_abs_error=max_abs_error,
                            worst_state=worst_state,
                            worst_gap_vector=_k5_gap_vector(worst_state),
                            worst_packet_type=packet_type(worst_state),
                            aw_value=worst_values[0],
                            wa_value=worst_values[1],
                        )
                    )
    return tuple(rows)


def _format_state_distribution(distribution: tuple[tuple[tuple[int, ...], float], ...]) -> str:
    return ", ".join(f"{state}:{probability:.3f}" for state, probability in distribution)


def print_k5_gap_proof_diagnostics(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10),
    n: int = 40,
) -> None:
    print("k=5 gap proof diagnostics")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print()

    print("1. Gap transition summaries")
    transition_rows = k5_gap_transition_rows(max_used)
    for row in transition_rows[:n]:
        print(
            f"  state={row.state}"
            f" gaps={row.gap_vector}"
            f" packet={row.packet_type}"
            f" action={row.action_name}"
            f" next_gaps={_format_state_distribution(row.next_gap_vectors)}"
        )
    transition_groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
    for row in transition_rows:
        transition_groups[(row.action_name, row.packet_type, row.packet_gaps)] += 1
    print("  top transition regimes:")
    for (action_name, ptype, gaps), count in sorted(transition_groups.items(), key=lambda item: (-item[1], item[0]))[:n]:
        print(f"    action={action_name:16s} packet={ptype} gaps={gaps} count={count}")

    print()
    print("2. Chase-vs-comb potential growth by gaps")
    centered_rows = k5_centered_potential_comb_vs_chase_rows(max_used)
    for potential_name in sorted({row.potential_name for row in centered_rows}):
        potential_rows = [row for row in centered_rows if row.potential_name == potential_name]
        print(f"  {potential_name}: Chase beats COMB")
        for row in sorted(potential_rows, key=lambda item: item.delta_diff, reverse=True)[:n]:
            print(
                f"    state={row.state}"
                f" gaps={_k5_gap_vector(row.state)}"
                f" packet={row.packet_type}"
                f" diff={row.delta_diff:.6f}"
                f" chase_delta={row.chase_delta:.6f}"
                f" comb_delta={row.comb_delta:.6f}"
            )
        print(f"  {potential_name}: COMB beats Chase")
        for row in sorted(potential_rows, key=lambda item: item.delta_diff)[:n]:
            print(
                f"    state={row.state}"
                f" gaps={_k5_gap_vector(row.state)}"
                f" packet={row.packet_type}"
                f" diff={row.delta_diff:.6f}"
            )

    print()
    print("  random-playout potential growth by gaps")
    for h in h_values:
        for playout_name in ("Q_comb", "Q_chase"):
            value_fn = _random_playout_value_function(5, h, playout_name)
            growth_rows: list[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
            for used in range(max_used + 1):
                for state in all_states(5, used):
                    canonical_state = canon(state)
                    comb_growth = sum(
                        probability * value_fn(canon(tuple(canonical_state[index] + action[index] for index in range(5))), max(h - 1, 0))
                        for probability, action in _k5_comb_support()
                    )
                    chase_growth = sum(
                        probability * value_fn(canon(tuple(canonical_state[index] + action[index] for index in range(5))), max(h - 1, 0))
                        for probability, action in _k5_chase_support()
                    )
                    growth_rows.append((chase_growth - comb_growth, canonical_state, _k5_gap_vector(canonical_state), packet_type(canonical_state)))
            print(f"    h={h} playout={playout_name}: Chase beats COMB")
            for diff, state, gaps, ptype in sorted(growth_rows, key=lambda item: item[0], reverse=True)[:n]:
                print(f"      state={state} gaps={gaps} packet={ptype} diff={diff:.6f}")
            print(f"    h={h} playout={playout_name}: COMB beats Chase")
            for diff, state, gaps, ptype in sorted(growth_rows, key=lambda item: item[0])[:n]:
                print(f"      state={state} gaps={gaps} packet={ptype} diff={diff:.6f}")

    print()
    print("3. certificate failures by gaps")
    for h in h_values:
        summary = chase5_potential_certificate_summary(h, n=n)
        print(f"  h={h} max_pm={summary.max_pm_residual:.6f} max_repaired={summary.max_repaired_alpha:.6f}")
        print("  top PM fail / repaired success rows:")
        for row in summary.pm_fail_repaired_success_rows[:n]:
            print(
                f"    rem={row.remaining_horizon}"
                f" state={row.state}"
                f" gaps={_k5_gap_vector(row.state)}"
                f" packet={packet_type(row.state)}"
                f" pm_resid={row.pm_max_residual:.6f}"
                f" pm_action={_format_action(row.pm_max_action)}"
                f" repaired_alpha={row.repaired_alpha:.6f}"
            )
        print("  top repaired failure rows:")
        for row in summary.repaired_failure_rows[:n]:
            print(
                f"    rem={row.remaining_horizon}"
                f" state={row.state}"
                f" gaps={_k5_gap_vector(row.state)}"
                f" packet={packet_type(row.state)}"
                f" alpha={row.repaired_alpha:.6f}"
                f" action={_format_action(row.repaired_max_action)}"
            )

    print()
    print("4. exchangeability mini-test")
    exchange_rows = k5_exchangeability_rows(max_used)
    for action_a in sorted({row.action_a for row in exchange_rows}):
        candidates = [row for row in exchange_rows if row.action_a == action_a]
        best = min(candidates, key=lambda row: row.max_tv_error)
        print(
            f"  A={action_a:16s}"
            f" best_W={best.action_w:16s}"
            f" max_TV={best.max_tv_error:.6f}"
            f" worst_state={best.worst_state}"
            f" gaps={best.worst_gap_vector}"
            f" packet={best.worst_packet_type}"
        )
    print("  largest exceptional pairwise errors:")
    for row in sorted(exchange_rows, key=lambda item: item.max_tv_error, reverse=True)[:n]:
        print(
            f"    A={row.action_a:16s}"
            f" W={row.action_w:16s}"
            f" max_TV={row.max_tv_error:.6f}"
            f" worst_state={row.worst_state}"
            f" gaps={row.worst_gap_vector}"
            f" packet={row.worst_packet_type}"
        )


def print_k5_scalar_potential_exchangeability(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20),
    playout_names: tuple[str, ...] = ("Q_chase", "Q_comb"),
    n: int = 40,
) -> None:
    rows = k5_scalar_potential_exchangeability_rows(
        max_used=max_used,
        h_values=h_values,
        playout_names=playout_names,
    )
    print("k=5 scalar potential exchangeability")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"playouts: {playout_names}")
    print()

    print("Summary by playout/horizon")
    for playout_name in playout_names:
        for horizon in h_values:
            group = [row for row in rows if row.playout_name == playout_name and row.horizon == horizon]
            if not group:
                continue
            max_error = max(row.max_abs_error for row in group)
            avg_error = sum(row.max_abs_error for row in group) / len(group)
            near_zero = sum(1 for row in group if row.max_abs_error <= 1e-9)
            print(
                f"  playout={playout_name:8s}"
                f" h={horizon:3d}"
                f" max_error={max_error:.6f}"
                f" avg_pair_error={avg_error:.6f}"
                f" exact_pairs={near_zero}/{len(group)}"
            )

    print()
    print("Best scalar commutator W for each A")
    for playout_name in playout_names:
        for horizon in h_values:
            print(f"  playout={playout_name} h={horizon}")
            group = [row for row in rows if row.playout_name == playout_name and row.horizon == horizon]
            for action_a in sorted({row.action_a for row in group}):
                best = min((row for row in group if row.action_a == action_a), key=lambda row: row.max_abs_error)
                print(
                    f"    A={action_a:16s}"
                    f" best_W={best.action_w:16s}"
                    f" max_abs_error={best.max_abs_error:.6f}"
                    f" worst_state={best.worst_state}"
                    f" gaps={best.worst_gap_vector}"
                    f" packet={best.worst_packet_type}"
                )

    print()
    print("Top scalar potential order failures")
    for row in sorted(rows, key=lambda item: item.max_abs_error, reverse=True)[:n]:
        print(
            f"  playout={row.playout_name:8s}"
            f" h={row.horizon:3d}"
            f" A={row.action_a:16s}"
            f" W={row.action_w:16s}"
            f" error={row.max_abs_error:.6f}"
            f" AW={row.aw_value:.6f}"
            f" WA={row.wa_value:.6f}"
            f" state={row.worst_state}"
            f" gaps={row.worst_gap_vector}"
            f" packet={row.worst_packet_type}"
        )

    print()
    print("Failures by gap/packet")
    grouped: dict[tuple[str, int, tuple[int, ...], tuple[int, ...]], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.playout_name, row.horizon, row.worst_gap_vector, row.worst_packet_type)].append(row.max_abs_error)
    for (playout_name, horizon, gaps, ptype), errors in sorted(
        grouped.items(),
        key=lambda item: (-max(item[1]), item[0]),
    )[:n]:
        print(
            f"  playout={playout_name:8s}"
            f" h={horizon:3d}"
            f" gaps={gaps}"
            f" packet={ptype}"
            f" max_error={max(errors):.6f}"
            f" count={len(errors)}"
        )


_K5_BOUNDARY_FEATURE_FAMILIES = (
    "packet_type",
    "zero_gap_pattern",
    "packet_zero",
    "gap_vector",
    "packet_h",
    "zero_h",
)


def _parse_k5_boundary_feature_families(text: str | None) -> tuple[str, ...]:
    if text is None or text.strip() == "" or text.strip() == "all":
        return _K5_BOUNDARY_FEATURE_FAMILIES
    families = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [family for family in families if family not in _K5_BOUNDARY_FEATURE_FAMILIES]
    if unknown:
        raise ValueError(f"unknown boundary feature families: {unknown}")
    return families


def _k5_zero_gap_pattern(state: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if gap == 0 else 0 for gap in _k5_gap_vector(state))


def _k5_boundary_feature_key(
    state: tuple[int, ...],
    horizon: int,
    feature_family: str,
) -> tuple[object, ...]:
    canonical_state = canon(state)
    ptype = packet_type(canonical_state)
    zero_pattern = _k5_zero_gap_pattern(canonical_state)
    gaps = _k5_gap_vector(canonical_state)
    if feature_family == "packet_type":
        return ("packet", ptype)
    if feature_family == "zero_gap_pattern":
        return ("zero", zero_pattern)
    if feature_family == "packet_zero":
        return ("packet_zero", ptype, zero_pattern)
    if feature_family == "gap_vector":
        return ("gap", gaps)
    if feature_family == "packet_h":
        return ("packet_h", horizon, ptype)
    if feature_family == "zero_h":
        return ("zero_h", horizon, zero_pattern)
    raise ValueError(f"unknown boundary feature family: {feature_family}")


def _format_k5_boundary_feature_key(key: tuple[object, ...]) -> str:
    return ":".join(str(part) for part in key)


def _two_step_feature_counts(
    state: tuple[int, ...],
    first_support: tuple[tuple[float, tuple[int, ...]], ...],
    second_support: tuple[tuple[float, tuple[int, ...]], ...],
    feature_family: str,
    feature_horizon: int,
) -> dict[tuple[object, ...], float]:
    totals: dict[tuple[object, ...], float] = defaultdict(float)
    for first_probability, first_action in first_support:
        first_state = canon(tuple(state[index] + first_action[index] for index in range(len(state))))
        for second_probability, second_action in second_support:
            second_state = canon(
                tuple(first_state[index] + second_action[index] for index in range(len(state)))
            )
            totals[_k5_boundary_feature_key(second_state, feature_horizon, feature_family)] += (
                first_probability * second_probability
            )
    return dict(totals)


def _k5_boundary_correction_samples(
    max_used: int,
    horizon: int,
    playout_name: str,
    feature_family: str,
) -> tuple[
    tuple[float, dict[tuple[object, ...], float], tuple[int, ...], str, str],
    ...,
]:
    action_pairs = _k5_proof_action_pairs()
    value_fn = _random_playout_value_function(5, horizon, playout_name)
    continuation_horizon = max(horizon - 2, 0)
    samples: list[tuple[float, dict[tuple[object, ...], float], tuple[int, ...], str, str]] = []
    for action_a, support_a in action_pairs.items():
        for action_w, support_w in action_pairs.items():
            if action_a == action_w:
                continue
            for used in range(max_used + 1):
                for state in all_states(5, used):
                    canonical_state = canon(state)
                    aw_value = _expected_two_step_potential_value(
                        canonical_state,
                        support_a,
                        support_w,
                        value_fn,
                        continuation_horizon,
                    )
                    wa_value = _expected_two_step_potential_value(
                        canonical_state,
                        support_w,
                        support_a,
                        value_fn,
                        continuation_horizon,
                    )
                    aw_features = _two_step_feature_counts(
                        canonical_state,
                        support_a,
                        support_w,
                        feature_family,
                        continuation_horizon,
                    )
                    wa_features = _two_step_feature_counts(
                        canonical_state,
                        support_w,
                        support_a,
                        feature_family,
                        continuation_horizon,
                    )
                    feature_delta: dict[tuple[object, ...], float] = defaultdict(float)
                    for key, value in aw_features.items():
                        feature_delta[key] += value
                    for key, value in wa_features.items():
                        feature_delta[key] -= value
                    samples.append((aw_value - wa_value, dict(feature_delta), canonical_state, action_a, action_w))
    return tuple(samples)


def _k5_boundary_design_matrix(
    max_used: int,
    horizon: int,
    playout_name: str,
    feature_family: str,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[object, ...], ...], tuple[tuple[int, ...], ...], tuple[str, ...], tuple[str, ...]]:
    samples = _k5_boundary_correction_samples(max_used, horizon, playout_name, feature_family)
    feature_keys = tuple(
        sorted(
            {key for _, feature_delta, _, _, _ in samples for key in feature_delta},
            key=_format_k5_boundary_feature_key,
        )
    )
    feature_index = {key: index for index, key in enumerate(feature_keys)}
    matrix = np.zeros((len(samples), len(feature_keys)), dtype=float)
    defects = np.zeros(len(samples), dtype=float)
    states: list[tuple[int, ...]] = []
    actions_a: list[str] = []
    actions_w: list[str] = []
    for row_index, (defect, feature_delta, state, action_a, action_w) in enumerate(samples):
        defects[row_index] = defect
        states.append(state)
        actions_a.append(action_a)
        actions_w.append(action_w)
        for key, coefficient in feature_delta.items():
            matrix[row_index, feature_index[key]] = coefficient
    return matrix, defects, feature_keys, tuple(states), tuple(actions_a), tuple(actions_w)


def _apply_k5_boundary_correction(defects: np.ndarray, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if matrix.size == 0 or weights.size == 0:
        return defects.copy()
    return defects + matrix @ weights


def _fit_k5_boundary_correction_row(
    max_used: int,
    horizon: int,
    playout_name: str,
    feature_family: str,
    ridge: float,
    n: int,
) -> K5BoundaryCorrectionFitRow:
    matrix, defects, feature_keys, states, actions_a, actions_w = _k5_boundary_design_matrix(
        max_used=max_used,
        horizon=horizon,
        playout_name=playout_name,
        feature_family=feature_family,
    )
    if len(defects) == 0 or len(feature_keys) == 0:
        weights = np.zeros(len(feature_keys), dtype=float)
    else:
        lhs = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
        rhs = -(matrix.T @ defects)
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    corrected = _apply_k5_boundary_correction(defects, matrix, weights)
    base_abs = np.abs(defects)
    corrected_abs = np.abs(corrected)
    base_avg = float(np.mean(base_abs)) if len(base_abs) else 0.0
    corrected_avg = float(np.mean(corrected_abs)) if len(corrected_abs) else 0.0
    improvement = base_avg / corrected_avg if corrected_avg > 1e-15 else float("inf")
    top_weight_indices = sorted(range(len(weights)), key=lambda index: abs(float(weights[index])), reverse=True)[:n]
    top_weights = tuple(
        (_format_k5_boundary_feature_key(feature_keys[index]), float(weights[index]))
        for index in top_weight_indices
        if abs(float(weights[index])) > 1e-12
    )
    top_failure_indices = sorted(range(len(corrected)), key=lambda index: abs(float(corrected[index])), reverse=True)[:n]
    top_failures = tuple(
        (
            float(corrected[index]),
            float(defects[index]),
            states[index],
            _k5_gap_vector(states[index]),
            packet_type(states[index]),
            actions_a[index],
            actions_w[index],
        )
        for index in top_failure_indices
    )
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]] = defaultdict(list)
    for index, state in enumerate(states):
        grouped[(_k5_gap_vector(state), packet_type(state))].append(abs(float(corrected[index])))
    grouped_failures = tuple(
        (gaps, ptype, max(values), len(values))
        for (gaps, ptype), values in sorted(grouped.items(), key=lambda item: (-max(item[1]), item[0]))[:n]
    )
    return K5BoundaryCorrectionFitRow(
        playout_name=playout_name,
        horizon=horizon,
        feature_family=feature_family,
        feature_count=len(feature_keys),
        sample_count=len(defects),
        base_max_abs_defect=float(np.max(base_abs)) if len(base_abs) else 0.0,
        base_avg_abs_defect=base_avg,
        corrected_max_abs_defect=float(np.max(corrected_abs)) if len(corrected_abs) else 0.0,
        corrected_avg_abs_defect=corrected_avg,
        improvement_factor=improvement,
        top_weights=top_weights,
        top_failures=top_failures,
        grouped_failures=grouped_failures,
    )


def k5_boundary_correction_fit_rows(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20),
    playout_names: tuple[str, ...] = ("Q_chase", "Q_comb"),
    feature_families: tuple[str, ...] = _K5_BOUNDARY_FEATURE_FAMILIES,
    ridge: float = 1e-8,
    n: int = 40,
) -> tuple[K5BoundaryCorrectionFitRow, ...]:
    rows: list[K5BoundaryCorrectionFitRow] = []
    for playout_name in playout_names:
        for horizon in h_values:
            for feature_family in feature_families:
                rows.append(
                    _fit_k5_boundary_correction_row(
                        max_used=max_used,
                        horizon=horizon,
                        playout_name=playout_name,
                        feature_family=feature_family,
                        ridge=ridge,
                        n=n,
                    )
                )
    return tuple(rows)


def print_k5_boundary_correction_fit_report(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20),
    playout_names: tuple[str, ...] = ("Q_chase", "Q_comb"),
    feature_families: tuple[str, ...] = _K5_BOUNDARY_FEATURE_FAMILIES,
    ridge: float = 1e-8,
    n: int = 40,
) -> None:
    rows = k5_boundary_correction_fit_rows(
        max_used=max_used,
        h_values=h_values,
        playout_names=playout_names,
        feature_families=feature_families,
        ridge=ridge,
        n=n,
    )
    print("k=5 boundary correction fit")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"playouts: {playout_names}")
    print(f"feature_families: {feature_families}")
    print(f"ridge: {ridge:g}")
    print()
    print(
        "playout  h feature_family        features samples"
        " base_max_abs_defect base_avg_abs_defect"
        " corrected max_abs_defect corrected_avg_abs_defect improvement"
    )
    for row in rows:
        print(
            f"{row.playout_name:8s}"
            f" {row.horizon:2d}"
            f" {row.feature_family:21s}"
            f" {row.feature_count:8d}"
            f" {row.sample_count:7d}"
            f" {row.base_max_abs_defect:19.6f}"
            f" {row.base_avg_abs_defect:19.6f}"
            f" {row.corrected_max_abs_defect:24.6f}"
            f" {row.corrected_avg_abs_defect:24.6f}"
            f" {row.improvement_factor:11.3f}"
        )

    print()
    print("top learned correction weights")
    for row in rows:
        print(f"  playout={row.playout_name} h={row.horizon} family={row.feature_family}")
        if not row.top_weights:
            print("    <all fitted weights approximately zero>")
        for key, weight in row.top_weights[:n]:
            print(f"    {key}: {weight:.6f}")

    print()
    print("top remaining failures after correction")
    for row in rows:
        print(f"  playout={row.playout_name} h={row.horizon} family={row.feature_family}")
        for corrected_defect, base_defect, state, gaps, ptype, action_a, action_w in row.top_failures[:n]:
            print(
                f"    corrected={corrected_defect:+.6f}"
                f" base={base_defect:+.6f}"
                f" state={state}"
                f" gaps={gaps}"
                f" packet={ptype}"
                f" A={action_a}"
                f" W={action_w}"
            )

    print()
    print("remaining failures by gap/packet")
    for row in rows:
        print(f"  playout={row.playout_name} h={row.horizon} family={row.feature_family}")
        for gaps, ptype, max_abs, count in row.grouped_failures[:n]:
            print(f"    gaps={gaps} packet={ptype} max_abs={max_abs:.6f} count={count}")


def _parse_k5_relaxation_potentials(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip() or text.strip() == "all":
        return ("Q_chase", "Q_comb", "Q_uniform_one_run", "softmax_c0.5", "softmax_c1", "softmax_c2")
    return tuple(piece.strip() for piece in text.split(",") if piece.strip())


def _k5_certificate_actions(action_family: str) -> tuple[tuple[int, ...], ...]:
    if action_family == "all_binary":
        return tuple(action for action in all_actions(5) if any(action))
    if action_family == "balanced_pairs_support":
        actions = {
            action
            for support in _k5_proof_action_pairs().values()
            for _, action in support
        }
        return tuple(sorted(actions, key=_format_action))
    raise ValueError(f"unknown action family: {action_family}")


def _k5_softmax_c_from_name(potential_name: str) -> float:
    if not potential_name.startswith("softmax_c"):
        raise ValueError(f"not a softmax potential name: {potential_name}")
    return float(potential_name.removeprefix("softmax_c"))


def _k5_relaxation_value_functions(potential_name: str, horizon: int):
    if potential_name.startswith("Q_"):
        value_fn = _random_playout_value_function(5, horizon, potential_name)

        def current_value(state: tuple[int, ...], remaining_horizon: int) -> float:
            return value_fn(canon(state), remaining_horizon)

        def next_value(raw_state: tuple[int, ...], remaining_horizon: int) -> float:
            return float(min(raw_state, default=0)) + value_fn(canon(raw_state), remaining_horizon)

        return current_value, next_value

    if potential_name.startswith("softmax_c"):
        c_value = _k5_softmax_c_from_name(potential_name)

        def tau(remaining_horizon: int) -> float:
            return c_value * (max(remaining_horizon, 1) ** 0.5)

        def current_value(state: tuple[int, ...], remaining_horizon: int) -> float:
            return _phi_centered_softmax(canon(state), tau(remaining_horizon))

        def next_value(raw_state: tuple[int, ...], remaining_horizon: int) -> float:
            return _phi_centered_softmax(canon(raw_state), tau(remaining_horizon))

        return current_value, next_value

    raise ValueError(f"unknown k=5 relaxation potential: {potential_name}")


def _k5_relaxation_natural_learner(
    potential_name: str,
    state: tuple[int, ...],
    remaining_horizon: int,
) -> tuple[float, ...] | None:
    if potential_name.startswith("Q_"):
        return _random_playout_terminal_winner_probabilities(5, potential_name, state, remaining_horizon)
    if potential_name.startswith("softmax_c"):
        c_value = _k5_softmax_c_from_name(potential_name)
        tau = c_value * (max(remaining_horizon, 1) ** 0.5)
        return _softmax_centered_gradient(state, tau)
    return None


def _k5_relaxation_residual(
    state: tuple[int, ...],
    remaining_horizon: int,
    action: tuple[int, ...],
    learner: tuple[float, ...],
    current_value,
    next_value,
) -> float:
    raw_next = tuple(state[index] + action[index] for index in range(5))
    learner_gain = sum(learner[index] * action[index] for index in range(5))
    return next_value(raw_next, max(remaining_horizon - 1, 0)) - learner_gain - current_value(state, remaining_horizon)


def _k5_relaxation_local_certificate_learner(
    state: tuple[int, ...],
    remaining_horizon: int,
    current_value,
    next_value,
    actions: tuple[tuple[int, ...], ...],
) -> tuple[tuple[float, ...], float, tuple[int, ...]]:
    potential = current_value(state, remaining_horizon)
    q_values = []
    for action in actions:
        raw_next = tuple(state[index] + action[index] for index in range(5))
        q_values.append(next_value(raw_next, max(remaining_horizon - 1, 0)) - potential)

    c = [0.0] * 5 + [1.0]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for action, q_value in zip(actions, q_values, strict=True):
        a_ub.append([-float(bit) for bit in action] + [-1.0])
        b_ub.append(-float(q_value))
    a_eq = [[1.0] * 5 + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * 5 + [(None, None)]
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"relaxation certificate LP failed at state {state}, h={remaining_horizon}: {result.message}")
    learner = tuple(float(value) for value in result.x[:5])
    residuals = tuple(
        _k5_relaxation_residual(state, remaining_horizon, action, learner, current_value, next_value)
        for action in actions
    )
    max_index = max(range(len(actions)), key=lambda index: residuals[index])
    return learner, float(residuals[max_index]), actions[max_index]


def _k5_relaxation_natural_alpha(
    state: tuple[int, ...],
    remaining_horizon: int,
    learner: tuple[float, ...],
    current_value,
    next_value,
    actions: tuple[tuple[int, ...], ...],
) -> float:
    return max(
        _k5_relaxation_residual(state, remaining_horizon, action, learner, current_value, next_value)
        for action in actions
    )


def k5_relaxation_certificate_rows(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50),
    potential_names: tuple[str, ...] = ("Q_chase", "Q_comb", "softmax_c1"),
    action_family: str = "all_binary",
) -> tuple[K5RelaxationCertificateRow, ...]:
    actions = _k5_certificate_actions(action_family)
    rows: list[K5RelaxationCertificateRow] = []
    for potential_name in potential_names:
        for horizon in h_values:
            current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
            for used in range(max_used + 1):
                for state in all_states(5, used):
                    canonical_state = canon(state)
                    natural_learner = _k5_relaxation_natural_learner(potential_name, canonical_state, horizon)
                    natural_alpha = (
                        _k5_relaxation_natural_alpha(
                            canonical_state,
                            horizon,
                            natural_learner,
                            current_value,
                            next_value,
                            actions,
                        )
                        if natural_learner is not None
                        else None
                    )
                    repaired_learner, repaired_alpha, repaired_action = _k5_relaxation_local_certificate_learner(
                        canonical_state,
                        horizon,
                        current_value,
                        next_value,
                        actions,
                    )
                    rows.append(
                        K5RelaxationCertificateRow(
                            potential_name=potential_name,
                            horizon=horizon,
                            state=canonical_state,
                            gap_vector=_k5_gap_vector(canonical_state),
                            packet_type=packet_type(canonical_state),
                            zero_gap_pattern=_k5_zero_gap_pattern(canonical_state),
                            rel_value=current_value(canonical_state, horizon),
                            natural_learner=natural_learner,
                            natural_alpha=natural_alpha,
                            repaired_learner=repaired_learner,
                            repaired_alpha=repaired_alpha,
                            repaired_worst_action=repaired_action,
                        )
                    )
    return tuple(rows)


def _positive_average(values: list[float]) -> float:
    positives = [value for value in values if value > 0.0]
    return sum(positives) / len(positives) if positives else 0.0


def _k5_relaxation_summary_rows(
    rows: tuple[K5RelaxationCertificateRow, ...],
) -> tuple[K5RelaxationCertificateSummaryRow, ...]:
    grouped: dict[tuple[str, int], list[K5RelaxationCertificateRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.potential_name, row.horizon)].append(row)
    summaries: list[K5RelaxationCertificateSummaryRow] = []
    for (potential_name, horizon), group_rows in sorted(grouped.items()):
        repaired = [row.repaired_alpha for row in group_rows]
        natural = [row.natural_alpha for row in group_rows if row.natural_alpha is not None]
        summaries.append(
            K5RelaxationCertificateSummaryRow(
                potential_name=potential_name,
                horizon=horizon,
                sample_count=len(group_rows),
                max_repaired_alpha=max(repaired) if repaired else 0.0,
                avg_positive_repaired_alpha=_positive_average(repaired),
                max_natural_alpha=max(natural) if natural else None,
                avg_positive_natural_alpha=_positive_average(natural) if natural else None,
            )
        )
    return tuple(summaries)


def _dedupe_k5_relaxation_certificate_rows(
    rows: tuple[K5RelaxationCertificateRow, ...],
    key_fn,
) -> tuple[K5RelaxationCertificateRow, ...]:
    deduped: dict[object, K5RelaxationCertificateRow] = {}
    for row in rows:
        key = key_fn(row)
        if key not in deduped:
            deduped[key] = row
    return tuple(deduped.values())


def print_k5_relaxation_certificate_report(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50),
    potential_names: tuple[str, ...] = ("Q_chase", "Q_comb", "softmax_c1"),
    action_family: str = "all_binary",
    n: int = 40,
) -> None:
    rows = k5_relaxation_certificate_rows(
        max_used=max_used,
        h_values=h_values,
        potential_names=potential_names,
        action_family=action_family,
    )
    rows = _dedupe_k5_relaxation_certificate_rows(
        rows,
        lambda row: (row.potential_name, row.horizon, row.state),
    )
    summaries = _k5_relaxation_summary_rows(rows)
    print("k=5 relaxation certificate report")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"potentials: {potential_names}")
    print(f"action_family: {action_family}")
    print()

    print("1. Certificate summary by potential/horizon")
    print(
        "potential          h samples max_repaired_alpha avg_positive_repaired_alpha"
        " h*max sqrt(h)*max h^(3/2)*max h*avg_pos sqrt(h)*avg_pos"
        " max_natural_alpha avg_positive_natural_alpha"
    )
    for row in summaries:
        sqrt_horizon = row.horizon ** 0.5 if row.horizon > 0 else 1.0
        print(
            f"{row.potential_name:18s}"
            f" {row.horizon:2d}"
            f" {row.sample_count:7d}"
            f" {row.max_repaired_alpha:18.6f}"
            f" {row.avg_positive_repaired_alpha:28.6f}"
            f" {row.horizon * row.max_repaired_alpha:9.6f}"
            f" {sqrt_horizon * row.max_repaired_alpha:11.6f}"
            f" {row.horizon * sqrt_horizon * row.max_repaired_alpha:13.6f}"
            f" {row.horizon * row.avg_positive_repaired_alpha:9.6f}"
            f" {sqrt_horizon * row.avg_positive_repaired_alpha:16.6f}"
            f" {_format_optional_float(row.max_natural_alpha):>17s}"
            f" {_format_optional_float(row.avg_positive_natural_alpha):>26s}"
        )

    print()
    print("2. Root row summary")
    root_state = (0, 0, 0, 0, 0)
    for row in sorted((row for row in rows if row.state == root_state), key=lambda item: (item.potential_name, item.horizon)):
        print(
            f"  potential={row.potential_name:18s}"
            f" h={row.horizon:2d}"
            f" natural_alpha={_format_optional_float(row.natural_alpha)}"
            f" repaired_alpha={row.repaired_alpha:.6f}"
            f" repaired_p={_format_float_tuple(row.repaired_learner)}"
        )

    print()
    print("3. PM vs repaired comparison")
    comparable = [row for row in rows if row.natural_alpha is not None]
    comparable = list(
        _dedupe_k5_relaxation_certificate_rows(
            tuple(comparable),
            lambda row: (row.potential_name, row.horizon, row.state),
        )
    )
    for row in sorted(comparable, key=lambda item: (item.natural_alpha or 0.0) - item.repaired_alpha, reverse=True)[:n]:
        print(
            f"  potential={row.potential_name:18s}"
            f" h={row.horizon:2d}"
            f" state={row.state}"
            f" gaps={row.gap_vector}"
            f" packet={row.packet_type}"
            f" natural_alpha={row.natural_alpha:.6f}"
            f" repaired_alpha={row.repaired_alpha:.6f}"
            f" repaired_p={_format_float_tuple(row.repaired_learner)}"
        )

    print()
    print("4. Top repaired failures")
    failure_rows = _dedupe_k5_relaxation_certificate_rows(
        rows,
        lambda row: (row.potential_name, row.horizon, row.state, row.repaired_worst_action),
    )
    for row in sorted(failure_rows, key=lambda item: item.repaired_alpha, reverse=True)[:n]:
        print(
            f"  potential={row.potential_name:18s}"
            f" h={row.horizon:2d}"
            f" state={row.state}"
            f" gaps={row.gap_vector}"
            f" packet={row.packet_type}"
            f" zero={row.zero_gap_pattern}"
            f" alpha={row.repaired_alpha:.6f}"
            f" action={_format_action(row.repaired_worst_action)}"
            f" p={_format_float_tuple(row.repaired_learner)}"
        )

    print()
    print("5. Failures grouped by gap/packet")
    grouped: dict[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[float]] = defaultdict(list)
    for row in failure_rows:
        grouped[(row.potential_name, row.horizon, row.gap_vector, row.packet_type, row.zero_gap_pattern)].append(
            row.repaired_alpha
        )
    for (potential_name, horizon, gaps, ptype, zero_pattern), values in sorted(
        grouped.items(),
        key=lambda item: (-max(item[1]), item[0]),
    )[:n]:
        print(
            f"  potential={potential_name:18s}"
            f" h={horizon:2d}"
            f" gaps={gaps}"
            f" packet={ptype}"
            f" zero={zero_pattern}"
            f" max_alpha={max(values):.6f}"
            f" count={len(values)}"
        )

    print()
    print("6. Best potential ranking")
    for row in sorted(summaries, key=lambda item: (item.max_repaired_alpha, item.avg_positive_repaired_alpha)):
        print(
            f"  potential={row.potential_name:18s}"
            f" h={row.horizon:2d}"
            f" max_alpha={row.max_repaired_alpha:.6f}"
            f" avg_pos_alpha={row.avg_positive_repaired_alpha:.6f}"
        )


def _packet_average_reconstruction(
    state: tuple[int, ...],
    learner: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    groups = packets(state)
    reconstructed = [0.0] * len(state)
    packet_averages: list[float] = []
    total_variance = 0.0
    for group in groups:
        average = sum(learner[index] for index in group) / len(group)
        packet_averages.append(average)
        for index in group:
            reconstructed[index] = average
            total_variance += (learner[index] - average) ** 2
    variance = total_variance / len(state) if state else 0.0
    return tuple(reconstructed), tuple(packet_averages), variance


def _k5_active_actions_for_learner(
    state: tuple[int, ...],
    remaining_horizon: int,
    learner: tuple[float, ...],
    current_value,
    next_value,
    actions: tuple[tuple[int, ...], ...],
    tolerance: float = 1e-8,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    residuals = tuple(
        (action, _k5_relaxation_residual(state, remaining_horizon, action, learner, current_value, next_value))
        for action in actions
    )
    max_residual = max((residual for _, residual in residuals), default=0.0)
    return tuple((action, residual) for action, residual in residuals if residual >= max_residual - tolerance)


def _project_to_simplex(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    sorted_values = sorted(values, reverse=True)
    cumulative = 0.0
    rho = 0
    for index, value in enumerate(sorted_values, start=1):
        cumulative += value
        theta = (cumulative - 1.0) / index
        if value - theta > 0:
            rho = index
    theta = (sum(sorted_values[:rho]) - 1.0) / rho if rho else 0.0
    return tuple(max(value - theta, 0.0) for value in values)


def _affine_gap_learner(state: tuple[int, ...], horizon: int, slope: float) -> tuple[float, ...]:
    mean_score = sum(state) / len(state) if state else 0.0
    scale = max(horizon, 1) ** 0.5
    raw = tuple((1.0 / len(state)) + slope * ((value - mean_score) / scale) for value in state)
    return _project_to_simplex(raw)


def k5_repaired_learner_anatomy_rows(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    action_family: str = "all_binary",
    active_tolerance: float = 1e-8,
) -> tuple[K5RepairedLearnerAnatomyRow, ...]:
    actions = _k5_certificate_actions(action_family)
    rows: list[K5RepairedLearnerAnatomyRow] = []
    softmax_grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    for horizon in h_values:
        current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
        for used in range(max_used + 1):
            for state in all_states(5, used):
                canonical_state = canon(state)
                repaired_p, repaired_alpha, worst_action = _k5_relaxation_local_certificate_learner(
                    canonical_state,
                    horizon,
                    current_value,
                    next_value,
                    actions,
                )
                natural_p = _k5_relaxation_natural_learner(potential_name, canonical_state, horizon)
                packet_uniform_p, packet_averages, within_packet_variance = _packet_average_reconstruction(
                    canonical_state,
                    repaired_p,
                )
                packet_uniform_alpha = _k5_relaxation_natural_alpha(
                    canonical_state,
                    horizon,
                    packet_uniform_p,
                    current_value,
                    next_value,
                    actions,
                )
                softmax_candidates = []
                for c_value in softmax_grid:
                    tau = c_value * (max(horizon, 1) ** 0.5)
                    learner = _softmax_centered_gradient(canonical_state, tau)
                    alpha = _k5_relaxation_natural_alpha(
                        canonical_state,
                        horizon,
                        learner,
                        current_value,
                        next_value,
                        actions,
                    )
                    softmax_candidates.append((alpha, c_value))
                best_softmax_alpha, best_softmax_c = min(softmax_candidates)
                active = _k5_active_actions_for_learner(
                    canonical_state,
                    horizon,
                    repaired_p,
                    current_value,
                    next_value,
                    actions,
                    tolerance=active_tolerance,
                )
                rows.append(
                    K5RepairedLearnerAnatomyRow(
                        horizon=horizon,
                        state=canonical_state,
                        gap_vector=_k5_gap_vector(canonical_state),
                        packet_type=packet_type(canonical_state),
                        zero_gap_pattern=_k5_zero_gap_pattern(canonical_state),
                        repaired_alpha=repaired_alpha,
                        repaired_p=repaired_p,
                        natural_p=natural_p,
                        p_minus_uniform=tuple(probability - (1.0 / 5.0) for probability in repaired_p),
                        packet_average_p=packet_uniform_p,
                        packet_average_by_packet=packet_averages,
                        within_packet_variance=within_packet_variance,
                        worst_action=worst_action,
                        active_worst_actions=tuple(action for action, _ in active),
                        packet_uniform_alpha=packet_uniform_alpha,
                        best_softmax_c=best_softmax_c,
                        best_softmax_alpha=best_softmax_alpha,
                    )
                )
    return tuple(rows)


def _format_packet_average_by_packet(row: K5RepairedLearnerAnatomyRow) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in row.packet_average_by_packet) + ")"


def print_k5_repaired_learner_anatomy_report(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    action_family: str = "all_binary",
    n: int = 40,
) -> None:
    rows = k5_repaired_learner_anatomy_rows(
        max_used=max_used,
        h_values=h_values,
        potential_name=potential_name,
        action_family=action_family,
    )
    print("k=5 repaired learner anatomy")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"potential: {potential_name}")
    print(f"action_family: {action_family}")
    print()

    print("A. Summary by h")
    print(
        "h samples max_alpha avg_positive_alpha max_within_packet_var"
        " avg_within_packet_var packet_constant_fraction"
    )
    for horizon in h_values:
        group = [row for row in rows if row.horizon == horizon]
        if not group:
            continue
        alphas = [row.repaired_alpha for row in group]
        variances = [row.within_packet_variance for row in group]
        packet_constant = sum(1 for row in group if row.within_packet_variance <= 1e-12)
        print(
            f"{horizon:3d}"
            f" {len(group):7d}"
            f" {max(alphas):9.6f}"
            f" {_positive_average(alphas):18.6f}"
            f" {max(variances):21.6g}"
            f" {sum(variances) / len(variances):21.6g}"
            f" {packet_constant / len(group):24.3f}"
        )

    print()
    print("B. Top alpha states")
    for row in sorted(rows, key=lambda item: item.repaired_alpha, reverse=True)[:n]:
        print(
            f"  h={row.horizon:3d}"
            f" state={row.state}"
            f" gaps={row.gap_vector}"
            f" packet={row.packet_type}"
            f" zero={row.zero_gap_pattern}"
            f" alpha={row.repaired_alpha:.6f}"
            f" p={_format_float_tuple(row.repaired_p)}"
            f" packet_avg={_format_packet_average_by_packet(row)}"
            f" worst={_format_action(row.worst_action)}"
            f" active={','.join(_format_action(action) for action in row.active_worst_actions[:8])}"
        )

    print()
    print("C. Packet-pattern templates")
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[K5RepairedLearnerAnatomyRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.packet_type, row.zero_gap_pattern)].append(row)
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.repaired_alpha for row in item[1]), item[0]))[:n]:
        representative = max(group, key=lambda row: row.repaired_alpha)
        avg_packet_count = len(representative.packet_average_by_packet)
        avg_by_packet = tuple(
            sum(row.packet_average_by_packet[index] for row in group if len(row.packet_average_by_packet) == avg_packet_count)
            / sum(1 for row in group if len(row.packet_average_by_packet) == avg_packet_count)
            for index in range(avg_packet_count)
        )
        print(
            f"  packet={ptype}"
            f" zero={zero_pattern}"
            f" count={len(group)}"
            f" max_alpha={max(row.repaired_alpha for row in group):.6f}"
            f" rep_state={representative.state}"
            f" rep_p={_format_float_tuple(representative.repaired_p)}"
            f" rep_packet_avg={_format_packet_average_by_packet(representative)}"
            f" avg_packet_avg={_format_float_tuple(avg_by_packet)}"
        )

    print()
    print("D. Scaling by packet pattern")
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.repaired_alpha for row in item[1]), item[0]))[:n]:
        by_h = {}
        for row in group:
            by_h[row.horizon] = max(by_h.get(row.horizon, float("-inf")), row.repaired_alpha)
        pieces = []
        for horizon in h_values:
            if horizon in by_h:
                scaled = (horizon ** 1.5) * by_h[horizon]
                pieces.append(f"h={horizon}:alpha={by_h[horizon]:.6f},h32={scaled:.6f}")
        print(f"  packet={ptype} zero={zero_pattern} " + "; ".join(pieces))

    print()
    print("E. Simple learner hypothesis")
    print("h true_max packet_uniform_max softmax_max true_avg_pos packet_uniform_avg_pos softmax_avg_pos")
    for horizon in h_values:
        group = [row for row in rows if row.horizon == horizon]
        if not group:
            continue
        true_alphas = [row.repaired_alpha for row in group]
        packet_alphas = [row.packet_uniform_alpha for row in group]
        softmax_alphas = [row.best_softmax_alpha for row in group]
        print(
            f"{horizon:3d}"
            f" {max(true_alphas):8.6f}"
            f" {max(packet_alphas):18.6f}"
            f" {max(softmax_alphas):11.6f}"
            f" {_positive_average(true_alphas):12.6f}"
            f" {_positive_average(packet_alphas):22.6f}"
            f" {_positive_average(softmax_alphas):15.6f}"
        )


def _packet_weights_from_packet_uniform_p(
    state: tuple[int, ...],
    learner: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(sum(learner[index] for index in group) / len(group) for group in packets(state))


def _reconstruct_from_packet_weights(
    state: tuple[int, ...],
    packet_weights: tuple[float, ...],
) -> tuple[float, ...]:
    reconstructed = [0.0] * len(state)
    for group_index, group in enumerate(packets(state)):
        for index in group:
            reconstructed[index] = packet_weights[group_index]
    return tuple(reconstructed)


def _rationalize_packet_weights(
    packet_type_value: tuple[int, ...],
    packet_weights: tuple[float, ...],
    max_denominator: int = 240,
) -> tuple[str, ...]:
    del packet_type_value
    return tuple(_format_fractionish(weight, max_denominator=max_denominator) for weight in packet_weights)


def _rationalized_packet_learner(
    state: tuple[int, ...],
    packet_weights: tuple[float, ...],
    max_denominator: int = 240,
) -> tuple[float, ...]:
    rational_weights = tuple(float(Fraction(weight).limit_denominator(max_denominator)) for weight in packet_weights)
    reconstructed = _reconstruct_from_packet_weights(state, rational_weights)
    total = sum(reconstructed)
    if total <= 0.0:
        return tuple(1.0 / len(state) for _ in state)
    return tuple(value / total for value in reconstructed)


def _packet_score_softmax_learner(
    state: tuple[int, ...],
    c_value: float,
    horizon: int,
) -> tuple[float, ...]:
    groups = packets(state)
    if not groups:
        return ()
    tau = c_value * (max(horizon, 1) ** 0.5)
    packet_scores = tuple(state[group[0]] for group in groups)
    max_score = max(packet_scores)
    packet_masses = tuple(len(group) * np.exp((score - max_score) / tau) for group, score in zip(groups, packet_scores))
    total_mass = sum(packet_masses)
    packet_weights = tuple(mass / total_mass / len(group) for mass, group in zip(packet_masses, groups))
    return _reconstruct_from_packet_weights(state, packet_weights)


def _affine_packet_learner(
    state: tuple[int, ...],
    slope: float,
    horizon: int,
) -> tuple[float, ...]:
    groups = packets(state)
    if not groups:
        return ()
    mean_score = sum(state) / len(state)
    scale = max(horizon, 1) ** 0.5
    packet_weights = tuple((1.0 / len(state)) + slope * ((state[group[0]] - mean_score) / scale) for group in groups)
    return _project_to_simplex(_reconstruct_from_packet_weights(state, packet_weights))


def k5_packet_learner_formula_rows(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    action_family: str = "all_binary",
    rational_max_denominator: int = 240,
) -> tuple[K5PacketLearnerFormulaRow, ...]:
    anatomy_rows = k5_repaired_learner_anatomy_rows(
        max_used=max_used,
        h_values=h_values,
        potential_name=potential_name,
        action_family=action_family,
    )
    actions = _k5_certificate_actions(action_family)
    softmax_grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    affine_grid = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
    rows: list[K5PacketLearnerFormulaRow] = []
    for row in anatomy_rows:
        current_value, next_value = _k5_relaxation_value_functions(potential_name, row.horizon)
        packet_weights = _packet_weights_from_packet_uniform_p(row.state, row.repaired_p)
        rationalized_p = _rationalized_packet_learner(
            row.state,
            packet_weights,
            max_denominator=rational_max_denominator,
        )
        rationalized_alpha = _k5_relaxation_natural_alpha(
            row.state,
            row.horizon,
            rationalized_p,
            current_value,
            next_value,
            actions,
        )
        softmax_candidates = []
        for c_value in softmax_grid:
            learner = _packet_score_softmax_learner(row.state, c_value, row.horizon)
            alpha = _k5_relaxation_natural_alpha(row.state, row.horizon, learner, current_value, next_value, actions)
            softmax_candidates.append((alpha, c_value))
        best_softmax_alpha, best_softmax_c = min(softmax_candidates)
        affine_candidates = []
        for slope in affine_grid:
            learner = _affine_packet_learner(row.state, slope, row.horizon)
            alpha = _k5_relaxation_natural_alpha(row.state, row.horizon, learner, current_value, next_value, actions)
            affine_candidates.append((alpha, slope))
        best_affine_alpha, best_affine_slope = min(affine_candidates)
        rows.append(
            K5PacketLearnerFormulaRow(
                source=row,
                packet_weights=packet_weights,
                rational_packet_weights=_rationalize_packet_weights(
                    row.packet_type,
                    packet_weights,
                    max_denominator=rational_max_denominator,
                ),
                rationalized_p=rationalized_p,
                rationalized_alpha=rationalized_alpha,
                packet_score_softmax_c=best_softmax_c,
                packet_score_softmax_alpha=best_softmax_alpha,
                affine_packet_slope=best_affine_slope,
                affine_packet_alpha=best_affine_alpha,
            )
        )
    return tuple(rows)


def _format_packet_weights(weights: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{weight:.6f}" for weight in weights) + ")"


def print_k5_packet_learner_formula_report(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    action_family: str = "all_binary",
    n: int = 40,
) -> None:
    rows = k5_packet_learner_formula_rows(
        max_used=max_used,
        h_values=h_values,
        potential_name=potential_name,
        action_family=action_family,
    )
    print("k=5 packet learner formula report")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"potential: {potential_name}")
    print(f"action_family: {action_family}")
    print()

    print("A. Packet learner summary")
    print("h max_alpha regimes weight_templates packet_uniform_fraction")
    for horizon in h_values:
        group = [row for row in rows if row.source.horizon == horizon]
        if not group:
            continue
        regimes = {
            (row.source.packet_type, row.source.gap_vector, row.source.zero_gap_pattern)
            for row in group
        }
        templates = {
            (row.source.packet_type, row.source.zero_gap_pattern, row.rational_packet_weights)
            for row in group
        }
        packet_uniform = sum(1 for row in group if row.source.within_packet_variance <= 1e-12)
        print(
            f"{horizon:3d}"
            f" {max(row.source.repaired_alpha for row in group):9.6f}"
            f" {len(regimes):7d}"
            f" {len(templates):16d}"
            f" {packet_uniform / len(group):23.3f}"
        )

    print()
    print("B. Rationalized packet-weight templates")
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[K5PacketLearnerFormulaRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.source.packet_type, row.source.zero_gap_pattern)].append(row)
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.source.repaired_alpha for row in item[1]), item[0]))[:n]:
        representative = max(group, key=lambda row: row.source.repaired_alpha)
        total_mass = sum(size * weight for size, weight in zip(ptype, representative.packet_weights))
        print(
            f"  packet={ptype}"
            f" zero={zero_pattern}"
            f" rep_gaps={representative.source.gap_vector}"
            f" state={representative.source.state}"
            f" alpha={representative.source.repaired_alpha:.6f}"
            f" weights={_format_packet_weights(representative.packet_weights)}"
            f" rational={representative.rational_packet_weights}"
            f" total_mass={total_mass:.6f}"
            f" worst={_format_action(representative.source.worst_action)}"
            f" active={','.join(_format_action(action) for action in representative.source.active_worst_actions[:8])}"
        )

    print()
    print("C. Scaling of packet templates")
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.source.repaired_alpha for row in item[1]), item[0]))[:n]:
        pieces = []
        for horizon in h_values:
            horizon_rows = [row for row in group if row.source.horizon == horizon]
            if not horizon_rows:
                continue
            representative = max(horizon_rows, key=lambda row: row.source.repaired_alpha)
            scaled = (horizon ** 1.5) * representative.source.repaired_alpha
            pieces.append(
                f"h={horizon}:alpha={representative.source.repaired_alpha:.6f},"
                f"h32={scaled:.6f},w={representative.rational_packet_weights}"
            )
        print(f"  packet={ptype} zero={zero_pattern} " + "; ".join(pieces))

    print()
    print("D. Simple formula candidates")
    print("h exact_max rational_max packet_softmax_max affine_packet_max exact_avg_pos rational_avg_pos")
    for horizon in h_values:
        group = [row for row in rows if row.source.horizon == horizon]
        if not group:
            continue
        exact = [row.source.repaired_alpha for row in group]
        rational = [row.rationalized_alpha for row in group]
        packet_softmax = [row.packet_score_softmax_alpha for row in group]
        affine = [row.affine_packet_alpha for row in group]
        print(
            f"{horizon:3d}"
            f" {max(exact):9.6f}"
            f" {max(rational):12.6f}"
            f" {max(packet_softmax):18.6f}"
            f" {max(affine):17.6f}"
            f" {_positive_average(exact):13.6f}"
            f" {_positive_average(rational):16.6f}"
        )

    print()
    print("  top rationalized failures")
    for row in sorted(rows, key=lambda item: item.rationalized_alpha, reverse=True)[:n]:
        print(
            f"    h={row.source.horizon:3d}"
            f" state={row.source.state}"
            f" packet={row.source.packet_type}"
            f" gaps={row.source.gap_vector}"
            f" alpha={row.rationalized_alpha:.6f}"
            f" rational={row.rational_packet_weights}"
        )

    print()
    print("E. Export proof table")
    print("packet_type zero_pattern gap_condition packet_weights alpha_bound_observed")
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.source.repaired_alpha for row in item[1]), item[0]))[:n]:
        representative = max(group, key=lambda row: row.source.repaired_alpha)
        print(
            f"{ptype} {zero_pattern}"
            f" gaps~{representative.source.gap_vector}"
            f" weights={representative.rational_packet_weights}"
            f" alpha<={max(row.source.repaired_alpha for row in group):.6f}"
        )


def _packet_count_vectors(packet_sizes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(product(*(range(size + 1) for size in packet_sizes)))


def _representative_action_from_packet_counts(
    state: tuple[int, ...],
    count_vector: tuple[int, ...],
) -> tuple[int, ...]:
    action = [0] * len(state)
    for group, count in zip(packets(state), count_vector):
        for index in group[:count]:
            action[index] = 1
    return tuple(action)


def _next_state_from_packet_counts(
    state: tuple[int, ...],
    count_vector: tuple[int, ...],
) -> tuple[int, ...]:
    action = _representative_action_from_packet_counts(state, count_vector)
    return canon(tuple(state[index] + action[index] for index in range(len(state))))


def _reduced_packet_lp(
    state: tuple[int, ...],
    horizon: int,
    current_value,
    next_value,
    active_tolerance: float = 1e-8,
) -> tuple[tuple[float, ...], float, tuple[tuple[tuple[int, ...], float, float], ...], tuple[tuple[tuple[int, ...], float], ...]]:
    canonical_state = canon(state)
    packet_sizes = packet_type(canonical_state)
    count_vectors = _packet_count_vectors(packet_sizes)
    potential = current_value(canonical_state, horizon)
    q_values: list[float] = []
    for count_vector in count_vectors:
        action = _representative_action_from_packet_counts(canonical_state, count_vector)
        raw_next = tuple(canonical_state[index] + action[index] for index in range(len(canonical_state)))
        q_values.append(next_value(raw_next, max(horizon - 1, 0)) - potential)

    variable_count = len(packet_sizes) + 1
    c = [0.0] * len(packet_sizes) + [1.0]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for count_vector, q_value in zip(count_vectors, q_values, strict=True):
        a_ub.append([-float(count) for count in count_vector] + [-1.0])
        b_ub.append(-float(q_value))
    a_eq = [[float(size) for size in packet_sizes] + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * len(packet_sizes) + [(None, None)]
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"reduced packet LP failed at state {canonical_state}, h={horizon}: {result.message}")
    weights = tuple(float(value) for value in result.x[: len(packet_sizes)])
    alpha = float(result.x[-1])
    residuals = tuple(
        q_value - sum(count * weight for count, weight in zip(count_vector, weights))
        for count_vector, q_value in zip(count_vectors, q_values, strict=True)
    )
    active = tuple(
        (count_vector, residual, q_value)
        for count_vector, residual, q_value in zip(count_vectors, residuals, q_values, strict=True)
        if residual >= alpha - active_tolerance
    )
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    dual_marginals: tuple[tuple[tuple[int, ...], float], ...]
    if marginals is None:
        dual_marginals = ()
    else:
        dual_marginals = tuple(
            (count_vector, float(-marginal))
            for count_vector, marginal in zip(count_vectors, marginals, strict=True)
            if abs(float(marginal)) > active_tolerance
        )
    del variable_count
    return weights, alpha, active, dual_marginals


def k5_reduced_packet_lp_rows(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    active_tolerance: float = 1e-8,
) -> tuple[K5ReducedPacketLPRow, ...]:
    rows: list[K5ReducedPacketLPRow] = []
    actions = _k5_certificate_actions("all_binary")
    for horizon in h_values:
        current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
        for used in range(max_used + 1):
            for state in all_states(5, used):
                canonical_state = canon(state)
                full_p, full_alpha, _ = _k5_relaxation_local_certificate_learner(
                    canonical_state,
                    horizon,
                    current_value,
                    next_value,
                    actions,
                )
                full_packet_weights = _packet_weights_from_packet_uniform_p(canonical_state, full_p)
                reduced_weights, reduced_alpha, active, dual_marginals = _reduced_packet_lp(
                    canonical_state,
                    horizon,
                    current_value,
                    next_value,
                    active_tolerance=active_tolerance,
                )
                max_weight_difference = max(
                    (abs(left - right) for left, right in zip(full_packet_weights, reduced_weights)),
                    default=0.0,
                )
                rows.append(
                    K5ReducedPacketLPRow(
                        horizon=horizon,
                        state=canonical_state,
                        packet_type=packet_type(canonical_state),
                        gap_vector=_k5_gap_vector(canonical_state),
                        zero_gap_pattern=_k5_zero_gap_pattern(canonical_state),
                        full_alpha=full_alpha,
                        full_packet_weights=full_packet_weights,
                        reduced_alpha=reduced_alpha,
                        reduced_weights=reduced_weights,
                        alpha_difference=reduced_alpha - full_alpha,
                        max_weight_difference=max_weight_difference,
                        count_vector_count=len(_packet_count_vectors(packet_type(canonical_state))),
                        active_constraints=active,
                        dual_marginals=dual_marginals,
                    )
                )
    return tuple(rows)


def _format_count_vector(count_vector: tuple[int, ...]) -> str:
    return "(" + ",".join(str(value) for value in count_vector) + ")"


def print_k5_reduced_packet_lp_report(
    max_used: int = 8,
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    potential_name: str = "Q_chase",
    n: int = 40,
) -> None:
    rows = k5_reduced_packet_lp_rows(
        max_used=max_used,
        h_values=h_values,
        potential_name=potential_name,
    )
    print("k=5 reduced packet LP report")
    print()
    print(f"max_used: {max_used}")
    print(f"h_values: {h_values}")
    print(f"potential: {potential_name}")
    print()

    print("A. Full LP vs reduced packet LP check")
    for horizon in h_values:
        group = [row for row in rows if row.horizon == horizon]
        if not group:
            continue
        print(
            f"  h={horizon:3d}"
            f" states={len(group):5d}"
            f" max_abs_alpha_difference={max(abs(row.alpha_difference) for row in group):.6g}"
            f" max_weight_difference={max(row.max_weight_difference for row in group):.6g}"
        )

    print()
    print("B. Reduced packet LP summary")
    print("h max_alpha avg_positive_alpha h^(3/2)*max packet_regimes avg_constraints avg_active")
    for horizon in h_values:
        group = [row for row in rows if row.horizon == horizon]
        if not group:
            continue
        regimes = {(row.packet_type, row.gap_vector, row.zero_gap_pattern) for row in group}
        print(
            f"{horizon:3d}"
            f" {max(row.reduced_alpha for row in group):9.6f}"
            f" {_positive_average([row.reduced_alpha for row in group]):18.6f}"
            f" {(horizon ** 1.5) * max(row.reduced_alpha for row in group):14.6f}"
            f" {len(regimes):14d}"
            f" {sum(row.count_vector_count for row in group) / len(group):15.3f}"
            f" {sum(len(row.active_constraints) for row in group) / len(group):10.3f}"
        )

    print()
    print("C. Active count-vector templates")
    for row in sorted(rows, key=lambda item: item.reduced_alpha, reverse=True)[:n]:
        rational_weights = _rationalize_packet_weights(row.packet_type, row.reduced_weights)
        active_text = "; ".join(
            f"a={_format_count_vector(count_vector)} residual={residual:.6f} q={q_value:.6f}"
            for count_vector, residual, q_value in row.active_constraints[:12]
        )
        dual_text = "; ".join(
            f"a={_format_count_vector(count_vector)} dual={dual:.6f}"
            for count_vector, dual in row.dual_marginals[:12]
        )
        print(
            f"  h={row.horizon:3d}"
            f" state={row.state}"
            f" packet={row.packet_type}"
            f" gaps={row.gap_vector}"
            f" weights={_format_packet_weights(row.reduced_weights)}"
            f" rational={rational_weights}"
            f" alpha={row.reduced_alpha:.6f}"
        )
        print(f"    active: {active_text}")
        if dual_text:
            print(f"    duals: {dual_text}")

    print()
    print("D. Closed-form candidate equations")
    for row in sorted(rows, key=lambda item: item.reduced_alpha, reverse=True)[:n]:
        print(
            f"  h={row.horizon:3d} state={row.state} packet={row.packet_type}"
            f" alpha={row.reduced_alpha:.6f}"
        )
        for count_vector, residual, q_value in row.active_constraints[:8]:
            terms = " + ".join(f"{count}*w{index + 1}" for index, count in enumerate(count_vector) if count)
            if not terms:
                terms = "0"
            print(
                f"    Rel(next_{_format_count_vector(count_vector)}) - ({terms})"
                f" = Rel(x) + alpha ; q={q_value:.6f}, residual={residual:.6f}"
            )

    print()
    print("E. Scaling by reduced packet pattern")
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[K5ReducedPacketLPRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.packet_type, row.zero_gap_pattern)].append(row)
    for (ptype, zero_pattern), group in sorted(grouped.items(), key=lambda item: (-max(row.reduced_alpha for row in item[1]), item[0]))[:n]:
        pieces = []
        for horizon in h_values:
            horizon_rows = [row for row in group if row.horizon == horizon]
            if not horizon_rows:
                continue
            representative = max(horizon_rows, key=lambda row: row.reduced_alpha)
            scaled = (horizon ** 1.5) * representative.reduced_alpha
            pieces.append(
                f"h={horizon}:alpha={representative.reduced_alpha:.6f},"
                f"h32={scaled:.6f},w={_rationalize_packet_weights(representative.packet_type, representative.reduced_weights)}"
            )
        print(f"  packet={ptype} zero={zero_pattern} " + "; ".join(pieces))

    print()
    print("F. Proof-ready theorem data")
    worst = max(rows, key=lambda row: (row.horizon ** 1.5) * row.reduced_alpha)
    constant = (worst.horizon ** 1.5) * worst.reduced_alpha
    print(f"  worst_observed_C={constant:.6f}")
    print(
        f"  attained_at h={worst.horizon} state={worst.state}"
        f" packet={worst.packet_type} gaps={worst.gap_vector}"
        f" alpha={worst.reduced_alpha:.6f}"
    )
    print(f"  observed alpha <= {constant:.6f} h^(-3/2) on sampled states")


def _gap_bucket(max_gap: int) -> str:
    if max_gap <= 5:
        return str(max_gap)
    if max_gap <= 10:
        return "6-10"
    return ">10"


def _k5_reachable_states_from_support(
    T: int,
    support: tuple[tuple[float, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    current: set[tuple[int, ...]] = {(0, 0, 0, 0, 0)}
    all_reachable: set[tuple[int, ...]] = set(current)
    for _ in range(T):
        next_states: set[tuple[int, ...]] = set()
        for state in current:
            for _, action in support:
                next_states.add(canon(tuple(state[index] + action[index] for index in range(5))))
        current = next_states
        all_reachable.update(current)
    return tuple(sorted(all_reachable, key=lambda state: (sum(state), state)))


def _k5_chase_reachable_states(T: int) -> tuple[tuple[int, ...], ...]:
    return _k5_reachable_states_from_support(T, _k5_chase_support())


def _k5_random_playout_reachable_states(T: int, playout_name: str) -> tuple[tuple[int, ...], ...]:
    return _k5_reachable_states_from_support(T, _random_playout_spec(5, playout_name).support)


def _summarize_k5_reduced_packet_rows(
    rows: tuple[K5ReducedPacketLPRow, ...],
    section: str,
    scope: str,
    horizon: int,
) -> K5ReducedPacketLPCoverageRow:
    if not rows:
        return K5ReducedPacketLPCoverageRow(
            section=section,
            scope=scope,
            horizon=horizon,
            state_count=0,
            max_alpha=0.0,
            avg_positive_alpha=0.0,
            scaled_max_alpha=0.0,
            worst_state=(),
            worst_packet_type=(),
            worst_gap_vector=(),
            worst_weights=(),
            worst_active_count_vectors=(),
        )
    worst = max(rows, key=lambda row: row.reduced_alpha)
    return K5ReducedPacketLPCoverageRow(
        section=section,
        scope=scope,
        horizon=horizon,
        state_count=len(rows),
        max_alpha=worst.reduced_alpha,
        avg_positive_alpha=_positive_average([row.reduced_alpha for row in rows]),
        scaled_max_alpha=(horizon ** 1.5) * worst.reduced_alpha,
        worst_state=worst.state,
        worst_packet_type=worst.packet_type,
        worst_gap_vector=worst.gap_vector,
        worst_weights=worst.reduced_weights,
        worst_active_count_vectors=tuple(count_vector for count_vector, _, _ in worst.active_constraints),
    )


def _k5_reduced_packet_rows_for_states(
    states: tuple[tuple[int, ...], ...],
    horizon: int,
    potential_name: str,
) -> tuple[K5ReducedPacketLPRow, ...]:
    current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
    rows: list[K5ReducedPacketLPRow] = []
    for state in states:
        rows.append(_k5_reduced_packet_row_for_state(state, horizon, current_value, next_value))
    return tuple(rows)


def _k5_reduced_packet_row_for_state(
    state: tuple[int, ...],
    horizon: int,
    current_value,
    next_value,
) -> K5ReducedPacketLPRow:
    canonical_state = canon(state)
    reduced_weights, reduced_alpha, active, dual_marginals = _reduced_packet_lp(
        canonical_state,
        horizon,
        current_value,
        next_value,
    )
    return K5ReducedPacketLPRow(
        horizon=horizon,
        state=canonical_state,
        packet_type=packet_type(canonical_state),
        gap_vector=_k5_gap_vector(canonical_state),
        zero_gap_pattern=_k5_zero_gap_pattern(canonical_state),
        full_alpha=reduced_alpha,
        full_packet_weights=reduced_weights,
        reduced_alpha=reduced_alpha,
        reduced_weights=reduced_weights,
        alpha_difference=0.0,
        max_weight_difference=0.0,
        count_vector_count=len(_packet_count_vectors(packet_type(canonical_state))),
        active_constraints=active,
        dual_marginals=dual_marginals,
    )


def _parse_reachable_mode(text: str) -> tuple[str, ...]:
    if text == "none":
        return ()
    if text == "both":
        return ("chase", "random_playout")
    if text in {"chase", "random_playout"}:
        return (text,)
    raise ValueError(f"unknown reachable mode: {text}")


def _coverage_progress(message: str, progress_file: str | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, file=sys.stderr, flush=True)
    if progress_file:
        with open(progress_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _k5_coverage_states_for_scope(
    section: str,
    scope: str,
    potential_name: str,
    sample_limit: int,
) -> tuple[tuple[int, ...], ...]:
    if section == "max_used":
        max_used = int(scope)
        states = tuple(state for used in range(max_used + 1) for state in all_states(5, used))
    elif section == "chase":
        states = _k5_chase_reachable_states(int(scope))
    elif section == "random_playout":
        states = _k5_random_playout_reachable_states(int(scope), potential_name)
    else:
        raise ValueError(f"unknown coverage section: {section}")
    if sample_limit > 0:
        states = states[:sample_limit]
    return states


def _k5_reduced_packet_lp_coverage_task(
    task: tuple[str, str, int, str, int],
) -> tuple[K5ReducedPacketLPCoverageRow, tuple[K5ReducedPacketLPRow, ...], float]:
    section, scope, horizon, potential_name, sample_limit = task
    start = time.time()
    states = _k5_coverage_states_for_scope(section, scope, potential_name, sample_limit)
    current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
    state_count = 0
    positive_sum = 0.0
    positive_count = 0
    worst_row: K5ReducedPacketLPRow | None = None
    compact_by_group: dict[tuple[str, tuple[int, ...], tuple[int, ...]], K5ReducedPacketLPRow] = {}
    top_rows: list[K5ReducedPacketLPRow] = []
    for state in states:
        row = _k5_reduced_packet_row_for_state(state, horizon, current_value, next_value)
        state_count += 1
        if row.reduced_alpha > 0.0:
            positive_sum += row.reduced_alpha
            positive_count += 1
        if worst_row is None or row.reduced_alpha > worst_row.reduced_alpha:
            worst_row = row
        max_gap = max(row.gap_vector, default=0)
        group_key = (_gap_bucket(max_gap), row.packet_type, row.zero_gap_pattern)
        existing = compact_by_group.get(group_key)
        if existing is None or (row.horizon ** 1.5) * row.reduced_alpha > (existing.horizon ** 1.5) * existing.reduced_alpha:
            compact_by_group[group_key] = row
        if max_gap >= 6 or len(top_rows) < 64:
            top_rows.append(row)
            top_rows = sorted(top_rows, key=lambda item: (item.horizon ** 1.5) * item.reduced_alpha, reverse=True)[:64]
    if worst_row is None:
        summary = _summarize_k5_reduced_packet_rows((), section, scope, horizon)
        return summary, (), time.time() - start
    summary = K5ReducedPacketLPCoverageRow(
        section=section,
        scope=scope,
        horizon=horizon,
        state_count=state_count,
        max_alpha=worst_row.reduced_alpha,
        avg_positive_alpha=positive_sum / positive_count if positive_count else 0.0,
        scaled_max_alpha=(horizon ** 1.5) * worst_row.reduced_alpha,
        worst_state=worst_row.state,
        worst_packet_type=worst_row.packet_type,
        worst_gap_vector=worst_row.gap_vector,
        worst_weights=worst_row.reduced_weights,
        worst_active_count_vectors=tuple(count_vector for count_vector, _, _ in worst_row.active_constraints),
    )
    compact_rows = {
        (row.horizon, row.state, row.packet_type, row.gap_vector): row
        for row in tuple(compact_by_group.values()) + tuple(top_rows) + (worst_row,)
    }
    return summary, tuple(compact_rows.values()), time.time() - start


def k5_reduced_packet_lp_coverage_rows(
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    max_used_values: tuple[int, ...] = (8, 12, 16, 20),
    potential_name: str = "Q_chase",
    reachable_mode: str = "both",
    reachable_T_values: tuple[int, ...] = (20, 50, 100),
    sample_limit: int = 0,
    jobs: int = 1,
    progress_file: str | None = None,
) -> tuple[tuple[K5ReducedPacketLPCoverageRow, ...], tuple[K5ReducedPacketLPRow, ...]]:
    summaries: list[K5ReducedPacketLPCoverageRow] = []
    detail_rows: list[K5ReducedPacketLPRow] = []
    tasks: list[tuple[str, str, int, str, int]] = []
    for max_used in max_used_values:
        for horizon in h_values:
            tasks.append(("max_used", str(max_used), horizon, potential_name, sample_limit))

    for mode in _parse_reachable_mode(reachable_mode):
        for T in reachable_T_values:
            for horizon in h_values:
                tasks.append((mode, str(T), horizon, potential_name, sample_limit))

    _coverage_progress(
        f"coverage start tasks={len(tasks)} jobs={max(jobs, 1)} potential={potential_name}",
        progress_file,
    )
    if jobs <= 1:
        for index, task in enumerate(tasks, start=1):
            section, scope, horizon, _, _ = task
            _coverage_progress(f"start {index}/{len(tasks)} section={section} scope={scope} h={horizon}", progress_file)
            summary, rows, elapsed = _k5_reduced_packet_lp_coverage_task(task)
            summaries.append(summary)
            detail_rows.extend(rows)
            _coverage_progress(
                f"done {index}/{len(tasks)} section={section} scope={scope} h={horizon}"
                f" states={summary.state_count} h32max={summary.scaled_max_alpha:.6f}"
                f" elapsed={elapsed:.2f}s",
                progress_file,
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_k5_reduced_packet_lp_coverage_task, task): task for task in tasks}
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                section, scope, horizon, _, _ = task
                summary, rows, elapsed = future.result()
                completed += 1
                summaries.append(summary)
                detail_rows.extend(rows)
                _coverage_progress(
                    f"done {completed}/{len(tasks)} section={section} scope={scope} h={horizon}"
                    f" states={summary.state_count} h32max={summary.scaled_max_alpha:.6f}"
                    f" elapsed={elapsed:.2f}s",
                    progress_file,
                )
    _coverage_progress(f"coverage complete tasks={len(tasks)} compact_detail_rows={len(detail_rows)}", progress_file)
    return tuple(summaries), tuple(detail_rows)


def print_k5_reduced_packet_lp_coverage_report(
    h_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    max_used_values: tuple[int, ...] = (8, 12, 16, 20),
    potential_name: str = "Q_chase",
    reachable_mode: str = "both",
    reachable_T_values: tuple[int, ...] = (20, 50, 100),
    sample_limit: int = 0,
    jobs: int = 1,
    progress_file: str | None = None,
    n: int = 40,
) -> None:
    if progress_file:
        with open(progress_file, "w", encoding="utf-8") as handle:
            handle.write("")
    summaries, detail_rows = k5_reduced_packet_lp_coverage_rows(
        h_values=h_values,
        max_used_values=max_used_values,
        potential_name=potential_name,
        reachable_mode=reachable_mode,
        reachable_T_values=reachable_T_values,
        sample_limit=sample_limit,
        jobs=jobs,
        progress_file=progress_file,
    )
    print("k=5 reduced packet LP coverage report")
    print()
    print(f"h_values: {h_values}")
    print(f"max_used_values: {max_used_values}")
    print(f"potential: {potential_name}")
    print(f"reachable_mode: {reachable_mode}")
    print(f"reachable_T_values: {reachable_T_values}")
    print(f"sample_limit: {sample_limit}")
    print(f"jobs: {jobs}")
    print(f"progress_file: {progress_file}")
    print()

    print("A. max_used sweep")
    for row in [summary for summary in summaries if summary.section == "max_used"]:
        print(
            f"  max_used={row.scope:>3s}"
            f" h={row.horizon:3d}"
            f" states={row.state_count:6d}"
            f" max_alpha={row.max_alpha:.6f}"
            f" avg_pos={row.avg_positive_alpha:.6f}"
            f" h32max={row.scaled_max_alpha:.6f}"
            f" worst_state={row.worst_state}"
            f" packet={row.worst_packet_type}"
            f" gaps={row.worst_gap_vector}"
        )

    print()
    print("B. reachable-state sweep")
    for row in [summary for summary in summaries if summary.section != "max_used"]:
        print(
            f"  mode={row.section:14s}"
            f" T={row.scope:>3s}"
            f" h={row.horizon:3d}"
            f" states={row.state_count:6d}"
            f" max_alpha={row.max_alpha:.6f}"
            f" avg_pos={row.avg_positive_alpha:.6f}"
            f" h32max={row.scaled_max_alpha:.6f}"
            f" worst_state={row.worst_state}"
            f" packet={row.worst_packet_type}"
            f" gaps={row.worst_gap_vector}"
        )

    print()
    print("C. gap-size stratification")
    grouped: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[K5ReducedPacketLPRow]] = defaultdict(list)
    for row in detail_rows:
        max_gap = max(row.gap_vector, default=0)
        grouped[(_gap_bucket(max_gap), row.packet_type, row.zero_gap_pattern)].append(row)
    for (bucket, ptype, zero_pattern), rows in sorted(
        grouped.items(),
        key=lambda item: (-max((row.horizon ** 1.5) * row.reduced_alpha for row in item[1]), item[0]),
    )[:n]:
        worst = max(rows, key=lambda row: (row.horizon ** 1.5) * row.reduced_alpha)
        print(
            f"  bucket={bucket:>4s}"
            f" packet={ptype}"
            f" zero={zero_pattern}"
            f" max_h32alpha={(worst.horizon ** 1.5) * worst.reduced_alpha:.6f}"
            f" state={worst.state}"
            f" h={worst.horizon}"
            f" alpha={worst.reduced_alpha:.6f}"
        )

    print()
    print("D. saturation / large-gap sanity")
    large_gap_rows = [row for row in detail_rows if max(row.gap_vector, default=0) >= 6]
    for row in sorted(large_gap_rows, key=lambda item: (item.horizon ** 1.5) * item.reduced_alpha, reverse=True)[:n]:
        print(
            f"  h={row.horizon:3d}"
            f" state={row.state}"
            f" gaps={row.gap_vector}"
            f" packet={row.packet_type}"
            f" alpha={row.reduced_alpha:.6f}"
            f" h32alpha={(row.horizon ** 1.5) * row.reduced_alpha:.6f}"
            f" weights={_format_packet_weights(row.reduced_weights)}"
            f" active={','.join(_format_count_vector(count_vector) for count_vector, _, _ in row.active_constraints[:8])}"
        )

    print()
    print("E. proof-ready bound summary")
    worst = max(detail_rows, key=lambda row: (row.horizon ** 1.5) * row.reduced_alpha)
    constant = (worst.horizon ** 1.5) * worst.reduced_alpha
    print(f"  global observed C = {constant:.6f}")
    print(
        f"  attained_at h={worst.horizon}"
        f" state={worst.state}"
        f" packet={worst.packet_type}"
        f" gaps={worst.gap_vector}"
        f" alpha={worst.reduced_alpha:.6f}"
    )
    print(f"  exceeds_0.7={constant > 0.7}")
    print(f"  exceeds_1.0={constant > 1.0}")
    print(f"  exceeds_2.0={constant > 2.0}")
    print(f"  observed alpha <= {constant:.6f} h^(-3/2) on checked states")


def _parse_float_grid(text: str | None) -> tuple[float, ...]:
    if text is None or not text.strip():
        return (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    return tuple(float(piece.strip()) for piece in text.split(",") if piece.strip())


def _k5_worst_family_state(
    family: str,
    horizon: int,
    s1: float,
    s2: float,
    s3: float = 0.0,
) -> tuple[int, ...]:
    scale = horizon ** 0.5
    a = int(round(s1 * scale))
    b = int(round(s2 * scale))
    c = int(round(s3 * scale))
    if family == "packet_122":
        return canon((a + b, b, b, 0, 0))
    if family == "packet_1211":
        # Use a tiny lower split by default so the bottom pair is broken.
        c = c if c > 0 else 1
        return canon((a + b + c, b + c, b + c, c, 0))
    if family == "packet_221":
        return canon((a + b, a + b, b, b, 0))
    if family == "packet_212":
        return canon((a + b, a + b, b, 0, 0))
    raise ValueError(f"unknown worst-family name: {family}")


def _parse_k5_worst_families(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip() or text.strip() == "all":
        return ("packet_122", "packet_1211", "packet_221", "packet_212")
    families = tuple(piece.strip() for piece in text.split(",") if piece.strip())
    for family in families:
        _k5_worst_family_state(family, 10, 1.0, 1.0)
    return families


def _k5_worst_family_scaling_rows_for_horizon(
    horizon: int,
    families: tuple[str, ...],
    grid: tuple[float, ...],
    potential_name: str,
    progress_file: str | None = None,
    progress_every: int = 25,
) -> tuple[K5WorstFamilyScalingRow, ...]:
    rows: list[K5WorstFamilyScalingRow] = []
    current_value, next_value = _k5_relaxation_value_functions(potential_name, horizon)
    seen: set[tuple[str, tuple[int, ...], float, float, float]] = set()
    total_grid_points = sum(
        len(grid) * len(grid) * (2 if family == "packet_1211" else 1)
        for family in families
    )
    completed_grid_points = 0
    best_scaled = float("-inf")
    for family in families:
        for s1 in grid:
            for s2 in grid:
                s3_values = (0.0, 1.0 / (horizon ** 0.5)) if family == "packet_1211" else (0.0,)
                for s3 in s3_values:
                    state = _k5_worst_family_state(family, horizon, s1, s2, s3)
                    key = (family, state, s1, s2, s3)
                    if key in seen:
                        continue
                    seen.add(key)
                    weights, alpha, active, _ = _reduced_packet_lp(state, horizon, current_value, next_value)
                    rows.append(
                        K5WorstFamilyScalingRow(
                            horizon=horizon,
                            family=family,
                            state=state,
                            s1=s1,
                            s2=s2,
                            s3=s3,
                            alpha=alpha,
                            scaled_alpha=(horizon ** 1.5) * alpha,
                            weights=weights,
                            active_count_vectors=tuple(count_vector for count_vector, _, _ in active),
                            active_constraints=active,
                        )
                    )
                    best_scaled = max(best_scaled, rows[-1].scaled_alpha)
                    completed_grid_points += 1
                    if progress_every > 0 and completed_grid_points % progress_every == 0:
                        _coverage_progress(
                            f"h={horizon} grid={completed_grid_points}/{total_grid_points}"
                            f" rows={len(rows)} best_C={best_scaled:.6f}",
                            progress_file,
                        )
    return tuple(rows)


def _k5_worst_family_scaling_task(
    task: tuple[int, tuple[str, ...], tuple[float, ...], str, str | None, int],
) -> tuple[int, tuple[K5WorstFamilyScalingRow, ...], float]:
    horizon, families, grid, potential_name, progress_file, progress_every = task
    start = time.time()
    rows = _k5_worst_family_scaling_rows_for_horizon(
        horizon,
        families,
        grid,
        potential_name,
        progress_file=progress_file,
        progress_every=progress_every,
    )
    return horizon, rows, time.time() - start


def k5_worst_family_scaling_rows(
    h_values: tuple[int, ...] = (50, 100, 200, 500),
    families: tuple[str, ...] = ("packet_122", "packet_1211", "packet_221", "packet_212"),
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    potential_name: str = "Q_chase",
    jobs: int = 1,
    progress_file: str | None = None,
    start_method: str = "spawn",
    memory_safe_above: int = 150,
    progress_every: int = 25,
) -> tuple[K5WorstFamilyScalingRow, ...]:
    rows: list[K5WorstFamilyScalingRow] = []
    tasks = tuple((horizon, families, grid, potential_name, progress_file, progress_every) for horizon in h_values)
    _coverage_progress(
        f"worst-family scaling start horizons={h_values} jobs={max(jobs, 1)}"
        f" start_method={start_method} memory_safe_above={memory_safe_above}",
        progress_file,
    )
    if jobs <= 1:
        for index, task in enumerate(tasks, start=1):
            horizon, horizon_rows, elapsed = _k5_worst_family_scaling_task(task)
            rows.extend(horizon_rows)
            worst = max(horizon_rows, key=lambda row: row.scaled_alpha)
            _coverage_progress(
                f"done {index}/{len(tasks)} h={horizon} rows={len(horizon_rows)}"
                f" max_C={worst.scaled_alpha:.6f} elapsed={elapsed:.2f}s",
                progress_file,
            )
    else:
        if memory_safe_above > 0:
            parallel_tasks = tuple(task for task in tasks if task[0] <= memory_safe_above)
            serial_tasks = tuple(task for task in tasks if task[0] > memory_safe_above)
        else:
            parallel_tasks = tasks
            serial_tasks = ()
        context = get_context(start_method)
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            futures = {executor.submit(_k5_worst_family_scaling_task, task): task for task in parallel_tasks}
            completed = 0
            for future in as_completed(futures):
                horizon, horizon_rows, elapsed = future.result()
                rows.extend(horizon_rows)
                completed += 1
                worst = max(horizon_rows, key=lambda row: row.scaled_alpha)
                _coverage_progress(
                    f"done parallel {completed}/{len(parallel_tasks)} h={horizon} rows={len(horizon_rows)}"
                    f" max_C={worst.scaled_alpha:.6f} elapsed={elapsed:.2f}s",
                    progress_file,
                )
        for serial_index, task in enumerate(serial_tasks, start=1):
            horizon = task[0]
            _coverage_progress(
                f"start memory-safe serial {serial_index}/{len(serial_tasks)} h={horizon}",
                progress_file,
            )
            horizon, horizon_rows, elapsed = _k5_worst_family_scaling_task(task)
            rows.extend(horizon_rows)
            worst = max(horizon_rows, key=lambda row: row.scaled_alpha)
            _coverage_progress(
                f"done memory-safe serial {serial_index}/{len(serial_tasks)} h={horizon}"
                f" rows={len(horizon_rows)} max_C={worst.scaled_alpha:.6f} elapsed={elapsed:.2f}s",
                progress_file,
            )
    _coverage_progress(f"worst-family scaling complete rows={len(rows)}", progress_file)
    return tuple(rows)


def print_k5_worst_family_scaling_report(
    h_values: tuple[int, ...] = (50, 100, 200, 500),
    families: tuple[str, ...] = ("packet_122", "packet_1211", "packet_221", "packet_212"),
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
    potential_name: str = "Q_chase",
    jobs: int = 1,
    progress_file: str | None = None,
    start_method: str = "spawn",
    memory_safe_above: int = 150,
    progress_every: int = 25,
    n: int = 40,
) -> None:
    if progress_file:
        with open(progress_file, "w", encoding="utf-8") as handle:
            handle.write("")
    rows = k5_worst_family_scaling_rows(
        h_values=h_values,
        families=families,
        grid=grid,
        potential_name=potential_name,
        jobs=jobs,
        progress_file=progress_file,
        start_method=start_method,
        memory_safe_above=memory_safe_above,
        progress_every=progress_every,
    )
    print("k=5 worst-family scaling report")
    print()
    print(f"h_values: {h_values}")
    print(f"families: {families}")
    print(f"grid: {grid}")
    print(f"potential: {potential_name}")
    print(f"jobs: {jobs}")
    print(f"start_method: {start_method}")
    print(f"memory_safe_above: {memory_safe_above}")
    print(f"progress_every: {progress_every}")
    print(f"progress_file: {progress_file}")
    print()

    print("A. Worst by h and family")
    for horizon in h_values:
        for family in families:
            group = [row for row in rows if row.horizon == horizon and row.family == family]
            if not group:
                continue
            worst = max(group, key=lambda row: row.scaled_alpha)
            print(
                f"  h={horizon:4d}"
                f" family={family:12s}"
                f" C={worst.scaled_alpha:.6f}"
                f" alpha={worst.alpha:.6f}"
                f" state={worst.state}"
                f" s=({worst.s1:.2f},{worst.s2:.2f},{worst.s3:.2f})"
                f" weights={_format_packet_weights(worst.weights)}"
                f" active={','.join(_format_count_vector(vector) for vector in worst.active_count_vectors)}"
            )

    print()
    print("B. Scaling convergence table")
    top_scaled_points = sorted(
        {(row.family, row.s1, row.s2, row.s3) for row in rows},
        key=lambda key: -max(
            row.scaled_alpha
            for row in rows
            if (row.family, row.s1, row.s2, row.s3) == key
        ),
    )[:n]
    for family, s1, s2, s3 in top_scaled_points:
        pieces = []
        for horizon in h_values:
            matches = [
                row
                for row in rows
                if row.horizon == horizon and row.family == family and row.s1 == s1 and row.s2 == s2 and row.s3 == s3
            ]
            if matches:
                pieces.append(f"h={horizon}:C={matches[0].scaled_alpha:.6f}")
        print(f"  family={family} s=({s1:.2f},{s2:.2f},{s3:.2f}) " + "; ".join(pieces))

    print()
    print("C. Heatmap-like grid table for packet_122")
    for horizon in h_values:
        print(f"  h={horizon}")
        header = "s1\\s2 " + " ".join(f"{s2:>7.2f}" for s2 in grid)
        print("    " + header)
        for s1 in grid:
            values = []
            for s2 in grid:
                match = next(
                    (
                        row
                        for row in rows
                        if row.horizon == horizon and row.family == "packet_122" and row.s1 == s1 and row.s2 == s2
                    ),
                    None,
                )
                values.append(f"{match.scaled_alpha:7.3f}" if match is not None else "   -   ")
            print(f"    {s1:5.2f} " + " ".join(values))

    print()
    print("D. Active-set stability")
    grouped: dict[tuple[str, tuple[tuple[int, ...], ...]], list[K5WorstFamilyScalingRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.active_count_vectors)].append(row)
    for (family, active_set), group in sorted(grouped.items(), key=lambda item: (-max(row.scaled_alpha for row in item[1]), item[0]))[:n]:
        worst = max(group, key=lambda row: row.scaled_alpha)
        print(
            f"  family={family:12s}"
            f" active={','.join(_format_count_vector(vector) for vector in active_set)}"
            f" count={len(group)}"
            f" max_C={worst.scaled_alpha:.6f}"
            f" worst_h={worst.horizon}"
            f" state={worst.state}"
        )

    print()
    print("E. Candidate limiting worst point")
    for family in families:
        group = [row for row in rows if row.family == family]
        if not group:
            continue
        worst = max(group, key=lambda row: row.scaled_alpha)
        print(
            f"  family={family:12s}"
            f" best_s=({worst.s1:.2f},{worst.s2:.2f},{worst.s3:.2f})"
            f" C={worst.scaled_alpha:.6f}"
            f" h={worst.horizon}"
            f" state={worst.state}"
        )

    print()
    print("F. Proof hint equations")
    worst = max(rows, key=lambda row: row.scaled_alpha)
    print(
        f"  worst family={worst.family} h={worst.horizon} state={worst.state}"
        f" C={worst.scaled_alpha:.6f} alpha={worst.alpha:.6f}"
    )
    simplex = " + ".join(f"{size}*w{index + 1}" for index, size in enumerate(packet_type(worst.state)))
    print(f"    simplex: {simplex} = 1")
    for count_vector, residual, q_value in worst.active_constraints:
        terms = " + ".join(f"{count}*w{index + 1}" for index, count in enumerate(count_vector) if count)
        if not terms:
            terms = "0"
        print(
            f"    q({_format_count_vector(count_vector)}) - ({terms}) = alpha"
            f" ; q={q_value:.6f} residual={residual:.6f}"
        )


def _k6_fixed_pattern_names() -> tuple[str, ...]:
    return (
        "fixed_101010_comb",
        "fixed_101011_truncated",
        "fixed_101000_prefix3",
        "fixed_100000_singleton",
        "top_prefix_all",
    )


def print_k6_adaptive_interval_anatomy_report(
    T: int,
    long_T_values: tuple[int, ...],
    support_tolerance: float = 1e-8,
    n: int = 40,
) -> None:
    zero = (0, 0, 0, 0, 0, 0)
    fixed_names = _k6_fixed_pattern_names()
    print("k=6 adaptive interval anatomy report")
    print()
    print(f"T: {T}")
    print(f"long T values: {long_T_values}")
    print()

    print("Fixed pattern comparison")
    print(
        "T policy                           V_policy V_policy/sqrt(T)"
        " gap_to_top_prefix gap_to_top_prefix/sqrt(T)"
    )
    for horizon in (T,) + tuple(long_T_values):
        top_prefix = evaluate_time_dependent_policy_reachable_root(
            6,
            horizon,
            _resolve_time_dependent_policy(6, "top_prefix_all"),
        ).value
        for name in fixed_names:
            policy_fn = _resolve_time_dependent_policy(6, name)
            if policy_fn is None:
                raise ValueError(f"unknown policy: {name}")
            value = top_prefix if name == "top_prefix_all" else evaluate_time_dependent_policy_reachable_root(
                6,
                horizon,
                policy_fn,
            ).value
            gap = top_prefix - value
            sqrt_horizon = horizon ** 0.5
            print(
                f"{horizon:3d} {name:31s}"
                f" {value:10.6f}"
                f" {value / sqrt_horizon:16.6f}"
                f" {gap:17.6f}"
                f" {gap / sqrt_horizon:26.6f}"
            )
    print()

    print("Reference restricted LP values at T")
    for library_name in ("top_prefix_all", "one_run", "two_run", "local_edges"):
        value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, library_name))[T][zero]
        print(f"  {library_name:14s} V={value:.6f}")
    print()

    _, top_rows = library_lp_dual_inspect_rows(6, T, "top_prefix_all", support_tolerance=support_tolerance)
    print("Top-prefix interval mass")
    global_interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
    by_horizon: dict[int, dict[tuple[int, int] | None, float]] = defaultdict(lambda: defaultdict(float))
    by_regime: dict[tuple[int, tuple[int, ...], tuple[int, ...]], dict[tuple[int, int] | None, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in top_rows:
        for weight, action, _ in row.support:
            interval = _interval_from_action(action)
            mass = row.occupancy_probability * weight
            global_interval_totals[interval] += mass
            by_horizon[row.remaining_horizon][interval] += mass
            by_regime[(row.remaining_horizon, row.packet_type, row.packet_gaps)][interval] += mass
    print(f"  global: {_format_interval_distribution(global_interval_totals)}")
    print("  by remaining horizon:")
    for remaining_horizon, totals in sorted(by_horizon.items(), reverse=True):
        print(f"    rem={remaining_horizon:2d}: {_format_interval_distribution(totals)}")
    print("  top regime interval distributions:")
    regime_masses = {
        key: sum(totals.values())
        for key, totals in by_regime.items()
    }
    for (remaining_horizon, ptype, gaps), _ in sorted(regime_masses.items(), key=lambda item: (-item[1], item[0]))[:n]:
        print(
            f"    rem={remaining_horizon:2d} packet={ptype} gaps={gaps}"
            f" mass={regime_masses[(remaining_horizon, ptype, gaps)]:.6f}"
            f" intervals={_format_interval_distribution(by_regime[(remaining_horizon, ptype, gaps)])}"
        )
    print()

    print("Top-prefix top rows by occupancy")
    for row in sorted(top_rows, key=lambda item: (-item.occupancy_probability, item.time, item.state))[:n]:
        interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
        for weight, action, _ in row.support:
            interval_totals[_interval_from_action(action)] += weight
        print(
            f"  rem={row.remaining_horizon:2d} state={row.state}"
            f" packet={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
            f" intervals={_format_interval_distribution(interval_totals)}"
        )
    print()

    print("One-run vs top-prefix differences")
    one_run_actions = _strategy_class_library_actions(6, T, "one_run")
    one_run_values = _library_lp_value_layers(6, T, one_run_actions)
    one_run_interval_rows = {
        (row.time, row.state): row
        for row in k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance)
    }
    correction_rows: list[tuple[float, LibraryLPDualInspectRow, float, tuple[tuple[tuple[int, int] | None, float], ...]]] = []
    for row in top_rows:
        continuation = one_run_values[row.remaining_horizon - 1]
        top_policy = tuple((weight, action) for weight, action, _ in row.support)
        top_step = _time_policy_step_value(6, row.state, top_policy, continuation)
        one_run_value = one_run_values[row.remaining_horizon][row.state]
        action_loss = one_run_value - top_step
        interval_row = one_run_interval_rows.get((row.time, row.state))
        if interval_row is None:
            q_by_action = {
                action: _next_state_value(row.state, action, continuation)
                for action in one_run_actions
            }
            dual = solve_adversary_dual(q_by_action, 6)
            support = tuple(
                K6OneRunIntervalActionRow(
                    weight=weight,
                    action=action,
                    edge_signature=_edge_signature(action),
                    interval=_interval_from_action(action),
                    relation_class=_classify_k6_one_run_interval_relation(_interval_from_action(action), row.state),
                    successor_packet_type=packet_type(canon(tuple(row.state[index] + action[index] for index in range(6)))),
                )
                for action, weight in dual.weights_by_action
                if weight >= support_tolerance
            )
        else:
            support = interval_row.support
        interval_distribution = _interval_distribution_for_interval_support(support)
        correction_rows.append((row.occupancy_probability * action_loss, row, action_loss, interval_distribution))
    for weighted_loss, row, action_loss, interval_distribution in sorted(
        correction_rows,
        key=lambda item: (-item[0], item[1].time, item[1].state),
    )[:n]:
        top_interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
        for weight, action, _ in row.support:
            top_interval_totals[_interval_from_action(action)] += weight
        one_run_intervals = ", ".join(
            f"{_format_interval(interval)}:{weight:.3f}"
            for interval, weight in interval_distribution[:5]
        )
        print(
            f"  rem={row.remaining_horizon:2d}"
            f" packet={row.packet_type} gaps={row.packet_gaps}"
            f" state={row.state}"
            f" occ={row.occupancy_probability:.6f}"
            f" action_loss={action_loss:.6f}"
            f" weighted_loss={weighted_loss:.6f}"
            f" top_prefix={_format_interval_distribution(top_interval_totals)}"
            f" one_run=[{one_run_intervals}]"
        )
    print()

    print("Candidate distillation target")
    dominant_rules = 0
    for (remaining_horizon, ptype, gaps), totals in sorted(regime_masses.items(), key=lambda item: (-item[1], item[0]))[:n]:
        interval_totals = by_regime[(remaining_horizon, ptype, gaps)]
        total = sum(interval_totals.values())
        best_interval, best_mass = max(interval_totals.items(), key=lambda item: item[1])
        if total > 0 and best_mass / total >= 0.9:
            dominant_rules += 1
            print(
                f"  top-prefix rule: rem={remaining_horizon} packet={ptype} gaps={gaps}"
                f" -> interval={_format_interval(best_interval)} mass_share={best_mass / total:.3f}"
            )
    if dominant_rules == 0:
        print("  no >=90% top-prefix regime rules in printed set")
    print("  missing correction candidates:")
    tried_regimes = {
        (patch.remaining_horizon, patch.packet_type, patch.packet_gaps)
        for patch in _k6_one_run_patch_list()
    }
    tried_regimes.add((5, (1, 2, 2, 1), (1, 1, 2)))
    for weighted_loss, row, _, interval_distribution in sorted(correction_rows, key=lambda item: (-item[0], item[1].state))[:n]:
        if not interval_distribution:
            continue
        interval, weight = interval_distribution[0]
        if interval is None or interval[0] == 0 or weight < 0.5:
            continue
        regime = (row.remaining_horizon, row.packet_type, row.packet_gaps)
        print(
            f"    rem={row.remaining_horizon} packet={row.packet_type} gaps={row.packet_gaps}"
            f" nonprefix={_format_interval(interval)} mass={weight:.3f}"
            f" weighted_loss={weighted_loss:.6f}"
            f" tried={'yes' if regime in tried_regimes else 'no'}"
        )


def _format_interval_distribution_tuple(
    distribution: tuple[tuple[tuple[int, int] | None, float], ...],
) -> str:
    return ", ".join(f"{_format_interval(interval)}:{weight:.3f}" for interval, weight in distribution) or "-"


def _barycenter_global_summary(rows: tuple[BarycenterActionRow, ...]):
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for packet in row.packet_summaries:
            key = (row.library_name, packet.status)
            weight = row.occupancy_probability * packet.packet_size
            totals[key]["mass"] += weight
            totals[key]["exposure"] += weight * packet.exposure
            totals[key]["split_rate"] += weight * packet.split_rate
    return totals


def _barycenter_top_bottom_summary(rows: tuple[BarycenterActionRow, ...]):
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        totals[row.library_name]["occupancy"] += row.occupancy_probability
        totals[row.library_name]["top1_bottom0"] += row.occupancy_probability * row.p_top1_bottom0
        totals[row.library_name]["top0_bottom1"] += row.occupancy_probability * row.p_top0_bottom1
        totals[row.library_name]["top_equals_bottom"] += row.occupancy_probability * row.p_top_equals_bottom
        totals[row.library_name]["covariance"] += row.occupancy_probability * row.top_bottom_covariance
    return totals


def _format_packet_statuses(row: BarycenterActionRow) -> str:
    return ",".join(
        f"{summary.packet_index}:{summary.status}:size{summary.packet_size}:exp{summary.exposure:.3f}:split{summary.split_rate:.3f}"
        for summary in row.packet_summaries
    )


def _classify_one_run_correction(distribution: tuple[tuple[tuple[int, int] | None, float], ...]) -> str:
    intervals = {interval for interval, weight in distribution if weight > 1e-8}
    if intervals == {(0, 0)}:
        return "top-alone"
    if intervals == {(0, 4)}:
        return "full-comb"
    if intervals == {(1, 4)}:
        return "tail-chase"
    if {(0, 0), (0, 4), (1, 4)}.issubset(intervals):
        return "mixed top/full/tail"
    if any(interval is not None and interval[0] > 0 and interval[1] < 4 for interval in intervals):
        return "middle-repair"
    return "other"


def print_k6_barycenter_action_anatomy_report(
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 30,
) -> None:
    _, top_dual_rows = library_lp_dual_inspect_rows(6, T, "top_prefix_all", support_tolerance=support_tolerance)
    _, one_dual_rows = library_lp_dual_inspect_rows(6, T, "one_run", support_tolerance=support_tolerance)
    rows = tuple(
        [_barycenter_action_row_from_dual_row("top_prefix_all", row) for row in top_dual_rows]
        + [_barycenter_action_row_from_dual_row("one_run", row) for row in one_dual_rows]
    )
    print("k=6 barycenter action anatomy report")
    print()
    print(f"T: {T}")
    print(f"support tolerance: {support_tolerance:.3g}")
    print()

    print("Global barycenter exposure summary")
    summary = _barycenter_global_summary(rows)
    for library_name in ("top_prefix_all", "one_run"):
        print(f"  library={library_name}")
        for status in ("above_mean", "crosses_mean", "below_mean"):
            item = summary.get((library_name, status), {})
            mass = item.get("mass", 0.0)
            exposure = item.get("exposure", 0.0) / mass if mass else 0.0
            split_rate = item.get("split_rate", 0.0) / mass if mass else 0.0
            print(
                f"    {status:12s}"
                f" packet_mass={mass:.6f}"
                f" exposure={exposure:.6f}"
                f" split_rate={split_rate:.6f}"
            )
    print()

    print("Top-bottom correlation summary")
    top_bottom = _barycenter_top_bottom_summary(rows)
    for library_name in ("top_prefix_all", "one_run"):
        item = top_bottom[library_name]
        occupancy = item["occupancy"] or 1.0
        print(
            f"  {library_name:14s}"
            f" P(top=1,bottom=0)={item['top1_bottom0'] / occupancy:.6f}"
            f" P(top=0,bottom=1)={item['top0_bottom1'] / occupancy:.6f}"
            f" P(top=bottom)={item['top_equals_bottom'] / occupancy:.6f}"
            f" covariance={item['covariance'] / occupancy:.6f}"
        )
    print()

    top_rows_by_key = {
        (row.time, row.state): row for row in rows if row.library_name == "top_prefix_all"
    }
    top_dual_rows_by_key = {(row.time, row.state): row for row in top_dual_rows}
    one_rows_by_key = {
        (row.time, row.state): row for row in rows if row.library_name == "one_run"
    }
    one_run_values = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "one_run"))
    differences: list[tuple[float, BarycenterActionRow, BarycenterActionRow, float]] = []
    for key, top_row in top_rows_by_key.items():
        if key not in one_rows_by_key:
            continue
        continuation = one_run_values[top_row.remaining_horizon - 1]
        top_support = tuple((weight, action) for weight, action, _ in top_dual_rows_by_key[key].support)
        top_step = _time_policy_step_value(6, top_row.state, top_support, continuation)
        one_value = one_run_values[top_row.remaining_horizon][top_row.state]
        action_loss = one_value - top_step
        differences.append((top_row.occupancy_probability * action_loss, top_row, one_rows_by_key[key], action_loss))

    print("Top one_run-vs-top_prefix difference regimes")
    for weighted_loss, top_row, one_row, action_loss in sorted(differences, key=lambda item: (-item[0], item[1].state))[:n]:
        print(
            f"  rem={top_row.remaining_horizon:2d}"
            f" state={top_row.state}"
            f" mean={top_row.mean_score:.3f}"
            f" packet={top_row.packet_type} gaps={top_row.packet_gaps}"
            f" occ={top_row.occupancy_probability:.6f}"
            f" action_loss={action_loss:.6f}"
            f" weighted_loss={weighted_loss:.6f}"
        )
        print(f"    packet statuses: {_format_packet_statuses(one_row)}")
        print(f"    top_prefix intervals: {_format_interval_distribution_tuple(top_row.interval_distribution)}")
        print(f"    one_run intervals:    {_format_interval_distribution_tuple(one_row.interval_distribution)}")
        print(
            f"    top_prefix top/bottom: p10={top_row.p_top1_bottom0:.3f}"
            f" p01={top_row.p_top0_bottom1:.3f}"
            f" same={top_row.p_top_equals_bottom:.3f}"
            f" cov={top_row.top_bottom_covariance:.3f}"
        )
        print(
            f"    one_run top/bottom:    p10={one_row.p_top1_bottom0:.3f}"
            f" p01={one_row.p_top0_bottom1:.3f}"
            f" same={one_row.p_top_equals_bottom:.3f}"
            f" cov={one_row.top_bottom_covariance:.3f}"
        )
    print()

    print("Rule hints")
    label_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.library_name != "one_run":
            continue
        label_totals[_classify_one_run_correction(row.interval_distribution)] += row.occupancy_probability
    for label, mass in sorted(label_totals.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {label:20s} occupancy={mass:.6f}")


def print_chase5_verification_report(
    T_values: tuple[int, ...],
    n: int = 20,
) -> None:
    print("Chase5 verification report")
    print()
    print(f"T values: {T_values}")
    print()
    for T in T_values:
        summary = chase5_verification_summary(T, n=n)
        print(f"T={T}")
        print(f"  F_T(0): {summary.value:.6f}")
        print(f"  F_T(0)/sqrt(T): {summary.normalized_value:.6f}")
        print(f"  max Bellman residual: {summary.max_residual:.9f}")
        print(f"  weighted positive residual: {summary.weighted_positive_residual:.9f}")
        print(f"  reachable states: {summary.reachable_state_count}")
        print("  top residual rows:")
        for row in summary.top_rows:
            support = ", ".join(f"{action}:{residual:.6f}" for action, residual in row.chase_support_residuals)
            print(
                f"    h={row.remaining_horizon:2d}"
                f" state={row.state}"
                f" action={_format_action(row.action)}"
                f" residual={row.residual:.9f}"
                f" occ={row.occupancy_probability:.6f}"
            )
            print(f"      p={_format_float_tuple(row.winner_probabilities)}")
            print(f"      Chase support residuals=[{support}]")
        print()


def _format_support_residuals(residuals: tuple[tuple[str, float], ...]) -> str:
    return ", ".join(f"{action}:{residual:.6f}" for action, residual in residuals)


def print_chase5_potential_certificate_report(
    T_values: tuple[int, ...],
    n: int = 20,
) -> None:
    print("Chase5 potential certificate report")
    print()
    print(f"T values: {T_values}")
    print()
    for T in T_values:
        summary = chase5_potential_certificate_summary(T, n=n)
        print(f"T={T}")
        print(f"  F_T(0): {summary.value:.6f}")
        print(f"  F_T(0)/sqrt(T): {summary.normalized_value:.6f}")
        print(f"  max PM residual: {summary.max_pm_residual:.9f}")
        print(f"  max repaired alpha: {summary.max_repaired_alpha:.9f}")
        print(f"  weighted positive PM residual: {summary.weighted_positive_pm_residual:.9f}")
        print(f"  weighted positive repaired alpha: {summary.weighted_positive_repaired_alpha:.9f}")
        print("  PM fails but repaired succeeds:")
        for row in summary.pm_fail_repaired_success_rows:
            print(
                f"    h={row.remaining_horizon:2d}"
                f" state={row.state}"
                f" occ={row.occupancy_probability:.6f}"
                f" PM_action={_format_action(row.pm_max_action)}"
                f" PM_residual={row.pm_max_residual:.9f}"
                f" repaired_alpha={row.repaired_alpha:.9f}"
            )
            print(f"      PM p={_format_float_tuple(row.pm_learner)}")
            print(f"      repaired p={_format_float_tuple(row.repaired_learner)}")
            print(f"      PM Chase support=[{_format_support_residuals(row.pm_chase_support_residuals)}]")
            print(f"      repaired Chase support=[{_format_support_residuals(row.repaired_chase_support_residuals)}]")
        print("  Repaired failures:")
        if not summary.repaired_failure_rows:
            print("    none")
        for row in summary.repaired_failure_rows:
            print(
                f"    h={row.remaining_horizon:2d}"
                f" state={row.state}"
                f" occ={row.occupancy_probability:.6f}"
                f" action={_format_action(row.repaired_max_action)}"
                f" alpha={row.repaired_alpha:.9f}"
                f" repaired p={_format_float_tuple(row.repaired_learner)}"
            )
        print()


def _k6_patch_fire_counts(
    T: int,
    policy_fn,
    patches: tuple[K6OneRunPatch, ...],
) -> tuple[tuple[str, int, float], ...]:
    occupancy = _time_policy_occupancy(6, T, policy_fn)
    counts: dict[str, int] = defaultdict(int)
    masses: dict[str, float] = defaultdict(float)
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            patch = _matching_k6_patch(state, remaining_horizon, patches)
            if patch is None:
                continue
            counts[patch.name] += 1
            masses[patch.name] += probability
    return tuple((name, counts[name], masses[name]) for name in counts)


def k6_one_run_patch_ladder_benchmark_rows(
    T: int,
) -> tuple[K6PatchLadderBenchmarkRow, ...]:
    zero = (0, 0, 0, 0, 0, 0)
    optimal_value = optimal_values(6, T)[T][zero]
    one_run_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "one_run"))[T][zero]
    top_prefix_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "top_prefix_all"))[T][zero]
    rows: list[K6PatchLadderBenchmarkRow] = []
    for policy_name, patch_names in _k6_patch_ladder_policy_specs():
        policy_fn = make_k6_one_run_patch_policy(
            "k6_alive_full_comb_control_policy",
            patch_names,
        )
        value = evaluate_time_dependent_policy(6, T, policy_fn)[T][zero]
        failures = k6_one_run_candidate_failure_rows(T, (policy_name,), support_tolerance=1e-8)
        failure_summary: dict[str, float] = defaultdict(float)
        for failure in failures:
            key = (
                f"rem={failure.remaining_horizon}"
                f" packet={failure.packet_type}"
                f" gaps={failure.packet_gaps}"
                f" lp={_format_interval(failure.lp_dominant_interval)}"
            )
            failure_summary[key] += failure.weighted_action_loss
        patches = tuple(_k6_one_run_patch_by_name()[name] for name in patch_names)
        rows.append(
            K6PatchLadderBenchmarkRow(
                policy_name=policy_name,
                patch_names=patch_names,
                value=value,
                one_run_value=one_run_value,
                gap_to_one_run=one_run_value - value,
                normalized_gap_to_one_run=(one_run_value - value) / (T ** 0.5),
                optimal_value=optimal_value,
                gap_to_optimal=optimal_value - value,
                normalized_gap_to_optimal=(optimal_value - value) / (T ** 0.5),
                top_prefix_value=top_prefix_value,
                gap_to_top_prefix=value - top_prefix_value,
                patch_fire_counts=tuple(sorted(_k6_patch_fire_counts(T, policy_fn, patches))),
                top_failure_summaries=tuple(
                    sorted(failure_summary.items(), key=lambda item: (-item[1], item[0]))[:10]
                ),
            )
        )
    return tuple(rows)


def print_k6_one_run_patch_ladder_benchmark(T: int) -> None:
    rows = k6_one_run_patch_ladder_benchmark_rows(T)
    if not rows:
        print("No k=6 patch ladder rows")
        return

    first = rows[0]
    print(f"k=6 one-run patch ladder benchmark, T={T}")
    print()
    print(f"V_star: {first.optimal_value:.6f}")
    print(f"one_run: {first.one_run_value:.6f}")
    print(f"top_prefix_all: {first.top_prefix_value:.6f}")
    print()
    print("Patch order:")
    for index, patch in enumerate(_k6_one_run_patch_list(), start=1):
        print(
            f"  {index}. {patch.name}:"
            f" rem={patch.remaining_horizon}"
            f" packet={patch.packet_type}"
            f" gaps={patch.packet_gaps}"
            f" interval={_format_interval(patch.interval)}"
        )
    print()
    print(
        "policy                    V_policy gap_to_one_run gap_to_one_run/sqrt(T)"
        " gap_to_Vstar gap_to_Vstar/sqrt(T) gap_vs_top_prefix"
    )
    for row in rows:
        print(
            f"{row.policy_name:25s}"
            f" {row.value:10.6f}"
            f" {row.gap_to_one_run:14.6f}"
            f" {row.normalized_gap_to_one_run:24.6f}"
            f" {row.gap_to_optimal:12.6f}"
            f" {row.normalized_gap_to_optimal:21.6f}"
            f" {row.gap_to_top_prefix:17.6f}"
        )
        if row.patch_fire_counts:
            print("  patch fired:")
            for name, count, mass in row.patch_fire_counts:
                print(f"    {name}: states={count} occupancy={mass:.6f}")
        else:
            print("  patch fired: none")
        print("  top failure summaries:")
        for label, loss in row.top_failure_summaries[:5]:
            print(f"    {label}: weighted_loss={loss:.6f}")
    print()


def print_k6_one_run_interval_rule_report(
    T: int,
    support_tolerance: float = 1e-8,
    n: int = 100,
) -> None:
    rows = k6_one_run_interval_rule_rows(T, support_tolerance=support_tolerance)
    zero = (0, 0, 0, 0, 0, 0)
    optimal = optimal_values(6, T)[T][zero]
    one_run_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "one_run"))[T][zero]
    top_prefix_value = _library_lp_value_layers(6, T, _strategy_class_library_actions(6, T, "top_prefix_all"))[T][zero]
    trunc_values = evaluate_time_dependent_policy(6, T, k6_alive_trunc_comb_policy)
    full_comb_values = evaluate_time_dependent_policy(6, T, k6_alive_full_comb_control_policy)
    sqrt_T = T ** 0.5

    print(f"k=6 one-run interval-rule report, T={T}")
    print()
    print(f"support tolerance: {support_tolerance:.3g}")
    print(f"rows: {len(rows)}")
    print()
    print("Value summary:")
    for name, value in (
        ("V_star", optimal),
        ("one_run", one_run_value),
        ("top_prefix_all", top_prefix_value),
        ("k6_alive_trunc_comb_policy", trunc_values[T][zero]),
        ("k6_alive_full_comb_control_policy", full_comb_values[T][zero]),
    ):
        gap = optimal - value
        print(f"  {name:34s} V={value:.6f} gap={gap:.6f} gap/sqrt(T)={gap / sqrt_T:.6f}")
    print()

    print("Overall LP support mass by interval relation:")
    dual_totals: dict[str, float] = defaultdict(float)
    occupancy_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for support in row.support:
            dual_totals[support.relation_class] += support.weight
            occupancy_totals[support.relation_class] += row.occupancy_probability * support.weight
    for relation in sorted(dual_totals, key=lambda key: (-occupancy_totals[key], key)):
        print(f"  {relation:28s} dual_weight={dual_totals[relation]:.6f} occupancy_weight={occupancy_totals[relation]:.6f}")
    print()

    print("Top interval distributions by regime:")
    regime_groups = _group_alive_rows(rows, lambda row: (row.remaining_horizon, row.packet_type, row.packet_gaps))
    for (remaining_horizon, ptype, gaps), group in regime_groups[:n]:
        interval_totals: dict[tuple[int, int] | None, float] = defaultdict(float)
        relation_totals: dict[str, float] = defaultdict(float)
        for row in group:
            for support in row.support:
                interval_totals[support.interval] += row.occupancy_probability * support.weight
                relation_totals[support.relation_class] += row.occupancy_probability * support.weight
        representative = max(group, key=lambda row: row.occupancy_probability)
        rendered_intervals = ", ".join(
            f"{_format_interval(interval)}:{weight:.6f}"
            for interval, weight in sorted(interval_totals.items(), key=lambda item: (-item[1], str(item[0])))[:5]
        )
        rendered_relations = ", ".join(
            f"{relation}:{weight:.6f}"
            for relation, weight in sorted(relation_totals.items(), key=lambda item: (-item[1], item[0]))[:5]
        )
        print(
            f"  rem={remaining_horizon:2d} packet={ptype} gaps={gaps}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
            f" intervals=[{rendered_intervals}]"
            f" relations=[{rendered_relations}]"
            f" rep={representative.state}"
        )
    print()

    print("Top interval distributions by alive_count:")
    for alive_count, group in _group_alive_rows(rows, lambda row: row.alive_count):
        interval_totals = defaultdict(float)
        relation_totals = defaultdict(float)
        for row in group:
            for support in row.support:
                interval_totals[support.interval] += row.occupancy_probability * support.weight
                relation_totals[support.relation_class] += row.occupancy_probability * support.weight
        print(
            f"  alive_count={alive_count}"
            f" rows={len(group)}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
        )
        print(
            "    intervals="
            + ", ".join(
                f"{_format_interval(interval)}:{weight:.6f}"
                for interval, weight in sorted(interval_totals.items(), key=lambda item: (-item[1], str(item[0])))[:8]
            )
        )
        print(
            "    relations="
            + ", ".join(
                f"{relation}:{weight:.6f}"
                for relation, weight in sorted(relation_totals.items(), key=lambda item: (-item[1], item[0]))[:8]
            )
        )
    print()

    print("Top candidate failure states:")
    failure_rows = k6_one_run_candidate_failure_rows(T, support_tolerance=support_tolerance)
    for row in sorted(failure_rows, key=lambda item: (-item.weighted_action_loss, item.policy_name, item.time, item.state))[:n]:
        intervals = ", ".join(
            f"{_format_interval(interval)}:{_format_fractionish(weight)}"
            for interval, weight in row.lp_interval_distribution[:5]
        )
        print(
            f"  policy={row.policy_name}"
            f" t={row.time:2d} rem={row.remaining_horizon:2d}"
            f" state={row.state} packet={row.packet_type} gaps={row.packet_gaps}"
            f" occ={row.occupancy_probability:.6f}"
            f" loss={row.action_loss:.6f}"
            f" weighted={row.weighted_action_loss:.6f}"
            f" candidate={row.candidate_patterns}"
            f" lp_intervals=[{intervals}]"
            f" dominant_relation={row.lp_dominant_relation}"
        )
    print()

    print("Candidate failure summary by policy/regime:")
    grouped_failures: dict[tuple[str, int, tuple[int, ...], tuple[int, ...], tuple[int, int] | None], float] = defaultdict(float)
    for row in failure_rows:
        grouped_failures[
            (row.policy_name, row.remaining_horizon, row.packet_type, row.packet_gaps, row.lp_dominant_interval)
        ] += row.weighted_action_loss
    for (policy_name, remaining_horizon, ptype, gaps, interval), weight in sorted(
        grouped_failures.items(),
        key=lambda item: (-item[1], str(item[0])),
    )[:n]:
        print(
            f"  policy={policy_name}"
            f" rem={remaining_horizon:2d}"
            f" packet={ptype} gaps={gaps}"
            f" dominant_interval={_format_interval(interval)}"
            f" weighted_loss={weight:.6f}"
        )
    print()

    print("Rule-hint summary:")
    hinted = 0
    for (remaining_horizon, ptype, gaps), group in regime_groups:
        interval_totals = defaultdict(float)
        total_support = 0.0
        for row in group:
            for support in row.support:
                weight = row.occupancy_probability * support.weight
                interval_totals[support.interval] += weight
                total_support += weight
        if total_support <= 0:
            continue
        interval, weight = max(interval_totals.items(), key=lambda item: item[1])
        if weight / total_support >= 0.8:
            hinted += 1
            representative = max(group, key=lambda row: row.occupancy_probability)
            actions = ", ".join(
                _format_action(support.action)
                for support in representative.support
                if support.interval == interval
            )
            print(
                f"  IF rem={remaining_horizon} packet={ptype} gaps={gaps}"
                f" THEN interval={_format_interval(interval)}"
                f" share={weight / total_support:.3f}"
                f" actions=[{actions}]"
            )
        if hinted >= n:
            break
    if hinted == 0:
        print("  no >=80% interval-dominant regime rules found")


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


def evaluate_time_dependent_policy_reachable_root(
    k: int,
    T: int,
    policy_fn,
) -> ReachablePolicyEvaluationResult:
    if T < 0:
        raise ValueError("T must be nonnegative")

    visited_by_horizon: dict[int, int] = defaultdict(int)
    max_support_size = 0

    @lru_cache(maxsize=None)
    def value(state: tuple[int, ...], remaining_horizon: int) -> float:
        nonlocal max_support_size
        canonical_state = canon(state)
        visited_by_horizon[remaining_horizon] += 1
        if remaining_horizon == 0:
            return float(max(canonical_state, default=0))
        policy = _policy_from_weighted_actions(tuple(policy_fn(canonical_state, remaining_horizon)))
        max_support_size = max(max_support_size, len(policy))
        expected_action = np.zeros(k)
        expected_continuation = 0.0
        for probability, action in policy:
            if len(action) != k:
                raise ValueError(f"policy action length {len(action)} does not match k={k}")
            expected_action += probability * np.asarray(action, dtype=float)
            raw_next = tuple(canonical_state[index] + action[index] for index in range(k))
            next_state = canon(raw_next)
            expected_continuation += probability * (
                float(min(raw_next, default=0)) + value(next_state, remaining_horizon - 1)
            )
        return expected_continuation - float(expected_action.max(initial=0.0))

    root = tuple(0 for _ in range(k))
    root_value = value(root, T)
    cache_info = value.cache_info()
    return ReachablePolicyEvaluationResult(
        value=root_value,
        visited_state_count=cache_info.currsize,
        visited_by_horizon=tuple(sorted(visited_by_horizon.items(), key=lambda item: item[0], reverse=True)),
        cache_hits=cache_info.hits,
        max_support_size=max_support_size,
    )


def _chase5_value_layers_reachable(T: int):
    if T < 0:
        raise ValueError("T must be nonnegative")
    support = _chase5_policy((0, 0, 0, 0, 0), T)

    @lru_cache(maxsize=None)
    def value(state: tuple[int, ...], remaining_horizon: int) -> float:
        canonical_state = canon(state)
        if remaining_horizon == 0:
            return float(max(canonical_state, default=0))
        expected = 0.0
        for probability, action in support:
            raw_next = tuple(canonical_state[index] + action[index] for index in range(5))
            expected += probability * (
                float(min(raw_next, default=0)) + value(canon(raw_next), remaining_horizon - 1)
            )
        return expected - 0.5

    return value


def _chase5_occupancy(T: int) -> list[dict[tuple[int, ...], float]]:
    support = _chase5_policy((0, 0, 0, 0, 0), T)
    occupancy: list[defaultdict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][(0, 0, 0, 0, 0)] = 1.0
    for time in range(T):
        for state, probability in tuple(occupancy[time].items()):
            for action_probability, action in support:
                raw_next = tuple(state[index] + action[index] for index in range(5))
                occupancy[time + 1][canon(raw_next)] += probability * action_probability
    return [dict(layer) for layer in occupancy]


def _chase5_named_winner_probabilities(
    named_state: tuple[int, ...],
    remaining_horizon: int,
    memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]],
) -> tuple[float, ...]:
    key = (named_state, remaining_horizon)
    if key in memo:
        return memo[key]
    if remaining_horizon == 0:
        result = _terminal_named_winner_probabilities(named_state)
        memo[key] = result
        return result
    probabilities = [0.0] * len(named_state)
    for canonical_probability, canonical_action in _chase5_policy(canon(named_state), remaining_horizon):
        for lift_probability, named_action in _lift_canonical_action_to_named_distribution(named_state, canonical_action):
            next_state = tuple(named_state[index] + named_action[index] for index in range(len(named_state)))
            next_probabilities = _chase5_named_winner_probabilities(next_state, remaining_horizon - 1, memo)
            for index, probability in enumerate(next_probabilities):
                probabilities[index] += canonical_probability * lift_probability * probability
    result = tuple(probabilities)
    memo[key] = result
    return result


def chase5_verification_summary(
    T: int,
    n: int = 20,
) -> Chase5VerificationSummary:
    value = _chase5_value_layers_reachable(T)
    occupancy = _chase5_occupancy(T)
    winner_memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]] = {}
    all_gain_actions = all_actions(5)
    chase_support = tuple(action for _, action in _chase5_policy((0, 0, 0, 0, 0), T))
    max_residual = float("-inf")
    weighted_positive_residual = 0.0
    rows: list[Chase5VerificationRow] = []
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            potential = value(state, remaining_horizon)
            winner_probabilities = _chase5_named_winner_probabilities(state, remaining_horizon, winner_memo)
            chase_support_residuals = tuple(
                (
                    _format_action(action),
                    float(min(tuple(state[index] + action[index] for index in range(5)), default=0))
                    + value(canon(tuple(state[index] + action[index] for index in range(5))), remaining_horizon - 1)
                    - sum(winner_probabilities[index] * action[index] for index in range(5))
                    - potential,
                )
                for action in chase_support
            )
            for action in all_gain_actions:
                raw_next = tuple(state[index] + action[index] for index in range(5))
                residual = (
                    float(min(raw_next, default=0))
                    + value(canon(raw_next), remaining_horizon - 1)
                    - sum(winner_probabilities[index] * action[index] for index in range(5))
                    - potential
                )
                max_residual = max(max_residual, residual)
                if residual > 0:
                    weighted_positive_residual += probability * residual
                rows.append(
                    Chase5VerificationRow(
                        T=T,
                        remaining_horizon=remaining_horizon,
                        state=state,
                        action=action,
                        occupancy_probability=probability,
                        residual=residual,
                        winner_probabilities=winner_probabilities,
                        chase_support_residuals=chase_support_residuals,
                    )
                )
    top_rows = tuple(sorted(rows, key=lambda row: (-row.residual, row.remaining_horizon, row.state, row.action))[:n])
    root_value = value((0, 0, 0, 0, 0), T)
    reachable_state_count = len({state for layer in occupancy for state in layer})
    return Chase5VerificationSummary(
        T=T,
        value=root_value,
        normalized_value=root_value / (T ** 0.5 if T > 0 else 1.0),
        max_residual=max_residual if max_residual != float("-inf") else 0.0,
        weighted_positive_residual=weighted_positive_residual,
        reachable_state_count=reachable_state_count,
        top_rows=top_rows,
    )


def _chase5_residual_for_action(
    state: tuple[int, ...],
    remaining_horizon: int,
    action: tuple[int, ...],
    learner: tuple[float, ...],
    value_fn,
) -> float:
    potential = value_fn(state, remaining_horizon)
    raw_next = tuple(state[index] + action[index] for index in range(5))
    return (
        float(min(raw_next, default=0))
        + value_fn(canon(raw_next), remaining_horizon - 1)
        - sum(learner[index] * action[index] for index in range(5))
        - potential
    )


def _chase5_certificate_learner(
    state: tuple[int, ...],
    remaining_horizon: int,
    value_fn,
) -> tuple[tuple[float, ...], float, tuple[int, ...]]:
    actions = all_actions(5)
    potential = value_fn(state, remaining_horizon)
    q_values = []
    for action in actions:
        raw_next = tuple(state[index] + action[index] for index in range(5))
        q_values.append(
            float(min(raw_next, default=0))
            + value_fn(canon(raw_next), remaining_horizon - 1)
            - potential
        )
    # Variables are p_0..p_4, alpha. Constraints q_g - p.g <= alpha.
    c = [0.0] * 5 + [1.0]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for action, q_value in zip(actions, q_values, strict=True):
        a_ub.append([-float(bit) for bit in action] + [-1.0])
        b_ub.append(-float(q_value))
    a_eq = [[1.0] * 5 + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * 5 + [(None, None)]
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"certificate LP failed at state {state}, h={remaining_horizon}: {result.message}")
    learner = tuple(float(value) for value in result.x[:5])
    residuals = tuple(
        _chase5_residual_for_action(state, remaining_horizon, action, learner, value_fn)
        for action in actions
    )
    max_index = max(range(len(actions)), key=lambda index: residuals[index])
    return learner, float(residuals[max_index]), actions[max_index]


def chase5_potential_certificate_summary(
    T: int,
    n: int = 20,
) -> Chase5CertificateSummary:
    value_fn = _chase5_value_layers_reachable(T)
    occupancy = _chase5_occupancy(T)
    winner_memo: dict[tuple[tuple[int, ...], int], tuple[float, ...]] = {}
    chase_support = tuple(action for _, action in _chase5_policy((0, 0, 0, 0, 0), T))
    all_gain_actions = all_actions(5)
    rows: list[Chase5CertificateRow] = []
    max_pm = float("-inf")
    max_repaired = float("-inf")
    weighted_pm = 0.0
    weighted_repaired = 0.0
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in occupancy[time].items():
            pm_learner = _chase5_named_winner_probabilities(state, remaining_horizon, winner_memo)
            pm_residuals = tuple(
                _chase5_residual_for_action(state, remaining_horizon, action, pm_learner, value_fn)
                for action in all_gain_actions
            )
            pm_max_index = max(range(len(all_gain_actions)), key=lambda index: pm_residuals[index])
            repaired_learner, repaired_alpha, repaired_action = _chase5_certificate_learner(
                state,
                remaining_horizon,
                value_fn,
            )
            max_pm = max(max_pm, pm_residuals[pm_max_index])
            max_repaired = max(max_repaired, repaired_alpha)
            weighted_pm += probability * max(0.0, pm_residuals[pm_max_index])
            weighted_repaired += probability * max(0.0, repaired_alpha)
            rows.append(
                Chase5CertificateRow(
                    T=T,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    occupancy_probability=probability,
                    potential_value=value_fn(state, remaining_horizon),
                    pm_learner=pm_learner,
                    pm_max_residual=pm_residuals[pm_max_index],
                    pm_max_action=all_gain_actions[pm_max_index],
                    repaired_learner=repaired_learner,
                    repaired_alpha=repaired_alpha,
                    repaired_max_action=repaired_action,
                    pm_chase_support_residuals=tuple(
                        (
                            _format_action(action),
                            _chase5_residual_for_action(state, remaining_horizon, action, pm_learner, value_fn),
                        )
                        for action in chase_support
                    ),
                    repaired_chase_support_residuals=tuple(
                        (
                            _format_action(action),
                            _chase5_residual_for_action(state, remaining_horizon, action, repaired_learner, value_fn),
                        )
                        for action in chase_support
                    ),
                )
            )
    pm_fail_repaired_success = tuple(
        sorted(
            (
                row
                for row in rows
                if row.pm_max_residual > 1e-9 and row.repaired_alpha <= 1e-8
            ),
            key=lambda row: (-row.pm_max_residual, row.remaining_horizon, row.state),
        )[:n]
    )
    repaired_failures = tuple(
        sorted(
            (row for row in rows if row.repaired_alpha > 1e-8),
            key=lambda row: (-row.repaired_alpha, row.remaining_horizon, row.state),
        )[:n]
    )
    root_value = value_fn((0, 0, 0, 0, 0), T)
    return Chase5CertificateSummary(
        T=T,
        value=root_value,
        normalized_value=root_value / (T ** 0.5 if T > 0 else 1.0),
        max_pm_residual=max_pm if max_pm != float("-inf") else 0.0,
        max_repaired_alpha=max_repaired if max_repaired != float("-inf") else 0.0,
        weighted_positive_pm_residual=weighted_pm,
        weighted_positive_repaired_alpha=weighted_repaired,
        pm_fail_repaired_success_rows=pm_fail_repaired_success,
        repaired_failure_rows=repaired_failures,
    )


def _time_policy_step_value(
    k: int,
    state: tuple[int, ...],
    policy: tuple[tuple[float, tuple[int, ...]], ...],
    continuation_values: dict[tuple[int, ...], float],
) -> float:
    expected_value = 0.0
    expected_action_vector = np.zeros(k)
    for probability, action in policy:
        expected_action_vector += probability * np.asarray(action, dtype=float)
        expected_value += probability * _next_state_value(state, action, continuation_values)
    return expected_value - float(expected_action_vector.max(initial=0.0))


def _time_policy_occupancy(k: int, T: int, policy_fn) -> list[dict[tuple[int, ...], float]]:
    zero = tuple(0 for _ in range(k))
    occupancy: list[dict[tuple[int, ...], float]] = [defaultdict(float) for _ in range(T + 1)]
    occupancy[0][zero] = 1.0
    for time in range(T):
        remaining_horizon = T - time
        for state, probability in tuple(occupancy[time].items()):
            for action_probability, action in policy_fn(state, remaining_horizon):
                next_state = canon(tuple(state[index] + action[index] for index in range(k)))
                occupancy[time + 1][next_state] += probability * action_probability
    return [dict(layer) for layer in occupancy]


def _time_dependent_policy_registry() -> dict[str, object]:
    policies: dict[str, object] = {
        "top_prefix_all": _top_prefix_all_reference_policy,
        "two_run_dual_support_replay_k9_T7": two_run_dual_support_replay_k9_T7_policy,
        "two_run_skeleton_v3a_late_policy": two_run_skeleton_v3a_late_policy,
        "two_run_skeleton_v3b_early_late_policy": two_run_skeleton_v3b_early_late_policy,
        "k6_alive_trunc_comb_policy": k6_alive_trunc_comb_policy,
        "k6_alive_trunc_comb_twin3_policy": k6_alive_trunc_comb_twin3_policy,
        "k6_alive_full_comb_control_policy": k6_alive_full_comb_control_policy,
        "k6_alive_trunc_comb_strict_policy": k6_alive_trunc_comb_strict_policy,
        "k6_alive_trunc_comb_twin3_strict_policy": k6_alive_trunc_comb_twin3_strict_policy,
        "k6_alive_full_comb_control_strict_policy": k6_alive_full_comb_control_strict_policy,
        "k6_one_run_rule_v1_policy": k6_one_run_rule_v1_policy,
        "k6_one_run_rule_v1_no_mixed_policy": k6_one_run_rule_v1_no_mixed_policy,
        "k6_one_run_rule_v1_aggressive_shoulder_policy": k6_one_run_rule_v1_aggressive_shoulder_policy,
        "k6_greedy_deterministic_patched_policy": k6_greedy_deterministic_patched_policy,
        "k6_barycentric_v1_equal": k6_barycentric_v1_equal_policy,
        "k6_barycentric_v1_late": k6_barycentric_v1_late_policy,
        "k6_barycentric_v1_middle": k6_barycentric_v1_middle_policy,
        "k6_barycentric_v1_tail_heavy": k6_barycentric_v1_tail_heavy_policy,
        "k6_barycentric_v1_top_tail": k6_barycentric_v1_top_tail_policy,
        "fixed_101010_comb": fixed_101010_comb_policy,
        "fixed_101011_truncated": fixed_101011_truncated_policy,
        "fixed_101000_prefix3": fixed_101000_prefix3_policy,
        "fixed_100000_singleton": fixed_100000_singleton_policy,
        "fixed_10101_comb": fixed_10101_comb_policy,
        "chase5": _chase5_policy,
        "potential_greedy_c0.25": _potential_greedy_policy_from_c(0.25),
        "potential_greedy_c0.5": _potential_greedy_policy_from_c(0.5),
        "potential_greedy_c1.0": _potential_greedy_policy_from_c(1.0),
        "potential_greedy_c2.0": _potential_greedy_policy_from_c(2.0),
    }
    for policy_name, patch_names in _k6_patch_ladder_policy_specs():
        policies[policy_name] = make_k6_one_run_patch_policy(
            "k6_alive_full_comb_control_policy",
            patch_names,
        )
    for remaining in range(1, 8):
        policies[f"replay_rem_ge_{remaining}_then_v3a"] = make_two_run_replay_cutoff_policy(
            replay_min_remaining=remaining,
            fallback_name="two_run_skeleton_v3a_late_policy",
        )
        policies[f"replay_rem_ge_{remaining}_then_v6"] = make_two_run_replay_cutoff_policy(
            replay_min_remaining=remaining,
            fallback_name="top_prefix_three_regime_v6",
        )
        policies[f"replay_rem_le_{remaining}_then_v3a"] = make_two_run_replay_cutoff_policy(
            replay_max_remaining=remaining,
            fallback_name="two_run_skeleton_v3a_late_policy",
        )
        policies[f"replay_rem_le_{remaining}_then_v6"] = make_two_run_replay_cutoff_policy(
            replay_max_remaining=remaining,
            fallback_name="top_prefix_three_regime_v6",
        )
    return policies


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


def _format_replay_horizons(replay_horizons: frozenset[int]) -> str:
    if not replay_horizons:
        return "{}"
    return "{" + ",".join(str(horizon) for horizon in sorted(replay_horizons)) + "}"


def _replay_layer_masks(T: int, mode: str) -> tuple[frozenset[int], ...]:
    horizons = tuple(range(1, T + 1))
    if mode == "all":
        masks: list[frozenset[int]] = []
        for size in range(T + 1):
            masks.extend(frozenset(mask) for mask in combinations(horizons, size))
        return tuple(masks)
    if mode == "singletons":
        return tuple(frozenset((horizon,)) for horizon in horizons)
    if mode == "pairs":
        return tuple(frozenset(mask) for mask in combinations(horizons, 2))
    if mode == "prefixes":
        return tuple(frozenset(range(minimum, T + 1)) for minimum in horizons)
    if mode == "suffixes":
        return tuple(frozenset(range(1, maximum + 1)) for maximum in horizons)
    raise ValueError(f"unknown replay mask mode: {mode}")


def replay_layer_mask_sweep_rows(
    k: int,
    T: int,
    fallback_policy_name: str,
    reference_policy_name: str = "two_run_dual_support_replay_k9_T7",
    mode: str = "all",
) -> tuple[ReplayLayerMaskSweepRow, ...]:
    if k != 9 or T != 7:
        raise ValueError("replay-layer mask sweep is currently defined only for k=9,T=7")

    reference_policy = _resolve_time_dependent_policy(k, reference_policy_name)
    if reference_policy is None:
        raise ValueError(f"unknown reference policy: {reference_policy_name}")
    if _resolve_time_dependent_policy(k, fallback_policy_name) is None:
        raise ValueError(f"unknown fallback policy: {fallback_policy_name}")

    zero = tuple(0 for _ in range(k))
    reference_values = evaluate_time_dependent_policy(k, T, reference_policy)
    reference_value = reference_values[T][zero]
    rows: list[ReplayLayerMaskSweepRow] = []
    for replay_horizons in _replay_layer_masks(T, mode):
        policy_fn = make_two_run_replay_mask_policy(replay_horizons, fallback_policy_name)
        values = evaluate_time_dependent_policy(k, T, policy_fn)
        value = values[T][zero]
        gap = reference_value - value
        normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
        rows.append(
            ReplayLayerMaskSweepRow(
                replay_horizons=replay_horizons,
                replay_count=len(replay_horizons),
                fallback_policy_name=fallback_policy_name,
                value=value,
                reference_value=reference_value,
                gap_to_reference=gap,
                normalized_gap_to_reference=normalized_gap,
                exact=abs(gap) <= 1e-9,
            )
        )
    return tuple(rows)


def _print_replay_mask_row(row: ReplayLayerMaskSweepRow) -> None:
    print(
        f"  mask={_format_replay_horizons(row.replay_horizons):18s}"
        f" count={row.replay_count}"
        f" V={row.value:.6f}"
        f" V_ref={row.reference_value:.6f}"
        f" gap={row.gap_to_reference:.6f}"
        f" gap/sqrt(T)={row.normalized_gap_to_reference:.6f}"
        f" exact={row.exact}"
        f" fallback={row.fallback_policy_name}"
    )


def _best_replay_mask_by_count(
    rows: tuple[ReplayLayerMaskSweepRow, ...],
) -> list[ReplayLayerMaskSweepRow]:
    by_count: dict[int, list[ReplayLayerMaskSweepRow]] = defaultdict(list)
    for row in rows:
        by_count[row.replay_count].append(row)
    best: list[ReplayLayerMaskSweepRow] = []
    for replay_count in sorted(by_count):
        best.append(
            min(
                by_count[replay_count],
                key=lambda item: (item.gap_to_reference, sorted(item.replay_horizons)),
            )
        )
    return best


def _row_by_mask(
    rows: tuple[ReplayLayerMaskSweepRow, ...],
) -> dict[frozenset[int], ReplayLayerMaskSweepRow]:
    return {row.replay_horizons: row for row in rows}


def _print_replay_mask_full_mode_diagnostics(
    rows: tuple[ReplayLayerMaskSweepRow, ...],
    T: int,
) -> None:
    rows_by_mask = _row_by_mask(rows)
    empty = rows_by_mask[frozenset()]
    full = rows_by_mask[frozenset(range(1, T + 1))]

    print("Best mask by replay_count:")
    for row in _best_replay_mask_by_count(rows):
        _print_replay_mask_row(row)
    print()

    print("Single-layer gains relative to no replay:")
    for horizon in range(1, T + 1):
        row = rows_by_mask[frozenset((horizon,))]
        print(
            f"  rem={horizon}"
            f" gain={empty.gap_to_reference - row.gap_to_reference:.6f}"
            f" mask_gap={row.gap_to_reference:.6f}"
            f" mask_gap/sqrt(T)={row.normalized_gap_to_reference:.6f}"
        )
    print()

    print("Greedy marginal gain ladder:")
    current = frozenset()
    current_row = empty
    remaining = set(range(1, T + 1))
    print(
        f"  start mask={_format_replay_horizons(current)}"
        f" gap={current_row.gap_to_reference:.6f}"
        f" gap/sqrt(T)={current_row.normalized_gap_to_reference:.6f}"
    )
    while remaining:
        candidates = []
        for horizon in sorted(remaining):
            candidate_mask = frozenset(set(current) | {horizon})
            candidate_row = rows_by_mask[candidate_mask]
            candidates.append((current_row.gap_to_reference - candidate_row.gap_to_reference, horizon, candidate_row))
        gain, horizon, best_row = max(candidates, key=lambda item: (item[0], -item[1]))
        current = best_row.replay_horizons
        current_row = best_row
        remaining.remove(horizon)
        print(
            f"  add rem={horizon}"
            f" gain={gain:.6f}"
            f" mask={_format_replay_horizons(current):18s}"
            f" gap={current_row.gap_to_reference:.6f}"
            f" gap/sqrt(T)={current_row.normalized_gap_to_reference:.6f}"
        )
    print()

    print("Reverse ablation from full replay:")
    for horizon in range(1, T + 1):
        mask = frozenset(set(range(1, T + 1)) - {horizon})
        row = rows_by_mask[mask]
        print(
            f"  remove rem={horizon}"
            f" damage={row.gap_to_reference - full.gap_to_reference:.6f}"
            f" mask={_format_replay_horizons(mask):18s}"
            f" gap={row.gap_to_reference:.6f}"
            f" gap/sqrt(T)={row.normalized_gap_to_reference:.6f}"
        )
    print()


def print_replay_layer_mask_sweep(
    k: int,
    T: int,
    fallback_policy_name: str,
    reference_policy_name: str = "two_run_dual_support_replay_k9_T7",
    mode: str = "all",
    n: int = 80,
) -> None:
    rows = replay_layer_mask_sweep_rows(
        k,
        T,
        fallback_policy_name,
        reference_policy_name=reference_policy_name,
        mode=mode,
    )
    print(f"Replay-layer mask sweep, k={k}, T={T}")
    print()
    print(f"fallback policy: {fallback_policy_name}")
    print(f"reference policy: {reference_policy_name}")
    print(f"mode: {mode}")
    print(f"rows: {len(rows)}")
    print()

    print("Overall top masks by smallest gap:")
    for row in sorted(rows, key=lambda item: (item.gap_to_reference, item.replay_count, sorted(item.replay_horizons)))[:n]:
        _print_replay_mask_row(row)
    print()

    if mode == "all":
        _print_replay_mask_full_mode_diagnostics(rows, T)
    else:
        print("Full-mask diagnostics are available in --mode all.")
        print()

    print("Rows:")
    for row in sorted(rows, key=lambda item: (item.replay_count, sorted(item.replay_horizons))):
        _print_replay_mask_row(row)


def time_policy_boundary_loss_rows(
    k: int,
    T: int,
    reference_policy_name: str,
    candidate_policy_name: str,
    occupancy_policy_name: str = "reference",
) -> tuple[TimePolicyBoundaryLossRow, ...]:
    reference_policy = _resolve_time_dependent_policy(k, reference_policy_name)
    candidate_policy = _resolve_time_dependent_policy(k, candidate_policy_name)
    if reference_policy is None:
        raise ValueError(f"unknown reference policy: {reference_policy_name}")
    if candidate_policy is None:
        raise ValueError(f"unknown candidate policy: {candidate_policy_name}")

    reference_values = evaluate_time_dependent_policy(k, T, reference_policy)
    candidate_values = evaluate_time_dependent_policy(k, T, candidate_policy)
    if occupancy_policy_name == "reference":
        occupancy = _time_policy_occupancy(k, T, reference_policy)
    elif occupancy_policy_name == "candidate":
        occupancy = _time_policy_occupancy(k, T, candidate_policy)
    else:
        raise ValueError("occupancy_policy_name must be 'reference' or 'candidate'")

    rows: list[TimePolicyBoundaryLossRow] = []
    for time in range(T):
        remaining_horizon = T - time
        reference_continuation = reference_values[remaining_horizon - 1]
        for state, probability in occupancy[time].items():
            reference_policy_at_state = reference_policy(state, remaining_horizon)
            candidate_policy_at_state = candidate_policy(state, remaining_horizon)
            reference_step = _time_policy_step_value(k, state, reference_policy_at_state, reference_continuation)
            candidate_step = _time_policy_step_value(k, state, candidate_policy_at_state, reference_continuation)
            reference_value = reference_values[remaining_horizon][state]
            candidate_value = candidate_values[remaining_horizon][state]
            action_loss = reference_step - candidate_step
            downstream_loss = candidate_step - candidate_value
            total_gap = reference_value - candidate_value
            rows.append(
                TimePolicyBoundaryLossRow(
                    time=time,
                    remaining_horizon=remaining_horizon,
                    state=state,
                    packet_type=packet_type(state),
                    packet_gaps=_packet_gaps(state),
                    adjacent_gaps=_gap_vector(state),
                    occupancy_probability=probability,
                    reference_value=reference_value,
                    candidate_value=candidate_value,
                    reference_step_value_on_reference_continuation=reference_step,
                    candidate_step_value_on_reference_continuation=candidate_step,
                    action_loss_on_reference_continuation=action_loss,
                    downstream_loss_after_candidate_action=downstream_loss,
                    total_value_gap=total_gap,
                    weighted_action_loss=probability * action_loss,
                    weighted_downstream_loss=probability * downstream_loss,
                    weighted_total_gap=probability * total_gap,
                    reference_support_count=len(reference_policy_at_state),
                    candidate_support_count=len(candidate_policy_at_state),
                )
            )
    return tuple(rows)


def _print_boundary_loss_row(row: TimePolicyBoundaryLossRow) -> None:
    print(
        f"  t={row.time} rem={row.remaining_horizon}"
        f" state={row.state}"
        f" packet={row.packet_type} gaps={row.packet_gaps}"
        f" occ={row.occupancy_probability:.6f}"
        f" action_loss={row.action_loss_on_reference_continuation:.6f}"
        f" downstream_loss={row.downstream_loss_after_candidate_action:.6f}"
        f" total_gap={row.total_value_gap:.6f}"
        f" weighted_action={row.weighted_action_loss:.6f}"
        f" weighted_downstream={row.weighted_downstream_loss:.6f}"
        f" weighted_total={row.weighted_total_gap:.6f}"
        f" support={row.reference_support_count}/{row.candidate_support_count}"
    )


def print_time_policy_boundary_loss(
    k: int,
    T: int,
    reference_policy_name: str,
    candidate_policy_name: str,
    occupancy_policy_name: str = "reference",
    n: int = 80,
) -> None:
    rows = time_policy_boundary_loss_rows(
        k,
        T,
        reference_policy_name,
        candidate_policy_name,
        occupancy_policy_name=occupancy_policy_name,
    )
    print(f"Time-policy boundary loss, k={k}, T={T}")
    print()
    print(f"reference policy: {reference_policy_name}")
    print(f"candidate policy: {candidate_policy_name}")
    print(f"occupancy policy: {occupancy_policy_name}")
    print(f"rows: {len(rows)}")
    print()
    print("Overall totals:")
    print(f"  occupancy mass: {sum(row.occupancy_probability for row in rows):.6f}")
    print(f"  weighted action loss: {sum(row.weighted_action_loss for row in rows):.6f}")
    print(f"  weighted downstream loss: {sum(row.weighted_downstream_loss for row in rows):.6f}")
    print(f"  weighted total gap: {sum(row.weighted_total_gap for row in rows):.6f}")
    print()

    by_remaining: dict[int, list[TimePolicyBoundaryLossRow]] = defaultdict(list)
    by_action_regime: dict[tuple[tuple[int, ...], tuple[int, ...]], list[TimePolicyBoundaryLossRow]] = defaultdict(list)
    by_total_regime: dict[tuple[tuple[int, ...], tuple[int, ...]], list[TimePolicyBoundaryLossRow]] = defaultdict(list)
    for row in rows:
        by_remaining[row.remaining_horizon].append(row)
        by_action_regime[(row.packet_type, row.packet_gaps)].append(row)
        by_total_regime[(row.packet_type, row.packet_gaps)].append(row)

    print("Loss by remaining horizon:")
    for remaining in sorted(by_remaining, reverse=True):
        group = by_remaining[remaining]
        print(
            f"  rem={remaining}"
            f" occupancy={sum(row.occupancy_probability for row in group):.6f}"
            f" action={sum(row.weighted_action_loss for row in group):.6f}"
            f" downstream={sum(row.weighted_downstream_loss for row in group):.6f}"
            f" total={sum(row.weighted_total_gap for row in group):.6f}"
        )
    print()

    print("Top canonical states by weighted action loss:")
    for row in sorted(rows, key=lambda item: (-item.weighted_action_loss, item.state))[:n]:
        _print_boundary_loss_row(row)
    print()

    print("Top canonical states by weighted downstream loss:")
    for row in sorted(rows, key=lambda item: (-item.weighted_downstream_loss, item.state))[:n]:
        _print_boundary_loss_row(row)
    print()

    print("Top packet regimes by weighted action loss:")
    action_regime_totals = [
        (
            key,
            sum(row.occupancy_probability for row in group),
            sum(row.weighted_action_loss for row in group),
            sum(row.weighted_downstream_loss for row in group),
            sum(row.weighted_total_gap for row in group),
        )
        for key, group in by_action_regime.items()
    ]
    for (ptype, gaps), occupancy, action_loss, downstream_loss, total_gap in sorted(
        action_regime_totals, key=lambda item: (-item[2], item[0])
    )[:n]:
        print(
            f"  packet={ptype} gaps={gaps}"
            f" occupancy={occupancy:.6f}"
            f" action={action_loss:.6f}"
            f" downstream={downstream_loss:.6f}"
            f" total={total_gap:.6f}"
        )
    print()

    print("Top packet regimes by weighted total gap:")
    for (ptype, gaps), occupancy, action_loss, downstream_loss, total_gap in sorted(
        action_regime_totals, key=lambda item: (-item[4], item[0])
    )[:n]:
        print(
            f"  packet={ptype} gaps={gaps}"
            f" occupancy={occupancy:.6f}"
            f" action={action_loss:.6f}"
            f" downstream={downstream_loss:.6f}"
            f" total={total_gap:.6f}"
        )


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

    explicit_time_policy_reachable_benchmark_parser = subparsers.add_parser("explicit-time-policy-reachable-benchmark")
    explicit_time_policy_reachable_benchmark_parser.add_argument(
        "--cases",
        default="6:12,6:20,6:30,6:50,6:100,6:200",
        help="comma-separated k:T cases, e.g. 6:50,6:100",
    )
    explicit_time_policy_reachable_benchmark_parser.add_argument(
        "--policies",
        default=(
            "top_prefix_all,"
            "k6_alive_full_comb_control_policy,"
            "k6_alive_trunc_comb_policy,"
            "k6_patch_ladder_5,"
            "k6_greedy_deterministic_patched_policy,"
            "k6_one_run_rule_v1_policy"
        ),
        help="comma-separated time-dependent policy names",
    )

    k6_barycentric_candidate_benchmark_parser = subparsers.add_parser("k6-barycentric-candidate-benchmark")
    k6_barycentric_candidate_benchmark_parser.add_argument(
        "--T-values",
        default="12,20,30,50,100",
        help="comma-separated horizons, e.g. 12,20,50",
    )
    k6_barycentric_candidate_benchmark_parser.add_argument(
        "--policies",
        default=",".join(_k6_barycentric_default_policy_names()),
        help="comma-separated time-dependent policy names",
    )
    k6_barycentric_candidate_benchmark_parser.add_argument("--trigger-n", type=int, default=20)
    k6_barycentric_candidate_benchmark_parser.add_argument("--trigger-stats-max-T", type=int, default=20)

    k6_barycentric_trigger_sweep_parser = subparsers.add_parser("k6-barycentric-trigger-sweep")
    k6_barycentric_trigger_sweep_parser.add_argument(
        "--T-values",
        default="12,20,30,50",
        help="comma-separated primary horizons, e.g. 12,20,30,50",
    )
    k6_barycentric_trigger_sweep_parser.add_argument(
        "--final-T-values",
        default="100",
        help="comma-separated final horizons for top candidates, e.g. 100",
    )
    k6_barycentric_trigger_sweep_parser.add_argument("--trigger-n", type=int, default=20)
    k6_barycentric_trigger_sweep_parser.add_argument("--trigger-stats-max-T", type=int, default=20)
    k6_barycentric_trigger_sweep_parser.add_argument("--finalist-count", type=int, default=10)

    k5_centered_potential_parser = subparsers.add_parser("k5-centered-potential-comb-vs-chase-report")
    k5_centered_potential_parser.add_argument("--max-used", type=int, default=8)
    k5_centered_potential_parser.add_argument("-n", type=int, default=40)

    k5_softmax_certificate_parser = subparsers.add_parser("k5-softmax-potential-certificate-report")
    k5_softmax_certificate_parser.add_argument("--max-used", type=int, default=8)
    k5_softmax_certificate_parser.add_argument("-n", type=int, default=40)

    potential_greedy_interval_benchmark_parser = subparsers.add_parser("potential-greedy-interval-benchmark")
    potential_greedy_interval_benchmark_parser.add_argument(
        "--cases",
        default="5:20,5:50,6:12,6:30,6:50",
        help="comma-separated k:T cases, e.g. 5:20,6:30",
    )
    potential_greedy_interval_benchmark_parser.add_argument(
        "--policies",
        default=",".join(_potential_greedy_default_policy_names()),
        help="comma-separated policy names",
    )
    potential_greedy_interval_benchmark_parser.add_argument("--print-action-stats", action="store_true")
    potential_greedy_interval_benchmark_parser.add_argument("-n", type=int, default=20)

    random_playout_potential_benchmark_parser = subparsers.add_parser("random-playout-potential-benchmark")
    random_playout_potential_benchmark_parser.add_argument(
        "--cases",
        default="5:20,5:50,6:12,6:30",
        help="comma-separated k:T cases, e.g. 5:20,6:30",
    )
    random_playout_potential_benchmark_parser.add_argument(
        "--playouts",
        default="all",
        help="comma-separated playout names, or all",
    )
    random_playout_potential_benchmark_parser.add_argument(
        "--mode",
        choices=("values", "greedy", "residuals", "all"),
        default="all",
    )
    random_playout_potential_benchmark_parser.add_argument("--print-action-stats", action="store_true")
    random_playout_potential_benchmark_parser.add_argument("--residual-max-T", type=int, default=12)
    random_playout_potential_benchmark_parser.add_argument("-n", type=int, default=20)

    k5_gap_proof_diagnostics_parser = subparsers.add_parser("k5-gap-proof-diagnostics")
    k5_gap_proof_diagnostics_parser.add_argument("--max-used", type=int, default=8)
    k5_gap_proof_diagnostics_parser.add_argument(
        "--h-values",
        default="5,10",
        help="comma-separated small horizons, e.g. 5,10",
    )
    k5_gap_proof_diagnostics_parser.add_argument("-n", type=int, default=40)

    k5_scalar_potential_exchangeability_parser = subparsers.add_parser("k5-scalar-potential-exchangeability")
    k5_scalar_potential_exchangeability_parser.add_argument("--max-used", type=int, default=8)
    k5_scalar_potential_exchangeability_parser.add_argument(
        "--h-values",
        default="5,10,20",
        help="comma-separated small horizons, e.g. 5,10,20",
    )
    k5_scalar_potential_exchangeability_parser.add_argument(
        "--playouts",
        default="Q_chase,Q_comb",
        help="comma-separated k=5 playout names",
    )
    k5_scalar_potential_exchangeability_parser.add_argument("-n", type=int, default=40)

    k5_boundary_correction_fit_parser = subparsers.add_parser("k5-boundary-correction-fit-report")
    k5_boundary_correction_fit_parser.add_argument("--max-used", type=int, default=8)
    k5_boundary_correction_fit_parser.add_argument(
        "--h-values",
        default="5,10,20",
        help="comma-separated small horizons, e.g. 5,10,20",
    )
    k5_boundary_correction_fit_parser.add_argument(
        "--playouts",
        default="Q_chase,Q_comb",
        help="comma-separated k=5 playout names",
    )
    k5_boundary_correction_fit_parser.add_argument(
        "--feature-families",
        default="all",
        help="comma-separated feature families, or all",
    )
    k5_boundary_correction_fit_parser.add_argument("--ridge", type=float, default=1e-8)
    k5_boundary_correction_fit_parser.add_argument("-n", type=int, default=40)

    k5_relaxation_certificate_parser = subparsers.add_parser("k5-relaxation-certificate-report")
    k5_relaxation_certificate_parser.add_argument("--max-used", type=int, default=8)
    k5_relaxation_certificate_parser.add_argument(
        "--h-values",
        default="5,10,20,50",
        help="comma-separated small horizons, e.g. 5,10,20,50",
    )
    k5_relaxation_certificate_parser.add_argument(
        "--potentials",
        default="Q_chase,Q_comb,softmax_c1",
        help="comma-separated potential names",
    )
    k5_relaxation_certificate_parser.add_argument(
        "--action-family",
        default="all_binary",
        choices=("all_binary", "balanced_pairs_support"),
    )
    k5_relaxation_certificate_parser.add_argument("-n", type=int, default=40)

    k5_repaired_learner_anatomy_parser = subparsers.add_parser("k5-repaired-learner-anatomy-report")
    k5_repaired_learner_anatomy_parser.add_argument("--max-used", type=int, default=8)
    k5_repaired_learner_anatomy_parser.add_argument(
        "--h-values",
        default="5,10,20,50,100",
        help="comma-separated small horizons, e.g. 5,10,20,50,100",
    )
    k5_repaired_learner_anatomy_parser.add_argument("--potential", default="Q_chase")
    k5_repaired_learner_anatomy_parser.add_argument(
        "--action-family",
        default="all_binary",
        choices=("all_binary", "balanced_pairs_support"),
    )
    k5_repaired_learner_anatomy_parser.add_argument("-n", type=int, default=40)

    k5_packet_learner_formula_parser = subparsers.add_parser("k5-packet-learner-formula-report")
    k5_packet_learner_formula_parser.add_argument("--max-used", type=int, default=8)
    k5_packet_learner_formula_parser.add_argument(
        "--h-values",
        default="5,10,20,50,100",
        help="comma-separated small horizons, e.g. 5,10,20,50,100",
    )
    k5_packet_learner_formula_parser.add_argument("--potential", default="Q_chase")
    k5_packet_learner_formula_parser.add_argument(
        "--action-family",
        default="all_binary",
        choices=("all_binary", "balanced_pairs_support"),
    )
    k5_packet_learner_formula_parser.add_argument("-n", type=int, default=40)

    k5_reduced_packet_lp_parser = subparsers.add_parser("k5-reduced-packet-lp-report")
    k5_reduced_packet_lp_parser.add_argument("--max-used", type=int, default=8)
    k5_reduced_packet_lp_parser.add_argument(
        "--h-values",
        default="5,10,20,50,100",
        help="comma-separated small horizons, e.g. 5,10,20,50,100",
    )
    k5_reduced_packet_lp_parser.add_argument("--potential", default="Q_chase")
    k5_reduced_packet_lp_parser.add_argument("-n", type=int, default=40)

    k5_reduced_packet_lp_coverage_parser = subparsers.add_parser("k5-reduced-packet-lp-coverage-report")
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--h-values",
        default="5,10,20,50,100",
        help="comma-separated small horizons, e.g. 5,10,20,50,100",
    )
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--max-used-values",
        default="8,12,16,20",
        help="comma-separated max-used cutoffs",
    )
    k5_reduced_packet_lp_coverage_parser.add_argument("--potential", default="Q_chase")
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--reachable-mode",
        default="both",
        choices=("none", "chase", "random_playout", "both"),
    )
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--reachable-T-values",
        default="20,50,100",
        help="comma-separated reachable horizons",
    )
    k5_reduced_packet_lp_coverage_parser.add_argument("--sample-limit", type=int, default=0)
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of parallel worker processes for independent coverage tasks",
    )
    k5_reduced_packet_lp_coverage_parser.add_argument(
        "--progress-file",
        default=None,
        help="optional file receiving timestamped progress updates; updates also go to stderr",
    )
    k5_reduced_packet_lp_coverage_parser.add_argument("-n", type=int, default=40)

    k5_worst_family_scaling_parser = subparsers.add_parser("k5-worst-family-scaling-report")
    k5_worst_family_scaling_parser.add_argument(
        "--h-values",
        default="50,100,200,500",
        help="comma-separated horizons, e.g. 50,100,200,500",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--families",
        default="packet_122,packet_1211,packet_221,packet_212",
        help="comma-separated family names or all",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--grid",
        default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3",
        help="comma-separated scaled gap grid",
    )
    k5_worst_family_scaling_parser.add_argument("--potential", default="Q_chase")
    k5_worst_family_scaling_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel worker count; split by horizon to avoid duplicated cache blowups",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--progress-file",
        default=None,
        help="optional timestamped progress log; updates also go to stderr",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--start-method",
        default="spawn",
        choices=("spawn", "forkserver", "fork"),
        help="multiprocessing start method; spawn is safest for avoiding fork freezes",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--memory-safe-above",
        type=int,
        default=150,
        help="horizons above this value run one at a time even when --jobs > 1; use 0 to disable",
    )
    k5_worst_family_scaling_parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="emit within-horizon progress every N scaled grid points; use 0 to disable",
    )
    k5_worst_family_scaling_parser.add_argument("-n", type=int, default=40)

    k6_adaptive_interval_anatomy_report_parser = subparsers.add_parser("k6-adaptive-interval-anatomy-report")
    k6_adaptive_interval_anatomy_report_parser.add_argument("--T", type=int, required=True)
    k6_adaptive_interval_anatomy_report_parser.add_argument(
        "--long-T-values",
        default="30,50,100",
        help="comma-separated T values for fixed-pattern comparison",
    )
    k6_adaptive_interval_anatomy_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    k6_adaptive_interval_anatomy_report_parser.add_argument("-n", type=int, default=40)

    k6_barycenter_action_anatomy_report_parser = subparsers.add_parser("k6-barycenter-action-anatomy-report")
    k6_barycenter_action_anatomy_report_parser.add_argument("--T", type=int, required=True)
    k6_barycenter_action_anatomy_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    k6_barycenter_action_anatomy_report_parser.add_argument("-n", type=int, default=30)

    chase5_verification_report_parser = subparsers.add_parser("chase5-verification-report")
    chase5_verification_report_parser.add_argument(
        "--T-values",
        default="5,10,20,50",
        help="comma-separated T values",
    )
    chase5_verification_report_parser.add_argument("-n", type=int, default=20)

    chase5_potential_certificate_report_parser = subparsers.add_parser("chase5-potential-certificate-report")
    chase5_potential_certificate_report_parser.add_argument(
        "--T-values",
        default="5,10,20,50",
        help="comma-separated T values",
    )
    chase5_potential_certificate_report_parser.add_argument("-n", type=int, default=20)

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

    two_run_skeleton_placement_report_parser = subparsers.add_parser("two-run-skeleton-placement-report")
    two_run_skeleton_placement_report_parser.add_argument("--k", type=int, required=True)
    two_run_skeleton_placement_report_parser.add_argument("--T", type=int, required=True)
    two_run_skeleton_placement_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_skeleton_placement_report_parser.add_argument("-n", type=int, default=80)

    two_run_replay_rule_mining_report_parser = subparsers.add_parser("two-run-replay-rule-mining-report")
    two_run_replay_rule_mining_report_parser.add_argument("--k", type=int, required=True)
    two_run_replay_rule_mining_report_parser.add_argument("--T", type=int, required=True)
    two_run_replay_rule_mining_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_replay_rule_mining_report_parser.add_argument("-n", type=int, default=80)

    two_run_successor_flow_report_parser = subparsers.add_parser("two-run-successor-flow-report")
    two_run_successor_flow_report_parser.add_argument("--k", type=int, required=True)
    two_run_successor_flow_report_parser.add_argument("--T", type=int, required=True)
    two_run_successor_flow_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_successor_flow_report_parser.add_argument("-n", type=int, default=100)

    two_run_flow_role_report_parser = subparsers.add_parser("two-run-flow-role-report")
    two_run_flow_role_report_parser.add_argument("--k", type=int, required=True)
    two_run_flow_role_report_parser.add_argument("--T", type=int, required=True)
    two_run_flow_role_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    two_run_flow_role_report_parser.add_argument("-n", type=int, default=100)

    k6_one_run_alive_pattern_report_parser = subparsers.add_parser("k6-one-run-alive-pattern-report")
    k6_one_run_alive_pattern_report_parser.add_argument("--T", type=int, required=True)
    k6_one_run_alive_pattern_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    k6_one_run_alive_pattern_report_parser.add_argument("-n", type=int, default=80)

    k6_alive_candidate_benchmark_parser = subparsers.add_parser("k6-alive-candidate-benchmark")
    k6_alive_candidate_benchmark_parser.add_argument(
        "--T",
        default="12",
        help="comma-separated T values, e.g. 8,10,12",
    )

    k6_one_run_interval_rule_report_parser = subparsers.add_parser("k6-one-run-interval-rule-report")
    k6_one_run_interval_rule_report_parser.add_argument("--T", type=int, required=True)
    k6_one_run_interval_rule_report_parser.add_argument("--support-tolerance", type=float, default=1e-8)
    k6_one_run_interval_rule_report_parser.add_argument("-n", type=int, default=100)

    k6_one_run_patch_ladder_benchmark_parser = subparsers.add_parser("k6-one-run-patch-ladder-benchmark")
    k6_one_run_patch_ladder_benchmark_parser.add_argument("--T", type=int, required=True)

    k6_one_run_greedy_patch_search_parser = subparsers.add_parser("k6-one-run-greedy-patch-search")
    k6_one_run_greedy_patch_search_parser.add_argument("--T", type=int, required=True)
    k6_one_run_greedy_patch_search_parser.add_argument("--max-patches", type=int, default=20)
    k6_one_run_greedy_patch_search_parser.add_argument("--candidate-n", type=int, default=50)

    k6_one_run_greedy_mixed_patch_search_parser = subparsers.add_parser("k6-one-run-greedy-mixed-patch-search")
    k6_one_run_greedy_mixed_patch_search_parser.add_argument("--T", type=int, required=True)
    k6_one_run_greedy_mixed_patch_search_parser.add_argument("--max-patches", type=int, default=20)
    k6_one_run_greedy_mixed_patch_search_parser.add_argument("--candidate-n", type=int, default=50)

    k6_explicit_long_horizon_sweep_parser = subparsers.add_parser("k6-explicit-long-horizon-sweep")
    k6_explicit_long_horizon_sweep_parser.add_argument(
        "--T",
        default="12,20,30,50,100,200",
        help="comma-separated T values, e.g. 12,20,30",
    )
    k6_explicit_long_horizon_sweep_parser.add_argument(
        "--policies",
        default=",".join(_default_k6_explicit_long_horizon_policies()),
        help="comma-separated explicit time-dependent policies",
    )

    time_policy_boundary_loss_parser = subparsers.add_parser("time-policy-boundary-loss")
    time_policy_boundary_loss_parser.add_argument("--k", type=int, required=True)
    time_policy_boundary_loss_parser.add_argument("--T", type=int, required=True)
    time_policy_boundary_loss_parser.add_argument("--reference-policy", required=True)
    time_policy_boundary_loss_parser.add_argument("--candidate-policy", required=True)
    time_policy_boundary_loss_parser.add_argument(
        "--occupancy-policy",
        choices=("reference", "candidate"),
        default="reference",
    )
    time_policy_boundary_loss_parser.add_argument("-n", type=int, default=80)

    replay_layer_mask_sweep_parser = subparsers.add_parser("replay-layer-mask-sweep")
    replay_layer_mask_sweep_parser.add_argument("--k", type=int, required=True)
    replay_layer_mask_sweep_parser.add_argument("--T", type=int, required=True)
    replay_layer_mask_sweep_parser.add_argument("--fallback-policy", required=True)
    replay_layer_mask_sweep_parser.add_argument(
        "--reference-policy",
        default="two_run_dual_support_replay_k9_T7",
    )
    replay_layer_mask_sweep_parser.add_argument(
        "--mode",
        choices=("all", "singletons", "pairs", "prefixes", "suffixes"),
        default="all",
    )
    replay_layer_mask_sweep_parser.add_argument("-n", type=int, default=80)

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
    if args.command == "explicit-time-policy-reachable-benchmark":
        print_explicit_time_policy_reachable_benchmark(
            _parse_cases(args.cases),
            _parse_policy_names(args.policies),
        )
        return
    if args.command == "k6-barycentric-candidate-benchmark":
        print_k6_barycentric_candidate_benchmark(
            _parse_T_values(args.T_values),
            _parse_policy_names(args.policies),
            trigger_n=args.trigger_n,
            trigger_stats_max_T=args.trigger_stats_max_T,
        )
        return
    if args.command == "k6-barycentric-trigger-sweep":
        print_k6_barycentric_trigger_sweep(
            _parse_T_values(args.T_values),
            final_T_values=_parse_T_values(args.final_T_values),
            trigger_n=args.trigger_n,
            trigger_stats_max_T=args.trigger_stats_max_T,
            finalist_count=args.finalist_count,
        )
        return
    if args.command == "k5-centered-potential-comb-vs-chase-report":
        print_k5_centered_potential_comb_vs_chase_report(
            args.max_used,
            n=args.n,
        )
        return
    if args.command == "k5-softmax-potential-certificate-report":
        print_k5_softmax_potential_certificate_report(
            args.max_used,
            n=args.n,
        )
        return
    if args.command == "potential-greedy-interval-benchmark":
        print_potential_greedy_interval_benchmark(
            _parse_cases(args.cases),
            _parse_policy_names(args.policies),
            print_action_stats=args.print_action_stats,
            n=args.n,
        )
        return
    if args.command == "random-playout-potential-benchmark":
        print_random_playout_potential_benchmark(
            _parse_cases(args.cases),
            playout_names=_parse_random_playout_names(args.playouts),
            mode=args.mode,
            print_action_stats=args.print_action_stats,
            residual_max_T=args.residual_max_T,
            n=args.n,
        )
        return
    if args.command == "k5-gap-proof-diagnostics":
        print_k5_gap_proof_diagnostics(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            n=args.n,
        )
        return
    if args.command == "k5-scalar-potential-exchangeability":
        print_k5_scalar_potential_exchangeability(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            playout_names=_parse_random_playout_names(args.playouts) or ("Q_chase", "Q_comb"),
            n=args.n,
        )
        return
    if args.command == "k5-boundary-correction-fit-report":
        print_k5_boundary_correction_fit_report(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            playout_names=_parse_random_playout_names(args.playouts) or ("Q_chase", "Q_comb"),
            feature_families=_parse_k5_boundary_feature_families(args.feature_families),
            ridge=args.ridge,
            n=args.n,
        )
        return
    if args.command == "k5-relaxation-certificate-report":
        print_k5_relaxation_certificate_report(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            potential_names=_parse_k5_relaxation_potentials(args.potentials),
            action_family=args.action_family,
            n=args.n,
        )
        return
    if args.command == "k5-repaired-learner-anatomy-report":
        print_k5_repaired_learner_anatomy_report(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            potential_name=args.potential,
            action_family=args.action_family,
            n=args.n,
        )
        return
    if args.command == "k5-packet-learner-formula-report":
        print_k5_packet_learner_formula_report(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            potential_name=args.potential,
            action_family=args.action_family,
            n=args.n,
        )
        return
    if args.command == "k5-reduced-packet-lp-report":
        print_k5_reduced_packet_lp_report(
            max_used=args.max_used,
            h_values=_parse_T_values(args.h_values),
            potential_name=args.potential,
            n=args.n,
        )
        return
    if args.command == "k5-reduced-packet-lp-coverage-report":
        print_k5_reduced_packet_lp_coverage_report(
            h_values=_parse_T_values(args.h_values),
            max_used_values=_parse_T_values(args.max_used_values),
            potential_name=args.potential,
            reachable_mode=args.reachable_mode,
            reachable_T_values=_parse_T_values(args.reachable_T_values),
            sample_limit=args.sample_limit,
            jobs=args.jobs,
            progress_file=args.progress_file,
            n=args.n,
        )
        return
    if args.command == "k5-worst-family-scaling-report":
        print_k5_worst_family_scaling_report(
            h_values=_parse_T_values(args.h_values),
            families=_parse_k5_worst_families(args.families),
            grid=_parse_float_grid(args.grid),
            potential_name=args.potential,
            jobs=args.jobs,
            progress_file=args.progress_file,
            start_method=args.start_method,
            memory_safe_above=args.memory_safe_above,
            progress_every=args.progress_every,
            n=args.n,
        )
        return
    if args.command == "k6-adaptive-interval-anatomy-report":
        print_k6_adaptive_interval_anatomy_report(
            args.T,
            _parse_T_values(args.long_T_values),
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "k6-barycenter-action-anatomy-report":
        print_k6_barycenter_action_anatomy_report(
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "chase5-verification-report":
        print_chase5_verification_report(
            _parse_T_values(args.T_values),
            n=args.n,
        )
        return
    if args.command == "chase5-potential-certificate-report":
        print_chase5_potential_certificate_report(
            _parse_T_values(args.T_values),
            n=args.n,
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
    if args.command == "two-run-skeleton-placement-report":
        print_two_run_skeleton_placement_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "two-run-replay-rule-mining-report":
        print_two_run_replay_rule_mining_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "two-run-successor-flow-report":
        print_two_run_successor_flow_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "two-run-flow-role-report":
        print_two_run_flow_role_report(
            args.k,
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "k6-one-run-alive-pattern-report":
        print_k6_one_run_alive_pattern_report(
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "k6-alive-candidate-benchmark":
        print_k6_alive_candidate_benchmark(_parse_T_values(args.T))
        return
    if args.command == "k6-one-run-interval-rule-report":
        print_k6_one_run_interval_rule_report(
            args.T,
            support_tolerance=args.support_tolerance,
            n=args.n,
        )
        return
    if args.command == "k6-one-run-patch-ladder-benchmark":
        print_k6_one_run_patch_ladder_benchmark(args.T)
        return
    if args.command == "k6-one-run-greedy-patch-search":
        print_k6_one_run_greedy_patch_search(
            args.T,
            max_patches=args.max_patches,
            candidate_n=args.candidate_n,
        )
        return
    if args.command == "k6-one-run-greedy-mixed-patch-search":
        print_k6_one_run_greedy_mixed_patch_search(
            args.T,
            max_patches=args.max_patches,
            candidate_n=args.candidate_n,
        )
        return
    if args.command == "k6-explicit-long-horizon-sweep":
        print_k6_explicit_long_horizon_sweep(
            _parse_T_values(args.T),
            tuple(item.strip() for item in args.policies.split(",") if item.strip()),
        )
        return
    if args.command == "time-policy-boundary-loss":
        print_time_policy_boundary_loss(
            args.k,
            args.T,
            args.reference_policy,
            args.candidate_policy,
            occupancy_policy_name=args.occupancy_policy,
            n=args.n,
        )
        return
    if args.command == "replay-layer-mask-sweep":
        print_replay_layer_mask_sweep(
            args.k,
            args.T,
            args.fallback_policy,
            reference_policy_name=args.reference_policy,
            mode=args.mode,
            n=args.n,
        )
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
