from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .actions import all_actions, comb_action, fixed_rank_action
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
    best_action_value: float
    policy_support: tuple[tuple[float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class PacketWeightedGreedySummary:
    packet_type: tuple[int, ...]
    total_contribution: float
    occupancy_mass: float
    average_local_defect: float
    top_best_actions: tuple[tuple[str, float], ...]
    top_policy_supports: tuple[tuple[str, float], ...]


def _format_action(action: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in action)


def _format_policy(policy: tuple[tuple[float, tuple[int, ...]], ...], max_items: int = 8) -> str:
    visible = [
        f"{_format_action(action)}:{probability:.3f}"
        for probability, action in policy
        if probability >= 1e-6
    ]
    if len(visible) <= max_items:
        return ", ".join(visible)
    return ", ".join(visible[:max_items]) + ", ..."


def summarize_weighted_greedy_by_packet(
    contributions: list[WeightedGreedyContribution],
    n: int = 10,
) -> list[PacketWeightedGreedySummary]:
    grouped: dict[tuple[int, ...], list[WeightedGreedyContribution]] = defaultdict(list)
    for item in contributions:
        grouped[item.packet_type].append(item)

    summaries: list[PacketWeightedGreedySummary] = []
    for ptype, items in grouped.items():
        total_contribution = sum(item.contribution for item in items)
        occupancy_mass = sum(item.occupancy_probability for item in items)
        weighted_defect_sum = sum(item.occupancy_probability * item.local_defect for item in items)
        average_local_defect = weighted_defect_sum / occupancy_mass if occupancy_mass > 0 else 0.0

        best_action_totals: dict[str, float] = defaultdict(float)
        policy_support_totals: dict[str, float] = defaultdict(float)
        for item in items:
            best_action_totals[_format_action(item.best_action)] += item.contribution
            policy_support_totals[_format_policy(item.policy_support)] += item.contribution

        top_best_actions = tuple(
            sorted(best_action_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
        )
        top_policy_supports = tuple(
            sorted(policy_support_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:n]
        )
        summaries.append(
            PacketWeightedGreedySummary(
                packet_type=ptype,
                total_contribution=total_contribution,
                occupancy_mass=occupancy_mass,
                average_local_defect=average_local_defect,
                top_best_actions=top_best_actions,
                top_policy_supports=top_policy_supports,
            )
        )

    summaries.sort(key=lambda item: item.total_contribution, reverse=True)
    return summaries


def _policy_registry(k: int) -> dict[str, object]:
    policies: dict[str, object] = {
        "comb": comb_policy,
        "packet_balanced_partition": packet_balanced_partition_policy,
        "packet_frontier": packet_frontier_policy,
        "packet_minimal_frontier": packet_minimal_frontier_policy,
    }
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
                    best_action_value=best_action_value,
                    policy_support=policy_support,
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
    print("time  remaining  state               packet type      occupancy    defect    contribution  best    value     policy support")
    for item in contributions[:n]:
        print(
            f"{item.time:4d} {item.remaining_horizon:10d} {str(item.state):18s}"
            f" {str(item.packet_type):14s} {item.occupancy_probability:10.6f}"
            f" {item.local_defect:9.6f} {item.contribution:13.6f}"
            f"  {_format_action(item.best_action):5s}  {item.best_action_value:8.6f}"
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
        print()


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
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
