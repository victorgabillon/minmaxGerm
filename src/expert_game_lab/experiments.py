from __future__ import annotations

import argparse
from itertools import combinations

from .actions import all_actions, comb_action, fixed_rank_action
from .defects import commutation_defect, greedy_defect
from .dp_optimal import optimal_values
from .dp_policy import evaluate_balanced_policy
from .lp_game import solve_minimax_step
from .policies import comb_policy, fixed_rank_policy, packet_frontier_policy
from .state import all_states, canon, packet_type


def _policy_registry(k: int) -> dict[str, object]:
    policies: dict[str, object] = {
        "comb": comb_policy,
        "packet_frontier": packet_frontier_policy,
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
    print("policy                value        gap      gap/sqrt(T)")
    for name, policy_fn in sorted(_policy_registry(k).items()):
        values = evaluate_balanced_policy(k, T, policy_fn)
        policy_zero = values[T][zero]
        gap = optimal_zero - policy_zero
        normalized_gap = gap / (T ** 0.5 if T > 0 else 1.0)
        print(f"{name:20s} {policy_zero:10.6f} {gap:10.6f} {normalized_gap:12.6f}")


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
            action: prev[canon(tuple(state[index] + action[index] for index in range(k)))]
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
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
