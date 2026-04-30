from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .actions import comb_action, complement, fixed_rank_action
from .state import canon, packets


Policy = list[tuple[float, tuple[int, ...]]]


def _balanced_from_base(action_weights: list[tuple[float, tuple[int, ...]]]) -> Policy:
    total = sum(weight for weight, _ in action_weights)
    if total <= 0:
        raise ValueError("policy must have positive total mass")

    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for weight, action in action_weights:
        normalized = weight / total
        aggregated[action] += 0.5 * normalized
        aggregated[complement(action)] += 0.5 * normalized

    return sorted(
        ((probability, action) for action, probability in aggregated.items() if probability > 0.0),
        key=lambda item: item[1],
    )


def comb_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    return _balanced_from_base([(1.0, comb_action(len(state)))])


def fixed_rank_policy(ones: set[int]):
    def policy(x: tuple[int, ...]) -> Policy:
        state = canon(x)
        return _balanced_from_base([(1.0, fixed_rank_action(len(state), ones))])

    return policy


def packet_frontier_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    state_packets = packets(state)
    top_packet = state_packets[0]
    k = len(state)

    if len(top_packet) == 1:
        action = [0] * k
        action[top_packet[0]] = 1
        return _balanced_from_base([(1.0, tuple(action))])

    split_size = len(top_packet) // 2
    base_actions: list[tuple[float, tuple[int, ...]]] = []
    for chosen in combinations(top_packet, split_size):
        action = [0] * k
        for index in chosen:
            action[index] = 1
        base_actions.append((1.0, tuple(action)))
    return _balanced_from_base(base_actions)
