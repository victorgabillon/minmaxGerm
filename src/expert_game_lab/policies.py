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


def packet_minimal_frontier_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    top_packet = packets(state)[0]
    k = len(state)

    if len(top_packet) == 1:
        action = [0] * k
        action[top_packet[0]] = 1
        return _balanced_from_base([(1.0, tuple(action))])

    base_actions: list[tuple[float, tuple[int, ...]]] = []
    for index in top_packet:
        action = [0] * k
        action[index] = 1
        base_actions.append((1.0, tuple(action)))
    return _balanced_from_base(base_actions)


def packet_balanced_partition_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    k = len(state)
    target = k // 2
    state_packets = packets(state)
    candidate_actions: list[tuple[int, ...]] = []
    for chosen in combinations(range(k), target):
        action = [0] * k
        for index in chosen:
            action[index] = 1
        candidate_actions.append(tuple(action))

    def split_count(action: tuple[int, ...]) -> int:
        count = 0
        for packet in state_packets:
            bits = {action[index] for index in packet}
            if len(bits) > 1:
                count += 1
        return count

    min_split_count = min(split_count(action) for action in candidate_actions)
    minimally_splitting = [
        action for action in candidate_actions if split_count(action) == min_split_count
    ]

    def selected_score_sum(action: tuple[int, ...]) -> int:
        return sum(state[index] for index in range(k) if action[index] == 1)

    min_selected_sum = min(selected_score_sum(action) for action in minimally_splitting)
    chosen_actions = [
        action for action in minimally_splitting if selected_score_sum(action) == min_selected_sum
    ]
    return _balanced_from_base([(1.0, action) for action in chosen_actions])


def packet_regime5_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    if len(state) != 5:
        return packet_minimal_frontier_policy(state)

    state_packets = packets(state)
    packet_sizes = tuple(len(packet) for packet in state_packets)

    def actions_with_size_two(indices: tuple[int, ...]) -> list[tuple[int, ...]]:
        base_actions: list[tuple[int, ...]] = []
        for chosen in combinations(indices, 2):
            action = [0] * len(state)
            for index in chosen:
                action[index] = 1
            base_actions.append(tuple(action))
        return base_actions

    if packet_sizes == (5,):
        base_actions = actions_with_size_two(tuple(range(len(state))))
        return _balanced_from_base([(1.0, action) for action in base_actions])

    if packet_sizes == (1, 4):
        base_actions = actions_with_size_two(tuple(state_packets[1]))
        return _balanced_from_base([(1.0, action) for action in base_actions])

    if packet_sizes == (4, 1):
        tail_index = state_packets[1][0]
        base_actions = []
        for top_index in state_packets[0]:
            action = [0] * len(state)
            action[top_index] = 1
            action[tail_index] = 1
            base_actions.append(tuple(action))
        return _balanced_from_base([(1.0, action) for action in base_actions])

    if packet_sizes in {(3, 2), (2, 3)}:
        size_three_packet = next(packet for packet in state_packets if len(packet) == 3)
        base_actions = []
        for index in size_three_packet:
            action = [0] * len(state)
            action[index] = 1
            base_actions.append(tuple(action))
        return _balanced_from_base([(1.0, action) for action in base_actions])

    return packet_minimal_frontier_policy(state)
