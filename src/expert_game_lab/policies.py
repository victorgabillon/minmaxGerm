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


def _top_prefix_action(k: int, length: int) -> tuple[int, ...]:
    if k == 0:
        return ()
    if length < 0 or length > k - 1:
        raise ValueError("prefix edge length must lie in [0, k - 1]")

    bits = [1]
    for index in range(k - 1):
        if index < length:
            bits.append(1 - bits[-1])
        else:
            bits.append(bits[-1])
    return tuple(bits)


def _top_prefix_policy_with_length(x: tuple[int, ...], length: int) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]
    return _balanced_from_base([(1.0, _top_prefix_action(len(state), length))])


def top_prefix_shortest_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    return _top_prefix_policy_with_length(state, min(1, max(len(state) - 1, 0)))


def top_prefix_longest_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    return _top_prefix_policy_with_length(state, max(len(state) - 1, 0))


def top_prefix_chase_length_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    return _top_prefix_policy_with_length(state, min(3, max(len(state) - 1, 0)))


def _top_prefix_gap_sum_policy(x: tuple[int, ...], prefer_long: bool) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]
    if len(state) == 1:
        return _top_prefix_policy_with_length(state, 0)

    gaps = tuple(state[index] - state[index + 1] for index in range(len(state) - 1))
    prefix_sums = tuple(sum(gaps[:length]) for length in range(1, len(state)))
    min_sum = min(prefix_sums)
    candidate_lengths = [
        length
        for length, prefix_sum in enumerate(prefix_sums, start=1)
        if prefix_sum == min_sum
    ]
    length = max(candidate_lengths) if prefer_long else min(candidate_lengths)
    return _top_prefix_policy_with_length(state, length)


def top_prefix_gap_sum_short_policy(x: tuple[int, ...]) -> Policy:
    return _top_prefix_gap_sum_policy(x, prefer_long=False)


def top_prefix_gap_sum_long_policy(x: tuple[int, ...]) -> Policy:
    return _top_prefix_gap_sum_policy(x, prefer_long=True)


def top_prefix_tie_mimic_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    diagnostic_lengths = {
        5: 1,
        6: 3,
        7: 3,
    }
    fallback_length = min(3, max(len(state) - 1, 0))
    return _top_prefix_policy_with_length(
        state,
        min(diagnostic_lengths.get(len(state), fallback_length), max(len(state) - 1, 0)),
    )


def _largest_odd_top_prefix_length(k: int) -> int:
    max_length = max(k - 1, 0)
    if max_length % 2 == 1:
        return max_length
    return max(max_length - 1, 0)


def top_prefix_three_regime_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    k = len(state)
    if k <= 1:
        return _top_prefix_policy_with_length(state, 0)

    largest_odd = _largest_odd_top_prefix_length(k)
    gaps = tuple(state[index] - state[index + 1] for index in range(k - 1))
    state_packets = packets(state)

    if all(gap == 0 for gap in gaps):
        length = largest_odd
    elif gaps[0] >= 2:
        length = 1
    elif (
        len(state_packets[0]) >= (k + 1) // 2
        or sum(1 for packet in state_packets if len(packet) >= 2) >= 2
    ):
        length = largest_odd
    else:
        length = min(3, largest_odd)

    return _top_prefix_policy_with_length(state, length)


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


def packet_regime5b_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    if len(state) != 5:
        return packet_regime5_policy(state)

    state_packets = packets(state)
    packet_sizes = tuple(len(packet) for packet in state_packets)
    packet_values = tuple(state[packet[0]] for packet in state_packets)
    packet_gaps = tuple(
        packet_values[index] - packet_values[index + 1]
        for index in range(len(packet_values) - 1)
    )

    if packet_sizes == (1, 2, 2) and packet_gaps == (1, 1):
        top_index = state_packets[0][0]
        middle_packet = state_packets[1]
        base_actions: list[tuple[int, ...]] = []
        for middle_index in middle_packet:
            action = [0] * len(state)
            action[top_index] = 1
            action[middle_index] = 1
            base_actions.append(tuple(action))
        return _balanced_from_base([(1.0, action) for action in base_actions])

    return packet_regime5_policy(state)


def packet_regime5c_policy(x: tuple[int, ...]) -> Policy:
    state = canon(x)
    if not state:
        return [(1.0, ())]

    if len(state) != 5:
        return packet_regime5b_policy(state)

    state_packets = packets(state)
    packet_sizes = tuple(len(packet) for packet in state_packets)
    packet_values = tuple(state[packet[0]] for packet in state_packets)
    packet_gaps = tuple(
        packet_values[index] - packet_values[index + 1]
        for index in range(len(packet_values) - 1)
    )

    if packet_sizes in {(2, 3), (3, 2)} and packet_gaps == (1,):
        return _balanced_from_base([(1.0, (1, 0, 1, 0, 0))])

    return packet_regime5b_policy(state)
