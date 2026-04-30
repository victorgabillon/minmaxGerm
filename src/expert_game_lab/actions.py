from __future__ import annotations

from itertools import product


Action = tuple[int, ...]


def all_actions(k: int, include_trivial: bool = True) -> list[Action]:
    actions = [tuple(bits) for bits in product((0, 1), repeat=k)]
    if include_trivial:
        return actions
    return [action for action in actions if any(action) and not all(action)]


def complement(a: Action) -> Action:
    return tuple(1 - bit for bit in a)


def comb_action(k: int) -> Action:
    return tuple(1 if index % 2 == 0 else 0 for index in range(k))


def fixed_rank_action(k: int, ones: set[int]) -> Action:
    if any(rank < 1 or rank > k for rank in ones):
        raise ValueError("ranks must be 1-based and lie in [1, k]")
    return tuple(1 if (index + 1) in ones else 0 for index in range(k))
