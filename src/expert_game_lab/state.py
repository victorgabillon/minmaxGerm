from __future__ import annotations

from functools import lru_cache
from typing import Iterable


State = tuple[int, ...]


def canon(x: tuple[int, ...]) -> State:
    if not x:
        return ()
    sorted_x = tuple(sorted(x, reverse=True))
    minimum = sorted_x[-1]
    return tuple(value - minimum for value in sorted_x)


def packets(x: tuple[int, ...]) -> list[list[int]]:
    state = canon(x)
    if not state:
        return []

    result: list[list[int]] = []
    current = [0]
    for index in range(1, len(state)):
        if state[index] == state[index - 1]:
            current.append(index)
            continue
        result.append(current)
        current = [index]
    result.append(current)
    return result


def packet_type(x: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(len(packet) for packet in packets(x))


@lru_cache(maxsize=None)
def _all_states_cached(k: int, t: int) -> tuple[State, ...]:
    states: list[State] = []

    def rec(prefix: list[int], remaining: int) -> None:
        if len(prefix) == k - 1:
            prefix.append(0)
            states.append(tuple(prefix))
            prefix.pop()
            return
        for value in range(min(remaining, t), -1, -1):
            prefix.append(value)
            rec(prefix, value)
            prefix.pop()

    rec([], t)
    return tuple(states)


def all_states(k: int, t: int) -> Iterable[State]:
    if k <= 0:
        raise ValueError("k must be positive")
    if t < 0:
        raise ValueError("t must be nonnegative")
    return _all_states_cached(k, t)
