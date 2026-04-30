from expert_game_lab.state import all_states, canon


def test_canon_is_permutation_invariant() -> None:
    assert canon((3, 1, 3, 0)) == canon((0, 3, 1, 3))


def test_canon_is_translation_invariant() -> None:
    assert canon((5, 2, 5, 4)) == canon((8, 5, 8, 7))


def test_all_states_contains_zero() -> None:
    states = set(all_states(4, 3))
    assert (0, 0, 0, 0) in states
