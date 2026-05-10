import pytest

from expert_game_lab.actions import complement
from expert_game_lab.lp_game import solve_adversary_dual, solve_minimax_step


def test_complement_is_involution() -> None:
    action = (1, 0, 1, 1)
    assert complement(complement(action)) == action


def test_lp_solves_symmetric_two_action_game() -> None:
    q_by_action = {
        (0, 0): 0.0,
        (1, 0): 1.0,
        (0, 1): 1.0,
        (1, 1): 2.0,
    }
    solution = solve_minimax_step(q_by_action, 2)
    assert solution.success
    assert solution.value == pytest.approx(1.0)
    worst_case = max(
        q_by_action[action] - sum(solution.p[index] * action[index] for index in range(2))
        for action in q_by_action
    )
    assert worst_case == pytest.approx(solution.value)


def test_lp_solves_single_coordinate_tradeoff() -> None:
    solution = solve_minimax_step(
        {
            (0, 0): 0.0,
            (1, 0): 1.0,
            (0, 1): 0.0,
            (1, 1): 1.0,
        },
        2,
    )
    assert solution.success
    assert solution.value == pytest.approx(0.0)
    assert solution.p == pytest.approx((1.0, 0.0), abs=1e-8)


def test_adversary_dual_matches_primal_small_game() -> None:
    q_by_action = {
        (0, 0): 0.0,
        (1, 0): 1.0,
        (0, 1): 1.0,
        (1, 1): 2.0,
    }

    primal = solve_minimax_step(q_by_action, 2)
    dual = solve_adversary_dual(q_by_action, 2)

    assert dual.success
    assert dual.value == pytest.approx(primal.value, abs=1e-8)
    assert sum(weight for _, weight in dual.weights_by_action) == pytest.approx(1.0)
    assert all(
        expected <= dual.max_expected_action + 1e-8
        for expected in dual.expected_action
    )


def test_adversary_dual_matches_single_coordinate_tradeoff() -> None:
    q_by_action = {
        (0, 0): 0.0,
        (1, 0): 1.0,
        (0, 1): 0.0,
        (1, 1): 1.0,
    }

    primal = solve_minimax_step(q_by_action, 2)
    dual = solve_adversary_dual(q_by_action, 2)

    assert dual.success
    assert dual.value == pytest.approx(primal.value, abs=1e-8)
    assert sum(weight for _, weight in dual.weights_by_action) == pytest.approx(1.0)
    assert dual.expected_action[0] <= dual.max_expected_action + 1e-8
    assert dual.expected_action[1] <= dual.max_expected_action + 1e-8
