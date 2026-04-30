import pytest

from expert_game_lab.dp_policy import state_occupancy
from expert_game_lab.experiments import occupation_weighted_greedy_defects
from expert_game_lab.policies import comb_policy, packet_minimal_frontier_policy


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_state_occupancy_layers_sum_to_one(k: int, T: int, policy_fn) -> None:
    occupancy = state_occupancy(k, T, policy_fn)
    for layer in occupancy:
        assert sum(layer.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_defect_is_nonnegative(k: int, T: int, policy_fn) -> None:
    total_defect, _ = occupation_weighted_greedy_defects(k, T, policy_fn)
    assert total_defect >= -1e-9


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 3, comb_policy), (3, 4, packet_minimal_frontier_policy)])
def test_weighted_greedy_runs_on_tiny_instances(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    assert total_defect >= -1e-9
    assert contributions