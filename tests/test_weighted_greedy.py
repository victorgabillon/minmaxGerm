from fractions import Fraction

import pytest
import numpy as np

from expert_game_lab.dp_policy import state_occupancy
from expert_game_lab.state import canon, packet_type
from expert_game_lab.experiments import (
    K6OneRunCandidateFailureRow,
    MixedIntervalPatch,
    _block_gap_stats,
    _edge_signature,
    _edge_run_intervals,
    _gap_vector,
    _is_one_run,
    _alive_positions,
    _alive_prefix_length,
    _classify_one_run_alive_relation,
    _classify_k6_one_run_interval_relation,
    _balanced_policy_from_one_run_interval,
    _fixed_balanced_action_policy_from_text,
    _interval_from_action,
    _packet_barycenter_status,
    _barycenter_action_row_from_dual_row,
    _chase5_policy,
    _chase5_value_layers_reachable,
    _chase5_named_winner_probabilities,
    _chase5_certificate_learner,
    _one_run_interval_distribution_for_regime,
    _mixed_one_run_interval_policy,
    _k6_barycentric_trigger,
    _k6_barycentric_trigger_fires,
    _k6_barycentric_trigger_sweep_candidates,
    _k6_barycentric_weights,
    _phi_centered_max,
    _phi_centered_softmax,
    _softmax_centered_gradient,
    _k5_comb_support,
    _k5_chase_support,
    _balanced_action_canonical_key,
    _balanced_action_is_balanced,
    _balanced_chase_centered_q_for_support,
    _balanced_chase_old_fixed_q_for_support,
    _balanced_chase_q_value,
    _balanced_support_common_marginal,
    _fraction_support_is_balanced,
    _k5_balanced_action_symmetry_classes,
    _k5_chase_balanced_vertex,
    _k5_chase_orbit_vertices,
    _k5_comb_balanced_vertex,
    _k5_symmetrized_chase_support,
    _potential_greedy_choice,
    _random_playout_value_function,
    _random_playout_terminal_winner_probabilities,
    _k5_gap_vector,
    _next_state_distribution,
    _two_step_distribution,
    _expected_two_step_potential_value,
    _k5_boundary_feature_key,
    _k5_boundary_design_matrix,
    _apply_k5_boundary_correction,
    _k5_relaxation_local_certificate_learner,
    _k5_relaxation_value_functions,
    _k5_relaxation_natural_alpha,
    _k5_certificate_actions,
    _packet_average_reconstruction,
    _packet_weights_from_packet_uniform_p,
    _reconstruct_from_packet_weights,
    _rationalized_packet_learner,
    _next_state_from_packet_counts,
    _representative_action_from_packet_counts,
    _reduced_packet_lp,
    _gap_bucket,
    _k5_chase_reachable_states,
    _k5_worst_family_state,
    _k5_packet122_state,
    _parse_k5_q_audit_states,
    _rank_chase_future_centered_value,
    _rank_chase_future_uncentered_value,
    _min_normalized_descending,
    _k6_mixed_patch_candidates_from_failures,
    _matching_k6_mixed_patch,
    _parse_T_values,
    _k6_one_run_patch_list,
    _k6_greedy_patch_candidates_from_failures,
    _matching_k6_patch,
    _effective_alive_pattern,
    _edge_run_restricted_candidates,
    _exchangeable_actions_for_orbit_key,
    _time_policy_occupancy,
    _one_run_edge_action_library,
    _one_run_restricted_candidates,
    _lift_canonical_action_to_named_distribution,
    _lift_alive_pair_to_k,
    _lift_canonical_vector_to_named,
    _parse_cases,
    _prefix_one_run_action_library,
    _prefix_plus_tail_anchor_action_library,
    _terminal_named_winner_probabilities,
    _two_run_dual_support_replay_table_k9_T7,
    _local_edge_action_library,
    _top_prefix_length_from_policy,
    evaluate_top_prefix_oracle,
    filter_weighted_greedy_contributions,
    k3_motif_sweep,
    k9_motif_library_sweep,
    library_oracle,
    probability_matching_inspect_rows,
    probability_matching_named_inspect_rows,
    probability_matching_named_residual_aggregates,
    library_lp_restricted_optimal,
    library_lp_dual_inspect_rows,
    one_run_tie_analysis,
    one_run_restricted_optimal,
    edge_run_restricted_optimal,
    occupation_weighted_greedy_defects,
    print_one_run_restricted_optimal,
    print_edge_run_restricted_optimal,
    print_library_lp_restricted_optimal,
    print_library_lp_dual_inspect,
    print_library_lp_dual_orbit_completion,
    print_library_lp_dual_orbits,
    print_two_run_replay_template_report,
    print_two_run_replay_coarse_template_report,
    print_two_run_replay_rule_mining_report,
    print_two_run_successor_flow_report,
    print_two_run_flow_role_report,
    print_k6_one_run_alive_pattern_report,
    print_k6_alive_candidate_benchmark,
    print_k6_one_run_interval_rule_report,
    print_k6_one_run_patch_ladder_benchmark,
    print_k6_one_run_greedy_patch_search,
    print_k6_one_run_greedy_mixed_patch_search,
    print_k6_explicit_long_horizon_sweep,
    print_explicit_time_policy_reachable_benchmark,
    print_k6_adaptive_interval_anatomy_report,
    print_k6_barycenter_action_anatomy_report,
    print_k6_barycentric_candidate_benchmark,
    print_k6_barycentric_trigger_sweep,
    print_k5_centered_potential_comb_vs_chase_report,
    print_k5_softmax_potential_certificate_report,
    print_potential_greedy_interval_benchmark,
    print_random_playout_potential_benchmark,
    print_k5_gap_proof_diagnostics,
    print_k5_scalar_potential_exchangeability,
    print_k5_boundary_correction_fit_report,
    print_k5_relaxation_certificate_report,
    print_k5_repaired_learner_anatomy_report,
    print_k5_packet_learner_formula_report,
    print_k5_reduced_packet_lp_report,
    print_k5_reduced_packet_lp_coverage_report,
    print_k5_worst_family_scaling_report,
    print_chase5_verification_report,
    print_chase5_potential_certificate_report,
    chase5_verification_summary,
    chase5_potential_certificate_summary,
    k6_barycenter_action_rows,
    print_two_run_skeleton_placement_report,
    print_time_policy_boundary_loss,
    print_k3_motif_sweep,
    print_k9_motif_library_sweep,
    print_probability_matching_inspect,
    print_probability_matching_dual_face_repair,
    print_probability_matching_named_inspect,
    print_probability_matching_named_residuals,
    probability_matching_dual_face_repair_rows,
    explicit_policy_benchmark_rows,
    explicit_time_policy_benchmark_rows,
    evaluate_time_dependent_policy,
    evaluate_time_dependent_policy_reachable_root,
    make_two_run_replay_mask_policy,
    make_two_run_replay_cutoff_policy,
    print_explicit_policy_benchmark,
    print_explicit_time_policy_benchmark,
    print_replay_layer_mask_sweep,
    print_strategy_class_relative_benchmark,
    print_strategy_class_benchmark,
    strategy_class_relative_benchmark_rows,
    strategy_class_benchmark_rows,
    summarize_weighted_greedy_by_packet,
    summarize_weighted_greedy_by_regime,
    print_top_prefix_length_regimes,
    print_top_prefix_next_state_debt,
    print_top_prefix_policy_vs_oracle_labels,
    print_top_prefix_restricted_optimal,
    print_policy_occupancy_diff,
    print_top_prefix_candidate_values,
    print_top_prefix_scale_rows,
    print_top_prefix_valid_length_structure,
    top_prefix_oracle_labels,
    top_prefix_restricted_optimal,
    top_prefix_tie_analysis,
    time_policy_boundary_loss_rows,
    replay_layer_mask_sweep_rows,
    two_run_orbit_mixture_v1_policy,
    two_run_dual_support_replay_k9_T7_policy,
    two_run_skeleton_v3a_late_policy,
    two_run_skeleton_v3b_early_late_policy,
    two_run_replay_coarse_template_aggregates,
    two_run_replay_rule_mining_rows,
    two_run_successor_flow_rows,
    two_run_flow_role_rows,
    k6_one_run_alive_pattern_rows,
    k6_one_run_interval_rule_rows,
    make_k6_one_run_patch_policy,
    make_k6_one_run_patch_policy_from_patches,
    make_k6_one_run_mixed_patch_policy,
    make_k6_barycentric_trigger_sweep_policy,
    k6_one_run_candidate_failure_rows,
    k6_one_run_patch_ladder_benchmark_rows,
    k6_one_run_greedy_patch_search_steps,
    k6_one_run_greedy_mixed_patch_search_steps,
    k6_explicit_long_horizon_sweep_rows,
    k6_barycentric_candidate_benchmark_rows,
    k5_centered_potential_comb_vs_chase_rows,
    k5_softmax_potential_certificate_rows,
    potential_greedy_interval_benchmark_rows,
    potential_greedy_interval_policy,
    random_playout_value_rows,
    random_playout_greedy_rows,
    k5_gap_transition_rows,
    k5_large_gap_influence_cone_certificate_rows,
    k5_large_gap_influence_cone_proof_export_rows,
    k5_large_gap_influence_cone_source_certificate_rows,
    k5_large_gap_one_leader_symbolic_rows,
    k5_large_gap_multi_gap_potential_rows,
    k5_large_gap_barrier_audit_report,
    k5_large_gap_one_dimensional_barrier_rows,
    k5_large_gap_one_dimensional_failure_analysis_rows,
    k5_one_leader_influence_recursive_certificate_rows,
    k5_balanced_action_vertices,
    k5_balanced_centered_orbit_defect_rows,
    k5_balanced_chase_greedy_defect_rows,
    k5_balanced_chase_orbit_defect_rows,
    k5_balanced_mdp_gap_scaling_rows,
    k5_balanced_mdp_gap_scaling_summary,
    k5_balanced_mdp_value_gap_rows,
    k5_balanced_varying_c_obstruction_rows,
    k5_c_half_orbit_flat_equality_holds,
    k5_c_half_augmented_cone_closure_rows,
    k5_c_half_augmented_cone_proof_export,
    k5_c_half_augmented_cone_audit_report,
    k5_c_half_augmented_cone_small_basis_report_data,
    k5_c_half_schur_guard_cone_result,
    k5_c_half_invariant_inequality_closure_rows,
    k5_c_half_orbit_dual_certificate_rows,
    k5_c_half_orbit_optimality_certificate_rows,
    k5_c_half_orbit_optimality_violation_rows,
    k5_direct_route_finite_quotient_validation_result,
    k5_direct_route_large_gap_hitting_bound_rows,
    k5_direct_route_large_gap_influence_decay_rows,
    k5_direct_route_large_gap_uniform_influence_rows,
    k5_direct_route_large_gap_domination_rows,
    k5_full_vs_orbit_delta_rows,
    k5_full_vs_orbit_delta_supersolution_rows,
    k5_full_vs_orbit_reachable_delta_rows,
    k5_full_vs_orbit_reachable_delta_supersolution_rows,
    k5_full_vs_orbit_time_packet_barrier_growth_packet_rows,
    k5_full_vs_orbit_time_packet_barrier_growth_summary_rows,
    k5_full_vs_orbit_time_packet_barrier_residual_rows,
    k5_full_vs_orbit_time_packet_barrier_rows,
    k5_low_gap_boundary_and_variance_lemma_rows,
    k5_low_gap_a0_c1_mechanism_lemma_rows,
    k5_low_gap_a0_c1_human_proof_rows,
    k5_low_gap_a0_c1_symbolic_mechanism_proof_rows,
    k5_low_gap_correction_pattern_proof_rows,
    k5_low_gap_correction_potential_rows,
    k5_low_gap_corrected_residual_taxonomy_data,
    k5_low_gap_corrected_tail_lyapunov_certificate_data,
    k5_low_gap_exact_residual_recurrence_rows,
    k5_low_gap_bellman_image_recurrence_rows,
    k5_low_gap_flat_split_tail_proof_data,
    k5_low_gap_flat_split_tail_telescoping_chains,
    k5_low_gap_flat_split_long_tail_rows,
    k5_low_gap_flat_split_tail_operator_certificate,
    k5_low_gap_flat_split_tail_contraction_lemma_data,
    k5_low_gap_flat_split_schema_proof_rows,
    k5_low_gap_adjacent_merge_schema_proof_rows,
    k5_low_gap_packet_count_schema_proof_rows,
    k5_low_gap_finite_residual_schema_proof_rows,
    k5_low_gap_boundary_schema_proof_rows,
    k5_low_gap_higher_refinement_schema_proof_rows,
    k5_low_gap_curvature_gap_action_rows,
    k5_low_gap_ordered_face_geometry_rows,
    k5_low_gap_discrete_maximum_principle_components,
    k5_complement_balancing_symmetry_rows,
    k5_low_gap_cyclic_component_closure_data,
    k5_low_gap_boundary_compression_rows,
    k5_low_gap_boundary_atom_rows,
    k5_low_gap_packet_convexity_schema_rows,
    k5_low_gap_pattern_curvature_recurrence_rows,
    k5_low_gap_richer_tail_envelope_certificate_data,
    k5_low_gap_scalar_envelope_lyapunov_certificate_data,
    k5_low_gap_scalar_envelope_exact_proof_certificate_rows,
    k5_low_gap_scalar_envelope_recurrence_proof_rows,
    k5_low_gap_scalar_residual_envelope_rows,
    k5_low_gap_signed_cancellation_identity_rows,
    k5_low_gap_unified_residual_automaton_certificate_data,
    k5_orbit_finite_difference_recurrence_rows,
    k5_packet_barrier_increment_source_rows,
    k5_variable_c_harmonic_barrier_compression_rows,
    k5_variable_c_harmonic_barrier_proof_exports,
    k5_variable_c_harmonic_barrier_lp_results,
    k5_variable_c_harmonic_barrier_simple_candidate_rows,
    k5_variable_c_mechanism_domination_rows,
    print_k5_balanced_action_enumeration_report,
    print_k5_low_gap_bellman_image_recurrence_report,
    print_k5_multi_step_bellman_pattern_miner_report,
    print_k5_low_gap_flat_split_tail_proof_report,
    print_k5_low_gap_flat_split_tail_telescoping_report,
    print_k5_low_gap_flat_split_long_tail_limit_report,
    print_k5_low_gap_flat_split_tail_operator_certificate_report,
    print_k5_low_gap_flat_split_tail_contraction_lemma_report,
    print_k5_low_gap_flat_split_schema_proof_report,
    print_k5_low_gap_adjacent_merge_schema_proof_report,
    print_k5_low_gap_packet_count_schema_proof_report,
    print_k5_low_gap_finite_residual_schema_proof_report,
    print_k5_low_gap_boundary_schema_proof_report,
    print_k5_low_gap_higher_refinement_schema_proof_report,
    print_k5_low_gap_curvature_gap_action_grouping_report,
    print_k5_low_gap_ordered_face_geometry_report,
    print_k5_low_gap_discrete_maximum_principle_report,
    print_k5_complement_balancing_symmetry_report,
    print_k5_low_gap_cyclic_component_closure_report,
    print_k5_low_gap_boundary_compression_report,
    print_k5_low_gap_boundary_atom_taxonomy_report,
    print_k5_low_gap_boundary_atom_symbolic_lemmas_report,
    print_k5_low_gap_packet_count_gap_closure_report,
    print_k5_low_gap_refinement_packet_count_fan_report,
    print_k5_low_gap_maximum_principle_proof_outline_report,
    print_k5_low_gap_maximum_principle_replacement_section_report,
    print_k5_balanced_centered_orbit_defect_report,
    print_k5_balanced_chase_greedy_defect_report,
    print_k5_balanced_chase_orbit_defect_report,
    print_k5_balanced_mdp_gap_scaling_report,
    print_k5_balanced_mdp_value_gap_report,
    print_k5_balanced_varying_c_obstruction_report,
    print_k5_c_half_augmented_cone_closure_report,
    print_k5_c_half_augmented_cone_proof_export_report,
    print_k5_c_half_augmented_cone_audit_report,
    print_k5_c_half_augmented_cone_certificate_verify_report,
    print_k5_c_half_augmented_cone_small_basis_report,
    print_k5_c_half_schur_guard_cone_report,
    print_k5_c_half_invariant_cone_closure_report,
    print_k5_c_half_invariant_inequality_closure_report,
    print_k5_c_half_orbit_dual_certificate_report,
    print_k5_c_half_orbit_optimality_certificate_report,
    print_k5_direct_route_finite_quotient_validation_report,
    print_k5_direct_route_large_gap_hitting_bound_report,
    print_k5_direct_route_large_gap_influence_decay_report,
    print_k5_direct_route_large_gap_uniform_influence_report,
    print_k5_direct_route_large_gap_domination_report,
    print_k5_large_gap_influence_cone_certificate_report,
    print_k5_large_gap_influence_cone_proof_export_report,
    print_k5_large_gap_influence_cone_source_certificate_report,
    print_k5_large_gap_one_leader_symbolic_report,
    print_k5_large_gap_multi_gap_potential_report,
    print_k5_large_gap_barrier_audit_report,
    print_k5_large_gap_barrier_certificate_verify_report,
    print_k5_large_gap_barrier_merge_verify_report,
    print_k5_large_gap_one_dimensional_barrier_report,
    print_k5_large_gap_one_dimensional_failure_analysis_report,
    print_k5_one_leader_influence_recursive_certificate_report,
    print_k5_full_vs_orbit_delta_barrier_report,
    print_k5_full_vs_orbit_delta_supersolution_report,
    print_k5_full_vs_orbit_packet_barrier_growth_inequality_report,
    print_k5_full_vs_orbit_packet_barrier_recurrence_report,
    print_k5_full_vs_orbit_reachable_delta_barrier_report,
    print_k5_full_vs_orbit_time_packet_barrier_growth_report,
    print_k5_full_vs_orbit_time_packet_barrier_report,
    print_k5_low_gap_boundary_and_variance_lemmas_report,
    print_k5_low_gap_a0_c1_mechanism_lemma_report,
    print_k5_low_gap_a0_c1_human_proof_report,
    print_k5_low_gap_a0_c1_symbolic_mechanism_proofs_report,
    print_k5_low_gap_correction_pattern_proof_report,
    print_k5_low_gap_correction_potential_report,
    print_k5_low_gap_corrected_residual_taxonomy_report,
    print_k5_low_gap_corrected_tail_lyapunov_certificate_report,
    print_k5_low_gap_exact_residual_recurrence_report,
    print_k5_low_gap_curvature_analytic_proof_report,
    print_k5_low_gap_packet_convexity_schema_report,
    print_k5_low_gap_pattern_curvature_recurrence_report,
    print_k5_low_gap_richer_tail_envelope_certificate_report,
    print_k5_low_gap_scalar_envelope_lyapunov_certificate_report,
    print_k5_low_gap_scalar_envelope_lyapunov_certificate_verify_report,
    print_k5_low_gap_scalar_envelope_exact_proof_certificate_report,
    print_k5_low_gap_scalar_envelope_recurrence_proof_report,
    print_k5_low_gap_scalar_residual_envelope_proof_report,
    print_k5_low_gap_signed_cancellation_identity_report,
    print_k5_low_gap_lyapunov_certificate_debug_report,
    print_k5_low_gap_unified_residual_automaton_certificate_report,
    print_k5_orbit_finite_difference_recurrence_report,
    print_k5_packet_barrier_increment_source_report,
    print_k5_packet_barrier_mechanism_source_bound_report,
    print_k5_variable_c_harmonic_barrier_compression_report,
    print_k5_variable_c_harmonic_barrier_proof_export_report,
    print_k5_variable_c_harmonic_barrier_lp_report,
    print_k5_variable_c_harmonic_barrier_simple_candidates_report,
    print_k5_variable_c_mechanism_domination_report,
    _k5_balanced_mdp_full_solution,
    _k5_balanced_mdp_half_solution,
    _k5_balanced_mdp_orbit_solution,
    _k5_chase_orbit_vertices,
    k5_scalar_potential_exchangeability_rows,
    k5_boundary_correction_fit_rows,
    k5_relaxation_certificate_rows,
    k5_repaired_learner_anatomy_rows,
    k5_packet_learner_formula_rows,
    k5_reduced_packet_lp_rows,
    k5_reduced_packet_lp_coverage_rows,
    k5_worst_family_scaling_rows,
    k5_packet122_asymptotic_equation_rows,
    print_k5_packet122_asymptotic_equations_report,
    k5_packet122_q_asymptotic_rows,
    print_k5_packet122_q_asymptotic_report,
    print_k5_q_convention_audit,
    print_k5_normalized_scalar_convention_audit,
    k5_rank_adaptive_dp_recursion_audit_rows,
    print_k5_rank_adaptive_dp_recursion_audit,
    k5_rank_adaptive_curvature_rows,
    print_k5_rank_adaptive_curvature_report,
    k5_packet122_curvature_growth_summaries,
    k5_packet122_curvature_growth_run,
    print_k5_packet122_curvature_growth_report,
    k5_packet122_active_regime_rows,
    k5_packet122_active_regime_run,
    print_k5_packet122_active_regime_map,
    k5_packet122_lp_state_probe_rows,
    print_k5_packet122_lp_state_probe,
    k5_top3_transition_check_rows,
    k5_top3_reflected_chain_rows,
    print_k5_top3_reflected_chain_report,
    k5_top3_local_time_identity_rows,
    k5_top3_local_time_curvature_rows,
    print_k5_top3_local_time_identity_report,
    _top3_m_value,
    _top3_m_recurrence_value,
    _top3_k_from_m,
    _top3_k_recurrence_value,
    k5_top3_k_recurrence_rows,
    print_k5_top3_k_recurrence_report,
    _top3_candidate_stencils,
    _top3_expand_stencil,
    _top3_stencil_value,
    k5_top3_difference_family_summaries,
    print_k5_top3_difference_closure_report,
    _top3_half_hitting_sum,
    _top3_kernel_term,
    k5_top3_dense_endpoint_barrier_rows,
    k5_top3_endpoint_barrier_rows,
    k5_top3_endpoint_coupling_graph_rows,
    k5_top3_endpoint_recurrence_rows,
    k5_top3_ballot_candidate_rows,
    _top3_endpoint_graph_label_summary,
    k5_top3_hitting_identity_rows,
    k5_top3_hitting_kernel_rows,
    print_k5_top3_hitting_kernel_report,
    k5_top3_kernel_pairing_rows,
    print_k5_top3_kernel_pairing_report,
    k5_top3_telescoping_candidate_rows,
    k5_top3_parity_telescope_rows,
    print_k5_top3_endpoint_barrier_report,
    print_k5_top3_endpoint_coupling_report,
    print_k5_top3_discrepancy_chain_symbolic_report,
    _top3_a_term,
    _fraction_sequence_for_endpoint,
    k5_top3_discrepancy_generating_recurrence_rows,
    print_k5_top3_discrepancy_generating_function_report,
    _top3_signed_endpoint_value,
    k5_top3_discrepancy_barrier_certificate_rows,
    print_k5_top3_discrepancy_barrier_certificate_report,
    k5_top3_endpoint_induction_certificate_rows,
    print_k5_top3_endpoint_induction_certificate_report,
    _top3_refined_label_aggregate_value,
    _top3_refined_aggregate_direct_value,
    k5_top3_refined_discrepancy_label_scan_rows,
    print_k5_top3_refined_discrepancy_barrier_report,
    _top3_cone_form_specs,
    _top3_cone_form_value,
    k5_top3_endpoint_cone_form_scan_rows,
    print_k5_top3_endpoint_cone_certificate_report,
    k5_top3_endpoint_cone_expansion_rows,
    print_k5_top3_endpoint_cone_expansion_report,
    k5_top3_kplus_companion_scan_rows,
    print_k5_top3_kplus_companion_barrier_report,
    _top3_endpoint_exact_image_rows,
    print_k5_top3_endpoint_exact_image_report,
    _top3_diagonal_discrepancy_value,
    _top3_diagonal_reflected_boundary_child_value,
    _top3_diagonal_e1_representation_value,
    _top3_diagonal_e1_reflected_boundary_contribution,
    _top3_diagonal_k_image_e1_residual,
    k5_top3_diagonal_discrepancy_recurrence_rows,
    _top3_reflected_source_gaussian_envelope_value,
    _top3_reflected_source_grid,
    k5_top3_reflected_convolution_analytic_scan_rows,
    k5_top3_reflected_convolution_gaussian_scan_rows,
    k5_top3_reflected_gradient_convolution_scan_rows,
    k5_top3_reflected_boundary_convolution_residual_rows,
    k5_top3_reflected_boundary_convolution_scan_rows,
    k5_top3_reflected_source_gaussian_envelope_rows,
    print_k5_top3_diagonal_discrepancy_hierarchy_report,
    print_k5_top3_diagonal_heat_kernel_report,
    print_k5_top3_reflected_gradient_convolution_report,
    print_k5_top3_reflected_convolution_analytic_bound_report,
    print_k5_top3_reflected_source_gaussian_report,
    print_k5_top3_reflected_boundary_convolution_report,
    k5_top3_endpoint_polytope_closure_rows,
    print_k5_top3_endpoint_polytope_barrier_report,
    print_k5_top3_telescoping_identity_report,
    explicit_time_policy_reachable_benchmark_rows,
    k6_alive_trunc_comb_policy,
    k6_alive_trunc_comb_twin3_policy,
    k6_alive_full_comb_control_policy,
    k6_one_run_rule_v1_policy,
    k6_one_run_rule_v1_no_mixed_policy,
    k6_one_run_rule_v1_aggressive_shoulder_policy,
    k6_greedy_deterministic_patched_policy,
    k6_barycentric_v1_equal_policy,
    k6_barycentric_v1_late_policy,
    k6_barycentric_v1_middle_policy,
    k6_barycentric_v1_tail_heavy_policy,
    k6_barycentric_v1_top_tail_policy,
    two_run_skeleton_placement_aggregates,
    two_run_replay_template_rows,
    weighted_top_prefix_oracle_labels,
)
from expert_game_lab.policies import comb_policy, packet_minimal_frontier_policy


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ((1, 0, 1, 0, 0), (1, 1, 1, 0)),
        ((1, 0, 1, 0, 1), (1, 1, 1, 1)),
    ],
)
def test_edge_signature(action: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert _edge_signature(action) == expected


def test_edge_run_intervals_and_one_run_predicate() -> None:
    assert _edge_run_intervals((0, 1, 1, 0, 1)) == [(1, 2), (4, 4)]
    assert _edge_run_intervals((1, 1, 1, 0)) == [(0, 2)]
    assert _is_one_run((1, 1, 1, 0))
    assert not _is_one_run((0, 1, 1, 0, 1))


def test_block_gap_stats_for_one_run_signature() -> None:
    stats = _block_gap_stats((5, 4, 4, 2, 0), (1, 1, 0, 0))

    assert _gap_vector((5, 4, 4, 2, 0)) == (1, 0, 2, 2)
    assert stats is not None
    assert stats.interval == (0, 1)
    assert stats.sum_gap == 1
    assert stats.max_gap == 1
    assert stats.length == 2
    assert _block_gap_stats((5, 4, 4, 2, 0), (1, 0, 1, 0)) is None


def test_local_edge_action_library_contains_expected_motifs() -> None:
    library5 = set(_local_edge_action_library(5))
    library6 = set(_local_edge_action_library(6))

    assert (1, 0, 1, 0, 0) in library5
    assert (1, 0, 1, 0, 1) in library5
    assert (1, 0, 1, 0, 0, 1) in library6


def test_strict_action_libraries_have_expected_membership() -> None:
    one_run6 = set(_one_run_edge_action_library(6))
    prefix6 = set(_prefix_one_run_action_library(6))
    prefix_tail6 = set(_prefix_plus_tail_anchor_action_library(6))

    assert (1, 0, 1, 0, 0, 0) in one_run6
    assert (1, 0, 1, 0, 0, 1) not in one_run6
    assert (1, 0, 1, 0, 0, 0) in prefix6
    assert (0, 0, 1, 0, 1, 0) not in prefix6
    assert (1, 0, 1, 0, 0, 1) in prefix_tail6
    assert (1, 0, 1, 0, 1, 0) not in prefix_tail6


def test_one_run_restricted_candidates_include_prefix_and_centered_intervals() -> None:
    candidates = _one_run_restricted_candidates(5)
    by_interval = {candidate.intervals[0]: candidate for candidate in candidates}

    assert (0, 2) in by_interval
    assert by_interval[(0, 2)].edge_signature == (1, 1, 1, 0)
    assert (1, 2) in by_interval
    assert by_interval[(1, 2)].edge_signature == (0, 1, 1, 0)


def test_two_run_restricted_candidates_include_one_run_and_two_run_patterns() -> None:
    candidates = _edge_run_restricted_candidates(9, 2)
    signatures = {candidate.edge_signature for candidate in candidates}

    assert (1, 1, 1, 0, 0, 0, 0, 0) in signatures
    assert (1, 0, 1, 1, 0, 0, 0, 0) in signatures


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


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 3, comb_policy), (3, 4, packet_minimal_frontier_policy)])
def test_policy_edge_signature_probabilities_sum_to_one(k: int, T: int, policy_fn) -> None:
    _, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    for item in contributions:
        assert sum(probability for probability, _ in item.policy_edge_signatures) == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_packet_summaries_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    summaries = summarize_weighted_greedy_by_packet(contributions, n=10)

    assert sum(summary.total_contribution for summary in summaries) == pytest.approx(total_defect)
    assert sum(summary.occupancy_mass for summary in summaries) == pytest.approx(float(T))
    assert summaries


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_regime_summaries_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    summaries = summarize_weighted_greedy_by_regime(contributions, n=10)

    assert sum(summary.total_contribution for summary in summaries) == pytest.approx(total_defect)
    assert sum(summary.occupancy_mass for summary in summaries) == pytest.approx(float(T))
    assert summaries


@pytest.mark.parametrize(("k", "T", "policy_fn"), [(2, 4, comb_policy), (3, 5, packet_minimal_frontier_policy)])
def test_weighted_greedy_signature_groupings_preserve_totals(k: int, T: int, policy_fn) -> None:
    total_defect, contributions = occupation_weighted_greedy_defects(k, T, policy_fn)
    best_signature_totals: dict[tuple[int, ...], float] = {}
    policy_signature_totals: dict[tuple[tuple[float, tuple[int, ...]], ...], float] = {}
    for item in contributions:
        best_signature_totals[item.best_edge_signature] = (
            best_signature_totals.get(item.best_edge_signature, 0.0) + item.contribution
        )
        policy_signature_totals[item.policy_edge_signatures] = (
            policy_signature_totals.get(item.policy_edge_signatures, 0.0) + item.contribution
        )

    assert sum(best_signature_totals.values()) == pytest.approx(total_defect)
    assert sum(policy_signature_totals.values()) == pytest.approx(total_defect)


def test_all_action_library_oracle_has_zero_restriction_loss() -> None:
    _, _, total_loss, contributions = library_oracle(3, 4, "all", comb_policy)

    assert total_loss == pytest.approx(0.0)
    assert sum(item.loss for item in contributions) == pytest.approx(0.0)


def test_one_run_tie_analysis_finds_optimal_one_run_rows() -> None:
    rows, total_occupancy, total_value, one_run_occupancy, one_run_value = one_run_tie_analysis(
        3,
        4,
        comb_policy,
    )

    assert rows
    assert total_occupancy > 0.0
    assert total_value > 0.0
    assert 0.0 < one_run_occupancy <= total_occupancy
    assert 0.0 < one_run_value <= total_value
    assert all(row.optimal_one_run_count >= 1 for row in rows)


def test_top_prefix_tie_analysis_finds_optimal_lengths() -> None:
    rows, total_occupancy, total_value, prefix_occupancy, prefix_value = top_prefix_tie_analysis(
        3,
        4,
        comb_policy,
    )

    assert rows
    assert total_occupancy > 0.0
    assert total_value > 0.0
    assert 0.0 < prefix_occupancy <= total_occupancy
    assert 0.0 < prefix_value <= total_value
    for row in rows:
        assert row.optimal_lengths
        assert row.min_optimal_length == min(row.optimal_lengths)
        assert row.max_optimal_length == max(row.optimal_lengths)


def test_top_prefix_oracle_eval_runs_for_all_selectors() -> None:
    for selector in ("min_valid", "max_valid", "median_valid", "chase_preferred"):
        result = evaluate_top_prefix_oracle(3, 4, selector)

        assert result.selector == selector
        assert result.value <= result.optimal_value + 1e-9
        assert result.gap >= -1e-9
        assert result.selected_length_counts


def test_top_prefix_oracle_labels_runs() -> None:
    rows = top_prefix_oracle_labels(3, 4, "min_valid")

    assert rows
    assert all(row.valid_lengths for row in rows)
    assert all(row.selected_length in row.valid_lengths for row in rows)


def test_weighted_top_prefix_oracle_labels_runs_for_all_selectors() -> None:
    for selector in ("min_valid", "max_valid", "median_valid", "chase_preferred"):
        rows = weighted_top_prefix_oracle_labels(3, 4, selector)

        assert rows
        assert sum(row.occupancy_probability for row in rows) == pytest.approx(4.0)
        assert all(row.valid_lengths for row in rows)
        assert all(row.selected_length in row.valid_lengths for row in rows)


def test_top_prefix_length_from_policy_infers_support_length() -> None:
    policy = [(0.5, (1, 0, 1, 0, 0)), (0.5, (0, 1, 0, 1, 1))]

    assert _top_prefix_length_from_policy(policy) == 3


def test_top_prefix_length_regimes_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_length_regimes(3, 4, "min_valid", n=2)

    captured = capsys.readouterr()
    assert "Selected L=1" in captured.out


def test_top_prefix_policy_vs_oracle_labels_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_policy_vs_oracle_labels(3, 4, "comb", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "invalid occupancy" in captured.out


def test_top_prefix_valid_length_structure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_valid_length_structure(3, 4, "min_valid", occupancy_policy_name="comb", n=2)

    captured = capsys.readouterr()
    assert "Valid parity histogram by value" in captured.out


def test_top_prefix_scale_rows_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_scale_rows(3, 4, "min_valid", occupancy_policy_name="comb", n=2)

    captured = capsys.readouterr()
    assert "near-global L" in captured.out


def test_policy_occupancy_diff_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_policy_occupancy_diff(3, 4, "comb", "top_prefix_shortest", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "total positive occupancy shift" in captured.out


def test_top_prefix_next_state_debt_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_next_state_debt(3, 4, "comb", "min_valid", n=2)

    captured = capsys.readouterr()
    assert "Candidate L cleanup debt" in captured.out


def test_top_prefix_candidate_values_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_candidate_values(3, 4, "comb", length_set="all", n=2)

    captured = capsys.readouterr()
    assert "total weighted regret" in captured.out


def test_top_prefix_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_top_prefix_restricted_optimal(3, 4, "all", n=2)

    captured = capsys.readouterr()
    assert "V_restricted" in captured.out


def test_one_run_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_one_run_restricted_optimal(3, 4, n=2)

    captured = capsys.readouterr()
    assert "library size" in captured.out


def test_two_run_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_edge_run_restricted_optimal(3, 4, 2, n=2)

    captured = capsys.readouterr()
    assert "library size" in captured.out


def test_one_run_restricted_value_dominates_top_prefix_restricted() -> None:
    one_run_result = one_run_restricted_optimal(3, 4)
    top_prefix_result = top_prefix_restricted_optimal(3, 4, "all")

    assert one_run_result.value >= top_prefix_result.value - 1e-9


def test_two_run_restricted_value_dominates_one_run_restricted() -> None:
    two_run_result = edge_run_restricted_optimal(3, 4, 2)
    one_run_result = one_run_restricted_optimal(3, 4)

    assert two_run_result.value >= one_run_result.value - 1e-9


def test_library_lp_all_matches_optimal_on_small_instance() -> None:
    result = library_lp_restricted_optimal(3, 4, "all")

    assert result.gap == pytest.approx(0.0)


def test_library_lp_monotone_for_run_libraries() -> None:
    one_run_result = library_lp_restricted_optimal(3, 4, "one_run")
    two_run_result = library_lp_restricted_optimal(3, 4, "two_run")

    assert two_run_result.value >= one_run_result.value - 1e-9


def test_library_lp_one_run_dominates_deterministic_one_run_pair() -> None:
    lp_result = library_lp_restricted_optimal(3, 4, "one_run")
    deterministic_result = one_run_restricted_optimal(3, 4)

    assert lp_result.value >= deterministic_result.value - 1e-9


def test_library_lp_restricted_optimal_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_restricted_optimal(3, 4, "one_run", n=2)

    captured = capsys.readouterr()
    assert "Library LP restricted optimal" in captured.out


def test_library_lp_dual_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_inspect(3, 4, "two_run", support_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual inspect" in captured.out
    assert "adversary dual support" in captured.out


def test_library_lp_dual_orbits_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_orbits(3, 4, "two_run", orbit_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual support orbits" in captured.out
    assert "support orbits" in captured.out


def test_library_lp_dual_orbit_completion_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_library_lp_dual_orbit_completion(3, 4, "two_run", orbit_n=2, n=2)

    captured = capsys.readouterr()
    assert "Library LP dual orbit completion" in captured.out
    assert "library_compatible" in captured.out


def test_k3_motif_sweep_runs() -> None:
    rows = k3_motif_sweep((4, 5))

    assert rows
    assert {row.library_name for row in rows} >= {
        "comb",
        "twin_comb3",
        "tail_singleton",
        "comb_twin_comb3",
        "all_three",
    }
    assert all(row.lp_value <= row.optimal_value + 1e-9 for row in rows)


def test_k3_motif_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k3_motif_sweep((4,))

    captured = capsys.readouterr()
    assert "k=3 motif sweep" in captured.out
    assert "comb_twin_comb3" in captured.out


def test_k9_motif_library_sweep_runs() -> None:
    rows = k9_motif_library_sweep((1,))

    assert rows
    assert {row.family_name for row in rows} >= {"top_prefix_all", "one_run", "two_run"}


def test_k9_motif_library_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k9_motif_library_sweep((1,), n=2)

    captured = capsys.readouterr()
    assert "k=9 motif-library sweep" in captured.out
    assert "top_prefix_all" in captured.out


def test_probability_matching_inspect_rows_have_winner_probabilities() -> None:
    _, rows = probability_matching_inspect_rows(3, 4, "all_three")

    assert rows
    for row in rows:
        assert sum(row.winner_probabilities) == pytest.approx(1.0)
        assert row.l1_error >= 0.0
        assert row.linf_error >= 0.0


def test_probability_matching_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_inspect(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Probability matching inspect" in captured.out
    assert "weighted L1 error" in captured.out


def test_lift_canonical_vector_to_named_averages_ties() -> None:
    lifted = _lift_canonical_vector_to_named((5, 3, 3, 1), (0.6, 0.2, 0.1, 0.1))

    assert lifted == pytest.approx((0.6, 0.15, 0.15, 0.1))


def test_exchangeable_action_lift_preserves_mass_and_packet_counts() -> None:
    lifted = _lift_canonical_action_to_named_distribution((5, 3, 3, 1), (1, 1, 0, 0))

    assert sum(probability for probability, _ in lifted) == pytest.approx(1.0)
    assert dict((action, probability) for probability, action in lifted) == {
        (1, 0, 1, 0): pytest.approx(0.5),
        (1, 1, 0, 0): pytest.approx(0.5),
    }
    for _, action in lifted:
        assert (action[0], action[1] + action[2], action[3]) == (1, 1, 0)


def test_terminal_named_winner_probabilities_sum_to_one() -> None:
    winners = _terminal_named_winner_probabilities((2, 3, 3, 1))

    assert winners == pytest.approx((0.0, 0.5, 0.5, 0.0))
    assert sum(winners) == pytest.approx(1.0)


def test_probability_matching_named_inspect_rows_run_and_keep_named_root() -> None:
    _, rows = probability_matching_named_inspect_rows(3, 4, "all_three")

    assert rows
    for row in rows:
        assert sum(row.winner_probabilities) == pytest.approx(1.0)
        assert row.l1_error >= 0.0
        assert row.linf_error >= 0.0

    root_row = next(row for row in rows if row.time == 0 and row.named_state == (0, 0, 0))
    assert root_row.winner_probabilities != pytest.approx((1.0, 0.0, 0.0))
    assert root_row.winner_probabilities == pytest.approx((1.0 / 3.0,) * 3)


def test_probability_matching_named_inspect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_named_inspect(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Named probability matching inspect" in captured.out
    assert "weighted L1 error" in captured.out


def test_probability_matching_named_residuals_are_zero_for_k3_all_three() -> None:
    rows, canonical_aggregates, regime_aggregates = probability_matching_named_residual_aggregates(
        3,
        4,
        "all_three",
    )

    assert rows
    assert canonical_aggregates
    assert regime_aggregates
    assert sum(row.weighted_l1_error for row in rows) == pytest.approx(0.0)
    assert sum(row.weighted_linf_error for row in rows) == pytest.approx(0.0)
    assert sum(item.weighted_linf_error for item in canonical_aggregates) == pytest.approx(0.0)


def test_probability_matching_named_residuals_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_named_residuals(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Named probability matching residuals" in captured.out
    assert "Top canonical-state residual aggregates" in captured.out


def test_probability_matching_dual_face_repair_keeps_k3_zero_residual() -> None:
    _, repair_rows = probability_matching_dual_face_repair_rows(3, 4, "all_three", n=5)

    assert repair_rows
    assert all(row.success for row in repair_rows)
    assert sum(row.original_weighted_l1_error for row in repair_rows) == pytest.approx(0.0)
    assert sum(row.repaired_weighted_l1_error for row in repair_rows) == pytest.approx(0.0, abs=1e-7)
    assert sum(row.repaired_weighted_linf_error for row in repair_rows) <= (
        sum(row.original_weighted_linf_error for row in repair_rows) + 1e-7
    )


def test_probability_matching_dual_face_repair_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_probability_matching_dual_face_repair(3, 4, "all_three", n=2)

    captured = capsys.readouterr()
    assert "Probability matching dual-face repair" in captured.out
    assert "Top repaired rows" in captured.out


def test_parse_cases() -> None:
    assert _parse_cases("3:20, 9:7") == ((3, 20), (9, 7))


def test_strategy_class_benchmark_rows_run_for_tiny_case() -> None:
    rows, skipped = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
    )

    assert not skipped
    assert {row.library_name for row in rows} == {"top_prefix_all", "one_run", "two_run"}
    assert all(row.library_size > 0 for row in rows)
    assert all(row.value <= row.optimal_value + 1e-9 for row in rows)


def test_strategy_class_benchmark_no_active_counts_preserves_values() -> None:
    rows_with_counts, skipped_with_counts = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run"),
    )
    rows_without_counts, skipped_without_counts = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run"),
        collect_active_counts=False,
    )

    assert not skipped_with_counts
    assert not skipped_without_counts
    by_library_with_counts = {row.library_name: row for row in rows_with_counts}
    by_library_without_counts = {row.library_name: row for row in rows_without_counts}
    assert by_library_with_counts.keys() == by_library_without_counts.keys()
    for library_name, row in by_library_with_counts.items():
        cheap_row = by_library_without_counts[library_name]
        assert cheap_row.value == pytest.approx(row.value)
        assert cheap_row.gap == pytest.approx(row.gap)
        assert cheap_row.active_action_count == -1
        assert cheap_row.active_edge_signature_count == -1


def test_strategy_class_relative_benchmark_matches_exact_values() -> None:
    exact_rows, exact_skipped = strategy_class_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        collect_active_counts=False,
    )
    relative_rows, relative_skipped = strategy_class_relative_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        reference_library_name="two_run",
    )

    assert not exact_skipped
    assert not relative_skipped
    exact_by_library = {row.library_name: row for row in exact_rows}
    relative_by_library = {row.library_name: row for row in relative_rows}
    reference_value = exact_by_library["two_run"].value
    assert relative_by_library["two_run"].gap_to_reference == pytest.approx(0.0)
    for library_name, row in relative_by_library.items():
        assert row.value == pytest.approx(exact_by_library[library_name].value)
        assert row.reference_value == pytest.approx(reference_value)
        assert row.gap_to_reference == pytest.approx(reference_value - row.value)


def test_strategy_class_relative_benchmark_reference_can_be_implicit() -> None:
    rows, skipped = strategy_class_relative_benchmark_rows(
        ((3, 4),),
        ("top_prefix_all",),
        reference_library_name="two_run",
    )

    assert not skipped
    assert {row.library_name for row in rows} == {"top_prefix_all"}
    assert rows[0].reference_library_name == "two_run"


def test_strategy_class_benchmark_all_library_is_valid_beyond_k3() -> None:
    rows, skipped = strategy_class_benchmark_rows(((4, 3),), ("all", "all_three"))

    assert any(row.library_name == "all" for row in rows)
    assert any("all_three is only valid for k=3" in message for message in skipped)


def test_strategy_class_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_strategy_class_benchmark(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        include_probability_matching=True,
        probability_matching_max_k=3,
        probability_matching_max_T=4,
        n=2,
    )

    captured = capsys.readouterr()
    assert "Strategy class benchmark" in captured.out
    assert "gap/sqrt(T)" in captured.out
    assert "pm_avg_linf" in captured.out


def test_strategy_class_relative_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_strategy_class_relative_benchmark(
        ((3, 4),),
        ("top_prefix_all", "one_run", "two_run"),
        reference_library_name="two_run",
        n=2,
    )

    captured = capsys.readouterr()
    assert "Strategy class relative benchmark" in captured.out
    assert "gap_to_reference/sqrt(T)" in captured.out


def test_exchangeable_actions_for_orbit_key_respects_packet_counts() -> None:
    policy = _exchangeable_actions_for_orbit_key(
        (1, 1, 1, 1, 0, 0, 0, 0, 0),
        (2, 1),
        max_runs=2,
    )

    assert policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    for _, action in policy:
        assert sum(action[:4]) == 2
        assert sum(action[4:]) == 1
        assert sum(action) == 3
        assert action in {candidate for _, candidate in policy}


def test_two_run_orbit_mixture_v1_policy_returns_distribution() -> None:
    policy = two_run_orbit_mixture_v1_policy((0,) * 9)

    assert policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    assert all(len(action) == 9 for _, action in policy)


def test_explicit_policy_benchmark_rows_run_without_reference_lp() -> None:
    rows, skipped = explicit_policy_benchmark_rows(
        ((9, 2),),
        ("two_run_orbit_mixture_v1",),
        include_reference=False,
    )

    assert not skipped
    assert len(rows) == 1
    assert rows[0].policy_name == "two_run_orbit_mixture_v1"
    assert rows[0].reference_value is None


def test_explicit_policy_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_explicit_policy_benchmark(
        ((9, 2),),
        ("two_run_orbit_mixture_v1",),
        include_reference=False,
    )

    captured = capsys.readouterr()
    assert "Explicit policy benchmark" in captured.out
    assert "two_run_orbit_mixture_v1" in captured.out


def test_two_run_dual_support_replay_table_contains_root() -> None:
    table = _two_run_dual_support_replay_table_k9_T7()
    root_policy = table[(7, (0,) * 9)]

    assert root_policy
    assert sum(probability for probability, _ in root_policy) == pytest.approx(1.0)


def test_two_run_dual_support_replay_policy_returns_distribution() -> None:
    policy = two_run_dual_support_replay_k9_T7_policy((0,) * 9, 7)

    assert policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    assert all(len(action) == 9 for _, action in policy)


def test_two_run_replay_template_rows_are_nonempty_and_include_root() -> None:
    rows = two_run_replay_template_rows(9, 7)

    assert rows
    root_rows = [row for row in rows if row.remaining_horizon == 7 and row.state == (0,) * 9]
    assert root_rows
    for row in rows[:10]:
        assert sum(weight for weight, *_ in row.support) == pytest.approx(1.0)


def test_two_run_replay_template_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_replay_template_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run replay template report" in captured.out
    assert "total distinct support templates" in captured.out


def test_two_run_replay_coarse_template_aggregates_are_nonempty() -> None:
    aggregates = two_run_replay_coarse_template_aggregates(9, 7)

    assert aggregates
    assert set(aggregates) == {
        "edge_weight_skeleton",
        "action_weight_skeleton",
        "weight_multiset_skeleton",
        "orbit_weight_shape",
        "successor_shape",
    }
    assert all(aggregates[family] for family in aggregates)


def test_two_run_replay_coarse_template_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_replay_coarse_template_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run replay coarse template report" in captured.out
    assert "edge_weight_skeleton" in captured.out


def test_two_run_skeleton_placement_aggregates_are_nonempty() -> None:
    aggregates = two_run_skeleton_placement_aggregates(9, 7)

    assert aggregates["action_weight_skeleton"]
    assert aggregates["edge_weight_skeleton"]
    root = [
        aggregate
        for aggregate in aggregates["action_weight_skeleton"]
        if aggregate.representative.state == (0,) * 9
    ]
    assert root
    recurring = [
        aggregate
        for aggregate in aggregates["edge_weight_skeleton"]
        if aggregate.row_count >= 20
    ]
    assert recurring


def test_two_run_skeleton_placement_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_skeleton_placement_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run skeleton placement report" in captured.out
    assert "action geometry" in captured.out
    assert "edge=" in captured.out


def test_two_run_replay_rule_mining_rows_are_nonempty_and_include_root() -> None:
    rows = two_run_replay_rule_mining_rows(9, 7)

    assert rows
    root_rows = [row for row in rows if row.remaining_horizon == 7 and row.state == (0,) * 9]
    assert root_rows
    root = root_rows[0]
    assert root.relative_placement_signature
    assert root.weight_multiset


def test_two_run_replay_rule_mining_exposure_signature_packet_internal_invariance() -> None:
    rows = two_run_replay_rule_mining_rows(9, 7)
    row = next(item for item in rows if item.remaining_horizon == 7 and item.state == (0,) * 9)
    exposure_vectors = {feature.exposure_vector for feature in row.action_features}

    assert exposure_vectors == {("4/9",)}


def test_two_run_replay_rule_mining_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_replay_rule_mining_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run replay rule-mining report" in captured.out
    assert "distinct relative placement signatures" in captured.out
    assert "Groups by weight_multiset + relative placement signature" in captured.out


def test_two_run_successor_flow_rows_include_expected_early_flow() -> None:
    rows = two_run_successor_flow_rows(9, 7)

    assert rows
    root = next(row for row in rows if row.source.remaining_horizon == 7 and row.source.state == (0,) * 9)
    assert root.successor_distribution == (((4, 5), "1"),)

    first_split = next(
        row
        for row in rows
        if row.source.packet_type == (4, 5) and row.source.packet_gaps == (1,)
    )
    assert ((2, 3, 4), "11/17") in first_split.successor_distribution
    assert ((1, 7, 1), "6/17") in first_split.successor_distribution


def test_two_run_successor_flow_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_successor_flow_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run successor-flow report" in captured.out
    assert "distinct successor-flow signatures" in captured.out
    assert "Early basin flow" in captured.out


def test_two_run_flow_role_rows_compress_successors_and_include_root() -> None:
    successor_rows = two_run_successor_flow_rows(9, 7)
    role_rows = two_run_flow_role_rows(9, 7)

    assert role_rows
    exact_successors = {row.exact_successor_distribution for row in role_rows}
    role_distributions = {row.role_distribution for row in role_rows}
    assert len(role_distributions) < len(exact_successors)

    root = next(row for row in role_rows if row.source.remaining_horizon == 7 and row.source.state == (0,) * 9)
    assert len(root.role_distribution) == 1
    assert root.role_distribution[0][1] == "1"
    assert len(successor_rows) == len(role_rows)


def test_two_run_flow_role_first_split_has_nontrivial_roles() -> None:
    rows = two_run_flow_role_rows(9, 7)
    first_split = next(
        row
        for row in rows
        if row.source.packet_type == (4, 5) and row.source.packet_gaps == (1,)
    )

    assert len(first_split.role_distribution) >= 2


def test_two_run_flow_role_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_two_run_flow_role_report(9, 7, n=2)

    captured = capsys.readouterr()
    assert "Two-run flow-role report" in captured.out
    assert "distinct role distributions" in captured.out
    assert "Early basin role chain" in captured.out


def test_alive_positions_for_simple_state() -> None:
    assert _alive_positions((5, 4, 3, 1, 0, 0), 2) == (0, 1, 2)
    assert _alive_positions((5, 4, 3, 1, 0, 0), 4) == (0, 1, 2, 3)


def test_alive_prefix_length_modes() -> None:
    assert _alive_prefix_length((3, 2, 2, 0, 0, 0), 2) == 3
    assert _alive_prefix_length((3, 2, 2, 1, 0, 0), 2, death_mode="weak") == 4
    assert _alive_prefix_length((3, 2, 2, 1, 0, 0), 2, death_mode="strict") == 3


def test_lift_alive_pair_keeps_dead_tail_zero() -> None:
    policy = _lift_alive_pair_to_k(6, 5, (1, 0, 1, 0, 0))

    assert policy == ((0.5, (0, 1, 0, 1, 1, 0)), (0.5, (1, 0, 1, 0, 0, 0)))
    assert all(action[-1] == 0 for _, action in policy)


def test_k6_alive_candidate_policy_root_and_dead_tail() -> None:
    root_policy = k6_alive_trunc_comb_policy((0, 0, 0, 0, 0, 0), 12)
    dead_tail_policy = k6_alive_trunc_comb_policy((3, 2, 2, 2, 2, 0), 2)

    assert {action for _, action in root_policy} == {
        (1, 0, 1, 0, 1, 1),
        (0, 1, 0, 1, 0, 0),
    }
    assert {action for _, action in dead_tail_policy} == {
        (1, 0, 1, 0, 0, 0),
        (0, 1, 0, 1, 1, 0),
    }
    assert all(action[-1] == 0 for _, action in dead_tail_policy)


def test_k6_alive_candidate_variants_use_expected_k3_tail() -> None:
    assert _effective_alive_pattern(3, "trunc6_chase5_comb_tail") == (1, 0, 1)
    assert _effective_alive_pattern(3, "trunc6_chase5_twin3_tail") == (1, 0, 0)
    assert sum(probability for probability, _ in k6_alive_trunc_comb_twin3_policy((4, 3, 3, 0, 0, 0), 1)) == pytest.approx(1.0)
    assert sum(probability for probability, _ in k6_alive_full_comb_control_policy((0, 0, 0, 0, 0, 0), 4)) == pytest.approx(1.0)


def test_classify_one_run_alive_relation_covers_cases() -> None:
    assert _classify_one_run_alive_relation((0, 3), (0, 4))[0] == "prefix_truncated_comb_on_full_k"
    assert _classify_one_run_alive_relation((1, 3), (1, 4))[0] == "alive_full_comb"
    assert _classify_one_run_alive_relation((2, 2), (1, 4))[0] == "alive_internal_repair"
    assert _classify_one_run_alive_relation((3, 4), (1, 3))[0] == "crosses_alive_dead_boundary"


def test_k6_one_run_interval_relation_classifier() -> None:
    assert _classify_k6_one_run_interval_relation((0, 4), (0, 0, 0, 0, 0, 0)) == "full_comb_like"
    assert _classify_k6_one_run_interval_relation((0, 3), (0, 0, 0, 0, 0, 0)) == "prefix_truncated"
    assert _classify_k6_one_run_interval_relation((1, 1), (2, 1, 1, 1, 1, 0)) in {
        "short_internal_repair",
        "left_shoulder_repair",
        "right_shoulder_repair",
        "crosses_packet_boundary",
    }


def test_balanced_policy_from_one_run_interval_examples() -> None:
    full_comb = _balanced_policy_from_one_run_interval(6, (0, 4))
    shoulder = _balanced_policy_from_one_run_interval(6, (1, 1))

    assert {action for _, action in full_comb} == {
        (0, 1, 0, 1, 0, 1),
        (1, 0, 1, 0, 1, 0),
    }
    assert {action for _, action in shoulder} == {
        (0, 0, 1, 1, 1, 1),
        (1, 1, 0, 0, 0, 0),
    }


def test_k6_one_run_rule_v1_branches() -> None:
    root = k6_one_run_rule_v1_policy((0, 0, 0, 0, 0, 0), 12)
    shoulder = k6_one_run_rule_v1_policy((2, 1, 1, 1, 1, 0), 10)
    staircase = k6_one_run_rule_v1_policy((4, 3, 2, 1, 1, 0), 8)
    mixed = k6_one_run_rule_v1_policy((5, 3, 3, 1, 1, 0), 7)

    assert {action for _, action in root} == {
        (0, 1, 0, 1, 0, 1),
        (1, 0, 1, 0, 1, 0),
    }
    assert {action for _, action in shoulder} == {
        (0, 0, 1, 1, 1, 1),
        (1, 1, 0, 0, 0, 0),
    }
    assert {action for _, action in staircase} == {
        (0, 1, 0, 1, 1, 1),
        (1, 0, 1, 0, 0, 0),
    }
    assert sum(probability for probability, _ in mixed) == pytest.approx(1.0)
    assert all(len(action) == 6 for _, action in mixed)


def test_k6_one_run_rule_v1_variants_return_valid_policies() -> None:
    state = (5, 3, 3, 1, 1, 0)
    for policy_fn in (
        k6_one_run_rule_v1_policy,
        k6_one_run_rule_v1_no_mixed_policy,
        k6_one_run_rule_v1_aggressive_shoulder_policy,
    ):
        policy = policy_fn(state, 7)
        assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
        assert all(len(action) == 6 for _, action in policy)


def test_k6_patch_ladder_base_matches_base_at_root() -> None:
    base_policy = k6_alive_full_comb_control_policy((0, 0, 0, 0, 0, 0), 12)
    patch_policy = make_k6_one_run_patch_policy("k6_alive_full_comb_control_policy", ())

    assert patch_policy((0, 0, 0, 0, 0, 0), 12) == base_policy


def test_k6_patch_ladder_patch_fires_on_exact_regime() -> None:
    patch = _k6_one_run_patch_list()[0]
    policy = make_k6_one_run_patch_policy("k6_alive_full_comb_control_policy", (patch.name,))
    state = (2, 1, 1, 1, 1, 0)

    assert _matching_k6_patch(state, 10, (patch,)) == patch
    assert {action for _, action in policy(state, 10)} == {
        (0, 0, 1, 1, 1, 1),
        (1, 1, 0, 0, 0, 0),
    }


def test_k6_patch_ladder_does_not_fire_on_nearby_regime() -> None:
    patch = _k6_one_run_patch_list()[0]
    nearby_state = (2, 1, 1, 1, 0, 0)

    assert _matching_k6_patch(nearby_state, 10, (patch,)) is None


def test_k6_patch_ladder_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_one_run_patch_ladder_benchmark(2)

    captured = capsys.readouterr()
    assert "k=6 one-run patch ladder benchmark" in captured.out
    assert "patch fired" in captured.out
    assert "gap_to_one_run" in captured.out


def test_k6_patch_ladder_benchmark_rows_are_valid() -> None:
    rows = k6_one_run_patch_ladder_benchmark_rows(2)

    assert len(rows) == len(_k6_one_run_patch_list()) + 1
    assert rows[0].policy_name == "k6_patch_ladder_0_base"


def test_k6_greedy_patch_candidates_include_late_cleanup() -> None:
    failure = K6OneRunCandidateFailureRow(
        policy_name="test",
        time=10,
        remaining_horizon=2,
        state=(3, 2, 2, 1, 1, 1),
        packet_type=(1, 2, 3),
        packet_gaps=(1, 1),
        adjacent_gaps=(1, 0, 1, 0, 0),
        occupancy_probability=0.25,
        one_run_value=0.0,
        candidate_step_value_on_one_run_continuation=-0.1,
        action_loss=0.1,
        weighted_action_loss=0.025,
        candidate_actions=(),
        candidate_patterns=(),
        lp_interval_distribution=(((0, 0), 1.0),),
        lp_dominant_interval=(0, 0),
        lp_dominant_relation="prefix_truncated",
    )

    candidates = _k6_greedy_patch_candidates_from_failures((failure,), (), 10)

    assert candidates
    assert candidates[0].remaining_horizon == 2
    assert candidates[0].packet_type == (1, 2, 3)
    assert candidates[0].packet_gaps == (1, 1)
    assert candidates[0].interval == (0, 0)


def test_k6_generated_patch_is_exact_regime_only() -> None:
    patch = _k6_greedy_patch_candidates_from_failures(
        (
            K6OneRunCandidateFailureRow(
                policy_name="test",
                time=10,
                remaining_horizon=2,
                state=(3, 2, 2, 1, 1, 1),
                packet_type=(1, 2, 3),
                packet_gaps=(1, 1),
                adjacent_gaps=(1, 0, 1, 0, 0),
                occupancy_probability=0.25,
                one_run_value=0.0,
                candidate_step_value_on_one_run_continuation=-0.1,
                action_loss=0.1,
                weighted_action_loss=0.025,
                candidate_actions=(),
                candidate_patterns=(),
                lp_interval_distribution=(((0, 0), 1.0),),
                lp_dominant_interval=(0, 0),
                lp_dominant_relation="prefix_truncated",
            ),
        ),
        (),
        10,
    )[0]
    policy = make_k6_one_run_patch_policy_from_patches("k6_alive_full_comb_control_policy", (patch,))
    matching_state = (3, 2, 2, 1, 1, 1)
    nearby_state = (3, 2, 2, 1, 1, 0)

    assert _matching_k6_patch(matching_state, 2, (patch,)) == patch
    assert _matching_k6_patch(nearby_state, 2, (patch,)) is None
    assert {action for _, action in policy(matching_state, 2)} == {
        (0, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0),
    }


def test_k6_greedy_patch_search_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_one_run_greedy_patch_search(2, max_patches=1, candidate_n=3)

    captured = capsys.readouterr()
    assert "selected patch" in captured.out
    assert "gap_vs_top_prefix" in captured.out
    assert "top remaining failure" in captured.out


def test_k6_greedy_patch_search_steps_are_valid() -> None:
    steps = k6_one_run_greedy_patch_search_steps(2, max_patches=1, candidate_n=3)

    assert steps
    assert steps[0].value <= steps[0].one_run_value + 1e-9


def test_k6_one_run_interval_distribution_for_regime_nonempty() -> None:
    distribution = _one_run_interval_distribution_for_regime(6, 2, 2, (6,), ())

    assert distribution
    assert sum(weight for weight, _ in distribution) == pytest.approx(1.0)


def test_mixed_one_run_interval_policy_probabilities_sum() -> None:
    policy = _mixed_one_run_interval_policy(6, ((0.25, (0, 0)), (0.75, (0, 4))))

    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    assert all(len(action) == 6 for _, action in policy)


def test_k6_mixed_patch_fires_only_on_exact_regime() -> None:
    patch = MixedIntervalPatch(
        "test",
        2,
        (1, 2, 3),
        (1, 1),
        ((1.0, (0, 0)),),
    )
    policy = make_k6_one_run_mixed_patch_policy(
        "k6_alive_full_comb_control_policy",
        (),
        (patch,),
    )
    matching_state = (3, 2, 2, 1, 1, 1)
    nearby_state = (3, 2, 2, 1, 1, 0)

    assert _matching_k6_mixed_patch(matching_state, 2, (patch,)) == patch
    assert _matching_k6_mixed_patch(nearby_state, 2, (patch,)) is None
    assert {action for _, action in policy(matching_state, 2)} == {
        (0, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0),
    }


def test_k6_mixed_patch_candidates_are_valid() -> None:
    failure = K6OneRunCandidateFailureRow(
        policy_name="test",
        time=0,
        remaining_horizon=2,
        state=(0, 0, 0, 0, 0, 0),
        packet_type=(6,),
        packet_gaps=(),
        adjacent_gaps=(0, 0, 0, 0, 0),
        occupancy_probability=1.0,
        one_run_value=1.0,
        candidate_step_value_on_one_run_continuation=0.9,
        action_loss=0.1,
        weighted_action_loss=0.1,
        candidate_actions=(),
        candidate_patterns=(),
        lp_interval_distribution=(((1, 1), 1.0),),
        lp_dominant_interval=(1, 1),
        lp_dominant_relation="short_internal_repair",
    )

    candidates = _k6_mixed_patch_candidates_from_failures(2, (failure,), (), 5)

    assert candidates
    assert candidates[0].weighted_intervals
    assert sum(weight for weight, _ in candidates[0].weighted_intervals) == pytest.approx(1.0)


def test_k6_greedy_mixed_patch_search_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_one_run_greedy_mixed_patch_search(2, max_patches=1, candidate_n=3)

    captured = capsys.readouterr()
    assert "selected mixed patch" in captured.out
    assert "weighted intervals" in captured.out
    assert "gap_vs_top_prefix" in captured.out


def test_k6_greedy_mixed_patch_search_steps_are_valid() -> None:
    steps = k6_one_run_greedy_mixed_patch_search_steps(2, max_patches=1, candidate_n=3)

    assert steps
    assert steps[0].value <= steps[0].one_run_value + 1e-9


def test_parse_T_values_for_k6_sweep() -> None:
    assert _parse_T_values("12,20,30") == (12, 20, 30)


def test_k6_explicit_long_horizon_sweep_rows_run() -> None:
    rows = k6_explicit_long_horizon_sweep_rows(
        (2,),
        ("top_prefix_all", "k6_alive_full_comb_control_policy"),
    )

    assert len(rows) == 2
    top_prefix = next(row for row in rows if row.policy_name == "top_prefix_all")
    assert top_prefix.gap_to_top_prefix == pytest.approx(0.0)
    assert top_prefix.normalized_gap_to_top_prefix == pytest.approx(0.0)


def test_k6_explicit_long_horizon_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_explicit_long_horizon_sweep(
        (2,),
        ("top_prefix_all", "k6_alive_full_comb_control_policy"),
    )

    captured = capsys.readouterr()
    assert "k=6 explicit long-horizon sweep" in captured.out
    assert "gap_to_top_prefix" in captured.out


def test_k6_one_run_alive_pattern_rows_are_nonempty() -> None:
    rows = k6_one_run_alive_pattern_rows(4)

    assert rows
    for row in rows:
        assert row.alive_count >= 1
        for support in row.support:
            assert support.relation_class in {
                "constant_or_no_run",
                "prefix_truncated_comb_on_full_k",
                "alive_full_comb",
                "alive_prefix_truncated_comb",
                "alive_internal_repair",
                "crosses_alive_dead_boundary",
                "outside_alive",
                "shifted_not_alive_explained",
            }


def test_k6_one_run_alive_pattern_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_one_run_alive_pattern_report(4, n=2)

    captured = capsys.readouterr()
    assert "k=6 one-run alive-pattern report" in captured.out
    assert "alive_count" in captured.out
    assert "relation class" in captured.out


def test_k6_alive_candidate_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_alive_candidate_benchmark((2,))

    captured = capsys.readouterr()
    assert "k=6 alive-prefix candidate benchmark" in captured.out
    assert "gap_to_one_run" in captured.out
    assert "pattern usage" in captured.out


def test_k6_one_run_interval_rule_rows_are_nonempty() -> None:
    rows = k6_one_run_interval_rule_rows(4)

    assert rows
    assert all(row.support for row in rows)
    assert {
        support.relation_class
        for row in rows
        for support in row.support
    } <= {
        "constant_or_no_run",
        "full_comb_like",
        "prefix_truncated",
        "short_internal_repair",
        "left_shoulder_repair",
        "right_shoulder_repair",
        "crosses_packet_boundary",
        "tail_cleanup",
        "other",
    }


def test_k6_one_run_interval_rule_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_one_run_interval_rule_report(4, n=2)

    captured = capsys.readouterr()
    assert "k=6 one-run interval-rule report" in captured.out
    assert "Top candidate failure states" in captured.out
    assert "Rule-hint summary" in captured.out


def test_evaluate_time_dependent_policy_runs_on_toy_policy() -> None:
    def toy_policy(state: tuple[int, ...], remaining_horizon: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        del state, remaining_horizon
        return ((0.5, (1, 0, 0)), (0.5, (0, 1, 1)))

    values = evaluate_time_dependent_policy(3, 2, toy_policy)

    assert (0, 0, 0) in values[2]


def test_reachable_time_policy_matches_full_evaluator_on_toy_policy() -> None:
    def toy_policy(state: tuple[int, ...], remaining_horizon: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        del state, remaining_horizon
        return ((0.5, (1, 0, 0)), (0.5, (0, 1, 1)))

    full_value = evaluate_time_dependent_policy(3, 4, toy_policy)[4][(0, 0, 0)]
    reachable = evaluate_time_dependent_policy_reachable_root(3, 4, toy_policy)

    assert reachable.value == pytest.approx(full_value)
    assert reachable.visited_state_count > 0
    assert sum(count for _, count in reachable.visited_by_horizon) == reachable.visited_state_count


def test_reachable_time_policy_matches_full_evaluator_on_k6_policy() -> None:
    full_value = evaluate_time_dependent_policy(6, 4, k6_alive_trunc_comb_policy)[4][(0, 0, 0, 0, 0, 0)]
    reachable = evaluate_time_dependent_policy_reachable_root(6, 4, k6_alive_trunc_comb_policy)

    assert reachable.value == pytest.approx(full_value)


def test_reachable_time_policy_visits_no_more_than_state_layers() -> None:
    def toy_policy(state: tuple[int, ...], remaining_horizon: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        del state, remaining_horizon
        return ((0.5, (1, 0, 0)), (0.5, (0, 1, 1)))

    reachable = evaluate_time_dependent_policy_reachable_root(3, 4, toy_policy)
    # For k=3 and T=4 there are C(t+k-1,k-1) canonical states at each layer.
    full_layer_count = sum(((used + 2) * (used + 1)) // 2 for used in range(5))

    assert reachable.visited_state_count <= full_layer_count


def test_explicit_time_policy_reachable_benchmark_rows_run() -> None:
    rows = explicit_time_policy_reachable_benchmark_rows(
        ((6, 4),),
        ("top_prefix_all", "k6_alive_trunc_comb_policy"),
    )

    assert len(rows) == 2
    top_prefix = next(row for row in rows if row.policy_name == "top_prefix_all")
    assert top_prefix.gap_to_top_prefix == pytest.approx(0.0)


def test_explicit_time_policy_reachable_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_explicit_time_policy_reachable_benchmark(
        ((6, 4),),
        ("top_prefix_all", "k6_alive_trunc_comb_policy"),
    )

    captured = capsys.readouterr()
    assert "visited_state_count" in captured.out
    assert "gap_to_top_prefix/sqrt(T)" in captured.out


def test_k6_barycentric_policy_distributions_are_valid() -> None:
    state = (4, 2, 2, 1, 0, 0)
    policies = (
        k6_barycentric_v1_equal_policy,
        k6_barycentric_v1_late_policy,
        k6_barycentric_v1_middle_policy,
        k6_barycentric_v1_tail_heavy_policy,
        k6_barycentric_v1_top_tail_policy,
    )

    for policy_fn in policies:
        support = policy_fn(state, 5)
        assert sum(probability for probability, _ in support) == pytest.approx(1.0)
        assert all(len(action) == 6 for _, action in support)


def test_k6_barycentric_interval_mixtures_sum_to_one() -> None:
    for variant in ("equal", "tail_heavy", "top_tail"):
        assert sum(weight for weight, _ in _k6_barycentric_weights(variant)) == pytest.approx(1.0)


def test_k6_barycentric_trigger_and_root_fallback() -> None:
    assert _k6_barycentric_trigger((4, 2, 2, 1, 0, 0), 5, "all")
    assert not _k6_barycentric_trigger((0, 0, 0, 0, 0, 0), 12, "all")

    root_policy = k6_barycentric_v1_equal_policy((0, 0, 0, 0, 0, 0), 12)
    assert set(root_policy) == {
        (0.5, (1, 0, 1, 0, 1, 0)),
        (0.5, (0, 1, 0, 1, 0, 1)),
    }


def test_k6_barycentric_candidate_benchmark_rows_run() -> None:
    rows = k6_barycentric_candidate_benchmark_rows(
        (2,),
        ("fixed_101010_comb", "k6_barycentric_v1_equal"),
        trigger_n=2,
    )

    assert len(rows) == 2
    fixed = next(row for row in rows if row.policy_name == "fixed_101010_comb")
    assert fixed.gap_to_fixed_comb == pytest.approx(0.0)


def test_k6_barycentric_candidate_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_barycentric_candidate_benchmark(
        (2,),
        ("fixed_101010_comb", "k6_barycentric_v1_equal"),
        trigger_n=2,
    )

    captured = capsys.readouterr()
    assert "k=6 barycentric candidate benchmark" in captured.out
    assert "gap_to_fixed_comb/sqrt(T)" in captured.out
    assert "trigger statistics" in captured.out


def test_k6_barycentric_trigger_sweep_candidates_are_valid() -> None:
    candidates = _k6_barycentric_trigger_sweep_candidates()
    assert candidates
    synthetic_state = (4, 2, 2, 1, 0, 0)
    root = (0, 0, 0, 0, 0, 0)

    assert all(isinstance(_k6_barycentric_trigger_fires(candidate, synthetic_state, 5), bool) for candidate in candidates)
    assert any(_k6_barycentric_trigger_fires(candidate, synthetic_state, 5) for candidate in candidates)
    assert not any(_k6_barycentric_trigger_fires(candidate, root, 12) for candidate in candidates)


def test_k6_barycentric_trigger_sweep_policy_distribution_is_valid() -> None:
    candidate = next(
        candidate
        for candidate in _k6_barycentric_trigger_sweep_candidates()
        if candidate.trigger_name == "top_singleton_second_pair"
        and candidate.mixture_name == "equal_top_full_tail"
    )
    policy = make_k6_barycentric_trigger_sweep_policy(candidate)
    support = policy((4, 2, 2, 1, 0, 0), 5)

    assert sum(probability for probability, _ in support) == pytest.approx(1.0)
    assert all(len(action) == 6 for _, action in support)


def test_k6_barycentric_trigger_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_barycentric_trigger_sweep(
        (2,),
        final_T_values=(),
        trigger_n=2,
        finalist_count=2,
    )

    captured = capsys.readouterr()
    assert "k=6 barycentric trigger sweep" in captured.out
    assert "best average normalized gap" in captured.out
    assert "top triggered states" in captured.out


def test_centered_max_invariant_under_translation() -> None:
    assert _phi_centered_max((3, 2, 0, 0, 0)) == pytest.approx(_phi_centered_max((8, 7, 5, 5, 5)))


def test_k5_comb_and_chase_have_balanced_marginals() -> None:
    for support in (_k5_comb_support(), _k5_chase_support()):
        marginals = [
            sum(probability * action[index] for probability, action in support)
            for index in range(5)
        ]
        assert marginals == pytest.approx([0.5] * 5)


def test_k5_centered_potential_rows_include_root() -> None:
    rows = k5_centered_potential_comb_vs_chase_rows(0)

    assert rows
    root_rows = [row for row in rows if row.state == (0, 0, 0, 0, 0)]
    assert root_rows


def test_k5_centered_potential_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_centered_potential_comb_vs_chase_report(2, n=2)

    captured = capsys.readouterr()
    assert "centered potential" in captured.out
    assert "Top states where Chase beats COMB" in captured.out


def test_softmax_centered_gradient_probabilities_sum_to_one() -> None:
    learner = _softmax_centered_gradient((3, 2, 0, 0, 0), 1.0)

    assert sum(learner) == pytest.approx(1.0)
    assert all(probability >= 0.0 for probability in learner)


def test_softmax_potential_invariant_under_translation() -> None:
    assert _phi_centered_softmax((3, 2, 0, 0, 0), 1.0) == pytest.approx(
        _phi_centered_softmax((8, 7, 5, 5, 5), 1.0)
    )


def test_k5_softmax_certificate_rows_are_nonempty() -> None:
    rows = k5_softmax_potential_certificate_rows(1, horizons=(5,))

    assert rows
    assert all(row.learner for row in rows)


def test_k5_softmax_potential_certificate_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_softmax_potential_certificate_report(1, n=2)

    captured = capsys.readouterr()
    assert "softmax potential certificate" in captured.out
    assert "Top residual failures" in captured.out


def test_potential_greedy_k5_root_returns_valid_distribution() -> None:
    policy = potential_greedy_interval_policy(5, 0.5)
    support = policy((0, 0, 0, 0, 0), 5)

    assert sum(probability for probability, _ in support) == pytest.approx(1.0)
    assert all(len(action) == 5 for _, action in support)


def test_potential_greedy_choice_names_are_from_grammar() -> None:
    choice = _potential_greedy_choice((0, 0, 0, 0, 0), 5, 0.5)

    assert choice.name
    assert sum(probability for probability, _ in choice.support) == pytest.approx(1.0)


def test_potential_greedy_benchmark_rows_run() -> None:
    rows = potential_greedy_interval_benchmark_rows(
        ((5, 2), (6, 2)),
        ("potential_greedy_c0.5", "fixed_10101_comb", "fixed_101010_comb"),
    )

    assert rows
    assert any(row.policy_name == "potential_greedy_c0.5" and row.k == 5 for row in rows)
    assert any(row.policy_name == "potential_greedy_c0.5" and row.k == 6 for row in rows)


def test_potential_greedy_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_potential_greedy_interval_benchmark(
        ((5, 2),),
        ("potential_greedy_c0.5", "fixed_10101_comb", "chase5"),
        print_action_stats=True,
        n=2,
    )

    captured = capsys.readouterr()
    assert "potential-greedy interval benchmark" in captured.out
    assert "action stats" in captured.out


def test_random_playout_phi_h0_equals_max() -> None:
    value = _random_playout_value_function(5, 3, "Q_chase")

    assert value((3, 2, 0, 0, 0), 0) == pytest.approx(3.0)


def test_random_playout_terminal_winner_probabilities_sum_to_one() -> None:
    probabilities = _random_playout_terminal_winner_probabilities(5, "Q_chase", (0, 0, 0, 0, 0), 2)

    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability >= 0.0 for probability in probabilities)


def test_random_playout_value_and_greedy_rows_run() -> None:
    value_rows = random_playout_value_rows(((5, 2),), ("Q_chase",))
    greedy_rows = random_playout_greedy_rows(((5, 2),), ("Q_chase",))

    assert value_rows
    assert greedy_rows


def test_random_playout_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_random_playout_potential_benchmark(
        ((5, 2),),
        playout_names=("Q_chase",),
        mode="all",
        print_action_stats=True,
        residual_max_T=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "random playout potential" in captured.out
    assert "Playout values" in captured.out
    assert "Potential-greedy values" in captured.out


def test_k5_gap_vector_extraction() -> None:
    assert _k5_gap_vector((5, 3, 3, 1, 0)) == (2, 0, 2, 1)


def test_k5_transition_distribution_sums_to_one() -> None:
    distribution = _next_state_distribution((0, 0, 0, 0, 0), _k5_chase_support())

    assert sum(probability for _, probability in distribution) == pytest.approx(1.0)


def test_k5_two_step_distribution_sums_to_one() -> None:
    distribution = _two_step_distribution((0, 0, 0, 0, 0), _k5_comb_support(), _k5_chase_support())

    assert sum(distribution.values()) == pytest.approx(1.0)


def test_k5_gap_transition_rows_are_nonempty() -> None:
    rows = k5_gap_transition_rows(1)

    assert rows
    assert all(row.gap_vector for row in rows)


def test_k5_gap_proof_diagnostics_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_gap_proof_diagnostics(max_used=1, h_values=(2,), n=2)

    captured = capsys.readouterr()
    assert "k=5 gap proof diagnostics" in captured.out
    assert "certificate failures by gaps" in captured.out
    assert "exchangeability mini-test" in captured.out


def test_k5_expected_two_step_potential_value_is_symmetric_for_same_action() -> None:
    value_fn = _random_playout_value_function(5, 3, "Q_chase")
    state = (1, 0, 0, 0, 0)
    support = _k5_chase_support()

    left = _expected_two_step_potential_value(state, support, support, value_fn, 1)
    right = _expected_two_step_potential_value(state, support, support, value_fn, 1)

    assert left == pytest.approx(right)


def test_balanced_action_enumeration_vertices_are_balanced() -> None:
    vertices = k5_balanced_action_vertices()

    assert len(vertices) == 1293
    assert all(_balanced_action_is_balanced(vertex.support) for vertex in vertices)


def test_balanced_action_enumeration_special_actions_included() -> None:
    assert _k5_chase_balanced_vertex() is not None
    assert _k5_comb_balanced_vertex() is not None


def test_balanced_action_enumeration_canonicalization_preserves_balance() -> None:
    chase = _k5_chase_balanced_vertex()
    assert chase is not None

    assert _balanced_action_canonical_key(chase.support) == chase.canonical_key
    assert _balanced_action_is_balanced(chase.support)


def test_balanced_action_enumeration_has_no_duplicate_canonical_classes() -> None:
    classes = _k5_balanced_action_symmetry_classes()
    keys = [key for key, _ in classes]

    assert len(keys) == len(set(keys))
    assert len(classes) == 25


def test_balanced_action_enumeration_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_action_enumeration_report(n=2)

    captured = capsys.readouterr()
    assert "k=5 balanced action enumeration report" in captured.out
    assert "symmetry classes" in captured.out
    assert "Chase" in captured.out


def test_balanced_chase_greedy_defect_rows_are_nonnegative() -> None:
    rows = k5_balanced_chase_greedy_defect_rows(h_values=(3,), max_used=1)

    assert rows
    assert all(row.epsilon >= -1e-10 for row in rows)


def test_balanced_chase_greedy_defect_chase_q_matches_itself() -> None:
    chase = _k5_chase_balanced_vertex()
    assert chase is not None

    state = (1, 0, 0, 0, 0)
    assert _balanced_chase_q_value(state, 3, chase) == pytest.approx(_balanced_chase_q_value(state, 3, chase))


def test_balanced_chase_greedy_defect_report_mentions_competitors(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_chase_greedy_defect_report(h_values=(3,), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "balanced Chase greedy defect" in captured.out
    assert "worst by packet" in captured.out
    assert "best_competing_balanced_action" in captured.out


def test_balanced_chase_orbit_contains_expected_complement_pairs() -> None:
    supports = {tuple(action for _, action in vertex.support) for vertex in _k5_chase_orbit_vertices()}

    assert ((0, 0, 1, 0, 1), (1, 1, 0, 1, 0)) in supports
    assert ((0, 1, 0, 1, 0), (1, 0, 1, 0, 1)) in supports
    assert ((0, 1, 0, 1, 1), (1, 0, 1, 0, 0)) in supports


def test_balanced_chase_orbit_defect_raw_reproduces_previous_report() -> None:
    rows = k5_balanced_chase_orbit_defect_rows(h_values=(10,), max_used=8)

    assert max(row.eps_raw for row in rows) == pytest.approx(0.5)


def test_balanced_chase_orbit_defect_is_no_larger_than_raw() -> None:
    rows = k5_balanced_chase_orbit_defect_rows(h_values=(3,), max_used=1)

    assert rows
    assert all(row.eps_orbit <= row.eps_raw + 1e-10 for row in rows)


def test_balanced_chase_symmetrized_support_is_balanced_and_invariant() -> None:
    state = (2, 1, 1, 0, 0)
    support = _k5_symmetrized_chase_support(state)

    assert _fraction_support_is_balanced(support)
    assert _k5_symmetrized_chase_support((2, 1, 1, 0, 0)) == _k5_symmetrized_chase_support((2, 1, 1, 0, 0))


def test_balanced_chase_orbit_defect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_chase_orbit_defect_report(h_values=(3,), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "balanced Chase orbit defect" in captured.out
    assert "max defects by packet" in captured.out
    assert "best orbit representatives" in captured.out


def test_balanced_centered_orbit_defect_all_vertices_have_common_marginal() -> None:
    for vertex in k5_balanced_action_vertices():
        assert _balanced_support_common_marginal(vertex.support) == vertex.marginal


def test_balanced_centered_orbit_defect_zero_and_one_actions_equivalent() -> None:
    state = (2, 1, 1, 0, 0)
    zero_support = ((Fraction(1), (0, 0, 0, 0, 0)),)
    one_support = ((Fraction(1), (1, 1, 1, 1, 1)),)

    assert _balanced_chase_centered_q_for_support(state, 5, zero_support) == pytest.approx(
        _balanced_chase_centered_q_for_support(state, 5, one_support)
    )


def test_balanced_centered_orbit_defect_correction_equals_marginal_minus_half() -> None:
    state = (8, 0, 0, 0, 0)
    support = (
        (Fraction(1, 4), (0, 0, 1, 1, 1)),
        (Fraction(1, 4), (1, 1, 0, 1, 1)),
        (Fraction(1, 4), (1, 1, 1, 0, 1)),
        (Fraction(1, 4), (1, 1, 1, 1, 0)),
    )
    old_q = _balanced_chase_old_fixed_q_for_support(state, 10, support)
    centered_q = _balanced_chase_centered_q_for_support(state, 10, support)

    assert old_q - centered_q == pytest.approx(float(_balanced_support_common_marginal(support) - Fraction(1, 2)))


def test_balanced_centered_orbit_defect_previous_raw_report_reproducible() -> None:
    rows = k5_balanced_centered_orbit_defect_rows(h_values=(10,), max_used=8)

    assert max(row.canonical_raw_eps for row in rows) == pytest.approx(0.5)


def test_balanced_centered_orbit_defect_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_centered_orbit_defect_report(h_values=(3,), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "balanced centered Chase orbit defect" in captured.out
    assert "packet (1,4) audit" in captured.out
    assert "baseline correction samples" in captured.out


def test_balanced_varying_c_obstruction_reproduces_one_eighth() -> None:
    rows = k5_balanced_varying_c_obstruction_rows(h_values=(10,), max_used=1)
    row = next(item for item in rows if item.state == (1, 0, 0, 0, 0))

    assert row.eps_full == pytest.approx(0.125)
    assert row.best_c == Fraction(1, 4)


def test_balanced_varying_c_obstruction_c_half_defect_zero() -> None:
    rows = k5_balanced_varying_c_obstruction_rows(h_values=(10,), max_used=2)

    assert max(row.eps_c_half for row in rows) <= 1e-9


def test_balanced_varying_c_obstruction_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_varying_c_obstruction_report(h_values=(3,), max_used=1, n=3)

    captured = capsys.readouterr()
    assert "varying-c obstruction" in captured.out
    assert "packet (1,4)" in captured.out
    assert "best c" in captured.out
    assert "exact 1/8" in captured.out


def test_balanced_mdp_value_gap_orders_initial_values() -> None:
    rows = k5_balanced_mdp_value_gap_rows(h_values=(1, 2, 3))

    assert rows
    assert all(row.v_full >= row.v_half - 1e-9 for row in rows)
    assert all(row.v_half >= row.v_orbit - 1e-9 for row in rows)
    assert all(row.v_orbit >= row.v_fixed_chase - 1e-9 for row in rows)


def test_balanced_mdp_value_gap_h1_reproduces_flat_one_step_values() -> None:
    row = k5_balanced_mdp_value_gap_rows(h_values=(1,))[0]

    assert row.v_full == pytest.approx(0.8)
    assert row.v_half == pytest.approx(0.5)
    assert row.v_orbit == pytest.approx(0.5)
    assert row.gap_full_minus_orbit == pytest.approx(0.3)
    assert row.initial_best_c == Fraction(1, 5)


def test_balanced_mdp_value_gap_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_mdp_value_gap_report(h_values=(1, 2), n=3)

    captured = capsys.readouterr()
    assert "balanced MDP value gap" in captured.out
    assert "V_full - V_orbit" in captured.out
    assert "policy c-usage" in captured.out
    assert "exceptional packets" in captured.out


def test_balanced_mdp_value_gap_summary_only_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_mdp_value_gap_report(h_values=(1, 2), n=3, summary_only=True)

    captured = capsys.readouterr()
    assert "balanced MDP value gap" in captured.out
    assert "summary_only: True" in captured.out
    assert "policy c-usage" not in captured.out
    assert "exceptional packets" not in captured.out


def test_balanced_mdp_gap_scaling_rows_reproduce_flat_gap_ratios() -> None:
    row = k5_balanced_mdp_gap_scaling_rows(h_values=(1,))[0]

    assert row.gap_full_minus_orbit == pytest.approx(0.3)
    assert row.gap_full_minus_half == pytest.approx(0.3)
    assert row.gap_half_minus_orbit == pytest.approx(0.0)
    assert row.gap_over_sqrt_h == pytest.approx(0.3)
    assert row.best_first_c == Fraction(1, 5)


def test_balanced_mdp_gap_scaling_summary_reports_monotonicity() -> None:
    summary = k5_balanced_mdp_gap_scaling_summary(h_values=(1, 2, 3, 4), tail_length=3)

    assert summary.max_gap == pytest.approx(0.3)
    assert summary.argmax_h == 1
    assert summary.monotonicity_holds
    assert summary.c_half_orbit_optimality_holds
    assert summary.tail_horizons == (2, 3, 4)


def test_balanced_mdp_gap_scaling_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_balanced_mdp_gap_scaling_report(h_values=(1, 2, 3), tail_length=2)

    captured = capsys.readouterr()
    assert "balanced MDP gap scaling" in captured.out
    assert "gap/sqrt(h)" in captured.out
    assert "monotonicity V_full >= V_half >= V_orbit holds" in captured.out
    assert "c=1/2 orbit optimality holds on tested flat horizons." in captured.out


def test_c_half_orbit_optimality_orbit_actions_are_in_b_half() -> None:
    assert all(vertex.marginal == Fraction(1, 2) for vertex in _k5_chase_orbit_vertices())


def test_c_half_orbit_optimality_rows_have_no_violation_on_small_grid() -> None:
    rows = k5_c_half_orbit_optimality_violation_rows(h_values=(1, 2, 3), max_used=2)

    assert rows
    assert max(row.eps_half_orbit for row in rows) <= 1e-10


def test_c_half_orbit_optimality_reproduces_flat_gap_scaling() -> None:
    assert k5_c_half_orbit_flat_equality_holds(h_values=tuple(range(1, 16)))


def test_c_half_orbit_optimality_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_c_half_orbit_optimality_certificate_rows(h_values=(1, 2), max_used=1)
    assert rows

    print_k5_c_half_orbit_optimality_certificate_report(h_values=(1, 2), max_used=1, n=3)

    captured = capsys.readouterr()
    assert "c=1/2 orbit optimality" in captured.out
    assert "transition containment" in captured.out
    assert "value domination" in captured.out
    assert "max violation" in captured.out


def test_c_half_orbit_dual_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_c_half_orbit_dual_certificate_rows(h_values=(1, 2), max_used=1)
    assert rows
    assert all(row.certificate_type in {"transition-hull", "value-domination"} for row in rows)

    print_k5_c_half_orbit_dual_certificate_report(h_values=(1, 2), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "c=1/2 orbit dual certificate" in captured.out
    assert "transition-hull certificates" in captured.out
    assert "value-domination inequalities" in captured.out
    assert "theorem-ready implication" in captured.out


def test_c_half_invariant_inequality_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_c_half_invariant_inequality_closure_rows(h_values=(2,), max_used=1)
    assert rows

    print_k5_c_half_invariant_inequality_closure_report(h_values=(2,), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "c=1/2 invariant inequality closure" in captured.out
    assert "closure summary" in captured.out
    assert "active-action case splits" in captured.out
    assert "theorem-ready closure table" in captured.out


def test_c_half_invariant_cone_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_c_half_invariant_cone_closure_report(h_values=(2,), max_used=1, n=2)

    captured = capsys.readouterr()
    assert "c=1/2 invariant cone closure" in captured.out
    assert "known value-domination inequality families" in captured.out
    assert "closure table" in captured.out
    assert "failures by exact packet/action/class" in captured.out


def test_c_half_augmented_cone_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_c_half_augmented_cone_closure_rows(h_values=(2,), max_used=1, cegar_iterations=1)
    assert rows

    print_k5_c_half_augmented_cone_closure_report(
        h_values=(2,),
        max_used=1,
        n=2,
        cegar_iterations=1,
    )

    captured = capsys.readouterr()
    assert "c=1/2 augmented cone closure" in captured.out
    assert "orbit guard inequalities" in captured.out
    assert "CEGAR" in captured.out
    assert "remaining failures" in captured.out


def test_c_half_augmented_cone_proof_export_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    export = k5_c_half_augmented_cone_proof_export(h_values=(2,), max_used=1, cegar_iterations=1)
    assert export.rows

    print_k5_c_half_augmented_cone_proof_export_report(
        h_values=(2,),
        max_used=1,
        n=2,
        cegar_iterations=1,
    )

    captured = capsys.readouterr()
    assert "c=1/2 augmented cone proof export" in captured.out
    assert "generator families" in captured.out
    assert "closed-row expansions" in captured.out
    assert "LaTeX-ready theorem statement" in captured.out


def test_c_half_augmented_cone_audit_export_and_verify_runs(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    certificate_path = tmp_path / "c_half_certificate.json"
    result = k5_c_half_augmented_cone_audit_report(
        h_values=(2,),
        max_used=0,
        export_certificate=str(certificate_path),
        cegar_iterations=1,
    )
    assert certificate_path.exists()
    assert result.pushed_rows >= result.closed_rows

    print_k5_c_half_augmented_cone_audit_report(
        h_values=(2,),
        max_used=0,
        export_certificate=str(certificate_path),
        cegar_iterations=1,
    )
    print_k5_c_half_augmented_cone_certificate_verify_report(str(certificate_path))

    captured = capsys.readouterr()
    assert "k5 c=1/2 augmented cone audit" in captured.out
    assert "finite representative basis" in captured.out
    assert "rationalized cone export" in captured.out
    assert "k5 c=1/2 augmented cone certificate verifier" in captured.out
    assert "certificate path:" in captured.out


def test_c_half_augmented_cone_small_basis_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    result = k5_c_half_augmented_cone_small_basis_report_data(h_values=(2,), max_used=0, cegar_iterations=1)
    assert result.groups
    assert result.basis_total >= result.basis_closed

    print_k5_c_half_augmented_cone_small_basis_report(
        h_values=(2,),
        max_used=0,
        n=2,
        cegar_iterations=1,
    )

    captured = capsys.readouterr()
    assert "c=1/2 augmented cone small human-readable basis" in captured.out
    assert "CEGAR generator classification" in captured.out
    assert "small-basis reproduction test" in captured.out


def test_c_half_schur_guard_cone_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    result = k5_c_half_schur_guard_cone_result(h_values=(2,), max_used=0)
    assert result.dominance_total >= result.dominance_closed

    print_k5_c_half_schur_guard_cone_report(
        h_values=(2,),
        max_used=0,
        n=2,
    )

    captured = capsys.readouterr()
    assert "c=1/2 cone as Schur-convexity plus active guards" in captured.out
    assert "Schur-convex / majorization inequalities" in captured.out
    assert "c=1/2 dominance from Schur + active guards" in captured.out
    assert "minimal additional exchange schemas" in captured.out


def test_full_vs_orbit_delta_barrier_reproduces_flat_delta_table() -> None:
    rows = k5_full_vs_orbit_delta_rows(h_max=1, max_used=0)
    flat = next(row for row in rows if row.horizon == 1 and row.state == (0, 0, 0, 0, 0))

    assert flat.delta == pytest.approx(0.3)
    assert flat.best_full_c == Fraction(1, 5)


def test_full_vs_orbit_delta_barrier_delta_is_nonnegative() -> None:
    rows = k5_full_vs_orbit_delta_rows(h_max=2, max_used=1)

    assert rows
    assert all(row.delta >= -1e-12 for row in rows)


def test_full_vs_orbit_delta_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_delta_barrier_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "full-vs-orbit Delta" in captured.out
    assert "barrier residual" in captured.out
    assert "packet barrier" in captured.out
    assert "flat bound" in captured.out


def test_delta_supersolution_reproduces_observed_flat_delta() -> None:
    rows = k5_full_vs_orbit_delta_rows(h_max=1, max_used=0)
    flat = next(row for row in rows if row.state == (0, 0, 0, 0, 0))

    assert flat.delta == pytest.approx(0.3)


def test_delta_supersolution_packet_constants_dominate_observed_grid() -> None:
    rows = k5_full_vs_orbit_delta_rows(h_max=2, max_used=1)
    constants: dict[tuple[int, ...], float] = {}
    for row in rows:
        constants[row.packet_type] = max(constants.get(row.packet_type, 0.0), row.delta)

    assert rows
    assert all(row.delta <= constants[row.packet_type] + 1e-12 for row in rows)


def test_delta_supersolution_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_full_vs_orbit_delta_supersolution_rows(h_max=1, max_used=0)
    assert rows

    print_k5_full_vs_orbit_delta_supersolution_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "Delta supersolution" in captured.out
    assert "packet barrier" in captured.out
    assert "Bellman residual" in captured.out
    assert "fitted constants" in captured.out


def test_reachable_delta_barrier_builds_union_dag() -> None:
    rows = k5_full_vs_orbit_reachable_delta_rows(h_max=2)
    reachabilities = {row.reachability for row in rows}

    assert {"full", "orbit", "union"} <= reachabilities
    assert all(row.delta >= -1e-12 for row in rows)


def test_reachable_delta_barrier_supersolution_rows_run() -> None:
    rows = k5_full_vs_orbit_reachable_delta_supersolution_rows(h_max=1)

    assert rows
    assert {row.barrier_name for row in rows} == {"constant_0.3", "reachable packet barrier"}


def test_reachable_delta_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_reachable_delta_barrier_report(h_max=1, n=2)

    captured = capsys.readouterr()
    assert "reachable full-vs-orbit Delta barrier" in captured.out
    assert "reachable DAG sizes" in captured.out
    assert "Bellman residual on reachable states" in captured.out
    assert "off-policy diagnostic" in captured.out


def test_time_packet_barrier_rows_dominate_flat_delta() -> None:
    rows = k5_full_vs_orbit_time_packet_barrier_rows(h_max=1, max_used=0)
    flat = next(row for row in rows if row.time == 1 and row.packet_type == (5,))

    assert flat.observed_delta == pytest.approx(0.3)
    assert flat.fitted_constant >= flat.flat_delta


def test_time_packet_barrier_residuals_are_feasible_on_tiny_grid() -> None:
    rows = k5_full_vs_orbit_time_packet_barrier_residual_rows(h_max=2, max_used=1)

    assert rows
    assert max(row.residual for row in rows) <= 1e-9


def test_time_packet_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_time_packet_barrier_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "full-vs-orbit time-packet barrier" in captured.out
    assert "feasibility" in captured.out
    assert "flat packet sequence" in captured.out
    assert "boundedness / periodicity hints" in captured.out


def test_packet_barrier_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_packet_barrier_recurrence_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "full-vs-orbit packet barrier recurrence" in captured.out
    assert "compact P_t(packet) table" in captured.out
    assert "requirement map argmax" in captured.out
    assert "fixed-point packet barrier" in captured.out


def test_time_packet_barrier_growth_rows_run() -> None:
    summary = k5_full_vs_orbit_time_packet_barrier_growth_summary_rows(h_max=2, max_used=1)
    packets = k5_full_vs_orbit_time_packet_barrier_growth_packet_rows(h_max=2, max_used=1)

    assert len(summary) == 2
    assert packets
    assert all(len(row.values) == 2 for row in packets)


def test_time_packet_barrier_growth_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_time_packet_barrier_growth_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "full-vs-orbit time-packet barrier growth" in captured.out
    assert "global growth summary" in captured.out
    assert "simple fits" in captured.out
    assert "growth evidence" in captured.out


def test_packet_barrier_growth_inequality_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_full_vs_orbit_packet_barrier_growth_inequality_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "full-vs-orbit packet barrier growth inequality" in captured.out
    assert "M_t increment diagnostics" in captured.out
    assert "empirical upper envelopes" in captured.out
    assert "candidate theorem" in captured.out


def test_packet_barrier_increment_source_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_packet_barrier_increment_source_rows(h_max=1, max_used=0)
    assert rows

    print_k5_packet_barrier_increment_source_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "packet barrier increment source" in captured.out
    assert "global increment witnesses" in captured.out
    assert "mechanism groups" in captured.out
    assert "candidate symbolic inequalities" in captured.out


def test_packet_barrier_mechanism_source_bound_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet_barrier_mechanism_source_bound_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "packet barrier mechanism source bound" in captured.out
    assert "clean mechanism table" in captured.out
    assert "representative exact formulas" in captured.out
    assert "harmonic envelope" in captured.out


def test_variable_c_mechanism_domination_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_variable_c_mechanism_domination_rows(h_max=1, max_used=0)
    assert rows

    print_k5_variable_c_mechanism_domination_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "variable-c mechanism domination" in captured.out
    assert "domination summary" in captured.out
    assert "mechanism coverage" in captured.out
    assert "failures" in captured.out


def test_variable_c_harmonic_barrier_lp_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    results = k5_variable_c_harmonic_barrier_lp_results(h_max=1, max_used=0)
    assert results

    print_k5_variable_c_harmonic_barrier_lp_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "variable-c harmonic barrier LP" in captured.out
    assert "refinement comparison" in captured.out
    assert "requested special labels" in captured.out
    assert "theorem candidate" in captured.out


def test_variable_c_harmonic_barrier_compression_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_variable_c_harmonic_barrier_compression_rows(h_max=1, max_used=0)
    assert rows

    print_k5_variable_c_harmonic_barrier_compression_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "variable-c low-gap harmonic barrier compression" in captured.out
    assert "simplest successful A(x)" in captured.out
    assert "failed simple forms" in captured.out
    assert "theorem candidate" in captured.out


def test_variable_c_harmonic_barrier_simple_candidates_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_variable_c_harmonic_barrier_simple_candidate_rows(h_max=1, max_used=0)
    assert rows

    print_k5_variable_c_harmonic_barrier_simple_candidates_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "variable-c human-simple low-gap harmonic supersolutions" in captured.out
    assert "simplest candidate that closes" in captured.out
    assert "worst rows for failed candidates" in captured.out
    assert "theorem candidate" in captured.out


def test_low_gap_a0_c1_human_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_a0_c1_human_proof_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_a0_c1_human_proof_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap A=0 C=1 human-proof report" in captured.out
    assert "mechanism summary" in captured.out
    assert "rationalized formulas" in captured.out
    assert "theorem candidate" in captured.out


def test_low_gap_a0_c1_mechanism_lemma_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_a0_c1_mechanism_lemma_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_a0_c1_mechanism_lemma_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap A=0 C=1 mechanism lemma report" in captured.out
    assert "compact lemma table" in captured.out
    assert "rational witnesses" in captured.out
    assert "theorem-ready statement" in captured.out


def test_low_gap_a0_c1_symbolic_mechanism_proofs_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_a0_c1_symbolic_mechanism_proof_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_a0_c1_symbolic_mechanism_proofs_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap A=0 C=1 symbolic mechanism proofs" in captured.out
    assert "LaTeX-ready lemma table" in captured.out
    assert "generic inequalities needed" in captured.out
    assert "theorem-ready proof skeleton" in captured.out


def test_low_gap_boundary_and_variance_lemmas_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_boundary_and_variance_lemma_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_boundary_and_variance_lemmas_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap boundary and variance lemmas" in captured.out
    assert "definitions and finite differences" in captured.out
    assert "LaTeX-ready lemmas" in captured.out
    assert "remaining proof work" in captured.out


def test_orbit_finite_difference_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_orbit_finite_difference_recurrence_rows(h_max=1)
    assert rows

    print_k5_orbit_finite_difference_recurrence_report(h_max=1, n=4)

    captured = capsys.readouterr()
    assert "k5 orbit finite-difference recurrence report" in captured.out
    assert "quantity summary and candidate constants" in captured.out
    assert "exact Bellman expansions" in captured.out
    assert "candidate inductive barriers" in captured.out


def test_low_gap_exact_residual_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows, vector_alpha, feasible, _message = k5_low_gap_exact_residual_recurrence_rows(h_max=1, max_used=0)
    assert rows
    assert vector_alpha
    assert isinstance(feasible, bool)

    print_k5_low_gap_exact_residual_recurrence_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap exact residual recurrence report" in captured.out
    assert "scalar alpha_m/t barriers" in captured.out
    assert "exact residual recurrence diagnostics" in captured.out
    assert "vector barrier fit" in captured.out


def test_low_gap_scalar_residual_envelope_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_scalar_residual_envelope_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_scalar_residual_envelope_proof_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap scalar residual envelope proof report" in captured.out
    assert "a_m(t)=t sup R_m(t) by mechanism" in captured.out
    assert "smallest sampled refinement" in captured.out
    assert "theorem candidate" in captured.out


def test_low_gap_scalar_envelope_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_scalar_envelope_recurrence_proof_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_scalar_envelope_recurrence_proof_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap scalar envelope recurrence proof report" in captured.out
    assert "scalar recurrence lemmas" in captured.out
    assert "induction" in captured.out
    assert "LaTeX-ready theorem template" in captured.out


def test_low_gap_scalar_envelope_exact_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_scalar_envelope_exact_proof_certificate_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_scalar_envelope_exact_proof_certificate_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap scalar envelope exact proof certificate" in captured.out
    assert "exact scalar envelope certificates" in captured.out
    assert "maximising representatives" in captured.out
    assert "proof gap" in captured.out


def test_low_gap_scalar_envelope_lyapunov_certificate_printer_runs(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    certificate = k5_low_gap_scalar_envelope_lyapunov_certificate_data(h_max=1, max_used=0)
    assert certificate["kind"] == "k5-low-gap-scalar-envelope-lyapunov-certificate"

    certificate_path = tmp_path / "lyapunov_certificate.json"
    print_k5_low_gap_scalar_envelope_lyapunov_certificate_report(
        h_max=1,
        max_used=0,
        certificate_path=str(certificate_path),
        n=2,
    )
    print_k5_low_gap_scalar_envelope_lyapunov_certificate_verify_report(str(certificate_path))

    captured = capsys.readouterr()
    assert "k5 low-gap scalar envelope Lyapunov certificate" in captured.out
    assert "independent verifier command" in captured.out
    assert "k5 low-gap scalar envelope Lyapunov certificate verifier" in captured.out
    assert certificate_path.exists()


def test_low_gap_unified_residual_automaton_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_unified_residual_automaton_certificate_data(h_max=1, max_used=0)
    assert data["kind"] == "k5-low-gap-unified-residual-automaton-certificate"

    print_k5_low_gap_unified_residual_automaton_certificate_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap unified residual automaton certificate" in captured.out
    assert "unified envelope" in captured.out
    assert "block decomposition" in captured.out
    assert "theorem-ready certificate" in captured.out


def test_low_gap_lyapunov_certificate_debug_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_lyapunov_certificate_debug_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap Lyapunov certificate debug report" in captured.out
    assert "failure classification" in captured.out
    assert "top 2 failed inequalities" in captured.out
    assert "corrected certificate status" in captured.out


def test_low_gap_corrected_residual_taxonomy_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_corrected_residual_taxonomy_data(h_max=1, max_used=0)
    assert data["kind"] == "k5-low-gap-corrected-residual-taxonomy"

    print_k5_low_gap_corrected_residual_taxonomy_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap corrected residual taxonomy report" in captured.out
    assert "corrected mechanism table" in captured.out
    assert "refined labels whose old alpha was too small" in captured.out
    assert "scalar envelope and Lyapunov verifier rerun" in captured.out


def test_low_gap_corrected_tail_lyapunov_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_corrected_tail_lyapunov_certificate_data(h_max=1, max_used=0)
    assert data["kind"] == "k5-low-gap-corrected-tail-lyapunov-certificate"

    print_k5_low_gap_corrected_tail_lyapunov_certificate_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap corrected tail Lyapunov certificate" in captured.out
    assert "cap attempts" in captured.out
    assert "selected certificate" in captured.out
    assert "theorem statement" in captured.out


def test_low_gap_richer_tail_envelope_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_richer_tail_envelope_certificate_data(h_max=1, max_used=0)
    assert data["kind"] == "k5-low-gap-richer-tail-envelope-certificate"

    print_k5_low_gap_richer_tail_envelope_certificate_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap richer tail envelope certificate" in captured.out
    assert "envelope attempts" in captured.out
    assert "selected certificate" in captured.out
    assert "signed envelope" in captured.out


def test_low_gap_signed_cancellation_identity_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_signed_cancellation_identity_rows(h_max=2, max_used=0, n=2)
    assert rows

    print_k5_low_gap_signed_cancellation_identity_report(h_max=2, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap signed cancellation identity report" in captured.out
    assert "signed Bellman residual identities" in captured.out
    assert "potential correction candidates" in captured.out or "no simple positive-coefficient feature correction" in captured.out
    assert "proposed next potential correction" in captured.out


def test_low_gap_correction_potential_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_correction_potential_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_correction_potential_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap correction potential report" in captured.out
    assert "candidate comparison" in captured.out
    assert "best simple correction potential" in captured.out
    assert "LaTeX-ready candidate lemma" in captured.out


def test_low_gap_correction_pattern_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_correction_pattern_proof_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_correction_pattern_proof_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap correction-potential finite pattern proof report" in captured.out
    assert "compact pattern table" in captured.out
    assert "verification summary" in captured.out
    assert "LaTeX-ready lemmas" in captured.out


def test_low_gap_packet_convexity_schema_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    schema_rows, appendix = k5_low_gap_packet_convexity_schema_rows(h_max=1, max_used=0)
    assert schema_rows
    assert appendix

    print_k5_low_gap_packet_convexity_schema_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap packet-convexity schema compression report" in captured.out
    assert "reusable packet-convexity schemas" in captured.out
    assert "theorem-proof skeleton" in captured.out
    assert "appendix mapping" in captured.out


def test_low_gap_curvature_analytic_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_curvature_analytic_proof_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap curvature analytic proof reduction report" in captured.out
    assert "curvature definitions" in captured.out
    assert "Bellman recurrence attempt" in captured.out
    assert "LaTeX-ready proof skeleton" in captured.out


def test_low_gap_pattern_curvature_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_pattern_curvature_recurrence_rows(h_max=1, max_used=0)
    assert rows

    print_k5_low_gap_pattern_curvature_recurrence_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap pattern-curvature recurrence report" in captured.out
    assert "refined variable list" in captured.out
    assert "recurrence matrix/source terms" in captured.out
    assert "LaTeX-ready induction lemma template" in captured.out


def test_low_gap_bellman_image_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_bellman_image_recurrence_rows(h_max=1, max_used=0, focus="F", max_new_generators=3)
    assert rows
    assert all(row.image_measure or row.image_constant for row in rows)

    print_k5_low_gap_bellman_image_recurrence_report(h_max=1, max_used=0, focus="F", max_new_generators=3, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap signed-packet Bellman image recurrence report" in captured.out
    assert "Bellman images" in captured.out
    assert "recurrence edges" in captured.out
    assert "theorem-ready recurrence template" in captured.out


def test_multi_step_bellman_pattern_miner_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_multi_step_bellman_pattern_miner_report(
        h_max=1,
        max_used=0,
        max_new_generators=2,
        finite_steps=2,
        large_gap_h_values=(1,),
        large_gap_max_used=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 multi-step Bellman pattern miner" in captured.out
    assert "low-gap residual SCC taxonomy" in captured.out
    assert "multi-step closure tests" in captured.out
    assert "large-gap influence graph coverage" in captured.out
    assert "c=1/2 orbit cone status" in captured.out


def test_low_gap_flat_split_tail_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_flat_split_tail_proof_data(h_max=1, max_used=0, max_new_generators=3, tail_time=2)
    assert data.rows
    assert data.open_rows or data.closed_rows

    print_k5_low_gap_flat_split_tail_proof_report(h_max=1, max_used=0, max_new_generators=3, tail_time=2, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split tail proof report" in captured.out
    assert "unclosed generator" in captured.out
    assert "recurrence graph structure" in captured.out
    assert "c_i/t barrier check" in captured.out
    assert "LaTeX-ready flat-split lemma status" in captured.out


def test_low_gap_flat_split_tail_telescoping_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    chains = k5_low_gap_flat_split_tail_telescoping_chains(h_max=1, max_used=0, max_new_generators=3)
    assert isinstance(chains, tuple)

    print_k5_low_gap_flat_split_tail_telescoping_report(h_max=1, max_used=0, max_new_generators=3, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split affine-tail telescoping report" in captured.out
    assert "exact source partial sums" in captured.out
    assert "cancellation against F roots" in captured.out
    assert "coboundary correction" in captured.out
    assert "LaTeX-ready flat-split tail lemma candidate" in captured.out


def test_low_gap_flat_split_long_tail_limit_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_flat_split_long_tail_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        steps=3,
    )
    assert len(rows) == 3

    print_k5_low_gap_flat_split_long_tail_limit_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        steps=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split long-tail limit report" in captured.out
    assert "affine source sums" in captured.out
    assert "X_80 / X_81 endpoint" in captured.out
    assert "limiting signed measure proxy" in captured.out
    assert "LaTeX-ready tail lemma candidate" in captured.out


def test_low_gap_flat_split_tail_operator_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    certificate = k5_low_gap_flat_split_tail_operator_certificate(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        steps=3,
        max_power=2,
    )
    assert certificate.packets
    assert certificate.mass_preserved

    print_k5_low_gap_flat_split_tail_operator_certificate_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        steps=3,
        max_power=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split tail operator certificate report" in captured.out
    assert "exact operator B and source" in captured.out
    assert "fixed-point equations" in captured.out
    assert "contraction certificate" in captured.out
    assert "theorem-ready tail lemma" in captured.out


def test_low_gap_flat_split_tail_contraction_lemma_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    data = k5_low_gap_flat_split_tail_contraction_lemma_data(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        finite_steps=3,
        max_power=2,
    )
    assert data.source_bound_c >= 0
    assert data.poisson_potential

    print_k5_low_gap_flat_split_tail_contraction_lemma_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        start_variable="X_1",
        finite_steps=3,
        max_power=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split tail contraction lemma report" in captured.out
    assert "source functional" in captured.out
    assert "exact Poisson/coboundary tail" in captured.out
    assert "LaTeX-ready lemma" in captured.out


def test_low_gap_flat_split_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_flat_split_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
        finite_steps=3,
        max_power=2,
    )
    assert rows

    print_k5_low_gap_flat_split_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        finite_steps=3,
        max_power=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap flat-split convexity schema proof report" in captured.out
    assert "F-root classification table" in captured.out
    assert "finite-chain lemmas" in captured.out
    assert "long-tail Poisson lemma" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_adjacent_merge_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_adjacent_merge_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_adjacent_merge_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap adjacent-merge convexity schema proof report" in captured.out
    assert "M-root classification table" in captured.out
    assert "proof summary" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_packet_count_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_packet_count_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_packet_count_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap packet-count correction schema proof report" in captured.out
    assert "Q-root classification table" in captured.out
    assert "proof summary" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_finite_residual_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_finite_residual_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_finite_residual_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap finite residual pattern schema proof report" in captured.out
    assert "E-root classification table" in captured.out
    assert "proof summary" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_boundary_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_boundary_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_boundary_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap one-one-three boundary convexity schema proof report" in captured.out
    assert "B-root classification table" in captured.out
    assert "proof summary" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_higher_refinement_schema_proof_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_higher_refinement_schema_proof_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_higher_refinement_schema_proof_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap higher-refinement averaging schema proof report" in captured.out
    assert "H-root classification table" in captured.out
    assert "failures" in captured.out
    assert "proof summary" in captured.out
    assert "LaTeX-ready theorem" in captured.out


def test_low_gap_curvature_gap_action_grouping_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_curvature_gap_action_rows(h_max=1, max_used=0)
    assert isinstance(rows, tuple)

    print_k5_low_gap_curvature_gap_action_grouping_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap curvature-gap action grouping report" in captured.out
    assert "competing action types" in captured.out
    assert "curvature-gap inequalities by competing action" in captured.out
    assert "theorem-reader note" in captured.out


def test_low_gap_ordered_face_geometry_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_ordered_face_geometry_rows(h_max=1, max_used=0)
    assert isinstance(rows, tuple)

    print_k5_low_gap_ordered_face_geometry_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 low-gap ordered-face geometry report" in captured.out
    assert "ordered chamber convention" in captured.out
    assert "packet partitions as ordered-chamber faces" in captured.out
    assert "schema names versus face-transition types" in captured.out
    assert "geometric interpretation" in captured.out


def test_low_gap_discrete_maximum_principle_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_discrete_maximum_principle_components(
        h_max=1,
        max_used=0,
        max_new_generators=2,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_discrete_maximum_principle_report(
        h_max=1,
        max_used=0,
        max_new_generators=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap discrete maximum principle report" in captured.out
    assert "strongly connected components" in captured.out
    assert "boundary" in captured.out
    assert "discrete maximum principle lemma" in captured.out
    assert "certificate status" in captured.out


def test_low_gap_cyclic_component_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_cyclic_component_closure_data(
        h_max=1,
        max_used=0,
        max_new_generators=2,
        finite_steps=2,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_cyclic_component_closure_report(
        h_max=1,
        max_used=0,
        max_new_generators=2,
        finite_steps=2,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap cyclic component closure report" in captured.out
    assert "cyclic component node lists" in captured.out
    assert "internal recurrence rows" in captured.out
    assert "flat-split X-tail" in captured.out
    assert "theorem-ready statement" in captured.out


def test_low_gap_boundary_compression_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_boundary_compression_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_boundary_compression_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap boundary compression report" in captured.out
    assert "essential boundary rows" in captured.out
    assert "cone-dominated boundary rows removed" in captured.out
    assert "old schema rows implied" in captured.out
    assert "theorem-ready proof skeleton" in captured.out


def test_low_gap_boundary_atom_taxonomy_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_low_gap_boundary_atom_rows(
        h_max=1,
        max_used=0,
        seed_generators=3,
    )
    assert isinstance(rows, tuple)

    print_k5_low_gap_boundary_atom_taxonomy_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap boundary atom taxonomy report" in captured.out
    assert "atom taxonomy table" in captured.out
    assert "representative formulas" in captured.out
    assert "proposed human-readable lemmas" in captured.out
    assert "rows not explained by simple atoms" in captured.out


def test_low_gap_boundary_atom_symbolic_lemmas_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_boundary_atom_symbolic_lemmas_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap boundary atom symbolic lemmas report" in captured.out
    assert "six theorem-ready atom lemmas" in captured.out
    assert "exact row specializations" in captured.out
    assert "primitive assumptions" in captured.out
    assert "remaining gaps" in captured.out
    assert "paper-ready conclusion" in captured.out


def test_low_gap_packet_count_gap_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_packet_count_gap_closure_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap packet-count symbolic gap closure report" in captured.out
    assert "exact decompositions" in captured.out
    assert "generic symbolic formula" in captured.out
    assert "Packet-count-with-remainder atom" in captured.out
    assert "theorem-ready statements" in captured.out
    assert "remaining appendix-only rows" in captured.out


def test_low_gap_refinement_packet_count_fan_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_refinement_packet_count_fan_report(
        h_max=1,
        max_used=0,
        seed_generators=3,
        n=2,
    )

    captured = capsys.readouterr()
    assert "k5 low-gap refinement packet-count fan report" in captured.out
    assert "common structure table" in captured.out
    assert "ordered-gap face formulas" in captured.out
    assert "shared fan decomposition test" in captured.out
    assert "proposed seventh atom" in captured.out
    assert "appendix-only status" in captured.out


def test_low_gap_maximum_principle_proof_outline_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_low_gap_maximum_principle_proof_outline_report()

    captured = capsys.readouterr()
    assert "k5 low-gap maximum-principle proof outline" in captured.out
    assert "quotient balanced actions" in captured.out
    assert "define curvature gaps" in captured.out
    assert "ordered-chamber faces" in captured.out
    assert "packet-face recurrence graph" in captured.out
    assert "unique cyclic component" in captured.out
    assert "Poisson/coboundary" in captured.out
    assert "LaTeX-ready theorem statement" in captured.out


def test_low_gap_maximum_principle_replacement_section_printer_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_k5_low_gap_maximum_principle_replacement_section_report()

    captured = capsys.readouterr()
    assert r"\subsection{Low-gap packet-face maximum principle}" in captured.out
    assert "Ordered gaps and packet faces" in captured.out
    assert "Curvature gaps" in captured.out
    assert "low-gap packet-face maximum principle" in captured.out
    assert "Packet-curvature atom lemmas" in captured.out
    assert "refinement packet-count fan" in captured.out
    assert "Appendix-only finite fan rows" in captured.out
    assert "flat-split Poisson tail" in captured.out
    assert "Conclusion of the low-gap supersolution" in captured.out


def test_complement_balancing_symmetry_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_complement_balancing_symmetry_rows(h_max=1, max_used=0)
    assert rows
    assert all(row.same_quotient_class for row in rows)

    print_k5_complement_balancing_symmetry_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "k5 complement/balancing symmetry report" in captured.out
    assert "action complement quotient table" in captured.out
    assert "residual symmetry audit" in captured.out
    assert "low-gap schema reduction" in captured.out
    assert "balanced complement symmetry" in captured.out


def test_variable_c_harmonic_barrier_proof_export_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exports = k5_variable_c_harmonic_barrier_proof_exports(h_max=1, max_used=0)
    assert exports

    print_k5_variable_c_harmonic_barrier_proof_export_report(h_max=1, max_used=0, n=2)

    captured = capsys.readouterr()
    assert "variable-c harmonic barrier proof export" in captured.out
    assert "theorem constants" in captured.out
    assert "LaTeX-ready barrier constants" in captured.out
    assert "sampled inequalities" in captured.out
    assert "LaTeX-ready theorem statement" in captured.out


def test_direct_route_finite_quotient_validation_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    result = k5_direct_route_finite_quotient_validation_result(max_used=0, h_max=1, max_refinement_rounds=1)
    assert result.concrete_state_count >= 1

    print_k5_direct_route_finite_quotient_validation_report(max_used=0, h_max=1, max_refinement_rounds=1, n=2)

    captured = capsys.readouterr()
    assert "direct route finite quotient validation" in captured.out
    assert "quotient refinement" in captured.out
    assert "abstract-label harmonic barrier LP" in captured.out
    assert "c=1/2 augmented cone closure cross-check" in captured.out
    assert "verdict" in captured.out


def test_direct_route_large_gap_domination_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_direct_route_large_gap_domination_rows(h_max=1, max_used=1)
    assert rows

    print_k5_direct_route_large_gap_domination_report(h_max=1, max_used=1, n=2)

    captured = capsys.readouterr()
    assert "direct route large-gap domination" in captured.out
    assert "monotonicity summary" in captured.out
    assert "threshold tests" in captured.out
    assert "requested quotient-conflict families" in captured.out
    assert "theorem candidate" in captured.out


def test_direct_route_large_gap_influence_decay_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_direct_route_large_gap_influence_decay_rows(h_max=1, max_used=1)
    assert rows

    print_k5_direct_route_large_gap_influence_decay_report(h_max=1, max_used=1, n=2)

    captured = capsys.readouterr()
    assert "direct route large-gap influence decay" in captured.out
    assert "influence by gap value" in captured.out
    assert "envelope fits" in captured.out
    assert "boundary versus true large-gap effects" in captured.out
    assert "candidate tail lemma" in captured.out


def test_direct_route_large_gap_hitting_bound_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_direct_route_large_gap_hitting_bound_rows(h_max=1, max_used=1)
    assert isinstance(rows, tuple)

    print_k5_direct_route_large_gap_hitting_bound_report(h_max=1, max_used=1, n=2)

    captured = capsys.readouterr()
    assert "direct route large-gap hitting bound" in captured.out
    assert "influence versus hitting probability" in captured.out
    assert "family crossing probabilities" in captured.out
    assert "simple crossing envelopes" in captured.out
    assert "theorem candidate" in captured.out


def test_direct_route_large_gap_uniform_influence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_direct_route_large_gap_uniform_influence_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)

    print_k5_direct_route_large_gap_uniform_influence_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "direct route large-gap uniform influence" in captured.out
    assert "uniform square-scaled bound" in captured.out
    assert "by horizon" in captured.out
    assert "by gap value" in captured.out
    assert "verdict" in captured.out


def test_large_gap_one_leader_symbolic_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_large_gap_one_leader_symbolic_rows(h_max=1, max_a=2)
    assert rows

    print_k5_large_gap_one_leader_symbolic_report(h_max=1, max_a=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap one-leader symbolic" in captured.out
    assert "I_h(a) table" in captured.out
    assert "active actions realizing the differences" in captured.out
    assert "recurrence diagnostics" in captured.out
    assert "theorem candidate" in captured.out


def test_one_leader_influence_recursive_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_one_leader_influence_recursive_certificate_rows(h_max=1, max_a=2)
    assert rows

    print_k5_one_leader_influence_recursive_certificate_report(h_max=1, max_a=2, n=2)

    captured = capsys.readouterr()
    assert "k5 one-leader influence recursive certificate" in captured.out
    assert "active regime summary" in captured.out
    assert "exact Bellman expressions" in captured.out
    assert "worst constants" in captured.out
    assert "theorem-ready lemma" in captured.out


def test_large_gap_influence_cone_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows, fitted_b, feasible, _message = k5_large_gap_influence_cone_certificate_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)
    assert fitted_b
    assert isinstance(feasible, bool)

    print_k5_large_gap_influence_cone_certificate_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap influence cone certificate" in captured.out
    assert "fitted vector barrier constants" in captured.out
    assert "active regimes" in captured.out
    assert "recurrence inequalities" in captured.out
    assert "target theorem" in captured.out


def test_large_gap_influence_cone_source_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows, fitted_b, feasible, _message = k5_large_gap_influence_cone_source_certificate_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)
    assert fitted_b
    assert isinstance(feasible, bool)

    print_k5_large_gap_influence_cone_source_certificate_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap influence cone source certificate" in captured.out
    assert "source-aware vector barrier constants" in captured.out
    assert "worst active row" in captured.out
    assert "B_F=1 safe certificate" in captured.out
    assert "theorem-ready inequalities" in captured.out


def test_large_gap_influence_cone_proof_export_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows, constants, used_default = k5_large_gap_influence_cone_proof_export_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)
    assert constants
    assert isinstance(used_default, bool)

    print_k5_large_gap_influence_cone_proof_export_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap influence cone proof export" in captured.out
    assert "rational B_F constants" in captured.out
    assert "LaTeX-ready certificate table" in captured.out
    assert "verification" in captured.out


def test_large_gap_one_dimensional_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_large_gap_one_dimensional_barrier_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)

    print_k5_large_gap_one_dimensional_barrier_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap one-dimensional barrier report" in captured.out
    assert "family-level summary" in captured.out
    assert "one-dimensional regime table" in captured.out
    assert "theorem candidate" in captured.out


def test_large_gap_one_dimensional_failure_analysis_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows = k5_large_gap_one_dimensional_failure_analysis_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)

    print_k5_large_gap_one_dimensional_failure_analysis_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap one-dimensional barrier failure analysis" in captured.out
    assert "failure groups by state family" in captured.out
    assert "extension tests" in captured.out
    assert "smallest human-readable extension" in captured.out


def test_large_gap_multi_gap_potential_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rows, fitted_b, feasible, _message = k5_large_gap_multi_gap_potential_rows(h_max=1, max_used=2)
    assert isinstance(rows, tuple)
    assert len(fitted_b) == 4
    assert isinstance(feasible, bool)

    print_k5_large_gap_multi_gap_potential_report(h_max=1, max_used=2, n=2)

    captured = capsys.readouterr()
    assert "k5 large-gap multi-gap potential report" in captured.out
    assert "Psi=sum_i B/(g_i+1)^2" in captured.out
    assert "fitted gap-position constants" in captured.out
    assert "theorem candidate" in captured.out


def test_large_gap_barrier_audit_export_and_verify_runs(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    certificate_path = tmp_path / "large_gap_certificate.json"
    result = k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        export_certificate=str(certificate_path),
    )
    assert certificate_path.exists()
    assert result.target_inequalities >= result.closed_rows

    print_k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        export_certificate=str(certificate_path),
        n=2,
    )
    print_k5_large_gap_barrier_certificate_verify_report(str(certificate_path))

    captured = capsys.readouterr()
    assert "k5 large-gap barrier audit" in captured.out
    assert "large-gap region" in captured.out
    assert "barrier/potential" in captured.out
    assert "exact rational verification" in captured.out
    assert "k5 large-gap barrier certificate verifier" in captured.out


def test_large_gap_barrier_streaming_chunks_and_merge_runs(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    certificate_dir = tmp_path / "chunks"
    manifest_path = tmp_path / "manifest.json"
    direct_chunk_path = tmp_path / "direct_h1_F1.json"

    print_k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        export_certificate=str(manifest_path),
        certificate_dir=str(certificate_dir),
        streaming=True,
        families=("F1",),
        n=2,
    )
    print_k5_large_gap_barrier_merge_verify_report(
        manifest_path=str(manifest_path),
    )
    print_k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        export_certificate=str(direct_chunk_path),
        families=("F1",),
        row_mode="true-large-gap",
        n=1,
    )
    print_k5_large_gap_barrier_certificate_verify_report(str(direct_chunk_path))
    print_k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        row_mode="coverage",
        n=1,
    )
    print_k5_large_gap_barrier_audit_report(
        h_values=(1,),
        max_used=2,
        families=("F1",),
        row_mode="true-large-gap",
        dry_run=True,
        n=1,
    )

    captured = capsys.readouterr()
    assert "k5 large-gap barrier audit (streaming)" in captured.out
    assert "chunk summary" in captured.out
    assert "merge verifier" in captured.out
    assert "k5 large-gap barrier merge verifier" in captured.out
    assert "single chunk direct export" in captured.out
    assert "row_mode: coverage" in captured.out
    assert "dry_run: True" in captured.out
    assert "No target inequalities were materialized" in captured.out
    assert manifest_path.exists()
    assert direct_chunk_path.exists()
    assert any(certificate_dir.iterdir())


def test_k5_scalar_potential_exchangeability_rows_run() -> None:
    rows = k5_scalar_potential_exchangeability_rows(
        max_used=1,
        h_values=(2,),
        playout_names=("Q_chase",),
    )

    assert rows
    assert all(row.playout_name == "Q_chase" for row in rows)


def test_k5_scalar_potential_exchangeability_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_scalar_potential_exchangeability(
        max_used=1,
        h_values=(2,),
        playout_names=("Q_chase",),
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 scalar potential exchangeability" in captured.out
    assert "Top scalar potential order failures" in captured.out


def test_k5_boundary_correction_feature_extraction_is_deterministic() -> None:
    state = (2, 1, 1, 0, 0)

    assert _k5_boundary_feature_key(state, 3, "packet_zero") == _k5_boundary_feature_key(
        state,
        3,
        "packet_zero",
    )


def test_k5_boundary_correction_design_matrix_dimensions() -> None:
    matrix, defects, feature_keys, states, actions_a, actions_w = _k5_boundary_design_matrix(
        max_used=1,
        horizon=2,
        playout_name="Q_chase",
        feature_family="packet_type",
    )

    assert matrix.shape[0] == len(defects) == len(states) == len(actions_a) == len(actions_w)
    assert matrix.shape[1] == len(feature_keys)
    assert matrix.shape[0] > 0
    assert matrix.shape[1] > 0


def test_k5_boundary_correction_zero_weights_leave_defects_unchanged() -> None:
    matrix, defects, *_ = _k5_boundary_design_matrix(
        max_used=1,
        horizon=2,
        playout_name="Q_chase",
        feature_family="packet_type",
    )
    corrected = _apply_k5_boundary_correction(defects, matrix, np.zeros(matrix.shape[1]))

    assert corrected.tolist() == pytest.approx(defects.tolist())


def test_k5_boundary_correction_rows_run() -> None:
    rows = k5_boundary_correction_fit_rows(
        max_used=1,
        h_values=(2,),
        playout_names=("Q_chase",),
        feature_families=("packet_type",),
        n=2,
    )

    assert rows
    assert rows[0].sample_count > 0
    assert rows[0].feature_count > 0


def test_k5_boundary_correction_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_boundary_correction_fit_report(
        max_used=1,
        h_values=(2,),
        playout_names=("Q_chase",),
        feature_families=("packet_type",),
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 boundary correction fit" in captured.out
    assert "corrected max_abs_defect" in captured.out
    assert "top learned correction weights" in captured.out


def test_k5_relaxation_certificate_local_lp_returns_simplex() -> None:
    current_value, next_value = _k5_relaxation_value_functions("softmax_c1", 2)
    learner, alpha, action = _k5_relaxation_local_certificate_learner(
        (1, 0, 0, 0, 0),
        2,
        current_value,
        next_value,
        _k5_certificate_actions("all_binary"),
    )

    assert sum(learner) == pytest.approx(1.0)
    assert all(probability >= -1e-9 for probability in learner)
    assert alpha < float("inf")
    assert any(action)


def test_k5_relaxation_certificate_constant_potential_alpha_is_finite() -> None:
    def current_value(state: tuple[int, ...], remaining_horizon: int) -> float:
        del state, remaining_horizon
        return 0.0

    def next_value(state: tuple[int, ...], remaining_horizon: int) -> float:
        del state, remaining_horizon
        return 0.0

    _, alpha, _ = _k5_relaxation_local_certificate_learner(
        (0, 0, 0, 0, 0),
        2,
        current_value,
        next_value,
        _k5_certificate_actions("all_binary"),
    )

    assert alpha < float("inf")


def test_k5_relaxation_certificate_repaired_alpha_leq_natural_alpha() -> None:
    current_value, next_value = _k5_relaxation_value_functions("softmax_c1", 2)
    actions = _k5_certificate_actions("all_binary")
    state = (1, 0, 0, 0, 0)
    natural = _softmax_centered_gradient(state, 2 ** 0.5)
    natural_alpha = _k5_relaxation_natural_alpha(state, 2, natural, current_value, next_value, actions)
    _, repaired_alpha, _ = _k5_relaxation_local_certificate_learner(
        state,
        2,
        current_value,
        next_value,
        actions,
    )

    assert repaired_alpha <= natural_alpha + 1e-9


def test_k5_relaxation_certificate_rows_run() -> None:
    rows = k5_relaxation_certificate_rows(
        max_used=1,
        h_values=(2,),
        potential_names=("Q_chase", "softmax_c1"),
    )

    assert rows
    assert all(row.repaired_learner for row in rows)


def test_k5_relaxation_certificate_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_relaxation_certificate_report(
        max_used=1,
        h_values=(2,),
        potential_names=("Q_chase",),
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 relaxation certificate report" in captured.out
    assert "h*max" in captured.out
    assert "sqrt(h)*max" in captured.out
    assert "Root row summary" in captured.out
    assert "Top repaired failures" in captured.out


def test_k5_repaired_learner_rows_have_valid_p() -> None:
    rows = k5_repaired_learner_anatomy_rows(max_used=1, h_values=(2,), potential_name="Q_chase")

    assert rows
    for row in rows:
        assert sum(row.repaired_p) == pytest.approx(1.0)
        assert all(probability >= -1e-8 for probability in row.repaired_p)


def test_packet_average_reconstruction_is_constant_within_packets() -> None:
    state = (2, 2, 1, 0, 0)
    learner = (0.3, 0.1, 0.2, 0.25, 0.15)
    reconstructed, packet_averages, variance = _packet_average_reconstruction(state, learner)

    assert reconstructed[0] == pytest.approx(reconstructed[1])
    assert reconstructed[3] == pytest.approx(reconstructed[4])
    assert packet_averages == pytest.approx((0.2, 0.2, 0.2))
    assert variance > 0.0


def test_k5_repaired_packet_uniform_alpha_is_finite() -> None:
    row = k5_repaired_learner_anatomy_rows(max_used=1, h_values=(2,), potential_name="Q_chase")[0]

    assert row.packet_uniform_alpha < float("inf")


def test_k5_repaired_learner_anatomy_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_repaired_learner_anatomy_report(
        max_used=1,
        h_values=(2,),
        potential_name="Q_chase",
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 repaired learner anatomy" in captured.out
    assert "Packet-pattern templates" in captured.out
    assert "Simple learner hypothesis" in captured.out


def test_k5_packet_learner_weights_reconstruct_p_and_sum_to_one() -> None:
    state = (2, 2, 1, 0, 0)
    learner = (0.3, 0.3, 0.2, 0.1, 0.1)
    packet_weights = _packet_weights_from_packet_uniform_p(state, learner)
    reconstructed = _reconstruct_from_packet_weights(state, packet_weights)

    assert reconstructed == pytest.approx(learner)
    assert sum(reconstructed) == pytest.approx(1.0)


def test_k5_packet_learner_rationalized_learner_close_to_original() -> None:
    state = (1, 1, 0, 0, 0)
    packet_weights = (0.35, 0.1)
    rationalized = _rationalized_packet_learner(state, packet_weights)
    original = _reconstruct_from_packet_weights(state, packet_weights)

    assert rationalized == pytest.approx(original, abs=1e-9)


def test_k5_packet_learner_reconstructed_alpha_matches_full_on_sample() -> None:
    row = k5_packet_learner_formula_rows(max_used=1, h_values=(2,), potential_name="Q_chase")[0]

    assert row.source.packet_uniform_alpha == pytest.approx(row.source.repaired_alpha)


def test_k5_packet_learner_formula_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet_learner_formula_report(
        max_used=1,
        h_values=(2,),
        potential_name="Q_chase",
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 packet learner formula report" in captured.out
    assert "Rationalized packet-weight templates" in captured.out
    assert "Export proof table" in captured.out


def test_k5_reduced_packet_lp_matches_full_lp_on_small_sample() -> None:
    rows = k5_reduced_packet_lp_rows(max_used=1, h_values=(2,), potential_name="Q_chase")

    assert rows
    for row in rows:
        assert row.reduced_alpha == pytest.approx(row.full_alpha, abs=1e-8)


def test_k5_reduced_packet_lp_weights_reconstruct_simplex() -> None:
    row = k5_reduced_packet_lp_rows(max_used=1, h_values=(2,), potential_name="Q_chase")[0]
    learner = _reconstruct_from_packet_weights(row.state, row.reduced_weights)

    assert sum(learner) == pytest.approx(1.0)
    assert all(probability >= -1e-9 for probability in learner)


def test_k5_reduced_packet_next_state_invariant_to_packet_choice() -> None:
    state = (1, 1, 0, 0, 0)
    count_vector = (1, 2)
    representative_next = _next_state_from_packet_counts(state, count_vector)
    action = _representative_action_from_packet_counts(state, count_vector)
    alternative_action = (action[1], action[0], action[2], action[4], action[3])
    alternative_next = canon(tuple(state[index] + alternative_action[index] for index in range(5)))

    assert representative_next == alternative_next


def test_k5_reduced_packet_active_constraints_are_tight() -> None:
    current_value, next_value = _k5_relaxation_value_functions("Q_chase", 2)
    _, alpha, active, _ = _reduced_packet_lp((1, 0, 0, 0, 0), 2, current_value, next_value)

    assert active
    assert all(residual == pytest.approx(alpha, abs=1e-8) for _, residual, _ in active)


def test_k5_reduced_packet_lp_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_reduced_packet_lp_report(
        max_used=1,
        h_values=(2,),
        potential_name="Q_chase",
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 reduced packet LP report" in captured.out
    assert "Full LP vs reduced packet LP check" in captured.out
    assert "Active count-vector templates" in captured.out


def test_k5_reduced_packet_lp_coverage_small_matches_full() -> None:
    rows = k5_reduced_packet_lp_rows(max_used=1, h_values=(2,), potential_name="Q_chase")

    assert rows
    assert max(abs(row.alpha_difference) for row in rows) <= 1e-8


def test_k5_reduced_packet_lp_reachable_states_include_root_and_successor() -> None:
    states = set(_k5_chase_reachable_states(1))

    assert (0, 0, 0, 0, 0) in states
    assert canon((1, 0, 1, 0, 0)) in states


def test_k5_reduced_packet_lp_gap_bucket() -> None:
    assert _gap_bucket(0) == "0"
    assert _gap_bucket(5) == "5"
    assert _gap_bucket(8) == "6-10"
    assert _gap_bucket(11) == ">10"


def test_k5_reduced_packet_lp_coverage_rows_run() -> None:
    summaries, detail_rows = k5_reduced_packet_lp_coverage_rows(
        h_values=(2,),
        max_used_values=(1,),
        potential_name="Q_chase",
        reachable_mode="chase",
        reachable_T_values=(1,),
    )

    assert summaries
    assert detail_rows


def test_k5_reduced_packet_lp_coverage_parallel_rows_run() -> None:
    summaries, detail_rows = k5_reduced_packet_lp_coverage_rows(
        h_values=(2,),
        max_used_values=(1,),
        potential_name="Q_chase",
        reachable_mode="none",
        jobs=2,
    )

    assert summaries
    assert detail_rows


def test_k5_reduced_packet_lp_coverage_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_reduced_packet_lp_coverage_report(
        h_values=(2,),
        max_used_values=(1,),
        potential_name="Q_chase",
        reachable_mode="chase",
        reachable_T_values=(1,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 reduced packet LP coverage report" in captured.out
    assert "jobs:" in captured.out
    assert "global observed C" in captured.out
    assert "gap-size stratification" in captured.out


def test_k5_worst_family_scaling_packet_122_generator() -> None:
    state = _k5_worst_family_state("packet_122", 100, 1.1, 1.2)

    assert packet_type(state) == (1, 2, 2)


def test_k5_worst_family_scaling_grid_has_known_nearby_state() -> None:
    state = _k5_worst_family_state("packet_122", 100, 1.1, 1.2)

    assert state == (23, 12, 12, 0, 0)


def test_k5_worst_family_scaling_rows_run() -> None:
    rows = k5_worst_family_scaling_rows(
        h_values=(10,),
        families=("packet_122",),
        grid=(1.0,),
    )

    assert rows
    assert rows[0].active_count_vectors


def test_k5_worst_family_scaling_parallel_rows_run() -> None:
    rows = k5_worst_family_scaling_rows(
        h_values=(10, 12),
        families=("packet_122",),
        grid=(1.0,),
        jobs=2,
    )

    assert rows
    assert {row.horizon for row in rows} == {10, 12}


def test_k5_worst_family_scaling_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_worst_family_scaling_report(
        h_values=(10,),
        families=("packet_122",),
        grid=(1.0,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "k=5 worst-family scaling report" in captured.out
    assert "Candidate limiting worst point" in captured.out
    assert "Active-set stability" in captured.out


def test_k5_packet122_asymptotic_state_generator() -> None:
    state = _k5_packet122_state(100, 1.1, 1.2)

    assert state == (23, 12, 12, 0, 0)
    assert packet_type(state) == (1, 2, 2)


def test_k5_packet122_asymptotic_rows_run() -> None:
    rows = k5_packet122_asymptotic_equation_rows(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert rows
    row = rows[0]
    assert abs(row.predicted_weights[0] + 2 * row.predicted_weights[1] + 2 * row.predicted_weights[2] - 1.0) < 1e-9
    assert row.lp_weights
    assert row.active_count_vectors


def test_k5_packet122_asymptotic_prediction_close_on_sample() -> None:
    rows = k5_packet122_asymptotic_equation_rows(
        h_values=(12,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert rows
    row = rows[0]
    assert abs(row.predicted_alpha - row.lp_alpha) < 1e-3
    assert max(abs(left - right) for left, right in zip(row.predicted_weights, row.lp_weights, strict=True)) < 1e-3


def test_k5_packet122_asymptotic_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet122_asymptotic_equations_report(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "packet_122 asymptotic equations" in captured.out
    assert "two-equation predicted weights" in captured.out
    assert "Prediction error" in captured.out


def test_k5_packet122_q_asymptotic_rows_formula_matches_lp_on_sample() -> None:
    rows = k5_packet122_q_asymptotic_rows(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert rows
    row = rows[0]
    formula_alpha = (row.q_top + 2.0 * row.q_mid - 1.0) / 3.0
    assert abs(formula_alpha - row.lp_alpha) < 1e-9


def test_k5_packet122_q_asymptotic_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet122_q_asymptotic_report(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "packet_122 q asymptotic report" in captured.out
    assert "Formula alpha versus LP alpha" in captured.out
    assert "Proof-ready table" in captured.out


def test_k5_q_convention_audit_parser_runs() -> None:
    cases = _parse_k5_q_audit_states("10:6,3,3,0,0")

    assert len(cases) == 1
    assert cases[0].horizon == 10
    assert cases[0].state == (6, 3, 3, 0, 0)


def test_rank_chase_formula_self_consistent() -> None:
    state = (6, 3, 3, 0, 0)
    horizon = 10

    assert abs(
        _rank_chase_future_uncentered_value(state, horizon)
        - (_rank_chase_future_centered_value(state, horizon) + 0.5 * horizon)
    ) < 1e-12


def test_k5_q_convention_audit_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_q_convention_audit(
        cases=_parse_k5_q_audit_states("10:6,3,3,0,0"),
    )

    captured = capsys.readouterr()
    assert "k=5 q convention audit" in captured.out
    assert "code-path q values" in captured.out
    assert "rank-wise scalar formula" in captured.out
    assert "final diagnosis" in captured.out


def test_min_normalized_descending_example() -> None:
    assert _min_normalized_descending((50, 38, 37, 1, 1)) == (49, 37, 36, 0, 0)


def test_normalized_scalar_value_translation_invariant() -> None:
    state = (50, 38, 37, 1, 1)
    shifted = tuple(value + 7 for value in state)

    assert _min_normalized_descending(state) == _min_normalized_descending(shifted)
    assert abs(
        _rank_chase_future_centered_value(_min_normalized_descending(state), 10)
        - _rank_chase_future_centered_value(_min_normalized_descending(shifted), 10)
    ) < 1e-12


def test_k5_normalized_scalar_convention_audit_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_normalized_scalar_convention_audit(
        cases=_parse_k5_q_audit_states("10:6,3,3,0,0"),
    )

    captured = capsys.readouterr()
    assert "k=5 normalized scalar convention audit" in captured.out
    assert "MATCH FOUND" in captured.out or "no normalized scalar variant matched" in captured.out


def test_k5_rank_adaptive_dp_recursion_residual_small() -> None:
    rows = k5_rank_adaptive_dp_recursion_audit_rows(
        _parse_k5_q_audit_states("10:6,3,3,0,0"),
    )

    assert rows
    row = rows[0]
    assert abs(row.recursion_residual) < 1e-10


def test_k5_rank_adaptive_dp_q_expr_decomposition_matches() -> None:
    rows = k5_rank_adaptive_dp_recursion_audit_rows(
        _parse_k5_q_audit_states("10:6,3,3,0,0"),
    )

    assert rows
    row = rows[0]
    assert abs(row.q_expr - row.decomposed_q_expr) < 1e-10


def test_k5_rank_adaptive_dp_recursion_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_rank_adaptive_dp_recursion_audit(
        cases=_parse_k5_q_audit_states("10:6,3,3,0,0"),
    )

    captured = capsys.readouterr()
    assert "rank-adaptive DP recursion audit" in captured.out
    assert "q_expr decomposition" in captured.out
    assert "proof-ready formula" in captured.out


def test_k5_rank_adaptive_curvature_formula_matches_decomposition() -> None:
    rows = k5_rank_adaptive_curvature_rows(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert rows
    row = rows[0]
    assert abs(row.q_expr_direct - row.q_expr_curvature) < 1e-10


def test_k5_rank_adaptive_curvature_known_h150_constant() -> None:
    rows = k5_rank_adaptive_curvature_rows(
        h_values=(150,),
        s1_values=(1.1,),
        s2_values=(3.0,),
        compare_lp=True,
    )

    assert rows
    row = rows[0]
    assert row.state == (50, 37, 37, 0, 0)
    assert abs(row.scaled_c - 1.327967) < 5e-4
    assert row.lp_scaled_c is not None
    assert abs(row.scaled_c - row.lp_scaled_c) < 5e-4


def test_k5_rank_adaptive_curvature_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_rank_adaptive_curvature_report(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "rank-adaptive curvature report" in captured.out
    assert "D1 = N_T - N_P" in captured.out
    assert "max C" in captured.out


def test_k5_packet122_curvature_growth_reproduces_h200_known_point() -> None:
    summaries = k5_packet122_curvature_growth_summaries(
        h_values=(200,),
        s1_values=(0.9,),
        s2_values=(1.5,),
    )

    assert summaries
    summary = summaries[0]
    assert summary.best_row.state == (34, 21, 21, 0, 0)
    assert abs(summary.best_row.scaled_c - 1.561969) < 5e-4


def test_k5_packet122_curvature_growth_alpha_sqrt_h_computed() -> None:
    summaries = k5_packet122_curvature_growth_summaries(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert summaries
    summary = summaries[0]
    assert abs(summary.alpha_sqrt_h - summary.alpha * (summary.horizon ** 0.5)) < 1e-12


def test_k5_packet122_curvature_growth_parallel_run() -> None:
    summaries, validations = k5_packet122_curvature_growth_run(
        h_values=(10, 12),
        s1_values=(1.0,),
        s2_values=(1.0,),
        compare_lp=True,
        lp_top_n=1,
        jobs=2,
    )

    assert {summary.horizon for summary in summaries} == {10, 12}
    assert {row.horizon for row in validations} == {10, 12}


def test_k5_packet122_curvature_growth_compare_lp_known_h150() -> None:
    rows = k5_rank_adaptive_curvature_rows(
        h_values=(150,),
        s1_values=(1.1,),
        s2_values=(3.0,),
        compare_lp=True,
    )

    assert rows
    row = rows[0]
    assert row.lp_alpha is not None
    assert abs(row.lp_alpha - row.q_expr_curvature / 3.0) < 1e-7


def test_k5_packet122_curvature_growth_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet122_curvature_growth_report(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
        compare_lp=True,
        lp_top_n=1,
    )

    captured = capsys.readouterr()
    assert "packet122 curvature growth report" in captured.out
    assert "LP-validated ridge" in captured.out
    assert "active_constraints" in captured.out
    assert "log-log beta" in captured.out
    assert "alpha_sqrt_h" in captured.out


def test_k5_packet122_active_regime_detects_formula_invalid_row() -> None:
    rows = k5_packet122_active_regime_rows(
        h_values=(12,),
        s1_values=(1.0,),
        s2_values=(1.0,),
    )

    assert rows
    assert not rows[0].formula_valid


def test_k5_packet122_active_regime_detects_formula_valid_ridge_row() -> None:
    rows = k5_packet122_active_regime_rows(
        h_values=(150,),
        s1_values=(1.1,),
        s2_values=(3.0,),
    )

    assert rows
    assert rows[0].state == (50, 37, 37, 0, 0)
    assert rows[0].formula_valid


def test_k5_packet122_active_regime_parallel_run() -> None:
    rows = k5_packet122_active_regime_run(
        h_values=(10, 12),
        s1_values=(1.0,),
        s2_values=(1.0,),
        jobs=2,
    )

    assert rows
    assert {row.horizon for row in rows} == {10, 12}


def test_k5_packet122_active_regime_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet122_active_regime_map(
        h_values=(10,),
        s1_values=(1.0,),
        s2_values=(1.0,),
        n=2,
    )

    captured = capsys.readouterr()
    assert "packet122 active regime map" in captured.out
    assert "Formula validity map" in captured.out
    assert "Best LP C by h" in captured.out


def test_k5_packet122_lp_state_probe_tiny_smoke() -> None:
    rows = k5_packet122_lp_state_probe_rows(
        _parse_k5_q_audit_states("20:9,5,5,0,0"),
        clear_cache_between_states=True,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.packet_type == (1, 2, 2)
    assert row.lp_alpha == pytest.approx(row.lp_alpha)
    assert row.active_count_vectors


def test_k5_packet122_lp_state_probe_known_h150_formula_valid() -> None:
    rows = k5_packet122_lp_state_probe_rows(
        _parse_k5_q_audit_states("150:50,37,37,0,0"),
        clear_cache_between_states=True,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.formula_valid
    assert abs(row.lp_alpha - row.formula_alpha) <= 1e-7


def test_k5_packet122_lp_state_probe_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_packet122_lp_state_probe(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
        clear_cache_between_states=True,
    )

    captured = capsys.readouterr()
    assert "packet122 LP state probe" in captured.out
    assert "formula_valid" in captured.out
    assert "active_constraints" in captured.out


def test_k5_top3_transition_check_matches_full_embedding() -> None:
    rows = k5_top3_transition_check_rows(((0, 0), (1, 0), (0, 1), (2, 3)), bottom_gap=20)

    assert rows
    for row in rows:
        assert row.plus_top3_gap == row.plus_expected_gap
        assert row.minus_top3_gap == row.minus_expected_gap


def test_k5_top3_reflected_chain_rows_run() -> None:
    rows = k5_top3_reflected_chain_rows(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.n == 19
    assert row.a == 4
    assert row.k3 == pytest.approx(row.k3)
    assert row.full_alpha == pytest.approx(row.full_alpha)


def test_k5_top3_reflected_chain_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_reflected_chain_report(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
        transition_pairs=((0, 0), (1, 0)),
    )

    captured = capsys.readouterr()
    assert "top3 reflected chain report" in captured.out
    assert "Transition verification" in captured.out
    assert "Proof-ready formula" in captured.out


def test_k5_top3_local_time_formula_equals_dp() -> None:
    rows = k5_top3_local_time_identity_rows(
        n_values=(0, 1, 5, 10),
        pairs=((0, 0), (1, 0), (0, 1), (4, 1)),
    )

    assert rows
    for row in rows:
        assert abs(row.residual) < 1e-12


def test_k5_top3_local_time_curvature_equals_dp() -> None:
    rows = k5_top3_local_time_curvature_rows(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
    )

    assert len(rows) == 1
    assert abs(rows[0].residual) < 1e-12


def test_k5_top3_local_time_identity_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_local_time_identity_report(
        n_values=(5,),
        pairs=((0, 0), (1, 0)),
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
    )

    captured = capsys.readouterr()
    assert "top3 local-time identity report" in captured.out
    assert "pathwise formula" in captured.out
    assert "K3 local-time curvature" in captured.out


def test_k5_top3_m_recurrence_matches_local_time_dp() -> None:
    for n in range(0, 5):
        for r, s in ((0, 0), (1, 0), (0, 1), (3, 2)):
            assert _top3_m_recurrence_value(n, r, s) == pytest.approx(_top3_m_value(n + 1, r, s))


def test_k5_top3_k_recurrence_matches_direct_k() -> None:
    for n in range(0, 5):
        for a in (1, 2, 4):
            assert _top3_k_recurrence_value(n, a) == pytest.approx(_top3_k_from_m(n + 1, a))


def test_k5_top3_k_recurrence_rows_run() -> None:
    rows = k5_top3_k_recurrence_rows(n_values=(1, 2, 5), max_a=4)

    assert rows
    for row in rows:
        assert abs(row.residual) < 1e-12


def test_k5_top3_k_recurrence_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_k_recurrence_report(
        max_n=10,
        sample_n_values=(1, 2, 5, 10),
        n=3,
    )

    captured = capsys.readouterr()
    assert "top3 K recurrence report" in captured.out
    assert "candidate barrier" in captured.out
    assert "LaTeX-ready recurrence" in captured.out


def test_k5_top3_difference_closure_reproduces_k_expansion() -> None:
    expanded = _top3_expand_stencil(_top3_candidate_stencils()["K"])

    assert expanded[(2, 1)] == pytest.approx(0.5)
    assert expanded[(1, 0)] == pytest.approx(0.5)
    assert expanded[(0, 0)] == pytest.approx(0.25)
    assert expanded[(0, -2)] == pytest.approx(0.25)
    assert expanded[(1, 1)] == pytest.approx(-0.75)
    assert expanded[(1, -1)] == pytest.approx(-0.75)


def test_k5_top3_difference_closure_k_matches_recurrence_report() -> None:
    stencil = _top3_candidate_stencils()["K"]

    assert _top3_stencil_value(stencil, 10, 4) == pytest.approx(_top3_k_from_m(10, 4))


def test_k5_top3_difference_family_summaries_run() -> None:
    summaries = k5_top3_difference_family_summaries(max_n=10, min_a=3)

    assert summaries
    assert any(summary.name == "K" for summary in summaries)


def test_k5_top3_difference_closure_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_difference_closure_report(max_n=10, min_a=3, n=3)

    captured = capsys.readouterr()
    assert "top3 difference closure report" in captured.out
    assert "candidate finite-difference family" in captured.out
    assert "closure attempt" in captured.out


def test_k5_top3_hitting_identity_matches_m() -> None:
    rows = k5_top3_hitting_identity_rows(
        n_values=(0, 1, 5, 10),
        pairs=((0, 0), (1, 0), (0, 1), (4, 1)),
    )

    assert rows
    for row in rows:
        assert row.m_value == pytest.approx(row.half_hitting_sum)


def test_k5_top3_hitting_kernel_matches_k() -> None:
    rows = k5_top3_hitting_kernel_rows(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.k_value == pytest.approx(row.half_kernel_sum)


def test_k5_top3_kernel_term_sums_to_k() -> None:
    n_value = 10
    a = 4
    half_sum = 0.5 * sum(_top3_kernel_term(t, a) for t in range(n_value))

    assert half_sum == pytest.approx(_top3_k_from_m(n_value, a))


def test_k5_top3_hitting_kernel_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_hitting_kernel_report(
        cases=_parse_k5_q_audit_states("20:9,5,5,0,0"),
        n_values=(5,),
        pairs=((0, 0), (1, 0)),
        n=3,
    )

    captured = capsys.readouterr()
    assert "top3 hitting kernel report" in captured.out
    assert "K as hitting-probability sum" in captured.out
    assert "telescoping search" in captured.out


def test_k5_top3_kernel_pairing_reconstructs_k() -> None:
    rows = k5_top3_kernel_pairing_rows(cases=((20, 4),))

    assert len(rows) == 1
    row = rows[0]
    assert row.k_value == pytest.approx(row.half_kernel_sum)
    assert row.pair_reconstruction_residual == pytest.approx(0.0)
    assert row.k_value == pytest.approx(row.even_half_sum + row.odd_half_sum)


def test_k5_top3_kernel_pairing_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_kernel_pairing_report(cases=((20, 4),), top_count=3)

    captured = capsys.readouterr()
    assert "top3 kernel pairing report" in captured.out
    assert "adjacent-pair cancellation" in captured.out
    assert "parity-separated sums" in captured.out


def test_k5_top3_telescoping_identity_residual_is_zero() -> None:
    rows = k5_top3_telescoping_candidate_rows(a_values=(1, 2, 4), max_t=20)
    by_name = {row.name: row for row in rows}

    assert by_name["I2: pair_t = A_{t+2} - A_t"].max_abs_residual == pytest.approx(0.0)
    assert by_name["I1: pair_t = A_{t+1} - A_t"].max_abs_residual > 0.0


def test_k5_top3_parity_telescope_matches_direct_k() -> None:
    rows = k5_top3_parity_telescope_rows(a_values=(1, 3, 4), max_n=40)

    assert rows
    for row in rows:
        assert row.k_value == pytest.approx(row.telescope_value)


def test_k5_top3_telescoping_identity_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_telescoping_identity_report(
        a_values=(1, 2, 4),
        identity_max_t=12,
        parity_max_n=20,
        barrier_max_n=40,
        barrier_a_values=(1, 2, 4),
    )

    captured = capsys.readouterr()
    assert "top3 telescoping identity report" in captured.out
    assert "exact adjacent-pair identity" in captured.out
    assert "parity telescope" in captured.out
    assert "endpoint barrier" in captured.out


def test_k5_top3_endpoint_barrier_recurrences_are_exact() -> None:
    rows = k5_top3_endpoint_recurrence_rows(a_values=(2, 3, 4), max_n=20)

    assert rows
    for row in rows:
        assert row.max_abs_residual == pytest.approx(0.0)


def test_k5_top3_endpoint_barrier_maxima_match_known_values() -> None:
    rows = {row.name: row for row in k5_top3_endpoint_barrier_rows(max_n=1000)}

    assert rows["A"].sign_holds is True
    assert rows["k"].sign_holds is True
    assert rows["A"].max_scaled == pytest.approx(1.0)
    assert rows["A"].argmax_n == 1
    assert rows["A"].argmax_a == 1
    assert rows["k"].max_scaled == pytest.approx(1.5)
    assert rows["k"].argmax_n == 1
    assert rows["k"].argmax_a == 1


def test_k5_top3_endpoint_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_barrier_report(
        a_values=(1, 2, 4),
        recurrence_max_n=12,
        max_n=40,
    )

    captured = capsys.readouterr()
    assert "top3 endpoint barrier report" in captured.out
    assert "A via |A_{n+1}|" in captured.out
    assert "k via interior k_{n+1} recurrence" in captured.out
    assert "induction does not close" in captured.out


def test_k5_top3_endpoint_coupling_transition_graph_agrees_with_direct_chain() -> None:
    rows = {
        (row.family, row.discrepancy): row
        for row in k5_top3_endpoint_coupling_graph_rows(max_steps=10)
    }

    assert rows[("A/B-adjacent", (1, -1))].plus_discrepancies == ((0, 1), (1, -1))
    assert rows[("A/B-adjacent", (1, -1))].minus_discrepancies == ((-1, 0), (1, -1))
    assert rows[("A/B-adjacent", (0, 1))].minus_discrepancies == ((0, 1),)
    assert rows[("A/B-adjacent", (-1, 0))].plus_discrepancies == ((-1, 0),)
    assert rows[("C-adjacent", (1, 0))].plus_discrepancies == ((1, 0),)
    assert rows[("C-adjacent", (1, 0))].minus_discrepancies == ((-1, 1), (1, 0))
    assert rows[("C-adjacent", (0, -1))].minus_discrepancies == ((0, -1),)
    assert rows[("C-adjacent", (-1, 1))].plus_discrepancies == ((-1, 1), (0, -1))


def test_k5_top3_dense_endpoint_barrier_reproduces_known_maxima() -> None:
    rows = {row.name: row for row in k5_top3_dense_endpoint_barrier_rows(max_n=1000)}

    assert rows["A"].sign_holds is True
    assert rows["k"].sign_holds is True
    assert rows["A"].max_scaled == pytest.approx(1.0)
    assert rows["A"].argmax_n == 1
    assert rows["A"].argmax_a == 1
    assert rows["k"].max_scaled == pytest.approx(1.5)
    assert rows["k"].argmax_n == 1
    assert rows["k"].argmax_a == 1


def test_k5_top3_dense_endpoint_barrier_accepts_a_limit() -> None:
    rows = {row.name: row for row in k5_top3_dense_endpoint_barrier_rows(max_n=20, max_a_per_n=3)}

    assert set(rows) == {"A", "k"}
    assert rows["A"].argmax_a is not None
    assert rows["A"].argmax_a <= 3
    assert rows["k"].argmax_a is not None
    assert rows["k"].argmax_a <= 3


def test_k5_top3_endpoint_coupling_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_coupling_report(
        a_values=(1, 2, 4),
        n_values=(0, 1, 2, 5),
        graph_max_steps=8,
        max_n=40,
    )

    captured = capsys.readouterr()
    assert "top3 endpoint coupling report" in captured.out
    assert "discrepancy transition graph" in captured.out
    assert "dense endpoint barrier" in captured.out


def test_k5_top3_endpoint_coupling_can_skip_dense_barrier(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_coupling_report(
        a_values=(1, 2),
        n_values=(0, 1),
        graph_max_steps=4,
        max_n=20,
        skip_dense_barrier=True,
        clear_top3_cache_after=True,
    )

    captured = capsys.readouterr()
    assert "top3 endpoint coupling report" in captured.out
    assert "skipped by --skip-dense-barrier" in captured.out


def test_k5_top3_discrepancy_chain_labels_cover_three_states() -> None:
    labels = _top3_endpoint_graph_label_summary()
    ab_labels = {label for family, _, label in labels if family == "A/B-adjacent"}
    c_labels = {label for family, _, label in labels if family == "C-adjacent"}

    assert "adjacent along same spread level" in ab_labels
    assert "adjacent across spread levels" in ab_labels
    assert "reflected discrepancy" in ab_labels
    assert "adjacent along same spread level" in c_labels
    assert "adjacent across spread levels" in c_labels
    assert "reflected discrepancy" in c_labels


def test_k5_top3_ballot_candidate_rows_run() -> None:
    rows = k5_top3_ballot_candidate_rows(a_values=(1, 2), n_values=(0, 1, 2))

    assert rows
    assert all(row.best_candidate for row in rows)
    assert all(row.best_abs_error >= 0.0 for row in rows)


def test_k5_top3_discrepancy_chain_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_discrepancy_chain_symbolic_report(
        a_values=(1, 2),
        n_values=(0, 1, 2),
        graph_max_steps=4,
        dense_max_n=20,
        n=8,
    )

    captured = capsys.readouterr()
    assert "discrepancy chain symbolic report" in captured.out
    assert "signed endpoint formula" in captured.out
    assert "candidate ballot formula" in captured.out


def test_k5_top3_discrepancy_generating_sequences_match_direct() -> None:
    sequence = _fraction_sequence_for_endpoint("A", 1, 6)

    assert len(sequence) == 6
    for n_value, value in enumerate(sequence):
        assert float(value) == pytest.approx(_top3_a_term(n_value, 1))


def test_k5_top3_discrepancy_generating_recurrence_rows_validate() -> None:
    rows = k5_top3_discrepancy_generating_recurrence_rows(
        a_values=(1,),
        fit_terms=20,
        validate_terms=40,
        max_order=8,
    )

    assert rows
    for row in rows:
        if row.order is not None:
            assert row.max_validate_residual == 0


def test_k5_top3_discrepancy_generating_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_discrepancy_generating_function_report(
        a_values=(1,),
        n_values=(0, 1, 2),
        fit_terms=20,
        validate_terms=40,
        max_order=8,
        n=12,
    )

    captured = capsys.readouterr()
    assert "discrepancy generating function report" in captured.out
    assert "guessed recurrence" in captured.out
    assert "endpoint coefficient bound" in captured.out


def test_k5_top3_discrepancy_barrier_signed_transforms_nonnegative() -> None:
    for n_value in range(20):
        for a in range(1, n_value + 3):
            assert _top3_signed_endpoint_value("A", n_value, a) >= -1e-15
            assert _top3_signed_endpoint_value("k", n_value, a) >= -1e-15


def test_k5_top3_discrepancy_barrier_reproduces_endpoint_maxima() -> None:
    rows = {
        (row.name, row.barrier): row
        for row in k5_top3_discrepancy_barrier_certificate_rows(
            max_n=40,
            dense_a_limit=20,
            barriers=("n",),
        )
    }

    assert rows[("A", "n")].constant == pytest.approx(1.0)
    assert rows[("k", "n")].constant == pytest.approx(1.5)


def test_k5_top3_discrepancy_barrier_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_discrepancy_barrier_certificate_report(
        max_n=40,
        dense_a_limit=20,
        recurrence_max_n=10,
        recurrence_a_values=(1, 2, 3),
        n=8,
    )

    captured = capsys.readouterr()
    assert "discrepancy barrier certificate" in captured.out
    assert "signed recurrence" in captured.out
    assert "super-solution" in captured.out
    assert "worst slack" in captured.out


def test_k5_top3_endpoint_induction_reproduces_value_scan() -> None:
    barrier_rows = {
        (row.name, row.barrier): row
        for row in k5_top3_discrepancy_barrier_certificate_rows(
            max_n=40,
            dense_a_limit=20,
            barriers=("n",),
        )
    }
    induction_rows = {
        (row.target, row.barrier): row
        for row in k5_top3_endpoint_induction_certificate_rows(
            max_n=40,
            dense_a_limit=20,
            barrier="n",
        )
    }

    assert induction_rows[("A", "n")].constant == pytest.approx(barrier_rows[("A", "n")].constant)


def test_k5_top3_endpoint_induction_distinguishes_value_and_step_barriers() -> None:
    rows = k5_top3_endpoint_induction_certificate_rows(
        max_n=40,
        dense_a_limit=20,
        barrier="n",
    )

    assert rows
    assert any((not row.closes) or row.obstruction != "none" for row in rows)
    assert all(row.exact_worst_slack >= -1e-12 for row in rows)


def test_k5_top3_endpoint_induction_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_induction_certificate_report(
        max_n=40,
        dense_a_limit=20,
        barriers=("n",),
        n=8,
    )

    captured = capsys.readouterr()
    assert "endpoint induction certificate" in captured.out
    assert "true induction inequality" in captured.out
    assert "worst induction slack" in captured.out


def test_k5_top3_refined_discrepancy_barrier_reconstructs_aggregate() -> None:
    for aggregate in ("A", "B", "C"):
        for n_value in range(8):
            for a in range(1, n_value + 3):
                assert _top3_refined_label_aggregate_value(
                    aggregate,
                    n_value,
                    a,
                ) == pytest.approx(_top3_refined_aggregate_direct_value(aggregate, n_value, a))


def test_k5_top3_refined_discrepancy_barrier_exposes_c_obstruction() -> None:
    rows = k5_top3_refined_discrepancy_label_scan_rows(max_n=20, dense_a_limit=12)

    assert rows
    assert any(row.aggregate == "C" for row in rows)
    assert any(row.first_negative is not None for row in rows if row.aggregate == "C")


def test_k5_top3_refined_discrepancy_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_refined_discrepancy_barrier_report(
        max_n=20,
        dense_a_limit=12,
        barriers=("n",),
        recurrence_max_n=8,
        n=8,
    )

    captured = capsys.readouterr()
    assert "refined discrepancy barrier" in captured.out
    assert "label-level recurrence" in captured.out
    assert "induction slack" in captured.out
    assert "reconstructs aggregate" in captured.out


def test_k5_top3_endpoint_cone_reconstructs_targets() -> None:
    specs = _top3_cone_form_specs()

    for n_value in range(8):
        for a in range(1, n_value + 3):
            assert _top3_cone_form_value(specs["Aplus"], n_value, a) == pytest.approx(
                _top3_signed_endpoint_value("A", n_value, a)
            )
            assert _top3_cone_form_value(specs["kplus"], n_value, a) == pytest.approx(
                _top3_signed_endpoint_value("k", n_value, a)
            )


def test_k5_top3_endpoint_cone_reproduces_label_box_failures() -> None:
    rows = {
        row.name: row
        for row in k5_top3_endpoint_cone_form_scan_rows(
            max_n=20,
            dense_a_limit=12,
        )
    }

    assert rows["Aplus"].first_negative is None
    assert rows["kplus"].first_negative is None
    assert rows["Cplus"].first_negative is not None


def test_k5_top3_endpoint_cone_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_cone_certificate_report(
        max_n=20,
        dense_a_limit=12,
        closure_max_n=8,
        n=8,
    )

    captured = capsys.readouterr()
    assert "endpoint cone certificate" in captured.out
    assert "candidate linear forms" in captured.out
    assert "cone closure" in captured.out
    assert "inductive super-solution" in captured.out


def test_k5_top3_endpoint_cone_expansion_reproduces_previous_fit() -> None:
    rows, residuals = k5_top3_endpoint_cone_expansion_rows(
        max_n=20,
        dense_a_limit=12,
        closure_max_n=8,
    )

    assert rows
    assert residuals
    assert any(row.basis_name == "base" and row.target == "kplus" for row in rows)
    assert any(row.basis_name == "expanded" and row.target == "kplus" for row in rows)


def test_k5_top3_endpoint_cone_expansion_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_cone_expansion_report(
        max_n=20,
        dense_a_limit=12,
        closure_max_n=8,
        n=8,
    )

    captured = capsys.readouterr()
    assert "endpoint cone expansion" in captured.out
    assert "residual-driven forms" in captured.out
    assert "true induction closure" in captured.out
    assert "weaker constants" in captured.out


def test_k5_top3_kplus_companion_reproduces_constants() -> None:
    rows = {
        row.name: row
        for row in k5_top3_kplus_companion_scan_rows(
            max_n=20,
            dense_a_limit=12,
        )
    }

    assert rows["A"].max_scaled == pytest.approx(1.0)
    assert rows["K"].max_scaled == pytest.approx(1.5)
    assert rows["Q1"].first_negative is None
    assert rows["Q3"].first_negative is None


def test_k5_top3_kplus_companion_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_kplus_companion_barrier_report(
        max_n=20,
        dense_a_limit=12,
        closure_max_n=8,
        n=8,
    )

    captured = capsys.readouterr()
    assert "kplus companion barrier" in captured.out
    assert "Q1 = kplus - Aplus" in captured.out
    assert "Q3 = kplus - 3/2 Aplus" in captured.out
    assert "K follows" in captured.out


def test_k5_top3_endpoint_exact_image_rows_are_exact() -> None:
    rows = _top3_endpoint_exact_image_rows(max_n=20, dense_a_limit=12)

    assert rows
    assert {row.name for row in rows} == {"A", "K", "Q3", "Q1", "B", "S"}
    for row in rows:
        assert row.max_abs_residual < 1e-12


def test_k5_top3_endpoint_exact_image_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_exact_image_report(
        max_n=20,
        dense_a_limit=12,
    )

    captured = capsys.readouterr()
    assert "endpoint exact image report" in captured.out
    assert "K image" in captured.out
    assert "missing residual term" in captured.out
    assert "boundary source" in captured.out


def test_k5_top3_diagonal_discrepancy_matches_a_and_d() -> None:
    for n_value in range(8):
        for a in range(1, n_value + 3):
            assert _top3_diagonal_discrepancy_value(0, n_value, a) == pytest.approx(
                _top3_signed_endpoint_value("A", n_value, a)
            )
            assert _top3_diagonal_discrepancy_value(1, n_value, a) == pytest.approx(
                _top3_signed_endpoint_value("D", n_value, a)
            )


def test_k5_top3_diagonal_discrepancy_recurrence_rows_are_exact() -> None:
    rows = k5_top3_diagonal_discrepancy_recurrence_rows(
        max_n=20,
        dense_a_limit=12,
        max_j=4,
    )

    assert rows
    for row in rows:
        assert row.max_abs_residual < 1e-12


def test_k5_top3_diagonal_discrepancy_k_image_residual_is_zero() -> None:
    for n_value in range(8):
        for a in range(2, n_value + 3):
            assert _top3_diagonal_k_image_e1_residual(n_value, a) == pytest.approx(0.0)


def test_k5_top3_diagonal_discrepancy_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_diagonal_discrepancy_hierarchy_report(
        max_n=20,
        dense_a_limit=12,
        max_j=4,
    )

    captured = capsys.readouterr()
    assert "diagonal discrepancy hierarchy" in captured.out
    assert "E_0 = Aplus" in captured.out
    assert "E_1 = Dplus" in captured.out
    assert "K image implication" in captured.out


def test_k5_top3_diagonal_heat_kernel_e1_representation_matches_direct() -> None:
    for a in range(1, 8):
        for n_value in range(0, 12):
            assert _top3_diagonal_e1_representation_value(n_value, a) == pytest.approx(
                _top3_diagonal_discrepancy_value(1, n_value, a)
            )


def test_k5_top3_diagonal_heat_kernel_reflected_child_matches_boundary_recurrence() -> None:
    for j in range(1, 5):
        for n_value in range(0, 12):
            left = _top3_diagonal_discrepancy_value(j, n_value + 1, j)
            right = 0.5 * (
                _top3_diagonal_discrepancy_value(j - 1, n_value, j)
                + _top3_diagonal_reflected_boundary_child_value(n_value, j)
            )
            assert left == pytest.approx(right)


def test_k5_top3_diagonal_heat_kernel_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_diagonal_heat_kernel_report(
        max_n=20,
        dense_a_limit=12,
        max_j=4,
    )

    captured = capsys.readouterr()
    assert "diagonal heat-kernel report" in captured.out
    assert "Duhamel representation" in captured.out
    assert "E_1 bound" in captured.out
    assert "reflected boundary child" in captured.out


def test_k5_top3_reflected_boundary_convolution_decomposition_residuals_are_small() -> None:
    rows = k5_top3_reflected_boundary_convolution_residual_rows(
        max_n=20,
        dense_a_limit=12,
    )

    assert rows
    assert max(row.max_abs_residual for row in rows) < 1e-12


def test_k5_top3_reflected_boundary_convolution_is_nonnegative_on_scan() -> None:
    rows = {
        row.name: row
        for row in k5_top3_reflected_boundary_convolution_scan_rows(
            max_n=20,
            dense_a_limit=12,
        )
    }

    assert rows["Rref"].sign_holds
    for a in range(1, 8):
        for n_value in range(max(0, a - 2), 12):
            assert _top3_diagonal_e1_reflected_boundary_contribution(n_value, a) >= -1e-12


def test_k5_top3_reflected_boundary_convolution_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_reflected_boundary_convolution_report(
        max_n=20,
        dense_a_limit=12,
    )

    captured = capsys.readouterr()
    assert "reflected-boundary convolution" in captured.out
    assert "Rref" in captured.out
    assert "a/sqrt(n) bins" in captured.out
    assert "candidate envelope" in captured.out


def test_k5_top3_reflected_convolution_analytic_scan_rows_are_finite() -> None:
    rows = k5_top3_reflected_convolution_analytic_scan_rows(
        max_n=20,
        dense_a_limit=12,
        verification_max_n=20,
        verification_a_limit=12,
    )

    assert rows
    assert all(np.isfinite(row.kernel_constant) for row in rows)
    assert all(np.isfinite(row.source_constant) for row in rows)
    assert all(np.isfinite(row.unit_max_scaled) for row in rows)
    assert all(np.isfinite(row.max_scaled_upper) for row in rows)
    assert all(row.max_scaled_upper >= 0.0 for row in rows)


def test_k5_top3_reflected_convolution_analytic_bound_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_reflected_convolution_analytic_bound_report(
        max_n=20,
        dense_a_limit=12,
    )

    captured = capsys.readouterr()
    assert "reflected convolution analytic bound" in captured.out
    assert "kernel bound" in captured.out
    assert "source envelope" in captured.out
    assert "abstract convolution inequality" in captured.out


def test_k5_top3_reflected_source_grid_matches_exact_child_on_small_grid() -> None:
    source_grid = _top3_reflected_source_grid(max_n=20, dense_a_limit=12)

    for t in range(21):
        for a in range(1, min(t + 2, 12) + 1):
            assert abs(source_grid[t, a] - _top3_diagonal_reflected_boundary_child_value(t, a)) < 1e-12


def test_k5_top3_reflected_source_gaussian_envelope_upper_bounds_small_grid() -> None:
    source_grid = _top3_reflected_source_grid(max_n=20, dense_a_limit=12)
    rows = k5_top3_reflected_source_gaussian_envelope_rows(
        max_n=20,
        dense_a_limit=12,
        source_grid=source_grid,
    )

    assert rows
    best_row = min(rows, key=lambda item: item.source_constant)
    for t in range(21):
        for a in range(1, min(t + 2, 12) + 1):
            envelope = best_row.source_constant * _top3_reflected_source_gaussian_envelope_value(
                t,
                a,
                best_row.family_name,
                best_row.heat_c,
            )
            assert source_grid[t, a] <= envelope + 1e-12


def test_k5_top3_reflected_source_gaussian_convolution_scan_is_finite() -> None:
    source_grid = _top3_reflected_source_grid(max_n=20, dense_a_limit=12)
    source_rows = k5_top3_reflected_source_gaussian_envelope_rows(
        max_n=20,
        dense_a_limit=12,
        source_grid=source_grid,
    )
    best_row = min(source_rows, key=lambda item: item.source_constant)
    rows = k5_top3_reflected_convolution_gaussian_scan_rows(
        max_n=20,
        dense_a_limit=12,
        source_row=best_row,
        verification_max_n=20,
        verification_a_limit=12,
    )

    assert rows
    assert all(np.isfinite(row.max_scaled_upper) for row in rows)
    assert all(row.max_scaled_upper >= 0.0 for row in rows)


def test_k5_top3_reflected_source_gaussian_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_reflected_source_gaussian_report(
        max_n=20,
        dense_a_limit=12,
    )

    captured = capsys.readouterr()
    assert "reflected source gaussian" in captured.out
    assert "Gaussian source envelope" in captured.out
    assert "new convolution upper" in captured.out
    assert "old envelope too crude" in captured.out


def test_k5_top3_reflected_gradient_convolution_source_bounds_small_grid() -> None:
    source_grid = _top3_reflected_source_grid(max_n=20, dense_a_limit=12)
    rows = [
        row
        for row in k5_top3_reflected_source_gaussian_envelope_rows(
            max_n=20,
            dense_a_limit=12,
            source_grid=source_grid,
        )
        if row.family_name == "(a+1)/(t+1)^(3/2)*exp"
    ]

    assert rows
    best_row = min(rows, key=lambda item: item.source_constant)
    for t in range(21):
        for a in range(1, min(t + 2, 12) + 1):
            envelope = best_row.source_constant * _top3_reflected_source_gaussian_envelope_value(
                t,
                a,
                best_row.family_name,
                best_row.heat_c,
            )
            assert source_grid[t, a] <= envelope + 1e-12


def test_k5_top3_reflected_gradient_convolution_upper_is_finite_and_improves() -> None:
    source_grid = _top3_reflected_source_grid(max_n=20, dense_a_limit=12)
    source_rows = k5_top3_reflected_source_gaussian_envelope_rows(
        max_n=20,
        dense_a_limit=12,
        source_grid=source_grid,
    )
    rows = k5_top3_reflected_gradient_convolution_scan_rows(
        max_n=20,
        dense_a_limit=12,
        source_rows=source_rows,
        verification_max_n=20,
        verification_a_limit=12,
    )
    old_rows = k5_top3_reflected_convolution_analytic_scan_rows(
        max_n=20,
        dense_a_limit=12,
        verification_max_n=20,
        verification_a_limit=12,
        heat_c_values=(2.0, 4.0, 8.0, 16.0, 32.0),
        source_name="min(1,(a+1)/(t+1))",
    )

    assert rows
    assert all(np.isfinite(row.max_scaled_upper) for row in rows)
    assert all(row.max_scaled_upper >= 0.0 for row in rows)
    assert min(row.max_scaled_upper for row in rows) <= min(row.max_scaled_upper for row in old_rows) + 1e-12


def test_k5_top3_reflected_gradient_convolution_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_reflected_gradient_convolution_report(
        max_n=20,
        dense_a_limit=12,
    )

    captured = capsys.readouterr()
    assert "reflected gradient convolution" in captured.out
    assert "source gradient envelope" in captured.out
    assert "gradient convolution upper" in captured.out
    assert "proof split" in captured.out


def test_k5_top3_endpoint_polytope_reproduces_companion_constants() -> None:
    rows = {
        row.name: row
        for row in k5_top3_kplus_companion_scan_rows(
            max_n=20,
            dense_a_limit=12,
        )
    }
    closure_rows = {
        row.target: row
        for row in k5_top3_endpoint_polytope_closure_rows(
            max_n=20,
            dense_a_limit=12,
            closure_max_n=8,
        )
    }

    assert rows["K"].max_scaled == pytest.approx(1.5)
    assert rows["Q3"].max_scaled == pytest.approx(0.5)
    assert "K<=1.5" in closure_rows


def test_k5_top3_endpoint_polytope_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k5_top3_endpoint_polytope_barrier_report(
        max_n=20,
        dense_a_limit=12,
        closure_max_n=8,
        n=8,
    )

    captured = capsys.readouterr()
    assert "endpoint polytope barrier" in captured.out
    assert "normalized convex hull" in captured.out
    assert "halfspace induction closure" in captured.out


def test_fixed_action_policy_returns_balanced_complement() -> None:
    policy = _fixed_balanced_action_policy_from_text("101010")
    support = policy((0, 0, 0, 0, 0, 0), 4)

    assert set(support) == {
        (0.5, (1, 0, 1, 0, 1, 0)),
        (0.5, (0, 1, 0, 1, 0, 1)),
    }


def test_fixed_comb_not_equal_to_adaptive_top_prefix_shorter_state() -> None:
    fixed_policy = _fixed_balanced_action_policy_from_text("101010")
    state = (2, 1, 1, 1, 1, 0)

    assert fixed_policy(state, 10) != k6_greedy_deterministic_patched_policy(state, 10)


def test_interval_from_action_examples() -> None:
    assert _interval_from_action((1, 0, 0, 0, 0, 0)) == (0, 0)
    assert _interval_from_action((1, 0, 1, 0, 0, 0)) == (0, 2)
    assert _interval_from_action((1, 0, 1, 0, 1, 0)) == (0, 4)


def test_k6_adaptive_interval_anatomy_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_adaptive_interval_anatomy_report(2, (2,), n=2)

    captured = capsys.readouterr()
    assert "Fixed pattern comparison" in captured.out
    assert "Top-prefix interval mass" in captured.out
    assert "One-run vs top-prefix differences" in captured.out
    assert "Candidate distillation target" in captured.out


def test_packet_barycenter_status_examples() -> None:
    state = (3, 3, 1, 1, 0, 0)
    mean = sum(state) / len(state)

    assert _packet_barycenter_status(3, mean) == "above_mean"
    assert _packet_barycenter_status(1, mean) == "below_mean"
    assert _packet_barycenter_status(mean, mean) == "crosses_mean"


def test_barycenter_action_rows_have_valid_probabilities() -> None:
    rows = k6_barycenter_action_rows(2)

    assert rows
    for row in rows:
        assert 0.0 <= row.p_top1_bottom0 <= 1.0
        assert 0.0 <= row.p_top0_bottom1 <= 1.0
        assert 0.0 <= row.p_top_equals_bottom <= 1.0
        for packet in row.packet_summaries:
            assert 0.0 <= packet.exposure <= 1.0
            assert 0.0 <= packet.split_rate <= 1.0


def test_barycenter_split_rate_zero_for_unsplit_packet() -> None:
    row = library_lp_dual_inspect_rows(6, 1, "top_prefix_all")[1][0]
    row = row.__class__(
        time=row.time,
        remaining_horizon=row.remaining_horizon,
        state=(1, 1, 0, 0, 0, 0),
        packet_type=(2, 4),
        packet_gaps=(1,),
        adjacent_gaps=(0, 1, 0, 0, 0),
        occupancy_probability=row.occupancy_probability,
        value=row.value,
        learner_distribution=row.learner_distribution,
        dual_value=row.dual_value,
        expected_action=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        max_expected_action=row.max_expected_action,
        support=(
            (0.5, (1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
            (0.5, (0, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0)),
        ),
    )
    bary_row = _barycenter_action_row_from_dual_row("top_prefix_all", row)

    assert all(packet.split_rate == pytest.approx(0.0) for packet in bary_row.packet_summaries)


def test_k6_barycenter_action_anatomy_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_k6_barycenter_action_anatomy_report(2, n=2)

    captured = capsys.readouterr()
    assert "Global barycenter exposure summary" in captured.out
    assert "Top-bottom correlation summary" in captured.out
    assert "Rule hints" in captured.out


def test_chase5_policy_returns_balanced_pair() -> None:
    assert _chase5_policy((0, 0, 0, 0, 0), 5) == (
        (0.5, (1, 0, 1, 0, 0)),
        (0.5, (0, 1, 0, 1, 1)),
    )


def test_chase5_value_recursion_runs() -> None:
    value = _chase5_value_layers_reachable(3)

    assert value((0, 0, 0, 0, 0), 3) >= 0.0


def test_chase5_winner_probabilities_sum_to_one() -> None:
    winners = _chase5_named_winner_probabilities((0, 0, 0, 0, 0), 3, {})

    assert sum(winners) == pytest.approx(1.0)


def test_chase5_verification_summary_runs() -> None:
    summary = chase5_verification_summary(3, n=2)

    assert summary.value >= 0.0
    assert summary.top_rows


def test_chase5_verification_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_chase5_verification_report((3,), n=2)

    captured = capsys.readouterr()
    assert "max Bellman residual" in captured.out


def test_chase5_certificate_repairs_h1_leader_state() -> None:
    value_fn = _chase5_value_layers_reachable(2)
    learner, alpha, _ = _chase5_certificate_learner((2, 1, 0, 0, 0), 1, value_fn)

    assert alpha <= 1e-9
    assert learner[0] == pytest.approx(1.0)


def test_chase5_potential_certificate_summary_runs() -> None:
    summary = chase5_potential_certificate_summary(3, n=2)

    assert summary.max_pm_residual >= summary.max_repaired_alpha - 1e-9


def test_chase5_potential_certificate_report_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_chase5_potential_certificate_report((3,), n=2)

    captured = capsys.readouterr()
    assert "PM fails but repaired succeeds" in captured.out


def test_explicit_time_policy_benchmark_rows_run() -> None:
    rows, skipped = explicit_time_policy_benchmark_rows(
        ((9, 2),),
        ("two_run_dual_support_replay_k9_T7", "top_prefix_three_regime_v6"),
        include_reference=False,
    )

    assert not skipped
    assert {row.policy_name for row in rows} == {
        "two_run_dual_support_replay_k9_T7",
        "top_prefix_three_regime_v6",
    }


def test_two_run_skeleton_v3a_late_policy_branches() -> None:
    equal_cleanup = two_run_skeleton_v3a_late_policy((5, 2, 2, 2, 2, 2, 1, 1, 0), 1)
    biased_repair = two_run_skeleton_v3a_late_policy((3, 3, 3, 1, 1, 1, 1, 0, 0), 3)
    central_repair = two_run_skeleton_v3a_late_policy((3, 2, 2, 2, 1, 1, 1, 0, 0), 2)

    assert sum(probability for probability, _ in equal_cleanup) == pytest.approx(1.0)
    assert {action for _, action in equal_cleanup} == {
        (0, 0, 0, 0, 1, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 1, 0),
        (0, 1, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 1, 0, 0),
    }
    assert sum(probability for probability, _ in biased_repair) == pytest.approx(1.0)
    assert len(biased_repair) == 5
    assert sum(probability for probability, _ in central_repair) == pytest.approx(1.0)
    assert len(central_repair) == 3
    assert all(len(action) == 9 for _, action in equal_cleanup + biased_repair + central_repair)


@pytest.mark.parametrize(
    ("state", "remaining_horizon", "expected_count"),
    [
        ((0, 0, 0, 0, 0, 0, 0, 0, 0), 7, 9),
        ((1, 1, 1, 1, 0, 0, 0, 0, 0), 6, 8),
        ((2, 2, 1, 1, 1, 0, 0, 0, 0), 5, 6),
        ((3, 2, 2, 1, 1, 1, 1, 0, 0), 4, 5),
    ],
)
def test_two_run_skeleton_v3b_early_late_policy_branches(
    state: tuple[int, ...],
    remaining_horizon: int,
    expected_count: int,
) -> None:
    policy = two_run_skeleton_v3b_early_late_policy(state, remaining_horizon)

    assert len(policy) == expected_count
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)
    assert all(len(action) == 9 for _, action in policy)


def test_two_run_skeleton_v3b_early_late_policy_fallback() -> None:
    policy = two_run_skeleton_v3b_early_late_policy((5, 2, 2, 2, 2, 2, 1, 1, 0), 1)

    assert len(policy) == 5
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_two_run_replay_cutoff_policy_uses_replay_inside_cutoff() -> None:
    cutoff_policy = make_two_run_replay_cutoff_policy(
        replay_min_remaining=7,
        fallback_name="two_run_skeleton_v3a_late_policy",
    )
    replay_policy = two_run_dual_support_replay_k9_T7_policy((0,) * 9, 7)

    policy = cutoff_policy((0,) * 9, 7)

    assert policy == replay_policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_two_run_replay_cutoff_policy_uses_fallback_outside_cutoff() -> None:
    cutoff_policy = make_two_run_replay_cutoff_policy(
        replay_min_remaining=7,
        fallback_name="two_run_skeleton_v3a_late_policy",
    )
    fallback_policy = two_run_skeleton_v3a_late_policy((5, 2, 2, 2, 2, 2, 1, 1, 0), 1)

    policy = cutoff_policy((5, 2, 2, 2, 2, 2, 1, 1, 0), 1)

    assert policy == fallback_policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_two_run_replay_mask_policy_uses_replay_inside_mask() -> None:
    mask_policy = make_two_run_replay_mask_policy(
        frozenset({7}),
        "top_prefix_three_regime_v6",
    )
    replay_policy = two_run_dual_support_replay_k9_T7_policy((0,) * 9, 7)

    policy = mask_policy((0,) * 9, 7)

    assert policy == replay_policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_two_run_replay_mask_policy_uses_fallback_outside_mask() -> None:
    mask_policy = make_two_run_replay_mask_policy(
        frozenset({7}),
        "two_run_skeleton_v3a_late_policy",
    )
    state = (5, 2, 2, 2, 2, 2, 1, 1, 0)
    fallback_policy = two_run_skeleton_v3a_late_policy(state, 1)

    policy = mask_policy(state, 1)

    assert policy == fallback_policy
    assert sum(probability for probability, _ in policy) == pytest.approx(1.0)


def test_replay_layer_mask_sweep_rows_singletons_run() -> None:
    rows = replay_layer_mask_sweep_rows(
        9,
        7,
        "top_prefix_three_regime_v6",
        mode="singletons",
    )

    assert len(rows) == 7
    assert all(row.replay_count == 1 for row in rows)


def test_replay_layer_mask_sweep_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_replay_layer_mask_sweep(
        9,
        7,
        "top_prefix_three_regime_v6",
        mode="singletons",
        n=2,
    )

    captured = capsys.readouterr()
    assert "Replay-layer mask sweep" in captured.out
    assert "gap_to_reference" not in captured.out
    assert "gap/sqrt(T)" in captured.out


def test_time_policy_boundary_loss_reference_equals_candidate_zero() -> None:
    rows = time_policy_boundary_loss_rows(
        3,
        3,
        "top_prefix_three_regime_v6",
        "top_prefix_three_regime_v6",
    )

    assert rows
    assert sum(row.weighted_action_loss for row in rows) == pytest.approx(0.0)
    assert sum(row.weighted_downstream_loss for row in rows) == pytest.approx(0.0)
    assert sum(row.weighted_total_gap for row in rows) == pytest.approx(0.0)


def test_time_policy_occupancy_nonterminal_mass_sums_to_T() -> None:
    def toy_policy(state: tuple[int, ...], remaining_horizon: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        del state, remaining_horizon
        return ((0.5, (1, 0, 0)), (0.5, (0, 1, 1)))

    occupancy = _time_policy_occupancy(3, 4, toy_policy)

    assert sum(sum(layer.values()) for layer in occupancy[:-1]) == pytest.approx(4.0)


def test_time_policy_boundary_loss_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown candidate policy"):
        time_policy_boundary_loss_rows(
            3,
            3,
            "top_prefix_three_regime_v6",
            "does_not_exist",
        )


def test_time_policy_boundary_loss_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_time_policy_boundary_loss(
        3,
        3,
        "top_prefix_three_regime_v6",
        "top_prefix_three_regime_v6",
        n=2,
    )

    captured = capsys.readouterr()
    assert "Time-policy boundary loss" in captured.out
    assert "Loss by remaining horizon" in captured.out


def test_explicit_time_policy_benchmark_printer_runs(capsys: pytest.CaptureFixture[str]) -> None:
    print_explicit_time_policy_benchmark(
        ((9, 2),),
        ("two_run_dual_support_replay_k9_T7",),
        include_reference=False,
    )

    captured = capsys.readouterr()
    assert "Explicit time-dependent policy benchmark" in captured.out
    assert "two_run_dual_support_replay_k9_T7" in captured.out


def test_weighted_greedy_filter_matches_requested_regime() -> None:
    _, contributions = occupation_weighted_greedy_defects(5, 5, comb_policy)
    filtered = filter_weighted_greedy_contributions(
        contributions,
        packet_type_filter=(1, 2, 2),
        packet_gaps_filter=(1, 1),
    )

    assert filtered
    for item in filtered:
        assert item.packet_type == (1, 2, 2)
        values = []
        for value in item.state:
            if not values or value != values[-1]:
                values.append(value)
        gaps = tuple(values[index] - values[index + 1] for index in range(len(values) - 1))
        assert gaps == (1, 1)
