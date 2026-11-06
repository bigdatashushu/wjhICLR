"""Explicit shared-source inventory shipped with the action runner."""

_FILES = {
    "": "__init__.py coords.py env_preflight.py",
    "adapters": "__init__.py episode_source.py frame_set.py split_builder.py vsibench_loader.py",
    "cluster": "README.md __init__.py identity.py perception_client.py publication.py queue.py scene_compute.py",
    "evaluation": (
        "__init__.py accuracy.py blink_experiment.py blink_recovery.py cached_episode_recovery_v11.py "
        "derived_published_fitset_v11.py direct_qwen_v11.py episode_pool_v11.py fitset_panel_v11.py "
        "fitset_v11.py gate_process_metrics.py mra.py multi_seed_aggregator.py process_metrics.py "
        "published_fitset_v11.py qa_partition_v11.py reviser_pool_v11.py skill_ablation_v11.py "
        "verified_publication_v11.py"
    ),
    "evolution": (
        "__init__.py adapters_v11.py campaign_v11.py case_records_v11.py deployment_continuation_v11.py "
        "diagnosis_v11.py evidence_graph_v11.py experience_v11.py fitset_revision_v11.py fitset_stream_v11.py "
        "gate_admission_v11.py leakage.py live_v11.py outer_v11.py prepared_binding_derivation_v11.py "
        "prepared_bindings_v11.py prepared_generation_reuse_v11.py process_feedback.py revision_edits_v11.py "
        "revision_guard_v11.py revision_input_v11.py revision_schedule.py revision_view_v11.py streaming_v11.py "
        "trajectory_analysis_v11.py trajectory_fragments_v11.py"
    ),
    "fsm": "__init__.py online_fsm.py",
    "gates": "__init__.py input_gate.py iqa.py",
    "governance": "__init__.py deepseek_client.py",
    "infra": "__init__.py experiment_campaign.py experiment_stages_v11.py experiment_startup.py m5_identity.py version_lock.py",
    "legacy": "readers.py",
    "online": "__init__.py config.py eval.py observation_reader.py progress.py recovery.py runner.py scene_overview.py submission.py synthetic.py",
    "reconstruction": "__init__.py image_grid.py metric_evidence.py metric_fusion.py run.py vggt_runner.py",
    "reconstruction_gate": "__init__.py confidence_map.py evidence_profile.py m4_main_gate.py quality_metrics.py scene_state.py world_frame.py",
    "routing": "__init__.py retrieval_policy.py skill_retriever.py task_classifier.py",
    "sandbox": "__init__.py action_check.py ast_guard.py kernel.py receipt.py safe_runtime.py",
    "scheduling": "__init__.py gpu_scheduler.py",
    "schemas": (
        "__init__.py answer.py authorization.py data_access.py episode.py evidence.py evolution.py "
        "evolution_v11.py image_ledger.py legacy.py program.py reconstruction.py retrieval.py skill.py tool.py "
        "tool_response.py trace.py"
    ),
    "segmentation": (
        "__init__.py binding_validation.py frozen_source_compatibility.py groundingdino_startup.py "
        "intern_binary_review.py intern_decision_client.py intern_decision_protocol.py intern_review_pool.py "
        "mask_semantic_review.py open_vocab_detector.py prepared_binding_store.py question_binding.py sam2_tracker.py"
    ),
    "skills": "__init__.py delivery.py promote_atomic.py registry.py state_baseline.py tool_baseline.py v11_library.py",
    "small_image": (
        "__init__.py backends.py blink_protocol.py cv3d_context.py cv3d_protocol.py cv3d_protocol_v1.py "
        "datasets.py depth_image.py precompute.py runner.py storage.py tools.py"
    ),
    "storage": (
        "__init__.py cold_history.py framework_state_io.py hot_cache.py offline_frames.py published_transport.py "
        "relocation.py scene_prefetch.py scene_runtime.py state_workspace.py"
    ),
    "synthesis": (
        "__init__.py bailian_client.py client_errors.py eval_visual_fallback.py execution_contract.py "
        "program_assembler.py prompt_builder.py request_audit.py request_context.py request_governor.py"
    ),
    "tools": (
        "__init__.py category_match.py connectivity.py contract.py distance_primitives.py docs_v11.py "
        "failure_guidance.py geometry_tools.py image_ledger.py mock_switch.py ranking_contract.py registry.py "
        "scene_handle.py task_policy.py vision_tools.py"
    ),
    "trace": "__init__.py store.py",
    "verifier": "__init__.py derivation.py geometry_oracle.py",
}

SHARED_SOURCES = tuple(sorted(
    "src/skill3d/" + (directory + "/" if directory else "") + filename
    for directory, names in _FILES.items() for filename in names.split()
))
