"""BBEH (BIG-Bench Extra Hard) tasks and their generation budgets.

The 23 tasks are run zero-shot with direct answers: the evaluation prompt is `{input}\\nAnswer: `, and training
appends the short answer (for example "131", "(A)", or "yes,no,unknown"). Each task maps to its `max_new_tokens` for
evaluation.

Paper: https://arxiv.org/abs/2502.19187
Data and official evaluator: https://github.com/google-deepmind/bbeh
"""

BBEH_TASKS: dict[str, int] = {
    # Short fixed-format answers (at most 9 characters).
    "bbeh_boardgame_qa":         32,
    "bbeh_boolean_expressions":  32,
    "bbeh_buggy_tables":         32,
    "bbeh_causal_understanding": 32,
    "bbeh_disambiguation_qa":    32,
    "bbeh_dyck_languages":       32,
    "bbeh_geometric_shapes":     32,
    "bbeh_hyperbaton":           32,
    "bbeh_movie_recommendation": 32,
    "bbeh_multistep_arithmetic": 32,
    "bbeh_nycc":                 32,
    "bbeh_object_counting":      32,
    "bbeh_object_properties":    32,
    "bbeh_sarc_triples":         32,
    "bbeh_shuffled_objects":     32,
    "bbeh_temporal_sequence":    32,
    "bbeh_zebra_puzzles":        32,

    # Medium answers (at most 30 characters).
    "bbeh_linguini":             64,
    "bbeh_sportqa":              64,
    "bbeh_web_of_lies":          64,

    # Longer free-form answers.
    "bbeh_time_arithmetic":      128,
    "bbeh_spatial_reasoning":    256,
    "bbeh_word_sorting":         384,
}
