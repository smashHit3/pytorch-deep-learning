"""Shared utilities for educational demos."""

from .D001_sequence_model import (
    SequenceDemoConfig,
    build_markov_features,
    evaluate_loss,
    get_net,
    init_weights,
    load_array,
    load_curve,
    plot,
    plot_one_step_predictions,
    plot_prediction_comparison,
    predict_multistep,
    run_sequence_model_demo,
    set_axes,
    train,
)

__all__ = [
    "SequenceDemoConfig",
    "build_markov_features",
    "evaluate_loss",
    "get_net",
    "init_weights",
    "load_array",
    "load_curve",
    "plot",
    "plot_one_step_predictions",
    "plot_prediction_comparison",
    "predict_multistep",
    "run_sequence_model_demo",
    "set_axes",
    "train",
]