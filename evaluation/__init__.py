"""Phase 7: Evaluation Framework (local, deterministic, no LLM or network)."""

from evaluation.evaluator import (  # noqa: F401
    EVALUATION_VERSION,
    evaluate_research_run,
    evaluate_retrieval_labels,
    format_evaluation_summary,
    load_retrieval_labels,
    save_evaluation,
)
