"""
Phase 7 Evaluation Framework: public entry point.

``evaluate_research_run`` inspects the outputs a research run already
produced and reports OBSERVABLE quality indicators. It never calls an LLM,
never touches the network, never re-runs retrieval, and never mutates its
inputs. It does NOT prove that the research is scientifically correct.

Result structure (JSON-serializable)
------------------------------------
{
  "evaluation_version": "1.0",
  "overall_status": "completed" | "completed_with_warnings" | "incomplete"
                    | "completed_with_errors",
  "requires_human_review": bool,
  "scientific_validity": "not_assessed",
  "metrics": {name: {status, description, value, reason?, details?}},
  "warnings": [str],
  "limitations": [str],
}

``overall_status`` describes the EVALUATION PROCESS, never the research:
  completed_with_errors  some metric code failed (error text is sanitized)
  incomplete             no metric errors, but a pipeline stage failed, was
                         skipped, partial or unknown
  completed_with_warnings all stages ran but problems were observed
                         (unknown citation IDs, unverified claims, ...)
  completed              no problems were observed by these checks
A metric that is merely ``not_evaluated`` (for example precision@k with no
labels) does not by itself change the overall status.

Metric statuses: calculated, heuristic, not_evaluated, error, needs_review
(see evaluation/metrics.py).
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents.analysis_agent import redact_secrets
from evaluation import metrics as m

EVALUATION_VERSION = "1.0"
MAX_ERROR_TEXT = 200
MAX_REPORT_BYTES = 2_000_000

STAGE_STATUSES = ("completed", "partial", "failed", "skipped", "unknown")
STAGES = ("planner", "retrieval", "iterative_retrieval", "graph_rag",
          "analysis", "verification", "report")

LIMITATIONS = (
    "These checks measure observable properties only; they do not show "
    "that any claim or conclusion is scientifically correct.",
    "Sub-question coverage is a keyword-overlap heuristic, not semantic "
    "verification.",
    "A citation that resolves to a known paper ID does not mean the paper "
    "supports the claim; a matching excerpt is a string match only.",
    "Verification statuses come from a language model and are not proof.",
    "Precision@k and recall@k exist only when trusted relevance labels are "
    "supplied; fixture labels are not universal ground truth.",
    "Stage statuses are derived from the outputs given; stages the "
    "application does not track are reported as unknown.",
)


# ---------------------------------------------------------------------------
# Sanitizing
# ---------------------------------------------------------------------------

def sanitize_error(error: Any) -> str:
    """Error text safe to show: secrets redacted, length capped. For an
    exception only its type and message are used (never a traceback)."""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    text = " ".join(redact_secrets(text).split())
    return text[:MAX_ERROR_TEXT]


# ---------------------------------------------------------------------------
# Stage reliability
# ---------------------------------------------------------------------------

def _stage(status: str, note: str = "") -> Dict[str, str]:
    return {"status": status, "note": note}


def _derive_stages(plan, papers, retrieval_result, graph_context, analysis,
                   verification_error, report, report_error,
                   overrides) -> Dict[str, Dict[str, str]]:
    stages = {name: _stage("unknown", "No information was provided.")
              for name in STAGES}

    if isinstance(plan, dict):
        stages["planner"] = (
            _stage("completed") if m._valid_subquestions(plan)
            else _stage("partial", "The plan has no sub-questions."))

    if isinstance(papers, list):
        stages["retrieval"] = (
            _stage("completed") if papers
            else _stage("partial", "Retrieval returned no papers."))

    if isinstance(retrieval_result, dict):
        errors = retrieval_result.get("errors")
        stages["iterative_retrieval"] = (
            _stage("partial", "Extra retrieval attempts recorded errors.")
            if isinstance(errors, list) and errors else _stage("completed"))

    if graph_context is not None:
        stages["graph_rag"] = (
            _stage("completed") if graph_context
            else _stage("partial", "No graph context was generated."))

    claims = m._claims(analysis)
    if isinstance(analysis, dict):
        stages["analysis"] = (
            _stage("completed") if analysis.get("status") == "success"
            else _stage("partial", "Analysis status: "
                        f"{m._short(str(analysis.get('status')), 40)}."))

        summary = analysis.get("verification_summary")
        has_record = any(isinstance(c.get("verification"), dict)
                         for c in claims or [])
        if verification_error:
            stages["verification"] = _stage(
                "failed", sanitize_error(verification_error))
        elif not claims:
            stages["verification"] = _stage(
                "skipped", "There were no claims to verify.")
        elif has_record:
            service = (summary or {}).get("service_status", "completed") \
                if isinstance(summary, dict) else "completed"
            stages["verification"] = (
                _stage("completed") if service == "completed" else
                _stage("failed" if service == "unavailable" else "partial",
                       f"Verification service status: {service}."))
        else:
            stages["verification"] = _stage(
                "unknown", "Claims exist but no verification results "
                "were provided.")
    elif verification_error:
        stages["verification"] = _stage(
            "failed", sanitize_error(verification_error))

    if report_error:
        stages["report"] = _stage("failed", sanitize_error(report_error))
    elif isinstance(report, dict):
        has_md, has_docx = bool(report.get("markdown_path")), bool(
            report.get("docx_path"))
        errs = report.get("errors")
        if has_md and has_docx and not errs:
            stages["report"] = _stage("completed")
        elif has_md or has_docx:
            stages["report"] = _stage(
                "partial", "Only some report files were written.")
        else:
            stages["report"] = _stage("failed", "No report file was written.")

    for name, status in (overrides or {}).items():
        if name in stages and status in STAGE_STATUSES:
            stages[name] = _stage(status, "Status supplied by the caller.")
    return stages


def _pipeline_reliability(stages: dict, report_info: Any,
                          retrieval_result: Any) -> Dict[str, Any]:
    desc = ("Stage outcomes derived from the outputs provided. Stages whose "
            "outcome is not exposed are 'unknown', never assumed to have "
            "succeeded.")
    counts = {s: 0 for s in STAGE_STATUSES}
    for stage in stages.values():
        counts[stage["status"]] += 1
    report_stage = stages["report"]["status"]
    return m.make_metric(m.STATUS_CALCULATED, desc, counts, details={
        "stages": stages,
        "usable_report_produced": (
            None if report_stage == "unknown"
            else report_stage in ("completed", "partial")),
        "retrieval_errors_recorded": len(retrieval_result.get("errors") or [])
        if isinstance(retrieval_result, dict)
        and isinstance(retrieval_result.get("errors"), list) else None,
        "execution_times_tracked": False,
    })


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

def _read_report_text(report: Any, report_markdown: Any):
    """Return (text or None, note). Reads the local Markdown file only."""
    if isinstance(report_markdown, str):
        return report_markdown, ""
    path = report.get("markdown_path") if isinstance(report, dict) else None
    if not path:
        return None, ""
    try:
        file_path = Path(path)
        if file_path.stat().st_size > MAX_REPORT_BYTES:
            return None, "The report file is too large to check."
        return file_path.read_text(encoding="utf-8"), ""
    except (OSError, UnicodeDecodeError) as exc:
        return None, "The report file could not be read: " + sanitize_error(
            exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_research_run(
        user_query: Optional[str] = None,
        plan: Optional[dict] = None,
        papers: Optional[list] = None,
        retrieval_result: Optional[dict] = None,
        graph_context: Optional[list] = None,
        analysis: Optional[dict] = None,
        verification_error: Optional[str] = None,
        report: Optional[dict] = None,
        report_markdown: Optional[str] = None,
        report_error: Optional[str] = None,
        ground_truth: Optional[dict] = None,
        stage_status: Optional[Dict[str, str]] = None,
        k: int = 5,
        strict: bool = False) -> Dict[str, Any]:
    """
    Evaluate one research run from the outputs the pipeline already has.

    Every argument is optional; anything missing yields ``not_evaluated`` /
    ``unknown`` results rather than an assumption of success.

    plan, papers, graph_context   as produced by planner / retriever / graph
    retrieval_result              dict from ``run_iterative_retrieval``
                                  (its "papers" are used if ``papers`` is
                                  None)
    analysis                      output of ``analyze_literature`` or, when
                                  verification ran, ``verify_analysis``
    verification_error            error text when verification raised
    report                        dict from ``generate_research_report``
    report_markdown               report text (else read from
                                  ``report["markdown_path"]``)
    report_error                  error text when report generation raised
    ground_truth                  optional {"relevant_ids": [...]} manual
                                  labels; enables precision@k / recall@k
    stage_status                  optional overrides {stage: status}
    k                             cut-off for precision@k / recall@k
    strict                        re-raise metric errors (for development)

    Inputs are never modified. Invalid ``k`` raises ValueError. A failure
    inside one metric is recorded as that metric's ``error`` status with a
    sanitized message and makes ``overall_status`` completed_with_errors;
    it is never reported as success.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    if papers is None and isinstance(retrieval_result, dict) and isinstance(
            retrieval_result.get("papers"), list):
        papers = retrieval_result["papers"]

    def run(description: str, function: Callable[[], Any]) -> Any:
        try:
            return function()
        except Exception as exc:       # noqa: BLE001 - see docstring
            if strict:
                raise
            return m.make_metric(
                m.STATUS_ERROR, description, None,
                reason=("The metric could not be calculated because of an "
                        "internal error: " + sanitize_error(exc)))

    results: Dict[str, Any] = {}

    def precision_recall() -> Dict[str, Any]:
        if not isinstance(papers, list):
            why = "No paper list was provided."
            return {
                "precision": m.not_evaluated(
                    f"Precision@{k} against supplied relevance labels.", why),
                "recall": m.not_evaluated(
                    f"Recall@{k} against supplied relevance labels.", why)}
        if ground_truth is not None and not isinstance(ground_truth, dict):
            raise ValueError("ground_truth must be a dictionary.")
        labels = ground_truth.get("relevant_ids") if ground_truth else None
        return m.precision_recall_at_k(papers, labels, k)

    pr = run("Precision/recall@k.", precision_recall)
    if "precision" not in pr:          # the call itself failed
        pr = {"precision": pr, "recall": pr}

    results["retrieval_record_count"] = run(
        "Retrieved record count.", lambda: m.retrieval_record_count(papers))
    results["retrieval_abstract_availability"] = run(
        "Abstract availability.", lambda: m.abstract_availability(papers))
    results["retrieval_duplicates"] = run(
        "Duplicate papers.", lambda: m.duplicate_papers(papers))
    results["retrieval_precision_at_k"] = pr["precision"]
    results["retrieval_recall_at_k"] = pr["recall"]
    results["iterative_retrieval"] = run(
        "Iterative retrieval.",
        lambda: m.iterative_retrieval_summary(retrieval_result))
    results["subquestion_coverage"] = run(
        "Sub-question coverage.",
        lambda: m.subquestion_coverage(user_query, plan, papers))
    results["citation_traceability"] = run(
        "Citation traceability.",
        lambda: m.citation_traceability(analysis, papers))
    results["claim_verification"] = run(
        "Claim verification.",
        lambda: m.claim_verification(
            analysis, sanitize_error(verification_error)
            if verification_error else None,
            (stage_status or {}).get("verification") == "skipped"))

    text, read_note = _read_report_text(report, report_markdown)
    results["report_completeness"] = run(
        "Report completeness.",
        lambda: m.report_completeness(
            text, papers, analysis,
            sanitize_error(verification_error)
            if verification_error else None, report,
            sanitize_error(report_error) if report_error else None))
    if read_note and results["report_completeness"]["status"] == \
            m.STATUS_NOT_EVALUATED:
        results["report_completeness"]["reason"] = read_note

    stages = _derive_stages(plan, papers, retrieval_result, graph_context,
                            analysis, verification_error, report,
                            report_error, stage_status)
    results["pipeline_reliability"] = run(
        "Pipeline reliability.",
        lambda: _pipeline_reliability(stages, report, retrieval_result))

    warnings = _collect_warnings(results, stages)
    statuses = [r.get("status") for r in results.values()]
    if m.STATUS_ERROR in statuses:
        overall = "completed_with_errors"
    elif any(s["status"] != "completed" for s in stages.values()):
        overall = "incomplete"
    elif warnings:
        overall = "completed_with_warnings"
    else:
        overall = "completed"

    return {
        "evaluation_version": EVALUATION_VERSION,
        "overall_status": overall,
        "requires_human_review": any(
            s in (m.STATUS_NEEDS_REVIEW, m.STATUS_ERROR) for s in statuses),
        "scientific_validity": "not_assessed",
        "metrics": results,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }


def _collect_warnings(results: dict, stages: dict) -> List[str]:
    warnings: List[str] = []
    for name, stage in stages.items():
        if stage["status"] in ("failed", "partial"):
            warnings.append(f"Stage '{name}' {stage['status']}"
                            + (f": {stage['note']}" if stage["note"] else ""))
    for name, metric in results.items():
        if metric.get("status") == m.STATUS_ERROR:
            warnings.append(f"Metric '{name}' failed: "
                            f"{metric.get('reason', '')}")

    dup = results["retrieval_duplicates"].get("details") or {}
    if dup.get("duplicate_count"):
        warnings.append(f"{dup['duplicate_count']} duplicate paper "
                        "record(s) found.")
    ab = results["retrieval_abstract_availability"].get("details") or {}
    if ab.get("total") and ab.get("with_abstract") == 0:
        warnings.append("No retrieved paper has a usable abstract.")
    cov = results["subquestion_coverage"].get("details") or {}
    if cov.get("uncovered"):
        warnings.append(f"{len(cov['uncovered'])} sub-question(s) not "
                        "covered by the keyword heuristic (estimate only).")

    cit = results["citation_traceability"].get("details") or {}
    if cit.get("claims_with_unknown_ids"):
        warnings.append(f"{cit['claims_with_unknown_ids']} claim(s) cite "
                        "unknown paper IDs.")
    if cit.get("claims_missing_references"):
        warnings.append(f"{cit['claims_missing_references']} claim(s) have "
                        "no references.")
    if cit.get("claims_missing_excerpt"):
        warnings.append(f"{cit['claims_missing_excerpt']} source-supported "
                        "claim(s) have no evidence excerpt.")

    ver = results["claim_verification"].get("details") or {}
    state = ver.get("verification_state")
    if state in ("failed", "unavailable", "skipped", "partial"):
        warnings.append(f"Claim verification state: {state}; affected "
                        "claims are NOT verified.")
    if ver.get("unrecognized_statuses"):
        warnings.append("Verification produced unrecognized status "
                        "value(s).")
    if ver.get("flagged_for_review"):
        warnings.append(f"{ver['flagged_for_review']} claim(s) flagged for "
                        "manual review.")
    if ver.get("summary_matches_records") is False:
        warnings.append("Verification summary does not match claim records.")

    rep = results["report_completeness"].get("details") or {}
    for problem in rep.get("problems", []):
        warnings.append("Report: " + problem + ".")
    return warnings


# ---------------------------------------------------------------------------
# Fixture-based retrieval evaluation
# ---------------------------------------------------------------------------

def load_retrieval_labels(path: Any) -> Dict[str, Any]:
    """Load a small manually labelled JSON fixture:
    {"research_question", "retrieved_ids", "relevant_ids", "note"}.
    The labels are fixture data, not universal ground truth."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("The labels file must contain a JSON object.")
    return data


def evaluate_retrieval_labels(data: Dict[str, Any], k: int = 5
                              ) -> Dict[str, Any]:
    """precision@k / recall@k for a fixture of identifiers (offline)."""
    if not isinstance(data, dict):
        raise ValueError("Labels must be a dictionary.")
    retrieved = data.get("retrieved_ids", [])
    outcome = m.precision_recall_at_k(retrieved, data.get("relevant_ids"), k)
    return {"research_question": data.get("research_question"),
            "label_source": "fixture/manual labels, not universal truth",
            "precision_at_k": outcome["precision"],
            "recall_at_k": outcome["recall"]}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_evaluation(result: Dict[str, Any], output_dir: Any = None) -> Path:
    """Write ``evaluation.json`` atomically. Returns the path. Only that
    one file is written; other files in the folder are untouched."""
    if output_dir is None:
        from agents.report_agent import DEFAULT_OUTPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "evaluation.json"
    temp = directory / "evaluation.json.tmp"
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, target)
    return target


def format_evaluation_summary(result: Dict[str, Any]) -> str:
    """Short console summary (a few lines)."""
    metrics = result.get("metrics", {})

    def value(name):
        metric = metrics.get(name, {})
        return metric.get("value"), metric.get("status")

    def fmt(name, percent=True):
        val, status = value(name)
        if val is None or status in (m.STATUS_NOT_EVALUATED, m.STATUS_ERROR):
            return status or "n/a"
        return f"{val:.0%}" if percent else str(val)

    ver = (metrics.get("claim_verification", {}).get("details") or {})
    lines = [
        f"Overall evaluation status: {result.get('overall_status')} "
        "(describes the evaluation, NOT scientific validity)",
        f"  Papers: {fmt('retrieval_record_count', False)} | "
        f"usable abstracts: {fmt('retrieval_abstract_availability')} | "
        f"duplicate rate: {fmt('retrieval_duplicates')}",
        f"  Sub-question coverage (heuristic estimate): "
        f"{fmt('subquestion_coverage')}",
        f"  Citations resolving to known papers: "
        f"{fmt('citation_traceability')}",
        f"  Verification: state={ver.get('verification_state', 'n/a')}, "
        f"flagged for review={ver.get('flagged_for_review', 'n/a')}",
        f"  Report sections present: {fmt('report_completeness')}",
        f"  Precision@k/recall@k: "
        f"{fmt('retrieval_precision_at_k')}/{fmt('retrieval_recall_at_k')}",
    ]
    for warning in result.get("warnings", [])[:8]:
        lines.append(f"  WARNING: {warning}")
    return "\n".join(lines)
