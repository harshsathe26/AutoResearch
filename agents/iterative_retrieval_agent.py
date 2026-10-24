"""
Iterative Retrieval Agent (Phase 6).

Runs AFTER the pipeline's initial retrieval and BEFORE Graph RAG, analysis
and verification. If the evidence collected so far looks insufficient, it
performs a small, bounded number of extra retrieval attempts with
reformulated queries, then returns the merged, deduplicated papers plus a
structured summary of what happened.

Everything here is deterministic: there is no LLM call. It reuses the
project's existing retrieval code:

* ``retrieve_for_sub_questions``  - does the actual searching and scoring
* ``calculate_relevance_score``   - the existing 0.0-1.0 keyword relevance
* ``normalize_paper_id`` / ``normalize_paper_title`` - duplicate detection
* ``build_search_query``          - the query the search layer really sends

Evidence is "sufficient" only when ALL of these hold (defaults in
``IterativeRetrievalConfig``):

1. At least ``min_papers`` (3) unique usable papers.
2. At least ``min_papers_with_abstracts`` (3) of them have a real abstract
   (the Analysis Agent can only use abstracts).
3. At least ``min_papers`` papers are relevant: relevance score >=
   ``min_relevance_score`` (0.30, the retriever's own threshold). The
   retriever's stored ``relevance_score`` is used when present. If it is
   missing, the existing scoring function is applied to the original
   question as a documented fallback; the value is never written into the
   paper record.
4. Sub-question coverage >= ``min_subquestion_coverage`` (1.0, i.e. every
   sub-question). A sub-question counts as covered when at least one paper
   WITH an abstract (a) scores >= ``min_relevance_score`` against that
   sub-question text and (b) mentions at least one term that is specific to
   the sub-question, i.e. not part of the plan's main topic. Without (b),
   every paper on the shared topic would "cover" every sub-question. If the
   plan has no main topic, only (a) is applied. This is a keyword-overlap
   heuristic, not proof that the paper answers the sub-question. With no
   valid sub-questions, coverage is "unknown" and is not a criterion.

Not implemented on purpose: the roadmap's 50% claim-support threshold.
Before analysis there are no claims, so no genuine support measurement
exists at this stage (it is reported as "not_assessed"). Retrieval success
never marks a claim as verified.

Known limitation: ``search_papers`` already catches network failures and
returns an empty list, so a failed search and a search with no results are
indistinguishable here. Both end as ``no_new_papers``. Only exceptions that
escape the retrieval layer are recorded as ``retrieval_error``.
"""

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from agents.analysis_agent import redact_secrets
from retrieval.paper_search import build_search_query
from retrieval.retriever import (
    MIN_RELEVANCE_SCORE,
    calculate_relevance_score,
    normalize_paper_id,
    normalize_paper_title,
    retrieve_for_sub_questions,
)

# ---------------------------------------------------------------------------
# Stop reasons (the complete, fixed set)
# ---------------------------------------------------------------------------

SUFFICIENT_EVIDENCE = "sufficient_evidence"
MAX_ATTEMPTS_REACHED = "max_attempts_reached"
NO_NEW_PAPERS = "no_new_papers"
NO_NEW_QUERIES = "no_new_queries"
RETRIEVAL_ERROR = "retrieval_error"
NO_SUBQUESTIONS_AVAILABLE = "no_subquestions_available"

STOP_REASONS = (
    SUFFICIENT_EVIDENCE, MAX_ATTEMPTS_REACHED, NO_NEW_PAPERS,
    NO_NEW_QUERIES, RETRIEVAL_ERROR, NO_SUBQUESTIONS_AVAILABLE,
)

# Errors that can legitimately escape the retrieval layer. Anything else
# (TypeError, KeyError, ...) is a programming error and is NOT caught.
RECOVERABLE_ERRORS = (
    requests.exceptions.RequestException, ConnectionError, TimeoutError,
)

MAX_ALLOWED_ATTEMPTS = 5     # hard ceiling so the loop can never be unbounded
MAX_ERROR_TEXT = 300
PLACEHOLDER_TITLES = {"title not available", "untitled", ""}
PLACEHOLDER_ABSTRACTS = {"abstract not available", ""}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass
class IterativeRetrievalConfig:
    """Tunable limits and thresholds. Invalid values raise ValueError."""

    max_additional_attempts: int = 2        # extra attempts after the initial
    min_papers: int = 3                     # unique usable papers
    min_papers_with_abstracts: int = 3
    min_relevance_score: float = MIN_RELEVANCE_SCORE
    min_subquestion_coverage: float = 1.0   # fraction of sub-questions
    queries_per_attempt: int = 3
    papers_per_query: int = 10

    def __post_init__(self) -> None:
        if not _is_int(self.max_additional_attempts) or not (
                0 <= self.max_additional_attempts <= MAX_ALLOWED_ATTEMPTS):
            raise ValueError(
                "max_additional_attempts must be an integer from 0 to "
                f"{MAX_ALLOWED_ATTEMPTS}.")
        if not _is_int(self.min_papers) or self.min_papers < 1:
            raise ValueError("min_papers must be an integer >= 1.")
        if (not _is_int(self.min_papers_with_abstracts)
                or self.min_papers_with_abstracts < 0):
            raise ValueError(
                "min_papers_with_abstracts must be an integer >= 0.")
        if not _is_number(self.min_relevance_score) or not (
                0.0 <= self.min_relevance_score <= 1.0):
            raise ValueError("min_relevance_score must be between 0 and 1.")
        if not _is_number(self.min_subquestion_coverage) or not (
                0.0 < self.min_subquestion_coverage <= 1.0):
            raise ValueError(
                "min_subquestion_coverage must be above 0 and at most 1.")
        if not _is_int(self.queries_per_attempt) or not (
                1 <= self.queries_per_attempt <= 5):
            raise ValueError("queries_per_attempt must be an integer 1-5.")
        if not _is_int(self.papers_per_query) or not (
                1 <= self.papers_per_query <= 25):
            raise ValueError("papers_per_query must be an integer 1-25.")


# ---------------------------------------------------------------------------
# Paper helpers
# ---------------------------------------------------------------------------

def _clean_str(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _real_title(paper: dict) -> str:
    title = _clean_str(paper.get("title"))
    return "" if title.casefold() in PLACEHOLDER_TITLES else title


def _real_abstract(paper: dict) -> str:
    abstract = _clean_str(paper.get("abstract"))
    return "" if abstract.casefold() in PLACEHOLDER_ABSTRACTS else abstract


def is_usable_paper(paper: Any) -> bool:
    """A usable paper is a dict with a real title. Abstract and other
    metadata are assessed separately."""
    return isinstance(paper, dict) and bool(_real_title(paper))


def _strong_ids(paper: dict) -> Dict[str, set]:
    """DOIs and OpenAlex IDs found in doi/url/id, via the retriever's own
    normalizer so identifier rules stay identical across the project."""
    found: Dict[str, set] = {"doi": set(), "openalex": set()}
    for field in ("doi", "url", "id"):
        value = paper.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        for probe in ({"doi": value}, {"id": value}):
            key = normalize_paper_id(probe)
            if key and key.startswith("doi:"):
                found["doi"].add(key)
            elif key and key.startswith("openalex:"):
                found["openalex"].add(key)
    return found


def _url_key(paper: dict) -> str:
    url = paper.get("url")
    if not isinstance(url, str) or not url.strip():
        return ""
    url = re.sub(r"^https?://(www\.)?", "", url.strip().casefold())
    return url.rstrip("/")


def _raw_id(paper: dict) -> str:
    value = paper.get("id")
    return str(value).strip().casefold() if value not in (None, "") else ""


def is_same_paper(a: dict, b: dict) -> bool:
    """
    Decide whether two records are the same paper.

    Shared DOI / OpenAlex ID, identical raw ID, or identical URL means yes.
    Otherwise an identical normalized title means yes, UNLESS both records
    carry different DOIs (or different OpenAlex IDs): then they are treated
    as distinct papers (e.g. two versions with their own DOIs). Titles are
    compared exactly after normalization, never fuzzily.
    """
    ids_a, ids_b = _strong_ids(a), _strong_ids(b)

    for kind in ("doi", "openalex"):
        if ids_a[kind] & ids_b[kind]:
            return True

    if _raw_id(a) and _raw_id(a) == _raw_id(b):
        return True
    if _url_key(a) and _url_key(a) == _url_key(b):
        return True

    title_a, title_b = normalize_paper_title(a), normalize_paper_title(b)
    if not title_a or title_a != title_b:
        return False

    for kind in ("doi", "openalex"):
        if ids_a[kind] and ids_b[kind] and not (ids_a[kind] & ids_b[kind]):
            return False
    return True


def _merge_two(base: dict, other: dict) -> dict:
    """Fold ``other`` into a copy of ``base``. Existing valid values are
    never overwritten by missing or weaker ones."""
    merged = dict(base)

    for key, value in other.items():
        if key in ("relevance_score", "source_questions", "abstract",
                   "title", "authors"):
            continue
        if merged.get(key) in (None, "", [], {}) and value not in (
                None, "", [], {}):
            merged[key] = value

    if not _real_title(merged) and _real_title(other):
        merged["title"] = other["title"]

    # Never overwrite a real abstract; only fill a missing/placeholder one.
    if not _real_abstract(merged) and _real_abstract(other):
        merged["abstract"] = other["abstract"]

    if not merged.get("authors") and isinstance(other.get("authors"), list) \
            and other["authors"]:
        merged["authors"] = list(other["authors"])

    scores = [s for s in (base.get("relevance_score"),
                          other.get("relevance_score")) if _is_number(s)]
    if scores:
        merged["relevance_score"] = max(scores)

    questions = []
    for source in (base, other):
        for question in source.get("source_questions") or []:
            if isinstance(question, str) and question not in questions:
                questions.append(question)
    if questions:
        merged["source_questions"] = questions

    return merged


def merge_papers(existing: List[dict], new: Any) -> Tuple[List[dict], int, int]:
    """
    Merge ``new`` records into ``existing`` without mutating either input.

    Returns (merged_list, number_of_truly_new_papers, invalid_dropped).
    Unusable records (non-dicts, no real title) are dropped.
    """
    merged: List[dict] = []
    dropped = 0
    added = 0

    for position, source in enumerate((existing, new)):
        if not isinstance(source, list):
            continue
        for paper in source:
            if not is_usable_paper(paper):
                dropped += 1
                continue
            for index, current in enumerate(merged):
                if is_same_paper(current, paper):
                    merged[index] = _merge_two(current, paper)
                    break
            else:
                merged.append(copy.deepcopy(paper))
                if position == 1:
                    added += 1

    return merged, added, dropped


# ---------------------------------------------------------------------------
# Evidence assessment
# ---------------------------------------------------------------------------

def _valid_subquestions(plan: Any) -> List[str]:
    """Non-empty, unique sub-question strings from the plan (order kept)."""
    raw = plan.get("sub_questions") if isinstance(plan, dict) else None
    result: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            text = _clean_str(item)
            if text and text not in result:
                result.append(text)
    return result


def _stem(word: str) -> str:
    """Tiny plural normalizer so "attacks" matches "attack"."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _paper_terms(paper: dict) -> set:
    text = f"{_clean_str(paper.get('title'))} {_clean_str(paper.get('abstract'))}"
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.casefold())}


def _specific_terms(sub_question: str, topic: Optional[str]) -> List[str]:
    """Keywords of the sub-question that are NOT part of the main topic.
    Empty when no topic is known (then no specificity is required)."""
    topic_terms = {_stem(w) for w in _keywords(topic)}
    if not topic_terms:
        return []
    return [w for w in _keywords(sub_question)
            if _stem(w) not in topic_terms]


def _covers(paper: dict, sub_question: str, specific: List[str],
            threshold: float) -> bool:
    if not _real_abstract(paper):
        return False
    if calculate_relevance_score(sub_question, paper) < threshold:
        return False
    return not specific or any(
        _stem(term) in _paper_terms(paper) for term in specific)


def assess_evidence(user_query: str, sub_questions: List[str],
                    papers: List[dict],
                    config: Optional[IterativeRetrievalConfig] = None,
                    topic: Optional[str] = None) -> Dict[str, Any]:
    """Apply the sufficiency rules to a paper list. Pure; no retrieval.
    ``topic`` is the plan's main topic (used for sub-question coverage)."""
    config = config or IterativeRetrievalConfig()
    usable = [p for p in papers if is_usable_paper(p)]
    with_abstract = [p for p in usable if _real_abstract(p)]

    from_retriever = fallback = relevant = 0
    for paper in usable:
        score = paper.get("relevance_score")
        if _is_number(score):
            from_retriever += 1
        else:
            fallback += 1
            score = calculate_relevance_score(user_query, paper)
        if score >= config.min_relevance_score:
            relevant += 1

    covered: List[str] = []
    uncovered: List[str] = []
    for question in sub_questions:
        specific = _specific_terms(question, topic)
        if any(_covers(p, question, specific, config.min_relevance_score)
               for p in with_abstract):
            covered.append(question)
        else:
            uncovered.append(question)

    if sub_questions:
        fraction = len(covered) / len(sub_questions)
        coverage_ok = fraction >= config.min_subquestion_coverage
        coverage = {"status": "assessed", "fraction": fraction,
                    "covered": covered, "uncovered": uncovered}
    else:
        coverage_ok = True   # unknown: not used as a criterion
        coverage = {"status": "unknown", "fraction": None,
                    "covered": [], "uncovered": []}

    criteria = {
        "enough_papers": len(usable) >= config.min_papers,
        "enough_abstracts":
            len(with_abstract) >= config.min_papers_with_abstracts,
        "enough_relevant": relevant >= config.min_papers,
        "coverage_ok": coverage_ok,
    }
    return {
        "usable_papers": len(usable),
        "papers_with_abstracts": len(with_abstract),
        "relevant_papers": relevant,
        "relevance_from_retriever": from_retriever,
        "relevance_fallback_used": fallback,
        "coverage": coverage,
        "criteria": criteria,
        "sufficient": all(criteria.values()),
        "claim_support": "not_assessed",
    }


# ---------------------------------------------------------------------------
# Query reformulation (deterministic)
# ---------------------------------------------------------------------------

def _keywords(text: Any) -> List[str]:
    """Content words, exactly as the search layer would extract them."""
    return build_search_query(text).split() if isinstance(text, str) else []


def _query_key(query: str) -> str:
    return build_search_query(query)


def generate_candidate_queries(
    user_query: str,
    plan: Any,
    targets: List[str],
    attempted_keys: set,
) -> List[Dict[str, Any]]:
    """
    Ordered, unique candidate queries not yet attempted.

    Strategies, in priority order (each is built from the plan/question):
      shortened_subquestion  - first 5 keywords of a target sub-question
      topic_plus_subquestion - main-topic keywords + that sub-question's
                               own keywords
      shortened_question     - first 6 keywords of the original question
      main_topic             - the plan's main topic
      research_objective     - first 6 keywords of the plan's objective

    A candidate is skipped if its search keywords equal a query already
    attempted, so the same search is never issued twice.
    """
    plan = plan if isinstance(plan, dict) else {}
    topic_words = (_keywords(plan.get("main_topic"))
                   or _keywords(user_query))[:4]

    raw: List[Tuple[str, str, Optional[str]]] = []

    for target in targets:
        words = _keywords(target)
        if len(words) > 5:
            raw.append(("shortened_subquestion", " ".join(words[:5]), target))
    for target in targets:
        own = [w for w in _keywords(target) if w not in topic_words][:3]
        if own:
            raw.append(("topic_plus_subquestion",
                        " ".join(topic_words[:3] + own), target))

    raw.append(("shortened_question", " ".join(_keywords(user_query)[:6]),
                None))
    raw.append(("main_topic", " ".join(topic_words), None))
    raw.append(("research_objective",
                " ".join(_keywords(plan.get("research_objective"))[:6]),
                None))

    seen = set(attempted_keys)
    candidates = []
    for strategy, query, target in raw:
        key = _query_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append({"strategy": strategy, "query": query,
                           "target_subquestion": target, "key": key})
    return candidates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _safe_error(exc: BaseException) -> str:
    text = redact_secrets(f"{type(exc).__name__}: {exc}")
    return text[:MAX_ERROR_TEXT]


def run_iterative_retrieval(
    user_query: str,
    plan: Optional[dict],
    papers: Optional[list],
    config: Optional[IterativeRetrievalConfig] = None,
    retrieve_fn: Optional[Callable[..., Any]] = None,
    initial_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assess the initial papers and, if needed, run bounded extra retrieval.

    ``retrieve_fn(queries, papers_per_question=n)`` defaults to the
    project's ``retrieve_for_sub_questions``. ``initial_queries`` are the
    queries the pipeline already used (default: the plan's sub-questions);
    they are never repeated.

    The input list and its papers are never mutated. Previously found
    papers are kept whatever happens later.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("Research question cannot be empty.")
    config = config or IterativeRetrievalConfig()

    sub_questions = _valid_subquestions(plan)
    topic = _clean_str(plan.get("main_topic")) if isinstance(
        plan, dict) else ""
    current, _, dropped = merge_papers([], papers)
    initial_count = len(current)

    attempted_keys = {
        _query_key(q) for q in (initial_queries
                                if initial_queries is not None
                                else sub_questions)
        if isinstance(q, str) and _query_key(q)
    }

    queries_attempted: List[str] = []
    attempt_log: List[Dict[str, Any]] = []
    errors: List[str] = []
    total_new = 0
    attempts_done = 0

    initial_assessment = assess_evidence(
        user_query, sub_questions, current, config, topic)
    assessment = initial_assessment
    stop_reason = None

    while True:
        if assessment["sufficient"]:
            stop_reason = SUFFICIENT_EVIDENCE
            break
        if attempts_done >= config.max_additional_attempts:
            stop_reason = MAX_ATTEMPTS_REACHED
            break

        # Aim at uncovered sub-questions; if all are covered but the
        # evidence is still thin, widen to every sub-question.
        targets = assessment["coverage"]["uncovered"] or sub_questions
        candidates = generate_candidate_queries(
            user_query, plan, targets, attempted_keys
        )[:config.queries_per_attempt]

        if not candidates:
            stop_reason = (NO_SUBQUESTIONS_AVAILABLE if not sub_questions
                           else NO_NEW_QUERIES)
            break

        for candidate in candidates:
            attempted_keys.add(candidate["key"])
            queries_attempted.append(candidate["query"])

        fetch = retrieve_fn or retrieve_for_sub_questions
        attempts_done += 1
        entry = {"attempt": attempts_done,
                 "queries": [c["query"] for c in candidates],
                 "strategies": [c["strategy"] for c in candidates],
                 "new_papers": 0, "error": None}
        attempt_log.append(entry)

        try:
            result = fetch([c["query"] for c in candidates],
                           papers_per_question=config.papers_per_query)
        except RECOVERABLE_ERRORS as exc:
            entry["error"] = _safe_error(exc)
            errors.append(f"Attempt {attempts_done}: {entry['error']}")
            stop_reason = RETRIEVAL_ERROR
            break

        if not isinstance(result, list):
            entry["error"] = ("Retrieval returned an unexpected result "
                              f"type: {type(result).__name__}")
            errors.append(f"Attempt {attempts_done}: {entry['error']}")
            stop_reason = RETRIEVAL_ERROR
            break

        current, added, bad = merge_papers(current, result)
        dropped += bad
        entry["new_papers"] = added
        total_new += added

        assessment = assess_evidence(user_query, sub_questions, current,
                                     config, topic)
        if added == 0 and not assessment["sufficient"]:
            stop_reason = NO_NEW_PAPERS
            break

    return {
        "papers": current,
        "initial_paper_count": initial_count,
        "final_paper_count": len(current),
        "additional_attempts": attempts_done,
        "max_additional_attempts": config.max_additional_attempts,
        "queries_attempted": queries_attempted,
        "attempts": attempt_log,
        "new_papers_found": total_new,
        "sufficient": assessment["sufficient"],
        "stop_reason": stop_reason,
        "initial_assessment": initial_assessment,
        "final_assessment": assessment,
        "invalid_records_dropped": dropped,
        "errors": errors,
    }


def format_retrieval_summary(result: Dict[str, Any]) -> str:
    """Short, terminal-friendly summary of a retrieval result."""
    if not isinstance(result, dict):
        return "Retrieval summary unavailable."

    final = result.get("final_assessment", {})
    coverage = final.get("coverage", {})
    if coverage.get("status") == "assessed":
        covered = len(coverage.get("covered", []))
        total = covered + len(coverage.get("uncovered", []))
        coverage_text = f"{covered}/{total} sub-questions covered"
    else:
        coverage_text = "sub-question coverage unknown"

    lines = [
        f"Initial papers: {result.get('initial_paper_count', 0)} | "
        f"Final papers: {result.get('final_paper_count', 0)} | "
        f"Extra attempts: {result.get('additional_attempts', 0)}"
        f"/{result.get('max_additional_attempts', 0)} | "
        f"Stop reason: {result.get('stop_reason')}",
        f"Evidence judged sufficient: "
        f"{'yes' if result.get('sufficient') else 'no'} "
        f"({final.get('papers_with_abstracts', 0)} with abstracts, "
        f"{coverage_text})",
    ]
    if not result.get("sufficient"):
        lines.append(
            "Note: evidence is limited; analysis and verification results "
            "should be treated as preliminary.")
    for query in result.get("queries_attempted", []):
        lines.append(f"  extra query: {query}")
    for error in result.get("errors", []):
        lines.append(f"  WARNING: {error}")
    return "\n".join(lines)