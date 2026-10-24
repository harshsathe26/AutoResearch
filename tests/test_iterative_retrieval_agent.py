"""
Tests for the Iterative Retrieval Agent (Phase 6).

Retrieval is always mocked: no network access and no Gemini key needed.
Run with:  python -m pytest -v
"""

import copy
from unittest.mock import MagicMock, patch

import pytest
import requests

from agents import iterative_retrieval_agent as ira
from agents.iterative_retrieval_agent import (
    IterativeRetrievalConfig,
    assess_evidence,
    format_retrieval_summary,
    generate_candidate_queries,
    is_same_paper,
    merge_papers,
    run_iterative_retrieval,
)

QUESTION = ("How do federated learning systems protect privacy "
            "against attacks")
SUB_Q1 = "How does federated learning protect privacy"
SUB_Q2 = "What attacks target federated learning"
PLAN = {
    "main_topic": "Federated learning privacy",
    "research_objective": "Understand privacy protection and attacks in "
                          "federated learning",
    "research_plan": ["Define", "Survey"],
    "sub_questions": [SUB_Q1, SUB_Q2],
}

# Contains the words of both sub-questions, so it "covers" both.
GOOD_ABSTRACT = ("federated learning protect privacy attacks target "
                 "federated learning systems")


def make_paper(n, abstract=GOOD_ABSTRACT, score=0.6, **extra):
    paper = {
        "id": f"https://openalex.org/W{n}",
        "title": f"Federated Privacy Study Number {n}",
        "abstract": abstract,
        "authors": [f"Author {n}"],
        "year": 2020 + n % 5,
        "doi": f"https://doi.org/10.1234/paper{n}",
        "url": f"https://doi.org/10.1234/paper{n}",
        "source_questions": [SUB_Q1],
    }
    if score is not None:
        paper["relevance_score"] = score
    paper.update(extra)
    return paper


def good_papers(count, start=1):
    return [make_paper(n) for n in range(start, start + count)]


def retriever(*batches):
    """Mock retrieval: returns each batch in turn; Exception => raised."""
    mock = MagicMock()
    mock.side_effect = [
        batch if isinstance(batch, BaseException) or not isinstance(
            batch, type) else batch
        for batch in batches
    ]
    return mock


def run(papers, fn, plan=PLAN, config=None, **kwargs):
    return run_iterative_retrieval(
        QUESTION, plan, papers, config=config, retrieve_fn=fn, **kwargs)


# ---------------------------------------------------------------------------
# 1-3. Sufficiency, triggering, discovering papers
# ---------------------------------------------------------------------------

def test_01_sufficient_evidence_means_no_additional_retrieval():
    fn = MagicMock()
    result = run(good_papers(3), fn)
    fn.assert_not_called()
    assert result["stop_reason"] == "sufficient_evidence"
    assert result["sufficient"] is True
    assert result["additional_attempts"] == 0
    assert result["queries_attempted"] == []
    assert result["final_paper_count"] == 3


def test_02_fewer_than_three_papers_triggers_retrieval():
    fn = retriever(good_papers(2, start=10))
    result = run(good_papers(2), fn)
    assert fn.call_count == 1
    assert result["additional_attempts"] == 1
    assert result["initial_paper_count"] == 2


def test_03_new_useful_papers_are_discovered_and_merged():
    fn = retriever(good_papers(2, start=10))
    result = run(good_papers(2), fn)
    assert result["final_paper_count"] == 4
    assert result["new_papers_found"] == 2
    assert result["sufficient"] is True
    assert result["stop_reason"] == "sufficient_evidence"
    titles = [p["title"] for p in result["papers"]]
    assert "Federated Privacy Study Number 10" in titles
    assert "Federated Privacy Study Number 1" in titles


def test_papers_without_abstracts_do_not_count_as_sufficient():
    no_abstract = [make_paper(n, abstract="Abstract not available")
                   for n in range(1, 5)]
    assessment = assess_evidence(QUESTION, [], no_abstract)
    assert assessment["usable_papers"] == 4
    assert assessment["papers_with_abstracts"] == 0
    assert assessment["criteria"]["enough_abstracts"] is False
    assert assessment["sufficient"] is False


# ---------------------------------------------------------------------------
# 4-5. Deduplication
# ---------------------------------------------------------------------------

def test_04_duplicates_across_attempts_are_removed():
    cfg = IterativeRetrievalConfig(min_papers=6, min_papers_with_abstracts=1)
    base = good_papers(2)
    attempt1 = [copy.deepcopy(base[0]), make_paper(10)]
    attempt2 = [make_paper(10), make_paper(11)]
    fn = retriever(attempt1, attempt2)
    result = run(base, fn, config=cfg)
    ids = [p["id"] for p in result["papers"]]
    assert len(ids) == len(set(ids)) == 4
    assert result["new_papers_found"] == 2


def test_05_duplicate_result_does_not_inflate_count():
    base = good_papers(2)
    fn = retriever([copy.deepcopy(base[0]), copy.deepcopy(base[1])])
    result = run(base, fn)
    assert result["final_paper_count"] == 2
    assert result["new_papers_found"] == 0
    assert result["stop_reason"] == "no_new_papers"


def test_same_paper_with_doi_in_one_record_and_openalex_in_other():
    a = {"id": "https://openalex.org/W5", "title": "Same Paper Title",
         "abstract": "x", "url": "https://openalex.org/W5"}
    b = {"id": "https://doi.org/10.9999/abc", "title": "same paper title!",
         "doi": "https://doi.org/10.9999/abc",
         "url": "https://doi.org/10.9999/abc", "abstract": "x"}
    assert is_same_paper(a, b)
    merged, added, _ = merge_papers([a], [b])
    assert len(merged) == 1 and added == 0


def test_different_dois_with_same_title_stay_distinct():
    a = make_paper(1, title="Introduction",
                   doi="https://doi.org/10.1111/aaa")
    b = make_paper(2, title="Introduction",
                   doi="https://doi.org/10.2222/bbb")
    a["url"] = a["doi"]
    b["url"] = b["doi"]
    assert not is_same_paper(a, b)
    merged, added, _ = merge_papers([a], [b])
    assert len(merged) == 2 and added == 1


def test_similar_but_not_identical_titles_are_not_merged():
    a = {"id": "a", "title": "Federated Learning for Privacy"}
    b = {"id": "b", "title": "Federated Learning for Privacy in Health"}
    assert not is_same_paper(a, b)


def test_url_fallback_matches_same_paper():
    a = {"id": "x1", "title": "Alpha", "url": "https://Example.org/p/1/"}
    b = {"id": "x2", "title": "Beta", "url": "http://www.example.org/p/1"}
    assert is_same_paper(a, b)


def test_merge_never_overwrites_valid_data_with_weaker_data():
    existing = make_paper(1, score=0.8)
    weaker = {
        "id": existing["id"], "title": "Different casing title",
        "abstract": "Abstract not available", "doi": None, "url": "",
        "authors": [], "relevance_score": 0.4,
        "source_questions": ["other query"],
    }
    merged, _, _ = merge_papers([existing], [weaker])
    paper = merged[0]
    assert paper["abstract"] == existing["abstract"]
    assert paper["doi"] == existing["doi"]
    assert paper["url"] == existing["url"]
    assert paper["authors"] == existing["authors"]
    assert paper["title"] == existing["title"]
    assert paper["relevance_score"] == 0.8          # best score kept
    assert paper["source_questions"] == [SUB_Q1, "other query"]
    assert paper["id"] == existing["id"]            # ID semantics unchanged


def test_merge_fills_missing_abstract_and_keeps_higher_score():
    existing = make_paper(1, abstract="Abstract not available", score=0.35)
    richer = make_paper(1, abstract="A real abstract about privacy.",
                        score=0.7)
    merged, _, _ = merge_papers([existing], [richer])
    assert merged[0]["abstract"] == "A real abstract about privacy."
    assert merged[0]["relevance_score"] == 0.7


def test_duplicate_can_supply_missing_abstract_and_make_evidence_sufficient():
    cfg = IterativeRetrievalConfig(min_papers=3, min_papers_with_abstracts=3)
    base = [make_paper(n, abstract="Abstract not available")
            for n in range(1, 4)]
    fn = retriever([make_paper(n) for n in range(1, 4)])  # same papers
    result = run(base, fn, config=cfg)
    assert result["new_papers_found"] == 0
    assert result["sufficient"] is True
    assert result["stop_reason"] == "sufficient_evidence"


def test_merge_does_not_mutate_inputs():
    existing = [make_paper(1, abstract="Abstract not available")]
    snapshot = copy.deepcopy(existing)
    incoming = [make_paper(1)]
    incoming_snapshot = copy.deepcopy(incoming)
    merge_papers(existing, incoming)
    assert existing == snapshot and incoming == incoming_snapshot


# ---------------------------------------------------------------------------
# 6-7. Limits and termination
# ---------------------------------------------------------------------------

def _endless_new_papers():
    counter = {"n": 100}

    def fetch(queries, papers_per_question=10):
        counter["n"] += 1
        return [make_paper(counter["n"])]
    return MagicMock(side_effect=fetch)


def test_06_maximum_additional_attempts_is_respected():
    cfg = IterativeRetrievalConfig(min_papers=50, min_papers_with_abstracts=1)
    fn = _endless_new_papers()
    result = run(good_papers(1), fn, config=cfg)
    assert fn.call_count == 2                      # default limit
    assert result["additional_attempts"] == 2
    assert result["stop_reason"] == "max_attempts_reached"
    assert result["sufficient"] is False


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_attempt_limit_is_configurable(limit):
    cfg = IterativeRetrievalConfig(max_additional_attempts=limit,
                                   queries_per_attempt=1,
                                   min_papers=50,
                                   min_papers_with_abstracts=1)
    fn = _endless_new_papers()
    result = run(good_papers(1), fn, config=cfg)
    assert fn.call_count == limit
    assert result["additional_attempts"] == limit
    assert result["stop_reason"] == "max_attempts_reached"


def test_07_no_new_papers_stops_retrieval():
    fn = retriever([])
    result = run(good_papers(1), fn)
    assert fn.call_count == 1
    assert result["stop_reason"] == "no_new_papers"
    assert result["final_paper_count"] == 1
    assert result["sufficient"] is False


# ---------------------------------------------------------------------------
# 8, 14. Errors
# ---------------------------------------------------------------------------

def test_08_recoverable_error_preserves_existing_papers():
    base = good_papers(2)
    fn = MagicMock(side_effect=requests.exceptions.ConnectionError("down"))
    result = run(base, fn)
    assert result["stop_reason"] == "retrieval_error"
    assert [p["id"] for p in result["papers"]] == [p["id"] for p in base]
    assert result["errors"] and "ConnectionError" in result["errors"][0]
    assert fn.call_count == 1                       # no hammering


@pytest.mark.parametrize("error", [
    TimeoutError("slow"), ConnectionError("reset"),
    requests.exceptions.Timeout("t"), requests.exceptions.HTTPError("429"),
])
def test_recoverable_error_types(error):
    result = run(good_papers(1), MagicMock(side_effect=error))
    assert result["stop_reason"] == "retrieval_error"
    assert result["final_paper_count"] == 1


def test_programming_errors_are_not_swallowed():
    fn = MagicMock(side_effect=TypeError("bug"))
    with pytest.raises(TypeError):
        run(good_papers(1), fn)


def test_14_later_failure_keeps_papers_found_earlier():
    cfg = IterativeRetrievalConfig(min_papers=50, min_papers_with_abstracts=1)
    found = [make_paper(20), make_paper(21)]
    fn = MagicMock(side_effect=[found, requests.exceptions.Timeout("late")])
    result = run(good_papers(1), fn, config=cfg)
    assert result["stop_reason"] == "retrieval_error"
    assert result["additional_attempts"] == 2
    assert result["new_papers_found"] == 2
    ids = {p["id"] for p in result["papers"]}
    assert {"https://openalex.org/W20", "https://openalex.org/W21"} <= ids
    assert result["final_paper_count"] == 3


def test_error_text_is_sanitized_and_bounded():
    secret = "AIza" + "s" * 30
    long_text = f"key={secret} " + "x" * 1000
    fn = MagicMock(side_effect=ConnectionError(long_text))
    result = run(good_papers(1), fn)
    text = " ".join(result["errors"])
    assert secret not in text
    assert len(result["errors"][0]) < 400


def test_unexpected_result_type_is_recorded_not_crashed():
    fn = MagicMock(return_value="not a list")
    result = run(good_papers(1), fn)
    assert result["stop_reason"] == "retrieval_error"
    assert "unexpected result type" in result["errors"][0]


# ---------------------------------------------------------------------------
# 9-12. Bad input
# ---------------------------------------------------------------------------

def test_09_empty_initial_result_is_handled():
    fn = retriever(good_papers(3))
    result = run([], fn)
    assert result["initial_paper_count"] == 0
    assert result["final_paper_count"] == 3
    assert result["stop_reason"] == "sufficient_evidence"


@pytest.mark.parametrize("empty", [None, [], "nope", 5, {}])
def test_empty_or_wrong_type_paper_input(empty):
    result = run(empty, MagicMock(return_value=[]))
    assert result["papers"] == []
    assert result["stop_reason"] == "no_new_papers"


def test_10_invalid_records_do_not_crash():
    junk = [None, "text", 7, {}, {"title": ""}, {"title": "Title not available"},
            {"abstract": "no title at all"}, [], make_paper(1)]
    fn = retriever(good_papers(2, start=10))
    result = run(junk, fn)
    assert result["initial_paper_count"] == 1
    assert result["invalid_records_dropped"] == 8
    assert result["final_paper_count"] == 3


def test_malformed_records_returned_by_retrieval_are_dropped():
    fn = retriever([None, {"title": ""}, make_paper(30), "junk"], [])
    result = run(good_papers(1), fn)
    assert result["final_paper_count"] == 2
    assert result["invalid_records_dropped"] == 3


def test_11_missing_relevance_scores_are_not_invented():
    papers = [make_paper(n, score=None) for n in range(1, 4)]
    assert all("relevance_score" not in p for p in papers)
    result = run(papers, MagicMock())
    for paper in result["papers"]:
        assert "relevance_score" not in paper          # nothing invented
    final = result["final_assessment"]
    assert final["relevance_fallback_used"] == 3
    assert final["relevance_from_retriever"] == 0


def test_missing_scores_with_irrelevant_papers_are_not_relevant():
    irrelevant = [make_paper(n, score=None, title=f"Cooking {n}",
                             abstract="pasta recipes and sauces")
                  for n in range(1, 4)]
    assessment = assess_evidence(QUESTION, [], irrelevant)
    assert assessment["relevant_papers"] == 0
    assert assessment["criteria"]["enough_relevant"] is False
    assert assessment["sufficient"] is False


def test_low_retriever_scores_are_respected():
    low = [make_paper(n, score=0.05) for n in range(1, 4)]
    assessment = assess_evidence(QUESTION, [], low)
    assert assessment["relevance_from_retriever"] == 3
    assert assessment["relevant_papers"] == 0
    assert assessment["sufficient"] is False


@pytest.mark.parametrize("plan", [
    None, {}, "plan", {"sub_questions": None}, {"sub_questions": "abc"},
    {"sub_questions": [None, 5, "", "   "]}, {"sub_questions": {"a": 1}},
])
def test_12_malformed_or_missing_subquestions(plan):
    result = run(good_papers(3), MagicMock(), plan=plan)
    assert result["final_assessment"]["coverage"]["status"] == "unknown"
    assert result["stop_reason"] == "sufficient_evidence"


def test_duplicate_subquestions_are_counted_once():
    plan = {"sub_questions": [SUB_Q1, SUB_Q1, f"  {SUB_Q1}  "]}
    result = run(good_papers(3), MagicMock(), plan=plan)
    coverage = result["final_assessment"]["coverage"]
    assert coverage["covered"] == [SUB_Q1]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        run_iterative_retrieval("  ", PLAN, [])


# ---------------------------------------------------------------------------
# Sub-question coverage
# ---------------------------------------------------------------------------

def test_uncovered_subquestion_triggers_retrieval_and_is_targeted():
    only_q1 = [make_paper(n, abstract="federated learning protect privacy")
               for n in range(1, 4)]
    assessment = assess_evidence(QUESTION, [SUB_Q1, SUB_Q2], only_q1,
                                 topic=PLAN["main_topic"])
    assert assessment["coverage"]["covered"] == [SUB_Q1]
    assert assessment["coverage"]["uncovered"] == [SUB_Q2]
    assert assessment["sufficient"] is False

    fn = retriever(good_papers(1, start=50))
    result = run(only_q1, fn)
    assert fn.call_count == 1
    assert result["final_assessment"]["coverage"]["uncovered"] == []
    assert result["stop_reason"] == "sufficient_evidence"
    # The first extra query is built from the UNCOVERED sub-question.
    sent = fn.call_args.args[0]
    assert any("attacks" in q for q in sent)


def test_title_word_alone_does_not_cover_a_subquestion():
    # Title mentions the topic, but there is no abstract at all.
    title_only = [make_paper(n, abstract="", title="Attacks target "
                             "federated learning") for n in range(1, 4)]
    assessment = assess_evidence(QUESTION, [SUB_Q2], title_only)
    assert assessment["coverage"]["covered"] == []
    assert assessment["coverage"]["uncovered"] == [SUB_Q2]


def test_topic_words_alone_do_not_cover_a_specific_subquestion():
    # Shares "federated learning" with the sub-question (score 0.30) but
    # never mentions attacks, so it must not count as covering it.
    topic_only = [make_paper(n, abstract="federated learning protect privacy")
                  for n in range(1, 4)]
    assert ira.calculate_relevance_score(SUB_Q2, topic_only[0]) >= 0.30
    with_topic = assess_evidence(QUESTION, [SUB_Q2], topic_only,
                                 topic=PLAN["main_topic"])
    assert with_topic["coverage"]["uncovered"] == [SUB_Q2]
    # Without a known topic, only the relevance score is applied.
    no_topic = assess_evidence(QUESTION, [SUB_Q2], topic_only)
    assert no_topic["coverage"]["covered"] == [SUB_Q2]


def test_plural_forms_match_specific_terms():
    papers = [make_paper(n, abstract="federated learning attack detection "
                         "target") for n in range(1, 4)]
    assessment = assess_evidence(QUESTION, [SUB_Q2], papers,
                                 topic=PLAN["main_topic"])
    assert assessment["coverage"]["covered"] == [SUB_Q2]


def test_coverage_threshold_is_configurable():
    only_q1 = [make_paper(n, abstract="federated learning protect privacy")
               for n in range(1, 4)]
    cfg = IterativeRetrievalConfig(min_subquestion_coverage=0.5)
    assessment = assess_evidence(QUESTION, [SUB_Q1, SUB_Q2], only_q1, cfg,
                                 topic=PLAN["main_topic"])
    assert assessment["coverage"]["fraction"] == 0.5
    assert assessment["criteria"]["coverage_ok"] is True


# ---------------------------------------------------------------------------
# 13, 16. Queries
# ---------------------------------------------------------------------------

def test_13_repeated_queries_are_not_reissued():
    cfg = IterativeRetrievalConfig(max_additional_attempts=5,
                                   min_papers=99,
                                   min_papers_with_abstracts=1)
    fn = _endless_new_papers()
    result = run(good_papers(1), fn, config=cfg)
    keys = [ira.build_search_query(q) for q in result["queries_attempted"]]
    assert len(keys) == len(set(keys))
    # None of the initial sub-question searches is repeated either.
    initial_keys = {ira.build_search_query(q) for q in (SUB_Q1, SUB_Q2)}
    assert not initial_keys & set(keys)


def test_exhausted_queries_stop_with_no_new_queries():
    cfg = IterativeRetrievalConfig(max_additional_attempts=5,
                                   min_papers=99,
                                   min_papers_with_abstracts=1)
    fn = _endless_new_papers()
    result = run(good_papers(1), fn, config=cfg)
    assert result["stop_reason"] == "no_new_queries"
    assert result["additional_attempts"] < 5


def test_no_subquestions_and_no_queries_left():
    plan = {"main_topic": "", "research_objective": ""}
    cfg = IterativeRetrievalConfig(max_additional_attempts=5,
                                   min_papers=99,
                                   min_papers_with_abstracts=1)
    fn = _endless_new_papers()
    result = run_iterative_retrieval(
        "federated learning privacy", plan, good_papers(1), config=cfg,
        retrieve_fn=fn)
    assert result["stop_reason"] == "no_subquestions_available"
    assert fn.call_count == 1


def test_16_query_generation_is_deterministic_and_uses_plan_terms():
    attempted = {ira.build_search_query(q) for q in (SUB_Q1, SUB_Q2)}
    first = generate_candidate_queries(QUESTION, PLAN, [SUB_Q2], attempted)
    second = generate_candidate_queries(QUESTION, PLAN, [SUB_Q2], attempted)
    assert first == second and first
    queries = [c["query"] for c in first]
    assert len(queries) == len(set(queries))
    joined = " ".join(queries)
    assert "federated" in joined and "attacks" in joined
    strategies = [c["strategy"] for c in first]
    assert "topic_plus_subquestion" in strategies
    assert "main_topic" in strategies or "shortened_question" in strategies
    # target sub-question is tracked
    assert any(c["target_subquestion"] == SUB_Q2 for c in first)


def test_candidates_skip_already_attempted_queries():
    attempted = {ira.build_search_query(PLAN["main_topic"])}
    candidates = generate_candidate_queries(QUESTION, PLAN, [], attempted)
    assert all(ira.build_search_query(c["query"]) not in attempted
               for c in candidates)


def test_queries_per_attempt_is_respected():
    cfg = IterativeRetrievalConfig(queries_per_attempt=1,
                                   max_additional_attempts=1,
                                   min_papers=99,
                                   min_papers_with_abstracts=1)
    fn = retriever([])
    run(good_papers(1), fn, config=cfg)
    assert len(fn.call_args.args[0]) == 1
    assert fn.call_args.kwargs["papers_per_question"] == 10


# ---------------------------------------------------------------------------
# 15. Result summary
# ---------------------------------------------------------------------------

def test_15_result_summary_and_stop_reason_are_accurate():
    fn = retriever(good_papers(1, start=10))
    result = run(good_papers(2), fn)
    assert result["initial_paper_count"] == 2
    assert result["final_paper_count"] == 3
    assert result["new_papers_found"] == 1
    assert result["additional_attempts"] == 1
    assert result["max_additional_attempts"] == 2
    assert result["stop_reason"] in ira.STOP_REASONS
    assert result["stop_reason"] == "sufficient_evidence"
    assert result["sufficient"] is True
    assert result["initial_assessment"]["sufficient"] is False
    assert result["final_assessment"]["sufficient"] is True
    assert len(result["queries_attempted"]) == len(
        result["attempts"][0]["queries"])
    assert result["attempts"][0]["new_papers"] == 1
    assert result["errors"] == []


def test_claim_support_is_not_assessed_at_this_stage():
    result = run(good_papers(3), MagicMock())
    assert result["final_assessment"]["claim_support"] == "not_assessed"


def test_result_contains_no_environment_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "z" * 30)
    result = run(good_papers(1), MagicMock(
        side_effect=ConnectionError("failed with AIza" + "z" * 30)))
    assert "AIza" + "z" * 30 not in repr(result)


def test_input_paper_list_is_not_mutated():
    papers = good_papers(2)
    snapshot = copy.deepcopy(papers)
    run(papers, retriever(good_papers(1, start=10)))
    assert papers == snapshot


def test_summary_text_is_readable():
    result = run(good_papers(1), retriever([]))
    text = format_retrieval_summary(result)
    assert "Initial papers: 1" in text
    assert "Final papers: 1" in text
    assert "Extra attempts: 1/2" in text
    assert "no_new_papers" in text
    assert "preliminary" in text
    assert format_retrieval_summary(None) == "Retrieval summary unavailable."


def test_summary_shows_errors_as_warnings():
    result = run(good_papers(1), MagicMock(side_effect=TimeoutError("slow")))
    assert "WARNING" in format_retrieval_summary(result)


# ---------------------------------------------------------------------------
# 17. Configuration validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"max_additional_attempts": -1},
    {"max_additional_attempts": 6},
    {"max_additional_attempts": 1.5},
    {"max_additional_attempts": True},
    {"max_additional_attempts": "2"},
    {"max_additional_attempts": None},
    {"min_papers": 0},
    {"min_papers": True},
    {"min_papers_with_abstracts": -1},
    {"min_relevance_score": 1.5},
    {"min_relevance_score": -0.1},
    {"min_relevance_score": float("nan")},
    {"min_subquestion_coverage": 0},
    {"min_subquestion_coverage": 1.1},
    {"queries_per_attempt": 0},
    {"queries_per_attempt": 6},
    {"papers_per_query": 0},
    {"papers_per_query": 26},
])
def test_17_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        IterativeRetrievalConfig(**kwargs)


def test_17_defaults_match_documented_values():
    cfg = IterativeRetrievalConfig()
    assert cfg.max_additional_attempts == 2
    assert cfg.min_papers == 3
    assert cfg.min_papers_with_abstracts == 3
    assert cfg.min_relevance_score == 0.30
    assert cfg.min_subquestion_coverage == 1.0


def test_default_retrieval_function_is_the_existing_retriever():
    """With no retrieve_fn the agent calls retrieve_for_sub_questions."""
    with patch.object(ira, "retrieve_for_sub_questions",
                      return_value=good_papers(1, start=10)) as mocked:
        result = run_iterative_retrieval(QUESTION, PLAN, good_papers(2))
    assert mocked.call_count == 1
    assert result["final_paper_count"] == 3


# ---------------------------------------------------------------------------
# 18. Integration with app.py
# ---------------------------------------------------------------------------

def _run_app(initial_papers, extra_papers, capsys):
    import app

    analysis = {"status": "success", "summary": "s",
                "evidence_linked_claims": [{"claim_id": "CLM-001",
                                            "claim": "c", "paper_ids": []}],
                "metadata": {}}
    verified = {**analysis, "verification_summary": {
        "service_status": "completed", "total_claims": 1, "supported": 0,
        "partially_supported": 0, "unsupported": 0, "unverifiable": 1,
        "flagged_for_review": 1}}

    mocks = {
        "retrieve": MagicMock(return_value=initial_papers),
        "extra": MagicMock(return_value=extra_papers),
        "graph": MagicMock(return_value=[]),
        "analyze": MagicMock(return_value=analysis),
        "verify": MagicMock(return_value=verified),
        "report": MagicMock(return_value={
            "markdown_path": "output/report.md",
            "docx_path": "output/report.docx", "errors": []}),
    }
    with patch("builtins.input", return_value=QUESTION), \
            patch.object(app, "create_research_plan", return_value=PLAN), \
            patch.object(app, "retrieve_for_sub_questions",
                         mocks["retrieve"]), \
            patch.object(ira, "retrieve_for_sub_questions",
                         mocks["extra"]), \
            patch.object(app, "build_graph_context", mocks["graph"]), \
            patch.object(app, "analyze_literature", mocks["analyze"]), \
            patch.object(app, "verify_analysis", mocks["verify"]), \
            patch.object(app, "generate_research_report", mocks["report"]):
        app.main()
    return mocks, capsys.readouterr().out


def test_18_app_passes_final_papers_to_every_downstream_stage(capsys):
    initial = good_papers(1)
    extra = good_papers(2, start=10)
    mocks, out = _run_app(initial, extra, capsys)

    # initial retrieval ran exactly once and was not repeated
    assert mocks["retrieve"].call_count == 1
    assert mocks["extra"].call_count == 1

    final_ids = {p["id"] for p in mocks["analyze"].call_args.kwargs["papers"]}
    expected = {p["id"] for p in initial + extra}
    assert final_ids == expected
    assert {p["id"] for p in mocks["graph"].call_args.args[1]} == expected
    assert {p["id"] for p in mocks["verify"].call_args.args[1]} == expected
    assert {p["id"] for p in
            mocks["report"].call_args.kwargs["papers"]} == expected

    assert "STEP 2B: ITERATIVE RETRIEVAL" in out
    assert "Initial papers: 1" in out and "Final papers: 3" in out
    assert "Stop reason:" in out
    assert "Retrieved 3 unique papers" in out


def test_18_app_skips_extra_search_when_evidence_is_sufficient(capsys):
    mocks, out = _run_app(good_papers(3), [], capsys)
    mocks["extra"].assert_not_called()
    assert mocks["retrieve"].call_count == 1
    assert "sufficient_evidence" in out
    assert mocks["analyze"].called


def test_18_app_exits_gracefully_when_no_papers_anywhere(capsys):
    mocks, out = _run_app([], [], capsys)
    assert "No relevant papers were retrieved" in out
    mocks["analyze"].assert_not_called()
    mocks["report"].assert_not_called()


def test_18_app_continues_with_initial_papers_when_extra_search_fails(capsys):
    import app

    initial = good_papers(2)
    analysis = {"status": "success", "summary": "s",
                "evidence_linked_claims": []}
    with patch("builtins.input", return_value=QUESTION), \
            patch.object(app, "create_research_plan", return_value=PLAN), \
            patch.object(app, "retrieve_for_sub_questions",
                         return_value=initial), \
            patch.object(ira, "retrieve_for_sub_questions",
                         side_effect=requests.exceptions.ConnectionError(
                             "offline")), \
            patch.object(app, "build_graph_context", return_value=[]), \
            patch.object(app, "analyze_literature",
                         return_value=analysis) as analyze, \
            patch.object(app, "generate_research_report",
                         return_value={"markdown_path": "m", "docx_path": "d",
                                       "errors": []}):
        app.main()
    out = capsys.readouterr().out
    assert {p["id"] for p in analyze.call_args.kwargs["papers"]} == {
        p["id"] for p in initial}
    assert "retrieval_error" in out and "WARNING" in out