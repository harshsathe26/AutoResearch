"""
Tests for the Phase 7 Evaluation Framework.

Everything is offline: no Gemini key, no network, no live retrieval. Test
numbers in the names follow the Phase 7 specification (1-48). Written with
unittest.TestCase so it runs under pytest (python -m pytest) or unittest.
"""

import copy
import importlib
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.report_agent import generate_research_report
from evaluation import (
    evaluate_research_run,
    evaluate_retrieval_labels,
    format_evaluation_summary,
    load_retrieval_labels,
    save_evaluation,
)
from evaluation import evaluator as evaluator_module
from evaluation import metrics

FIXTURE = (Path(__file__).parent / "fixtures"
           / "retrieval_labels_example.json")

QUESTION = "How do federated learning systems protect privacy against attacks"
SUB_Q1 = "How does federated learning protect privacy"
SUB_Q2 = "What attacks target federated learning"
PLAN = {
    "main_topic": "Federated learning privacy",
    "research_objective": "Understand privacy protection and attacks",
    "research_plan": ["Define", "Survey"],
    "sub_questions": [SUB_Q1, SUB_Q2],
}
GOOD_ABSTRACT = ("federated learning protect privacy attacks target "
                 "federated learning systems")


def make_paper(n, abstract=GOOD_ABSTRACT, **extra):
    paper = {
        "id": f"https://openalex.org/W{n}",
        "title": f"Federated Privacy Study Number {n}",
        "abstract": abstract,
        "authors": [f"Author {n}"],
        "year": 2020 + n % 5,
        "doi": f"https://doi.org/10.1234/paper{n}",
        "url": f"https://doi.org/10.1234/paper{n}",
    }
    paper.update(extra)
    return paper


def make_claim(n=1, ids=None, invalid=None, excerpt=None,
               support_type="source_supported"):
    return {
        "claim_id": f"CLM-{n:03d}",
        "claim": f"Claim number {n}.",
        "support_type": support_type,
        "paper_ids": ["https://openalex.org/W1"] if ids is None else ids,
        "invalid_paper_ids": invalid or [],
        "evidence_excerpt": ("federated learning protect privacy"
                             if excerpt is None else excerpt),
    }


def make_analysis(claims=None, status="success"):
    claims = [make_claim(1), make_claim(2)] if claims is None else claims
    return {
        "status": status, "summary": "s",
        "key_themes_and_consensus": [], "limitations_and_research_gaps": [],
        "divergent_findings_and_contradictions": [],
        "methodologies_and_benchmarks": [],
        "evidence_linked_claims": claims,
    }


def verified(analysis, statuses, service="completed"):
    result = copy.deepcopy(analysis)
    for claim, status in zip(result["evidence_linked_claims"], statuses):
        claim["verification"] = {
            "status": status, "needs_review": status != "SUPPORTED",
            "missing_paper_ids": [], "warnings": []}
    result["verification_summary"] = {
        "service_status": service, "service_note": "",
        "total_claims": len(result["evidence_linked_claims"])}
    return result


def good_papers(count=3):
    return [make_paper(n) for n in range(1, count + 1)]


def retrieval_result(papers, errors=None):
    return {"papers": papers, "sufficient": True, "stop_reason":
            "sufficient_evidence", "additional_attempts": 0,
            "new_papers_found": 0, "initial_paper_count": len(papers),
            "final_paper_count": len(papers), "errors": errors or []}


def build_report(tmp, papers, analysis, **kwargs):
    info = generate_research_report(
        user_query=QUESTION, plan=PLAN, papers=papers, graph_context=[],
        analysis=analysis, output_dir=tmp, **kwargs)
    return info, Path(info["markdown_path"]).read_text(encoding="utf-8")


def full_run(tmp, statuses=("SUPPORTED", "SUPPORTED")):
    papers = good_papers(3)
    analysis = verified(make_analysis(), list(statuses))
    info, text = build_report(tmp, papers, analysis)
    return dict(user_query=QUESTION, plan=PLAN, papers=papers,
                retrieval_result=retrieval_result(papers),
                graph_context=[{"title": "x"}], analysis=analysis,
                report=info)


def load_app():
    """Import app.py without needing a real .env: importing the planner
    requires GEMINI_API_KEY to exist (a dummy value; no request is made)."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": os.environ.get(
            "GEMINI_API_KEY") or "dummy-key-for-import-only"}):
        return importlib.import_module("app")


def metric(result, name):
    return result["metrics"][name]


# ---------------------------------------------------------------------------
# 1-10 Basic behaviour
# ---------------------------------------------------------------------------

class TestBasic(unittest.TestCase):

    def test_01_complete_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_research_run(**full_run(tmp))
        self.assertEqual(result["overall_status"], "completed")
        self.assertEqual(result["scientific_validity"], "not_assessed")
        self.assertEqual(metric(result, "retrieval_record_count")["value"], 3)
        self.assertEqual(
            metric(result, "citation_traceability")["value"], 1.0)
        self.assertEqual(
            metric(result, "subquestion_coverage")["value"], 1.0)
        self.assertEqual(
            metric(result, "report_completeness")["value"], 1.0)
        self.assertTrue(result["limitations"])

    def test_02_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_research_run(**full_run(tmp))
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_03_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = full_run(tmp)
            a = json.dumps(evaluate_research_run(**kwargs), sort_keys=True)
            b = json.dumps(evaluate_research_run(**kwargs), sort_keys=True)
        self.assertEqual(a, b)

    def test_04_missing_optional_fields(self):
        result = evaluate_research_run(user_query=QUESTION, plan=PLAN,
                                       papers=good_papers(3))
        self.assertEqual(metric(result, "retrieval_record_count")["value"], 3)
        self.assertEqual(
            metric(result, "claim_verification")["status"], "not_evaluated")
        self.assertEqual(result["overall_status"], "incomplete")

    def test_05_empty_paper_list(self):
        result = evaluate_research_run(plan=PLAN, papers=[])
        self.assertEqual(metric(result, "retrieval_record_count")["value"], 0)
        self.assertEqual(
            metric(result, "retrieval_abstract_availability")["status"],
            "not_evaluated")
        self.assertEqual(
            metric(result, "retrieval_duplicates")["status"],
            "not_evaluated")
        self.assertEqual(
            metric(result, "subquestion_coverage")["value"], 0.0)

    def test_06_empty_claims(self):
        result = evaluate_research_run(
            papers=good_papers(3), analysis=make_analysis(claims=[]))
        ver = metric(result, "claim_verification")
        self.assertEqual(ver["status"], "not_evaluated")
        self.assertEqual(ver["details"]["verification_state"], "no_claims")
        self.assertEqual(
            metric(result, "citation_traceability")["status"],
            "not_evaluated")

    def test_07_empty_subquestions(self):
        plan = dict(PLAN, sub_questions=[])
        result = evaluate_research_run(plan=plan, papers=good_papers(3))
        cov = metric(result, "subquestion_coverage")
        self.assertEqual(cov["status"], "not_evaluated")
        self.assertIsNone(cov["value"])
        self.assertIn("not applicable", cov["reason"])

    def test_08_missing_analysis(self):
        result = evaluate_research_run(papers=good_papers(3))
        self.assertEqual(
            metric(result, "citation_traceability")["status"],
            "not_evaluated")
        self.assertEqual(metric(result, "claim_verification")["details"]
                         ["verification_state"], "no_analysis")

    def test_09_missing_verification(self):
        result = evaluate_research_run(
            papers=good_papers(3), analysis=make_analysis())
        ver = metric(result, "claim_verification")
        self.assertEqual(ver["status"], "not_evaluated")
        self.assertEqual(ver["details"]["verification_state"], "unavailable")

    def test_10_missing_report(self):
        result = evaluate_research_run(papers=good_papers(3))
        rep = metric(result, "report_completeness")
        self.assertEqual(rep["status"], "not_evaluated")
        stages = metric(result, "pipeline_reliability")["details"]["stages"]
        self.assertEqual(stages["report"]["status"], "unknown")


# ---------------------------------------------------------------------------
# 11-18 Retrieval metrics
# ---------------------------------------------------------------------------

class TestRetrieval(unittest.TestCase):

    def test_11_usable_abstract_count(self):
        papers = [make_paper(1), make_paper(2, abstract=""),
                  make_paper(3, abstract="Abstract not available"),
                  make_paper(4)]
        m = metrics.abstract_availability(papers)
        self.assertEqual(m["value"], 0.5)
        self.assertEqual(m["details"]["with_abstract"], 2)
        self.assertEqual(m["details"]["without_abstract"], 2)

    def test_12_duplicate_detection(self):
        papers = [make_paper(1), make_paper(1), make_paper(2)]
        m = metrics.duplicate_papers(papers)
        self.assertEqual(m["details"]["duplicate_count"], 1)
        self.assertAlmostEqual(m["value"], 0.3333, places=4)

    def test_12b_similar_titles_are_not_merged(self):
        a = make_paper(1, title="Privacy in Federated Learning")
        b = make_paper(2, title="Privacy in Federated Learning Systems")
        self.assertEqual(
            metrics.duplicate_papers([a, b])["details"]["duplicate_count"], 0)

    def test_12c_different_dois_same_title_are_distinct(self):
        a = make_paper(1, title="Same Title")
        b = make_paper(2, title="Same Title")
        self.assertEqual(
            metrics.duplicate_papers([a, b])["details"]["duplicate_count"], 0)

    def test_13_precision_at_k(self):
        r = metrics.precision_recall_at_k(
            ["a", "b", "c", "d"], ["a", "c"], k=4)
        self.assertEqual(r["precision"]["value"], 0.5)

    def test_14_recall_at_k(self):
        r = metrics.precision_recall_at_k(
            ["a", "b", "c", "d"], ["a", "c", "z"], k=4)
        self.assertAlmostEqual(r["recall"]["value"], 0.6667, places=4)
        r2 = metrics.precision_recall_at_k(
            ["a", "b", "c", "d"], ["a", "c", "z"], k=1)
        self.assertAlmostEqual(r2["recall"]["value"], 0.3333, places=4)

    def test_13b_fixture_file_values(self):
        data = load_retrieval_labels(FIXTURE)
        self.assertIn("FIXTURE", data["note"])
        out = evaluate_retrieval_labels(data, k=3)
        self.assertAlmostEqual(out["precision_at_k"]["value"], 0.6667, 4)
        self.assertAlmostEqual(out["recall_at_k"]["value"], 0.6667, 4)
        self.assertIn("not universal", out["label_source"])
        out5 = evaluate_retrieval_labels(data, k=5)
        self.assertEqual(out5["precision_at_k"]["value"], 0.4)

    def test_15_missing_ground_truth(self):
        result = evaluate_research_run(papers=good_papers(3))
        for name in ("retrieval_precision_at_k", "retrieval_recall_at_k"):
            m = metric(result, name)
            self.assertEqual(m["status"], "not_evaluated")
            self.assertIsNone(m["value"])
            self.assertIn("ground-truth", m["reason"])

    def test_16_empty_retrieval_list_with_labels(self):
        r = metrics.precision_recall_at_k([], ["a"], k=5)
        self.assertEqual(r["precision"]["status"], "not_evaluated")
        self.assertEqual(r["recall"]["status"], "calculated")
        self.assertEqual(r["recall"]["value"], 0.0)   # genuine zero

    def test_17_zero_denominators(self):
        self.assertIsNone(metrics.safe_ratio(0, 0))
        r = metrics.precision_recall_at_k(["a"], [], k=5)
        self.assertEqual(r["recall"]["status"], "not_evaluated")
        self.assertEqual(
            metrics.abstract_availability([])["status"], "not_evaluated")

    def test_18_no_invented_relevance_labels(self):
        # relevance_score on a paper is NOT a label.
        papers = [make_paper(n, relevance_score=0.99) for n in (1, 2, 3)]
        result = evaluate_research_run(papers=papers)
        self.assertEqual(metric(result, "retrieval_precision_at_k")["status"],
                         "not_evaluated")

    def test_18b_precision_with_real_papers_and_labels(self):
        papers = good_papers(4)
        result = evaluate_research_run(
            papers=papers, k=2, ground_truth={
                "relevant_ids": ["10.1234/paper1", "10.1234/paper4"]})
        self.assertEqual(
            metric(result, "retrieval_precision_at_k")["value"], 0.5)
        self.assertEqual(
            metric(result, "retrieval_recall_at_k")["value"], 0.5)

    def test_18c_invalid_k_rejected(self):
        for bad in (0, -1, True, "5"):
            with self.assertRaises(ValueError):
                evaluate_research_run(k=bad)

    def test_18d_iterative_retrieval_reported(self):
        papers = good_papers(3)
        result = evaluate_research_run(
            papers=papers, retrieval_result=retrieval_result(
                papers, errors=["Attempt 1: boom"]))
        it = metric(result, "iterative_retrieval")
        self.assertEqual(it["details"]["error_count"], 1)
        stages = metric(result, "pipeline_reliability")["details"]["stages"]
        self.assertEqual(stages["iterative_retrieval"]["status"], "partial")


# ---------------------------------------------------------------------------
# 19-23 Coverage
# ---------------------------------------------------------------------------

class TestCoverage(unittest.TestCase):

    def test_19_all_covered(self):
        m = metrics.subquestion_coverage(QUESTION, PLAN, good_papers(3))
        self.assertEqual(m["value"], 1.0)
        self.assertEqual(m["details"]["uncovered"], [])
        for row in m["details"]["per_sub_question"]:
            self.assertTrue(row["covered"])
            self.assertTrue(row["matched_papers"])

    def test_20_some_uncovered(self):
        papers = [make_paper(n, abstract="federated learning protect privacy")
                  for n in (1, 2, 3)]
        m = metrics.subquestion_coverage(QUESTION, PLAN, papers)
        self.assertEqual(m["value"], 0.5)
        self.assertEqual(m["details"]["uncovered"], [SUB_Q2])
        rows = {r["sub_question"]: r for r in m["details"]["per_sub_question"]}
        self.assertFalse(rows[SUB_Q2]["covered"])
        self.assertEqual(rows[SUB_Q2]["matched_papers"], [])

    def test_21_no_subquestions(self):
        m = metrics.subquestion_coverage(QUESTION, {"sub_questions": []}, [])
        self.assertEqual(m["status"], "not_evaluated")

    def test_22_heuristic_limitations_disclosed(self):
        m = metrics.subquestion_coverage(QUESTION, PLAN, good_papers(3))
        self.assertEqual(m["status"], "heuristic")
        self.assertIn("ESTIMATE", m["details"]["limitations"])
        self.assertIn("not semantic", m["details"]["limitations"])
        self.assertFalse(m["details"]["semantic_verification"])

    def test_23_topic_word_overlap_is_not_semantic_proof(self):
        # Title shares topic words but there is no abstract: not covered.
        papers = [make_paper(n, abstract="") for n in (1, 2, 3)]
        m = metrics.subquestion_coverage(QUESTION, PLAN, papers)
        self.assertEqual(m["value"], 0.0)
        self.assertNotIn("semantic_verification", m["status"])
        self.assertFalse(m["details"]["semantic_verification"])

    def test_23b_matches_phase6_assessment(self):
        from agents.iterative_retrieval_agent import assess_evidence
        papers = [make_paper(1, abstract="federated learning protect "
                                         "privacy")]
        phase6 = assess_evidence(QUESTION, [SUB_Q1, SUB_Q2], papers,
                                 topic=PLAN["main_topic"])
        m = metrics.subquestion_coverage(QUESTION, PLAN, papers)
        self.assertEqual(m["value"], phase6["coverage"]["fraction"])


# ---------------------------------------------------------------------------
# 24-29 Citations and evidence
# ---------------------------------------------------------------------------

class TestCitations(unittest.TestCase):

    def trace(self, claims, papers=None):
        return metrics.citation_traceability(
            make_analysis(claims=claims), papers or good_papers(3))

    def test_24_all_references_resolve(self):
        m = self.trace([make_claim(1), make_claim(2, ids=[
            "https://openalex.org/W2"])])
        self.assertEqual(m["value"], 1.0)
        self.assertEqual(m["status"], "calculated")
        self.assertEqual(m["details"]["claims_with_valid_reference"], 2)

    def test_25_unknown_ids_detected(self):
        m = self.trace([make_claim(1, ids=["https://openalex.org/W1"],
                                   invalid=["https://openalex.org/W999"])])
        self.assertEqual(m["details"]["claims_with_unknown_ids"], 1)
        self.assertEqual(m["value"], 0.5)
        self.assertEqual(m["status"], "needs_review")

    def test_25b_id_not_in_current_corpus_is_unknown(self):
        m = self.trace([make_claim(1, ids=["https://openalex.org/W77"])])
        self.assertEqual(m["details"]["claims_with_unknown_ids"], 1)
        self.assertEqual(m["details"]["claims_with_valid_reference"], 0)

    def test_26_missing_citations_detected(self):
        m = self.trace([make_claim(1, ids=[]), make_claim(2)])
        self.assertEqual(m["details"]["claims_missing_references"], 1)
        self.assertEqual(m["status"], "needs_review")

    def test_26b_no_citations_at_all_has_undefined_rate(self):
        m = self.trace([make_claim(1, ids=[])])
        self.assertIsNone(m["value"])
        self.assertIn("undefined", m["reason"])

    def test_27_missing_excerpts_handled(self):
        m = self.trace([make_claim(1, excerpt=" "),
                        make_claim(2, support_type="inference", excerpt=" ")])
        self.assertEqual(m["details"]["claims_missing_excerpt"], 1)
        self.assertEqual(m["details"]["inference_claims"], 1)

    def test_28_excerpt_match_is_not_semantic_verification(self):
        m = self.trace([make_claim(1)])
        self.assertEqual(m["details"]["excerpt_found_in_cited_abstract"], 1)
        self.assertFalse(m["details"]["semantic_support_verified"])
        self.assertIn("string match", m["details"]["note"])
        self.assertIn("not that it supports", m["details"]["note"])

    def test_29_duplicate_records_handled_consistently(self):
        papers = [make_paper(1), make_paper(1, id="https://openalex.org/W1b"),
                  make_paper(2)]
        claims = [make_claim(1, ids=["https://openalex.org/W1",
                                     "https://openalex.org/W1b"])]
        m = self.trace(claims, papers)
        self.assertEqual(m["value"], 1.0)
        self.assertEqual(m["details"]["references_resolved"], 2)
        self.assertEqual(m["details"]["distinct_papers_cited"], 1)


# ---------------------------------------------------------------------------
# 30-35 Verification
# ---------------------------------------------------------------------------

class TestVerification(unittest.TestCase):

    def test_30_mixed_statuses(self):
        analysis = verified(make_analysis(
            claims=[make_claim(n) for n in range(1, 5)]),
            ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
             "UNVERIFIABLE"])
        m = metrics.claim_verification(analysis)
        counts = m["details"]["counts"]
        self.assertEqual(counts, {"supported": 1, "partially_supported": 1,
                                  "unsupported": 1, "unverifiable": 1})
        self.assertEqual(m["details"]["ratios_of_total_claims"]["supported"],
                         0.25)
        self.assertEqual(m["status"], "needs_review")
        self.assertIn("not proof", m["details"]["note"])

    def test_31_empty_claims_not_a_success(self):
        m = metrics.claim_verification(make_analysis(claims=[]))
        self.assertEqual(m["status"], "not_evaluated")
        self.assertEqual(m["details"]["verification_state"], "no_claims")

    def test_32_skipped_verification(self):
        m = metrics.claim_verification(
            make_analysis(), verification_skipped=True)
        self.assertEqual(m["details"]["verification_state"], "skipped")
        self.assertEqual(m["status"], "not_evaluated")

    def test_33_service_failure(self):
        m = metrics.claim_verification(
            make_analysis(), verification_error="RuntimeError: down")
        self.assertEqual(m["details"]["verification_state"], "failed")
        analysis = verified(make_analysis(), ["UNVERIFIABLE", "UNVERIFIABLE"],
                            service="unavailable")
        m2 = metrics.claim_verification(analysis)
        self.assertEqual(m2["details"]["verification_state"], "unavailable")
        self.assertEqual(m2["status"], "needs_review")
        result = evaluate_research_run(
            papers=good_papers(3), analysis=analysis)
        self.assertTrue(any("unavailable" in w for w in result["warnings"]))

    def test_34_unknown_status_values(self):
        analysis = verified(make_analysis(), ["SUPPORTED", "MAYBE"])
        m = metrics.claim_verification(analysis)
        self.assertEqual(m["details"]["unrecognized_statuses"], {"MAYBE": 1})
        self.assertEqual(m["status"], "needs_review")

    def test_35_review_flags_counted(self):
        analysis = verified(make_analysis(), ["SUPPORTED", "SUPPORTED"])
        analysis["evidence_linked_claims"][0]["verification"][
            "needs_review"] = True
        analysis["evidence_linked_claims"][1]["verification"][
            "missing_paper_ids"] = ["x"]
        m = metrics.claim_verification(analysis)
        self.assertEqual(m["details"]["flagged_for_review"], 1)
        self.assertEqual(
            m["details"]["claims_with_missing_or_invalid_citations"], 1)

    def test_35b_claims_without_result_are_flagged(self):
        analysis = verified(make_analysis(), ["SUPPORTED", "SUPPORTED"])
        del analysis["evidence_linked_claims"][1]["verification"]
        m = metrics.claim_verification(analysis)
        self.assertEqual(m["details"]["claims_without_result"], 1)
        self.assertEqual(m["details"]["flagged_for_review"], 1)


# ---------------------------------------------------------------------------
# 36-40 Report completeness
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.papers = good_papers(3)
        self.analysis = verified(make_analysis(), ["SUPPORTED", "SUPPORTED"])
        self.info, self.text = build_report(
            self.tmp.name, self.papers, self.analysis)

    def check(self, text=None, **kwargs):
        args = dict(papers=self.papers, analysis=self.analysis,
                    report_info=self.info)
        args.update(kwargs)
        return metrics.report_completeness(
            self.text if text is None else text, **args)

    def test_36_required_sections_present(self):
        m = self.check()
        self.assertEqual(m["details"]["missing_sections"], [])
        self.assertEqual(m["details"]["problems"], [])
        self.assertEqual(m["value"], 1.0)
        self.assertIn("not a scientifically", m["details"]["note"])

    def test_37_required_sections_missing(self):
        text = re.sub(r"^## 9\. Conclusion.*$", "", self.text,
                      flags=re.M)
        m = self.check(text)
        self.assertEqual(m["details"]["missing_sections"], ["conclusion"])
        self.assertEqual(m["status"], "needs_review")
        self.assertEqual(m["value"], 0.9)

    def test_38_missing_references(self):
        head = self.text.split("## 10.")[0]
        m = self.check(head + "## 10. References and Source Index\n")
        self.assertIn("references expected but none listed",
                      m["details"]["problems"])

    def test_38b_unresolved_citation_numbers(self):
        m = self.check(self.text.replace(
            "## 10.", "See [99].\n\n## 10.", 1))
        self.assertIn(99, m["details"]["references"][
            "unresolved_citation_numbers"])

    def test_39_verification_limitations_preserved(self):
        m = self.check()
        self.assertTrue(m["details"]["disclosure"][
            "limitations_section_present"])
        # Verification failed: the generated report must say so.
        with tempfile.TemporaryDirectory() as tmp:
            info, text = build_report(
                tmp, self.papers, make_analysis(),
                verification_error="RuntimeError: down")
            m2 = metrics.report_completeness(
                text, self.papers, make_analysis(), "RuntimeError: down",
                info)
        self.assertTrue(m2["details"]["disclosure"]["degradation_disclosed"])
        # A report that hides the failure is flagged.
        stripped = re.sub(r"(?i)not verified|failed|unavailable|partial|"
                          r"unverifiable", "ok", self.text)
        m3 = self.check(stripped, verification_error="boom")
        self.assertIn("verification problems not disclosed",
                      m3["details"]["problems"])

    def test_39b_insufficient_evidence_disclosed(self):
        analysis = make_analysis(claims=[], status="insufficient_evidence")
        with tempfile.TemporaryDirectory() as tmp:
            info, text = build_report(tmp, self.papers, analysis)
            m = metrics.report_completeness(text, self.papers, analysis,
                                            None, info)
        self.assertTrue(m["details"]["disclosure"]["insufficient_evidence"])
        self.assertNotIn("insufficient evidence not disclosed",
                         m["details"]["problems"])

    def test_40_generation_failure_represented(self):
        m = metrics.report_completeness(None, report_error="ValueError: x")
        self.assertEqual(m["status"], "not_evaluated")
        self.assertTrue(
            m["details"]["outputs"]["report_generation_failed"])
        result = evaluate_research_run(
            papers=self.papers, report_error="ValueError: bad")
        stages = metric(result, "pipeline_reliability")["details"]["stages"]
        self.assertEqual(stages["report"]["status"], "failed")
        self.assertFalse(metric(result, "pipeline_reliability")
                         ["details"]["usable_report_produced"])

    def test_40b_missing_docx_reported(self):
        info = dict(self.info, docx_path=None, errors=["Word not saved"])
        m = self.check(report_info=info)
        self.assertIn("Word file not written", m["details"]["problems"])

    def test_40c_report_read_from_markdown_path(self):
        result = evaluate_research_run(
            papers=self.papers, analysis=self.analysis, report=self.info)
        self.assertEqual(metric(result, "report_completeness")["value"], 1.0)
        missing = dict(self.info, markdown_path=str(
            Path(self.tmp.name) / "nope.md"))
        result = evaluate_research_run(papers=self.papers, report=missing)
        rep = metric(result, "report_completeness")
        self.assertEqual(rep["status"], "not_evaluated")
        self.assertIn("could not be read", rep["reason"])


# ---------------------------------------------------------------------------
# 41-48 Reliability, safety, integration
# ---------------------------------------------------------------------------

class TestReliabilityAndSafety(unittest.TestCase):

    def test_41_partial_pipeline_outputs(self):
        result = evaluate_research_run(
            user_query=QUESTION, plan=PLAN, papers=good_papers(3),
            analysis=make_analysis())
        self.assertEqual(result["overall_status"], "incomplete")
        stages = metric(result, "pipeline_reliability")["details"]["stages"]
        self.assertEqual(stages["retrieval"]["status"], "completed")
        self.assertEqual(stages["verification"]["status"], "unknown")

    def test_42_missing_stages_are_unknown_not_success(self):
        result = evaluate_research_run()
        counts = metric(result, "pipeline_reliability")["value"]
        self.assertEqual(counts["unknown"], 7)
        self.assertEqual(counts["completed"], 0)

    def test_42b_stage_overrides(self):
        result = evaluate_research_run(
            stage_status={"graph_rag": "skipped", "bogus": "completed",
                          "planner": "not-a-status"})
        stages = metric(result, "pipeline_reliability")["details"]["stages"]
        self.assertEqual(stages["graph_rag"]["status"], "skipped")
        self.assertNotIn("bogus", stages)
        self.assertEqual(stages["planner"]["status"], "unknown")

    def test_43_errors_are_sanitized(self):
        secret = "AIzaSy" + "A" * 33
        text = evaluator_module.sanitize_error(
            RuntimeError(f"request failed key={secret} " + "x" * 500))
        self.assertNotIn(secret, text)
        self.assertLessEqual(len(text), 200)
        self.assertTrue(text.startswith("RuntimeError"))

    def test_44_no_secret_leakage_in_result(self):
        secret = "AIzaSy" + "B" * 33
        result = evaluate_research_run(
            papers=good_papers(3), analysis=make_analysis(),
            verification_error=f"RuntimeError: bad key {secret}",
            report_error=f"OSError: token {secret}")
        self.assertNotIn(secret, json.dumps(result))

    def test_45_inputs_are_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = full_run(tmp)
            snapshot = copy.deepcopy(kwargs)
            evaluate_research_run(
                **kwargs, ground_truth={"relevant_ids": ["10.1234/paper1"]})
        self.assertEqual(kwargs, snapshot)

    def test_46_no_llm_or_network_calls(self):
        boom = AssertionError("evaluation must not use LLM or network")
        with patch("agents.analysis_agent.get_gemini_client",
                   side_effect=boom) as client, \
                patch("agents.verification_agent.get_gemini_client",
                      side_effect=boom), \
                patch("requests.get", side_effect=boom) as get, \
                patch("requests.post", side_effect=boom) as post, \
                patch("socket.socket.connect", side_effect=boom), \
                tempfile.TemporaryDirectory() as tmp:
            result = evaluate_research_run(**full_run(tmp))
        client.assert_not_called()
        get.assert_not_called()
        post.assert_not_called()
        self.assertEqual(result["overall_status"], "completed")

    def test_47_application_is_compatible(self):
        """app.main() still runs end to end (all agents mocked) and now
        also prints and saves the evaluation."""
        app = load_app()
        papers = good_papers(3)
        analysis = make_analysis()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("builtins.input", return_value=QUESTION), \
                patch.object(app, "create_research_plan",
                             return_value=copy.deepcopy(PLAN)), \
                patch.object(app, "retrieve_for_sub_questions",
                             return_value=papers), \
                patch.object(app, "run_iterative_retrieval",
                             return_value=retrieval_result(papers)), \
                patch.object(app, "build_graph_context", return_value=[]), \
                patch.object(app, "analyze_literature",
                             return_value=analysis), \
                patch.object(app, "verify_analysis", return_value=verified(
                    analysis, ["SUPPORTED", "SUPPORTED"])), \
                patch.object(app, "generate_research_report",
                             side_effect=lambda **kw: generate_research_report(
                                 output_dir=tmp, **kw)), \
                patch.object(app, "save_evaluation",
                             side_effect=lambda r: save_evaluation(r, tmp)):
            out = io.StringIO()
            with redirect_stdout(out):
                app.main()
            text = out.getvalue()
            self.assertIn("RESEARCH PIPELINE COMPLETED", text)
            self.assertIn("STEP 7: EVALUATION", text)
            self.assertIn("NOT scientific validity", text)
            saved = json.loads(
                (Path(tmp) / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["evaluation_version"], "1.0")
            self.assertTrue((Path(tmp) / "report.md").exists())

    def test_47b_app_survives_unsavable_evaluation(self):
        app = load_app()
        with patch.object(app, "save_evaluation",
                          side_effect=PermissionError("denied")):
            out = io.StringIO()
            with redirect_stdout(out):
                result = app.run_evaluation_step(
                    user_query=QUESTION, plan=PLAN)
        self.assertIn("not saved", out.getvalue())
        self.assertIn("overall_status", result)

    def test_48_programming_errors_are_not_reported_as_success(self):
        with patch.object(metrics, "duplicate_papers",
                          side_effect=KeyError("bug")):
            result = evaluate_research_run(
                plan=PLAN, papers=good_papers(3))
            dup = metric(result, "retrieval_duplicates")
            self.assertEqual(dup["status"], "error")
            self.assertEqual(result["overall_status"],
                             "completed_with_errors")
            self.assertTrue(result["requires_human_review"])
            self.assertTrue(any("failed" in w for w in result["warnings"]))
            with self.assertRaises(KeyError):      # strict re-raises
                evaluate_research_run(papers=good_papers(3), strict=True)

    def test_48b_summary_text_and_saved_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_research_run(**full_run(tmp))
            path = save_evaluation(result, Path(tmp) / "sub")
            self.assertEqual(path.name, "evaluation.json")
            self.assertEqual(json.loads(path.read_text("utf-8")), result)
            self.assertFalse(list(Path(tmp, "sub").glob("*.tmp")))
        summary = format_evaluation_summary(result)
        self.assertIn("NOT scientific validity", summary)
        self.assertNotIn("abstract", result["metrics"][
            "citation_traceability"].get("details", {}))

    def test_48c_output_stays_small_and_has_no_abstracts(self):
        papers = [make_paper(n, abstract="secret abstract text " * 50)
                  for n in range(1, 40)]
        result = evaluate_research_run(plan=PLAN, papers=papers)
        dumped = json.dumps(result)
        self.assertNotIn("secret abstract text", dumped)
        self.assertLess(len(dumped), 20000)


if __name__ == "__main__":
    unittest.main()
