import unittest
from unittest.mock import MagicMock, patch

from agents.analysis_agent import (
    analyze_literature,
    build_corpus_index,
    build_insufficient_evidence_response,
    format_analysis_report,
    validate_and_enrich_analysis,
    verify_excerpt_against_abstract,
)


PAPER_A = "https://openalex.org/W1"
PAPER_B = "https://openalex.org/W2"


def valid_analysis():
    return {
        "summary": "Two papers examine retrieval techniques.",
        "key_themes_and_consensus": [
            {
                "theme": "Retrieval",
                "description": "Retrieval is used to ground generation.",
                "supporting_paper_ids": [PAPER_A],
            }
        ],
        "divergent_findings_and_contradictions": [],
        "methodologies_and_benchmarks": [
            {
                "approach_or_dataset": "Adaptive retrieval",
                "description": "Retrieval is adjusted to the query.",
                "papers_using": [PAPER_A],
            }
        ],
        "limitations_and_research_gaps": [
            {
                "gap_or_limitation": "Further evaluation is needed.",
                "reported_by": [PAPER_B],
                "type": "reported_limitation",
            }
        ],
        "evidence_linked_claims": [
            {
                "claim_id": "CLM-001",
                "claim": "Self-RAG improves factual accuracy.",
                "support_type": "source_supported",
                "paper_ids": [PAPER_A],
                "evidence_excerpt": (
                    "improves language model quality and factual accuracy"
                ),
            }
        ],
    }


class TestAnalysisAgent(unittest.TestCase):

    def setUp(self):
        self.papers = [
            {
                "id": PAPER_A,
                "title": "Self-RAG",
                "abstract": (
                    "We introduce Self-RAG, a framework that improves "
                    "language model quality and factual accuracy through "
                    "adaptive retrieval and self-reflection tokens."
                ),
                "authors": ["Author A"],
                "year": 2024,
                "url": PAPER_A,
            },
            {
                "id": PAPER_B,
                "title": "Corrective Retrieval",
                "abstract": (
                    "We propose corrective retrieval to improve robustness "
                    "by evaluating retrieval quality and triggering web search."
                ),
                "authors": ["Author B"],
                "year": 2024,
                "url": PAPER_B,
            },
        ]

    # 1. Exact excerpt verification
    def test_exact_excerpt_is_verified(self):
        abstract = self.papers[0]["abstract"]
        quote = "improves language model quality and factual accuracy"
        self.assertTrue(
            verify_excerpt_against_abstract(quote, abstract)
        )

    # 2. Punctuation and capitalization normalization
    def test_punctuation_normalized_excerpt_is_verified(self):
        abstract = "This method improves accuracy, reliability!"
        quote = "IMPROVES accuracy reliability"
        self.assertTrue(
            verify_excerpt_against_abstract(quote, abstract)
        )

    # 3. Fabricated quote with high word overlap must fail
    def test_high_word_overlap_fabricated_quote_fails(self):
        abstract = (
            "The method improves language model quality and factual "
            "accuracy through adaptive retrieval."
        )
        fabricated = (
            "improves language model quality and factual "
            "accuracy through adaptive generation"
        )
        self.assertFalse(
            verify_excerpt_against_abstract(fabricated, abstract)
        )

    # 4. Empty quotes fail verification
    def test_empty_excerpt_fails(self):
        self.assertFalse(
            verify_excerpt_against_abstract(
                "", self.papers[0]["abstract"]
            )
        )

    # 5. Corpus retains abstract and metadata
    def test_corpus_index(self):
        corpus = build_corpus_index(self.papers)
        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[PAPER_A]["year"], 2024)
        self.assertIn("adaptive retrieval", corpus[PAPER_A]["abstract"])

    # 6. Empty papers return insufficient evidence
    def test_empty_retrieval(self):
        result = analyze_literature("What is RAG?", [])
        self.assertEqual(result["status"], "insufficient_evidence")

    # 7. Missing abstracts return insufficient evidence
    def test_missing_abstracts(self):
        papers = [
            {"id": "W3", "title": "Paper", "abstract": ""},
            {
                "id": "W4",
                "title": "Paper 2",
                "abstract": "Abstract not available",
            },
        ]
        result = analyze_literature("Research question", papers)
        self.assertEqual(result["status"], "insufficient_evidence")

    # 8. All five sections are retained
    @patch("agents.analysis_agent.get_gemini_client")
    def test_all_required_sections(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.models.generate_content.return_value.text = __import__(
            "json"
        ).dumps(valid_analysis())

        result = analyze_literature("Research question", self.papers)

        self.assertEqual(result["status"], "success")
        for section in (
            "key_themes_and_consensus",
            "divergent_findings_and_contradictions",
            "methodologies_and_benchmarks",
            "limitations_and_research_gaps",
            "evidence_linked_claims",
        ):
            self.assertIn(section, result)

    # 9. Invalid theme IDs remain visible
    def test_invalid_theme_reference_is_flagged(self):
        data = valid_analysis()
        data["key_themes_and_consensus"][0][
            "supporting_paper_ids"
        ] = ["FAKE-ID"]

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        theme = result["key_themes_and_consensus"][0]

        self.assertEqual(theme["invalid_paper_ids"], ["FAKE-ID"])
        self.assertFalse(theme["references_valid"])

    # 10. Invalid method references are flagged
    def test_invalid_method_reference_is_flagged(self):
        data = valid_analysis()
        data["methodologies_and_benchmarks"][0][
            "papers_using"
        ] = ["FAKE-ID"]

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        self.assertEqual(
            result["methodologies_and_benchmarks"][0]["invalid_paper_ids"],
            ["FAKE-ID"],
        )

    # 11. Invalid limitation references are flagged
    def test_invalid_limitation_reference_is_flagged(self):
        data = valid_analysis()
        data["limitations_and_research_gaps"][0][
            "reported_by"
        ] = ["FAKE-ID"]

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        self.assertEqual(
            result["limitations_and_research_gaps"][0]["invalid_paper_ids"],
            ["FAKE-ID"],
        )

    # 12. Invalid contradiction references are flagged
    def test_invalid_contradiction_references_are_flagged(self):
        data = valid_analysis()
        data["divergent_findings_and_contradictions"] = [
            {
                "topic": "Performance",
                "finding_a": "A",
                "paper_a_id": PAPER_A,
                "finding_b": "B",
                "paper_b_id": "FAKE-ID",
                "explanation": "Different outcomes",
            }
        ]

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        item = result["divergent_findings_and_contradictions"][0]

        self.assertTrue(item["paper_a_valid"])
        self.assertFalse(item["paper_b_valid"])

    # 13. A source claim with invalid citation is not verified
    def test_invalid_claim_citation_is_unverified(self):
        data = valid_analysis()
        data["evidence_linked_claims"][0]["paper_ids"] = ["FAKE-ID"]

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        claim = result["evidence_linked_claims"][0]

        self.assertFalse(claim["citation_valid"])
        self.assertEqual(claim["evidence_status"], "unverified")

    # 14. An unverified quote is not labelled verified
    def test_fabricated_claim_excerpt_is_unverified(self):
        data = valid_analysis()
        data["evidence_linked_claims"][0]["evidence_excerpt"] = (
            "language model quality improves quantum teleportation"
        )

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        claim = result["evidence_linked_claims"][0]

        self.assertFalse(claim["excerpt_verified"])
        self.assertEqual(claim["evidence_status"], "unverified")

    # 15. Inferences remain labelled separately
    def test_inference_is_not_marked_as_verified_source_claim(self):
        data = valid_analysis()
        data["evidence_linked_claims"][0]["support_type"] = "inference"
        data["evidence_linked_claims"][0]["evidence_excerpt"] = ""

        result = validate_and_enrich_analysis(
            data, build_corpus_index(self.papers)
        )
        claim = result["evidence_linked_claims"][0]

        self.assertEqual(claim["evidence_status"], "inference")
        self.assertEqual(result["metadata"]["inference_claims"], 1)

    # 16. Missing sections fail schema validation
    def test_missing_section_fails_validation(self):
        data = valid_analysis()
        del data["methodologies_and_benchmarks"]

        with self.assertRaises(ValueError):
            validate_and_enrich_analysis(
                data, build_corpus_index(self.papers)
            )

    # 17. Incorrect section type fails schema validation
    def test_wrong_section_type_fails_validation(self):
        data = valid_analysis()
        data["key_themes_and_consensus"] = "not a list"

        with self.assertRaises(ValueError):
            validate_and_enrich_analysis(
                data, build_corpus_index(self.papers)
            )

    # 18. Malformed model JSON raises a clear error
    @patch("agents.analysis_agent.get_gemini_client")
    def test_malformed_json(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.models.generate_content.return_value.text = "not JSON"

        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            analyze_literature("Research question", self.papers)

    # 19. Gemini API failures are wrapped
    @patch("agents.analysis_agent.get_gemini_client")
    def test_api_error(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.models.generate_content.side_effect = Exception("API offline")

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            analyze_literature("Research question", self.papers)

    # 20. The formatted report shows verification status
    def test_report_contains_evidence_status(self):
        result = validate_and_enrich_analysis(
            valid_analysis(), build_corpus_index(self.papers)
        )
        report = format_analysis_report(result)

        self.assertIn("EVIDENCE-LINKED CLAIMS", report)
        self.assertIn("EXCERPT_MATCHED", report)
        self.assertNotIn("[CLM-001] VERIFIED", report)

    # 21. Fallback structure is consistent
    def test_insufficient_evidence_structure(self):
        result = build_insufficient_evidence_response("Not enough papers.")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["evidence_linked_claims"], [])

    def test_existing_excerpt_does_not_prove_claim_semantically(self):
        from agents.analysis_agent import validate_and_enrich_analysis

        corpus = {
            "P1": {
                "id": "P1",
                "title": "RAG Evaluation",
                "abstract": (
                    "The proposed method reduced hallucinations "
                    "on the tested benchmark."
                ),
                "authors": [],
                "year": 2025,
                "url": "https://example.com/paper1",
            }
        }

        analysis = {
            "summary": "The method eliminates hallucinations everywhere.",
            "key_themes_and_consensus": [],
            "divergent_findings_and_contradictions": [],
            "methodologies_and_benchmarks": [],
            "limitations_and_research_gaps": [],
            "evidence_linked_claims": [
                {
                    "claim_id": "CLM-001",
                    "claim": (
                        "The method eliminates hallucinations "
                        "in all models and settings."
                    ),
                    "support_type": "source_supported",
                    "paper_ids": ["P1"],
                    "evidence_excerpt": (
                        "The proposed method reduced hallucinations "
                        "on the tested benchmark."
                    ),
                }
            ],
        }

        result = validate_and_enrich_analysis(analysis, corpus)

        claim = result["evidence_linked_claims"][0]

        # This documents the current limitation:
        # matching text alone is not semantic claim verification.
        # Exact excerpt matching is not semantic claim verification.
        self.assertTrue(claim["excerpt_verified"])
        self.assertEqual(claim["evidence_status"], "excerpt_matched")
        self.assertEqual(
            result["metadata"]["verified_source_supported_claims"], 0
        )
        self.assertEqual(result["metadata"]["verified_excerpt_count"], 1)



# ---------------------------------------------------------------------------
# Error handling, retries and extra validation tests (no real API calls)
# ---------------------------------------------------------------------------
import json
from google.genai import errors as genai_errors

from agents import analysis_agent as agent


FAKE_KEY = "AIzaSyFAKEKEYFORTESTING1234567890abcd"


def api_error(code, status, message="error", details=None):
    """Build a real google-genai error the way the SDK would."""
    body = {"error": {"code": code, "status": status, "message": message}}
    if details:
        body["error"]["details"] = details
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


def quota_details(quota_id, retry_delay=None):
    details = [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/"
                           "generate_content_free_tier_requests",
            "quotaId": quota_id,
        }],
    }]
    if retry_delay:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": retry_delay,
        })
    return details


DAILY_429 = api_error(
    429, "RESOURCE_EXHAUSTED", "Quota exceeded",
    quota_details("GenerateRequestsPerDayPerModel-FreeTier"),
)
MINUTE_429 = api_error(
    429, "RESOURCE_EXHAUSTED", "Quota exceeded",
    quota_details("GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                  "37s"),
)
ERROR_503 = api_error(503, "UNAVAILABLE", "This model is currently "
                      "experiencing high demand.")


def ok_response(text=None):
    response = MagicMock()
    response.text = text if text is not None else json.dumps(valid_analysis())
    return response


def papers_for_tests():
    return [
        {"id": PAPER_A, "title": "Self-RAG", "year": 2024, "url": PAPER_A,
         "abstract": "We introduce Self-RAG, a framework that improves "
                     "language model quality and factual accuracy through "
                     "adaptive retrieval and self-reflection tokens."},
        {"id": PAPER_B, "title": "Corrective Retrieval", "year": 2024,
         "url": PAPER_B,
         "abstract": "We propose corrective retrieval to improve robustness "
                     "by evaluating retrieval quality and triggering web "
                     "search."},
    ]


class TestErrorClassification(unittest.TestCase):

    def test_503_is_transient(self):
        self.assertEqual(agent.classify_gemini_error(ERROR_503),
                         agent.TRANSIENT)

    def test_500_internal_is_transient(self):
        err = api_error(500, "INTERNAL")
        self.assertEqual(agent.classify_gemini_error(err), agent.TRANSIENT)

    def test_daily_429_is_daily_quota(self):
        self.assertEqual(agent.classify_gemini_error(DAILY_429),
                         agent.DAILY_QUOTA)

    def test_per_minute_429_is_rate_limit_not_daily(self):
        # Same metric name as the daily limit, but a per-minute quotaId.
        self.assertEqual(agent.classify_gemini_error(MINUTE_429),
                         agent.RATE_LIMIT)

    def test_plain_429_without_details_is_rate_limit(self):
        err = api_error(429, "RESOURCE_EXHAUSTED", "Slow down")
        self.assertEqual(agent.classify_gemini_error(err), agent.RATE_LIMIT)

    def test_auth_and_bad_request_errors_are_fatal(self):
        for code, status in ((400, "INVALID_ARGUMENT"),
                             (401, "UNAUTHENTICATED"),
                             (403, "PERMISSION_DENIED"),
                             (404, "NOT_FOUND")):
            with self.subTest(code=code):
                self.assertEqual(
                    agent.classify_gemini_error(api_error(code, status)),
                    agent.FATAL,
                )

    def test_numbers_inside_message_do_not_trigger_retry(self):
        # "500" appears in the text, but the real code is 400.
        err = api_error(400, "INVALID_ARGUMENT",
                        "Prompt has 500 tokens; model not found")
        self.assertEqual(agent.classify_gemini_error(err), agent.FATAL)

    def test_plain_exception_is_fatal(self):
        self.assertEqual(agent.classify_gemini_error(Exception("boom")),
                         agent.FATAL)

    def test_message_fallback_when_error_has_no_code_attribute(self):
        self.assertEqual(
            agent.classify_gemini_error(
                Exception("503 UNAVAILABLE. high demand")),
            agent.TRANSIENT,
        )

    def test_connection_errors_are_transient(self):
        self.assertEqual(agent.classify_gemini_error(ConnectionError("x")),
                         agent.TRANSIENT)


@patch("agents.analysis_agent.time.sleep")
@patch("agents.analysis_agent.get_gemini_client")
class TestRetryBehaviour(unittest.TestCase):

    def setUp(self):
        self.papers = papers_for_tests()

    def _client(self, mock_get_client, side_effect):
        client = MagicMock()
        client.models.generate_content.side_effect = side_effect
        mock_get_client.return_value = client
        return client

    def test_503_then_success(self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client,
                              [ERROR_503, ERROR_503, ok_response()])

        result = analyze_literature("Question?", self.papers)

        self.assertEqual(result["status"], "success")
        self.assertEqual(client.models.generate_content.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_repeated_503_stops_after_max_attempts(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client,
                              [ERROR_503] * agent.MAX_ATTEMPTS)

        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            analyze_literature("Question?", self.papers)

        self.assertEqual(client.models.generate_content.call_count,
                         agent.MAX_ATTEMPTS)
        # No sleep after the final failed attempt.
        self.assertEqual(mock_sleep.call_count, agent.MAX_ATTEMPTS - 1)

    def test_backoff_grows_and_is_bounded(self, mock_get_client, mock_sleep):
        self._client(mock_get_client, [ERROR_503] * agent.MAX_ATTEMPTS)

        with self.assertRaises(RuntimeError):
            analyze_literature("Question?", self.papers)

        waits = [call.args[0] for call in mock_sleep.call_args_list]
        self.assertEqual(waits, sorted(waits))
        self.assertTrue(all(w <= agent.MAX_DELAY_SECONDS for w in waits))

    def test_daily_quota_stops_immediately(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client, [DAILY_429, ok_response()])

        with self.assertRaisesRegex(RuntimeError, "daily quota exhausted"):
            analyze_literature("Question?", self.papers)

        self.assertEqual(client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    def test_per_minute_429_is_retried_and_honours_server_delay(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client, [MINUTE_429, ok_response()])

        result = analyze_literature("Question?", self.papers)

        self.assertEqual(result["status"], "success")
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertGreaterEqual(mock_sleep.call_args.args[0], 37)

    def test_auth_error_is_not_retried(self, mock_get_client, mock_sleep):
        client = self._client(
            mock_get_client, [api_error(403, "PERMISSION_DENIED", "bad key")])

        with self.assertRaisesRegex(RuntimeError, "not temporary"):
            analyze_literature("Question?", self.papers)

        self.assertEqual(client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    def test_invalid_model_name_is_not_retried(
            self, mock_get_client, mock_sleep):
        client = self._client(
            mock_get_client, [api_error(404, "NOT_FOUND", "model not found")])

        with self.assertRaises(RuntimeError):
            analyze_literature("Question?", self.papers)

        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_configured_model_is_used_on_every_attempt(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client, [ERROR_503, ok_response()])

        analyze_literature("Question?", self.papers)

        models = [c.kwargs["model"]
                  for c in client.models.generate_content.call_args_list]
        self.assertEqual(models, ["gemini-3.6-flash"] * 2)

    def test_api_key_is_not_leaked_in_error_message(
            self, mock_get_client, mock_sleep):
        self._client(mock_get_client, [api_error(
            400, "INVALID_ARGUMENT", f"bad request key={FAKE_KEY}")])

        with patch.dict("os.environ", {"GEMINI_API_KEY": FAKE_KEY}):
            with self.assertRaises(RuntimeError) as ctx:
                analyze_literature("Question?", self.papers)

        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_api_key_is_not_leaked_in_retry_logs(
            self, mock_get_client, mock_sleep):
        self._client(mock_get_client, [ERROR_503, ok_response()])

        with patch.dict("os.environ", {"GEMINI_API_KEY": FAKE_KEY}):
            with patch("builtins.print") as mock_print:
                analyze_literature("Question?", self.papers)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertNotIn(FAKE_KEY, printed)


@patch("agents.analysis_agent.time.sleep")
@patch("agents.analysis_agent.get_gemini_client")
class TestModelOutputHandling(unittest.TestCase):

    def setUp(self):
        self.papers = papers_for_tests()

    def _client(self, mock_get_client, side_effect):
        client = MagicMock()
        client.models.generate_content.side_effect = side_effect
        mock_get_client.return_value = client
        return client

    def test_empty_response_then_valid_response_recovers(
            self, mock_get_client, mock_sleep):
        empty = MagicMock()
        empty.text = None
        client = self._client(mock_get_client, [empty, ok_response()])

        result = analyze_literature("Question?", self.papers)

        self.assertEqual(result["status"], "success")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_persistently_empty_response_raises_clear_error(
            self, mock_get_client, mock_sleep):
        empty = MagicMock()
        empty.text = ""
        client = self._client(mock_get_client, [empty, empty, empty])

        with self.assertRaisesRegex(ValueError, "empty response"):
            analyze_literature("Question?", self.papers)

        # Bounded: one re-ask only, not an endless loop.
        self.assertEqual(client.models.generate_content.call_count,
                         agent.MAX_FORMAT_ATTEMPTS)

    def test_persistently_malformed_json_is_not_turned_into_a_report(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client,
                              [ok_response("{not json"),
                               ok_response("{still not json")])

        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            analyze_literature("Question?", self.papers)

        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_markdown_fenced_json_is_accepted(
            self, mock_get_client, mock_sleep):
        fenced = "```json\n" + json.dumps(valid_analysis()) + "\n```"
        self._client(mock_get_client, [ok_response(fenced)])

        result = analyze_literature("Question?", self.papers)

        self.assertEqual(result["status"], "success")

    def test_wrong_field_type_raises_instead_of_fabricating(
            self, mock_get_client, mock_sleep):
        bad = valid_analysis()
        bad["evidence_linked_claims"][0]["paper_ids"] = "not-a-list"
        self._client(mock_get_client, [ok_response(json.dumps(bad))])

        with self.assertRaises(ValueError):
            analyze_literature("Question?", self.papers)

    def test_unknown_paper_in_model_output_is_flagged(
            self, mock_get_client, mock_sleep):
        data = valid_analysis()
        data["evidence_linked_claims"][0]["paper_ids"] = [PAPER_A, "W999"]
        self._client(mock_get_client, [ok_response(json.dumps(data))])

        result = analyze_literature("Question?", self.papers)
        claim = result["evidence_linked_claims"][0]

        self.assertEqual(claim["invalid_paper_ids"], ["W999"])
        self.assertEqual(claim["evidence_status"], "unverified")
        self.assertEqual(result["metadata"]["invalid_reference_count"], 1)

    def test_api_failure_is_runtime_error_not_insufficient_evidence(
            self, mock_get_client, mock_sleep):
        # A temporary outage must not look like "no evidence found".
        self._client(mock_get_client, [ERROR_503] * agent.MAX_ATTEMPTS)

        with self.assertRaises(RuntimeError):
            analyze_literature("Question?", self.papers)

    def test_no_abstracts_means_no_api_call(
            self, mock_get_client, mock_sleep):
        papers = [{"id": "W1", "title": "T", "abstract": ""}]

        result = analyze_literature("Question?", papers)

        self.assertEqual(result["status"], "insufficient_evidence")
        mock_get_client.assert_not_called()

    def test_prompt_reports_how_many_abstracts_exist(
            self, mock_get_client, mock_sleep):
        papers = papers_for_tests() + [
            {"id": "W3", "title": "No abstract", "abstract": ""}]
        client = self._client(mock_get_client, [ok_response()])

        result = analyze_literature("Question?", papers)

        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("2 of 3 papers have an abstract", prompt)
        self.assertEqual(
            result["metadata"]["papers_with_usable_abstracts"], 2)
        self.assertEqual(result["metadata"]["total_papers_analyzed"], 3)

    def test_verified_claim_note_does_not_overstate(
            self, mock_get_client, mock_sleep):
        self._client(mock_get_client, [ok_response()])

        result = analyze_literature("Question?", self.papers)
        claim = result["evidence_linked_claims"][0]

        self.assertEqual(claim["evidence_status"], "excerpt_matched")
        self.assertIn(
            "semantic support for the complete claim has not been verified",
            claim["verification_note"],
        )

    def test_formatted_report_matches_app_usage(
            self, mock_get_client, mock_sleep):
        # app.py calls analyze_literature(user_query=, papers=, plan=)
        # and then prints format_analysis_report(analysis).
        self._client(mock_get_client, [ok_response()])

        analysis = analyze_literature(
            user_query="Question?", papers=self.papers,
            plan={"sub_questions": ["Sub question 1", "Sub question 2"]},
        )
        report = format_analysis_report(analysis)

        self.assertIn("LITERATURE SYNTHESIS", report)
        self.assertIn("2 with abstracts", report)

    def test_plan_sub_questions_reach_the_prompt(
            self, mock_get_client, mock_sleep):
        client = self._client(mock_get_client, [ok_response()])

        analyze_literature("Q?", self.papers,
                           plan={"sub_questions": ["What is X?"]})

        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("What is X?", prompt)


class TestRedaction(unittest.TestCase):

    def test_redacts_known_key_and_key_like_strings(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "my-secret-value"}):
            text = agent.redact_secrets(
                f"fail my-secret-value and {FAKE_KEY} and api_key=abc123")
        self.assertNotIn("my-secret-value", text)
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("abc123", text)


if __name__ == "__main__":
    unittest.main()
