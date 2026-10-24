"""
Tests for the Claim Verification Agent.

No test makes a real Gemini call: the client is a mock that returns
scripted JSON. Run with:  python -m pytest -q
"""

import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from google.genai import errors as genai_errors

from agents import verification_agent as va
from agents.analysis_agent import validate_and_enrich_analysis, build_corpus_index
from agents.verification_agent import (
    format_verification_report,
    verify_analysis,
)


PAPER_A = "https://openalex.org/W1"
PAPER_B = "https://openalex.org/W2"
PAPER_NO_ABSTRACT = "https://doi.org/10.1/none"

ABSTRACT_A = (
    "We introduce Self-RAG, a framework that improves language model "
    "quality and factual accuracy through adaptive retrieval and "
    "self-reflection tokens. Experiments show it outperforms ChatGPT "
    "on open-domain question answering."
)
ABSTRACT_B = (
    "We propose corrective retrieval to improve robustness by evaluating "
    "retrieval quality and triggering web search."
)

PAPERS = [
    {"id": PAPER_A, "title": "Self-RAG", "abstract": ABSTRACT_A,
     "authors": ["A"], "year": 2024, "url": PAPER_A},
    {"id": PAPER_B, "title": "Corrective Retrieval", "abstract": ABSTRACT_B,
     "authors": ["B"], "year": 2024, "url": PAPER_B},
    {"id": PAPER_NO_ABSTRACT, "title": "Crossref Record", "abstract": "",
     "authors": [], "year": 2023, "url": PAPER_NO_ABSTRACT},
]


def make_claim(claim_id="CLM-001", text="Self-RAG improves factual accuracy.",
               ids=None, support_type="source_supported", invalid=None):
    return {
        "claim_id": claim_id,
        "claim": text,
        "support_type": support_type,
        "paper_ids": [PAPER_A] if ids is None else ids,
        "invalid_paper_ids": invalid or [],
        "paper_titles": [],
        "citation_valid": True,
        "evidence_excerpt": "improves language model quality",
        "excerpt_verified": True,
        "evidence_status": "excerpt_matched",
        "verification_note": "original note",
    }


def make_analysis(*claims):
    return {
        "status": "success",
        "summary": "A narrative summary.",
        "key_themes_and_consensus": [],
        "divergent_findings_and_contradictions": [],
        "methodologies_and_benchmarks": [],
        "limitations_and_research_gaps": [],
        "evidence_linked_claims": list(claims),
        "metadata": {"total_claims": len(claims)},
    }


def verdict(status="SUPPORTED", confidence=0.9, evidence=None,
            justification="Because the abstract says so.", warning=""):
    return {
        "status": status,
        "confidence": confidence,
        "justification": justification,
        "evidence": evidence if evidence is not None else [],
        "warning": warning,
    }


def response(payload):
    resp = MagicMock()
    resp.text = payload if isinstance(payload, str) else json.dumps(payload)
    return resp


def client_returning(*payloads):
    """Mock client whose generate_content returns the payloads in order."""
    client = MagicMock()
    client.models.generate_content.side_effect = [
        response(p) for p in payloads
    ]
    return client


def api_error(code, status, message="error"):
    return genai_errors.APIError(
        code, {"error": {"code": code, "status": status,
                          "message": message}}
    )


class VerificationBase(unittest.TestCase):
    def setUp(self):
        # Never really sleep during retry tests.
        patcher = patch.object(va.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def verify(self, analysis, client):
        return verify_analysis(analysis, PAPERS, client=client)

    @staticmethod
    def record(result, index=0):
        return result["evidence_linked_claims"][index]["verification"]


class TestVerdicts(VerificationBase):

    def test_1_directly_supported_claim(self):
        client = client_returning(verdict("SUPPORTED", 0.95, [
            {"paper_id": PAPER_A,
             "excerpt": "improves language model quality and factual accuracy"}
        ]))
        result = self.verify(make_analysis(make_claim()), client)
        rec = self.record(result)
        self.assertEqual(rec["status"], "SUPPORTED")
        self.assertEqual(rec["evidence_paper_id"], PAPER_A)
        self.assertEqual(rec["evidence_paper_title"], "Self-RAG")
        self.assertFalse(rec["needs_review"])
        self.assertEqual(result["verification_summary"]["supported"], 1)
        self.assertEqual(
            result["verification_summary"]["flagged_for_review"], 0
        )

    def test_2_partially_supported_claim(self):
        client = client_returning(verdict("PARTIALLY_SUPPORTED", 0.6, [
            {"paper_id": PAPER_A, "excerpt": "outperforms ChatGPT"}
        ], warning="No percentage is given."))
        claim = make_claim(text="Self-RAG beats ChatGPT by 20%.")
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["status"], "PARTIALLY_SUPPORTED")
        self.assertTrue(rec["needs_review"])
        self.assertTrue(any("percentage" in w for w in rec["warnings"]))

    def test_3_false_claim_without_evidence(self):
        client = client_returning(verdict("UNSUPPORTED", 0.9))
        claim = make_claim(text="Self-RAG cures cancer.")
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["status"], "UNSUPPORTED")
        self.assertEqual(rec["evidence_excerpt"], "")
        self.assertTrue(rec["needs_review"])

    def test_4_contradicted_claim(self):
        # Even if the model attaches a quote, no "supporting" quote is
        # shown next to an UNSUPPORTED verdict.
        client = client_returning(verdict("UNSUPPORTED", 0.85, [
            {"paper_id": PAPER_A, "excerpt": "outperforms ChatGPT"}
        ], justification="The abstract says it outperforms ChatGPT."))
        claim = make_claim(text="Self-RAG performs worse than ChatGPT.")
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["status"], "UNSUPPORTED")
        self.assertEqual(rec["evidence_excerpt"], "")

    def test_13_inference_is_never_supported(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}
        ]))
        claim = make_claim(support_type="inference")
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["status"], "PARTIALLY_SUPPORTED")
        self.assertTrue(rec["is_inference"])
        self.assertTrue(rec["needs_review"])
        self.assertTrue(any("inference" in w for w in rec["warnings"]))

    def test_invalid_status_from_model_is_unverifiable(self):
        client = client_returning(verdict("TOTALLY_TRUE"))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")

    def test_confidence_is_clamped(self):
        client = client_returning(verdict("UNSUPPORTED", 7))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["confidence"], 1.0)

        client = client_returning(verdict("UNSUPPORTED", "high"))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["confidence"], 0.0)


class TestEvidenceLookup(VerificationBase):

    def test_5_unknown_paper_id(self):
        client = client_returning()  # must not be called
        claim = make_claim(ids=[], invalid=["https://openalex.org/W999"])
        result = self.verify(make_analysis(claim), client)
        rec = self.record(result)
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        self.assertEqual(rec["missing_paper_ids"],
                         ["https://openalex.org/W999"])
        client.models.generate_content.assert_not_called()

    def test_unknown_id_alongside_known_id_is_flagged(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}
        ]))
        claim = make_claim(invalid=["https://openalex.org/W999"])
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["missing_paper_ids"],
                         ["https://openalex.org/W999"])
        self.assertTrue(rec["needs_review"])
        self.assertTrue(any("not found" in w for w in rec["warnings"]))

    def test_6_paper_without_abstract(self):
        client = client_returning()  # must not be called
        claim = make_claim(ids=[PAPER_NO_ABSTRACT])
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        self.assertTrue(any("no usable abstract" in w
                            for w in rec["warnings"]))
        client.models.generate_content.assert_not_called()

    def test_claim_with_no_citations(self):
        client = client_returning()
        rec = self.record(self.verify(
            make_analysis(make_claim(ids=[])), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        client.models.generate_content.assert_not_called()

    def test_only_cited_papers_are_sent_to_the_model(self):
        client = client_returning(verdict("UNSUPPORTED", 0.8))
        self.verify(make_analysis(make_claim(ids=[PAPER_A])), client)
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn(ABSTRACT_A, prompt)
        self.assertNotIn(ABSTRACT_B, prompt)


class TestQuoteValidation(VerificationBase):

    def test_7_hallucinated_quote_is_discarded(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_A,
             "excerpt": "cuts hallucinations by ninety percent"}
        ]))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["evidence_excerpt"], "")
        self.assertEqual(rec["evidence"], [])
        # No verifiable quote -> cannot stay SUPPORTED.
        self.assertEqual(rec["status"], "PARTIALLY_SUPPORTED")
        self.assertTrue(any("discarded" in w for w in rec["warnings"]))

    def test_8_valid_quote_attributed_to_wrong_paper(self):
        # The quote is real, but it belongs to paper B, not paper A.
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_A,
             "excerpt": "evaluating retrieval quality and triggering web "
                        "search"}
        ]))
        claim = make_claim(ids=[PAPER_A, PAPER_B])
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["evidence_excerpt"], "")
        self.assertNotEqual(rec["status"], "SUPPORTED")

    def test_quote_attributed_to_uncited_paper_is_discarded(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_B, "excerpt": "corrective retrieval"}
        ]))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["evidence_excerpt"], "")

    def test_quote_maps_to_the_right_paper_when_two_are_cited(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_B,
             "excerpt": "evaluating retrieval quality"}
        ]))
        claim = make_claim(ids=[PAPER_A, PAPER_B])
        rec = self.record(self.verify(make_analysis(claim), client))
        self.assertEqual(rec["evidence_paper_id"], PAPER_B)
        self.assertEqual(rec["evidence_paper_title"],
                         "Corrective Retrieval")


class TestFailures(VerificationBase):

    def test_9_malformed_json_is_unverifiable_not_unsupported(self):
        client = client_returning("this is not json", "{broken")
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        self.assertNotEqual(rec["status"], "UNSUPPORTED")
        # one initial request + one re-ask, then give up
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_malformed_then_valid_json_recovers(self):
        client = client_returning("not json", verdict("UNSUPPORTED", 0.7))
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNSUPPORTED")

    def test_empty_response_is_unverifiable(self):
        client = client_returning("", "")
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")

    def test_json_that_is_not_an_object(self):
        client = client_returning("[1, 2]", "[3]")
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNVERIFIABLE")

    def test_10_temporary_error_is_retried_then_succeeds(self):
        client = MagicMock()
        client.models.generate_content.side_effect = [
            api_error(503, "UNAVAILABLE"),
            response(verdict("UNSUPPORTED", 0.8)),
        ]
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNSUPPORTED")
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.sleep.assert_called()

    def test_persistent_temporary_error_is_bounded_and_unverifiable(self):
        client = MagicMock()
        client.models.generate_content.side_effect = api_error(
            503, "UNAVAILABLE")
        result = self.verify(make_analysis(make_claim()), client)
        rec = self.record(result)
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        self.assertEqual(client.models.generate_content.call_count,
                         va.MAX_API_ATTEMPTS)
        self.assertEqual(
            result["verification_summary"]["service_status"],
            "unavailable")
        self.assertEqual(result["verification_summary"]["unsupported"], 0)
        for call in self.sleep.call_args_list:
            self.assertLessEqual(call.args[0], va.MAX_DELAY_SECONDS)

    def test_rate_limit_429_is_retried(self):
        client = MagicMock()
        client.models.generate_content.side_effect = [
            api_error(429, "RESOURCE_EXHAUSTED"),
            response(verdict("UNSUPPORTED", 0.8)),
        ]
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertEqual(rec["status"], "UNSUPPORTED")

    def test_fatal_error_is_not_retried_and_stops_remaining_claims(self):
        client = MagicMock()
        client.models.generate_content.side_effect = api_error(
            400, "INVALID_ARGUMENT")
        analysis = make_analysis(make_claim("CLM-001"), make_claim("CLM-002"))
        result = self.verify(analysis, client)
        self.assertEqual(client.models.generate_content.call_count, 1)
        self.assertEqual(result["verification_summary"]["unverifiable"], 2)

    def test_missing_api_key_degrades_gracefully(self):
        with patch.object(va, "get_gemini_client",
                          side_effect=ValueError("GEMINI_API_KEY is missing")):
            result = verify_analysis(make_analysis(make_claim()), PAPERS)
        rec = self.record(result)
        self.assertEqual(rec["status"], "UNVERIFIABLE")
        self.assertEqual(
            result["verification_summary"]["service_status"],
            "unavailable")

    def test_api_key_is_not_leaked_in_error_text(self):
        secret = "AIza" + "x" * 30
        client = MagicMock()
        client.models.generate_content.side_effect = api_error(
            400, "INVALID_ARGUMENT", f"bad key {secret}")
        rec = self.record(self.verify(make_analysis(make_claim()), client))
        self.assertNotIn(secret, json.dumps(rec))


class TestSummaryAndAudit(VerificationBase):

    def test_11_empty_claim_list(self):
        client = client_returning()
        result = self.verify(make_analysis(), client)
        summary = result["verification_summary"]
        self.assertEqual(summary["total_claims"], 0)
        self.assertEqual(summary["supported"], 0)
        client.models.generate_content.assert_not_called()
        self.assertIn("No claims", format_verification_report(result))

    def test_12_mixed_claims_have_correct_counts(self):
        client = client_returning(
            verdict("SUPPORTED", 0.9, [
                {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}]),
            verdict("PARTIALLY_SUPPORTED", 0.5, [
                {"paper_id": PAPER_A, "excerpt": "outperforms ChatGPT"}]),
            verdict("UNSUPPORTED", 0.9),
        )
        analysis = make_analysis(
            make_claim("CLM-001"),
            make_claim("CLM-002"),
            make_claim("CLM-003"),
            make_claim("CLM-004", ids=[PAPER_NO_ABSTRACT]),
        )
        summary = self.verify(analysis, client)["verification_summary"]
        self.assertEqual(summary["total_claims"], 4)
        self.assertEqual(summary["supported"], 1)
        self.assertEqual(summary["partially_supported"], 1)
        self.assertEqual(summary["unsupported"], 1)
        self.assertEqual(summary["unverifiable"], 1)
        self.assertEqual(summary["flagged_for_review"], 3)
        self.assertEqual(
            summary["supported"] + summary["partially_supported"]
            + summary["unsupported"] + summary["unverifiable"],
            summary["total_claims"])

    def test_14_original_analysis_is_preserved(self):
        analysis = make_analysis(make_claim())
        snapshot = copy.deepcopy(analysis)
        client = client_returning(verdict("UNSUPPORTED", 0.9))
        result = self.verify(analysis, client)

        # The input object was not mutated.
        self.assertEqual(analysis, snapshot)
        self.assertNotIn("verification_summary", analysis)
        self.assertNotIn("verification", analysis["evidence_linked_claims"][0])

        # Every original claim field is unchanged in the copy.
        new_claim = result["evidence_linked_claims"][0]
        for key, value in snapshot["evidence_linked_claims"][0].items():
            self.assertEqual(new_claim[key], value)
        self.assertEqual(result["summary"], snapshot["summary"])

    def test_narrative_is_never_marked_verified(self):
        client = client_returning(verdict("SUPPORTED", 0.9, [
            {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}]))
        result = self.verify(make_analysis(make_claim()), client)
        summary = result["verification_summary"]
        self.assertFalse(summary["narrative_verified"])
        self.assertIn("NOT verified", summary["scope_notice"])
        self.assertNotIn("verified", result["status"])

    def test_excerpt_match_does_not_count_as_verification(self):
        # The claim already has evidence_status "excerpt_matched", but the
        # model says UNSUPPORTED. The verifier's verdict must win, and the
        # original status stays visible for auditing.
        client = client_returning(verdict("UNSUPPORTED", 0.9))
        result = self.verify(make_analysis(make_claim()), client)
        claim = result["evidence_linked_claims"][0]
        self.assertEqual(claim["evidence_status"], "excerpt_matched")
        self.assertEqual(claim["verification"]["status"], "UNSUPPORTED")

    def test_invalid_analysis_input(self):
        with self.assertRaises(ValueError):
            verify_analysis("not a dict", PAPERS)
        with self.assertRaises(ValueError):
            verify_analysis({"evidence_linked_claims": "oops"}, PAPERS)


class TestReport(VerificationBase):

    def test_report_shows_all_required_parts(self):
        client = client_returning(
            verdict("SUPPORTED", 0.93, [
                {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}]),
            verdict("UNSUPPORTED", 0.8, justification="Not in abstract."),
        )
        analysis = make_analysis(
            make_claim("CLM-001"), make_claim("CLM-002", text="Bad claim."))
        text = format_verification_report(self.verify(analysis, client))
        for expected in ("VERIFICATION REPORT", "CLM-001", "SUPPORTED",
                         "CLM-002", "UNSUPPORTED", "0.93",
                         "adaptive retrieval", "Self-RAG",
                         "Not in abstract.", "REVIEW NEEDED",
                         "not a calibrated probability",
                         "NOT verified by this step"):
            self.assertIn(expected, text)

    def test_report_without_verification_says_not_verified(self):
        text = format_verification_report(make_analysis(make_claim()))
        self.assertIn("VERIFICATION NOT RUN", text)

    def test_report_states_service_outage(self):
        client = MagicMock()
        client.models.generate_content.side_effect = api_error(
            503, "UNAVAILABLE")
        text = format_verification_report(
            self.verify(make_analysis(make_claim()), client))
        self.assertIn("UNAVAILABLE", text)
        self.assertIn("UNVERIFIABLE, not unsupported", text)


class TestAnalysisAgentCompatibility(VerificationBase):

    def test_15_works_on_real_analysis_agent_output(self):
        """Feed output of the real validate_and_enrich_analysis()."""
        raw = {
            "summary": "Summary.",
            "key_themes_and_consensus": [],
            "divergent_findings_and_contradictions": [],
            "methodologies_and_benchmarks": [],
            "limitations_and_research_gaps": [],
            "evidence_linked_claims": [
                {"claim_id": "CLM-001",
                 "claim": "Self-RAG improves factual accuracy.",
                 "support_type": "source_supported",
                 "paper_ids": [PAPER_A],
                 "evidence_excerpt":
                     "improves language model quality and factual accuracy"},
                {"claim_id": "CLM-002",
                 "claim": "Retrieval will be central to future systems.",
                 "support_type": "inference",
                 "paper_ids": [PAPER_A, "https://openalex.org/W404"],
                 "evidence_excerpt": ""},
            ],
        }
        analysis = validate_and_enrich_analysis(
            raw, build_corpus_index(PAPERS))
        client = client_returning(
            verdict("SUPPORTED", 0.9, [
                {"paper_id": PAPER_A,
                 "excerpt": "improves language model quality and factual "
                            "accuracy"}]),
            verdict("PARTIALLY_SUPPORTED", 0.4, [
                {"paper_id": PAPER_A, "excerpt": "adaptive retrieval"}]),
        )
        result = self.verify(analysis, client)

        first, second = result["evidence_linked_claims"]
        self.assertEqual(first["verification"]["status"], "SUPPORTED")
        # Analysis Agent fields survive untouched.
        self.assertEqual(first["evidence_status"], "excerpt_matched")
        self.assertEqual(second["evidence_status"], "inference")
        # The invalid citation the Analysis Agent recorded is carried over.
        self.assertEqual(second["verification"]["missing_paper_ids"],
                         ["https://openalex.org/W404"])
        self.assertEqual(result["metadata"], analysis["metadata"])

    def test_insufficient_evidence_response(self):
        from agents.analysis_agent import build_insufficient_evidence_response
        analysis = build_insufficient_evidence_response("No papers.")
        result = self.verify(analysis, client_returning())
        self.assertEqual(result["verification_summary"]["total_claims"], 0)


if __name__ == "__main__":
    unittest.main()
