"""
Pure, deterministic metric functions for the Phase 7 Evaluation Framework.

Nothing here calls an LLM, the network, or mutates its inputs.

Every function returns a *metric*: a JSON-serializable dict with

    status       one of the STATUS_* constants below
    description  what the metric measures
    value        the headline number/object (None if not calculable)
    reason       why it is not_evaluated / error (when applicable)
    details      optional breakdown

Status meanings
    calculated     computed exactly from the inputs
    heuristic      an estimate from a keyword rule; NOT semantic verification
    not_evaluated  could not be computed (missing input / zero denominator)
    error          the metric code itself failed (the error text is sanitized)
    needs_review   computed, but the result calls for a human to look at it

A zero is a real result ("0 of 12"); ``None`` + ``not_evaluated`` means
"unknown". The two are never mixed up.

Identity rules (duplicates, label matching) reuse Phase 6 (``is_same_paper``)
so the whole project agrees on when two records are the same paper. Titles
are never compared fuzzily.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agents.analysis_agent import (
    build_corpus_index,
    verify_excerpt_against_abstract,
)
from agents.iterative_retrieval_agent import (
    IterativeRetrievalConfig,
    _covers,          # Phase 6 coverage rule, reused on purpose
    _raw_id,
    _real_abstract,
    _strong_ids,
    _url_key,
    _valid_subquestions,
    assess_evidence,
    is_same_paper,
    is_usable_paper,
)
from agents.verification_agent import VALID_STATUSES

STATUS_CALCULATED = "calculated"
STATUS_HEURISTIC = "heuristic"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_ERROR = "error"
STATUS_NEEDS_REVIEW = "needs_review"

METRIC_STATUSES = (STATUS_CALCULATED, STATUS_HEURISTIC, STATUS_NOT_EVALUATED,
                   STATUS_ERROR, STATUS_NEEDS_REVIEW)

MAX_LISTED = 10      # cap on list-valued details so output stays small
MAX_TEXT = 120       # cap on any title / claim text copied into details


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def make_metric(status: str, description: str, value: Any = None,
                reason: Optional[str] = None,
                details: Optional[dict] = None) -> Dict[str, Any]:
    metric: Dict[str, Any] = {"status": status, "description": description,
                              "value": value}
    if reason:
        metric["reason"] = reason
    if details is not None:
        metric["details"] = details
    return metric


def not_evaluated(description: str, reason: str,
                  details: Optional[dict] = None) -> Dict[str, Any]:
    return make_metric(STATUS_NOT_EVALUATED, description, None, reason,
                       details)


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    """numerator/denominator rounded to 4 places; None if denominator is 0."""
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _short(text: Any, limit: int = MAX_TEXT) -> str:
    text = " ".join(text.split()) if isinstance(text, str) else ""
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _dict_papers(papers: Any) -> List[dict]:
    return [p for p in papers if isinstance(p, dict)] \
        if isinstance(papers, list) else []


def _paper_id(paper: dict, index: int) -> str:
    """Same ID rule as the Analysis Agent's corpus index."""
    raw = paper.get("id") or paper.get("url") or f"paper_{index}"
    return str(raw).strip() or f"paper_{index}"


def identity_keys(paper: dict) -> set:
    """All strong identifiers of a record (DOI, OpenAlex ID, raw ID, URL).
    Titles are deliberately excluded."""
    keys = set()
    strong = _strong_ids(paper)
    keys |= strong["doi"] | strong["openalex"]
    if _raw_id(paper):
        keys.add("id:" + _raw_id(paper))
    if _url_key(paper):
        keys.add("url:" + _url_key(paper))
    return keys


def label_keys(label: Any) -> set:
    """Identity keys for a bare identifier string (DOI, URL, OpenAlex ID)."""
    if not isinstance(label, str) or not label.strip():
        return set()
    return identity_keys({"id": label, "doi": label, "url": label})


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def duplicate_clusters(papers: Sequence[dict]) -> List[List[int]]:
    """Group record indexes that Phase 6 considers the same paper."""
    parent = list(range(len(papers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if is_same_paper(papers[i], papers[j]):
                parent[find(j)] = find(i)

    groups: Dict[int, List[int]] = {}
    for i in range(len(papers)):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def retrieval_record_count(papers: Any) -> Dict[str, Any]:
    desc = "Number of retrieved paper records."
    if not isinstance(papers, list):
        return not_evaluated(desc, "No paper list was provided.")
    records = _dict_papers(papers)
    return make_metric(STATUS_CALCULATED, desc, len(records), details={
        "records": len(records),
        "non_dict_records_ignored": len(papers) - len(records),
        "records_with_real_title": sum(is_usable_paper(p) for p in records),
    })


def abstract_availability(papers: Any) -> Dict[str, Any]:
    desc = ("Papers with a usable abstract (not empty and not the "
            "'Abstract not available' placeholder).")
    if not isinstance(papers, list):
        return not_evaluated(desc, "No paper list was provided.")
    records = _dict_papers(papers)
    with_abs = sum(1 for p in records if _real_abstract(p))
    ratio = safe_ratio(with_abs, len(records))
    if ratio is None:
        return not_evaluated(desc, "There are no paper records, so the "
                             "proportion is undefined.",
                             {"with_abstract": 0, "total": 0})
    return make_metric(STATUS_CALCULATED, desc, ratio, details={
        "with_abstract": with_abs, "total": len(records),
        "without_abstract": len(records) - with_abs})


def duplicate_papers(papers: Any) -> Dict[str, Any]:
    desc = ("Duplicate paper records, using the Phase 6 identity rule "
            "(shared DOI / OpenAlex ID / raw ID / URL, or identical "
            "normalized title unless DOIs differ). No fuzzy matching.")
    if not isinstance(papers, list):
        return not_evaluated(desc, "No paper list was provided.")
    records = _dict_papers(papers)
    clusters = duplicate_clusters(records)
    duplicates = len(records) - len(clusters)
    groups = [c for c in clusters if len(c) > 1]
    rate = safe_ratio(duplicates, len(records))
    if rate is None:
        return not_evaluated(desc, "There are no paper records, so the "
                             "duplicate rate is undefined.",
                             {"duplicate_count": 0})
    return make_metric(STATUS_CALCULATED, desc, rate, details={
        "duplicate_count": duplicates,
        "unique_papers": len(clusters),
        "total_records": len(records),
        "duplicate_groups": [
            {"record_indexes": g,
             "title": _short(records[g[0]].get("title"))}
            for g in groups[:MAX_LISTED]],
    })


def _ranked_key_sets(items: Iterable[Any]) -> List[set]:
    """Identity key sets for a ranked list of papers (dicts) or ID strings,
    with later duplicates of an earlier item removed."""
    ranked: List[set] = []
    for item in items:
        keys = identity_keys(item) if isinstance(item, dict) \
            else label_keys(item)
        if not keys:
            # Unidentifiable records still occupy a rank (they were
            # retrieved) but can never match a label.
            ranked.append(set())
        elif not any(keys & seen for seen in ranked if seen):
            ranked.append(keys)
    return ranked


def precision_recall_at_k(retrieved: Any, relevant_ids: Any,
                          k: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Precision@k and recall@k against MANUALLY SUPPLIED relevance labels.

    ``retrieved``     ranked list of paper dicts or identifier strings
    ``relevant_ids``  identifiers of papers a human judged relevant

    * Duplicates in ``retrieved`` count once (first occurrence kept).
    * precision@k = relevant in top-k / items in top-k. A list shorter than
      k is NOT padded (divides by the number actually retrieved).
    * recall@k = distinct labels found in top-k / distinct labels.
      Supply ONE label per paper, or recall's denominator is inflated.
    * No labels (None / not a list / empty) -> both ``not_evaluated``.
      Labels are never invented and a missing label is never a 0.
    """
    p_desc = f"Precision@{k} against supplied relevance labels."
    r_desc = f"Recall@{k} against supplied relevance labels."

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if relevant_ids is None:
        why = ("No ground-truth relevance labels were supplied; "
               "relevance is not guessed.")
        return {"precision": not_evaluated(p_desc, why),
                "recall": not_evaluated(r_desc, why)}
    if not isinstance(relevant_ids, list) or not isinstance(
            retrieved, (list, tuple)):
        raise ValueError("retrieved and relevant_ids must be lists.")

    labels = []
    for label in relevant_ids:
        if isinstance(label, str) and label.strip() \
                and label.strip().casefold() not in labels:
            labels.append(label.strip().casefold())
    if not labels:
        why = "The ground truth contains no relevant-paper labels."
        return {"precision": not_evaluated(p_desc, why),
                "recall": not_evaluated(r_desc, why)}

    top = _ranked_key_sets(retrieved)[:k]
    label_sets = [label_keys(label) for label in labels]
    hits = sum(1 for keys in top
               if keys and any(keys & ls for ls in label_sets))
    found = sum(1 for ls in label_sets
                if ls and any(ls & keys for keys in top))
    details = {"k": k, "items_in_top_k": len(top), "relevant_in_top_k": hits,
               "labels_total": len(labels), "labels_found_in_top_k": found,
               "labels_are_fixture_or_manual_not_universal_truth": True}

    precision = safe_ratio(hits, len(top))
    if precision is None:
        p = not_evaluated(p_desc, "Nothing was retrieved, so precision is "
                          "undefined (0 retrieved items).", details)
    else:
        p = make_metric(STATUS_CALCULATED, p_desc, precision, details=details)
    r = make_metric(STATUS_CALCULATED, r_desc, safe_ratio(found, len(labels)),
                    details=details)
    return {"precision": p, "recall": r}


def iterative_retrieval_summary(retrieval_result: Any) -> Dict[str, Any]:
    desc = "Phase 6 iterative retrieval outcome (reported, not re-run)."
    if not isinstance(retrieval_result, dict):
        return not_evaluated(desc, "No iterative retrieval result was "
                             "provided.")
    errors = retrieval_result.get("errors")
    return make_metric(STATUS_CALCULATED, desc,
                       retrieval_result.get("sufficient"), details={
        "stop_reason": retrieval_result.get("stop_reason"),
        "additional_attempts": retrieval_result.get("additional_attempts"),
        "new_papers_found": retrieval_result.get("new_papers_found"),
        "initial_paper_count": retrieval_result.get("initial_paper_count"),
        "final_paper_count": retrieval_result.get("final_paper_count"),
        "error_count": len(errors) if isinstance(errors, list) else 0,
    })


# ---------------------------------------------------------------------------
# Sub-question coverage (heuristic; reuses Phase 6 logic)
# ---------------------------------------------------------------------------

COVERAGE_METHOD = (
    "Phase 6 keyword rule: a paper 'covers' a sub-question when it has a "
    "usable abstract, its keyword-overlap relevance score reaches the "
    "threshold, and (when the plan has a main topic) the title/abstract "
    "contains at least one term specific to the sub-question.")
COVERAGE_LIMITS = (
    "This is a heuristic ESTIMATE based on word overlap. It does not show "
    "that any paper answers the sub-question and is not semantic "
    "verification. A shared topic word proves nothing by itself.")


def subquestion_coverage(user_query: Any, plan: Any, papers: Any
                         ) -> Dict[str, Any]:
    desc = "Estimated coverage of planner sub-questions by retrieved papers."
    sub_questions = _valid_subquestions(plan)
    if not sub_questions:
        return not_evaluated(
            desc, "The plan has no sub-questions, so coverage is not "
            "applicable (it is NOT reported as 100%).")
    if not isinstance(papers, list):
        return not_evaluated(desc, "No paper list was provided.")

    config = IterativeRetrievalConfig()
    topic = plan.get("main_topic") if isinstance(plan, dict) else None
    topic = topic if isinstance(topic, str) else None
    records = _dict_papers(papers)

    # One algorithm: coverage verdicts come from Phase 6's assess_evidence.
    assessment = assess_evidence(
        user_query if isinstance(user_query, str) else "",
        sub_questions, records, config, topic)
    covered = set(assessment["coverage"]["covered"])

    # Which papers triggered each verdict (same _covers predicate).
    from agents.iterative_retrieval_agent import _specific_terms
    candidates = [p for p in records
                  if is_usable_paper(p) and _real_abstract(p)]
    breakdown = []
    for question in sub_questions:
        specific = _specific_terms(question, topic)
        triggers = [p for p in candidates if _covers(
            p, question, specific, config.min_relevance_score)]
        breakdown.append({
            "sub_question": _short(question, 200),
            "covered": question in covered,
            "matching_paper_count": len(triggers),
            "matched_papers": [
                {"title": _short(p.get("title")),
                 "id": _short(str(p.get("id") or p.get("url") or ""), 80)}
                for p in triggers[:3]],
            "basis": ("keyword relevance >= threshold and a "
                      "sub-question-specific term in title/abstract"
                      if question in covered else
                      "no paper with an abstract met the keyword rule"),
        })

    return make_metric(
        STATUS_HEURISTIC, desc,
        safe_ratio(len(covered), len(sub_questions)), details={
            "method": COVERAGE_METHOD, "limitations": COVERAGE_LIMITS,
            "total_sub_questions": len(sub_questions),
            "covered_count": len(covered),
            "uncovered": [_short(q, 200) for q in sub_questions
                          if q not in covered],
            "per_sub_question": breakdown,
            "semantic_verification": False,
        })


# ---------------------------------------------------------------------------
# Citation and evidence traceability
# ---------------------------------------------------------------------------

TRACE_NOTE = (
    "A resolving paper ID shows the cited paper exists in the retrieved "
    "set, not that it supports the claim. An excerpt found in an abstract "
    "is a string match, not semantic proof of the claim.")


def _claims(analysis: Any) -> Optional[List[dict]]:
    if not isinstance(analysis, dict):
        return None
    claims = analysis.get("evidence_linked_claims")
    return [c for c in claims if isinstance(c, dict)] \
        if isinstance(claims, list) else None


def citation_traceability(analysis: Any, papers: Any) -> Dict[str, Any]:
    desc = "Whether analyzed claims cite papers that exist in the paper set."
    claims = _claims(analysis)
    if claims is None:
        return not_evaluated(desc, "No analysis output with "
                             "'evidence_linked_claims' was provided.")
    if not isinstance(papers, list):
        return not_evaluated(desc, "No paper list was provided to resolve "
                             "citations against.")
    if not claims:
        return not_evaluated(desc, "The analysis contains zero claims, so "
                             "there is nothing to trace.",
                             {"total_claims": 0})

    records = _dict_papers(papers)
    corpus = build_corpus_index(records)
    cluster_of = {}
    for number, group in enumerate(duplicate_clusters(records)):
        for index in group:
            cluster_of[_paper_id(records[index], index + 1)] = number

    totals = dict(claims_with_references=0, claims_with_valid_reference=0,
                  claims_with_unknown_ids=0, claims_missing_references=0,
                  claims_missing_excerpt=0, inference_claims=0,
                  excerpt_found_in_cited_abstract=0)
    refs_submitted = refs_resolved = 0
    cited_papers = set()
    unknown_claims = []

    for index, claim in enumerate(claims, start=1):
        good = [str(i).strip() for i in claim.get("paper_ids") or []
                if str(i).strip()]
        bad = [str(i).strip() for i in claim.get("invalid_paper_ids") or []
               if str(i).strip()]
        resolved = [i for i in good if i in corpus]
        unknown = bad + [i for i in good if i not in corpus]
        refs_submitted += len(resolved) + len(unknown)
        refs_resolved += len(resolved)
        cited_papers |= {cluster_of.get(i, i) for i in resolved}

        if resolved or unknown:
            totals["claims_with_references"] += 1
        else:
            totals["claims_missing_references"] += 1
        if resolved:
            totals["claims_with_valid_reference"] += 1
        if unknown:
            totals["claims_with_unknown_ids"] += 1
            if len(unknown_claims) < MAX_LISTED:
                unknown_claims.append({
                    "claim_id": str(claim.get("claim_id") or f"#{index}"),
                    "unknown_ids": [_short(i, 80) for i in unknown[:5]]})

        if claim.get("support_type") == "inference":
            totals["inference_claims"] += 1       # no quote expected
        else:
            excerpt = claim.get("evidence_excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip():
                totals["claims_missing_excerpt"] += 1
            elif any(verify_excerpt_against_abstract(
                    excerpt, corpus[i]["abstract"]) for i in resolved):
                totals["excerpt_found_in_cited_abstract"] += 1

    rate = safe_ratio(refs_resolved, refs_submitted)
    problems = (totals["claims_with_unknown_ids"]
                + totals["claims_missing_references"])
    details = dict(totals, total_claims=len(claims),
                   references_submitted=refs_submitted,
                   references_resolved=refs_resolved,
                   distinct_papers_cited=len(cited_papers),
                   claims_with_unknown_ids_sample=unknown_claims,
                   semantic_support_verified=False, note=TRACE_NOTE)
    return make_metric(
        STATUS_NEEDS_REVIEW if problems else STATUS_CALCULATED, desc, rate,
        reason=None if rate is not None else
        "No citations were submitted, so the resolution rate is undefined.",
        details=details)


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

VERIFICATION_NOTE = (
    "Statuses are a language model's judgement of abstracts, not proof "
    "that a claim is true.")


def claim_verification(analysis: Any, verification_error: Optional[str] = None,
                       verification_skipped: bool = False) -> Dict[str, Any]:
    """
    ``details['verification_state']`` is one of:
    no_analysis, no_claims, skipped, failed, unavailable, partial, completed.
    """
    desc = "Claim-verification outcomes reported by the Verification Agent."
    claims = _claims(analysis)
    if claims is None:
        return not_evaluated(desc, "No analysis output was provided.",
                             {"verification_state": "no_analysis"})
    if not claims:
        return not_evaluated(
            desc, "The analysis has zero claims. Nothing was verified; "
            "this is not a successful verification.",
            {"verification_state": "no_claims", "total_claims": 0})

    summary = analysis.get("verification_summary")
    summary = summary if isinstance(summary, dict) else {}
    records = [c.get("verification") for c in claims]
    has = [r for r in records if isinstance(r, dict) and r.get("status")]

    if verification_error:
        state, reason = "failed", "Verification raised an error."
    elif not has and verification_skipped:
        state, reason = "skipped", "Verification was skipped."
    elif not has:
        state, reason = "unavailable", ("Claims exist but no verification "
                                        "results were provided.")
    else:
        state, reason = summary.get("service_status") or "completed", None
        if state not in ("completed", "partial", "unavailable"):
            state = "completed"

    if not has:
        return not_evaluated(desc, reason, {
            "verification_state": state, "total_claims": len(claims),
            "claims_without_result": len(claims)})

    counts = {s: 0 for s in VALID_STATUSES}
    unrecognized: Dict[str, int] = {}
    flagged = missing_cited = 0
    for record in has:
        status = record["status"]
        if status in counts:
            counts[status] += 1
        else:
            key = _short(str(status), 40)
            unrecognized[key] = unrecognized.get(key, 0) + 1
        if record.get("needs_review") or status != "SUPPORTED":
            flagged += 1
        if record.get("missing_paper_ids"):
            missing_cited += 1

    without = len(claims) - len(has)
    flagged += without
    total = len(claims)
    ratios = {s.lower(): safe_ratio(n, total) for s, n in counts.items()}
    details = {
        "verification_state": state, "total_claims": total,
        "counts": {s.lower(): n for s, n in counts.items()},
        "ratios_of_total_claims": ratios,
        "unrecognized_statuses": unrecognized,
        "claims_without_result": without,
        "flagged_for_review": flagged,
        "claims_with_missing_or_invalid_citations": missing_cited,
        "verification_error_recorded": bool(verification_error),
        "service_note": _short(summary.get("service_note"), 200),
        "summary_matches_records": (
            not summary or summary.get("total_claims") == total),
        "note": VERIFICATION_NOTE,
    }
    needs_review = (flagged or unrecognized or state != "completed"
                    or details["summary_matches_records"] is False)
    return make_metric(
        STATUS_NEEDS_REVIEW if needs_review else STATUS_CALCULATED, desc,
        ratios["supported"], details=details)


# ---------------------------------------------------------------------------
# Report completeness
# ---------------------------------------------------------------------------

# (key, phrase that must appear in a heading) - from report_agent sections 1-10
REQUIRED_SECTIONS = (
    ("metadata", "title and research metadata"),
    ("executive_summary", "executive summary"),
    ("research_plan", "research plan"),
    ("literature_landscape", "literature landscape"),
    ("findings_synthesis", "comparative findings"),
    ("contradictions_and_gaps", "contradictions"),
    ("claims_and_evidence", "verified claims"),
    ("verification_summary", "verification summary"),
    ("conclusion", "conclusion"),
    ("references", "references"),
)
REPORT_NOTE = ("A structurally complete report is not a scientifically "
               "correct report. These checks look at structure only.")


def _headings(markdown: str) -> List[Tuple[int, str, int]]:
    found = []
    for number, line in enumerate(markdown.splitlines()):
        stripped = line.strip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 6 and stripped[level:level + 1] == " ":
            found.append((level, stripped[level:].strip().casefold(),
                          number))
    return found


def _section_span(markdown: str, headings: list, phrase: str
                  ) -> Optional[Tuple[int, int]]:
    """(first_line, end_line) of the body of the first heading containing
    ``phrase``, up to the next heading of the same or higher level."""
    total = len(markdown.splitlines())
    for position, (level, text, number) in enumerate(headings):
        if phrase in text:
            end = total
            for later_level, _, later in headings[position + 1:]:
                if later_level <= level:
                    end = later
                    break
            return number + 1, end
    return None


def report_completeness(report_markdown: Any, papers: Any = None,
                        analysis: Any = None,
                        verification_error: Optional[str] = None,
                        report_info: Any = None,
                        report_error: Optional[str] = None
                        ) -> Dict[str, Any]:
    import re
    from agents.report_agent import build_references

    desc = "Structural completeness of the generated report (Markdown)."
    info = report_info if isinstance(report_info, dict) else {}
    outputs = {
        "markdown_generated": bool(info.get("markdown_path")),
        "docx_generated": bool(info.get("docx_path")),
        "report_errors_recorded": len(info.get("errors") or [])
        if isinstance(info.get("errors"), list) else 0,
        "report_generation_failed": bool(report_error),
    }
    if report_error:
        return not_evaluated(
            desc, "Report generation failed, so there is no report to "
            "check.", {"outputs": outputs})
    if not isinstance(report_markdown, str) or not report_markdown.strip():
        return not_evaluated(
            desc, "No report text was available to check.",
            {"outputs": outputs})

    headings = _headings(report_markdown)
    present = {key: any(phrase in text for _, text, _ in headings)
               for key, phrase in REQUIRED_SECTIONS}
    missing = [k for k, ok in present.items() if not ok]
    problems: List[str] = []
    if missing:
        problems.append("missing sections: " + ", ".join(missing))

    # References
    entries = build_references(papers)["entries"] \
        if isinstance(papers, list) else None
    lines = report_markdown.splitlines()
    span = _section_span(report_markdown, headings, "references")
    ref_lines = lines[span[0]:span[1]] if span else []
    listed = sum(1 for line in ref_lines if re.match(r"\s*\d+\.\s", line))
    references = {"expected_entries": None if entries is None
                  else len(entries), "listed_entries": listed}
    if entries and listed == 0:
        problems.append("references expected but none listed")

    # Citations like [3] must point at an existing reference number.
    body = "\n".join(lines[:span[0]] + lines[span[1]:] if span else lines)
    cited = {int(n) for n in re.findall(r"\[(\d{1,4})\](?!\()", body)}
    limit = listed or (len(entries) if entries else 0)
    unresolved = sorted(n for n in cited if n < 1 or n > limit)
    references.update({"distinct_numeric_citations": len(cited),
                       "unresolved_citation_numbers": unresolved[:MAX_LISTED]})
    if unresolved:
        problems.append("citations that do not resolve to a reference")

    # Limitations / verification disclosure
    text = report_markdown.casefold()
    claims = _claims(analysis) or []
    summary = analysis.get("verification_summary") if isinstance(
        analysis, dict) else None
    degraded = bool(verification_error) or (
        isinstance(summary, dict)
        and summary.get("service_status", "completed") != "completed")
    disclosure = {
        "limitations_section_present": any(
            t == "limitations" for _, t, _ in headings),
        "verification_degraded": degraded,
        "degradation_disclosed": (
            None if not degraded else any(
                w in text for w in ("not verified", "failed", "unavailable",
                                    "partial", "unverifiable"))),
        "unverified_claims_disclosed": (
            None if not claims or any(
                isinstance(c.get("verification"), dict) for c in claims)
            else "not verified" in text),
    }
    if not disclosure["limitations_section_present"]:
        problems.append("no Limitations section")
    if disclosure["degradation_disclosed"] is False:
        problems.append("verification problems not disclosed")
    if disclosure["unverified_claims_disclosed"] is False:
        problems.append("unverified claims not disclosed")

    insufficient = isinstance(analysis, dict) and analysis.get(
        "status") == "insufficient_evidence"
    disclosure["insufficient_evidence"] = insufficient
    if insufficient and "insufficient" not in text:
        problems.append("insufficient evidence not disclosed")

    for flag, label in (("markdown_generated", "Markdown file not written"),
                        ("docx_generated", "Word file not written")):
        if info and not outputs[flag]:
            problems.append(label)

    return make_metric(
        STATUS_NEEDS_REVIEW if problems else STATUS_CALCULATED, desc,
        safe_ratio(len(present) - len(missing), len(present)), details={
            "sections_present": present, "missing_sections": missing,
            "references": references, "disclosure": disclosure,
            "outputs": outputs, "problems": problems,
            "note": REPORT_NOTE})
