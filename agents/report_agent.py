"""
Report Agent (Phase 5).

Turns the outputs of Phases 1-4 into a Markdown report and a Word report.

Design rules:

* Fully deterministic. No LLM call and no API key are needed: the report
  only organizes data the earlier agents already produced.
* One structured document (a list of "blocks") is built once, then rendered
  to Markdown and to DOCX, so both files always carry the same content.
* Nothing is invented. Missing data is reported as "Not available".
* Verification statuses come from the Verification Agent (Phase 4) and are
  never upgraded. A claim without a verification result is shown as
  NOT VERIFIED, never as verified.
* An exact excerpt match is shown as an excerpt match only. It does not
  prove that the abstract supports the whole claim.
"""

import datetime
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.retriever import normalize_paper_id, normalize_paper_title

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

NOT_AVAILABLE = "Not available"
NOT_VERIFIED = "NOT VERIFIED"

CONFIDENCE_NOTE = (
    "Confidence is the verifier model's assessment score. It is not a "
    "calibrated probability."
)
EXCERPT_NOTE = (
    "\"Excerpt matched\" means the quoted text was found in a cited "
    "abstract. It does not show that the abstract supports the whole claim."
)
NARRATIVE_NOTE = (
    "Only the structured evidence-linked claims were checked by the "
    "Verification Agent. The summary, themes, methodologies, "
    "contradictions and gaps in this report are the Analysis Agent's "
    "synthesis and have NOT been independently verified."
)

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Safe data access
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    """Return single-line, XML-safe text (control characters removed,
    all whitespace including newlines collapsed to single spaces)."""
    if value is None:
        return ""
    text = _CONTROL_RE.sub("", str(value))
    return " ".join(text.split())


def _split_urls(text: str) -> list:
    """Split text into [(is_url, chunk), ...]. Trailing punctuation and
    unbalanced closing parentheses are not part of a URL, but balanced
    parentheses inside it (as in some DOIs) are kept."""
    parts, last = [], 0
    for match in _URL_RE.finditer(text):
        url = match.group(0)
        while url and (url[-1] in ".,;:!?]" or (
                url[-1] == ")" and url.count(")") > url.count("("))):
            url = url[:-1]
        if len(url) <= len("https://"):
            continue
        end = match.start() + len(url)
        if match.start() > last:
            parts.append((False, text[last:match.start()]))
        parts.append((True, url))
        last = end
    if last < len(text):
        parts.append((False, text[last:]))
    return parts


def _or_na(value: Any) -> str:
    return _clean(value) or NOT_AVAILABLE


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dicts(value: Any) -> List[dict]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# References: one entry per distinct paper
# ---------------------------------------------------------------------------

def _analysis_paper_id(paper: dict, index: int) -> str:
    """Same ID rule as analysis_agent.build_corpus_index, so claim
    citations can be traced to the right reference."""
    raw_id = paper.get("id") or paper.get("url") or f"paper_{index}"
    return str(raw_id).strip() or f"paper_{index}"


def _usable_abstract(paper: dict) -> str:
    abstract = paper.get("abstract")
    if not isinstance(abstract, str):
        return ""
    abstract = abstract.strip()
    if abstract.casefold() == "abstract not available":
        return ""
    return abstract


def build_references(papers: Any) -> Dict[str, Any]:
    """
    Deduplicate retrieved papers (by DOI / OpenAlex ID / title, using the
    retriever's own normalizer) and number them.

    Returns {"entries": [...], "lookup": {alias -> number}}.
    """
    entries: List[dict] = []
    by_key: Dict[str, dict] = {}
    by_title: Dict[str, dict] = {}
    lookup: Dict[str, int] = {}

    for index, paper in enumerate(_as_list(papers), start=1):
        if not isinstance(paper, dict):
            continue

        key = normalize_paper_id(paper) or f"index:{index}"
        title_key = normalize_paper_title(paper) if isinstance(
            paper.get("title"), str) else ""
        # Same DOI/ID first; otherwise the same normalized title (a paper
        # can arrive once with a DOI and once with only an OpenAlex ID).
        entry = by_key.get(key) or (by_title.get(title_key)
                                    if title_key else None)

        if entry is None:
            entry = {
                "number": len(entries) + 1,
                "key": key,
                "title": "",
                "authors": [],
                "year": None,
                "doi": "",
                "url": "",
                "abstract": "",
                "aliases": [],
            }
            entries.append(entry)
        by_key.setdefault(key, entry)
        if title_key:
            by_title.setdefault(title_key, entry)

        # Fill gaps only; never overwrite data we already have.
        if not entry["title"] and paper.get("title"):
            entry["title"] = _clean(paper.get("title"))
        authors = [_clean(a) for a in _as_list(paper.get("authors")) if _clean(a)]
        if not entry["authors"] and authors:
            entry["authors"] = authors
        if entry["year"] in (None, "") and paper.get("year"):
            entry["year"] = paper.get("year")
        if not entry["doi"] and paper.get("doi"):
            entry["doi"] = _clean(paper.get("doi"))
        if not entry["url"] and paper.get("url"):
            entry["url"] = _clean(paper.get("url"))
        if not entry["abstract"]:
            entry["abstract"] = _usable_abstract(paper)

        for alias in (_analysis_paper_id(paper, index), paper.get("id"),
                      paper.get("url"), paper.get("doi")):
            alias = _clean(alias)
            if alias:
                if alias not in entry["aliases"]:
                    entry["aliases"].append(alias)
                lookup.setdefault(alias, entry["number"])

        # Also index the normalized key so Graph RAG items (no "id")
        # can be matched by DOI/URL/title.
        lookup.setdefault(key, entry["number"])

    return {"entries": entries, "lookup": lookup}


def _ref_label(paper_id: Any, refs: dict) -> str:
    """Citation label such as [3], or the raw ID if it is unresolved."""
    pid = _clean(paper_id)
    number = refs["lookup"].get(pid)
    return f"[{number}]" if number else f"(unresolved ID: {pid or 'none'})"


def _ref_labels(ids: Any, refs: dict) -> str:
    labels = [_ref_label(i, refs) for i in _as_list(ids)]
    return ", ".join(labels) if labels else "none"


def _short_authors(authors: List[str], limit: int = 3) -> str:
    if not authors:
        return NOT_AVAILABLE
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + f", et al. ({len(authors)} authors)"


def _paper_link(entry: dict) -> str:
    return entry["doi"] or entry["url"] or NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Claims and verification
# ---------------------------------------------------------------------------

def _claim_rows(analysis: dict, refs: dict) -> List[dict]:
    """Flatten claims into display rows combining Analysis (Phase 3) and
    Verification (Phase 4) data without altering either."""
    rows = []

    for index, claim in enumerate(
            _as_dicts(analysis.get("evidence_linked_claims")), start=1):
        record = claim.get("verification")
        has_record = isinstance(record, dict) and record.get("status")
        status = _clean(record["status"]) if has_record else NOT_VERIFIED
        record = record if has_record else {}

        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(
                confidence, bool):
            confidence_text = f"{float(confidence):.2f}"
        else:
            confidence_text = "—"

        verifier_excerpt = _clean(record.get("evidence_excerpt"))
        analysis_excerpt = _clean(claim.get("evidence_excerpt"))

        if verifier_excerpt:
            excerpt = verifier_excerpt
            matched = "Yes (found in the cited abstract by the verifier)"
        elif analysis_excerpt:
            excerpt = "[Analysis Agent excerpt] " + analysis_excerpt
            matched = (
                "Yes" if claim.get("excerpt_verified") is True
                else "No (not found in the cited abstracts)"
            )
        else:
            excerpt = "—"
            matched = "No excerpt"

        notes = []
        if has_record:
            if record.get("justification"):
                notes.append(_clean(record["justification"]))
            notes.extend(_clean(w) for w in _as_list(record.get("warnings")))
        else:
            notes.append("No verification result exists for this claim.")
        if claim.get("support_type") == "inference":
            notes.append("Labelled as an inference by the Analysis Agent.")

        needs_review = (
            not has_record
            or bool(record.get("needs_review"))
            or status != "SUPPORTED"
        )
        if needs_review:
            notes.append("MANUAL REVIEW NEEDED.")

        ids = _as_list(claim.get("paper_ids")) + _as_list(
            claim.get("invalid_paper_ids"))

        rows.append({
            "id": _clean(claim.get("claim_id")) or f"CLM-{index:03d}",
            "claim": _or_na(claim.get("claim")),
            "status": status,
            "confidence": confidence_text,
            "papers": _ref_labels(ids, refs),
            "excerpt": excerpt,
            "matched": matched,
            "notes": " ".join(n for n in notes if n),
            "needs_review": needs_review,
            "has_record": bool(has_record),
        })

    return rows


def _status_counts(rows: List[dict]) -> Counter:
    return Counter(row["status"] for row in rows)


def _verification_one_liner(rows: List[dict], summary: dict,
                            verification_error: Optional[str]) -> str:
    if not rows:
        return "No claims were available to verify."
    counts = _status_counts(rows)
    if not any(r["has_record"] for r in rows):
        reason = (f" Reason: {_clean(verification_error)}"
                  if verification_error else "")
        return (f"Verification did NOT run. All {len(rows)} claim(s) are "
                f"unverified.{reason}")
    parts = [f"{counts[s]} {s}" for s in
             ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
              "UNVERIFIABLE") if counts.get(s)]
    for status, n in counts.items():
        if status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                          "UNVERIFIABLE"):
            parts.append(f"{n} {status}")
    flagged = sum(1 for r in rows if r["needs_review"])
    return (f"{len(rows)} claim(s): " + ", ".join(parts)
            + f". {flagged} flagged for manual review. "
            "Narrative sections are not verified.")


# ---------------------------------------------------------------------------
# Building the report (a list of structured blocks)
# ---------------------------------------------------------------------------

Block = dict


def _h(level: int, text: str) -> Block:
    return {"type": "heading", "level": level, "text": text}


def _p(text: str) -> Block:
    return {"type": "paragraph", "text": text}


def _note(text: str) -> Block:
    return {"type": "note", "text": text}


def _bullets(items: list) -> Block:
    return {"type": "bullets", "items": items}


def _numbered(items: list) -> Block:
    return {"type": "numbered", "items": items}


def _table(headers: List[str], rows: List[List[str]], wide: bool = False,
           widths: Optional[List[float]] = None) -> Block:
    return {"type": "table", "headers": headers, "rows": rows,
            "wide": wide, "widths": widths}


def _title_from_question(question: str) -> str:
    text = _clean(question).rstrip("?.! ")
    if len(text) > 110:
        text = text[:110].rsplit(" ", 1)[0] + "…"
    return f"Research Report: {text[:1].upper() + text[1:]}"


def build_report_blocks(
    user_query: str,
    plan: Optional[dict],
    papers: Optional[list],
    graph_context: Optional[list],
    analysis: Optional[dict],
    verification_error: Optional[str] = None,
    generated_at: Optional[datetime.date] = None,
) -> List[Block]:
    """Build the full report as a list of blocks (sections 1-10)."""
    if not isinstance(user_query, str) or not _clean(user_query):
        raise ValueError("Research question cannot be empty.")

    plan = _as_dict(plan)
    analysis = _as_dict(analysis)
    papers = _as_list(papers)
    graph_context = _as_dicts(graph_context)
    generated_at = generated_at or datetime.date.today()

    refs = build_references(papers)
    entries = refs["entries"]
    rows = _claim_rows(analysis, refs)
    v_summary = _as_dict(analysis.get("verification_summary"))
    v_ran = any(r["has_record"] for r in rows)

    themes = _as_dicts(analysis.get("key_themes_and_consensus"))
    contradictions = _as_dicts(
        analysis.get("divergent_findings_and_contradictions"))
    methods = _as_dicts(analysis.get("methodologies_and_benchmarks"))
    gaps = _as_dicts(analysis.get("limitations_and_research_gaps"))
    insufficient = analysis.get("status") == "insufficient_evidence"
    with_abstract = sum(1 for e in entries if e["abstract"])

    blocks: List[Block] = [
        {"type": "title", "text": _title_from_question(user_query)}]

    # ---- 1. Metadata ----------------------------------------------------
    blocks += [
        _h(1, "1. Title and Research Metadata"),
        _table(["Item", "Value"], [
            ["Research question", _clean(user_query)],
            ["Report generated", generated_at.isoformat()],
            ["Retrieved papers (unique)", str(len(entries))],
            ["Papers with abstracts", str(with_abstract)],
            ["Analyzed claims", str(len(rows))],
            ["Verification summary",
             _verification_one_liner(rows, v_summary, verification_error)],
        ], widths=[1.8, 4.7]),
        _note("This report was assembled automatically from the outputs of "
              "the AutoResearch pipeline. It is based on paper abstracts "
              "only and is preliminary."),
    ]

    # ---- 2. Executive summary --------------------------------------------
    blocks.append(_h(1, "2. Executive Summary"))
    objective = _clean(plan.get("research_objective"))
    if objective:
        blocks.append(_p("Research objective: " + objective))

    if insufficient:
        blocks.append(_p(
            "The Analysis Agent reported insufficient evidence: "
            + _or_na(analysis.get("summary"))))
    elif not analysis:
        blocks.append(_p("No literature analysis was available."))
    else:
        blocks.append(_p(
            "Analysis Agent summary: " + _or_na(analysis.get("summary"))))
        facts = [
            f"{len(entries)} unique paper(s) were retrieved, "
            f"{with_abstract} with usable abstracts.",
        ]
        if themes:
            facts.append("Themes identified: " + "; ".join(
                _or_na(t.get("theme")) for t in themes) + ".")
        else:
            facts.append("No themes were identified.")
        facts.append(
            f"{len(contradictions)} contradiction(s) and "
            f"{len(gaps)} limitation(s) or research gap(s) were reported "
            "by the Analysis Agent.")
        blocks.append(_bullets(facts))

    status_text = (
        "Findings are preliminary. " + _verification_one_liner(
            rows, v_summary, verification_error))
    blocks.append(_p(status_text))
    blocks.append(_note(NARRATIVE_NOTE))

    # ---- 3. Plan and sub-questions ----------------------------------------
    blocks.append(_h(1, "3. Research Plan and Sub-Questions"))
    if _clean(plan.get("main_topic")):
        blocks.append(_p("Main topic: " + _clean(plan["main_topic"])))
    steps = [_clean(s) for s in _as_list(plan.get("research_plan"))
             if _clean(s)]
    if steps:
        blocks += [_p("Research plan:"), _numbered(steps)]
    else:
        blocks.append(_p("No research plan was available."))

    sub_questions = [_clean(q) for q in _as_list(plan.get("sub_questions"))
                     if _clean(q)]
    if sub_questions:
        counts = Counter()
        recorded = False
        for paper in papers:
            if isinstance(paper, dict) and isinstance(
                    paper.get("source_questions"), list):
                recorded = True
                for question in paper["source_questions"]:
                    counts[_clean(question)] += 1
        blocks.append(_p(
            "The Planner split the research question into the "
            "sub-questions below. Retrieval searched each sub-question "
            "separately and merged the results"
            + (", so the second column shows how many of the unique "
               "retrieved papers matched each one." if recorded
               else ". Per-question paper counts were not recorded.")))
        blocks.append(_table(
            ["#", "Sub-question"] + (["Papers matched"] if recorded else []),
            [[str(i), q] + ([str(counts.get(q, 0))] if recorded else [])
             for i, q in enumerate(sub_questions, start=1)],
            widths=[0.4, 4.9, 1.2] if recorded else [0.4, 6.1]))
    else:
        blocks.append(_p("No sub-questions were available."))

    # ---- 4. Literature landscape and graph --------------------------------
    blocks.append(_h(1, "4. Literature Landscape and Paper Graph"))
    if not entries:
        blocks.append(_p("No papers were retrieved."))
    else:
        years = [int(e["year"]) for e in entries
                 if str(e["year"]).isdigit()]
        span = (f"Publication years range from {min(years)} to {max(years)}."
                if years else "Publication years are not available.")
        blocks.append(_p(
            f"{len(entries)} unique paper(s) were retrieved; "
            f"{with_abstract} have a usable abstract and "
            f"{len(entries) - with_abstract} do not. {span}"))
        blocks.append(_table(
            ["Ref", "Title", "Year", "Authors", "DOI / URL"],
            [[f"[{e['number']}]", _or_na(e["title"]), _or_na(e["year"]),
              _short_authors(e["authors"]), _paper_link(e)]
             for e in entries],
            widths=[0.4, 2.2, 0.85, 1.4, 1.7]))

    blocks.append(_h(2, "Graph RAG results"))
    if not graph_context:
        blocks.append(_p(
            "Graph RAG output is unavailable or empty, so no paper "
            "relationships are reported."))
    else:
        blocks.append(_p(
            "Graph RAG links papers through shared concepts extracted from "
            "their text; these are not citation links. Scores are the "
            "heuristic values returned by the Graph RAG stage. The "
            f"{len(graph_context)} paper(s) it selected as most relevant "
            "are listed below."))
        g_rows = []
        for item in graph_context:
            number = refs["lookup"].get(normalize_paper_id(item) or "")
            label = f"[{number}] " if number else ""
            related = [
                f"{_or_na(r.get('title'))} "
                f"({_clean(r.get('shared_concepts', 0))} shared concepts)"
                for r in _as_dicts(item.get("related_papers"))]
            concepts = ", ".join(
                _clean(c) for c in _as_list(item.get("matched_concepts"))
                if _clean(c))
            score = item.get("combined_graph_score")
            g_rows.append([
                label + _or_na(item.get("title")),
                f"{score:.2f}" if isinstance(score, (int, float))
                and not isinstance(score, bool) else NOT_AVAILABLE,
                concepts or NOT_AVAILABLE,
                "; ".join(related) or "None reported",
            ])
        blocks.append(_table(
            ["Paper", "Combined graph score", "Matched concepts",
             "Related papers"], g_rows, widths=[2.0, 0.8, 1.6, 2.1]))

    # ---- 5. Comparative findings -------------------------------------------
    blocks.append(_h(1, "5. Comparative Findings and Synthesis"))
    blocks.append(_note(
        "Everything in this section is the Analysis Agent's synthesis of "
        "the abstracts. Only the claims in Section 7 were independently "
        "checked."))
    if not (themes or methods):
        blocks.append(_p("No themes or methodologies were reported."))
    if themes:
        blocks.append(_h(2, "Key themes and consensus"))
        for i, theme in enumerate(themes, start=1):
            blocks.append(_h(3, f"{i}. {_or_na(theme.get('theme'))}"))
            blocks.append(_p(_or_na(theme.get("description"))))
            blocks.append(_p(
                "Cited papers: "
                + _ref_labels(theme.get("supporting_paper_ids"), refs)
                + _invalid_note(theme)))
    if methods:
        blocks.append(_h(2, "Methodologies and benchmarks"))
        blocks.append(_table(
            ["Approach or dataset", "Description", "Papers"],
            [[_or_na(m.get("approach_or_dataset")),
              _or_na(m.get("description")),
              _ref_labels(m.get("papers_using"), refs) + _invalid_note(m)]
             for m in methods],
            widths=[1.7, 3.6, 1.2]))

    # ---- 6. Contradictions and gaps ----------------------------------------
    blocks.append(_h(1, "6. Contradictions and Open Research Gaps"))
    blocks.append(_h(2, "Contradictions"))
    if not contradictions:
        blocks.append(_p(
            "The Analysis Agent did not identify a contradiction between "
            "the retrieved papers, so none is reported."))
    for i, c in enumerate(contradictions, start=1):
        blocks.append(_h(3, f"{i}. {_or_na(c.get('topic'))}"))
        blocks.append(_bullets([
            ("Finding A " + _ref_label(c.get("paper_a_id"), refs),
             _or_na(c.get("finding_a"))),
            ("Finding B " + _ref_label(c.get("paper_b_id"), refs),
             _or_na(c.get("finding_b"))),
            ("Analysis", _or_na(c.get("explanation"))),
        ]))

    blocks.append(_h(2, "Limitations and research gaps"))
    if not gaps:
        blocks.append(_p("No limitations or research gaps were reported."))
    else:
        blocks.append(_note(
            "\"Reported\" means the Analysis Agent attributed the "
            "limitation to a cited abstract; this attribution was not "
            "independently verified. \"Inferred\" means the model derived "
            "the gap itself or gave no valid citation."))
        g_rows = []
        for gap in gaps:
            ids = _as_list(gap.get("reported_by"))
            reported = (gap.get("type") == "reported_limitation"
                        and bool(ids) and not gap.get("invalid_paper_ids"))
            g_rows.append([
                _or_na(gap.get("gap_or_limitation")),
                "Reported in cited abstract(s)" if reported
                else "Inferred by the model",
                _ref_labels(ids, refs) + _invalid_note(gap),
            ])
        blocks.append(_table(["Limitation or gap", "Basis", "Papers"],
                             g_rows, widths=[3.6, 1.7, 1.2]))

    # ---- 7. Claims and evidence matrix --------------------------------------
    matrix_heading = _h(1, "7. Verified Claims and Evidence Matrix")
    matrix_heading["landscape"] = True  # wide table: landscape pages
    blocks.append(matrix_heading)
    if not rows:
        blocks.append(_p("No evidence-linked claims were available."))
    else:
        blocks.append(_note(
            "Status comes from the Verification Agent. A claim without a "
            "verification result is shown as NOT VERIFIED. "
            + CONFIDENCE_NOTE + " " + EXCERPT_NOTE))
        blocks.append(_table(
            ["Claim ID", "Claim", "Status", "Confidence",
             "Cited papers", "Evidence excerpt", "Excerpt matched",
             "Verification note"],
            [[r["id"], r["claim"], r["status"], r["confidence"],
              r["papers"], r["excerpt"], r["matched"], r["notes"]]
             for r in rows],
            wide=True,
            widths=[0.7, 1.8, 1.65, 0.8, 0.7, 1.9, 1.0, 2.0]))

    # ---- 8. Verification summary and limitations -----------------------------
    blocks.append(_h(1, "8. Verification Summary and Limitations"))
    counts = _status_counts(rows)
    summary_rows = [["Total claims reviewed", str(len(rows))]]
    for status in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                   "UNVERIFIABLE"):
        summary_rows.append([status, str(counts.get(status, 0))])
    for status, n in counts.items():
        if status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                          "UNVERIFIABLE", NOT_VERIFIED):
            summary_rows.append([status, str(n)])
    summary_rows.append([NOT_VERIFIED, str(counts.get(NOT_VERIFIED, 0))])
    flagged = [r["id"] for r in rows if r["needs_review"]]
    summary_rows.append(["Flagged for manual review", str(len(flagged))])
    blocks.append(_table(["Measure", "Count"], summary_rows,
                         widths=[3.0, 1.0]))
    if flagged:
        blocks.append(_p("Claims needing manual review: "
                         + ", ".join(flagged) + "."))

    service_status = _clean(v_summary.get("service_status"))
    if verification_error:
        blocks.append(_p(
            "Verification service status: FAILED. "
            + _clean(verification_error)
            + " Claims without a verification result are NOT VERIFIED."))
    elif not v_ran and rows:
        blocks.append(_p(
            "Verification service status: no verification results were "
            "provided. Claims are NOT VERIFIED."))
    elif service_status and service_status != "completed":
        blocks.append(_p(
            f"Verification service status: {service_status.upper()}. "
            + _clean(v_summary.get("service_note"))))
    elif v_ran:
        blocks.append(_p("Verification service status: completed."))

    limitations = [
        "Only titles and abstracts were available, not full texts. Claims "
        "that depend on details found only in full texts cannot be "
        "confirmed.",
        f"{len(entries) - with_abstract} of {len(entries)} retrieved "
        "paper(s) have no usable abstract and cannot serve as evidence.",
        "Retrieval relies on keyword search of OpenAlex (with a Crossref "
        "fallback) and a relevance filter, so relevant papers may be "
        "missing.",
        NARRATIVE_NOTE,
        "The verifier is a language model. " + CONFIDENCE_NOTE,
        EXCERPT_NOTE,
    ]
    blocks += [_h(2, "Limitations"), _bullets(limitations)]

    # ---- 9. Conclusion and future research -----------------------------------
    blocks.append(_h(1, "9. Conclusion and Future Research"))
    if insufficient or not analysis:
        blocks.append(_p(
            "The available evidence was insufficient for a literature "
            "synthesis, so no findings are concluded."))
    else:
        supported = counts.get("SUPPORTED", 0)
        blocks.append(_p(
            f"Based on the abstracts of {with_abstract} paper(s), the "
            f"Analysis Agent reported {len(themes)} theme(s). "
            + (f"The Verification Agent rated {supported} of {len(rows)} "
               "claim(s) as SUPPORTED"
               + ("; the remainder require manual review. "
                  if supported < len(rows) else ". ")
               if v_ran else
               "The claims have not been verified. ")
            + "These conclusions are preliminary."))

    future = []
    for gap in gaps:
        text = _clean(gap.get("gap_or_limitation"))
        if text:
            future.append("Address: " + text)
    for c in contradictions:
        topic = _clean(c.get("topic"))
        if topic:
            future.append("Resolve the disagreement on: " + topic)
    if future:
        blocks += [_p("Directions drawn from the identified gaps and "
                      "contradictions:"), _bullets(future)]
    else:
        blocks.append(_p(
            "The Analysis Agent identified no specific gaps or "
            "contradictions to build future directions on."))
    follow_ups = []
    if flagged:
        follow_ups.append(
            "Manually review the flagged claims in Section 7.")
    if len(entries) - with_abstract:
        follow_ups.append(
            "Obtain full texts or abstracts for papers that lack them.")
    if follow_ups:
        blocks += [_p("Process follow-ups:"), _bullets(follow_ups)]

    # ---- 10. References -------------------------------------------------------
    blocks.append(_h(1, "10. References and Source Index"))
    if not entries:
        blocks.append(_p("No papers were retrieved."))
    else:
        blocks.append(_p(
            "Duplicates were merged by DOI, OpenAlex ID or title. The "
            "paper ID is the identifier used by the pipeline."))
        blocks.append(_numbered([
            _reference_text(e) for e in entries]))

    return blocks


def _invalid_note(item: dict) -> str:
    invalid = [_clean(i) for i in _as_list(item.get("invalid_paper_ids"))]
    if invalid:
        return " (invalid citation(s): " + ", ".join(invalid) + ")"
    return ""


def _reference_text(entry: dict) -> str:
    authors = ", ".join(entry["authors"]) if entry["authors"] \
        else NOT_AVAILABLE
    return (
        f"{authors} ({_or_na(entry['year'])}). {_or_na(entry['title'])}. "
        f"DOI: {entry['doi'] or NOT_AVAILABLE}. "
        f"URL: {entry['url'] or NOT_AVAILABLE}. "
        f"Paper ID: {entry['aliases'][0] if entry['aliases'] else NOT_AVAILABLE}. "
        f"Abstract available: {'yes' if entry['abstract'] else 'no'}."
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_escape(text: str) -> str:
    """Escape characters that would change Markdown meaning, while keeping
    ordinary text such as [1], PARTIALLY_SUPPORTED and C# readable."""
    text = re.sub(r"([\\`*<>|~])", r"\\\1", text)
    # Underscores only matter at word edges (intra-word ones are literal).
    text = re.sub(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])", r"\\_", text)
    # "[text](url)" in model output must not become a live link.
    text = text.replace("](", "\\](")
    return re.sub(r"^(#+)", r"\\\1", text)


def _md_inline(text: Any) -> str:
    """Escape Markdown syntax but turn bare URLs into valid links."""
    out = []
    for is_url, chunk in _split_urls(_clean(text)):
        if is_url:
            target = chunk.replace("(", "%28").replace(")", "%29")
            out.append("[" + chunk.replace("_", "\\_") + "](" + target + ")")
        else:
            out.append(_md_escape(chunk))
    return "".join(out)


def _md_item(item: Any) -> str:
    if isinstance(item, tuple):
        return f"**{_md_inline(item[0])}:** {_md_inline(item[1])}"
    return _md_inline(item)


def render_markdown(blocks: List[Block]) -> str:
    """Render blocks as standard Markdown."""
    lines: List[str] = []

    for block in blocks:
        kind = block["type"]

        if kind == "title":
            lines += [f"# {_md_inline(block['text'])}", ""]
        elif kind == "heading":
            level = min(block["level"] + 1, 6)
            lines += [f"{'#' * level} {_md_inline(block['text'])}", ""]
        elif kind == "paragraph":
            lines += [_md_inline(block["text"]), ""]
        elif kind == "note":
            lines += [f"> {_md_inline(block['text'])}", ""]
        elif kind == "bullets":
            lines += [f"- {_md_item(i)}" for i in block["items"]] + [""]
        elif kind == "numbered":
            lines += [f"{n}. {_md_item(i)}"
                      for n, i in enumerate(block["items"], start=1)] + [""]
        elif kind == "table":
            lines.append("| " + " | ".join(
                _md_inline(h) for h in block["headers"]) + " |")
            lines.append("| " + " | ".join("---" for _ in block["headers"])
                         + " |")
            for row in block["rows"]:
                lines.append("| " + " | ".join(
                    _md_inline(c) for c in row) + " |")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_report_markdown(
    user_query: str,
    plan: Optional[dict],
    papers: Optional[list],
    graph_context: Optional[list],
    analysis: Optional[dict],
    verification_error: Optional[str] = None,
    generated_at: Optional[datetime.date] = None,
) -> str:
    """Return the complete report as a Markdown string."""
    return render_markdown(build_report_blocks(
        user_query, plan, papers, graph_context, analysis,
        verification_error, generated_at))


# ---------------------------------------------------------------------------
# DOCX rendering (python-docx)
# ---------------------------------------------------------------------------

def _add_hyperlink(paragraph, url: str, text: str, size=None) -> None:
    """Append a clickable hyperlink run to a python-docx paragraph."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    rel_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/"
        "relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel_id)

    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(color)
    props.append(underline)
    if size is not None:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        props.append(sz)
    run.append(props)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def _add_inline(paragraph, text: Any, bold=False, italic=False,
                size=None) -> None:
    """Add text to a paragraph, turning bare URLs into hyperlinks."""
    from docx.shared import Pt

    for is_url, chunk in _split_urls(_clean(text)):
        if is_url:
            _add_hyperlink(paragraph, chunk, chunk, size)
        elif chunk:
            run = paragraph.add_run(chunk)
            run.bold = bold
            run.italic = italic
            if size:
                run.font.size = Pt(size)


def _add_page_number_footer(section) -> None:
    """Centered 'Page N' footer using a PAGE field."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    footer = section.footer
    footer.is_linked_to_previous = False
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run("Page ")

    run = paragraph.add_run()
    for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)


def _shade(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    props.append(shading)


def _repeat_header(row) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    props = row._tr.get_or_add_trPr()
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    props.append(element)


def _no_row_split(row) -> None:
    from docx.oxml import OxmlElement

    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def _set_orientation(section, landscape: bool) -> None:
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches

    wanted_w, wanted_h = (11, 8.5) if landscape else (8.5, 11)
    section.orientation = (WD_ORIENT.LANDSCAPE if landscape
                           else WD_ORIENT.PORTRAIT)
    section.page_width = Inches(wanted_w)
    section.page_height = Inches(wanted_h)
    margin = Inches(0.75 if landscape else 1)
    section.left_margin = section.right_margin = margin
    section.top_margin = section.bottom_margin = Inches(1 if not landscape
                                                        else 0.8)


def _add_table(document, block: dict, usable_inches: float) -> None:
    from docx.shared import Inches, Pt

    headers = block["headers"]
    font_size = 8 if block["wide"] else 9

    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False

    widths = block.get("widths")
    if widths and len(widths) == len(headers):
        total = sum(widths)
        widths = [w * usable_inches / total for w in widths]
    else:
        widths = [usable_inches / len(headers)] * len(headers)

    for column, width in zip(table.columns, widths):
        column.width = Inches(width)

    header_row = table.rows[0]
    _repeat_header(header_row)
    for cell, text, width in zip(header_row.cells, headers, widths):
        cell.width = Inches(width)
        _shade(cell, "D9E2F3")
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
        _add_inline(cell.paragraphs[0], text, bold=True, size=font_size)

    for values in block["rows"]:
        row = table.add_row()
        _no_row_split(row)
        for cell, text, width in zip(row.cells, values, widths):
            cell.width = Inches(width)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            _add_inline(cell.paragraphs[0], text, size=font_size)

    document.add_paragraph().paragraph_format.space_after = Pt(4)


def _add_item(paragraph, item: Any) -> None:
    if isinstance(item, tuple):
        _add_inline(paragraph, item[0] + ":", bold=True)
        paragraph.add_run(" ")
        _add_inline(paragraph, item[1])
    else:
        _add_inline(paragraph, item)


def save_report_docx(blocks: List[Block], path) -> Path:
    """Render report blocks into a formatted Word document at ``path``."""
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.shared import Inches, Pt, RGBColor

    document = Document()

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for name, size in (("Title", 22), ("Heading 1", 16),
                       ("Heading 2", 13), ("Heading 3", 11.5)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)

    section = document.sections[0]
    _set_orientation(section, landscape=False)
    _add_page_number_footer(section)
    current_landscape = False

    def usable_width() -> float:
        return 9.5 if current_landscape else 6.5

    for block in blocks:
        kind = block["type"]

        # A level-1 heading flagged "landscape" starts landscape pages;
        # the next unflagged level-1 heading returns to portrait.
        if kind == "heading" and block["level"] == 1:
            wants_landscape = bool(block.get("landscape"))
            if wants_landscape != current_landscape:
                new_section = document.add_section(WD_SECTION.NEW_PAGE)
                _set_orientation(new_section, wants_landscape)
                _add_page_number_footer(new_section)
                current_landscape = wants_landscape

        if kind == "title":
            document.add_paragraph(_clean(block["text"]), style="Title")
        elif kind == "heading":
            document.add_heading(_clean(block["text"]),
                                 level=block["level"])
        elif kind == "paragraph":
            _add_inline(document.add_paragraph(), block["text"])
        elif kind == "note":
            _add_inline(document.add_paragraph(), block["text"],
                        italic=True, size=10)
        elif kind == "bullets":
            for item in block["items"]:
                _add_item(document.add_paragraph(style="List Bullet"), item)
        elif kind == "numbered":
            for number, item in enumerate(block["items"], start=1):
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.left_indent = Inches(0.35)
                paragraph.paragraph_format.first_line_indent = Inches(-0.35)
                paragraph.paragraph_format.space_after = Pt(4)
                paragraph.add_run(f"{number}.\t")
                _add_item(paragraph, item)
        elif kind == "table":
            _add_table(document, block, usable_width())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def generate_research_report(
    user_query: str,
    plan: Optional[dict],
    papers: Optional[list],
    graph_context: Optional[list],
    analysis: Optional[dict],
    output_dir: Optional[Any] = None,
    verification_error: Optional[str] = None,
    generated_at: Optional[datetime.date] = None,
) -> Dict[str, Any]:
    """
    Build the report and save ``report.md`` and ``report.docx``.

    ``analysis`` should be the output of ``verify_analysis`` when
    verification ran. If verification failed, pass the original analysis
    and the error text in ``verification_error``; the report then states
    that the claims are NOT verified.

    Returns {"markdown_path", "docx_path", "errors"}. A path is set only
    if that file was really written. Invalid input raises ValueError.
    """
    blocks = build_report_blocks(
        user_query, plan, papers, graph_context, analysis,
        verification_error, generated_at)

    directory = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "markdown_path": None, "docx_path": None, "errors": []}

    markdown_path = directory / "report.md"
    try:
        _atomic_write_text(markdown_path, render_markdown(blocks))
        result["markdown_path"] = str(markdown_path)
    except Exception as exc:
        result["errors"].append(
            f"Markdown report was not saved: {type(exc).__name__}: {exc}")

    docx_path = directory / "report.docx"
    temp_docx = directory / "report.docx.tmp"
    try:
        save_report_docx(blocks, temp_docx)
        os.replace(temp_docx, docx_path)
        result["docx_path"] = str(docx_path)
    except Exception as exc:
        result["errors"].append(
            f"Word report was not saved: {type(exc).__name__}: {exc}")
        try:
            temp_docx.unlink()
        except OSError:
            pass

    return result