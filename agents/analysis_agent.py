import json
import os
import re
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.6-flash"
_client = None

REQUIRED_LIST_SECTIONS = (
    "key_themes_and_consensus",
    "divergent_findings_and_contradictions",
    "methodologies_and_benchmarks",
    "limitations_and_research_gaps",
    "evidence_linked_claims",
)


def get_gemini_client():
    """Initialize Gemini only when a live request is needed."""
    global _client

    if _client is not None:
        return _client

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to the .env file "
            "in the AutoResearch project root."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def normalize_tokens(text):
    """Normalize text while preserving the order of words."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.casefold())


def verify_excerpt_against_abstract(excerpt, abstract):
    """
    Verify a complete excerpt as a contiguous sequence of normalized
    tokens in the abstract. Punctuation and capitalization may differ.

    Partial word overlap is deliberately insufficient.
    """
    excerpt_tokens = normalize_tokens(excerpt)
    abstract_tokens = normalize_tokens(abstract)

    if not excerpt_tokens or not abstract_tokens:
        return False

    width = len(excerpt_tokens)

    return any(
        abstract_tokens[index:index + width] == excerpt_tokens
        for index in range(len(abstract_tokens) - width + 1)
    )


def build_corpus_index(papers):
    """Index papers by their stable IDs while preserving source metadata."""
    corpus = {}

    if not isinstance(papers, list):
        return corpus

    for index, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            continue

        raw_id = paper.get("id") or paper.get("url")
        if not raw_id:
            raw_id = f"paper_{index}"

        paper_id = str(raw_id).strip()
        if not paper_id:
            paper_id = f"paper_{index}"

        abstract = paper.get("abstract", "")
        if not isinstance(abstract, str):
            abstract = ""

        if abstract.strip().casefold() == "abstract not available":
            abstract = ""

        authors = paper.get("authors", [])
        if not isinstance(authors, list):
            authors = []

        corpus[paper_id] = {
            "id": paper_id,
            "title": str(paper.get("title") or "Title not available"),
            "abstract": abstract.strip(),
            "authors": authors,
            "year": paper.get("year"),
            "url": str(paper.get("url") or paper_id),
        }

    return corpus


def build_insufficient_evidence_response(reason):
    """Return a consistent report when synthesis cannot be supported."""
    return {
        "status": "insufficient_evidence",
        "summary": reason,
        "key_themes_and_consensus": [],
        "divergent_findings_and_contradictions": [],
        "methodologies_and_benchmarks": [],
        "limitations_and_research_gaps": [],
        "evidence_linked_claims": [],
        "metadata": {
            "total_papers_analyzed": 0,
            "total_claims": 0,
            "verified_source_supported_claims": 0,
            "inference_claims": 0,
            "invalid_reference_count": 0,
            "verified_excerpt_count": 0,
            "papers_with_usable_abstracts": 0,
        },
    }


def extract_json_text(text):
    """Extract a JSON object from a model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Analysis Agent returned an empty response.")

    cleaned = text.strip()

    # Accept a response enclosed in Markdown code fences.
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"\A```(?:json)?\s*",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```\s*\Z", "", cleaned, count=1)

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Handle explanatory text surrounding one JSON object.
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char == "{":
            try:
                _, end = decoder.raw_decode(cleaned[index:])
                candidate = cleaned[index:index + end]
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                continue

    raise ValueError("Analysis Agent returned malformed JSON.")


def _require_list_of_objects(data, section):
    """Enforce the common section schema."""
    value = data.get(section)

    if not isinstance(value, list):
        raise ValueError(
            f"Analysis section '{section}' must be a list."
        )

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Item {index + 1} in '{section}' must be a JSON object."
            )


def validate_analysis_schema(data):
    """Validate section and field types before processing model output."""
    if not isinstance(data, dict):
        raise ValueError("Analysis response must be a JSON object.")

    if not isinstance(data.get("summary"), str):
        raise ValueError("Analysis field 'summary' must be a string.")

    for section in REQUIRED_LIST_SECTIONS:
        _require_list_of_objects(data, section)

    # Validate citation-list fields in each section.
    list_fields = {
        "key_themes_and_consensus": "supporting_paper_ids",
        "methodologies_and_benchmarks": "papers_using",
        "limitations_and_research_gaps": "reported_by",
        "evidence_linked_claims": "paper_ids",
    }

    for section, field in list_fields.items():
        for index, item in enumerate(data[section]):
            value = item.get(field, [])
            if not isinstance(value, list) or not all(
                isinstance(entry, (str, int)) for entry in value
            ):
                raise ValueError(
                    f"'{section}[{index}].{field}' must be a list "
                    "of paper ID strings."
                )

    for index, item in enumerate(
        data["divergent_findings_and_contradictions"]
    ):
        for field in ("paper_a_id", "paper_b_id"):
            value = item.get(field)
            if value is not None and not isinstance(value, (str, int)):
                raise ValueError(
                    f"'divergent_findings_and_contradictions"
                    f"[{index}].{field}' must be a paper ID string."
                )

    for index, item in enumerate(data["evidence_linked_claims"]):
        for field in ("claim", "support_type", "evidence_excerpt"):
            value = item.get(field, "")
            if not isinstance(value, str):
                raise ValueError(
                    f"'evidence_linked_claims[{index}].{field}' "
                    "must be a string."
                )

    return data


def _validate_id_list(values, valid_ids):
    """Preserve submitted references and identify invalid ones."""
    valid = []
    invalid = []

    for value in values:
        paper_id = str(value).strip()
        if paper_id in valid_ids:
            valid.append(paper_id)
        else:
            invalid.append(paper_id)

    return valid, invalid


def validate_and_enrich_analysis(raw_analysis, corpus):
    """
    Validate all references and attach evidence-verification statuses.

    A source-supported claim is marked verified only when it has at least
    one valid citation and a contiguous excerpt found in a cited abstract.
    """
    data = validate_analysis_schema(raw_analysis)
    valid_ids = set(corpus)
    invalid_reference_count = 0
    verified_excerpt_count = 0
    verified_source_claim_count = 0
    inference_count = 0

    # Themes: keep invalid IDs visible as warnings.
    for item in data["key_themes_and_consensus"]:
        ids, invalid = _validate_id_list(
            item.get("supporting_paper_ids", []), valid_ids
        )
        item["supporting_paper_ids"] = ids
        item["invalid_paper_ids"] = invalid
        item["references_valid"] = not invalid and bool(ids)
        invalid_reference_count += len(invalid)

    # Contradictions: validate both sides independently.
    for item in data["divergent_findings_and_contradictions"]:
        for field in ("paper_a_id", "paper_b_id"):
            raw_id = item.get(field)
            paper_id = str(raw_id).strip() if raw_id is not None else ""
            is_valid = paper_id in valid_ids
            item[f"{field.removesuffix('_id')}_valid"] = is_valid
            if not is_valid:
                invalid_reference_count += 1

    # Methods and benchmarks.
    for item in data["methodologies_and_benchmarks"]:
        ids, invalid = _validate_id_list(
            item.get("papers_using", []), valid_ids
        )
        item["papers_using"] = ids
        item["invalid_paper_ids"] = invalid
        item["references_valid"] = not invalid and bool(ids)
        invalid_reference_count += len(invalid)

    # Limitations and research gaps.
    for item in data["limitations_and_research_gaps"]:
        ids, invalid = _validate_id_list(
            item.get("reported_by", []), valid_ids
        )
        item["reported_by"] = ids
        item["invalid_paper_ids"] = invalid
        item["references_valid"] = not invalid and bool(ids)
        invalid_reference_count += len(invalid)

    # Claims: distinguish citation validity from evidence verification.
    validated_claims = []

    for index, item in enumerate(data["evidence_linked_claims"], start=1):
        ids, invalid = _validate_id_list(
            item.get("paper_ids", []), valid_ids
        )
        support_type = item.get("support_type", "source_supported")
        if support_type not in ("source_supported", "inference"):
            support_type = "inference"

        excerpt = item.get("evidence_excerpt", "")
        citation_valid = bool(ids) and not invalid
        excerpt_verified = False
        evidence_status = "unverified"
        verification_note = ""

        if support_type == "source_supported":
            excerpt_verified = any(
                verify_excerpt_against_abstract(
                    excerpt, corpus[paper_id]["abstract"]
                )
                for paper_id in ids
            )

            if citation_valid and excerpt_verified:
                # Matching an excerpt verifies its presence in the abstract,
                # not whether it supports the complete claim.
                evidence_status = "excerpt_matched"
                verification_note = (
                    "The excerpt was found in a cited abstract. "
                    "Citation validity and excerpt matching were checked, "
                    "but semantic support for the complete claim has not "
                    "been verified."
                )
                verified_excerpt_count += 1
            elif not citation_valid:
                verification_note = (
                    "Unverified: at least one citation is missing or invalid."
                )
            else:
                verification_note = (
                    "Unverified: excerpt was not found as a complete "
                    "normalized sequence in any cited abstract."
                )
        else:
            evidence_status = "inference"
            verification_note = (
                "This is a reasoned inference, not a directly verified quote."
            )
            inference_count += 1

        validated_claims.append({
            "claim_id": item.get("claim_id") or f"CLM-{index:03d}",
            "claim": item.get("claim", ""),
            "support_type": support_type,
            "paper_ids": ids,
            "invalid_paper_ids": invalid,
            "paper_titles": [corpus[pid]["title"] for pid in ids],
            "citation_valid": citation_valid,
            "evidence_excerpt": excerpt,
            "excerpt_verified": excerpt_verified,
            "evidence_status": evidence_status,
            "verification_note": verification_note,
        })

        invalid_reference_count += len(invalid)

    data["evidence_linked_claims"] = validated_claims
    data["status"] = "success"
    data["metadata"] = {
        "total_papers_analyzed": len(corpus),
        "total_claims": len(validated_claims),
        "verified_source_supported_claims": verified_source_claim_count,
        "inference_claims": inference_count,
        "invalid_reference_count": invalid_reference_count,
        "verified_excerpt_count": verified_excerpt_count,
    }

    return data


# ---------------------------------------------------------------------------
# Gemini error handling
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 5          # total tries per request (1 try + 4 retries)
BASE_DELAY_SECONDS = 2    # waits grow 2, 4, 8, 16 seconds (plus jitter)
MAX_DELAY_SECONDS = 60    # never sleep longer than this in one go
MAX_FORMAT_ATTEMPTS = 2   # one re-ask if the reply is empty or not JSON

TRANSIENT_HTTP_CODES = {500, 502, 503, 504}
TRANSIENT_STATUSES = {"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}

# Error kinds returned by classify_gemini_error().
DAILY_QUOTA = "daily_quota"
RATE_LIMIT = "rate_limit"
TRANSIENT = "transient"
FATAL = "fatal"


def redact_secrets(text):
    """Remove API keys from text before it is shown or logged."""
    text = str(text)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        text = text.replace(api_key, "[REDACTED]")

    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key[=:]\s*)[^\s&\"',]+", r"\1[REDACTED]", text
    )
    return text


def _error_code_and_status(exc):
    """
    Read the HTTP code and status (e.g. 503, "UNAVAILABLE") from an error.

    google-genai errors carry .code and .status. If they are missing, fall
    back to the START of the message, which the SDK formats as
    "503 UNAVAILABLE. {...}". Only the start is checked so that numbers
    elsewhere in a message (token counts, URLs) are never mistaken for
    an HTTP code.
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)

    if not isinstance(code, int) or isinstance(code, bool):
        code = None
    if not isinstance(status, str):
        status = None

    if code is None:
        match = re.match(r"\s*(\d{3})\b\s*([A-Z_]+)?", str(exc))
        if match:
            code = int(match.group(1))
            status = status or match.group(2)

    return code, (status or "").upper()


def _error_details_text(exc):
    """Text of the error details, used to look for quota identifiers."""
    details = getattr(exc, "details", None)
    if details is not None:
        try:
            return json.dumps(details, default=str) + " " + str(exc)
        except (TypeError, ValueError):
            pass
    return str(exc)


def _suggested_retry_delay(exc):
    """Seconds the server asked us to wait (RetryInfo or message), if any."""
    text = _error_details_text(exc)

    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', text)
    if not match:
        match = re.search(
            r"retry in (\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE
        )

    return float(match.group(1)) if match else None


def is_network_error(exc):
    """True for connection problems and timeouts (not HTTP error replies)."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    try:
        import httpx  # installed as a dependency of google-genai
    except ImportError:
        return False

    return isinstance(exc, httpx.TransportError)


def classify_gemini_error(exc):
    """
    Decide what to do about a failed Gemini request.

    Returns one of:
      DAILY_QUOTA - daily quota used up. Retrying cannot help.
      RATE_LIMIT  - 429 that looks temporary (e.g. per-minute). Retry.
      TRANSIENT   - 5xx / UNAVAILABLE / network problem. Retry.
      FATAL       - bad key, bad model name, bad request... Do not retry.

    A 429 is only called DAILY_QUOTA when its details say so (for example
    quotaId "GenerateRequestsPerDayPerModel-FreeTier"). The metric name
    "generate_content_free_tier_requests" alone is NOT enough, because
    per-minute limits use the same metric name.
    """
    code, status = _error_code_and_status(exc)

    if code == 429 or status == "RESOURCE_EXHAUSTED":
        details = _error_details_text(exc).casefold()
        if "perday" in details or "per_day" in details:
            return DAILY_QUOTA
        return RATE_LIMIT

    if code in TRANSIENT_HTTP_CODES or status in TRANSIENT_STATUSES:
        return TRANSIENT

    if code is None and is_network_error(exc):
        return TRANSIENT

    return FATAL


def _wait_before_retry(attempt, kind, exc):
    """Sleep with exponential backoff + jitter; honour the server's hint."""
    wait_seconds = BASE_DELAY_SECONDS * (2 ** attempt) + random.random()

    hint = _suggested_retry_delay(exc) if kind == RATE_LIMIT else None
    if hint is not None:
        wait_seconds = max(wait_seconds, hint + random.random())

    wait_seconds = min(wait_seconds, MAX_DELAY_SECONDS)
    time.sleep(wait_seconds)
    return wait_seconds


def generate_with_retries(client, model_name, prompt):
    """
    Call Gemini, retrying only temporary failures (bounded).

    Raises RuntimeError with a safe, actionable message when the request
    cannot succeed. The model name is never changed.
    """
    last_kind = None
    last_error = None
    last_reason = ""

    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
        except Exception as exc:
            kind = classify_gemini_error(exc)
            reason = redact_secrets(exc)
            last_kind, last_error, last_reason = kind, exc, reason

            if kind == DAILY_QUOTA:
                raise RuntimeError(
                    f"Gemini daily quota exhausted for {model_name}. "
                    "Retrying will not help until the quota resets. Check "
                    "the quota of the Google project that owns "
                    "GEMINI_API_KEY. The request was not retried."
                ) from exc

            if kind == FATAL:
                raise RuntimeError(
                    f"Gemini Analysis Agent request failed "
                    f"({model_name}): {reason}. "
                    "This error is not temporary, so it was not retried. "
                    "Check GEMINI_API_KEY, the model name and the request."
                ) from exc

            if attempt == MAX_ATTEMPTS - 1:
                break

            waited = _wait_before_retry(attempt, kind, exc)
            label = (
                "rate limit reached" if kind == RATE_LIMIT
                else "service temporarily unavailable"
            )
            print(
                f"Gemini {label}. Retrying in {waited:.1f}s "
                f"(attempt {attempt + 2}/{MAX_ATTEMPTS})..."
            )

    if last_kind == RATE_LIMIT:
        advice = (
            "The rate limit did not clear. Wait a few minutes, or check "
            "the project's quota."
        )
    else:
        advice = (
            "The model is overloaded or unreachable. This is on the "
            "provider's side; try again later."
        )

    raise RuntimeError(
        f"Gemini Analysis Agent request failed ({model_name}) after "
        f"{MAX_ATTEMPTS} attempts: {last_reason}. {advice}"
    ) from last_error


def parse_model_response(response):
    """Turn a Gemini response into a dict. Raises ValueError if unusable."""
    cleaned = extract_json_text(getattr(response, "text", None))

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Analysis Agent returned malformed JSON: {exc}"
        ) from exc


def analyze_literature(user_query, papers, plan=None, model=None):
    """Generate a structured, evidence-grounded literature synthesis."""
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("Research question cannot be empty.")

    corpus = build_corpus_index(papers)

    if not corpus:
        return build_insufficient_evidence_response(
            "No academic papers were retrieved. Insufficient evidence "
            "to perform a literature synthesis."
        )

    if not any(paper["abstract"] for paper in corpus.values()):
        return build_insufficient_evidence_response(
            "Retrieved papers have no usable abstracts. Insufficient "
            "evidence to synthesize findings."
        )

    usable_abstracts = sum(1 for p in corpus.values() if p["abstract"])

    paper_blocks = []
    for paper in corpus.values():
        paper_blocks.append(
            f"ID: {paper['id']}\n"
            f"Title: {paper['title']}\n"
            f"Year: {paper['year']}\n"
            f"Abstract:\n{paper['abstract'] or '[No abstract available]'}"
        )

    sub_questions = []
    if isinstance(plan, dict):
        candidate = plan.get("sub_questions", [])
        if isinstance(candidate, list):
            sub_questions = [
                question for question in candidate
                if isinstance(question, str)
            ]

    prompt = f"""
You are the Analysis Agent in AutoResearch.

Research question:
{user_query}

Research sub-questions:
{json.dumps(sub_questions, ensure_ascii=False)}

Retrieved papers:
{chr(10).join(paper_blocks)}

Evidence available: {usable_abstracts} of {len(corpus)} papers have an abstract.
Only abstracts were provided, not full texts. If this is too little to answer
the research question, say so in the summary.

Use only the abstracts above. Do not invent findings, quotations, citations,
methods, benchmarks, contradictions, or reported limitations.

Use exact paper IDs supplied above. If evidence is insufficient, say so.
Only report a contradiction when the papers genuinely disagree.
Label direct abstract-supported claims as "source_supported" and reasoned
synthesis deductions as "inference". For source_supported claims, provide
a verbatim excerpt from a cited abstract.

Return ONLY JSON with this schema:
{{
  "summary": "string",
  "key_themes_and_consensus": [
    {{"theme":"string","description":"string","supporting_paper_ids":["ID"]}}
  ],
  "divergent_findings_and_contradictions": [
    {{"topic":"string","finding_a":"string","paper_a_id":"ID",
      "finding_b":"string","paper_b_id":"ID","explanation":"string"}}
  ],
  "methodologies_and_benchmarks": [
    {{"approach_or_dataset":"string","description":"string","papers_using":["ID"]}}
  ],
  "limitations_and_research_gaps": [
    {{"gap_or_limitation":"string","reported_by":["ID"],
      "type":"reported_limitation"}}
  ],
  "evidence_linked_claims": [
    {{"claim_id":"CLM-001","claim":"string",
      "support_type":"source_supported","paper_ids":["ID"],
      "evidence_excerpt":"verbatim abstract excerpt"}}
  ]
}}
"""
    print(f"Analysis prompt length: {len(prompt)} characters")

    target_model = (
        model
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_MODEL
    )

    client = get_gemini_client()

    # Ask again (once) if the reply is empty or is not valid JSON. This is
    # separate from API-error retries and never invents a result.
    raw_analysis = None
    for format_attempt in range(MAX_FORMAT_ATTEMPTS):
        response = generate_with_retries(client, target_model, prompt)
        try:
            raw_analysis = parse_model_response(response)
            break
        except ValueError as exc:
            if format_attempt == MAX_FORMAT_ATTEMPTS - 1:
                raise
            print(f"{exc} Asking the model once more...")

    analysis = validate_and_enrich_analysis(raw_analysis, corpus)
    analysis["metadata"]["papers_with_usable_abstracts"] = usable_abstracts
    return analysis


def format_analysis_report(analysis):
    """Format a validated analysis as a readable terminal report."""
    if not isinstance(analysis, dict):
        return "Invalid analysis data."

    if analysis.get("status") == "insufficient_evidence":
        return (
            "\n[INSUFFICIENT EVIDENCE]\n"
            + str(analysis.get("summary", "No summary available."))
        )

    lines = [
        "",
        "=" * 78,
        "AUTORESEARCH — LITERATURE SYNTHESIS",
        "=" * 78,
        "",
        "1. EXECUTIVE SUMMARY",
        str(analysis.get("summary", "No summary provided.")),
    ]

    sections = [
        ("2. KEY THEMES AND CONSENSUS", "key_themes_and_consensus"),
        ("3. DIVERGENT FINDINGS AND CONTRADICTIONS",
         "divergent_findings_and_contradictions"),
        ("4. METHODOLOGIES AND BENCHMARKS",
         "methodologies_and_benchmarks"),
        ("5. LIMITATIONS AND RESEARCH GAPS",
         "limitations_and_research_gaps"),
    ]

    display_fields = {
        "key_themes_and_consensus": (
            "theme", "description", "supporting_paper_ids"
        ),
        "divergent_findings_and_contradictions": (
            "topic", "explanation", ("paper_a_id", "paper_b_id")
        ),
        "methodologies_and_benchmarks": (
            "approach_or_dataset", "description", "papers_using"
        ),
        "limitations_and_research_gaps": (
            "gap_or_limitation", "type", "reported_by"
        ),
    }

    for heading, section in sections:
        lines.extend(["", heading])
        items = analysis.get(section, [])
        if not items:
            lines.append("No findings reported.")
            continue

        first, second, references = display_fields[section]
        for index, item in enumerate(items, start=1):
            lines.append(f"{index}. {item.get(first, 'Not specified')}")
            lines.append(f"   {item.get(second, '')}")

            if isinstance(references, tuple):
                refs = [str(item.get(key, "")) for key in references]
                validity = [
                    item.get("paper_a_valid", False),
                    item.get("paper_b_valid", False),
                ]
                lines.append(
                    "   Paper references: "
                    + ", ".join(
                        f"{ref} ({'valid' if valid else 'INVALID'})"
                        for ref, valid in zip(refs, validity)
                    )
                )
            else:
                refs = item.get(references, [])
                lines.append(
                    "   Paper references: "
                    + (", ".join(refs) if refs else "None")
                )

            invalid = item.get("invalid_paper_ids", [])
            if invalid:
                lines.append("   INVALID references: " + ", ".join(invalid))

    lines.extend(["", "6. EVIDENCE-LINKED CLAIMS"])

    claims = analysis.get("evidence_linked_claims", [])
    if not claims:
        lines.append("No claims generated.")
    else:
        for claim in claims:
            lines.extend([
                "",
                f"[{claim.get('claim_id', 'CLM')}] "
                f"{claim.get('evidence_status', 'unverified').upper()}",
                str(claim.get("claim", "")),
                "   Cited papers: "
                + (", ".join(claim.get("paper_ids", [])) or "None"),
            ])

            if claim.get("evidence_excerpt"):
                lines.append(
                    "   Excerpt: " + str(claim["evidence_excerpt"])
                )

            if claim.get("verification_note"):
                lines.append("   Note: " + claim["verification_note"])

            if claim.get("invalid_paper_ids"):
                lines.append(
                    "   INVALID citations: "
                    + ", ".join(claim["invalid_paper_ids"])
                )

    metadata = analysis.get("metadata", {})
    lines.extend([
        "",
        "=" * 78,
        f"Papers: {metadata.get('total_papers_analyzed', 0)} "
        f"({metadata.get('papers_with_usable_abstracts', 0)} with abstracts) | "
        f"Claims: {metadata.get('total_claims', 0)} | "
        f"Semantically verified source claims: "
        f"{metadata.get('verified_source_supported_claims', 0)} | "
        f"Excerpt-matched claims: "
        f"{metadata.get('verified_excerpt_count', 0)} | "
        f"Inferences: {metadata.get('inference_claims', 0)} | "
        f"Invalid references: {metadata.get('invalid_reference_count', 0)}",
        "=" * 78,
    ])

    return "\n".join(lines)