import html
import re
import time

import requests


STOP_WORDS = {
    "what", "are", "the", "is", "a", "an",
    "and", "or", "of", "to", "in", "on",
    "for", "how", "do", "does", "did",
    "with", "from", "used", "use",
    "current", "main", "primary",
    "key", "systems", "system",
    "their", "this", "that",
    "these", "those", "into",
    "while", "such", "as",
    "observed", "identified",
    "utilized", "measure",
    "measuring", "effectiveness",
}

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"

HEADERS = {
    "User-Agent": "AutoResearch/1.0 (academic research tool)"
}

# Reuse connections across searches.
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def build_search_query(question):
    """Convert a research question into academic search keywords."""
    words = re.findall(r"[A-Za-z0-9-]+", question.lower())

    keywords = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(keywords)


def decode_abstract(inverted_index):
    """Convert an OpenAlex inverted abstract index into readable text."""
    if not inverted_index:
        return "Abstract not available"

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in words)


def clean_abstract(text):
    """Remove basic HTML/XML markup from a publisher abstract."""
    if not text or not isinstance(text, str):
        return "Abstract not available"

    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text or "Abstract not available"


def _request_json(url, params, max_attempts=3):
    """
    Request JSON with limited retries for rate limits and temporary errors.
    Respects Retry-After when the server provides it.
    """
    for attempt in range(max_attempts):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=25,
            )

            if response.status_code == 429 or response.status_code in {
                500, 502, 503, 504
            }:
                if attempt == max_attempts - 1:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")

                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = 2 ** (attempt + 1)

                delay = max(1, min(delay, 30))
                print(
                    f"Temporary API error ({response.status_code}). "
                    f"Retrying in {delay:.0f} seconds "
                    f"({attempt + 2}/{max_attempts})..."
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as exc:
            if attempt == max_attempts - 1:
                raise

            delay = 2 ** (attempt + 1)
            print(
                f"Temporary search error: {exc}. "
                f"Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    return None


def _search_openalex(search_query, limit):
    """Search OpenAlex and normalize results to the project paper format."""
    data = _request_json(
        OPENALEX_URL,
        {
            "search": search_query,
            "per-page": limit,
        },
    )

    papers = []

    for paper in data.get("results", []):
        authors = []

        for authorship in paper.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name")

            if name:
                authors.append(name)

        papers.append({
            "id": paper.get("id"),
            "title": paper.get("display_name") or "Title not available",
            "abstract": decode_abstract(
                paper.get("abstract_inverted_index")
            ),
            "authors": authors,
            "year": paper.get("publication_year"),
            "doi": paper.get("doi"),
            "url": paper.get("doi") or paper.get("id"),
        })

    return papers


def _search_crossref(search_query, limit):
    """Fallback academic search using Crossref."""
    data = _request_json(
        CROSSREF_URL,
        {
            "query.bibliographic": search_query,
            "rows": limit,
            "select": (
                "DOI,title,author,published,created,URL,abstract,type"
            ),
        },
    )

    papers = []

    for paper in data.get("message", {}).get("items", []):
        titles = paper.get("title") or []
        title = titles[0] if titles else "Title not available"

        authors = []
        for author in paper.get("author", []):
            name = " ".join(
                part for part in [
                    author.get("given", ""),
                    author.get("family", ""),
                ]
                if part
            ).strip()

            if name:
                authors.append(name)

        date_parts = (
            paper.get("published", {}).get("date-parts")
            or paper.get("created", {}).get("date-parts")
            or []
        )

        year = (
            date_parts[0][0]
            if date_parts and date_parts[0]
            else None
        )

        doi = paper.get("DOI")
        url = f"https://doi.org/{doi}" if doi else paper.get("URL")

        papers.append({
            "id": doi or url,
            "title": title,
            "abstract": clean_abstract(
                paper.get("abstract", "")
            ),
            "authors": authors,
            "year": year,
            "doi": f"https://doi.org/{doi}" if doi else None,
            "url": url,
        })

    return papers


def search_papers(query, limit=5):
    """
    Search OpenAlex first, then use Crossref if OpenAlex fails.
    Returns papers in the existing project format.
    """
    search_query = build_search_query(query)

    if not search_query:
        print("No usable keywords found for this paper search.")
        return []

    try:
        papers = _search_openalex(search_query, limit)

        if papers:
            return papers

        print("OpenAlex returned no results. Trying Crossref...")

    except requests.exceptions.RequestException as exc:
        print(f"OpenAlex search failed: {exc}")
        print("Trying Crossref as a fallback...")

    except (ValueError, KeyError, TypeError) as exc:
        print(f"Could not process OpenAlex results: {exc}")
        print("Trying Crossref as a fallback...")

    try:
        papers = _search_crossref(search_query, limit)

        if papers:
            print(f"Crossref returned {len(papers)} papers.")
            return papers

        print("Crossref returned no papers for this query.")

    except requests.exceptions.RequestException as exc:
        print(f"Crossref search failed: {exc}")

    except (ValueError, KeyError, TypeError) as exc:
        print(f"Could not process Crossref results: {exc}")

    return []
