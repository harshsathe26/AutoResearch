import re
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
    "measuring", "effectiveness"
}


def build_search_query(question):
    """
    Convert a natural-language research question
    into a shorter academic search query.
    """

    words = re.findall(
        r"[A-Za-z0-9-]+",
        question.lower()
    )

    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]

    return " ".join(keywords)


def decode_abstract(inverted_index):
    """
    Convert OpenAlex's abstract inverted index
    into normal readable text.
    """

    if not inverted_index:
        return "Abstract not available"

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda x: x[0])

    return " ".join(
        word for _, word in words
    )


def search_papers(query, limit=5):
    """
    Search OpenAlex for academic papers.
    """

    url = "https://api.openalex.org/works"

    search_query = build_search_query(query)

    params = {
        "search": search_query,
        "per-page": limit
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 429:
            print(
                "OpenAlex rate limit reached. "
                "Please wait before trying again."
            )
            return []

        response.raise_for_status()

        data = response.json()
        papers = []

        for paper in data.get("results", []):

            authors = []

            for authorship in paper.get(
                "authorships",
                []
            ):
                author = authorship.get(
                    "author",
                    {}
                )

                if author.get("display_name"):
                    authors.append(
                        author["display_name"]
                    )

            abstract = decode_abstract(
                paper.get(
                    "abstract_inverted_index"
                )
            )

            papers.append({
                "id": paper.get("id"),

                "title": paper.get(
                    "display_name",
                    "Title not available"
                ),

                "abstract": abstract,

                "authors": authors,

                "year": paper.get(
                    "publication_year"
                ),

                "url": (
                    paper.get("doi")
                    or paper.get("id")
                )
            })

        return papers

    except requests.exceptions.RequestException as e:
        print(
            f"Error searching papers: {e}"
        )
        return []