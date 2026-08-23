import requests


def decode_abstract(inverted_index):
    if not inverted_index:
        return "Abstract not available"

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(word for _, word in words)


def search_papers(query, limit=5):
    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per-page": limit
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()
        papers = []

        for paper in data.get("results", []):
            authors = []

            for authorship in paper.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])

            abstract = decode_abstract(
                paper.get("abstract_inverted_index")
            )

            papers.append({
                "title": paper.get("display_name", "Title not available"),
                "abstract": abstract,
                "authors": authors,
                "year": paper.get("publication_year"),
                "url": paper.get("doi") or paper.get("id")
            })

        return papers

    except requests.exceptions.RequestException as e:
        print(f"Error searching papers: {e}")
        return []