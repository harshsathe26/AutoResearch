from retrieval.paper_search import search_papers


MIN_RELEVANCE_SCORE = 0.30


def normalize_text(text):
    """
    Normalize text for simple keyword matching.
    """
    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
    )




import re


def normalize_paper_id(paper):
    """Create a normalized identifier for a paper."""

    for value in (paper.get("doi"), paper.get("url")):
        if not value:
            continue

        value = value.strip().lower()
        match = re.search(
            r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?"
            r"(10\.\d{4,9}/\S+)",
            value,
        )

        if match:
            return f"doi:{match.group(1).rstrip('.,;)')}"

    for value in (paper.get("id"), paper.get("url")):
        if not value:
            continue

        match = re.search(
            r"openalex\.org/(W\d+)",
            value.strip(),
            re.IGNORECASE,
        )

        if match:
            return f"openalex:{match.group(1).upper()}"

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        paper.get("title", "").lower(),
    ).strip()

    return f"title:{title}" if title else None
def normalize_paper_title(paper):
    """Normalize a paper title for duplicate comparison."""
    return " ".join(
        re.sub(
            r"[^a-z0-9]+",
            " ",
            paper.get("title", "").lower()
        ).split()
    )




def calculate_relevance_score(question, paper):
    """
    Calculate a relevance score using keyword overlap.

    Title matches receive more importance than
    abstract-only matches.
    """

    stop_words = {
        "what", "are", "the", "is", "a", "an",
        "and", "or", "of", "to", "in", "on",
        "for", "how", "do", "does", "did",
        "with", "from", "used", "use",
        "current", "main", "primary",
        "key", "systems", "system",
        "their", "this", "that",
        "these", "those",
        "into", "while", "such", "as",
        "observed", "identified",
        "utilized", "measure",
        "measuring", "effectiveness",
        "which", "where", "when",
        "can", "still", "reduce",
        "reduces", "reduction"
    }

    question_words = {
        word
        for word in normalize_text(question).split()
        if word not in stop_words
        and len(word) > 2
    }

    if not question_words:
        return 0.0

    title = normalize_text(
        paper.get("title", "")
    )

    abstract = normalize_text(
        paper.get("abstract", "")
    )

    title_words = set(title.split())
    abstract_words = set(abstract.split())

    title_matches = (
        question_words.intersection(title_words)
    )

    abstract_matches = (
        question_words.intersection(abstract_words)
    )

    # Title matches are more valuable because the
    # paper title directly describes its research focus.
    title_score = (
        len(title_matches) /
        len(question_words)
    )

    abstract_score = (
        len(abstract_matches) /
        len(question_words)
    )

    score = (
        title_score * 0.40 +
        abstract_score * 0.60
    )

    return min(score, 1.0)


def retrieve_for_sub_questions(
    sub_questions,
    papers_per_question=5,
    relevance_threshold=MIN_RELEVANCE_SCORE
):
    """
    Retrieve, score, filter, deduplicate and rank
    papers for all research sub-questions.
    """


    unique_papers = {}
    title_to_key = {}

    for question in sub_questions:
        papers = search_papers(
            question,
            limit=papers_per_question
        )

        for paper in papers:
            score = calculate_relevance_score(
                question,
                paper
            )

            if score < relevance_threshold:
                continue

            paper_id = normalize_paper_id(paper)
            normalized_title = normalize_paper_title(paper)

            if not paper_id and not normalized_title:
                continue

            # Prefer an exact identifier match; otherwise
            # check whether the normalized title already exists.
            existing_key = (
                paper_id
                if paper_id in unique_papers
                else title_to_key.get(normalized_title)
            )

            if existing_key is None:
                existing_key = paper_id or f"title:{normalized_title}"

                unique_papers[existing_key] = {
                    **paper,
                    "relevance_score": score,
                    "source_questions": [question]
                }

                if normalized_title:
                    title_to_key[normalized_title] = existing_key

            else:
                existing = unique_papers[existing_key]

                if score > existing["relevance_score"]:
                    existing["relevance_score"] = score

                    # Keep the best-scoring record's metadata,
                    # while preserving questions already associated.
                    existing.update({
                        key: value
                        for key, value in paper.items()
                        if value
                    })

                if question not in existing["source_questions"]:
                    existing["source_questions"].append(question)

                if normalized_title:
                    title_to_key[normalized_title] = existing_key


    for question in sub_questions:

        papers = search_papers(
            question,
            limit=papers_per_question
        )

        for paper in papers:

            score = calculate_relevance_score(
                question,
                paper
            )

            if score < relevance_threshold:
                continue

            paper_id = normalize_paper_id(
                paper
            )

            if not paper_id:
                continue

            if paper_id not in unique_papers:

                unique_papers[paper_id] = {
                    **paper,
                    "relevance_score": score,
                    "source_questions": [
                        question
                    ]
                }

            else:

                existing_score = (
                    unique_papers[paper_id]
                    ["relevance_score"]
                )

                if score > existing_score:

                    unique_papers[
                        paper_id
                    ]["relevance_score"] = score

                if question not in (
                    unique_papers[paper_id]
                    ["source_questions"]
                ):

                    unique_papers[
                        paper_id
                    ]["source_questions"].append(
                        question
                    )

    return sorted(
        unique_papers.values(),
        key=lambda paper:
            paper["relevance_score"],
        reverse=True
    )
