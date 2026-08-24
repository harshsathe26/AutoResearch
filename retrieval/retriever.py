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


def normalize_paper_id(paper):
    """
    Create a stable identifier for deduplication.

    Prefer the OpenAlex ID, then DOI/URL,
    then title.
    """

    if paper.get("id"):
        return paper["id"].lower().strip()

    if paper.get("url"):
        return paper["url"].lower().strip().rstrip("/")

    if paper.get("title"):
        return paper["title"].lower().strip()

    return None


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