from retrieval.paper_search import search_papers


def retrieve_for_sub_questions(sub_questions, papers_per_question=5):
    results = {}

    for question in sub_questions:
        papers = search_papers(
            question,
            limit=papers_per_question
        )

        results[question] = papers

    return results