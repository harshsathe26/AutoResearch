from collections import defaultdict
import re


def normalize_text(text):
    """Normalize text for concept extraction."""
    if not text:
        return ""

    return re.sub(
        r"[^a-zA-Z0-9\s-]",
        " ",
        str(text).lower()
    )


def extract_concepts(text, min_length=4):
    """
    Extract lightweight concepts/keywords from text.

    Graph RAG uses these concepts to create relationships
    between retrieved papers.
    """

    stop_words = {
        "about", "after", "again", "against", "also",
        "among", "because", "before", "being", "between",
        "could", "during", "each", "from", "further",
        "have", "having", "into", "more", "most",
        "other", "over", "same", "some", "such",
        "than", "that", "their", "these", "they",
        "this", "those", "through", "using", "were",
        "which", "while", "with", "would",

        "large", "language", "models", "model",
        "paper", "study", "research", "approach",
        "method", "methods", "system", "systems",
        "results", "based", "proposed", "using"
    }

    words = normalize_text(text).split()

    concepts = set()

    for word in words:
        word = word.strip("-")

        if (
            len(word) >= min_length
            and word not in stop_words
            and not word.isdigit()
        ):
            concepts.add(word)

    return concepts


def paper_text(paper):
    """Create searchable text from a retrieved paper."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Protect Graph RAG from malformed abstract data.
    if isinstance(abstract, dict):
        abstract = ""

    authors = paper.get("authors", [])

    if isinstance(authors, list):
        authors_text = " ".join(
            str(author)
            for author in authors
        )
    else:
        authors_text = str(authors)

    return (
        f"{title} "
        f"{abstract} "
        f"{authors_text}"
    )


def build_knowledge_graph(papers):
    """
    Build a lightweight paper-concept knowledge graph.

    Nodes:
        - papers
        - concepts

    Relationships:
        - paper -> concept
        - concept -> paper
    """

    graph = {
        "papers": {},
        "concepts": {},
        "paper_concept_edges": []
    }

    for index, paper in enumerate(papers):

        paper_id = (
            paper.get("id")
            or paper.get("url")
            or paper.get("title")
            or f"paper_{index}"
        )

        paper_id = str(paper_id)

        concepts = extract_concepts(
            paper_text(paper)
        )

        graph["papers"][paper_id] = {
            "id": paper_id,
            "title": paper.get(
                "title",
                "Title not available"
            ),
            "year": paper.get("year"),
            "authors": paper.get(
                "authors",
                []
            ),
            "url": paper.get("url"),
            "relevance_score": paper.get(
                "relevance_score",
                0.0
            ),
            "concepts": sorted(concepts)
        }

        for concept in concepts:

            graph["concepts"].setdefault(
                concept,
                {
                    "name": concept,
                    "papers": []
                }
            )

            graph["concepts"][concept]["papers"].append(
                paper_id
            )

            graph["paper_concept_edges"].append(
                {
                    "paper_id": paper_id,
                    "concept": concept
                }
            )

    return graph


def find_related_papers(
    paper_id,
    graph,
    limit=5
):
    """
    Find papers related to a paper through shared concepts.

    The number of shared concepts determines the relationship
    strength.
    """

    if paper_id not in graph["papers"]:
        return []

    source_concepts = set(
        graph["papers"][paper_id].get(
            "concepts",
            []
        )
    )

    related_scores = defaultdict(int)

    for concept in source_concepts:

        concept_data = graph["concepts"].get(
            concept,
            {}
        )

        for related_id in concept_data.get(
            "papers",
            []
        ):

            if related_id == paper_id:
                continue

            related_scores[related_id] += 1

    ranked = sorted(
        related_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for related_id, shared_concepts in ranked[:limit]:

        paper = graph["papers"][related_id].copy()

        paper["shared_concepts"] = shared_concepts

        results.append(paper)

    return results


def calculate_graph_connections(
    graph,
    paper_id
):
    """
    Calculate how strongly a paper is connected to
    other papers in the graph.

    Returns:
        dict:
            {
                related_paper_id: shared_concept_count
            }
    """

    if paper_id not in graph["papers"]:
        return {}

    source_concepts = set(
        graph["papers"][paper_id].get(
            "concepts",
            []
        )
    )

    connection_scores = defaultdict(int)

    for concept in source_concepts:

        concept_data = graph["concepts"].get(
            concept,
            {}
        )

        for related_id in concept_data.get(
            "papers",
            []
        ):

            if related_id == paper_id:
                continue

            connection_scores[related_id] += 1

    return dict(connection_scores)


def graph_rag_retrieve(
    query,
    papers,
    top_k=5
):
    """
    Perform graph-based retrieval over papers already
    retrieved by the Retrieval Agent.

    Ranking combines:

        1. Direct query-concept matching
        2. Original retrieval relevance score
        3. Graph connectivity to directly relevant papers

    This makes graph relationships influence retrieval
    instead of merely being returned as metadata.
    """

    if not papers:
        return []

    graph = build_knowledge_graph(papers)

    query_concepts = extract_concepts(query)

    if not query_concepts:
        return []

    # --------------------------------------------------
    # STEP 1: Direct query-to-paper matching
    # --------------------------------------------------

    direct_matches = {}

    for paper_id, paper in graph["papers"].items():

        paper_concepts = set(
            paper.get("concepts", [])
        )

        matched_concepts = (
            query_concepts &
            paper_concepts
        )

        direct_matches[paper_id] = {
            "score": len(matched_concepts),
            "matched_concepts": matched_concepts
        }

    # --------------------------------------------------
    # STEP 2: Identify directly relevant papers
    # --------------------------------------------------

    directly_relevant = {
        paper_id
        for paper_id, data in direct_matches.items()
        if data["score"] > 0
    }

    # --------------------------------------------------
    # STEP 3: Expand through graph relationships
    # --------------------------------------------------

    graph_scores = defaultdict(float)
    graph_sources = defaultdict(set)

    for source_id in directly_relevant:

        connections = calculate_graph_connections(
            graph,
            source_id
        )

        for related_id, shared_count in connections.items():

            # A paper receives graph evidence when it is
            # connected to a directly relevant paper.
            graph_scores[related_id] += shared_count

            graph_sources[related_id].add(
                source_id
            )

    # --------------------------------------------------
    # STEP 4: Calculate combined scores
    # --------------------------------------------------

    ranked_results = []

    for paper_id, paper in graph["papers"].items():

        direct_score = direct_matches[
            paper_id
        ]["score"]

        retrieval_score = float(
            paper.get(
                "relevance_score",
                0.0
            ) or 0.0
        )

        graph_score = graph_scores.get(
            paper_id,
            0.0
        )

        # Direct query matches are strongest.
        direct_weight = direct_score * 3.0

        # Original retrieval score provides additional
        # evidence from the Retrieval Agent.
        retrieval_weight = retrieval_score

        # Graph relationships provide expansion evidence.
        graph_weight = graph_score * 0.5

        combined_score = (
            direct_weight
            + retrieval_weight
            + graph_weight
        )

        # Ignore papers with absolutely no connection
        # to the query or relevant graph.
        if combined_score <= 0:
            continue

        ranked_results.append(
            {
                "paper_id": paper_id,
                "paper": paper,
                "direct_score": direct_score,
                "retrieval_score": retrieval_score,
                "graph_score": graph_score,
                "combined_score": combined_score,
                "matched_concepts": direct_matches[
                    paper_id
                ]["matched_concepts"],
                "graph_sources": graph_sources.get(
                    paper_id,
                    set()
                )
            }
        )

    # --------------------------------------------------
    # STEP 5: Rank papers
    # --------------------------------------------------

    ranked_results.sort(
        key=lambda item: (
            item["combined_score"],
            item["direct_score"],
            item["graph_score"],
            item["retrieval_score"]
        ),
        reverse=True
    )

    # --------------------------------------------------
    # STEP 6: Build final Graph RAG results
    # --------------------------------------------------

    selected = []

    for item in ranked_results[:top_k]:

        paper_id = item["paper_id"]

        result = item["paper"].copy()

        result["graph_score"] = item[
            "graph_score"
        ]

        result["combined_graph_score"] = round(
            item["combined_score"],
            4
        )

        result["matched_concepts"] = sorted(
            item["matched_concepts"]
        )

        result["graph_sources"] = [
            graph["papers"][source_id]["title"]
            for source_id in item[
                "graph_sources"
            ]
            if source_id in graph["papers"]
        ]

        result["related_papers"] = (
            find_related_papers(
                paper_id,
                graph,
                limit=3
            )
        )

        selected.append(result)

    return selected


def build_graph_context(
    query,
    papers,
    top_k=5
):
    """
    Create a compact Graph RAG context for downstream
    synthesis/research agents.

    This is the main public interface for Graph RAG.
    """

    results = graph_rag_retrieve(
        query,
        papers,
        top_k=top_k
    )

    context = []

    for paper in results:

        context.append(
            {
                "title": paper.get(
                    "title"
                ),
                "year": paper.get(
                    "year"
                ),
                "authors": paper.get(
                    "authors",
                    []
                ),
                "url": paper.get(
                    "url"
                ),
                "relevance_score": paper.get(
                    "relevance_score",
                    0.0
                ),
                "graph_score": paper.get(
                    "graph_score",
                    0.0
                ),
                "combined_graph_score": paper.get(
                    "combined_graph_score",
                    0.0
                ),
                "matched_concepts": paper.get(
                    "matched_concepts",
                    []
                ),
                "graph_sources": paper.get(
                    "graph_sources",
                    []
                ),
                "related_papers": [
                    {
                        "title": related.get(
                            "title"
                        ),
                        "year": related.get(
                            "year"
                        ),
                        "url": related.get(
                            "url"
                        ),
                        "shared_concepts": related.get(
                            "shared_concepts",
                            0
                        )
                    }
                    for related in paper.get(
                        "related_papers",
                        []
                    )
                ]
            }
        )

    return context