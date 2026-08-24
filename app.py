from agents.planner import create_research_plan
from retrieval.retriever import retrieve_for_sub_questions
from retrieval.graph_rag import build_graph_context

import json


def main():
    print("Welcome to AutoResearch")

    user_query = input("\nEnter your research question: ")

    try:
        # --------------------------------------------------
        # 1. PLANNER AGENT
        # --------------------------------------------------

        plan = create_research_plan(user_query)

        print("\nGenerated Research Plan:\n")
        print(json.dumps(plan, indent=4))

        print("\nSub-questions for further research:")

        for i, question in enumerate(
            plan["sub_questions"],
            start=1
        ):
            print(f"{i}. {question}")

        # --------------------------------------------------
        # 2. RETRIEVAL AGENT
        # --------------------------------------------------

        print("\nStarting academic paper retrieval...")

        retrieved_papers = retrieve_for_sub_questions(
            plan["sub_questions"],
            papers_per_question=10
        )

        print("\nResearch retrieval complete.")
        print(
            f"Total unique papers: "
            f"{len(retrieved_papers)}"
        )

        # --------------------------------------------------
        # 3. DISPLAY RETRIEVED PAPERS
        # --------------------------------------------------

        print("\n" + "=" * 80)
        print("UNIQUE RETRIEVED PAPERS")
        print("=" * 80)

        if not retrieved_papers:
            print("No papers found.")
        else:
            for i, paper in enumerate(
                retrieved_papers,
                start=1
            ):
                print(f"\n{i}. {paper['title']}")
                print(f"Year: {paper['year']}")
                print(
                    f"Relevance Score: "
                    f"{paper['relevance_score']:.2f}"
                )
                print(
                    f"Authors: "
                    f"{', '.join(paper['authors'])}"
                )
                print(f"URL: {paper['url']}")

                print("Retrieved for:")
                for question in paper["source_questions"]:
                    print(f"  - {question}")

        # --------------------------------------------------
        # 4. GRAPH RAG
        # --------------------------------------------------

        print("\nStarting Graph RAG context building...")

        graph_context = build_graph_context(
            user_query,
            retrieved_papers,
            top_k=5
        )

        print("\nGraph RAG context building complete.")

        print("\n" + "=" * 80)
        print("GRAPH RAG CONTEXT")
        print("=" * 80)

        print(
            json.dumps(
                graph_context,
                indent=4
            )
        )

    except ValueError as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()