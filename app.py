from agents.planner import create_research_plan
from retrieval.retriever import retrieve_for_sub_questions
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
            papers_per_question=3
        )

        # --------------------------------------------------
        # 3. DISPLAY RETRIEVED PAPERS
        # --------------------------------------------------

        for question, papers in retrieved_papers.items():

            print("\n" + "=" * 80)

            print("SUB-QUESTION:")
            print(question)

            print("\nRETRIEVED PAPERS:")

            if not papers:
                print("No papers found.")
                continue

            for i, paper in enumerate(
                papers,
                start=1
            ):
                print(f"\n{i}. {paper['title']}")
                print(f"Year: {paper['year']}")
                print(f"Authors: {', '.join(paper['authors'])}")
                print(f"URL: {paper['url']}")

    except ValueError as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()