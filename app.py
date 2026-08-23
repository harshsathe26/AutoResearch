from agents.planner import create_research_plan
import json


def main():
    print("Welcome to AutoResearch")

    user_query = input("\nEnter your research question: ")

    try:
        plan = create_research_plan(user_query)

        print("\nGenerated Research Plan:\n")
        print(json.dumps(plan, indent=4))

        print("\nSub-questions for further research:")
        for i, question in enumerate(plan["sub_questions"], start=1):
            print(f"{i}. {question}")

    except ValueError as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()