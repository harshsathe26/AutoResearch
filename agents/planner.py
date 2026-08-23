import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=api_key)


def validate_plan(plan):
    required_fields = {
        "main_topic": str,
        "research_objective": str,
        "research_plan": list,
        "sub_questions": list
    }

    for field, expected_type in required_fields.items():
        if field not in plan:
            raise ValueError(f"Missing required field: {field}")

        if not isinstance(plan[field], expected_type):
            raise ValueError(
                f"'{field}' must be of type {expected_type.__name__}."
            )

    if not plan["main_topic"].strip():
        raise ValueError("Main topic cannot be empty.")

    if not plan["research_objective"].strip():
        raise ValueError("Research objective cannot be empty.")

    if not 3 <= len(plan["sub_questions"]) <= 5:
        raise ValueError(
            "Planner must generate between 3 and 5 sub-questions."
        )

    if len(plan["research_plan"]) == 0:
        raise ValueError("Research plan cannot be empty.")

    for question in plan["sub_questions"]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Every sub-question must be a non-empty string."
            )

    normalized_questions = [
        question.strip().lower()
        for question in plan["sub_questions"]
    ]

    if len(normalized_questions) != len(set(normalized_questions)):
        raise ValueError("Planner generated duplicate sub-questions.")

    return plan


def create_research_plan(user_query):
    if not user_query or not user_query.strip():
        raise ValueError("Research question cannot be empty.")

    prompt = f"""
You are the Planner Agent in an AI research system called AutoResearch.

Your job is to create a research plan for the user's research question.

You must NOT answer the research question yourself.
You must only analyze the request and plan how the system should research it.

User research question:
{user_query}

Return ONLY valid JSON. Do not use Markdown or code fences.

Use exactly this structure:

{{
    "main_topic": "The main topic of the research",
    "research_objective": "What the research should achieve",
    "research_plan": [
        "Step 1",
        "Step 2",
        "Step 3",
        "Step 4"
    ],
    "sub_questions": [
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4"
    ]
}}

Rules:
- Generate 3 to 5 focused sub-questions.
- The sub-questions should together cover the user's research request.
- Do not generate duplicate questions.
- Do not answer the questions.
- Keep the plan focused on academic research.
- Make each sub-question suitable for searching academic papers.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        plan = json.loads(response.text)

        return validate_plan(plan)

    except json.JSONDecodeError:
        raise ValueError(
            "Planner Agent returned invalid JSON."
        )


if __name__ == "__main__":
    try:
        user_query = input("Enter your research question: ")

        plan = create_research_plan(user_query)

        print("\nGenerated Research Plan:\n")
        print(json.dumps(plan, indent=4))

    except ValueError as error:
        print(f"\nPlanner Error: {error}")