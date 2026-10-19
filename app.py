import json

from agents.planner import create_research_plan
from agents.analysis_agent import (
    analyze_literature,
    format_analysis_report,
)
from agents.verification_agent import (
    verify_analysis,
    format_verification_report,
)
from agents.report_agent import generate_research_report
from retrieval.retriever import retrieve_for_sub_questions
from retrieval.graph_rag import build_graph_context


def main():
    user_query = input(
        "\nEnter your research question: "
    ).strip()

    if not user_query:
        print("Please enter a research question.")
        return

    try:
        # STEP 1: Research Planner
        print("\n" + "=" * 70)
        print("STEP 1: RESEARCH PLANNER")
        print("=" * 70)

        plan = create_research_plan(user_query)
        print(json.dumps(plan, indent=2, ensure_ascii=False))

        # STEP 2: Paper Retrieval
        print("\n" + "=" * 70)
        print("STEP 2: PAPER RETRIEVAL")
        print("=" * 70)

        sub_questions = plan.get("sub_questions", [])

        if not sub_questions:
            print("The Planner did not return any sub-questions.")
            return

        papers = retrieve_for_sub_questions(
            sub_questions,
            papers_per_question=10,
        )

        if not papers:
            print(
                "No relevant papers were retrieved. "
                "The Analysis Agent will not run without evidence."
            )
            return

        print(f"\nRetrieved {len(papers)} unique papers:\n")

        for index, paper in enumerate(papers, start=1):
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "Year unavailable"
            url = paper.get("url") or "URL unavailable"

            print(f"{index}. {title} ({year})")
            print(f"   URL: {url}")

        # STEP 3: Graph RAG
        print("\n" + "=" * 70)
        print("STEP 3: GRAPH RAG")
        print("=" * 70)

        graph_context = build_graph_context(
        user_query,
        papers,
        top_k=5,
        )

        if graph_context:
            print(graph_context)
        else:
            print("No graph context was generated.")

        # STEP 4: Analysis Agent
        print("\n" + "=" * 70)
        print("STEP 4: LITERATURE ANALYSIS AGENT")
        print("=" * 70)

        print(
            "\nAnalyzing retrieved paper abstracts, "
            "identifying themes, comparing findings, "
            "and checking evidence-linked claims..."
        )

        analysis = analyze_literature(
            user_query=user_query,
            papers=papers,
            plan=plan,
        )

        print("\n" + format_analysis_report(analysis))
        print(
            "\nNote: the statuses above (e.g. EXCERPT_MATCHED) only show "
            "that a quote exists in a cited abstract. They do NOT mean the "
            "claim is verified. See the Verification Report below."
        )

        # STEP 5: Claim Verification Agent
        print("\n" + "=" * 70)
        print("STEP 5: CLAIM VERIFICATION AGENT")
        print("=" * 70)

        # The report uses the verified analysis when verification worked.
        # Otherwise it receives the original analysis plus the error, and
        # states that the claims are NOT verified.
        report_analysis = analysis
        verification_error = None

        if not analysis.get("evidence_linked_claims"):
            print("No evidence-linked claims were available to verify.")
        else:
            print(
                "\nChecking each claim against its cited abstracts..."
            )
            try:
                verified_analysis = verify_analysis(analysis, papers)
                report_analysis = verified_analysis
                print(format_verification_report(verified_analysis))
            except Exception as error:
                # Verification problems must never hide the analysis.
                verification_error = f"{type(error).__name__}: {error}"
                print(
                    f"\nClaim verification is unavailable: "
                    f"{type(error).__name__}: {error}"
                )
                print(
                    "The analysis above has NOT been verified. Treat all "
                    "of its claims as unverified."
                )

        # STEP 6: Report Generation (needs the verification results)
        print("\n" + "=" * 70)
        print("STEP 6: REPORT GENERATION")
        print("=" * 70)

        try:
            report = generate_research_report(
                user_query=user_query,
                plan=plan,
                papers=papers,
                graph_context=graph_context,
                analysis=report_analysis,
                verification_error=verification_error,
            )

            if report["markdown_path"]:
                print(f"\nMarkdown report saved: {report['markdown_path']}")
            if report["docx_path"]:
                print(f"Word report saved:     {report['docx_path']}")
            for message in report["errors"]:
                print(f"WARNING: {message}")

            v_summary = report_analysis.get("verification_summary")
            if isinstance(v_summary, dict):
                print(
                    "\nVerification summary: "
                    f"{v_summary.get('supported', 0)} supported, "
                    f"{v_summary.get('partially_supported', 0)} partially "
                    f"supported, {v_summary.get('unsupported', 0)} "
                    f"unsupported, {v_summary.get('unverifiable', 0)} "
                    f"unverifiable of {v_summary.get('total_claims', 0)} "
                    f"claims; {v_summary.get('flagged_for_review', 0)} "
                    "flagged for manual review."
                )
                if v_summary.get("service_status", "completed") != "completed":
                    print(
                        "WARNING: verification was "
                        f"{v_summary['service_status']}. "
                        f"{v_summary.get('service_note', '')}"
                    )
            else:
                print(
                    "\nWARNING: the claims in this report are NOT verified."
                )
        except Exception as error:
            print(
                f"\nReport generation failed: "
                f"{type(error).__name__}: {error}"
            )
            print(
                "No report file was confirmed as created. The analysis "
                "and verification output above are unaffected."
            )

        print("\n" + "=" * 70)
        print("RESEARCH PIPELINE COMPLETED")
        print("=" * 70)

    except ValueError as error:
        print(f"\nInput or validation error: {error}")

    except RuntimeError as error:
        print(f"\nAnalysis/AI service error: {error}")
        print(
            "Check the error details. If the model is temporarily "
            "unavailable, retry later. The Planner and .env "
            "configuration have not been changed."
        )

    except KeyboardInterrupt:
        print("\n\nResearch cancelled by user.")

    except Exception as error:
        print(
            f"\nUnexpected pipeline error: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()