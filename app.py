from retrieval.paper_search import search_papers


papers = search_papers("Retrieval Augmented Generation hallucination", limit=5)

for i, paper in enumerate(papers, start=1):
    print(f"\n{i}. {paper['title']}")
    print(f"Year: {paper['year']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Abstract: {paper['abstract'][:300]}...")
    print(f"URL: {paper['url']}")