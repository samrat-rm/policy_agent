"""Test retrieval on its own. No LLM involved.

    python -m scripts.check_retrieval

If the right text shows up here, retrieval is fine and the model is the
problem. If it doesn't, the bug is in chunking or embedding and no amount of
prompt tuning will fix it.

Always run this before debugging model behaviour.
"""

from app.config import TOP_K
from app.ingest import get_vectorstore

QUERIES = [
    "monitor reimbursement percentage cap",
    "sick leave days medical certificate",
    "office chair desk reimbursement",
    "dog vet bills pet expenses",
]


def main() -> None:
    store = get_vectorstore()

    count = store._collection.count()
    print(f"Chunks in store: {count}")
    if count == 0:
        print("Store is empty. Run: python -m app.ingest")
        return

    for query in QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = store.similarity_search_with_score(query, k=TOP_K)
        if not results:
            print("  no results")
            continue

        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "?")
            section = doc.metadata.get("section", "?")
            text = doc.page_content.replace("\n", " ")[:220]
            print(f"\n  [{i}] score={score:.4f}  {source} / {section}")
            print(f"      {text}...")


if __name__ == "__main__":
    main()
