"""Print similarity scores for in-scope vs out-of-scope queries to help choose RELEVANCE_THRESHOLD.

Usage: python -m scripts.calibrate_threshold   (uses the configured embedding model and index)
Run this once with EMBEDDING_PROVIDER=huggingface on a machine that can download the model.
"""

from __future__ import annotations

from app.config import Settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import DestinationKnowledgeBase

IN_SCOPE = [
    "What are the local customs in Tokyo?",
    "What should I pack for Reykjavik in winter?",
    "Do I need a visa for Thailand?",
    "Is tap water safe to drink in Bangkok?",
    "Is Barcelona safe? Pickpockets?",
    "Best time of year to visit Tokyo",
]
OUT_OF_SCOPE = [
    "How do I bake sourdough bread?",
    "What is the capital of France?",
    "Explain quantum entanglement",
    "Best stock to buy this year",
    "Write me a poem about the sea",
]


def best_score(kb: DestinationKnowledgeBase, query: str) -> float:
    hits = kb.search(query, top_k=1, threshold=0.0)
    return hits[0].score if hits else 0.0


def main() -> None:
    settings = Settings.from_env()
    kb = DestinationKnowledgeBase.load(settings, get_embeddings(settings))
    in_scores = {q: best_score(kb, q) for q in IN_SCOPE}
    out_scores = {q: best_score(kb, q) for q in OUT_OF_SCOPE}
    for label, scores in (("IN-SCOPE", in_scores), ("OUT-OF-SCOPE", out_scores)):
        print(label)
        for query, score in scores.items():
            print(f"  {score:.3f}  {query}")
    low_in, high_out = min(in_scores.values()), max(out_scores.values())
    print(f"\nlowest in-scope best score:  {low_in:.3f}\nhighest out-of-scope score: {high_out:.3f}")
    if low_in > high_out:
        print(f"Suggested RELEVANCE_THRESHOLD ~ {(low_in + high_out) / 2:.2f}")
    else:
        print("Scores overlap: no single threshold separates these queries; rely on the LLM to judge.")
    print(f"Current threshold: {settings.relevance_threshold}")


if __name__ == "__main__":
    main()
