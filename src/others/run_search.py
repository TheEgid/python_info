from typing import Callable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def wrap_text(text: str, width: int = 80) -> str:
    """Форматирует текст, вписывая строки в заданную ширину."""
    lines = []
    while len(text) > width:
        split_index = text.rfind(" ", 0, width)
        if split_index == -1:
            split_index = width
        lines.append(text[:split_index])
        text = text[split_index:].strip()
    lines.append(text)
    return "\n".join(lines)


def run_search(user_prompt, vector_store, embedding_function: Callable[[List[str]], np.ndarray]) -> None:  # noqa: ANN001
    """
    Выполняет поиск по векторному хранилищу на основе пользовательского запроса.
    Использует vector_store.search(query_emb, top_k, match_threshold).
    """

    query_emb = embedding_function([user_prompt], device="cpu")[0]

    search_results: List[Tuple[int, str, dict, float]] = vector_store.search(
        query_emb=query_emb, top_k=5, match_threshold=0.3
    )

    if not search_results:
        print("❌ Ничего не найдено.")
        return

    top_id, top_text, top_metadata, top_score = search_results[0]

    # print("\n" + "─" * 80)
    # print("🔍 Ваш запрос:")
    # print(wrap_text(user_prompt))
    # print("─" * 80)

    # print("📈 Лучший результат поиска:")
    # print(f"🆔 ID: {top_id}")
    # print(f"⭐ Score: {top_score:.4f}")
    # print(f"📚 Source: {top_metadata.get('source', 'unknown')}")
    # print("📝 Text:\n")
    # print(wrap_text(top_text))
    # print("─" * 80 + "\n")

    return top_text


def calculate_cosine_similarity_with_embeddings(text1: str, text2:str) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]
