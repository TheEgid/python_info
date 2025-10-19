from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def get_frida_embeddings(
    sentences: List[str], model_name: str = "ai-forever/FRIDA", batch_size: int = 32) -> np.ndarray:
    """
    Возвращает эмбеддинги предложений с использованием модели FRIDA.

    Args:
        sentences (List[str]): Список предложений (русский или английский).
        model_name (str): Название модели SentenceTransformer или путь к локальной модели.
        batch_size (int): Размер батча для encode (для ускорения при больших данных).

    Returns:
        np.ndarray: Массив эмбеддингов shape=(len(sentences), 1536), dtype=float32
    """
    model: SentenceTransformer = SentenceTransformer(model_name, device="cpu")
    embeddings: np.ndarray = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    return embeddings
