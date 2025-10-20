import sys
from typing import Callable, List, Tuple

import numpy as np
from tqdm import tqdm

from classes.VectorStoreFRIDA import VectorStoreFRIDA
from others.frida import get_frida_embeddings
from others.wiki_scraper import process_text_file


def process_and_add_chunks(
    source_text: str,
    chunk_size: int,
    batch_size: int,
    embedding_function: Callable[[List[str]], np.ndarray],
    vector_store: VectorStoreFRIDA,
) -> Tuple[int, int]:
    """Читает текст, разбивает на чанки, получает эмбеддинги и добавляет в хранилище."""
    chunked_text: List[str] = process_text_file(source_text, chunk_size)
    total_chunks = len(chunked_text)
    total_batches = (total_chunks - 1) // batch_size + 1

    print("📊 Статистика:")
    print(f"  • Всего чанков: {total_chunks}")
    print(f"  • Размер батча: {batch_size}")
    print(f"  • Количество батчей: {total_batches}\n")

    total_added = 0
    total_skipped = 0

    print("➕ Добавление чанков в хранилище:\n")

    for batch_index, batch_start in enumerate(
        tqdm(range(0, total_chunks, batch_size), total=total_batches, desc="Батчи", ncols=100, unit="batch")
    ):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = chunked_text[batch_start:batch_end]

        # Получаем эмбеддинги для батча
        batch_embeddings: np.ndarray = embedding_function(batch_chunks, device="cpu")

        # Метаданные
        batch_metadatas = [
            {
                "source": "simple_scraper",
                "url": source_text,
                "chunk_index": batch_start + i,
                "total_chunks": total_chunks,
                "batch_index": batch_index,
            }
            for i in range(len(batch_chunks))
        ]

        added, skipped = vector_store.add_chunks_batch(
            chunks=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

        total_added += added
        total_skipped += skipped

        tqdm.write(f"  📦 Батч {batch_index + 1}/{total_batches}: добавлено {added}, пропущено {skipped}")

    return total_added, total_skipped


def run_pipeline() -> None:
    """Настраивает и запускает основной процесс добавления чанков."""
    try:
        add_to_vector_store = True
        source_text = "llm.txt"
        CHUNK_SIZE = 1000
        BATCH_SIZE = 20

        if not add_to_vector_store:
            print("ℹ️ Добавление в хранилище отключено.")
            return

        vector_store = VectorStoreFRIDA()
        embedding_function: Callable[[List[str]], np.ndarray] = get_frida_embeddings

        total_added, total_skipped = process_and_add_chunks(
            source_text=source_text,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            embedding_function=embedding_function,
            vector_store=vector_store,
        )

        print("\n✅ Процесс завершен:")
        print(f"  • Добавлено чанков: {total_added}")
        print(f"  • Пропущено (уже существуют): {total_skipped}")
        print(f"  • Всего обработано: {total_added + total_skipped}")
        print("🎉 Все чанки успешно добавлены в векторное хранилище!")

    except Exception as e:
        print("❌ Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)
