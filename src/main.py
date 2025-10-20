import sys
from typing import Callable, List, NoReturn

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from classes.VectorStoreFRIDA import VectorStoreFRIDA
from others.frida import get_frida_embeddings
from others.wiki_scraper import process_text_file


def main() -> NoReturn:
    load_dotenv()
    try:
        add_to_vector_store = True
        source_text = "llm.txt"
        CHUNK_SIZE = 1000
        BATCH_SIZE = 20

        if add_to_vector_store:
            # Чтение текста и разбиение на чанки
            chunked_text: List[str] = process_text_file(source_text, CHUNK_SIZE)
            total_chunks = len(chunked_text)
            total_batches = (total_chunks - 1) // BATCH_SIZE + 1

            print("📊 Статистика:")
            print(f"  • Всего чанков: {total_chunks}")
            print(f"  • Размер батча: {BATCH_SIZE}")
            print(f"  • Количество батчей: {total_batches}\n")

            vector_store = VectorStoreFRIDA()
            embedding_function: Callable[[List[str]], np.ndarray] = get_frida_embeddings

            total_added = 0
            total_skipped = 0

            # Обработка батчами
            print("➕ Добавление чанков в хранилище:\n")
            for batch_index, batch_start in enumerate(
                tqdm(range(0, total_chunks, BATCH_SIZE), total=total_batches, desc="Батчи", ncols=100, unit="batch")
            ):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                batch_chunks = chunked_text[batch_start:batch_end]

                # Получаем эмбеддинги для батча
                batch_embeddings: np.ndarray = embedding_function(batch_chunks, device="cpu")

                # Подготавливаем метаданные для батча
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

                # Добавляем батч в хранилище с проверкой существования
                added, skipped = vector_store.add_chunks_batch(
                    chunks=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )

                total_added += added
                total_skipped += skipped

                # Логирование после прогрессбара
                tqdm.write(
                    f"  📦 Батч {batch_index + 1}/{total_batches}: "
                    f"добавлено {added}, пропущено {skipped}"
                )

            print("\n✅ Процесс завершен:")
            print(f"  • Добавлено чанков: {total_added}")
            print(f"  • Пропущено (уже существуют): {total_skipped}")
            print(f"  • Всего обработано: {total_added + total_skipped}")
            print("🎉 Все чанки успешно добавлены в векторное хранилище!")

    except Exception as e:
        print("❌ Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
