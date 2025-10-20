import sys
from typing import Callable, List, NoReturn

import numpy as np
from dotenv import load_dotenv

from classes.VectorStoreFRIDA import VectorStoreFRIDA
from others.frida import get_frida_embeddings
from others.wiki_scraper import process_text_file


def main() -> NoReturn:
    load_dotenv()
    try:
        add_to_vector_store = True
        source_text = "llm.txt"
        CHUNK_SIZE = 1000

        if add_to_vector_store:
            # Чтение текста и разбиение на чанки
            chunked_text: List[str] = process_text_file(source_text, CHUNK_SIZE)
            total_chunks = len(chunked_text)

            print(f"Всего чанков: {total_chunks}\n")

            vector_store = VectorStoreFRIDA()
            embedding_function: Callable[[List[str]], np.ndarray] = get_frida_embeddings

            for i, chunk in enumerate(chunked_text):
                embedding_data: np.ndarray = embedding_function([chunk], device="cpu")  # shape (1, 1536)

                # Метаданные с информацией о чанке
                metadata: dict = {
                    "source": "simple_scraper",
                    "url": source_text,  # Можно заменить на реальный URL
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                }

                vector_store.add_chunk(chunk, embedding_data, metadata=metadata)

                # Печать примера первых 5 значений эмбеддинга для первых 5 чанков
                if i < 5:
                    print(f"Чанк {i + 1}: {chunk[:60]}...")
                    print(f"Эмбеддинг shape: {embedding_data.shape}, dtype: {embedding_data.dtype}")
                    print(f"Пример эмбеддинга (первые 5 значений): {embedding_data[0][:5]}\n")

                # Остановить после 8-го чанка (для теста)
                if i == 8:
                    break

    except Exception as e:
        print("Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
