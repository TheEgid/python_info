import sys
from typing import List, NoReturn

import numpy as np
from dotenv import load_dotenv

from classes.VectorStoreFRIDA import VectorStoreFRIDA
from others.frida import get_frida_embeddings
from others.wiki_scraper import process_text_file, run_scraper  # noqa: F401


def main() -> NoReturn:
    load_dotenv()
    try:
        file_path = "llm.txt"
        chunks: List[str] = process_text_file(file_path)

        print(f"Всего чанков: {len(chunks)}\n")

        vs = VectorStoreFRIDA()

        # Берем первые 10 чанков
        for i, chunk in enumerate(chunks[:5]):
            embedding: np.ndarray = get_frida_embeddings([chunk])

            print(f"Чанк {i + 1}: {chunk[:60]}...")
            print(f"Эмбеддинг shape: {embedding.shape}, dtype: {embedding.dtype}")
            print(f"Пример эмбеддинга (первые 5 значений): {embedding[0][:5]}\n")

            vs.add_chunk(chunk, embedding)

        # sentences = ["Пример на русском", "Another example in English"]
        # emb = get_frida_embeddings(sentences)
        # print(emb.shape)  # ожидается (2, 1536)
        # print(emb.dtype)  # обычно float32

    except Exception as e:
        print("Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
