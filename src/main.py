import sys
from typing import List, NoReturn

from classes import VectorStoreFRIDA  # noqa: F401
from others.frida import get_frida_embeddings
from others.helpers import multiply
from others.wiki_scraper import process_text_file, run_scraper  # noqa: F401


def main() -> NoReturn:
    try:
        result = multiply(300, 400)
        # run_scraper()
        print(result)

        file_path = "llm.txt"
        chunks: List[str] = process_text_file(file_path)

        print("Пример первого чанка:", chunks[0][:100], "...")

        print(f"Количество чанков: {len(chunks)}")

        chunk = chunks[-1]
        embedding = get_frida_embeddings([chunk])

        print(embedding.shape)  # (1, 1536)
        print(embedding.dtype)  # float32

        print(f"Размер эмбеддингов: {embedding.shape}")
        print(f"Тип эмбеддингов: {embedding.dtype}")
        print("Пример эмбеддинга:", embedding)

        SUPABASE_URL = "YOUR_SUPABASE_URL"  # noqa: F841
        SUPABASE_KEY = "YOUR_SUPABASE_KEY"  # noqa: F841

        # vs = VectorStoreFRIDA(SUPABASE_URL, SUPABASE_KEY)

        # 3. Добавление в Supabase
        # vs.add_chunks(chunks, embeddings)

        # sentences = ["Пример на русском", "Another example in English"]
        # emb = get_frida_embeddings(sentences)
        # print(emb.shape)  # ожидается (2, 1536)
        # print(emb.dtype)  # обычно float32

    except Exception as e:
        print("Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
