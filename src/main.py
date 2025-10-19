import sys
from typing import NoReturn

from others.frida import get_frida_embeddings
from others.helpers import multiply
from others.wiki_scraper import run_scraper  # noqa: F401


def main() -> NoReturn:
    try:
        result = multiply(300, 400)
        # run_scraper()
        print(result)

        sentences = ["Пример на русском", "Another example in English"]
        emb = get_frida_embeddings(sentences)
        print(emb.shape)  # ожидается (2, 1536)
        print(emb.dtype)  # обычно float32

    except Exception as e:
        print("Ошибка запуска:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
