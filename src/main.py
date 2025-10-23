import logging
import os
import sys
import textwrap

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openrouter import OpenRouter

from others.frida import FridaEmbedding
from others.lance_dataset import load_or_fill_lance

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main() -> None:
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
            return

        vector_store, nodes = load_or_fill_lance()

        test_embedding = vector_store._compute_query_embedding("test")
        logging.info(f"Embedding shape: {test_embedding.shape}")
        logging.info(f"Embedding norm: {np.linalg.norm(test_embedding)}")  # Должно быть ~1.0

        if vector_store is None:
            logging.error("❌ Векторный store не создан, возможно, нет документов.")
            return

        llm = OpenRouter(
            model="z-ai/glm-4.5-air:free",
            max_tokens=3000,
            temperature=0.3,
            api_key=api_key,
            context_window=4096,
        )
        embed_model = FridaEmbedding()

        Settings.embed_model = embed_model
        Settings.llm = llm

        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

        # Получаем ответ на вопрос
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            streaming=False,
        )

        response = query_engine.query("Какая рыба плавает быстрее всех?")
        response_text = str(response).strip()
        wrapped_text = textwrap.fill(response_text, width=120)

        print("\n" + "=" * 50)
        print("ОТВЕТ:")
        print("=" * 50)
        print(wrapped_text)
        print("=" * 50)

        # Дополнительная информация
        print("\n📊 Статистика ответа:")
        print(f"   Общая длина: {len(response_text)} символов")
        print(f"   Количество строк после форматирования: {wrapped_text.count(chr(10)) + 1}")

    except KeyboardInterrupt:
        logging.info("🛑 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
