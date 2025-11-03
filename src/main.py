import logging
import os
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore", module=r"^pydantic(\.|$)")
warnings.filterwarnings("ignore", module=r"^pydantic_core(\.|$)")

from dotenv import load_dotenv  # noqa: E402
from llama_index.core import Settings, SimpleKeywordTableIndex, VectorStoreIndex  # noqa: E402
from llama_index.core.indices.composability import ComposableGraph  # noqa: E402
from llama_index.core.response_synthesizers import ResponseMode  # noqa: E402
from llama_index.llms.openrouter import OpenRouter  # noqa: E402

from others.frida import FridaEmbedding  # noqa: E402
from others.lance_dataset import load_or_fill_lance  # noqa: E402
from others.tools import calculate_enhanced_similarity  # noqa: E402

logging.basicConfig(level=logging.INFO)
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Остальной код без изменений...
def main() -> None:
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
            sys.exit(1)

        vector_store, nodes = load_or_fill_lance()

        if vector_store is None or not nodes:
            logging.error("❌ Векторный store не создан или нет документов для обработки.")
            sys.exit(1)

        llm = OpenRouter(
            model="tngtech/deepseek-r1t2-chimera:free",
            max_tokens=3000,
            temperature=0.3,
            api_key=api_key,
            context_window=4096,
            system_prompt="Ты - полезный AI-ассистент. Всегда отвечай на русском языке.",
        )

        embed_model = FridaEmbedding()

        Settings.embed_model = embed_model
        Settings.llm = llm

        vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        keyword_index = SimpleKeywordTableIndex(nodes=nodes)

        graph = ComposableGraph.from_indices(
            VectorStoreIndex,
            children_indices=[vector_index, keyword_index],
            index_summaries=[
                "Векторный индекс для семантического поиска по LanceDB",
                "Таблица ключевых слов для быстрого поиска",
            ],
        )

        query_engine = graph.as_query_engine(
            similarity_top_k=3,
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            streaming=False,
        )

        queries = ["какая рыба плавает быстро"]

        for i, my_query in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"ЗАПРОС {i}: {my_query}")
            print(f"{'=' * 60}")

            try:
                response = query_engine.query(my_query)
                response_text = str(response).strip()

                if response_text:
                    wrapped_text = textwrap.fill(response_text, width=80)
                    print("ОТВЕТ:")
                    print(wrapped_text)

                    score = calculate_enhanced_similarity(my_query, response_text)
                    print(f"\nScore схожести: {score:.3f}")
                else:
                    print("❌ Пустой ответ от модели")

            except Exception as e:
                logging.error(f"Ошибка при обработке запроса: {e}")
                continue

    except KeyboardInterrupt:
        logging.info("🛑 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"❌ Критическая ошибка выполнения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
