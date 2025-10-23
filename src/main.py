import logging
import os
import sys
import textwrap

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openrouter import OpenRouter

from others.frida import FridaEmbedding
from others.lance_dataset import load_or_fill_lance
from others.tools import calculate_enhanced_similarity

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main() -> None:
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
            return

        vector_store, nodes = load_or_fill_lance()

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

        vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        keyword_index = SimpleKeywordTableIndex.from_documents(nodes)

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

        my_query = "Какая рыба плавает быстрее всех?"

        response = query_engine.query(my_query)
        response_text = str(response).strip()
        wrapped_text = textwrap.fill(response_text, width=120)

        print("\n" + "=" * 50)
        print("ОТВЕТ:")
        print("=" * 50)
        print(wrapped_text)
        print("=" * 50)

        score = calculate_enhanced_similarity(my_query, response_text)
        print(f"Best Cosine Similarity Score: {score:.3f}")

    except KeyboardInterrupt:
        logging.info("🛑 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
