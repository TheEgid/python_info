# file: rag_openrouter_frida.py
import logging
import os
from typing import List

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

from others.frida import get_frida_embeddings


class FridaEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        return get_frida_embeddings([query])[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return get_frida_embeddings([text])[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


logger = logging.getLogger(__name__)


def main() -> int:
    """Точка входа в приложение."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        documents = SimpleDirectoryReader("articles/").load_data()

        vector_store = DeepLakeVectorStore(
            dataset_path="./deeplake_datasets/articles_index",
            overwrite=True,
        )

        embed_model = FridaEmbedding()

        # Настройка OpenRouter LLM
        llm = OpenRouter(
            model="z-ai/glm-4-5-air:free",  # Или любая другая модель с OpenRouter
            max_tokens=512,
            context_window=4096,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        Settings.llm = llm
        Settings.embed_model = embed_model

        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
        )

        query_engine = index.as_query_engine(response_mode="compact")
        response = query_engine.query("Расскажи о телескопах")

        print("\n=== Ответ ===\n", response.response)
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Процесс прерван пользователем")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main()
