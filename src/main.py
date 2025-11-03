import logging
import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore", module=r"^pydantic(\.|$)")
warnings.filterwarnings("ignore", module=r"^pydantic_core(\.|$)")

from dotenv import load_dotenv  # noqa: E402
from llama_index.core import Settings, SimpleKeywordTableIndex, VectorStoreIndex  # noqa: E402
from llama_index.core.indices.composability import ComposableGraph  # noqa: E402
from llama_index.core.response_synthesizers import ResponseMode  # noqa: E402
from llama_index.llms.openrouter import OpenRouter  # noqa: E402

from others.frida import FridaEmbedding  # noqa: E402
from others.lance_dataset import display_lance_db_contents, load_or_fill_lance  # noqa: E402, F401
from others.lance_sources import sync_supabase_to_lance  # noqa: E402, F401
from others.tools import calculate_enhanced_similarity  # noqa: E402

logging.basicConfig(level=logging.INFO)
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_vector_index(vector_store: object, embed_model: FridaEmbedding) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


def create_keyword_index(nodes: list) -> SimpleKeywordTableIndex:
    return SimpleKeywordTableIndex(nodes=nodes)


def create_composable_graph(vector_index: VectorStoreIndex, keyword_index: SimpleKeywordTableIndex) -> ComposableGraph:
    return ComposableGraph.from_indices(
        VectorStoreIndex,
        children_indices=[vector_index, keyword_index],
        index_summaries=[
            "Векторный индекс для семантического поиска по LanceDB",
            "Таблица ключевых слов для быстрого поиска",
        ],
    )


def create_query_engine(graph: ComposableGraph) -> object:
    return graph.as_query_engine(
        similarity_top_k=3,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        streaming=False,
    )


def process_single_query(query_engine: object, query: str, query_index: int) -> None:
    """
    Обрабатывает один запрос и выводит результат.

    Args:
        query_engine (object): Query engine
        query (str): Текст запроса
        query_index (int): Номер запроса для вывода
    """
    print(f"\n{'=' * 60}")
    print(f"ЗАПРОС {query_index}: {query}")
    print(f"{'=' * 60}")

    try:
        response = query_engine.query(query)
        response_text = str(response).strip()

        if response_text:
            wrapped_text = textwrap.fill(response_text, width=80)
            print("ОТВЕТ:")
            print(wrapped_text)

            score = calculate_enhanced_similarity(query, response_text)
            print(f"\nScore схожести: {score:.3f}")
        else:
            print("❌ Пустой ответ от модели")

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")


def execute_queries(query_engine: object, queries: list[str]) -> None:
    """
    Выполняет список запросов через query engine.

    Args:
        query_engine (object): Query engine для выполнения запросов
        queries (list[str]): Список запросов
    """
    for i, query in enumerate(queries, 1):
        process_single_query(query_engine, query, i)


def setup_models_and_settings(api_key: str) -> Tuple[OpenRouter, FridaEmbedding]:
    def configure_llm_model(api_key: str) -> OpenRouter:
        return OpenRouter(
            model="tngtech/deepseek-r1t2-chimera:free",
            max_tokens=3000,
            temperature=0.3,
            api_key=api_key,
            context_window=4096,
            system_prompt="Ты - полезный AI-ассистент. Всегда отвечай на русском языке.",
        )

    llm = configure_llm_model(api_key)
    embed_model = FridaEmbedding()

    Settings.embed_model = embed_model
    Settings.llm = llm

    return llm, embed_model


def create_indices_and_graph(vector_store: object, nodes: list, embed_model: FridaEmbedding) -> ComposableGraph:
    """
    Создает все индексы и композиционный граф.

    Args:
        vector_store (object): Векторное хранилище
        nodes (list): Список узлов документов
        embed_model (FridaEmbedding): Embedding модель

    Returns:
        ComposableGraph: Готовый композиционный граф
    """
    vector_index = create_vector_index(vector_store, embed_model)
    keyword_index = create_keyword_index(nodes)
    return create_composable_graph(vector_index, keyword_index)


def run_rag_system(lance_db_path: Path) -> None:
    """
    Запускает полную систему RAG (Retrieval-Augmented Generation).

    Координирует работу всех компонентов системы:
    - Валидация окружения
    - Загрузка данных
    - Настройка моделей
    - Создание индексов
    - Выполнение запросов
    """

    vector_store, nodes = load_or_fill_lance(db_path=lance_db_path, documents_source="articles")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
        sys.exit(1)

    # Настройка моделей
    _llm, embed_model = setup_models_and_settings(api_key)

    # Создание индексов и графа
    graph = create_indices_and_graph(vector_store, nodes, embed_model)

    # Создание query engine
    query_engine = create_query_engine(graph)

    # Выполнение запросов

    queries = ["отличия эмблемы от логотипа"]

    execute_queries(query_engine, queries)


def main() -> None:
    """
    Основная функция программы с обработкой исключений.

    Обеспечивает graceful shutdown и логирование ошибок.
    """
    LANCE_DB_PATH = Path("./lance_db/lance_db")

    try:
        # 1. Заполнение LanceDB
        vector_store, nodes = sync_supabase_to_lance(lance_db_path=LANCE_DB_PATH)
        if nodes:
            print(f"\n✅ Синхронизировано {len(nodes)} документов")
        else:
            print("\n❌ Синхронизация не выполнена")

        # run_rag_system(lance_db_path=LANCE_DB_PATH)
        # # vector_store, nodes = sync_supabase_to_lance(lance_db_path=LANCE_DB_PATH)
        # if nodes:
        #     print(f"\n✅ Синхронизировано {len(nodes)} документов")
        # else:
        #     print("\n❌ Синхронизация не выполнена")

        display_lance_db_contents(limit=3,db_path=LANCE_DB_PATH, show_vectors=True)

    except KeyboardInterrupt:
        logging.info("🛑 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"❌ Критическая ошибка выполнения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
