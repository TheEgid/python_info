import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import lancedb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from others.frida import FridaEmbedding

logging.basicConfig(level=logging.INFO)
load_dotenv()

LANCE_DB_PATH = Path("./lancedb/articles_index").resolve()


class PatchedLanceDBVectorStore(LanceDBVectorStore):
    """Исправляет несовместимость LlamaIndex ↔ LanceDB при возврате Series."""

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Переопределяем метод query для корректной обработки результатов."""
        try:
            # Выполняем поиск в LanceDB
            query_embedding = query.query_embedding
            if query_embedding is None:
                raise ValueError("Query embedding is required")

            # Конвертируем в numpy array если нужно
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            # Выполняем поиск
            results = self._table.search(query_embedding).limit(query.similarity_top_k or 10).to_pandas()

            # Обрабатываем результаты
            nodes = []
            similarities = []
            ids = []

            for idx, row in results.iterrows():
                # Извлекаем метаданные
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"__node_content__": metadata}
                elif isinstance(metadata, pd.Series):
                    metadata = metadata.to_dict()

                # Убеждаемся, что есть __node_content__
                text_content = row.get("text", "")
                if "__node_content__" not in metadata:
                    metadata["__node_content__"] = text_content

                # Создаем TextNode
                node = TextNode(text=text_content, metadata=metadata, id_=metadata.get("doc_id", f"doc_{idx}"))

                nodes.append(node)
                similarities.append(float(row.get("_distance", 0.0)))
                ids.append(metadata.get("doc_id", f"doc_{idx}"))

            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        except Exception as e:
            logging.error(f"Ошибка в query: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


def fill_lance_dataset(
    documents: List[Document],
    db_path: Path = LANCE_DB_PATH,
) -> Tuple[Optional[PatchedLanceDBVectorStore], List[TextNode]]:
    """Создает и заполняет LanceDB базу данных."""
    if not documents:
        logging.warning("⚠️ Нет документов для индексации.")
        return None, []

    logging.info(f"Создаём LanceDB по пути: {db_path}")

    # Инициализация LanceDB
    db = lancedb.connect(db_path)
    table_name = "articles"

    # Если таблица уже есть — удаляем (для чистого старта)
    if table_name in db.table_names():
        db.drop_table(table_name)

    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    # Подготавливаем данные
    table_data = []
    nodes = []

    for i, doc in enumerate(documents):
        text = doc.text.strip()
        if not text:  # Пропускаем пустые документы
            continue

        # Получаем embedding
        embedding = embed_model._get_text_embedding(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        # Создаем метаданные
        doc_id = doc.doc_id or f"doc_{i}"
        metadata = {
            "doc_id": doc_id,
            "__node_content__": text,
            "source": getattr(doc, "extra_info", {}).get("file_path", "unknown"),
        }

        # Добавляем в данные для таблицы
        table_data.append(
            {
                "id": doc_id,
                "text": text,
                "vector": embedding_array.tolist(),
                "metadata": json.dumps(metadata),  # Сериализуем метаданные
            }
        )

        # Создаем TextNode
        node = TextNode(text=text, embedding=embedding_array.tolist(), metadata=metadata, id_=doc_id)
        nodes.append(node)

    if not table_data:
        logging.warning("⚠️ Нет валидных документов для индексации.")
        return None, []

    # Создание таблицы
    table = db.create_table(
        table_name,
        data=table_data,
        mode="overwrite",
    )

    vector_store = PatchedLanceDBVectorStore(table=table)
    logging.info(f"✅ LanceDB создан: {len(table_data)} документов")

    return vector_store, nodes


def load_or_fill_lance(
    db_path: Path = LANCE_DB_PATH,
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    """Загружает существующую базу или создает новую."""
    try:
        db = lancedb.connect(db_path)
        table_name = "articles"

        if table_name in db.table_names():
            logging.info(f"📦 LanceDB '{table_name}' найден, загружаем...")
            table = db.open_table(table_name)
            vector_store = PatchedLanceDBVectorStore(table=table)

            # Логируем информацию о таблице
            logging.info(f"📜 Lance schema: {table.schema}")
            sample_data = table.to_pandas().head(1)
            if not sample_data.empty:
                logging.info(f"📊 Пример строки: {sample_data.to_dict(orient='records')}")

            return vector_store, None
        else:
            logging.info("🆕 LanceDB не найден, создаём заново...")

            # Проверяем наличие директории с документами
            articles_dir = Path("articles/")
            if not articles_dir.exists():
                logging.error(f"❌ Директория {articles_dir} не найдена!")
                return None, None

            documents = SimpleDirectoryReader(str(articles_dir)).load_data()
            if not documents:
                logging.error("❌ Не найдено документов для индексации!")
                return None, None

            vector_store, nodes = fill_lance_dataset(documents, db_path=db_path)
            return vector_store, nodes

    except Exception as e:
        logging.exception(f"❌ Ошибка при работе с LanceDB: {e}")
        return None, None


def main() -> None:
    """Основная функция."""
    try:
        # Проверяем наличие API ключа
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
            return

        vector_store, nodes = load_or_fill_lance()

        if vector_store is None:
            logging.error("❌ Векторный store не создан, возможно, нет документов.")
            return

        # Настройка модели
        embed_model = FridaEmbedding()
        Settings.embed_model = embed_model

        llm = OpenRouter(
            model="z-ai/glm-4.5-air:free",
            max_tokens=512,
            context_window=4096,
            api_key=api_key,
        )
        Settings.llm = llm

        # Создание индекса
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )

        # Создание query engine
        query_engine = index.as_query_engine(
            response_mode="compact",
            verbose=True,
        )

        logging.info("🔍 Выполнение тестового запроса...")
        response = query_engine.query("Расскажи о телескопах")
        print("\n" + "=" * 50)
        print("ОТВЕТ:")
        print("=" * 50)
        print(str(response))
        print("=" * 50)

    except Exception as e:
        logging.exception(f"❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
