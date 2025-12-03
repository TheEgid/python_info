import hashlib
import logging
from typing import List, Optional, Set, Tuple

import pymorphy3
from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode

from classes.PatchedClickhouseVectorStore import PatchedClickhouseVectorStore
from others.frida import FridaEmbedding

morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    """Базовая лемматизация текста"""
    return " ".join([morph.parse(word)[0].normal_form for word in text.split()])


def generate_stable_doc_id(doc: Document) -> str:
    """
    Генерирует СТАБИЛЬНЫЙ уникальный ID для документа.
    """
    # Приоритет 1: URL
    url = doc.metadata.get("url")
    if url:
        url_hash = hashlib.md5(str(url).encode()).hexdigest()
        return f"doc_url_{url_hash}"

    # Приоритет 2: File path
    file_path = doc.metadata.get("file_path") or getattr(doc, "extra_info", {}).get(
        "file_path"
    )
    if file_path:
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return f"doc_file_{path_hash}"

    # Приоритет 3: Содержимое
    content = doc.text[:1000] if len(doc.text) > 1000 else doc.text
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"doc_content_{content_hash}"


def get_existing_doc_ids(vector_store: PatchedClickhouseVectorStore) -> Set[str]:
    """
    Получает множество всех существующих ID документов из ClickHouse.
    """
    existing_ids = set()

    try:
        # Получаем все документы из ClickHouse
        query_result = vector_store.query(
            VectorStoreQuery( # type: ignore
                query_embedding=[0] * 1536,  # Пустой эмбеддинг
                similarity_top_k=10000,  # Большое число чтобы получить все
            )
        )

        # Собираем ID из узлов
        for node in query_result.nodes:
            if hasattr(node, "node_id") and node.node_id:
                existing_ids.add(node.node_id)
            elif hasattr(node, "metadata") and node.metadata:
                doc_id = node.metadata.get("doc_id")
                if doc_id:
                    existing_ids.add(str(doc_id))

        logging.info(f"📊 Найдено {len(existing_ids)} существующих ID в ClickHouse")
        return existing_ids

    except Exception as e:
        logging.error(f"❌ Ошибка при получении существующих ID: {e}")
        return set()


def add_documents_to_clickhouse(
    documents: List[Document],
    vector_store: PatchedClickhouseVectorStore,
) -> Tuple[Optional[PatchedClickhouseVectorStore], Optional[List[TextNode]]]:
    """
    Добавляет документы в ClickHouse с проверкой дубликатов.
    """
    try:
        if not documents:
            logging.warning("⚠️ Нет документов для добавления")
            return vector_store, []

        # Получаем существующие ID
        existing_ids = get_existing_doc_ids(vector_store)

        # Фильтруем новые документы
        new_documents = []
        duplicates_count = 0

        for doc in documents:
            doc_id = generate_stable_doc_id(doc)

            if doc_id not in existing_ids:
                doc.doc_id = doc_id
                doc.metadata["doc_id"] = doc_id
                new_documents.append(doc)
                logging.debug(f"✅ Новый документ: {doc_id}")
            else:
                duplicates_count += 1
                logging.debug(f"⏭️ Дубликат пропущен: {doc_id}")

        logging.info(f"📝 Новые: {len(new_documents)}, дубликаты: {duplicates_count}")

        if not new_documents:
            logging.info("✅ Все документы уже существуют")
            return vector_store, []

        # Добавляем в ClickHouse
        return fill_clickhouse_dataset(
            documents=new_documents, vector_store=vector_store
        )

    except Exception as e:
        logging.exception(f"❌ Ошибка при добавлении документов: {e}")
        return vector_store, []


def fill_clickhouse_dataset(
    documents: List[Document],
    vector_store: PatchedClickhouseVectorStore,
) -> Tuple[PatchedClickhouseVectorStore, List[TextNode]]:
    """
    Добавляет документы в ClickHouse векторное хранилище.
    """
    if not documents:
        logging.warning("⚠️ Нет документов для индексации")
        return vector_store, []

    # Инициализируем эмбеддинг модель
    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    nodes: List[TextNode] = []

    # Обрабатываем каждый документ
    for doc in documents:
        text_orig = (doc.text or "").strip()
        if not text_orig:
            logging.warning("⚠️ Пропускаем пустой документ")
            continue

        # Лемматизация
        text = lemmatize_text(text_orig)

        # Получаем эмбеддинг
        try:
            embedding = embed_model._get_text_embedding(text)
        except Exception as e:
            logging.error(f"❌ Ошибка получения эмбеддинга: {e}")
            continue

        # Используем уже установленный ID
        doc_id = doc.doc_id or generate_stable_doc_id(doc)

        # Подготавливаем метаданные
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata["doc_id"] = doc_id

        if not metadata.get("source"):
            metadata["source"] = getattr(doc, "extra_info", {}).get(
                "file_path", "unknown"
            )

        # Создаем узел
        node = TextNode(text=text, embedding=embedding, metadata=metadata, id_=doc_id)
        nodes.append(node)

    if not nodes:
        logging.error("❌ Нет валидных данных для добавления")
        return vector_store, []

    # Добавляем в ClickHouse
    try:
        vector_store.add(nodes)
        logging.info(f"✅ Добавлено {len(nodes)} документов в ClickHouse")
        return vector_store, nodes

    except Exception as e:
        logging.error(f"❌ Ошибка при сохранении в ClickHouse: {e}")
        return vector_store, []


def load_existing_clickhouse_db(
    connection_params: dict,
) -> Tuple[Optional[PatchedClickhouseVectorStore], Optional[List[TextNode]]]:
    """
    Загружает существующую базу ClickHouse.
    """
    try:
        vector_store = PatchedClickhouseVectorStore(**connection_params)

        # Получаем все документы для проверки
        query_result = vector_store.query(
            VectorStoreQuery(query_embedding=[0] * 1536, similarity_top_k=10000)
        )

        logging.info(f"📖 Загружено {len(query_result.nodes)} документов из ClickHouse")
        return vector_store, query_result.nodes

    except Exception as e:
        logging.exception(f"❌ Ошибка загрузки базы: {e}")
        return None, None


def load_or_fill_clickhouse(
    connection_params: dict,
    documents_source: Optional[str] = "supabase",
    limit: int = 25,
) -> Tuple[Optional[PatchedClickhouseVectorStore], Optional[List[TextNode]]]:
    """
    Загружает существующую базу или создает новую из источника.
    """
    try:
        # 1. Пытаемся загрузить существующую базу
        vector_store, nodes = load_existing_clickhouse_db(connection_params)

        if vector_store is not None:
            logging.info(
                f"📦 ClickHouse база загружена: {len(nodes) if nodes else 0} документов"
            )

            # Если нужно добавить новые документы из Supabase
            if documents_source == "supabase" and vector_store:
                logging.info("🔄 Проверяем наличие новых документов в Supabase...")

                from others.clickhouse_sources import get_documents_from_supabase

                new_documents = get_documents_from_supabase(
                    table_name="documents", limit=limit
                )

                if new_documents:
                    # Добавляем ТОЛЬКО новые
                    updated_store, updated_nodes = add_documents_to_clickhouse(
                        documents=new_documents, vector_store=vector_store
                    )
                    return updated_store, updated_nodes

            return vector_store, nodes

        # 2. Если базы нет - создаем новую
        logging.info("🆕 Создаем новую ClickHouse базу...")
        vector_store = PatchedClickhouseVectorStore(**connection_params)

        if documents_source == "supabase":
            from others.clickhouse_sources import get_documents_from_supabase

            documents = get_documents_from_supabase(table_name="documents", limit=limit)

            if documents:
                return add_documents_to_clickhouse(
                    documents=documents, vector_store=vector_store
                )

        logging.error("❌ Нет документов для создания базы")
        return None, None

    except Exception as e:
        logging.exception(f"❌ Ошибка в load_or_fill_clickhouse: {e}")
        return None, None


def display_clickhouse_contents(
    limit: int, connection_params: dict, show_vectors: bool = False
) -> None:
    """Показывает содержимое ClickHouse в читаемом формате."""
    try:
        vector_store, nodes = load_existing_clickhouse_db(connection_params)

        if vector_store is None or not nodes:
            print("❌ ClickHouse не доступен или пуст")
            return

        print(f"\n{'=' * 80}")
        print("СОДЕРЖИМОЕ CLICKHOUSE")
        print(f"{'=' * 80}")

        print("\n📊 Информация об узлах:")
        print(f"   • Количество узлов: {len(nodes)}")

        print(f"\n📄 Первые {limit} документов:")
        print(f"{'-' * 80}")

        for i, node in enumerate(nodes[:limit]):
            print(f"\n📖 Документ {i + 1}:")
            print(f"   ID: {node.node_id}")
            print(f"   Метаданные: {node.metadata}")

            text_preview = (
                node.text[:300] + "..." if len(node.text) > 300 else node.text
            )
            print(f"   Текст: {text_preview}")

            if hasattr(node, "embedding") and node.embedding:
                vector_length = len(node.embedding)
                print(f"   Размер вектора: {vector_length}")

                if show_vectors and vector_length > 0:
                    embedding_preview = node.embedding[:5]
                    print(f"   Вектор (первые 5 значений): {embedding_preview}")

            print(f"   {'-' * 40}")

        if len(nodes) > limit:
            print(f"\n⚠️  Показано {limit} из {len(nodes)} документов")

        # Статистика
        total_chars = sum(len(node.text) for node in nodes)
        avg_chars = total_chars / len(nodes)

        print("\n📈 Статистика текстов:")
        print(f"   • Общий объем текста: {total_chars} символов")
        print(f"   • Средняя длина документа: {avg_chars:.0f} символов")

    except Exception as e:
        print(f"❌ Ошибка при чтении ClickHouse: {e}")
