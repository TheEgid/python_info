import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

import lancedb
import numpy as np
import pymorphy3
from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode

from classes.PatchedLanceDBVectorStore import PatchedLanceDBVectorStore
from others.frida import FridaEmbedding

morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    """Базовая лемматизация текста"""
    return " ".join([morph.parse(word)[0].normal_form for word in text.split()])


def generate_stable_doc_id(doc: Document) -> str:
    """
    Генерирует СТАБИЛЬНЫЙ уникальный ID для документа.
    Идемпотентная функция - один документ всегда получает один ID.

    Приоритет:
    1. URL (если есть)
    2. File path (если есть)
    3. Хеш содержимого
    """
    # Приоритет 1: URL
    url = doc.metadata.get("url")
    if url:
        url_hash = hashlib.md5(str(url).encode()).hexdigest()
        return f"doc_url_{url_hash}"

    # Приоритет 2: File path
    file_path = doc.metadata.get("file_path") or getattr(doc, "extra_info", {}).get("file_path")
    if file_path:
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return f"doc_file_{path_hash}"

    # Приоритет 3: Содержимое
    # Используем только первые 1000 символов для стабильности
    content = doc.text[:1000] if len(doc.text) > 1000 else doc.text
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return f"doc_content_{content_hash}"


def get_existing_doc_ids(db_path: Path, table_name: str = "novaya") -> Set[str]:
    """
    Получает множество всех существующих ID документов из LanceDB.

    Returns:
        Множество существующих ID
    """
    existing_ids = set()

    try:
        db = lancedb.connect(str(db_path))

        if table_name not in db.table_names():
            logging.info(f"📌 Таблица '{table_name}' не существует, возвращаем пустое множество")
            return existing_ids

        table = db.open_table(table_name)
        df = table.to_pandas()

        # Собираем ID из колонки "id"
        if "id" in df.columns:
            existing_ids.update(df["id"].astype(str).tolist())

        # Также проверяем метаданные на всякий случай
        if "metadata" in df.columns:
            for metadata_value in df["metadata"]:
                try:
                    if isinstance(metadata_value, str):
                        metadata = json.loads(metadata_value)
                    else:
                        metadata = metadata_value

                    if isinstance(metadata, dict):
                        doc_id = metadata.get("doc_id")
                        if doc_id:
                            existing_ids.add(str(doc_id))
                except Exception as e:
                    logging.debug(f"⚠️ Ошибка парсинга метаданных: {e}")
                    continue

        logging.info(f"📊 Найдено {len(existing_ids)} существующих ID в базе")

        # Логируем первые несколько ID для отладки
        if existing_ids:
            sample_ids = list(existing_ids)[:5]
            logging.debug(f"🔍 Примеры существующих ID: {sample_ids}")

        return existing_ids

    except Exception as e:
        logging.error(f"❌ Ошибка при получении существующих ID: {e}")
        return set()


def build_documents_from_chunks(documents: List[Document]) -> List[Document]:
    """
    Собирает документы из чанков по URL.
    Документы без URL остаются отдельными.
    """
    try:
        chunks_by_url = {}
        standalone_docs = []

        for doc in documents:
            url = doc.metadata.get("url")

            if url:
                if url not in chunks_by_url:
                    chunks_by_url[url] = []
                chunks_by_url[url].append(doc)
            else:
                standalone_docs.append(doc)

        # Собираем чанки по URL
        assembled_documents = []

        for url, chunks in chunks_by_url.items():
            # Сортируем по chunk_index
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x.metadata.get("chunk_index", 0)
            )

            # Объединяем текст
            combined_text = " ".join(chunk.text for chunk in sorted_chunks)

            # Создаем метаданные
            base_metadata = sorted_chunks[0].metadata.copy()
            base_metadata.update({
                "chunks_count": len(chunks),
                "assembled_from_chunks": True,
                "url": url
            })

            assembled_doc = Document(
                text=combined_text,
                metadata=base_metadata
            )
            assembled_documents.append(assembled_doc)

        result = assembled_documents + standalone_docs
        logging.info(
            f"🧩 Собрано {len(assembled_documents)} документов из чанков, "
            f"{len(standalone_docs)} отдельных документов"
        )

        return result

    except Exception as e:
        logging.exception(f"❌ Ошибка при сборке документов: {e}")
        return documents


def filter_new_documents(
    documents: List[Document],
    existing_ids: Set[str]
) -> Tuple[List[Document], int]:
    """
    Фильтрует документы, оставляя только новые.

    Args:
        documents: Список документов для проверки
        existing_ids: Множество существующих ID

    Returns:
        Tuple[новые_документы, количество_дубликатов]
    """
    new_documents = []
    duplicates_count = 0

    for doc in documents:
        # Генерируем стабильный ID
        doc_id = generate_stable_doc_id(doc)

        # Проверяем наличие
        if doc_id not in existing_ids:
            # Устанавливаем ID в документ
            doc.doc_id = doc_id
            doc.metadata["doc_id"] = doc_id

            new_documents.append(doc)
            logging.debug(f"✅ Новый документ: {doc_id}")
        else:
            duplicates_count += 1
            logging.debug(f"⏭️ Дубликат пропущен: {doc_id}")

    logging.info(
        f"📝 Отфильтровано: {len(new_documents)} новых, "
        f"{duplicates_count} дубликатов"
    )

    return new_documents, duplicates_count


def add_documents_to_lance(
    documents: List[Document],
    db_path: Path,
    table_name: str = "novaya",
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    """
    Добавляет ТОЛЬКО новые документы в LanceDB.
    Гарантия отсутствия дубликатов через проверку ID.
    """
    try:
        if not documents:
            logging.warning("⚠️ Нет документов для добавления")
            return None, None

        # 1. Собираем чанки в полные документы
        processed_documents = build_documents_from_chunks(documents)

        if not processed_documents:
            logging.error("❌ Нет документов после обработки чанков")
            return None, None

        # 2. Получаем существующие ID из базы
        existing_ids = get_existing_doc_ids(db_path, table_name)

        # 3. Фильтруем новые документы
        new_documents, duplicates_count = filter_new_documents(
            processed_documents,
            existing_ids
        )

        if not new_documents:
            logging.info("✅ Все документы уже существуют, ничего не добавляем")
            return load_existing_lance_db(db_path, table_name)

        logging.info(
            f"📝 Будет добавлено {len(new_documents)} новых документов "
            f"(обработано: {len(processed_documents)}, дубликатов: {duplicates_count})"
        )

        # 4. Добавляем ТОЛЬКО новые документы
        return fill_lance_dataset(
            documents=new_documents,
            db_path=db_path,
            table_name=table_name,
            force_recreate=False
        )

    except Exception as e:
        logging.exception(f"❌ Ошибка при добавлении документов: {e}")
        return None, None


def fill_lance_dataset(
    documents: List[Document],
    db_path: Path,
    table_name: str = "novaya",
    force_recreate: bool = False,
) -> Tuple[Optional[PatchedLanceDBVectorStore], List[TextNode]]:
    """
    Создает или дополняет LanceDB dataset.
    В режиме append добавляет документы в существующую таблицу.
    """
    if not documents:
        logging.warning("⚠️ Нет документов для индексации")
        return None, []

    logging.info(f"Подключаемся к LanceDB: {db_path}")
    db = lancedb.connect(str(db_path))

    table_exists = table_name in db.table_names()

    # Определяем режим работы
    if force_recreate and table_exists:
        logging.warning(f"🔄 Принудительное пересоздание таблицы '{table_name}'")
        db.drop_table(table_name)
        table_exists = False

    # Инициализируем эмбеддинг модель
    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    table_data = []
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
            embedding_array = np.array(embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"❌ Ошибка получения эмбеддинга: {e}")
            continue

        # Используем уже установленный ID (из filter_new_documents)
        doc_id = doc.doc_id or generate_stable_doc_id(doc)

        # Подготавливаем метаданные
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata["doc_id"] = doc_id

        if not metadata.get("source"):
            metadata["source"] = getattr(doc, "extra_info", {}).get("file_path", "unknown")

        # Добавляем в структуры данных
        table_data.append({
            "id": doc_id,
            "text": text,
            "embedding": embedding_array.tolist(),
            "metadata": json.dumps(metadata, ensure_ascii=False)
        })

        nodes.append(TextNode(
            text=text,
            embedding=embedding_array.tolist(),
            metadata=metadata,
            id_=doc_id
        ))

    if not table_data:
        logging.error("❌ Нет валидных данных для добавления")
        return None, []

    # Сохраняем в LanceDB
    try:
        if table_exists:
            # Режим APPEND - добавляем к существующим данным
            table = db.open_table(table_name)
            table.add(table_data)
            logging.info(f"✅ Добавлено {len(table_data)} документов (режим: append)")
        else:
            # Создаем новую таблицу
            table = db.create_table(table_name, data=table_data, mode="overwrite")
            logging.info(f"✅ Создана новая таблица с {len(table_data)} документами")

        vector_store = PatchedLanceDBVectorStore(table=table)
        return vector_store, nodes

    except Exception as e:
        logging.error(f"❌ Ошибка при сохранении в LanceDB: {e}")
        return None, []


def load_existing_lance_db(
    db_path: Path,
    table_name: str = "novaya"
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    """
    Загружает существующую базу LanceDB.
    """
    try:
        db = lancedb.connect(str(db_path))

        if table_name not in db.table_names():
            logging.info(f"📌 Таблица '{table_name}' не найдена")
            return None, None

        table = db.open_table(table_name)
        vector_store = PatchedLanceDBVectorStore(table=table)
        nodes: List[TextNode] = []

        df = table.to_pandas()

        for record in df.to_dict(orient="records"):
            metadata = record.get("metadata", {})

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}

            text = metadata.get("content", record.get("text", ""))

            nodes.append(TextNode(
                text=text,
                metadata=metadata,
                embedding=record.get("embedding"),
                id_=metadata.get("doc_id")
            ))

        logging.info(f"📖 Загружено {len(nodes)} документов из базы")
        return vector_store, nodes

    except Exception as e:
        logging.exception(f"❌ Ошибка загрузки базы: {e}")
        return None, None


def load_or_fill_lance(
    db_path: Path,
    documents_source: Optional[str] = "supabase",
    limit: int = 25
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    """
    Загружает существующую базу или создает новую из Supabase.
    Автоматически проверяет дубликаты перед добавлением.
    """
    try:
        table_name = "novaya"

        # 1. Пытаемся загрузить существующую базу
        vector_store, nodes = load_existing_lance_db(db_path, table_name)

        if vector_store is not None and nodes:
            logging.info(f"📦 База '{table_name}' загружена: {len(nodes)} документов")

            # Если нужно добавить новые документы из Supabase
            if documents_source == "supabase":
                logging.info("🔄 Проверяем наличие новых документов в Supabase...")

                from others.lance_sources import get_documents_from_supabase

                # Загружаем последние документы
                new_documents = get_documents_from_supabase(
                    table_name=table_name,
                    limit=limit * 2,  # Берем с запасом
                )

                # Сортируем и берем топ-N
                new_documents = sorted(
                    new_documents,
                    key=lambda d: d.metadata.get("created_at", 0),
                    reverse=True
                )[:limit]

                if new_documents:
                    # Добавляем ТОЛЬКО новые
                    updated_store, updated_nodes = add_documents_to_lance(
                        documents=new_documents,
                        db_path=db_path,
                        table_name=table_name
                    )

                    if updated_store and updated_nodes:
                        return updated_store, updated_nodes

            return vector_store, nodes

        # 2. Если базы нет - создаем с нуля
        logging.info("🆕 База не найдена, создаем новую...")

        if documents_source == "supabase":
            from others.lance_sources import get_documents_from_supabase

            documents = get_documents_from_supabase(
                table_name=table_name,
                limit=limit * 2
            )

            documents = sorted(
                documents,
                key=lambda d: d.metadata.get("created_at", 0),
                reverse=True
            )[:limit]

            if not documents:
                logging.error("❌ Нет документов в Supabase")
                return None, None

            return add_documents_to_lance(
                documents=documents,
                db_path=db_path,
                table_name=table_name
            )

        logging.error("❌ Неизвестный источник документов")
        return None, None

    except Exception as e:
        logging.exception(f"❌ Ошибка в load_or_fill_lance: {e}")
        return None, None


def display_lance_db_contents(limit: int, db_path: Path, show_vectors: bool = False) -> None:
    """Показывает содержимое LanceDB в читаемом формате."""
    try:
        vector_store, nodes = load_or_fill_lance(db_path)

        if vector_store is None:
            print("❌ Vector store не доступен")
            return

        if not nodes:
            print("❌ Нет узлов в базе данных")
            return

        print(f"\n{'=' * 80}")
        print("СОДЕРЖИМОЕ LANCE DB")
        print(f"{'=' * 80}")

        try:
            table = vector_store.table
            df = table.to_pandas()
            print("📊 Общая информация из таблицы:")
            print(f"   • Количество записей в таблице: {len(df)}")
            print(f"   • Колонки: {list(df.columns)}")
        except Exception as e:
            print(f"   ⚠️ Не удалось получить данные таблицы: {e}")

        print("\n📊 Информация об узлах:")
        print(f"   • Количество узлов: {len(nodes)}")

        print(f"\n📄 Первые {limit} документов:")
        print(f"{'-' * 80}")

        for i, node in enumerate(nodes[:limit]):
            print(f"\n📖 Документ {i + 1}:")
            print(f"   ID: {node.node_id}")
            print(f"   Метаданные: {node.metadata}")

            text_preview = node.text[:300] + "..." if len(node.text) > 300 else node.text
            print(f"   Текст: {text_preview}")

            if hasattr(node, "embedding") and node.embedding is not None:
                vector_length = len(node.embedding) if node.embedding else 0
                print(f"   Размер вектора: {vector_length}")

                if show_vectors and vector_length > 0:
                    embedding_preview = node.embedding[:5]
                    print(f"   Вектор (первые 5 значений): {embedding_preview}")
            else:
                print("   Вектор: не задан")

            print(f"   {'-' * 40}")

        if len(nodes) > limit:
            print(f"\n⚠️  Показано {limit} из {len(nodes)} документов")

        if nodes:
            metadata_keys = set()
            sources = set()
            for node in nodes:
                if hasattr(node, "metadata") and node.metadata:
                    metadata_keys.update(node.metadata.keys())
                    if "source" in node.metadata:
                        sources.add(node.metadata["source"])

            if metadata_keys:
                print(f"\n🏷️  Ключи метаданных: {list(metadata_keys)}")
            if sources:
                print(f"📁 Источники документов: {list(sources)}")

        total_chars = sum(len(node.text) for node in nodes) if nodes else 0
        avg_chars = total_chars / len(nodes) if nodes else 0

        print("\n📈 Статистика текстов:")
        print(f"   • Общий объем текста: {total_chars} символов")
        print(f"   • Средняя длина документа: {avg_chars:.0f} символов")

    except Exception as e:
        print(f"❌ Ошибка при чтении LanceDB: {e}")
        logging.exception("Подробности ошибки:")


# def quick_lance_db_check(db_path: Path) -> None:
#     """Быстрая проверка содержимого LanceDB без запуска всей RAG системы."""
#     print("🔍 Быстрая проверка LanceDB...")
#     display_lance_db_contents(limit=5, db_path=db_path)
