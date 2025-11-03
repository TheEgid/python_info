import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import lancedb
import numpy as np
import pymorphy3
from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode

from classes.PatchedLanceDBVectorStore import PatchedLanceDBVectorStore
from others.frida import FridaEmbedding
from src.others.lance_sources import get_documents_from_directory, get_documents_from_sql

LANCE_DB_PATH = Path("./lancedb/articles_index").resolve()

morph = pymorphy3.MorphAnalyzer()


def lemmatize_text(text: str) -> str:
    """Базовая лемматизация текста"""
    return " ".join([morph.parse(word)[0].normal_form for word in text.split()])


def fill_lance_dataset(
    documents: List[Document], db_path: Path = LANCE_DB_PATH, table_name: str = "articles"
) -> Tuple[Optional[PatchedLanceDBVectorStore], List[TextNode]]:
    """Создает или обновляет LanceDB dataset из документов"""
    if not documents:
        logging.warning("⚠️ Нет документов для индексации.")
        return None, []

    logging.info(f"Создаём LanceDB по пути: {db_path}")
    db = lancedb.connect(db_path)

    if table_name in db.table_names():
        logging.info(f"🔄 Обновляем существующую таблицу: {table_name}")
        db.drop_table(table_name)

    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    table_data = []
    nodes: List[TextNode] = []

    for i, doc in enumerate(documents):
        text_orig = (doc.text or "").strip()
        if not text_orig:
            continue

        text = lemmatize_text(text_orig)
        embedding = embed_model._get_text_embedding(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        doc_id = doc.doc_id or f"doc_{i}"

        # Расширяем метаданные с информацией об источнике
        metadata = doc.metadata or {}
        if not metadata.get("source"):
            metadata["source"] = getattr(doc, "extra_info", {}).get("file_path", "unknown")
        metadata["doc_id"] = doc_id

        table_data.append(
            {
                "id": doc_id,
                "text": text,  # сохраняем лемматизированный текст
                "embedding": embedding_array.tolist(),
                "metadata": json.dumps(metadata),
            }
        )

        nodes.append(TextNode(text=text, embedding=embedding_array.tolist(), metadata=metadata, id_=doc_id))

    if not table_data:
        logging.warning("⚠️ Нет валидных документов для индексации.")
        return None, []

    table = db.create_table(table_name, data=table_data, mode="overwrite")
    vector_store = PatchedLanceDBVectorStore(table=table)

    logging.info(f"✅ LanceDB создан: {len(table_data)} документов")
    return vector_store, nodes


def load_or_fill_lance(
    db_path: Path = LANCE_DB_PATH,
    documents_source: Optional[str] = "directory",
    table_name: str = "articles",
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    """
    Загружает существующую базу LanceDB или создает новую из указанного источника

    Args:
        db_path: Путь к базе данных LanceDB
        documents_source: Источник документов ("directory", "sql")
        custom_documents_loader: Кастомная функция для загрузки документов
        table_name: Название таблицы в базе данных

    Returns:
        Tuple[vector_store, nodes] или (None, None) при ошибке
    """
    try:
        db = lancedb.connect(db_path)

        if table_name in db.table_names():
            logging.info(f"📦 LanceDB '{table_name}' найден, проверяем данные...")
            table = db.open_table(table_name)
            df = table.to_pandas()

            if df.empty:
                logging.warning(f"⚠️ Таблица '{table_name}' существует, но пуста. Пересоздаём...")
                db.drop_table(table_name)
            else:
                vector_store = PatchedLanceDBVectorStore(table=table)
                nodes: List[TextNode] = []
                for record in df.to_dict(orient="records"):
                    meta = record.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {"content": meta}

                    text = meta.get("content", record.get("text", ""))
                    nodes.append(
                        TextNode(text=text, metadata=meta, embedding=record.get("embedding"), id_=meta.get("doc_id"))
                    )
                return vector_store, nodes

        logging.info("🆕 LanceDB не найден, создаём заново...")

        # Определяем источник документов
        documents = []

        if documents_source == "sql":
            logging.info("🗃️ Загружаем документы из SQL базы...")
            documents = get_documents_from_sql()
        else:  # directory - значение по умолчанию
            logging.info("📁 Загружаем документы из директории...")
            documents = get_documents_from_directory()

        if not documents:
            logging.error("❌ Не найдено документов для индексации!")
            return None, None

        vector_store, nodes = fill_lance_dataset(documents, db_path=db_path, table_name=table_name)
        return vector_store, nodes

    except Exception as e:
        logging.exception(f"❌ Ошибка при работе с LanceDB: {e}")
        return None, None


def display_lance_db_contents(limit: int = 10, show_vectors: bool = False) -> None:
    """
    Показывает содержимое LanceDB в читаемом формате.
    """
    try:
        vector_store, nodes = load_or_fill_lance()

        if vector_store is None:
            print("❌ Vector store не доступен")
            return

        if not nodes:
            print("❌ Нет узлов в базе данных")
            return

        print(f"\n{'=' * 80}")
        print("СОДЕРЖИМОЕ LANCE DB")
        print(f"{'=' * 80}")

        # Получаем сырые данные из таблицы для дополнительной информации
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

        # Показываем первые N узлов
        print(f"\n📄 Первые {limit} документов:")
        print(f"{'-' * 80}")

        for i, node in enumerate(nodes[:limit]):
            print(f"\n📖 Документ {i + 1}:")
            print(f"   ID: {node.node_id}")
            print(f"   Метаданные: {node.metadata}")

            # Отображаем текст (уже лемматизированный в вашей реализации)
            text_preview = node.text[:300] + "..." if len(node.text) > 300 else node.text
            print(f"   Текст (лемматизированный): {text_preview}")

            # Информация о векторе
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

        # Статистика по метаданным
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

        # Дополнительная статистика
        total_chars = sum(len(node.text) for node in nodes) if nodes else 0
        avg_chars = total_chars / len(nodes) if nodes else 0

        print("\n📈 Статистика текстов:")
        print(f"   • Общий объем текста: {total_chars} символов")
        print(f"   • Средняя длина документа: {avg_chars:.0f} символов")

    except Exception as e:
        print(f"❌ Ошибка при чтении LanceDB: {e}")
        logging.exception("Подробности ошибки:")


def display_detailed_document(doc_id: Optional[str] = None, index: Optional[int] = None) -> None:
    """
    Показывает детальную информацию о конкретном документе.
    """
    try:
        _vector_store, nodes = load_or_fill_lance()

        if not nodes:
            print("❌ Нет документов в базе")
            return

        target_node = None

        if doc_id:
            for node in nodes:
                if node.node_id == doc_id:
                    target_node = node
                    break
        elif index is not None and 0 <= index < len(nodes):
            target_node = nodes[index]
        else:
            print("❌ Укажите либо doc_id, либо index")
            return

        if target_node:
            print(f"\n{'=' * 80}")
            print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ДОКУМЕНТЕ")
            print(f"{'=' * 80}")
            print(f"📋 ID: {target_node.node_id}")
            print(f"📊 Метаданные: {target_node.metadata}")
            print("\n📖 Полный текст (лемматизированный):")
            print(f"{'=' * 80}")
            print(target_node.text)
            print(f"{'=' * 80}")

            if hasattr(target_node, "embedding") and target_node.embedding:
                vector_length = len(target_node.embedding)
                print(f"\n🔢 Размер вектора: {vector_length}")
                if vector_length > 10:
                    print(f"   Первые 10 значений: {target_node.embedding[:10]}")
                else:
                    print(f"   Значения: {target_node.embedding}")
        else:
            print("❌ Документ не найден")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def quick_lance_db_check() -> None:
    """
    Быстрая проверка содержимого LanceDB без запуска всей RAG системы.
    """
    print("🔍 Быстрая проверка LanceDB...")
    display_lance_db_contents(limit=5)
