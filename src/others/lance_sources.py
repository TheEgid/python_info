import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode
from supabase import Client, create_client

try:
    from classes.PatchedLanceDBVectorStore import PatchedLanceDBVectorStore
except ImportError:
    PatchedLanceDBVectorStore = None
    logging.warning("⚠️ PatchedLanceDBVectorStore не найден, используется None")


# ==================== СУPABASE CLIENT ====================
class SupabaseManager:
    """Менеджер для работы с Supabase SDK v2.23.1"""

    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL и SUPABASE_KEY не установлены в переменных окружения"
            )

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        logging.info("✅ Подключение к Supabase успешно")

    def close(self) -> None:
        """Закрыть подключение"""
        try:
            if hasattr(self.supabase, 'close'):
                self.supabase.close()
                logging.info("✅ Подключение Supabase закрыто")
        except Exception as e:
            logging.error(f"❌ Ошибка при закрытии подключения: {e}")


# Глобальный менеджер
_supabase_manager: Optional[SupabaseManager] = None


def get_supabase_client() -> Client:
    """Получить Supabase клиент"""
    global _supabase_manager

    if _supabase_manager is None:
        _supabase_manager = SupabaseManager()

    return _supabase_manager.supabase


# ==================== ДОКУМЕНТЫ ИЗ ДИРЕКТОРИИ ====================
def get_documents_from_directory(directory: str = "articles/") -> List[Document]:
    """Загружает документы из директории"""
    articles_dir = Path(directory)

    if not articles_dir.exists():
        logging.error(f"❌ Директория {articles_dir} не найдена!")
        return []

    try:
        documents = SimpleDirectoryReader(str(articles_dir)).load_data()
        logging.info(f"📁 Загружено {len(documents)} документов из {directory}")
        return documents
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки документов из {directory}: {e}")
        return []


# ==================== РАБОТА С SUPABASE ====================
def calculate_content_hash(content: str) -> str:
    """Вычисляет MD5 хеш содержимого для проверки дубликатов"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием.

    Args:
        text: Исходный текст
        chunk_size: Размер чанка
        overlap: Перекрытие между чанками

    Returns:
        Список чанков
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

        if start >= len(text):
            break

    return chunks


def get_documents_from_supabase(
    table_name: str = "novaya",
    limit: Optional[int] = None,
    where_condition: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
    cache_path: Path = Path("./cache_supabase.json"),
    cache_ttl: int = 3600,  # Время жизни кеша в секундах (по умолчанию 1 час)
    force_refresh: bool = False,  # Принудительно обновить кеш
) -> List[Document]:
    """
    Получает документы из Supabase с восстановлением чанков и кешированием результата.
    """
    try:
        # --- Проверяем кеш ---
        if cache_path.exists() and not force_refresh:
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime < cache_ttl:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    logging.info(f"🧠 Загрузка документов из кеша ({cache_path})")
                    return [
                        Document(text=item["text"], metadata=item["metadata"], doc_id=item["doc_id"])
                        for item in cached
                    ]
                except Exception as e:
                    logging.warning(f"⚠️ Ошибка чтения кеша: {e}")

        # --- Загружаем из Supabase ---
        supabase = get_supabase_client()
        query = supabase.table(table_name).select("*")

        if where_condition:
            for key, value in where_condition.items():
                if isinstance(value, dict):
                    for op, val in value.items():
                        if op == "eq":
                            query = query.eq(key, val)
                        elif op == "gt":
                            query = query.gt(key, val)
                        elif op == "lt":
                            query = query.lt(key, val)
                else:
                    query = query.eq(key, value)

        if limit:
            query = query.limit(limit)

        response = query.execute()

        rows = None

        # 🧩 1. Новая версия клиента (response.data)
        if hasattr(response, "data"):
            rows = response.data

        # 🧩 2. Старый клиент — вернул словарь
        elif isinstance(response, dict):
            rows = response.get("data")

        # 🧩 3. API вернул JSON-строку
        elif isinstance(response, str):
            try:
                parsed = json.loads(response)
                rows = parsed.get("data") if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                logging.error("❌ Supabase вернул невалидный JSON-ответ.")
                return []

        # 🧩 4. Финальная проверка
        if not isinstance(rows, list):
            logging.error(f"❌ Неожиданный формат ответа от Supabase: {type(rows)}")
            return []

        if not rows:
            logging.warning(f"⚠️ Нет данных в таблице {table_name}")
            return []

        documents_by_url = {}

        for row in rows:
            content = row.get("content", "")
            metadata = row.get("metadata") or {}

            if isinstance(metadata, str) and metadata.strip():
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            url = metadata.get("url", f"unknown_{row.get('id', 'no_id')}")
            chunk_index = metadata.get("chunk_index")
            total_chunks = metadata.get("total_chunks", 1)
            source = metadata.get("source", "supabase")

            if url not in documents_by_url:
                documents_by_url[url] = {
                    "chunks": {},
                    "total_chunks": total_chunks,
                    "source": source,
                    "metadata_template": metadata or {},
                }

            if chunk_index is not None:
                documents_by_url[url]["chunks"][chunk_index] = content
                documents_by_url[url]["total_chunks"] = max(
                    documents_by_url[url]["total_chunks"], chunk_index + 1
                )
            else:
                documents_by_url[url]["chunks"][0] = content

        documents = []

        for url, doc_data in documents_by_url.items():
            chunks = doc_data["chunks"]
            total_chunks = doc_data["total_chunks"]

            available_chunks = len(chunks)
            if available_chunks != total_chunks:
                logging.warning(
                    f"⚠️ Для URL {url} ожидалось {total_chunks} чанков, найдено {available_chunks}"
                )

            full_content = " ".join(chunks[i] for i in sorted(chunks.keys())).strip()
            if not full_content:
                continue

            metadata = doc_data["metadata_template"].copy() if doc_data["metadata_template"] else {}
            if not include_metadata:
                metadata = {}

            metadata.update({
                "source": "supabase",
                "table": table_name,
                "url": url,
                "content_hash": calculate_content_hash(full_content),
                "reconstructed_from_chunks": available_chunks,
                "total_chunks_expected": total_chunks,
                "chunks_missing": total_chunks - available_chunks,
                "supabase_source": doc_data["source"],
            })

            doc = Document(
                text=full_content,
                metadata=metadata,
                doc_id=f"supabase_{hashlib.md5(url.encode()).hexdigest()[:16]}",
            )
            documents.append(doc)

        # --- Сохраняем кеш ---
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"text": d.text, "metadata": d.metadata, "doc_id": d.doc_id} for d in documents],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logging.info(f"💾 Результат сохранён в кеш ({cache_path})")
        except Exception as e:
            logging.warning(f"⚠️ Ошибка сохранения кеша: {e}")

        logging.info(f"📊 Восстановлено {len(documents)} документов из {len(documents_by_url)} URL")
        total_reconstructed = sum(
            1 for doc in documents if doc.metadata.get("reconstructed_from_chunks", 0) > 1
        )
        logging.info(f"📦 Документов восстановлено из чанков: {total_reconstructed}")

        return documents

    except Exception as e:
        logging.error(f"❌ Ошибка загрузки документов из Supabase: {e}")
        return []


def insert_documents_to_supabase(
    documents: List[Document],
    table_name: str = "novaya",
    chunk_size: int = 1000,
    overlap: int = 200
) -> Dict[str, Any]:
    """
    Вставляет документы в Supabase с автоматическим разбиением на чанки.

    Args:
        documents: Список документов для вставки
        table_name: Название таблицы
        chunk_size: Максимальный размер чанка в символах
        overlap: Перекрытие между чанками

    Returns:
        Статистика операции
    """
    try:
        supabase = get_supabase_client()

        inserted_count = 0
        skipped_count = 0
        total_chunks_created = 0
        errors = []
        records_to_insert = []

        for doc in documents:
            try:
                content = doc.text or ""
                url = doc.metadata.get("url", "unknown")

                # Разбиваем текст на чанки
                chunks = split_text_into_chunks(content, chunk_size, overlap)
                total_chunks = len(chunks)

                for chunk_index, chunk_content in enumerate(chunks):
                    content_hash = calculate_content_hash(chunk_content)

                    # Подготавливаем метаданные для чанка
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": chunk_index,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "url": url,
                        }
                    )

                    record = {
                        "content": chunk_content,
                        "content_hash": content_hash,
                        "metadata": json.dumps(chunk_metadata),
                    }
                    records_to_insert.append(record)
                    total_chunks_created += 1

                inserted_count += 1

            except Exception as e:
                errors.append(f"Документ {doc.doc_id}: {str(e)}")
                continue

        # Вставляем все записи одной операцией
        if records_to_insert:
            try:
                response = supabase.table(table_name).upsert(records_to_insert).execute()  # noqa: F841
                logging.info(f"✅ Успешно вставлено {len(records_to_insert)} записей")
            except Exception as e:
                logging.error(f"❌ Ошибка вставки данных: {e}")
                errors.append(f"Ошибка массовой вставки: {str(e)}")

        return {
            "inserted_documents": inserted_count,
            "skipped_chunks": skipped_count,
            "total_chunks_created": total_chunks_created,
            "errors": errors,
        }

    except Exception as e:
        logging.error(f"❌ Ошибка вставки в Supabase: {e}")
        return {"error": str(e)}


def delete_documents_from_supabase(
    table_name: str = "novaya",
    where_condition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Удаляет документы из Supabase.

    Args:
        table_name: Название таблицы
        where_condition: Условие для удаления

    Returns:
        Статистика операции
    """
    try:
        supabase = get_supabase_client()

        query = supabase.table(table_name).delete()

        if where_condition:
            for key, value in where_condition.items():
                query = query.eq(key, value)
        else:
            logging.warning("⚠️ Условие удаления не указано, удаление отменено")
            return {"error": "No delete condition provided"}

        query.execute()

        return {
            "status": "success",
            "message": "Documents deleted successfully"
        }

    except Exception as e:
        logging.error(f"❌ Ошибка удаления документов: {e}")
        return {"error": str(e)}


def sync_supabase_to_lance(
    supabase_table: str = "novaya",
    lance_db_path: Path = None,
    where_condition: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> Tuple[Optional[Any], Optional[List[TextNode]]]:
    """
    Синхронизирует данные из Supabase в LanceDB с восстановлением структуры чанков.

    Args:
        supabase_table: Таблица в Supabase
        lance_db_path: Путь к LanceDB
        where_condition: Условие фильтрации
        limit: Ограничение количества документов

    Returns:
        Tuple[vector_store, nodes] или (None, None) при ошибке
    """

    try:
        logging.info("🔄 Синхронизация Supabase → LanceDB...")

        documents = get_documents_from_supabase(
            table_name=supabase_table,
            where_condition=where_condition,
            limit=limit,
        )

        if not documents:
            logging.error("❌ Не найдено документов в Supabase для синхронизации")
            return None, None

        # Импортируем функцию из другого файла
        from others.lance_dataset import fill_lance_dataset

        # Создаем/обновляем LanceDB
        vector_store, nodes = fill_lance_dataset(documents, db_path=lance_db_path)

        if vector_store and nodes:
            logging.info(f"✅ Успешно синхронизировано {len(nodes)} документов")

            # Логируем статистику
            urls = set()
            sources = set()
            for node in nodes:
                if node.metadata.get("url"):
                    urls.add(node.metadata["url"])
                if node.metadata.get("source"):
                    sources.add(node.metadata["source"])

            logging.info(f"🌐 Уникальных URL: {len(urls)}")
            logging.info(f"📁 Источников: {len(sources)}")

            # Логируем информацию о восстановленных чанках
            reconstructed_docs = sum(
                1 for node in nodes if node.metadata.get("reconstructed_from_chunks", 0) > 1
            )
            logging.info(f"🧩 Документов восстановлено из чанков: {reconstructed_docs}")

        return vector_store, nodes

    except Exception as e:
        logging.error(f"❌ Ошибка синхронизации Supabase → LanceDB: {e}")
        return None, None


def get_table_stats(table_name: str = "novaya") -> Dict[str, Any]:
    """
    Получить статистику таблицы из Supabase.

    Args:
        table_name: Название таблицы

    Returns:
        Словарь со статистикой
    """
    try:
        supabase = get_supabase_client()

        # Получаем количество записей
        response = supabase.table(table_name).select("id", count="exact").limit(1).execute()
        total_records = response.count if hasattr(response, 'count') else len(response.data)

        # Получаем информацию об источниках
        response = supabase.table(table_name).select("metadata").execute()

        sources = set()
        urls = set()
        chunks_count = 0

        for row in response.data:
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    pass

            if isinstance(metadata, dict):
                if metadata.get("source"):
                    sources.add(metadata["source"])
                if metadata.get("url"):
                    urls.add(metadata["url"])
                if metadata.get("chunk_index") is not None:
                    chunks_count += 1

        return {
            "table_name": table_name,
            "total_records": total_records,
            "unique_sources": len(sources),
            "sources": list(sources),
            "unique_urls": len(urls),
            "total_chunks": chunks_count,
            "statistics": {
                "records": total_records,
                "urls": len(urls),
                "sources": len(sources),
            }
        }

    except Exception as e:
        logging.error(f"❌ Ошибка получения статистики: {e}")
        return {"error": str(e)}


# # ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     # 1. Тест подключения
#     print("\n=== Тест 1: Проверка подключения ===")
#     try:
#         client = get_supabase_client()
#         print("✅ Подключение к Supabase успешно!")
#     except Exception as e:
#         print(f"❌ Ошибка подключения: {e}")

#     # 2. Получить статистику таблицы
#     print("\n=== Тест 2: Статистика таблицы ===")
#     stats = get_table_stats("novaya")
#     print(f"📊 Статистика: {stats}")

#     # 3. Синхронизация всех данных
#     print("\n=== Тест 3: Синхронизация ===")
#     vector_store, nodes = sync_supabase_to_lance()
#     if nodes:
#         print(f"✅ Загружено {len(nodes)} документов")

#     # 4. Вставка тестовых данных
#     print("\n=== Тест 4: Вставка тестовых данных ===")
#     sample_docs = [
#         Document(
#             text="Это первый тестовый документ для проверки работы системы.",
#             metadata={"url": "https://example.com/doc1", "source": "test_supabase_sdk"},
#         ),
#     ]
#     result = insert_documents_to_supabase(sample_docs)
#     print(f"📊 Результат: {result}")
