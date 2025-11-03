import logging
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def get_documents_from_sql() -> List[Document]:
    """
    Заглушка для получения документов из SQL базы данных.
    Переопределите эту функцию в соответствии с вашей структурой БД.
    """
    # Пример реализации - замените на вашу логику
    try:
        # Здесь ваш код для подключения к БД и извлечения данных
        # Например:
        # import psycopg2
        # conn = psycopg2.connect(...)
        # cursor = conn.cursor()
        # cursor.execute("SELECT id, content, metadata FROM documents")
        # rows = cursor.fetchall()

        documents = []
        # for row in rows:
        #     doc = Document(
        #         text=row[1],
        #         metadata=row[2] or {},
        #         doc_id=str(row[0])
        #     )
        #     documents.append(doc)

        logging.info("📊 Документы из SQL базы загружены")
        return documents

    except Exception as e:
        logging.error(f"❌ Ошибка загрузки документов из SQL: {e}")
        return []


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
