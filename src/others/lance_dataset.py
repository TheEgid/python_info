import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import lancedb
import numpy as np
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode

from classes.PatchedLanceDBVectorStore import PatchedLanceDBVectorStore
from others.frida import FridaEmbedding

LANCE_DB_PATH = Path("./lancedb/articles_index").resolve()


def fill_lance_dataset(
    documents: List[Document],
    db_path: Path = LANCE_DB_PATH,
) -> Tuple[Optional[PatchedLanceDBVectorStore], List[TextNode]]:
    if not documents:
        logging.warning("⚠️ Нет документов для индексации.")
        return None, []

    logging.info(f"Создаём LanceDB по пути: {db_path}")
    db = lancedb.connect(db_path)
    table_name = "articles"

    if table_name in db.table_names():
        db.drop_table(table_name)

    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    table_data = []
    nodes: List[TextNode] = []

    for i, doc in enumerate(documents):
        text = (doc.text or "").strip()
        if not text:
            continue

        embedding = embed_model._get_text_embedding(text)
        embedding_array = np.array(embedding, dtype=np.float32)

        doc_id = doc.doc_id or f"doc_{i}"
        metadata = {
            "doc_id": doc_id,
            "source": getattr(doc, "extra_info", {}).get("file_path", "unknown"),
        }

        table_data.append(
            {
                "id": doc_id,
                "text": text,
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
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    try:
        db = lancedb.connect(db_path)
        table_name = "articles"

        if table_name in db.table_names():
            logging.info(f"📦 LanceDB '{table_name}' найден, проверяем данные...")
            table = db.open_table(table_name)
            df = table.to_pandas()  # Проверяем содержимое
            if df.empty:
                logging.warning(f"⚠️ Таблица '{table_name}' существует, но пуста. Пересоздаём...")
                db.drop_table(table_name)  # Удаляем пустую и перейдём к созданию
            else:
                vector_store = PatchedLanceDBVectorStore(table=table)
                nodes: List[TextNode] = []
                for r in df.to_dict(orient="records"):
                    meta = r.get("metadata", {})
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {"__node_content__": meta}
                    text = meta.get("__node_content__", r.get("text", ""))
                    nodes.append(
                        TextNode(text=text, metadata=meta, embedding=r.get("embedding"), id_=meta.get("doc_id"))
                    )
                return vector_store, nodes
        # Если таблица не существовала или была сброшена, создаём заново
        logging.info("🆕 LanceDB не найден, создаём заново...")
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
