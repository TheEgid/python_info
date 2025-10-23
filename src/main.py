# src/main.py
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Tuple

import lancedb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from others.frida import FridaEmbedding

logging.basicConfig(level=logging.INFO)
load_dotenv()

LANCE_DB_PATH = Path("./lancedb/articles_index").resolve()


class PatchedLanceDBVectorStore(LanceDBVectorStore):
    """Patched LanceDB store: compute embeddings via Frida when needed and normalize results."""

    def _compute_query_embedding(self, query_obj: VectorStoreQuery) -> np.ndarray:
        # Prefer provided embedding
        q_emb = getattr(query_obj, "query_embedding", None)
        if q_emb is not None:
            return np.array(q_emb, dtype=np.float32)

        # Try to extract a query text from common attributes
        possible_text = None
        for attr in ("query", "query_str", "query_text", "text", "query_str_value"):
            if hasattr(query_obj, attr):
                val = getattr(query_obj, attr)
                if isinstance(val, str) and val.strip():
                    possible_text = val
                    break

        # If query_obj itself is str-like, use str()
        if possible_text is None:
            try:
                possible_text = str(query_obj)
            except Exception:
                possible_text = ""

        embed_model = FridaEmbedding()
        emb = embed_model._get_text_embedding(possible_text)
        return np.array(emb, dtype=np.float32)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ANN401
        try:
            q_emb = self._compute_query_embedding(query)
            top_k = getattr(query, "similarity_top_k", None) or 10

            # LanceDB search expects a list/array
            results_df = self._table.search(q_emb).limit(top_k).to_pandas()

            nodes: List[TextNode] = []
            similarities: List[float] = []
            ids: List[str] = []

            for idx, row in results_df.iterrows():
                # Extract metadata (string or struct/pandas.Series)
                metadata = row.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {"__node_content__": metadata}
                elif isinstance(metadata, pd.Series):
                    metadata = metadata.to_dict()
                elif metadata is None:
                    metadata = {}

                # Ensure __node_content__ present
                text_content = row.get("text", "") or metadata.get("__node_content__", "")
                if "__node_content__" not in metadata or not metadata.get("__node_content__"):
                    metadata["__node_content__"] = text_content

                # Build TextNode (include embedding if present in row)
                row_embedding = None
                if "embedding" in row:
                    row_embedding = row["embedding"]
                elif "vector" in row:
                    row_embedding = row["vector"]

                node = TextNode(
                    text=text_content,
                    metadata=metadata,
                    embedding=(list(row_embedding) if row_embedding is not None else None),
                    id_=metadata.get("doc_id", f"doc_{idx}"),
                )

                nodes.append(node)

                # similarity/distance: support both _distance and _score
                dist = row.get("_distance", None)
                if dist is None:
                    dist = row.get("_score", 0.0)
                try:
                    similarities.append(float(dist))
                except Exception:
                    similarities.append(0.0)

                ids.append(metadata.get("doc_id", f"doc_{idx}"))

            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        except Exception as e:
            logging.exception(f"Ошибка в PatchedLanceDBVectorStore.query: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


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
            "__node_content__": text,
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

        nodes.append(
            TextNode(text=text, embedding=embedding_array.tolist(), metadata=metadata, id_=doc_id)
        )

    if not table_data:
        logging.warning("⚠️ Нет валидных документов для индексации.")
        return None, []

    table = db.create_table(table_name, data=table_data, mode="overwrite")

    vector_store = PatchedLanceDBVectorStore(table=table)
    logging.info(f"✅ LanceDB создан: {len(table_data)} документов")
    logging.info(f"📜 Lance schema: {table.schema}")

    return vector_store, nodes


def load_or_fill_lance(
    db_path: Path = LANCE_DB_PATH,
) -> Tuple[Optional[PatchedLanceDBVectorStore], Optional[List[TextNode]]]:
    try:
        db = lancedb.connect(db_path)
        table_name = "articles"

        if table_name in db.table_names():
            logging.info(f"📦 LanceDB '{table_name}' найден, загружаем...")
            table = db.open_table(table_name)
            vector_store = PatchedLanceDBVectorStore(table=table)

            # Build nodes list from existing table (useful for some LlamaIndex flows)
            df = table.to_pandas()
            nodes: List[TextNode] = []
            for r in df.to_dict(orient="records"):
                meta = r.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {"__node_content__": meta}
                text = meta.get("__node_content__", r.get("text", ""))
                nodes.append(TextNode(text=text, metadata=meta, embedding=r.get("embedding"), id_=meta.get("doc_id")))

            return vector_store, nodes
        else:
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

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )

        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("Ваш запрос")
        logging.info(f"Найдено чанков: {len(nodes)}")

        for i, node in enumerate(nodes):
            logging.info(f"Чанк {i+1}: {len(node.text)} символов")

        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            streaming=False,
        )

        logging.info("🔍 Выполнение тестового запроса...")
        response = query_engine.query("Расскажи о телескопах")
        response_text = str(response).strip()
        wrapped_text = textwrap.fill(response_text, width=120)  # Ширина 80 символов

        print("\n" + "=" * 50)
        print("ОТВЕТ:")
        print("=" * 50)
        print(wrapped_text)
        print("=" * 50)

        # Дополнительная информация
        print("\n📊 Статистика ответа:")
        print(f"   Общая длина: {len(response_text)} символов")
        print(f"   Количество строк после форматирования: {wrapped_text.count(chr(10)) + 1}")

    except Exception as e:
        logging.exception(f"❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
