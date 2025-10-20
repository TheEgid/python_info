import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from supabase import Client, create_client

from others.frida import get_frida_embeddings

logger = logging.getLogger(__name__)


class VectorStoreFRIDA:
    """Векторное хранилище на Supabase с поддержкой FRIDA эмбеддингов."""

    def __init__(self, table_name: str = "test_novaya") -> None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise EnvironmentError("SUPABASE_URL и SUPABASE_KEY не заданы в .env")

        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name

    def add_chunks_batch(
        self, chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None
    ) -> Tuple[int, int]:
        if not chunks:
            logger.warning("Получен пустой батч чанков")
            return 0, 0

        if len(chunks) != len(embeddings):
            raise ValueError("Количество чанков и эмбеддингов должно совпадать")

        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]

        rows_to_insert = []
        skipped_count = 0

        try:
            existing = self.client.table(self.table_name).select("content").in_("content", chunks).execute()
            existing_chunks = {item["content"] for item in existing.data} if existing.data else set()
        except Exception as e:
            logger.error(f"Ошибка при проверке существующих чанков: {e}")
            raise

        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas):  # noqa: B905
            if chunk in existing_chunks:
                skipped_count += 1
                continue

            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            embedding_list = embedding.astype(float).tolist()
            rows_to_insert.append({"content": chunk, "metadata": metadata, "embedding": embedding_list})

        successful_count = 0
        if rows_to_insert:
            try:
                self.client.table(self.table_name).insert(rows_to_insert).execute()
                successful_count = len(rows_to_insert)
                logger.info(f"Добавлено {successful_count} чанков")
            except Exception as e:
                logger.error(f"Ошибка при вставке батча: {e}")
                raise

        return successful_count, skipped_count

    def search(
        self, query_emb: np.ndarray, top_k: int = 5, match_threshold: float = 0.3
    ) -> List[Tuple[int, str, dict, float]]:
        try:
            if query_emb.ndim == 2 and query_emb.shape[0] == 1:
                query_emb = query_emb[0]

            response = self.client.rpc(
                "match_documents_test_novaya",
                {
                    "query_embedding": query_emb.tolist(),
                    "match_threshold": match_threshold,
                    "match_count": top_k,
                },
            ).execute()

            results = response.data if response.data else []
            return [(r["id"], r["content"], r["metadata"], r["similarity"]) for r in results]
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise

    # 🔥 Новый метод
    def load_and_index_directory(
        self,
        directory: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        batch_size: int = 64,
    ) -> Tuple[int, int]:
        """
        Загружает файлы из директории, разбивает на чанки и индексирует в Supabase.

        Args:
            directory: путь к папке с документами
            chunk_size: длина чанка
            chunk_overlap: перекрытие между чанками
            batch_size: размер батча для вставки
        """
        reader = SimpleDirectoryReader(input_dir=directory)
        docs = reader.load_data()
        if not docs:
            logger.warning(f"Папка '{directory}' пуста")
            return 0, 0

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        all_chunks, all_metadatas = [], []
        for doc in docs:
            chunks = splitter.split_text(doc.get_content())
            all_chunks.extend(chunks)
            all_metadatas.extend([doc.metadata or {} for _ in chunks])

        logger.info(f"📄 Разбито {len(docs)} документов на {len(all_chunks)} чанков")

        total_added, total_skipped = 0, 0
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_meta = all_metadatas[i : i + batch_size]
            embeddings = get_frida_embeddings(batch_chunks, device="cpu")

            added, skipped = self.add_chunks_batch(batch_chunks, embeddings, batch_meta)
            total_added += added
            total_skipped += skipped

        logger.info(f"✅ Индексация завершена: добавлено {total_added}, пропущено {total_skipped}")
        return total_added, total_skipped
