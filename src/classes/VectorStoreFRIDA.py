import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from llama_index.core.vector_stores import VectorStore
from supabase import Client, create_client

from others.frida import get_frida_embeddings

logger = logging.getLogger(__name__)


class VectorStoreFRIDA:
    def __init__(self) -> None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = "test_novaya"

    def add_chunks_batch(
        self, chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None
    ) -> Tuple[int, int]:
        """
        Добавляет батч чанков в векторное хранилище.

        Args:
            chunks: Список текстов чанков
            embeddings: Массив эмбеддингов shape=(n_chunks, embedding_dim)
            metadatas: Список метаданных для каждого чанка

        Returns:
            Tuple[int, int]: количество успешно добавленных и пропущенных чанков
        """
        if not chunks:
            logger.warning("Получен пустой батч чанков")
            return 0, 0

        if len(chunks) != len(embeddings):
            raise ValueError("Количество чанков и эмбеддингов должно совпадать")

        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]
        elif len(chunks) != len(metadatas):
            raise ValueError("Количество чанков и метаданных должно совпадать")

        rows_to_insert = []
        skipped_count = 0

        try:
            existing = self.client.table(self.table_name).select("content").in_("content", chunks).execute()
            existing_chunks = {item["content"] for item in existing.data} if existing.data else set()
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке существующих чанков: {e}")
            raise

        # Подготавливаем данные для вставки
        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas):  # noqa: B905
            if chunk in existing_chunks:
                skipped_count += 1
                continue

            # Нормализуем эмбеддинг
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            elif embedding.ndim > 2:
                logger.warning(f"Неожиданная размерность эмбеддинга: {embedding.ndim}")
                continue

            embedding_list = embedding.astype(float).tolist()
            rows_to_insert.append({"content": chunk, "metadata": metadata, "embedding": embedding_list})

        successful_count = 0
        if rows_to_insert:
            try:
                self.client.table(self.table_name).insert(rows_to_insert).execute()
                successful_count = len(rows_to_insert)
                logger.info(f"✅ Добавлено {successful_count} чанков")
            except Exception as e:
                logger.error(f"❌ Ошибка при вставке батча: {e}")
                raise

        return successful_count, skipped_count

    def add_chunk(self, chunk: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Добавляет один чанк в хранилище."""
        if metadata is None:
            metadata = {}

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        embedding_list = embedding.astype(float).tolist()

        try:
            existing = self.client.table(self.table_name).select("id").eq("content", chunk).execute()
            if existing.data and len(existing.data) > 0:
                logger.info(f"⚠️  Чанк уже существует, пропускаем: {chunk[:50]}...")
                return

            row = {"content": chunk, "metadata": metadata, "embedding": embedding_list}
            self.client.table(self.table_name).insert(row).execute()
            logger.info(f"✅ Добавлен чанк: {chunk[:50]}...")
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении чанка: {e}")
            raise

    def search(
        self, query_emb: np.ndarray, top_k: int = 5, match_threshold: float = 0.3
    ) -> List[Tuple[int, str, dict, float]]:
        """Поиск по векторному сходству."""
        try:
            # Нормализуем эмбеддинг запроса
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
            logger.error(f"❌ Ошибка при поиске: {e}")
            raise


class LlamaIndexFRIDA(VectorStore):
    """
    Adapter для использования VectorStoreFRIDA с LlamaIndex.
    """

    stores_text = True

    def __init__(
        self,
        vector_store: Optional[VectorStoreFRIDA] = None,
        embedding_function: Any = None,  # noqa: ANN401
    ) -> None:
        """
        Args:
            vector_store: экземпляр VectorStoreFRIDA (если None, создаётся новый)
            embedding_function: функция для получения эмбеддингов (по умолчанию get_frida_embeddings)
        """
        self.vector_store = vector_store or VectorStoreFRIDA()
        self.embedding_function = embedding_function or get_frida_embeddings

    def add(self, nodes: List[Any], **add_kwargs: Any) -> List[str]:  # noqa: ANN401, ANN002, ANN003
        """
        Добавляет ноды (документы) в хранилище.
        Требуется LlamaIndex интерфейсом.
        """
        if not nodes:
            logger.warning("Получен пустой список нодов")
            return []

        try:
            texts = [node.get_content() for node in nodes]
            metadatas = [
                node.metadata if node.metadata is not None else {} for node in nodes
            ]
            embeddings = self.embedding_function(texts, device="cpu")
            added, skipped = self.vector_store.add_chunks_batch(texts, embeddings, metadatas)
            logger.info(f"Добавлено {added} нодов, пропущено {skipped}")
            return [node.node_id for node in nodes]
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении нодов: {e}")
            raise

    def add_documents(
        self, documents: List[str], metadatas: Optional[List[dict]] = None
    ) -> Tuple[int, int]:
        """
        Добавляет список документов в хранилище.

        Returns:
            Tuple[int, int]: (успешно добавлено, пропущено)
        """
        if not documents:
            logger.warning("Получен пустой список документов")
            return 0, 0

        try:
            embeddings = self.embedding_function(documents, device="cpu")
            return self.vector_store.add_chunks_batch(documents, embeddings, metadatas)
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении документов: {e}")
            raise

    def query(
        self, query: str, top_k: int = 5, match_threshold: float = 0.3
    ) -> List[Tuple[int, str, dict, float]]:
        """
        Поиск топ-N результатов по запросу.

        Returns:
            Список кортежей: (id, text, metadata, similarity)
        """
        if not query:
            logger.warning("Получен пустой запрос")
            return []

        try:
            embeddings = self.embedding_function([query], device="cpu")

            # Гарантируем 1D array
            if embeddings.ndim == 2:
                if embeddings.shape[0] != 1:
                    logger.warning(f"Неожиданная форма эмбеддинга: {embeddings.shape}")
                query_emb = embeddings[0]
            else:
                query_emb = embeddings

            results = self.vector_store.search(query_emb, top_k=top_k, match_threshold=match_threshold)
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка при поиске: {e}")
            raise

    def query_texts(self, query: str, top_k: int = 5, match_threshold: float = 0.3) -> List[str]:
        """
        Быстрый метод для получения только текстов результатов.
        """
        try:
            results = self.query(query, top_k=top_k, match_threshold=match_threshold)
            return [r[1] for r in results]
        except Exception as e:
            logger.error(f"❌ Ошибка при получении текстов: {e}")
            raise
