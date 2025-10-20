import os
from typing import List, Optional, Tuple

import numpy as np
from supabase import Client, create_client


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
        if len(chunks) != len(embeddings):
            raise ValueError("Количество чанков и эмбеддингов должно совпадать")

        if metadatas is None:
            metadatas = [{} for _ in range(len(chunks))]
        elif len(chunks) != len(metadatas):
            raise ValueError("Количество чанков и метаданных должно совпадать")

        rows_to_insert = []
        skipped_count = 0

        # Проверяем существующие чанки для всего батча
        existing_chunks = set()
        for chunk in chunks:
            existing = self.client.table(self.table_name).select("id").eq("content", chunk).execute()
            if existing.data and len(existing.data) > 0:
                existing_chunks.add(chunk)

        # Подготавливаем данные для вставки
        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas):  # noqa: B905
            if chunk in existing_chunks:
                skipped_count += 1
                continue

            # Нормализуем эмбеддинг
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            embedding_list = embedding.astype(float).tolist()

            rows_to_insert.append({"content": chunk, "metadata": metadata, "embedding": embedding_list})

        # Вставляем батч в БД
        successful_count = 0
        if rows_to_insert:
            self.client.table(self.table_name).insert(rows_to_insert).execute()
            successful_count = len(rows_to_insert)

        return successful_count, skipped_count

    def add_chunk(self, chunk: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Добавляет один чанк в хранилище."""
        if metadata is None:
            metadata = {}

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        embedding_list = embedding.astype(float).tolist()

        existing = self.client.table(self.table_name).select("id").eq("content", chunk).execute()
        if existing.data and len(existing.data) > 0:
            print(f"⚠️  Чанк уже существует, пропускаем: {chunk[:50]}...")
            return

        row = {"content": chunk, "metadata": metadata, "embedding": embedding_list}
        self.client.table(self.table_name).insert(row).execute()
        print(f"✅ Добавлен чанк: {chunk[:50]}...")

    def search(
        self, query_emb: np.ndarray, top_k: int = 5, match_threshold: float = 0.3
    ) -> List[Tuple[int, str, dict, float]]:
        """Поиск по векторному сходству."""
        response = self.client.rpc(
            "match_documents_test_novaya",
            {"query_embedding": query_emb.tolist(), "match_threshold": match_threshold, "match_count": top_k},
        ).execute()

        results = response.data
        return [(r["id"], r["content"], r["metadata"], r["similarity"]) for r in results]
