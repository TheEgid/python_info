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

    def add_chunk(
        self,
        chunk: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None
    ) -> None:

        if metadata is None:
            metadata = {}

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        embedding_list = embedding.astype(float).tolist()

        existing = self.client.table(self.table_name).select("id").eq("content", chunk).execute()
        if existing.data and len(existing.data) > 0:
            print(f"⚠️ Чанк уже существует в {self.table_name}, пропускаем вставку: {chunk[:50]}...")
            return

        row = {
            "content": chunk,
            "metadata": metadata,
            "embedding": embedding_list
        }

        self.client.table(self.table_name).insert(row).execute()
        print(f"✅ Добавлен чанк в {self.table_name}: {chunk[:50]}...")

    def search(
        self, query_emb: np.ndarray, top_k: int = 5, match_threshold: float = 0.3
    ) -> List[Tuple[int, str, dict, float]]:
        # Вызываем функцию PostgreSQL через RPC
        response = self.client.rpc(
            "match_documents_test_novaya",
            {"query_embedding": query_emb.tolist(), "match_threshold": match_threshold, "match_count": top_k},
        ).execute()

        results = response.data  # список dict с полями id, content, metadata, similarity
        return [(r["id"], r["content"], r["metadata"], r["similarity"]) for r in results]
