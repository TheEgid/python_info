from typing import List, Optional, Tuple

import numpy as np
from supabase import Client, create_client


class VectorStoreFRIDA:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = "test_novaya"

    def add_chunks(self, chunks: List[str], embeddings: np.ndarray, metadata: Optional[List[dict]] = None) -> None:
        if metadata is None:
            metadata = [{} for _ in chunks]

        rows = [
            {"content": text, "metadata": meta, "embedding": emb.tolist()}
            for text, meta, emb in zip(chunks, metadata, embeddings, strict=False)
        ]

        for row in rows:
            self.client.table(self.table_name).insert(row).execute()

        print(f"✅ Добавлено {len(chunks)} записей в {self.table_name}")

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
