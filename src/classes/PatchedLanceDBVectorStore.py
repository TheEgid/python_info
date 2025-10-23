import json
import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from others.frida import FridaEmbedding

logger = logging.getLogger(__name__)


class PatchedLanceDBVectorStore(LanceDBVectorStore):
    """Patched LanceDB store with enhanced error handling and normalization."""

    def __init__(self, *args: Any, metric: str = "cosine", embedding_dim: int = 1536, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        # Используем приватные атрибуты, чтобы избежать конфликтов с Pydantic
        self._metric = metric
        self._embedding_dim = embedding_dim
        self._embed_model: Optional[FridaEmbedding] = None

    @property
    def embed_model(self) -> FridaEmbedding:
        """Lazy loading embed model."""
        if self._embed_model is None:
            self._embed_model = FridaEmbedding()
        return self._embed_model

    @property
    def metric(self) -> str:
        """Get the distance metric."""
        return self._metric

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Нормализует эмбеддинг для косинусного сходства."""
        if not self._validate_embedding(embedding):
            logger.warning("Invalid embedding provided, returning zero vector")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """Валидирует эмбеддинг."""
        return (
            isinstance(embedding, np.ndarray) and
            embedding.size > 0 and
            not np.any(np.isnan(embedding)) and
            not np.any(np.isinf(embedding))
        )

    def _compute_query_embedding(self, query_obj: VectorStoreQuery) -> np.ndarray:
        """Вычисляет эмбеддинг для запроса."""
        try:
            # Prefer provided embedding
            q_emb = getattr(query_obj, "query_embedding", None)
            if q_emb is not None:
                embedding = np.array(q_emb, dtype=np.float32)
                return self._normalize_embedding(embedding)

            # Extract query text
            possible_text = self._extract_query_text(query_obj)

            if not possible_text:
                logger.warning("No query text found, using empty string")
                possible_text = ""

            # Get and normalize embedding
            emb = self.embed_model._get_text_embedding(possible_text)
            embedding = np.array(emb, dtype=np.float32)
            return self._normalize_embedding(embedding)

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error computing query embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Unexpected error in query embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _extract_query_text(self, query_obj: VectorStoreQuery) -> Optional[str]:
        """Извлекает текст запроса из объекта запроса."""
        for attr in ("query", "query_str", "query_text", "text", "query_str_value"):
            if hasattr(query_obj, attr):
                val = getattr(query_obj, attr)
                if isinstance(val, str) and val.strip():
                    return val

        # Fallback to string representation
        query_str = str(query_obj) if query_obj else ""
        return query_str if query_str.strip() else None

    def _distance_to_similarity(self, distance: float) -> float:
        """Конвертирует расстояние в схожесть."""
        SIMILARITY_METRICS = {"cosine"}
        DISTANCE_METRICS = {"l2", "euclidean"}

        try:
            distance = float(distance)
        except (ValueError, TypeError):
            logger.warning(f"Invalid distance value: {distance}, using 0.0")
            distance = 0.0

        if self.metric in SIMILARITY_METRICS:
            return max(0.0, min(1.0, 1.0 - distance))  # Clamp to [0, 1]
        elif self.metric in DISTANCE_METRICS:
            return 1.0 / (1.0 + abs(distance))
        else:
            logger.warning(f"Unknown metric: {self.metric}, returning raw distance")
            return distance

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ANN401
        """Выполняет поиск по векторному хранилищу."""
        try:
            # Compute query embedding
            q_emb = self._compute_query_embedding(query)
            top_k = getattr(query, "similarity_top_k", 10)

            logger.debug(f"Performing vector search with top_k={top_k}")

            # Выполняем поиск
            results_df = self._table.search(q_emb).limit(top_k).to_pandas()

            if results_df.empty:
                logger.warning("No results found for query")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            nodes: List[TextNode] = []
            similarities: List[float] = []
            ids: List[str] = []

            # Process results
            for idx, row in results_df.iterrows():
                try:
                    node_data = self._process_search_result(row, idx)
                    if node_data:
                        nodes.append(node_data['node'])
                        similarities.append(node_data['similarity'])
                        ids.append(node_data['id'])

                except Exception as node_error:
                    logger.warning(f"Error processing row {idx}: {node_error}")
                    continue

            logger.debug(f"Successfully processed {len(nodes)} results")
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        except Exception as e:
            logger.exception(f"Error in PatchedLanceDBVectorStore.query: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

    def _process_search_result(self, row: pd.Series, idx: int) -> Optional[dict]:
        """Обрабатывает одну строку результатов поиска."""
        try:
            # Обработка метаданных
            metadata = self._extract_metadata(row)
            text_content = self._extract_text_content(row, metadata)

            # Создаем узел
            node = self._create_text_node(row, metadata, text_content, idx)

            # Вычисляем схожесть
            distance = self._extract_distance(row)
            similarity = self._distance_to_similarity(distance)

            # ID
            doc_id = metadata.get("doc_id", f"doc_{idx}")

            return {
                'node': node,
                'similarity': similarity,
                'id': doc_id
            }

        except Exception as e:
            logger.error(f"Error processing search result at index {idx}: {e}")
            return None

    def _extract_metadata(self, row: pd.Series) -> dict:
        """Извлекает и нормализует метаданные."""
        metadata = row.get("metadata", {})

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Failed to parse metadata JSON, using as string")
                metadata = {"__node_content__": metadata}
        elif isinstance(metadata, pd.Series):
            metadata = metadata.to_dict()
        elif metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            # Convert other types to dict
            metadata = {"__node_content__": str(metadata)}

        return metadata

    def _extract_text_content(self, row: pd.Series, metadata: dict) -> str:
        """Извлекает текстовое содержимое."""
        # Try to get text from row first
        text_content = row.get("text", "")

        # If no text in row, try metadata
        if not text_content:
            text_content = metadata.get("__node_content__", "")

        # Ensure text_content is string
        if not isinstance(text_content, str):
            text_content = str(text_content) if text_content is not None else ""

        # Ensure __node_content__ is present in metadata
        if "__node_content__" not in metadata or not metadata.get("__node_content__"):
            metadata["__node_content__"] = text_content

        return text_content

    def _create_text_node(
        self,
        row: pd.Series,
        metadata: dict,
        text_content: str,
        idx: int
    ) -> TextNode:
        """Создает TextNode из данных строки."""
        # Extract embedding
        row_embedding = None
        for emb_field in ["embedding", "vector"]:
            if emb_field in row:
                emb_value = row[emb_field]
                if emb_value is not None:
                    try:
                        # Convert to list if it's numpy array or similar
                        if hasattr(emb_value, 'tolist'):
                            row_embedding = emb_value.tolist()
                        elif isinstance(emb_value, (list, tuple)):
                            row_embedding = list(emb_value)
                        else:
                            row_embedding = [float(emb_value)] if np.isscalar(emb_value) else None
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert embedding: {e}")
                        row_embedding = None
                break

        node_id = metadata.get("doc_id", f"doc_{idx}")

        return TextNode(
            text=text_content,
            metadata=metadata,
            embedding=row_embedding,
            id_=node_id,
        )

    def _extract_distance(self, row: pd.Series) -> float:
        """Извлекает расстояние из результатов поиска."""
        # Try different possible distance field names
        for field_name in ["_distance", "_score", "distance", "score"]:
            if field_name in row:
                distance = row[field_name]
                try:
                    return float(distance)
                except (ValueError, TypeError):
                    continue

        # Fallback
        logger.warning("No distance field found in search results, using 0.0")
        return 0.0
