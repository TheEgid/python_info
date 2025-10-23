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
        self._metric = metric
        self._embedding_dim = embedding_dim
        self._embed_model: Optional[FridaEmbedding] = None
        logger.info(f"Initialized LanceDB vector store with metric: {metric}, dim: {embedding_dim}")

    @property
    def embed_model(self) -> FridaEmbedding:
        """Lazy loading embed model."""
        if self._embed_model is None:
            self._embed_model = FridaEmbedding()
        return self._embed_model

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Нормализует эмбеддинг для косинусного сходства."""
        if not self._validate_embedding(embedding):
            logger.warning("Invalid embedding provided, returning zero vector")
            return np.zeros(self._embedding_dim, dtype=np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """Валидирует эмбеддинг."""
        return (
            isinstance(embedding, np.ndarray)
            and embedding.size > 0
            and embedding.shape[0] == self._embedding_dim
            and not np.any(np.isnan(embedding))
            and not np.any(np.isinf(embedding))
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

        except Exception as e:
            logger.error(f"Error computing query embedding: {e}")
            return np.zeros(self._embedding_dim, dtype=np.float32)

    def _extract_query_text(self, query_obj: VectorStoreQuery) -> Optional[str]:
        """Извлекает текст запроса из объекта запроса."""
        # Основные атрибуты где может быть текст запроса
        for attr in ["query_str", "query"]:
            if hasattr(query_obj, attr):
                val = getattr(query_obj, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        return None

    def _distance_to_similarity(self, distance: float) -> float:
        """Конвертирует расстояние в схожесть."""
        try:
            distance = float(distance)
        except (ValueError, TypeError):
            logger.warning(f"Invalid distance value: {distance}, using 0.0")
            return 0.0

        # Для cosine distance: similarity = 1 - distance
        if self._metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        # Для L2/евклидова расстояния: similarity = 1 / (1 + distance)
        elif self._metric in ["l2", "euclidean"]:
            return 1.0 / (1.0 + max(0, distance))
        else:
            logger.warning(f"Unknown metric: {self._metric}, returning 1 - distance")
            return 1.0 - distance

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:  # noqa: ANN401
        """Выполняет поиск по векторному хранилищу."""
        try:
            # Проверяем что таблица существует
            if self._table is None:
                logger.error("LanceDB table is not initialized")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            # Вычисляем эмбеддинг запроса
            query_embedding = self._compute_query_embedding(query)

            # Если нет эмбеддинга - возвращаем пустой результат
            if query_embedding is None or query_embedding.size == 0:
                logger.error("Failed to compute query embedding")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            # Получаем top_k
            top_k = query.similarity_top_k or 10

            logger.info(f"Performing vector search with top_k={top_k}, metric={self._metric}")

            # ВАЖНО: Правильный вызов поиска LanceDB
            # Используем правильный API для поиска
            if hasattr(self._table, "search"):
                # Современный API
                results = self._table.search(query_embedding).limit(top_k).to_list()
            else:
                # Совместимость со старыми версиями
                results = self._table.query().nearest_neighbors("vector", query_embedding).limit(top_k).to_pandas()
                results = results.to_dict("records")

            if not results:
                logger.warning("No results found for query")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            # Обрабатываем результаты
            nodes: List[TextNode] = []
            similarities: List[float] = []
            ids: List[str] = []

            for i, result in enumerate(results):
                try:
                    node_data = self._process_search_result(result, i)
                    if node_data:
                        nodes.append(node_data["node"])
                        similarities.append(node_data["similarity"])
                        ids.append(node_data["id"])
                except Exception as e:
                    logger.warning(f"Error processing result {i}: {e}")
                    continue

            logger.info(f"Successfully processed {len(nodes)} results")
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        except Exception as e:
            logger.exception(f"Error in PatchedLanceDBVectorStore.query: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

    def _process_search_result(self, result: dict, idx: int) -> Optional[dict]:
        """Обрабатывает одну строку результатов поиска."""
        try:
            # Конвертируем в Series для совместимости
            if not isinstance(result, dict):
                result = dict(result)

            row = pd.Series(result)

            # Обработка метаданных
            metadata = self._extract_metadata(row)
            text_content = self._extract_text_content(row, metadata)

            # Создаем узел
            node = self._create_text_node(row, metadata, text_content, idx)

            # Вычисляем схожесть
            distance = self._extract_distance(row)
            similarity = self._distance_to_similarity(distance)

            # ID
            doc_id = metadata.get("doc_id") or metadata.get("id") or f"doc_{idx}"

            return {"node": node, "similarity": similarity, "id": doc_id}

        except Exception as e:
            logger.error(f"Error processing search result at index {idx}: {e}")
            return None

    def _extract_metadata(self, row: pd.Series) -> dict:
        """Извлекает и нормализует метаданные."""
        metadata = row.get("metadata", {})

        # Если metadata строка, пытаемся распарсить JSON
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"content": metadata}

        # Если metadata не dict, конвертируем
        if not isinstance(metadata, dict):
            metadata = {"content": str(metadata)}

        return metadata

    def _extract_text_content(self, row: pd.Series, metadata: dict) -> str:
        """Извлекает текстовое содержимое."""
        # Пробуем разные поля где может быть текст
        for field in ["text", "content", "document", "body"]:
            if field in row and row[field]:
                text = row[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()

        # Пробуем из метаданных
        for field in ["text", "content", "document"]:
            if field in metadata and metadata[field]:
                text = metadata[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()

        # Fallback
        return metadata.get("content", "") or ""

    def _create_text_node(self, row: pd.Series, metadata: dict, text_content: str, idx: int) -> TextNode:
        """Создает TextNode из данных строки."""
        # Extract embedding
        embedding = None
        for emb_field in ["embedding", "vector"]:
            if emb_field in row and row[emb_field] is not None:
                emb_value = row[emb_field]
                try:
                    if hasattr(emb_value, "tolist"):
                        embedding = emb_value.tolist()
                    elif isinstance(emb_value, (list, tuple, np.ndarray)):
                        embedding = list(emb_value)
                    elif isinstance(emb_value, (int, float)):
                        embedding = [float(emb_value)]
                    break
                except Exception as e:
                    logger.warning(f"Failed to extract embedding from {emb_field}: {e}")
                    continue

        node_id = metadata.get("doc_id") or metadata.get("id") or f"doc_{idx}"

        return TextNode(
            text=text_content,
            metadata=metadata,
            embedding=embedding,
            id_=node_id,
        )

    def _extract_distance(self, row: pd.Series) -> float:
        """Извлекает расстояние из результатов поиска."""
        # Поля которые могут содержать расстояние в LanceDB
        for field_name in ["_distance", "_score", "distance", "score", "similarity"]:
            if field_name in row:
                distance = row[field_name]
                try:
                    return float(distance)
                except (ValueError, TypeError):
                    continue

        # Если расстояние не найдено, используем дефолтное
        return 0.0
