import logging
import os
from typing import List, Tuple

from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.supabase import SupabaseVectorStore
from supabase import create_client

from others.frida import get_frida_embeddings

logger = logging.getLogger(__name__)


class FRIDAEmbedding(BaseEmbedding):
    """Обёртка для LlamaIndex, чтобы использовать FRIDA локально."""

    def __init__(self, batch_size: int = 32, device: str = "cpu") -> None:
        self.batch_size = batch_size
        self.device = device

    def _get_nodes_embedding(self, texts: List[str]) -> List[List[float]]:
        # texts — список текстов
        embs = get_frida_embeddings(texts, batch_size=self.batch_size, device=self.device)
        return embs.tolist()


class VectorStoreFRIDA:
    """
    FRIDA + LlamaIndex + Supabase векторное хранилище.
    Автоматическая загрузка, разбиение и вставка с кастомными эмбеддингами.
    """

    def __init__(self, table_name: str = "test_novaya") -> None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase_postgres_url = os.getenv("SUPABASE_POSTGRES_URL")

        if not supabase_url or not supabase_key or not supabase_postgres_url:
            raise EnvironmentError("❌ SUPABASE_URL, SUPABASE_KEY и SUPABASE_POSTGRES_URL должны быть заданы в .env")

        self.client = create_client(supabase_url, supabase_key)
        self.table_name = table_name

        vector_store = SupabaseVectorStore(
            postgres_connection_string=supabase_postgres_url,
            collection_name=table_name,
        )

        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.embed_model = FRIDAEmbedding(batch_size=32, device="cpu")
        self.vector_store = vector_store

        logger.info(f"✅ Инициализирован VectorStoreFRIDA (таблица: {table_name})")

    # ----------------------------------------------------------------------

    def load_documents(self, directory: str) -> List[Document]:
        """Загружает документы через LlamaIndex SimpleDirectoryReader."""
        docs = SimpleDirectoryReader(input_dir=directory).load_data()
        if not docs:
            raise FileNotFoundError(f"Папка '{directory}' пуста или не содержит документов")
        logger.info(f"📄 Загружено {len(docs)} документов")
        return docs

    # ----------------------------------------------------------------------

    def index_directory(
        self,
        directory: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        batch_size: int = 32,
    ) -> Tuple[int, int]:
        """
        Загружает, разбивает документы, получает FRIDA-эмбеддинги
        и сохраняет всё в Supabase через LlamaIndex.
        """
        docs = self.load_documents(directory)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # разбиение на ноды
        nodes = splitter.get_nodes_from_documents(docs)
        logger.info(f"✂️ Разбито на {len(nodes)} чанков")

        # получаем эмбеддинги через FRIDA
        texts = [n.get_content() for n in nodes]
        embeddings = get_frida_embeddings(texts, batch_size=batch_size, device="cpu")

        for node, emb in zip(nodes, embeddings, strict=False):
            node.embedding = emb.tolist()

        # создаём индекс напрямую через VectorStoreIndex с FRIDA
        VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info(f"✅ Индексация завершена. Добавлено {len(nodes)} записей.")
        return len(nodes), 0  # добавлено, пропущено

    # ----------------------------------------------------------------------

    def query(self, text: str, top_k: int = 5) -> List[dict]:
        """Поиск по кастомному FRIDA-эмбеддингу через VectorStore."""
        emb = get_frida_embeddings([text])[0]
        results = self.vector_store.query(query_embedding=emb.tolist(), top_k=top_k)
        logger.info(f"🔍 Найдено {len(results)} совпадений")
        return results

    # ----------------------------------------------------------------------

    def stats(self) -> int:
        """Простая статистика таблицы Supabase (кол-во записей)."""
        response = self.client.table(self.table_name).select("id", count="exact").execute()
        return getattr(response, "count", 0)
