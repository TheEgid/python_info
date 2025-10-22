import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex  # noqa: F401
from llama_index.core.schema import TextNode
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

from others.frida import FridaEmbedding

logging.basicConfig(level=logging.INFO)
load_dotenv()


def fill_deeplake_dataset(
    documents: list, dataset_path: Path | str = "./deeplake_datasets/articles_index"
) -> tuple[DeepLakeVectorStore, list]:
    """
    Создаёт и заполняет DeepLake датасет с документами и эмбеддингами.

    Все TextNode имеют корректное '__node_content__' в metadata для LlamaIndex.
    """

    if not documents:
        logging.warning("Нет документов для индексации.")
        return None, []

    # Создаём DeepLake vector store
    vector_store = DeepLakeVectorStore(
        dataset_path=dataset_path,
        overwrite=True,
    )

    # Кастомный эмбеддинг
    embed_model = FridaEmbedding()
    Settings.embed_model = embed_model

    texts = []
    metadatas = []
    embeddings = []
    nodes = []

    for i, doc in enumerate(documents):
        text = doc.text
        embedding_array = np.array(embed_model._get_text_embedding(text), dtype=np.float32)

        # ✅ Метаданные для DeepLake (ключ '__node_content__' обязателен)
        metadata = {
            "doc_id": doc.doc_id or f"doc_{i}",
            "__node_content__": text
        }

        # Добавляем в списки
        texts.append(text)
        metadatas.append(metadata)
        embeddings.append(embedding_array)

        # Создаём TextNode с той же метадатой
        node = TextNode(
            text=text,
            embedding=embedding_array.tolist(),
            metadata=metadata
        )
        nodes.append(node)

    logging.info(f"Добавление {len(texts)} документов в DeepLake...")

    # Преобразуем эмбеддинги в массив
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Добавляем в DeepLake с правильными метаданными
    vector_store.vectorstore.add(
        text=texts,
        embedding=embeddings_array,
        metadata=metadatas
    )
    vector_store.vectorstore.dataset.flush()

    # ✅ Проверка сохранённых метаданных
    dataset = vector_store.vectorstore.dataset
    logging.info(f"Тензоры в датасете: {list(dataset.tensors.keys())}")
    logging.info(f"Размер тензора 'text': {len(dataset.text)}")
    logging.info(f"Размер тензора 'embedding': {len(dataset.embedding)}")
    logging.info(f"Размер тензора 'metadata': {len(dataset.metadata)}")

    return vector_store, nodes


def load_or_fill_deeplake(
    dataset_path: Path | str = "./deeplake_datasets/articles_index",
) -> tuple[DeepLakeVectorStore, list]:
    dataset_dir = Path(dataset_path)

    # Проверяем, существует ли папка с датасетом
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logging.info(f"Датасет {dataset_path} уже существует. Загружаем...")
        vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
        nodes = None
    else:
        logging.info(f"Датасет {dataset_path} не найден. Загружаем документы и создаем...")
        documents = SimpleDirectoryReader("articles/").load_data()
        vector_store, nodes = fill_deeplake_dataset(documents, dataset_path=dataset_path)

    dataset = vector_store.vectorstore.dataset
    logging.info(f"Тензоры в датасете: {list(dataset.tensors.keys())}")
    logging.info(f"Размер тензора 'text': {len(dataset.text)}")
    logging.info(f"Размер тензора 'embedding': {len(dataset.embedding)}")
    logging.info(f"Размер тензора 'metadata': {len(dataset.metadata)}")

    return vector_store, nodes

def main() -> None:
    try:
        vector_store, nodes = load_or_fill_deeplake()

        llm = OpenRouter(
            model="z-ai/glm-4-5-air:free",
            max_tokens=512,
            context_window=4096,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=FridaEmbedding(),
            llm=llm,
        )

        query_engine = index.as_query_engine(response_mode="compact", verbose=True, llm=llm)

        logging.info("Выполнение тестового запроса...")

        response = query_engine.query("Расскажи о телескопах")

        print("Ответ:", response)

    except Exception as e:
        logging.exception(f"❌ Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
