import logging
from typing import NoReturn

from dotenv import load_dotenv

from classes.VectorStoreFRIDA import VectorStoreFRIDA

# from classes.LLMService import LLMService
# from classes.VectorStoreFRIDA import VectorStoreFRIDA
# from others.frida import get_frida_embeddings
# from others.process_and_add_chunks import run_pipeline  # noqa: F401
# from others.run_search import calculate_cosine_similarity_with_embeddings, run_search
    # run_scraper_separate_files()

# Настройка логирования
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

logger = logging.getLogger(__name__)

def main() -> NoReturn:
    """Точка входа в приложение."""
    load_dotenv()

    try:
        vs = VectorStoreFRIDA(table_name="test_novaya")

        before = vs.stats()
        logger.info(f"📊 До индексации: {before}")

        vs.index_directory(
            directory="./articles",
            chunk_size=1024,
            chunk_overlap=64,
            batch_size=32,
        )

        after = vs.stats()
        logger.info(f"📊 После индексации: {after}")

        # пример запроса
        query = "What is the Hubble Space Telescope?"
        results = vs.query(query)
        for r in results:
            logger.info(f"⭐ {r['content'][:120]}...")

        # # ЭТАП 2: Тестирование поиска
        # logger.info("▶️  ЭТАП 2: Тестирование поиска")
        # logger.info("─" * 80)

        # test_query = "Какую самую важную информацию содержит этот набор документов?"
        # logger.info(f"📝 Тестовый запрос: '{test_query}'\n")

        # # Получаем эмбеддинг запроса
        # logger.info("🔄 Получение эмбеддинга запроса...")
        # q_emb = get_frida_embeddings(
        #     [test_query],
        #     device="cpu",
        # )

        # # Выполняем поиск
        # results = vs.search(
        #     q_emb[0],
        #     top_k=5,
        #     match_threshold=0.0,  # Показываем все результаты
        # )

        # # Выводим результаты
        # logger.info(f"\n🔍 Результаты поиска: найдено {len(results)} документов\n")
        # logger.info("─" * 80)

        # if results:
        #     for idx, (doc_id, content, metadata, similarity) in enumerate(results, 1):
        #         logger.info(f"\n{idx}. 📄 Сходство: {similarity:.4f}")
        #         logger.info(f"   ID: {doc_id}")
        #         logger.info(f"   Текст: {content[:100]}...")
        #         if metadata:
        #             logger.info("   Метаданные:")
        #             for key, value in metadata.items():
        #                 logger.info(f"      {key}: {value}")
        # else:
        #     logger.warning("⚠️  Результаты не найдены")

        # logger.info("\n" + "=" * 80)
        # logger.info("✅ ПРОЦЕСС УСПЕШНО ЗАВЕРШЕН!")
        # logger.info("=" * 80 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Процесс прерван пользователем")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Критическая ошибка: {e}", exc_info=True)
        return 1

    # run_pipeline()

    # vector_store = VectorStoreFRIDA()
    # embedding_function = get_frida_embeddings
    # user_prompt = "Tell me about space exploration on the Moon and Mars."

    # top_text = run_search(user_prompt, vector_store, embedding_function)

    # similarity = calculate_cosine_similarity_with_embeddings(user_prompt, top_text)
    # print(f"Similarity: {similarity}")

    # augmented_input = user_prompt + " " + top_text

    # llm_service = LLMService()

    # start_time = time.time()
    # gpt_response = llm_service.summarize_texts(augmented_input)
    # response_time = time.time() - start_time

    # llm_service.print_formatted_response(gpt_response)
    # print(f"Response Time: {response_time:.2f} seconds")

    # similarity = calculate_cosine_similarity_with_embeddings(user_prompt, gpt_response)
    # print(f"Similarity: {similarity}")


if __name__ == "__main__":
    main()
