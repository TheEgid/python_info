from typing import NoReturn

from dotenv import load_dotenv

from others.wiki_scraper import run_scraper_separate_files

# from classes.LLMService import LLMService
# from classes.VectorStoreFRIDA import VectorStoreFRIDA
# from others.frida import get_frida_embeddings
# from others.process_and_add_chunks import run_pipeline  # noqa: F401
# from others.run_search import calculate_cosine_similarity_with_embeddings, run_search


def main() -> NoReturn:
    """Точка входа."""
    load_dotenv()

    run_scraper_separate_files()
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
