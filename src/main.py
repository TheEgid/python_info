from typing import NoReturn

from dotenv import load_dotenv

from classes.VectorStoreFRIDA import VectorStoreFRIDA

# from classes.LLMService import LLMService
# from classes.VectorStoreFRIDA import VectorStoreFRIDA
# from others.frida import get_frida_embeddings
# from others.process_and_add_chunks import run_pipeline  # noqa: F401
# from others.run_search import calculate_cosine_similarity_with_embeddings, run_search


def main() -> NoReturn:
    """Точка входа."""
    load_dotenv()

    # run_scraper_separate_files()

    vs = VectorStoreFRIDA(table_name="test_novaya")


    res = vs.load_and_index_directory(directory="./articles", chunk_size=1024, chunk_overlap=64, batch_size=64)
    print(res)
    # затем можно искать:
    from others.frida import get_frida_embeddings

    q_emb = get_frida_embeddings(["Какую самую важную информацию содержит этот набор документов?"], device="cpu")
    print(vs.search(q_emb[0], top_k=5))
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
