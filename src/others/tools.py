from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_enhanced_similarity(text1, text2):  # noqa: ANN001, ANN201
    """
    Упрощенная функция вычисления косинусного сходства между двумя текстами
    """
    vectorizer = TfidfVectorizer(stop_words=["english", "russian"], ngram_range=(1, 2), use_idf=True, norm="l2")

    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return max(0, float(f"{cosine_sim:.4f}"))
    except Exception:
        return 0.0
