from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    *,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=ngram_range,
        stop_words=stop_words,
    )


def vectorize_text(
    vectorizer: TfidfVectorizer,
    *,
    train_text,
    test_text,
):
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test
