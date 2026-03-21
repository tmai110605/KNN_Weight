from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfFeatures:
    vectorizer: TfidfVectorizer
    X_train: sparse.spmatrix
    X_val: sparse.spmatrix
    X_test: sparse.spmatrix
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def build_tfidf_features(
    *,
    train_df,
    val_df,
    test_df,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> TfidfFeatures:
    """Fit TF-IDF on train and transform val/test."""

    tfidf = TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    X_train = tfidf.fit_transform(train_df["full_text"])
    X_val = tfidf.transform(val_df["full_text"])
    X_test = tfidf.transform(test_df["full_text"])

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    return TfidfFeatures(
        vectorizer=tfidf,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
