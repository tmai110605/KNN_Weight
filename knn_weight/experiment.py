from __future__ import annotations

import argparse

from sklearn.metrics import accuracy_score

from .data import load_ag_news_splits
from .models import KNNOptimized, KNNWeightedCosine
from .vectorize import build_tfidf_features


def run(*, k: int = 5, train_size: int = 8000, test_size: int = 600, max_features: int = 5000) -> None:
    splits = load_ag_news_splits(train_size=train_size, test_size=test_size)
    feats = build_tfidf_features(
        train_df=splits.train_df,
        val_df=splits.val_df,
        test_df=splits.test_df,
        max_features=max_features,
    )

    knn = KNNOptimized(k=k)
    knn.fit(feats.X_train, feats.y_train)
    _ = knn.predict_proba(feats.X_val)
    _ = knn.predict_proba(feats.X_test)
    acc_knn = accuracy_score(feats.y_test, knn.predict(feats.X_test))
    print(f"KNNOptimized: {acc_knn:.4f}")

    knn_w = KNNWeightedCosine(k=k)
    knn_w.fit(feats.X_train, feats.y_train)
    _ = knn_w.predict_proba(feats.X_val)
    _ = knn_w.predict_proba(feats.X_test)
    acc_knn_w = accuracy_score(feats.y_test, knn_w.predict(feats.X_test))
    print(f"KNNWeightedCosine: {acc_knn_w:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AG News TF-IDF + KNN experiments (from notebook).")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    run(k=args.k, train_size=args.train_size, test_size=args.test_size, max_features=args.max_features)


if __name__ == "__main__":
    main()
