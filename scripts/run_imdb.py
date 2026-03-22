from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()

    from knn_weight import KNN_Optimized, KNN_WeightedCosine
    from knn_weight.data import load_imdb
    from knn_weight.vectorize import build_tfidf, vectorize_text

    parser = argparse.ArgumentParser(description="Run (weighted) cosine KNN on IMDb")
    parser.add_argument("--train-size", type=int, default=25000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()

    print("Loading IMDb dataset...")
    splits = load_imdb(train_size=args.train_size, test_size=args.test_size, seed=args.seed, val_size=0.2)
    train_df = splits.train
    test_df = splits.test

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nLabel distribution:")
    print(train_df["label"].value_counts().sort_index())

    tfidf = build_tfidf(max_features=args.max_features)
    X_train, X_test = vectorize_text(tfidf, train_text=train_df["full_text"], test_text=test_df["full_text"])

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    knn = KNN_Optimized(k=args.k)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    print(f"KNN_Optimized (vote) acc: {acc_knn:.4f}")

    knn_w = KNN_WeightedCosine(k=args.k)
    knn_w.fit(X_train, y_train)
    acc_knn_w = accuracy_score(y_test, knn_w.predict(X_test))
    print(f"KNN_WeightedCosine acc: {acc_knn_w:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
