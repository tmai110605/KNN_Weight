from __future__ import annotations

import numpy as np


class KNN_Optimized:
    def __init__(self, k: int = 5):
        self.k = int(k)

    def fit(self, X, y):
        self.X_train = X.toarray() if hasattr(X, "toarray") else X
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)

        self.X_train_norm = self._normalize(self.X_train)
        return self

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return X / norm

    def _cosine_similarity(self, X):
        X = X.toarray() if hasattr(X, "toarray") else X
        X_norm = self._normalize(X)
        return np.dot(X_norm, self.X_train_norm.T)

    def predict(self, X):
        sims = self._cosine_similarity(X)

        knn_idx = np.argsort(sims, axis=1)[:, -self.k :]
        knn_labels = self.y_train[knn_idx]

        preds = []
        for labels in knn_labels:
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.asarray(preds)

    def predict_proba(self, X):
        sims = self._cosine_similarity(X)
        knn_idx = np.argsort(sims, axis=1)[:, -self.k :]
        knn_labels = self.y_train[knn_idx]

        proba = np.zeros((X.shape[0], len(self.classes_)))

        for i, labels in enumerate(knn_labels):
            for j, c in enumerate(self.classes_):
                proba[i, j] = np.sum(labels == c)

        proba /= self.k
        return proba


class KNN_WeightedCosine:
    def __init__(self, k: int = 5):
        self.k = int(k)

    def fit(self, X, y):
        self.X_train = X.toarray() if hasattr(X, "toarray") else X
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)

        self.X_train_norm = self._normalize(self.X_train)
        return self

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        return X / norm

    def _cosine_similarity(self, X):
        X = X.toarray() if hasattr(X, "toarray") else X
        X_norm = self._normalize(X)
        return np.dot(X_norm, self.X_train_norm.T)

    def predict(self, X):
        sims = self._cosine_similarity(X)

        knn_idx = np.argpartition(sims, -self.k, axis=1)[:, -self.k :]

        preds = []
        for i, idx in enumerate(knn_idx):
            labels = self.y_train[idx]
            weights = np.clip(sims[i, idx], 0, None)

            class_scores = {}
            for c in self.classes_:
                class_scores[c] = weights[labels == c].sum()

            preds.append(max(class_scores, key=class_scores.get))

        return np.asarray(preds)

    def predict_proba(self, X):
        sims = self._cosine_similarity(X)
        knn_idx = np.argpartition(sims, -self.k, axis=1)[:, -self.k :]

        proba = np.zeros((X.shape[0], len(self.classes_)))

        for i, idx in enumerate(knn_idx):
            labels = self.y_train[idx]
            weights = np.clip(sims[i, idx], 0, None)

            for j, c in enumerate(self.classes_):
                proba[i, j] = weights[labels == c].sum()

            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
            else:
                proba[i] = 1.0 / len(self.classes_)

        return proba
