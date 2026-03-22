"""Weighted cosine KNN experiments (extracted from notebooks)."""

from .knn import KNN_Optimized, KNN_WeightedCosine
from .data import load_ag_news, load_imdb
from .vectorize import build_tfidf, vectorize_text

__all__ = [
    "KNN_Optimized",
    "KNN_WeightedCosine",
    "load_imdb",
    "load_ag_news",
    "build_tfidf",
    "vectorize_text",
]
