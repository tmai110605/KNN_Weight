from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TextDatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame | None = None


def load_imdb(
    *,
    train_size: int = 25_000,
    test_size: int = 3_000,
    seed: int = 42,
    val_size: float | None = 0.2,
) -> TextDatasetSplits:
    """Load IMDb from HuggingFace datasets and return DataFrames.

    Labels: 0 = negative, 1 = positive.

    Note: this mirrors the notebook behavior by shuffling then selecting a subset.
    """

    dataset = load_dataset("imdb")

    train_data = dataset["train"].shuffle(seed=seed).select(range(int(train_size)))
    test_data = dataset["test"].shuffle(seed=seed).select(range(int(test_size)))

    train_df = pd.DataFrame({"full_text": train_data["text"], "label": train_data["label"]})
    test_df = pd.DataFrame({"full_text": test_data["text"], "label": test_data["label"]})

    if val_size is None or val_size <= 0:
        return TextDatasetSplits(train=train_df, test=test_df, val=None)

    train_df, val_df = train_test_split(
        train_df,
        test_size=float(val_size),
        stratify=train_df["label"],
        random_state=seed,
    )
    return TextDatasetSplits(train=train_df, test=test_df, val=val_df)


def load_ag_news(
    *,
    train_size: int = 8_000,
    test_size: int = 600,
) -> TextDatasetSplits:
    """Load AG News from HuggingFace datasets and return DataFrames.

    Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """

    dataset = load_dataset("ag_news")

    train_data = dataset["train"].select(range(int(train_size)))
    test_data = dataset["test"].select(range(int(test_size)))

    train_df = pd.DataFrame({"full_text": train_data["text"], "label": train_data["label"]})
    test_df = pd.DataFrame({"full_text": test_data["text"], "label": test_data["label"]})

    return TextDatasetSplits(train=train_df, test=test_df, val=None)
