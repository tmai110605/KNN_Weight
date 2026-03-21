from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class AgNewsSplits:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_map: dict[int, str]


def load_ag_news_splits(
    *,
    train_size: int = 8000,
    test_size: int = 600,
    val_size: float = 0.2,
    random_state: int = 42,
) -> AgNewsSplits:
    """Load AG News and return train/val/test dataframes.

    DataFrames have columns:
        - full_text: str
        - label: int
    """

    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")

    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    train_data = dataset["train"].select(range(int(train_size)))
    test_data = dataset["test"].select(range(int(test_size)))

    train_df = pd.DataFrame({"full_text": train_data["text"], "label": train_data["label"]})
    test_df = pd.DataFrame({"full_text": test_data["text"], "label": test_data["label"]})

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nLabel distribution:")
    print(train_df["label"].value_counts().sort_index())

    train_df, val_df = train_test_split(
        train_df,
        test_size=float(val_size),
        stratify=train_df["label"],
        random_state=int(random_state),
    )

    return AgNewsSplits(train_df=train_df, val_df=val_df, test_df=test_df, label_map=label_map)
