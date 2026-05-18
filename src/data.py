"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from typing import Any
import pandas as pd


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.
    """
    # Point to your final processed master data split from your notebooks
    # Note: Make sure this file is inside your datasets/ folder!
    df = pd.read_csv("datasets/df_engineered_outfield.csv")
    
    # Split features and target
    X = df.drop(columns=["market_value", "log_market_value"], errors="ignore")
    y = df["log_market_value"]
    
    # Simulating a simple train/test split or returning subsets
    X_train, X_test = X.iloc[:int(len(X)*0.8)], X.iloc[int(len(X)*0.8):]
    y_train, y_test = y.iloc[:int(len(y)*0.8)], y.iloc[int(len(y)*0.8):]
    
    return X_train, X_test, y_train, y_test