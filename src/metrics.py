"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values.
    """
    # Convert inputs to clean numpy arrays to prevent indexing mismatch crashes
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate your core performance metrics
    # Note: Since your target variable was log-transformed in D02/D03, 
    # taking the standard RMSE of log values gives you the RMSLE perfectly!
    rmsle = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "RMSLE": rmsle,
        "MAE": mae,
        "R2_Score": r2
    }