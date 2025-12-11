# src/eval_utils.py
from scipy.stats import pearsonr
import numpy as np

def pearson_corr(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return pearsonr(y_true, y_pred)[0]
