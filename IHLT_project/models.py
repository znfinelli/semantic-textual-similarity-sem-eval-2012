
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from eval_utils import pearson_corr


# models.py
import numpy as np

def build_feature_matrix(df, feature_fn, has_score=True):
    """
    df: DataFrame with at least columns 's1', 's2'.
    feature_fn: function(s1, s2) -> 1D np.ndarray
    has_score: 
        - True: expects df['score'] and returns (X, y)
        - False: ignores gold, returns only X
    """
    feats = []
    for s1, s2 in zip(df["s1"], df["s2"]):
        feats.append(feature_fn(s1, s2))
    X = np.vstack(feats).astype(float)

    if has_score:
        y = df["score"].values.astype(float)
        return X, y
    else:
        return X



def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Simple train/validation split, stratification is not needed here.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# ==== Model factories =======================================================

def make_ridge_model():
    """
    Ridge regression with scaling.
    Returns a GridSearchCV object ready to .fit().
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    param_grid = {
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # we select by MSE, evaluate by Pearson
        cv=5,
        n_jobs=-1
    )
    return grid


def make_svr_model():
    """
    RBF SVR with scaling.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf"))
    ])

    param_grid = {
        "model__C": [1.0, 10.0, 100.0],
        "model__gamma": ["scale", 0.1, 0.01],
        "model__epsilon": [0.1, 0.2]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )
    return grid


def make_rf_model():
    """
    Random Forest Regressor (no scaling needed).
    """
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3]
    }

    grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )
    return grid


# ==== Training + evaluation helper =========================================

def fit_and_eval(model, X_train, y_train, X_val, y_val, label="model"):
    """
    Fit a (GridSearchCV or plain sklearn) model and return Pearson on val.
    """
    model.fit(X_train, y_train)
    if hasattr(model, "best_estimator_"):
        print(f"[{label}] best params:", model.best_params_)
        best_model = model.best_estimator_
    else:
        best_model = model

    y_pred_val = best_model.predict(X_val)
    pearson = pearson_corr(y_val, y_pred_val)
    print(f"[{label}] Val Pearson: {pearson:.4f}")
    return best_model, pearson
