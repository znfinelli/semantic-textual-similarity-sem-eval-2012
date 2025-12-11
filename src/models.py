"""
models.py
---------
Machine Learning pipelines for Semantic Textual Similarity.

METHODOLOGY JUSTIFICATION:
1. Support Vector Regression (SVR):
   Used by the TakeLab team (Rank 2). They specifically found SVR with an RBF kernel 
   effective for combining heterogeneous features (lexical + syntactic).
   
2. Linear Regression (Ridge):
   Used by the UKP Lab team (Rank 1). They employed a "Simple Log-Linear Regression" 
   classifier, finding that linear combinations of diverse metrics avoided overfitting.

3. Random Forest:
   Included as a non-linear baseline to capture feature interactions (e.g., Negation + Overlap) 
   that linear models might miss.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from src.eval_utils import pearson_corr

def build_feature_matrix(df, feature_fn, has_score=True):
    """
    Transforms a DataFrame into a feature matrix X and target y.
    """
    feats = []
    # Extract features for every pair
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
    Standard stratified split to tune hyperparameters without touching Test data.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

# ==== Model Factories =======================================================

def make_ridge_model():
    """
    Ridge Regression (L2 Regularization).
    INSPIRATION: UKP Lab (SemEval Rank 1) used simple linear regression.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()), # Linear models need scaling
        ("model", Ridge())
    ])
    
    # Grid Search for Alpha (Regularization strength)
    param_grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    
    return GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

def make_svr_model():
    """
    Support Vector Regression (RBF Kernel).
    INSPIRATION: TakeLab (SemEval Rank 2) used SVR to model non-linear boundaries.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()), # SVM requires scaling
        ("model", SVR(kernel="rbf"))
    ])
    
    # Grid Search for C (Penalty) and Gamma (Kernel width)
    param_grid = {
        "model__C": [1.0, 10.0, 100.0],
        "model__gamma": ["scale", 0.1, 0.01],
        "model__epsilon": [0.1, 0.2]
    }
    
    return GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

def make_rf_model():
    """
    Random Forest Regressor.
    Used for Feature Importance analysis and capturing complex interactions 
    (e.g., "High Overlap" is good ONLY if "Negation Mismatch" is False).
    """
    rf = RandomForestRegressor(random_state=42)
    
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3]
    }
    
    return GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

# ==== Evaluation Helper ====================================================

def fit_and_eval(model, X_train, y_train, X_val, y_val, label="model"):
    """
    Trains a model and evaluates it using Pearson Correlation (Official SemEval Metric).
    """
    model.fit(X_train, y_train)
    
    # Get best estimator from GridSearch
    if hasattr(model, "best_estimator_"):
        best_model = model.best_estimator_
        # Print best params to show optimization was done
        print(f"[{label}] Best Params: {model.best_params_}")
    else:
        best_model = model

    y_pred = best_model.predict(X_val)
    score = pearson_corr(y_val, y_pred)
    
    return best_model, score