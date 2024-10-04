from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import bootstrap
from sklearn.inspection import permutation_importance

from recovery_factor.validate import get_independent_data, get_split_data, train_model


def calculate_all_feature_importances(
    params: dict[str, float], training_data: str
) -> dict[str, pd.DataFrame]:
    X_train, X_test, _y_train, y_test = get_split_data(training_data)
    estimator = train_model(params, training_data)
    importances = {
        "SHAP training": compute_shap_feature_importance(estimator, X_train),
        "Permutation testing": compute_permutation_feature_importance(
            estimator, X_test, y_test, scoring="accuracy"
        ),
    }

    X_independent, y_independent = get_independent_data(training_data)
    if X_independent is not None:
        importances["Permutation independent"] = compute_permutation_feature_importance(
            estimator, X_independent, y_independent, scoring="accuracy"
        )

    return importances


def compute_shap_feature_importance(
    estimator: xgb.XGBClassifier, X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute SHAP values for a trained XGBoost model.

    Args:
        estimator (xgb.XGBClassifier): A trained XGBoost classifier.
        X_train (pd.DataFrame): Training features used to fit the model.
        y_train (pd.Series): Training labels (target variable) used to fit the model.

    Returns:
        pd.DataFrame: A dataframe with feature names and corresponding mean absolute SHAP values
        which represent feature importance. Features are sorted in descending order of importance.
    """
    N_RESAMPLES = 200

    if not hasattr(estimator, "feature_importances_"):
        msg = "The estimator should be trained before computing SHAP values."
        raise ValueError(msg)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_train)

    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_importance_se = np.array(
        [
            bootstrap((row,), np.mean, vectorized=False, n_resamples=N_RESAMPLES).standard_error
            for row in shap_importance
        ]
    )

    return pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": shap_importance.mean(axis=1),
            "Importance SE": shap_importance_se,
        }
    ).sort_values(by="Importance", ascending=False)


def compute_permutation_feature_importance(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "accuracy",
    n_repeats: int = 40,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a trained estimator using the given data.

    Args:
        estimator: A trained estimator (e.g., XGBoost, RandomForest) supporting scikit-learn API.
        X_train (pd.DataFrame): Training features used to evaluate the model.
        y_train (pd.Series): Training labels (target variable) used to evaluate the model.
        scoring (str): Scoring method to evaluate model performance (default: 'accuracy').
        n_repeats (int): Number of times to permute a feature (default: 10).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        pd.DataFrame: A dataframe with feature names and their corresponding permutation
        importances scores, sorted in descending order of importance.
    """
    N_RESAMPLES = 100
    if not hasattr(estimator, "predict"):
        msg = "The estimator should be trained before computing SHAP values."
        raise ValueError(msg)
    result = permutation_importance(
        estimator,
        X_train,
        y_train,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=42,
    )
    bootstrap_se = np.array(
        [
            bootstrap((row,), np.mean, vectorized=False, n_resamples=N_RESAMPLES).standard_error
            for row in result.importances
        ]
    )

    return pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": result.importances_mean,
            "Importance SE": bootstrap_se,
        }
    ).sort_values(by="Importance", ascending=False)
