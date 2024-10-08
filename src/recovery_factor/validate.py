from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from recovery_factor.preprocess import clean

data_root = Path(__file__).parent / "../../data_run"
fixed_params = {
    "objective": "multi:softmax",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "nthread": 8,
    "colsample_bylevel": 0.9,
    "gamma": 0.01,
    "alpha": 0.2,
    "lambda": 0.01,
    "max_delta_step": 0.1,
    "num_class": 10,
    "device": "cpu",
}
CLASS_DICT = {"1% to 10%": "0"} | {f"{n}0% to {n+1}0%": str(n) for n in range(1, 10)}


def within_1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the proportion of predictions that are within ±1 of the true class.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.

    Returns:
        float: The within-1 accuracy.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate if the prediction is within ±1 of the true value
    within_1 = np.abs(y_true - y_pred) <= 1

    # Return the proportion of predictions that are within ±1
    return np.mean(within_1)


DEFAULT_METRICS = {
    "accuracy": accuracy_score,
    "within-1 accuracy": within_1_accuracy,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
    "R2": r2_score,
}


def get_scores(
    params: dict[str, float], training_data: str, metrics: dict[str, Callable] | None = None
) -> dict[str, dict[str, float]]:
    """
    Train an XGBoost model and evaluates performance on the training and test set.

    Args:
        params (dict): Hyperparameters for the XGBoost model.
        training_data (str): Name of the CSV file (without extension) to load the training data.
        metrics (dict): ML metrics to report. Accepts a dictionary of name:callable.
            The callables should take y_true and y_pred as arguments and return a float.
            By default, includes accuracy, precision, recall, f1, and R-squared.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 scores for
        training and test data.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    X_train, X_test, y_train, y_test = get_split_data(training_data)
    estimator = train_model(params, training_data)

    scores = {}
    train_predict = estimator.predict(X_train)
    scores["Train"] = calculate_metrics(y_train, train_predict, metrics)

    test_predict = estimator.predict(X_test)
    scores["Test"] = calculate_metrics(y_test, test_predict, metrics)

    X_independent, y_independent = get_independent_data(training_data)
    if X_independent is not None:
        independent_predict = estimator.predict(X_independent)
        scores["Independent"] = calculate_metrics(y_independent, independent_predict, metrics)

    return scores


def get_confusion_matrices(params: dict[str, float], training_data: str):
    """
    Trains an XGBoost model and outputs confusion matrices for both
    the training and test datasets.

    Args:
        params (dict): Hyperparameters for the XGBoost model.
        training_data (str): Name of the CSV file (without extension) to load the training data.
    Choices are TC, TA, CA, and TCA
        data_root (Path): The root directory containing the CSV files.
        fixed_params (dict): Fixed parameters for the XGBoost model.

    Returns:
        dict: A dictionary containing the confusion matrices for both the training and test data.
    """
    estimator = train_model(params, training_data)
    X_train, X_test, y_train, y_test = get_split_data(training_data)

    confusion_matrices = {}
    train_predict = estimator.predict(X_train)
    confusion_matrices["Train"] = confusion_matrix(y_train, train_predict)

    test_predict = estimator.predict(X_test)
    confusion_matrices["Test"] = confusion_matrix(y_test, test_predict)

    X_independent, y_independent = get_independent_data(training_data)
    if X_independent is not None:
        independent_predict = estimator.predict(X_independent)
        confusion_matrices["Independent"] = confusion_matrix(y_independent, independent_predict)
    return confusion_matrices


def calculate_metrics(y_true, y_pred, metrics):
    """
    Calculate the given metrics on the provided true and predicted values.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        metrics (dict): A dictionary where keys are metric names and values are metric functions.

    Returns:
        dict: A dictionary containing calculated metric values.
    """
    return {metric_name: metric_fn(y_true, y_pred) for metric_name, metric_fn in metrics.items()}


def train_model(params: dict[str, float], training_data: str):
    X_train, _, y_train, _ = get_split_data(training_data)
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    estimator = xgb.XGBClassifier(**fixed_params, **params)
    estimator.fit(X_train, y_train)
    return estimator


def get_split_data(training_data: str):
    df_raw = pd.read_csv(data_root / f"{training_data}.csv")
    df_clean = clean(df_raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, X_test, y_train, y_test = train_test_split(
            df_clean.drop(columns=["RECOVERY FACTOR", "Class"]),
            df_clean["Class"].replace(CLASS_DICT).astype(int),
            test_size=0.2,
            random_state=42,
        )
    return X_train, X_test, y_train, y_test


def get_independent_data(training_data: str):
    independent = set("TCA") - set(training_data)
    if independent:
        csv_independent = next(
            (data_root / "independent/").glob(independent.pop().lower() + "*.csv")
        )
        df_independent = pd.read_csv(csv_independent).pipe(clean)
        X_independent = df_independent.drop(columns=["RECOVERY FACTOR", "Class"])
        y_independent = df_independent["Class"].replace(CLASS_DICT).astype(int)

        return X_independent, y_independent
    return None, None
