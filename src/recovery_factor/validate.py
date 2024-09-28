from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from recovery_factor.preprocess import clean

data_root = Path(__file__) / "../../data_run"
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


def get_scores(params: dict[str, float], training_data: str) -> dict[str, dict[str, float]]:
    """
    Trains an XGBoost model and evaluates performance on the training and test set.

    Args:
        params (dict): Hyperparameters for the XGBoost model.
        training_data (str): Name of the CSV file (without extension) to load the training data.
        data_root (Path): The root directory containing the CSV files.
        fixed_params (dict): Fixed parameters for the XGBoost model.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 scores for
        training and test data.
    """
    # convert float to int for some hyperparams
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    # Load and preprocess the data
    df_raw = pd.read_csv(data_root / f"{training_data}.csv")
    df_clean = clean(df_raw)  # Assuming you have a 'clean' function for data cleaning

    X_train, X_test, y_train, y_test = train_test_split(
        df_clean.drop(columns=["RECOVERY FACTOR", "Class"]),
        df_clean["Class"].astype("category").cat.codes,
        test_size=0.2,
        random_state=42,
    )

    estimator = xgb.XGBClassifier(**fixed_params, **params)
    estimator.fit(X_train, y_train)

    scores = {}
    # Predictions and metrics on the training set
    train_predict = estimator.predict(X_train)
    scores["Train"] = {}
    scores["Train"]["accuracy"] = accuracy_score(y_train, train_predict)
    scores["Train"]["precision"] = precision_score(y_train, train_predict, average="weighted")
    scores["Train"]["recall"] = recall_score(y_train, train_predict, average="weighted")
    scores["Train"]["f1"] = f1_score(y_train, train_predict, average="weighted")

    # Predictions and metrics on the test set
    test_predict = estimator.predict(X_test)
    scores["Test"] = {}
    scores["Test"]["accuracy"] = accuracy_score(y_test, test_predict)
    scores["Test"]["precision"] = precision_score(y_test, test_predict, average="weighted")
    scores["Test"]["recall"] = recall_score(y_test, test_predict, average="weighted")
    scores["Test"]["f1"] = f1_score(y_test, test_predict, average="weighted")

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
    # convert float to int for some hyperparams
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])

    df_raw = pd.read_csv(data_root / f"{training_data}.csv")
    df_clean = clean(df_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean.drop(columns=["RECOVERY FACTOR", "Class"]),
        df_clean["Class"].astype("category").cat.codes,
        test_size=0.2,
        random_state=42,
    )

    estimator = xgb.XGBClassifier(**fixed_params, **params)
    estimator.fit(X_train, y_train)

    confusion_matrices = {}
    train_predict = estimator.predict(X_train)
    confusion_matrices["Train"] = confusion_matrix(y_train, train_predict)

    test_predict = estimator.predict(X_test)
    confusion_matrices["Test"] = confusion_matrix(y_test, test_predict)

    return confusion_matrices


# def ali_plots():
#     plt.figure(figsize=(8, 4))
#     plt.scatter(y_test, y_pred, c="r", label="Test Measured RF")
#     plt.xlim([0, 10])
#     plt.ylim([0, 10])
#     plt.plot([0, 10], [0, 10], color="black", linewidth=1, label="y = x")
#     plt.xlabel("Measured RF")
#     plt.ylabel("Estimated RF")
#     plt.title("Measured RF Vs Estimated RF (Test Dataset)")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("Measured RF Vs Estimated RF (Test Dataset)", bbox_inches="tight")
#
#     plt.figure(figsize=(8, 4))
#     plt.scatter(y_train, y_pred1, c="b", label="Train Measured RF")
#     plt.plot([0, 10], [0, 10], color="black", linewidth=1, label="y = x")
#     plt.xlim([0, 10])
#     plt.ylim([0, 10])
#     plt.xlabel("Measured RF")
#     plt.ylabel("Estimated RF")
#     plt.title("Measured RF Vs Estimated RF (Train Dataset)")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("Measured RF Vs Estimated RF (Train Dataset)", bbox_inches="tight")
#
#     plt.figure(figsize=(8, 4))
#     x_ax = range(len(y_test))
#     plt.plot(x_ax, y_test, ".", label="Measured", color="orange")
#     plt.plot(x_ax, y_pred, ".", label="Estimated", color="blue")
#     plt.xlabel("Sample")
#     plt.ylabel("Measured/Estimated RF")
#     plt.title("Measured RF Vs. Estimated RF Test Dataset")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("Measured Ksat Vs. Estimated Ksat (Test Dataset).png", bbox_inches="tight")
#
#     plt.figure(figsize=(8, 4))
#     x_ax = range(len(y_train))
#     plt.plot(x_ax, y_train, ".", label="Measured", color="orange")
#     plt.plot(x_ax, y_pred1, ".", label="Estimated", color="blue")
#     plt.xlabel("Sample")
#     plt.ylabel("Measured/Estimated RF")
#     plt.title("Measured RF Vs. Estimated RF Train Dataset")
#     plt.legend()
#     plt.grid(False)
#     plt.savefig("Measured Ksat Vs. Estimated Ksat (Train Dataset).png", bbox_inches="tight")
