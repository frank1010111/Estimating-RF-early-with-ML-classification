from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope

from recovery_factor.preprocess import clean, split

fixed_params = {
    "objective": "multi:softmax",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "nthread": 8,
    "colsample_bylevel": 0.9,
    "alpha": 0.2,
    "lambda": 0.01,
    "gamma": 0.01,
    "max_delta_step": 0.1,
    "num_class": 10,
    "device": "cpu",
}


def objective(params, x_train, y_train, n_folds=5):
    """Objective function for HyperOpt to minimize during hyperparameter tuning.

    This function performs k-fold cross-validation on the provided training data
    using the XGBoost classifier, and returns the cross-validated log-loss.

    Args:
        params (dict): Hyperparameters to tune.
        x_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target labels.
        n_folds (int, optional): Number of folds for k-fold cross-validation. Default is 5.

    Returns:
        dict: A dictionary containing the loss (log-loss) and status for HyperOpt.
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    full_params = fixed_params | params

    cv_results = xgb.cv(
        full_params,
        dtrain,
        num_boost_round=150,
        nfold=n_folds,
        early_stopping_rounds=10,  # Early stopping to prevent overfitting
        as_pandas=True,
        seed=222,
    )

    mlogloss = cv_results["test-mlogloss-mean"].mean()

    return {"loss": mlogloss, "status": STATUS_OK}


def tune_xgboost(x_train, y_train, n_trials: int = 50, n_folds: int = 10):
    """
    Perform hyperparameter tuning for an XGBoost classifier.

    This function searches for the best hyperparameters based on cross-validated log-loss
    and returns the best hyperparameter configuration found.

    Args:
        x_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target labels.
        n_trials (int, optional): Number of hyperparameter optimization trials. Default is 50.
        n_folds (int, optional): Number of folds for k-fold cross-validation. Default is 5.

    Returns:
        best (dict): The best hyperparameter configuration found.
    """

    def objective_with_data(params):
        return objective(params, x_train, y_train, n_folds)

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 12, 1)),
        "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 11, 1)),
        "eta": hp.loguniform("eta", np.log(1e-3), np.log(0.5)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        # "alpha": hp.loguniform("alpha", -3, 1),
        # "lambda": hp.loguniform("lambda", -3, 1),
    }

    trials = Trials()
    best = fmin(
        fn=objective_with_data,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    return best


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--training-data", "-t", type=click.Choice(["CA", "TA", "TC", "TCA"], case_sensitive=False)
)
@click.option("--n-trials", "-n", type=int, default=1)
def main(training_data, n_trials):
    data_root = Path(__file__).parent.parent.parent / "data_run"
    df_raw = pd.read_csv(data_root / f"{training_data}.csv")
    df_clean = clean(df_raw)
    click.echo(f"Training on these features: {df_clean.columns.to_list()}")
    X_train, _, y_train, _ = split(
        df_clean.drop(columns=["RECOVERY FACTOR", "Class"]),
        df_clean["Class"].astype("category").cat.codes,
    )

    best_estimator = tune_xgboost(X_train, y_train, n_trials)
    click.echo(best_estimator)
    with (data_root.parent / f"results/{training_data}.json").open("w") as f:
        json.dump(best_estimator, f)


if __name__ == "__main__":
    main()
