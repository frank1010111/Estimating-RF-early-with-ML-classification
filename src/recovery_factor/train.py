from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import xgboost as xgb
from ray import tune
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from recovery_factor.preprocess import clean, split


def tune_xgboost(x_train, y_train, n_trials: int = 10):
    search_space = {
        "max_depth": tune.randint(1, 12),
        "min_child_weight": tune.randint(1, 11),
        "eta": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
    }
    locked_params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "booster": "gbtree",
        "nthread": 4,
        "validate_parameters": "True",
        "alpha": 0.2,
        "lambda": 0.001,
        "colsample_bylevel": 0.9,
        "gamma": 0.01,
        "max_delta_step": 0.1,
        "num_class": 10,
    }

    def train_xgboost(config):
        model = xgb.XGBClassifier(**locked_params, **config)
        model.fit(x_train, y_train)

    # Use the ASHA scheduler for early stopping
    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)

    # Use HyperOptSearch for optimization
    search_algo = HyperOptSearch(metric="mlogloss", mode="min")

    # Create a Ray Tuner object with the search space, algorithm, and scheduler
    tuner = tune.Tuner(
        tune.with_resources(train_xgboost, {"cpu": 4}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            scheduler=scheduler,
            num_samples=n_trials,
        ),
        run_config=RunConfig(verbose=1),
    )

    # Run the hyperparameter search
    results = tuner.fit()

    return results


@click.command()
@click.option(
    "--training-data", type=click.Choice(["CA", "TA", "TC", "TCA"], case_sensitive=False)
)
@click.option("--n-trials", "-n", type=int, default=1)
def main(training_data, n_trials):
    data_root = Path(__file__).parent.parent.parent / "data_run"
    df_raw = pd.read_csv(data_root / f"{training_data}.csv")
    df_clean = clean(df_raw)
    click.echo("Training on these features: {df_clean.columns.to_list()}")
    X_train, _, y_train, _ = split(
        df_clean.drop(columns=["RECOVERY FACTOR", "Class"]),
        df_clean["Class"].astype("category").cat.codes,
    )

    results = tune_xgboost(X_train, y_train, n_trials)
    print(results.best_estimator)  # noqa: T201
    best_bst = results.best_estimator
    best_bst.save_model(data_root.parent / f"results/{training_data}.model")
    #     json.dump(results.get_best_result()["config"], f)
    click.echo(best_bst.get_params())


if __name__ == "__main__":
    main()
