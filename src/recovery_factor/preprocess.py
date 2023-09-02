"""Pre-process data to prepare it for machine learning."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def clean(
    raw_data: pd.DataFrame,
    column_missing_fraction: float = 0.55,
    row_missing_fraction: float = 0.7,
) -> pd.DataFrame:
    """Clean dataframe, removing invalid numbers and columns with too many NaNs.

    Args:
        raw_data (pd.DataFrame): Input dataframe
        column_missing_fraction (float): Fraction of missing values to justify removing a column
        row_missing_fraction (float): Fraction of missing values to justify remove

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    cleaned_data = raw_data.copy().rename(columns=lambda x: x.split(" (")[0])
    rf_col = "RECOVERY FACTOR"
    cleaned_data = cleaned_data.dropna(how="any", subset=[rf_col])
    for col in (rf_col, "POROSITY", "WATER SATURATION"):
        cleaned_data = cleaned_data[
            lambda x: x[col].isnull() | ((x[col] >= 0) & (x[col] <= 1))  # noqa: B023
        ]
    cleaned_data = cleaned_data[(cleaned_data["FVF"] >= 0) | (cleaned_data["FVF"].isnull())]
    cleaned_data = cleaned_data[(cleaned_data["FVF"] <= 10) | (cleaned_data["FVF"].isnull())]
    cleaned_data = cleaned_data[(cleaned_data["GOR"] >= 0) | (cleaned_data["GOR"].isnull())]
    cleaned_data = cleaned_data[(cleaned_data["GOR"] <= 60) | (cleaned_data["GOR"].isnull())]
    cleaned_data = cleaned_data[
        (cleaned_data["RESERVES"] >= 0) | (cleaned_data["RESERVES"].isnull())
    ]
    cleaned_data = cleaned_data[
        (cleaned_data["RESERVES"] <= 5e11) | (cleaned_data["RESERVES"].isnull())
    ]

    column_missing_threshold = round(column_missing_fraction * cleaned_data.shape[1])
    row_missing_threshold = round(row_missing_fraction * cleaned_data.shape[0])

    cleaned_data = cleaned_data.dropna(axis=0, thresh=column_missing_threshold)
    cleaned_data = cleaned_data.dropna(axis=1, thresh=row_missing_threshold)
    cleaned_data = cleaned_data.sort_values(by="RECOVERY FACTOR", ascending=True)
    cleaned_data = cleaned_data.reset_index(drop=True)

    dfnum = cleaned_data.select_dtypes(include=["number"])

    dfcat = cleaned_data.select_dtypes(include=["object"])
    dfcat = dfcat.astype("category")
    dfcat = dfcat.fillna(dfcat.mode().iloc[0])

    df_out = pd.concat([dfnum, dfcat], axis=1)
    return df_out


def split(df_x: pd.DataFrame, df_y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.1, random_state=42, stratify=df_y
    )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return X_train, X_test, y_train, y_test
