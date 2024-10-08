# Estimating oil RF at exploration

This repository is the companion to the manuscript _Estimating oil recovery
factor at exploration stage using XGBoost classification_. It contains _all_ the
code used to generate the figures and tables used in that manuscript. The
software is meant specifically for training and evaluating machine learning
models for predicting oil recovery factors, but much of it is sufficiently
general-purpose that it could be adapted for any ordinal categorical regression
task with multiple databases.

## Project setup

Run

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install .[dev]
```

You might have to install `"numba==0.60.0"`, then `".[dev]"`

The code used for the 2024 submission is in `src/recovery_factor`. The notebooks
that generate the plots are in `notebooks`.

The models were trained at the command line with the bash one-liner

```sh
echo "TCA TC CA TA" | tr ' ' '\n' | parallel -j 2 rf-train -n 200 --training-data {}
```

## The databases

**TORIS**:
<https://edx.netl.doe.gov/dataset/toris-an-integrated-decision-support-system-for-petroleum-e-p-policy-evaluation>

TORIS comes from the Department of Energy.

**Atlas**: <https://www.data.boem.gov/Main/GandG.aspx>

This is an atlas produced by the Bureau of Ocean Energy Management.

**Commercial** This data is not publicly available.
