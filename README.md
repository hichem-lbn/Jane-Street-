# Jane Street Real-Time Market Data Forecasting (Kaggle Competition Project) 


Research-oriented machine learning workflow for short-horizon financial prediction on the Jane Street Real-Time Market Data Forecasting dataset.

The project investigates weak-signal prediction under non-stationary time-series conditions using strict forward-looking validation, leakage-aware feature design, and robustness-focused evaluation.

The objective is not purely leaderboard optimization, but the study of how validation protocol, temporal sampling, and preprocessing decisions affect out-of-sample predictive stability.

---

# Project Overview

This project explores predictive modeling on large-scale anonymized financial market data using a statistically conservative research workflow.

Main research themes:

* strict chronological validation
* leakage-aware feature selection
* weak-signal detection
* non-stationary time-series modeling
* robustness diagnostics
* temporal stability analysis
* weighted financial evaluation metrics
* LightGBM-based baseline modeling

The workflow was intentionally designed under conservative financial ML assumptions in order to obtain more realistic out-of-sample evaluation.

---

# Dataset

Dataset source:

Jane Street Real-Time Market Data Forecasting Competition (Kaggle)

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting

Dataset characteristics:

* ~25 million observations
* 90+ anonymized market features
* multiple financial instruments (`symbol_id`)
* intraday timestamps (`date_id`, `time_id`)
* weighted evaluation metric
* low signal-to-noise ratio typical of financial prediction tasks

The dataset itself is not included in this repository because of licensing and storage constraints.

Because the features are anonymized, the project focuses on predictive structure and validation methodology rather than direct economic interpretation.

---

# Methodological Design

The workflow was intentionally designed to reduce unrealistic signal inflation caused by temporal leakage or unstable validation procedures.

Core methodological choices:

* strictly forward-looking train/validation split
* no random shuffling
* explicit exclusion of target-derived predictors
* exclusion of raw temporal index (`date_id`)
* weighted training and weighted evaluation
* sparse-feature filtering
* heavy-tail stabilization through feature clipping
* robustness-focused diagnostics

The project emphasizes conservative evaluation and reproducibility over aggressive optimization.

---

# Data Diagnostics

Initial exploratory analysis included:

* missing-value analysis
* target inspection
* feature sparsity analysis
* temporal coverage verification
* validation-date consistency checks

The dataset contains substantial missing-value structure, which was preserved because LightGBM handles missing values natively.

---

# Feature Engineering & Preprocessing

Several preprocessing decisions were introduced to improve robustness:

## Leakage-aware feature selection

The following variables were excluded from the feature space:

* `date_id`
* `weight`
* non-target `responder_*` variables

This prevents the model from exploiting target-related or regime-specific information that may not generalize out-of-sample.

## Sparse-feature filtering

Features with excessive missing-value ratios were removed using:

```python
missing_ratio < 0.8
```

## Heavy-tail stabilization

Financial predictors exhibited heavy-tailed distributions.

Feature values were clipped using:

```python
X.clip(-5, 5)
```

to reduce instability in tree-based splits.

---

# Temporal Sampling Strategy

An early baseline using contiguous recent periods produced unstable validation performance.

To improve regime coverage while preserving manageable local training constraints, the project used sparse chronological sampling:

```python
date_id[::20]
```

This allowed the model to observe multiple temporal regimes while keeping memory usage tractable.

---

# Target Diagnostics

The target exhibits heavy-tailed behavior and unstable cumulative dynamics over time, which motivates the use of strict chronological validation instead of random train/test splits.

## Distribution of responder_6

<p align="center">
<img src="responder6_distribution.png" width="550" >
</p>

## Rolling weighted responder_6 dynamics

<p align="center">
  <img src="rolling_responder6.png" width="800">
</p>

# Validation Framework

Validation was performed using a strictly chronological split:

```python
train: date_id < cutoff_date
valid: date_id ≥ cutoff_date
```

No shuffling or random cross-validation was used.

This setup provides a more realistic approximation of forward-looking financial prediction pipelines.

---

# Baseline Model

Main modeling choices:

* LightGBM regression
* weighted training
* competition-aligned weighted zero-mean R²
* native missing-value handling
* categorical handling of `symbol_id`
* early stopping
* moderate regularization

Main hyperparameters:

```python
learning_rate = 0.03
num_leaves = 128
min_data_in_leaf = 50
feature_fraction = 0.8
bagging_fraction = 0.8
lambda_l2 = 2.0
```

---

# Validation Results

| Experiment | Weighted R² |
|---|---|
| Initial baseline | ~0.0045 |
| Improved baseline | ~0.0086 |
| Harder OOS split | ~0.0031 |
| Stronger regularization | ~0.0082 |

Additional observations:

* ~85.7% of validation dates produced positive scores
* shuffled-target testing produced near-zero performance
* harder out-of-sample splits significantly reduced predictive performance
* validation performance varied across temporal regimes

These observations are consistent with low signal-to-noise financial prediction problems.

---

# Robustness Diagnostics

Several robustness checks were implemented:

* leakage diagnostics
* shuffled-target testing
* harder out-of-sample validation
* date-level validation stability
* overfitting analysis
* regularization sensitivity testing

Main observations:

* no obvious leakage patterns detected
* positive out-of-sample predictive signal
* expected overfitting structure typical of financial ML tasks
* weaker performance under harder validation regimes

---

# Research Observations

The experiments highlighted several important properties of financial ML workflows:

* methodological choices impacted performance more than model complexity alone
* stronger regularization slightly reduced predictive strength
* signal quality varied across time regimes
* validation protocol design strongly affected measured performance
* temporal sampling materially influenced signal stability

The project reinforces the importance of conservative validation design in non-stationary financial datasets.

---

# Limitations

This project uses a fully anonymized dataset, which prevents direct economic interpretation of the learned predictive structure.

The workflow therefore focuses on statistical prediction methodology rather than trading strategy construction, portfolio optimization, or execution modeling.

---

# Future Work

Potential extensions include:

* rolling walk-forward validation
* lag-based feature engineering
* cross-sectional normalization
* regime-aware modeling
* feature-neutralization experiments
* ensemble methods
* more advanced temporal validation frameworks

---

# Technologies Used

* Python
* Pandas
* NumPy
* LightGBM
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# Repository Structure

```text
jane-street-financial-forecasting-research/
│
├── notebooks/
├── models/
├── artifacts/
├── figures/
├── README.md
├── requirements.txt
└── .gitignore
```
