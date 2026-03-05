# Fairlogue — Component 1  
## Observational Intersectional Fairness

---

## Overview

Component 1 quantifies **observed disparities** across intersectional subgroups using standard classification fairness metrics.

It answers:

> Do model error rates differ across intersectional groups?

This component evaluates:
- Single protected axes (e.g., race)
- Second protected axis (e.g., gender)
- Intersectional groups (e.g., Black | Female)

---

## Metrics Computed

For each subgroup:

- False Negative Rate (FNR)
- False Positive Rate (FPR)
- True Positive Rate (TPR)
- True Negative Rate (TNR)
- Intersectional disparity gaps

Optional controls:
- Minimum group size filtering
- Class balance requirement
- Plot generation

---

## Basic Usage

```python
from fairlogue.component1 import evaluate_intersectional_fairness

results, figures, intermediates = evaluate_intersectional_fairness(
    df=df,
    outcome="target",
    protected_1="race_factor",
    protected_2="gender_factor",
    features=features,
    model_type="lgbm",
    model_params=MODEL_PARAMS,
    threshold=0.5,
    return_intermediates=True,
    return_non_intersectional=True,
    min_group_size=20,
    require_class_balance=True,
)
```

---

Using a Development/Test Split

By default, Component 1 performs an internal random split.

To train on a development set and evaluate on a test set:

```python
results, figures, intermediates = evaluate_intersectional_fairness(
    df=df,
    train_df=dev_df,
    test_df=test_df,
    outcome="target",
    protected_1="race_factor",
    protected_2="gender_factor",
    features=features,
    model_type="lgbm",
    model_params=MODEL_PARAMS,
)
```

This:
-Fits the model on dev_df
-Evaluates fairness metrics on test_df

Supported Models
-Logistic Regression
-Random Forest
-LightGBM

Any estimator implementing:
-fit(X, y)
-predict_proba(X)

---

Interpretation

Component 1 detects observed disparities only.

It does not determine whether disparities arise due to:
-Structural covariate imbalance
-Systematic model dependence on group membership
-Random variation