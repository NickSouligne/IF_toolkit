
---

# 📄 component3.md

```markdown
# Fairlogue — Component 3  
## Counterfactual Group-Membership Fairness

---

## Overview

Component 3 evaluates whether disparities persist **after conditioning on covariates**.

It answers:

> Would predictions change if group membership were counterfactually reassigned while holding covariates fixed?

This is a model-dependence fairness evaluation.

---

## Conceptual Framework

For each individual:

1. Fit an outcome model
2. Generate counterfactual predictions by replacing group membership
3. Compute:
   - Counterfactual False Negative Rate (cFNR)
   - Counterfactual False Positive Rate (cFPR)
4. Estimate:
   - Disparity contrasts (Δ)
   - U-values (retained unfairness metric)

---

## Methods

- **SR (Single Robust)** — outcome modeling only  
- **DR (Doubly Robust)** — outcome + propensity modeling  

---

## Basic Usage

```python
from lightgbm import LGBMClassifier
import iftoolkit as ift

lgbm = LGBMClassifier(**MODEL_PARAMS)

m = ift.Component3.Model(
    data=df,
    outcome="target",
    protected_characteristics=("race_factor", "gender_factor"),
    covariates=features,
    outcome_estimator=lgbm,
    method="sr",
    n_splits=5,
    random_state=42,
)

m.pre_process_data()
m.fit_fairness()

summary = m.summarize()
estimates, deltas, uvalues = m.plots(delta_uval=0.10)
```

Using a Development/Test Split
Always split after preprocessing so the internal intersectional column exists.

```python
m.pre_process_data()

proc = m.data

dev_df_proc = proc[proc["state_of_residence_source_value"].isin(development_states)]
test_df_proc = proc[proc["state_of_residence_source_value"].isin(test_states)]

m.fit_fairness(
    train_df=dev_df_proc,
    test_df=test_df_proc,
    gen_null=True,
    bootstrap="rescaled"
)
```


This:
-Fits the outcome model on dev_df_proc
-Computes counterfactual metrics on test_df_proc


---
Outputs

Component 3 provides:
-Counterfactual FNR / FPR
-Null distributions (optional)
-Bootstrap confidence intervals
-Disparity contrasts (Δ)
-U-values

U-Value Interpretation
The U-value quantifies retained unfairness.
-U ≈ 0 → No systematic unfairness after conditioning
-U > δ → Meaningful model dependence on group membership
δ is the acceptable fairness threshold (e.g., 0.1).


| Component   | Measures                   | Conditioning     | Interpretation                      |
| ----------- | -------------------------- | ---------------- | ----------------------------------- |
| Component 1 | Observed disparities       | None             | Detects raw group differences       |
| Component 3 | Counterfactual disparities | Covariates fixed | Detects systematic model dependence |


---
Recommended Workflow

1.Run Component 1
2.Identify intersectional disparities
3.Run Component 3
4.Determine whether disparities persist under counterfactual group reassignment

---

Model Compatibility

Supports:
-Logistic Regression
-Random Forest
-LightGBM
-Custom scikit-learn estimators