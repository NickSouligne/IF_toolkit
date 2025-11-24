# Model Facade – Detailed API

This document provides full narrative documentation for the functions in `model.py`.
Use short docstrings in code; keep rationale, edge cases, and examples here.

## Table of Contents
- [ensure_probabilistic_estimator](#ensure_probabilistic_estimator)
- (add others as you go)

---

## ensure_probabilistic_estimator

**Signature**

```python
ensure_probabilistic_estimator(estimator, *, method: str = "isotonic", cv: int = 3)
```
What this does

Ensures an sklearn-compatible classifier can produce calibrated probability outputs via predict_proba(X).
If the supplied estimator already exposes predict_proba, it is returned unchanged. Otherwise, it is wrapped in CalibratedClassifierCV, which converts decision scores to calibrated probabilities.

Why it’s needed

This framework estimates counterfactual rates (cFPR/cFNR) from probability predictions μ(X). Some estimators (e.g., SVC without probability=True) don’t natively provide predict_proba. This helper makes the system model-agnostic without imposing constraints on the user’s choice of estimator.

Parameters

estimator (object): Any sklearn-like classifier with fit(X, y). If it already implements predict_proba, it is returned as-is. Otherwise it will be wrapped.

method {"isotonic", "sigmoid"} (kw-only, default "isotonic"): Calibration method.

"isotonic": non-parametric, flexible; needs more data.

"sigmoid": Platt scaling; parametric; stabler on small data.

cv (int, kw-only, default 3): Number of CV folds used internally by CalibratedClassifierCV to fit the calibration mapping.

Returns

estimator_or_wrapper (object):

The original estimator (if it already supports predict_proba), or

A CalibratedClassifierCV(base_estimator=estimator, method, cv) wrapper providing predict_proba.

Notes & Edge Cases

SVC quirk: sklearn.svm.SVC only exposes predict_proba when constructed with probability=True. You can use that, or rely on this wrapper (which calibrates the decision function).

Calibration affects the mapping from scores to probabilities, not the underlying classifier’s separating surface.

For multiclass propensity models (π_g(X)), calibration yields a valid probability distribution over classes.

Examples
```python
from sklearn.svm import SVC
base = SVC(kernel="rbf")  # no predict_proba
prob_est = ensure_probabilistic_estimator(base, method="sigmoid", cv=5)
assert hasattr(prob_est, "predict_proba")
```


# Model Facade – Detailed API

## Class `Model`

A stateful, estimator-agnostic wrapper around the fairness pipeline. It keeps your data and configuration together, exposes a clean API (`pre_process_data` → `fit_fairness` → `summarize`/`plots`), and supports both single-robust (SR/g-formula) and doubly-robust (DR/AIPW) estimation. Works with any sklearn-compatible classifier (LightGBM/XGBoost/RandomForest/SVM/etc.); non-probabilistic models are automatically calibrated to provide `predict_proba`.

---

## `__init__`

**Signature**
```python
Model(
  data: pd.DataFrame,
  model_type: str = "rf",
  *,
  outcome_estimator: Optional[object] = None,
  propensity_estimator: Optional[object] = None,
  treatment: Optional[str] = None,
  outcome: Optional[str] = None,
  covariates: Optional[List[str]] = None,
  protected_characteristics: tuple = (),
  risk_score: Optional[str] = None,
  treatment_flag: bool = True,
  group_label_map: Optional[dict] = None,
  coeff_map: Optional[dict] = None,
  random_state: int = 42,
  n_splits: int = 5,
  method: str = "sr",
  auto_compute_propensity: bool = True,
  calibration_method: str = "isotonic",
  calibration_cv: int = 3,
)
```

What this does

Stores your dataset and configuration. You may pass a custom outcome_estimator (any classifier with fit and predict_proba or calibratable), and—if using DR mode—a propensity_estimator for modeling A1A2 | X.

Why it’s needed

Centralizes configuration and makes the pipeline ergonomic to use and reuse.

Parameters (highlights)

data (DataFrame): input data with A1, A2, Y, and feature columns.

outcome (str): name of the binary outcome column (Y).

protected_characteristics (tuple[str, str]): names of the protected columns (A1, A2).

covariates (list[str] | None): feature columns. If None, inferred as numeric columns excluding {Y, A1, A2, A1A2, (D if present)} after preprocessing.

outcome_estimator (sklearn classifier | None): any model for P(Y=1 | X, A). If None, model_type factory supplies a baseline.

propensity_estimator (sklearn classifier | None): any multiclass model for P(A=g | X) in DR mode. Optional if you allow auto-compute.

method {"sr","dr"}: estimation approach.

auto_compute_propensity (bool): if True and method="dr", fits propensities if missing.

calibration_method/calibration_cv: controls calibration when wrapping non-probabilistic models.

Returns

An initialized Model instance.


pre_process_data

Signature

pre_process_data() -> None

What this does

Validates that Y, A1, and A2 exist.

Casts Y (and D if present) to categorical.

Creates the intersection label A1A2 = str(A1) + str(A2).

Infers covariates if not provided (numeric columns excluding {Y, A1, A2, A1A2, (D)}).

Why it’s needed

Downstream functions assume consistent types and the presence of A1A2.

Notes & Edge Cases

Does not impute missingness; handle NAs upstream if needed.

If covariates are not numeric, convert or one-hot encode before calling.

_ensure_dr_inputs

Signature

_ensure_dr_inputs() -> None

What this does

If method="dr", verifies that propensity columns group_<g>_prob exist for each observed A1A2 class. If missing and auto_compute_propensity=True, it calls add_group_propensities_general to fit a multiclass model and append the columns.

Why it’s needed

DR/AIPW estimators require π_g(X) to construct doubly-robust ratios.

Notes

No-op in SR mode.

Raises an error if required columns are missing and auto_compute_propensity=False.

add_group_propensities_general

Signature

@staticmethod
add_group_propensities_general(
  df: pd.DataFrame,
  covariates: List[str],
  group_col: str = "A1A2",
  estimator: Optional[object] = None,
  random_state: int = 42,
  calibration_method: str = "isotonic",
  calibration_cv: int = 3,
) -> pd.DataFrame

What this does

Fits a multiclass classifier to estimate π_g(X) = P(A=g | X) and writes one column per group: group_<g>_prob. Uses calibration if the model lacks predict_proba. Returns a new DataFrame with added columns.

Why it’s needed

Provides propensities for DR without forcing users to create them manually.

Notes

Default estimator is a reasonable RandomForest if none is supplied.

Handles any sklearn classifier; wraps with calibration to ensure probabilities.

get_model_info

Signature

get_model_info() -> Dict[str, Any]

What this does

Returns a dictionary describing the configured models, method, key column names, and basic shapes (rows, number of covariates, CV splits, random seed).

Why it’s needed

For logging, reproducibility, and quick sanity checks.

fit_fairness

Signature

fit_fairness(
  cutoff: Optional[float] = None,
  gen_null: bool = True,
  R_null: int = 200,
  bootstrap: str = "rescaled",
  B: int = 500,
  m_factor: float = 0.75
) -> Dict[str, object]

What this does

Ensures preprocessing and (for DR) propensities.

Runs the engine to cross-fit outcome models, produce counterfactual μ columns, pick a global threshold τ (Youden if cutoff=None), and compute fairness definitions (defs).

Optionally generates a permutation null (table_null) and/or rescaled bootstrap samples (boot_out).

Why it’s needed

This is the single call that executes the full fairness estimation workflow.

Returns

A results dict with (at least): defs, est_choice, tau, groups; and optionally table_null, boot_out.

Notes

bootstrap="none" skips bootstrap; set to "rescaled" to populate CIs via plots.

gen_null=True is required if you want U-values in plots.

summarize

Signature

summarize() -> pd.DataFrame

What this does

Converts results["defs"] into a tidy DataFrame with two columns: stat, value. Includes aggregate disparity stats and group-wise rates.

Why it’s needed

Easy printing, downstream reporting, and testing.

plots

Signature

plots(
  alpha: float = 0.05,
  m_factor: float = 0.75,
  delta_uval: float = 0.05,
  u_mode: str = 'two_sided',
  include_groupwise_uvals: bool = True
)

What this does

Assembles plotting tables and (optionally) renders figures. Returns:

est_summaries: point estimates (+ CIs if bootstrap provided),

table_null_delta: obs–null draws for aggregate stats,

table_uval: U-values computed from the null.

Why it’s needed

Centralizes inference visualization and the reporting artifacts you’ll send onward.

Notes

U-values require results["table_null"] (i.e., gen_null=True during fit).

CIs require results["boot_out"] (i.e., bootstrap="rescaled" during fit).