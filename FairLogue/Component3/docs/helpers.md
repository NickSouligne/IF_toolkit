# Helpers – Detailed API

Utility functions and lightweight interfaces used throughout the fairness pipeline.
They provide stable encodings for groups, numeric safety for probabilities, a global
operating threshold, fixed-order group dummies, and a simple estimator protocol/factory.

## Table of Contents
- [_as_str_groups](#_as_str_groups)
- [_clip_probs](#_clip_probs)
- [choose_threshold_youden](#choose_threshold_youden)
- [_add_group_dummies](#_add_group_dummies)
- [ProbaEstimator](#ProbaEstimator)
- [make_outcome_estimator](#make_outcome_estimator)

---

## _as_str_groups

**Signature**
```python
_as_str_groups(s: pd.Series) -> pd.Series
```

What this does

Casts any group label series to str. This creates a stable, comparable dtype for
joins, mapping, and column selection downstream.

Why it’s needed

Different pandas dtypes (categorical, int, object) can cause subtle mismatches when
used as keys. Normalizing to str avoids those errors.

Parameters

s (pd.Series): group labels (e.g., "A1A2").

Returns

(pd.Series): string-typed labels.

_clip_probs

Signature

_clip_probs(arr: np.ndarray, lo: float = 1e-6, hi: float = 1 - 1e-6) -> np.ndarray

What this does

Clips probabilities to the open interval (lo, hi). Protects against numerical
issues like division by zero or log(0) and stabilizes AIPW weights.

Why it’s needed

Propensities and μ predictions can be extremely small or large; clipping improves
numerical stability for ratio estimators and CI routines.

Parameters

arr (np.ndarray): probability array.

lo/hi (float): lower/upper bounds (exclusive).

Returns

(np.ndarray): clipped probabilities.

Notes

The defaults 1e-6 and 1-1e-6 are conservative; you may adjust if needed.

choose_threshold_youden

Signature

choose_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float

What this does

Computes a global decision threshold τ that maximizes Youden’s J:
J = TPR − FPR evaluated over the ROC curve of pooled out-of-fold factual
predictions.

Why it’s needed

The framework compares groups at a single global operating point, avoiding
per-group thresholds that can obscure disparity.

Parameters

y_true (np.ndarray): binary ground truth (0/1).

y_prob (np.ndarray): predicted probabilities for the factual group/row.

Returns

(float): the selected threshold τ.

Notes

Ensure y_prob are out-of-fold to avoid optimistic bias in τ selection.

_add_group_dummies

Signature

_add_group_dummies(X: pd.DataFrame, group_vals: pd.Series, groups: List[str], prefix: str = "G__") -> pd.DataFrame

What this does

Creates one-hot encodings for a known set of groups and concatenates them to X.
Unlike pd.get_dummies, this preserves a fixed column order across folds, bootstraps,
and permutations by using the explicit groups list.

Why it’s needed

Stable column layouts are essential for cross-fitting and for generating muY_<g> columns
consistently across resamples.

Parameters

X (DataFrame): feature matrix.

group_vals (Series): group label per row (any dtype; casted to str internally).

groups (List[str]): the full, ordered set of group labels to one-hot encode.

prefix (str, default "G__"): column prefix for dummy variables.

Returns

(DataFrame): a new frame with X plus dummy columns in fixed order.

ProbaEstimator

Definition

class ProbaEstimator(Protocol):
    def fit(self, X, y) -> Any: ...
    def predict_proba(self, X) -> np.ndarray: ...

What this is

A typing protocol that defines the minimal interface required by the outcome model:

fit(X, y)

predict_proba(X) -> (n_samples, 2) for binary outcomes.

Why it’s needed

Enables estimator-agnostic code while keeping static type checkers helpful. Any
sklearn-compatible classifier that matches this interface will work in the pipeline.

Notes

If your model lacks predict_proba, use the facade’s calibration helper to wrap it.

make_outcome_estimator

Signature

make_outcome_estimator(kind: str, random_state: int = 42) -> ProbaEstimator

What this does

Returns a baseline outcome classifier:

"rf" → RandomForestClassifier(n_estimators=700, max_depth=3, class_weight="balanced")

"glm" → LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")

Both are sensible defaults and work well as pooled models on [X, one-hot(A)].

Why it’s needed

Provides a ready-to-go fallback when the user does not pass a custom estimator.

Parameters

kind {"rf","glm"}: which baseline to return.

random_state (int): seed for reproducibility.

Returns

(ProbaEstimator): a classifier with predict_proba.

Notes

These defaults use class_weight="balanced" to mitigate outcome imbalance.

You can always pass your own estimator (e.g., LightGBM, XGBoost, SVC) through the facade.