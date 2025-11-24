# Outcome Models – Detailed API

This module produces **counterfactual risk columns** `muY_<g>` via cross-fitting a pooled outcome model with group dummies, and then converts those into **group-wise rates** (cFPR/cFNR) and aggregate disparity summaries.

## Table of Contents
- [build_outcome_models_and_scores](#build_outcome_models_and_scores)
- [CfRates](#cfrates)
- [_select_mu_fact](#_select_mu_fact)
- [compute_cf_group_rates_sr](#compute_cf_group_rates_sr)
- [compute_cf_group_rates_dr](#compute_cf_group_rates_dr)
- [_pairwise_abs_diffs](#_pairwise_abs_diffs)
- [get_defs_from_rates](#get_defs_from_rates)

---

## build_outcome_models_and_scores

**Signature**
```python
build_outcome_models_and_scores(
  data, group_col, outcome_col, covariates,
  model=None, model_type="rf", n_splits=5, random_state=42,
  groups_universe=None
) -> (pd.DataFrame, float, List[str])
```
What this does

Cross-fits a pooled outcome model of Y | X, 1{A=g} using n_splits stratified folds:

For each fold, train the model on training rows with features [X, one-hot(A)].

On the test fold, predict counterfactual probabilities for every group g:

Replace the group with each g in turn,

Compute muY_<g>(X_test) = P(Y=1 | X_test, A:=g).

Collect factual out-of-fold probabilities by selecting muY_<A_i>(X_i) for each test row.

Choose a single global threshold τ using Youden’s J on the factual OOF probabilities.

Return the input data augmented with all muY_<g> columns, the selected τ, and the fixed group set used.

Why it’s needed

Produces counterfactual risks consistent across folds and groups.

The single τ ensures a global operating point shared across groups for fair comparisons.

Cross-fitting reduces overfitting bias in threshold selection.

Parameters

data (DataFrame): original dataset (not modified in place).

group_col (str): the intersection label (e.g., "A1A2").

outcome_col (str): binary outcome (0/1).

covariates (list[str]): feature columns for the model.

model (ProbaEstimator | None): custom classifier with fit/predict_proba. If None, uses make_outcome_estimator(model_type, random_state).

model_type {"rf","glm"}: factory shortcut for common baselines.

n_splits (int): stratified K folds for cross-fitting.

random_state (int): seed for fold construction and factory models.

groups_universe (list[str] | None): optional fixed set to stabilize columns; inferred from data if None.

Returns

df_with_mu (DataFrame): copy of data with muY_<g> columns filled OOF.

tau (float): global decision threshold (Youden).

groups (list[str]): sorted group labels used to create columns.

Notes & Edge Cases

The model must provide calibrated probabilities; the facade ensures this via calibration when needed.

Ensures consistent column order via an explicit groups list.

CfRates

A dataclass container:

cfpr (dict[str,float]): {g: cFPR_g}

cfnr (dict[str,float]): {g: cFNR_g}

fpr_obs (dict[str,float]): observed FPR by factual group

fnr_obs (dict[str,float]): observed FNR by factual group

Used as the transport object between rate estimation and summary aggregation.

_select_mu_fact

Signature

_select_mu_fact(df, A, groups, mu_prefix="muY_") -> np.ndarray

What this does

From the wide muY_<g> matrix, return the factual μ for each row by indexing the column that corresponds to the row’s actual group label A.

Why it’s needed

To compute observed rates (FPR/FNR) under factual group assignment and to derive factual decisions at the global threshold τ.

compute_cf_group_rates_sr

Signature

compute_cf_group_rates_sr(
  data, group_col, outcome_col, tau,
  mu_prefix="muY_", groups_universe=None, eps=1e-8
) -> CfRates

What this does

Implements plug-in (g-formula) estimators under do(A=g):

With μ^g = muY_<g>(X) and decision S^g = 1[μ^g ≥ τ],

cFPR_g = Σ S^g (1 − μ^g) / Σ (1 − μ^g)

cFNR_g = Σ (1 − S^g) μ^g / Σ μ^g

Also computes observed FPR/FNR by factual A using the factual decision S_fact = 1[μ^{A} ≥ τ].

Why it’s needed

Provides counterfactual error rates for each group without propensities; simple and stable if the outcome model is well-specified.

Parameters

eps prevents divide-by-zero when μ is 0 or 1 for all rows.

Returns

A CfRates with cfpr, cfnr, fpr_obs, fnr_obs.

compute_cf_group_rates_dr

Signature

compute_cf_group_rates_dr(
  data, group_col, outcome_col, tau,
  mu_prefix="muY_", pi_prefix="group_", groups_universe=None
) -> CfRates

What this does

Implements doubly-robust (AIPW) estimators for ratio targets under do(A=g). Requires propensity columns π_g(X) = P(A=g|X) named {pi_prefix}{g}_prob.

For each group g:

Let μ1_g = muY_<g>, μ0_g = 1 − μ1_g, S_g = 1[μ1_g ≥ τ], and A_is_g = 1[A=g].

With weights w = A_is_g / π_g(X), form AIPW numerators/denominators:

cFPR:

Ỹ0 = S_g * (1 − y), μỸ0 = S_g * μ0_g, Z0 = (1 − y), μZ0 = μ0_g

num = E[w Ỹ0 − (w − 1) μỸ0], den = E[w Z0 − (w − 1) μZ0]

cFPR_g = num / den

cFNR:

Ỹ1 = (1 − S_g) * y, μỸ1 = (1 − S_g) * μ1_g, Z1 = y, μZ1 = μ1_g

cFNR_g = (E[w Ỹ1 − (w − 1) μỸ1]) / (E[w Z1 − (w − 1) μZ1])

Also returns observed FPR/FNR computed as in SR.

Why it’s needed

Remains consistent if either the outcome model or the propensity model is correctly specified (double robustness).

Notes

Propensities must be clipped away from {0,1} (handled upstream via _clip_probs).

If any group has zero support, the corresponding estimates become NaN.

_pairwise_abs_diffs

Signature

_pairwise_abs_diffs(vals: List[float]) -> np.ndarray

What this does

Computes all pairwise absolute differences among a set of group rates, ignoring NaNs/infinities. Used to summarize disparities via average, maximum, and variance.

get_defs_from_rates

Signature

get_defs_from_rates(rates: CfRates) -> Dict[str, float]

What this does

Aggregates the CfRates object into:

Aggregate disparity summaries

avg_pos, max_pos, var_pos over pairwise diffs of cFPRs,

avg_neg, max_neg, var_neg over pairwise diffs of cFNRs.

Per-group statistics

cfpr_<g>, cfnr_<g>, fpr_<g>, fnr_<g>.

Why it’s needed

Transforms detailed groupwise rates into a compact set of KPIs suitable for comparisons, testing, and plotting.

Implementation Notes

Fixed group set: A consistent groups_universe stabilizes column order and rate computation, especially across folds, null permutations, and bootstrap resamples.

Global τ: Chosen via Youden on factual OOF probabilities to avoid per-group operating points (which would confound disparity comparisons).

Numerical safety: Slight clipping and NaN-aware operations guard against degenerate splits.