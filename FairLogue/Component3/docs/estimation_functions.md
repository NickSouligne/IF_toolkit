# Estimation Functions – Detailed API

This module orchestrates estimation steps around the cross-fitted outcome models:
- Converts muY_* and τ into fairness definitions (SR/DR).
- Builds permutation null distributions for u-values.
- Provides a rescaled bootstrap and CI utilities.

## Table of Contents
- [get_defs_analysis](#get_defs_analysis)
- [analysis_nulldist](#analysis_nulldist)
- [bs_rescaled_analysis](#bs_rescaled_analysis)
- [analysis_estimation](#analysis_estimation)
- [get_bs_rescaled](#get_bs_rescaled)
- [ci_norm](#ci_norm)
- [ci_tint](#ci_tint)
- [ci_trunc](#ci_trunc)

---

## get_defs_analysis

**Signature**
```python
get_defs_analysis(data_with_mu, group_col, outcome_col, tau, method="sr", groups_universe=None) -> Dict[str, float]
```

What this does

Given a dataset augmented with counterfactual risk columns muY_<g> and a global decision threshold τ, compute group-wise counterfactual rates (cFPR, cFNR) using either:

SR (plug-in / g-formula), or

DR (AIPW, requires group_<g>_prob columns),

and aggregate them into disparity definitions: avg_pos, max_pos, var_pos, avg_neg, max_neg, var_neg, plus per-group keys (cfpr_*, cfnr_*, fpr_*, fnr_*).

Why it’s needed

Separates the rate estimation stage from the modeling stage; lets you switch SR/DR without changing modeling code.

Parameters

data_with_mu (DataFrame): output of build_outcome_models_and_scores; must include muY_<g> columns.

group_col (str): intersection label (e.g., "A1A2").

outcome_col (str): binary outcome.

tau (float): global decision threshold.

method {"sr","dr"}: plug-in vs AIPW.

groups_universe (list[str] | None): optional fixed group set (for stability / reproducibility).

Returns

defs (dict): aggregate and per-group stats.

analysis_nulldist

Signature

analysis_nulldist(data, group_col, outcome_col, covariates, tau_fixed, R=200,
                  model=None, model_type="rf", n_splits=5, random_state=42,
                  method="sr", groups_universe=None) -> Dict[str, List[float]]

What this does

Constructs a permutation null for the disparity statistics by shuffling group_col (A1A2), refitting the outcome model each time, recomputing muY_*, and then computing defs with τ held fixed to tau_fixed.

Why it’s needed

Enables u-values: empirical evidence that the observed disparity exceeds what would occur by chance under random group assignment.

Parameters (highlights)

tau_fixed: keep the same operating point across permutations; isolates the effect of group assignment.

R: number of permutations.

model / model_type: pass a custom estimator or use the factory kind; cross-fitted each replicate.

Returns

A dict of arrays where each key is a stat (e.g., avg_pos) and values are length-R draws.

Notes

Uses the same groups_universe across permutations for column stability.

If the model is stochastic, random_state + r is used per replicate.

bs_rescaled_analysis

Signature

bs_rescaled_analysis(data, group_col, outcome_col, covariates, B=500, m_factor=0.75,
                     model=None, model_type="rf", n_splits=5, random_state=42,
                     method="sr", groups_universe=None) -> List[Dict[str, float]]

What this does

Runs a rescaled bootstrap:

Resample m = floor(n**m_factor) rows with replacement.

Refit the outcome model, recompute muY_*, pick τ on the bootstrap sample.

Compute defs on that sample.

Why it’s needed

Provides sampling variability for CIs without refitting on the full sample each time. The rescaling step (see get_bs_rescaled) aligns bootstrap draws for CI construction.

Parameters

B (int): number of bootstrap draws.

m_factor (float): controls bootstrap sample size.

Returns

List of defs-shaped dicts, length B.

analysis_estimation

Signature

analysis_estimation(data, group_col, outcome_col, covariates, cutoff=None,
                    model=None, model_type="rf", n_splits=5, random_state=42,
                    gen_null=True, R_null=200, bootstrap="none", B=500, m_factor=0.75,
                    method="sr") -> Dict[str, object]

What this does

A one-stop routine that:

Cross-fits the outcome model and computes muY_* columns.

Selects the global threshold τ (Youden if cutoff=None).

Computes defs via SR/DR.

Optionally builds the permutation null and/or rescaled bootstrap.

Why it’s needed

This is the engine entry point called by the facade; it returns everything downstream utilities expect.

Returns

A dict with:

defs (dict), est_choice (DataFrame with muY_)*, tau (float), groups (list[str]),

optionally table_null and boot_out.

Notes

groups_universe is inferred from data[group_col] for stability.

If you need a fixed τ (policy threshold), pass cutoff=....

get_bs_rescaled

Signature

get_bs_rescaled(bs_table, est_vals, sampsize, m_factor) -> pd.DataFrame

What this does

Converts a list of bootstrap defs dicts into a rescaled matrix aligned to the observed estimate keys:

Rows = bootstrap draws

Columns = keys of est_vals

Entries = sqrt(m) * (boot - est) (where m is number of bootstrap draws)

Why it’s needed

Standardizes bootstrap output for CI construction via ci_norm or ci_tint.

Parameters

bs_table (list[dict]): from bs_rescaled_analysis.

est_vals (dict): observed defs.

sampsize (int): original sample size (used later for SE scaling).

m_factor (float): stored for completeness (not directly used here).

Returns

DataFrame: rescaled bootstrap table with columns matching est_vals.

Notes

Missing keys in individual bootstrap dicts are filled with NaN.

ci_norm

Signature

ci_norm(bs_rescaled, est_named, parameter, sampsize, alpha) -> pd.DataFrame

What this does

Builds a normal-approximation CI for a single parameter using the rescaled bootstrap variance:

var_est = sample variance of rescaled draws

se_est = sqrt(var_est / n)

CI = point_est ± z_{1-α/2} * se_est

Why it’s needed

Fast, interpretable CIs when normality is reasonable.

Returns

A one-row DataFrame: var_est, point_est, se_est, ci_low, ci_high.

ci_tint

Signature

ci_tint(bs_rescaled, est_named, parameter, sampsize, alpha, m_factor) -> pd.DataFrame

What this does

Constructs a transformed interval (t-int) CI:

Forms bootstrap t-statistics using the rescaled draws and an estimated SE.

Uses empirical quantiles (percentile-t style) to back-transform bounds.

Why it’s needed

More robust than plain normal CIs for skewed/heteroskedastic bootstrap draws.

Returns

A one-row DataFrame: point_est, se_est, low_trans, high_trans.

ci_trunc

Signature

ci_trunc(ci_result, type2) -> pd.DataFrame

What this does

Clips CI bounds to [0, 1] for probability-like parameters.

Why it’s needed

Ensures interpretable bounds for metrics that must lie in [0, 1].

Parameters

type2 {"norm","tint"}: which columns to clip (ci_low/ci_high vs low_trans/high_trans).

Returns

A DataFrame with clipped bounds.

Suggested usage order

analysis_estimation(...) to get results.

If you bootstrapped, pass results["boot_out"] and results["defs"] to get_bs_rescaled.

Use ci_norm or ci_tint on selected parameters; clip with ci_trunc.

For u-values, rely on analysis_nulldist(...) and downstream plotting helpers.