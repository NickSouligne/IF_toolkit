from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from helpers import _as_str_groups, _clip_probs, choose_threshold_youden, _add_group_dummies, ProbaEstimator, make_outcome_estimator



#-------Cross-fitting & muY outputs -----------

def build_outcome_models_and_scores(
    data: pd.DataFrame,
    group_col: str,             # e.g., 'A1A2' (string codes)
    outcome_col: str,           # e.g., 'Y' (binary 0/1)
    covariates: List[str],
    model: Optional[ProbaEstimator] = None,
    model_type: str = "rf",
    n_splits: int = 5,
    outcome_model_kwargs: Optional[dict] = None,
    random_state: int = 42,
    groups_universe: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, float, List[str]]:
    """
    Cross-fit pooled outcome model, produce muY_<g> for all groups, choose τ (cutoff threshold).

    Trains a pooled classifier on [X, one-hot(A)] using stratified K-folds,
    writes counterfactual probabilities muY_<g>(X) for each group g on OOF
    folds, and selects a global threshold τ via Youden on factual OOF preds.
    Returns (df_with_mu, tau, groups). 
    
    See docs/outcome_models.md#build_outcome_models_and_scores
    """
    df = data.copy()
    y = df[outcome_col].astype(int).to_numpy()

    #Build columns for each group
    groups = sorted(groups_universe or _as_str_groups(df[group_col]).unique().tolist())
    mu_cols = [f"muY_{g}" for g in groups]
    for c in mu_cols:
        df[c] = np.nan

    #Wrapper to create new model if none provided
    def _make():
        return model if model is not None else make_outcome_estimator(model_type, random_state=random_state, outcome_model_kwargs=outcome_model_kwargs or {})

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    factual_pred = np.zeros(len(df), dtype=float)
    colpos = {g: j for j, g in enumerate(groups)}

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df[covariates], y), start=1):
        #Train and Test splitting
        X_tr = df.iloc[train_idx][covariates]
        X_te = df.iloc[test_idx][covariates]
        A_tr = _as_str_groups(df.iloc[train_idx][group_col])
        A_te = _as_str_groups(df.iloc[test_idx][group_col])

        #Add group dummies to training data and fit
        X_tr_aug = _add_group_dummies(X_tr, A_tr, groups)
        clf = _make()
        clf.fit(X_tr_aug, y[train_idx])

        #Predict counterfactual risk for each g on test fold
        for g in groups:
            A_te_g = pd.Series(g, index=X_te.index)
            X_te_aug_g = _add_group_dummies(X_te, A_te_g, groups)
            X_te_aug_g = X_te_aug_g.reindex(columns=X_tr_aug.columns, fill_value=0)
            p = clf.predict_proba(X_te_aug_g)[:, 1]
            df.loc[X_te.index, f"muY_{g}"] = p

        #factual out of fold (OOF) probability for τ
        mu_mat = df.loc[X_te.index, [f"muY_{g}" for g in groups]].to_numpy()
        j = A_te.map(colpos).to_numpy()
        factual_pred[test_idx] = mu_mat[np.arange(mu_mat.shape[0]), j]

    tau = choose_threshold_youden(y, factual_pred)
    return df, tau, groups



@dataclass
class CfRates:
    """
    Container for groupwise rates.

    cfpr, cfnr: counterfactual FPR/FNR under do(A=g)
    fpr_obs, fnr_obs: observed rates under factual A.
    See docs/outcome_models.md#cfrates
    """
    cfpr: Dict[str, float]
    cfnr: Dict[str, float]
    fpr_obs: Dict[str, float]
    fnr_obs: Dict[str, float]


def _select_mu_fact(df: pd.DataFrame, A: pd.Series, groups: List[str], mu_prefix: str = "muY_") -> np.ndarray:
    """
    Extract factual predicted values from counterfactual mean outcome columns.

    Given a DataFrame containing counterfactual predictions (e.g., columns like 'muY_group'),
    this function selects, for each observation, the value corresponding to that individual's
    actual (factual) group membership.

    See docs/outcome_models.md#_select_mu_fact
    """
    #Build list of column names representing each groups predicted mu (outcome)
    mu_cols = [f"{mu_prefix}{g}" for g in groups]
    mu_mat = df[mu_cols].to_numpy()

    #Map group labels to column positions
    colpos = {g: j for j, g in enumerate(groups)}
    j = A.map(colpos).to_numpy()

    #Return factual prediction value
    return mu_mat[np.arange(len(df)), j]



#------- Group-wise rates (sr/DR) -----------
def compute_cf_group_rates_sr(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    tau: float,
    mu_prefix: str = "muY_",
    groups_universe: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> CfRates:
    """
    Singly Robust estimators for cFPR/cFNR plus observed FPR/FNR.

    Uses μ^g(X) and a fixed τ to compute cFPR_g and cFNR_g; also reports
    observed FPR/FNR by group under factual A. See docs/outcome_models.md#compute_cf_group_rates_sr
    """
    df = data.copy()
    A = _as_str_groups(df[group_col])
    y = df[outcome_col].astype(int).to_numpy()
    groups = sorted(groups_universe or A.unique().tolist())

    mu_fact = _select_mu_fact(df, A, groups, mu_prefix=mu_prefix)
    S_fact = (mu_fact >= tau).astype(int)

    cfpr, cfnr, fpr_obs, fnr_obs = {}, {}, {}, {}

    for g in groups:
        mu_g = df[f"{mu_prefix}{g}"].to_numpy()
        mu0_g = np.clip(1.0 - mu_g, eps, 1.0)
        mu1_g = np.clip(mu_g, eps, 1.0)
        S_g = (mu_g >= tau).astype(int)

        #counterfactual rates
        cfpr[g] = float((S_g * mu0_g).sum() / mu0_g.sum()) if np.isfinite(mu0_g.sum()) and mu0_g.sum() > 0 else np.nan
        cfnr[g] = float(((1 - S_g) * mu1_g).sum() / mu1_g.sum()) if np.isfinite(mu1_g.sum()) and mu1_g.sum() > 0 else np.nan

        #observed rates under factual group
        mask = (A == g).to_numpy()
        y0 = (y == 0)
        y1 = (y == 1)
        denom_fpr = float((y0 & mask).sum())
        denom_fnr = float((y1 & mask).sum())
        fpr_obs[g] = float(((S_fact == 1) & y0 & mask).sum()) / denom_fpr if denom_fpr > 0 else np.nan
        fnr_obs[g] = float(((S_fact == 0) & y1 & mask).sum()) / denom_fnr if denom_fnr > 0 else np.nan

    return CfRates(cfpr=cfpr, cfnr=cfnr, fpr_obs=fpr_obs, fnr_obs=fnr_obs)


def compute_cf_group_rates_dr(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    tau: float,
    mu_prefix: str = "muY_",
    pi_prefix: str = "group_",   #expects columns like 'group_<g>_prob'
    groups_universe: Optional[List[str]] = None,
) -> CfRates:
    """
    Doubly-robust (AIPW) estimators for cFPR/cFNR plus observed FPR/FNR.

    Requires propensity columns '{pi_prefix}{g}_prob'. Forms AIPW ratios for
    cFPR_g and cFNR_g, keeping τ fixed. See docs/outcome_models.md#compute_cf_group_rates_dr
    """
    df = data.copy()
    A = _as_str_groups(df[group_col])
    y = df[outcome_col].astype(int).to_numpy()
    groups = sorted(groups_universe or A.unique().tolist())

    mu_fact = _select_mu_fact(df, A, groups, mu_prefix=mu_prefix)
    S_fact = (mu_fact >= tau).astype(int)

    cfpr, cfnr, fpr_obs, fnr_obs = {}, {}, {}, {}

    for g in groups:
        mu1_g = df[f"{mu_prefix}{g}"].to_numpy()
        mu0_g = 1.0 - mu1_g
        S_g = (mu1_g >= tau).astype(int)
        pi_g = _clip_probs(df.get(f"{pi_prefix}{g}_prob", pd.Series(np.nan, index=df.index)).to_numpy())
        A_is_g = (A == g).to_numpy().astype(float)

        #cFPR
        Ytilde0 = (S_g * (1 - y)).astype(float)
        muYtilde0 = (S_g * mu0_g).astype(float)
        Z0 = (1 - y).astype(float)
        muZ0 = mu0_g.astype(float)
        w = A_is_g / pi_g
        num = np.nanmean(w * Ytilde0 - (w - 1.0) * muYtilde0)
        den = np.nanmean(w * Z0      - (w - 1.0) * muZ0)
        cfpr[g] = num / den if den > 0 else np.nan

        #cFNR
        Ytilde1 = ((1 - S_g) * y).astype(float)
        muYtilde1 = ((1 - S_g) * mu1_g).astype(float)
        Z1 = y.astype(float)
        muZ1 = mu1_g.astype(float)
        num2 = np.nanmean(w * Ytilde1 - (w - 1.0) * muYtilde1)
        den2 = np.nanmean(w * Z1      - (w - 1.0) * muZ1)
        cfnr[g] = num2 / den2 if den2 > 0 else np.nan

        #observed rates
        mask = (A == g).to_numpy()
        y0 = (y == 0)
        y1 = (y == 1)
        denom_fpr = float((y0 & mask).sum())
        denom_fnr = float((y1 & mask).sum())
        fpr_obs[g] = float(((S_fact == 1) & y0 & mask).sum()) / denom_fpr if denom_fpr > 0 else np.nan
        fnr_obs[g] = float(((S_fact == 0) & y1 & mask).sum()) / denom_fnr if denom_fnr > 0 else np.nan

    return CfRates(cfpr=cfpr, cfnr=cfnr, fpr_obs=fpr_obs, fnr_obs=fnr_obs)


#-------Pairwise summaries (defs)-----------

def _pairwise_abs_diffs(vals: List[float]) -> np.ndarray:
    """
    All pairwise absolute differences, ignoring NaNs/infs.

    Utility for disparity summaries (avg/max/var). See docs/outcome_models.md#_pairwise_abs_diffs
    """
    v = np.array(vals, dtype=float)
    n = len(v)
    if n <= 1:
        return np.array([np.nan])
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(v[i]) and np.isfinite(v[j]):
                diffs.append(abs(v[i] - v[j]))
    return np.array(diffs) if diffs else np.array([np.nan])


def get_defs_from_rates(rates: CfRates) -> Dict[str, float]:
    """
    Summaries from groupwise rates.

    Aggregates cFPR/cFNR into avg/max/var (pos/neg), and includes per-group
    cfpr_*, cfnr_*, fpr_*, fnr_*. See docs/outcome_models.md#get_defs_from_rates
    """
    groups = sorted(rates.cfpr.keys())
    cfpr_vec = [rates.cfpr[g] for g in groups]
    cfnr_vec = [rates.cfnr[g] for g in groups]

    dpos = _pairwise_abs_diffs(cfpr_vec)
    dneg = _pairwise_abs_diffs(cfnr_vec)

    defs = {
        "avg_pos": float(np.nanmean(dpos)),
        "max_pos": float(np.nanmax(dpos)),
        "var_pos": float(np.nanvar(dpos)),
        "avg_neg": float(np.nanmean(dneg)),
        "max_neg": float(np.nanmax(dneg)),
        "var_neg": float(np.nanvar(dneg)),
    }
    for g, v in rates.cfpr.items():
        defs[f"cfpr_{g}"] = float(v)
    for g, v in rates.cfnr.items():
        defs[f"cfnr_{g}"] = float(v)
    for g, v in rates.fpr_obs.items():
        defs[f"fpr_{g}"] = float(v)
    for g, v in rates.fnr_obs.items():
        defs[f"fnr_{g}"] = float(v)
    return defs
