from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from .containers import FairnessResults, GroupRates
from .plots import _plot_bar, _plot_bar_series_by_group, _plot_grouped_eods_components, _plot_fairness_matrix
from .utilities import _compute_group_rates, _make_ohe, _get_model, _as_prob


def evaluate_intersectional_fairness(
    df: pd.DataFrame,
    outcome: str,
    protected_1: str,
    protected_2: str,
    features: Optional[List[str]] = None,
    model_type: str = "logreg",
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    positive_label: Any = 1,
    threshold: float = 0.5,
    make_plots: bool = True,
    *,
    return_intermediates = False,
    return_non_intersectional: bool = False,
    min_group_size: int = 0,           #drop groups with n < this
    require_class_balance: bool = False #require ≥1 pos & ≥1 neg per group
) -> Tuple[FairnessResults, Dict[str, plt.Figure]]:
    
    #Check that protected characteristics and outcome are in data
    for col in (outcome, protected_1, protected_2):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")
        

    def _compute_fairness_for_groups(group_labels, *, dropped_groups_local=None, kept_summary_local=None) -> FairnessResults:
        g_series_local = pd.Series(group_labels, index=np.arange(len(group_labels)))
        group_rates_local = _compute_group_rates(y_true=y_test, y_pred=y_hat, groups=g_series_local)
        df_groups_local = pd.DataFrame([gr.__dict__ for gr in group_rates_local])

        df_groups_filtered_local = df_groups_local.copy()
        if min_group_size > 0:
            df_groups_filtered_local = df_groups_filtered_local[df_groups_filtered_local["n"] >= min_group_size]
        if require_class_balance:
            df_groups_filtered_local = df_groups_filtered_local[
                (df_groups_filtered_local["pos_true"] >= 1) & (df_groups_filtered_local["neg_true"] >= 1)
            ]

        if df_groups_filtered_local.empty:
            df_for_metrics_local = df_groups_local.copy()
        else:
            df_for_metrics_local = df_groups_filtered_local

        if df_for_metrics_local.empty:
            privileged_group_local = None
            tpr_priv_local = fpr_priv_local = np.nan
        else:
            privileged_group_local = df_for_metrics_local.sort_values("n", ascending=False).iloc[0]["group"]
            tpr_priv_local = float(df_for_metrics_local.loc[df_for_metrics_local["group"] == privileged_group_local, "tpr"].iloc[0])
            fpr_priv_local = float(df_for_metrics_local.loc[df_for_metrics_local["group"] == privileged_group_local, "fpr"].iloc[0])

        df_disp_local = df_for_metrics_local.copy()
        df_disp_local["eo_diff"] = tpr_priv_local - df_disp_local["tpr"]
        df_disp_local["eod_tpr_diff"] = tpr_priv_local - df_disp_local["tpr"]
        df_disp_local["eod_fpr_diff"] = df_disp_local["fpr"] - fpr_priv_local
        df_disp_local["eod_max_abs"] = np.maximum(df_disp_local["eod_tpr_diff"].abs(), df_disp_local["eod_fpr_diff"].abs())
        per_group_with_diffs_local = df_disp_local.reset_index(drop=True)

        def _gap(s: pd.Series) -> float:
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return float(s.max() - s.min()) if not s.empty else float("nan")

        return FairnessResults(
            model=mt,
            groups=[GroupRates(**row) for row in df_for_metrics_local.to_dict(orient="records")],
            demographic_parity_gap=_gap(df_for_metrics_local["positive_rate"]),
            equalized_odds_gap_tpr=_gap(df_for_metrics_local["tpr"]),
            equalized_odds_gap_fpr=_gap(df_for_metrics_local["fpr"]),
            equal_opportunity_gap=_gap(df_for_metrics_local["tpr"]),
            per_group_df=per_group_with_diffs_local,
            dropped_groups=dropped_groups_local or [],
            kept_groups_summary=(kept_summary_local if kept_summary_local is not None else pd.DataFrame({"group": [], "n": []})),
        )



    #Filter out small intersectional groups pre-training
    inter_series = df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
    counts = inter_series.value_counts()
    if min_group_size > 0:
        keep_groups = counts[counts >= min_group_size].index
        dropped_groups = counts[counts < min_group_size].index.tolist()
        df = df[inter_series.isin(keep_groups)].copy()
        #refresh series/counts to reflect the filtered df
        inter_series = df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
        counts = inter_series.value_counts()
    else:
        keep_groups = counts.index
        dropped_groups = []

    #pre-training summary of group sizes
    kept_summary = counts.rename("n").reset_index().rename(columns={"index": "group"})

    #Recompute binary target and intersectional groups after filtering
    y = (df[outcome].values == positive_label).astype(int)
    inter = (df[protected_1].astype(str) + "|" + df[protected_2].astype(str)).values


    #Remove protected groups from feature set
    if features is None:
        X = df.drop(columns=[outcome, protected_1, protected_2])
        feature_cols = X.columns.tolist()
    else:
        #strip protecteds/target if a user accidentally included them
        feature_cols = [c for c in features if c not in (outcome, protected_1, protected_2)]
        X = df[feature_cols].copy()

    #Drop columns that are entirely NaN after filtering
    all_nan_cols = [c for c in feature_cols if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    if not feature_cols:
        raise ValueError("No usable feature columns remain after filtering and dropping all-NaN columns.")

    #Check feature types
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    #preprocessing: imputer + scaler for numeric, imputer + OHE for categorical
    #sparse output for tree-based models, dense for others (esp. neural nets)
    num_pipe_sparse = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    cat_pipe_sparse = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe(dense=False)),
    ])
    pre_sparse = ColumnTransformer([
        ("num", num_pipe_sparse, numeric_cols),
        ("cat", cat_pipe_sparse, categorical_cols),
    ], remainder="drop", sparse_threshold=0.3)

    num_pipe_dense = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True)),
    ])
    cat_pipe_dense = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe(dense=True)),
    ])
    pre_dense = ColumnTransformer([
        ("num", num_pipe_dense, numeric_cols),
        ("cat", cat_pipe_dense, categorical_cols),
    ], remainder="drop", sparse_threshold=1.1)

    #Get the model type and parameters
    clf = _get_model(model_type, model_params)
    mt = model_type.lower()
    if mt in ("nn", "mlp", "neural", "neural_network"):
        densify = FunctionTransformer(lambda A: A.toarray() if hasattr(A, "toarray") else A, accept_sparse=True)
        pipe = Pipeline([("prep", pre_dense), ("densify", densify), ("model", clf)])
    else:
        pipe = Pipeline([("prep", pre_sparse), ("model", clf)])

    #Split & train
    p1 = df[protected_1].astype(str).values
    p2 = df[protected_2].astype(str).values

    X_train, X_test, y_train, y_test, g_train, g_test, p1_train, p1_test, p2_train, p2_test = train_test_split(
        X, y, inter, p1, p2,
        test_size=test_size, random_state=random_state, stratify=y
    )
    #X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    #    X, y, inter, test_size=test_size, random_state=random_state, stratify=y
    #)
    pipe.fit(X_train, y_train)

    #Predict on test
    proba = _as_prob(pipe, X_test)
    y_hat = (proba >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_test, proba)
    except Exception:
        auroc = float("nan")   #if y_test has a single class

    accuracy = float(accuracy_score(y_test, y_hat))

    #(optional) quick console summary
    print(f"[Model performance] accuracy={accuracy:.3f} | AUROC={auroc:.3f}")

    #Metrics by group (on test)
    g_series = pd.Series(g_test, index=np.arange(len(g_test)))
    group_rates = _compute_group_rates(y_true=y_test, y_pred=y_hat, groups=g_series)
    df_groups = pd.DataFrame([gr.__dict__ for gr in group_rates])
    

    df_groups_filtered = df_groups.copy()

    #small-n filter on the test fold (Avoid reporting groups under a certain size)
    if min_group_size > 0:
        df_groups_filtered = df_groups_filtered[df_groups_filtered["n"] >= min_group_size]

    #optional class-balance requirement on the test fold (Avoids reporting NaN or 0 rates)
    if require_class_balance:
        df_groups_filtered = df_groups_filtered[
            (df_groups_filtered["pos_true"] >= 1) & (df_groups_filtered["neg_true"] >= 1)
        ]

    #If everything got filtered out, keep the original (but warn in metrics)
    if df_groups_filtered.empty:
        df_for_metrics = df_groups.copy()
        filtered_note = True
    else:
        df_for_metrics = df_groups_filtered
        filtered_note = False
        
    #Find the privileged group (largest n in filtered view)    
    if df_for_metrics.empty:
        privileged_group = None
        tpr_priv = fpr_priv = np.nan
    else:
        privileged_group = df_for_metrics.sort_values("n", ascending=False).iloc[0]["group"]
        tpr_priv = float(df_for_metrics.loc[df_for_metrics["group"] == privileged_group, "tpr"].iloc[0])
        fpr_priv = float(df_for_metrics.loc[df_for_metrics["group"] == privileged_group, "fpr"].iloc[0])

    #Find per-group disparities vs privileged
    df_disp = df_for_metrics.copy()
    df_disp["eo_diff"] = tpr_priv - df_disp["tpr"]                      #Equal Opportunity difference (TPR)
    df_disp["eod_tpr_diff"] = tpr_priv - df_disp["tpr"]                #EOds component on TPR
    df_disp["eod_fpr_diff"] = df_disp["fpr"] - fpr_priv                #EOds component on FPR (higher is worse)
    df_disp["eod_max_abs"] = np.maximum(df_disp["eod_tpr_diff"].abs(),
                                        df_disp["eod_fpr_diff"].abs())

    #Push these back onto results table
    per_group_with_diffs = df_disp.copy().reset_index(drop=True)

    #Fix gaps with NaN values (from single-class groups)
    def _gap(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.max() - s.min()) if not s.empty else float("nan")
    

    if filtered_note:
        print("Warning: All groups were filtered out by min_group_size or require_class_balance. "
              "Returning metrics on unfiltered groups, which may include NaN rates.")

    """
    results = FairnessResults(
        model=mt,
        groups=[GroupRates(**row) for row in df_for_metrics.to_dict(orient="records")],
        demographic_parity_gap=_gap(df_for_metrics["positive_rate"]),
        equalized_odds_gap_tpr=_gap(df_for_metrics["tpr"]),
        equalized_odds_gap_fpr=_gap(df_for_metrics["fpr"]),
        equal_opportunity_gap=_gap(df_for_metrics["tpr"]),
        per_group_df=per_group_with_diffs,
        dropped_groups=dropped_groups,
        kept_groups_summary=kept_summary.reset_index().rename(columns={"g": "group"}),
    )
    """

    results = _compute_fairness_for_groups(
        g_test,
        dropped_groups_local=dropped_groups,
        kept_summary_local=kept_summary.reset_index(drop=True),
    )

    non_intersectional = None
    if return_non_intersectional:
        p1_summary = pd.Series(p1, name="group").value_counts().rename("n").reset_index().rename(columns={"index": "group"})
        p2_summary = pd.Series(p2, name="group").value_counts().rename("n").reset_index().rename(columns={"index": "group"})

        non_intersectional = {
            protected_1: _compute_fairness_for_groups(p1_test, kept_summary_local=p1_summary),
            protected_2: _compute_fairness_for_groups(p2_test, kept_summary_local=p2_summary),
        }

    #Plots use the same filtered view
    figs: Dict[str, plt.Figure] = {}
    if make_plots:
        base = df_for_metrics.set_index("group")
        figs["demographic_parity"] = _plot_bar(
            base["positive_rate"], "Demographic Parity by Group", "P(Ŷ=1)"
        )
        #Per-group Equal Opportunity difference (TPR vs privileged)
        figs["eo_diff_by_group"] = _plot_bar_series_by_group(
            df=per_group_with_diffs,
            value_col="eo_diff",
            title=f"Equal Opportunity Difference by Group (privileged: {privileged_group})",
            ylabel="TPR_priv - TPR_group"
        )

        #Per-group Equalized Odds (single-number, max abs of TPR/FPR diffs)
        figs["eods_maxabs_by_group"] = _plot_bar_series_by_group(
            df=per_group_with_diffs,
            value_col="eod_max_abs",
            title=f"Equalized Odds (max |TPR/FPR diff|) by Group (privileged: {privileged_group})",
            ylabel="max(|ΔTPR|, |ΔFPR|)"
        )

        figs["eods_components_grouped"] = _plot_grouped_eods_components(results.per_group_df)

        figs["fairness_landscape"] = _plot_fairness_matrix(
            per_group_with_diffs,
            metric_cols=[
                "positive_rate", "tpr", "fpr",
                "eo_diff", "eod_fpr_diff", "eod_max_abs",
            ],
            title=f"Fairness Metrics Matrix (privileged: {privileged_group})",
            annotate=True,
            sort_by="eod_max_abs",
            max_groups=40,      # tune as needed; set None for all groups
            normalize="zscore", # or "none" if you prefer raw-color scaling
        )

        
    if return_intermediates:
       intermediates = {
            "y_test": y_test,
            "y_hat": y_hat,
            "groups_test": g_test,
            "proba": proba,
            "model_metrics": {         
                "accuracy": accuracy,
                "auroc": auroc,
                "threshold": float(threshold),
                "test_size": float(test_size),
                "model_type": mt,
            },
            "non_intersectional": non_intersectional,
        }
       return results, figs, intermediates

    return results, figs