from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


sns.set_theme(style="whitegrid")

def _plot_bar(series: pd.Series, title: str, ylabel: str):
    """Sleek seaborn barplot for a 1D Series."""
    fig, ax = plt.subplots()

    data = series.reset_index()
    data.columns = ["category", "value"]

    sns.barplot(
        data=data,
        x="category",
        y="value",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Light horizontal grid only
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()
    return fig

def _plot_bar_series_by_group(df: pd.DataFrame, value_col: str, title: str, ylabel: str):
    """
    Sleek seaborn barplot: one bar per group (y = value_col).
    Expects a 'group' column in df.
    """
    fig, ax = plt.subplots()

    sns.barplot(
        data=df,
        x="group",
        y=value_col,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Zero line (important for signed fairness diffs)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()
    return fig

def _plot_grouped_eods_components(df: pd.DataFrame):
    """
    Two bars per group using seaborn:
    TPR diff and FPR diff in a grouped barplot.
    Expects columns: 'group', 'eod_tpr_diff', 'eod_fpr_diff'.
    """
    fig, ax = plt.subplots()

    df_long = df.melt(
        id_vars="group",
        value_vars=["eod_tpr_diff", "eod_fpr_diff"],
        var_name="component",
        value_name="difference",
    )

    # Friendlier labels
    component_map = {
        "eod_tpr_diff": "TPR diff",
        "eod_fpr_diff": "FPR diff",
    }
    df_long["component"] = df_long["component"].map(component_map)

    sns.barplot(
        data=df_long,
        x="group",
        y="difference",
        hue="component",
        ax=ax,
    )

    ax.set_title("Equalized Odds Components by Group (vs privileged)")
    ax.set_ylabel("Difference")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    sns.despine(ax=ax, left=False, bottom=False)

    ax.legend(title="")
    fig.tight_layout()
    return fig

