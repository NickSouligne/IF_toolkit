from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _plot_bar(series: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

def _plot_bar_series_by_group(df: pd.DataFrame, value_col: str, title: str, ylabel: str):
    fig, ax = plt.subplots()
    s = df.set_index("group")[value_col]
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

def _plot_grouped_eods_components(df: pd.DataFrame):
    #two bars per group: eod_tpr_diff and eod_fpr_diff
    fig, ax = plt.subplots()
    idx = np.arange(len(df))
    width = 0.4
    ax.bar(idx - width/2, df["eod_tpr_diff"].values, width, label="TPR diff")
    ax.bar(idx + width/2, df["eod_fpr_diff"].values, width, label="FPR diff")
    ax.set_title("Equalized Odds Components by Group (vs privileged)")
    ax.set_ylabel("Difference")
    ax.set_xticks(idx)
    ax.set_xticklabels(df["group"].tolist(), rotation=45, ha="right")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
