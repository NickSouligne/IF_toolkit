import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .functions_format import get_bs_rescaled, ci_tint, ci_trunc, ci_norm
import seaborn as sns

#----- Plotting function ----- 

def get_plots(results, sampsize, alpha, m_factor,
              plot_labels=None, plot_values=None, plot_colors=None, delta_uval=0.1):
    """
    Assemble plot tables, optional figures, and compute u-values.

    Returns:
        est_summaries: tidy table of estimates + CIs
        table_null_delta: null vs observed difference table (or None)
        table_uval: u-values table (or None)
    """

    # ---------- 0) Basic checks ----------
    if 'table_null' in results.keys() and delta_uval is None:
        raise ValueError("Must specify 'delta_uval' if null distribution results are provided.")

    est_named = results['defs']

    # ---------- 1) Rescaled bootstrap + aggregate CIs ----------
    bs_rescaled = get_bs_rescaled(results['boot_out'], est_named, sampsize, m_factor)

    # Aggregate metrics
    agg_stats = ["avg_neg", "avg_pos", "max_neg", "max_pos", "var_neg", "var_pos"]
    ci_trunc_list = []
    for stat in agg_stats:
        ci_t = ci_tint(bs_rescaled, est_named, stat, sampsize, alpha, m_factor)
        ci_trunc_list.append(ci_trunc(ci_t, "tint"))

    ci_trunc_df = pd.concat(ci_trunc_list, ignore_index=True)
    ci_trunc_df["stat"] = agg_stats
    ci_trunc_df = ci_trunc_df[["stat", "point_est", "se_est", "low_trans", "high_trans"]]

    # ---------- 2) Counterfactual error rates: group-wise cfpr/cfnr ----------
    # Filter keys containing "cfpr" / "cfnr" and remove those containing "marg"
    results_cfpr_temp = {k: v for k, v in est_named.items() if "cfpr" in k and "marg" not in k}
    results_cfnr_temp = {k: v for k, v in est_named.items() if "cfnr" in k and "marg" not in k}

    # cfpr CIs
    cfpr_ci_list = []
    for stat in results_cfpr_temp.keys():
        ci_t = ci_tint(bs_rescaled, est_named, stat, sampsize, alpha, m_factor)
        cfpr_ci_list.append(ci_trunc(ci_t, "tint"))
    ci_trunc_cfpr = pd.concat(cfpr_ci_list, ignore_index=True)
    ci_trunc_cfpr["stat"] = list(results_cfpr_temp.keys())
    ci_trunc_cfpr = ci_trunc_cfpr[["stat", "point_est", "se_est", "low_trans", "high_trans"]]

    # cfnr CIs
    cfnr_ci_list = []
    for stat in results_cfnr_temp.keys():
        ci_t = ci_tint(bs_rescaled, est_named, stat, sampsize, alpha, m_factor)
        cfnr_ci_list.append(ci_trunc(ci_t, "tint"))
    ci_trunc_cfnr = pd.concat(cfnr_ci_list, ignore_index=True)
    ci_trunc_cfnr["stat"] = list(results_cfnr_temp.keys())
    ci_trunc_cfnr = ci_trunc_cfnr[["stat", "point_est", "se_est", "low_trans", "high_trans"]]

    # ---------- 3) Null distribution / u-values (optional) ----------
    table_null_delta = None
    table_uval = None

    if "table_null" in results.keys():
        est_named_df = pd.DataFrame(list(est_named.items()), columns=["stat", "value_obs"])
        table_null = results["table_null"]

        # Select aggregate columns and melt to long
        table_null_delta = table_null[agg_stats].melt(
            var_name="stat", value_name="value_null"
        )
        table_null_delta = table_null_delta.merge(est_named_df, on="stat")
        table_null_delta["obs_minus_null"] = (
            table_null_delta["value_obs"] - table_null_delta["value_null"]
        )

        # u-values per aggregate stat
        table_uval = pd.DataFrame(
            {
                "avg_neg": np.mean((est_named["avg_neg"] - table_null["avg_neg"]) > delta_uval),
                "avg_pos": np.mean((est_named["avg_pos"] - table_null["avg_pos"]) > delta_uval),
                "max_neg": np.mean((est_named["max_neg"] - table_null["max_neg"]) > delta_uval),
                "max_pos": np.mean((est_named["max_pos"] - table_null["max_pos"]) > delta_uval),
                "var_neg": np.mean((est_named["var_neg"] - table_null["var_neg"]) > delta_uval),
                "var_pos": np.mean((est_named["var_pos"] - table_null["var_pos"]) > delta_uval),
            },
            index=[0],
        ).round(3)

    # ---------- 4) Combine CIs with point estimates ----------
    allcis_trunc = pd.concat(
        [ci_trunc_df, ci_trunc_cfpr, ci_trunc_cfnr], ignore_index=True
    )

    est_named_df = pd.DataFrame(list(est_named.items()), columns=["stat", "value"])
    # Use helper to categorize stats
    est_named_df["sign"] = est_named_df["stat"].apply(categorize_stat)

    est_summaries = est_named_df.merge(allcis_trunc, on="stat", how="left")

    # Override sign for aggregates explicitly
    est_summaries.loc[est_summaries["stat"].isin(["avg_neg", "max_neg", "var_neg"]), "sign"] = "aggregate_neg"
    est_summaries.loc[est_summaries["stat"].isin(["avg_pos", "max_pos", "var_pos"]), "sign"] = "aggregate_pos"

    # ---------- 5) cFPR plot ----------
    est_summaries_cfpr = est_summaries[est_summaries["sign"] == "cfpr"]

    if not est_summaries_cfpr.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="stat",
            y="value",
            hue="stat",
            data=est_summaries_cfpr,
            s=100,
            palette="Set1",
        )

        y_center = est_summaries_cfpr["point_est"]
        yerr_lower = y_center - est_summaries_cfpr["low_trans"]
        yerr_upper = est_summaries_cfpr["high_trans"] - y_center

        # ensure non-negative (numerical safety)
        yerr_lower = yerr_lower.clip(lower=0)
        yerr_upper = yerr_upper.clip(lower=0)
        yerr = np.vstack([yerr_lower, yerr_upper])

        plt.errorbar(
            x=np.arange(len(est_summaries_cfpr)),
            y=y_center,
            yerr=yerr,
            fmt="none",
            capsize=5,
            elinewidth=2,
            alpha=0.7,
            color="gray",
        )

        plt.xticks(
            ticks=np.arange(len(est_summaries_cfpr)),
            labels=est_summaries_cfpr["stat"],
            rotation=45,
            ha="right",
        )
        plt.xlabel(None)
        plt.ylabel("Group cFPR estimate")
        plt.title(None)
        plt.legend(title="Group", fontsize="large", title_fontsize="large")
        plt.tight_layout()

    # ---------- 6) cFNR plot ----------
    est_summaries_cfnr = est_summaries[est_summaries["sign"] == "cfnr"]

    if not est_summaries_cfnr.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="stat",
            y="value",
            hue="stat",
            data=est_summaries_cfnr,
            s=100,
            palette="Set1",
        )

        y_center = est_summaries_cfnr["point_est"]
        yerr_lower = y_center - est_summaries_cfnr["low_trans"]
        yerr_upper = est_summaries_cfnr["high_trans"] - y_center
        yerr_lower = yerr_lower.clip(lower=0)
        yerr_upper = yerr_upper.clip(lower=0)
        yerr = np.vstack([yerr_lower, yerr_upper])

        plt.errorbar(
            x=np.arange(len(est_summaries_cfnr)),
            y=est_summaries_cfnr["value"],
            yerr=yerr,
            fmt="none",
            capsize=5,
            elinewidth=2,
            alpha=0.7,
            color="gray",
        )

        plt.xticks(
            ticks=np.arange(len(est_summaries_cfnr)),
            labels=est_summaries_cfnr["stat"],
            rotation=45,
            ha="right",
        )
        plt.xlabel(None)
        plt.ylabel("Group cFNR estimate")
        plt.title(None)
        plt.legend(title="Group", fontsize="large", title_fontsize="large")
        plt.tight_layout()

    # ---------- 7) Aggregate negative metrics ----------
    est_summaries_neg = est_summaries[est_summaries["sign"] == "aggregate_neg"]

    if not est_summaries_neg.empty:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x="stat",
            y="value",
            hue="stat",
            data=est_summaries_neg,
            s=100,
            palette="Set1",
        )

        y_center = est_summaries_neg["point_est"]
        yerr_lower = y_center - est_summaries_neg["low_trans"]
        yerr_upper = est_summaries_neg["high_trans"] - y_center
        yerr_lower = yerr_lower.clip(lower=0)
        yerr_upper = yerr_upper.clip(lower=0)
        yerr = np.vstack([yerr_lower, yerr_upper])

        plt.errorbar(
            x=np.arange(len(est_summaries_neg)),
            y=est_summaries_neg["value"],
            yerr=yerr,
            fmt="none",
            capsize=5,
            elinewidth=2,
            alpha=0.7,
            color="gray",
        )

        plt.xticks(
            ticks=np.arange(len(est_summaries_neg)),
            labels=est_summaries_neg["stat"],
            rotation=0,
        )
        plt.xlabel(None)
        plt.ylabel("Negative metrics (estimate, 95% CI)")
        plt.title(None)
        plt.legend(title="Metric", fontsize="large", title_fontsize="large")
        plt.tight_layout()

    # ---------- 8) Aggregate positive metrics ----------
    est_summaries_pos = est_summaries[est_summaries["sign"] == "aggregate_pos"]

    if not est_summaries_pos.empty:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x="stat",
            y="value",
            hue="stat",
            data=est_summaries_pos,
            s=100,
            palette="Set1",
        )

        y_center = est_summaries_pos["point_est"]
        yerr_lower = y_center - est_summaries_pos["low_trans"]
        yerr_upper = est_summaries_pos["high_trans"] - y_center
        yerr_lower = yerr_lower.clip(lower=0)
        yerr_upper = yerr_upper.clip(lower=0)
        yerr = np.vstack([yerr_lower, yerr_upper])

        plt.errorbar(
            x=np.arange(len(est_summaries_pos)),
            y=est_summaries_pos["value"],
            yerr=yerr,
            fmt="none",
            capsize=5,
            elinewidth=2,
            alpha=0.7,
            color="gray",
        )

        plt.xticks(
            ticks=np.arange(len(est_summaries_pos)),
            labels=est_summaries_pos["stat"],
            rotation=0,
        )
        plt.xlabel(None)
        plt.ylabel("Positive metrics (estimate, 95% CI)")
        plt.title(None)
        plt.legend(title="Metric", fontsize="large", title_fontsize="large")
        plt.tight_layout()

    # ---------- 9) Null distribution KDE plots (if available) ----------
    if table_null_delta is not None and table_uval is not None:
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))

        # Make sure table_uval is 1-row DataFrame
        u = table_uval.iloc[0]

        # Average (negative)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "avg_neg"],
            x="obs_minus_null",
            ax=axes[0, 0],
        )
        annotate_plot(axes[0, 0], "avg_neg", u["avg_neg"])
        axes[0, 0].set_title("Average (negative)")
        axes[0, 0].set_xlabel("Obs. - Null")
        axes[0, 0].set_ylabel("Density")

        # Maximum (negative)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "max_neg"],
            x="obs_minus_null",
            ax=axes[0, 1],
        )
        annotate_plot(axes[0, 1], "max_neg", u["max_neg"])
        axes[0, 1].set_title("Maximum (negative)")
        axes[0, 1].set_xlabel("Obs. - Null")
        axes[0, 1].set_ylabel("Density")

        # Variational (negative)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "var_neg"],
            x="obs_minus_null",
            ax=axes[1, 0],
        )
        annotate_plot(axes[1, 0], "var_neg", u["var_neg"])
        axes[1, 0].set_title("Variational (negative)")
        axes[1, 0].set_xlabel("Obs. - Null")
        axes[1, 0].set_ylabel("Density")

        # Average (positive)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "avg_pos"],
            x="obs_minus_null",
            ax=axes[1, 1],
        )
        annotate_plot(axes[1, 1], "avg_pos", u["avg_pos"])
        axes[1, 1].set_title("Average (positive)")
        axes[1, 1].set_xlabel("Obs. - Null")
        axes[1, 1].set_ylabel("Density")

        # Maximum (positive)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "max_pos"],
            x="obs_minus_null",
            ax=axes[2, 0],
        )
        annotate_plot(axes[2, 0], "max_pos", u["max_pos"])
        axes[2, 0].set_title("Maximum (positive)")
        axes[2, 0].set_xlabel("Obs. - Null")
        axes[2, 0].set_ylabel("Density")

        # Variational (positive)
        sns.kdeplot(
            data=table_null_delta[table_null_delta["stat"] == "var_pos"],
            x="obs_minus_null",
            ax=axes[2, 1],
        )
        annotate_plot(axes[2, 1], "var_pos", u["var_pos"])
        axes[2, 1].set_title("Variational (positive)")
        axes[2, 1].set_xlabel("Obs. - Null")
        axes[2, 1].set_ylabel("Density")

        plt.tight_layout()

    plt.show()

    return est_summaries, table_null_delta, table_uval





#Helper function to annotate plots
def annotate_plot(ax, stat, value):
    ax.axvline(value, linestyle='--', color='r')
    max_y = ax.get_ylim()[1]
    ax.annotate(f'u-value = {value}', xy=(value, max_y * 0.8), xytext=(value, max_y * 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right')


#Define a function to categorize 'stat' into different groups
def categorize_stat(stat):
    if 'cfpr_marg' in stat:
        return 'cfpr_marg'
    elif 'cfnr_marg' in stat:
        return 'cfnr_marg'
    elif 'cfpr' in stat:
        return 'cfpr'
    elif 'cfnr' in stat:
        return 'cfnr'
    elif 'fpr' in stat:
        return 'fpr'
    elif 'fnr' in stat:
        return 'fnr'
    elif stat in ['avg_neg', 'max_neg', 'var_neg']:
        return 'aggregate_neg'
    elif stat in ['avg_pos', 'max_pos', 'var_pos']:
        return 'aggregate_pos'
    else:
        return 'other'