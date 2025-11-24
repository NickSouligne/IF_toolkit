# Plot Utilities – Detailed API

The plotting helpers build publication-ready summary tables and (optionally) render
figures that visualize null contrasts, confidence intervals, and groupwise estimates.
They operate on the `results` dict returned by `analysis_estimation` / `Model.fit_fairness`.

## Table of Contents
- [annotate_plot](#annotate_plot)
- [get_plots](#get_plots)

---

## annotate_plot

**Signature**
```python
annotate_plot(ax: plt.Axes, stat: str, value: float) -> None
```
What this does

Adds a vertical dashed line at value with a text label “u-value = {value}” inside
the given axis. Used to visually mark the u-value location on histograms of
obs_minus_null.

Why it’s needed

When comparing the observed disparity against its permutation-null distribution,
placing a clear reference line helps communicate whether the observed statistic is
“unusually large” given the null.

Parameters

ax (matplotlib.axes.Axes): axis to annotate.

stat (str): label of the statistic being plotted (used in the title upstream).

value (float): u-value to display.

Returns

None (draws directly on the axis).

get_plots

Signature

get_plots(
  results: Dict[str, object],
  sampsize: Optional[int] = None,
  alpha: float = 0.05,
  m_factor: float = 0.75,
  delta_uval: float = 0.10,
  u_mode: str = 'one_sided',
  include_groupwise_uvals: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]

What this does

Builds three reporting artifacts from the results dictionary:

est_summaries (pd.DataFrame) – A tidy table of point estimates (stat, value),
optionally augmented with CI columns when bootstrap results are available:

For aggregate stats (avg_neg, max_neg, var_neg, avg_pos, max_pos, var_pos),
CIs are computed via the transformed interval method (ci_tint) if
results["boot_out"] is present.

For groupwise stats (cfpr_*, cfnr_*), CIs are likewise included when available.

table_null_delta (pd.DataFrame | None) – If results["table_null"] exists
(i.e., you fit with gen_null=True), this is a long table of obs − null
differences for the aggregate stats. Each row is one null draw and one statistic.

table_uval (pd.DataFrame | None) – A 1×k table of u-values for the
aggregate stats, using the rule:

one_sided: u = mean(obs − null > δ)

two_sided: u = mean(|obs| − |null| > δ) ← your implementation
(note: this is different from |obs − null|; it thresholds the magnitude
difference rather than absolute deviation)

Additionally, get_plots renders figures (if a matplotlib backend is active):

A 3×2 panel of histograms for obs_minus_null on the six aggregate stats,
with a dashed vertical line at the u-value for each pane.

Optional scatter plots of groupwise cfpr_* and cfnr_* (with error bars
if CIs are available from bootstrap).

Why it’s needed

Centralizes the post-estimation visualization and reporting logic:

presents point estimates alongside interval estimates,

visualizes how far the observed stats sit relative to their permutation null,

provides compact u-values for quick hypothesis-style summaries.

Parameters

results (dict): output of analysis_estimation / Model.fit_fairness.
Must contain defs; may contain boot_out and table_null.

sampsize (int | None): base sample size for SE scaling. Defaults to the
number of rows in results["est_choice"] if available.

alpha (float, default 0.05): two-sided CI level (used in ci_tint).

m_factor (float, default 0.75): bootstrap rescale factor (passed through
to the CI computation).

delta_uval (float, default 0.10): tolerance for u-values. Only
differences larger than δ are counted as evidence against the null.

u_mode {"one_sided", "two_sided"}: u-value definition (see above).

include_groupwise_uvals (bool, default True): reserved for future use to
compute per-group u-values; currently affects only figure rendering choices.

Returns

A tuple (est_summaries, table_null_delta, table_uval):

est_summaries: always present.

table_null_delta: None unless table_null exists in results.

table_uval: None unless table_null exists in results.

Notes & Edge Cases

Bootstrap not present: CI columns are omitted; est_summaries falls back to
point estimates only.

Null not present: no histograms or u-values; table_null_delta and table_uval
are None.

Two-sided definition: Your current two-sided u-value uses |obs| − |null| > δ.
If you prefer the classical |obs − null| > δ, change the _uval function accordingly.

Matplotlib backends: In headless environments (e.g., scripts on servers),
figures are created but may not display unless you call plt.savefig(...)
or configure a non-interactive backend.

Error handling: plot rendering is wrapped in try/except to avoid
interfering with numeric outputs.

Example
# After running Model.fit_fairness(...):
est_summaries, table_null_delta, table_uval = get_plots(
    results=results,
    sampsize=len(results.get('est_choice', [])),
    alpha=0.05,
    m_factor=0.75,
    delta_uval=0.05,
    u_mode='two_sided'
)

print(est_summaries.head())
if table_uval is not None:
    print("U-values:", table_uval.to_dict(orient="records")[0])
