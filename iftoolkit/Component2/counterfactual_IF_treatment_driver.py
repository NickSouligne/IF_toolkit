import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.api as sm
import statsmodels.formula.api as smf
from functions_fairness import analysis_estimation
from functions_plots import  get_plots
from functions_smallsubgroups import get_pa_ext_small


# Otherwise, make sure ex_data_estimation, ex_data_small, ex_data_external
# are pandas DataFrames with the same column names as in the R examples.
# Load generated datasets
ex_data_estimation = pd.read_csv("ex_data_estimation.csv")
ex_data_small = pd.read_csv("ex_data_small.csv")
ex_data_external = pd.read_csv("ex_data_external.csv")

ex_data_estimation = ex_data_estimation.rename(columns=lambda c: c.replace(".", "_"))
ex_data_small = ex_data_small.rename(columns=lambda c: c.replace(".", "_"))
ex_data_external = ex_data_external.rename(columns=lambda c: c.replace(".", "_"))


# -------------------------------
# 1) Propensity score model
# -------------------------------
# NOTE: In patsy formulas, variable names like `X.1` usually work,
# but if you run into issues, rename them to `X_1`, etc., and
# adjust the formula and pi_xvars lists accordingly.

pi_model_ex = smf.glm(
    formula="D ~ A1*A2 + A1 + A2 + S_prob + X_1 + X_2 + X_3 + X_4",
    data=ex_data_estimation,
    family=sm.families.Binomial()
).fit()

# -------------------------------
# 2) Main analysis (standard estimator)
# -------------------------------
results = analysis_estimation(
    data=ex_data_estimation,
    cutoff=0.5,
    estimator_type="standard",
    pi_model=pi_model_ex,
    pi_model_type="glm",
    pi_xvars=["X_1", "X_2", "X_3", "X_4"],
    gen_null=True,
    bootstrap="rescaled",
)



# -------------------------------
# 3) Small subgroup option, no borrowing
# -------------------------------
pi_model_ex_small = smf.glm(
    formula="D ~ A1*A2 + A1 + A2 + S_prob + X_1 + X_2 + X_3 + X_4",
    data=ex_data_small,
    family=sm.families.Binomial()
).fit()

results_smallsub = analysis_estimation(
    data=ex_data_small,
    cutoff=0.5,
    estimator_type="small_internal",
    gen_null=True,
    R_null=100,
    bootstrap="rescaled",
    B=100,
    pi_model=pi_model_ex_small,
    pi_model_type="glm",
    pi_xvars=["X_1", "X_2", "X_3", "X_4"],
    outcome_model_type="glm",
    outcome_xvars=[
        "X_outcome_1", "X_outcome_2", "X_outcome_3", "X_outcome_4",
        "X_outcome_5", "X_outcome_6", "X_outcome_7", "X_outcome_8",
    ],
    fit_method_int="multinomial",
    nfolds=5,
    pa_xvars_int=[
        "X_pa_1", "X_pa_2", "X_pa_3", "X_pa_4",
        "X_pa_5", "X_pa_6", "X_pa_7",
    ],
)


# -------------------------------
# 4) Small subgroup option, with borrowing
# -------------------------------
# Optional external PA model fit
pa_model_ext_ex = get_pa_ext_small(
    data_external=ex_data_external,
    fit_method_ext="multinomial",
    pa_xvars_ext=["X_pa_1", "X_pa_2", "X_pa_3", "X_pa_4", "X_pa_5"],
)

results_borrow = analysis_estimation(
    data=ex_data_small,
    cutoff=0.5,
    estimator_type="small_borrow",
    gen_null=False,
    R_null=100,
    bootstrap="none",
    B=100,
    pi_model=pi_model_ex_small,
    pi_model_type="glm",
    pi_xvars=["X_1", "X_2", "X_3", "X_4"],
    outcome_model_type="glm",
    outcome_xvars=[
        "X_outcome_1", "X_outcome_2", "X_outcome_3", "X_outcome_4",
        "X_outcome_5", "X_outcome_6", "X_outcome_7", "X_outcome_8",
    ],
    fit_method_int="multinomial",
    nfolds=5,
    pa_xvars_int=[
        "X_pa_1", "X_pa_2", "X_pa_3", "X_pa_4",
        "X_pa_5", "X_pa_6", "X_pa_7",
    ],
    data_external=ex_data_external,
    fit_method_ext="multinomial",
    pa_xvars_ext=["X_pa_1", "X_pa_2", "X_pa_3", "X_pa_4", "X_pa_5"],
    borrow_metric="brier",
    pa_model_ext=pa_model_ext_ex,
)


# -------------------------------
# 5) Plots
# -------------------------------
results_plots = get_plots(
    results_smallsub,
    sampsize=5000,
    alpha=0.05,
    m_factor=0.75,
)

# Example: negative (cFNR) versions of small subgroup metrics
# metrics_pos = results_plots["metrics_pos"]
