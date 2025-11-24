"""
Intersectional Fairness Toolkit (iftoolkit)

Top-level public API aggregating Components 1–3.

Naming convention:
- ifm_*  : Component 1 (Intersectional Fairness Metrics)
- cft_*  : Component 2 (Counterfactual IF w/ treatment)
- cf_*   : Component 3 (Counterfactual IF no treatment)

Users can still import component-specific APIs via:
    from iftoolkit.component2 import get_defs_analysis
"""


#Component 1: Intersectional Fairness Metrics
from .Component1 import (
    GroupRates,
    FairnessResults,
    evaluate_intersectional_fairness as ifm_evaluate_intersectional_fairness,
    filter_intersectional_groups as ifm_filter_intersectional_groups,
    confusion_by_group as ifm_confusion_by_group,
)


#Component 2: Counterfactual IF w/ treatment
from .Component2 import (
    prob_trunc as cft_prob_trunc,
    manhattan as cft_manhattan,
    get_defs_analysis as cft_get_defs_analysis,
    bs_rescaled_analysis as cft_bs_rescaled_analysis,
    get_est_analysis as cft_get_est_analysis,
    analysis_nulldist as cft_analysis_nulldist,
    analysis_estimation as cft_analysis_estimation,
    cond_round_3 as cft_cond_round_3,
    select_coef as cft_select_coef,
    get_bs_rescaled as cft_get_bs_rescaled,
    ci_norm as cft_ci_norm,
    ci_tint as cft_ci_tint,
    ci_trunc as cft_ci_trunc,
    get_plots as cft_get_plots,
    annotate_plot as cft_annotate_plot,
    categorize_stat as cft_categorize_stat,
    borrow_alpha as cft_borrow_alpha,
    get_models_small as cft_get_models_small,
    get_pa_int_small as cft_get_pa_int_small,
    get_pa_ext_small as cft_get_pa_ext_small,
)

#Component 3: Counterfactual IF no treatment
from .Component3 import (
    build_outcome_models_and_scores as cf_build_outcome_models_and_scores,
    get_bs_rescaled as cf_get_bs_rescaled,
    get_defs_analysis as cf_get_defs_analysis,
    get_defs_from_rates as cf_get_defs_from_rates,
    FairnessPipeline,
    make_outcome_estimator as cf_make_outcome_estimator,
    ProbaEstimator,
    choose_threshold_youden as cf_choose_threshold_youden,
    ensure_probabilistic_estimator as cf_ensure_probabilistic_estimator,
    Model,
    get_plots as cf_get_plots,
    annotate_plot as cf_annotate_plot,
)

__all__ = [
    #----Component 1----
    "GroupRates",
    "FairnessResults",
    "ifm_evaluate_intersectional_fairness",
    "ifm_filter_intersectional_groups",
    "ifm_confusion_by_group",

    #----Component 2----
    "cft_prob_trunc",
    "cft_manhattan",
    "cft_get_defs_analysis",
    "cft_bs_rescaled_analysis",
    "cft_get_est_analysis",
    "cft_analysis_nulldist",
    "cft_analysis_estimation",
    "cft_cond_round_3",
    "cft_select_coef",
    "cft_get_bs_rescaled",
    "cft_ci_norm",
    "cft_ci_tint",
    "cft_ci_trunc",
    "cft_get_plots",
    "cft_annotate_plot",
    "cft_categorize_stat",
    "cft_borrow_alpha",
    "cft_get_models_small",
    "cft_get_pa_int_small",
    "cft_get_pa_ext_small",

    #----Component 3----
    "cf_build_outcome_models_and_scores",
    "cf_get_bs_rescaled",
    "cf_get_defs_analysis",
    "cf_get_defs_from_rates",
    "FairnessPipeline",
    "cf_make_outcome_estimator",
    "ProbaEstimator",
    "cf_choose_threshold_youden",
    "cf_ensure_probabilistic_estimator",
    "Model",
    "cf_get_plots",
    "cf_annotate_plot",
]

