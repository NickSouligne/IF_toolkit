import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedShuffleSplit
from .functions_smallsubgroups import get_models_small
from sklearn.preprocessing import StandardScaler




def prob_trunc(p):
    return np.maximum(np.minimum(p, 0.995), 0.005)

#function to calculate Manhattan distance 
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def get_defs_analysis(data, cutoff, estimator_type, preds_out=None, preds_pa=None):
    #Grid of values for protected characteristic combinations
    if data['A2'].nunique() > 1:
        data['A1'] = pd.Categorical(data['A1'])
        data['A2'] = pd.Categorical(data['A2'])
        #Get the unique levels of A1 and A2
        A1_levels = np.sort(data['A1'].unique())
        A2_levels = np.sort(data['A2'].unique())
        #Create a grid of combinations
        A1A2_grid = pd.DataFrame([(a2, a1) for a1 in A1_levels for a2 in A2_levels], columns=['Var1', 'Var2'])
    else:
        #If A2 has only one level
        A1A2_grid = pd.DataFrame({'Var2': data['A1'].unique(), 'Var1': [1]*len(data['A1'].unique())})

    

    #Set up A1A2 factor
    for i in range(len(A1A2_grid)):
        varname = f"A1A2_{i+1}"
        data[varname] = np.where((data['A1'] == A1A2_grid.loc[i, 'Var2']) & 
                                (data['A2'] == A1A2_grid.loc[i, 'Var1']), 1, 0)
        

    data['A1A2'] = data['A1'].astype(str) + data['A2'].astype(str)

    #Other variables for fairness definitions
    Y0 = data['Y0']
    S = (data['S_prob'] >= cutoff).astype(int)
    Y = data['Y'].astype(int)
    D = data['D'].astype(int)
    pi = data['pi']
    pahat_mat = preds_pa

    #Intersectional FPRs and FNRs (observational)
    fpr_vec = []
    fnr_vec = []
    cfpr_vec = []
    cfnr_vec = []


    for i, row in A1A2_grid.iterrows():
        varname = f"A1A2_{i + 1}"
        fpr = (data[varname] * S * (1 - Y)).mean() / (data[varname] * (1 - Y)).mean()
        fnr = (data[varname] * (1 - S) * Y).mean() / (data[varname] * Y).mean()
        cfpr = (data[varname] * S * (1 - Y) * (1 - D) / (1 - pi)).mean() / (data[varname] * (1 - Y) * (1 - D) / (1 - pi)).mean()
        cfnr = (data[varname] * (1 - S) * Y0).mean() / (data[varname] * Y0).mean()

        fpr_vec.append(fpr)
        fnr_vec.append(fnr)
        cfpr_vec.append(cfpr)
        cfnr_vec.append(cfnr)

    #Remove groups with 0 or 1 and give a warning
    cfpr_vec_dropped = [x for x in cfpr_vec if x not in [0, 1]]
    cfnr_vec_dropped = [x for x in cfnr_vec if x not in [0, 1]]
    fpr_vec_dropped = [x for x in fpr_vec if x not in [0, 1]]
    fnr_vec_dropped = [x for x in fnr_vec if x not in [0, 1]]

    if len(cfpr_vec_dropped) != len(cfpr_vec):
        print("cFPR of 0, 1, or NULL for at least one group.")
    if len(cfnr_vec_dropped) != len(cfnr_vec):
        print("cFNR of 0, 1, or NULL for at least one group.")
    if len(fpr_vec_dropped) != len(fpr_vec):
        print("Observational FPR of 0, 1, or NULL for at least one group.")
    if len(fnr_vec_dropped) != len(fnr_vec):
        print("Observational FNR of 0, 1, or NULL for at least one group.")


    #Marginal cFPRs and cFNRs
    cfpr_marg_vec_A1 = []
    cfpr_marg_vec_A2 = []
    cfnr_marg_vec_A1 = []
    cfnr_marg_vec_A2 = []

    for level in data['A1'].unique():
        varvec = (data['A1'] == level).astype(int)
        cfpr_marg = (varvec * S * (1 - Y) * (1 - D) / (1 - pi)).mean() / (varvec * (1 - Y) * (1 - D) / (1 - pi)).mean()
        cfpr_marg_vec_A1.append(cfpr_marg)

    for level in data['A2'].unique():
        varvec = (data['A2'] == level).astype(int)
        cfpr_marg = (varvec * S * (1 - Y) * (1 - D) / (1 - pi)).mean() / (varvec * (1 - Y) * (1 - D) / (1 - pi)).mean()
        cfpr_marg_vec_A2.append(cfpr_marg)

    for level in data['A1'].unique():
        varvec = (data['A1'] == level).astype(int)
        cfnr_marg = (varvec * (1 - S) * Y0).mean() / (varvec * Y0).mean()
        cfnr_marg_vec_A1.append(cfnr_marg)

    for level in data['A2'].unique():
        varvec = (data['A2'] == level).astype(int)
        cfnr_marg = (varvec * (1 - S) * Y0).mean() / (varvec * Y0).mean()
        cfnr_marg_vec_A2.append(cfnr_marg)

    #Intersectional deltas
    cdeltaps = []
    cdeltans = []

    #Find pairwise Manhattan distances
    cfpr_vec_arr = np.array(cfpr_vec_dropped).reshape(-1,1)
    distances = pdist(cfpr_vec_arr, metric='cityblock')
    cdeltaps.append(distances)

    cfnr_vec_arr = np.array(cfnr_vec_dropped).reshape(-1,1)
    distances2 = pdist(cfnr_vec_arr, metric = 'cityblock')
    cdeltans.append(distances2)


    #Calculate cdeltans and cdeltaps as arrays
    #Average
    if len(cdeltans) > 0:
        cdelta_avg_neg = np.nanmean(np.abs(cdeltans))
    else:
        cdelta_avg_neg = np.nan

    if len(cdeltaps) > 0:
        cdelta_avg_pos = np.nanmean(np.abs(cdeltaps))
    else:
        cdelta_avg_pos = np.nan

    #Max
    if len(cdeltans) > 0:
        cdelta_max_neg = np.nanmax(np.abs(cdeltans))
    else:
        cdelta_max_neg = np.nan

    if len(cdeltaps) > 0:
        cdelta_max_pos = np.nanmax(np.abs(cdeltaps))
    else:
        cdelta_max_pos = np.nan

    #Variational
    if len(cdeltans) > 0:
        cdelta_var_neg = np.nanvar(np.abs(cdeltans), ddof=1)
    else:
        cdelta_var_neg = np.nan

    if len(cdeltaps) > 0:
        cdelta_var_pos = np.nanvar(np.abs(cdeltaps), ddof=1)
    else:
        cdelta_var_pos = np.nan


    
    ############# SMALL SUBGROUP METRICS ############################
    if estimator_type in ["small_internal", "small_borrow"]:
        #Overall cFPR and cFNR
        pi = data['pi']  #Assuming pi is defined somewhere in your data
        S = data['S']
        Y = data['Y']
        D = data['D']


        cfpr_all = np.mean(S * (1 - Y) * (1 - D) / (1 - pi)) / np.mean((1 - Y) * (1 - D) / (1 - pi))
        cfnr_all = np.mean(((1 - S) * Y * (1 - D)) / (1 - pi)) / np.mean((Y * (1 - D)) / (1 - pi))

        #Estimate numerators
        cfprnum_vec_small = np.array([
            np.mean((1 - preds_out['mu0S1hat']) * S * data[f'A1A2_{i + 1}']) / 
            np.mean((1 - preds_out['mu0S1hat']) * S)
            for i in range(len(A1A2_grid))
        ])

        cfnrnum_vec_small = np.array([
            np.mean(preds_out['mu0S0hat'] * (1 - S) * data[f'A1A2_{i + 1}']) / 
            np.mean(preds_out['mu0S0hat'] * (1 - S))
            for i in range(len(A1A2_grid))
        ])


        
        #Estimate cFPR and cFNR small subgroup versions
        cfpr_vec_small = np.array([
            cfpr_all * cfprnum_vec_small[r] / 
            (np.mean(pahat_mat.iloc[:, r] * (1 - preds_out['mu0hat'])) / np.mean(1 - preds_out['mu0hat']))
            for r in range(len(A1A2_grid))
        ])

        cfnr_vec_small = np.array([
            cfnr_all * cfnrnum_vec_small[r] / 
            (np.mean(pahat_mat.iloc[:, r] * preds_out['mu0hat']) / np.mean(preds_out['mu0hat']))
            for r in range(len(A1A2_grid))
        ])

        #Remove groups with 0 or 1 and give a warning
        cfpr_vec_small_dropped = cfpr_vec_small[(cfpr_vec_small != 0) & (cfpr_vec_small != 1)]
        cfnr_vec_small_dropped = cfnr_vec_small[(cfnr_vec_small != 0) & (cfnr_vec_small != 1)]
        
        if len(cfpr_vec_small_dropped) != len(cfpr_vec_small):
            print("Warning: Small subgroup cFPR of 0 or 1 for at least one group.")
        
        if len(cfnr_vec_small_dropped) != len(cfnr_vec_small):
            print("Warning: Small subgroup cFNR of 0 or 1 for at least one group.")

        #Intersectional deltas
        cdeltaps_small = None
        try:
            cdeltaps_small = pdist(cfpr_vec_small_dropped.reshape(-1, 1), metric='cityblock')
        except:
            pass

        cdeltans_small = None
        try:
            cdeltans_small = pdist(cfnr_vec_small_dropped.reshape(-1, 1), metric='cityblock')
        except:
            pass

        #Average, Max, Variational
        cdelta_avg_neg_small = np.mean(np.abs(cdeltans_small)) if cdeltans_small is not None else np.nan
        cdelta_max_neg_small = np.max(np.abs(cdeltans_small)) if cdeltans_small is not None else np.nan
        cdelta_var_neg_small = np.var(np.abs(cdeltans_small)) if cdeltans_small is not None else np.nan

        cdelta_avg_pos_small = np.mean(np.abs(cdeltaps_small)) if cdeltaps_small is not None else np.nan
        cdelta_max_pos_small = np.max(np.abs(cdeltaps_small)) if cdeltaps_small is not None else np.nan
        cdelta_var_pos_small = np.var(np.abs(cdeltaps_small)) if cdeltaps_small is not None else np.nan


    #Return vector of metrics
    defs = (fpr_vec + fnr_vec + cfpr_vec + cfnr_vec +
            cfpr_marg_vec_A1 + cfpr_marg_vec_A2 +
            cfnr_marg_vec_A1 + cfnr_marg_vec_A2 +
            [cdelta_avg_neg] + [cdelta_max_neg] + [cdelta_var_neg] +
            [cdelta_avg_pos] + [cdelta_max_pos] + [cdelta_var_pos])
    
    if estimator_type in ["small_internal", "small_borrow"]:
        defs = np.concatenate([
            defs,
            cfpr_vec_small, cfnr_vec_small,
            [cdelta_avg_neg_small, cdelta_max_neg_small, cdelta_var_neg_small],
            [cdelta_avg_pos_small, cdelta_max_pos_small, cdelta_var_pos_small]
        ])
    
    return {'defs': defs}
    
# Rescaled, stratified bootstrap for unfairness metrics.
#
# @inheritParams analysis_estimation
# @param B Number of bootstrap replications.
# @param m_factor Fractional power for calculating resample size (default 0.75).
# @param defs_names Character vector of names for get_defs_analysis output.
#
# @returns A matrix of definitions, one row for each of the B bootstrap replications.
def bs_rescaled_analysis(data, cutoff, B, m_factor,
                         pi_model_type, pi_model_seed=None, pi_xvars=None,
                         defs_names=None, estimator_type=None,
                         outcome_model_type=None, outcome_xvars=None,
                         fit_method_int=None, nfolds=None, pa_xvars_int=None,
                         data_external=None, fit_method_ext=None, borrow_metric=None,
                         pa_xvars_ext=None, pa_model_ext=None):
    
    #Empty matrix for results
    boot_out = np.zeros((B, len(defs_names)))
    boot_out = pd.DataFrame(boot_out, columns=defs_names)
    
    #m size for m of n bootstrap
    n_rescaled = int(np.floor(len(data) ** m_factor))
    
    for b in range(B):
        #Stratified sample conditional on proportions of (A1, A2), Y, S (binary)
        data['A1A2'] = data['A1'].astype(str) + data['A2'].astype(str)
        data['S_char'] = np.where(data['S_prob'] >= cutoff, '1', '0')

        #Stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_rescaled / len(data), random_state=b)
        for train_index, _ in sss.split(data, data[['A1A2', 'Y', 'S_char']]):
            data_bs = data.iloc[train_index]

        #Get Y0 estimator
        ests = get_est_analysis(data_bs, cutoff, None, pi_model_type, pi_model_seed, pi_xvars)
        
        #Get small subgroup unfairness metrics
        if estimator_type in ["small_internal", "small_borrow"]:
            #Get outcome and P(A=a) models
            models_temp = get_models_small(data=data_bs, cutoff=cutoff, estimator_type=estimator_type,
                                           outcome_xvars=outcome_xvars, outcome_model_type=outcome_model_type,
                                           pa_xvars_int=pa_xvars_int, fit_method_int=fit_method_int, nfolds=nfolds,
                                           data_external=data_external, fit_method_ext=fit_method_ext, borrow_metric=borrow_metric,
                                           pa_xvars_ext=pa_xvars_ext, pa_model_ext=pa_model_ext)
            #Get metrics
            defs_temp = get_defs_analysis(data_bs.assign(Y0=ests['Y0est'], pi=ests['pi']), cutoff=cutoff,
                                          estimator_type=estimator_type, preds_out=models_temp['preds_out'],
                                          preds_pa=models_temp['preds_pa'])['defs']
        else:
            #Get original unfairness metrics
            defs_temp = get_defs_analysis(data_bs.assign(Y0=ests['Y0est'], pi=ests['pi']), cutoff=cutoff,
                                          estimator_type=estimator_type)['defs']
        
        #Add row to matrix
        boot_out.iloc[b, :] = defs_temp
    
    return boot_out


#Helper function to truncate probability between .001 and .999
def prob_trunc(probs):
    return np.clip(probs, 0.001, 0.999)

# Get estimated propensity score and $Y^0$ weighted estimator.
#
# @inheritParams analysis_estimation
#
# @returns List with the following components:
#  * Y0est: Vector of inverse probability weighted estimates of $Y^0$.
#  * pi: Vector of estimated propensity scores.
def get_est_analysis(data, cutoff, pi_model, pi_model_type, pi_model_seed, pi_xvars):
    #Blank dictionary for storing results
    est_choice = {"Y0est": None, "pi": None}

    #Classify predictions according to cutoff
    data['S'] = np.where(data['S_prob'] >= cutoff, 1, 0)

    #Get propensity score model if pi_model not specified
    if pi_model is None:
        if pi_model_type == "glm":
            pi_model_formula = 'D ~ ' + ' + '.join(pi_xvars)
            pi_model = sm.GLM.from_formula(pi_model_formula, data=data, family=sm.families.Binomial()).fit()
        elif pi_model_type == "rf":
            data_x = data[pi_xvars]
            matrix_x = StandardScaler().fit_transform(data_x)
            np.random.seed(pi_model_seed[0])
            pi_model = RandomForestClassifier(n_estimators=2000, max_features=90, min_samples_leaf=50, random_state=pi_model_seed[0])
            pi_model.fit(matrix_x, data['D'])

    #Get propensity score predictions
    if pi_model_type == 'glm':
        pihat = prob_trunc(pi_model.predict(data))
    elif pi_model_type == 'rf':
        data_x = data[pi_xvars]
        matrix_x = StandardScaler().fit_transform(data_x)
        np.random.seed(pi_model_seed[1])
        pihat = prob_trunc(pi_model.predict_proba(matrix_x)[:, 1])
    else:
        raise ValueError("'pi_model_type' must be 'glm' or 'rf'")

    #IPW estimate of Y0
    ipwhat = ((1 - data['D']) * data['Y']) / (1 - pihat)

    #Save Y0 and pi estimates
    est_choice['Y0est'] = ipwhat
    est_choice['pi'] = pihat

   

    return est_choice



# Simulate null distributions.
#
# @inheritParams bs_rescaled_analysis
# @param R Number of replications.
#
# @returns List with the following components:
#  * ipw: Matrix of null unfairness metrics, one row for each replication.
def analysis_nulldist(data, R, cutoff, pi_model_type, pi_model_seed, pi_xvars,
                      defs_names, estimator_type, outcome_model_type, outcome_xvars,
                      fit_method_int, nfolds, pa_xvars_int, data_external, fit_method_ext,
                      borrow_metric, pa_xvars_ext, pa_model_ext):
    

    #Matrix for storing results
    table_null = np.empty((R, len(defs_names)))
    table_null[:] = np.nan
    table_null_df = pd.DataFrame(table_null, columns=defs_names)

    for i in range(R):
        data_permute = data.copy()
        N_permute = data_permute.shape[0]

        #Sample A1, A2 jointly (without replacement)
        data_A = data_permute[['A1', 'A2']].copy()
        inds_A = np.random.permutation(N_permute)
        data_A_sampled = data_A.iloc[inds_A].reset_index(drop=True)

        data_permute['A1'] = data_A_sampled['A1']
        data_permute['A2'] = data_A_sampled['A2']

        #Estimate nuisance parameters and Y0
        est_list_permute = get_est_analysis(data_permute, cutoff=cutoff, pi_model=None,
                                            pi_model_type=pi_model_type, pi_model_seed=pi_model_seed, pi_xvars=pi_xvars)

        #Get small subgroup unfairness metrics
        if estimator_type in ["small_internal", "small_borrow"]:
            #Get outcome and P(A=a) models
            models_temp = get_models_small(data=data_permute, cutoff=cutoff, estimator_type=estimator_type,
                                           outcome_xvars=outcome_xvars, outcome_model_type=outcome_model_type,
                                           pa_xvars_int=pa_xvars_int, fit_method_int=fit_method_int, nfolds=nfolds,
                                           data_external=data_external, fit_method_ext=fit_method_ext,
                                           borrow_metric=borrow_metric, pa_xvars_ext=pa_xvars_ext,
                                           pa_model_ext=pa_model_ext)

            #Get metrics
            data_permute = data_permute.assign(Y0=est_list_permute['Y0est'], pi=est_list_permute['pi'])
            defs_permute = get_defs_analysis(data_permute, cutoff=cutoff, estimator_type=estimator_type,
                                             preds_out=models_temp['preds_out'], preds_pa=models_temp['preds_pa'])['defs']
        else:
            #Get original unfairness metrics
            data_permute = data_permute.assign(Y0=est_list_permute['Y0est'], pi=est_list_permute['pi'])
            defs_permute = get_defs_analysis(data_permute, cutoff=cutoff, estimator_type=estimator_type)['defs']


        
        #Add a row to the null distribution table
        table_null_df.iloc[i, :] = defs_permute


    return table_null_df

#' Main function: Estimation of nuisance parameters and unfairness metrics.
#'
#' @param data Dataframe or tibble. Must include at least A1, A2, Y (binary 0/1),
#'  D (binary 0/1), covariates used to train propensity score model,
#'  S_prob (probability).
#' @param cutoff Classification cutoff. Must be a single numeric value.
#' @param estimator_type Type of estimation: standard ('standard'), small subgroup internal-only ('small_internal'),
#'  or small subgroup with data borrowing ('small_borrow').
#' @param gen_null T/F: generate null distributions.
#' @param R_null Number of replications for permutation null distribution (default 500).
#' @param bootstrap Obtain bootstrap estimates using rescaled method ('rescaled') or
#'  no bootstrap ('none', the default).
#' @param B Number of bootstrap resamples (default 500).
#' @param m_factor Fractional power for calculating resample size (default 0.75).
#' @param pi_model Pre-fit propensity score model.
#' @param pi_model_type Type of propensity score model: logistic regression ('glm') or random forest ('rf')
#' @param pi_model_seed Numeric vector of random seeds for random forest propensity score model. Required if pi_model_type is 'rf'.
#'  Must be length 2 if pi_model is not specified and length 1 if pi_model is specified.
#'  First element is set prior to model fitting; second element (or only element if pi_model specified) is set prior to prediction.
#' @param pi_xvars Character vector of covariates used in propensity score model.
#' @param outcome_model_type Type of outcome model: logistic regression ('glm'), random forest ('rf'),
#'  or neural network ('neural_net')
#' @param outcome_xvars Character vector of covariates used in outcome models.
#' @param fit_method_int Type of internal data model for P(A=a): multinomial ('multinomial') or neural network ('neural_net')
#' @param nfolds Number of folds for cross-fitting (default 5).
#' @param pa_xvars_int Character vector of covariates used in internal P(A=a) model.
#' @param data_external External data set. Must contain at least A1, A2, and a subset of covariates used to train P(A=a) model.
#' @param fit_method_ext Type of external data model for P(A=a): multinomial ('multinomial') or neural network ('neural_net')
#' @param pa_xvars_ext Character vector of covariates used in external P(A=a) model.
#' @param borrow_metric Metric for data borrowing: AUC ('auc') or Brier score ('brier')
#' @param pa_model_ext Pre-fit external P(A=a) model. Can be either multinomial or neural network model produced by the package nnet.
#'  Model type must match fit_method_ext.
#'
#' @returns List with the following components:
#'  * defs: estimated definitions
#'  * estY0: mean estimated Y0 for each protected group
#'  * table_null: null distribution table (if gen_null = T)
#'  * boot_out: bootstrap estimates for metrics and error rates (if bootstrap = 'rescaled')
#'  * est.choice: Input data frame with Y0 estimates added
#'
def analysis_estimation(data, cutoff, estimator_type='standard', gen_null=False, R_null=500, bootstrap='none', B=500,
                        m_factor=0.75, pi_model=None, pi_model_type='glm', pi_model_seed = None, pi_xvars=None, outcome_model_type=None,
                        outcome_xvars=None, fit_method_int=None, nfolds=5, pa_xvars_int = None, data_external=None, fit_method_ext=None,
                        pa_xvars_ext=None, borrow_metric=None, pa_model_ext=None, formula = None):

    if pi_model_type == 'rf':
        if pi_model is None:
            assert len(pi_model_seed) == 2
        else:
            assert len(pi_model_seed) == 1

    if estimator_type == "small_borrow" and ((data_external is None and pa_model_ext is None) or fit_method_ext is None or borrow_metric is None or pa_xvars_ext is None):
        raise ValueError("'fit_method_ext', 'borrow_metric', 'pa_xvars_ext', and either 'data_external' or 'pa_model_ext' must be specified if 'estimator_type' is 'small_borrow'.")

    if estimator_type in ["small_internal", "small_borrow"] and (pa_xvars_int is None or fit_method_int is None or nfolds is None or outcome_xvars is None or outcome_model_type is None):
        raise ValueError("'pa_xvars_int', 'fit_method_int', 'nfolds', 'outcome_xvars', 'outcome_model_type' must be specified if 'estimator_type' is 'small_borrow' or 'small_internal'.")

    #Required data columns and types
    required_cols = ["A1", "A2", "Y", "D", "S_prob"]
    if not all(col in data.columns for col in required_cols):
        raise ValueError("'data' must contain columns: A1, A2, Y (binary), D (binary), S_prob (continuous), and propensity score model covariates.")
    if len(data['Y'].unique()) != 2 or len(data['D'].unique()) != 2:
        raise ValueError("'data' must have exactly two unique values for columns Y (binary) and D (binary).")

    if not all(pi_xvars in data.columns for pi_xvars in pi_xvars):
        raise ValueError("'data' must contain all propensity score model covariates (pi_xvars).")

    if estimator_type in ["small_internal", "small_borrow"]:
        if not all(outcome_xvars in data.columns for outcome_xvars in outcome_xvars):
            raise ValueError("'data' must contain all outcome model covariates (outcome_xvars).")
        if not all(pa_xvars_int in data.columns for pa_xvars_int in pa_xvars_int):
            raise ValueError("'data' must contain all internal P(A=a) model covariates (pa_xvars_int).")

    if estimator_type == "small_borrow":
        if not all(pa_xvars_ext in pa_xvars_int for pa_xvars_ext in pa_xvars_ext):
            raise ValueError("All covariates in 'pa_xvars_ext' must be present in 'pa_xvars_int'.")
        if not all(pa_xvars_ext in data_external.columns for pa_xvars_ext in pa_xvars_ext):
            raise ValueError("'data_external' must contain all external P(A=a) model covariates (pa_xvars_ext).")




    #Convert Y and D to 0/1 if not already
    data['Y'] = data['Y'].astype(int)
    data['D'] = data['D'].astype(int)

    #Drop rows with missing values
    data.dropna(inplace=True)


    #External data: required columns and types
    if data_external is not None:
        required_cols_ext = ["A1", "A2"]
        if not all(col in data_external.columns for col in required_cols_ext):
            raise ValueError("'data_external' must contain columns: A1, A2, and covariates for P(A=a) model.")

        #Dropping rows with missing values
        data_ext_complete = data_external.dropna()
        if len(data_ext_complete) != len(data_external):
            print("Warning: External data contains missing values, dropping incomplete rows")
            data_external = data_ext_complete.copy()

        #Creating a new column 'A1A2'
        data_external['A1A2'] = data_external['A1'].astype(str) + data_external['A2'].astype(str)
        #data_external['A1A2'] = data_external['A1A2'].astype('category')

    if pa_xvars_ext is not None and not set(pa_xvars_ext).issubset(set(pa_xvars_int)):
        raise ValueError("All external P(A=a) covariates must be included in internal covariates ('pa_xvars_ext' must be a subset of 'pa_xvars_int').")


    #Convert A1 and A2 to categorical variables
    if estimator_type != 'small_internal':
        #data['A1'] = pd.Categorical(data['A1'], categories=[0, 1], ordered=False)
        #data['A2'] = pd.Categorical(data['A2'], categories=[0, 1], ordered=False)
        data['A1'] = data['A1'].astype(int)
        data['A2'] = data['A2'].astype(int)

    #Set metric names and return vector length
    if len(data['A2'].unique()) > 1:
        A1_levels = data['A1'].unique()
        A2_levels = data['A2'].unique()
        A1A2_grid = pd.DataFrame(list(itertools.product(A2_levels, A1_levels)), columns=['Var2', 'Var1'])
    else:
        A1A2_grid = pd.DataFrame({'Var2': data['A1'].unique(), 'Var1': 1})

    A1A2_names = [f"{row['Var2']}{row['Var1']}" for idx, row in A1A2_grid.iterrows()]

    defs_names_standard = (["fpr_" + str(name) for name in A1A2_names] +
                        ["fnr_" + str(name) for name in A1A2_names] +
                        ["cfpr_" + str(name) for name in A1A2_names] +
                        ["cfnr_" + str(name) for name in A1A2_names] +
                        ["cfpr_marg_A1_" + str(level) for level in data['A1'].unique()] +
                        ["cfpr_marg_A2_" + str(level) for level in data['A2'].unique()] +
                        ["cfnr_marg_A1_" + str(level) for level in data['A1'].unique()] +
                        ["cfnr_marg_A2_" + str(level) for level in data['A2'].unique()] +
                        ["avg_neg", "max_neg", "var_neg", "avg_pos", "max_pos", "var_pos"])

    defs_names_small = defs_names_standard + (["cfpr_small_" + name for name in A1A2_names] +
                                            ["cfnr_small_" + name for name in A1A2_names] +
                                            ["avg_neg_small", "max_neg_small", "var_neg_small",
                                            "avg_pos_small", "max_pos_small", "var_pos_small"])


    if estimator_type in ["small_internal", "small_borrow"]:
        defs_names = defs_names_small
    else:
        defs_names = defs_names_standard

    if gen_null:
        #Call the function or perform actions when gen_null is True
        table_null_temp = analysis_nulldist(data, R=R_null, cutoff=cutoff,
                                            pi_model_type=pi_model_type, pi_model_seed=pi_model_seed, pi_xvars=pi_xvars,
                                            defs_names=defs_names, estimator_type=estimator_type,
                                            outcome_model_type=outcome_model_type, outcome_xvars=outcome_xvars,
                                            fit_method_int=fit_method_int, nfolds=nfolds, pa_xvars_int=pa_xvars_int,
                                            data_external=data_external, fit_method_ext=fit_method_ext, borrow_metric=borrow_metric, pa_xvars_ext=pa_xvars_ext,
                                            pa_model_ext=pa_model_ext)
        
    if bootstrap == 'rescaled':
        #Call the function or perform actions when bootstrap is 'rescaled'
        boot_out = bs_rescaled_analysis(data, B=B, m_factor=m_factor, cutoff=cutoff,
                                        pi_model_type=pi_model_type, pi_model_seed=pi_model_seed, pi_xvars=pi_xvars,
                                        defs_names=defs_names, estimator_type=estimator_type,
                                        outcome_model_type=outcome_model_type, outcome_xvars=outcome_xvars,
                                        fit_method_int=fit_method_int, nfolds=nfolds, pa_xvars_int=pa_xvars_int,
                                        data_external=data_external, fit_method_ext=fit_method_ext, borrow_metric=borrow_metric, pa_xvars_ext=pa_xvars_ext,
                                        pa_model_ext=pa_model_ext)


    #Estimate nuisance parameters and Y^0
    est_choice_list = get_est_analysis(data, cutoff=cutoff,
                                        pi_model=pi_model, pi_model_type=pi_model_type, pi_model_seed= 1, pi_xvars=pi_xvars)
    

    est_choice_temp = data.copy()  

    #Add estimated Y0 and pi to test data for return
    est_choice_temp['Y0est'] = est_choice_list['Y0est']
    est_choice_temp['pi'] = est_choice_list['pi']


    if estimator_type in ["small_internal", "small_borrow"]:
        #Get outcome and P(A=a) models
        models_temp = get_models_small(data=data, cutoff=cutoff, estimator_type=estimator_type,
                                    outcome_xvars=outcome_xvars, outcome_model_type=outcome_model_type,
                                    pa_xvars_int=pa_xvars_int, fit_method_int=fit_method_int, nfolds=nfolds,
                                    data_external=data_external, fit_method_ext=fit_method_ext, borrow_metric=borrow_metric, pa_xvars_ext=pa_xvars_ext,
                                    pa_model_ext=pa_model_ext)
    
    

        #Get metrics
        defs_temp = get_defs_analysis(data.assign(Y0=est_choice_list['Y0est'], pi=est_choice_list['pi']), cutoff=cutoff,
                                    estimator_type=estimator_type,
                                    preds_out=models_temp['preds_out'], preds_pa=models_temp['preds_pa'])
        
        

    else:
        #Get original unfairness metrics
        defs_temp = get_defs_analysis(data.assign(Y0=est_choice_list['Y0est'], pi=est_choice_list['pi']), cutoff=cutoff,
                                    estimator_type=estimator_type)
    

    #Assigning column names to each array in defs_temp
    named_defs_temp = {name: array for name, array in zip(defs_names, defs_temp['defs'])}
    defs_temp = named_defs_temp

    #Initialize return dictionary
    return_list = {'defs': defs_temp, 'est_choice': est_choice_temp}

    #Add components based on conditions
    if 'table_null_temp' in locals():
        return_list['table_null'] = table_null_temp

    if 'boot_out' in locals():
        return_list['boot_out'] = boot_out

    if estimator_type == "small_borrow" and 'models_temp' in locals():
        return_list['alpha'] = models_temp['alpha']

    return return_list