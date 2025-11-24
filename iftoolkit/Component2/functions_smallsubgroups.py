import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.api import Logit
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
from sklearn.utils import shuffle

def prob_trunc(probs, epsilon=1e-6):
    return np.clip(probs, epsilon, 1 - epsilon)

def borrow_alpha(alpha, preds_ext, preds_int, data, borrow_metric):
    """
    Optimize multi-class AUC or Brier score for data borrowing parameter.

    Parameters:
    alpha (float): Borrowing parameter.
    preds_ext (numpy.ndarray): Predicted P(A=a) probabilities on internal data from external model.
    preds_int (numpy.ndarray): Predicted P(A=a) probabilities on internal data from internal model.
    data (pandas.DataFrame): Internal data set.
    borrow_metric (str): Which performance metric to use: Multi-class AUC ("auc") or Brier score ("brier").

    Returns:
    float: Multi-class AUC or Brier score for a given value of alpha.
    """
    # Make sure we're in array space
    preds_ext = np.asarray(preds_ext, dtype=float)
    preds_int = np.asarray(preds_int, dtype=float)
    #Weight predicted probabilities
    preds_borrow = alpha * preds_ext + (1 - alpha) * preds_int

    y_true = label_binarize(data['A1A2'], classes=np.unique(data['A1A2']))

    if borrow_metric == "auc":
        #Multi-class AUC
        auc_vals = []
        for i in range(y_true.shape[1]):
            auc_vals.append(roc_auc_score(y_true[:, i], preds_borrow[:, i]))
        auc_return = -np.mean(auc_vals)  #Negative because we are optimizing
        return auc_return

    elif borrow_metric == "brier":
        #Brier score
        brier_vals = []
        y_true_arr = np.asarray(y_true)
        preds_borrow_arr = np.asarray(preds_borrow)
        for i in range(y_true_arr.shape[1]):
            brier_vals.append(
                brier_score_loss(y_true_arr[:, i], preds_borrow_arr[:, i])
            )
        brier_return = np.mean(brier_vals)
        return brier_return

def get_models_small(
    data, cutoff, outcome_xvars, pa_xvars_int, estimator_type,
    outcome_model_type, fit_method_int, nfolds,
    data_external, fit_method_ext, borrow_metric, pa_xvars_ext, pa_model_ext
):
    # Set S using cutoff
    data['S'] = np.where(data['S_prob'] >= cutoff, 1, 0)

    # Formulas for outcome models
    f_mu0_model = f"Y ~ D + {' + '.join(outcome_xvars)}"
    f_mu0_S_model = f"Y ~ S + D + {' + '.join(outcome_xvars)}"

    # Standardize numeric outcome features
    outcome_xvars_numeric = data[outcome_xvars].select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[outcome_xvars_numeric] = scaler.fit_transform(data[outcome_xvars_numeric])

    # ----- Outcome models -----
    if outcome_model_type == "glm":
        mu0_model = sm.Logit.from_formula(f_mu0_model, data).fit()
        mu0hat = prob_trunc(mu0_model.predict(data.assign(D=0)))

        mu0_S_model = sm.Logit.from_formula(f_mu0_S_model, data).fit()
        mu0S1hat = prob_trunc(mu0_S_model.predict(data.assign(D=0, S=1)))
        mu0S0hat = prob_trunc(mu0_S_model.predict(data.assign(D=0, S=0)))

    elif outcome_model_type == "neural_net":
        # (unchanged from your version)
        mu0hat = mu0S1hat = mu0S0hat = np.zeros(len(data))
        data_cv = data.copy()
        data_cv['row_id'] = range(len(data_cv))
        data_cv['A1A2'] = data_cv['A1'].astype(str) + data_cv['A2'].astype(str)
        data_cv = data_cv.reset_index(drop=True)

        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        data_cv['fold'] = -1
        for fold, (_, test_idx) in enumerate(skf.split(data_cv, data_cv['A1A2'])):
            data_cv.loc[test_idx, 'fold'] = fold

        for fold in range(nfolds):
            train_idx = data_cv[data_cv['fold'] != fold].index
            test_idx = data_cv[data_cv['fold'] == fold].index
            train_data, test_data = data.loc[train_idx], data.loc[test_idx]

            mu0_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000, alpha=1, random_state=42)
            mu0_model.fit(train_data[outcome_xvars], train_data['Y'])
            mu0hat[test_idx] = prob_trunc(mu0_model.predict_proba(test_data[outcome_xvars])[:, 1])

            mu0_S_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000, alpha=1, random_state=42)
            mu0_S_model.fit(train_data[['S'] + outcome_xvars], train_data['Y'])
            mu0S1hat[test_idx] = prob_trunc(
                mu0_S_model.predict_proba(test_data.assign(S=1)[['S'] + outcome_xvars])[:, 1]
            )
            mu0S0hat[test_idx] = prob_trunc(
                mu0_S_model.predict_proba(test_data.assign(S=0)[['S'] + outcome_xvars])[:, 1]
            )

    pred_out_list = {"mu0hat": mu0hat, "mu0S1hat": mu0S1hat, "mu0S0hat": mu0S0hat}

    # ----- Internal P(A=a) -----
    pahat_mat_int_df = get_pa_int_small(data, pa_xvars_int, fit_method_int, nfolds)
    # prob_trunc on the numeric values only
    pahat_mat_int_arr = prob_trunc(pahat_mat_int_df.to_numpy(dtype=float))

    if estimator_type == "small_borrow":
        # External P(A=a)
        if pa_model_ext is None:
            pa_model_ext = get_pa_ext_small(data_external, pa_xvars_ext, fit_method_ext)

        if fit_method_ext == "multinomial":
            # statsmodels MNLogit: predict returns probabilities
            pahat_mat_ext_raw = pa_model_ext.predict(data)
        elif fit_method_ext == "neural_net":
            pahat_mat_ext_raw = pa_model_ext.predict_proba(data[pa_xvars_ext])

        pahat_mat_ext_arr = prob_trunc(np.asarray(pahat_mat_ext_raw, dtype=float))

        # Ensure shapes match before borrowing
        if pahat_mat_ext_arr.shape != pahat_mat_int_arr.shape:
            raise ValueError(
                f"Shape mismatch between external and internal PA matrices: "
                f"{pahat_mat_ext_arr.shape} vs {pahat_mat_int_arr.shape}"
            )

        # Prepare A1A2 for label_binarize in borrow_alpha
        data = data.copy()
        data['A1A2'] = data['A1'].astype(str) + data['A2'].astype(str)

        # Optimize alpha (scalar in [0,1])
        result_temp = minimize_scalar(
            borrow_alpha,
            bounds=(0, 1),
            args=(pahat_mat_ext_arr, pahat_mat_int_arr, data, borrow_metric),
            method='bounded',
        )
        alpha_temp = result_temp.x

        # Final borrowed probabilities (as array)
        pahat_borrow_arr = alpha_temp * pahat_mat_ext_arr + (1 - alpha_temp) * pahat_mat_int_arr

        # IMPORTANT: return as DataFrame so downstream code can use .iloc
        pahat_mat_df = pd.DataFrame(
            pahat_borrow_arr,
            index=data.index,
            columns=pahat_mat_int_df.columns,  # class labels align with internal model
        )

        return {"preds_out": pred_out_list, "preds_pa": pahat_mat_df, "alpha": alpha_temp}

    elif estimator_type == "small_internal":
        # No borrowing: return internal P(A=a) as DataFrame (unchanged behavior)
        return {"preds_out": pred_out_list, "preds_pa": pahat_mat_int_df}




def get_pa_int_small(data, pa_xvars_int, fit_method_int, nfolds):
    #Select numeric columns
    pa_xvars_int_numeric = data[pa_xvars_int].select_dtypes(include=[np.number]).columns

    #Create a copy of the data
    data_cv = data.copy()

    #Standardize numeric columns
    scaler = StandardScaler()
    data_cv[pa_xvars_int_numeric] = scaler.fit_transform(data_cv[pa_xvars_int_numeric])

    #Create row_id and A1A2 columns
    data_cv['row_id'] = np.arange(len(data_cv))
    data_cv['A1A2'] = data['A1'].astype(str) + data['A2'].astype(str)
    
    #Encode A1A2 as a categorical variable
    label_encoder_A1A2 = LabelEncoder()
    data_cv['A1A2'] = label_encoder_A1A2.fit_transform(data_cv['A1A2'])

    #Group by A1A2 and shuffle within groups
    data_cv = data_cv.sample(frac=1, random_state=1).reset_index(drop=True)

    #Assign folds
    data_cv['fold'] = np.tile(np.arange(1, nfolds + 1), len(data_cv) // nfolds + 1)[:len(data_cv)]

    #Arrange data by row_id
    data_cv = data_cv.sort_values('row_id').reset_index(drop=True)

    #Initialize prediction array
    pa_pred_int = np.zeros((len(data_cv), len(data_cv['A1A2'].unique())))

    #Define the formula for the model
    f_pa_model_int = 'A1A2 ~ ' + ' + '.join(pa_xvars_int)

    skf = StratifiedKFold(n_splits=nfolds)

    for train_index, test_index in skf.split(data_cv, data_cv['fold']):
        trainData = data_cv.iloc[train_index]
        testData = data_cv.iloc[test_index]

        X_train = trainData[pa_xvars_int]
        y_train = trainData['A1A2']
        X_test = testData[pa_xvars_int]

        if fit_method_int == "multinomial":
            if data['A1'].nunique() == 2 and data['A2'].nunique() == 1:
                pa_model_int = Logit(y_train, add_constant(X_train)).fit(disp=False)
                pa_pred_int[test_index, 1] = pa_model_int.predict(add_constant(X_test))
                pa_pred_int[test_index, 0] = 1 - pa_pred_int[test_index, 1]
            else:
                pa_model_int = LogisticRegression(solver='lbfgs', max_iter=1000)
                pa_model_int.fit(X_train, y_train)
                pa_pred_int[test_index, :] = pa_model_int.predict_proba(X_test)
        
        elif fit_method_int == "neural_net":
            pa_model_int = MLPClassifier(hidden_layer_sizes=(50,), alpha=1, max_iter=3000, random_state=1)
            pa_model_int.fit(X_train, y_train)
            if data['A1'].nunique() == 2 and data['A2'].nunique() == 1:
                pa_pred_int[test_index, 1] = pa_model_int.predict_proba(X_test)[:, 1]
                pa_pred_int[test_index, 0] = 1 - pa_pred_int[test_index, 1]
            else:
                pa_pred_int[test_index, :] = pa_model_int.predict_proba(X_test)
            if pa_model_int.n_iter_ < pa_model_int.max_iter:
                print("Internal model: converged")
            else:
                print("Internal model: did not converge")
    
    pa_pred_int = pd.DataFrame(pa_pred_int, columns=label_encoder_A1A2.classes_)
    return pa_pred_int


def get_pa_ext_small(data_external, pa_xvars_ext, fit_method_ext, maxit=1000):
    data_external['A1A2'] = data_external['A1'].astype(str) + data_external['A2'].astype(str)
    data_external['A1A2'] = pd.Categorical(data_external['A1A2']).codes
    pa_xvars_ext_numeric = data_external[pa_xvars_ext].select_dtypes(include=[np.number]).columns

    #Standardize numeric features
    scaler = StandardScaler()
    data_external[pa_xvars_ext_numeric] = scaler.fit_transform(data_external[pa_xvars_ext_numeric])

    f_pa_model_ext = f"A1A2 ~ {' + '.join(pa_xvars_ext)}"

    #Fit model (no CV needed because we will predict on internal data)
    if fit_method_ext == "multinomial":
        #Use statsmodels for multinomial logistic regression
        pa_model_ext = smf.mnlogit(formula=f_pa_model_ext, data=data_external).fit(disp=False)
    elif fit_method_ext == "neural_net":
        #Use scikit-learn for neural network model
        pa_model_ext = MLPClassifier(hidden_layer_sizes=(100,), max_iter=maxit, alpha=1, random_state=42)
        X_ext = data_external[pa_xvars_ext]
        y_ext = data_external['A1A2']
        pa_model_ext.fit(X_ext, y_ext)
        if pa_model_ext.n_iter_ < maxit:
            print("External model: converged")
        else:
            print("External model: did not converge")

    return pa_model_ext