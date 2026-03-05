import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Helper functions
# -------------------------------
def logit(x):
    x = np.asarray(x)
    return np.log(x / (1 - x))

def prob_trunc(p):
    p = np.asarray(p)
    return np.maximum(np.minimum(p, 0.995), 0.005)


# -------------------------------
# Parameters (params_1 and params_list)
# -------------------------------
params_1 = {
    "nr_marg": 0.5,
    "nr_int": 0.4,
    "nr_maj": 0.6,
    "or_marg": 0.4,
    "or_int": 0.6,
    "or_maj": 0.2,
    "orOpp_maj": 0.3,
    "orOpp_margint": 0.2,
}

params_list = {
    "pa": np.array([0.49, 0.23, 0.13, 0.15]),
    "p": 4,
    "mean_X": np.array([1.0, -1.0, 2.0, -2.0]),
    "z_trt_int": 0.5,
    "z_trt_marg": 0.4,
    "z_trt_A11_A20": 0.3,
    "z_trt_maj": 0.2,
}

# Add the derived parameters
params_list.update(
    alpha_Y0 = logit(params_1["nr_maj"]),
    beta_Y0 = np.array([1.0, 1.0, 1.0, 0.75]),
    betaA_Y0 = np.array([
        logit(params_1["nr_marg"]) - logit(params_1["nr_maj"]),
        logit(params_1["nr_marg"]) - logit(params_1["nr_maj"]),
        logit(params_1["nr_maj"])
        - 2 * logit(params_1["nr_marg"])
        + logit(params_1["nr_int"]),
    ]),
    alpha_D = logit(params_1["or_maj"]),
    beta_D = np.array([1.0, 1.0]),
    betaA_D = np.array([
        logit(params_1["or_marg"]) - logit(params_1["or_maj"]),
        logit(params_1["or_marg"]) - logit(params_1["or_maj"]),
        logit(params_1["or_maj"])
        - 2 * logit(params_1["or_marg"])
        + logit(params_1["or_int"]),
    ]),
    alpha_DOpp = logit(params_1["orOpp_maj"]),
    beta_DOpp = np.array([1.0, 1.0]),
    betaA_DOpp = np.array([
        logit(params_1["orOpp_margint"]) - logit(params_1["orOpp_maj"]),
        logit(params_1["orOpp_margint"]) - logit(params_1["orOpp_maj"]),
        logit(params_1["orOpp_maj"]) - logit(params_1["orOpp_margint"]),
    ]),
    betaS_D = logit(0.1),
    betaS_DOpp = logit(0.1),
)


# -------------------------------
# Data generator (translation of data_gen)
# -------------------------------
def data_gen(params, type_, N, rai=None, cutoff=0.5):
    pa = params["pa"]
    p = params["p"]
    mean_X = params["mean_X"]

    alpha_Y0 = params["alpha_Y0"]
    beta_Y0 = params["beta_Y0"]
    betaA_Y0 = params["betaA_Y0"]

    alpha_D = params["alpha_D"]
    beta_D = params["beta_D"]
    betaA_D = params["betaA_D"]

    alpha_DOpp = params["alpha_DOpp"]
    beta_DOpp = params["beta_DOpp"]
    betaA_DOpp = params["betaA_DOpp"]

    z_trt_int = params["z_trt_int"]
    z_trt_marg = params["z_trt_marg"]
    z_trt_A11_A20 = params["z_trt_A11_A20"]
    z_trt_maj = params["z_trt_maj"]

    betaS_D = params["betaS_D"]
    betaS_DOpp = params["betaS_DOpp"]

    # A1_A2 group order:
    # 1: A1=0, A2=0
    # 2: A1=1, A2=0
    # 3: A1=0, A2=1
    # 4: A1=1, A2=1
    A1_A2 = np.random.multinomial(1, pa, size=N)   # N x 4 one-hot
    A1_A2_vec = A1_A2.argmax(axis=1) + 1           # classes 1..4

    A1 = np.isin(A1_A2_vec, [2, 4]).astype(int)
    A2 = np.isin(A1_A2_vec, [3, 4]).astype(int)

    # Covariates X ~ MVN(mean_X, diag(0.3))
    cov_X = np.eye(p) * 0.3
    X = np.random.multivariate_normal(mean=mean_X, cov=cov_X, size=N)

    # 1) Probability of event under no treatment (Y0)
    XA = np.column_stack([X, A1, A2, A1 * A2])
    lin_Y0 = alpha_Y0 + XA @ np.concatenate([beta_Y0, betaA_Y0])
    prob_Y0 = prob_trunc(expit(lin_Y0))
    Y0 = np.random.binomial(1, prob_Y0)

    # 2) Probability of no event under treatment, given Y0=1
    prob_Y1 = np.zeros(N)

    mask_int = (A1 == 1) & (A2 == 1) & (Y0 == 1)
    mask_marg = (A1 == 0) & (A2 == 1) & (Y0 == 1)
    mask_A11A20 = (A1 == 1) & (A2 == 0) & (Y0 == 1)
    mask_maj = (A1 == 0) & (A2 == 0) & (Y0 == 1)

    prob_Y1[mask_int] = 1 - z_trt_int
    prob_Y1[mask_marg] = 1 - z_trt_marg
    prob_Y1[mask_A11A20] = 1 - z_trt_A11_A20
    prob_Y1[mask_maj] = 1 - z_trt_maj

    Y1 = np.random.binomial(1, prob_Y1)

    # 3) Opportunity rate models (D) and opposite
    S = None
    S_prob = None

    if type_ == "pre":
        XA_D = np.column_stack([X[:, :2], A1, A2, A1 * A2])
        lin_or = alpha_D + XA_D @ np.concatenate([beta_D, betaA_D])
        lin_orOpp = alpha_DOpp + XA_D @ np.concatenate([beta_DOpp, betaA_DOpp])

    elif type_ == "post":
        if rai is None:
            raise ValueError("Specify rai if type_='post'.")

        # Features for RAI
        new_data = pd.DataFrame({
            "A1": A1,
            "A2": A2,
            "X.1": X[:, 0],
            "X.2": X[:, 1],
            "X.3": X[:, 2],
            "X.4": X[:, 3],
        })

        S_prob = rai.predict_proba(new_data)[:, 1]
        S = (S_prob >= cutoff).astype(int)

        XA_D = np.column_stack([X[:, :2], A1, A2, A1 * A2, S])
        lin_or = alpha_D + XA_D @ np.concatenate([beta_D, betaA_D, np.array([betaS_D])])
        lin_orOpp = alpha_DOpp + XA_D @ np.concatenate([beta_DOpp, betaA_DOpp, np.array([betaS_DOpp])])

    else:
        raise ValueError("type_ must be 'pre' or 'post'")

    # Generate D and Y
    prob_D = prob_trunc(expit(lin_or))
    D = np.random.binomial(1, prob_D)
    Y = (1 - D) * Y0 + D * Y1

    # Assemble output DataFrame (mimic as.data.frame(list(...)) in R)
    data = {
        "A1": A1.astype(str),  # mimic factor-as-character
        "A2": A2.astype(str),
    }
    for j in range(p):
        data[f"X.{j+1}"] = X[:, j]
    data["Y"] = Y
    data["D"] = D

    if type_ == "post":
        data["S"] = S
        data["S_prob"] = S_prob

    return pd.DataFrame(data)


# -------------------------------
# 1) RAI training data (type = "pre")
# -------------------------------
data_train = data_gen(params_list, type_="pre", N=1000)
data_train["Y"] = data_train["Y"].astype(int)

# Random forest RAI (similar to randomForest::randomForest with mtry=6)
feature_cols = ["A1", "A2", "X.1", "X.2", "X.3", "X.4"]
X_train = data_train[feature_cols].astype(float)
y_train = data_train["Y"]

rai = RandomForestClassifier(
    n_estimators=500,
    max_features=6,   # all 6 predictors, analogous to mtry=6
    random_state=123
)
rai.fit(X_train, y_train)

# -------------------------------
# 2) Estimation data (type = "post")
# -------------------------------
ex_data_estimation = data_gen(
    params_list,
    type_="post",
    N=5000,
    rai=rai,
    cutoff=0.5
)

# -------------------------------
# 3) Save to CSV
# -------------------------------
ex_data_estimation.to_csv("ex_data_estimation.csv", index=False)
print("Saved ex_data_estimation.csv")
