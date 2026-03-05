import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------
# Helper functions
# -------------------------------------
def logit(x):
    x = np.asarray(x)
    return np.log(x / (1 - x))

def prob_trunc(p):
    p = np.asarray(p)
    return np.maximum(np.minimum(p, 0.995), 0.005)


# -------------------------------------
# Parameter list (translated from R)
# -------------------------------------
params_list = {
    "mean_X_ps": np.array([1, -1, 2, -2]),
    "mean_X_outcome": np.array([0.5, 0.5, -0.5, -0.5, 1, -1]),
    "mean_X_pa": np.array([1, -1, 0.5, -0.5, 1]),
    "A1_A2_means": np.array([
        [0.3, -0.05, 0,     0,     0,    -0.2],
        [-0.05, 0,   0.05,  0,     0.1,   0],
        [0.3,   0,  -0.1,   0.4,  -0.05, -0.2],
        [-0.4, -0.1, 0,    -0.3,   0,     0.05],
        [0,     0,   0,     0,     0.1,  -0.4],
        [0.3,   0.1, 0.1,  -0.05, -0.2,  -0.2],
        [0.4,  -0.1, -0.1,  0.05, -0.05, -0.3]
    ]),
    "z_trt": np.array([0.5, 0.4, 0.3, 0.2, 0.2, 0.2]),
    "alpha_Y0": logit(0.4),
    "beta_Y0": np.ones(8),
    "betaA_Y0": logit(np.array([0.5, 0.4, 0.4, 0.6, 0.4, 0.3])),
    "alpha_D": logit(0.6),
    "beta_D": np.repeat(0.7, 4),
    "betaA_D": logit(np.array([0.4, 0.6, 0.6, 0.2, 0.3, 0.3])),
    "alpha_DOpp": logit(0.3),
    "beta_DOpp": np.repeat(0.2, 4),
    "betaA_DOpp": logit(np.array([0.3, 0.2, 0.2, 0.2, 0.3, 0.3])),
    "betaS_D": logit(0.1),
    "betaS_DOpp": logit(0.1),
}

# -------------------------------------
# Core Data Generator (R → Python)
# -------------------------------------
def data_gen_small(params, type_, N, rai=None, cutoff=0.5):

    mean_X_ps = params["mean_X_ps"]
    mean_X_outcome = params["mean_X_outcome"]
    mean_X_pa = params["mean_X_pa"]

    A1_A2_means = params["A1_A2_means"]

    z_trt = params["z_trt"]

    alpha_Y0 = params["alpha_Y0"]
    beta_Y0 = params["beta_Y0"]
    betaA_Y0 = params["betaA_Y0"]

    alpha_D = params["alpha_D"]
    beta_D = params["beta_D"]
    betaA_D = params["betaA_D"]

    alpha_DOpp = params["alpha_DOpp"]
    beta_DOpp = params["beta_DOpp"]
    betaA_DOpp = params["betaA_DOpp"]

    betaS_D = params["betaS_D"]
    betaS_DOpp = params["betaS_DOpp"]

    # -------------------------------------
    # 1. X_ps covariates
    # -------------------------------------
    Xmat_ps = np.random.multivariate_normal(mean=mean_X_ps, cov=np.eye(len(mean_X_ps)) * 0.3, size=N)

    # -------------------------------------
    # 2. X_outcome covariates
    # -------------------------------------
    Xmat_out = np.random.multivariate_normal(
        mean=mean_X_outcome,
        cov=np.eye(len(mean_X_outcome)) * 0.5,
        size=N
    )

    # Add X_ps[,2:3]
    Xmat_out = np.column_stack([Xmat_out, Xmat_ps[:,1], Xmat_ps[:,2]])

    # -------------------------------------
    # 3. X_pa covariates
    # -------------------------------------
    Xmat_pa = np.random.multivariate_normal(
        mean=mean_X_pa,
        cov=np.eye(len(mean_X_pa)) * 0.5,
        size=N
    )
    Xmat_pa = np.column_stack([Xmat_pa, Xmat_ps[:,0], Xmat_out[:,0]])

    # -------------------------------------
    # 4. Sample A class (1..7)
    # -------------------------------------
    A1_A2_vec = np.zeros(N, dtype=int)
    for j in range(N):
        p_vec = expit(Xmat_pa[j, :] @ A1_A2_means)
        p_vec = p_vec / p_vec.sum()
        A1_A2_vec[j] = np.argmax(np.random.multinomial(1, p_vec)) + 1

    # -------------------------------------
    # 5. Translate to A1, A2
    # -------------------------------------
    A1 = np.isin(A1_A2_vec, [2,4,6]).astype(int)
    A2 = np.select(
        [
            np.isin(A1_A2_vec, [1,2]),
            np.isin(A1_A2_vec, [3,4]),
            np.isin(A1_A2_vec, [5,6])
        ],
        [0,1,2]
    )

    # One-hot encoding for A1_A2_vec
    Amat = np.zeros((N, 6))
    for j in range(N):
        Amat[j, A1_A2_vec[j]-1] = 1

    # -------------------------------------
    # 6. Y0 generation
    # -------------------------------------
    XA0 = np.column_stack([Xmat_out, Amat])
    lin_Y0 = alpha_Y0 + XA0 @ np.concatenate([beta_Y0, betaA_Y0])
    Y0 = np.random.binomial(1, prob_trunc(expit(lin_Y0)))

    # -------------------------------------
    # 7. Y1 generation via z_trt
    # -------------------------------------
    Y1 = np.zeros(N)
    conds = [
        (A1 == 1) & (A2 == 2) & (Y0 == 1),
        (A1 == 0) & (A2 == 2) & (Y0 == 1),
        (A1 == 1) & (A2 == 1) & (Y0 == 1),
        (A1 == 0) & (A2 == 1) & (Y0 == 1),
        (A1 == 1) & (A2 == 0) & (Y0 == 1),
        (A1 == 0) & (A2 == 0) & (Y0 == 1),
    ]
    z_indices = [5,4,3,2,1,0]

    prob_Y1 = np.zeros(N)
    for c, zi in zip(conds, z_indices):
        prob_Y1[c] = 1 - z_trt[zi]
    Y1 = np.random.binomial(1, prob_Y1)

    # -------------------------------------
    # 8. Treatment (D)
    # -------------------------------------
    if type_ == "pre":
        XA_D = np.column_stack([Xmat_ps, Amat])
        lin_or = alpha_D + XA_D @ np.concatenate([beta_D, betaA_D])
        lin_orOpp = alpha_DOpp + XA_D @ np.concatenate([beta_DOpp, betaA_DOpp])

    elif type_ == "post":
        new_data = pd.DataFrame({
            "A1": A1,
            "A2": A2,
            **{f"X_outcome.{k+1}": Xmat_out[:,k] for k in range(8)}
        })
        S_prob = rai.predict_proba(new_data)[:, 1]
        S = (S_prob >= cutoff).astype(int)

        XA_D = np.column_stack([Xmat_ps, Amat, S])
        lin_or = alpha_D + XA_D @ np.concatenate([beta_D, betaA_D, np.array([betaS_D])])
        lin_orOpp = alpha_DOpp + XA_D @ np.concatenate([beta_DOpp, betaA_DOpp, np.array([betaS_DOpp])])

    prob_D = prob_trunc(expit(lin_or))
    D = np.random.binomial(1, prob_D)

    Y = (1 - D) * Y0 + D * Y1

    # -------------------------------------
    # Build final DataFrame
    # -------------------------------------
    df = {
        "A1": A1.astype(str),
        "A2": A2.astype(str),
        "Y": Y,
        "D": D,
    }

    for j in range(Xmat_ps.shape[1]):
        df[f"X.{j+1}"] = Xmat_ps[:,j]

    for j in range(Xmat_out.shape[1]):
        df[f"X_outcome.{j+1}"] = Xmat_out[:,j]

    for j in range(Xmat_pa.shape[1]):
        df[f"X_pa.{j+1}"] = Xmat_pa[:,j]

    if type_ == "post":
        df["S"] = S
        df["S_prob"] = S_prob

    return pd.DataFrame(df)


# -------------------------------------
# Train RAI (Random Forest)
# -------------------------------------
data_train = data_gen_small(params_list, type_="pre", N=1000)
X_train = data_train[[col for col in data_train.columns if col.startswith("X_outcome.") or col in ["A1","A2"]]].astype(float)
y_train = data_train["Y"].astype(int)

rai = RandomForestClassifier(n_estimators=500, max_features=6, random_state=42)
rai.fit(X_train, y_train)

# -------------------------------------
# Generate estimation dataset (post)
# -------------------------------------
ex_data_small = data_gen_small(params_list, type_="post", N=5000, rai=rai, cutoff=0.5)

# -------------------------------------
# Save CSV
# -------------------------------------
ex_data_small.to_csv("ex_data_small.csv", index=False)
print("Saved ex_data_small.csv")
