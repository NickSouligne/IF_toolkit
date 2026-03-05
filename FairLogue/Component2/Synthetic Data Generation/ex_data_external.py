import numpy as np
import pandas as pd
from scipy.special import expit
from numpy.random import multivariate_normal

# -------------------------------
# Utility functions
# -------------------------------
def logit(x):
    return np.log(x / (1 - x))

def prob_trunc(p):
    return np.maximum(np.minimum(p, 0.995), 0.005)

# -------------------------------
# Parameters (translated from R)
# -------------------------------
params_list = {
    "mean_X_pa": np.array([1, -1, 0.5, -0.5, 1]),
    "A1_A2_means": np.array([
        [0.3, -0.05, 0,    0,    0,   -0.2],
        [-0.05, 0,   0.05, 0,    0.1,  0],
        [0.3,   0,  -0.1,  0.4, -0.05, -0.2],
        [-0.4, -0.1, 0,   -0.3,  0,    0.05],
        [0,     0,   0,    0,    0.1, -0.4]
    ])
}

# -------------------------------
# Data generator (R equivalent of data_gen_external)
# -------------------------------
def data_gen_external(params, N=10000):

    mean_X_pa = params["mean_X_pa"]
    A1_A2_means = params["A1_A2_means"]

    # Generate multivariate normal covariates
    p = len(mean_X_pa)
    Xmat_pa = multivariate_normal(mean=mean_X_pa, cov=np.eye(p) * 0.5, size=N)

    # Allocate result containers
    A1_A2_vec = np.zeros(N, dtype=int)
    pa_df = np.zeros((N, A1_A2_means.shape[1]))

    # Multinomial sampling for class membership (1..6)
    for j in range(N):
        p_vec = expit(Xmat_pa[j, :] @ A1_A2_means)
        p_vec = p_vec / p_vec.sum()

        pa_df[j, :] = p_vec
        A1_A2_vec[j] = np.argmax(np.random.multinomial(1, p_vec)) + 1  # classes 1..6

    # -------------------------
    # Create A1 and A2
    # -------------------------
    A1 = np.isin(A1_A2_vec, [2, 4, 6]).astype(int)

    A2 = np.select(
        [np.isin(A1_A2_vec, [1, 2]),
         np.isin(A1_A2_vec, [3, 4]),
         np.isin(A1_A2_vec, [5, 6])],
        [0, 1, 2]
    )

    # -------------------------
    # Build result as pandas DataFrame
    # -------------------------
    df = pd.DataFrame({
        "A1": A1.astype(str),   # mimic factor-as-character from R
        "A2": A2.astype(str)
    })

    # Append covariates
    for i in range(Xmat_pa.shape[1]):
        df[f"X_pa.{i+1}"] = Xmat_pa[:, i]

    return df


# -------------------------------
# Generate external data
# -------------------------------
ex_data_external = data_gen_external(params_list, N=10000)

# -------------------------------
# Save to CSV
# -------------------------------
ex_data_external.to_csv("ex_data_external.csv", index=False)

print("Saved ex_data_external.csv")
