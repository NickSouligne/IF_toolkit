import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .model import Model
from .Fairness import FairnessPipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# ------------- 1) Synthetic data generator -------------
def generate_synthetic_df(
    n: int = 20000,
    seed: int = 123,
    p_features: int = 10,
    gender_probs=(0.48, 0.52),            #(Female=0, Male=1)
    race_vals=(2, 3, 4, 5),               #Asian=2, Black=3, White=4, Hispanic=5
    race_probs=(0.08, 0.12, 0.6, 0.2)     #Define probabilities of each race assignment
) -> pd.DataFrame:
    """
    Create a dataset with:
      - A1 in {0,1} (Female=0, Male=1)
      - A2 in {2,3,4,5} (Asian, Black, White, Hispanic)
      - X1..Xp numeric features with correlation
      - Y ~ Bernoulli(sigmoid(β·X + γ_group))
    Includes true group effects so fairness metrics show signal.
    """
    rng = np.random.default_rng(seed)

    #Protected attributes
    A1 = rng.choice([0, 1], size=n, p=list(gender_probs)) #Gender
    A2 = rng.choice(race_vals, size=n, p=list(race_probs)) #Race

    #Correlated feature matrix
    #start with standard normals then inject correlation
    X = rng.normal(size=(n, p_features))
    #add shared latent to induce correlation
    latent = rng.normal(size=(n, 1))
    X = 0.7 * X + 0.3 * latent

    cols = [f"X{i+1}" for i in range(p_features)]
    df = pd.DataFrame(X, columns=cols)

    #Base linear predictor from features
    beta = rng.normal(0, 0.6, size=p_features)
    lp = X @ beta

    #Intersectional group effects (γ) to induce unfairness patterns.
    #Codes for A1A2 strings "04", "14", etc. (A1=str 0/1, A2=str 2/3/4/5)
    A1A2 = (A1.astype(str) + A2.astype(str))

    #Define a few non-zero group shifts; others default to 0.
    gamma_map = {
        "14":  0.60,  #Male/White: higher log-odds
        "13": -0.50,  #Male/Black: lower log-odds
        "02": -0.20,  #Female/Asian: slightly lower
        "15":  0.25,  #Male/Hispanic: moderately higher
    }
    gamma = np.array([gamma_map.get(g, 0.0) for g in A1A2])

    #Nonlinear feature interaction to avoid purely linear separation
    nl = 0.35 * np.tanh(X[:, 0] * X[:, 1])

    #Final logit
    logit = -0.25 + lp + gamma + nl
    p = 1.0 / (1.0 + np.exp(-logit))

    Y = rng.binomial(1, p).astype(int)

    #Assemble DF with protected attributes
    df["A1"] = A1
    df["A2"] = A2
    df["Y"]  = Y

    return df


# ------------- 2) (Optional) add π_g(X) for DR mode -------------
def add_group_propensities(
    df: pd.DataFrame,
    covariates: list,
    group_col: str = "A1A2",
    *,
    categorical_cols: list = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    class_weight = "balanced",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit a LightGBM multiclass model π_g(X)=P(A=g|X) on the covariates and
    write columns:  group_<g>_prob  for every group label g in df[group_col].
    """
    out = df.copy()

    y = out[group_col].astype(str).values
    X = out[covariates]

    #(Optional) handle categorical features
    cat_feats = None
    if categorical_cols:
        #Ensure provided categorical columns are a subset of covariates
        cat_feats = [c for c in categorical_cols if c in covariates]

    #Define LGBM multiclass classifier
    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )

    #Fit lgbm model and predict outcomes
    clf.fit(X, y, categorical_feature=cat_feats)
    probs = clf.predict_proba(X) 
    classes = clf.classes_.astype(str)

    #Write one column per group label
    for j, g in enumerate(classes):
        out[f"group_{g}_prob"] = probs[:, j]

    return out


# ------------- 3) Run demo through the Model facade -------------
if __name__ == "__main__":
    #Load data
    df = pd.read_csv("iftoolkit\\Component3\\glaucoma_synth_component3.csv")

    #Covariates = everything except protected + outcome
    df["A1A2"] = df.get("A1A2", df["A1"].astype(str) + df["A2"].astype(str)).astype(str)
    protected = {"A1", "A2", "A1A2"}
    covariates = [c for c in df.columns if c not in protected | {"Y"}]

    #Outcome model (binary)
    lgbm_outcome = LGBMClassifier(
        n_estimators=700,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        verbosity=-1,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42
    )

    #------ single-robust -----
    #Create the Model class instance
    m_sr = Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=covariates,
        outcome_estimator=lgbm_outcome,   #LightGBM binary
        method="sr",
        n_splits=5,
        random_state=42
    )

    #Pre-process data
    m_sr.pre_process_data()
    
    #Fit the fairness pipeline
    res_plugin = m_sr.fit_fairness(
        cutoff=.5,       #If none, cutoff chosen via Youden threshold
        gen_null=True,     #enable permutation-null to get U-values
        R_null=200,
        bootstrap="rescaled"
    )

    #Report model info, point estimates, and u-values
    print("\n[m_sr | LGBM] Model info")
    print(m_sr.get_model_info())

    print("\n[m_sr | LGBM] Point estimates")
    print(m_sr.summarize().sort_values("stat").to_string(index=False))

    _, _, uvals_plugin = m_sr.plots(alpha=0.05, delta_uval=0.10)
    print("\n[PLUGIN | LGBM] U-values (aggregate)")
    print(uvals_plugin.to_string(index=False))


    #----------------- DR (AIPW doubly-robust) ------------------
    n_groups = df["A1A2"].nunique()

    #Propensity model — only needed for DR
    lgbm_propensity = LGBMClassifier(
        n_estimators=700,
        max_depth=-1,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        objective="multiclass",
        num_class=n_groups,  #number of intersectional groups
        random_state=42
    )

    #Uses the same LightGBM outcome model; for propensities we pass the multiclass LGBM
    m_dr = Model(
        data=df.copy(),
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=covariates,
        outcome_estimator=lgbm_outcome,       #LightGBM binary
        propensity_estimator=lgbm_propensity, #LightGBM multiclass
        method="dr",
        auto_compute_propensity=True,         #will fit π_g(X) = P(A=g|X) if missing
        n_splits=5,
        random_state=42
    )
    m_dr.pre_process_data()

    res_dr = m_dr.fit_fairness(
        cutoff=.5,
        gen_null=True,
        R_null=200,
        bootstrap="rescaled"
    )

    #Report model info, point estimates, and u-values
    print("\n[DR | LGBM] Model info")
    print(m_dr.get_model_info())

    print("\n[DR | LGBM] Point estimates")
    print(m_dr.summarize().sort_values("stat").to_string(index=False))

    _, _, uvals_dr = m_dr.plots(alpha=0.05, delta_uval=0.10)
    print("\n[DR | LGBM] U-values (aggregate)")
    print(uvals_dr.to_string(index=False))