from matplotlib import pyplot as plt
from .intersectional_metrics import evaluate_intersectional_fairness
import numpy as np
import pandas as pd


#-----Example usage (synthetic)------

if __name__ == "__main__":
    '''
    #Create a small synthetic example to demonstrate usage
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(0),
        "income": rng.normal(60000, 15000, n),
        "feature_cat": rng.choice(["X", "Y", "Z"], size=n, p=[0.4, 0.4, 0.2]),
        "prot1": rng.choice(["M", "F"], size=n, p=[0.6, 0.4]),
        "prot2": rng.choice(["Wh", "Bl", "As"], size=n, p=[0.5, 0.3, 0.2]),
    })
    #Create a target with some bias structures
    logits = (
        -8.0
        + 0.05 * df["age"].values
        + 0.00003 * df["income"].values
        + (df["feature_cat"].values == "X").astype(int) * 0.8
        + (df["prot1"].values == "F").astype(int) * 0.6
        + (df["prot2"].values == "B").astype(int) * 0.5
    )
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    df["y"] = y

    '''

    #Load the synthetic data
    df = pd.read_csv("C:\\Users\\nicks\\Documents\\UA_Classes\\Python Code\\Clinical Data Generation\\synthetic_glaucoma_intervention.csv")

    #Define model parameters for lightGBM
    MODEL_PARAMS = {
        "objective": "binary",
        "n_estimators": 600,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced"  
    }   
    #Run the intersectional fairness evaluation
    res, figs, inter = evaluate_intersectional_fairness(
        df=df,
        outcome="glaucoma_intervention",
        protected_1="Race",
        protected_2="Gender",
        model_type="lgbm",
        model_params=MODEL_PARAMS,
        test_size=0.3,
        random_state=42,
        positive_label=1,
        threshold=0.5,
        require_class_balance=True, #Require at least one positive and negative in each group for metrics
        min_group_size=20,  #Minimum size of each intersectional group to be included
        make_plots=True,
        return_intermediates=True,
        return_non_intersectional=True
    )

    print("Model:", res.model)
    print("Demographic parity gap (max-min P(Ŷ=1)):", res.demographic_parity_gap)
    print("Equalized odds TPR gap:", res.equalized_odds_gap_tpr)
    print("Equalized odds FPR gap:", res.equalized_odds_gap_fpr)
    print("Equal opportunity gap (TPR gap):", res.equal_opportunity_gap)
    print("\nPer-group metrics:")
    print(res.per_group_df)
    print(inter["non_intersectional"]["Race"].per_group_df)
    print(inter["non_intersectional"]["Gender"].demographic_parity_gap)

    # Show plots when running as a script
    for name, fig in figs.items():
        fig.show()
        plt.show()