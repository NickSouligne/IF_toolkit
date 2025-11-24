# Intersectional Fairness Framework

This package provides a **generalized, estimator-agnostic framework** for assessing
**intersectional fairness** in predictive models. It extends the counterfactual fairness
work of Wastvedt et al. (2024, *Biostatistics*) with support for:

- Any sklearn-compatible model (Random Forest, Logistic Regression, LightGBM, SVM, etc.)
- Intersectional protected groups (e.g., Gender × Race)
- **Single-robust (SR)** plugin estimators and **Doubly-robust (DR)** AIPW estimators
- Cross-fitted outcome models with counterfactual predictions (`muY_<group>`)
- Groupwise and summary disparity metrics (cFPR, cFNR, avg/max/var diffs)
- Permutation-based null distributions and rescaled bootstrap CIs
- U-values for hypothesis-style fairness testing
- Built-in visualization utilities for aggregate and groupwise results

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/fairness-framework.git
cd fairness-framework
pip install -r requirements.txt
```

Repository Structure

fairness-framework/
│
├── model.py               # High-level Model facade (user-facing entrypoint)
├── helpers.py             # Utility functions & estimator protocols
├── outcome_models.py      # Cross-fitted outcome models + groupwise rate functions
├── estimation_functions.py# Estimation orchestrators, nulls, bootstrap, CIs
├── plots.py               # Visualization & reporting
├── synthetic_test.py      # Example driver script with synthetic data
│
└── docs/
    ├── model.md           # Long-form docs for model.py
    ├── helpers.md         # Long-form docs for helpers.py
    ├── outcome_models.md  # Long-form docs for outcome_models.py
    ├── estimation_functions.md # Long-form docs for estimation_functions.py
    └── plots.md           # Long-form docs for plots.py




Quick Start
```python
import pandas as pd
from model import Model
from lightgbm import LGBMClassifier

# Load data
df = pd.read_csv("synthetic_glaucoma_intervention.csv")
df = df.rename(columns={"Race": "A2", "Gender": "A1", "glaucoma_intervention": "Y"})
df["Y"] = df["Y"].astype(int)

# Define covariates (all except protected + outcome)
covariates = [c for c in df.columns if c not in {"A1", "A2", "Y"}]

# Define outcome model
lgbm_outcome = LGBMClassifier(
    n_estimators=700, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    num_leaves=64, random_state=42
)

# Single-robust analysis
m_sr = Model(
    data=df,
    outcome="Y",
    protected_characteristics=("A1", "A2"),
    covariates=covariates,
    outcome_estimator=lgbm_outcome,
    method="sr"
)
m_sr.pre_process_data()
results = m_sr.fit_fairness()

print(m_sr.get_model_info())
print(m_sr.summarize().sort_values("stat"))

# U-values and plots
_, _, uvals = m_sr.plots(delta_uval=0.05, u_mode="two_sided")
print("U-values:", uvals.to_dict(orient="records")[0])
```

