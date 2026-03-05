# Fairlogue — Component 2  
## Treatment-Aware Counterfactual Fairness

---

## Overview

Component 2 evaluates fairness in settings where **treatment, intervention, or exposure variables influence outcomes**.

It extends counterfactual fairness analysis by incorporating **causal structure**, enabling evaluation of disparities that arise through:

- Treatment assignment mechanisms
- Differential exposure pathways
- Mediated relationships between group membership and outcomes

It answers:

> Are disparities driven by treatment allocation mechanisms rather than direct model dependence on protected characteristics?

---

## Motivation

In many clinical and policy applications:

- Outcomes are influenced by treatments or interventions
- Treatment assignment may differ across demographic groups
- Observed disparities may reflect structural exposure differences

Component 2 provides a framework to separate:

- Direct model bias
- Treatment-driven disparities
- Structural covariate imbalance

---

## Conceptual Framework

Component 2 incorporates:

1. An outcome model  
2. A treatment (or exposure) model  
3. Counterfactual prediction under alternative treatment or group assignments  

This allows evaluation of:

- Counterfactual disparities under equalized treatment
- Mediated pathways of unfairness
- Disparities attributable to treatment mechanisms

---

## Core Quantities

Component 2 estimates:

- Counterfactual outcome probabilities under alternative treatment assignments
- Disparity contrasts across groups
- Mediation-aware fairness metrics

Depending on implementation, it may use:

- Inverse probability weighting
- Doubly robust estimation
- Structural modeling of treatment effects

---

## Basic Usage

```python
import iftoolkit as ift
from lightgbm import LGBMClassifier

outcome_model = LGBMClassifier(**OUTCOME_PARAMS)
treatment_model = LGBMClassifier(**TREATMENT_PARAMS)

m2 = ift.Component2.Model(
    data=df,
    outcome="target",
    treatment="treatment_variable",
    protected_characteristics=("race_factor", "gender_factor"),
    covariates=features,
    outcome_estimator=outcome_model,
    treatment_estimator=treatment_model,
    method="dr",
    random_state=42,
)

m2.pre_process_data()
m2.fit_fairness()

summary = m2.summarize()
estimates, deltas = m2.plots()
```

