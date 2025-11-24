import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def _make_component3_df(n=600, seed=0):
    rng = np.random.default_rng(seed)

    # Protected attributes
    A1 = rng.integers(0, 2, size=n)          # 0/1
    A2 = rng.choice([2, 3, 4, 5], size=n, p=[0.1, 0.15, 0.6, 0.15])  # multi-level

    # Numeric covariates
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    X3 = rng.normal(size=n)

    # Create outcome with intersectional signal
    A1A2 = A1.astype(str) + A2.astype(str)
    gamma_map = {"14": 0.6, "13": -0.5, "02": -0.2, "15": 0.25}
    gamma = np.array([gamma_map.get(g, 0.0) for g in A1A2])

    logits = -0.3 + 0.7 * X1 - 0.4 * X2 + 0.2 * X3 + gamma
    p = 1 / (1 + np.exp(-logits))
    Y = rng.binomial(1, p).astype(int)

    df = pd.DataFrame(
        {"A1": A1, "A2": A2, "Y": Y, "X1": X1, "X2": X2, "X3": X3}
    )
    return df


def test_pre_process_data_creates_intersection_and_covariates():
    import iftoolkit as ift

    df = _make_component3_df(n=200, seed=1)

    m = ift.Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=["X1", "X2", "X3"],
        outcome_estimator=LogisticRegression(max_iter=200),
        method="sr",
        n_splits=2,
        random_state=0,
    )

    m.pre_process_data()

    assert "A1A2" in m.data.columns
    assert m.A == "A1A2"
    assert set(m.covariates) == {"X1", "X2", "X3"}
    assert m.data[m.Y].dtype.name == "category"


def test_fit_fairness_sr_happy_path_no_null_no_bootstrap_fast():
    import iftoolkit as ift

    df = _make_component3_df(n=500, seed=2)

    m = ift.Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=["X1", "X2", "X3"],
        outcome_estimator=LogisticRegression(max_iter=300),
        method="sr",
        n_splits=3,
        random_state=0,
    )

    m.pre_process_data()

    res = m.fit_fairness(
        cutoff=0.5,
        gen_null=False,        # keep test fast/stable
        bootstrap="none",
    )

    assert isinstance(res, dict)
    assert "defs" in res
    assert "est_choice" in res

    # summarize() should return tidy stat/value df
    summ = m.summarize()
    assert set(summ.columns) == {"stat", "value"}
    assert len(summ) > 0


def test_fit_fairness_dr_autocompute_propensity_adds_group_probs():
    import iftoolkit as ift

    df = _make_component3_df(n=600, seed=3)

    m = ift.Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=["X1", "X2", "X3"],
        outcome_estimator=LogisticRegression(max_iter=300),
        # DR mode, let pipeline auto-fit propensities
        method="dr",
        auto_compute_propensity=True,
        n_splits=3,
        random_state=0,
        calibration_method="isotonic",
        calibration_cv=2,
    )

    m.pre_process_data()

    # Should not error; should auto-add group_<g>_prob columns
    res = m.fit_fairness(
        cutoff=0.5,
        gen_null=False,
        bootstrap="none",
    )

    assert "defs" in res

    # Verify propensity columns exist for all observed groups
    groups = m.data["A1A2"].astype(str).unique().tolist()
    for g in groups:
        col = f"group_{g}_prob"
        assert col in m.data.columns
        assert m.data[col].between(0, 1).all()


def test_fit_fairness_requires_preprocess():
    import iftoolkit as ift

    df = _make_component3_df(n=120, seed=4)

    m = ift.Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=["X1", "X2", "X3"],
        outcome_estimator=LogisticRegression(max_iter=200),
        method="sr",
        n_splits=2,
        random_state=0,
    )

    # No pre_process_data() call
    with pytest.raises(RuntimeError, match="pre_process_data"):
        m.fit_fairness(gen_null=False, bootstrap="none")


def test_pre_process_data_missing_columns_raises():
    import iftoolkit as ift

    df = _make_component3_df(n=80, seed=5).drop(columns=["A2"])

    m = ift.Model(
        data=df,
        outcome="Y",
        protected_characteristics=("A1", "A2"),
        covariates=["X1", "X2", "X3"],
        outcome_estimator=LogisticRegression(max_iter=200),
        method="sr",
        n_splits=2,
        random_state=0,
    )

    with pytest.raises(ValueError, match="A1 or A2 not found"):
        m.pre_process_data()


def test_ensure_probabilistic_estimator_wraps_nonproba_model():
    import iftoolkit as ift

    base = LinearSVC()  # no predict_proba
    wrapped = ift.cf_ensure_probabilistic_estimator(base, method="isotonic", cv=2)

    # Wrapped model should now have predict_proba
    assert hasattr(wrapped, "predict_proba")
