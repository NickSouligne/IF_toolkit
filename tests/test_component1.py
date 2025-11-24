import numpy as np
import pandas as pd
import pytest


def _make_synth_df(n=200, seed=0):
    rng = np.random.default_rng(seed)

    # Protected attributes (2x2 intersection => 4 groups)
    race = rng.choice(["White", "Black"], size=n, p=[0.6, 0.4])
    sex = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])

    # Features: numeric + categorical
    age = rng.normal(60, 10, size=n)
    bmi = rng.normal(28, 4, size=n)
    smoker = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    region = rng.choice(["NE", "SW", "MW"], size=n)

    # Outcome depends a bit on features and group to create disparities
    logits = (
        -2.0
        + 0.03 * (age - 60)
        + 0.05 * (bmi - 28)
        + 0.8 * smoker
        + (race == "Black") * 0.4
        + (sex == "M") * 0.2
    )
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p, size=n)

    df = pd.DataFrame(
        {
            "Y": y,
            "race": race,
            "sex": sex,
            "age": age,
            "bmi": bmi,
            "smoker": smoker,
            "region": region,
        }
    )
    return df


def test_evaluate_intersectional_fairness_basic_run_no_plots():
    import iftoolkit as ift

    df = _make_synth_df(n=250, seed=1)

    results, figs = ift.ifm_evaluate_intersectional_fairness(
        df=df,
        outcome="Y",
        protected_1="race",
        protected_2="sex",
        features=None,           # uses all non-protected, non-outcome features
        model_type="logreg",
        test_size=0.3,
        random_state=42,
        threshold=0.5,
        make_plots=False,        # avoid matplotlib backend issues in CI
    )

    # Basic shape checks
    assert results.model == "logreg"
    assert isinstance(results.per_group_df, pd.DataFrame)
    assert len(results.groups) > 0

    # Gaps should be floats (may be nan if degenerate, but not error)
    assert isinstance(results.demographic_parity_gap, float)
    assert isinstance(results.equalized_odds_gap_tpr, float)
    assert isinstance(results.equalized_odds_gap_fpr, float)
    assert isinstance(results.equal_opportunity_gap, float)

    # With make_plots=False, figs should be empty dict
    assert isinstance(figs, dict)
    assert len(figs) == 0


def test_min_group_size_filters_small_groups_and_records_drops():
    import iftoolkit as ift

    df = _make_synth_df(n=180, seed=2)

    # Force one tiny group
    df.loc[:3, "race"] = "TinyRace"
    df.loc[:3, "sex"] = "TinySex"

    results, figs = ift.ifm_evaluate_intersectional_fairness(
        df=df,
        outcome="Y",
        protected_1="race",
        protected_2="sex",
        model_type="logreg",
        make_plots=False,
        min_group_size=10,   # should drop the tiny group pre-training
    )

    # The tiny intersection group should be listed as dropped
    assert any("TinyRace|TinySex" == g for g in results.dropped_groups)

    # Kept summary should have no rows below threshold
    assert (results.kept_groups_summary["n"] >= 10).all()


def test_require_class_balance_filters_single_class_groups_on_test_fold():
    import iftoolkit as ift

    df = _make_synth_df(n=220, seed=3)

    # Create a group that's nearly all positive to risk single-class in test
    mask = (df["race"] == "White") & (df["sex"] == "F")
    df.loc[mask, "Y"] = 1

    results, figs = ift.ifm_evaluate_intersectional_fairness(
        df=df,
        outcome="Y",
        protected_1="race",
        protected_2="sex",
        model_type="logreg",
        make_plots=False,
        require_class_balance=True,
        min_group_size=0,
    )

    # If filtered view is non-empty, then any kept group must have >=1 pos and >=1 neg
    per_group = results.per_group_df
    if not per_group.empty:
        assert (per_group["pos_true"] >= 1).all()
        assert (per_group["neg_true"] >= 1).all()


def test_all_nan_feature_is_dropped_not_crashing():
    import iftoolkit as ift

    df = _make_synth_df(n=160, seed=4)
    df["all_nan"] = np.nan

    results, figs = ift.ifm_evaluate_intersectional_fairness(
        df=df,
        outcome="Y",
        protected_1="race",
        protected_2="sex",
        features=None,
        model_type="logreg",
        make_plots=False,
    )

    assert results is not None
    assert "all_nan" not in results.per_group_df.columns  # not a strict guarantee, but sanity
    # Main point: function runs after dropping all-NaN columns.


def test_missing_required_columns_raises_keyerror():
    import iftoolkit as ift

    df = _make_synth_df(n=50, seed=5).drop(columns=["sex"])

    with pytest.raises(KeyError):
        ift.ifm_evaluate_intersectional_fairness(
            df=df,
            outcome="Y",
            protected_1="race",
            protected_2="sex",
            make_plots=False,
        )


def test_no_usable_features_raises_valueerror():
    import iftoolkit as ift

    df = _make_synth_df(n=60, seed=6)

    # If we pass features consisting only of protected/outcome cols, it should error
    with pytest.raises(ValueError):
        ift.ifm_evaluate_intersectional_fairness(
            df=df,
            outcome="Y",
            protected_1="race",
            protected_2="sex",
            features=["Y", "race", "sex"],
            make_plots=False,
        )
