import numpy as np
import pandas as pd
import pytest


def _make_component2_df(n=200, seed=0):
    rng = np.random.default_rng(seed)

    # Protected attrs (binary for simplicity)
    A1 = rng.integers(0, 2, size=n)
    A2 = rng.integers(0, 2, size=n)

    # Covariates for pi model
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)

    # Treatment D depends on covariates a bit
    pD = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    D = rng.binomial(1, pD, size=n)

    # Sampling prob column (continuous)
    S_prob = np.clip(0.2 + 0.6 * rng.random(n), 1e-3, 0.999)

    # Outcome depends on D, covars, and intersection to yield disparities
    logits = (
        -1.5
        + 0.8 * D
        + 0.4 * x1
        - 0.3 * x2
        + 0.3 * (A1 == 1)
        + 0.2 * (A2 == 1)
        + 0.2 * ((A1 == 1) & (A2 == 1))
    )
    pY = 1 / (1 + np.exp(-logits))
    Y = rng.binomial(1, pY, size=n)

    return pd.DataFrame(
        {
            "A1": A1,
            "A2": A2,
            "Y": Y,
            "D": D,
            "S_prob": S_prob,
            "x1": x1,
            "x2": x2,
        }
    )


def _make_component2_df_balanced(n_per_cell=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            n = n_per_cell
            x1 = rng.normal(0, 1, size=n)
            x2 = rng.normal(0, 1, size=n)

            # Force D to have both classes with good margin
            D = rng.binomial(1, 0.5, size=n)

            # Make Y depend on D and covars but keep probabilities away from 0/1
            logits = -0.2 + 0.6 * D + 0.2 * x1 - 0.2 * x2 + 0.1 * a1 + 0.1 * a2
            pY = 1 / (1 + np.exp(-logits))
            pY = np.clip(pY, 0.1, 0.9)  # avoid near-deterministic outcomes
            Y = rng.binomial(1, pY, size=n)

            S_prob = np.clip(0.2 + 0.6 * rng.random(n), 1e-3, 0.999)

            rows.append(pd.DataFrame({
                "A1": np.full(n, a1),
                "A2": np.full(n, a2),
                "Y": Y,
                "D": D,
                "S_prob": S_prob,
                "x1": x1,
                "x2": x2,
            }))

    return pd.concat(rows, ignore_index=True)


def test_analysis_estimation_standard_happy_path_fast():
    import iftoolkit as ift

    data = _make_component2_df(n=250, seed=1)

    out = ift.cft_analysis_estimation(
        data=data,
        cutoff=0.5,
        estimator_type="standard",
        gen_null=False,
        bootstrap="none",
        pi_model_type="glm",
        pi_model_seed=[1],   # accepted for glm
        pi_xvars=["x1", "x2"],
    )

    assert isinstance(out, dict)
    assert "defs" in out
    assert "est_choice" in out

    # defs should be a dict of named metrics to arrays
    assert isinstance(out["defs"], dict)
    assert len(out["defs"]) > 0

    # est_choice should include Y0est and pi columns
    est_choice = out["est_choice"]
    assert "Y0est" in est_choice.columns
    assert "pi" in est_choice.columns
    assert len(est_choice) == len(data)


def test_missing_required_columns_raises_valueerror():
    import iftoolkit as ift

    data = _make_component2_df(n=50, seed=2).drop(columns=["S_prob"])

    with pytest.raises(ValueError, match="must contain columns"):
        ift.cft_analysis_estimation(
            data=data,
            cutoff=0.5,
            pi_model_seed=[1],
            pi_xvars=["x1", "x2"],
        )


def test_nonbinary_Y_or_D_raises_valueerror():
    import iftoolkit as ift

    data = _make_component2_df(n=60, seed=3)
    data.loc[:5, "Y"] = 2  # introduce 3rd class

    with pytest.raises(ValueError, match="exactly two unique values"):
        ift.cft_analysis_estimation(
            data=data,
            cutoff=0.5,
            pi_model_seed=[1],
            pi_xvars=["x1", "x2"],
        )


def test_missing_pi_xvars_raises_valueerror():
    import iftoolkit as ift

    data = _make_component2_df(n=80, seed=4).drop(columns=["x2"])

    with pytest.raises(ValueError, match="propensity score model covariates"):
        ift.cft_analysis_estimation(
            data=data,
            cutoff=0.5,
            pi_model_seed=[1],
            pi_xvars=["x1", "x2"],  # x2 missing
        )


def test_small_internal_missing_arguments_raises_valueerror():
    import iftoolkit as ift

    data = _make_component2_df(n=120, seed=5)

    with pytest.raises(ValueError, match="must be specified"):
        ift.cft_analysis_estimation(
            data=data,
            cutoff=0.5,
            estimator_type="small_internal",
            pi_model_seed=[1],
            pi_xvars=["x1", "x2"],
            # missing outcome_xvars, outcome_model_type, pa_xvars_int, fit_method_int, nfolds
        )


def test_small_borrow_missing_external_requirements_raises_valueerror():
    import iftoolkit as ift

    data = _make_component2_df(n=120, seed=6)

    with pytest.raises(ValueError, match="must be specified if 'estimator_type' is 'small_borrow'"):
        ift.cft_analysis_estimation(
            data=data,
            cutoff=0.5,
            estimator_type="small_borrow",
            pi_model_seed=[1],
            pi_xvars=["x1", "x2"],
            # missing data_external or pa_model_ext, plus borrow args
            outcome_model_type="glm",
            outcome_xvars=["x1", "x2"],
            pa_xvars_int=["x1", "x2"],
            fit_method_int="glm",
            nfolds=2,
        )


def test_gen_null_adds_table_null_key_fast():
    import iftoolkit as ift

    data = _make_component2_df(n=180, seed=7)

    out = ift.cft_analysis_estimation(
        data=data,
        cutoff=0.5,
        estimator_type="standard",
        gen_null=True,
        R_null=5,           # keep test fast
        bootstrap="none",
        pi_model_seed=[1],
        pi_xvars=["x1", "x2"],
    )

    assert "table_null" in out
    assert out["table_null"] is not None


def test_bootstrap_rescaled_adds_boot_out_key_fast():
    import iftoolkit as ift

    data = _make_component2_df_balanced(n_per_cell=120, seed=8)  # 480 rows total

    out = ift.cft_analysis_estimation(
        data=data,
        cutoff=0.5,
        estimator_type="standard",
        gen_null=False,
        bootstrap="rescaled",
        B=10,               # still fast, less chance of degeneracy
        m_factor=0.75,
        pi_model_seed=[1],
        pi_xvars=["x1", "x2"],
    )

    assert "boot_out" in out
    assert out["boot_out"] is not None

