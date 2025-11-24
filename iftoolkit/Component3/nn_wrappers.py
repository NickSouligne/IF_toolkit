from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

@dataclass
class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for a Keras Sequential binary classifier.

    - build_fn: callable that returns a *compiled* tf.keras.Model
    - epochs, batch_size, verbose, callbacks, etc. are stored as params for clone()
    """

    build_fn: Callable[..., Any]
    build_kwargs: Optional[Dict[str, Any]] = None
    epochs: int = 25
    batch_size: int = 32
    verbose: int = 0
    random_state: int = 42

    def __post_init__(self):
        self.build_kwargs = dict(self.build_kwargs or {})
        self.model_ = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "build_fn": self.build_fn,
            "build_kwargs": self.build_kwargs,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, sample_weight=None):
        # Import TF lazily so base installs don’t require it
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.model_ = self.build_fn(**self.build_kwargs)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)

        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            **fit_kwargs
        )
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fit yet.")

        X = np.asarray(X, dtype=np.float32)
        p1 = self.model_.predict(X, verbose=0).reshape(-1)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T  # (n, 2)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
