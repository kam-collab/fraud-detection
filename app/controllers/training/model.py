"""The deployable model artefact: preprocessing + classifier + calibration + decision policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from app.controllers.features.builder import CATEGORICAL, FEATURES, NUMERIC
from app.utils import config

APPROVE, REVIEW, BLOCK = "approve", "review", "block"


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_logreg(class_weight=None, C: float = 1.0) -> Pipeline:
    """Baseline. Linear models need imputation, scaling and one-hot encoding."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(_signed_log)),  # tame heavy-tailed counts and sums
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
        ]
    )
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_gbm(class_weight=None, learning_rate: float = 0.05, max_iter: int = 300) -> Pipeline:
    """Tree model. Missing values and categoricals are handled natively, no scaling needed."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
    pre = ColumnTransformer([("cat", encoder, CATEGORICAL), ("num", "passthrough", NUMERIC)])
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=list(range(len(CATEGORICAL))),
        class_weight=class_weight,
        early_stopping=False,
        random_state=config.SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


@dataclass
class FraudModel:
    pipeline: Pipeline
    calib_a: float = 1.0  # Platt scaling on the logit: strictly monotone,
    calib_b: float = 0.0  # so it never changes the ranking
    review_threshold: float = 0.5
    block_threshold: float = 0.9
    version: str = "0.0.0"
    reference: dict = field(default_factory=dict)  # typical feature values, for explanations
    features: list = field(default_factory=lambda: list(FEATURES))

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probability."""
        return 1 / (1 + np.exp(-(self.calib_a * _logit(self.raw_score(X)) + self.calib_b)))

    def decide(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score)
        return np.where(score >= self.block_threshold, BLOCK, np.where(score >= self.review_threshold, REVIEW, APPROVE))

    def contributions(self, x: pd.DataFrame, top: int = 5) -> list[dict]:
        """Why did this one transaction get its score?

        Occlusion attribution: replace one feature at a time with its typical training
        value and measure how much the log-odds drop. Simple and model-agnostic, but it
        ignores feature interactions, so treat it as a guide for analysts, not an audit.
        """
        row = x[self.features].iloc[[0]]
        variants = pd.concat([row] * (len(self.features) + 1), ignore_index=True)
        for i, feat in enumerate(self.features, start=1):
            variants.loc[i, feat] = self.reference.get(feat, np.nan)
        variants = variants.astype(row.dtypes.to_dict(), errors="ignore")
        logits = self.calib_a * _logit(self.raw_score(variants)) + self.calib_b
        deltas = logits[0] - logits[1:]
        order = np.argsort(-deltas)[:top]
        out = []
        for i in order:
            if deltas[i] <= 0.05:
                break
            value = row.iloc[0][self.features[i]]
            out.append(
                {
                    "feature": self.features[i],
                    "value": None if pd.isna(value) else (value if isinstance(value, str) else float(value)),
                    "typical": self.reference.get(self.features[i]),
                    "log_odds_contribution": round(float(deltas[i]), 3),
                }
            )
        return out
