# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Router training module for QueryRouter++.

description: Train routing classifiers (XGBoost, Random Forest, Logistic)
    with cross-validation and Optuna hyperparameter tuning.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class TrainingResult:
    """Results from a training run.

    Attributes:
        method: Algorithm used (xgboost, random_forest, logistic).
        accuracy: Mean cross-validation accuracy.
        f1_macro: Mean cross-validation F1 score (macro average).
        confusion_matrix: Confusion matrix from the final fold.
        best_params: Best hyperparameters found by Optuna.
        model_path: Path where the trained model was saved.
        cv_scores: Per-fold accuracy scores.
    """

    method: str
    accuracy: float
    f1_macro: float
    confusion_matrix: list[list[int]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    cv_scores: list[float] = field(default_factory=list)


class RouterTrainer:
    """Train routing classifiers with hyperparameter optimization.

    Supports XGBoost, Random Forest, and Logistic Regression methods.
    Uses Optuna for hyperparameter search with stratified k-fold
    cross-validation.

    Args:
        model_dir: Directory to save trained models.
        n_trials: Number of Optuna trials for hyperparameter search.
        n_folds: Number of cross-validation folds.

    Example:
        >>> trainer = RouterTrainer(model_dir=Path("models"))
        >>> result = trainer.train(X_train, y_train, method="xgboost")
        >>> result.accuracy
        0.85
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        n_trials: int = 20,
        n_folds: int = 5,
    ) -> None:
        """Initialize the trainer.

        Args:
            model_dir: Directory to save trained models. Created if needed.
            n_trials: Number of Optuna hyperparameter search trials.
            n_folds: Number of stratified k-fold splits.
        """
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.n_folds = n_folds

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: Literal["xgboost", "random_forest", "logistic"] = "xgboost",
    ) -> TrainingResult:
        """Train a routing classifier with hyperparameter optimization.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,) with integer class labels.
            method: Classification algorithm to use.

        Returns:
            TrainingResult with metrics, best params, and model path.
        """
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        def objective(trial: optuna.Trial) -> float:
            model = self._create_model(trial, method, n_classes=len(np.unique(y)))
            scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_trial.params
        best_model = self._build_model(best_params, method, n_classes=len(np.unique(y)))
        best_model.fit(X, y)

        # Final CV scores with best model
        cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(best_model, X, y, cv=skf, scoring="f1_macro")

        # Confusion matrix on full training set
        from sklearn.metrics import confusion_matrix

        y_pred = best_model.predict(X)
        cm = confusion_matrix(y, y_pred).tolist()

        # Save model
        import pickle

        model_path = self.model_dir / f"router_{method}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        # Save params
        params_path = self.model_dir / f"router_{method}_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

        return TrainingResult(
            method=method,
            accuracy=float(cv_scores.mean()),
            f1_macro=float(f1_scores.mean()),
            confusion_matrix=cm,
            best_params=best_params,
            model_path=str(model_path),
            cv_scores=cv_scores.tolist(),
        )

    # -- Hyperparameter search spaces per method --

    _SEARCH_SPACES: dict[str, dict[str, tuple]] = {
        "xgboost": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float", 0.01, 0.3, True),
            "subsample": ("float", 0.6, 1.0, False),
            "colsample_bytree": ("float", 0.6, 1.0, False),
        },
        "random_forest": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 20),
            "min_samples_split": ("int", 2, 10),
            "min_samples_leaf": ("int", 1, 5),
        },
        "logistic": {
            "C": ("float", 0.01, 100.0, True),
            "max_iter": ("int", 200, 1000),
        },
    }

    _DEFAULTS: dict[str, dict[str, Any]] = {
        "xgboost": {
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        "random_forest": {
            "n_estimators": 100, "max_depth": 10,
            "min_samples_split": 2, "min_samples_leaf": 1,
        },
        "logistic": {"C": 1.0, "max_iter": 500},
    }

    def _suggest_params(self, trial: Any, method: str) -> dict[str, Any]:
        """Use Optuna trial to suggest hyperparameters for the method."""
        params: dict[str, Any] = {}
        for key, spec in self._SEARCH_SPACES[method].items():
            if spec[0] == "int":
                params[key] = trial.suggest_int(key, spec[1], spec[2])
            else:
                params[key] = trial.suggest_float(key, spec[1], spec[2], log=spec[3])
        return params

    def _create_model(
        self, trial: Any, method: str, n_classes: int
    ) -> Any:
        """Create a model with Optuna-suggested hyperparameters."""
        params = self._suggest_params(trial, method)
        return self._build_model(params, method, n_classes)

    def _build_model(
        self, params: dict[str, Any], method: str, n_classes: int
    ) -> Any:
        """Build a classifier from a parameter dict.

        Args:
            params: Hyperparameter dict (may be partial — defaults fill gaps).
            method: Algorithm name.
            n_classes: Number of target classes.

        Returns:
            Scikit-learn compatible classifier.
        """
        defaults = self._DEFAULTS[method]
        p = {k: params.get(k, v) for k, v in defaults.items()}

        if method == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                **p,
                num_class=n_classes if n_classes > 2 else None,
                objective="multi:softmax" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42,
                verbosity=0,
            )

        if method == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**p, random_state=42)

        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            **p,
            solver="lbfgs",
            multi_class="multinomial" if n_classes > 2 else "auto",
            random_state=42,
        )
