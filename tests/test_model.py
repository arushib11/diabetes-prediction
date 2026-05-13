import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics
from src.preprocess import FEATURES, TARGET, build_preprocessor, clean_data

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'diabetes.csv'


@pytest.fixture(scope='module')
def trained_pipeline():
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def test_predict_returns_correct_length(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)


def test_predict_binary_output(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    preds = pipeline.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_predict_proba_sums_to_one(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    probs = pipeline.predict_proba(X_test)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_proba_positive_class_in_range(trained_pipeline):
    pipeline, X_test, _ = trained_pipeline
    probs = pipeline.predict_proba(X_test)[:, 1]
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_metrics_has_all_keys(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    metrics = compute_metrics(pipeline, X_test, y_test)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}


def test_roc_auc_above_random(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    metrics = compute_metrics(pipeline, X_test, y_test)
    assert metrics['roc_auc'] > 0.65, 'ROC-AUC should be well above random baseline'


def test_single_patient_prediction():
    """Model should accept a single-row DataFrame with all features."""
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    X, y = df[FEATURES], df[TARGET]
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)
    sample = X.iloc[[0]]
    prob = pipeline.predict_proba(sample)[0, 1]
    assert 0.0 <= prob <= 1.0
