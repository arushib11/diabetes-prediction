"""Tests for the LLM interface (src/app.py)."""
import json
import pathlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.app import parse_features, run_inference, explain_prediction, KEY_MAP


# ── Helpers ───────────────────────────────────────────────────────────────────

def mock_client(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


FULL = {
    'pregnancies': 2, 'glucose': 148, 'blood_pressure': 72,
    'skin_thickness': 35, 'insulin': 0, 'bmi': 33.6,
    'diabetes_pedigree_function': 0.627, 'age': 50,
    'out_of_scope': False,
}

PARTIAL = {
    'pregnancies': 1, 'glucose': 120, 'blood_pressure': None,
    'skin_thickness': None, 'insulin': None, 'bmi': 28.5,
    'diabetes_pedigree_function': None, 'age': 38,
    'out_of_scope': False,
}

OUT_OF_SCOPE = {
    'pregnancies': None, 'glucose': None, 'blood_pressure': None,
    'skin_thickness': None, 'insulin': None, 'bmi': None,
    'diabetes_pedigree_function': None, 'age': None,
    'out_of_scope': True,
}


# ── parse_features ────────────────────────────────────────────────────────────

def test_parse_features_returns_all_keys():
    result = parse_features(mock_client(json.dumps(FULL)), 'any text')
    assert set(KEY_MAP.keys()).issubset(set(result.keys()))


def test_parse_features_out_of_scope_flagged():
    result = parse_features(mock_client(json.dumps(OUT_OF_SCOPE)), 'hello there')
    assert result.get('out_of_scope') is True


def test_parse_features_in_scope_not_flagged():
    result = parse_features(mock_client(json.dumps(FULL)), 'patient is 45, BMI 32')
    assert result.get('out_of_scope') is False


def test_parse_features_correct_values():
    result = parse_features(mock_client(json.dumps(FULL)), 'any text')
    assert result['glucose'] == 148
    assert result['age'] == 50


def test_parse_features_handles_nulls():
    result = parse_features(mock_client(json.dumps(PARTIAL)), 'any text')
    assert result['blood_pressure'] is None
    assert result['insulin'] is None


def test_parse_features_strips_markdown_fences():
    wrapped = '```json\n' + json.dumps(FULL) + '\n```'
    result = parse_features(mock_client(wrapped), 'any text')
    assert result['bmi'] == 33.6


# ── run_inference ─────────────────────────────────────────────────────────────

@pytest.fixture
def trained_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from src.preprocess import build_preprocessor, FEATURES

    X = pd.DataFrame([
        [2, 148, 72, 35, 0,   33.6, 0.627, 50],
        [1,  85, 66, 29, 0,   26.6, 0.351, 31],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        [8, 183, 64,  0, 0,   23.3, 0.672, 32],
    ], columns=FEATURES)
    y = [1, 0, 1, 1]
    pipe = Pipeline([('preprocessor', build_preprocessor()), ('classifier', LogisticRegression())])
    pipe.fit(X, y)
    return pipe


def test_run_inference_returns_valid_types(trained_pipeline):
    pred, prob = run_inference(trained_pipeline, FULL)
    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0


def test_run_inference_prob_matches_pred(trained_pipeline):
    pred, prob = run_inference(trained_pipeline, FULL)
    assert pred == int(prob >= 0.5)


def test_run_inference_handles_missing_values(trained_pipeline):
    pred, prob = run_inference(trained_pipeline, PARTIAL)
    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0


# ── explain_prediction ────────────────────────────────────────────────────────

def test_explain_prediction_returns_string():
    client = mock_client('This patient shows elevated risk.')
    result = explain_prediction(client, FULL, pred=1, prob=0.72)
    assert isinstance(result, str) and len(result) > 0


def test_explain_prediction_low_risk():
    client = mock_client('The patient appears to be at low risk.')
    result = explain_prediction(client, PARTIAL, pred=0, prob=0.28)
    assert isinstance(result, str) and len(result) > 0


def test_explain_prediction_calls_api_once():
    client = mock_client('Some explanation.')
    explain_prediction(client, FULL, pred=0, prob=0.3)
    client.chat.completions.create.assert_called_once()
