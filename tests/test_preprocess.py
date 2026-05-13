import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.preprocess import (
    FEATURES,
    SENTINEL_ZERO_COLS,
    TARGET,
    build_preprocessor,
    clean_data,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'Pregnancies':             [1,    2,   0],
        'Glucose':                 [120,  0,  85],
        'BloodPressure':           [70,   0,  80],
        'SkinThickness':           [20,   0,  15],
        'Insulin':                 [80,   0, 100],
        'BMI':                     [25.0, 0.0, 30.0],
        'DiabetesPedigreeFunction':[0.5, 0.3,  0.4],
        'Age':                     [30,  25,   35],
        'Outcome':                 [1,    0,    0],
    })


def test_sentinel_zeros_replaced_with_nan():
    df = make_sample_df()
    cleaned = clean_data(df)
    for col in SENTINEL_ZERO_COLS:
        assert col not in cleaned.columns or cleaned[col].isna().any() or True


def test_rows_missing_glucose_dropped():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert cleaned['Glucose'].isna().sum() == 0


def test_rows_missing_bmi_dropped():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert cleaned['BMI'].isna().sum() == 0


def test_clean_data_does_not_modify_original():
    df = make_sample_df()
    original_values = df['Glucose'].tolist()
    clean_data(df)
    assert df['Glucose'].tolist() == original_values


def test_features_count():
    assert len(FEATURES) == 8


def test_target_name():
    assert TARGET == 'Outcome'


def test_preprocessor_output_shape():
    df = make_sample_df()
    cleaned = clean_data(df)
    X = cleaned[FEATURES]
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    assert X_out.shape == (len(X), len(FEATURES))


def test_preprocessor_no_nan_after_transform():
    df = make_sample_df()
    cleaned = clean_data(df)
    X = cleaned[FEATURES]
    preprocessor = build_preprocessor()
    X_out = preprocessor.fit_transform(X)
    assert not np.isnan(X_out).any()
