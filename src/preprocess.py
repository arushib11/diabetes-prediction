import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
]
TARGET = 'Outcome'

# These columns use 0 as a sentinel for missing data (biologically impossible)
SENTINEL_ZERO_COLS = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel zeros with NaN; drop rows missing Glucose or BMI."""
    df = df.copy()
    df[SENTINEL_ZERO_COLS] = df[SENTINEL_ZERO_COLS].replace(0, np.nan)
    df = df.dropna(subset=['Glucose', 'BMI'])
    return df


def build_preprocessor() -> Pipeline:
    """Median imputation then standard scaling — returned unfitted."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
