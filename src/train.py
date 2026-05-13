"""
Standalone training script — mirrors notebooks/model_training.ipynb.
Trains 6 model configurations, logs all runs to MLflow, saves best_model.json.

Usage (from project root):
    python src/train.py
"""
import json
import pathlib
import sys
import warnings
import logging

warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')
logging.getLogger('mlflow.models.model').setLevel(logging.ERROR)
logging.getLogger('mlflow.sklearn').setLevel(logging.ERROR)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.preprocess import clean_data, build_preprocessor, FEATURES, TARGET
from src.evaluate import compute_metrics


def load_config() -> dict:
    with open(PROJECT_ROOT / 'configs' / 'config.yaml') as f:
        return yaml.safe_load(f)


def build_model_configs(config: dict) -> list:
    """Return list of (run_name, classifier, params_to_log) tuples."""
    lr  = config['models']['logistic_regression']
    rf  = config['models']['random_forest']
    xgb_cfg = config['models']['xgboost']
    return [
        ('logistic_regression',    LogisticRegression(**lr),                                      lr),
        ('logistic_regression_v2', LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs'),     {'C': 0.01, 'max_iter': 1000, 'solver': 'lbfgs'}),
        ('random_forest',          RandomForestClassifier(**rf),                                   rf),
        ('random_forest_v2',       RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42), {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 8, 'random_state': 42}),
        ('xgboost',                xgb.XGBClassifier(**xgb_cfg, eval_metric='logloss', verbosity=0), xgb_cfg),
        ('xgboost_v2',             xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0), {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8}),
    ]


def train_all(config: dict) -> dict:
    df_raw = pd.read_csv(PROJECT_ROOT / config['data']['raw_path'])
    df = clean_data(df_raw)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y,
    )
    print(f'Train: {len(X_train)} rows | Test: {len(X_test)} rows')

    track_uri = str(PROJECT_ROOT / config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(track_uri)
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    results = {}
    for run_name, classifier, params in build_model_configs(config):
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', classifier),
        ])
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag('data_description',
                f'Pima Indians Diabetes | {len(X_train)} train / {len(X_test)} test | '
                f'sentinel zeros replaced with NaN | rows missing Glucose or BMI dropped')
            mlflow.log_param('train_rows', len(X_train))
            mlflow.log_param('test_rows', len(X_test))
            mlflow.log_param('n_features', len(FEATURES))
            mlflow.log_param('imputation', 'median')
            mlflow.log_param('scaling', 'standard_scaler')
            mlflow.log_params(params)

            pipeline.fit(X_train, y_train)

            metrics = compute_metrics(pipeline, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name='model')

            run_id = run.info.run_id
            results[run_name] = {'run_id': run_id, 'metrics': metrics, 'pipeline': pipeline}
            print(f'  {run_name}: ROC-AUC={metrics["roc_auc"]:.4f}  F1={metrics["f1"]:.4f}')

    best_name = max(results, key=lambda n: results[n]['metrics']['roc_auc'])
    best = results[best_name]
    best_info = {
        'model_name':      best_name,
        'run_id':          best['run_id'],
        'metrics':         best['metrics'],
        'features':        FEATURES,
        'tracking_uri':    track_uri,
        'experiment_name': config['mlflow']['experiment_name'],
    }
    out = PROJECT_ROOT / 'configs' / 'best_model.json'
    with open(out, 'w') as f:
        json.dump(best_info, f, indent=2)
    print(f'\nBest model : {best_name}')
    print(f'ROC-AUC    : {best["metrics"]["roc_auc"]:.4f}')
    print(f'Saved to   : {out}')
    return best_info


if __name__ == '__main__':
    cfg = load_config()
    train_all(cfg)
