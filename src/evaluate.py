from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(pipeline, X_test, y_test) -> dict:
    """Return the five standard classification metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        'accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred),    4),
        'f1':        round(f1_score(y_test, y_pred),        4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob),  4),
    }


def print_report(pipeline, X_test, y_test) -> None:
    """Print a full sklearn classification report."""
    y_pred = pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=['No Diabetes', 'Diabetes'],
    ))
