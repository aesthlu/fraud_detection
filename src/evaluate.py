import mlflow

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
import numpy as np

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)
    pr  = average_precision_score(y_test, probs)
    
    with mlflow.start_run():
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("pr_auc", pr)

    return roc, pr, probs

def business_threshold(y_true, probs, cost_fn=100, cost_fp=1):
    thresholds = np.linspace(0.01, 0.9, 100)
    costs = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        fn = ((y_true == 1) & (preds == 0)).sum()
        fp = ((y_true == 0) & (preds == 1)).sum()
        costs.append(fn * cost_fn + fp * cost_fp)

    return thresholds[np.argmin(costs)]