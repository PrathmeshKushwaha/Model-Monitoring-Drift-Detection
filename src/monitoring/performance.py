import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

TARGET_COLUMN = "y"
LABEL_MAPPING = {"no": 0, "yes": 1}
POS_LABEL = 1


def precision_at_k(y_true, y_proba, k=0.1):
    cutoff = int(len(y_proba) * k)
    top_idx = np.argsort(y_proba)[-cutoff:]
    return precision_score(y_true[top_idx], [POS_LABEL] * cutoff, pos_label=POS_LABEL)


def expected_calibration_error(y_true, y_proba, bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.cut(df["p"], bins)

    ece = 0.0
    for _, g in df.groupby("bin"):
        if len(g) == 0:
            continue
        acc = (g["y"] == POS_LABEL).mean()
        conf = g["p"].mean()
        ece += abs(acc - conf) * len(g) / len(df)

    return ece


def evaluate_performance(model, df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(LABEL_MAPPING)

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return {
        "auc": roc_auc_score((y == POS_LABEL).astype(int), y_proba),
        "precision": precision_score(y, y_pred, pos_label=POS_LABEL),
        "recall": recall_score(y, y_pred, pos_label=POS_LABEL),
        "precision_at_10": precision_at_k(y.values, y_proba, k=0.1),
        "ece": expected_calibration_error(y.values, y_proba),
        "proba_mean": np.mean(y_proba),
    }
