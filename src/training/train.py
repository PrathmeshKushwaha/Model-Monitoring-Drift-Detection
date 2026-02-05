import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.utils.schema import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
from src.utils.data_checks import validate_data


DATA_DIR = Path("data")

def main():
    # Load data
    train_df = validate_data(DATA_DIR / "train.csv")
    test_df = validate_data(DATA_DIR / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = (train_df[TARGET_COLUMN] == "yes").astype(int)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = (test_df[TARGET_COLUMN] == "yes").astype(int)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    # Model
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("bank-marketing-baseline")

    with mlflow.start_run(run_name="logreg_v1"):
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
