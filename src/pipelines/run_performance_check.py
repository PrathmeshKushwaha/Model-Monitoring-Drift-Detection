import json
import pandas as pd
import mlflow
import mlflow.sklearn

from src.monitoring.performance import evaluate_performance
from src.monitoring.evaluator import evaluate_rules
from src.monitoring.data_drift import evaluate_data_drift
from src.monitoring.feature_drift import evaluate_feature_drift

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bank-marketing-monitoring")

MODEL_URI = "models:/bank-marketing@champion"
REFERENCE_PATH = "data/reference.csv"
PRODUCTION_PATH = "data/production/batch_feature_drift.csv"


def main():
    model = mlflow.sklearn.load_model(MODEL_URI)

    ref_df = pd.read_csv(REFERENCE_PATH)
    prod_df = pd.read_csv(PRODUCTION_PATH)

    # ---- DATA DRIFT ----
    drift_status, drift_metrics = evaluate_data_drift(ref_df, prod_df)

    # ---- PERFORMANCE ----
    ref_metrics = evaluate_performance(model, ref_df)
    prod_metrics = evaluate_performance(model, prod_df)

    perf_status, rule_results = evaluate_rules(ref_metrics, prod_metrics)

    feature_drift_status, feature_drift_metrics = evaluate_feature_drift(
    ref_df, prod_df)



    # ---- FINAL DECISION ----
    overall_status = "FAIL" if (
        drift_status == "FAIL" or perf_status == "FAIL"
    ) else "PASS"

    final_status = "FAIL" if (
    overall_status == "FAIL" or feature_drift_status == "FAIL"
    ) else "PASS"


    report = {
    "overall_status": overall_status,
    "performance": {
        "reference": ref_metrics,
        "production": prod_metrics
    },
    "performance_rules": rule_results,
    "feature_drift_status": feature_drift_status,
    "feature_drift_metrics": feature_drift_metrics
    }


    with mlflow.start_run(run_name="monitoring_check"):
        # Performance metrics
        for k, v in ref_metrics.items():
            mlflow.log_metric(f"ref_{k}", v)

        for k, v in prod_metrics.items():
            mlflow.log_metric(f"prod_{k}", v)

        for k in ref_metrics:
            mlflow.log_metric(f"delta_{k}", prod_metrics[k] - ref_metrics[k])

        # Drift metrics
        for k, v in drift_metrics.items():
            mlflow.log_metric(f"drift_{k}", v)

        # Status flags
        mlflow.log_param("performance_status", perf_status)
        mlflow.log_param("data_drift_status", drift_status)
        mlflow.log_param("overall_status", overall_status)

        # Save report
        mlflow.log_dict(report, "monitoring_report.json")

        mlflow.log_param("feature_drift_status", feature_drift_status)

        for feature, values in feature_drift_metrics.items():
            mlflow.log_metric(f"drift_{feature}", values["score"])


    print("\nMONITORING STATUS:", overall_status)
    print(f"\nMONITORING STATUS: {final_status}")

    if overall_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
