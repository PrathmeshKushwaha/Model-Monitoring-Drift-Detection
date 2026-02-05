from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

def split_feature_types(df: pd.DataFrame, target_col: str):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if target_col in numerical:
        numerical.remove(target_col)
    if target_col in categorical:
        categorical.remove(target_col)

    return numerical, categorical


def numerical_drift(reference: pd.Series, production: pd.Series, alpha=0.05):
    stat, p_value = ks_2samp(reference, production)
    drifted = p_value < alpha
    return drifted, p_value

def categorical_drift(reference: pd.Series, production: pd.Series, alpha=0.05):
    ref_counts = reference.value_counts()
    prod_counts = production.value_counts()

    combined = pd.concat([ref_counts, prod_counts], axis=1).fillna(0)
    chi2, p_value, _, _ = chi2_contingency(combined)

    drifted = p_value < alpha
    return drifted, p_value


def detect_feature_drift(
    reference_path: Path,
    production_path: Path,
    target_col: str,
    drift_threshold: float = 0.3
):
    ref_df = pd.read_csv(reference_path)
    prod_df = pd.read_csv(production_path)

    num_features, cat_features = split_feature_types(ref_df, target_col)

    drift_report = {}
    drifted_count = 0
    total_features = len(num_features) + len(cat_features)

    for col in num_features:
        drifted, p = numerical_drift(ref_df[col], prod_df[col])
        drift_report[col] = {"type": "numerical", "p_value": p, "drifted": drifted}
        if drifted:
            drifted_count += 1

    for col in cat_features:
        drifted, p = categorical_drift(ref_df[col], prod_df[col])
        drift_report[col] = {"type": "categorical", "p_value": p, "drifted": drifted}
        if drifted:
            drifted_count += 1

    drift_ratio = drifted_count / total_features
    overall_drift = drift_ratio >= drift_threshold

    return {
        "drift_ratio": drift_ratio,
        "drifted_features": drifted_count,
        "total_features": total_features,
        "overall_drift": overall_drift,
        "details": drift_report,
    }
