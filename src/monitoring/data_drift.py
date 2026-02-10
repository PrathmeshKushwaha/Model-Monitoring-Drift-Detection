import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.schema import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def calculate_psi(ref_series, prod_series, eps=1e-6):
    ref_dist = ref_series.value_counts(normalize=True)
    prod_dist = prod_series.value_counts(normalize=True)

    all_categories = set(ref_dist.index).union(set(prod_dist.index))

    psi = 0.0
    for cat in all_categories:
        ref_p = ref_dist.get(cat, eps)
        prod_p = prod_dist.get(cat, eps)
        psi += (prod_p - ref_p) * np.log(prod_p / ref_p)

    return psi


def evaluate_data_drift(ref_df: pd.DataFrame, prod_df: pd.DataFrame):
    drift_results = {}
    drift_flags = []

    # Numerical drift (KS test)
    for col in NUMERICAL_FEATURES:
        stat, p_value = ks_2samp(ref_df[col], prod_df[col])
        drift = p_value < KS_PVALUE_THRESHOLD

        drift_results[f"{col}_ks_pvalue"] = p_value
        drift_flags.append(drift)

    # Categorical drift (PSI)
    for col in CATEGORICAL_FEATURES:
        psi = calculate_psi(ref_df[col], prod_df[col])
        drift = psi > PSI_THRESHOLD

        drift_results[f"{col}_psi"] = psi
        drift_flags.append(drift)

    overall_status = "FAIL" if any(drift_flags) else "PASS"

    return overall_status, drift_results
