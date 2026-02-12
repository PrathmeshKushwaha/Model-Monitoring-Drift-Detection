import numpy as np
import pandas as pd
from scipy.stats import entropy


def psi(expected, actual, bins=10):
    eps = 1e-6

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))

    expected_counts = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, breakpoints)[0] / len(actual)

    expected_counts = np.clip(expected_counts, eps, None)
    actual_counts = np.clip(actual_counts, eps, None)

    return np.sum((expected_counts - actual_counts) * np.log(expected_counts / actual_counts))


def js_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def evaluate_feature_drift(ref_df, prod_df, psi_threshold=0.2):
    drift_results = {}
    overall_status = "PASS"

    for col in ref_df.columns:
        if col not in prod_df.columns:
            continue

        if pd.api.types.is_numeric_dtype(ref_df[col]):
            score = psi(ref_df[col].dropna(), prod_df[col].dropna())
            status = "FAIL" if score > psi_threshold else "PASS"

        else:
            ref_dist = ref_df[col].value_counts(normalize=True)
            prod_dist = prod_df[col].value_counts(normalize=True)

            aligned = ref_dist.align(prod_dist, fill_value=0)
            score = js_divergence(aligned[0], aligned[1])
            status = "FAIL" if score > 0.1 else "PASS"

        drift_results[col] = {
            "score": round(float(score), 4),
            "status": status
        }

        if status == "FAIL":
            overall_status = "FAIL"

    return overall_status, drift_results
