MONITORING_RULES = {
    "auc": {
        "max_drop": 0.02,
        "severity": "critical"
    },
    "precision_at_10": {
        "max_drop": 0.05,
        "severity": "business"
    },
    "recall": {
        "max_drop": 0.05,
        "severity": "warning"
    },
    "ece": {
        "max_increase": 0.02,
        "severity": "calibration"
    },
    "proba_mean": {
        "max_relative_change": 0.20,
        "severity": "data"
    }
}