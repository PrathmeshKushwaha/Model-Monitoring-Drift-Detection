from src.monitoring.rules import MONITORING_RULES


def evaluate_rules(ref_metrics, prod_metrics):
    results = []
    overall_status = "PASS"

    for metric, rule in MONITORING_RULES.items():
        ref = ref_metrics[metric]
        prod = prod_metrics[metric]
        delta = prod - ref

        status = "PASS"

        if "max_drop" in rule and delta < -rule["max_drop"]:
            status = "FAIL"

        if "max_increase" in rule and delta > rule["max_increase"]:
            status = "FAIL"

        if "max_relative_change" in rule:
            rel_change = abs(delta) / (abs(ref) + 1e-9)
            if rel_change > rule["max_relative_change"]:
                status = "FAIL"

        if status == "FAIL":
            overall_status = "FAIL"

        results.append({
            "metric": metric,
            "ref": ref,
            "prod": prod,
            "delta": delta,
            "status": status,
            "severity": rule["severity"]
        })

    return overall_status, results
