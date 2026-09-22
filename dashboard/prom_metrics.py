"""
Prometheus metrics for the AfyaPlus dashboard, scraped at /metrics.
Gauges are refreshed from the latest data_loader output on every scrape,
which keeps this a valid pull-based exporter (no stale background thread).
"""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest

registry = CollectorRegistry()

service_up = Gauge(
    "afyaplus_service_up", "1 if all required observability outputs are present", registry=registry
)
exception_total = Gauge(
    "afyaplus_exception_total", "Count of data-loading exceptions observed", registry=registry
)
quality_score = Gauge(
    "afyaplus_quality_score",
    "Mean evaluation metric score per model/feature/metric",
    ["model", "feature", "metric_name"],
    registry=registry,
)
drift_p_value = Gauge(
    "afyaplus_drift_p_value",
    "K-S test p-value for the current month vs Month-1 baseline",
    ["column"],
    registry=registry,
)
drift_detected = Gauge(
    "afyaplus_drift_detected",
    "1 if drift detected (p < 0.05) for the current month, else 0",
    ["column"],
    registry=registry,
)
daily_cost_usd = Gauge("afyaplus_daily_cost_usd", "Simulated spend for the latest day", registry=registry)
monthly_cost_usd = Gauge(
    "afyaplus_monthly_cost_usd", "Simulated 30-day cumulative spend", registry=registry
)
daily_budget_utilization = Gauge(
    "afyaplus_daily_budget_utilization_ratio", "Daily spend / daily budget cap", registry=registry
)
monthly_budget_utilization = Gauge(
    "afyaplus_monthly_budget_utilization_ratio", "Monthly spend / monthly budget cap", registry=registry
)


def refresh(health: dict, quality_rows: list[dict], drift: dict, budget: dict) -> None:
    service_up.set(1 if health.get("status") == "UP" else 0)
    exception_total.set(health.get("exception_count", 0))

    quality_score.clear()
    for row in quality_rows:
        for metric_name in ("overall_alignment", "correctness", "groundedness", "rouge_l", "token_f1"):
            quality_score.labels(
                model=row["model"], feature=row["feature"], metric_name=metric_name
            ).set(row[metric_name])

    drift_p_value.clear()
    drift_detected.clear()
    for col in drift.get("columns", []):
        if col["p_value"] is not None:
            drift_p_value.labels(column=col["column"]).set(col["p_value"])
        drift_detected.labels(column=col["column"]).set(1 if col["drifted"] else 0)

    if budget:
        daily_cost_usd.set(budget["daily_spend_usd"])
        monthly_cost_usd.set(budget["monthly_spend_usd"])
        daily_budget_utilization.set(budget["daily_utilization"])
        monthly_budget_utilization.set(budget["monthly_utilization"])


def render_latest() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
