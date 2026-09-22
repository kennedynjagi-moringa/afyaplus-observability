"""
Dashboard smoke tests. These read the CSV outputs already produced by
Phases 1-3, so run `evaluation.run_evaluation`, `drift.run_drift_detection`,
and `cost.run_cost_analysis` at least once before running this test file.
"""

from fastapi.testclient import TestClient

from dashboard.app import app

client = TestClient(app)


def test_dashboard_page_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "AfyaPlus Executive Observability Dashboard" in response.text
    assert "System Health" in response.text.replace("&mdash;", "-")


def test_health_endpoint_reports_up():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UP"


def test_metrics_endpoint_is_prometheus_text():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "afyaplus_service_up" in response.text


def test_dashboard_json_has_all_four_sections():
    response = client.get("/api/dashboard")
    body = response.json()
    assert set(body.keys()) == {"health", "quality", "drift", "budget"}
    assert len(body["quality"]) > 0
