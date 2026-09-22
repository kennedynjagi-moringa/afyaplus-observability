"""
Phase 4: FastAPI Web Dashboard.

Unified monitoring console for AfyaPlus, serving:
  - GET /                render the dashboard (4 mandatory sections)
  - GET /api/dashboard    same data as JSON
  - GET /metrics          Prometheus scrape endpoint
  - GET /health           lightweight liveness/health JSON

Run from the project root:
    .venv/bin/uvicorn dashboard.app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from dashboard import data_loader, prom_metrics
from dashboard.templates import render_dashboard

app = FastAPI(title="AfyaPlus Executive Observability Dashboard")


def _collect() -> dict:
    health = data_loader.get_system_health()
    quality_rows = data_loader.get_quality_matrix()
    drift = data_loader.get_drift_status()
    budget = data_loader.get_budget_status()
    prom_metrics.refresh(health, quality_rows, drift, budget)
    return {"health": health, "quality": quality_rows, "drift": drift, "budget": budget}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    data = _collect()
    return render_dashboard(data["health"], data["quality"], data["drift"], data["budget"])


@app.get("/api/dashboard")
def dashboard_json() -> dict:
    return _collect()


@app.get("/health")
def health() -> dict:
    return data_loader.get_system_health()


@app.get("/metrics")
def metrics() -> Response:
    _collect()
    body, content_type = prom_metrics.render_latest()
    return Response(content=body, media_type=content_type)
