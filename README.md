# AfyaPlus Executive Observability Dashboard

Week 3 AI Engineering capstone. AfyaPlus has been running its clinical AI
assistant (triage routing, insurance verification, medication calculation
— see the [Week 1](../week-1/afya-plus-triage) and [Week 2](../week-2/afyaplus-rag-agent)
capstones) in production for six weeks with **zero automated evaluation, zero
drift monitoring, and zero API cost tracking**. This project builds all three
observability layers, a unified FastAPI dashboard on top of them, and a
one-page data-driven memo for the CTO and Medical Director.

## What's inside

| Path | Phase | What it does |
|---|---|---|
| [`evaluation/`](evaluation) | 1 | Scores gpt-4o-mini vs gpt-4o on a 15-question clinical dataset (BLEU / ROUGE-L / Token F1 + LLM-as-judge), applies a quality gate |
| [`drift/`](drift) | 2 | Simulates 3 months of production traffic and detects statistical drift with Evidently AI |
| [`cost/`](cost) | 3 | Simulates 30 days of traffic volume and projects spend, cost-per-request, and quality-gated routing savings |
| [`dashboard/`](dashboard) | 4 | FastAPI console unifying all of the above, plus a Prometheus `/metrics` endpoint |
| [`make_executive_summary.py`](make_executive_summary.py) | 5 | Builds `executive_summary.pdf` from the real numbers produced by phases 1-3 |
| [`common/`](common) | — | Shared OpenAI client + token counting used by phases 1-3 |
| [`tests/`](tests) | — | Offline unit tests for metrics, quality gates, pricing, and the dashboard API |

All numbers in the dashboard and the executive summary are read back from the
CSV files these scripts produce — nothing is hard-coded into the write-ups.

**Why "simulated" traffic?** There is no real 6-week production log to replay.
Phases 2 and 3 simulate traffic exactly as the brief asks ("simulate three
months of production traffic", "simulate 30 days of production payload
volumes") — Phase 2 makes real, small OpenAI calls with progressively
noisier patient messages so drift is genuinely measured, not faked; Phase 3
is a volume/pricing simulation using real average token counts measured in
Phase 1. Assumptions (base daily volumes, budget caps, staffing cost for the
human-review comparison) are documented in code comments where they're used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_EVAL_MODEL=gpt-4o-mini
OPENAI_PREMIUM_MODEL=gpt-4o
```

## Running it, in order

Each phase reads from earlier phases' outputs, so run them in this order
from the project root. Phases 1 and 2 make real (small, cheap) OpenAI API
calls; Phase 3, the dashboard, and the executive summary do not.

```bash
# Phase 1: clinical evaluation (30 OpenAI calls: 15 questions x 2 models)
.venv/bin/python -m evaluation.run_evaluation

# Phase 2: drift simulation + detection (90 OpenAI calls: 30 requests x 3 months)
.venv/bin/python -m drift.run_drift_detection

# Phase 3: cost & efficiency analysis (no API calls, uses Phase 1's token counts)
.venv/bin/python -m cost.run_cost_analysis

# Phase 4: dashboard (reads Phase 1-3 outputs)
.venv/bin/uvicorn dashboard.app:app --reload --port 8000
# then open http://localhost:8000  (Prometheus metrics at http://localhost:8000/metrics)

# Phase 5: executive summary PDF (reads Phase 1-3 outputs)
.venv/bin/python make_executive_summary.py
```

Run the offline test suite any time (no API key required beyond `.env`
existing so imports don't fail):

```bash
.venv/bin/python -m pytest
```

## Key outputs

- `evaluation/full_evaluation_results.csv`, `model_comparison_aggregation.csv`, `quality_gate_log.csv`
- `drift/month1_drift_report.html`, `month2_drift_report.html`, `month3_drift_report.html`, `drift_trend_table.csv`, `drift_alert_log.csv`
- `cost/cost_projection_30day.csv`, `cost_projection_summary.csv`, `cost_per_request_comparison.csv`, `structural_savings_analysis.csv`
- `executive_summary.pdf`
