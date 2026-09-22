"""Server-rendered HTML for the AfyaPlus executive dashboard. Plain
string templating on purpose - the data is all internally generated
(no user input reaches this page), and it keeps the dependency list
short (no Jinja2 needed) for a small, single-page monitoring view."""

import html


def _status_pill(is_good: bool, good_text: str, bad_text: str) -> str:
    cls = "pill pill-good" if is_good else "pill pill-bad"
    text = good_text if is_good else bad_text
    return f'<span class="{cls}">{html.escape(text)}</span>'


def _progress_bar(ratio: float) -> str:
    pct = min(ratio, 1.5) * 100
    cls = "bar-fill"
    if ratio >= 1.0:
        cls += " bar-danger"
    elif ratio >= 0.8:
        cls += " bar-warning"
    return f'<div class="bar"><div class="{cls}" style="width:{min(pct, 100):.1f}%"></div></div>'


def render_dashboard(health: dict, quality_rows: list, drift: dict, budget: dict) -> str:
    checks_html = "".join(
        f"<li>{_status_pill(ok, 'OK', 'MISSING')} {html.escape(label)}</li>"
        for label, ok in health.get("checks", {}).items()
    )

    quality_html = "".join(
        f"""<tr>
            <td>{html.escape(row['feature'])}</td>
            <td>{html.escape(row['model'])}</td>
            <td>{row['overall_alignment']} / 5</td>
            <td>{row['correctness']} / 5</td>
            <td>{row['groundedness']} / 5</td>
            <td>{row['rouge_l']}</td>
            <td>{row['token_f1']}</td>
            <td>{_status_pill(row['quality_gate_pass'], 'PASS', 'FAIL')}</td>
            <td><strong>{html.escape(row['recommended_routing'])}</strong></td>
        </tr>"""
        for row in quality_rows
    )

    drift_html = "".join(
        f"""<tr>
            <td>{html.escape(col['column'])}</td>
            <td>{col['baseline_month1_mean']}</td>
            <td>{col['current_mean']}</td>
            <td>{col['p_value'] if col['p_value'] is not None else '-'}</td>
            <td>{_status_pill(not col['drifted'], 'STABLE', 'DRIFTED')}</td>
        </tr>"""
        for col in drift.get("columns", [])
    )

    budget_html = ""
    if budget:
        budget_html = f"""
        <div class="budget-row">
            <div class="budget-label">Daily spend</div>
            <div>${budget['daily_spend_usd']:.4f} / ${budget['daily_cap_usd']:.2f} cap</div>
            {_progress_bar(budget['daily_utilization'])}
        </div>
        <div class="budget-row">
            <div class="budget-label">30-day spend</div>
            <div>${budget['monthly_spend_usd']:.2f} / ${budget['monthly_cap_usd']:.2f} cap</div>
            {_progress_bar(budget['monthly_utilization'])}
        </div>
        """
    else:
        budget_html = "<p>Cost data not available yet. Run Phase 3.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="refresh" content="30" />
<title>AfyaPlus Executive Observability Dashboard</title>
<style>
  :root {{
    --bg: #0b1220; --card: #131c2e; --border: #24304a; --text: #e6ecf5;
    --muted: #93a3bf; --good: #1f9d55; --bad: #d64545; --warn: #d69e2e; --accent: #4f8cff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; }}
  .subtitle {{ color: var(--muted); margin: 0 0 28px 0; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; grid-column: span 1;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{ font-size: 15px; margin: 0 0 14px 0; color: var(--accent); letter-spacing: 0.03em; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  ul {{ list-style: none; padding: 0; margin: 0; font-size: 13px; }}
  li {{ padding: 6px 0; display: flex; gap: 8px; align-items: center; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
  .pill-good {{ background: rgba(31,157,85,0.18); color: #4fd88a; }}
  .pill-bad {{ background: rgba(214,69,69,0.18); color: #ff7a7a; }}
  .bar {{ background: #1e2940; border-radius: 6px; height: 10px; overflow: hidden; margin-top: 6px; }}
  .bar-fill {{ background: var(--good); height: 100%; }}
  .bar-fill.bar-warning {{ background: var(--warn); }}
  .bar-fill.bar-danger {{ background: var(--bad); }}
  .budget-row {{ margin-bottom: 18px; font-size: 13px; }}
  .budget-label {{ color: var(--muted); margin-bottom: 2px; }}
  .footer {{ margin-top: 24px; color: var(--muted); font-size: 12px; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
  <h1>AfyaPlus Executive Observability Dashboard</h1>
  <p class="subtitle">Live status as of {html.escape(health.get('checked_at', ''))} &middot; auto-refreshes every 30s &middot; <a href="/metrics">Prometheus /metrics</a></p>

  <div class="grid">
    <div class="card">
      <h2>System Health</h2>
      <p>Overall status: {_status_pill(health.get('status') == 'UP', 'UP', 'DOWN')}
         &nbsp;|&nbsp; Exceptions observed: <strong>{health.get('exception_count', 0)}</strong></p>
      <ul>{checks_html}</ul>
    </div>

    <div class="card">
      <h2>Budget Capital Utilisation</h2>
      {budget_html}
    </div>

    <div class="card full">
      <h2>Feature Quality Matrix</h2>
      <table>
        <thead><tr>
          <th>Feature</th><th>Model</th><th>Overall Alignment</th><th>Correctness</th>
          <th>Groundedness</th><th>ROUGE-L</th><th>Token F1</th><th>Quality Gate</th><th>Recommended Routing</th>
        </tr></thead>
        <tbody>{quality_html}</tbody>
      </table>
    </div>

    <div class="card full">
      <h2>Drift Vector Status &mdash; Month {drift.get('current_month', '-')} vs Month 1 Baseline</h2>
      <table>
        <thead><tr><th>Column</th><th>Month 1 Mean</th><th>Current Mean</th><th>p-value (K-S)</th><th>Status</th></tr></thead>
        <tbody>{drift_html}</tbody>
      </table>
    </div>
  </div>

  <p class="footer">AfyaPlus Week 3 Observability Capstone &middot; data sourced from /evaluation, /drift, /cost pipeline outputs.</p>
</body>
</html>"""
