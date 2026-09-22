"""
Phase 3: Token Cost & Efficiency Analysis.

Combines:
  - real average token counts measured in Phase 1 (evaluation/full_evaluation_results.csv)
  - a simulated 30-day, 75% gpt-4o-mini / 25% gpt-4o traffic volume
  - real published OpenAI pricing (cost/pricing.py)
  - Phase 1's quality gate outcomes (evaluation/quality_gate_log.csv)

to produce cost projections and a "route to the cheapest model that still
clears the clinical quality gate" savings analysis.

Outputs (written into cost/):
  - cost_projection_30day.csv          (day x feature x model spend)
  - cost_projection_summary.csv        (spend rolled up by model, and by feature)
  - cost_per_request_comparison.csv    (gpt-4o-mini vs gpt-4o, per feature)
  - structural_savings_analysis.csv    (savings from quality-gated model routing)

Run from the project root:
    .venv/bin/python -m cost.run_cost_analysis
"""

import os

import pandas as pd

from cost.pricing import request_cost_usd
from cost.volume_simulator import simulate_volume

OUTPUT_DIR = os.path.dirname(__file__)
EVALUATION_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "evaluation")


def load_avg_tokens() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(EVALUATION_DIR, "full_evaluation_results.csv"))
    return (
        df.groupby(["model", "feature"])[["input_tokens", "output_tokens"]]
        .mean()
        .reset_index()
        .rename(columns={"input_tokens": "avg_input_tokens", "output_tokens": "avg_output_tokens"})
    )


def load_quality_gate() -> pd.DataFrame:
    return pd.read_csv(os.path.join(EVALUATION_DIR, "quality_gate_log.csv"))


def run() -> None:
    avg_tokens = load_avg_tokens()
    quality_gate = load_quality_gate()
    volume = simulate_volume()

    # ---- 30-day cost projection ----
    projection = volume.merge(avg_tokens, on=["model", "feature"], how="left")
    projection["cost_usd"] = projection.apply(
        lambda r: request_cost_usd(r["model"], r["avg_input_tokens"], r["avg_output_tokens"])
        * r["requests"],
        axis=1,
    ).round(4)

    projection_path = os.path.join(OUTPUT_DIR, "cost_projection_30day.csv")
    projection.to_csv(projection_path, index=False)
    print(f"Saved 30-day cost projection -> {projection_path}")

    by_model = (
        projection.groupby("model")[["requests", "cost_usd"]]
        .sum()
        .reset_index()
        .assign(group_type="model")
        .rename(columns={"model": "group_value"})
    )
    by_feature = (
        projection.groupby("feature")[["requests", "cost_usd"]]
        .sum()
        .reset_index()
        .assign(group_type="feature")
        .rename(columns={"feature": "group_value"})
    )
    summary = pd.concat([by_model, by_feature], ignore_index=True)[
        ["group_type", "group_value", "requests", "cost_usd"]
    ]
    summary["cost_usd"] = summary["cost_usd"].round(2)
    summary_path = os.path.join(OUTPUT_DIR, "cost_projection_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved cost projection summary -> {summary_path}")

    # ---- Cost-per-request comparison ----
    per_request = avg_tokens.copy()
    per_request["cost_per_request_usd"] = per_request.apply(
        lambda r: request_cost_usd(r["model"], r["avg_input_tokens"], r["avg_output_tokens"]),
        axis=1,
    ).round(6)
    comparison = per_request.pivot(index="feature", columns="model", values="cost_per_request_usd")
    comparison["gpt-4o_premium_multiplier"] = (
        comparison["gpt-4o"] / comparison["gpt-4o-mini"]
    ).round(1)
    comparison_path = os.path.join(OUTPUT_DIR, "cost_per_request_comparison.csv")
    comparison.reset_index().to_csv(comparison_path, index=False)
    print(f"Saved cost-per-request comparison -> {comparison_path}")

    # ---- Structural savings analysis ----
    # For each feature: what's the cheapest model that clears Phase 1's quality gate?
    savings_rows = []
    for feature in avg_tokens["feature"].unique():
        feature_gate = quality_gate[quality_gate["feature"] == feature]
        passing_models = feature_gate[feature_gate["overall_pass"]]["model"].tolist()

        feature_costs = per_request[per_request["feature"] == feature].set_index("model")[
            "cost_per_request_usd"
        ]

        if passing_models:
            optimal_model = min(passing_models, key=lambda m: feature_costs[m])
            note = "Cheapest model that clears the clinical quality gate."
        else:
            optimal_model = "gpt-4o"
            note = (
                "No model cleared the quality gate for this feature in Phase 1 - "
                "defaulting to the higher-quality model until prompts/thresholds are revisited."
            )

        current_cost_30day = projection[projection["feature"] == feature]["cost_usd"].sum()
        optimal_requests = projection[projection["feature"] == feature].groupby("day")[
            "requests"
        ].sum()
        optimal_cost_30day = (
            optimal_requests * feature_costs[optimal_model]
        ).sum()

        savings_rows.append(
            {
                "feature": feature,
                "current_canary_cost_30day_usd": round(current_cost_30day, 2),
                "optimal_model": optimal_model,
                "optimal_cost_30day_usd": round(optimal_cost_30day, 2),
                "savings_usd": round(current_cost_30day - optimal_cost_30day, 2),
                "note": note,
            }
        )

    savings_df = pd.DataFrame(savings_rows)
    savings_path = os.path.join(OUTPUT_DIR, "structural_savings_analysis.csv")
    savings_df.to_csv(savings_path, index=False)
    print(f"Saved structural savings analysis -> {savings_path}")

    print("\n=== Structural Savings Summary ===")
    print(savings_df.to_string(index=False))


if __name__ == "__main__":
    run()
