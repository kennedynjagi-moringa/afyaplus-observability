"""
Phase 1: Clinical Evaluation Pipeline.

Runs the 15-question AfyaPlus dataset through both gpt-4o-mini and
gpt-4o, scores each answer with BLEU / ROUGE-L / Token F1 and an
LLM-as-a-Judge, then writes:

  - full_evaluation_results.csv     (row-by-row raw metrics + judge scores)
  - model_comparison_aggregation.csv (mean scores per model per feature)
  - quality_gate_log.csv             (pass/fail per model per feature)

Run from the project root:
    .venv/bin/python -m evaluation.run_evaluation
"""

import os
import time

import pandas as pd

from common.llm_client import MODELS, get_client
from common.tokens import count_tokens
from evaluation.evaluation_data import EVALUATION_DATASET
from evaluation.knowledge_context import SYSTEM_PROMPT
from evaluation.llm_judge import judge_answer
from evaluation.metrics import compute_bleu, compute_rouge_l, compute_token_f1
from evaluation.quality_gates import QUALITY_GATE_THRESHOLDS, evaluate_quality_gate

OUTPUT_DIR = os.path.dirname(__file__)


def generate_answer(client, model: str, question: str) -> dict:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    answer = response.choices[0].message.content or ""
    usage = response.usage

    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "input_tokens": usage.prompt_tokens if usage else count_tokens(question),
        "output_tokens": usage.completion_tokens if usage else count_tokens(answer),
    }


def run() -> pd.DataFrame:
    client = get_client()
    rows = []

    total = len(MODELS) * len(EVALUATION_DATASET)
    step = 0

    for model in MODELS:
        for item in EVALUATION_DATASET:
            step += 1
            print(f"[{step}/{total}] {model} -> {item['id']} ({item['feature']}/{item['channel']})")

            gen = generate_answer(client, model, item["question"])
            reference = item["clinical_reference"]
            candidate = gen["answer"]

            judge_scores = judge_answer(
                client,
                question=item["question"],
                reference=reference,
                candidate=candidate,
            )

            rows.append(
                {
                    "id": item["id"],
                    "feature": item["feature"],
                    "channel": item["channel"],
                    "model": model,
                    "question": item["question"],
                    "clinical_reference": reference,
                    "model_answer": candidate,
                    "bleu": compute_bleu(candidate, reference),
                    "rouge_l": compute_rouge_l(candidate, reference),
                    "token_f1": compute_token_f1(candidate, reference),
                    **judge_scores,
                    "latency_ms": gen["latency_ms"],
                    "input_tokens": gen["input_tokens"],
                    "output_tokens": gen["output_tokens"],
                }
            )

    df = pd.DataFrame(rows)
    raw_path = os.path.join(OUTPUT_DIR, "full_evaluation_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw metrics log -> {raw_path}")

    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "bleu",
        "rouge_l",
        "token_f1",
        "correctness",
        "groundedness",
        "relevance",
        "helpfulness",
        "overall_alignment",
        "latency_ms",
    ]
    agg = (
        df.groupby(["model", "feature"])[metric_cols]
        .mean()
        .round(3)
        .reset_index()
    )
    agg_path = os.path.join(OUTPUT_DIR, "model_comparison_aggregation.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Saved model comparison aggregation -> {agg_path}")
    return agg


def quality_gate(agg: pd.DataFrame) -> pd.DataFrame:
    gate_rows = []
    for _, row in agg.iterrows():
        mean_scores = {
            "rouge_l": row["rouge_l"],
            "token_f1": row["token_f1"],
            "correctness": row["correctness"],
            "groundedness": row["groundedness"],
            "overall_alignment": row["overall_alignment"],
        }
        gate_result = evaluate_quality_gate(mean_scores)
        gate_rows.append(
            {
                "model": row["model"],
                "feature": row["feature"],
                **mean_scores,
                **gate_result,
            }
        )

    gate_df = pd.DataFrame(gate_rows)
    gate_path = os.path.join(OUTPUT_DIR, "quality_gate_log.csv")
    gate_df.to_csv(gate_path, index=False)
    print(f"Saved quality gate log -> {gate_path}")
    print(f"Thresholds used: {QUALITY_GATE_THRESHOLDS}")
    return gate_df


if __name__ == "__main__":
    results = run()
    aggregation = aggregate(results)
    gates = quality_gate(aggregation)

    print("\n=== Quality Gate Summary ===")
    print(gates[["model", "feature", "overall_pass"]].to_string(index=False))
