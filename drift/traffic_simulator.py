"""
Simulates 3 months of AfyaPlus production traffic.

There is no real historical traffic to replay, so this module simulates
the thing the brief describes: patient messages drifting over time
(longer, noisier, more informal) as real-world usage patterns creep in
after launch. Month 1 is the clean baseline the production model was
validated against; Months 2 and 3 progressively inject noise into the
same underlying questions, which is what should eventually show up as
statistical drift in Evidently's reports.

Each simulated request is actually sent to the production model
(gpt-4o-mini, matching the default deployed in the Week 2 RAG agent), so
rouge_l / latency_ms / input_token_length are real measurements, not
fabricated numbers.
"""

import random
import time

from common.llm_client import EVAL_MODEL, get_client
from common.tokens import count_tokens
from evaluation.evaluation_data import EVALUATION_DATASET
from evaluation.knowledge_context import SYSTEM_PROMPT
from evaluation.metrics import compute_rouge_l

REQUESTS_PER_MONTH = 30
PRODUCTION_MODEL = EVAL_MODEL  # gpt-4o-mini: the model actually deployed

# Noise pools used to simulate messier real-world patient input over time.
_MONTH2_PREFIXES = [
    "Hi, sorry to bother you, ",
    "Umm, quick question, ",
    "Hello, hope you're well. ",
]
_MONTH2_SUFFIXES = [
    " Also, I'm not sure if that's related.",
    " Please advise when you can.",
    " Sorry for the long message.",
]
_MONTH3_PREFIXES = [
    "Hi sorry to disturb you again, I already messaged before but ",
    "Sasa, I really need help, aki this is urgent, ",
    "Hello hello, hope you are fine, I have been trying to reach someone, ",
]
_MONTH3_SUFFIXES = [
    " Also my neighbor has similar issue, should she also come, and also is this covered, "
    "and one more thing, how long will it take, thank you so much, God bless.",
    " By the way I forgot to mention I also have some other minor issues, not sure if "
    "important, please just confirm everything is fine, thanks in advance.",
    " Kindly note network is bad on my side so reply might be delayed, but this is really "
    "urgent please help as soon as possible, thank you thank you.",
]


def _apply_drift_noise(question: str, month: int, rng: random.Random) -> str:
    if month == 1:
        return question
    if month == 2:
        prefix = rng.choice(_MONTH2_PREFIXES) if rng.random() < 0.7 else ""
        suffix = rng.choice(_MONTH2_SUFFIXES) if rng.random() < 0.7 else ""
        return f"{prefix}{question}{suffix}"
    # month 3: heavier drift
    prefix = rng.choice(_MONTH3_PREFIXES)
    suffix = rng.choice(_MONTH3_SUFFIXES)
    return f"{prefix}{question}{suffix}"


def simulate_month(month: int, seed: int) -> list[dict]:
    """
    Simulate one month of traffic: REQUESTS_PER_MONTH real calls to the
    production model, sampled with replacement from the 15-question
    dataset, with month-appropriate input drift applied.
    """
    rng = random.Random(seed)
    client = get_client()
    rows = []

    for i in range(REQUESTS_PER_MONTH):
        base_item = rng.choice(EVALUATION_DATASET)
        simulated_question = _apply_drift_noise(base_item["question"], month, rng)

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=PRODUCTION_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": simulated_question},
            ],
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        answer = response.choices[0].message.content or ""

        rows.append(
            {
                "month": month,
                "request_index": i,
                "base_question_id": base_item["id"],
                "feature": base_item["feature"],
                "channel": base_item["channel"],
                "simulated_question": simulated_question,
                "model_answer": answer,
                "rouge_l": compute_rouge_l(answer, base_item["clinical_reference"]),
                "latency_ms": latency_ms,
                "input_token_length": count_tokens(simulated_question),
            }
        )
        print(f"  month {month} [{i + 1}/{REQUESTS_PER_MONTH}] {base_item['id']}")

    return rows
