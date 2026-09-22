"""
LLM-as-a-Judge: an independent model scores a model answer against the
clinical_reference on five clinical-quality dimensions, each 0-5.
"""

import json

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_SYSTEM_PROMPT = """You are an impartial clinical QA auditor for AfyaPlus.
You compare an AI assistant's answer against a verified clinical reference
answer and score it on five dimensions, each on a 0-5 integer scale
(0 = completely fails, 5 = fully meets the bar):

- correctness: does the answer reach the same clinical/policy/numeric
  conclusion as the reference?
- groundedness: does the answer avoid inventing facts, symptoms, policy
  details, or numbers not supported by the reference?
- relevance: does the answer directly address what was asked?
- helpfulness: is the answer clear and actionable for the intended
  channel (USSD/Mobile/Web) user?
- overall_alignment: your holistic judgment of clinical safety and
  quality alignment with the reference.

Respond with ONLY a JSON object with exactly these integer fields:
{"correctness": int, "groundedness": int, "relevance": int,
 "helpfulness": int, "overall_alignment": int}
"""


def judge_answer(client, question: str, reference: str, candidate: str) -> dict:
    user_prompt = (
        f"Question: {question}\n\n"
        f"Clinical reference answer: {reference}\n\n"
        f"AI assistant's answer to score: {candidate}"
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        scores = {}

    fields = [
        "correctness",
        "groundedness",
        "relevance",
        "helpfulness",
        "overall_alignment",
    ]
    return {field: int(scores.get(field, 0)) for field in fields}
