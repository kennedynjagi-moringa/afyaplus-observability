"""
OpenAI list pricing used for cost simulation (USD per 1,000,000 tokens).
These are published API prices, not billing data pulled from an account.
"""

PRICING_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def request_cost_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    rates = PRICING_PER_MILLION_TOKENS[model]
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]
