from cost.pricing import request_cost_usd


def test_gpt4o_mini_cheaper_than_gpt4o_for_same_tokens():
    mini_cost = request_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=200)
    premium_cost = request_cost_usd("gpt-4o", input_tokens=1000, output_tokens=200)
    assert mini_cost < premium_cost


def test_zero_tokens_costs_zero():
    assert request_cost_usd("gpt-4o-mini", input_tokens=0, output_tokens=0) == 0.0


def test_cost_scales_linearly_with_tokens():
    single = request_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=0)
    double = request_cost_usd("gpt-4o-mini", input_tokens=2000, output_tokens=0)
    assert double == single * 2
