"""
Simulates 30 days of AfyaPlus production request volume, split 75%
gpt-4o-mini / 25% gpt-4o (the canary split specified in the brief),
across the 3 clinical features.

Base daily volumes are illustrative assumptions for a mid-size digital
health platform (documented here, not hidden), with day-to-day noise and
a mild growth trend layered on top to look like real traffic.
"""

import random

import pandas as pd

BASE_DAILY_VOLUME = {
    "triage_routing": 900,
    "insurance_verification": 600,
    "medication_calculation": 350,
}

DAYS = 30
CANARY_SPLIT = {"gpt-4o-mini": 0.75, "gpt-4o": 0.25}
DAILY_GROWTH_RATE = 0.01  # +1%/day compounding, mild organic growth
NOISE_STD = 0.08  # +/-8% day-to-day noise


def simulate_volume(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for day in range(1, DAYS + 1):
        for feature, base_volume in BASE_DAILY_VOLUME.items():
            growth_factor = (1 + DAILY_GROWTH_RATE) ** (day - 1)
            noise_factor = 1 + rng.gauss(0, NOISE_STD)
            total_requests = max(0, round(base_volume * growth_factor * noise_factor))

            mini_requests = round(total_requests * CANARY_SPLIT["gpt-4o-mini"])
            gpt4o_requests = total_requests - mini_requests

            rows.append(
                {"day": day, "feature": feature, "model": "gpt-4o-mini", "requests": mini_requests}
            )
            rows.append(
                {"day": day, "feature": feature, "model": "gpt-4o", "requests": gpt4o_requests}
            )

    return pd.DataFrame(rows)
