"""
Shared OpenAI client + model constants for evaluation, drift, and cost
phases, all of which call gpt-4o-mini and gpt-4o against the AfyaPlus
domain.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY is not configured. Add it to your .env file."
    )

EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini")
PREMIUM_MODEL = os.getenv("OPENAI_PREMIUM_MODEL", "gpt-4o")
MODELS = [EVAL_MODEL, PREMIUM_MODEL]

_client = OpenAI(api_key=_api_key)


def get_client() -> OpenAI:
    return _client
