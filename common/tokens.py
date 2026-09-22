"""Token counting shared by evaluation, drift, and cost phases."""

import tiktoken

_encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text or ""))
