"""
Automated text-similarity metrics used in Phase 1: BLEU, ROUGE-L, and
Token F1, each comparing a model answer against the clinical_reference.
"""

import re

import sacrebleu
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_bleu(candidate: str, reference: str) -> float:
    """Sentence-level BLEU (0-100 scale, via sacrebleu)."""
    if not candidate.strip():
        return 0.0
    score = sacrebleu.sentence_bleu(candidate, [reference])
    return round(score.score, 2)


def compute_rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure (0-1 scale)."""
    if not candidate.strip():
        return 0.0
    scores = _rouge.score(reference, candidate)
    return round(scores["rougeL"].fmeasure, 4)


def compute_token_f1(candidate: str, reference: str) -> float:
    """SQuAD-style token overlap F1 (0-1 scale)."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    common = {}
    for tok in cand_tokens:
        common[tok] = common.get(tok, 0) + 1

    overlap = 0
    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    for tok, count in common.items():
        if tok in ref_counts:
            overlap += min(count, ref_counts[tok])

    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)
