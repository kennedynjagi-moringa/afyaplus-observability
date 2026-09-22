from evaluation.metrics import compute_bleu, compute_rouge_l, compute_token_f1


def test_identical_strings_score_high():
    text = "Route to emergency care immediately for severe chest pain."
    assert compute_rouge_l(text, text) == 1.0
    assert compute_token_f1(text, text) == 1.0
    assert compute_bleu(text, text) > 90


def test_unrelated_strings_score_low():
    candidate = "The weather today is sunny with a light breeze."
    reference = "Route to emergency care immediately for severe chest pain."
    assert compute_rouge_l(candidate, reference) < 0.2
    assert compute_token_f1(candidate, reference) < 0.2


def test_empty_candidate_scores_zero():
    reference = "Route to emergency care immediately."
    assert compute_rouge_l("", reference) == 0.0
    assert compute_token_f1("", reference) == 0.0
    assert compute_bleu("", reference) == 0.0
