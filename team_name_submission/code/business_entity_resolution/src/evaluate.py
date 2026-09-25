"""
Macro-averaged F_0.5 scoring, exactly as defined in the problem statement:
computed per Source-1 entity (singletons included, worth 1.0 when correctly
predicted empty and 0.0 if a match is wrongly predicted for them), then
averaged across all Source-1 entities.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f_beta(pred: Iterable[str], truth: Iterable[str], beta: float = 0.5) -> float:
    pred_set, true_set = set(pred), set(truth)
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    inter = len(pred_set & true_set)
    precision = inter / len(pred_set)
    recall = inter / len(true_set)
    return _f_beta(precision, recall, beta)


def macro_f_beta(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    beta: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Returns (macro_score, per_entity_scores). Every S1 id in
    ``ground_truth`` must have an entry in ``predictions`` (missing ids are
    treated as an empty prediction, matching how the real scorer would
    penalize a missing row).
    """
    scores = {}
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, [])
        scores[s1_id] = entity_f_beta(pred, truth, beta)
    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return macro, scores


def error_summary(
    predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]
) -> Dict[str, object]:
    """Lightweight false-positive / false-negative breakdown for the
    methodology write-up's error-analysis section.
    """
    false_positive_examples = []
    false_negative_examples = []
    n_singletons = n_singletons_correct = 0

    for s1_id, truth in ground_truth.items():
        pred = set(predictions.get(s1_id, []))
        truth_set = set(truth)
        if not truth_set:
            n_singletons += 1
            if not pred:
                n_singletons_correct += 1
        fp = pred - truth_set
        fn = truth_set - pred
        if fp and len(false_positive_examples) < 10:
            false_positive_examples.append((s1_id, sorted(fp)))
        if fn and len(false_negative_examples) < 10:
            false_negative_examples.append((s1_id, sorted(fn)))

    return {
        "n_entities": len(ground_truth),
        "n_singletons": n_singletons,
        "n_singletons_correct": n_singletons_correct,
        "singleton_accuracy": (
            n_singletons_correct / n_singletons if n_singletons else None
        ),
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
    }
