#!/usr/bin/env python3
"""
Train the pairwise entity-matching classifier on the training data.

Pipeline:
  1. Read train_source{1,2,3}.tsv + train_ground_truth.tsv.
  2. Normalize every record (normalize.py) and run blocking (blocking.py)
     to get, for every Source-1 training entity, the candidate set the
     *same* blocking stage would produce at inference time.
  3. Label every (S1, candidate) pair using the ground truth. Ground-truth
     pairs that blocking missed are *added back in* for training only (so
     the classifier still learns from them) and separately reported as the
     blocking-stage recall ceiling - the honest number is the one printed
     as "blocking recall", not the augmented training set.
  4. Split by Source-1 entity id (not by pair!) into train/validation so no
     entity's pairs leak across the split.
  5. Train a LightGBM classifier (MIT-licensed, well under the 8B-parameter
     ceiling - a gradient-boosted tree ensemble has no "parameters" in the
     LLM sense at all) on the engineered pairwise features.
  6. Sweep the decision threshold on the validation split to maximize the
     challenge's macro F_0.5, using the *real* blocking-produced candidates
     (not the ground-truth-augmented set) so the reported score reflects
     the whole pipeline, blocking included.
  7. Save the model, the chosen threshold, and the feature list.

Usage:
    python3 -m src.train \
        --train-dir dataset/train \
        --model-dir models \
        --max-candidates 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import blocking, data_io, evaluate, features


def build_pairs_for_split(
    s1_ids, s1_lookup, candidates, ground_truth, augment_with_truth=True
):
    """Return list of (s1_id, cand_id, label) for the given S1 ids."""
    pairs = []
    for s1_id in s1_ids:
        cand_ids = set(candidates.get(s1_id, []))
        truth_ids = set(ground_truth.get(s1_id, []))
        if augment_with_truth:
            cand_ids |= truth_ids
        for cid in cand_ids:
            pairs.append((s1_id, cid, 1 if cid in truth_ids else 0))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--max-candidates", type=int, default=50)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading training data ...")
    s1_df = data_io.read_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2_df = data_io.read_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3_df = data_io.read_source(os.path.join(args.train_dir, "train_source3.tsv"))
    ground_truth = data_io.read_ground_truth(
        os.path.join(args.train_dir, "train_ground_truth.tsv")
    )
    print(f"  S1: {len(s1_df)}  S2: {len(s2_df)}  S3: {len(s3_df)}  "
          f"labeled S1 entities: {len(ground_truth)}")

    print("Normalizing records ...")
    s1_records = blocking.prepare_records(s1_df, "S1")
    s2_records = blocking.prepare_records(s2_df, "S2")
    s3_records = blocking.prepare_records(s3_df, "S3")

    record_lookup = {r.entity_id: r for r in s1_records + s2_records + s3_records}

    print("Building blocking indices and generating candidates ...")
    country_indices = blocking.build_country_indices(s2_records, s3_records)
    candidates = blocking.generate_candidates(
        s1_records, country_indices, max_candidates=args.max_candidates
    )

    # --- Blocking recall diagnostic (the honest number) ------------------
    total_truth_pairs = sum(len(v) for v in ground_truth.values())
    found_truth_pairs = sum(
        len(set(v) & set(candidates.get(s1_id, [])))
        for s1_id, v in ground_truth.items()
    )
    blocking_recall = (
        found_truth_pairs / total_truth_pairs if total_truth_pairs else float("nan")
    )
    print(
        f"Blocking recall ceiling: {found_truth_pairs}/{total_truth_pairs} "
        f"= {blocking_recall:.4f} of true matches are reachable by the "
        f"trained classifier."
    )

    # --- Split by S1 entity (never by pair) -------------------------------
    all_s1_ids = [r.entity_id for r in s1_records if r.entity_id in ground_truth]
    random.shuffle(all_s1_ids)
    n_val = max(1, int(len(all_s1_ids) * args.val_fraction))
    val_ids = set(all_s1_ids[:n_val])
    train_ids = set(all_s1_ids[n_val:])
    print(f"Split: {len(train_ids)} train S1 entities, {len(val_ids)} val S1 entities")

    train_pairs = build_pairs_for_split(
        train_ids, record_lookup, candidates, ground_truth, augment_with_truth=True
    )
    # Validation uses ONLY real blocking candidates - no cheating with
    # ground-truth augmentation - so the validation score reflects the full
    # pipeline (blocking + classifier + threshold) end to end.
    val_pairs = build_pairs_for_split(
        val_ids, record_lookup, candidates, ground_truth, augment_with_truth=False
    )

    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    def to_frame(pairs):
        rows, feat_names = features.build_feature_matrix(
            [(s1, cid) for s1, cid, _ in pairs], record_lookup
        )
        X = pd.DataFrame(rows, columns=feat_names)
        y = np.array([lab for _, _, lab in pairs])
        return X, y

    X_train, y_train = to_frame(train_pairs)
    X_val, y_val = to_frame(val_pairs)

    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    print(f"Train label balance: {n_pos} positive / {n_neg} negative")

    # min_child_samples scales with the amount of training data available -
    # the full challenge dataset will have millions of pairs (default 20 is
    # fine there), but small dev/smoke-test runs need a much smaller floor
    # or LightGBM cannot legally split at all.
    min_child_samples = max(1, min(20, len(X_train) // 20))
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=min_child_samples,
        objective="binary",
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_state=args.seed,
        verbosity=-1,
    )
    if len(X_train) == 0:
        raise SystemExit("No training pairs were generated - check your data paths.")

    fit_kwargs = {}
    if len(X_val) > 0 and y_val.sum() > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)

    # --- Threshold sweep on validation, optimizing macro F_0.5 ------------
    val_pred_by_s1 = defaultdict(list)  # s1_id -> [(cand_id, prob)]
    if len(X_val) > 0:
        val_probs = model.predict_proba(X_val)[:, 1]
        for (s1_id, cand_id, _), p in zip(val_pairs, val_probs):
            val_pred_by_s1[s1_id].append((cand_id, p))

    best_threshold, best_score = 0.5, -1.0
    val_truth = {s1: ground_truth.get(s1, []) for s1 in val_ids}
    for threshold in np.arange(0.05, 0.96, 0.05):
        preds = {
            s1: [cid for cid, p in items if p >= threshold]
            for s1, items in val_pred_by_s1.items()
        }
        macro, _ = evaluate.macro_f_beta(preds, val_truth, beta=0.5)
        if macro > best_score:
            best_score, best_threshold = macro, float(threshold)

    print(f"Best validation macro F_0.5 = {best_score:.4f} at threshold {best_threshold:.2f}")

    final_preds = {
        s1: [cid for cid, p in items if p >= best_threshold]
        for s1, items in val_pred_by_s1.items()
    }
    summary = evaluate.error_summary(final_preds, val_truth)
    print(
        f"Validation singleton accuracy: {summary['singleton_accuracy']} "
        f"({summary['n_singletons_correct']}/{summary['n_singletons']})"
    )

    os.makedirs(args.model_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(args.model_dir, "matcher.lgb.txt"))
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "feature_names": features.FEATURE_NAMES,
                "max_candidates": args.max_candidates,
                "validation_macro_f0.5": best_score,
                "blocking_recall": blocking_recall,
            },
            f,
            indent=2,
        )
    print(f"Saved model + config to {args.model_dir}/")


if __name__ == "__main__":
    main()
