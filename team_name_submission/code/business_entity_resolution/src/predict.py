#!/usr/bin/env python3
"""
Run the trained pipeline on the test set and write both submission files:

  output/candidate_pairs.tsv   - the blocking stage's final candidate set
                                  (exactly what the model scores)
  output/matching_results.tsv  - candidates with predicted probability >=
                                  the trained threshold, after an optional
                                  greedy "each S2/S3 record is claimed by at
                                  most one Source-1 entity" precision pass

Every Source-1 entity in test_source1.tsv gets exactly one row in both
output files, per the format spec (empty matched/candidate list for
entities blocking finds nothing for).

Usage:
    python3 -m src.predict \
        --test-dir dataset/test \
        --model-dir models \
        --output-dir output \
        --max-candidates 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import blocking, data_io, features


def enforce_unique_targets(scored_matches):
    """Greedy 1:1 assignment: if the same S2/S3 id is a match candidate for
    several Source-1 entities, keep it only for the highest-probability one.
    This is a precision-oriented post-processing pass (F_0.5 weights
    precision 2x) built on the reasonable real-world assumption that a
    single Source-2/3 record describes exactly one real business, even
    though a Source-1 entity may legitimately match several Source-2/3
    duplicates of itself.

    ``scored_matches``: {s1_id: [(cand_id, prob), ...]} (already thresholded)
    Returns the same shape, with conflicts resolved.
    """
    best_owner: dict = {}
    for s1_id, items in scored_matches.items():
        for cand_id, prob in items:
            if cand_id not in best_owner or prob > best_owner[cand_id][1]:
                best_owner[cand_id] = (s1_id, prob)

    resolved: dict = {s1_id: [] for s1_id in scored_matches}
    for cand_id, (owner_s1, prob) in best_owner.items():
        resolved[owner_s1].append((cand_id, prob))
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--max-candidates", type=int, default=None,
                     help="Override the trained config's max_candidates.")
    ap.add_argument("--no-unique-targets", action="store_true",
                     help="Disable the greedy 1:1 de-duplication pass.")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    max_candidates = args.max_candidates or config["max_candidates"]
    threshold = config["threshold"]
    feature_names = config["feature_names"]

    model = lgb.Booster(model_file=os.path.join(args.model_dir, "matcher.lgb.txt"))

    print("Loading test data ...")
    s1_df = data_io.read_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2_df = data_io.read_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3_df = data_io.read_source(os.path.join(args.test_dir, "test_source3.tsv"))
    print(f"  S1: {len(s1_df)}  S2: {len(s2_df)}  S3: {len(s3_df)}")

    print("Normalizing records ...")
    s1_records = blocking.prepare_records(s1_df, "S1")
    s2_records = blocking.prepare_records(s2_df, "S2")
    s3_records = blocking.prepare_records(s3_df, "S3")
    record_lookup = {r.entity_id: r for r in s1_records + s2_records + s3_records}

    print("Building blocking indices and generating candidates ...")
    country_indices = blocking.build_country_indices(s2_records, s3_records)
    candidates = blocking.generate_candidates(
        s1_records, country_indices, max_candidates=max_candidates
    )

    # Every Source-1 test entity must appear, even with an empty list.
    candidate_pairs = {r.entity_id: candidates.get(r.entity_id, []) for r in s1_records}

    print("Scoring candidates ...")
    all_pairs = [
        (s1_id, cid) for s1_id, cids in candidate_pairs.items() for cid in cids
    ]
    if all_pairs:
        rows, _ = features.build_feature_matrix(all_pairs, record_lookup)
        X = pd.DataFrame(rows, columns=feature_names)
        probs = model.predict(X)
    else:
        probs = np.array([])

    scored_matches: dict = {r.entity_id: [] for r in s1_records}
    for (s1_id, cid), p in zip(all_pairs, probs):
        if p >= threshold:
            scored_matches[s1_id].append((cid, float(p)))

    if not args.no_unique_targets:
        scored_matches = enforce_unique_targets(scored_matches)

    matching_results = {
        s1_id: [cid for cid, _ in sorted(items, key=lambda t: -t[1])]
        for s1_id, items in scored_matches.items()
    }

    os.makedirs(args.output_dir, exist_ok=True)
    data_io.write_candidate_pairs(
        os.path.join(args.output_dir, "candidate_pairs.tsv"), candidate_pairs
    )
    data_io.write_matching_results(
        os.path.join(args.output_dir, "matching_results.tsv"), matching_results
    )

    n_matched = sum(1 for v in matching_results.values() if v)
    print(
        f"Wrote {len(matching_results)} rows to matching_results.tsv "
        f"({n_matched} with >=1 match, {len(matching_results) - n_matched} singletons)"
    )
    print(f"Wrote candidate_pairs.tsv with {len(candidate_pairs)} rows")


if __name__ == "__main__":
    main()
