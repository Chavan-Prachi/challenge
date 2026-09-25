"""
Pairwise feature engineering for the entity-matching classifier.

Every feature is computed from the two normalized records only (see
normalize.py / blocking.py) - nothing here touches an external service, in
line with the "no external data lookup" rule. Features are grouped into
name similarity, address similarity, and a couple of structural/agreement
signals; see the methodology document for the full rationale.
"""

from __future__ import annotations

from typing import Dict, List

from rapidfuzz import fuzz

from .blocking import PreparedRecord, _jaccard  # reuse the same jaccard helper

FEATURE_NAMES = [
    "name_jaccard",
    "name_core_jaccard",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_prefix4_match",
    "name_phonetic_match",
    "name_len_ratio",
    "addr_jaccard",
    "addr_levenshtein_ratio",
    "addr_token_overlap_count",
    "addr_len_ratio",
    "country_match",
]


def pair_features(s1: PreparedRecord, cand: PreparedRecord) -> Dict[str, float]:
    name_jaccard = _jaccard(s1.name_tokens, cand.name_tokens)
    name_lev = fuzz.ratio(s1.name_norm, cand.name_norm) / 100.0
    name_tsort = fuzz.token_sort_ratio(s1.name_core, cand.name_core) / 100.0
    name_partial = fuzz.partial_ratio(s1.name_core, cand.name_core) / 100.0

    len1, len2 = len(s1.name_core), len(cand.name_core)
    name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) else 0.0

    addr_jaccard = _jaccard(s1.addr_tokens, cand.addr_tokens)
    addr_lev = fuzz.ratio(s1.addr_norm, cand.addr_norm) / 100.0
    addr_overlap = len(s1.addr_tokens & cand.addr_tokens)

    alen1, alen2 = len(s1.addr_norm), len(cand.addr_norm)
    addr_len_ratio = (
        min(alen1, alen2) / max(alen1, alen2) if max(alen1, alen2) else 0.0
    )

    return {
        "name_jaccard": name_jaccard,
        "name_core_jaccard": name_jaccard,  # kept as a separate, explicit name
        "name_levenshtein_ratio": name_lev,
        "name_token_sort_ratio": name_tsort,
        "name_partial_ratio": name_partial,
        "name_prefix4_match": float(
            bool(s1.prefix4) and s1.prefix4 == cand.prefix4
        ),
        "name_phonetic_match": float(
            bool(s1.phonetic) and s1.phonetic == cand.phonetic
        ),
        "name_len_ratio": name_len_ratio,
        "addr_jaccard": addr_jaccard,
        "addr_levenshtein_ratio": addr_lev,
        "addr_token_overlap_count": float(addr_overlap),
        "addr_len_ratio": addr_len_ratio,
        "country_match": float(s1.country == cand.country),
    }


def build_feature_matrix(
    pairs: List[tuple], record_lookup: Dict[str, PreparedRecord]
):
    """pairs: list of (s1_id, cand_id). Returns (list_of_rows, feature_names).

    ``record_lookup`` must map every id (S1 and S2/S3) appearing in ``pairs``
    to its PreparedRecord.
    """
    rows = []
    for s1_id, cand_id in pairs:
        s1 = record_lookup[s1_id]
        cand = record_lookup[cand_id]
        rows.append(pair_features(s1, cand))
    return rows, FEATURE_NAMES
