"""
Candidate generation (blocking) for the Business Entity Resolution challenge.

Comparing every Source-1 entity against every Source-2/3 record is
quadratic and infeasible at the challenge's scale (the test set has on the
order of 10^6 entities per source). This module cuts the comparison space
down with two complementary, purely-statistical techniques (no external
data, per the fair-play rules):

1. Exact-key blocking - four cheap, high-precision keys (canonical core
   name, first name token, NYSIIS phonetic code, 4-char alnum prefix), all
   scoped *within country* so a US "Apex Inc" is never blocked against an
   Indian "Apex Inc". This is O(N) to build and O(1) amortized to query.

2. MinHash/LSH blocking on character 3-gram shingles of the normalized
   name - catches the typo / reordering / partial-name cases exact keys
   miss, while staying sub-linear (LSH banding) instead of all-pairs.

Both indices are built once per country over the combined Source-2 + 3
pool, then queried per Source-1 entity. The union of hits is scored with a
cheap heuristic (token Jaccard on name + address) and truncated to
``max_candidates`` - this truncated, scored list is exactly what
``candidate_pairs.tsv`` reports and what the matching model scores at
inference time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasketch import MinHash, MinHashLSH

from .normalize import normalize_address, normalize_country, normalize_name

NUM_PERM = 64
SHINGLE_SIZE = 3
LSH_THRESHOLD = 0.35  # Jaccard threshold on 3-gram shingles for a LSH hit


def _shingles(text: str, k: int = SHINGLE_SIZE) -> List[str]:
    text = text.replace(" ", "")
    if len(text) < k:
        return [text] if text else []
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def _minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for sh in _shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh


@dataclass
class PreparedRecord:
    entity_id: str
    source: str  # "S2" or "S3"
    country: str
    name_norm: str
    name_core: str
    name_tokens: frozenset
    first_token: str
    phonetic: str
    prefix4: str
    addr_norm: str
    addr_tokens: frozenset
    minhash: MinHash


def _prefix4(core: str) -> str:
    alnum = "".join(ch for ch in core if ch.isalnum())
    return alnum[:4]


def prepare_records(df: pd.DataFrame, source: str) -> List[PreparedRecord]:
    """Normalize every record once and cache everything blocking/features need."""
    records = []
    for row in df.itertuples(index=False):
        nm = normalize_name(row.business_name)
        addr = normalize_address(row.business_address)
        country = normalize_country(row.country)
        records.append(
            PreparedRecord(
                entity_id=row.entity_id,
                source=source,
                country=country,
                name_norm=nm.norm,
                name_core=nm.core,
                name_tokens=nm.core_tokens,
                first_token=nm.first_token,
                phonetic=nm.phonetic,
                prefix4=_prefix4(nm.core),
                addr_norm=addr.norm,
                addr_tokens=addr.tokens,
                minhash=_minhash(nm.core),
            )
        )
    return records


class CountryIndex:
    """All lookup structures for one country's Source-2 + Source-3 pool."""

    def __init__(self, records: List[PreparedRecord]):
        self.records: Dict[str, PreparedRecord] = {r.entity_id: r for r in records}
        self.by_core: Dict[str, List[str]] = defaultdict(list)
        self.by_first_token: Dict[str, List[str]] = defaultdict(list)
        self.by_phonetic: Dict[str, List[str]] = defaultdict(list)
        self.by_prefix4: Dict[str, List[str]] = defaultdict(list)
        self.lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)

        for r in records:
            if r.name_core:
                self.by_core[r.name_core].append(r.entity_id)
            if r.first_token:
                self.by_first_token[r.first_token].append(r.entity_id)
            if r.phonetic:
                self.by_phonetic[r.phonetic].append(r.entity_id)
            if r.prefix4:
                self.by_prefix4[r.prefix4].append(r.entity_id)
            try:
                self.lsh.insert(r.entity_id, r.minhash)
            except ValueError:
                # datasketch rejects duplicate keys; entity_ids are unique
                # across sources by construction, so this should not fire,
                # but skip defensively rather than crash a large batch job.
                pass

    def candidates_for(self, q: PreparedRecord) -> set:
        hits: set = set()
        if q.name_core:
            hits.update(self.by_core.get(q.name_core, ()))
        if q.first_token:
            hits.update(self.by_first_token.get(q.first_token, ()))
        if q.phonetic:
            hits.update(self.by_phonetic.get(q.phonetic, ()))
        if q.prefix4:
            hits.update(self.by_prefix4.get(q.prefix4, ()))
        try:
            hits.update(self.lsh.query(q.minhash))
        except ValueError:
            pass
        return hits


def build_country_indices(
    s2_records: List[PreparedRecord], s3_records: List[PreparedRecord]
) -> Dict[str, CountryIndex]:
    by_country: Dict[str, List[PreparedRecord]] = defaultdict(list)
    for r in s2_records + s3_records:
        by_country[r.country].append(r)
    return {country: CountryIndex(recs) for country, recs in by_country.items()}


def _cheap_score(q: PreparedRecord, cand: PreparedRecord) -> float:
    """Fast heuristic used only to rank/truncate candidates, never to decide
    a match - the trained classifier (features.py + the model) makes that
    call. Kept cheap on purpose: it runs once per (S1, candidate) pair.
    """
    name_j = _jaccard(q.name_tokens, cand.name_tokens)
    addr_j = _jaccard(q.addr_tokens, cand.addr_tokens)
    return 0.7 * name_j + 0.3 * addr_j


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def generate_candidates(
    s1_records: List[PreparedRecord],
    country_indices: Dict[str, CountryIndex],
    max_candidates: int = 50,
) -> Dict[str, List[str]]:
    """Return {s1_entity_id: [candidate_entity_id, ...]} sorted by the cheap
    heuristic score, truncated to ``max_candidates``. This truncated list IS
    the candidate_pairs.tsv content and is exactly what gets scored by the
    matching model at inference time.
    """
    result: Dict[str, List[str]] = {}
    for q in s1_records:
        idx = country_indices.get(q.country)
        if idx is None:
            result[q.entity_id] = []
            continue
        hit_ids = idx.candidates_for(q)
        scored = [(cid, _cheap_score(q, idx.records[cid])) for cid in hit_ids]
        scored.sort(key=lambda t: t[1], reverse=True)
        result[q.entity_id] = [cid for cid, _ in scored[:max_candidates]]
    return result
