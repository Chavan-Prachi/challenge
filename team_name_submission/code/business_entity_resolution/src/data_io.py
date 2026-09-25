"""I/O helpers: reading the challenge's TSVs and writing submission-format
output files (matching_results.tsv / candidate_pairs.tsv).

All reads are explicit ``sep="\\t"`` - the problem statement calls out that
reading without it silently collapses every column into one.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

import pandas as pd


def read_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[]
    )
    df.columns = [c.strip() for c in df.columns]
    # tolerate the occasional header typo (seen in the sample docx export)
    df = df.rename(columns={"Sentity_id": "entity_id"})
    for col in ("business_name", "business_address", "country"):
        if col not in df.columns:
            df[col] = ""
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def read_ground_truth(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[]
    )
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for row in df.itertuples(index=False):
        ids = row.matched_entity_ids.strip() if row.matched_entity_ids else ""
        out[row.source1_entity_id] = ids.split(",") if ids else []
    return out


def write_id_list_tsv(
    path: str, mapping: Dict[str, Iterable[str]], id_col: str, list_col: str
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in mapping.items():
            # de-duplicate while preserving order, drop any accidental blanks
            seen = set()
            clean = []
            for i in ids:
                if i and i not in seen:
                    seen.add(i)
                    clean.append(i)
            f.write(f"{s1_id}\t{','.join(clean)}\n")


def write_matching_results(path: str, mapping: Dict[str, Iterable[str]]) -> None:
    write_id_list_tsv(path, mapping, "source1_entity_id", "matched_entity_ids")


def write_candidate_pairs(path: str, mapping: Dict[str, Iterable[str]]) -> None:
    write_id_list_tsv(path, mapping, "source1_entity_id", "candidate_entity_ids")
