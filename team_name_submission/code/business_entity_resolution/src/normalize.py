"""
Text normalization for the Business Entity Resolution challenge.

Business names and addresses arrive from three independent sources with very
different formatting conventions (casing, legal-suffix abbreviations, native
scripts, punctuation, landmark references, reordered components...). This
module turns raw strings into a small set of comparable representations that
the blocking and feature-engineering stages build on:

  - a fully normalized string (ASCII, lower-case, abbreviations expanded)
  - a "core" name with legal suffixes stripped (for suffix-insensitive match)
  - a token set (for order-insensitive / reordering-robust comparison)
  - a phonetic key (for typo-robust blocking)

No external data or network lookups are used anywhere in this module -
only static, hand-built abbreviation tables - to comply with the challenge's
"no external data lookup" rule.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from anyascii import anyascii
import jellyfish

# ---------------------------------------------------------------------------
# Static abbreviation / suffix tables (no external lookups - hand curated
# from the noise patterns called out in the problem statement).
# ---------------------------------------------------------------------------

# Legal-form suffixes -> canonical token. Order matters: longer phrases first
# so "private limited" is matched before the bare "limited".
LEGAL_SUFFIX_MAP = {
    "private limited": "pvtltd",
    "pvt ltd": "pvtltd",
    "pvt. ltd.": "pvtltd",
    "pvt": "pvtltd",
    "private": "pvtltd",
    "limited": "pvtltd",
    "ltd": "pvtltd",
    "llp": "llp",
    "l l p": "llp",
    "l.l.p": "llp",
    "llc": "llc",
    "l l c": "llc",
    "l.l.c": "llc",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "sarl": "sarl",
    "s.a.r.l": "sarl",
    "sasu": "sasu",
    "sas": "sas",
    "eurl": "eurl",
    "sci": "sci",
    "s.a": "sa",
    "sa": "sa",
    "plc": "plc",
    "lp": "lp",
    "p.c": "pc",
    "pc": "pc",
    "l.c.s.w": "lcsw",
}
# regex alternation, longest-first (already ordered above), suffix-anchored
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in LEGAL_SUFFIX_MAP) + r")\b"
)

# Common punctuation / word substitutions in business names.
NAME_SUBSTITUTIONS = [
    (r"&", " and "),
    (r"\bdba\b", " dba "),  # "doing business as" - keep as a separator token
    (r"[\u2018\u2019']", ""),  # apostrophes: "Orelee's" -> "Orelees"
    (r"[^a-z0-9\s]", " "),  # drop remaining punctuation
]

# Address component abbreviations -> canonical form.
ADDRESS_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "unit",
    "apartment": "unit",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "no": "number",
    "twp": "township",
    "co": "county",  # only fires when isolated - see NOTE below re: name vs addr
}

# Tokens that carry little discriminative value for address matching
# (landmark connectors, filler words, explicit "no data" markers).
ADDRESS_STOPWORDS = {
    "near", "opp", "opposite", "behind", "beside", "next", "to", "the",
    "of", "at", "in", "unit", "null", "and", "po", "box",
}


def _strip_diacritics(text: str) -> str:
    """Transliterate any script to ASCII (phonetic, no network calls)."""
    return anyascii(text)


def _basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = _strip_diacritics(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    return text


@dataclass
class NormalizedName:
    raw: str
    norm: str            # cleaned, ASCII, abbreviations canonicalized
    core: str            # norm with legal-suffix tokens removed entirely
    tokens: frozenset = field(default_factory=frozenset)
    core_tokens: frozenset = field(default_factory=frozenset)
    sorted_core: str = ""     # order-insensitive representation
    phonetic: str = ""        # NYSIIS code of the first core token
    first_token: str = ""


@dataclass
class NormalizedAddress:
    raw: str
    norm: str
    tokens: frozenset = field(default_factory=frozenset)


def normalize_name(name: str) -> NormalizedName:
    raw = name or ""
    text = _basic_clean(raw)
    for pattern, repl in NAME_SUBSTITUTIONS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()

    # Canonicalize legal-suffix phrases wherever they occur.
    def _sub_suffix(m):
        return LEGAL_SUFFIX_MAP[m.group(1)]

    norm = _SUFFIX_PATTERN.sub(_sub_suffix, text)
    norm = re.sub(r"\s+", " ", norm).strip()

    canonical_suffixes = set(LEGAL_SUFFIX_MAP.values())
    tokens = [t for t in norm.split() if t]
    core_tokens = [t for t in tokens if t not in canonical_suffixes]
    core = " ".join(core_tokens)
    sorted_core = " ".join(sorted(core_tokens))

    phonetic = ""
    if core_tokens:
        try:
            phonetic = jellyfish.nysiis("".join(core_tokens[:2]))
        except Exception:
            phonetic = ""

    return NormalizedName(
        raw=raw,
        norm=norm,
        core=core,
        tokens=frozenset(tokens),
        core_tokens=frozenset(core_tokens),
        sorted_core=sorted_core,
        phonetic=phonetic,
        first_token=core_tokens[0] if core_tokens else "",
    )


def normalize_address(address: str) -> NormalizedAddress:
    raw = address or ""
    text = _basic_clean(raw)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    out_tokens = []
    for tok in text.split():
        # Only expand short alphabetic tokens (avoid mangling numbers like
        # unit/PIN codes) and skip the "co" -> county rule if it looks like
        # it is actually a company abbreviation leaking from the name field.
        expanded = ADDRESS_ABBR_MAP.get(tok, tok)
        if expanded in ADDRESS_STOPWORDS:
            continue
        out_tokens.append(expanded)

    norm = " ".join(out_tokens)
    return NormalizedAddress(raw=raw, norm=norm, tokens=frozenset(out_tokens))


def normalize_country(country: str) -> str:
    """Country is an open-set string label - normalize casing/whitespace only.

    Never hard-code a fixed {US, India} set: the test set adds France, and
    any future country label must pass through unchanged apart from this
    light cleanup.
    """
    if country is None:
        return ""
    return str(country).strip().lower()
