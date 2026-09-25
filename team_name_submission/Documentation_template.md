# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat entity resolution as a two-stage **blocking + pairwise classifier** pipeline. A
country-scoped combination of exact-key indices (canonical name, first token, NYSIIS
phonetic code, 4-character prefix) and MinHash/LSH fuzzy blocking generates a bounded
candidate set per Source-1 entity in near-linear time; a LightGBM classifier trained on
~13 hand-engineered name/address similarity features then scores each candidate, with the
decision threshold tuned directly against the challenge's macro F_0.5 metric. A final
greedy pass gives each Source-2/3 record to at most one Source-1 entity, trading a small
amount of recall for the precision F_0.5 rewards most.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory review of the provided preview rows surfaced the noise patterns the problem
statement calls out, plus a few practical wrinkles worth flagging explicitly:

- **Script diversity.** Source-2/3 business names appear in Devanagari, Tamil, Telugu,
  Gujarati, Punjabi, Malayalam, and Bengali scripts, sometimes for the *same* business
  whose Source-1 record is in Latin transliteration (e.g. a Hindi rendering of "Ram
  Marketing Private Limited"). We transliterate everything to ASCII before comparison
  (`anyascii`) so this becomes a fuzzy-match problem instead of a script-mismatch problem.
- **Legal-suffix inconsistency** across *and within* records: `Pvt`/`Private`, `Ltd`/
  `Limited`, `Inc`/`Incorporated`, `Corp`/`Corporation`, French forms (`SARL`, `SAS`,
  `SASU`, `EURL`). We canonicalize these to shared tokens and also compute a
  suffix-stripped "core name" so suffix noise never drives a false negative.
  ("`Global Hovnanian LLC`" — legal suffix simply appended.)
- **Address noise**: abbreviations (`Rd`/`Road`, `Ave`/`Avenue`), landmark references
  ("Near Fortis Hospital", "Opp.Rta Office"), missing PIN/state, reordered components
  (some US rows list state before street, e.g. `"OH, Columbus, 5559 Orville Avenue"`), and
  the occasional entirely-blank address (`S3-859268022`). Token-set (bag-of-words)
  similarity is inherently robust to reordering; landmark/filler stopwords are dropped.
- **Open country set.** Training data is US/India only; the test set adds France. The
  pipeline never special-cases a fixed country list — every blocking index and the
  `country_match` feature key off whatever string is present, so France (and any future
  country) is handled automatically.
- **The provided sample files are independent excerpts, not a matched mini-corpus.** None
  of the sample `train_source1.tsv` entity ids appear in the sample
  `train_ground_truth.tsv` (they're evidently head-of-file cuts of much larger, unordered
  files). We verified this, then built a small **synthetic, self-consistent** smoke-test
  fixture (`dev_tools/make_synthetic_smoke_test.py`) purely to exercise and validate the
  training/evaluation code path before pointing the same, unmodified code at the real
  full-size training data (see §5).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (Hybrid)

**Core Innovation:** Country-scoped dual blocking (exact multi-key + MinHash/LSH) feeding
a compact, fully-interpretable feature set into a LightGBM classifier, with the decision
threshold selected by directly optimizing macro F_0.5 (not accuracy or AUC) on an
entity-level validation split, followed by a global greedy 1-record-per-Source-2/3-id
de-duplication pass to further protect precision.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all scoped within `country`, over the combined Source-2+3 pool):
  1. Exact canonical "core" name (legal suffixes stripped, ASCII-normalized)
  2. First core-name token
  3. NYSIIS phonetic code of the first two core-name tokens
  4. 4-character alphanumeric prefix of the core name
  5. MinHash/LSH over character 3-gram shingles of the core name (Jaccard threshold 0.35,
     64 permutations) — catches typos, dropped words, and partial-name matches the exact
     keys miss.
- **Candidate pairs generated:** capped at `max_candidates` (default 50) per Source-1
  entity, ranked by a cheap heuristic (0.7 × name-token Jaccard + 0.3 × address-token
  Jaccard) before truncation — this ranked, truncated list *is* `candidate_pairs.tsv` and
  is exactly what the classifier scores at inference time.
- **How true matches are not lost:** `train.py` reports a **blocking recall ceiling** —
  the fraction of ground-truth matches actually present in the blocking output — every
  training run, since the classifier can never recover a pair blocking never proposed. On
  the synthetic smoke-test fixture, blocking recall was **100% (81/81)**. On the real
  data, we recommend monitoring this number per country and, if it's below target,
  loosening `LSH_THRESHOLD` or adding a key rather than touching the classifier.

---

## 4. Matching Model

**Features used** (`src/features.py`, 13 features total):
- Name features: token-set Jaccard, Levenshtein ratio, token-sort ratio, partial ratio
  (all via `rapidfuzz` after normalization), 4-char-prefix exact-match flag, NYSIIS
  phonetic-match flag, name-length ratio.
- Address features: token-set Jaccard, Levenshtein ratio, raw token-overlap count,
  address-length ratio.
- Other: `country_match` (near-constant given country-scoped blocking, kept as a guard).

**Model type:** LightGBM gradient-boosted trees (`objective="binary"`,
`scale_pos_weight` set from the train-split class balance to counter the heavy
candidate-set class imbalance). MIT-licensed, and — having no "parameters" in the LLM
sense — trivially satisfies the ≤8B-parameter / MIT-or-Apache-2.0 model constraint.

**Threshold selection method:** a sweep over thresholds 0.05–0.95 (step 0.05) on the
entity-level validation split (never seen during training, split by Source-1 id to avoid
leakage), each scored with the challenge's exact macro F_0.5 formula; the threshold
maximizing that macro score is what ships in `models/config.json` and is used unchanged
at inference time. A subsequent greedy pass (`enforce_unique_targets` in `src/predict.py`)
resolves any Source-2/3 record claimed by multiple Source-1 entities in favor of the
highest-probability claimant, trading a small amount of recall for the precision F_0.5
weights 2× over recall.

---

## 5. Results & Error Analysis

We could not compute a meaningful score on the provided sample files, because (as noted
in §2.1) the sample `train_source{1,2,3}.tsv` rows and the sample `train_ground_truth.tsv`
rows do not correspond to the same entities — this is a size-limited-preview artifact, not
a code issue. To validate the pipeline mechanics end to end before running it on the real
data, we ran it against the fabricated, self-consistent smoke-test fixture instead:

- **F_0.5 Score (macro), synthetic smoke test:** **1.00** on a 13-entity held-out
  validation split (44 total labeled entities; deliberately includes singletons and
  multi-match entities). This number certifies the pipeline mechanics (blocking → feature
  engineering → classifier → threshold tuning → evaluation) work correctly; it is **not**
  a claim about real-world accuracy, since the fixture's synthetic noise is simpler than
  the full challenge data's.
- **Blocking recall ceiling, synthetic smoke test:** 100% (81/81 true matches reachable).
- **Format validation:** running the organizers' `utils/validate_submission.py` against
  the pipeline's output on the real (preview-sample) `dataset/test` files returns
  `PASS — no blocking issues found. Safe to submit.`, including with `--check-ids`.
- **Common false positives (wrong merges):** none observed on the small synthetic
  validation split; anticipated real-data risk areas are (a) very generic/short names
  sharing a token or phonetic code within the same country (e.g. many "XYZ Private
  Limited" records) and (b) near-duplicate franchise-style names at different locations —
  both are mitigated by the address-similarity features and the country scoping, but are
  worth watching once real error analysis is possible.
- **Common false negatives (missed matches):** anticipated real-data risk areas are heavy
  address truncation (city/state present, street missing or vice versa) combined with a
  materially reworded name, since that can push a true pair below both the blocking keys
  and the LSH threshold simultaneously; the blocking-recall diagnostic in `train.py` is
  designed to surface exactly this on the real data.

---

## 6. Conclusion

The pipeline combines linear-time, country-scoped blocking (exact keys + MinHash/LSH)
with a small, interpretable LightGBM classifier whose decision threshold is tuned
directly against the challenge's own macro F_0.5 metric, plus a precision-oriented
de-duplication pass. Every stage was validated end-to-end (format compliance via the
organizers' own validator, and pipeline correctness via a fabricated self-consistent
smoke test), since the real training/ground-truth sample rows provided did not overlap.
The main lesson from this phase was to build that blocking-recall diagnostic *before*
trusting any downstream score — it is what will tell us, once run on the real data,
whether classifier accuracy or blocking coverage is the bottleneck to improve next.

---

## Appendix

### A. Code Artefacts

Full runnable pipeline ships under `code/business_entity_resolution/`:

- `src/normalize.py`, `src/blocking.py`, `src/features.py`, `src/data_io.py`,
  `src/evaluate.py` — shared library code.
- `src/train.py` — entry point: `python3 -m src.train --train-dir dataset/train
  --model-dir models` → `models/matcher.lgb.txt` + `models/config.json`.
- `src/predict.py` — entry point: `python3 -m src.predict --test-dir dataset/test
  --model-dir models --output-dir output` → `output/matching_results.tsv` +
  `output/candidate_pairs.tsv`.
- `dev_tools/make_synthetic_smoke_test.py` — regenerates the fabricated smoke-test
  fixture referenced in §5.
- `utils/validate_submission.py` — the organizers' validator, included unmodified.

Exact reproduction steps, including environment setup, are in
`code/business_entity_resolution/README.md`.

### B. Additional Results

See `code/business_entity_resolution/README.md` "Design notes / scaling to the full
dataset" for blocking-complexity notes, and `models/config.json` (written by `train.py`)
for the exact threshold, feature list, and blocking-recall/validation-F_0.5 numbers of
whichever training run produced the shipped model.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity
and technical depth.
