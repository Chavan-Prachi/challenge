# Business Entity Resolution — ML Challenge 2026

Blocking + gradient-boosted-tree classifier pipeline that matches Source-1
business records against Source-2/3 records across US, India, and (test-set
only) France.

## 1. Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.10+ recommended (developed/tested on 3.12).

## 2. Put the data in place

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

The real challenge files are large (test alone is on the order of 10^6 rows
per source) and are **not** bundled in this zip. Copy/symlink them into
`dataset/train` and `dataset/test` before running the commands below.

For convenience, this folder ships two small illustrative fixtures that
are **not** the real challenge data:

- `dataset/train/` and `dataset/test/` (as delivered here) contain the
  small preview rows the challenge portal supplied. **Note:** the preview
  rows for `train_source{1,2,3}.tsv` and `train_ground_truth.tsv` are
  independent head-of-file excerpts of the real files and do not actually
  correspond to the same entities (no sample source1 id appears in the
  sample ground truth) — that's expected for a size-limited preview, not a
  bug. They're enough to sanity-check file formats and run the scripts
  end-to-end, but a model trained on them alone learns nothing.
- `dataset/dev_synthetic/train/` is a small, **fabricated** self-consistent
  dataset (see `dev_tools/make_synthetic_smoke_test.py`) used only to
  smoke-test that training/evaluation actually works end to end before
  pointing the same code at the real data. Regenerate it with:
  ```bash
  python3 dev_tools/make_synthetic_smoke_test.py dataset/dev_synthetic/train
  ```

## 3. Train

```bash
python3 -m src.train \
    --train-dir dataset/train \
    --model-dir models \
    --max-candidates 50 \
    --val-fraction 0.2
```

This normalizes every record, runs the blocking stage to build candidate
pairs, labels them from `train_ground_truth.tsv`, splits **by Source-1
entity id** (never by pair, to avoid leakage) into train/validation, fits
a LightGBM classifier on the pairwise features, sweeps the decision
threshold to maximize macro F_0.5 on the validation split, and writes:

- `models/matcher.lgb.txt` — the trained LightGBM booster
- `models/config.json` — chosen threshold, feature list, blocking recall,
  and validation macro F_0.5

It prints the **blocking recall ceiling** (share of true matches the
blocking stage actually retrieves) — this is the number to watch when
tuning blocking, since the classifier can never recover a pair blocking
never proposed.

## 4. Predict on the test set

```bash
python3 -m src.predict \
    --test-dir dataset/test \
    --model-dir models \
    --output-dir output \
    --max-candidates 50
```

Writes `output/candidate_pairs.tsv` (the blocking stage's final candidate
set) and `output/matching_results.tsv` (thresholded predictions, after a
greedy pass that gives each Source-2/3 record to at most one Source-1
entity — see `enforce_unique_targets` in `src/predict.py`; disable with
`--no-unique-targets`).

## 5. Validate before submitting

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

(This is the organizers' own validator, copied verbatim from the challenge
resources.)

## 6. Score your own held-out split

`src/train.py` already reports validation macro F_0.5 during training. To
score any prediction file against a ground truth file directly:

```python
from src import data_io, evaluate
preds = data_io.read_ground_truth("output/matching_results.tsv")  # same 2-col format
truth = data_io.read_ground_truth("dataset/train/train_ground_truth.tsv")
macro, per_entity = evaluate.macro_f_beta(preds, truth)
print(macro)
```

## Code layout

```
src/
├── normalize.py   # name/address text normalization (no external lookups)
├── blocking.py     # candidate generation: exact-key + MinHash/LSH blocking
├── features.py     # pairwise similarity feature engineering
├── data_io.py       # TSV readers/writers in the exact submission format
├── evaluate.py      # macro F_0.5 scorer (matches the problem statement)
├── train.py         # end-to-end training + threshold tuning entry point
└── predict.py        # end-to-end inference entry point
dev_tools/
└── make_synthetic_smoke_test.py   # fabricates a tiny self-consistent set
utils/
└── validate_submission.py         # organizers' validator (unmodified)
```

## Design notes / scaling to the full dataset

- **Blocking is O(N)+LSH, not O(N²).** Exact-key indices (canonical core
  name, first token, NYSIIS phonetic code, 4-char prefix) and MinHash/LSH
  over 3-gram shingles are each built once per country over the combined
  Source-2+3 pool, then queried per Source-1 record — no all-pairs
  comparison anywhere.
- **Country scoping.** Country is treated as an open string label (never
  hard-coded to {US, India}); every index/lookup is scoped within a
  country so France (test-set only) is handled automatically.
- **No external data.** Only static, hand-built abbreviation tables and
  the training data itself are used — no geocoding, no registries, no
  network calls — per the challenge's fair-play rules.
- **Model size/license.** The classifier is LightGBM (MIT license), which
  has no "parameters" in the LLM sense at all — trivially inside the
  8B-parameter, MIT/Apache-2.0 constraint.
- **At full scale (~10^6 rows/source):** the exact-key blocking indices
  are plain hash maps and scale linearly; MinHash/LSH insertion and query
  are each O(1) amortized. The one part worth profiling on the real data
  is per-country LSH bucket size for very generic names (e.g. many
  "XYZ Private Limited" records sharing a token) — `max_candidates`
  truncates this, but consider raising `LSH_THRESHOLD` in `blocking.py`
  for such countries if candidate lists get too large.
