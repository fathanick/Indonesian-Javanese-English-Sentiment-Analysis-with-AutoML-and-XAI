# Indonesian-Javanese-English Sentiment Analysis with AutoML and XAI

## Description

Reproducible Python pipeline for three-class sentiment classification of
Indonesian-Javanese-English (IJE) code-mixed text. The pipeline compares three
AutoML frameworks on three feature sets and explains the selected model with
LIME.

Classes: `negative` (`label_id = 0`), `neutral` (`1`), `positive` (`2`).

This repository holds the **corrected pipeline**. Every executing stage is
gated by an approval file that binds the run to an exact protocol hash, code
hash, and stage list — see [Execution is approval-gated](#execution-is-approval-gated).

## Reported results

Selection uses **development folds only**; the test split is _not_ used to pick
the configuration.

| Scenario | Dev mean outer macro-F1 | Test macro F1 | Test weighted F1 | Test accuracy |
|---|---:|---:|---:|---:|
| `autogluon_tfidf` **(selected)** | 0.912916 | 0.890872 | 0.892321 | 0.892009 |
| `optuna_tfidf_cm` | 0.907453 | 0.915000 | 0.916000 | 0.915767 |
| `optuna_tfidf` | 0.908907 | 0.913000 | 0.913900 | 0.913607 |
| `flaml_full` | 0.893236 | 0.906038 | 0.907121 | 0.907127 |
| `autogluon_full` | 0.899557 | 0.890200 | 0.891500 | 0.892009 |
| `flaml_tfidf_cm` | 0.901454 | 0.889194 | 0.890453 | 0.889849 |
| `autogluon_tfidf_cm` | 0.906207 | 0.888591 | 0.889807 | 0.889849 |
| `flaml_tfidf` | 0.902260 | 0.880047 | 0.881651 | 0.881210 |
| `optuna_full` | 0.889397 | 0.896800 | 0.898300 | 0.898488 |

The development-selected configuration is `autogluon_tfidf`. It is retained
**regardless of its test ranking** — the honest consequence of selecting on
development folds is that the selected configuration need not be the best
performer on test. Here it ranks fifth of nine on test macro F1, while
`optuna_tfidf_cm` reaches 0.9150. Reporting the test-best configuration as the
outcome would be test-set selection; it is reported here as what it is.

Macro-F1 95% interval for the selected configuration: **0.8617 – 0.9172**
(stratified percentile bootstrap, 2,000 replicates, conditional on the fitted
models and observed test class counts; it does not correct for prior test
exposure or multiple comparisons).

## Repository structure

```text
.
├── data/                        # dataset (unchanged from the earlier release)
│   ├── train.csv  valid.csv  test.csv
│   ├── train_raw.xlsx  valid_raw.xlsx  test_raw.xlsx
│   └── meta.json
├── config/
│   ├── protocol.json            # single configuration consumed by every stage
│   ├── baselines.json           # historical baselines, disabled pending verification
│   ├── approval.example.json    # template, NOT approval
│   └── lid_audit_plan.example.json
├── ije/
│   ├── storage.py               # config/code fingerprints, stage markers, checksums
│   ├── data.py                  # preprocessing, split validation, grouped folds
│   ├── features.py              # TF-IDF, text statistics, CM, embeddings
│   ├── engines.py               # FLAML, AutoGluon, Optuna adapters
│   ├── experiment.py            # development selection, final fit, test scoring
│   ├── explain.py               # LIME on the exact saved pipeline
│   ├── metrics.py               # metrics and bootstrap intervals
│   ├── audit.py                 # IJELID overlap-audit draft
│   ├── validation.py            # protocol checks, offline tests, smoke checks
│   └── report.py                # tables and figures from saved artifacts
├── tests/test_protocol.py
├── run.py                       # approval-gated stage dispatcher
├── make_hardware_record.py
├── make_comparability_audit.py
├── requirements.txt
└── README.md
```

Run directories, fitted models, feature matrices, and validation evidence are
written under `runs/` and `validation_evidence/`, which are excluded from
version control.

> **History.** The earlier release of this repository shipped a flat
> `scripts/` directory written before peer review. That pipeline placed
> preprocessing inside the cross-validation loop, which leaks into the reported
> scores. `ije/features.py` fits vectorizers, IDF, scale, and statistics on each
> inner-training subset only, never on its validation subset. The previous
> scripts remain available in the Git history of this repository and in the
> published archived release for the earlier version.

## Dataset

1,929 code-mixed social-media texts. The source dataset reports agreement
between two annotators with Cohen's kappa of `0.9767`.

| Split | Samples | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 1,350 | 461 | 437 | 452 |
| Validation | 116 | 40 | 36 | 40 |
| Test | 463 | 143 | 163 | 157 |
| Total | 1,929 | 644 | 636 | 649 |

The CSVs carry `text_raw`, `text` (cleaned), `label_raw`, `label`, `label_id`,
and `split`. `data/meta.json` records label mappings, split statistics, the
source URL, and the preprocessing settings.

Cleaning, applied in this order: remove URLs, remove `@mentions`, drop the `#`
symbol while keeping the word, lowercase, collapse whitespace. Punctuation,
digits, emoji, and character repetitions are **retained** (unlike the 2024
comparison pipeline, which removed them — a disclosed difference).

## Method summary

- **Features.** `tfidf`: word 1–3 grams and character 2–4 grams, each capped at
  10,000 features, `min_df=2`, sublinear term frequency, plus scaled text
  statistics. `tfidf_cm`: adds Indonesian/Javanese/English ratios, code-mixing
  indices and switch points. `full`: adds 384-dimensional sentence embeddings.
- **Grouping.** Rows sharing an exact cleaned-text SHA-256 stay in the same
  fold, so duplicate texts cannot straddle a train/validation boundary.
- **Selection.** Five outer folds; framework tuning uses a common inner holdout;
  the winner is the highest mean outer-fold macro-F1, ties broken by
  configuration order. The selection is frozen before final fitting.
- **Final fit and test.** The selected pipeline is fitted once, then scored on
  the held-out test split. Class-level results and the confusion matrix are kept.
- **Explanation.** LIME calls the saved complete prediction pipeline; no
  classifier substitution or refitting is permitted.
- **Language identification.** Contextual language labels come from the
  fine-tuned `fathan/ijelid-ft-indojave-indobertweet` model at a pinned revision.
  The code-mixing definition is explicitly a tag-diversity variant, not the
  unqualified standard CMI.

## Requirements

The reported environment used Python 3.11 with the exact versions pinned in
`requirements.txt`. A virtual environment is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

AutoGluon is pinned because it silently drops CatBoost from its candidate pool
when the package is absent, which would change the search space relative to the
reported run.

## Execution is approval-gated

**No stage runs without an explicit approval file.** `config/approval.json` is
not distributed — it is machine-bound and is listed in `.gitignore`. Copy the
template and fill it yourself:

```bash
cp config/approval.example.json config/approval.json
```

Then set `execution_enabled: true` in `config/protocol.json`, populate
`approved_by`, `approved_at`, the SHA-256 of `config/protocol.json`, the code
hash from `ije.storage.code_digest`, and the stages you actually authorise in
`allowed_stages`. Nothing in this repository populates an approval for you.

```bash
python run.py validate  --run-dir runs/my-run
python run.py preflight --run-dir runs/my-run
python run.py audit     --run-dir runs/my-run --audit-plan config/lid_audit_plan.json
python run.py smoke     --run-dir runs/my-run
python run.py encoders  --run-dir runs/my-run
python run.py develop   --run-dir runs/my-run
python run.py fit       --run-dir runs/my-run
python run.py evaluate  --run-dir runs/my-run
python run.py explain   --run-dir runs/my-run
python run.py report    --run-dir runs/my-run
```

Each stage verifies the artifacts of its prerequisites. A failed stage writes a
`.failed.json` and never silently resumes or overwrites earlier results. Do not
execute the modules directly to bypass the guards.

`config/lid_audit_plan.example.json` is a template. The `audit` stage produces a
draft, never a signed conclusion; a human reviewer must examine the evidence and
sign. `make_comparability_audit.py` and `make_hardware_record.py` are the two
standalone evidence scripts from the reported study; both read inputs that are
**not** redistributed here and say so in their module docstrings.

## Reported environment

- Hardware: Apple M4 Pro, 12 logical cores (8 performance + 4 efficiency), 24 GiB
- Operating system: macOS 26.5.2 (build 25F84) arm64
- Compute: CPU only; GPU/MPS not enabled; thread pool limited to 4
- Schedulers: FLAML 2.5.0, AutoGluon 1.5.0, Optuna 4.7.0, scikit-learn 1.7.2

## Limits and disclosures

- **The test split has been inspected during earlier work.** These results are
  not a newly untouched external evaluation.
- **The selected configuration is not the test-best configuration** (see above).
- **Historical baselines stay disabled.** `config/baselines.json` is empty by
  design: no historical value is copied into a new result table, and weighted F1
  is never relabelled as macro F1. The 2024 comparison is made at pipeline level,
  a difference the accompanying manuscript states.
- **Documented preprocessing difference** against the 2024 comparison pipeline.
- Duplicate grouping is exact-match; it does not establish the absence of all
  near-duplicates.
- IJELID is a pretrained transformer. CPU-only sentiment training does not make
  the full pipeline transformer-free, and IJELID itself required GPU training.

## Citation

The dataset was obtained from:

> Fathanick. *Code-mixed Sentiment Analysis IJE*.
> https://github.com/fathanick/Code-mixed-Sentiment-analysis-IJE

The comparison refers to:

> Hidayatullah, A. F. (2024). *Code-Mixed Sentiment Analysis on
> Indonesian-Javanese-English Text Using Transformer Models*.

Cite the source dataset, this repository, and the associated publication once
its final bibliographic details are available.

## License and contributions

No software or dataset license has yet been declared. Unless a license is added
by the copyright holder, reuse and redistribution rights are not granted
automatically.

Contributions are welcome through issues and pull requests. A contribution
should describe the change, preserve the fixed test split, document new
dependencies, and include enough commands and random-seed information to
reproduce new results. Do not commit credentials, virtual environments,
generated run directories, or Python bytecode.
