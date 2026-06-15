# Indonesian-Javanese-English Sentiment Analysis with AutoML and XAI

## Description

This repository contains the dataset and reproducible Python pipeline used to
study sentiment classification in Indonesian-Javanese-English (IJE) code-mixed
text. The pipeline compares multiple AutoML approaches and explains model
predictions with LIME.

The task is three-class sentiment classification:

- `negative` (`label_id = 0`)
- `neutral` (`label_id = 1`)
- `positive` (`label_id = 2`)

The best configuration recorded in the experiment report was
Optuna + scikit-learn with TF-IDF features, with a weighted test F1 of `0.9160`
and accuracy of `0.9158`.

## Repository Structure

```text
.
├── data/
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   ├── train_raw.xlsx
│   ├── valid_raw.xlsx
│   ├── test_raw.xlsx
│   └── meta.json
├── scripts/
│   ├── preprocess.py
│   ├── features.py
│   ├── run_optuna.py
│   ├── run_flaml.py
│   ├── run_autogluon.py
│   ├── run_autosklearn.py
│   ├── evaluate.py
│   ├── lime_analysis.py
│   ├── visualize_results.py
│   ├── compare_prior_study.py
│   └── generate_report.py
└── requirements.txt
```

Generated models, feature matrices, result files, reports, and figures are
written to `outputs/`, which is excluded from version control.

## Dataset Information

The dataset contains 1,929 code-mixed social-media texts in Indonesian,
Javanese, and English. The source dataset reports agreement between two
annotators with Cohen's kappa of `0.9767`.

| Split | Samples | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 1,350 | 461 | 437 | 452 |
| Validation | 116 | 40 | 36 | 40 |
| Test | 463 | 143 | 163 | 157 |
| Total | 1,929 | 644 | 636 | 649 |

The processed CSV files contain:

| Column | Description |
|---|---|
| `text_raw` | Original text |
| `text` | Preprocessed text used for modeling |
| `label_raw` | Original sentiment label |
| `label` | Canonical label: negative, neutral, or positive |
| `label_id` | Numeric class identifier |
| `split` | Train, validation, or test assignment |

`data/meta.json` records label mappings, split statistics, the source URL, and
the preprocessing settings.

## Materials and Methods

### Data Preprocessing

`scripts/preprocess.py` downloads or loads the original XLSX splits, detects the
text and label columns, and applies these operations in order:

1. Remove HTTP, HTTPS, and `www` URLs.
2. Remove `@mentions`.
3. Remove the `#` symbol while retaining the hashtag word.
4. Convert text to lowercase.
5. Normalize repeated whitespace.
6. Map labels to `negative`, `neutral`, and `positive`.
7. Remove rows with unsupported labels or empty cleaned text.

If an official validation or test split is unavailable, the script creates a
stratified split with `random_state=42`. The files currently included in
`data/` are the fixed splits used in the reported experiments.

### Feature Engineering

`scripts/features.py` fits preprocessing components on training data and
creates three feature sets:

1. **TF-IDF (`tfidf`)**: word n-grams `(1, 3)`, character n-grams `(2, 4)`,
   and scaled text statistics. Each TF-IDF vectorizer is limited to 10,000
   features and uses `min_df=2` and sublinear term frequency.
2. **TF-IDF + code-mixing (`tfidf_cm`)**: the TF-IDF features plus
   dictionary- and heuristic-based Indonesian, Javanese, and English ratios,
   Code-Mixing Index, language switch points, and dominant-language indicators.
3. **Full (`full`)**: all preceding features plus 384-dimensional sentence
   embeddings from
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

The text statistics include word count, character count, average word length,
punctuation count and density, digit count, and uppercase-word count.

### Algorithms and Implementation

| Script | Algorithm or role |
|---|---|
| `run_optuna.py` | Optuna TPE search over logistic regression, calibrated linear SVM, SGD, random forest, extra trees, and histogram gradient boosting |
| `run_flaml.py` | FLAML search over LightGBM, XGBoost, random forest, extra trees, and L1/L2 logistic regression |
| `run_autogluon.py` | AutoGluon Tabular model selection and ensembling |
| `run_autosklearn.py` | Auto-sklearn model and preprocessing search; intended primarily for Linux |
| `evaluate.py` | Weighted precision, recall, F1, accuracy, per-class F1, ranking, ablation, and available significance tests |
| `lime_analysis.py` | Local Interpretable Model-agnostic Explanations, global aggregation, language contribution analysis, errors, and stability |
| `visualize_results.py` | Publication-quality result and explanation figures |
| `compare_prior_study.py` | Comparison figures for AutoML and prior transformer results |
| `generate_report.py` | Builds the Markdown experiment report from saved outputs |

The AutoML experiments optimize F1 through five-fold stratified
cross-validation. Training and validation are combined for the final FLAML,
Optuna, and auto-sklearn searches; the held-out test split is used only for
final evaluation. Random seeds are set to `42` where supported.

## Requirements

The reported environment used Python `3.11.15`. A Python 3.11 virtual
environment is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install optional frameworks only when needed:

```bash
# AutoGluon
python -m pip install autogluon.tabular

# auto-sklearn is best installed on a supported Linux environment
python -m pip install auto-sklearn
```

Auto-sklearn was not installed in the reported macOS experiment. AutoGluon,
FLAML, and Optuna were run on CPU only. The first full-feature extraction run
downloads the multilingual MiniLM model.

## Reproducibility and Usage

Run commands from the repository root.

### 1. Use or rebuild the dataset

The processed and raw data are already included. To rebuild the CSV files from
the original source:

```bash
python scripts/preprocess.py
```

### 2. Extract features

```bash
python scripts/features.py
```

This creates sparse feature matrices, labels, fitted extractors, and feature
metadata in `outputs/`.

### 3. Run AutoML experiments

The following examples use shorter budgets for a verification run:

```bash
python scripts/run_optuna.py --time 600 --trials 100
python scripts/run_flaml.py --time 1200
python scripts/run_autogluon.py --time 3600 --preset good_quality
```

Optional Linux-only auto-sklearn run:

```bash
python scripts/run_autosklearn.py --time 3600 --per-run-time 360
```

To run one feature set with Optuna or FLAML:

```bash
python scripts/run_optuna.py --time 600 --trials 100 --fs tfidf
python scripts/run_flaml.py --time 1200 --fs tfidf_cm
```

### 4. Evaluate and explain results

```bash
python scripts/evaluate.py
python scripts/lime_analysis.py
python scripts/visualize_results.py
python scripts/compare_prior_study.py
python scripts/generate_report.py
```

Some reporting scripts require result JSON files from the preceding experiment
steps. Runtime and exact selected models can vary with operating system,
processor, library version, and AutoML time budget.

### Reported Environment

- Hardware: Apple M4 Pro, 12 CPU cores, 24 GB unified memory
- Operating system: macOS 26.2 arm64
- Compute: CPU only; GPU/MPS was not enabled
- Main libraries: pandas 2.3.3, scikit-learn 1.7.2, FLAML 2.5.0,
  sentence-transformers 5.2.3, NumPy 2.3.5, SciPy 1.16.3, PyTorch 2.10.0

## Citation

The dataset was obtained from:

> Fathanick. *Code-mixed Sentiment Analysis IJE*.  
> https://github.com/fathanick/Code-mixed-Sentiment-analysis-IJE

The comparison script also refers to:

> Hidayatullah, A. F. (2024). *Code-Mixed Sentiment Analysis on
> Indonesian-Javanese-English Text Using Transformer Models*.

When using this repository in research, cite the source dataset, this
repository, and the associated publication once its final bibliographic details
are available.

## License and Contributions

No software or dataset license has yet been declared in this repository.
Unless a license is added by the copyright holder, reuse and redistribution
rights are not granted automatically.

Contributions are welcome through GitHub issues and pull requests. A
contribution should describe the change, preserve the fixed test split, document
new dependencies, and include enough commands and random-seed information to
reproduce new results. Do not commit credentials, virtual environments,
generated `outputs/`, or Python bytecode.

