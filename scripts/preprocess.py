"""
preprocess.py
=============
Download, clean, and prepare the IJE Code-Mixed Sentiment dataset.

Dataset source:
    https://github.com/fathanick/Code-mixed-Sentiment-analysis-IJE/tree/main/dataset

Usage:
    python preprocess.py
"""

import os
import re
import json
import sys
import pathlib
import requests
import pandas as pd
import numpy as np
from io import BytesIO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# GitHub raw URLs — XLSX files (actual repo structure)
# ---------------------------------------------------------------------------
BASE_RAW = "https://raw.githubusercontent.com/fathanick/Code-mixed-Sentiment-analysis-IJE/main/dataset"

SPLIT_URL_CANDIDATES = {
    "train": [
        f"{BASE_RAW}/train_set.xlsx",
    ],
    "valid": [
        f"{BASE_RAW}/validation_set.xlsx",
    ],
    "test": [
        f"{BASE_RAW}/test_set.xlsx",
    ],
}

LABEL_NORMALISATION = {
    # common string → canonical
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "2": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "0": "neutral",
    "negative": "negative",
    "neg": "negative",
    "-1": "negative",
}

CANONICAL_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(CANONICAL_LABELS)}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single tweet.

    Steps applied in order:
    1. Remove URLs (http/https/www)
    2. Remove @mentions
    3. Remove the '#' character but keep the following word
    4. Lowercase
    5. Normalise whitespace
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)
    # Strip hashtag symbol but keep word
    text = re.sub(r"#(\w+)", r"\1", text)
    # Lowercase
    text = text.lower()
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def try_download(urls: list[str]) -> bytes | None:
    """Try each URL in *urls* and return the binary content of the first successful one."""
    for url in urls:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                print(f"  Downloaded: {url}")
                return resp.content
        except requests.RequestException:
            pass
    return None


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (text_col, label_col) by inspecting column names heuristically."""
    lower_cols = {c.lower(): c for c in df.columns}

    text_col = None
    for candidate in ["text", "tweet", "content", "sentence", "kalimat"]:
        if candidate in lower_cols:
            text_col = lower_cols[candidate]
            break
    if text_col is None:
        # pick the column with the longest average string length
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in str_cols}
            text_col = max(avg_lens, key=avg_lens.get)

    label_col = None
    for candidate in ["label", "sentiment", "class", "category", "polaritas"]:
        if candidate in lower_cols and lower_cols[candidate] != text_col:
            label_col = lower_cols[candidate]
            break
    if label_col is None:
        remaining = [c for c in df.columns if c != text_col]
        if remaining:
            label_col = remaining[0]

    return text_col, label_col


def normalise_label(raw: str) -> str | None:
    """Map a raw label string to a canonical label."""
    raw_str = str(raw).strip().lower()
    return LABEL_NORMALISATION.get(raw_str, None)


def load_split(split: str) -> pd.DataFrame | None:
    """Download or load from cache a split XLSX and return a normalised DataFrame."""
    cache_path = DATA_DIR / f"{split}_raw.xlsx"

    if cache_path.exists():
        print(f"  [{split}] Loading from cache: {cache_path}")
        raw_bytes = cache_path.read_bytes()
    else:
        print(f"  [{split}] Trying to download…")
        raw_bytes = try_download(SPLIT_URL_CANDIDATES[split])
        if raw_bytes is None:
            print(f"  [{split}] WARNING: could not download any URL for split '{split}'.")
            return None
        cache_path.write_bytes(raw_bytes)

    # Parse XLSX
    df = None
    try:
        df = pd.read_excel(BytesIO(raw_bytes), engine="openpyxl")
    except Exception as e:
        print(f"  [{split}] ERROR: could not parse XLSX: {e}")
        return None

    if df is None or len(df.columns) < 2:
        print(f"  [{split}] ERROR: unexpected file format.")
        return None

    text_col, label_col = detect_columns(df)
    if text_col is None or label_col is None:
        print(f"  [{split}] ERROR: could not identify text/label columns. Columns: {df.columns.tolist()}")
        return None

    print(f"  [{split}] Detected columns — text: '{text_col}', label: '{label_col}'")

    out = pd.DataFrame()
    out["text_raw"] = df[text_col].astype(str)
    out["text"] = out["text_raw"].apply(clean_text)
    out["label_raw"] = df[label_col].astype(str)
    out["label"] = out["label_raw"].apply(normalise_label)

    before = len(out)
    out = out.dropna(subset=["label"])
    after = len(out)
    if before != after:
        print(f"  [{split}] Dropped {before - after} rows with unrecognised labels.")

    # Drop empty texts
    out = out[out["text"].str.strip().ne("")]
    out = out.reset_index(drop=True)
    out["label_id"] = out["label"].map(LABEL2ID)
    out["split"] = split
    return out


# ---------------------------------------------------------------------------
# Fallback: synthesise valid split from train if not available
# ---------------------------------------------------------------------------

def split_train_valid(train_df: pd.DataFrame, valid_frac: float = 0.1, seed: int = 42):
    """Carve out a validation set from train when the valid split is unavailable."""
    from sklearn.model_selection import train_test_split

    train_new, valid_new = train_test_split(
        train_df, test_size=valid_frac, stratify=train_df["label_id"], random_state=seed
    )
    train_new = train_new.copy()
    valid_new = valid_new.copy()
    train_new["split"] = "train"
    valid_new["split"] = "valid"
    return train_new.reset_index(drop=True), valid_new.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, split_name: str) -> dict:
    counts = df["label"].value_counts().to_dict()
    return {
        "split": split_name,
        "n_samples": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
    }


def print_dataset_stats(splits: dict[str, pd.DataFrame]) -> None:
    """Pretty-print dataset statistics across all splits."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    all_counts: dict[str, dict] = {}
    for split_name, df in splits.items():
        stats = compute_stats(df, split_name)
        all_counts[split_name] = stats
        print(f"\nSplit: {split_name.upper()}  ({stats['n_samples']} samples)")
        for lbl in CANONICAL_LABELS:
            n = stats["class_counts"].get(lbl, 0)
            pct = 100.0 * n / max(stats["n_samples"], 1)
            print(f"  {lbl:10s}: {n:5d}  ({pct:.1f}%)")

    # Totals
    total = sum(s["n_samples"] for s in all_counts.values())
    print(f"\n{'Total':10s}: {total:5d}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, pd.DataFrame]:
    print("=" * 60)
    print("Preprocessing IJE Code-Mixed Sentiment Dataset")
    print("=" * 60)

    splits_raw: dict[str, pd.DataFrame | None] = {}
    for split in ["train", "valid", "test"]:
        print(f"\n[{split.upper()}]")
        splits_raw[split] = load_split(split)

    # If valid is missing, carve from train
    if splits_raw["valid"] is None and splits_raw["train"] is not None:
        print("\nValidation split unavailable — carving 10% from train.")
        splits_raw["train"], splits_raw["valid"] = split_train_valid(splits_raw["train"])

    # If test is missing, carve from train
    if splits_raw["test"] is None and splits_raw["train"] is not None:
        print("\nTest split unavailable — carving 15% from train.")
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(
            splits_raw["train"],
            test_size=0.15,
            stratify=splits_raw["train"]["label_id"],
            random_state=42,
        )
        splits_raw["train"] = tr.reset_index(drop=True)
        splits_raw["test"] = te.reset_index(drop=True)
        splits_raw["test"]["split"] = "test"

    splits: dict[str, pd.DataFrame] = {k: v for k, v in splits_raw.items() if v is not None}

    # Save cleaned CSVs
    for split_name, df in splits.items():
        out_path = DATA_DIR / f"{split_name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved {split_name}: {out_path}  ({len(df)} rows)")

    # Build meta.json
    split_stats = {name: compute_stats(df, name) for name, df in splits.items()}
    total_counts: dict[str, int] = {}
    for lbl in CANONICAL_LABELS:
        total_counts[lbl] = sum(
            s["class_counts"].get(lbl, 0) for s in split_stats.values()
        )
    total_n = sum(total_counts.values())

    meta = {
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "canonical_labels": CANONICAL_LABELS,
        "splits": split_stats,
        "total": {
            "n_samples": int(total_n),
            "class_counts": {k: int(v) for k, v in total_counts.items()},
        },
        "source": "https://github.com/fathanick/Code-mixed-Sentiment-analysis-IJE",
        "preprocessing": {
            "remove_urls": True,
            "remove_mentions": True,
            "strip_hashtag_symbol": True,
            "lowercase": True,
            "normalise_whitespace": True,
        },
    }
    meta_path = DATA_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved meta: {meta_path}")

    print_dataset_stats(splits)
    return splits


def load_data() -> dict[str, pd.DataFrame]:
    """Load pre-processed splits from disk (run main() first)."""
    splits = {}
    for split in ["train", "valid", "test"]:
        path = DATA_DIR / f"{split}.csv"
        if path.exists():
            splits[split] = pd.read_csv(path, encoding="utf-8")
        else:
            raise FileNotFoundError(
                f"Split '{split}' not found at {path}. Run preprocess.py first."
            )
    return splits


if __name__ == "__main__":
    main()
