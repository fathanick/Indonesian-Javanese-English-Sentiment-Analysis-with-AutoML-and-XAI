"""Data loading. Development functions never open the test split."""
import hashlib
import re
from pathlib import Path


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def source_dir(config, config_path):
    return (Path(config_path).parent / config["source_data"]).resolve()


def load_split(config, config_path, split):
    import pandas as pd
    df = pd.read_csv(source_dir(config, config_path) / f"{split}.csv", keep_default_na=False)
    required = {"text_raw", "text", "label_id"}
    if not required <= set(df.columns):
        raise ValueError(f"{split}: missing fields {required - set(df.columns)}")
    if len(df) != config["expected_rows"][split]:
        raise ValueError(f"{split}: unexpected sample count")
    if df["text_raw"].eq("").any() or df["text"].str.strip().eq("").any():
        raise ValueError(f"{split}: missing/empty text")
    if not df["label_id"].isin([0, 1, 2]).all():
        raise ValueError(f"{split}: invalid labels")
    if "label" in df:
        mapping = {name: i for i, name in enumerate(config["labels"])}
        mapped = df["label"].astype(str).str.strip().str.lower().map(mapping)
        if mapped.isna().any() or not mapped.eq(df["label_id"]).all():
            raise ValueError(f"{split}: label names and numeric IDs disagree")
    if "split" in df and not df["split"].eq(split).all():
        raise ValueError(f"{split}: records claim another source partition")
    if not df["text_raw"].map(clean_text).eq(df["text"]).all():
        raise ValueError(f"{split}: supplied cleaned text differs from the documented preprocessing")
    df["label_id"] = df["label_id"].astype(int)
    df["sample_id"] = [f"{split}:{i:05d}" for i in range(len(df))]
    df["group"] = df["text"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return df


def development(config, config_path):
    import pandas as pd
    return pd.concat([load_split(config, config_path, "train"),
                      load_split(config, config_path, "valid")], ignore_index=True)


def grouped_folds(frame, n_splits, seed):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(frame)), frame.label_id, frame.group))
    for train, valid in folds:
        if set(frame.iloc[train].group) & set(frame.iloc[valid].group):
            raise ValueError("A duplicate group crossed a fold boundary")
        if set(frame.iloc[train].label_id) != {0, 1, 2} or set(frame.iloc[valid].label_id) != {0, 1, 2}:
            raise ValueError("A fold lacks a sentiment class")
    return folds


def inner_partition(frame, config, seed):
    return grouped_folds(frame, config["inner_folds"], seed)[config["inner_holdout_fold"]]


def preflight(config, config_path, run_dir):
    """No ML models loaded; test inspection here is restricted to integrity checks."""
    import importlib.metadata
    import platform
    from .storage import digest, write_json
    splits = {s: load_split(config, config_path, s) for s in ("train", "valid", "test")}
    report = {"rows": {}, "duplicates": {}, "overlaps": {}, "preprocessing_examples": [],
              "python": platform.python_version(), "platform": platform.platform(),
              "source_hashes": {}, "dependencies": {}}
    for s, df in splits.items():
        report["rows"][s] = {"count": len(df), "class_counts": {str(k): int(v) for k, v in df.label_id.value_counts().items()}}
        report["duplicates"][s] = df.loc[df.duplicated("text", keep=False), "sample_id"].tolist()
        report["source_hashes"][s] = digest(source_dir(config, config_path) / f"{s}.csv")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        report["overlaps"][f"{a}/{b}"] = {
            "kind": "exact_cleaned_text_match_not_fuzzy_near_duplicate",
            "rows_in_second_split": splits[b].loc[splits[b].group.isin(splits[a].group), "sample_id"].tolist()}
    # Examples chosen from development data only.
    patterns = {"hashtag": r"#\w+", "mention": r"@\w+", "url": r"https?://|www\.",
                "repetition": r"(.)\1\1", "digits": r"\d", "punctuation": r"[!?]"}
    for category, pattern in patterns.items():
        candidates = splits["train"][splits["train"].text_raw.str.contains(pattern, regex=True)]
        if len(candidates):
            row = candidates.iloc[0]
            report["preprocessing_examples"].append({"category": category, "sample_id": row.sample_id,
                                                     "before": row.text_raw, "after": row.text})
    from .storage import ROOT
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, required = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != required:
            raise RuntimeError(f"Dependency {name}: expected {required}, found {actual}")
        report["dependencies"][name] = actual
    report["disclosures"] = config["disclosures"]
    path = Path(run_dir) / "preflight.json"
    write_json(path, report)
    return [path]


def verify_sources(config, config_path, run_dir, include_test=False):
    from .storage import read_json, digest
    expected = read_json(Path(run_dir) / "preflight.json")["source_hashes"]
    for split in (("train", "valid", "test") if include_test else ("train", "valid")):
        if digest(source_dir(config, config_path) / f"{split}.csv") != expected[split]:
            raise RuntimeError(f"Source {split} changed after preflight")
