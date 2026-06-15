"""
run_autogluon.py
================
AutoGluon experiment runner for IJE Code-Mixed Sentiment Analysis.

Runs AutoGluon on three feature sets (pre-extracted features as tabular DataFrames).
Saves results to outputs/results_autogluon.json.

Usage:
    python run_autogluon.py [--time 3600]
"""

import argparse
import json
import pathlib
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_LABELS = ["negative", "neutral", "positive"]
FEATURE_SETS = ["tfidf", "tfidf_cm", "full"]
LABEL_COL = "sentiment"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    import scipy.sparse as sp

    features = {}
    for fs in FEATURE_SETS:
        features[fs] = {}
        for split in ["train", "valid", "test"]:
            path = OUTPUTS_DIR / f"X_{fs}_{split}.npz"
            features[fs][split] = sp.load_npz(str(path))

    labels = {}
    for split in ["train", "valid", "test"]:
        labels[split] = np.load(str(OUTPUTS_DIR / f"y_{split}.npy"))

    return features, labels


def sparse_to_df(X, y=None, label_col=LABEL_COL):
    """Convert a sparse matrix to a pandas DataFrame for AutoGluon."""
    # AutoGluon needs dense DataFrames; dataset is small enough
    df = pd.DataFrame(X.toarray(), columns=[f"f{i}" for i in range(X.shape[1])])
    if y is not None:
        df[label_col] = [CANONICAL_LABELS[yi] for yi in y]
    return df


def compute_metrics(y_true_labels, y_pred_labels):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )
    # Convert string labels to ids for sklearn
    label2id = {l: i for i, l in enumerate(CANONICAL_LABELS)}
    y_true = [label2id[l] for l in y_true_labels]
    y_pred = [label2id.get(l, 0) for l in y_pred_labels]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_per_class": {
            lbl: float(v)
            for lbl, v in zip(
                CANONICAL_LABELS,
                f1_score(y_true, y_pred, average=None, zero_division=0),
            )
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=CANONICAL_LABELS, zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Run AutoGluon on one feature set
# ---------------------------------------------------------------------------

def run_autogluon_one(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    time_limit: int,
    feature_set_name: str,
    presets: str = "best_quality",
):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("  autogluon not installed. Skipping.")
        return None

    model_path = str(OUTPUTS_DIR / f"autogluon_{feature_set_name}")

    print(f"\n  [AutoGluon | {feature_set_name}] Preparing DataFrames…")
    train_df = sparse_to_df(X_train, y_train)
    valid_df = sparse_to_df(X_valid, y_valid)
    test_df = sparse_to_df(X_test)

    print(f"  [AutoGluon | {feature_set_name}] Fitting (time_limit={time_limit}s)…")
    start = time.time()

    # In bagged mode (best_quality), train+valid must be merged;
    # use use_bag_holdout=True to keep valid as holdout
    predictor = TabularPredictor(
        label=LABEL_COL,
        eval_metric="f1_weighted",
        path=model_path,
        problem_type="multiclass",
        verbosity=1,
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=valid_df,
        use_bag_holdout=True,
        presets=presets,
        time_limit=time_limit,
        num_cpus=4,
        num_gpus=0,
    )

    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Test evaluation
    y_pred_series = predictor.predict(test_df)
    y_true_labels = [CANONICAL_LABELS[yi] for yi in y_test]
    metrics = compute_metrics(y_true_labels, y_pred_series.tolist())

    # Leaderboard
    try:
        lb = predictor.leaderboard(test_df, silent=True)
        leaderboard_str = lb.to_string()
        best_model = predictor.get_model_best()
    except Exception:
        leaderboard_str = ""
        best_model = "unknown"

    # CV-like evaluation: AutoGluon uses internal bagging; extract fit summary
    try:
        fit_summary = predictor.fit_summary(verbosity=0)
        fit_summary_str = str(fit_summary)
    except Exception:
        fit_summary_str = ""

    result = {
        "framework": "autogluon",
        "feature_set": feature_set_name,
        "training_time_s": elapsed,
        "test_metrics": metrics,
        "best_model": best_model,
        "leaderboard": leaderboard_str,
        "fit_summary": fit_summary_str,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(time_limit: int = 3600, presets: str = "good_quality"):
    sys.path.insert(0, str(SCRIPT_DIR))

    print("=" * 60)
    print("AutoGluon Experiment Runner")
    print("=" * 60)

    try:
        features, labels = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run features.py first.")
        sys.exit(1)

    all_results = []

    for fs_name in FEATURE_SETS:
        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")

        try:
            result = run_autogluon_one(
                features[fs_name]["train"], labels["train"],
                features[fs_name]["valid"], labels["valid"],
                features[fs_name]["test"], labels["test"],
                time_limit=time_limit,
                feature_set_name=fs_name,
                presets=presets,
            )
            if result:
                all_results.append(result)
                print(f"\n  Test F1 (weighted): {result['test_metrics']['f1_weighted']:.4f}")
        except Exception:
            print(f"  ERROR running AutoGluon on {fs_name}:")
            traceback.print_exc()

    # Save results
    out_path = OUTPUTS_DIR / "results_autogluon.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("AUTOGLUON RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Feature Set':<15} {'F1 (weighted)':<16} {'Accuracy':<12} {'Time (s)'}")
    print("-" * 60)
    for r in all_results:
        m = r["test_metrics"]
        print(
            f"{r['feature_set']:<15} {m['f1_weighted']:.4f}           "
            f"{m['accuracy']:.4f}       {r['training_time_s']:.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=3600, help="Time budget per feature set (s)")
    parser.add_argument(
        "--preset",
        default="good_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
        help="AutoGluon preset (default: best_quality)",
    )
    args = parser.parse_args()
    main(time_limit=args.time, presets=args.preset)
