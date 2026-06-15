"""
run_autosklearn.py
==================
Auto-sklearn experiment runner for IJE Code-Mixed Sentiment Analysis.

Runs Auto-sklearn on three feature sets:
  1. tfidf  — TF-IDF only
  2. tfidf_cm — TF-IDF + code-mixing features
  3. full   — TF-IDF + CM + embeddings

For each feature set, performs 5-fold stratified cross-validation and
evaluates on the held-out test set. Saves results to outputs/results_autosklearn.json.

Usage:
    python run_autosklearn.py [--time 3600] [--per-run-time 360]
"""

import argparse
import json
import pathlib
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_LABELS = ["negative", "neutral", "positive"]
FEATURE_SETS = ["tfidf", "tfidf_cm", "full"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load feature matrices and labels from disk."""
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


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )

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


def cv_evaluate(clf, X, y, n_splits=5):
    """Stratified K-fold CV — returns per-fold metrics."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1 = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        fold_f1.append(float(f1))
        print(f"    Fold {fold_idx + 1}: F1={f1:.4f}")
    return fold_f1


# ---------------------------------------------------------------------------
# Run Auto-sklearn on one feature set
# ---------------------------------------------------------------------------

def run_autosklearn_one(
    X_train, y_train, X_test, y_test,
    time_left: int,
    per_run_time: int,
    feature_set_name: str,
):
    try:
        import autosklearn.classification
        import autosklearn.metrics
    except ImportError:
        print("  auto-sklearn not installed. Skipping.")
        return None

    print(f"\n  [Auto-sklearn | {feature_set_name}] Fitting…")
    start = time.time()

    # Convert sparse to dense if very small (auto-sklearn can struggle with sparse)
    # For safety, keep sparse but ensure CSR format
    from scipy.sparse import issparse, csr_matrix
    if issparse(X_train):
        X_train = csr_matrix(X_train, dtype=np.float32)
        X_test = csr_matrix(X_test, dtype=np.float32)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_left,
        per_run_time_limit=per_run_time,
        ensemble_size=50,
        ensemble_nbest=50,
        memory_limit=8192,
        seed=42,
        n_jobs=-1,
        metric=autosklearn.metrics.f1_weighted,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 5},
        initial_configurations_via_metalearning=0,
    )

    automl.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Test set evaluation
    y_pred = automl.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # CV scores from auto-sklearn internals
    try:
        cv_results = automl.cv_results_
        cv_scores = list(cv_results.get("mean_test_score", []))
    except Exception:
        cv_scores = []

    # Leaderboard / model info
    try:
        lb = automl.leaderboard(detailed=True, ensemble_only=True)
        leaderboard_str = lb.to_string()
    except Exception:
        leaderboard_str = str(automl.leaderboard())

    result = {
        "framework": "auto-sklearn",
        "feature_set": feature_set_name,
        "training_time_s": elapsed,
        "test_metrics": metrics,
        "cv_scores_internal": cv_scores,
        "leaderboard": leaderboard_str,
        "best_models": str(automl.show_models()),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(time_left: int = 3600, per_run_time: int = 360):
    sys.path.insert(0, str(SCRIPT_DIR))

    print("=" * 60)
    print("Auto-sklearn Experiment Runner")
    print("=" * 60)

    # Load data
    try:
        features, labels = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run features.py first.")
        sys.exit(1)

    # Combine train + valid for final model (auto-sklearn uses CV internally)
    from scipy.sparse import vstack

    all_results = []

    for fs_name in FEATURE_SETS:
        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")

        X_train_full = vstack([features[fs_name]["train"], features[fs_name]["valid"]])
        y_train_full = np.concatenate([labels["train"], labels["valid"]])
        X_test = features[fs_name]["test"]
        y_test = labels["test"]

        try:
            result = run_autosklearn_one(
                X_train_full, y_train_full, X_test, y_test,
                time_left=time_left,
                per_run_time=per_run_time,
                feature_set_name=fs_name,
            )
            if result:
                all_results.append(result)
                print(f"\n  Test F1 (weighted): {result['test_metrics']['f1_weighted']:.4f}")
        except Exception:
            print(f"  ERROR running auto-sklearn on {fs_name}:")
            traceback.print_exc()

    # Save results
    out_path = OUTPUTS_DIR / "results_autosklearn.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("AUTO-SKLEARN RESULTS SUMMARY")
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
    parser.add_argument("--time", type=int, default=3600, help="Total time budget (s)")
    parser.add_argument("--per-run-time", type=int, default=360, help="Per-model time limit (s)")
    args = parser.parse_args()
    main(time_left=args.time, per_run_time=args.per_run_time)
