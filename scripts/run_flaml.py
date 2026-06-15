"""
run_flaml.py
============
FLAML experiment runner for IJE Code-Mixed Sentiment Analysis.

Runs FLAML on three feature sets with 5-fold stratified cross-validation.
Saves results to outputs/results_flaml.json.

Usage:
    python run_flaml.py [--time 3600]
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


def cv_evaluate(estimator_cls, X, y, n_splits=5, seed=42):
    """Manual 5-fold CV for the best FLAML estimator."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_f1 = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf = estimator_cls()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        fold_f1.append(f1)
        print(f"    Fold {fold_idx + 1}: F1={f1:.4f}")
    return fold_f1


# ---------------------------------------------------------------------------
# Run FLAML on one feature set
# ---------------------------------------------------------------------------

def run_flaml_one(
    X_train, y_train,
    X_test, y_test,
    time_budget: int,
    feature_set_name: str,
):
    try:
        from flaml import AutoML
    except ImportError:
        print("  FLAML not installed. Skipping.")
        return None

    # FLAML works best with dense arrays for tabular tasks
    from scipy.sparse import issparse
    if issparse(X_train):
        print(f"  Converting sparse to dense (shape: {X_train.shape})…")
        X_train = X_train.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    print(f"\n  [FLAML | {feature_set_name}] Fitting (time_budget={time_budget}s)…")
    start = time.time()

    log_file = str(OUTPUTS_DIR / f"flaml_{feature_set_name}.log")

    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": "macro_f1",        # multiclass: use macro_f1 (not "f1" which is binary-only)
        "task": "classification",
        "estimator_list": [
            "lgbm", "xgboost", "xgb_limitdepth",
            "rf", "extra_tree", "lrl1", "lrl2",
        ],
        "log_file_name": log_file,
        "seed": 42,
        "n_jobs": -1,
        "eval_method": "cv",
        "n_splits": 5,
        "verbose": 1,
    }

    automl.fit(X_train, y_train, **settings)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    # Test set evaluation
    y_pred = automl.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # FLAML internals
    best_estimator = str(automl.best_estimator)
    best_config = automl.best_config
    # FLAML minimizes 1 - metric, so best metric = 1 - best_loss
    best_cv_f1 = float(1.0 - automl.best_loss) if automl.best_loss is not None else None

    best_config_per_estimator = {}
    try:
        for est, info in automl.best_config_per_estimator.items():
            best_config_per_estimator[est] = str(info)
    except Exception:
        pass

    result = {
        "framework": "flaml",
        "feature_set": feature_set_name,
        "training_time_s": elapsed,
        "test_metrics": metrics,
        "best_estimator": best_estimator,
        "best_config": best_config,
        "best_cv_f1": best_cv_f1,
        "best_config_per_estimator": best_config_per_estimator,
        "log_file": log_file,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(time_budget: int = 1200, feature_sets: list[str] | None = None):
    sys.path.insert(0, str(SCRIPT_DIR))

    run_sets = feature_sets if feature_sets else FEATURE_SETS

    print("=" * 60)
    print(f"FLAML Experiment Runner — feature sets: {run_sets}")
    print("=" * 60)

    try:
        features, labels = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run features.py first.")
        sys.exit(1)

    # Load existing results to avoid re-running completed sets
    out_path = OUTPUTS_DIR / "results_flaml.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["feature_set"] for r in existing}
    else:
        existing = []
        done = set()

    from scipy.sparse import vstack

    all_results = list(existing)

    for fs_name in run_sets:
        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")

        # Combine train + valid
        X_train_full = vstack([features[fs_name]["train"], features[fs_name]["valid"]])
        y_train_full = np.concatenate([labels["train"], labels["valid"]])
        X_test = features[fs_name]["test"]
        y_test = labels["test"]

        if fs_name in done:
            print(f"  Skipping {fs_name} (already completed).")
            continue

        try:
            result = run_flaml_one(
                X_train_full, y_train_full,
                X_test, y_test,
                time_budget=time_budget,
                feature_set_name=fs_name,
            )
            if result:
                all_results.append(result)
                print(f"\n  Test F1 (weighted): {result['test_metrics']['f1_weighted']:.4f}")
                print(f"  Best estimator:     {result['best_estimator']}")
                print(f"  Best CV F1:         {result['best_cv_f1']}")
        except Exception:
            print(f"  ERROR running FLAML on {fs_name}:")
            traceback.print_exc()

    # Save results
    out_path = OUTPUTS_DIR / "results_flaml.json"
    out_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("FLAML RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Feature Set':<15} {'F1 (weighted)':<16} {'Accuracy':<12} {'Best Model':<20} {'Time (s)'}")
    print("-" * 75)
    for r in all_results:
        m = r["test_metrics"]
        print(
            f"{r['feature_set']:<15} {m['f1_weighted']:.4f}           "
            f"{m['accuracy']:.4f}       {r['best_estimator']:<20} {r['training_time_s']:.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=1200, help="Time budget per feature set (s)")
    parser.add_argument("--fs", nargs="+", choices=FEATURE_SETS, default=None,
                        help="Which feature sets to run (default: all)")
    args = parser.parse_args()
    main(time_budget=args.time, feature_sets=args.fs)
