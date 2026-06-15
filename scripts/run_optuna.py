"""
run_optuna.py
=============
Optuna + scikit-learn experiment runner for IJE Code-Mixed Sentiment Analysis.
Replaces Auto-sklearn (not available on macOS/Python 3.11+).

Searches over a broad set of sklearn classifiers and their hyperparameters
using Optuna's TPE sampler, optimising weighted F1 via 5-fold stratified CV.

Runs on three feature sets: tfidf, tfidf_cm, full.
Saves results to outputs/results_optuna.json.

Usage:
    python run_optuna.py [--time 600] [--trials 100] [--fs tfidf tfidf_cm full]
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
        accuracy_score, f1_score, precision_score,
        recall_score, classification_report,
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


def build_classifier(trial):
    """Define the search space over sklearn classifiers and hyperparameters."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier,
    )
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    clf_name = trial.suggest_categorical("classifier", [
        "logreg_l1", "logreg_l2", "linearsvc",
        "sgd", "random_forest", "extra_trees",
        "hist_gbm",
    ])

    if clf_name == "logreg_l1":
        C = trial.suggest_float("lr_l1_C", 1e-3, 1e2, log=True)
        return LogisticRegression(penalty="l1", C=C, solver="saga",
                                  max_iter=2000, n_jobs=-1, random_state=42)

    elif clf_name == "logreg_l2":
        C = trial.suggest_float("lr_l2_C", 1e-3, 1e2, log=True)
        return LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                  max_iter=2000, n_jobs=-1, random_state=42)

    elif clf_name == "linearsvc":
        C = trial.suggest_float("svc_C", 1e-3, 1e2, log=True)
        clf = LinearSVC(C=C, max_iter=3000, random_state=42)
        return CalibratedClassifierCV(clf, cv=3)

    elif clf_name == "sgd":
        loss = trial.suggest_categorical("sgd_loss", ["hinge", "modified_huber", "log_loss"])
        alpha = trial.suggest_float("sgd_alpha", 1e-6, 1e-1, log=True)
        return SGDClassifier(loss=loss, alpha=alpha, max_iter=1000,
                             n_jobs=-1, random_state=42)

    elif clf_name == "random_forest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        max_depth = trial.suggest_int("rf_max_depth", 5, 30)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1, random_state=42,
        )

    elif clf_name == "extra_trees":
        n_estimators = trial.suggest_int("et_n_estimators", 50, 300)
        max_depth = trial.suggest_int("et_max_depth", 5, 30)
        return ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=42,
        )

    else:  # hist_gbm
        learning_rate = trial.suggest_float("hgb_lr", 1e-3, 0.3, log=True)
        max_iter = trial.suggest_int("hgb_max_iter", 50, 300)
        max_depth = trial.suggest_int("hgb_max_depth", 3, 10)
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter,
            max_depth=max_depth, random_state=42,
        )


# ---------------------------------------------------------------------------
# Run Optuna on one feature set
# ---------------------------------------------------------------------------

def run_optuna_one(
    X_train, y_train,
    X_test, y_test,
    time_budget: int,
    n_trials: int,
    feature_set_name: str,
):
    import optuna
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Convert sparse to dense (needed for tree-based models)
    if issparse(X_train):
        print(f"  Converting sparse to dense (shape: {X_train.shape})…")
        X_train = X_train.toarray().astype(np.float32)
        X_test = X_test.toarray().astype(np.float32)

    print(f"\n  [Optuna | {feature_set_name}] Starting search "
          f"(time_budget={time_budget}s, max_trials={n_trials})…")

    start = time.time()
    deadline = start + time_budget

    def objective(trial):
        if time.time() > deadline:
            raise optuna.exceptions.TrialPruned()
        clf = build_classifier(trial)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=skf, scoring="f1_weighted", n_jobs=-1,
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=time_budget,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    print(f"  Search time: {elapsed:.1f}s  |  Trials completed: {len(study.trials)}")

    # Refit best model on full training set and evaluate on test
    best_params = study.best_params
    best_trial = study.best_trial
    best_cv_f1 = float(study.best_value)

    # Rebuild best classifier from best params
    best_clf = build_classifier(best_trial)
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print(f"  Best CV F1:   {best_cv_f1:.4f}")
    print(f"  Test F1:      {metrics['f1_weighted']:.4f}")
    print(f"  Best model:   {best_params.get('classifier', 'unknown')}")

    # Trial summary
    trials_summary = [
        {
            "number": t.number,
            "value": t.value,
            "classifier": t.params.get("classifier"),
        }
        for t in study.trials
        if t.value is not None
    ]
    # Top 5 by CV F1
    top5 = sorted(trials_summary, key=lambda x: x["value"] or 0, reverse=True)[:5]

    return {
        "framework": "optuna+sklearn",
        "feature_set": feature_set_name,
        "training_time_s": elapsed,
        "n_trials_completed": len(study.trials),
        "test_metrics": metrics,
        "best_cv_f1": best_cv_f1,
        "best_params": best_params,
        "top5_trials": top5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(time_budget: int = 600, n_trials: int = 100, feature_sets: list = None):
    sys.path.insert(0, str(SCRIPT_DIR))

    run_sets = feature_sets if feature_sets else FEATURE_SETS

    print("=" * 60)
    print(f"Optuna+sklearn Experiment Runner — feature sets: {run_sets}")
    print("=" * 60)

    try:
        features, labels = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run features.py first.")
        sys.exit(1)

    from scipy.sparse import vstack

    # Load existing results to allow resuming
    out_path = OUTPUTS_DIR / "results_optuna.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["feature_set"] for r in existing}
    else:
        existing = []
        done = set()

    all_results = list(existing)

    for fs_name in run_sets:
        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")

        if fs_name in done:
            print(f"  Skipping {fs_name} (already completed).")
            continue

        X_train_full = vstack([features[fs_name]["train"], features[fs_name]["valid"]])
        y_train_full = np.concatenate([labels["train"], labels["valid"]])
        X_test = features[fs_name]["test"]
        y_test = labels["test"]

        try:
            result = run_optuna_one(
                X_train_full, y_train_full,
                X_test, y_test,
                time_budget=time_budget,
                n_trials=n_trials,
                feature_set_name=fs_name,
            )
            if result:
                all_results.append(result)
        except Exception:
            print(f"  ERROR running Optuna on {fs_name}:")
            traceback.print_exc()

        # Save after each feature set
        out_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("OPTUNA+SKLEARN RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Feature Set':<15} {'F1 (weighted)':<16} {'Accuracy':<12} {'Best Model':<20} {'Trials':<8} {'Time (s)'}")
    print("-" * 85)
    for r in all_results:
        m = r["test_metrics"]
        best_clf = r["best_params"].get("classifier", "unknown")
        print(
            f"{r['feature_set']:<15} {m['f1_weighted']:.4f}           "
            f"{m['accuracy']:.4f}       {best_clf:<20} {r['n_trials_completed']:<8} {r['training_time_s']:.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=600, help="Time budget per feature set (s)")
    parser.add_argument("--trials", type=int, default=100, help="Max Optuna trials per feature set")
    parser.add_argument("--fs", nargs="+", choices=FEATURE_SETS, default=None,
                        help="Which feature sets to run (default: all)")
    args = parser.parse_args()
    main(time_budget=args.time, n_trials=args.trials, feature_sets=args.fs)
