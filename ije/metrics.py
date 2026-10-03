"""One set of explicit metrics for every framework and every stage."""
import numpy as np


def metrics(y, pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0 or not set(y) <= {0, 1, 2} or not set(pred) <= {0, 1, 2}:
        raise ValueError("Invalid or empty predictions")
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "precision_macro": float(p.mean()), "recall_macro": float(r.mean()),
            "f1_macro": float(f.mean()),
            "f1_weighted": float(f1_score(y, pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])}
                          for i, name in enumerate(("negative", "neutral", "positive"))},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def bootstrap(y, pred, replicates, seed, other=None):
    """Stratified paired bootstrap, fixed-model uncertainty, not selection correction."""
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in (0, 1, 2)]
    values = []
    for _ in range(replicates):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        value = f1_score(y[ix], pred[ix], labels=[0, 1, 2], average="macro", zero_division=0)
        if other is not None:
            value -= f1_score(y[ix], np.asarray(other)[ix], labels=[0, 1, 2], average="macro", zero_division=0)
        values.append(float(value))
    return {"lower_95": float(np.quantile(values, .025)),
            "upper_95": float(np.quantile(values, .975)), "replicates": replicates,
            "method": "stratified_percentile_bootstrap",
            "scope": "conditional_on_fitted_models_and_observed_test_class_counts",
            "caution": "does_not_correct_prior_test_exposure_or_multiple_comparisons"}
