"""
lime_analysis.py
================
LIME explanation generation for the best AutoML model on IJE Code-Mixed Sentiment.

What this script does:
  1. Identifies the best-performing model from FLAML / AutoGluon / Auto-sklearn results
  2. Re-trains that model on train+valid using a TF-IDF pipeline (so LIME gets raw text)
  3. Generates per-sample LIME explanations for:
     - Representative correctly-classified samples (2 per class)
     - Misclassified samples (up to 5)
  4. Aggregates LIME weights across test set for global feature importance
  5. Analyses language contributions (Indonesian / Javanese / English)
  6. Checks LIME stability (Jaccard similarity across 10 runs)
  7. Saves all figures (PNG, PDF) to outputs/figures/ and results to outputs/lime_results.json

Usage:
    python lime_analysis.py
"""

import json
import pathlib
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}


# ---------------------------------------------------------------------------
# Identify best model from saved results
# ---------------------------------------------------------------------------

def load_best_result() -> dict:
    """
    Load all AutoML results and return the entry with the highest test F1.
    Falls back to FLAML's best if others are unavailable.
    """
    best = None
    best_f1 = -1.0

    for fname in ["results_flaml.json", "results_autosklearn.json", "results_autogluon.json", "results_optuna.json"]:
        path = OUTPUTS_DIR / fname
        if not path.exists():
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        for r in results:
            f1 = r.get("test_metrics", {}).get("f1_weighted", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best = r

    if best is None:
        raise FileNotFoundError(
            "No AutoML results found. Run run_flaml.py / run_autogluon.py / run_autosklearn.py first."
        )

    print(f"Best model: {best['framework']} | feature_set={best['feature_set']} | F1={best_f1:.4f}")
    return best


# ---------------------------------------------------------------------------
# Re-train best model with TF-IDF pipeline for LIME compatibility
# ---------------------------------------------------------------------------

def build_lime_pipeline(train_texts, train_labels, feature_set_name: str):
    """
    Build a sklearn Pipeline (TF-IDF → best_classifier) so LIME can call
    predict_proba on raw text.  We pick the best single-model estimator
    from FLAML (lgbm / xgboost / rf / extra_tree / logistic) since sklearn
    Pipelines work cleanly with these.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    best_clf = None

    # Try Optuna results first (highest overall F1)
    optuna_path = OUTPUTS_DIR / "results_optuna.json"
    if optuna_path.exists():
        results = json.loads(optuna_path.read_text(encoding="utf-8"))
        for r in results:
            if r["feature_set"] == feature_set_name:
                best_clf = _instantiate_estimator_optuna(r.get("best_params", {}))
                break

    # Fall back to FLAML
    if best_clf is None:
        flaml_path = OUTPUTS_DIR / "results_flaml.json"
        if flaml_path.exists():
            results = json.loads(flaml_path.read_text(encoding="utf-8"))
            for r in results:
                if r["feature_set"] == feature_set_name:
                    est_name = r.get("best_estimator", "")
                    best_config = r.get("best_config", {})
                    best_clf = _instantiate_estimator(est_name, best_config)
                    break

    if best_clf is None:
        print("  Using default LogisticRegression for LIME pipeline.")
        best_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=10000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("clf", best_clf),
    ])

    print(f"  Fitting LIME pipeline ({type(best_clf).__name__})…")
    pipeline.fit(train_texts, train_labels)
    return pipeline


def _instantiate_estimator(name: str, config: dict):
    """Instantiate an sklearn-compatible estimator from a FLAML name + config."""
    try:
        if "lgbm" in name:
            from lightgbm import LGBMClassifier
            safe = {k: v for k, v in config.items() if k in [
                "n_estimators", "num_leaves", "min_child_samples",
                "learning_rate", "colsample_bytree", "reg_alpha", "reg_lambda",
            ]}
            return LGBMClassifier(random_state=42, verbose=-1, **safe)
        elif "xgb" in name:
            from xgboost import XGBClassifier
            safe = {k: v for k, v in config.items() if k in [
                "n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            ]}
            return XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                 random_state=42, verbosity=0, **safe)
        elif "rf" in name:
            from sklearn.ensemble import RandomForestClassifier
            safe = {k: v for k, v in config.items() if k in [
                "n_estimators", "max_depth", "min_samples_leaf", "max_features",
            ]}
            return RandomForestClassifier(random_state=42, **safe)
        elif "extra_tree" in name:
            from sklearn.ensemble import ExtraTreesClassifier
            safe = {k: v for k, v in config.items() if k in [
                "n_estimators", "max_depth", "min_samples_leaf",
            ]}
            return ExtraTreesClassifier(random_state=42, **safe)
        elif "lrl1" in name or "lrl2" in name:
            from sklearn.linear_model import LogisticRegression
            penalty = "l1" if "lrl1" in name else "l2"
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            return LogisticRegression(
                penalty=penalty, solver=solver,
                C=config.get("C", 1.0), max_iter=1000, random_state=42,
            )
    except Exception as e:
        print(f"  Warning: could not instantiate {name}: {e}")
    return None


def _instantiate_estimator_optuna(params: dict):
    """Instantiate an sklearn estimator from Optuna best_params dict."""
    from sklearn.calibration import CalibratedClassifierCV

    clf_name = params.get("classifier", "")
    try:
        if clf_name == "logreg_l1":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty="l1", C=params.get("lr_l1_C", 1.0),
                                      solver="saga", max_iter=2000, n_jobs=-1, random_state=42)
        elif clf_name == "logreg_l2":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty="l2", C=params.get("lr_l2_C", 1.0),
                                      solver="lbfgs", max_iter=2000, n_jobs=-1, random_state=42)
        elif clf_name == "linearsvc":
            from sklearn.svm import LinearSVC
            clf = LinearSVC(C=params.get("svc_C", 1.0), max_iter=3000, random_state=42)
            return CalibratedClassifierCV(clf, cv=3)
        elif clf_name == "sgd":
            from sklearn.linear_model import SGDClassifier
            return SGDClassifier(loss=params.get("sgd_loss", "log_loss"),
                                 alpha=params.get("sgd_alpha", 1e-4),
                                 max_iter=1000, n_jobs=-1, random_state=42)
        elif clf_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get("rf_n_estimators", 100),
                max_depth=params.get("rf_max_depth", None),
                n_jobs=-1, random_state=42,
            )
        elif clf_name == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=params.get("et_n_estimators", 100),
                max_depth=params.get("et_max_depth", None),
                n_jobs=-1, random_state=42,
            )
        elif clf_name == "hist_gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                learning_rate=params.get("hgb_lr", 0.1),
                max_iter=params.get("hgb_max_iter", 100),
                max_depth=params.get("hgb_max_depth", None),
                random_state=42,
            )
    except Exception as e:
        print(f"  Warning: could not instantiate Optuna estimator '{clf_name}': {e}")
    return None


# ---------------------------------------------------------------------------
# Language identification helper (reuse from features.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from features import identify_word_language
except ImportError:
    def identify_word_language(word: str) -> str:
        return "other"


# ---------------------------------------------------------------------------
# LIME explanation functions
# ---------------------------------------------------------------------------

def get_lime_explainer():
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(
        class_names=CANONICAL_LABELS,
        split_expression=r"\s+",
        random_state=42,
    )


def explain_sample(explainer, text: str, predict_fn, num_features: int = 10, num_samples: int = 5000):
    return explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=3,
    )


def explanation_to_dict(exp, predicted_label_id: int) -> dict:
    """Serialise a LIME explanation to a JSON-friendly dict."""
    label = CANONICAL_LABELS[predicted_label_id]
    feature_weights = exp.as_list(label=predicted_label_id)
    return {
        "predicted_label": label,
        "features": [
            {
                "word": w,
                "weight": float(wt),
                "language": identify_word_language(w),
            }
            for w, wt in feature_weights
        ],
    }


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

def check_stability(text: str, predict_fn, n_runs: int = 10, num_features: int = 10) -> dict:
    from lime.lime_text import LimeTextExplainer

    all_top = []
    for seed in range(n_runs):
        exp_i = LimeTextExplainer(
            class_names=CANONICAL_LABELS,
            split_expression=r"\s+",
            random_state=seed,
        ).explain_instance(text, predict_fn, num_features=num_features, top_labels=1)
        top_label = exp_i.top_labels[0]
        top_words = {w for w, _ in exp_i.as_list(label=top_label)}
        all_top.append(top_words)

    similarities = []
    for i in range(len(all_top)):
        for j in range(i + 1, len(all_top)):
            inter = len(all_top[i] & all_top[j])
            union = len(all_top[i] | all_top[j])
            if union > 0:
                similarities.append(inter / union)

    return {
        "n_runs": n_runs,
        "mean_jaccard": float(np.mean(similarities)) if similarities else 0.0,
        "std_jaccard": float(np.std(similarities)) if similarities else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregate LIME weights across test set
# ---------------------------------------------------------------------------

def aggregate_lime_weights(test_texts, predict_fn, pipeline, num_features: int = 20):
    """Compute average LIME weight per word across all test samples, per class."""
    explainer = get_lime_explainer()
    word_weights_per_class = {lbl: defaultdict(list) for lbl in CANONICAL_LABELS}

    for idx, text in enumerate(test_texts):
        if (idx + 1) % 20 == 0:
            print(f"  Aggregate LIME: {idx + 1}/{len(test_texts)}")
        try:
            pred_proba = pipeline.predict_proba([text])[0]
            pred_label_id = int(np.argmax(pred_proba))
            pred_label = CANONICAL_LABELS[pred_label_id]

            exp = explainer.explain_instance(
                text, predict_fn, num_features=num_features,
                top_labels=3, num_samples=2000,
            )
            for label_id in range(len(CANONICAL_LABELS)):
                lbl = CANONICAL_LABELS[label_id]
                for word, weight in exp.as_list(label=label_id):
                    word_weights_per_class[lbl][word.lower()].append(float(weight))
        except Exception:
            continue

    # Aggregate: mean weight per word (min 3 appearances)
    aggregated = {}
    for lbl in CANONICAL_LABELS:
        avg = {
            w: np.mean(ws)
            for w, ws in word_weights_per_class[lbl].items()
            if len(ws) >= 3
        }
        top_pos = sorted(
            [(w, wt) for w, wt in avg.items() if wt > 0],
            key=lambda x: x[1], reverse=True,
        )[:10]
        top_neg = sorted(
            [(w, wt) for w, wt in avg.items() if wt < 0],
            key=lambda x: x[1],
        )[:10]
        aggregated[lbl] = {
            "top_positive_words": [
                {"word": w, "avg_weight": float(wt), "language": identify_word_language(w)}
                for w, wt in top_pos
            ],
            "top_negative_words": [
                {"word": w, "avg_weight": float(wt), "language": identify_word_language(w)}
                for w, wt in top_neg
            ],
        }
    return aggregated


# ---------------------------------------------------------------------------
# Language contribution analysis
# ---------------------------------------------------------------------------

def analyze_language_contributions(explanation_dicts: list[dict]) -> dict:
    """Compute average positive/negative LIME weight contribution per language."""
    lang_pos = defaultdict(list)
    lang_neg = defaultdict(list)

    for exp_dict in explanation_dicts:
        for f in exp_dict.get("features", []):
            lang = f["language"]
            wt = f["weight"]
            if wt > 0:
                lang_pos[lang].append(wt)
            else:
                lang_neg[lang].append(wt)

    result = {}
    for lang in ["id", "jv", "en", "other"]:
        result[lang] = {
            "avg_positive": float(np.mean(lang_pos[lang])) if lang_pos[lang] else 0.0,
            "avg_negative": float(np.mean(lang_neg[lang])) if lang_neg[lang] else 0.0,
            "n_positive_occurrences": len(lang_pos[lang]),
            "n_negative_occurrences": len(lang_neg[lang]),
        }
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_lime_explanation(exp, label_id: int, title: str, save_path: pathlib.Path):
    import matplotlib.pyplot as plt

    fig = exp.as_pyplot_figure(label=label_id)
    fig.set_size_inches(3.5, 2.5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_language_contributions(lang_contributions: dict, save_path: pathlib.Path):
    import matplotlib.pyplot as plt

    langs = ["Indonesian", "Javanese", "English"]
    lang_codes = ["id", "jv", "en"]
    colors_pos = ["#2196F3", "#4CAF50", "#FF9800"]
    colors_neg = ["#1565C0", "#2E7D32", "#E65100"]

    avg_pos = [lang_contributions.get(lc, {}).get("avg_positive", 0.0) for lc in lang_codes]
    avg_neg = [lang_contributions.get(lc, {}).get("avg_negative", 0.0) for lc in lang_codes]

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x - width / 2, avg_pos, width, label="Positive", color=colors_pos, alpha=0.85)
    ax.bar(x + width / 2, avg_neg, width, label="Negative", color=colors_neg, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_ylabel("Avg. LIME Weight", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("Language Contribution to Predictions", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_words(aggregated: dict, save_path: pathlib.Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, lbl in zip(axes, CANONICAL_LABELS):
        top_pos = aggregated[lbl]["top_positive_words"][:8]
        top_neg = aggregated[lbl]["top_negative_words"][:8]
        words = [f["word"] for f in top_neg] + [f["word"] for f in top_pos]
        weights = [f["avg_weight"] for f in top_neg] + [f["avg_weight"] for f in top_pos]
        colors = ["#F44336" if w < 0 else "#4CAF50" for w in weights]
        y_pos = range(len(words))
        ax.barh(list(y_pos), weights, color=colors, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(words, fontsize=7)
        ax.set_title(f"{lbl.capitalize()}", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Global LIME Feature Importance by Sentiment Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(SCRIPT_DIR))
    from preprocess import load_data as load_splits

    print("=" * 60)
    print("LIME Analysis")
    print("=" * 60)

    # 1. Load best result metadata
    best_result = load_best_result()
    best_fs = best_result["feature_set"]
    best_framework = best_result["framework"]

    # 2. Load pre-processed splits
    splits = load_splits()
    train_texts = splits["train"]["text"].tolist()
    valid_texts = splits["valid"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label_id"].values
    test_raw_texts = splits["test"]["text_raw"].tolist()

    # Combine train + valid for pipeline training
    all_train_texts = train_texts + valid_texts
    all_train_labels = np.concatenate([
        splits["train"]["label_id"].values,
        splits["valid"]["label_id"].values,
    ])

    # 3. Build LIME-compatible pipeline
    print(f"\nBuilding LIME pipeline using best feature set: {best_fs}")
    pipeline = build_lime_pipeline(all_train_texts, all_train_labels, best_fs)

    def predict_fn(texts):
        return pipeline.predict_proba(texts)

    # 4. Test set predictions
    y_pred = pipeline.predict(test_texts)
    correct_mask = y_pred == test_labels
    wrong_mask = ~correct_mask

    print(f"\nTest accuracy (TF-IDF pipeline): {correct_mask.mean():.4f}")
    print(f"Correct: {correct_mask.sum()} / Wrong: {wrong_mask.sum()}")

    # 5. Select representative samples
    explainer = get_lime_explainer()
    lime_results = {
        "per_class_correct": {},
        "misclassifications": [],
        "stability": [],
        "language_contributions": {},
        "aggregated_global": {},
        "figures": {},
    }

    # Per-class correct examples (up to 2 per class)
    print("\nGenerating per-class correct explanations…")
    per_class_exps = []
    for label_id, lbl in enumerate(CANONICAL_LABELS):
        mask = (test_labels == label_id) & correct_mask
        indices = np.where(mask)[0][:2]
        class_exps = []
        for rank, idx in enumerate(indices):
            text = test_texts[idx]
            exp = explain_sample(explainer, text, predict_fn)
            exp_dict = explanation_to_dict(exp, label_id)
            exp_dict["text"] = text
            exp_dict["true_label"] = lbl
            exp_dict["sample_index"] = int(idx)
            class_exps.append(exp_dict)
            per_class_exps.append(exp_dict)

            # Save individual figure
            fig_path = FIGURES_DIR / f"lime_{lbl}_correct_{rank}.png"
            plot_lime_explanation(exp, label_id, f"Correctly classified: {lbl}", fig_path)
            print(f"  [{lbl}] sample {rank}: saved {fig_path.name}")

        lime_results["per_class_correct"][lbl] = class_exps

    # Language contributions from per-class examples
    lime_results["language_contributions"] = analyze_language_contributions(per_class_exps)

    # Plot language contributions
    lang_fig = FIGURES_DIR / "lime_language_contributions.png"
    plot_language_contributions(lime_results["language_contributions"], lang_fig)
    lime_results["figures"]["language_contributions"] = str(lang_fig)

    # Misclassifications (up to 5)
    print("\nGenerating misclassification explanations…")
    wrong_indices = np.where(wrong_mask)[0][:5]
    for idx in wrong_indices:
        text = test_texts[idx]
        true_id = int(test_labels[idx])
        pred_id = int(y_pred[idx])
        exp = explain_sample(explainer, text, predict_fn)
        exp_dict = explanation_to_dict(exp, pred_id)
        exp_dict["text"] = text
        exp_dict["true_label"] = CANONICAL_LABELS[true_id]
        exp_dict["predicted_label"] = CANONICAL_LABELS[pred_id]
        exp_dict["sample_index"] = int(idx)
        lime_results["misclassifications"].append(exp_dict)
        print(f"  [Wrong] true={CANONICAL_LABELS[true_id]} pred={CANONICAL_LABELS[pred_id]} idx={idx}")

    # Stability check (on 3 samples)
    print("\nChecking LIME stability…")
    stability_indices = [
        np.where((test_labels == 0) & correct_mask)[0][0] if np.any((test_labels == 0) & correct_mask) else 0,
        np.where((test_labels == 1) & correct_mask)[0][0] if np.any((test_labels == 1) & correct_mask) else 0,
        np.where((test_labels == 2) & correct_mask)[0][0] if np.any((test_labels == 2) & correct_mask) else 0,
    ]
    for idx in stability_indices:
        stab = check_stability(test_texts[idx], predict_fn, n_runs=10)
        stab["text"] = test_texts[idx]
        stab["label"] = CANONICAL_LABELS[int(test_labels[idx])]
        lime_results["stability"].append(stab)
        print(f"  Jaccard: {stab['mean_jaccard']:.3f} ± {stab['std_jaccard']:.3f}  [{stab['label']}]")

    # Aggregate global importance (subset of test for speed)
    print(f"\nComputing aggregate LIME weights across test set ({min(len(test_texts), 100)} samples)…")
    sample_texts = test_texts[:100]
    lime_results["aggregated_global"] = aggregate_lime_weights(sample_texts, predict_fn, pipeline)

    # Plot top words per class
    top_words_fig = FIGURES_DIR / "lime_top_words.png"
    plot_top_words(lime_results["aggregated_global"], top_words_fig)
    lime_results["figures"]["top_words"] = str(top_words_fig)

    # Save results
    out_path = OUTPUTS_DIR / "lime_results.json"

    # Make serialisable
    def _make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serialisable(i) for i in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    out_path.write_text(
        json.dumps(_make_serialisable(lime_results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nLIME results saved to {out_path}")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
