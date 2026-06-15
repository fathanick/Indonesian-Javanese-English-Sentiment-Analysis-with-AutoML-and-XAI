"""
visualize_results.py
====================
Generates all publication-quality figures for the IJE AutoML + LIME paper.

Figures saved to outputs/figures/:
  fig1_f1_heatmap.png          — F1 heatmap: Framework × Feature Set
  fig2_f1_grouped_bar.png      — Grouped bar: F1 per framework per feature set
  fig3_per_class_f1.png        — Per-class F1 for best config per framework
  fig4_ablation.png            — Feature ablation delta chart
  fig5_comparison_baselines.png — AutoML vs PEFT baselines (horizontal bar)
  fig6_confusion_matrix.png    — Confusion matrix of best model (Optuna/TF-IDF)
  fig7_lime_top_words.png      — LIME global top words per class (improved)
  fig8_lime_language.png       — LIME per-language contribution (improved)
  fig9_lime_stability.png      — LIME Jaccard stability per class

Usage:
    python scripts/visualize_results.py
"""

import json
import pathlib
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 10,
    "figure.dpi": 150,
})

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUTS    = ROOT_DIR / "outputs"
FIGURES    = OUTPUTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CANONICAL_LABELS = ["negative", "neutral", "positive"]
FRAMEWORKS = ["FLAML", "AutoGluon", "Optuna+sklearn"]
FEATURE_SETS = ["TF-IDF", "TF-IDF+CM", "TF-IDF+CM+Emb"]
FS_MAP = {"tfidf": "TF-IDF", "tfidf_cm": "TF-IDF+CM", "full": "TF-IDF+CM+Emb"}
FW_MAP = {"flaml": "FLAML", "autogluon": "AutoGluon", "optuna+sklearn": "Optuna+sklearn"}

# Colour palette
C_FLAML   = "#2196F3"   # blue
C_GLUON   = "#4CAF50"   # green
C_OPTUNA  = "#9C27B0"   # purple
FW_COLORS = {"FLAML": C_FLAML, "AutoGluon": C_GLUON, "Optuna+sklearn": C_OPTUNA}

# -----------------------------------------------------------------------
# Load results
# -----------------------------------------------------------------------

def load_results():
    all_r = []
    for fname, fw_key in [
        ("results_flaml.json",   "flaml"),
        ("results_autogluon.json","autogluon"),
        ("results_optuna.json",  "optuna+sklearn"),
    ]:
        p = OUTPUTS / fname
        if not p.exists():
            print(f"  WARNING: {fname} not found — skipping.")
            continue
        data = json.loads(p.read_text())
        for r in data:
            r["framework"] = FW_MAP.get(r.get("framework",""), r.get("framework",""))
            r["feature_set_label"] = FS_MAP.get(r.get("feature_set",""), r.get("feature_set",""))
        all_r.extend(data)
    return all_r

def load_lime():
    p = OUTPUTS / "lime_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def matrix(results):
    """dict[(framework_label, fs_label)] -> result"""
    return {(r["framework"], r["feature_set_label"]): r for r in results}


def f1(r): return r["test_metrics"]["f1_weighted"]
def save(fig, name, tight=True):
    path = FIGURES / name
    fig.savefig(str(path), dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# -----------------------------------------------------------------------
# Fig 1 — F1 Heatmap
# -----------------------------------------------------------------------

def fig_f1_heatmap(results):
    mat = matrix(results)
    data = np.zeros((len(FRAMEWORKS), len(FEATURE_SETS)))
    for i, fw in enumerate(FRAMEWORKS):
        for j, fs in enumerate(FEATURE_SETS):
            r = mat.get((fw, fs))
            data[i, j] = f1(r) if r else np.nan

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    im = ax.imshow(data, cmap="YlGn", vmin=0.88, vmax=0.93)
    ax.set_xticks(range(len(FEATURE_SETS)))
    ax.set_xticklabels(FEATURE_SETS, rotation=15, ha="right")
    ax.set_yticks(range(len(FRAMEWORKS)))
    ax.set_yticklabels(FRAMEWORKS)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Weighted F1", fontsize=8)

    for i in range(len(FRAMEWORKS)):
        for j in range(len(FEATURE_SETS)):
            val = data[i, j]
            if not np.isnan(val):
                weight = "bold" if val == np.nanmax(data) else "normal"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, fontweight=weight,
                        color="black" if val < 0.915 else "white")

    ax.set_title("Weighted F1 Score — Framework × Feature Set")
    plt.tight_layout()
    save(fig, "fig1_f1_heatmap.png")


# -----------------------------------------------------------------------
# Fig 2 — Grouped Bar: F1 per feature set, grouped by framework
# -----------------------------------------------------------------------

def fig_f1_grouped_bar(results):
    mat = matrix(results)
    x = np.arange(len(FEATURE_SETS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, fw in enumerate(FRAMEWORKS):
        vals = [f1(mat[(fw, fs)]) * 100 if (fw, fs) in mat else 0 for fs in FEATURE_SETS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=fw,
                      color=FW_COLORS[fw], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=6.5)

    # Baseline reference line
    ax.axhline(91.12, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(2.42, 91.18, "IndoBERTweet+LoRA (91.12%)", color="red", fontsize=7.5, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_SETS)
    ax.set_ylabel("Weighted F1 (%)")
    ax.set_ylim(86, 93.5)
    ax.legend(loc="lower right")
    ax.set_title("AutoML Framework Performance Across Feature Sets")
    plt.tight_layout()
    save(fig, "fig2_f1_grouped_bar.png")


# -----------------------------------------------------------------------
# Fig 3 — Per-class F1 (best config per framework)
# -----------------------------------------------------------------------

def fig_per_class_f1(results):
    # Best config per framework
    best = {}
    for r in results:
        fw = r["framework"]
        if fw not in best or f1(r) > f1(best[fw]):
            best[fw] = r

    x = np.arange(len(CANONICAL_LABELS))
    width = 0.25
    cap_labels = [c.capitalize() for c in CANONICAL_LABELS]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for i, fw in enumerate(FRAMEWORKS):
        r = best.get(fw)
        if r is None:
            continue
        pc = r["test_metrics"].get("f1_per_class", {})
        vals = [pc.get(lbl, 0) for lbl in CANONICAL_LABELS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=fw,
                      color=FW_COLORS[fw], alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cap_labels)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.84, 0.97)
    ax.legend()
    ax.set_title("Per-Class F1 — Best Config per Framework")
    plt.tight_layout()
    save(fig, "fig3_per_class_f1.png")


# -----------------------------------------------------------------------
# Fig 4 — Feature Ablation Deltas
# -----------------------------------------------------------------------

def fig_ablation(results):
    mat = matrix(results)
    fw_deltas = {}
    for fw in FRAMEWORKS:
        r_base = mat.get((fw, "TF-IDF"))
        r_cm   = mat.get((fw, "TF-IDF+CM"))
        r_full = mat.get((fw, "TF-IDF+CM+Emb"))
        if not (r_base and r_cm and r_full):
            continue
        fw_deltas[fw] = {
            "delta_cm":  (f1(r_cm)   - f1(r_base)) * 100,
            "delta_emb": (f1(r_full) - f1(r_cm))   * 100,
        }

    labels = list(fw_deltas.keys())
    d_cm  = [fw_deltas[fw]["delta_cm"]  for fw in labels]
    d_emb = [fw_deltas[fw]["delta_emb"] for fw in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    bars1 = ax.bar(x - width/2, d_cm,  width, label="+CM Features",
                   color="#FF9800", alpha=0.88, edgecolor="white")
    bars2 = ax.bar(x + width/2, d_emb, width, label="+Embeddings",
                   color="#607D8B", alpha=0.88, edgecolor="white")

    for bar in list(bars1) + list(bars2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.05 if v >= 0 else -0.18),
                f"{v:+.2f} pp", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ Weighted F1 (pp)")
    ax.legend()
    ax.set_title("Feature Ablation: Marginal Contribution")
    plt.tight_layout()
    save(fig, "fig4_ablation.png")


# -----------------------------------------------------------------------
# Fig 5 — Comparison with PEFT Baselines (horizontal bar)
# -----------------------------------------------------------------------

def fig_comparison_baselines(results):
    best_automl = max(results, key=f1)

    methods = [
        ("MBERT + LoRA",            0.8701, "#BDBDBD", False),
        ("IndoBERT + LoRA",         0.8821, "#BDBDBD", False),
        ("XLM-RoBERTa + LoRA+SLU", 0.8851, "#BDBDBD", False),
        ("IndoBERTweet + SLU",      0.8943, "#BDBDBD", False),
        ("IndoBERTweet + LoRA",     0.9112, "#EF5350", False),
        (f"Optuna+sklearn\n(TF-IDF, ours)",
                                    f1(best_automl), C_OPTUNA, True),
    ]

    labels = [m[0] for m in methods]
    vals   = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    bold   = [m[3] for m in methods]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.barh(range(len(methods)), vals, color=colors, alpha=0.88,
                   edgecolor="white", height=0.65)

    for i, (bar, v, bld) in enumerate(zip(bars, vals, bold)):
        ax.text(v + 0.001, i, f"{v:.4f}",
                va="center", fontsize=8,
                fontweight="bold" if bld else "normal")

    ax.axvline(0.9112, color="#EF5350", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Weighted F1")
    ax.set_xlim(0.84, 0.935)
    ax.set_title("Comparison with Transformer PEFT Baselines")

    patch_peft = mpatches.Patch(color="#BDBDBD", alpha=0.88, label="PEFT (transformer, GPU)")
    patch_best = mpatches.Patch(color="#EF5350", alpha=0.88, label="Best PEFT baseline")
    patch_ours = mpatches.Patch(color=C_OPTUNA,  alpha=0.88, label="AutoML (ours, no GPU)")
    ax.legend(handles=[patch_peft, patch_best, patch_ours], loc="lower right", fontsize=7.5)
    plt.tight_layout()
    save(fig, "fig5_comparison_baselines.png")


# -----------------------------------------------------------------------
# Fig 6 — Confusion Matrix (best model: Optuna+sklearn / TF-IDF)
# -----------------------------------------------------------------------

def fig_confusion_matrix():
    try:
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import scipy.sparse as sp
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))

        # Load best Optuna params
        optuna_path = OUTPUTS / "results_optuna.json"
        results_opt = json.loads(optuna_path.read_text())
        best_r = next((r for r in results_opt if r["feature_set"] == "tfidf"), None)
        if best_r is None:
            print("  Optuna tfidf result not found — skipping confusion matrix.")
            return

        # Load data
        from preprocess import load_data as load_splits
        splits = load_splits()
        train_texts = splits["train"]["text"].tolist() + splits["valid"]["text"].tolist()
        train_labels = np.concatenate([
            splits["train"]["label_id"].values,
            splits["valid"]["label_id"].values,
        ])
        test_texts  = splits["test"]["text"].tolist()
        test_labels = splits["test"]["label_id"].values

        # Rebuild best pipeline
        params = best_r.get("best_params", {})
        C_val = params.get("svc_C", 1.0)
        clf = CalibratedClassifierCV(
            LinearSVC(C=C_val, max_iter=3000, random_state=42), cv=3
        )
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 3),
                max_features=10000, sublinear_tf=True,
                min_df=2, strip_accents="unicode",
            )),
            ("clf", clf),
        ])
        pipeline.fit(train_texts, train_labels)
        y_pred = pipeline.predict(test_texts)

        cm = confusion_matrix(test_labels, y_pred, normalize="true")
        cap_labels = [c.capitalize() for c in CANONICAL_LABELS]

        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cap_labels)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
        ax.set_title("Confusion Matrix — Optuna+sklearn (TF-IDF)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        save(fig, "fig6_confusion_matrix.png")

    except Exception as e:
        print(f"  Could not generate confusion matrix: {e}")


# -----------------------------------------------------------------------
# Fig 7 — LIME Global Top Words (improved, per-class side-by-side)
# -----------------------------------------------------------------------

def fig_lime_top_words(lime_data):
    if lime_data is None:
        print("  No LIME data — skipping fig7.")
        return

    agg = lime_data.get("aggregated_global", {})
    if not agg:
        print("  No aggregated_global in LIME data — skipping fig7.")
        return

    lang_colors = {
        "ID":    "#2196F3",   # blue   — Indonesian
        "JV":    "#4CAF50",   # green  — Javanese
        "EN":    "#FF9800",   # orange — English
        "MIXED": "#9C27B0",   # purple — intra-word mixed (e.g. kamerane, vibesnya)
        "OTHER": "#9E9E9E",   # gray   — platform/internet terms (rt, wkwk, …)
    }
    lang_labels = {
        "ID": "Indonesian", "JV": "Javanese", "EN": "English",
        "MIXED": "Intra-word Mixed", "OTHER": "Other",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=False)
    for ax, lbl in zip(axes, CANONICAL_LABELS):
        top_pos = agg[lbl]["top_positive_words"][:7]
        top_neg = agg[lbl]["top_negative_words"][:7]
        items = top_neg[::-1] + top_pos   # neg at bottom, pos at top
        words   = [i["word"] for i in items]
        weights = [i["avg_weight"] for i in items]
        langs   = [i.get("language","OTHER").upper() for i in items]
        colors  = [lang_colors.get(l, "#9E9E9E") for l in langs]

        y = np.arange(len(words))
        ax.barh(y, weights, color=colors, alpha=0.88, edgecolor="white", height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=7.5)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(lbl.capitalize(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Avg. LIME Weight", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    # Single shared legend on the rightmost axis
    handles = [mpatches.Patch(color=c, label=lang_labels[k], alpha=0.88)
               for k, c in lang_colors.items()]
    axes[-1].legend(handles=handles, fontsize=6.5, loc="lower right",
                    title="Language", title_fontsize=7)

    fig.suptitle("Global LIME Feature Importance by Sentiment Class",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig7_lime_top_words.png")


# -----------------------------------------------------------------------
# Fig 8 — LIME Language Contributions (improved)
# -----------------------------------------------------------------------

def fig_lime_language(lime_data):
    if lime_data is None:
        return

    lc = lime_data.get("language_contributions", {})
    if not lc:
        return

    lang_display = {"id": "Indonesian", "jv": "Javanese", "en": "English", "other": "Other/Mixed"}
    lang_keys    = ["id", "jv", "en", "other"]
    colors_pos   = ["#1565C0", "#2E7D32", "#E65100", "#757575"]
    colors_neg   = ["#90CAF9", "#A5D6A7", "#FFCC80", "#CFD8DC"]

    names    = [lang_display[k] for k in lang_keys]
    avg_pos  = [lc.get(k, {}).get("avg_positive", 0) for k in lang_keys]
    avg_neg  = [lc.get(k, {}).get("avg_negative", 0) for k in lang_keys]
    n_pos    = [lc.get(k, {}).get("n_positive_occurrences", 0) for k in lang_keys]
    n_neg    = [lc.get(k, {}).get("n_negative_occurrences", 0) for k in lang_keys]

    x = np.arange(len(lang_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    b1 = ax.bar(x - width/2, avg_pos, width, label="Avg. Positive Weight",
                color=colors_pos, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + width/2, avg_neg, width, label="Avg. Negative Weight",
                color=colors_neg, alpha=0.88, edgecolor="white")

    for bar, n in zip(b1, n_pos):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"n={n}", ha="center", fontsize=6.5)
    for bar, n in zip(b2, n_neg):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.006,
                    f"n={n}", ha="center", fontsize=6.5, va="top")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Avg. LIME Weight")
    ax.set_title("LIME Language Contribution to Predictions")
    p1 = mpatches.Patch(color="#424242", alpha=0.88, label="Avg. Positive Weight")
    p2 = mpatches.Patch(color="#BDBDBD", alpha=0.88, label="Avg. Negative Weight")
    ax.legend(handles=[p1, p2])
    plt.tight_layout()
    save(fig, "fig8_lime_language.png")


# -----------------------------------------------------------------------
# Fig 9 — LIME Explanation Stability (Jaccard per class)
# -----------------------------------------------------------------------

def fig_lime_stability(lime_data):
    if lime_data is None:
        return

    stab = lime_data.get("stability", [])
    if not stab:
        return

    labels = [s.get("label", "?").capitalize() for s in stab]
    means  = [s.get("mean_jaccard", 0) for s in stab]
    stds   = [s.get("std_jaccard",  0) for s in stab]

    x = np.arange(len(labels))
    colors = ["#EF5350", "#42A5F5", "#66BB6A"]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    bars = ax.bar(x, means, color=colors[:len(labels)], alpha=0.88,
                  yerr=stds, capsize=5, edgecolor="white")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{m:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Jaccard Similarity")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_title("LIME Explanation Stability\n(10 runs per class, top-10 words)")
    plt.tight_layout()
    save(fig, "fig9_lime_stability.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 55)
    print("Generating Visualizations")
    print("=" * 55)

    results   = load_results()
    lime_data = load_lime()

    print(f"\nLoaded {len(results)} result entries.")
    print(f"LIME data: {'available' if lime_data else 'not found'}\n")

    print("--- Framework × Feature Set figures ---")
    fig_f1_heatmap(results)
    fig_f1_grouped_bar(results)
    fig_per_class_f1(results)
    fig_ablation(results)
    fig_comparison_baselines(results)

    print("\n--- Confusion matrix ---")
    fig_confusion_matrix()

    print("\n--- LIME figures ---")
    fig_lime_top_words(lime_data)
    fig_lime_language(lime_data)
    fig_lime_stability(lime_data)

    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
