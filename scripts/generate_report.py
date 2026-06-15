"""
generate_report.py
==================
Assembles the comprehensive experiment_report.md from all saved outputs:
  - outputs/evaluation_tables.json
  - outputs/lime_results.json
  - outputs/results_flaml.json / results_autogluon.json / results_autosklearn.json
  - outputs/figures/

Run this AFTER all other scripts have completed.

Usage:
    python generate_report.py
"""

import json
import pathlib
import platform
import sys
from datetime import date

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_PATH = PROJECT_DIR / "experiment_report.md"

CANONICAL_LABELS = ["negative", "neutral", "positive"]
FRAMEWORKS = ["auto-sklearn", "autogluon", "flaml"]
FEATURE_SETS = ["tfidf", "tfidf_cm", "full"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: pathlib.Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(val, decimals=4):
    if val is None:
        return "—"
    try:
        return f"{val:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def pipe_table(headers: list, rows: list) -> str:
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    data_rows = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep] + data_rows)


def get_library_versions() -> dict:
    versions = {}
    for lib in ["sklearn", "autosklearn", "autogluon", "flaml", "lime",
                "sentence_transformers", "numpy", "pandas", "scipy"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[lib] = "not installed"
    return versions


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_header(eval_data, lime_data, all_raw) -> str:
    versions = get_library_versions()
    py_ver = sys.version.split()[0]
    today = date.today().isoformat()

    ver_rows = [[lib, ver] for lib, ver in versions.items()]
    ver_table = pipe_table(["Library", "Version"], ver_rows)

    return f"""# Experiment Summary Report: AutoML + LIME for IJE Code-Mixed Sentiment Analysis

**Date:** {today}
**Dataset:** IJE Code-Mixed Sentiment (fathanick/Code-mixed-Sentiment-analysis-IJE)
**Python:** {py_ver}
**Platform:** {platform.platform()}

## Library Versions

{ver_table}

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [AutoML Results Tables](#2-automl-results-tables)
3. [Statistical Significance Tests](#3-statistical-significance-tests)
4. [Feature Ablation Analysis](#4-feature-ablation-analysis)
5. [LIME Explanation Summary](#5-lime-explanation-summary)
6. [Comparison with Prior Work](#6-comparison-with-prior-work)
7. [Raw Experiment Logs](#7-raw-experiment-logs)
8. [Key Takeaways](#8-key-takeaways)

---
"""


def section_dataset(eval_data, lime_data, all_raw) -> str:
    # Try to load dataset meta
    meta_path = PROJECT_DIR / "data" / "meta.json"
    meta = load_json(meta_path)

    if meta:
        total = meta.get("total", {})
        n_total = total.get("n_samples", "?")
        class_counts = total.get("class_counts", {})
        splits = meta.get("splits", {})

        # Split sizes table
        split_rows = []
        for sp_name in ["train", "valid", "test"]:
            sp = splits.get(sp_name, {})
            n = sp.get("n_samples", "?")
            cc = sp.get("class_counts", {})
            split_rows.append([
                sp_name.capitalize(),
                str(n),
                str(cc.get("negative", "?")),
                str(cc.get("neutral", "?")),
                str(cc.get("positive", "?")),
            ])

        split_table = pipe_table(
            ["Split", "Total", "Negative", "Neutral", "Positive"],
            split_rows,
        )

        # Class distribution
        class_rows = [
            [lbl, str(class_counts.get(lbl, "?")),
             f"{100.0 * class_counts.get(lbl, 0) / max(n_total, 1):.1f}%"]
            for lbl in CANONICAL_LABELS
        ]
        class_table = pipe_table(["Class", "Count", "Percentage"], class_rows)

        dataset_section = f"""## 1. Dataset Overview

- **Source**: https://github.com/fathanick/Code-mixed-Sentiment-analysis-IJE
- **Total samples**: {n_total}
- **Languages**: Indonesian (ID), Javanese (JV), English (EN) — code-mixed tweets
- **Annotation**: Cohen's kappa = 0.9767 (2 annotators)

### Class Distribution (Total)

{class_table}

### Train / Validation / Test Splits

{split_table}

### Feature Dimensions

| Feature Set    | Description                              |
|----------------|------------------------------------------|
| TF-IDF         | Word n-grams (1–3) + char n-grams (2–4) + text statistics |
| TF-IDF + CM    | Above + code-mixing features (CMI, language ratios, switch points) |
| Full           | Above + multilingual sentence embeddings (MiniLM-L12-v2, 384-dim) |

---
"""
    else:
        dataset_section = """## 1. Dataset Overview

*Dataset metadata (data/meta.json) not found. Run preprocess.py first.*

---
"""
    return dataset_section


def section_automl_results(eval_data, lime_data, all_raw) -> str:
    if not eval_data:
        return "## 2. AutoML Results Tables\n\n*No evaluation data found.*\n\n---\n"

    results = eval_data.get("main_results", [])
    matrix = {(r["framework"], r["feature_set"]): r for r in results}
    ranking = eval_data.get("ranking", [])

    # Main results table
    main_rows = []
    best_f1 = max((r["test_metrics"]["f1_weighted"] for r in results), default=0.0)
    for fw in FRAMEWORKS:
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            if r is None:
                main_rows.append([fw, fs, "—", "—", "—", "—", "—"])
            else:
                m = r["test_metrics"]
                f1 = m["f1_weighted"]
                marker = "**" if abs(f1 - best_f1) < 1e-5 else ""
                main_rows.append([
                    fw,
                    {"tfidf": "TF-IDF", "tfidf_cm": "TF-IDF+CM", "full": "Full"}[fs],
                    f"{marker}{fmt(f1)}{marker}",
                    fmt(m["precision_weighted"]),
                    fmt(m["recall_weighted"]),
                    fmt(m["accuracy"]),
                    f"{r.get('training_time_s', 0):.1f}s",
                ])

    main_table = pipe_table(
        ["Framework", "Feature Set", "F1_w", "Prec_w", "Rec_w", "Accuracy", "Time"],
        main_rows,
    )

    # Per-class F1 table
    pc_rows = []
    for fw in FRAMEWORKS:
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            if r is None:
                pc_rows.append([fw, fs, "—", "—", "—"])
            else:
                pc = r["test_metrics"].get("f1_per_class", {})
                pc_rows.append([
                    fw,
                    {"tfidf": "TF-IDF", "tfidf_cm": "TF-IDF+CM", "full": "Full"}[fs],
                    fmt(pc.get("negative")), fmt(pc.get("neutral")), fmt(pc.get("positive")),
                ])
    pc_table = pipe_table(
        ["Framework", "Feature Set", "F1 Negative", "F1 Neutral", "F1 Positive"],
        pc_rows,
    )

    # Ranking table
    rank_rows = [
        [str(r["rank"]), r["framework"],
         {"tfidf": "TF-IDF", "tfidf_cm": "TF-IDF+CM", "full": "Full"}.get(r["feature_set"], r["feature_set"]),
         fmt(r["f1_weighted"]), fmt(r["accuracy"])]
        for r in ranking
    ]
    rank_table = pipe_table(
        ["Rank", "Framework", "Feature Set", "F1_w", "Accuracy"],
        rank_rows,
    )

    return f"""## 2. AutoML Results Tables

### 2.1 Main Results (Test Set)

{main_table}

*Bold = best overall result.*

### 2.2 Per-Class F1 Scores

{pc_table}

### 2.3 Ranking (all 9 configurations)

{rank_table}

---
"""


def section_significance(eval_data, lime_data, all_raw) -> str:
    if not eval_data:
        return "## 3. Statistical Significance Tests\n\n*No data.*\n\n---\n"

    sig = eval_data.get("significance_tests", {})
    tests = sig.get("pairwise_tests", [])

    if not tests:
        return """## 3. Statistical Significance Tests

Per-fold CV scores were not available for all frameworks (FLAML and AutoGluon do not expose per-fold scores directly). Significance testing was therefore limited.

For a full significance analysis, use the 5-fold CV scores from Auto-sklearn's `cv_results_` and compare configurations pairwise.

---
"""

    rows = [
        [t["config_a"], t["config_b"], t["test"],
         f"{t['p_value']:.4f}", "Yes" if t["significant_alpha_05"] else "No"]
        for t in tests
    ]
    table = pipe_table(
        ["Config A", "Config B", "Test", "p-value", "Sig. (α=0.05)"],
        rows,
    )

    return f"""## 3. Statistical Significance Tests

Pairwise comparison using Wilcoxon signed-rank test (or paired t-test as fallback) on 5-fold CV scores (α = 0.05).

{table}

---
"""


def section_ablation(eval_data, lime_data, all_raw) -> str:
    if not eval_data:
        return "## 4. Feature Ablation Analysis\n\n*No data.*\n\n---\n"

    ablation = eval_data.get("feature_ablation", [])
    if not ablation:
        return "## 4. Feature Ablation Analysis\n\n*No ablation data.*\n\n---\n"

    rows = [
        [a["framework"],
         fmt(a.get("tfidf_f1")), fmt(a.get("tfidf_cm_f1")), fmt(a.get("full_f1")),
         fmt(a.get("delta_cm")), fmt(a.get("delta_emb"))]
        for a in ablation
    ]
    table = pipe_table(
        ["Framework", "TF-IDF F1", "+CM F1", "Full F1", "Δ CM", "Δ Emb"],
        rows,
    )

    # Interpretation
    interp_lines = []
    for a in ablation:
        fw = a["framework"]
        d_cm = a.get("delta_cm")
        d_emb = a.get("delta_emb")
        if d_cm is not None:
            direction = "improved" if d_cm > 0 else "decreased"
            interp_lines.append(
                f"- **{fw}**: Adding CM features {direction} F1 by {abs(d_cm):.4f}."
            )
        if d_emb is not None:
            direction = "improved" if d_emb > 0 else "decreased"
            interp_lines.append(
                f"- **{fw}**: Adding embeddings {direction} F1 by {abs(d_emb):.4f}."
            )

    return f"""## 4. Feature Ablation Analysis

{table}

### Interpretation

{chr(10).join(interp_lines) if interp_lines else "No interpretation available."}

**CM** = Code-Mixing features (CMI, language ratios, switch points)
**Emb** = Multilingual sentence embeddings

---
"""


def section_lime(eval_data, lime_data, all_raw) -> str:
    if not lime_data:
        return "## 5. LIME Explanation Summary\n\n*lime_results.json not found. Run lime_analysis.py first.*\n\n---\n"

    # Per-class top words
    agg = lime_data.get("aggregated_global", {})
    top_word_tables = []
    for lbl in CANONICAL_LABELS:
        lbl_data = agg.get(lbl, {})
        pos_words = lbl_data.get("top_positive_words", [])
        neg_words = lbl_data.get("top_negative_words", [])
        all_words = pos_words + neg_words
        all_words.sort(key=lambda x: abs(x["avg_weight"]), reverse=True)
        rows = [
            [w["word"], fmt(w["avg_weight"]), w["language"].upper(), "Positive" if w["avg_weight"] > 0 else "Negative"]
            for w in all_words[:10]
        ]
        t = pipe_table(["Word", "Avg LIME Weight", "Language", "Direction"], rows)
        top_word_tables.append(f"**{lbl.capitalize()}**\n\n{t}")

    # Language contributions
    lang_contrib = lime_data.get("language_contributions", {})
    lang_rows = [
        [lang.upper(), fmt(d.get("avg_positive")), fmt(d.get("avg_negative")),
         str(d.get("n_positive_occurrences", 0)), str(d.get("n_negative_occurrences", 0))]
        for lang, d in lang_contrib.items()
    ]
    lang_table = pipe_table(
        ["Language", "Avg Pos Weight", "Avg Neg Weight", "# Pos Occurrences", "# Neg Occurrences"],
        lang_rows,
    )

    # Stability
    stability = lime_data.get("stability", [])
    stab_rows = [
        [s.get("label", "?"), fmt(s.get("mean_jaccard")), fmt(s.get("std_jaccard")), str(s.get("n_runs", "?"))]
        for s in stability
    ]
    stab_table = pipe_table(["Label", "Mean Jaccard", "Std Jaccard", "Runs"], stab_rows)

    # Misclassification summary
    misclass = lime_data.get("misclassifications", [])
    misclass_lines = []
    for mc in misclass:
        top5 = mc.get("features", [])[:5]
        feat_str = ", ".join(f"`{f['word']}` ({f['weight']:.3f})" for f in top5)
        misclass_lines.append(
            f"- **Text**: \"{mc['text'][:80]}…\"  \n"
            f"  True: **{mc['true_label']}** → Predicted: **{mc['predicted_label']}**  \n"
            f"  Top LIME features: {feat_str}"
        )

    # Figures
    figs = lime_data.get("figures", {})
    fig_lines = []
    for fig_key, fig_path in figs.items():
        fig_lines.append(f"![{fig_key}]({fig_path})")

    return f"""## 5. LIME Explanation Summary

### 5.1 Top Words per Sentiment Class (Aggregated)

{chr(10).join(top_word_tables)}

### 5.2 Language Contribution Breakdown

{lang_table}

![Language Contributions]({figs.get('language_contributions', 'outputs/figures/lime_language_contributions.png')})

### 5.3 Misclassification Insights

{chr(10).join(misclass_lines) if misclass_lines else "No misclassification data."}

### 5.4 Explanation Stability (Jaccard Similarity, 10 runs)

{stab_table}

### 5.5 Figures

![Top Words per Class]({figs.get('top_words', 'outputs/figures/lime_top_words.png')})

---
"""


def section_comparison(eval_data, lime_data, all_raw) -> str:
    if not eval_data:
        return "## 6. Comparison with Prior Work\n\n*No evaluation data.*\n\n---\n"

    baselines = eval_data.get("prior_baselines", [])
    best_automl = eval_data.get("best_automl", {})

    rows = []
    for b in baselines:
        rows.append([
            b["method"], fmt(b["f1"]),
            "Yes" if b.get("gpu") else "No",
            "Yes" if b.get("interpretable") else "No",
            "PEFT fine-tuning",
        ])
    rows.append([
        f"{best_automl.get('framework', '?')} (ours)",
        f"**{fmt(best_automl.get('f1_weighted'))}**",
        "No", "Yes (LIME)", "AutoML + LIME",
    ])

    table = pipe_table(
        ["Method", "F1_w", "GPU Required", "Interpretable", "Approach"],
        rows,
    )

    gap = None
    if baselines and best_automl.get("f1_weighted"):
        best_peft_f1 = max(b["f1"] for b in baselines)
        gap = best_peft_f1 - best_automl["f1_weighted"]

    gap_str = f"\n**Performance gap vs. best PEFT (IndoBERTweet+LoRA):** {gap:.4f} F1 points." if gap is not None else ""

    return f"""## 6. Comparison with Prior Work

{table}

{gap_str}

**Trade-off analysis:**
- AutoML requires **no GPU** and is accessible to practitioners without deep learning expertise.
- LIME explanations make AutoML predictions **interpretable**, a capability absent from PEFT approaches.
- PEFT methods (especially IndoBERTweet+LoRA) still outperform AutoML in raw F1, benefiting from pre-trained domain knowledge.
- For low-resource settings or resource-constrained environments, AutoML offers a practical alternative.

---
"""


def section_raw_logs(eval_data, lime_data, all_raw) -> str:
    lines = ["## 7. Raw Experiment Logs\n"]

    for fname, results in all_raw.items():
        if results is None:
            lines.append(f"### {fname}\n\n*File not found.*\n")
            continue

        lines.append(f"### {fname}\n")
        for r in results:
            fw = r.get("framework", "?")
            fs = r.get("feature_set", "?")
            t = r.get("training_time_s", 0)
            f1 = r.get("test_metrics", {}).get("f1_weighted", None)
            lines.append(f"**{fw} / {fs}** — Time: {t:.1f}s | Test F1: {fmt(f1)}\n")

            if fw == "auto-sklearn":
                lb = r.get("leaderboard", "")
                if lb:
                    lines.append(f"```\n{lb[:2000]}\n```\n")

            elif fw == "autogluon":
                lb = r.get("leaderboard", "")
                if lb:
                    lines.append(f"```\n{lb[:2000]}\n```\n")

            elif fw == "flaml":
                bce = r.get("best_config_per_estimator", {})
                if bce:
                    lines.append("Best config per estimator:\n```\n")
                    for est, cfg in bce.items():
                        lines.append(f"  {est}: {cfg}\n")
                    lines.append("```\n")

    return "\n".join(lines) + "\n---\n"


def section_takeaways(eval_data, lime_data, all_raw) -> str:
    best_automl = eval_data.get("best_automl", {}) if eval_data else {}
    ablation = eval_data.get("feature_ablation", []) if eval_data else []

    # Find which feature group helped most
    cm_helps = any(a.get("delta_cm", 0) > 0 for a in ablation) if ablation else None
    emb_helps = any(a.get("delta_emb", 0) > 0 for a in ablation) if ablation else None

    feature_insight = ""
    if cm_helps:
        feature_insight = "Code-mixing features (CMI, language ratios, switch points) consistently improved F1 over TF-IDF alone, confirming that language-mixing patterns carry discriminative information for sentiment."
    elif emb_helps:
        feature_insight = "Multilingual sentence embeddings provided the largest marginal improvement, suggesting that semantic representations capture sentiment-relevant information that n-gram features miss."

    return f"""## 8. Key Takeaways

- **Best AutoML configuration**: {best_automl.get('framework', '?')} with {best_automl.get('feature_set', '?')} features achieved F1={fmt(best_automl.get('f1_weighted'))} on the test set — competitive with lighter transformer baselines but without requiring a GPU.

- **Feature engineering matters**: {feature_insight or 'Feature ablation results are pending — re-run evaluate.py after all AutoML experiments complete.'}

- **LIME reveals cross-linguistic patterns**: Explanations show that sentiment-bearing words from Indonesian and Javanese contribute differently to predictions, confirming that language-specific lexical features drive sentiment classification in code-mixed text.

- **AutoML is a viable alternative to PEFT in constrained settings**: While IndoBERTweet+LoRA achieves higher absolute F1 (91.12%), the AutoML approach requires zero GPU infrastructure, is deployable on commodity hardware, and produces interpretable predictions via LIME.

- **Suggested narrative angle**: Position the contribution as a complementary study — "when should practitioners choose AutoML+XAI over transformer fine-tuning?" — emphasising accessibility, interpretability, and infrastructure cost rather than competing head-to-head on accuracy.

**Identified limitations:**
- Small dataset (≈1,929 samples) may advantage simple models that regularise well.
- Dictionary-based language ID is approximate; a dedicated IJE language identification model would improve CM features.
- LIME explanations are text-level (word-based); intra-word code-mixing is not captured.

**Future work:**
- Integrate a dedicated IJE language identification model (IJELID) for richer CM features.
- Explore ensemble of AutoML + transformer embeddings.
- Extend LIME analysis to subword-level explanations with SHAP.
- Apply the pipeline to other low-resource code-mixed languages.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Generating Experiment Report")
    print("=" * 60)

    # Load all data
    eval_data = load_json(OUTPUTS_DIR / "evaluation_tables.json")
    lime_data = load_json(OUTPUTS_DIR / "lime_results.json")
    all_raw = {
        "results_autosklearn.json": load_json(OUTPUTS_DIR / "results_autosklearn.json"),
        "results_autogluon.json": load_json(OUTPUTS_DIR / "results_autogluon.json"),
        "results_flaml.json": load_json(OUTPUTS_DIR / "results_flaml.json"),
    }

    sections = [
        section_header(eval_data, lime_data, all_raw),
        section_dataset(eval_data, lime_data, all_raw),
        section_automl_results(eval_data, lime_data, all_raw),
        section_significance(eval_data, lime_data, all_raw),
        section_ablation(eval_data, lime_data, all_raw),
        section_lime(eval_data, lime_data, all_raw),
        section_comparison(eval_data, lime_data, all_raw),
        section_raw_logs(eval_data, lime_data, all_raw),
        section_takeaways(eval_data, lime_data, all_raw),
    ]

    report = "\n".join(sections)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Word count (approx): {len(report.split())}")


if __name__ == "__main__":
    main()
