"""
evaluate.py
===========
Collects all AutoML results and produces:
  - Comparison tables (Framework × Feature Set)
  - Per-class F1 breakdown
  - Feature ablation analysis
  - Statistical significance tests (paired t-test / Wilcoxon)
  - Comparison with prior PEFT/LoRA baselines
  - Saves all tables to outputs/evaluation_tables.json
  - Prints LaTeX-formatted tables for the paper

Usage:
    python evaluate.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_LABELS = ["negative", "neutral", "positive"]
FEATURE_SETS = ["tfidf", "tfidf_cm", "full"]
FRAMEWORKS = ["autogluon", "flaml", "optuna+sklearn"]

# Prior work baselines (from the PEFT paper)
PRIOR_BASELINES = [
    {"method": "IndoBERTweet + LoRA", "f1": 0.9112, "acc": None, "gpu": True, "interpretable": False},
    {"method": "IndoBERTweet + SLU",  "f1": 0.8943, "acc": None, "gpu": True, "interpretable": False},
    {"method": "XLM-RoBERTa + LoRA+SLU", "f1": 0.8851, "acc": None, "gpu": True, "interpretable": False},
    {"method": "IndoBERT + LoRA",     "f1": 0.8821, "acc": None, "gpu": True, "interpretable": False},
    {"method": "MBERT + LoRA",        "f1": 0.8701, "acc": None, "gpu": True, "interpretable": False},
]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_all_results() -> list[dict]:
    """Load results from all three AutoML frameworks."""
    all_results = []
    for fname in ["results_autogluon.json", "results_flaml.json", "results_optuna.json"]:
        path = OUTPUTS_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found — skipping.")
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        all_results.extend(results)
    return all_results


def result_key(r: dict) -> tuple:
    return (r["framework"], r["feature_set"])


def build_matrix(results: list[dict]) -> dict:
    """Build a lookup dict: (framework, feature_set) -> result."""
    return {result_key(r): r for r in results}


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def print_markdown_table(headers: list, rows: list):
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    print()


def latex_table_main(matrix: dict) -> str:
    """Generate LaTeX table: Framework × Feature Set → F1, Prec, Rec, Acc."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AutoML Framework Comparison on IJE Code-Mixed Sentiment (Test Set)}",
        r"\label{tab:automl_results}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Framework & Feature Set & F1$_w$ & Prec$_w$ & Rec$_w$ & Acc \\",
        r"\midrule",
    ]
    best_f1 = max(
        (r["test_metrics"]["f1_weighted"] for r in matrix.values()),
        default=0.0,
    )
    for fw in FRAMEWORKS:
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            if r is None:
                continue
            m = r["test_metrics"]
            f1 = m["f1_weighted"]
            prec = m["precision_weighted"]
            rec = m["recall_weighted"]
            acc = m["accuracy"]
            f1_str = f"\\textbf{{{f1:.4f}}}" if abs(f1 - best_f1) < 1e-5 else f"{f1:.4f}"
            fs_label = {"tfidf": "TF-IDF", "tfidf_cm": "TF-IDF+CM", "full": "TF-IDF+CM+Emb"}[fs]
            lines.append(
                f"{fw} & {fs_label} & {f1_str} & {prec:.4f} & {rec:.4f} & {acc:.4f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"  # replace last \midrule with \bottomrule
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table_comparison(results: list[dict]) -> str:
    """Comparison table: AutoML best vs. PEFT baselines."""
    best_r = max(results, key=lambda r: r["test_metrics"]["f1_weighted"])
    best_m = best_r["test_metrics"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with Transformer-Based PEFT Methods}",
        r"\label{tab:comparison}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & F1$_w$ & GPU Req. & Interpretable \\",
        r"\midrule",
    ]
    for b in PRIOR_BASELINES:
        f1_str = f"{b['f1']:.4f}"
        gpu = "Yes" if b["gpu"] else "No"
        interp = "No"
        lines.append(f"{b['method']} & {f1_str} & {gpu} & {interp} \\\\")

    lines.append(r"\midrule")
    fw_label = f"{best_r['framework']} ({best_r['feature_set']})"
    f1_str = f"\\textbf{{{best_m['f1_weighted']:.4f}}}"
    lines.append(f"{fw_label} (ours) & {f1_str} & No & Yes \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def significance_tests(results: list[dict]) -> dict:
    """
    Pairwise significance between top-3 configurations.
    Uses Wilcoxon signed-rank test when CV fold scores are available.
    """
    from scipy.stats import wilcoxon, ttest_rel

    # Get CV scores per config (some frameworks store them differently)
    configs_with_scores = []
    for r in results:
        scores = r.get("cv_scores_internal") or []
        if not scores:
            # FLAML stores best_cv_f1 but not per-fold; skip
            continue
        configs_with_scores.append({
            "label": f"{r['framework']}/{r['feature_set']}",
            "scores": scores,
        })

    sig_results = []
    n = len(configs_with_scores)
    for i in range(n):
        for j in range(i + 1, n):
            a = configs_with_scores[i]
            b = configs_with_scores[j]
            # Align lengths
            min_len = min(len(a["scores"]), len(b["scores"]))
            if min_len < 2:
                continue
            s_a = a["scores"][:min_len]
            s_b = b["scores"][:min_len]
            try:
                stat, p = wilcoxon(s_a, s_b)
                test = "wilcoxon"
            except Exception:
                stat, p = ttest_rel(s_a, s_b)
                test = "ttest_rel"
            sig_results.append({
                "config_a": a["label"],
                "config_b": b["label"],
                "test": test,
                "statistic": float(stat),
                "p_value": float(p),
                "significant_alpha_05": bool(p < 0.05),
            })

    return {"pairwise_tests": sig_results}


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------

def feature_ablation(matrix: dict) -> list[dict]:
    rows = []
    for fw in FRAMEWORKS:
        r_base = matrix.get((fw, "tfidf"))
        r_cm = matrix.get((fw, "tfidf_cm"))
        r_full = matrix.get((fw, "full"))
        if not any([r_base, r_cm, r_full]):
            continue
        base_f1 = r_base["test_metrics"]["f1_weighted"] if r_base else None
        cm_f1 = r_cm["test_metrics"]["f1_weighted"] if r_cm else None
        full_f1 = r_full["test_metrics"]["f1_weighted"] if r_full else None
        rows.append({
            "framework": fw,
            "tfidf_f1": base_f1,
            "tfidf_cm_f1": cm_f1,
            "full_f1": full_f1,
            "delta_cm": round(cm_f1 - base_f1, 4) if (cm_f1 and base_f1) else None,
            "delta_emb": round(full_f1 - cm_f1, 4) if (full_f1 and cm_f1) else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Ranking table
# ---------------------------------------------------------------------------

def ranking_table(results: list[dict]) -> list[dict]:
    ranked = sorted(results, key=lambda r: r["test_metrics"]["f1_weighted"], reverse=True)
    table = []
    for rank, r in enumerate(ranked, 1):
        table.append({
            "rank": rank,
            "framework": r["framework"],
            "feature_set": r["feature_set"],
            "f1_weighted": r["test_metrics"]["f1_weighted"],
            "accuracy": r["test_metrics"]["accuracy"],
        })
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_f1_comparison(matrix: dict, save_path: pathlib.Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(FEATURE_SETS))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fw_labels = {"autogluon": "AutoGluon", "flaml": "FLAML", "optuna+sklearn": "Optuna+sklearn"}

    for i, fw in enumerate(FRAMEWORKS):
        f1_vals = []
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            f1_vals.append(r["test_metrics"]["f1_weighted"] if r else 0.0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1_vals, width, label=fw_labels.get(fw, fw),
                      color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["TF-IDF", "TF-IDF+CM", "TF-IDF+CM+Emb"], fontsize=9)
    ax.set_ylabel("Weighted F1", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("AutoML Frameworks vs. Feature Sets", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(SCRIPT_DIR))

    print("=" * 60)
    print("Evaluation & Comparison Tables")
    print("=" * 60)

    results = load_all_results()
    if not results:
        print("No results found. Run AutoML scripts first.")
        sys.exit(1)

    matrix = build_matrix(results)

    # Main results table (Markdown)
    print("\n### Main Results Table (Framework × Feature Set)")
    print("-" * 80)
    headers = ["Framework", "Feature Set", "F1_w", "Prec_w", "Rec_w", "Accuracy", "Time(s)"]
    rows = []
    for fw in FRAMEWORKS:
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            if r is None:
                rows.append([fw, fs, "—", "—", "—", "—", "—"])
            else:
                m = r["test_metrics"]
                rows.append([
                    fw, fs,
                    fmt(m["f1_weighted"]), fmt(m["precision_weighted"]),
                    fmt(m["recall_weighted"]), fmt(m["accuracy"]),
                    f"{r['training_time_s']:.1f}",
                ])
    print_markdown_table(headers, rows)

    # Per-class F1
    print("\n### Per-Class F1")
    print("-" * 80)
    headers2 = ["Framework", "Feature Set", "F1_neg", "F1_neu", "F1_pos"]
    rows2 = []
    for fw in FRAMEWORKS:
        for fs in FEATURE_SETS:
            r = matrix.get((fw, fs))
            if r is None:
                rows2.append([fw, fs, "—", "—", "—"])
            else:
                pc = r["test_metrics"].get("f1_per_class", {})
                rows2.append([
                    fw, fs,
                    fmt(pc.get("negative")), fmt(pc.get("neutral")), fmt(pc.get("positive")),
                ])
    print_markdown_table(headers2, rows2)

    # Ranking
    print("\n### Ranking (all configurations)")
    print("-" * 80)
    ranked = ranking_table(results)
    headers3 = ["Rank", "Framework", "Feature Set", "F1_w", "Accuracy"]
    rows3 = [[r["rank"], r["framework"], r["feature_set"],
              fmt(r["f1_weighted"]), fmt(r["accuracy"])] for r in ranked]
    print_markdown_table(headers3, rows3)

    # Feature ablation
    print("\n### Feature Ablation")
    print("-" * 80)
    ablation = feature_ablation(matrix)
    headers4 = ["Framework", "TF-IDF F1", "+CM F1", "Full F1", "Δ CM", "Δ Emb"]
    rows4 = [[
        a["framework"],
        fmt(a["tfidf_f1"]), fmt(a["tfidf_cm_f1"]), fmt(a["full_f1"]),
        fmt(a["delta_cm"]), fmt(a["delta_emb"]),
    ] for a in ablation]
    print_markdown_table(headers4, rows4)

    # Comparison with prior work
    print("\n### Comparison with Prior Work (PEFT Baselines)")
    print("-" * 80)
    best_r = max(results, key=lambda r: r["test_metrics"]["f1_weighted"])
    best_m = best_r["test_metrics"]
    print(f"Best AutoML: {best_r['framework']} / {best_r['feature_set']} — F1={best_m['f1_weighted']:.4f}")
    headers5 = ["Method", "F1_w", "GPU Required", "Interpretable"]
    rows5 = []
    for b in PRIOR_BASELINES:
        rows5.append([b["method"], fmt(b["f1"]), "Yes" if b["gpu"] else "No", "No"])
    rows5.append([
        f"{best_r['framework']} (ours)",
        fmt(best_m["f1_weighted"]), "No", "Yes (LIME)",
    ])
    print_markdown_table(headers5, rows5)

    # Significance tests
    print("\n### Statistical Significance Tests")
    sig = significance_tests(results)
    if sig["pairwise_tests"]:
        print_markdown_table(
            ["Config A", "Config B", "Test", "p-value", "Sig. (α=0.05)"],
            [[t["config_a"], t["config_b"], t["test"],
              f"{t['p_value']:.4f}", str(t["significant_alpha_05"])]
             for t in sig["pairwise_tests"]],
        )
    else:
        print("  (Not enough per-fold CV scores available for significance testing)")

    # LaTeX tables
    print("\n" + "=" * 60)
    print("LATEX TABLE: Main Results")
    print("=" * 60)
    print(latex_table_main(matrix))

    print("\n" + "=" * 60)
    print("LATEX TABLE: Comparison with Prior Work")
    print("=" * 60)
    print(latex_table_comparison(results))

    # Figure
    fig_path = FIGURES_DIR / "f1_comparison.png"
    try:
        plot_f1_comparison(matrix, fig_path)
        print(f"\nFigure saved: {fig_path}")
    except Exception as e:
        print(f"Could not generate figure: {e}")

    # Save all evaluation data
    eval_data = {
        "main_results": [
            {
                "framework": r["framework"],
                "feature_set": r["feature_set"],
                "test_metrics": r["test_metrics"],
                "training_time_s": r.get("training_time_s"),
                "best_model": r.get("best_estimator") or r.get("best_model"),
            }
            for r in results
        ],
        "ranking": ranked,
        "feature_ablation": ablation,
        "significance_tests": sig,
        "prior_baselines": PRIOR_BASELINES,
        "best_automl": {
            "framework": best_r["framework"],
            "feature_set": best_r["feature_set"],
            "f1_weighted": best_m["f1_weighted"],
            "accuracy": best_m["accuracy"],
        },
    }

    out_path = OUTPUTS_DIR / "evaluation_tables.json"
    out_path.write_text(
        json.dumps(eval_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nEvaluation data saved to {out_path}")


if __name__ == "__main__":
    main()
