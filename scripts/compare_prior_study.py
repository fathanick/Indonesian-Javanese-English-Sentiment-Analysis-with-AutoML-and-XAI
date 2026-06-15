"""
compare_prior_study.py
======================
Generates comparison visualizations between our AutoML results and the
prior study:

  Hidayatullah, A. F. (2024). "Code-Mixed Sentiment Analysis on
  Indonesian-Javanese-English Text Using Transformer Models."

Prior study Table IV (full fine-tuning, 4× NVIDIA Tesla V100):
  DistilMBERT  : P=83.25  R=82.49  F1=82.34
  MBERT        : P=85.71  R=83.76  F1=83.56
  XLM-RoBERTa  : P=89.26  R=89.03  F1=88.78
  IndoBERT     : P=91.56  R=91.57  F1=91.50
  IndoBERTweet : P=94.21  R=94.27  F1=94.14  ← best transformer

Our AutoML best per framework (this study, CPU only):
  AutoGluon (TF-IDF+CM+Emb) : P=90.74  R=90.50  F1=90.55
  FLAML     (TF-IDF+CM)     : P=90.84  R=90.71  F1=90.74
  Optuna+sk (TF-IDF)        : P=91.71  R=91.58  F1=91.60  ← best AutoML

Saved to:  visualizations/
  cmp1_f1_all_methods.png    — sorted horizontal bar (all 8 models)
  cmp2_prf_grouped.png       — Precision / Recall / F1 grouped bar
  cmp3_tradeoff.png          — F1 vs. resource-cost scatter
  cmp4_gap_to_best.png       — F1 gap vs. IndoBERTweet per method
"""

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

ROOT = pathlib.Path(__file__).parent.parent.resolve()
VIZ  = ROOT / "visualizations"
VIZ.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    p = VIZ / name
    fig.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------

# (label, precision, recall, f1, category, gpu_required)
# Values in percent (×100)
ALL_METHODS = [
    # --- Transformer models (prior study, full fine-tuning) ---
    ("DistilMBERT",             83.25, 82.49, 82.34, "Multilingual\nTransformer", True),
    ("MBERT",                   85.71, 83.76, 83.56, "Multilingual\nTransformer", True),
    ("XLM-RoBERTa",            89.26, 89.03, 88.78, "Multilingual\nTransformer", True),
    ("IndoBERT",                91.56, 91.57, 91.50, "Monolingual\nTransformer",  True),
    ("IndoBERTweet",            94.21, 94.27, 94.14, "Monolingual\nTransformer",  True),
    # --- AutoML (this study, CPU only) ---
    ("AutoGluon\n(TF-IDF+CM+Emb)", 90.74, 90.50, 90.55, "AutoML\n(ours)", False),
    ("FLAML\n(TF-IDF+CM)",         90.84, 90.71, 90.74, "AutoML\n(ours)", False),
    ("Optuna+sklearn\n(TF-IDF)",   91.71, 91.58, 91.60, "AutoML\n(ours)", False),
]

CATEGORY_COLORS = {
    "Multilingual\nTransformer": "#90CAF9",   # light blue
    "Monolingual\nTransformer":  "#1565C0",   # dark blue
    "AutoML\n(ours)":            "#9C27B0",   # purple
}

BEST_TRANSFORMER_F1 = 94.14   # IndoBERTweet


# -----------------------------------------------------------------------
# Fig 1 — Horizontal bar: all methods sorted by F1
# -----------------------------------------------------------------------

def cmp1_f1_all_methods():
    data = sorted(ALL_METHODS, key=lambda x: x[3])   # sort by F1
    labels  = [d[0] for d in data]
    f1s     = [d[3] for d in data]
    cats    = [d[4] for d in data]
    colors  = [CATEGORY_COLORS[c] for c in cats]
    is_best = [d[0] == "IndoBERTweet" for d in data]
    is_ours_best = [d[0] == "Optuna+sklearn\n(TF-IDF)" for d in data]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.barh(range(len(data)), f1s, color=colors, alpha=0.9,
                   edgecolor="white", height=0.7)

    for i, (bar, v, bo, bours) in enumerate(zip(bars, f1s, is_best, is_ours_best)):
        fw = "bold" if (bo or bours) else "normal"
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=8, fontweight=fw)

    # Dashed line at IndoBERTweet F1
    ax.axvline(BEST_TRANSFORMER_F1, color="#1565C0", linewidth=1.1,
               linestyle="--", alpha=0.7)
    ax.text(BEST_TRANSFORMER_F1 - 0.15, len(data) - 0.6,
            "IndoBERTweet\n94.14%", color="#1565C0", fontsize=7,
            ha="right", va="top")

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Weighted F1 (%)")
    ax.set_xlim(79, 97)
    ax.set_title("F1 Comparison: AutoML (Ours) vs. Transformer Models\n"
                 "(Hidayatullah 2024, full fine-tuning on 4× Tesla V100)")

    patches = [mpatches.Patch(color=c, label=k.replace("\n", " "), alpha=0.9)
               for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5)
    plt.tight_layout()
    save(fig, "cmp1_f1_all_methods.png")


# -----------------------------------------------------------------------
# Fig 2 — Grouped bar: Precision / Recall / F1 for all models
# -----------------------------------------------------------------------

def cmp2_prf_grouped():
    data = sorted(ALL_METHODS, key=lambda x: x[3])
    short_labels = [d[0].replace("\n", " ") for d in data]
    precs  = [d[1] for d in data]
    recalls= [d[2] for d in data]
    f1s    = [d[3] for d in data]
    cats   = [d[4] for d in data]

    x = np.arange(len(data))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    b1 = ax.bar(x - w, precs,   w, label="Precision", color="#42A5F5", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x,     recalls, w, label="Recall",    color="#66BB6A", alpha=0.88, edgecolor="white")
    b3 = ax.bar(x + w, f1s,     w, label="F1",        color="#AB47BC", alpha=0.88, edgecolor="white")

    # Hatch AutoML bars to distinguish
    automl_idx = [i for i, d in enumerate(data) if d[4] == "AutoML\n(ours)"]
    for bars in [b1, b2, b3]:
        for i in automl_idx:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("#333333")
            bars[i].set_linewidth(0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(78, 97)
    ax.set_title("Precision, Recall, and F1: AutoML vs. Transformer Models\n"
                 "(hatched bars = AutoML, solid bars = Transformer fine-tuning)")
    ax.axhline(BEST_TRANSFORMER_F1, color="#1565C0", linewidth=0.9,
               linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    save(fig, "cmp2_prf_grouped.png")


# -----------------------------------------------------------------------
# Fig 3 — Scatter: F1 vs. Interpretability / Resource Cost
# -----------------------------------------------------------------------

def cmp3_tradeoff():
    """
    X-axis: Weighted F1 (%)
    Y-axis: 'Accessibility Score' (0–10, higher = more accessible/lightweight)
             Transformers (GPU fine-tuning) = 1–2
             AutoML (CPU, no GPU) = 8–9
    Bubble size: proportional to model parameter count (approx)
    """
    # (label, f1, accessibility, params_M, category)
    methods = [
        ("DistilMBERT",             82.34, 2.0, 135,  "Multilingual\nTransformer"),
        ("MBERT",                   83.56, 1.5, 178,  "Multilingual\nTransformer"),
        ("XLM-RoBERTa",            88.78, 1.5, 270,  "Multilingual\nTransformer"),
        ("IndoBERT",                91.50, 2.0, 125,  "Monolingual\nTransformer"),
        ("IndoBERTweet",            94.14, 2.0, 125,  "Monolingual\nTransformer"),
        ("AutoGluon\n(ours)",       90.55, 8.5, 5,    "AutoML\n(ours)"),
        ("FLAML\n(ours)",           90.74, 9.0, 3,    "AutoML\n(ours)"),
        ("Optuna+sklearn\n(ours)",  91.60, 9.5, 2,    "AutoML\n(ours)"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for label, f1, acc, params, cat in methods:
        color = CATEGORY_COLORS[cat]
        size  = max(80, params * 1.8)
        ax.scatter(f1, acc, s=size, color=color, alpha=0.85,
                   edgecolors="#333333", linewidths=0.6, zorder=3)
        offset_x = 0.15
        offset_y = 0.25 if "ours" in cat else -0.35
        ax.annotate(label.replace("\n", "\n"), (f1, acc),
                    xytext=(f1 + offset_x, acc + offset_y),
                    fontsize=6.5, ha="left", va="center")

    ax.set_xlabel("Weighted F1 (%)")
    ax.set_ylabel("Accessibility Score\n(higher = lighter, no GPU, interpretable)")
    ax.set_title("F1 vs. Accessibility Trade-off\n"
                 "AutoML (ours) vs. Transformer Models (prior study)")
    ax.set_xlim(80, 97)
    ax.set_ylim(0, 11)
    ax.axhline(5, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.text(80.2, 5.2, "← GPU required above this line", fontsize=7,
            color="gray")

    patches = [mpatches.Patch(color=c, label=k.replace("\n", " "), alpha=0.88)
               for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=7.5)
    plt.tight_layout()
    save(fig, "cmp3_tradeoff.png")


# -----------------------------------------------------------------------
# Fig 4 — Gap to IndoBERTweet (best transformer)
# -----------------------------------------------------------------------

def cmp4_gap_to_best():
    """
    Shows the F1 gap of each model relative to IndoBERTweet (94.14%).
    Positive gap = outperforms IndoBERTweet, negative = below.
    """
    data = sorted(ALL_METHODS, key=lambda x: x[3])
    # Exclude IndoBERTweet itself (gap=0)
    data = [d for d in data if d[0] != "IndoBERTweet"]

    labels = [d[0].replace("\n", " ") for d in data]
    gaps   = [d[3] - BEST_TRANSFORMER_F1 for d in data]
    cats   = [d[4] for d in data]
    colors = []
    for g, cat in zip(gaps, cats):
        if cat == "AutoML\n(ours)":
            colors.append("#9C27B0")
        elif g >= 0:
            colors.append("#43A047")
        else:
            colors.append("#90CAF9" if "Multi" in cat else "#1565C0")

    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.barh(x, gaps, color=colors, alpha=0.88, edgecolor="white", height=0.65)

    for bar, g in zip(bars, gaps):
        xpos = g + 0.05 if g >= 0 else g - 0.05
        ha   = "left"   if g >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{g:+.2f}%", va="center", fontsize=8, ha=ha)

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("F1 Gap vs. IndoBERTweet (%) — best full fine-tuning")
    ax.set_title("F1 Gap Relative to Best Transformer (IndoBERTweet 94.14%)\n"
                 "Positive = surpasses transformer; hatched = AutoML (no GPU)")

    # Mark IndoBERTweet reference
    ax.text(0.1, len(data) - 0.5, "IndoBERTweet\n= 94.14%",
            fontsize=7, color="black", va="top")

    # Hatching on AutoML bars
    automl_idx = [i for i, d in enumerate(data) if d[4] == "AutoML\n(ours)"]
    for i in automl_idx:
        bars[i].set_hatch("//")
        bars[i].set_edgecolor("#333333")

    patches = [
        mpatches.Patch(color="#90CAF9", alpha=0.88, label="Multilingual Transformer (GPU)"),
        mpatches.Patch(color="#1565C0", alpha=0.88, label="Monolingual Transformer (GPU)"),
        mpatches.Patch(color="#9C27B0", alpha=0.88, label="AutoML — ours (CPU only)"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower right")
    plt.tight_layout()
    save(fig, "cmp4_gap_to_best.png")


# -----------------------------------------------------------------------
# Fig 5 — Radar chart: best AutoML vs. best Transformers
# -----------------------------------------------------------------------

def cmp5_radar():
    """
    Radar comparing Optuna+sklearn, IndoBERTweet, IndoBERT on
    Precision, Recall, F1, Accessibility (GPU-free score), Interpretability.
    """
    categories = ["Precision", "Recall", "F1 Score",
                  "Accessibility\n(no GPU)", "Interpretability"]
    N = len(categories)

    # Normalise to 0–10 scale. P/R/F1 mapped from 80–95 → 0–10
    def norm_prf(v):
        return (v - 80) / (95 - 80) * 10

    models = {
        "IndoBERTweet\n(prior study)":       [norm_prf(94.21), norm_prf(94.27), norm_prf(94.14), 1.0, 1.0],
        "IndoBERT\n(prior study)":            [norm_prf(91.56), norm_prf(91.57), norm_prf(91.50), 1.5, 1.5],
        "Optuna+sklearn\n(ours, TF-IDF)":    [norm_prf(91.71), norm_prf(91.58), norm_prf(91.60), 9.5, 9.5],
    }
    model_colors = {
        "IndoBERTweet\n(prior study)":    "#1565C0",
        "IndoBERT\n(prior study)":         "#42A5F5",
        "Optuna+sklearn\n(ours, TF-IDF)": "#9C27B0",
    }

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(4.8, 4.2),
                           subplot_kw=dict(polar=True))

    for label, values in models.items():
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle="solid",
                color=model_colors[label], label=label.replace("\n", " "))
        ax.fill(angles, values, color=model_colors[label], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=6)
    ax.set_title("Multi-Dimensional Comparison:\nAutoML vs. Best Transformers",
                 pad=15, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)
    plt.tight_layout()
    save(fig, "cmp5_radar.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 55)
    print("Generating Prior-Study Comparison Figures")
    print("=" * 55)
    cmp1_f1_all_methods()
    cmp2_prf_grouped()
    cmp3_tradeoff()
    cmp4_gap_to_best()
    cmp5_radar()
    print(f"\nAll comparison figures saved to: {VIZ}")


if __name__ == "__main__":
    main()
