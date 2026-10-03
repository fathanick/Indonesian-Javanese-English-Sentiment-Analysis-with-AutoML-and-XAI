"""Tables and figures driven by actual artifacts; no test-based winner selection."""
from pathlib import Path
import numpy as np
from .storage import read_json


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def report(config, config_path, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    selection = read_json(run_dir / "selection.json")
    dev = read_json(run_dir / "development.json")
    test = read_json(run_dir / "test_results.json")
    fits = read_json(run_dir / "final_fit.json")
    lime = read_json(run_dir / "lime_results.json")
    preflight = read_json(run_dir / "preflight.json")
    selected = selection["selected_scenario"]
    if test["selected_scenario"] != selected or lime["selected_scenario"] != selected:
        raise RuntimeError("Inconsistent selected configuration across artifacts")
    final_fit = {r["scenario_id"]: r["fit"] for r in fits["scenarios"]}
    rows = test["scenarios"]
    selected_result = next(r for r in rows if r["scenario_id"] == selected)
    lines = ["# Corrected experiment results", "",
             f"Development-selected configuration: **{selected}**. Selection preceded test evaluation.", "",
             "The test partition was inspected during earlier work. These results do not represent a newly untouched external test.", "",
             "## Development evaluation", "",
             "Common grouped outer folds; framework tuning uses a common inner holdout. Scores used for selection are development scores.", "",
             table(["Scenario", "Mean outer macro-F1", "SD across folds"],
                   [[r["scenario_id"], f"{r['mean_outer_macro_f1']:.6f}", f"{r['std_outer_macro_f1']:.6f}"] for r in dev]), "",
             "## Final test results", "",
             "Configuration order is fixed. The selected flag comes from development evaluation, not test ranking.", "",
             table(["Scenario", "Selected", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Accuracy"],
                   [[r["scenario_id"], r["selected_by_development"]]
                    + [f"{r['metrics'][m]:.6f}" for m in ("precision_macro", "recall_macro", "f1_macro", "f1_weighted", "accuracy")]
                    for r in rows]), "",
             "## Runtime", "",
             "Search limits are soft. Search time includes candidate fitting; final classifier refit is timed separately. Feature fit includes encoder inference where applicable. First model loading can be cold; later calls reuse in-memory pretrained encoders.", "",
             table(["Scenario", "Feature fit s", "Feature validation transform s", "Search s", "Refit s", "Total training s", "Search overrun s"],
                   [[r["scenario_id"]] + [f"{final_fit[r['scenario_id']]['timing'][m]:.2f}" for m in
                     ("feature_fit_s", "feature_transform_inner_valid_s", "search_s", "refit_s", "training_total_s", "search_overrun_s")]
                    for r in rows]), "",
             "These are measured training times, not a controlled comparison against historical GPU timings. No energy or memory claim is made.", "",
             "## Duplicate exposure sensitivity", "",
             table(["Scenario", "Exposed test rows", "Macro F1 excluding those rows", "Accuracy excluding those rows"],
                   [[r["scenario_id"], len(r["exposed_test_ids"]),
                     f"{r['overlap_excluded_metrics']['f1_macro']:.6f}" if r["overlap_excluded_metrics"] else "N/A",
                     f"{r['overlap_excluded_metrics']['accuracy']:.6f}" if r["overlap_excluded_metrics"] else "N/A"] for r in rows]), "",
             "This sensitivity removes exact cleaned-text overlaps only; it does not establish absence of near-duplicates or author-level dependence.", "",
             "## Uncertainty", "",
             f"Selected model macro-F1 percentile bootstrap interval: {test['selected_macro_f1_ci']['lower_95']:.6f} to {test['selected_macro_f1_ci']['upper_95']:.6f}.", "",
             "The interval conditions on the fitted model and observed test class counts. Paired intervals in test_results.json are exploratory, unadjusted for multiple comparisons, and do not establish superiority over historical baselines.", "",
             "## Preprocessing examples", "",
             table(["Category", "Before", "After"], [[r["category"], r["before"], r["after"]] for r in preflight["preprocessing_examples"]]), "",
             "## Language-level LIME observations", "",
             "Weights explain the saved evaluated pipeline. Counts below are retained token-position attributions, not all corpus tokens. Interpret these as sample-specific associations.", "",
             table(["Target class", "Language", "Mean weight", "Documents", "Distinct words", "Positive attributions", "Negative attributions"],
                   [[r["target_class"], r["language"], f"{r['mean_weight']:.6f}", r["n_documents"], r["n_distinct_words"],
                     r["n_positive_attributions"], r["n_negative_attributions"]] for r in lime["language_aggregation"]]), "",
             "## Actual misclassifications", "",
             "Authors must add verified English translations and contextual interpretations before transferring examples into the manuscript.", "",
             table(["Sample", "True", "Predicted", "Complete cleaned text"],
                   [[r["sample_id"], r["true_label"], r["predicted_label"], r["cleaned_text"]] for r in lime["error_examples"]]), "",
             "## Historical comparison", "",
             "Historical metrics, exact splits, and preprocessing have not been verified. No numerical transformer comparison or significance claim is generated.", "",
             "## Remaining manuscript work", "",
             "Correct regularization language, remove duplicated citations, disclose transformer-based IJELID and its training overlap audit, replace all obsolete numbers, and moderate linguistic and hardware claims. Code execution alone does not close manuscript comments."]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=False)
    files = [path]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["scenario_id"] for r in rows]
    values = [r["metrics"]["f1_macro"] for r in rows]
    ax.barh(labels, values, color=["#225ea8" if label == selected else "#969696" for label in labels])
    ax.set(xlim=(0, 1), xlabel="Test macro-F1", title="Blue marks the development-selected configuration")
    fig.tight_layout()
    output = figures / "test_macro_f1.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    files.append(output)
    cm = np.asarray(selected_result["metrics"]["confusion_matrix"])
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({normalized[i,j]:.2f})", ha="center", va="center",
                    color="white" if normalized[i,j] > .5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=config["labels"], yticklabels=config["labels"],
           xlabel="Predicted", ylabel="True", title=selected)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output = figures / "selected_confusion_matrix.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    files.append(output)
    return files
