"""Explain the locked, saved prediction pipeline, never a reconstructed model."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from .data import load_split, verify_sources
from .experiment import load_pipeline, check_model_provenance
from .features import language_tags
from .storage import read_json, write_json


def stratified_sample(y, size, seed):
    from sklearn.model_selection import train_test_split
    if size >= len(y):
        return np.arange(len(y))
    selected, _ = train_test_split(np.arange(len(y)), train_size=size, stratify=y, random_state=seed)
    return np.sort(selected)


def aggregate_weights(rows, minimum_documents):
    """Counts are selected token-position attributions, not all corpus tokens."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_class"], row["word"], row["language"])].append(row)
    words = []
    retained = []
    for (target, word, language), occurrences in grouped.items():
        documents = {r["sample_id"] for r in occurrences}
        if len(documents) < minimum_documents:
            continue
        weights = [r["weight"] for r in occurrences]
        words.append({"target_class": target, "word": word, "language": language,
                      "mean_weight": float(np.mean(weights)), "mean_abs_weight": float(np.mean(np.abs(weights))),
                      "n_documents": len(documents), "n_attributions": len(weights),
                      "n_positive_attributions": sum(w > 0 for w in weights),
                      "n_negative_attributions": sum(w < 0 for w in weights)})
        retained.extend(occurrences)
    by_language = defaultdict(list)
    for row in retained:
        by_language[(row["target_class"], row["language"])].append(row)
    languages = []
    for (target, language), occurrences in by_language.items():
        weights = [r["weight"] for r in occurrences]
        languages.append({"target_class": target, "language": language,
                          "mean_weight": float(np.mean(weights)),
                          "mean_abs_weight": float(np.mean(np.abs(weights))),
                          "n_distinct_words": len({r["word"] for r in occurrences}),
                          "n_documents": len({r["sample_id"] for r in occurrences}),
                          "n_attributions": len(weights),
                          "n_positive_attributions": sum(w > 0 for w in weights),
                          "n_negative_attributions": sum(w < 0 for w in weights)})
    return words, languages


def explain(config, config_path, run_dir):
    from lime.lime_text import LimeTextExplainer
    verify_sources(config, config_path, run_dir, include_test=True)
    audit = check_model_provenance(config, config_path)
    run_dir = Path(run_dir)
    selection = read_json(run_dir / "selection.json")
    if audit != selection["provenance_audit"]:
        raise RuntimeError("IJELID provenance changed")
    sid = selection["selected_scenario"]
    saved = read_json(run_dir / "test_predictions" / f"{sid}.json")
    directory = run_dir / "final_models" / sid
    pipeline = load_pipeline(directory)
    test = load_split(config, config_path, "test")
    if saved["sample_ids"] != test.sample_id.tolist():
        raise RuntimeError("Test row order differs from evaluation")
    texts = test.text.tolist()
    predict = lambda batch: pipeline.predict_proba(list(batch), directory)
    reproduced = predict(texts)
    if not np.allclose(reproduced, saved["probabilities"], atol=1e-7, rtol=1e-6):
        raise RuntimeError("LIME prediction interface does not reproduce evaluated probabilities")
    if not np.array_equal(reproduced.argmax(axis=1), saved["y_pred"]):
        raise RuntimeError("LIME predictions differ from saved classifier")
    opts = config["lime"]
    sample = stratified_sample(test.label_id.to_numpy(), opts["sample_size"], config["seed"])
    rows, fidelity = [], []
    def make_explanation(i, target, seed, perturbations, num_features):
        explainer = LimeTextExplainer(class_names=config["labels"], random_state=seed,
                                       split_expression=r"\s+", bow=False, mask_string="")
        explanation = explainer.explain_instance(texts[i], predict, labels=(target,),
                                                 num_samples=perturbations, num_features=num_features)
        words = texts[i].split()
        indexed = explanation.domain_mapper.indexed_string
        if indexed.num_words() != len(words) or any(indexed.word(j) != w for j, w in enumerate(words)):
            raise RuntimeError("LIME token positions do not match contextual language tags")
        return explanation

    tags_by_index = {}
    for i in sample:
        i = int(i)
        tags_by_index[i] = language_tags([texts[i]], config)[0]
        for target in range(3):
            e = make_explanation(i, target, config["seed"] + i, opts["perturbations"], opts["num_features"])
            fidelity.append({"sample_id": test.iloc[i].sample_id, "target_class": config["labels"][target],
                             "local_surrogate_r2": float(e.score) if np.isfinite(e.score) else None})
            for position, weight in e.local_exp[target]:
                rows.append({"sample_id": test.iloc[i].sample_id, "target_class": config["labels"][target],
                             "position": int(position), "word": texts[i].split()[position],
                             "language": tags_by_index[i][position], "weight": float(weight)})
    words, languages = aggregate_weights(rows, opts["min_distinct_documents"])
    # Select up to one actual error for each ordered true/predicted class pair.
    errors = []
    truth, pred = test.label_id.to_numpy(), np.asarray(saved["y_pred"])
    for actual in range(3):
        for predicted in range(3):
            if actual == predicted:
                continue
            candidates = np.flatnonzero((truth == actual) & (pred == predicted))
            if not len(candidates):
                continue
            i = int(candidates[0])
            tags = tags_by_index.get(i)
            if tags is None:
                tags = language_tags([texts[i]], config)[0]
            e = make_explanation(i, predicted, config["seed"] + i, opts["perturbations"], opts["num_features"])
            errors.append({"sample_id": test.iloc[i].sample_id, "raw_text": test.iloc[i].text_raw,
                           "cleaned_text": texts[i], "english_translation": None,
                           "author_interpretation": None,
                           "true_label": config["labels"][actual], "predicted_label": config["labels"][predicted],
                           "features": [{"word": texts[i].split()[p], "position": int(p),
                                         "language": tags[p], "weight": float(w)} for p, w in e.local_exp[predicted]]})
    stability = []
    rng = np.random.default_rng(config["seed"])
    for target in range(3):
        candidates = np.flatnonzero(pred == target)
        chosen = rng.choice(candidates, min(len(candidates), opts["stability_examples_per_class"]), replace=False)
        for i in chosen:
            i = int(i)
            sets = []
            for repeat in range(opts["stability_runs"]):
                e = make_explanation(i, target, config["seed"] + 100000 + 100 * i + repeat,
                                     opts["stability_perturbations"], opts["stability_features"])
                sets.append({int(p) for p, _ in e.local_exp[target]})
            similarities = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(sets, 2)]
            stability.append({"sample_id": test.iloc[i].sample_id, "target_class": config["labels"][target],
                              "n_tokens": len(texts[i].split()),
                              "trivial_top_k_covers_all_tokens": len(texts[i].split()) <= opts["stability_features"],
                              "mean_pairwise_jaccard": float(np.mean(similarities)),
                              "runs": opts["stability_runs"]})
    summary = []
    for label in config["labels"]:
        eligible = [r["mean_pairwise_jaccard"] for r in stability
                    if r["target_class"] == label and not r["trivial_top_k_covers_all_tokens"]]
        summary.append({"target_class": label, "n_nontrivial_instances": len(eligible),
                        "mean_instance_jaccard": float(np.mean(eligible)) if eligible else None,
                        "std_across_instances": float(np.std(eligible)) if eligible else None})
    out = {"selected_scenario": sid, "prediction_equivalence_verified": True,
           "sample_ids": test.iloc[sample].sample_id.tolist(), "sampling": "seeded_stratified_by_true_class",
           "class_counts": {str(k): int(v) for k, v in test.iloc[sample].label_id.value_counts().items()},
           "settings": opts, "tokenization": "whitespace_token_positions_empty_mask",
           "scope": "complete_saved_pipeline_all_features_recomputed_after_text_perturbation",
           "language_assignment": "IJELID_at_each_original_contextual_token_position",
           "aggregation": "occurrence_weighted_over_selected_LIME_attributions_min_distinct_document_filter",
           "limitations": ["local_posthoc_associations_not_linguistic_causation",
                           "perturbations_can_be_ungrammatical", "not_all_corpus_token_occurrences_are_selected_features",
                           "stability_does_not_establish_fidelity", "language_tags_are_model_predictions"],
           "word_aggregation": words, "language_aggregation": languages, "fidelity": fidelity,
           "error_examples": errors, "stability_instances": stability, "stability_summary": summary}
    write_json(run_dir / "lime_attributions.json", rows)
    write_json(run_dir / "lime_results.json", out)
    return [run_dir / "lime_results.json", run_dir / "lime_attributions.json"]
