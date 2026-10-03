"""Development selection, final fitting, and test evaluation are separate stages."""
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
from .data import development, grouped_folds, inner_partition, load_split, verify_sources
from .storage import read_json, write_json, digest, now
from .metrics import metrics, bootstrap


def check_model_provenance(config, config_path):
    """Required before any CM training or IJELID explanation."""
    import re
    lid = config["features"]["lid"]
    if not lid["provenance_reviewed"] or lid["overlap_status"] not in ("no_overlap_verified", "overlap_documented"):
        raise RuntimeError("Documented IJELID overlap audit is required")
    if not lid.get("evidence_file"):
        raise RuntimeError("IJELID audit file missing")
    evidence = (Path(config_path).parent / lid["evidence_file"]).resolve()
    audit = read_json(evidence)
    if audit.get("status") != lid["overlap_status"] or not audit.get("reviewed_by") or not audit.get("reviewed_at"):
        raise RuntimeError("IJELID audit status or reviewer missing")
    if audit.get("proposed_status") != audit["status"]:
        raise RuntimeError("Signed audit must agree with its documented overlap finding")
    if audit.get("model_revision") != lid["revision"]:
        raise RuntimeError("IJELID audit does not cover this checkpoint revision")
    if not audit.get("method") or not audit.get("evidence_sources") or not audit.get("limitations"):
        raise RuntimeError("IJELID audit lacks method, evidence, or limitations")
    if audit["status"] == "no_overlap_verified" and not audit.get("coverage_complete"):
        raise RuntimeError("No-overlap conclusion requires documented complete coverage of relevant IJELID fitting/tuning records")
    from .data import source_dir
    expected = {s: digest(source_dir(config, config_path) / f"{s}.csv") for s in ("train", "valid", "test")}
    if audit.get("sentiment_source_hashes") != expected:
        raise RuntimeError("IJELID audit is not bound to these exact sentiment splits")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", lid.get("revision") or ""):
        raise RuntimeError("Pin the IJELID model revision")
    if "full" in config["feature_sets"] and not re.fullmatch(r"[0-9a-fA-F]{40}", config["features"]["embedding"].get("revision") or ""):
        raise RuntimeError("Pin the sentence embedding model revision")
    return {"audit_path": str(evidence), "audit_sha256": digest(evidence), "status": audit["status"],
            "limitations": audit["limitations"]}


def scenarios(config):
    return [(f"{fw}_{fs}", fw, fs) for fw in config["frameworks"] for fs in config["feature_sets"]]


def select_development(summaries):
    if not summaries or not all(np.isfinite(r["mean_outer_macro_f1"]) for r in summaries):
        raise ValueError("Finite development scores are required")
    return max(summaries, key=lambda r: r["mean_outer_macro_f1"])


class SavedPipeline:
    def __init__(self, features, handle):
        self.features, self.handle = features, handle

    def predict(self, texts, artifact_dir):
        return self.handle.predict(self.features.transform(texts).toarray(), artifact_dir)

    def predict_proba(self, texts, artifact_dir):
        batch_size = self.features.config["lime"]["batch_size"]
        batches = []
        for start in range(0, len(texts), batch_size):
            matrix = self.features.transform(texts[start:start + batch_size]).toarray()
            batches.append(self.handle.proba(matrix, artifact_dir))
        return np.concatenate(batches) if batches else np.empty((0, 3))


def fit_procedure(pool, inner_train, inner_valid, fw, fs, config, seed, artifact_dir):
    from .features import TextFeatures
    from .engines import train_engine
    start_total = perf_counter()
    features = TextFeatures(fs, config)
    start = perf_counter()
    x_train = features.fit_transform(pool.iloc[inner_train].text_raw.tolist()).toarray()
    feature_fit_s = perf_counter() - start
    start = perf_counter()
    x_valid = features.transform(pool.iloc[inner_valid].text_raw.tolist()).toarray()
    feature_transform_s = perf_counter() - start
    y_train = pool.iloc[inner_train].label_id.to_numpy()
    y_valid = pool.iloc[inner_valid].label_id.to_numpy()
    # Preserve exact feature map selected in search. All outer-training rows
    # enter classifier refitting; none of the outer-validation rows do.
    x_refit = np.concatenate([x_train, x_valid])
    y_refit = np.concatenate([y_train, y_valid])
    handle, meta, timing = train_engine(fw, x_train, y_train, x_valid, y_valid,
                                        x_refit, y_refit, config, seed, artifact_dir)
    timing.update({"feature_fit_s": feature_fit_s, "feature_transform_inner_valid_s": feature_transform_s,
                   "training_total_s": perf_counter() - start_total})
    pipeline = SavedPipeline(features, handle)
    artifact_dir = Path(artifact_dir)
    with (artifact_dir / "pipeline.pkl").open("xb") as f:
        pickle.dump(pipeline, f)
    meta.update({"feature_set": fs, "n_features": features.n_features,
                 "feature_fit_sample_ids": pool.iloc[inner_train].sample_id.tolist(),
                 "classifier_refit_sample_ids": pool.sample_id.tolist(),
                 "constant_dense_columns": features.constant_dense_columns,
                 "timing": timing})
    write_json(artifact_dir / "fit.json", meta)
    return pipeline, meta


def load_pipeline(directory):
    directory = Path(directory)
    with (directory / "pipeline.pkl").open("rb") as f:
        return pickle.load(f)


def all_files(directory):
    # Framework diagnostic logs may append on model loading; model binaries and
    # numeric results remain checksum-bound, while logs are retained separately.
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix != ".log")


def develop(config, config_path, run_dir):
    verify_sources(config, config_path, run_dir)
    audit = check_model_provenance(config, config_path)
    dev = development(config, config_path)
    folds = grouped_folds(dev, config["outer_folds"], config["seed"])
    plan = []
    for fold, (train, valid) in enumerate(folds):
        pool = dev.iloc[train].reset_index(drop=True)
        a, b = inner_partition(pool, config, config["seed"] + fold + 1)
        plan.append({"fold": fold, "outer_train": train.tolist(), "outer_valid": valid.tolist(),
                     "inner_train": a.tolist(), "inner_valid": b.tolist(),
                     "outer_train_ids": pool.sample_id.tolist(),
                     "outer_valid_ids": dev.iloc[valid].sample_id.tolist(),
                     "inner_train_ids": pool.iloc[a].sample_id.tolist(),
                     "inner_valid_ids": pool.iloc[b].sample_id.tolist()})
    run_dir = Path(run_dir)
    write_json(run_dir / "fold_plan.json", plan)
    summaries = []
    for sid, fw, fs in scenarios(config):
        fold_rows, predictions = [], np.full(len(dev), -1, dtype=int)
        for item in plan:
            fold = item["fold"]
            pool = dev.iloc[item["outer_train"]].reset_index(drop=True)
            holdout = dev.iloc[item["outer_valid"]]
            directory = run_dir / "development_models" / sid / f"fold_{fold}"
            pipeline, fit = fit_procedure(pool, item["inner_train"], item["inner_valid"], fw, fs,
                                          config, config["seed"] + fold + 1, directory)
            start = perf_counter()
            pred = pipeline.predict(holdout.text_raw.tolist(), directory)
            evaluation_s = perf_counter() - start
            predictions[item["outer_valid"]] = pred
            fold_rows.append({"fold": fold, "metrics": metrics(holdout.label_id, pred),
                              "fit": fit, "outer_evaluation_s": evaluation_s})
        if (predictions < 0).any():
            raise RuntimeError("Missing out-of-fold predictions")
        scores = [r["metrics"]["f1_macro"] for r in fold_rows]
        row = {"scenario_id": sid, "framework": fw, "feature_set": fs,
               "mean_outer_macro_f1": float(np.mean(scores)), "std_outer_macro_f1": float(np.std(scores, ddof=1)),
               "pooled_oof_metrics": metrics(dev.label_id, predictions), "folds": fold_rows}
        summaries.append(row)
        write_json(run_dir / "development_predictions" / f"{sid}.json", {
            "sample_ids": dev.sample_id.tolist(), "y_true": dev.label_id.tolist(), "y_pred": predictions.tolist()})
    # Stable configuration order breaks exact ties. Test scores are not loaded.
    winner = select_development(summaries)
    selection = {"selected_at": now(), "selected_scenario": winner["scenario_id"],
                 "framework": winner["framework"], "feature_set": winner["feature_set"],
                 "criterion": config["selection_rule"], "score": winner["mean_outer_macro_f1"],
                 "test_scores_used": False, "provenance_audit": audit,
                 "caution": "development_selection_score_not_an_unbiased_final_performance_estimate"}
    write_json(run_dir / "development.json", summaries)
    write_json(run_dir / "selection.json", selection)
    return ([run_dir / "fold_plan.json", run_dir / "development.json", run_dir / "selection.json"]
            + all_files(run_dir / "development_models") + all_files(run_dir / "development_predictions"))


def fit_final(config, config_path, run_dir):
    verify_sources(config, config_path, run_dir)
    selection = read_json(Path(run_dir) / "selection.json")
    audit = check_model_provenance(config, config_path)
    if audit != selection["provenance_audit"]:
        raise RuntimeError("IJELID provenance changed after development selection")
    dev = development(config, config_path)
    a, b = inner_partition(dev, config, config["seed"])
    results = []
    for sid, fw, fs in scenarios(config):
        directory = Path(run_dir) / "final_models" / sid
        _, fit = fit_procedure(dev, a, b, fw, fs, config, config["seed"], directory)
        results.append({"scenario_id": sid, "selected": sid == selection["selected_scenario"], "fit": fit})
    path = Path(run_dir) / "final_fit.json"
    write_json(path, {"selection_sha256": digest(Path(run_dir) / "selection.json"), "scenarios": results,
                      "inner_train_ids": dev.iloc[a].sample_id.tolist(), "inner_valid_ids": dev.iloc[b].sample_id.tolist()})
    return [path] + all_files(Path(run_dir) / "final_models")


def evaluate(config, config_path, run_dir):
    verify_sources(config, config_path, run_dir, include_test=True)
    run_dir = Path(run_dir)
    selection = read_json(run_dir / "selection.json")
    if read_json(run_dir / "final_fit.json")["selection_sha256"] != digest(run_dir / "selection.json"):
        raise RuntimeError("Selection changed after final fitting")
    test = load_split(config, config_path, "test")
    dev = development(config, config_path)
    exposed = test.group.isin(set(dev.group)).to_numpy()
    rows, saved = [], {}
    for sid, fw, fs in scenarios(config):
        directory = run_dir / "final_models" / sid
        pipeline = load_pipeline(directory)
        start = perf_counter()
        probability = pipeline.predict_proba(test.text_raw.tolist(), directory)
        pred = pipeline.predict(test.text_raw.tolist(), directory)
        if not np.array_equal(pred, probability.argmax(axis=1)):
            raise RuntimeError(f"{sid}: native labels disagree with probability argmax")
        elapsed = perf_counter() - start
        row = {"scenario_id": sid, "selected_by_development": sid == selection["selected_scenario"],
               "metrics": metrics(test.label_id, pred), "prediction_and_probability_verification_s": elapsed,
               "exposed_test_ids": test.loc[exposed, "sample_id"].tolist(),
               "overlap_excluded_metrics": metrics(test.label_id.to_numpy()[~exposed], pred[~exposed]) if (~exposed).any() else None}
        rows.append(row)
        saved[sid] = pred
        write_json(run_dir / "test_predictions" / f"{sid}.json", {
            "sample_ids": test.sample_id.tolist(), "y_true": test.label_id.tolist(),
            "y_pred": pred.tolist(), "probabilities": probability.tolist()})
    selected = selection["selected_scenario"]
    ci = bootstrap(test.label_id, saved[selected], config["bootstrap_replicates"], config["seed"])
    comparisons = [{"selected_scenario": selected, "other_scenario": sid,
                    "macro_f1_difference_ci": bootstrap(test.label_id, saved[selected], config["bootstrap_replicates"],
                                                         config["seed"], other=pred)}
                   for sid, pred in saved.items() if sid != selected]
    write_json(run_dir / "test_results.json", {"selected_scenario": selected, "scenarios": rows,
               "selected_macro_f1_ci": ci, "exploratory_paired_comparisons": comparisons,
               "test_previously_inspected": config["disclosures"]["test_previously_inspected"],
               "note": "Winner remains the development-selected configuration regardless of test ranking."})
    return [run_dir / "test_results.json"] + all_files(run_dir / "test_predictions")
