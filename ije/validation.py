"""Explicit future validation stages. Nothing in this module runs on import."""
from pathlib import Path
import copy
import io
import unittest
from .storage import ROOT, write_json


def validate_protocol(config):
    if config["schema_version"] != 1 or config["primary_metric"] != "f1_macro":
        raise ValueError("Unsupported protocol schema or selection metric")
    if config["frameworks"] != ["flaml", "autogluon", "optuna"]:
        raise ValueError("This approved design requires all three frameworks in fixed order")
    if config["feature_sets"] != ["tfidf", "tfidf_cm", "full"]:
        raise ValueError("This approved design requires all three feature sets in fixed order")
    if config["labels"] != ["negative", "neutral", "positive"]:
        raise ValueError("Unexpected label mapping")
    if config["outer_folds"] != 5 or config["inner_folds"] != 5 or config["inner_holdout_fold"] != 0:
        raise ValueError("Review required for changes to the fixed five-fold/inner-holdout design")
    if config["group_by"] != "exact_cleaned_text_sha256":
        raise ValueError("Unsupported duplicate-group policy")
    if config["selection_rule"] != "highest_mean_outer_fold_macro_f1_then_configuration_order":
        raise ValueError("Unsupported selection rule")
    for key in ("search_budget_seconds", "cpu_threads", "optuna_trials", "bootstrap_replicates"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["budget_policy"] != "soft_search_budget_finish_current_trial_report_overrun":
        raise ValueError("Only explicitly reported soft search budgets are implemented")
    if not config["disclosures"]["test_previously_inspected"]:
        raise ValueError("Prior test exposure must remain disclosed")
    for key, value in config["lime"].items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"lime.{key} must be positive")
    if config["lime"]["stability_runs"] < 2:
        raise ValueError("Stability requires at least two runs")


def unit_validation(config, config_path, run_dir):
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = Path(run_dir) / "unit_validation.txt"
    log.write_text(stream.getvalue(), encoding="utf-8")
    if not result.wasSuccessful() or result.testsRun == 0:
        raise RuntimeError("Offline unit validation failed; see unit_validation.txt")
    output = Path(run_dir) / "unit_validation.json"
    write_json(output, {"passed": True, "tests_run": result.testsRun,
                        "scope": "offline_logic_no_model_downloads_no_benchmark_training"})
    return [log, output]


def smoke(config, config_path, run_dir):
    """Tiny synthetic training validates the installed framework APIs, after approval."""
    import numpy as np
    import pandas as pd
    from .experiment import fit_procedure, load_pipeline, all_files
    smoke_config = copy.deepcopy(config)
    smoke_config.update(search_budget_seconds=3, optuna_trials=1)
    stems = ["elek bad kecewa", "informasi biasa netral", "apik good senang"]
    texts = [f"{stems[i % 3]} produk layanan contoh{i}" for i in range(90)]
    pool = pd.DataFrame({"text_raw": texts, "label_id": [i % 3 for i in range(90)],
                         "sample_id": [f"synthetic:{i}" for i in range(90)]})
    rows = []
    for framework in config["frameworks"]:
        directory = Path(run_dir) / "smoke_models" / framework
        before, meta = fit_procedure(pool, np.arange(72), np.arange(72, 90), framework, "tfidf",
                                     smoke_config, config["seed"], directory)
        after = load_pipeline(directory)
        probe = [stems[0], stems[1], stems[2], "", "unseen text emoji 😊"]
        a, b = before.predict_proba(probe, directory), after.predict_proba(probe, directory)
        if not np.allclose(a, b, atol=1e-7) or not np.array_equal(after.predict(probe, directory), b.argmax(axis=1)):
            raise RuntimeError(f"{framework}: saved pipeline parity failed")
        rows.append({"framework": framework, "serialization_probability_parity": True,
                     "fit": meta, "scope": "synthetic_api_check_not_scientific_performance"})
    output = Path(run_dir) / "smoke.json"
    write_json(output, rows)
    return [output] + all_files(Path(run_dir) / "smoke_models")


def encoders(config, config_path, run_dir):
    """Model download/inference check, separately approved and provenance-gated."""
    import pickle
    import numpy as np
    from .experiment import check_model_provenance
    from .features import TextFeatures, language_tags, embedding_backend
    audit = check_model_provenance(config, config_path)
    texts = ["apik good layanan bagus", "elek bad layanan buruk", "informasi neutral layanan biasa"] * 3
    features = TextFeatures("full", config)
    matrix = features.fit_transform(texts)
    restored = pickle.loads(pickle.dumps(features))
    if not np.allclose(matrix.toarray(), restored.transform(texts).toarray(), atol=1e-6):
        raise RuntimeError("Encoder/feature serialization check failed")
    probes = ["", "produk 😊 apik", "tidak baguuus #good", "hello English bgt"]
    tags = language_tags(probes, config)
    if [len(t) for t in tags] != [len(t.split()) for t in probes]:
        raise RuntimeError("Contextual token alignment failed")
    if not np.isfinite(restored.transform(probes).data).all():
        raise RuntimeError("Non-finite feature values")
    e = config["features"]["embedding"]
    encoder = embedding_backend(e["model"], e["revision"], config["cpu_threads"])
    if str(encoder.device) != "cpu":
        raise RuntimeError("Embedding device is not CPU")
    output = Path(run_dir) / "encoder_validation.json"
    write_json(output, {"audit": audit, "device": str(encoder.device), "feature_dimension": matrix.shape[1],
                        "roundtrip_verified": True, "token_alignment_verified": True})
    return [output]
