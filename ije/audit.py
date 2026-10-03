"""Produce an overlap audit draft from supplied IJELID training records.

This does not download or guess training data, and never signs off its own audit.
"""
from pathlib import Path
from .data import load_split, source_dir, clean_text
from .storage import read_json, write_json, digest


def overlap_audit(config, config_path, run_dir, plan_path):
    import pandas as pd
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if not plan.get("sources") or not plan.get("checkpoint_training_evidence"):
        raise RuntimeError("Supply actual IJELID training sources and checkpoint training evidence")
    if plan.get("model_revision") != config["features"]["lid"]["revision"]:
        raise RuntimeError("Audit plan must identify the configured IJELID checkpoint revision")
    splits = {s: load_split(config, config_path, s) for s in ("train", "valid", "test")}
    results, evidence, any_overlap = [], [], False
    for source in plan["sources"]:
        path = (plan_path.parent / source["path"]).resolve()
        if not source.get("role") or not source.get("evidence"):
            raise RuntimeError("Each source needs a role and checkpoint linkage evidence")
        frame = pd.read_csv(path, keep_default_na=False)
        texts = frame[source["text_column"]].astype(str)
        if texts.str.strip().eq("").any():
            raise ValueError("Blank IJELID training records need review")
        raw, cleaned = set(texts), set(texts.map(clean_text))
        source_result = {"path": str(path), "sha256": digest(path), "role": source["role"], "n_rows": len(texts), "overlaps": {}}
        for split, df in splits.items():
            ids = df.loc[df.text.isin(cleaned), "sample_id"].tolist()
            any_overlap |= bool(ids)
            source_result["overlaps"][split] = {"exact_cleaned_sample_ids": ids,
                                                "exact_raw_sample_ids": df.loc[df.text_raw.isin(raw), "sample_id"].tolist()}
        results.append(source_result)
        evidence.append({"source": str(path), "sha256": digest(path), "description": source["evidence"]})
    proposed = "overlap_documented" if any_overlap else (
        "no_overlap_verified" if plan.get("coverage_complete") else "unknown")
    audit = {"status": "draft_requires_human_review", "proposed_status": proposed,
             "reviewed_by": None, "reviewed_at": None, "model_revision": plan["model_revision"],
             "coverage_complete": plan.get("coverage_complete", False),
             "method": "exact_raw_and_documented_cleaned_text_matching",
             "checkpoint_training_evidence": plan["checkpoint_training_evidence"],
             "evidence_sources": evidence, "results": results,
             "sentiment_source_hashes": {s: digest(source_dir(config, config_path) / f"{s}.csv") for s in splits},
             "limitations": plan.get("limitations", [])}
    path = Path(run_dir) / "ijelid_overlap_audit_draft.json"
    write_json(path, audit)
    return [path]
