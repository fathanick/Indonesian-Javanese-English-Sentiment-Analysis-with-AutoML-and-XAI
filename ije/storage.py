"""Standard-library-only provenance and execution guards."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest():
    paths = sorted((ROOT / "ije").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + [ROOT / "run.py", ROOT / "requirements.txt"]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    def scalar(obj):
        if isinstance(obj, Path):
            return str(obj)
        # NumPy scalar hyperparameters occur in FLAML; preserve numeric types.
        if type(obj).__module__.startswith("numpy") and hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Unsupported JSON value: {type(obj)}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=scalar), encoding="utf-8")
    temp.replace(path)


def authorize(config_path, approval_path, stage):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if not config.get("execution_enabled"):
        raise RuntimeError("Execution locked. Protocol review and explicit approval are required.")
    approval = read_json(approval_path)
    if not approval.get("approved_by") or not approval.get("approved_at"):
        raise RuntimeError("Approval identity and date are missing.")
    if approval.get("protocol_sha256") != digest(config_path):
        raise RuntimeError("Approval does not match this exact protocol.")
    if approval.get("code_sha256") != code_digest():
        raise RuntimeError("Approval does not match this exact code and requirements.")
    if stage not in approval.get("allowed_stages", []):
        raise RuntimeError(f"Stage {stage!r} has not been approved.")
    return config


def bind_run(config_path, run_dir, stage):
    """Never resume with changed code/configuration or overwrite a stage."""
    run_dir = Path(run_dir).resolve()
    if ROOT not in run_dir.parents or run_dir == ROOT:
        raise ValueError("Use a dedicated run directory below REVISED SCRIPT/.")
    fingerprint = {"protocol_sha256": digest(config_path), "code_sha256": code_digest()}
    meta = run_dir / "provenance.json"
    if meta.exists():
        if read_json(meta)["fingerprint"] != fingerprint:
            raise RuntimeError("Run provenance changed. Start a new approved run.")
    else:
        write_json(meta, {"fingerprint": fingerprint, "created_at": now()})
    marker = run_dir / f"{stage}.started.json"
    # Exclusive creation: failed or partial stages must be inspected, not silently overwritten.
    with marker.open("x", encoding="utf-8") as f:
        json.dump({"started_at": now()}, f)
    return run_dir


def require_stage(run_dir, stage):
    manifest = Path(run_dir) / f"{stage}.complete.json"
    if not manifest.exists():
        raise RuntimeError(f"Required stage {stage!r} is not complete.")
    data = read_json(manifest)
    for relative, expected in data["files"].items():
        if digest(Path(run_dir) / relative) != expected:
            raise RuntimeError(f"Stage artifact changed: {relative}")
    return data


def finish_stage(run_dir, stage, files):
    run_dir = Path(run_dir)
    write_json(run_dir / f"{stage}.complete.json", {
        "finished_at": now(),
        "files": {str(Path(p).relative_to(run_dir)): digest(p) for p in files}
    })
