#!/usr/bin/env python3
"""Stage dispatcher. Every executing stage requires explicit, hash-bound approval."""
import argparse
import os
import sys
import traceback
from pathlib import Path


def main():
    from ije.storage import ROOT, authorize, bind_run, require_stage, finish_stage, now, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validate", "preflight", "audit", "smoke", "encoders", "develop", "fit", "evaluate", "explain", "report"])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "protocol.json")
    parser.add_argument("--approval", type=Path, default=ROOT / "config" / "approval.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-plan", type=Path)
    args = parser.parse_args()
    # No ML libraries imported before authorization; no implicit approval creation.
    config = authorize(args.config, args.approval, args.stage)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    from ije.validation import validate_protocol
    validate_protocol(config)
    prerequisites = {
        "validate": [], "preflight": ["validate"], "audit": ["preflight"],
        "smoke": ["validate", "preflight"], "encoders": ["validate", "preflight"],
        "develop": ["validate", "preflight", "smoke", "encoders"],
        "fit": ["validate", "preflight", "develop"],
        "evaluate": ["preflight", "develop", "fit"],
        "explain": ["preflight", "develop", "fit", "evaluate"],
        "report": ["preflight", "develop", "fit", "evaluate", "explain"]
    }
    for stage in prerequisites[args.stage]:
        require_stage(args.run_dir, stage)
    run_dir = bind_run(args.config, args.run_dir, args.stage)
    try:
        if args.stage == "validate":
            from ije.validation import unit_validation as execute
        elif args.stage == "preflight":
            from ije.data import preflight as execute
        elif args.stage == "audit":
            if not args.audit_plan:
                raise ValueError("--audit-plan is required")
            from ije.audit import overlap_audit
            execute = lambda c, p, r: overlap_audit(c, p, r, args.audit_plan)
        elif args.stage == "smoke":
            from ije.validation import smoke as execute
        elif args.stage == "encoders":
            from ije.validation import encoders as execute
        elif args.stage == "develop":
            from ije.experiment import develop as execute
        elif args.stage == "fit":
            from ije.experiment import fit_final as execute
        elif args.stage == "evaluate":
            from ije.experiment import evaluate as execute
        elif args.stage == "explain":
            from ije.explain import explain as execute
        else:
            from ije.report import report as execute
        files = execute(config, args.config.resolve(), run_dir)
        finish_stage(run_dir, args.stage, files)
        print(f"Completed {args.stage}: {run_dir}")
    except Exception:
        write_json(run_dir / f"{args.stage}.failed.json", {"at": now(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        sys.exit(1)
