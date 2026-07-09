# -*- coding: utf-8 -*-
"""
Master runner for QIHC three-law experiments (NE1–NE9, Tier 0/1/2).

Usage:
  python experiments/nsfc_evidence/run_three_law_suite.py --profile smoke
  python experiments/nsfc_evidence/run_three_law_suite.py --profile full
  python experiments/nsfc_evidence/run_three_law_suite.py --profile tier0
  python experiments/nsfc_evidence/run_three_law_suite.py --profile tier012
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.logging_utils import ExperimentLogger  # noqa: E402

PYTHON = sys.executable

# (name, script, smoke_args_extra, full_args_extra)
NE_STEPS = [
    ("ne2_contraction", "experiments/nsfc_evidence/run_ne2_contraction.py"),
    ("ne4_stale_field", "experiments/nsfc_evidence/run_ne4_stale_field.py"),
    ("ne9_free_energy", "experiments/nsfc_evidence/run_ne9_free_energy.py"),
    ("ne3_refresh", "experiments/nsfc_evidence/run_ne3_refresh.py"),
    ("ne6_trust_gate", "experiments/nsfc_evidence/run_ne6_trust_gate.py"),
    ("ne1_division", "experiments/nsfc_evidence/run_ne1_division.py"),
    ("ne5_snr_lambda", "experiments/nsfc_evidence/run_ne5_snr_lambda.py"),
    ("ne7_trust_proxy", "experiments/nsfc_evidence/run_ne7_trust_proxy.py"),
    ("ne8_pareto", "experiments/nsfc_evidence/run_ne8_pareto.py"),
]

TIER0 = {"ne2_contraction", "ne4_stale_field", "ne9_free_energy"}
TIER1 = {"ne3_refresh", "ne6_trust_gate"}
TIER2 = {"ne1_division", "ne5_snr_lambda", "ne7_trust_proxy", "ne8_pareto"}


def _select(names: set[str]) -> list[tuple[str, str]]:
    return [s for s in NE_STEPS if s[0] in names]


PROFILES = {
    "smoke": {
        "description": "Fast local smoke for all NE1–NE9 (~5–15 min CPU)",
        "steps": NE_STEPS,
        "profile_flag": "smoke",
    },
    "tier0": {
        "description": "Tier0 only: NE2/NE4/NE9",
        "steps": _select(TIER0),
        "profile_flag": "full",
    },
    "tier1": {
        "description": "Tier1 only: NE3/NE6",
        "steps": _select(TIER1),
        "profile_flag": "full",
    },
    "tier2": {
        "description": "Tier2 only: NE1/NE5/NE7/NE8",
        "steps": _select(TIER2),
        "profile_flag": "full",
    },
    "tier012": {
        "description": "Full Tier0+1+2 for NSFC preliminary evidence",
        "steps": NE_STEPS,
        "profile_flag": "full",
    },
    "full": {
        "description": "Alias of tier012",
        "steps": NE_STEPS,
        "profile_flag": "full",
    },
}


def run_py(logger: ExperimentLogger, name: str, script: str, args: list[str]) -> int:
    argv = [PYTHON, str(REPO_ROOT / script)] + args
    return logger.run_subprocess(name, argv, REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="QIHC three-law suite (NE1–NE9)")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="smoke")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Run only these step names")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "experiments" / "outputs" / "nsfc_evidence" / f"run_three_law_{ts}"
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(run_dir, run_name=f"three_law_{args.profile}")
    logger.log(f"Profile: {args.profile} — {profile['description']}")
    logger.log(f"Repo: {REPO_ROOT}")
    logger.save_json(
        "profile.json",
        {
            "profile": args.profile,
            "description": profile["description"],
            "steps": [s[0] for s in profile["steps"]],
            "ne_profile_flag": profile["profile_flag"],
        },
    )

    failed = []
    for name, script in profile["steps"]:
        if args.only and name not in args.only:
            continue
        sub = run_dir / name
        step_args = ["--profile", profile["profile_flag"], "--output-dir", str(sub)]
        code = run_py(logger, name, script, step_args)
        if code != 0:
            failed.append(name)

    # Aggregate index
    index = {"profile": args.profile, "steps": {}, "failed": failed}
    for name, _ in profile["steps"]:
        summary_path = run_dir / name / "summary.json"
        index["steps"][name] = {
            "ok": name not in failed,
            "summary": str(summary_path) if summary_path.exists() else None,
        }
    logger.save_json("three_law_index.json", index)

    status = "completed" if not failed else "completed_with_errors"
    logger.save_json("failed_steps.json", {"failed": failed})
    logger.finalize(status)

    if failed:
        logger.log(f"Failed steps: {failed}", level="WARN")
        return 1
    logger.log("All three-law steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
