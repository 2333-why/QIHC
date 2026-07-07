# -*- coding: utf-8 -*-
"""
Master runner for NSFC evidence chain (all experiments + logging).

Usage:
    python experiments/nsfc_evidence/run_evidence_suite.py --profile smoke
    python experiments/nsfc_evidence/run_evidence_suite.py --profile server
    python experiments/nsfc_evidence/run_evidence_suite.py --profile p012
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.nsfc_evidence.logging_utils import ExperimentLogger  # noqa: E402

PYTHON = sys.executable


def _model_name() -> str:
    return os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


def _model_14b() -> str:
    return os.environ.get("MODEL_NAME_14B", "Qwen/Qwen2.5-14B-Instruct")


def run_py(logger: ExperimentLogger, name: str, script: str, args: list[str]) -> int:
    argv = [PYTHON, str(REPO_ROOT / script)] + args
    return logger.run_subprocess(name, argv, REPO_ROOT)


def _p012_steps() -> list[tuple[str, str, list[str]]]:
    """P0–P2 cases + mechanism evidence (handoff/pareto/F/sampler)."""
    m = _model_name()
    m14 = _model_14b()
    return [
        # P0
        (
            "case_a_cr_subset",
            "experiments/nsfc_evidence/run_case_a_cr_subset.py",
            ["--n-tasks", "200", "--n-samples", "50", "--budget-steps", "400"],
        ),
        (
            "case_b_dual_synthetic",
            "experiments/nsfc_evidence/run_case_b_dual_synthetic.py",
            ["--n-tasks", "200", "--budget-steps", "400"],
        ),
        # P1
        (
            "case_d_cr_by_task",
            "experiments/nsfc_evidence/run_case_d_cr_by_task.py",
            ["--n-tasks", "200", "--n-samples", "50", "--budget-steps", "400"],
        ),
        (
            "case_e_cr_bundled_full",
            "experiments/nsfc_evidence/run_case_e_cr_bundled_full.py",
            ["--n-samples", "50", "--budget-steps", "400"],
        ),
        (
            "case_f_vci_ablation",
            "experiments/nsfc_evidence/run_case_f_vci_ablation.py",
            ["--n-tasks", "200", "--budget-steps", "400"],
        ),
        # Mechanism (fast, supports narrative)
        (
            "dual_bundled",
            "experiments/nsfc_evidence/run_dual_evidence.py",
            ["--source", "bundled", "--budget-steps", "400"],
        ),
        (
            "cr_bundled",
            "experiments/nsfc_evidence/run_cr_bbh.py",
            ["--source", "bundled", "--n-samples", "50", "--budget-steps", "400"],
        ),
        ("f_trajectories", "experiments/nsfc_evidence/run_f_trajectories.py", ["--budget-steps", "400"]),
        ("sampler_ablation", "experiments/nsfc_evidence/run_sampler_ablation.py", ["--steps", "400"]),
        (
            "handoff",
            "experiments/run_vci_handoff.py",
            [
                "--source", "bundled", "--logits", "llm", "--model-name", m,
                "--limit", "40", "--budget-steps", "400", "--noise-scales-dense",
                "--seeds", "0", "1", "2",
            ],
        ),
        (
            "pareto",
            "experiments/run_vci_pareto.py",
            ["--budgets", "100", "200", "300", "400", "600", "--include-cr"],
        ),
        (
            "unified_ablation_bundled",
            "experiments/nsfc_evidence/run_unified_ablation.py",
            ["--dataset", "bundled", "--budget-steps", "400", "--n-samples", "50", "--seeds", "0", "1", "2"],
        ),
        (
            "unified_ablation_synthetic",
            "experiments/nsfc_evidence/run_unified_ablation.py",
            [
                "--dataset", "synthetic", "--n-tasks", "200",
                "--budget-steps", "400", "--n-samples", "50", "--seeds", "0", "1", "2",
            ],
        ),
        (
            "scaling",
            "experiments/run_sampler_scaling.py",
            ["--nodes", "50", "100", "200", "500", "--trials", "5", "--measure-tts"],
        ),
        # P2 (GPU-heavy)
        (
            "case_g_constrained_bbh",
            "experiments/nsfc_evidence/run_case_g_constrained_bbh.py",
            [
                "--limit-per-task", "50",
                "--logits", "llm",
                "--model-name", m,
                "--budget-steps", "400",
            ],
        ),
        (
            "case_h_model_compare",
            "experiments/nsfc_evidence/run_case_h_model_compare.py",
            [
                "--n-tasks", "200",
                "--model-7b", m,
                "--model-14b", m14,
                "--budget-steps", "400",
            ],
        ),
    ]


PROFILES = {
    "smoke": {
        "description": "Fast local smoke test (~2 min CPU)",
        "steps": [
            ("dual_bundled", "experiments/nsfc_evidence/run_dual_evidence.py", ["--source", "bundled", "--budget-steps", "150"]),
            ("f_trajectories", "experiments/nsfc_evidence/run_f_trajectories.py", ["--budget-steps", "150"]),
            ("sampler_ablation", "experiments/nsfc_evidence/run_sampler_ablation.py", ["--steps", "150"]),
            ("cr_mock", "experiments/nsfc_evidence/run_cr_bbh.py", ["--source", "bundled", "--limit", "15", "--n-samples", "20", "--budget-steps", "150"]),
            ("handoff", "experiments/run_vci_handoff.py", ["--source", "bundled", "--limit", "20", "--budget-steps", "150"]),
            ("pareto", "experiments/run_vci_pareto.py", ["--budgets", "100", "200", "--limit", "20"]),
            ("scaling", "experiments/run_sampler_scaling.py", ["--nodes", "12", "14", "16", "--trials", "3", "--steps", "400"]),
        ],
    },
    "server": {
        "description": "Full H200 server run for NSFC evidence (~2-6 hours)",
        "steps": [
            (
                "dual_bundled",
                "experiments/nsfc_evidence/run_dual_evidence.py",
                ["--source", "bundled", "--budget-steps", "400"],
            ),
            (
                "dual_hf_llm",
                "experiments/nsfc_evidence/run_dual_evidence.py",
                [
                    "--source", "hf",
                    "--logits", "llm",
                    "--model-name", _model_name(),
                    "--limit-per-task", "50",
                    "--budget-steps", "400",
                ],
            ),
            (
                "cr_protocol_llm",
                "experiments/nsfc_evidence/run_cr_bbh.py",
                [
                    "--source", "hf",
                    "--use-llm",
                    "--model-name", _model_name(),
                    "--n-samples", "50",
                    "--limit-per-task", "30",
                    "--budget-steps", "400",
                ],
            ),
            (
                "cr_bundled",
                "experiments/nsfc_evidence/run_cr_bbh.py",
                ["--source", "bundled", "--n-samples", "50", "--budget-steps", "400"],
            ),
            ("f_trajectories", "experiments/nsfc_evidence/run_f_trajectories.py", ["--budget-steps", "400"]),
            ("sampler_ablation", "experiments/nsfc_evidence/run_sampler_ablation.py", ["--steps", "400"]),
            (
                "handoff",
                "experiments/run_vci_handoff.py",
                ["--source", "bundled", "--logits", "llm", "--model-name", _model_name(), "--limit", "40", "--budget-steps", "400"],
            ),
            ("pareto", "experiments/run_vci_pareto.py", ["--budgets", "100", "200", "300", "400", "600"]),
            (
                "scaling",
                "experiments/run_sampler_scaling.py",
                ["--nodes", "12", "14", "16", "18", "20", "22", "--trials", "10", "--steps", "800"],
            ),
            (
                "bbh_hf_full",
                "experiments/run_vci_bbh.py",
                [
                    "--source", "hf",
                    "--logits", "llm",
                    "--model-name", _model_name(),
                    "--limit-per-task", "50",
                    "--budget-steps", "400",
                ],
            ),
        ],
    },
    "p012": {
        "description": "P0–P2 NSFC cases on Pro 6000 (~4–10 hours with 7B+14B)",
        "steps": _p012_steps(),
        "post_steps": ["c_compute_budget"],
    },
    "pre_submission": {
        "description": "立项前补全七项证据（unified ablation + handoff dense + pareto + scaling TTS）",
        "steps": [
            s for s in _p012_steps()
            if s[0]
            in (
                "case_e_cr_bundled_full",
                "unified_ablation_bundled",
                "unified_ablation_synthetic",
                "handoff",
                "pareto",
                "scaling",
                "case_g_constrained_bbh",
            )
        ],
        "post_steps": ["c_compute_budget"],
    },
}


def _redirect_output_dirs(profile_steps: list, run_dir: Path) -> list[tuple[str, str, list[str]]]:
    """Inject --output-dir for each step into run_dir subfolder."""
    out: list[tuple[str, str, list[str]]] = []
    for name, script, args in profile_steps:
        sub = run_dir / name.replace("/", "_")
        new_args = list(args) + ["--output-dir", str(sub)]
        out.append((name, script, new_args))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="NSFC evidence suite")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="smoke")
    parser.add_argument("--run-dir", default=None, help="Override output run directory")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "experiments" / "outputs" / "nsfc_evidence" / f"run_{ts}"
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(run_dir, run_name=f"nsfc_{args.profile}")
    logger.log(f"Profile: {args.profile} — {profile['description']}")
    logger.log(f"Repo: {REPO_ROOT}")
    logger.log(f"Python: {PYTHON}")
    logger.log(f"MODEL_NAME: {_model_name()}")
    logger.log(f"MODEL_NAME_14B: {_model_14b()}")
    logger.save_json(
        "profile.json",
        {
            "profile": args.profile,
            **profile,
            "steps": [s[0] for s in profile["steps"]],
            "model_name": _model_name(),
            "model_name_14b": _model_14b(),
        },
    )

    steps = _redirect_output_dirs(profile["steps"], run_dir)
    failed = []
    for name, script, step_args in steps:
        code = run_py(logger, name, script, step_args)
        if code != 0:
            failed.append(name)

    # P0-C: aggregate compute budget from this run
    if profile.get("post_steps"):
        for post in profile["post_steps"]:
            post_args = ["--run-dir", str(run_dir), "--output-dir", str(run_dir / post)]
            code = run_py(logger, post, f"experiments/nsfc_evidence/run_case_{post}.py", post_args)
            if code != 0:
                failed.append(post)

    if not args.skip_plot:
        plot_argv = [
            PYTHON,
            str(REPO_ROOT / "experiments/nsfc_evidence/plot_evidence_report.py"),
            "--run-dir",
            str(run_dir),
        ]
        logger.run_subprocess("plot_report", plot_argv, REPO_ROOT)

    status = "completed" if not failed else "completed_with_errors"
    logger.save_json("failed_steps.json", {"failed": failed})
    logger.finalize(status)

    if failed:
        logger.log(f"Failed steps: {failed}", level="WARN")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
