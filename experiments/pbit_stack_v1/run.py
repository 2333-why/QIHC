"""CPU and four-card entry point for the frozen and improved comparisons."""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=(
            "prepare-current",
            "prepare-improved",
            "gpu-current",
            "gpu-improved",
            "finish-current",
            "finish-improved",
            "report",
            "worker",
        ),
    )
    p.add_argument(
        "stage", nargs="?", choices=("current", "improved"), default="improved"
    )
    p.add_argument("task", nargs="?")
    args = p.parse_args()
    if args.command == "worker":
        from .gpu import worker

        worker(args.stage, args.task)
    elif args.command == "report":
        from .report import report

        report(args.stage)
    else:
        op, stage = args.command.split("-")
        if op == "prepare":
            from .prepare import prepare_current, prepare_improved

            (prepare_current if stage == "current" else prepare_improved)()
        elif op == "gpu":
            from .gpu import launch

            launch(stage)
        else:
            from .cpu import finish

            finish(stage)


if __name__ == "__main__":
    main()
