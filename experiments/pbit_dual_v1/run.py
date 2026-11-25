"""Stage entry point for E005-1.18."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "gpu", "finish", "report", "worker")
    )
    parser.add_argument("task", nargs="?")
    args = parser.parse_args()
    if args.stage == "prepare":
        from .prepare import prepare

        prepare()
    elif args.stage == "gpu":
        from .gpu import launch

        launch()
    elif args.stage == "worker":
        from .gpu import worker

        worker(args.task)
    elif args.stage == "finish":
        from .cpu import finish

        finish()
    else:
        from .report import report

        report()


if __name__ == "__main__":
    main()
