import argparse


def main():
    parser = argparse.ArgumentParser(description="E005-1.20 DCP transfer")
    parser.add_argument(
        "command", choices=("prepare", "gpu", "worker", "finish", "report")
    )
    parser.add_argument("task", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare

        prepare()
    elif args.command == "gpu":
        from .gpu import launch

        launch()
    elif args.command == "worker":
        from .gpu import worker

        worker(args.task)
    elif args.command == "finish":
        from .score import finish

        finish()
    else:
        from .score import report

        report()


if __name__ == "__main__":
    main()
