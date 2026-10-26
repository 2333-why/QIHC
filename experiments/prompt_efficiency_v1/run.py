"""Thin entrypoint: experiment definition + reusable modeling runner."""
from qihc.experiment_runtime.runner import main
from . import study


if __name__ == "__main__":
    raise SystemExit(main(study))
