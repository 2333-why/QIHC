#!/usr/bin/env python3
"""Load the exact QIHC LLM path and run one short structured generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    generator = LocalDirectLLMGenerator(
        args.model_path,
        device=args.device,
        max_new_tokens=64,
        temperature=0.0,
        max_input_tokens=1024,
    )
    output = generator.generate(
        'Return exactly one compact JSON object with this content: {"status":"ok"}'
    )
    print(json.dumps({"device": args.device, "output": output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
