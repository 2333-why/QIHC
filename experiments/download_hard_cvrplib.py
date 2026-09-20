#!/usr/bin/env python3
"""Download a reproducible X-family pilot directly from official CVRPLIB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://galgos.inf.puc-rio.br/cvrplib"


def fetch(url: str) -> bytes:
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "QIHC research benchmark downloader/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def discover(html: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r'href="(?P<path>[^\"]+/download/instance/(?P<id>\d+))"[^>]*>\s*'
        r'(?P<name>X-n\d+-k\d+)\s*</a>', re.IGNORECASE | re.DOTALL,
    )
    return [(match.group("name"), match.group("id")) for match in pattern.finditer(html)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-customers", type=int, default=200)
    parser.add_argument("--max-customers", type=int, default=400)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--selection", choices=["smallest", "spread"], default="smallest")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html = fetch(BASE + "/en/instances").decode("utf-8", errors="replace")
    cases = [(name, code) for name, code in discover(html)
             if args.min_customers <= int(name.split("-")[1][1:]) - 1 <= args.max_customers]
    if not cases:
        raise RuntimeError("No matching X instances found on the official index")
    if args.limit and args.selection == "spread" and len(cases) > args.limit:
        if args.limit == 1:
            cases = cases[:1]
        else:
            indices = [round(i * (len(cases) - 1) / (args.limit - 1)) for i in range(args.limit)]
            cases = [cases[index] for index in indices]
    else:
        cases = cases[:args.limit or None]
    manifest = []
    for name, code in cases:
        entry = {"name": name, "official_id": code, "files": {}}
        for suffix, kind, marker in (("vrp", "instance", b"NODE_COORD_SECTION"),
                                     ("sol", "bks", b"Route #")):
            url = f"{BASE}/en/download/{kind}/{code}"
            target = args.output_dir / f"{name}.{suffix}"
            payload = target.read_bytes() if target.exists() else fetch(url)
            if marker not in payload:
                raise ValueError(f"Unexpected official response for {url}")
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(target)
            entry["files"][suffix] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
        manifest.append(entry)
        print(json.dumps({"downloaded": name}, ensure_ascii=False), flush=True)
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
