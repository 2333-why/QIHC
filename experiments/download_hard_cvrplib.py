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
    parser.add_argument("--reserve-factor", type=int, default=1, help="Download a larger candidate pool, capped by available instances")
    parser.add_argument("--selection", choices=["smallest", "spread"], default="smallest")
    parser.add_argument("--manifest-name", default="download_manifest.json")
    parser.add_argument("--verify-cached", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.reserve_factor < 1:
        parser.error("limit must be nonnegative and reserve-factor must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = args.output_dir / args.manifest_name
    previous = {}
    if previous_manifest.exists():
        try:
            previous = {row["name"]: row for row in json.loads(previous_manifest.read_text(encoding="utf-8"))}
        except (ValueError, KeyError):
            previous = {}
    html = fetch(BASE + "/en/instances").decode("utf-8", errors="replace")
    cases = [(name, code) for name, code in discover(html)
             if args.min_customers <= int(name.split("-")[1][1:]) - 1 <= args.max_customers]
    cases.sort(key=lambda item: (int(item[0].split("-")[1][1:]) - 1, item[0]))
    if not cases:
        raise RuntimeError("No matching X instances found on the official index")
    if args.limit and len(cases) < args.limit:
        raise ValueError(f"Requested {args.limit} X instances, but only {len(cases)} are available in the selected size range")
    pool_limit = min(len(cases), args.limit * args.reserve_factor) if args.limit else len(cases)
    def choose(items, count):
        if count <= 0:
            return []
        if args.selection != "spread" or len(items) <= count:
            return items[:count]
        if count == 1:
            return items[:1]
        return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]
    # Keep the original primary selection stable; add backups rather than reselect it.
    primary = choose(cases, args.limit or len(cases))
    primary_names = {name for name, _ in primary}
    reserves = choose([item for item in cases if item[0] not in primary_names], pool_limit - len(primary))
    cases = primary + reserves
    manifest = []
    for name, code in cases:
        entry = {"name": name, "official_id": code, "candidate_role": "primary" if name in primary_names else "reserve", "files": {}}
        for suffix, kind, marker in (("vrp", "instance", b"NODE_COORD_SECTION"),
                                     ("sol", "bks", b"Route #")):
            url = f"{BASE}/en/download/{kind}/{code}"
            target = args.output_dir / f"{name}.{suffix}"
            payload = target.read_bytes() if target.exists() else None
            expected_sha = previous.get(name, {}).get("files", {}).get(suffix, {}).get("sha256")
            cached_valid = (payload is not None and marker in payload and (
                not args.verify_cached or expected_sha == hashlib.sha256(payload).hexdigest()))
            replace_cache = not cached_valid
            if replace_cache:
                payload = fetch(url)
            if marker not in payload:
                raise ValueError(f"Unexpected official response for {url}")
            if replace_cache:
                if target.exists():
                    backup = target.with_suffix(target.suffix + f".invalid-{time.time_ns()}.bak")
                    backup.write_bytes(target.read_bytes())
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(target)
            entry["files"][suffix] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
        manifest.append(entry)
        print(json.dumps({"downloaded": name}, ensure_ascii=False), flush=True)
    (args.output_dir / args.manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
