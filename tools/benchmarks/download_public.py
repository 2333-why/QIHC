#!/usr/bin/env python3
"""Pinned public test acquisition. Standard library only; never executes dataset code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unicodedata
import urllib.request

LOCK = Path(__file__).with_name('public_benchmarks.lock.json')
ENDPOINTS = ('https://hf-mirror.com', 'https://huggingface.co')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid(data, spec):
    return len(data) == spec['bytes'] and digest(data) == spec['sha256']


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download(spec, lock, destination, endpoints, attempts=2, timeout=45):
    if destination.exists() and valid(destination.read_bytes(), spec):
        return 'verified_cache'
    # urllib does not read HF_TOKEN or cached Hub credentials. Public files only.
    errors = []
    for endpoint in endpoints:
        url = f"{endpoint}/datasets/{lock['repo']}/resolve/{lock['revision']}/{spec['name']}"
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'QIHC-benchmark-audit/1'})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read(spec['bytes'] + 1)
                if not valid(data, spec):
                    raise ValueError('size/SHA256 mismatch (not accepted)')
                atomic_write(destination, data)
                return endpoint
            except (OSError, ValueError) as exc:
                message = f"{spec['name']} {endpoint} attempt {attempt + 1}: {type(exc).__name__}: {exc}"
                print(message, flush=True)
                errors.append(message)
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 8))
    raise RuntimeError('\n'.join(errors))


def read_rows(path):
    text = path.read_text(encoding='utf-8-sig')
    # IndustryOR_fixedV2.json is JSONL despite its .json suffix.
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'{path.name}: expected object records')
    return rows


def question_hash(question):
    return digest(' '.join(unicodedata.normalize('NFKC', question).casefold().split()).encode())


def finite_number(value):
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def audit(lock, raw):
    inventory, inputs, labels = [], [], []
    groups = defaultdict(list)
    for spec in lock['files']:
        path = raw / spec['name']
        if not valid(path.read_bytes(), spec):
            raise ValueError(f"Integrity failure: {spec['name']}")
        if 'rows' not in spec:
            continue
        rows = read_rows(path)
        if len(rows) != spec['rows']:
            raise ValueError(f"Row count mismatch: {spec['name']}")
        fields = Counter()
        numeric = 0
        for index, row in enumerate(rows):
            question = row.get('en_question')
            if not isinstance(question, str) or not question.strip() or 'en_answer' not in row:
                raise ValueError(f"Invalid question/answer schema: {spec['name']}:{index}")
            fields.update(row.keys())
            numeric += finite_number(row['en_answer'])
            key = f"{Path(spec['name']).stem}:{index:04d}"
            qhash = question_hash(question)
            groups[qhash].append(key)
            # No answers, gold IR, solver scripts, or hidden hints enter model inputs.
            inputs.append({'id': key, 'dataset': Path(spec['name']).stem,
                           'question': question, 'question_sha256': qhash,
                           'split': 'external_test_quarantine'})
            labels.append({'id': key, 'reference_answer': row['en_answer'],
                           'reference_optimality': 'not_independently_verified',
                           'independent_constraint_verifier': None,
                           'pbit_eligibility': 'pending_semantic_and_encoding_review'})
        inventory.append({'file': spec['name'], 'rows': len(rows), 'fields': dict(fields),
                          'finite_numeric_answers': numeric,
                          'other_answers': len(rows) - numeric,
                          'pbit_formal_ready': False})
    report = {'status': 'download_verified_adaptation_pending', 'repo': lock['repo'],
              'revision': lock['revision'], 'total_rows': len(inputs),
              'datasets': inventory, 'unique_normalized_questions': len(groups),
              'duplicate_groups': [v for v in groups.values() if len(v) > 1],
              'warning': 'Objective agreement does not establish correct modeling or feasibility. '
                         'No rows approved for p-bit formal evaluation yet. Do not train on these test data.'}
    return report, inputs, labels


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', action='append', choices=ENDPOINTS,
                        help='Repeat for fallback order; default mirror then official.')
    parser.add_argument('--verify-only', action='store_true', help='No network; verify and rebuild audit.')
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    if args.attempts < 1 or args.timeout <= 0:
        parser.error('attempts and timeout must be positive')
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    root = args.output.resolve() / lock['revision']
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two jobs replacing each other's manifest or reports.
    guard = root / '.download.lock'
    try:
        fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'Active/stale lock: {guard}. Check no downloader is running before removing it.')
    os.close(fd)
    try:
        manifest = {'status': 'in_progress', 'started_utc': datetime.now(timezone.utc).isoformat(),
                    'lock_sha256': digest(LOCK.read_bytes()), 'files': []}
        atomic_write(root / 'manifest.json', json_bytes(manifest))
        for spec in lock['files']:
            target = root / 'raw' / spec['name']
            if args.verify_only:
                if not target.exists() or not valid(target.read_bytes(), spec):
                    raise ValueError(f"Missing/corrupt local file: {spec['name']}")
                origin = 'verified_local'
            else:
                origin = download(spec, lock, target, args.endpoint or ENDPOINTS,
                                  args.attempts, args.timeout)
            manifest['files'].append(dict(spec, source=origin))
            print(f"VERIFIED {spec['name']} ({origin})", flush=True)
        report, inputs, labels = audit(lock, root / 'raw')
        for name, rows in [('questions.jsonl', inputs), ('reference_answers.DO_NOT_PROMPT.jsonl', labels)]:
            atomic_write(root / name, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8'))
        atomic_write(root / 'audit.json', json_bytes(report))
        atomic_write(root / 'source.lock.json', LOCK.read_bytes())
        manifest['status'] = report['status']
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_write(root / 'manifest.json', json_bytes(manifest))
        print(json.dumps({'status': manifest['status'], 'rows': len(inputs), 'output': str(root)}), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        atomic_write(root / 'manifest.json', json_bytes(manifest))
        raise
    finally:
        guard.unlink()


if __name__ == '__main__':
    main()
