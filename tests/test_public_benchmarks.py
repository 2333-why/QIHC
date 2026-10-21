import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import io
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('public_download', Path(__file__).resolve().parents[1] / 'tools/benchmarks/download_public.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class PublicBenchmarksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, rows):
        data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        spec = {'name': 'sample.json', 'bytes': len(data), 'rows': len(rows),
                'sha256': hashlib.sha256(data).hexdigest()}
        (self.root / spec['name']).write_bytes(data)
        return spec, {'repo': 'owner/repo', 'revision': 'abc', 'files': [spec]}

    def test_json_suffix_jsonl_and_no_answer_leak(self):
        _, lock = self.fixture([{'en_question': ' A question ', 'en_answer': 1},
                                {'en_question': 'a QUESTION', 'en_answer': 'infeasible'}])
        report, inputs, labels = mod.audit(lock, self.root)
        self.assertEqual(report['unique_normalized_questions'], 1)
        self.assertEqual(report['datasets'][0]['finite_numeric_answers'], 1)
        self.assertEqual(len(report['duplicate_groups']), 1)
        self.assertNotIn('reference_answer', inputs[0])
        self.assertIsNone(labels[0]['independent_constraint_verifier'])
        self.assertFalse(report['datasets'][0]['pbit_formal_ready'])

    def test_array(self):
        path = self.root / 'array.jsonl'
        path.write_text('[{"x": 1}]')
        self.assertEqual(mod.read_rows(path), [{'x': 1}])

    def test_corruption_rejected(self):
        spec, lock = self.fixture([{'en_question': 'q', 'en_answer': 1}])
        (self.root / spec['name']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'Integrity'):
            mod.audit(lock, self.root)

    def test_schema_and_counts_rejected(self):
        spec, lock = self.fixture([{'en_question': 'q'}])
        with self.assertRaisesRegex(ValueError, 'schema'):
            mod.audit(lock, self.root)
        spec['rows'] = 2
        with self.assertRaisesRegex(ValueError, 'count'):
            mod.audit(lock, self.root)

    def test_cached_file_does_not_network(self):
        spec, lock = self.fixture([{'en_question': 'q', 'en_answer': 1}])
        with patch.object(mod.urllib.request, 'urlopen', side_effect=AssertionError):
            self.assertEqual(mod.download(spec, lock, self.root / spec['name'], mod.ENDPOINTS), 'verified_cache')

    def test_mirror_bad_content_falls_back_and_no_token(self):
        spec, lock = self.fixture([{'en_question': 'q', 'en_answer': 1}])
        good = (self.root / spec['name']).read_bytes()
        target = self.root / 'new.json'
        with patch.object(mod.urllib.request, 'urlopen', side_effect=[io.BytesIO(b'bad'), io.BytesIO(good)]) as call:
            origin = mod.download(spec, lock, target, mod.ENDPOINTS, attempts=1)
        self.assertEqual(origin, mod.ENDPOINTS[1])
        self.assertEqual(target.read_bytes(), good)
        for args in call.call_args_list:
            self.assertNotIn('Authorization', args.args[0].headers)

    def test_failure_preserves_existing_bytes(self):
        spec, lock = self.fixture([{'en_question': 'q', 'en_answer': 1}])
        target = self.root / 'damaged.json'
        target.write_bytes(b'existing')
        with patch.object(mod.urllib.request, 'urlopen', side_effect=OSError('unavailable')):
            with self.assertRaises(RuntimeError):
                mod.download(spec, lock, target, mod.ENDPOINTS, attempts=1)
        self.assertEqual(target.read_bytes(), b'existing')

    def test_numeric_answer(self):
        for value in [True, None, 'NaN', 'inf', 'infeasible']:
            self.assertFalse(mod.finite_number(value))
        self.assertTrue(mod.finite_number('12.5'))


if __name__ == '__main__':
    unittest.main()
