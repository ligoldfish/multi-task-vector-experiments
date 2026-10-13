"""Local fixture checks; no model weights, network hosts or experiment data."""
import hashlib
import importlib.util
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import types
import unittest
from unittest.mock import patch
import zipfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('asset_downloader', HERE / 'download_task_vector_assets.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FIXTURES = HERE / ('verification-fixtures-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
FIXTURES.mkdir()
PAYLOAD = b'pinned original bytes, resume safely\n' * 100
RANGES = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        RANGES.append((self.path, self.headers.get('Range')))
        if self.path == '/fail':
            self.send_response(503)
            self.end_headers()
            return
        data = b'X' * len(PAYLOAD) if self.path == '/corrupt' else PAYLOAD
        offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        ranged = self.headers.get('Range') and self.path != '/ignore'
        self.send_response(206 if ranged else 200)
        if ranged:
            self.send_header('Content-Range', f'bytes {offset}-{len(data)-1}/{len(data)}')
            data = data[offset:]
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def run_download(self, name, paths, partial=None):
        root = FIXTURES / name
        root.mkdir()
        if partial is not None:
            (root / 'asset.bin.partial').write_bytes(partial)
        item = {'destination': 'asset.bin', 'size': len(PAYLOAD),
                'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [self.url + path for path in paths]}
        with patch.object(module.time, 'sleep'):
            result = module.acquire(root, item)
        self.assertEqual((root / 'asset.bin').read_bytes(), PAYLOAD)
        self.assertEqual(result['sha256'], hashlib.sha256(PAYLOAD).hexdigest())
        return root, item

    def test_fresh_download_and_reuse(self):
        root, item = self.run_download('fresh', ['/ok'])
        count = len(RANGES)
        result = module.acquire(root, item)
        self.assertTrue(result['reused'])
        self.assertEqual(count, len(RANGES))

    def test_resume_206(self):
        self.run_download('resume', ['/ok'], PAYLOAD[:123])
        self.assertEqual(RANGES[-1], ('/ok', 'bytes=123-'))

    def test_server_ignores_range_200(self):
        self.run_download('ignore', ['/ignore'], PAYLOAD[:123])

    def test_mirror_failure_uses_second_pinned_source(self):
        self.run_download('fallback', ['/fail', '/ok'])
        self.assertEqual(RANGES[-1][0], '/ok')

    def test_checksum_failure_preserved_then_fallback(self):
        root, _ = self.run_download('corruption', ['/corrupt', '/ok'])
        self.assertEqual(len(list(root.glob('*.invalid-*'))), 3)

    def test_existing_corrupt_asset_is_never_replaced(self):
        root = FIXTURES / 'existing-corrupt'
        root.mkdir()
        path = root / 'asset.bin'
        path.write_bytes(b'X' * len(PAYLOAD))
        item = {'destination': 'asset.bin', 'size': len(PAYLOAD),
                'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [self.url + '/ok']}
        with self.assertRaises(ValueError):
            module.acquire(root, item)
        self.assertEqual(path.read_bytes(), b'X' * len(PAYLOAD))

    def test_path_boundaries(self):
        for raw in ('../outside', '/absolute', 'C:/outside', 'a\\b'):
            with self.assertRaises(ValueError):
                module.child(FIXTURES, raw)

    def test_exact_proofwriter_members_and_unknown_retained(self):
        root = FIXTURES / 'proofwriter'
        (root / 'archives').mkdir(parents=True)
        members = []
        with zipfile.ZipFile(root / 'archives/test.zip', 'w') as archive:
            for depth in range(4):
                for split in ('train', 'dev', 'test'):
                    path = f'release/OWA/depth-{depth}/meta-{split}.jsonl'
                    row = {'id': str(depth), 'theory': 'Original fact.', 'questions': {
                        'Q1': {'question': 'Fact?', 'answer': True, 'QDep': 0},
                        'Q2': {'question': 'Negated fact?', 'answer': False, 'QDep': 0},
                        'Q3': {'question': 'Unprovable?', 'answer': 'Unknown', 'QDep': -1}}}
                    payload = (json.dumps(row) + '\n').encode()
                    archive.writestr(path, payload)
                    members.append({'path': path, 'size': len(payload), 'split': split, 'depth': depth})
            archive.writestr('release/OWA/depth-3/meta-abduct-train.jsonl', 'unrelated task')
        paths, records = module.extract_proofwriter(root, {'filename': 'test.zip', 'members': members})
        self.assertEqual(set(paths), {'train', 'validation', 'test'})
        self.assertEqual(len(records), 12)
        self.assertTrue(all(record['labels']['Unknown'] == 1 for record in records))
        self.assertEqual(list(root.rglob('meta-abduct-*')), [])

    def test_samsum_empty_dialogue_preserved_and_reuse_hashes_checked(self):
        root = FIXTURES / 'samsum'
        (root / 'archives').mkdir(parents=True)
        (root / 'archives/corpus.7z').write_bytes(b'fixture archive')
        rows = {'train.json': [{'id': 'train1', 'dialogue': '', 'summary': 'Original summary'}],
                'val.json': [{'id': 'val1', 'dialogue': 'D', 'summary': 'S'}],
                'test.json': [{'id': 'test1', 'dialogue': 'D', 'summary': 'S'}]}

        class FakeArchive:
            def __init__(self, *args):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def list(self):
                return [types.SimpleNamespace(filename=name, is_symlink=False, is_directory=False) for name in rows]
            def extractall(self, path):
                for name, values in rows.items():
                    (path / name).write_text(json.dumps(values), encoding='utf-8')

        archive = {'filename': 'corpus.7z', 'sha256': '123456789abcdef0',
                   'expected_rows': {'train': 1, 'validation': 1, 'test': 1}}
        with patch.dict(module.sys.modules, {'py7zr': types.SimpleNamespace(SevenZipFile=FakeArchive)}):
            paths, _ = module.extract_samsum(root, archive)
            self.assertEqual(json.loads(Path(paths['train']).read_text())[0]['dialogue'], '')
            reused, _ = module.extract_samsum(root, archive)
            self.assertEqual(paths, reused)
            changed = rows['train.json'][0].copy()
            changed['summary'] = 'Changed summary'
            Path(paths['train']).write_text(json.dumps([changed]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed after extraction'):
                module.extract_samsum(root, archive)


if __name__ == '__main__':
    unittest.main(verbosity=2)
