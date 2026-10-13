"""Local fixture checks; no model weights, network hosts or experiment data."""
import hashlib
import importlib.util
import json
import ipaddress
import ssl
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import types
import unittest
from unittest.mock import patch
import zipfile
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('asset_downloader', HERE / 'download_task_vector_assets_tlsfix.py')
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
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/ok')
            self.end_headers()
            return
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

    def do_HEAD(self):
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/ok')
        else:
            self.send_response(200)
        self.end_headers()


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

    def test_existing_ca_bundle_repairs_empty_platform_ca_store(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.assertEqual(context.cert_store_stats()['x509_ca'], 0)
        with patch.object(module.ssl, 'create_default_context', return_value=context):
            evidence = module.configure_tls()
        self.assertGreater(evidence['effective_store']['x509_ca'], 0)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        module.TLS_CONTEXT = None

    def test_tls_verification_and_redirect_context(self):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Local fixture CA')])
        now = datetime.now(timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
                .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
                .sign(key, hashes.SHA256()))
        cert_path, key_path = FIXTURES / 'fixture-ca.pem', FIXTURES / 'fixture-key.pem'
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(str(cert_path), str(key_path))
        server.socket = server_context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'https://127.0.0.1:{server.server_port}/redirect'
        try:
            module.TLS_CONTEXT = ssl.create_default_context()
            with self.assertRaises((module.urllib.error.URLError, RuntimeError)):
                module.probe_source('untrusted-fixture', url)
            module.TLS_CONTEXT = ssl.create_default_context(cafile=str(cert_path))
            result = module.probe_source('trusted-fixture', url)
            self.assertIn(result['http_status'], (200, 206))
            self.assertTrue(result['resolved_url'].endswith('/ok'))
            self.assertTrue(module.TLS_CONTEXT.check_hostname)
            self.assertEqual(module.TLS_CONTEXT.verify_mode, ssl.CERT_REQUIRED)
        finally:
            module.TLS_CONTEXT = None
            server.shutdown()
            server.server_close()
            thread.join()

    def test_network_probe_failure_is_bounded_before_asset_downloads(self):
        error = module.urllib.error.URLError(ssl.SSLCertVerificationError('fixture missing issuer'))
        with patch.object(module, 'probe_source', side_effect=error) as probe:
            records = module.network_preflight()
        self.assertEqual(probe.call_count, 5)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record.get('error') for record in records))

    def test_model_mirror_probe_accepts_verified_fallback(self):
        def probe(name, url):
            if 'modelscope.cn' in url:
                raise module.urllib.error.URLError('fixture primary unavailable')
            return {'source': name, 'requested_url': url, 'http_status': 200}
        with patch.object(module, 'probe_source', side_effect=probe):
            records = module.network_preflight()
        self.assertFalse(any(record.get('error') for record in records))
        model = next(record for record in records if record['source'] == 'model_mirror')
        self.assertIn('hf-mirror.com', model['requested_url'])
        self.assertEqual(len(model['previous_route_errors']), 1)

    @unittest.skipUnless(module.shutil.which('curl'), 'curl not installed locally')
    def test_system_curl_fallback_preserves_resume_and_checksum(self):
        root = FIXTURES / 'curl-resume'
        root.mkdir()
        partial = root / 'asset.bin.partial'
        partial.write_bytes(PAYLOAD[:123])
        item = {'destination': 'asset.bin', 'size': len(PAYLOAD),
                'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [self.url + '/ok']}
        failure = module.urllib.error.URLError(ssl.SSLCertVerificationError('fixture custom-Python CA missing'))
        with patch.object(module.urllib.request, 'urlopen', side_effect=failure):
            result = module.acquire(root, item)
        self.assertEqual(result['origin']['transport'], 'curl')
        self.assertEqual((root / 'asset.bin').read_bytes(), PAYLOAD)
        self.assertEqual(RANGES[-1], ('/ok', 'bytes=123-'))

    @unittest.skipUnless(module.shutil.which('curl'), 'curl not installed locally')
    def test_curl_probe_reads_real_route_and_reports_transport(self):
        result = module.curl_probe('local-fixture', self.url + '/ok')
        self.assertEqual(result['transport'], 'curl')
        self.assertEqual(result['http_status'], 206)

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
