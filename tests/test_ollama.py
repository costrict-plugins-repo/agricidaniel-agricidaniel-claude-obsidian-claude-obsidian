#!/usr/bin/env python3
"""Hermetic tests for the shared Ollama local-model proof and transport.

The transport tests use loopback-only fake servers defined here. No real
network service or model is contacted. ``FakeOllamaServer`` and
``SlowDripServer`` are also imported by the tiling and retrieval suites.
"""

import http.server
import json
import socket
import sys
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claude_obsidian.ollama import (
    OllamaLocalProofError,
    OllamaModelUnavailable,
    model_is_available,
    prove_local_model,
    validate_model_response,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
FAKE_DIGEST = "a" * 64
GARBAGE_STATUS = b"GARBAGE\r\n\r\n"


class FakeOllamaServer:
    """Threaded 127.0.0.1 fake Ollama that records every request.

    ``mode`` selects one deterministic behavior: ``good``, ``badstatus`` (every
    reply is a malformed status line), ``badproof`` (only ``/api/status`` is
    malformed), ``dupkey`` (status repeats ``remote_host``), ``othercloud``
    (an unrelated installed cloud model), ``selfcloud`` (the requested model
    is remote), ``embremote``, ``emb500``, or ``dim``.
    """

    def __init__(self, mode="good", model="m1:latest"):
        self.mode = mode
        self.model = model
        self.requests = []
        self._lock = threading.Lock()
        self._embeds = 0

    def __enter__(self):
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return None

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with fake._lock:
                    fake.requests.append((self.command, self.path, body))
                reply = fake._reply(self.path)
                if reply is None:
                    self.wfile.write(GARBAGE_STATUS)
                    self.close_connection = True
                    return
                status, document = reply
                raw = document if isinstance(document, bytes) else json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _handle
            do_POST = _handle

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        return False

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def count(self, method, path):
        with self._lock:
            return sum(1 for item in self.requests if item[:2] == (method, path))

    def _reply(self, path):
        mode = self.mode
        if mode == "badstatus" or (mode == "badproof" and path == "/api/status"):
            return None
        if path == "/api/version":
            return 200, {"version": "0.0.0-fixture"}
        if path == "/api/status":
            if mode == "dupkey":
                return 200, (
                    b'{"cloud":{"disabled":true},'
                    b'"remote_host":"https://ollama.com","remote_host":null}'
                )
            return 200, {"cloud": {"disabled": True}}
        if path == "/api/tags":
            entry = {"name": self.model, "model": self.model, "digest": FAKE_DIGEST}
            if mode == "selfcloud":
                entry["remote_host"] = "https://ollama.com:443"
            models = [entry]
            if mode == "othercloud":
                models.append({
                    "name": "gpt-oss:120b-cloud",
                    "model": "gpt-oss:120b-cloud",
                    "remote_host": "https://ollama.com:443",
                    "remote_model": "gpt-oss:120b",
                    "digest": "b" * 64,
                })
            return 200, {"models": models}
        if path == "/api/show":
            return 200, {"model_info": {"general.architecture": "fixture"}}
        if path in ("/api/embeddings", "/api/embed"):
            with self._lock:
                self._embeds += 1
                count = self._embeds
            vector = [0.1, 0.2, 0.3]
            if mode == "dim" and count % 2 == 0:
                vector = [0.1, 0.2]
            if mode == "emb500":
                return 500, {"error": "fixture failure"}
            if mode == "embremote":
                return 200, {"embedding": vector, "embeddings": [vector],
                             "remote_host": "https://ollama.com:443"}
            if path == "/api/embed":
                return 200, {"model": self.model, "embeddings": [vector]}
            return 200, {"embedding": vector}
        return 404, {"error": "not found"}


class SlowDripServer:
    """127.0.0.1 server that drips one valid JSON reply a byte at a time.

    Every individual socket read completes well inside a short socket timeout,
    so only an overall request deadline can stop the request early.
    ``phase="body"`` sends headers at once; ``phase="headers"`` drips all.
    """

    BODY = b'{"cloud": {"disabled": true}, "padding": "................"}'

    def __init__(self, *, phase="body", interval=0.2):
        self.phase = phase
        self.interval = interval
        self._stop = threading.Event()

    def __enter__(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self.thread.join(5)
        self.listener.close()
        return False

    @property
    def url(self):
        return f"http://127.0.0.1:{self.listener.getsockname()[1]}"

    def _serve(self):
        while not self._stop.is_set():
            try:
                connection, _address = self.listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._drip, args=(connection,), daemon=True).start()

    def _drip(self, connection):
        head = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(self.BODY)
        )
        try:
            connection.recv(65536)
            if self.phase == "body":
                connection.sendall(head)
                slow = self.BODY
            else:
                slow = head + self.BODY
            for value in slow:
                if self.interval and self._stop.wait(self.interval):
                    break
                connection.sendall(bytes([value]))
        except OSError:
            pass
        finally:
            connection.close()


def proof_responses(*, status=None, models=None, show=None):
    documents = {
        "status": status if status is not None else {"cloud": {"disabled": True}},
        "tags": {"models": models if models is not None else [
            {"name": "fixture", "digest": DIGEST_A},
        ]},
    }
    shown = show if show is not None else {"model_info": {"general.architecture": "fixture"}}
    events = []

    def get_json(endpoint):
        events.append(("get", endpoint))
        return documents[endpoint]

    def post_json(endpoint, payload):
        events.append(("post", endpoint, payload))
        return shown

    return get_json, post_json, events


class LocalModelProofTests(unittest.TestCase):
    def test_status_must_prove_cloud_disabled_with_exact_boolean_true(self):
        for status in (
            {},
            {"cloud": {}},
            {"cloud": {"disabled": False}},
            {"cloud": {"disabled": 1}},
            {"cloud": {"disabled": "true"}},
        ):
            get_json, post_json, events = proof_responses(status=status)
            with self.assertRaises(OllamaLocalProofError):
                prove_local_model("fixture", get_json, post_json)
            self.assertEqual([("get", "status")], events)

    def test_untagged_prefers_exact_and_allows_only_latest_alias(self):
        models = [
            {"name": "fixture:latest", "digest": DIGEST_A},
            {"name": "fixture", "digest": DIGEST_A},
            {"name": "fixture:other", "digest": DIGEST_B},
        ]
        get_json, post_json, events = proof_responses(models=models)
        proof = prove_local_model("fixture", get_json, post_json)
        self.assertEqual("fixture", proof["model"])
        self.assertEqual(DIGEST_A, proof["digest"])
        self.assertEqual(("post", "show", {"model": "fixture"}), events[-1])
        self.assertTrue(model_is_available("fixture", models))
        self.assertFalse(model_is_available("fixture:missing", models))

    def test_untagged_aliases_with_conflicting_digests_are_rejected(self):
        get_json, post_json, _events = proof_responses(models=[
            {"name": "fixture", "digest": DIGEST_A},
            {"name": "fixture:latest", "digest": DIGEST_B},
        ])
        with self.assertRaisesRegex(OllamaLocalProofError, "digest"):
            prove_local_model("fixture", get_json, post_json)

    def test_digest_prefix_spelling_is_normalized_before_alias_comparison(self):
        bare = "a" * 64
        get_json, post_json, _events = proof_responses(
            models=[
                {"name": "fixture", "digest": bare},
                {"name": "fixture:latest", "digest": "sha256:" + bare},
            ],
            show={
                "model_info": {"general.architecture": "fixture"},
                "digest": bare,
            },
        )
        proof = prove_local_model("fixture", get_json, post_json)
        self.assertEqual("sha256:" + bare, proof["digest"])

    def test_explicit_tag_requires_exact_identity(self):
        get_json, post_json, _events = proof_responses(models=[
            {"name": "fixture:latest", "digest": DIGEST_A},
        ])
        proof = prove_local_model("fixture:latest", get_json, post_json)
        self.assertEqual("fixture:latest", proof["model"])
        with self.assertRaises(OllamaModelUnavailable):
            prove_local_model("fixture:other", get_json, post_json)
        with self.assertRaises(OllamaLocalProofError):
            prove_local_model("fixture", get_json, post_json, require_explicit_tag=True)

    def test_cloud_tags_remote_metadata_and_bad_digests_are_rejected(self):
        cases = (
            [{"name": "fixture:cloud", "digest": DIGEST_A}],
            [{"name": "fixture", "digest": "bad"}],
            [{"name": "fixture", "digest": DIGEST_A, "remote_host": "https://example.test"}],
            [{"name": "fixture", "digest": DIGEST_A, "remote_model": "remote:tag"}],
            [{"name": "fixture", "model": "other", "digest": DIGEST_A}],
        )
        requested = ("fixture:cloud", "fixture", "fixture", "fixture", "fixture")
        for model, models in zip(requested, cases):
            get_json, post_json, _events = proof_responses(models=models)
            with self.assertRaises(OllamaLocalProofError):
                prove_local_model(model, get_json, post_json)

    def test_show_requires_local_metadata_and_consistent_identity(self):
        cases = (
            {},
            {"model_info": {}},
            {"model_info": {"x": 1}, "remote_host": "https://example.test"},
            {"model_info": {"x": 1}, "remote_model": "remote:tag"},
            {"model_info": {"x": 1}, "model": "other"},
            {"model_info": {"x": 1}, "digest": DIGEST_B},
        )
        for show in cases:
            get_json, post_json, _events = proof_responses(show=show)
            with self.assertRaises(OllamaLocalProofError):
                prove_local_model("fixture", get_json, post_json)

    def test_embedding_response_rejects_remote_or_conflicting_identity(self):
        proof = {"model": "fixture", "digest": DIGEST_A}
        validate_model_response({}, proof)
        validate_model_response({"model": "fixture", "digest": DIGEST_A}, proof)
        for response in (
            {"remote_host": "https://example.test"},
            {"remote_model": "remote:tag"},
            {"model": "other"},
            {"name": "other"},
            {"digest": DIGEST_B},
        ):
            with self.assertRaises(OllamaLocalProofError):
                validate_model_response(response, proof)

    def test_unrelated_remote_model_does_not_block_a_local_model(self):
        models = [
            {"name": "fixture", "digest": DIGEST_A},
            {
                "name": "gpt-oss:120b-cloud",
                "model": "gpt-oss:120b-cloud",
                "remote_host": "https://ollama.com:443",
                "remote_model": "gpt-oss:120b",
                "digest": DIGEST_B,
            },
            {"name": "other", "digest": DIGEST_B, "remote_host": "https://x.test"},
        ]
        get_json, post_json, _events = proof_responses(models=models)
        proof = prove_local_model("fixture", get_json, post_json)
        self.assertEqual({"model": "fixture", "digest": DIGEST_A}, proof)
        self.assertTrue(model_is_available("fixture", models))

    def test_requested_remote_model_and_remote_list_still_fail_closed(self):
        get_json, post_json, events = proof_responses(models=[
            {"name": "fixture", "digest": DIGEST_A},
            {"name": "fixture:latest", "digest": DIGEST_A,
             "remote_model": "fixture:remote"},
        ])
        with self.assertRaisesRegex(OllamaLocalProofError, "remote identity"):
            prove_local_model("fixture", get_json, post_json)
        self.assertNotIn("show", [event[1] for event in events])

        def remote_list(endpoint):
            if endpoint == "status":
                return {"cloud": {"disabled": True}}
            return {"models": [{"name": "fixture", "digest": DIGEST_A}],
                    "remote_host": "https://ollama.com"}

        with self.assertRaisesRegex(OllamaLocalProofError, "remote identity"):
            prove_local_model("fixture", remote_list, post_json)


class DirectTransportTests(unittest.TestCase):
    """The shared no-proxy, no-redirect JSON transport is bounded per request."""

    def _get(self, url, timeout, *, limit=1024 * 1024):
        from claude_obsidian import ollama as shared

        request = urllib.request.Request(url + "/api/status", method="GET")
        deadline = time.monotonic() + timeout
        with shared.open_direct(request, timeout) as response:
            return shared.read_json_object(
                response, request, limit=limit, deadline=deadline, label="Ollama"
            )

    def test_slow_drip_body_is_stopped_by_the_request_deadline(self):
        with SlowDripServer(phase="body", interval=0.2) as server:
            started = time.monotonic()
            with self.assertRaises(OSError):
                self._get(server.url, 1.0)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 3.0)

    def test_slow_drip_headers_are_stopped_by_the_request_deadline(self):
        with SlowDripServer(phase="headers", interval=0.2) as server:
            started = time.monotonic()
            with self.assertRaises(OSError):
                self._get(server.url, 1.0)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 3.0)

    def test_tls_handshake_gets_only_the_time_left_after_a_slow_connect(self):
        from claude_obsidian import ollama as shared

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        accepted = []
        acceptor = threading.Thread(
            target=lambda: accepted.append(listener.accept()[0]), daemon=True
        )
        acceptor.start()
        real_create = socket.create_connection

        def slow_create(*args, **kwargs):
            # The TCP connect itself consumes most of the request budget, then
            # the server never answers the TLS ClientHello.
            time.sleep(0.8)
            return real_create(*args, **kwargs)

        port = listener.getsockname()[1]
        request = urllib.request.Request(f"https://127.0.0.1:{port}/api/status")
        try:
            with unittest.mock.patch.object(socket, "create_connection", slow_create):
                started = time.monotonic()
                with self.assertRaises(OSError):
                    shared.open_direct(request, 1.2)
                elapsed = time.monotonic() - started
        finally:
            acceptor.join(2)
            for connection in accepted:
                connection.close()
            listener.close()
        self.assertLess(elapsed, 1.7)

    def test_prompt_reply_within_deadline_is_decoded(self):
        with SlowDripServer(phase="body", interval=0.0) as server:
            document = self._get(server.url, 3.0)
        self.assertEqual({"disabled": True}, document["cloud"])

    def test_duplicate_keys_nesting_and_size_are_rejected(self):
        from claude_obsidian import ollama as shared

        for raw in (
            b'{"remote_host":"https://ollama.com","remote_host":null}',
            b'{"cloud":{"disabled":true,"disabled":true}}',
            b"[" * 100000 + b"]" * 100000,
            b'"not an object"',
            b'{"x":NaN}',
            b"\xff",
        ):
            with self.assertRaises(ValueError):
                shared.parse_json_object(raw, "Ollama")
        with FakeOllamaServer() as server:
            with self.assertRaisesRegex(ValueError, "byte limit"):
                self._get(server.url, 3.0, limit=8)
            self.assertEqual({"cloud": {"disabled": True}}, self._get(server.url, 3.0))

    def test_redirects_are_refused_and_proxies_are_bypassed(self):
        from claude_obsidian import ollama as shared

        class Redirect(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return None

            def do_GET(self):
                self.send_response(307)
                self.send_header("Location", "http://127.0.0.2:9/api/status")
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self._get(url, 3.0)
            self.assertEqual(307, caught.exception.code)
        finally:
            server.shutdown()
            server.server_close()
        with FakeOllamaServer() as fake:
            proxies = {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                       "no_proxy": "", "NO_PROXY": ""}
            with unittest.mock.patch.dict("os.environ", proxies):
                self.assertEqual({"cloud": {"disabled": True}}, self._get(fake.url, 3.0))
            self.assertEqual(1, fake.count("GET", "/api/status"))
        handler_types = [type(handler) for handler in shared.direct_handlers(1.0)]
        self.assertIn(shared.NoRedirectHandler, handler_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
