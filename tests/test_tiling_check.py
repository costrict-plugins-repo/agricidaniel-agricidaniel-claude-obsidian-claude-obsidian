#!/usr/bin/env python3
"""test_tiling_check.py — unit tests for scripts/tiling-check.py.

Does NOT require ollama; tests cover parsing, cosine, inclusion logic,
hash properties, cache schema, and the localhost-URL guard. Tests that
need ollama are marked and skipped cleanly when the helper reports
exit 10/11.

Usage:
  python3 tests/test_tiling_check.py
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "scripts" / "tiling-check.py"
os.environ["CLAUDE_OBSIDIAN_VAULT"] = str(ROOT)

spec = importlib.util.spec_from_file_location("tc", HELPER)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)
DIGEST = "sha256:" + "a" * 64

if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from test_ollama import FakeOllamaServer, SlowDripServer  # noqa: E402


def local_proof(model="m1"):
    return {"model": model, "digest": DIGEST}


class Fail(SystemExit):
    pass


def assert_eq(label, expected, actual):
    if expected != actual:
        raise Fail(f"FAIL {label}: expected {expected!r}, got {actual!r}")
    print(f"OK   {label}")


def assert_true(label, cond):
    if not cond:
        raise Fail(f"FAIL {label}")
    print(f"OK   {label}")


def test_cosine():
    assert_eq("cosine identical", 1.0, tc.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert_true(
        "cosine huge identical",
        abs(1.0 - tc.cosine([1e308, 1e308], [1e308, 1e308])) < 1e-12,
    )
    assert_true(
        "cosine tiny identical",
        abs(1.0 - tc.cosine([1e-308], [1e-308])) < 1e-12,
    )
    assert_eq("cosine orthogonal", 0.0, tc.cosine([1.0, 0.0], [0.0, 1.0]))
    assert_eq("cosine anti-parallel", -1.0, tc.cosine([1.0, 0.0], [-1.0, 0.0]))
    assert_eq("cosine zero vector", 0.0, tc.cosine([0.0, 0.0], [1.0, 2.0]))
    try:
        tc.cosine([1.0], [1.0, 2.0])
        raise Fail("FAIL dim mismatch should raise")
    except ValueError:
        print("OK   cosine dim mismatch raises ValueError")


def test_frontmatter():
    fm, body = tc.parse_frontmatter("---\ntype: concept\ntitle: Foo\n---\n# Body\n")
    assert_eq("parse type", "concept", fm.get("type"))
    assert_eq("parse body", "# Body\n", body)
    fm, body = tc.parse_frontmatter("# Just a title\n")
    assert_eq("no frontmatter -> empty", {}, fm)
    fm, _ = tc.parse_frontmatter('---\ntype: "meta"\n---\nbody\n')
    assert_eq("quoted type stripped", "meta", fm.get("type"))


def test_body_hash_model_scoped():
    h1 = tc.body_hash("body", "model-A")
    h2 = tc.body_hash("body", "model-B")
    h3 = tc.body_hash("body", "model-A")
    assert_true("different models hash differently", h1 != h2)
    assert_eq("same body+model hashes identically", h1, h3)


def test_included_basic():
    cases = [
        (ROOT / "wiki/concepts/Foo.md", {"type": "concept"}, True, "included"),
        (ROOT / "wiki/index.md", {"type": "meta"}, False, "excluded filename"),
        (ROOT / "wiki/folds/fold-1.md", {"type": "fold"}, False, "under wiki/folds/"),
        (ROOT / "wiki/meta/session.md", {"type": "session"}, False, "under wiki/meta/"),
        (ROOT / "wiki/entities/Person.md", {"type": "entity"}, True, "included"),
    ]
    for path, fm, expected_ok, expected_reason in cases:
        ok, reason = tc.included(path, fm)
        label = f"included({path.relative_to(ROOT)}, {fm.get('type')})"
        assert_eq(label + ".ok", expected_ok, ok)
        assert_eq(label + ".reason", expected_reason, reason)


def test_is_local_url():
    assert_true("127.0.0.1 is local", tc._is_local_url("http://127.0.0.1:11434"))
    assert_true("localhost is local", tc._is_local_url("http://localhost:11434"))
    assert_true("::1 is local", tc._is_local_url("http://[::1]:11434"))
    assert_true("example.com NOT local", not tc._is_local_url("http://example.com"))
    assert_true("1.2.3.4 NOT local", not tc._is_local_url("http://1.2.3.4"))


def test_transport_disables_proxies_and_rejects_redirects():
    request = tc.urllib.request.Request(
        "http://127.0.0.1:11434/api/version", method="GET"
    )
    opener = mock.Mock()
    sentinel = object()
    opener.open.return_value = sentinel
    with mock.patch.object(
        tc.urllib.request, "build_opener", return_value=opener
    ) as builder:
        result = tc._open_ollama_request(request, 3)
    assert_true("tiling transport returns opener response", result is sentinel)
    handlers = builder.call_args.args
    proxy_handlers = [
        handler for handler in handlers
        if isinstance(handler, tc.urllib.request.ProxyHandler)
    ]
    assert_eq("tiling transport installs one proxy handler", 1, len(proxy_handlers))
    assert_eq("tiling transport disables ambient proxies", {}, proxy_handlers[0].proxies)
    assert_true(
        "tiling transport refuses redirects",
        any(isinstance(handler, tc._NoOllamaRedirect) for handler in handlers),
    )

    read_limits = []

    class ExactResponse:
        headers = {"Content-Type": "application/json; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def getcode():
            return 200

        @staticmethod
        def geturl():
            return "http://127.0.0.1:11434/api/version"

        @staticmethod
        def read(limit):
            read_limits.append(limit)
            return b'{"version":"fixture"}'

    with mock.patch.object(
        tc, "_open_ollama_request", return_value=ExactResponse()
    ):
        payload = tc._http_get_json(
            "http://127.0.0.1:11434/api/version", 1
        )
    assert_eq("tiling exact-target GET decodes JSON", {"version": "fixture"}, payload)
    assert_eq(
        "tiling response read is bounded",
        [tc.MAX_RESPONSE_BYTES + 1],
        read_limits,
    )

    class RedirectedResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def getcode():
            return 200

        @staticmethod
        def geturl():
            return "https://collector.example/api/embeddings"

        @staticmethod
        def read(_limit):
            return b'{"embedding":[1.0,2.0]}'

    with mock.patch.object(
        tc, "_open_ollama_request", return_value=RedirectedResponse()
    ):
        try:
            tc._http_post_json(
                "http://127.0.0.1:11434/api/embeddings",
                {"model": "fixture", "prompt": "private"},
                1,
            )
            raise Fail("FAIL redirected tiling response should be rejected")
        except ValueError as exc:
            assert_true("tiling redirect rejection is explicit", "target changed" in str(exc))


def test_embedding_vectors_are_bounded_finite_and_nonboolean():
    invalid_vectors = (
        [],
        [True],
        [float("nan")],
        [float("inf")],
        [10 ** 400],
        [0.0],
        [0.0] * (tc.MAX_EMBEDDING_DIMENSIONS + 1),
    )
    for vector in invalid_vectors:
        with mock.patch.object(tc, "_http_post_json", return_value={"embedding": vector}):
            try:
                tc.embed("body", "model", "http://127.0.0.1:11434")
                raise Fail("FAIL invalid live embedding should be rejected")
            except RuntimeError:
                pass
        assert_true("invalid live embedding rejected", True)
    with mock.patch.object(tc, "_http_post_json", return_value={"embedding": [0.0, 1.0]}):
        assert_eq(
            "finite numeric embedding accepted",
            [0.0, 1.0],
            tc.embed("body", "model", "http://127.0.0.1:11434"),
        )
    for response in (
        {"embedding": [0.0, 1.0], "remote_host": "https://example.test"},
        {"embedding": [0.0, 1.0], "model": "other"},
    ):
        with mock.patch.object(tc, "_http_post_json", return_value=response):
            try:
                tc.embed(
                    "body",
                    "m1",
                    "http://127.0.0.1:11434",
                    proof=local_proof(),
                )
                raise Fail("FAIL remote or mismatched embedding response accepted")
            except tc.OllamaLocalProofError:
                print("OK   remote or mismatched embedding response rejected")


def test_cache_schema():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        original_cache = tc.CACHE_PATH
        original_meta = tc.META_DIR
        tc.CACHE_PATH = tmp / "cache.json"
        tc.META_DIR = tmp
        try:
            c = tc.load_cache("m1", DIGEST)
            assert_eq("empty cache -> version 2", 2, c["version"])
            assert_eq("empty cache -> empty embeddings", {}, c["embeddings"])

            tc.CACHE_PATH.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": "m1",
                        "embeddings": {"a.md": {"hash": "h", "embedding": [1.0]}},
                    }
                )
            )
            c = tc.load_cache("m1", DIGEST)
            assert_eq("valid legacy cache is a miss", 0, len(c["embeddings"]))

            tc.CACHE_PATH.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "model": "m1",
                        "digest": DIGEST,
                        "embeddings": {"a.md": {"hash": "h", "embedding": [1.0]}},
                    }
                )
            )
            c = tc.load_cache("m1", DIGEST)
            assert_eq("valid version 2 cache loads", 1, len(c["embeddings"]))
            tc.CACHE_PATH.write_text(
                json.dumps({
                    "version": 2,
                    "model": "m1",
                    "digest": "a" * 64,
                    "embeddings": {"a.md": {"hash": "h", "embedding": [1.0]}},
                })
            )
            c = tc.load_cache("m1", DIGEST)
            assert_eq("bare cache digest reuses equivalent proof", 1, len(c["embeddings"]))
            assert_eq("cache digest normalizes in memory", DIGEST, c["digest"])
            c = tc.load_cache("m1", "sha256:" + "b" * 64)
            assert_eq("model digest drift invalidates cache", {}, c["embeddings"])

            c = tc.load_cache("m2", DIGEST)
            assert_eq("model drift -> empty", {}, c["embeddings"])
            assert_eq("model drift -> new model", "m2", c["model"])

            tc.CACHE_PATH.write_text("not-json{{")
            try:
                tc.load_cache("m1", DIGEST)
                raise Fail("FAIL corrupt cache should SystemExit")
            except SystemExit as e:
                assert_eq("corrupt cache exit", 3, e.code)

            tc.CACHE_PATH.write_bytes(b"\xff")
            try:
                tc.load_cache("m1", DIGEST)
                raise Fail("FAIL invalid UTF-8 cache should SystemExit")
            except SystemExit as e:
                assert_eq("invalid UTF-8 cache exit", 3, e.code)

            for value in (None, [], "text", 7, True):
                tc.CACHE_PATH.write_text(json.dumps(value), encoding="utf-8")
                try:
                    tc.load_cache("m1", DIGEST)
                    raise Fail("FAIL non-object cache should SystemExit")
                except SystemExit as e:
                    assert_eq("non-object cache exit", 3, e.code)

            invalid_entries = (
                {"a.md": 1},
                {"a.md": {}},
                {"a.md": {"hash": "h"}},
                {"a.md": {"hash": "h", "embedding": "1,2"}},
                {"a.md": {"hash": "h", "embedding": [True]}},
                {"a.md": {"hash": "h", "embedding": [float("nan")]}},
                {"a.md": {"hash": "h", "embedding": [float("inf")]}},
                {"a.md": {"hash": "h", "embedding": [10 ** 400]}},
                {"a.md": {"hash": "h", "embedding": [0.0]}},
                {"a.md": {"hash": "h", "embedding": [0.0] * (tc.MAX_EMBEDDING_DIMENSIONS + 1)}},
                {
                    "a.md": {"hash": "h1", "embedding": [1.0, 0.0]},
                    "b.md": {"hash": "h2", "embedding": [1.0, 0.0, 0.0]},
                },
            )
            for embeddings in invalid_entries:
                tc.CACHE_PATH.write_text(
                    json.dumps({"version": 1, "model": "m1", "embeddings": embeddings}),
                    encoding="utf-8",
                )
                try:
                    tc.load_cache("m1", DIGEST)
                    raise Fail("FAIL invalid cache entry should SystemExit")
                except SystemExit as e:
                    assert_eq("invalid cache entry exit", 3, e.code)

            tc.CACHE_PATH.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": "m1",
                        "embeddings": {"a.md": {"hash": "h", "embedding": "bad"}},
                    }
                ),
                encoding="utf-8",
            )
            writes = []
            with (
                mock.patch.object(
                    tc, "load_thresholds",
                    return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
                ),
                mock.patch.object(tc, "detect_ollama", return_value=True),
                mock.patch.object(tc, "detect_model", return_value=True),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
                mock.patch.object(tc, "save_cache", side_effect=lambda _cache: writes.append("save")),
            ):
                try:
                    tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
                    raise Fail("FAIL corrupt cache run should SystemExit")
                except SystemExit as e:
                    assert_eq("corrupt cache run exit", tc.EXIT_CACHE_CORRUPT, e.code)
            assert_eq("corrupt cache is not published", [], writes)

            wiki = tmp / "wiki"
            wiki.mkdir()
            (wiki / "A.md").write_text("cached body", encoding="utf-8")
            (wiki / "B.md").write_text("new body", encoding="utf-8")
            tc.CACHE_PATH.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "model": "m1",
                        "digest": DIGEST,
                        "embeddings": {
                            "wiki/A.md": {
                                "hash": tc.body_hash("cached body", "m1", DIGEST),
                                "embedding": [1.0, 0.0],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            prior_cache = tc.CACHE_PATH.read_bytes()
            writes = []
            output = io.StringIO()
            with (
                mock.patch.object(tc, "VAULT_ROOT", tmp),
                mock.patch.object(tc, "WIKI_DIR", wiki),
                mock.patch.object(
                    tc, "load_thresholds",
                    return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
                ),
                mock.patch.object(tc, "detect_ollama", return_value=True),
                mock.patch.object(tc, "detect_model", return_value=True),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
                mock.patch.object(tc, "embed", return_value=[1.0, 0.0, 0.0]),
                mock.patch.object(tc, "save_cache", side_effect=lambda _cache: writes.append("save")),
                contextlib.redirect_stdout(output),
            ):
                result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
            assert_eq("new embedding dimension mismatch exit", tc.EXIT_INCOMPLETE, result)
            assert_eq("dimension mismatch is not published", [], writes)
            assert_eq("dimension mismatch preserves prior cache", prior_cache, tc.CACHE_PATH.read_bytes())
            assert_true("dimension mismatch report is incomplete", "status: incomplete" in output.getvalue())
            assert_true("dimension mismatch report is provisional", "provisional" in output.getvalue())

            tc.CACHE_PATH.write_text(json.dumps({"version": 999, "embeddings": {}}))
            try:
                tc.load_cache("m1", DIGEST)
                raise Fail("FAIL wrong version should SystemExit")
            except SystemExit as e:
                assert_eq("wrong version exit", 3, e.code)

            tc.CACHE_PATH.write_text(
                json.dumps({"version": True, "model": "m1", "embeddings": {}}),
                encoding="utf-8",
            )
            try:
                tc.load_cache("m1", DIGEST)
                raise Fail("FAIL boolean cache version should SystemExit")
            except SystemExit as e:
                assert_eq("boolean cache version exit", 3, e.code)

            tc.CACHE_PATH.write_text("null", encoding="utf-8")
            with (
                mock.patch.object(tc, "THRESHOLDS_PATH", tmp / "missing-thresholds.json"),
                mock.patch.object(tc, "detect_ollama", return_value=True),
                mock.patch.object(tc, "detect_model", return_value=True),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
            ):
                assert_eq(
                    "peek non-object cache exit",
                    tc.EXIT_CACHE_CORRUPT,
                    tc.cmd_peek("http://127.0.0.1:11434", "m1"),
                )
        finally:
            tc.CACHE_PATH = original_cache
            tc.META_DIR = original_meta


def test_threshold_validation_precedes_provider_and_cache():
    invalid_values = (
        {},
        {"version": "1", "bands": {"review": 0.8, "error": 0.9}},
        {"version": 1, "bands": {"review": "0.8", "error": 0.9}},
        {"version": 1, "bands": {"review": float("nan"), "error": 0.9}},
        {"version": 1, "bands": {"review": 10 ** 400, "error": 0.9}},
        {"version": 1, "bands": {"review": 0.9, "error": 0.8}},
        {"version": 1, "bands": {"review": -0.1, "error": 0.9}},
        {"version": 1, "bands": {"review": 0.8, "error": 1.1}},
    )
    with tempfile.TemporaryDirectory() as directory:
        wiki = Path(directory) / "wiki"
        wiki.mkdir()
        for invalid in invalid_values:
            events = []
            with (
                mock.patch.object(tc, "WIKI_DIR", wiki),
                mock.patch.object(tc, "load_thresholds", return_value=invalid),
                mock.patch.object(
                    tc, "detect_ollama",
                    side_effect=lambda _url: events.append("detect_ollama") or True,
                ),
                mock.patch.object(
                    tc, "detect_model",
                    side_effect=lambda _url, _model: events.append("detect_model") or True,
                ),
                mock.patch.object(
                    tc, "load_cache", side_effect=lambda _model: events.append("load_cache") or {}
                ),
                mock.patch.object(
                    tc, "save_cache", side_effect=lambda _cache: events.append("save_cache")
                ),
            ):
                result = tc.run_check(
                    False, None, "http://127.0.0.1:11434", "fixture"
                )
            assert_eq("invalid thresholds use controlled exit", tc.EXIT_USAGE, result)
            assert_eq("invalid thresholds have no provider or cache events", [], events)

    with tempfile.TemporaryDirectory() as directory:
        vault = Path(directory)
        (vault / "wiki").mkdir()
        meta = vault / ".vault-meta"
        meta.mkdir()
        (meta / "tiling-thresholds.json").write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(HELPER), "--vault", str(vault)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert_eq("invalid thresholds CLI exit", tc.EXIT_USAGE, result.returncode)
        assert_true("invalid thresholds CLI error is controlled", "Traceback" not in result.stderr)

    with tempfile.TemporaryDirectory() as directory:
        threshold_path = Path(directory) / "tiling-thresholds.json"
        threshold_path.write_bytes(b"\xff")
        events = []
        with (
            mock.patch.object(tc, "THRESHOLDS_PATH", threshold_path),
            mock.patch.object(
                tc, "detect_ollama",
                side_effect=lambda _url: events.append("detect_ollama") or True,
            ),
            mock.patch.object(
                tc, "load_cache", side_effect=lambda _model: events.append("load_cache") or {}
            ),
        ):
            result = tc.run_check(
                False, None, "http://127.0.0.1:11434", "fixture"
            )
        assert_eq("invalid UTF-8 thresholds use controlled exit", tc.EXIT_USAGE, result)
        assert_eq("invalid UTF-8 thresholds precede provider and cache", [], events)


def test_url_guard_via_subprocess():
    with tempfile.TemporaryDirectory() as directory:
        vault = Path(directory)
        (vault / "wiki").mkdir()
        env = os.environ.copy()
        env["OLLAMA_URL"] = "http://example.com:11434"
        result = subprocess.run(
            [sys.executable, str(HELPER), "--vault", str(vault), "--peek"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("remote URL without flag exit", 2, result.returncode)
        assert_true("remote URL error message", "not localhost" in result.stderr)

        for invalid in ("file:///etc/passwd", "http://user:secret@localhost:11434"):
            env["OLLAMA_URL"] = invalid
            result = subprocess.run(
                [
                    sys.executable,
                    str(HELPER),
                    "--vault",
                    str(vault),
                    "--peek",
                    "--allow-remote-ollama",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert_eq("invalid URL rejected with remote opt-in", 2, result.returncode)
            assert_true(
                "invalid URL error message", "credential-free http(s)" in result.stderr
            )


def test_report_path_cannot_overwrite_vault_content():
    original = (
        tc.VAULT_ROOT,
        tc.WIKI_DIR,
        tc.META_DIR,
        tc.CACHE_PATH,
        tc.CACHE_LOCK,
        tc.THRESHOLDS_PATH,
    )
    try:
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            (vault / "wiki").mkdir()
            raw = vault / ".raw"
            raw.mkdir()
            precious = raw / "precious.pdf"
            precious.write_bytes(b"immutable source bytes")
            assert_true("temporary vault config", tc.configure_vault(str(vault)))
            result = tc.run_check(
                rebuild=False,
                report_path=precious,
                ollama_url="http://127.0.0.1:11434",
                model="unused",
            )
            assert_eq("--report rejected", tc.EXIT_USAGE, result)
            assert_eq(
                "raw source preserved", b"immutable source bytes", precious.read_bytes()
            )
    finally:
        (
            tc.VAULT_ROOT,
            tc.WIKI_DIR,
            tc.META_DIR,
            tc.CACHE_PATH,
            tc.CACHE_LOCK,
            tc.THRESHOLDS_PATH,
        ) = original


def test_peek_localhost_transport_uses_numeric_loopback():
    original = (
        tc.VAULT_ROOT,
        tc.WIKI_DIR,
        tc.META_DIR,
        tc.CACHE_PATH,
        tc.CACHE_LOCK,
        tc.THRESHOLDS_PATH,
    )
    requested = []

    def transport(url, timeout):
        requested.append(url)
        if url.endswith("/api/tags"):
            return {"models": [{"name": tc.DEFAULT_MODEL, "digest": DIGEST}]}
        return {"version": "fixture"}

    try:
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            (vault / "wiki").mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {"OLLAMA_URL": "http://localhost:12121/private-base"},
                ),
                mock.patch.object(tc, "_http_get_json", side_effect=transport),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof(tc.DEFAULT_MODEL)),
            ):
                result = tc.main(["--vault", str(vault), "--peek"])
            assert_eq("numeric-loopback tiling peek succeeds", tc.EXIT_OK, result)
            assert_eq(
                "tiling peek transport receives only numeric loopback",
                [
                    "http://127.0.0.1:12121/private-base/api/version",
                    "http://127.0.0.1:12121/private-base/api/tags",
                ],
                requested,
            )
    finally:
        (
            tc.VAULT_ROOT,
            tc.WIKI_DIR,
            tc.META_DIR,
            tc.CACHE_PATH,
            tc.CACHE_LOCK,
            tc.THRESHOLDS_PATH,
        ) = original


def test_json_conversion_failures_use_controlled_cache_and_threshold_errors():
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache.json"
        thresholds = Path(directory) / "thresholds.json"
        cache.write_text("{}", encoding="utf-8")
        thresholds.write_text("{}", encoding="utf-8")
        for failure in (ValueError("integer conversion limit"), RecursionError("nesting limit")):
            with (
                mock.patch.object(tc, "CACHE_PATH", cache),
                mock.patch.object(tc, "THRESHOLDS_PATH", thresholds),
                mock.patch.object(tc.json, "load", side_effect=failure),
            ):
                try:
                    tc.load_cache(tc.DEFAULT_MODEL, DIGEST)
                except SystemExit as exc:
                    assert_eq("JSON conversion failure has controlled cache exit", tc.EXIT_CACHE_CORRUPT, exc.code)
                else:
                    raise Fail("FAIL cache conversion failure was accepted")
                try:
                    tc.load_thresholds()
                except tc.ThresholdValidationError:
                    print("OK   JSON conversion failure has controlled threshold error")
                else:
                    raise Fail("FAIL threshold conversion failure was accepted")
                assert_eq("failed JSON reads preserve cache", "{}", cache.read_text())
                assert_eq("failed JSON reads preserve thresholds", "{}", thresholds.read_text())


def test_unproved_local_state_precedes_cache_and_page_reads():
    events = []
    with tempfile.TemporaryDirectory() as directory:
        wiki = Path(directory) / "wiki"
        wiki.mkdir()
        (wiki / "private.md").write_text("private body", encoding="utf-8")
        with (
            mock.patch.object(tc, "WIKI_DIR", wiki),
            mock.patch.object(
                tc,
                "load_thresholds",
                return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
            ),
            mock.patch.object(tc, "detect_ollama", return_value=True),
            mock.patch.object(tc, "detect_model", return_value=True),
            mock.patch.object(
                tc,
                "local_model_proof",
                side_effect=tc.OllamaLocalProofError("cloud enabled"),
            ),
            mock.patch.object(
                tc, "load_cache", side_effect=lambda *_args: events.append("cache")
            ),
            mock.patch.object(
                tc, "embed", side_effect=lambda *_args: events.append("embed")
            ),
        ):
            result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
    assert_eq("unproved local state uses incomplete exit", tc.EXIT_INCOMPLETE, result)
    assert_eq("unproved local state has no cache or body payload work", [], events)


def test_partial_and_total_embedding_failures_preserve_prior_cache():
    for fail_after in (0, 1):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            wiki = vault / "wiki"
            wiki.mkdir()
            (wiki / "A.md").write_text("body A", encoding="utf-8")
            (wiki / "B.md").write_text("body B", encoding="utf-8")
            meta = vault / ".vault-meta"
            meta.mkdir()
            cache_path = meta / "tiling-cache.json"
            cache_path.write_text('{"legacy":"precious"}\n', encoding="utf-8")
            before = cache_path.read_bytes()
            calls = []

            def fake_embed(text, model, url, **_kwargs):
                calls.append(text)
                if len(calls) > fail_after:
                    raise RuntimeError("fixture failure")
                return [1.0, 0.5]

            output = io.StringIO()
            with (
                mock.patch.object(tc, "VAULT_ROOT", vault),
                mock.patch.object(tc, "WIKI_DIR", wiki),
                mock.patch.object(tc, "CACHE_PATH", cache_path),
                mock.patch.object(
                    tc,
                    "load_thresholds",
                    return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
                ),
                mock.patch.object(tc, "detect_ollama", return_value=True),
                mock.patch.object(tc, "detect_model", return_value=True),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
                mock.patch.object(
                    tc,
                    "load_cache",
                    return_value={
                        "version": 2,
                        "model": "m1",
                        "digest": DIGEST,
                        "embeddings": {},
                    },
                ),
                mock.patch.object(tc, "embed", side_effect=fake_embed),
                mock.patch.object(tc, "save_cache") as save,
                contextlib.redirect_stdout(output),
            ):
                result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
            assert_eq("incomplete embedding uses exit 12", tc.EXIT_INCOMPLETE, result)
            assert_eq("incomplete embedding does not publish cache", 0, save.call_count)
            assert_eq("incomplete embedding preserves prior cache", before, cache_path.read_bytes())
            assert_true("incomplete report is explicit", "status: incomplete" in output.getvalue())
            assert_true("incomplete report is provisional", "provisional" in output.getvalue())


def test_fresh_embedding_dimension_mismatch_is_incomplete():
    with tempfile.TemporaryDirectory() as directory:
        vault = Path(directory)
        wiki = vault / "wiki"
        wiki.mkdir()
        (wiki / "A.md").write_text("body A", encoding="utf-8")
        (wiki / "B.md").write_text("body B", encoding="utf-8")
        cache_path = vault / ".vault-meta" / "tiling-cache.json"
        cache_path.parent.mkdir()
        cache_path.write_bytes(b'{"prior":"precious"}\n')
        before = cache_path.read_bytes()
        vectors = iter(([1.0, 0.5], [1.0, 0.5, 0.25]))
        output = io.StringIO()
        with (
            mock.patch.object(tc, "VAULT_ROOT", vault),
            mock.patch.object(tc, "WIKI_DIR", wiki),
            mock.patch.object(tc, "CACHE_PATH", cache_path),
            mock.patch.object(
                tc,
                "load_thresholds",
                return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
            ),
            mock.patch.object(tc, "detect_ollama", return_value=True),
            mock.patch.object(tc, "detect_model", return_value=True),
            mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
            mock.patch.object(
                tc,
                "load_cache",
                return_value={
                    "version": 2,
                    "model": "m1",
                    "digest": DIGEST,
                    "embeddings": {},
                },
            ),
            mock.patch.object(tc, "embed", side_effect=lambda *_args, **_kwargs: next(vectors)),
            mock.patch.object(tc, "save_cache") as save,
            contextlib.redirect_stdout(output),
        ):
            result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
        assert_eq("fresh dimension mismatch uses exit 12", tc.EXIT_INCOMPLETE, result)
        assert_eq("fresh dimension mismatch does not publish cache", 0, save.call_count)
        assert_eq("fresh dimension mismatch preserves prior cache", before, cache_path.read_bytes())
        assert_true("fresh dimension mismatch report is incomplete", "status: incomplete" in output.getvalue())
        assert_true("fresh dimension mismatch report is provisional", "provisional" in output.getvalue())


def test_empty_and_complete_runs_publish_once():
    for page_names, expected_status in (
        (["index.md"], "complete-empty"),
        (["A.md", "B.md"], "complete"),
    ):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            wiki = vault / "wiki"
            wiki.mkdir()
            for name in page_names:
                (wiki / name).write_text("body " + name, encoding="utf-8")
            output = io.StringIO()
            with (
                mock.patch.object(tc, "VAULT_ROOT", vault),
                mock.patch.object(tc, "WIKI_DIR", wiki),
                mock.patch.object(
                    tc,
                    "load_thresholds",
                    return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
                ),
                mock.patch.object(tc, "detect_ollama", return_value=True),
                mock.patch.object(tc, "detect_model", return_value=True),
                mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
                mock.patch.object(
                    tc,
                    "load_cache",
                    return_value={
                        "version": 2,
                        "model": "m1",
                        "digest": DIGEST,
                        "embeddings": {},
                    },
                ),
                mock.patch.object(tc, "embed", return_value=[1.0, 0.5]),
                mock.patch.object(tc, "save_cache") as save,
                contextlib.redirect_stdout(output),
            ):
                result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
            assert_eq(f"{expected_status} run succeeds", tc.EXIT_OK, result)
            assert_eq(f"{expected_status} run publishes at most once", 1, save.call_count)
            assert_true(
                f"{expected_status} status is explicit",
                f"status: {expected_status}" in output.getvalue(),
            )


def test_complete_cached_and_fresh_run_publishes_once():
    with tempfile.TemporaryDirectory() as directory:
        vault = Path(directory)
        wiki = vault / "wiki"
        wiki.mkdir()
        (wiki / "A.md").write_text("body A", encoding="utf-8")
        (wiki / "B.md").write_text("body B", encoding="utf-8")
        cached = {
            "version": 2,
            "model": "m1",
            "digest": DIGEST,
            "embeddings": {
                "wiki/A.md": {
                    "hash": tc.body_hash("body A", "m1", DIGEST),
                    "embedding": [1.0, 0.5],
                }
            },
        }
        output = io.StringIO()
        with (
            mock.patch.object(tc, "VAULT_ROOT", vault),
            mock.patch.object(tc, "WIKI_DIR", wiki),
            mock.patch.object(
                tc,
                "load_thresholds",
                return_value={"version": 1, "bands": {"review": 0.8, "error": 0.9}},
            ),
            mock.patch.object(tc, "detect_ollama", return_value=True),
            mock.patch.object(tc, "detect_model", return_value=True),
            mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
            mock.patch.object(tc, "load_cache", return_value=cached),
            mock.patch.object(tc, "embed", return_value=[0.5, 1.0]) as embed,
            mock.patch.object(tc, "save_cache") as save,
            contextlib.redirect_stdout(output),
        ):
            result = tc.run_check(False, None, "http://127.0.0.1:11434", "m1")
    assert_eq("cached and fresh complete run succeeds", tc.EXIT_OK, result)
    assert_eq("cached and fresh run embeds one page", 1, embed.call_count)
    assert_eq("cached and fresh run publishes once", 1, save.call_count)
    assert_true(
        "cached and fresh report accounts for both paths",
        "cache hits: 1; recomputed: 1" in output.getvalue(),
    )


def test_peek_proves_local_state_without_writes():
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "tiling-cache.json"
        output = io.StringIO()
        with (
            mock.patch.object(tc, "CACHE_PATH", cache),
            mock.patch.object(tc, "THRESHOLDS_PATH", Path(directory) / "thresholds.json"),
            mock.patch.object(tc, "detect_ollama", return_value=True),
            mock.patch.object(tc, "detect_model", return_value=True),
            mock.patch.object(tc, "local_model_proof", return_value=local_proof()),
            mock.patch.object(tc, "save_cache") as save,
            contextlib.redirect_stdout(output),
        ):
            result = tc.cmd_peek("http://127.0.0.1:11434", "m1")
    assert_eq("peek with local proof succeeds", tc.EXIT_OK, result)
    assert_eq("peek writes no cache", 0, save.call_count)
    assert_true("peek reports local proof", json.loads(output.getvalue())["local_model_proved"])


FAKE_MODEL = "m1:latest"


def _fake_server_vault(directory, pages=4):
    vault = Path(directory)
    wiki = vault / "wiki"
    wiki.mkdir()
    for index in range(pages):
        (wiki / f"P{index}.md").write_text(
            f"---\ntype: concept\n---\nprivate page body {index}\n", encoding="utf-8"
        )
    meta = vault / ".vault-meta"
    meta.mkdir()
    cache = meta / "tiling-cache.json"
    prior = {"version": 2, "model": FAKE_MODEL, "digest": DIGEST, "embeddings": {}}
    cache.write_bytes((json.dumps(prior, indent=2) + "\n").encode("utf-8"))
    return vault, cache


def _run_tiling_against(server_url, vault):
    """Run the real CLI entry point against a loopback fake server."""
    original = (
        tc.VAULT_ROOT,
        tc.WIKI_DIR,
        tc.META_DIR,
        tc.CACHE_PATH,
        tc.CACHE_LOCK,
        tc.THRESHOLDS_PATH,
    )
    output = io.StringIO()
    errors = io.StringIO()
    try:
        with (
            mock.patch.dict(os.environ, {"OLLAMA_URL": server_url}),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            try:
                result = tc.main(["--vault", str(vault), "--model", FAKE_MODEL])
            except Exception as exc:  # a traceback is the regression
                raise Fail(f"FAIL tiling raised {type(exc).__name__}: {exc}") from exc
    finally:
        (
            tc.VAULT_ROOT,
            tc.WIKI_DIR,
            tc.META_DIR,
            tc.CACHE_PATH,
            tc.CACHE_LOCK,
            tc.THRESHOLDS_PATH,
        ) = original
    return result, output.getvalue(), errors.getvalue()


def test_provider_failure_circuit_stops_after_first_page_body():
    for mode, expected_posts in (("embremote", 1), ("emb500", 1), ("dim", 2)):
        with tempfile.TemporaryDirectory() as directory, FakeOllamaServer(mode, FAKE_MODEL) as server:
            vault, cache = _fake_server_vault(directory, pages=4)
            before = cache.read_bytes()
            result, report, errors = _run_tiling_against(server.url, vault)
            posts = server.count("POST", "/api/embeddings")
            after = cache.read_bytes()
        assert_eq(f"{mode}: provider failure is incomplete", tc.EXIT_INCOMPLETE, result)
        assert_eq(f"{mode}: no page body is sent after the first failure", expected_posts, posts)
        assert_eq(f"{mode}: incomplete run preserves cache bytes", before, after)
        assert_true(f"{mode}: report is provisional", "status: incomplete" in report)
        assert_true(f"{mode}: circuit is reported", "provider circuit opened" in errors)


def test_duplicate_remote_status_keys_block_every_page_body():
    with tempfile.TemporaryDirectory() as directory, FakeOllamaServer("dupkey", FAKE_MODEL) as server:
        vault, cache = _fake_server_vault(directory)
        before = cache.read_bytes()
        result, _report, errors = _run_tiling_against(server.url, vault)
        posts = server.count("POST", "/api/embeddings")
        shows = server.count("POST", "/api/show")
        after = cache.read_bytes()
    assert_eq("duplicate remote key is an unproved local state", tc.EXIT_INCOMPLETE, result)
    assert_eq("duplicate remote key sends no page body", 0, posts)
    assert_eq("duplicate remote key stops before model metadata", 0, shows)
    assert_true("duplicate key is named", "duplicate JSON object key" in errors)
    assert_eq("duplicate remote key preserves cache", before, after)


def test_malformed_http_status_uses_documented_exits():
    for mode, expected in (("badstatus", tc.EXIT_NO_OLLAMA), ("badproof", tc.EXIT_INCOMPLETE)):
        with tempfile.TemporaryDirectory() as directory, FakeOllamaServer(mode, FAKE_MODEL) as server:
            vault, _cache = _fake_server_vault(directory)
            result, _report, _errors = _run_tiling_against(server.url, vault)
            posts = server.count("POST", "/api/embeddings")
        assert_eq(f"{mode}: malformed status line has a controlled exit", expected, result)
        assert_eq(f"{mode}: malformed status line sends no page body", 0, posts)


def test_unrelated_cloud_model_does_not_block_local_tiling():
    with tempfile.TemporaryDirectory() as directory, FakeOllamaServer("othercloud", FAKE_MODEL) as server:
        vault, _cache = _fake_server_vault(directory, pages=3)
        result, report, errors = _run_tiling_against(server.url, vault)
        posts = server.count("POST", "/api/embeddings")
    assert_eq("unrelated cloud model leaves local tiling available", tc.EXIT_OK, result)
    assert_eq("every local page is embedded once", 3, posts)
    assert_true("complete report", "status: complete" in report)
    assert_true("no misleading malformed-list message", "malformed" not in errors)
    with tempfile.TemporaryDirectory() as directory, FakeOllamaServer("selfcloud", FAKE_MODEL) as server:
        vault, _cache = _fake_server_vault(directory, pages=3)
        result, _report, errors = _run_tiling_against(server.url, vault)
        posts = server.count("POST", "/api/embeddings")
    assert_eq("requested remote model still fails closed", tc.EXIT_INCOMPLETE, result)
    assert_eq("requested remote model receives no page body", 0, posts)
    assert_true("remote identity is named accurately", "remote identity" in errors)
    assert_true("requested remote model is not called malformed", "malformed" not in errors)


def test_request_deadline_bounds_slow_drip_transport():
    with SlowDripServer(phase="body", interval=0.2) as server:
        started = time.monotonic()
        try:
            tc._http_get_json(server.url + "/api/status", 1)
        except OSError:
            pass
        else:
            raise Fail("FAIL slow-drip response was accepted past the request deadline")
        elapsed = time.monotonic() - started
    assert_true(f"slow-drip request stops near its deadline ({elapsed:.1f}s)", elapsed < 3.0)


if __name__ == "__main__":
    try:
        test_cosine()
        test_frontmatter()
        test_body_hash_model_scoped()
        test_included_basic()
        test_is_local_url()
        test_transport_disables_proxies_and_rejects_redirects()
        test_embedding_vectors_are_bounded_finite_and_nonboolean()
        test_cache_schema()
        test_threshold_validation_precedes_provider_and_cache()
        test_url_guard_via_subprocess()
        test_report_path_cannot_overwrite_vault_content()
        test_peek_localhost_transport_uses_numeric_loopback()
        test_json_conversion_failures_use_controlled_cache_and_threshold_errors()
        test_unproved_local_state_precedes_cache_and_page_reads()
        test_partial_and_total_embedding_failures_preserve_prior_cache()
        test_fresh_embedding_dimension_mismatch_is_incomplete()
        test_empty_and_complete_runs_publish_once()
        test_complete_cached_and_fresh_run_publishes_once()
        test_peek_proves_local_state_without_writes()
        test_provider_failure_circuit_stops_after_first_page_body()
        test_duplicate_remote_status_keys_block_every_page_body()
        test_malformed_http_status_uses_documented_exits()
        test_unrelated_cloud_model_does_not_block_local_tiling()
        test_request_deadline_bounds_slow_drip_transport()
    except Fail as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print("\nAll tests passed.")
