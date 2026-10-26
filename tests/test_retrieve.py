#!/usr/bin/env python3
"""test_retrieve.py — hermetic tests for scripts/retrieve.py and scripts/rerank.py.

No network, no ollama, no LLM calls. Tests cover:
  - import_sibling resolves hyphenated module names
  - chunk_snippet truncation behavior
  - rerank.cosine math correctness
  - rerank.rerank() no-op behavior when ollama is unreachable
  - retrieve.py exit 10 (not provisioned) when chunks/index are missing
  - dedupe-by-page logic via integration smoke test on synthetic fixtures

Usage:
  python3 tests/test_retrieve.py
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRIEVE = ROOT / "scripts" / "retrieve.py"
RERANK = ROOT / "scripts" / "rerank.py"
BM25 = ROOT / "scripts" / "bm25-index.py"


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retrieve = import_script("retrieve", RETRIEVE)
rerank = import_script("rerank", RERANK)
bm25 = import_script("bm25", BM25)

if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from test_ollama import FakeOllamaServer, SlowDripServer  # noqa: E402

PROOF_DIGEST = "sha256:" + "a" * 64


def local_proof(model=None):
    return {"model": model or rerank.DEFAULT_MODEL, "digest": PROOF_DIGEST}


class Fail(SystemExit):
    pass


def assert_eq(label, expected, actual):
    if expected != actual:
        raise Fail(f"FAIL {label}: expected {expected!r}, got {actual!r}")
    print(f"OK   {label}")


def assert_true(label, cond, hint=""):
    if not cond:
        raise Fail(f"FAIL {label}{(': ' + hint) if hint else ''}")
    print(f"OK   {label}")


def assert_close(label, expected, actual, eps=1e-6):
    if abs(expected - actual) > eps:
        raise Fail(f"FAIL {label}: expected ~{expected}, got {actual}")
    print(f"OK   {label}")


def sha256_text(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sandbox_env(vault):
    """Let copied helpers import the real package while selecting the sandbox."""
    env = os.environ.copy()
    env["CLAUDE_OBSIDIAN_VAULT"] = str(vault)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return env


# ─── import_sibling ──────────────────────────────────────────────────────────
def test_import_sibling_resolves_hyphenated_names():
    """retrieve.import_sibling('bm25_index', 'bm25-index.py') must succeed."""
    mod = retrieve.import_sibling("bm25_index", "bm25-index.py")
    assert_true("import_sibling returns module", mod is not None)
    assert_true("module has tokenize()", callable(getattr(mod, "tokenize", None)))


# ─── chunk_snippet ───────────────────────────────────────────────────────────
def test_chunk_snippet_short():
    """Short chunks should pass through unchanged."""
    out = retrieve.chunk_snippet({"raw_text": "short text"}, max_chars=200)
    assert_eq("chunk_snippet short pass-through", "short text", out)


def test_chunk_snippet_truncates_with_ellipsis():
    """Long chunks should be truncated with an ellipsis."""
    long_text = "x" * 500
    out = retrieve.chunk_snippet({"raw_text": long_text}, max_chars=100)
    assert_true("snippet length under cap", len(out) <= 110, hint=f"len={len(out)}")
    assert_true("snippet ends with ellipsis", out.endswith("…"))


# ─── rerank.cosine() ─────────────────────────────────────────────────────────
def test_cosine_identical():
    assert_close(
        "cosine identical vectors", 1.0, rerank.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    )
    assert_close("cosine huge identical vectors", 1.0, rerank.cosine([1e308], [1e308]))
    assert_close("cosine tiny identical vectors", 1.0, rerank.cosine([1e-308], [1e-308]))


def test_cosine_orthogonal():
    assert_close("cosine orthogonal", 0.0, rerank.cosine([1.0, 0.0], [0.0, 1.0]))


def test_cosine_anti_parallel():
    assert_close("cosine anti-parallel", -1.0, rerank.cosine([1.0, 0.0], [-1.0, 0.0]))


def test_cosine_length_mismatch():
    """Mismatched vector lengths should return 0.0 (defensive, not crash)."""
    assert_close("cosine length mismatch", 0.0, rerank.cosine([1.0], [1.0, 2.0]))


def test_cosine_zero_vector():
    assert_close("cosine zero vector", 0.0, rerank.cosine([0.0, 0.0], [1.0, 2.0]))


def test_multilingual_model_default_and_validation():
    assert_eq(
        "multilingual Nomic v2 is the default",
        "nomic-embed-text-v2-moe",
        rerank.DEFAULT_MODEL,
    )
    for valid in (
        "nomic-embed-text-v2-moe",
        "nomic-embed-text:v1.5",
        "registry.example/library/nomic-embed-text:latest",
    ):
        assert_eq("valid model name round-trips", valid, rerank.validate_model_name(valid))
    for invalid in (
        "",
        "https://example.com/model",
        "model name",
        "namespace/../model",
        "x" * 256,
    ):
        try:
            rerank.validate_model_name(invalid)
            raise Fail(f"invalid model name accepted: {invalid!r}")
        except ValueError:
            print(f"OK   invalid model name rejected: {invalid!r}")


def test_model_availability_handles_exact_and_tagged_names():
    assert_true(
        "untagged model matches installed latest tag",
        rerank.model_is_available(
            "nomic-embed-text-v2-moe", ["nomic-embed-text-v2-moe:latest"]
        ),
    )
    assert_true(
        "exact tagged model matches",
        rerank.model_is_available(
            "nomic-embed-text:v1.5", ["nomic-embed-text:v1.5"]
        ),
    )
    assert_true(
        "different explicit tag does not match",
        not rerank.model_is_available(
            "nomic-embed-text:v1.5", ["nomic-embed-text:latest"]
        ),
    )
    assert_true(
        "untagged model does not match an arbitrary installed tag",
        not rerank.model_is_available(
            "nomic-embed-text", ["nomic-embed-text:v1.5"]
        ),
    )


def test_retrieval_top_arguments_are_positive_and_bounded():
    cases = (
        (RETRIEVE, ["query", "--top", "0"]),
        (RETRIEVE, ["query", "--bm25-top", "-1"]),
        (RERANK, ["query", "--candidates", "-", "--top", "1001"]),
    )
    for script, arguments in cases:
        result = subprocess.run(
            [sys.executable, str(script), *arguments],
            input="[]" if script == RERANK else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert_eq(f"{script.name} rejects invalid top bound", 2, result.returncode)


def test_embed_one_uses_current_single_input_ollama_api():
    class Response:
        def __init__(self, url):
            self.url = url
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read(limit):
            assert_eq("embedding response read is bounded", 4 * 1024 * 1024 + 1, limit)
            return json.dumps({"embeddings": [[1.0, 2.0, 3.0]]}).encode()

        @staticmethod
        def getcode():
            return 200

        def geturl(self):
            return self.url

    def transport(request, timeout):
        return Response(request.full_url)

    with unittest.mock.patch.object(
        rerank, "_open_ollama_request", side_effect=transport
    ) as request_call:
        vector = rerank.embed_one(
            "http://127.0.0.1:11434",
            "nomic-embed-text-v2-moe",
            "search_query: 日本語",
        )
    request = request_call.call_args.args[0]
    payload = json.loads(request.data)
    assert_eq(
        "current Ollama embed endpoint",
        "http://127.0.0.1:11434/api/embed",
        request.full_url,
    )
    assert_eq(
        "single-input Ollama payload",
        {
            "model": "nomic-embed-text-v2-moe",
            "input": "search_query: 日本語",
        },
        payload,
    )
    assert_eq("single embedding is unwrapped", [1.0, 2.0, 3.0], vector)
    for response in (
        {"embeddings": [[1.0]], "remote_model": "remote:tag"},
        {"embeddings": [[1.0]], "model": "other"},
    ):
        with unittest.mock.patch.object(rerank, "_ollama_json", return_value=response):
            try:
                rerank.embed_one(
                    "http://127.0.0.1:11434",
                    rerank.DEFAULT_MODEL,
                    "private query",
                    proof=local_proof(),
                )
                raise Fail("remote or mismatched rerank response accepted")
            except rerank.OllamaLocalProofError:
                print("OK   remote or mismatched rerank response rejected")


def test_rerank_transport_disables_proxies_and_rejects_redirects():
    request = rerank.urllib.request.Request(
        "http://127.0.0.1:11434/api/tags", method="GET"
    )
    opener = unittest.mock.Mock()
    sentinel = object()
    opener.open.return_value = sentinel
    with unittest.mock.patch.object(
        rerank.urllib.request, "build_opener", return_value=opener
    ) as builder:
        result = rerank._open_ollama_request(request, 3)
    assert_true("rerank transport returns opener response", result is sentinel)
    handlers = builder.call_args.args
    proxy_handlers = [
        handler for handler in handlers
        if isinstance(handler, rerank.urllib.request.ProxyHandler)
    ]
    assert_eq("rerank transport installs one proxy handler", 1, len(proxy_handlers))
    assert_eq("rerank transport disables ambient proxies", {}, proxy_handlers[0].proxies)
    assert_true(
        "rerank transport refuses redirects",
        any(isinstance(handler, rerank._NoOllamaRedirect) for handler in handlers),
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
            return "https://collector.example/api/embed"

        @staticmethod
        def read(_limit):
            return b'{"embeddings":[[1.0,2.0]]}'

    with unittest.mock.patch.object(
        rerank, "_open_ollama_request", return_value=RedirectedResponse()
    ):
        try:
            rerank.embed_one("http://127.0.0.1:11434", "fixture", "private")
            raise Fail("FAIL redirected rerank response should be rejected")
        except ValueError as exc:
            assert_true("rerank redirect rejection is explicit", "target changed" in str(exc))


def test_rerank_input_validation_precedes_model_provider_and_cache():
    valid = {"chunk_id": "c-001:0", "bm25_score": 1.0, "path": "x"}
    cases = (
        ("\ufb03" * 2667, [valid], "normalized query bound"),
        ("query", [1], "candidate object"),
        ("query", [valid] * 1001, "candidate count"),
        ("query", [{**valid, "path": "../outside.json"}], "candidate path"),
        ("query", [{**valid, "bm25_score": float("inf")}], "candidate score"),
        ("query", [{**valid, "bm25_score": 10 ** 400}], "candidate score overflow"),
    )
    for query, candidates, label in cases:
        events = []
        with (
            unittest.mock.patch.object(
                rerank, "validate_model_name",
                side_effect=lambda value: events.append("model") or value,
            ),
            unittest.mock.patch.object(
                rerank, "ollama_url", side_effect=lambda _allow: events.append("url") or "unused"
            ),
            unittest.mock.patch.object(
                rerank, "load_cache", side_effect=lambda: events.append("cache") or {}
            ),
            unittest.mock.patch.object(
                rerank, "embed_one", side_effect=lambda *_args: events.append("embed") or [1.0]
            ),
        ):
            try:
                rerank.rerank(query, candidates, top_k=1)
                raise Fail(f"FAIL invalid {label} should be rejected")
            except ValueError:
                pass
        assert_eq(f"invalid {label} has no provider or cache events", [], events)

    stderr = io.StringIO()
    with (
        unittest.mock.patch.object(rerank, "configure_vault", return_value=True),
        unittest.mock.patch.object(sys, "stdin", io.StringIO("[1]")),
        contextlib.redirect_stderr(stderr),
    ):
        result = rerank.main(["query", "--candidates", "-"])
    assert_eq("malformed candidate CLI exit", rerank.EXIT_CANDIDATES, result)
    assert_true("malformed candidate CLI error is controlled", "Traceback" not in stderr.getvalue())


def test_rerank_invalid_utf8_cache_is_a_miss():
    with tempfile.TemporaryDirectory() as directory:
        cache_path = Path(directory) / "embed-cache.json"
        cache_path.write_bytes(b"\xff")
        with unittest.mock.patch.object(rerank, "EMBED_CACHE_PATH", cache_path):
            assert_eq("invalid UTF-8 rerank cache is a miss", {}, rerank.load_cache())


def test_rerank_malformed_candidate_and_chunk_files_are_controlled():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate_path = root / "candidates.json"
        candidate_path.write_bytes(b"\xff")
        events = []
        stderr = io.StringIO()
        with (
            unittest.mock.patch.object(rerank, "configure_vault", return_value=True),
            unittest.mock.patch.object(rerank, "assert_within", return_value=candidate_path),
            unittest.mock.patch.object(
                rerank, "ollama_url", side_effect=lambda _allow: events.append("provider")
            ),
            unittest.mock.patch.object(
                rerank, "load_cache", side_effect=lambda: events.append("cache") or {}
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = rerank.main(["query", "--candidates", "candidates.json"])
        assert_eq("invalid UTF-8 candidate file exit", rerank.EXIT_CANDIDATES, result)
        assert_true("invalid UTF-8 candidate file is controlled", "Traceback" not in stderr.getvalue())
        assert_eq("invalid UTF-8 candidate file precedes provider and cache", [], events)

        class InvalidStdin:
            @staticmethod
            def read():
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        stderr = io.StringIO()
        with (
            unittest.mock.patch.object(rerank, "configure_vault", return_value=True),
            unittest.mock.patch.object(sys, "stdin", InvalidStdin()),
            contextlib.redirect_stderr(stderr),
        ):
            result = rerank.main(["query", "--candidates", "-"])
        assert_eq("invalid UTF-8 candidate stdin exit", rerank.EXIT_CANDIDATES, result)
        assert_true("invalid UTF-8 candidate stdin is controlled", "Traceback" not in stderr.getvalue())

        chunks = root / ".vault-meta" / "chunks"
        chunks.mkdir(parents=True)
        chunk_path = chunks / "bad.json"
        with (
            unittest.mock.patch.object(rerank, "VAULT_ROOT", root),
            unittest.mock.patch.object(rerank, "META_DIR", root / ".vault-meta"),
        ):
            chunk_path.write_bytes(b"\xff")
            assert_eq("invalid UTF-8 chunk is unavailable", None, rerank.load_chunk(".vault-meta/chunks/bad.json"))
            chunk_path.write_text("[]", encoding="utf-8")
            assert_eq("non-object chunk is unavailable", None, rerank.load_chunk(".vault-meta/chunks/bad.json"))
            chunk_path.write_text('{"raw_text":1}', encoding="utf-8")
            assert_eq("non-text chunk is unavailable", None, rerank.load_chunk(".vault-meta/chunks/bad.json"))

            candidates = [
                {"chunk_id": "c-bad:0", "bm25_score": 2.0, "path": ".vault-meta/chunks/bad.json"}
            ]
            with (
                unittest.mock.patch.object(
                    rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
                ),
                unittest.mock.patch.object(
                    rerank, "local_model_proof", return_value=local_proof()
                ),
                unittest.mock.patch.object(rerank, "load_cache", return_value={}),
                unittest.mock.patch.object(rerank, "embed_one", return_value=[1.0]),
            ):
                result = rerank.rerank("query", candidates, top_k=1)
            assert_eq("non-text chunk keeps whole-set fallback", "noop-missing-chunk", result[0]["rerank_source"])
            assert_eq("non-text chunk keeps BM25 score", 2.0, result[0]["rerank_score"])


# ─── rerank.rerank() no-op fallback ──────────────────────────────────────────
def test_rerank_noop_when_ollama_unreachable():
    """When ollama is not reachable, rerank should pass candidates through with
    rerank_source='noop-no-ollama'. We force this by patching ollama_alive."""
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        candidates = [
            {"chunk_id": "c-001:0", "bm25_score": 7.5, "path": "fake/p1.json"},
            {"chunk_id": "c-002:0", "bm25_score": 5.1, "path": "fake/p2.json"},
        ]
        out = rerank.rerank("query", candidates, top_k=5)
        assert_eq(
            "rerank no-op preserves order",
            ["c-001:0", "c-002:0"],
            [c["chunk_id"] for c in out],
        )
        assert_true(
            "rerank no-op tags source",
            all(c.get("rerank_source") == "noop-no-ollama" for c in out),
        )
        assert_true(
            "rerank no-op copies score to rerank_score",
            all(c["rerank_score"] == c["bm25_score"] for c in out),
        )


def test_rerank_noop_when_model_missing():
    """When ollama is up but model isn't pulled, rerank should still no-op."""
    with unittest.mock.patch.object(
        rerank, "ollama_alive", return_value=(True, ["other-model"])
    ):
        candidates = [{"chunk_id": "c-001:0", "bm25_score": 5.0, "path": "x"}]
        out = rerank.rerank("query", candidates, top_k=5)
        assert_eq(
            "rerank no-op for missing model", "noop-no-model", out[0]["rerank_source"]
        )
        assert_eq(
            "missing-model fallback keeps BM25 score", 5.0, out[0]["rerank_score"]
        )


def test_rerank_local_proof_refusal_precedes_cache_and_payload():
    events = []
    candidates = [
        {"chunk_id": "c-001:0", "bm25_score": 5.0, "path": "private.json"}
    ]
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank,
            "local_model_proof",
            side_effect=rerank.OllamaLocalProofError("cloud is enabled"),
        ),
        unittest.mock.patch.object(
            rerank,
            "load_cache",
            side_effect=lambda: events.append("cache") or {},
        ),
        unittest.mock.patch.object(
            rerank,
            "embed_one",
            side_effect=lambda *_args, **_kwargs: events.append("embed"),
        ),
    ):
        output = rerank.rerank("private query", candidates, top_k=1)
    assert_eq("local proof refusal has no cache or embed work", [], events)
    assert_eq(
        "local proof refusal keeps BM25 diagnostic",
        "noop-local-proof",
        output[0]["rerank_source"],
    )
    assert_eq("local proof refusal keeps BM25 score", 5.0, output[0]["rerank_score"])


def test_rerank_truncates_to_top_k():
    with unittest.mock.patch.object(rerank, "ollama_alive", return_value=(False, [])):
        candidates = [
            {"chunk_id": f"c-{i:03}:0", "score": float(i), "path": "x"}
            for i in range(10)
        ]
        out = rerank.rerank("query", candidates, top_k=3)
        assert_eq("rerank truncates to top_k", 3, len(out))


def test_rerank_query_embed_failure_preserves_bm25():
    candidates = [
        {"chunk_id": "c-001:0", "bm25_score": 9.0, "path": "x"},
        {"chunk_id": "c-002:0", "bm25_score": 4.0, "path": "y"},
    ]
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=OSError("boom")),
    ):
        out = rerank.rerank("query", candidates, top_k=5)
    assert_eq(
        "query-embed failure preserves BM25 order",
        ["c-001:0", "c-002:0"],
        [c["chunk_id"] for c in out],
    )
    assert_eq(
        "query-embed failure preserves BM25 scores",
        [9.0, 4.0],
        [c["rerank_score"] for c in out],
    )


def test_rerank_document_failure_preserves_candidate_bm25_score():
    candidates = [
        {"chunk_id": "c-001:0", "bm25_score": 8.0, "path": "x"},
        {"chunk_id": "c-002:0", "bm25_score": 6.25, "path": "y"},
    ]
    chunks = [
        {
            "body_hash": "sha256:first",
            "contextualized_text": "first",
            "raw_text": "first",
        },
        {
            "body_hash": "sha256:second",
            "contextualized_text": "second",
            "raw_text": "second",
        },
    ]

    def fake_embed(url, model, text, **_kwargs):
        if text.startswith("search_query: "):
            return [1.0, 2.0]
        if text == "search_document: first":
            return [2.0, 1.0]
        raise OSError("document embed failed")

    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=fake_embed),
        unittest.mock.patch.object(rerank, "load_chunk", side_effect=chunks),
        unittest.mock.patch.object(rerank, "load_cache", return_value={}),
    ):
        out = rerank.rerank("query", candidates, top_k=2)
    assert_eq(
        "partial document failure restores BM25 order",
        ["c-001:0", "c-002:0"],
        [c["chunk_id"] for c in out],
    )
    assert_eq(
        "partial document failure restores all BM25 scores",
        [8.0, 6.25],
        [c["rerank_score"] for c in out],
    )
    assert_true(
        "document failure marks complete no-op",
        all(c["rerank_source"] == "noop-embed-error" for c in out),
    )


def test_rerank_missing_chunk_preserves_candidate_bm25_score():
    candidate = {"chunk_id": "c-001:0", "bm25_score": 3.5, "path": "missing"}
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", return_value=[1.0, 2.0]),
        unittest.mock.patch.object(rerank, "load_chunk", return_value=None),
        unittest.mock.patch.object(rerank, "load_cache", return_value={}),
    ):
        out = rerank.rerank("query", [candidate], top_k=1)
    assert_eq("missing chunk keeps BM25 score", 3.5, out[0]["rerank_score"])
    assert_eq(
        "missing chunk is identified", "noop-missing-chunk", out[0]["rerank_source"]
    )


# ─── Nomic asymmetric task prefixes and cache scheme ──────────────────────
def test_with_task_prefix_contract():
    assert_eq(
        "Nomic query prefix",
        "search_query: question",
        rerank.with_task_prefix("question", "query", "nomic-embed-text"),
    )
    assert_eq(
        "Nomic document prefix",
        "search_document: passage",
        rerank.with_task_prefix("passage", "document", "nomic-embed-text:latest"),
    )
    assert_eq(
        "non-Nomic input remains raw",
        "question",
        rerank.with_task_prefix("question", "query", "mxbai-embed-large"),
    )
    try:
        rerank.with_task_prefix("text", "passage", "nomic-embed-text")
        raise Fail("with_task_prefix should reject an unknown role")
    except ValueError:
        print("OK   Nomic prefix rejects unknown role")


def test_rerank_sends_nomic_task_prefixes():
    sent = []

    def fake_embed(url, model, text, **_kwargs):
        sent.append(text)
        return [1.0, 2.0, 3.0]

    chunk = {
        "body_hash": "sha256:deadbeef",
        "contextualized_text": "advisory locks are age-based",
        "raw_text": "advisory locks are age-based",
    }
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=fake_embed),
        unittest.mock.patch.object(rerank, "load_chunk", return_value=chunk),
        unittest.mock.patch.object(rerank, "load_cache", return_value={}),
        unittest.mock.patch.object(rerank, "save_cache"),
    ):
        rerank.rerank(
            "how does locking work",
            [{"chunk_id": "c-001:0", "bm25_score": 5.0, "path": "x"}],
            top_k=1,
        )
    assert_eq(
        "embedder receives asymmetric Nomic inputs",
        [
            "search_query: how does locking work",
            "search_document: advisory locks are age-based",
        ],
        sent,
    )


def test_selected_model_is_used_for_embeddings_cache_and_source():
    selected_model = "nomic-embed-text-v2-moe:latest"
    sent = []
    saved = {}

    def fake_embed(url, model, text, **_kwargs):
        sent.append((model, text))
        return [1.0, 2.0, 3.0]

    def capture_cache(cache):
        saved.update(cache)

    chunk = {
        "body_hash": "sha256:deadbeef",
        "contextualized_text": "日本語の全文検索機能",
        "raw_text": "日本語の全文検索機能",
    }
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [selected_model])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof(selected_model)
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=fake_embed),
        unittest.mock.patch.object(rerank, "load_chunk", return_value=chunk),
        unittest.mock.patch.object(rerank, "load_cache", return_value={}),
        unittest.mock.patch.object(rerank, "save_cache", side_effect=capture_cache),
    ):
        output = rerank.rerank(
            "日本語の検索",
            [{"chunk_id": "c-001:0", "bm25_score": 5.0, "path": "x"}],
            top_k=1,
            model=selected_model,
        )

    assert_true(
        "selected model reaches every embedding request",
        sent and all(model == selected_model for model, _ in sent),
        hint=repr(sent),
    )
    assert_eq(
        "CJK query receives Nomic prefix",
        "search_query: 日本語の検索",
        sent[0][1],
    )
    assert_true(
        "CJK document receives Nomic prefix",
        any(text == "search_document: 日本語の全文検索機能" for _, text in sent),
    )
    assert_true(
        "cache key contains exact selected model",
        saved and all(key.startswith(selected_model + ":") for key in saved),
        hint=repr(saved),
    )
    assert_eq(
        "rerank source contains exact selected model",
        f"cosine:{selected_model}",
        output[0]["rerank_source"],
    )


def test_peek_reports_selected_tagged_model():
    output = io.StringIO()
    selected_model = "nomic-embed-text-v2-moe"
    with (
        unittest.mock.patch.object(rerank, "configure_vault", return_value=True),
        unittest.mock.patch.object(
            rerank, "ollama_url", return_value="http://127.0.0.1:11434"
        ),
        unittest.mock.patch.object(
            rerank,
            "ollama_alive",
            return_value=(True, [selected_model + ":latest"]),
        ),
        unittest.mock.patch.object(
            rerank,
            "local_model_proof",
            return_value=local_proof(selected_model),
        ),
        contextlib.redirect_stdout(output),
    ):
        result = rerank.main(["query", "--peek", "--model", selected_model])
    payload = json.loads(output.getvalue())
    assert_eq("peek succeeds", rerank.EXIT_OK, result)
    assert_eq("peek reports selected model", selected_model, payload["model"])
    assert_true("peek recognizes installed tagged model", payload["model_present"])
    assert_eq(
        "peek reports exact strategy",
        f"cosine:{selected_model}",
        payload["strategy"],
    )


def test_embedding_cache_key_versions_model_and_scheme():
    input_hash = rerank.embedding_input_hash("search_document: passage")
    key = rerank.embedding_cache_key("nomic-embed-text", input_hash)
    assert_true(
        "cache key contains model", key.startswith("nomic-embed-text:"), hint=key
    )
    assert_true(
        "cache key contains Nomic scheme", rerank.NOMIC_EMBED_SCHEME in key, hint=key
    )
    assert_true(
        "old prefix-less cache key differs",
        key != f"nomic-embed-text:{input_hash}",
        hint=key,
    )
    assert_true(
        "model change invalidates cache key",
        key != rerank.embedding_cache_key("mxbai-embed-large", input_hash),
    )
    changed_input = rerank.embedding_input_hash(
        "search_document: updated context\n\npassage"
    )
    assert_true(
        "context change invalidates cache key",
        key != rerank.embedding_cache_key("nomic-embed-text", changed_input),
    )


def test_stale_prefixless_cache_is_not_reused():
    body_hash = "sha256:deadbeef"
    stale = {f"{rerank.DEFAULT_MODEL}:{body_hash}": [99.0, 99.0, 99.0]}
    embedded = []

    def fake_embed(url, model, text, **_kwargs):
        embedded.append(text)
        return [1.0, 2.0, 3.0]

    chunk = {"body_hash": body_hash, "contextualized_text": "text", "raw_text": "text"}
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=fake_embed),
        unittest.mock.patch.object(rerank, "load_chunk", return_value=chunk),
        unittest.mock.patch.object(rerank, "load_cache", return_value=stale),
        unittest.mock.patch.object(rerank, "save_cache"),
    ):
        rerank.rerank("query", [{"chunk_id": "c-001:0", "score": 1.0, "path": "x"}], 1)
    assert_true(
        "old cache entry forces document re-embed",
        any(text.startswith("search_document: ") for text in embedded),
        hint=f"embedded={embedded}",
    )


def test_verified_digest_change_misses_rerank_cache():
    text = "search_document: text"
    input_hash = rerank.embedding_input_hash(text)
    old_key = rerank.embedding_cache_key(
        rerank.DEFAULT_MODEL,
        input_hash,
        digest="sha256:" + "b" * 64,
    )
    stale = {old_key: [99.0, 99.0, 99.0]}
    embedded = []
    saved = {}

    def fake_embed(url, model, value, **_kwargs):
        embedded.append(value)
        return [1.0, 2.0, 3.0]

    chunk = {"contextualized_text": "text", "raw_text": "text"}
    with (
        unittest.mock.patch.object(
            rerank, "ollama_alive", return_value=(True, [rerank.DEFAULT_MODEL])
        ),
        unittest.mock.patch.object(
            rerank, "local_model_proof", return_value=local_proof()
        ),
        unittest.mock.patch.object(rerank, "embed_one", side_effect=fake_embed),
        unittest.mock.patch.object(rerank, "load_chunk", return_value=chunk),
        unittest.mock.patch.object(rerank, "load_cache", return_value=stale),
        unittest.mock.patch.object(
            rerank, "save_cache", side_effect=lambda cache: saved.update(cache)
        ),
    ):
        rerank.rerank(
            "query",
            [{"chunk_id": "c-001:0", "score": 1.0, "path": "x"}],
            1,
        )
    assert_true(
        "digest drift forces document re-embed",
        text in embedded,
        hint=repr(embedded),
    )
    new_key = rerank.embedding_cache_key(
        rerank.DEFAULT_MODEL,
        input_hash,
        digest=PROOF_DIGEST,
    )
    assert_true("verified digest is present in new cache key", new_key in saved)


def test_remote_ollama_remains_explicit_opt_in():
    with unittest.mock.patch.dict(os.environ, {"OLLAMA_URL": "https://example.com"}):
        try:
            rerank.ollama_url(False)
            raise Fail("remote OLLAMA_URL should require explicit consent")
        except SystemExit as e:
            assert_eq("remote Ollama default-deny exit", rerank.EXIT_USAGE, e.code)
        assert_eq(
            "remote Ollama explicit opt-in",
            "https://example.com",
            rerank.ollama_url(True),
        )

    for invalid in (
        "file:///etc/passwd",
        "ftp://localhost/resource",
        "http://user:secret@localhost:11434",
        "http://localhost:11434?token=secret",
    ):
        with unittest.mock.patch.dict(os.environ, {"OLLAMA_URL": invalid}):
            for allow_remote in (False, True):
                try:
                    rerank.ollama_url(allow_remote)
                    raise Fail(f"invalid OLLAMA_URL accepted: {invalid}")
                except SystemExit as e:
                    assert_eq(
                        "invalid Ollama scheme/authority exit",
                        rerank.EXIT_USAGE,
                        e.code,
                    )


def test_rerank_localhost_transport_uses_numeric_loopback():
    requested = []

    class Response:
        def __init__(self, url):
            self.url = url
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read(limit):
            return b'{"models": []}'

        @staticmethod
        def getcode():
            return 200

        def geturl(self):
            return self.url

    def transport(request, timeout):
        requested.append(request.full_url)
        return Response(request.full_url)

    with (
        unittest.mock.patch.dict(
            os.environ,
            {"OLLAMA_URL": "http://localhost:12121/private-base"},
        ),
        unittest.mock.patch.object(
            rerank, "_open_ollama_request", side_effect=transport
        ),
    ):
        url = rerank.ollama_url(False)
        alive, _models = rerank.ollama_alive(url)
    assert_true("numeric-loopback rerank probe succeeds", alive)
    assert_eq(
        "rerank normalizes localhost and preserves port and base path",
        "http://127.0.0.1:12121/private-base",
        url,
    )
    assert_eq(
        "rerank transport receives only numeric loopback",
        ["http://127.0.0.1:12121/private-base/api/tags"],
        requested,
    )


def test_rerank_rejects_chunk_path_escape():
    assert_eq(
        "relative chunk escape rejected", None, rerank.load_chunk("../../outside.json")
    )
    assert_eq(
        "absolute chunk path rejected", None, rerank.load_chunk("/tmp/outside.json")
    )


# ─── retrieve.py CLI: exit 10 when not provisioned ────────────────────────────
def test_retrieve_exits_10_without_index():
    """End-to-end CLI test: with no .vault-meta/bm25/index.json, retrieve.py must exit 10."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a minimal vault layout under tmpdir
        sandbox = Path(tmpdir)
        (sandbox / ".vault-meta").mkdir()
        (sandbox / "wiki").mkdir()
        # Run product code against a separate vault. It should exit 10 because
        # no BM25 index exists, without treating the product root as storage.
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "test query",
            ],
            env=sandbox_env(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("retrieve.py exit 10 when not provisioned", 10, result.returncode)
        assert_true(
            "retrieve.py prints friendly error",
            "no BM25 index" in result.stderr,
            hint=result.stderr[:200],
        )


def test_retrieve_validates_and_forwards_explicit_model():
    selected_model = "nomic-embed-text-v2-moe:latest"
    received = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        chunks_dir = sandbox / ".vault-meta/chunks/c-000001"
        index_dir = sandbox / ".vault-meta/bm25"
        page = sandbox / "wiki/example.md"
        chunks_dir.mkdir(parents=True)
        index_dir.mkdir(parents=True)
        page.parent.mkdir(parents=True)

        page_text = "日本語の全文検索機能"
        chunk_text = "日本語の全文検索"
        page.write_text(page_text, encoding="utf-8")
        chunk_path = chunks_dir / "chunk-000.json"
        chunk_path.write_text(
            json.dumps(
                {
                    "page_address": "c-000001",
                    "page_path": "wiki/example.md",
                    "chunk_index": 0,
                    "raw_text": chunk_text,
                    "contextualized_text": chunk_text,
                    "body_hash": sha256_text(chunk_text),
                    "page_body_hash": sha256_text(page_text),
                }
            ),
            encoding="utf-8",
        )
        (index_dir / "index.json").write_text("{}\n", encoding="utf-8")

        class FakeBm25:
            EXIT_INDEX_MISSING = 3

            @staticmethod
            def configure_vault(_vault):
                return True

            @staticmethod
            def query(_query, top_k):
                del top_k
                return [
                    {
                        "chunk_id": "c-000001:0",
                        "score": 5.0,
                        "path": ".vault-meta/chunks/c-000001/chunk-000.json",
                        "body_hash": sha256_text(chunk_text),
                        "page_body_hash": sha256_text(page_text),
                    }
                ]

        class FakeReranker:
            DEFAULT_MODEL = rerank.DEFAULT_MODEL
            validate_model_name = staticmethod(rerank.validate_model_name)

            @staticmethod
            def configure_vault(_vault):
                return True

            @staticmethod
            def rerank(query, candidates, top_k, allow_remote, model):
                received.update(
                    {
                        "query": query,
                        "top_k": top_k,
                        "allow_remote": allow_remote,
                        "model": model,
                    }
                )
                for candidate in candidates:
                    candidate["rerank_score"] = 1.0
                    candidate["rerank_source"] = f"cosine:{model}"
                return candidates

        def fake_import(_name, filename):
            return FakeBm25 if filename == "bm25-index.py" else FakeReranker

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            unittest.mock.patch.object(
                retrieve, "import_sibling", side_effect=fake_import
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = retrieve.main(
                [
                    "--vault",
                    str(sandbox),
                    "日本語の検索",
                    "--model",
                    selected_model,
                    "--explain",
                ]
            )
        payload = json.loads(stdout.getvalue())
        assert_eq("retrieve explicit-model run succeeds", retrieve.EXIT_OK, result)
        assert_eq("retrieve forwards exact model", selected_model, received["model"])
        assert_eq(
            "retrieve strategy includes exact model",
            f"bm25+rerank:cosine:{selected_model}",
            payload["strategy"],
        )
        assert_eq(
            "retrieve explain reports exact model",
            selected_model,
            payload["explain"]["rerank_model"],
        )


def test_retrieve_translates_corrupt_index_to_documented_fallback_exit():
    """A malformed cache is unavailable, not an uncaught helper traceback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        chunks_dir = sandbox / ".vault-meta" / "chunks" / "c-000001"
        index_dir = sandbox / ".vault-meta" / "bm25"
        chunks_dir.mkdir(parents=True)
        index_dir.mkdir(parents=True)
        (sandbox / "wiki").mkdir()
        (index_dir / "index.json").write_text(
            json.dumps({"schema_version": 1, "params": []}),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "test query",
                "--no-rerank",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("corrupt index maps to retrieve exit 10", 10, result.returncode)
        assert_true(
            "corrupt helper emits stable diagnostic",
            "ERR BM25_INDEX_CORRUPT:" in result.stderr,
            hint=result.stderr,
        )
        assert_true(
            "retrieve emits stable unavailable diagnostic",
            "ERR RETRIEVAL_INDEX_UNAVAILABLE:" in result.stderr,
            hint=result.stderr,
        )
        assert_true(
            "retrieve provides exact rebuild guidance",
            "python3 scripts/bm25-index.py" in result.stderr
            and " build" in result.stderr,
            hint=result.stderr,
        )
        assert_true(
            "retrieve provides fallback guidance",
            "fall back to the standard vault query/text-search path" in result.stderr,
            hint=result.stderr,
        )
        assert_true(
            "corrupt retrieval has no traceback",
            "Traceback" not in result.stderr and "KeyError" not in result.stderr,
            hint=result.stderr,
        )


# ─── Integration smoke test: end-to-end with synthetic data ──────────────────
def test_end_to_end_with_synthetic_chunks():
    """Build a minimal vault with 2 chunks, index it, run retrieve, verify output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        pages_dir = sandbox / "wiki" / "fake"
        pages_dir.mkdir(parents=True)

        # Write 2 synthetic chunks
        def chunk(addr, idx, text):
            return {
                "schema_version": 1,
                "page_path": f"wiki/fake/{addr}.md",
                "page_address": addr,
                "chunk_index": idx,
                "raw_text": text,
                "contextualized_text": text,
                "prefix": "",
                "prefix_source": "synthetic",
                "char_count": len(text),
                "body_hash": sha256_text(text),
                "page_body_hash": sha256_text(text),
                "created_at": "2026-05-17T00:00:00Z",
            }

        (chunks_dir / "c-000001").mkdir()
        (chunks_dir / "c-000002").mkdir()
        (pages_dir / "c-000001.md").write_text(
            "compounding wiki vault pattern by karpathy"
        )
        (pages_dir / "c-000002.md").write_text("obsidian cli transport detection")
        (chunks_dir / "c-000001" / "chunk-000.json").write_text(
            json.dumps(
                chunk("c-000001", 0, "compounding wiki vault pattern by karpathy")
            )
        )
        (chunks_dir / "c-000002" / "chunk-000.json").write_text(
            json.dumps(chunk("c-000002", 0, "obsidian cli transport detection"))
        )
        # Product scripts operate on the separately selected sandbox vault.
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bm25-index.py"),
                "--vault",
                str(sandbox),
                "build",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("bm25 build rc=0", 0, result.returncode)
        # Run retrieve
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "karpathy wiki",
                "--top",
                "2",
                "--no-rerank",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("retrieve rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_eq("retrieve.strategy is bm25-only", "bm25-only", out["strategy"])
        assert_true(
            "retrieve returns at least 1 candidate", len(out["candidates"]) >= 1
        )
        # c-000001 should rank above c-000002 for "karpathy wiki"
        first = out["candidates"][0]
        assert_eq("top hit is c-000001", "c-000001", first["page_address"])


def test_dedupe_happens_before_final_top_k():
    """Multiple top chunks from one page must not crowd out another page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        chunks_dir = sandbox / ".vault-meta" / "chunks"
        chunks_dir.mkdir(parents=True)
        pages_dir = sandbox / "wiki" / "fake"
        pages_dir.mkdir(parents=True)

        page_bodies = {"c-000001": "page one", "c-000002": "page two"}
        for address, page_body in page_bodies.items():
            (pages_dir / f"{address}.md").write_text(page_body)
            (chunks_dir / address).mkdir()

        def write_chunk(address, index, text):
            record = {
                "schema_version": 1,
                "page_path": f"wiki/fake/{address}.md",
                "page_address": address,
                "chunk_index": index,
                "raw_text": text,
                "contextualized_text": text,
                "body_hash": sha256_text(text),
                "page_body_hash": sha256_text(page_bodies[address]),
            }
            (chunks_dir / address / f"chunk-{index:03d}.json").write_text(
                json.dumps(record)
            )

        write_chunk("c-000001", 0, "alpha alpha alpha alpha")
        write_chunk("c-000001", 1, "alpha alpha alpha beta")
        write_chunk("c-000002", 0, "alpha gamma delta epsilon")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bm25-index.py"),
                "--vault",
                str(sandbox),
                "build",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "alpha",
                "--top",
                "2",
                "--bm25-top",
                "3",
                "--no-rerank",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        addresses = [
            item["page_address"] for item in json.loads(result.stdout)["candidates"]
        ]
        assert_eq(
            "dedupe fills requested page count", ["c-000001", "c-000002"], addresses
        )


def test_retrieve_skips_page_deleted_after_index_build():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        chunks_dir = sandbox / ".vault-meta" / "chunks" / "c-000001"
        chunks_dir.mkdir(parents=True)
        page = sandbox / "wiki" / "fake" / "c-000001.md"
        page.parent.mkdir(parents=True)
        page_text = "stale retrieval sentinel"
        page.write_text(page_text)
        (chunks_dir / "chunk-000.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "page_path": "wiki/fake/c-000001.md",
                    "page_address": "c-000001",
                    "chunk_index": 0,
                    "raw_text": page_text,
                    "contextualized_text": page_text,
                    "body_hash": sha256_text(page_text),
                    "page_body_hash": sha256_text(page_text),
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bm25-index.py"),
                "--vault",
                str(sandbox),
                "build",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        page.unlink()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "sentinel",
                "--no-rerank",
                "--explain",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        out = json.loads(result.stdout)
        assert_eq("deleted source yields no candidates", [], out["candidates"])
        assert_eq(
            "deleted source counted as stale",
            1,
            out["explain"]["stale_candidates_skipped"],
        )


def test_chunk_currency_rejects_hashless_or_malformed_legacy_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        page = Path(tmpdir) / "wiki/fake/c-000001.md"
        page.parent.mkdir(parents=True)
        page_text = "current page text"
        page.write_text(page_text, encoding="utf-8")
        chunk = {
            "page_address": "c-000001",
            "chunk_index": 0,
            "raw_text": page_text,
            "body_hash": sha256_text(page_text),
            "page_body_hash": sha256_text(page_text),
        }
        hit = {
            "chunk_id": "c-000001:0",
            "body_hash": chunk["body_hash"],
            "page_body_hash": chunk["page_body_hash"],
        }
        assert_true(
            "fully hashed chunk is current",
            retrieve.chunk_is_current(hit, chunk, page, {}),
        )

        for label, target, field, value in (
            ("hashless chunk", chunk, "page_body_hash", None),
            ("malformed chunk hash", chunk, "body_hash", "sha256:nope"),
            ("hashless index hit", hit, "page_body_hash", None),
        ):
            changed = dict(target)
            changed[field] = value
            candidate_chunk = changed if target is chunk else chunk
            candidate_hit = changed if target is hit else hit
            assert_true(
                f"{label} rejected",
                not retrieve.chunk_is_current(candidate_hit, candidate_chunk, page, {}),
            )


def test_chunk_currency_accepts_crlf_pages():
    # Regression: chunk_is_current hashed pages via read_text (CRLF-normalizing),
    # so vaults with CRLF line endings dropped every candidate as stale and
    # retrieval returned zero results.  The hash must match the chunker's
    # byte-exact reading of the same page.
    with tempfile.TemporaryDirectory() as tmpdir:
        page = Path(tmpdir) / "wiki/fake/c-000001.md"
        page.parent.mkdir(parents=True)
        page.write_bytes(b"crlf line one\r\ncrlf line two\r\n")
        page_text = page.read_bytes().decode("utf-8", errors="replace")
        chunk = {
            "page_address": "c-000001",
            "chunk_index": 0,
            "raw_text": page_text,
            "body_hash": sha256_text(page_text),
            "page_body_hash": sha256_text(page_text),
        }
        hit = {
            "chunk_id": "c-000001:0",
            "body_hash": chunk["body_hash"],
            "page_body_hash": chunk["page_body_hash"],
        }
        assert_true(
            "CRLF page chunk is current",
            retrieve.chunk_is_current(hit, chunk, page, {}),
        )


# ─── M8 closure: --explain and --no-rerank flag coverage ─────────────────────
def test_explain_flag_adds_diagnostics_block():
    """v1.7.2 / closes audit M8: --explain must include an 'explain' diagnostics block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        page_dir = sandbox / "wiki" / "fake"
        page_dir.mkdir(parents=True)
        # 2 synthetic chunks
        (chunks_dir / "c-000010").mkdir()
        page_text = "hybrid retrieval pipeline"
        (page_dir / "c-000010.md").write_text(page_text)
        (chunks_dir / "c-000010" / "chunk-000.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "page_path": "wiki/fake/c-000010.md",
                    "page_address": "c-000010",
                    "chunk_index": 0,
                    "raw_text": "hybrid retrieval pipeline",
                    "contextualized_text": "hybrid retrieval pipeline",
                    "prefix": "",
                    "prefix_source": "synthetic",
                    "char_count": 25,
                    "body_hash": sha256_text(page_text),
                    "page_body_hash": sha256_text(page_text),
                    "created_at": "2026-05-17T00:00:00Z",
                }
            )
        )
        # Build index
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bm25-index.py"),
                "--vault",
                str(sandbox),
                "build",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            timeout=10,
            check=True,
        )
        # Run with --explain --no-rerank
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "hybrid",
                "--top",
                "1",
                "--no-rerank",
                "--explain",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("retrieve --explain --no-rerank rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_true(
            "--explain produces 'explain' key",
            "explain" in out,
            hint=f"keys={list(out.keys())}",
        )
        explain = out.get("explain", {})
        assert_true(
            "--explain reports BM25 candidate count",
            "bm25_candidates" in explain or "bm25" in str(explain).lower(),
            hint=f"explain={explain}",
        )


def test_no_rerank_flag_strategy_bm25_only():
    """v1.7.2 / closes audit M8: --no-rerank must produce strategy='bm25-only'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        meta = sandbox / ".vault-meta"
        chunks_dir = meta / "chunks"
        bm25_dir = meta / "bm25"
        chunks_dir.mkdir(parents=True)
        bm25_dir.mkdir(parents=True)
        page_dir = sandbox / "wiki" / "fake"
        page_dir.mkdir(parents=True)
        (chunks_dir / "c-000020").mkdir()
        page_text = "transport detection fallback chain"
        (page_dir / "c-000020.md").write_text(page_text)
        (chunks_dir / "c-000020" / "chunk-000.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "page_path": "wiki/fake/c-000020.md",
                    "page_address": "c-000020",
                    "chunk_index": 0,
                    "raw_text": "transport detection fallback chain",
                    "contextualized_text": "transport detection fallback chain",
                    "prefix": "",
                    "prefix_source": "synthetic",
                    "char_count": 35,
                    "body_hash": sha256_text(page_text),
                    "page_body_hash": sha256_text(page_text),
                    "created_at": "2026-05-17T00:00:00Z",
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "bm25-index.py"),
                "--vault",
                str(sandbox),
                "build",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            timeout=10,
            check=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrieve.py"),
                "--vault",
                str(sandbox),
                "transport",
                "--top",
                "1",
                "--no-rerank",
            ],
            env=sandbox_env(sandbox),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert_eq("retrieve --no-rerank rc=0", 0, result.returncode)
        out = json.loads(result.stdout)
        assert_eq(
            "--no-rerank sets strategy='bm25-only'", "bm25-only", out.get("strategy")
        )
        # --no-rerank produces a consistent shape: rerank fields are populated
        # but rerank_source is "skipped" (so callers don't have to special-case).
        candidates = out.get("candidates", [])
        assert_true("--no-rerank still returns candidates", len(candidates) >= 1)
        first = candidates[0]
        assert_eq(
            "--no-rerank candidate rerank_source='skipped'",
            "skipped",
            first.get("rerank_source"),
        )
        assert_eq(
            "--no-rerank candidate rerank_score equals bm25_score",
            first.get("bm25_score"),
            first.get("rerank_score"),
        )


def test_json_conversion_failures_preserve_rerank_fallbacks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chunks = root / ".vault-meta" / "chunks"
        chunks.mkdir(parents=True)
        chunk_path = chunks / "bad.json"
        cache_path = root / ".vault-meta" / "embed-cache.json"
        chunk_path.write_text("{}", encoding="utf-8")
        cache_path.write_text("{}", encoding="utf-8")
        for failure in (ValueError("integer conversion limit"), RecursionError("nesting limit")):
            with (
                unittest.mock.patch.object(rerank, "VAULT_ROOT", root),
                unittest.mock.patch.object(rerank, "META_DIR", root / ".vault-meta"),
                unittest.mock.patch.object(rerank, "EMBED_CACHE_PATH", cache_path),
                unittest.mock.patch.object(rerank.json, "loads", side_effect=failure),
            ):
                assert_eq("JSON conversion failure makes rerank cache a miss", {}, rerank.load_cache())
                assert_eq("JSON conversion failure makes chunk unavailable", None, rerank.load_chunk(".vault-meta/chunks/bad.json"))
            with (
                unittest.mock.patch.object(rerank, "configure_vault", return_value=True),
                unittest.mock.patch.object(sys, "stdin", io.StringIO("[]")),
                unittest.mock.patch.object(rerank.json, "loads", side_effect=failure),
                unittest.mock.patch.object(rerank, "ollama_alive", side_effect=AssertionError("provider must not run")),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                assert_eq("JSON conversion failure has controlled candidate exit", rerank.EXIT_CANDIDATES, rerank.main(["query", "--candidates", "-"]))


def test_retrieve_keeps_healthy_hits_when_one_chunk_is_corrupt():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "wiki/concepts").mkdir(parents=True)
        first = None
        for number, name in enumerate(("alpha", "beta"), 1):
            address = f"c-{number:06d}"
            text = "verification harbor guide " + name + " \ufffd"
            page = f"wiki/concepts/{name}.md"
            (root / page).write_text(text, encoding="utf-8")
            chunk = {
                "schema_version": 1, "page_path": page, "page_address": address,
                "chunk_index": 0, "raw_text": text, "contextualized_text": text,
                "prefix": "", "prefix_source": "synthetic", "char_count": len(text),
                "body_hash": sha256_text(text), "page_body_hash": sha256_text(text),
                "created_at": "2026-10-18T00:00:00Z",
            }
            folder = root / ".vault-meta/chunks" / address
            folder.mkdir(parents=True)
            (folder / "chunk-000.json").write_text(json.dumps(chunk), encoding="utf-8")
            if first is None:
                first = chunk
        env = sandbox_env(root)
        build = subprocess.run([sys.executable, str(BM25), "--vault", str(root), "build"], env=env, capture_output=True, text=True, timeout=10)
        assert_eq("corruption fixture index build", 0, build.returncode)
        command = [sys.executable, str(RETRIEVE), "--vault", str(root), "verification harbor", "--no-rerank", "--explain"]
        healthy = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        assert_eq("healthy corruption fixture retrieval", 0, healthy.returncode)
        assert_eq("both fixture pages are actually indexed", 2, len(json.loads(healthy.stdout)["candidates"]))
        variants = {
            "invalid UTF-8": b"\xff", "null": b"null", "array": b"[]",
            "integer conversion limit": ('{"x":' + "1" * 5000 + "}").encode(),
            "surrogate text": json.dumps(dict(first, raw_text="\ud800")).encode(),
            "invalid UTF-8 page": json.dumps(first).encode(),
        }
        for name, payload in variants.items():
            target = root / ".vault-meta/chunks/c-000001/chunk-000.json"
            target.write_bytes(payload)
            page = root / first["page_path"]
            page_bytes = first["raw_text"].encode("utf-8")
            if name == "invalid UTF-8 page":
                page_bytes = page_bytes.replace(b"\xef\xbf\xbd", b"\xff")
            page.write_bytes(page_bytes)
            before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
            assert_eq(f"{name} retrieval remains available", 0, result.returncode)
            assert_true(f"{name} emits no traceback", "Traceback" not in result.stderr, result.stderr)
            value = json.loads(result.stdout)
            assert_eq(f"{name} preserves healthy result", ["wiki/concepts/beta.md"], [c["page_path"] for c in value["candidates"]])
            assert_eq(f"{name} accounts for skipped chunk", 1, value["explain"]["stale_candidates_skipped"])
            after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            assert_eq(f"{name} keeps vault bytes unchanged", before, after)


FAKE_MODEL = "m1:latest"


def _rerank_fixture(sandbox):
    """One indexed synthetic chunk; retrieval itself never needs a provider."""
    chunks_dir = sandbox / ".vault-meta" / "chunks" / "c-000030"
    chunks_dir.mkdir(parents=True)
    page_dir = sandbox / "wiki" / "fake"
    page_dir.mkdir(parents=True)
    page_text = "transport detection fallback chain"
    (page_dir / "c-000030.md").write_text(page_text)
    (chunks_dir / "chunk-000.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "page_path": "wiki/fake/c-000030.md",
                "page_address": "c-000030",
                "chunk_index": 0,
                "raw_text": page_text,
                "contextualized_text": page_text,
                "prefix": "",
                "prefix_source": "synthetic",
                "char_count": len(page_text),
                "body_hash": sha256_text(page_text),
                "page_body_hash": sha256_text(page_text),
                "created_at": "2026-05-17T00:00:00Z",
            }
        )
    )
    subprocess.run(
        [sys.executable, str(BM25), "--vault", str(sandbox), "build"],
        env=sandbox_env(sandbox),
        capture_output=True,
        timeout=10,
        check=True,
    )


def _retrieve_with_fake(sandbox, server_url):
    env = sandbox_env(sandbox)
    env["OLLAMA_URL"] = server_url
    return subprocess.run(
        [sys.executable, str(RETRIEVE), "--vault", str(sandbox), "transport",
         "--top", "1", "--model", FAKE_MODEL],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_retrieve_falls_back_when_ollama_port_returns_garbage():
    for mode, strategy in (
        ("badstatus", "bm25+rerank:noop-no-ollama"),
        ("badproof", "bm25+rerank:noop-local-proof"),
    ):
        with tempfile.TemporaryDirectory() as tmpdir, FakeOllamaServer(mode, FAKE_MODEL) as server:
            sandbox = Path(tmpdir)
            _rerank_fixture(sandbox)
            result = _retrieve_with_fake(sandbox, server.url)
            posts = server.count("POST", "/api/embed")
        assert_eq(f"{mode}: retrieve exits 0 instead of a traceback", 0, result.returncode)
        assert_true(f"{mode}: no traceback", "Traceback" not in result.stderr, result.stderr[-400:])
        out = json.loads(result.stdout)
        assert_eq(f"{mode}: BM25 fallback strategy", strategy, out["strategy"])
        assert_true(f"{mode}: BM25 candidates survive", len(out["candidates"]) == 1)
        assert_eq(f"{mode}: no query or chunk text is embedded", 0, posts)


def test_rerank_duplicate_remote_status_keys_send_no_query_text():
    candidates = [{"chunk_id": "c-1:0", "path": ".vault-meta/chunks/c-1/chunk-000.json", "bm25_score": 2.0}]
    with FakeOllamaServer("dupkey", FAKE_MODEL) as server:
        with unittest.mock.patch.dict(os.environ, {"OLLAMA_URL": server.url}):
            result = rerank.rerank("private query", candidates, top_k=1, model=FAKE_MODEL)
        posts = server.count("POST", "/api/embed")
        shows = server.count("POST", "/api/show")
    assert_eq("duplicate remote key is a local-proof refusal", "noop-local-proof", result[0]["rerank_source"])
    assert_eq("duplicate remote key sends no query text", 0, posts)
    assert_eq("duplicate remote key stops before model metadata", 0, shows)


def test_rerank_unrelated_cloud_model_still_reranks_locally():
    with tempfile.TemporaryDirectory() as tmpdir, FakeOllamaServer("othercloud", FAKE_MODEL) as server:
        sandbox = Path(tmpdir)
        _rerank_fixture(sandbox)
        result = _retrieve_with_fake(sandbox, server.url)
        posts = server.count("POST", "/api/embed")
    assert_eq("unrelated cloud model retrieve rc=0", 0, result.returncode)
    out = json.loads(result.stdout)
    assert_eq("unrelated cloud model keeps cosine rerank", f"bm25+rerank:cosine:{FAKE_MODEL}", out["strategy"])
    assert_eq("query and one chunk are embedded locally", 2, posts)
    with tempfile.TemporaryDirectory() as tmpdir, FakeOllamaServer("selfcloud", FAKE_MODEL) as server:
        sandbox = Path(tmpdir)
        _rerank_fixture(sandbox)
        result = _retrieve_with_fake(sandbox, server.url)
        posts = server.count("POST", "/api/embed")
    out = json.loads(result.stdout)
    assert_eq("requested remote model still refuses rerank", "bm25+rerank:noop-local-proof", out["strategy"])
    assert_eq("requested remote model receives no text", 0, posts)


def test_rerank_request_deadline_bounds_slow_drip_transport():
    import time

    with SlowDripServer(phase="body", interval=0.2) as server:
        request = rerank.urllib.request.Request(server.url + "/api/tags", method="GET")
        started = time.monotonic()
        try:
            rerank._ollama_json(request, 1)
        except OSError:
            pass
        else:
            raise Fail("FAIL slow-drip rerank response accepted past the request deadline")
        elapsed = time.monotonic() - started
    assert_true(f"slow-drip rerank request stops near its deadline ({elapsed:.1f}s)", elapsed < 3.0)


def main():
    print("=== test_retrieve.py ===")
    test_import_sibling_resolves_hyphenated_names()
    test_chunk_snippet_short()
    test_chunk_snippet_truncates_with_ellipsis()
    test_cosine_identical()
    test_cosine_orthogonal()
    test_cosine_anti_parallel()
    test_cosine_length_mismatch()
    test_cosine_zero_vector()
    test_multilingual_model_default_and_validation()
    test_model_availability_handles_exact_and_tagged_names()
    test_retrieval_top_arguments_are_positive_and_bounded()
    test_embed_one_uses_current_single_input_ollama_api()
    test_rerank_transport_disables_proxies_and_rejects_redirects()
    test_rerank_input_validation_precedes_model_provider_and_cache()
    test_rerank_invalid_utf8_cache_is_a_miss()
    test_rerank_malformed_candidate_and_chunk_files_are_controlled()
    test_rerank_noop_when_ollama_unreachable()
    test_rerank_noop_when_model_missing()
    test_rerank_local_proof_refusal_precedes_cache_and_payload()
    test_rerank_truncates_to_top_k()
    test_rerank_query_embed_failure_preserves_bm25()
    test_rerank_document_failure_preserves_candidate_bm25_score()
    test_rerank_missing_chunk_preserves_candidate_bm25_score()
    test_with_task_prefix_contract()
    test_rerank_sends_nomic_task_prefixes()
    test_selected_model_is_used_for_embeddings_cache_and_source()
    test_peek_reports_selected_tagged_model()
    test_embedding_cache_key_versions_model_and_scheme()
    test_stale_prefixless_cache_is_not_reused()
    test_verified_digest_change_misses_rerank_cache()
    test_remote_ollama_remains_explicit_opt_in()
    test_rerank_localhost_transport_uses_numeric_loopback()
    test_rerank_rejects_chunk_path_escape()
    test_retrieve_exits_10_without_index()
    test_retrieve_validates_and_forwards_explicit_model()
    test_retrieve_translates_corrupt_index_to_documented_fallback_exit()
    test_end_to_end_with_synthetic_chunks()
    test_dedupe_happens_before_final_top_k()
    test_retrieve_skips_page_deleted_after_index_build()
    test_chunk_currency_rejects_hashless_or_malformed_legacy_records()
    test_chunk_currency_accepts_crlf_pages()
    test_explain_flag_adds_diagnostics_block()
    test_no_rerank_flag_strategy_bm25_only()
    test_json_conversion_failures_preserve_rerank_fallbacks()
    test_retrieve_keeps_healthy_hits_when_one_chunk_is_corrupt()
    test_retrieve_falls_back_when_ollama_port_returns_garbage()
    test_rerank_duplicate_remote_status_keys_send_no_query_text()
    test_rerank_unrelated_cloud_model_still_reranks_locally()
    test_rerank_request_deadline_bounds_slow_drip_transport()
    print("\nAll retrieve tests passed.")


if __name__ == "__main__":
    main()
