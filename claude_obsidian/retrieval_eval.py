"""Read-only, offline page-retrieval evaluation against explicit local labels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from .json_utils import strict_json_loads
from .paths import is_name_surrogate, linked_path_hint, supports_confined_dirfd


QUERY_SCHEMA = "claude-obsidian.retrieval-queries.v1"
REPORT_SCHEMA = "claude-obsidian.retrieval-evaluation.v1"
MAX_QUERY_FILE_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_QUERIES = 1000
MAX_GOLD_PAGES_PER_QUERY = 64
MAX_QUERY_CHARS = 8000
INDEX_PATH = ".vault-meta/bm25/index.json"


class EvaluationError(ValueError):
    """Invalid labels, unsafe paths, or an incomparable evaluation snapshot."""


class IndexNotReadyError(EvaluationError):
    """Derived retrieval state is unavailable; the command should exit 10."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path, *, limit: int, metadata_only: bool = False) -> bytes:
    """Open bounded regular data without following a link in any component."""

    path = Path(os.path.abspath(path))
    directory_fd = -1
    descriptor = -1
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        if supports_confined_dirfd():
            directory_fd = os.open(path.anchor, directory_flags)
            for part in path.parts[1:-1]:
                child = os.open(part, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = child
            descriptor = os.open(path.name, file_flags, dir_fd=directory_fd)
        else:
            cursor = Path(path.anchor)
            for part in path.parts[1:]:
                cursor = cursor / part
                if is_name_surrogate(cursor.lstat()):
                    raise EvaluationError(f"symlinks and junctions are not allowed: {path}")
            descriptor = os.open(path, file_flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationError(f"expected a regular file: {path}")
        if metadata_only:
            return b""
        if before.st_size > limit:
            raise EvaluationError(f"file has {before.st_size} bytes; limit is {limit}: {path}")
        chunks = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > limit:
                raise EvaluationError(f"file grew beyond the {limit}-byte limit: {path}")
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise EvaluationError(f"file changed during evaluation read: {path}")
        return b"".join(chunks)
    except FileNotFoundError:
        raise
    except OSError as exc:
        hint = linked_path_hint(path)
        if hint:
            raise EvaluationError(f"cannot safely read regular file {path}: {hint}") from exc
        raise EvaluationError(f"cannot safely read regular file {path}: {exc.strerror}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)


def _page_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("wiki/") or not value.endswith(".md"):
        raise EvaluationError("expected_pages and returned pages must be wiki-relative .md paths")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF or character == "\\" for character in value)
    ):
        raise EvaluationError(f"page path is not a canonical vault-relative path: {value}")
    return value


def load_queries(query_file: Path | str) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = _read_regular(Path(query_file), limit=MAX_QUERY_FILE_BYTES)
        document = strict_json_loads(raw.decode("utf-8"))
    except (FileNotFoundError, UnicodeError, ValueError, RecursionError) as exc:
        raise EvaluationError(f"cannot read query labels: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != QUERY_SCHEMA:
        raise EvaluationError(f"query labels must declare schema={QUERY_SCHEMA}")
    queries = document.get("queries")
    if not isinstance(queries, list) or len(queries) > MAX_QUERIES:
        raise EvaluationError(f"queries must be an array with at most {MAX_QUERIES} entries")
    seen: set[str] = set()
    for number, item in enumerate(queries):
        if not isinstance(item, dict):
            raise EvaluationError(f"queries[{number}] must be an object")
        identity = item.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", identity) or identity in seen:
            raise EvaluationError(f"queries[{number}].id must be unique portable text of 1 to 128 characters")
        seen.add(identity)
        query = item.get("query")
        if (
            not isinstance(query, str) or not query.strip()
            or len(unicodedata.normalize("NFKC", query)) > MAX_QUERY_CHARS
            or any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in query)
        ):
            raise EvaluationError(f"queries[{number}].query must be non-empty text within {MAX_QUERY_CHARS} normalized characters")
        expected = item.get("expected_pages")
        if not isinstance(expected, list) or len(expected) > MAX_GOLD_PAGES_PER_QUERY:
            raise EvaluationError(f"queries[{number}].expected_pages must be an explicit array of at most {MAX_GOLD_PAGES_PER_QUERY} page paths")
        normalized = [_page_path(path) for path in expected]
        if len(set(normalized)) != len(normalized):
            raise EvaluationError(f"queries[{number}].expected_pages contains duplicates")
        if "skip_reason" in item and (not isinstance(item["skip_reason"], str) or not item["skip_reason"].strip()):
            raise EvaluationError(f"queries[{number}].skip_reason must be non-empty text")
        if "category" in item and not isinstance(item["category"], str):
            raise EvaluationError(f"queries[{number}].category must be text")
    return queries, _sha256(raw)


def score_ranked_pages(ranked: Sequence[str], expected: Sequence[str], *, mrr_depth: int = 10) -> dict[str, Any]:
    """Macro recall input and reciprocal rank, with empty gold kept separate."""

    unique = list(dict.fromkeys(ranked))
    gold = set(expected)
    if not gold:
        return {"recall_at_1": None, "recall_at_5": None, "recall_at_10": None,
                "reciprocal_rank": None, "first_relevant_rank": None,
                "unanswerable_correct": not unique}
    first = next((rank for rank, page in enumerate(unique[:mrr_depth], 1) if page in gold), None)
    return {
        **{f"recall_at_{cutoff}": len(set(unique[:cutoff]) & gold) / len(gold) for cutoff in (1, 5, 10)},
        "reciprocal_rank": 1 / first if first is not None else 0.0,
        "first_relevant_rank": first,
        "unanswerable_correct": None,
    }


def _mean(values: Iterable[float]) -> dict[str, Any]:
    selected = list(values)
    return {"value": sum(selected) / len(selected) if selected else None,
            "n": len(selected), "status": "measured" if selected else "no data"}


def _index_snapshot(vault: Path, product_root: Path) -> bytes:
    try:
        raw = _read_regular(vault / INDEX_PATH, limit=MAX_INDEX_BYTES)
        document = strict_json_loads(raw.decode("utf-8"))
    except (FileNotFoundError, UnicodeError, ValueError, RecursionError) as exc:
        raise IndexNotReadyError(f"BM25 index is missing, unsafe, oversized, or corrupt; rebuild derived caches explicitly: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("docs"), dict):
        raise IndexNotReadyError("BM25 index has no docs mapping; rebuild derived caches explicitly")
    helper = product_root / "scripts/bm25-index.py"
    try:
        spec = importlib.util.spec_from_file_location("claude_obsidian_eval_bm25", helper)
        if spec is None or spec.loader is None:
            raise EvaluationError("BM25 validation helper cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_index(document)
    except (OSError, ValueError, ImportError, SyntaxError, AttributeError) as exc:
        raise IndexNotReadyError(f"BM25 index or validation helper is unavailable: {exc}") from exc
    return raw


def _execute_query(vault: Path, query: str, product_root: Path, depth: int) -> tuple[dict[str, Any], float]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(product_root / "scripts/retrieve.py"), "--vault", str(vault),
               "--top", str(depth), "--bm25-top", str(depth), "--no-rerank", "--explain", "--", query]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=product_root, env=env, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationError(f"offline retrieval could not finish within its process budget: {exc}") from exc
    elapsed = (time.perf_counter() - started) * 1000
    if completed.returncode == 10:
        raise IndexNotReadyError("retrieval index is not ready: " + completed.stderr[-2000:].strip())
    if completed.returncode:
        raise EvaluationError(f"offline retrieval exited {completed.returncode}: {completed.stderr[-2000:].strip()}")
    if len(completed.stdout.encode("utf-8")) > 4 * 1024 * 1024:
        raise EvaluationError("offline retrieval response exceeds its 4 MiB evaluation bound")
    try:
        result = strict_json_loads(completed.stdout)
    except (ValueError, RecursionError) as exc:
        raise EvaluationError("offline retrieval returned malformed JSON") from exc
    if not isinstance(result, dict) or result.get("strategy") != "bm25-only" or not isinstance(result.get("candidates"), list):
        raise EvaluationError("offline evaluation requires an explicit bm25-only result with candidates")
    return result, elapsed


def run_quality(vault: Path | str, query_file: Path | str, product_root: Path | str, top_k: int = 10) -> dict[str, Any]:
    """Evaluate a prebuilt index without provisioning, reranking, or writes."""

    if type(top_k) is not int or not 1 <= top_k <= 1000:
        raise EvaluationError("top_k must be an integer between 1 and 1000")
    root = Path(vault).expanduser().resolve()
    product = Path(product_root).expanduser().resolve()
    queries, labels_hash = load_queries(query_file)
    index_bytes = _index_snapshot(root, product)
    index_hash = _sha256(index_bytes)
    ranking_depth = max(10, top_k)
    pool_depth = max(20, ranking_depth)
    missing_gold: set[str] = set()
    gold_exists: dict[str, bool] = {}
    for item in queries:
        for path in item["expected_pages"]:
            if path in gold_exists:
                continue
            try:
                _read_regular(root / path, limit=1, metadata_only=True)
            except FileNotFoundError:
                gold_exists[path] = False
                missing_gold.add(path)
            else:
                gold_exists[path] = True
    per_query = []
    for item in queries:
        missing = [path for path in item["expected_pages"] if not gold_exists[path]]
        row: dict[str, Any] = {"id": item["id"], "query": item["query"],
                               "category": item.get("category"), "expected_pages": item["expected_pages"]}
        if item.get("skip_reason") or missing:
            row.update(status="skipped", skip_reason=item.get("skip_reason", "missing gold pages"), missing_gold_pages=missing)
            per_query.append(row)
            continue
        result, latency = _execute_query(root, item["query"], product, pool_depth)
        paths = []
        for candidate in result["candidates"]:
            if not isinstance(candidate, dict):
                raise EvaluationError("retrieval candidates must be objects")
            path = _page_path(candidate.get("page_path"))
            try:
                _read_regular(root / path, limit=1, metadata_only=True)
            except FileNotFoundError as exc:
                raise EvaluationError("a returned page disappeared during evaluation; rerun on a stable vault") from exc
            if path not in paths:
                paths.append(path)
        measured = paths[:ranking_depth]
        metrics = score_ranked_pages(measured, item["expected_pages"], mrr_depth=ranking_depth)
        gold = set(item["expected_pages"])
        row.update(status="scored", ranked_pages=measured, usable_candidate_pages=paths,
                   missing_expected_pages=[path for path in item["expected_pages"] if path not in measured],
                   missing_at_cutoff={str(cutoff): [path for path in item["expected_pages"] if path not in measured[:cutoff]] for cutoff in (1, 5, 10)},
                   missing_from_usable_candidates=[path for path in item["expected_pages"] if path not in paths],
                   latency_ms=latency, metrics=metrics,
                   usable_candidate_recall=len(set(paths) & gold) / len(gold) if gold else None,
                   explain=result.get("explain", {}))
        per_query.append(row)
    if _sha256(_index_snapshot(root, product)) != index_hash:
        raise EvaluationError("BM25 index changed during evaluation; rerun against one stable snapshot")
    if load_queries(query_file)[1] != labels_hash:
        raise EvaluationError("query labels changed during evaluation; results were not accepted")
    scored = [row for row in per_query if row["status"] == "scored"]
    answerable = [row for row in scored if row["expected_pages"]]
    unanswerable = [row for row in scored if not row["expected_pages"]]
    latencies = sorted(row["latency_ms"] for row in scored)
    return {
        "schema": REPORT_SCHEMA, "evidence_class": "offline-local-prebuilt-index",
        "strategy": "bm25-only", "query_set_sha256": labels_hash, "index_snapshot_sha256": index_hash,
        "query_count": len(queries), "scored_query_count": len(scored),
        "scored_answerable_query_count": len(answerable), "scored_unanswerable_query_count": len(unanswerable),
        "skipped_query_count": len(queries) - len(scored),
        "missing_gold_query_count": sum(bool(row.get("missing_gold_pages")) for row in per_query),
        "missing_gold_page_count": len(missing_gold), "missing_gold_pages": sorted(missing_gold),
        "configured_top_k": top_k, "ranking_depth": ranking_depth, "candidate_pool_depth": pool_depth,
        "metric_definition": "macro recall over scored non-empty gold queries; MRR over ranking_depth; empty gold scored separately for no-result correctness",
        "metrics": {
            **{f"recall_at_{cutoff}": _mean(row["metrics"][f"recall_at_{cutoff}"] for row in answerable) for cutoff in (1, 5, 10)},
            "mrr": {**_mean(row["metrics"]["reciprocal_rank"] for row in answerable), "cutoff": ranking_depth},
            "usable_candidate_recall": {**_mean(row["usable_candidate_recall"] for row in answerable), "pool_depth": pool_depth,
                                        "scope": "after stale filtering and page deduplication; raw BM25 candidate recall is not exposed"},
            "unanswerable_no_result_accuracy": _mean(float(row["metrics"]["unanswerable_correct"]) for row in unanswerable),
        },
        "latency_ms": {"n": len(latencies), "includes": "retrieval process startup, index load, and query",
                       "p50": latencies[(len(latencies) - 1) // 2] if latencies else None,
                       "p95": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
                       "status": "measured" if latencies else "no data"},
        "stale_candidates_skipped": sum(row.get("explain", {}).get("stale_candidates_skipped", 0) for row in scored),
        "per_query": per_query,
    }


def provision_fixture(destination: Path | str, product_root: Path | str) -> Path:
    """Provision only an absent, explicitly selected fixture directory."""

    root = Path(destination)
    product = Path(product_root).resolve()
    fixture = product / "tests/fixtures/retrieval-quality"
    if root.exists() or root.is_symlink():
        raise EvaluationError("fixture destination must be absent; an existing vault is never overwritten")
    if not (fixture / "pages").is_dir():
        raise EvaluationError("original retrieval fixture is unavailable in this distribution; use a source checkout or supply your own read-only labels")
    root.mkdir(parents=True)
    (root / ".obsidian").mkdir()
    (root / ".raw").mkdir()
    (root / "wiki").mkdir()
    for source in sorted((fixture / "pages").rglob("*.md")):
        target = root / "wiki" / source.relative_to(fixture / "pages")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_read_regular(source, limit=1024 * 1024))
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script, arguments in (
        ("contextual-prefix.py", ["--all", "--no-llm"]),
        ("bm25-index.py", ["build"]),
    ):
        completed = subprocess.run(
            [sys.executable, str(product / "scripts" / script), "--vault", str(root), *arguments],
            cwd=product, env=environment, capture_output=True, text=True, timeout=30,
        )
        if completed.returncode:
            raise EvaluationError(f"fixture provisioning failed in {script}: {completed.stderr[-2000:]}")
    for source in sorted((fixture / "post-index").rglob("*.md")):
        target = root / "wiki" / source.relative_to(fixture / "post-index")
        if not target.is_file() or target.is_symlink():
            raise EvaluationError("post-index fixture replacement must name an existing regular fixture page")
        target.write_bytes(_read_regular(source, limit=1024 * 1024))
    return fixture / "queries.json"


def run_fixture(product_root: Path | str) -> dict[str, Any]:
    """Measure the original frozen fixture inside a disposable vault only."""

    product = Path(product_root).resolve()
    fixture = product / "tests/fixtures/retrieval-quality"
    with tempfile.TemporaryDirectory(prefix="claude-obsidian-quality-") as temporary:
        vault = Path(temporary) / "fixture-vault"
        labels = provision_fixture(vault, product)
        report = run_quality(vault, labels, product)
        report["evidence_class"] = "offline-original-synthetic-fixture"
        files = {path.relative_to(fixture).as_posix(): _sha256(_read_regular(path, limit=1024 * 1024))
                 for path in sorted(fixture.rglob("*")) if path.is_file()}
        report["fixture_content_sha256"] = _sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        report["fixture_files"] = files
        report["prefix_strategy"] = "synthetic --no-llm"
        return report
