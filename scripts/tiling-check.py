#!/usr/bin/env python3
"""tiling-check.py — DragonScale Mechanism 3: semantic tiling lint.

Computes per-page embeddings via a local ollama instance and reports
candidate duplicate page pairs. Read-only; never modifies wiki pages.

Security model:
- Defaults to http://127.0.0.1:11434. Remote ollama endpoints require
  --allow-remote-ollama explicitly (vault bodies are POSTed as embedding
  input; a hostile env var would otherwise exfiltrate content).
- Rejects symlinked page files to prevent escape outside the vault root.

Feature-gated: exits 10 if ollama is unreachable or 11 if the embedding
model is not pulled, so the calling skill can no-op gracefully. Exits 0
on success. Exit 3 on cache corruption, 12 on unproved or incomplete local
embedding work, and 2 on usage error.

Concurrency:
- Embedding work runs without holding the vault-wide mutation lock.
- The derived cache is published in one short canonical mutation through
  pinned vault-root and `.vault-meta` descriptors.

Usage:
  tiling-check.py --vault PATH           # select a vault explicitly
  tiling-check.py                      # run; exit 10/11 if ollama/model missing
  tiling-check.py                      # report is stdout-only
  tiling-check.py --rebuild-cache      # ignore cached embeddings
  tiling-check.py --peek               # structured diagnostics; no compute
  tiling-check.py --allow-remote-ollama # accept non-localhost OLLAMA_URL
"""

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT_SEC = 3
EMBED_TIMEOUT_SEC = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MB; embeddings can be ~10 KB each
MAX_EMBEDDING_DIMENSIONS = 65_536

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from claude_obsidian.paths import VaultSelectionError, resolve_vault_root
from claude_obsidian.ollama import (
    NoRedirectHandler,
    OllamaLocalProofError,
    OllamaModelUnavailable,
    normalize_model_digest,
    open_direct,
    prove_local_model,
    read_json_object,
    validate_model_response,
)
from claude_obsidian.transaction import (
    MutationLock,
    TransactionError,
    _atomic_vault_write,
    _safe_vault_path,
)

VAULT_ROOT = Path.cwd().resolve()
WIKI_DIR = VAULT_ROOT / "wiki"
META_DIR = VAULT_ROOT / ".vault-meta"
CACHE_PATH = META_DIR / "tiling-cache.json"
CACHE_LOCK = META_DIR / ".tiling.lock"
THRESHOLDS_PATH = META_DIR / "tiling-thresholds.json"

EXCLUDE_TYPES = {"meta", "fold"}
EXCLUDE_FILENAMES = {
    "_index.md", "index.md", "log.md", "hot.md", "overview.md",
    "dashboard.md", "Wiki Map.md", "getting-started.md",
}
EXCLUDE_PATH_PREFIXES = ("wiki/folds/", "wiki/meta/")
MAX_BODY_BYTES = 128 * 1024
SCALE_WARN_PAGES = 500
SCALE_HARD_FAIL_PAGES = 5000

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CACHE_CORRUPT = 3
EXIT_SCALE_EXCEEDED = 4
EXIT_NO_OLLAMA = 10
EXIT_NO_MODEL = 11
EXIT_INCOMPLETE = 12

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TYPE_RE = re.compile(r"^type:\s*(\S+)", re.MULTILINE)


def configure_vault(explicit=None):
    global VAULT_ROOT, WIKI_DIR, META_DIR, CACHE_PATH, CACHE_LOCK, THRESHOLDS_PATH
    try:
        selected = resolve_vault_root(
            explicit,
            start=Path.cwd(),
            plugin_root=PLUGIN_ROOT,
        )
    except VaultSelectionError as exc:
        log(f"ERR {exc.code}: {exc}")
        return False
    VAULT_ROOT = selected.root
    WIKI_DIR = VAULT_ROOT / "wiki"
    META_DIR = VAULT_ROOT / ".vault-meta"
    CACHE_PATH = META_DIR / "tiling-cache.json"
    CACHE_LOCK = META_DIR / ".tiling.lock"
    THRESHOLDS_PATH = META_DIR / "tiling-thresholds.json"
    try:
        for relative in (
            "wiki",
            ".vault-meta",
            ".vault-meta/tiling-cache.json",
            ".vault-meta/.tiling.lock",
            ".vault-meta/tiling-thresholds.json",
        ):
            _safe_vault_path(VAULT_ROOT, relative)
    except (VaultSelectionError, TransactionError) as exc:
        log(f"ERR {exc.code}: {exc}")
        return False
    return True


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _is_local_url(url: str) -> bool:
    parsed = _validate_ollama_base_url(url)
    return parsed is not None and (parsed.hostname or "") in (
        "127.0.0.1",
        "localhost",
        "::1",
    )


def _validate_ollama_base_url(url: str) -> urllib.parse.ParseResult | None:
    try:
        parsed = urllib.parse.urlparse(url)
        _ = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed


_NoOllamaRedirect = NoRedirectHandler

# Transport failures that mean "no usable local provider", never a traceback.
OLLAMA_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    OSError,
    ValueError,
    UnicodeError,
    TimeoutError,
    RuntimeError,
)


def _open_ollama_request(request: urllib.request.Request, timeout: float):
    """Open an Ollama request without proxies or redirects, within one deadline."""
    return open_direct(request, timeout)


def _ollama_json(request: urllib.request.Request, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    with _open_ollama_request(request, timeout) as response:
        return read_json_object(
            response,
            request,
            limit=MAX_RESPONSE_BYTES,
            deadline=deadline,
            label="Ollama",
        )


def _http_get_json(url: str, timeout: float) -> dict:
    if _validate_ollama_base_url(url.rsplit("/api/", 1)[0]) is None:
        raise ValueError("invalid Ollama HTTP(S) URL")
    request = urllib.request.Request(url, method="GET")
    return _ollama_json(request, timeout)


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    if _validate_ollama_base_url(url.rsplit("/api/", 1)[0]) is None:
        raise ValueError("invalid Ollama HTTP(S) URL")
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _ollama_json(request, timeout)


def detect_ollama(url: str) -> bool:
    try:
        _http_get_json(f"{url}/api/version", OLLAMA_TIMEOUT_SEC)
        return True
    except OLLAMA_TRANSPORT_ERRORS:
        return False


def detect_model(url: str, model: str) -> bool:
    try:
        data = _http_get_json(f"{url}/api/tags", OLLAMA_TIMEOUT_SEC)
    except OLLAMA_TRANSPORT_ERRORS:
        return False
    models = data.get("models")
    if not isinstance(models, list):
        return True
    aliases = (model,) if ":" in model else (model, f"{model}:latest")
    malformed = False
    for entry in models:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            malformed = True
            continue
        if entry["name"] in aliases:
            return True
    return malformed


def local_model_proof(url: str, model: str) -> dict:
    """Prove cloud-disabled local model state without sending page text."""
    return prove_local_model(
        model,
        lambda endpoint: _http_get_json(
            f"{url}/api/{endpoint}", OLLAMA_TIMEOUT_SEC
        ),
        lambda endpoint, payload: _http_post_json(
            f"{url}/api/{endpoint}", payload, OLLAMA_TIMEOUT_SEC
        ),
    )


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_raw = m.group(1)
    body = text[m.end():]
    fm: dict = {}
    tm = TYPE_RE.search(fm_raw)
    if tm:
        fm["type"] = tm.group(1).strip().strip('"').strip("'")
    return fm, body


def body_hash(body: str, model: str, digest: str | None = None) -> str:
    h = hashlib.sha256()
    h.update(f"model={model}\ndigest={digest or ''}\n".encode("utf-8"))
    h.update(body.encode("utf-8"))
    return h.hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    if not all(finite_number(item) for item in (*a, *b)):
        raise ValueError("vector contains a non-finite or non-numeric value")
    scale_a = max((abs(item) for item in a), default=0.0)
    scale_b = max((abs(item) for item in b), default=0.0)
    if scale_a == 0 or scale_b == 0:
        return 0.0
    scaled_a = [item / scale_a for item in a]
    scaled_b = [item / scale_b for item in b]
    dot = sum(x * y for x, y in zip(scaled_a, scaled_b))
    na = math.sqrt(sum(x * x for x in scaled_a))
    nb = math.sqrt(sum(y * y for y in scaled_b))
    result = dot / (na * nb)
    if not math.isfinite(result):
        raise ValueError("cosine result is not finite")
    return max(-1.0, min(1.0, result))


def finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_embedding_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and 0 < len(value) <= MAX_EMBEDDING_DIMENSIONS
        and all(
            finite_number(item)
            for item in value
        )
        and any(item != 0 for item in value)
    )


def validate_cache_document(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("cache document is not an object")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 2}:
        raise ValueError(f"unknown cache version: {data.get('version')}")
    if not isinstance(data.get("model"), str):
        raise ValueError("cache.model is not a string")
    if version == 2:
        try:
            normalize_model_digest(data.get("digest"))
        except OllamaLocalProofError:
            raise ValueError("cache.digest is not a valid model digest")
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("cache.embeddings is not an object")
    expected_dimensions = None
    for path, entry in embeddings.items():
        if not isinstance(path, str) or not isinstance(entry, dict):
            raise ValueError("cache entry must map a path string to an object")
        if not isinstance(entry.get("hash"), str) or not entry["hash"]:
            raise ValueError(f"cache entry {path!r} has no content hash")
        if not valid_embedding_vector(entry.get("embedding")):
            raise ValueError(f"cache entry {path!r} has an invalid embedding")
        dimensions = len(entry["embedding"])
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError("cache embeddings have inconsistent dimensions")
    return data


def load_cache(current_model: str, current_digest: str) -> dict:
    empty = {
        "version": 2,
        "model": current_model,
        "digest": current_digest,
        "embeddings": {},
    }
    if not CACHE_PATH.exists():
        return empty
    try:
        with CACHE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        log(f"ERR: cache read failed: {exc}")
        sys.exit(EXIT_CACHE_CORRUPT)
    try:
        validate_cache_document(data)
    except ValueError as exc:
        log(f"ERR: cache schema invalid: {exc}")
        sys.exit(EXIT_CACHE_CORRUPT)
    if data["version"] == 1:
        log("INFO: valid legacy tiling cache has no model digest; treating it as a miss")
        return empty
    cached_model = data.get("model", "")
    cached_digest = normalize_model_digest(data.get("digest"))
    if cached_model != current_model or cached_digest != current_digest:
        log("INFO: cached model identity differs from verified model; invalidating cache")
        return empty
    data["digest"] = cached_digest
    return data


def save_cache(cache: dict) -> None:
    """Publish the derived cache without following replaced vault aliases."""

    payload = (json.dumps(cache, indent=2) + "\n").encode("utf-8")
    root_fd = -1
    meta_fd = -1
    try:
        with MutationLock(VAULT_ROOT) as mutation_lock:
            root_fd = mutation_lock.duplicate_root_fd()
            meta_fd = mutation_lock.duplicate_parent_fd()
            _atomic_vault_write(
                VAULT_ROOT,
                ".vault-meta/tiling-cache.json",
                payload,
                mode=0o600,
                root_fd=root_fd,
                meta_fd=meta_fd,
            )
    finally:
        if meta_fd >= 0:
            os.close(meta_fd)
        if root_fd >= 0:
            os.close(root_fd)


class ThresholdValidationError(ValueError):
    pass


def validate_thresholds(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ThresholdValidationError("threshold document must be an object")
    version = value.get("version")
    bands = value.get("bands")
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ThresholdValidationError("threshold version must be integer 1")
    if not isinstance(bands, dict):
        raise ThresholdValidationError("threshold bands must be an object")
    review = bands.get("review")
    error = bands.get("error")
    for name, threshold in (("review", review), ("error", error)):
        if (
            not finite_number(threshold)
        ):
            raise ThresholdValidationError(f"threshold {name} must be finite")
    if not 0 <= review < error <= 1:
        raise ThresholdValidationError(
            "threshold bands must satisfy 0 <= review < error <= 1"
        )
    return value


def load_thresholds() -> dict:
    if not THRESHOLDS_PATH.exists():
        return validate_thresholds({
            "version": 1, "model": DEFAULT_MODEL,
            "bands": {"error": 0.90, "review": 0.80},
            "calibrated": False, "calibration_pairs_labeled": 0,
        })
    try:
        with THRESHOLDS_PATH.open(encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ThresholdValidationError(f"threshold read failed: {exc}") from exc
    return validate_thresholds(value)


def included(path: Path, fm: dict) -> tuple[bool, str]:
    rel = path.relative_to(VAULT_ROOT).as_posix()
    if path.is_symlink():
        return False, "symlink"
    resolved = path.resolve()
    try:
        resolved.relative_to(VAULT_ROOT.resolve())
    except ValueError:
        return False, "escapes vault"
    if path.name in EXCLUDE_FILENAMES:
        return False, "excluded filename"
    for prefix in EXCLUDE_PATH_PREFIXES:
        if rel.startswith(prefix):
            return False, f"under {prefix}"
    if fm.get("type") in EXCLUDE_TYPES:
        return False, f"type={fm['type']}"
    return True, "included"


def embed(text: str, model: str, url: str, *, proof: dict | None = None) -> list[float]:
    data = _http_post_json(
        f"{url}/api/embeddings",
        {"model": model, "prompt": text},
        EMBED_TIMEOUT_SEC,
    )
    if proof is not None:
        validate_model_response(data, proof)
    emb = data.get("embedding")
    if not valid_embedding_vector(emb):
        raise RuntimeError("ollama returned an invalid embedding vector")
    return emb


def run_check(
    rebuild: bool,
    report_path: Path | None,
    ollama_url: str,
    model: str,
) -> int:
    if report_path is not None:
        log(
            "ERR: --report is disabled; capture stdout for review, then use a "
            "claude-obsidian transaction to file a canonical report"
        )
        return EXIT_USAGE
    try:
        thresholds = validate_thresholds(load_thresholds())
    except ThresholdValidationError as exc:
        log(f"ERR: invalid tiling thresholds: {exc}")
        return EXIT_USAGE
    if not detect_ollama(ollama_url):
        log(f"ollama not reachable at {ollama_url}; skipping tiling check")
        return EXIT_NO_OLLAMA
    if not detect_model(ollama_url, model):
        log(f"model '{model}' not pulled; run: ollama pull {model}")
        return EXIT_NO_MODEL

    try:
        proof = local_model_proof(ollama_url, model)
    except OllamaModelUnavailable:
        log(f"model '{model}' not pulled; run: ollama pull {model}")
        return EXIT_NO_MODEL
    except (
        OllamaLocalProofError,
        http.client.HTTPException,
        OSError,
        RecursionError,
        UnicodeError,
        ValueError,
    ) as exc:
        log(f"ERR: local model proof failed: {type(exc).__name__}: {exc}")
        return EXIT_INCOMPLETE
    resolved_model = proof["model"]
    digest = proof["digest"]

    cache: dict[str, Any] = (
        load_cache(resolved_model, digest)
        if not rebuild
        else {
            "version": 2,
            "model": resolved_model,
            "digest": digest,
            "embeddings": {},
        }
    )
    expected_dimensions = next(
        (
            len(entry["embedding"])
            for entry in cache["embeddings"].values()
        ),
        None,
    )

    pages: list[tuple[str, list[float]]] = []
    scanned = 0
    computed = 0
    cached_hits = 0
    skipped_counts: dict[str, int] = {}
    live_paths: set[str] = set()
    # One provider failure opens a circuit for the whole run: no further page
    # body is sent, cached vectors still count, and the run is incomplete.
    provider_failed = False

    candidates = sorted(WIKI_DIR.rglob("*.md"))
    scale_n = len(candidates)
    if scale_n > SCALE_HARD_FAIL_PAGES:
        log(f"ERR: {scale_n} pages exceed hard-fail limit {SCALE_HARD_FAIL_PAGES}")
        return EXIT_SCALE_EXCEEDED
    if scale_n > SCALE_WARN_PAGES:
        log(f"WARN: {scale_n} pages; cold-cache embed will issue ~{scale_n} POSTs to ollama")

    for md in candidates:
        scanned += 1
        # Symlink and vault-root guards must run BEFORE read_text so a
        # hostile symlink cannot cause off-vault content to be read and
        # POSTed to the embedding endpoint.
        if md.is_symlink():
            skipped_counts["symlink"] = skipped_counts.get("symlink", 0) + 1
            continue
        try:
            resolved = md.resolve(strict=True)
            resolved.relative_to(VAULT_ROOT.resolve())
        except (OSError, ValueError):
            skipped_counts["escapes vault"] = skipped_counts.get("escapes vault", 0) + 1
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped_counts["read_error"] = skipped_counts.get("read_error", 0) + 1
            continue
        if len(text.encode("utf-8")) > MAX_BODY_BYTES:
            skipped_counts["too_large"] = skipped_counts.get("too_large", 0) + 1
            continue
        fm, body = parse_frontmatter(text)
        ok, reason = included(md, fm)
        if not ok:
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
            continue
        rel = md.relative_to(VAULT_ROOT).as_posix()
        live_paths.add(rel)
        h = body_hash(body, resolved_model, digest)
        entry = cache["embeddings"].get(rel)
        if entry and entry.get("hash") == h:
            pages.append((rel, entry["embedding"]))
            cached_hits += 1
            continue
        if provider_failed:
            skipped_counts["provider_circuit_open"] = (
                skipped_counts.get("provider_circuit_open", 0) + 1
            )
            continue
        failure = None
        try:
            emb = embed(body, resolved_model, ollama_url, proof=proof)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        else:
            if expected_dimensions is None:
                expected_dimensions = len(emb)
            elif len(emb) != expected_dimensions:
                failure = "provider embedding dimensions differ"
        if failure is not None:
            log(f"ERR embedding {rel}: {failure}")
            log(
                "ERR: provider circuit opened; no further page text will be "
                "sent in this run"
            )
            skipped_counts["embed_error"] = skipped_counts.get("embed_error", 0) + 1
            provider_failed = True
            continue
        cache["embeddings"][rel] = {
            "hash": h,
            "embedding": emb,
            "computed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        pages.append((rel, emb))
        computed += 1

    incomplete = provider_failed
    orphans = []
    if not incomplete:
        # Orphan GC is safe only after every eligible page was embedded.
        orphans = [k for k in cache["embeddings"] if k not in live_paths]
        for k in orphans:
            del cache["embeddings"][k]
        try:
            save_cache(cache)
        except (OSError, TransactionError) as exc:
            log(f"ERR: cache write failed: {exc}")
            return EXIT_CACHE_CORRUPT

    review = thresholds["bands"]["review"]
    error_ = thresholds["bands"]["error"]
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            a_path, a_emb = pages[i]
            b_path, b_emb = pages[j]
            try:
                sim = cosine(a_emb, b_emb)
            except ValueError as exc:
                log(f"WARN cosine skip ({a_path}, {b_path}): {exc}")
                continue
            if sim >= review:
                pairs.append((sim, a_path, b_path))
    pairs.sort(reverse=True)

    errors = [p for p in pairs if p[0] >= error_]
    reviews = [p for p in pairs if review <= p[0] < error_]

    out_lines: list[str] = []
    out_lines.append("# Semantic Tiling Report")
    out_lines.append("")
    out_lines.append(f"- generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    status = "incomplete" if incomplete else ("complete-empty" if not pages else "complete")
    out_lines.append(
        f"- status: {status}"
        + (" (provisional; prior cache preserved)" if incomplete else "")
    )
    out_lines.append(f"- model: {resolved_model}")
    out_lines.append(f"- ollama_url: {ollama_url}")
    out_lines.append(f"- thresholds: error>={error_}, review={review}-{error_}")
    out_lines.append(f"- calibrated: {thresholds.get('calibrated', False)}"
                     + (" (using uncalibrated defaults)" if not thresholds.get("calibrated") else ""))
    out_lines.append(f"- pages scanned: {scanned}; embedded: {len(pages)}; skipped: {sum(skipped_counts.values())}")
    if skipped_counts:
        out_lines.append("- skipped reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped_counts.items())))
    out_lines.append(f"- cache hits: {cached_hits}; recomputed: {computed}; orphans pruned: {len(orphans)}")
    out_lines.append("")
    out_lines.append(f"## Errors (similarity >= {error_})")
    out_lines.append("")
    if not errors:
        out_lines.append("- none")
    else:
        for sim, a, b in errors:
            out_lines.append(f"- `{sim:.4f}` {a} -- {b}")
    out_lines.append("")
    out_lines.append(f"## Review ({review} <= similarity < {error_})")
    out_lines.append("")
    if not reviews:
        out_lines.append("- none")
    else:
        for sim, a, b in reviews:
            out_lines.append(f"- `{sim:.4f}` {a} -- {b}")
    report = "\n".join(out_lines) + "\n"

    print(report)

    return EXIT_INCOMPLETE if incomplete else EXIT_OK


def cmd_peek(ollama_url: str, model: str) -> int:
    """Structured diagnostics. Prints a JSON object and a plain summary."""
    try:
        thresholds = validate_thresholds(load_thresholds())
    except ThresholdValidationError as exc:
        log(f"ERR: invalid tiling thresholds: {exc}")
        return EXIT_USAGE
    diag: dict = {}
    script_path = Path(__file__).resolve()
    diag["script_path"] = str(script_path)
    diag["script_executable"] = os.access(script_path, os.X_OK)
    diag["python"] = sys.executable
    diag["vault_root"] = str(VAULT_ROOT)
    diag["ollama_url"] = ollama_url
    diag["ollama_reachable"] = detect_ollama(ollama_url)
    diag["model_requested"] = model
    diag["model_present"] = detect_model(ollama_url, model) if diag["ollama_reachable"] else False
    diag["local_model_proved"] = False
    if diag["model_present"]:
        try:
            proof = local_model_proof(ollama_url, model)
            diag["local_model_proved"] = True
            diag["model_resolved"] = proof["model"]
            diag["model_digest"] = proof["digest"]
        except (
            OllamaLocalProofError,
            http.client.HTTPException,
            OSError,
            RecursionError,
            UnicodeError,
            ValueError,
        ) as exc:
            diag["local_model_proof_error"] = str(exc)
    if not diag["ollama_reachable"]:
        print(json.dumps(diag, indent=2))
        return EXIT_NO_OLLAMA
    if not diag["model_present"]:
        print(json.dumps(diag, indent=2))
        return EXIT_NO_MODEL
    if not diag["local_model_proved"]:
        print(json.dumps(diag, indent=2))
        return EXIT_INCOMPLETE
    diag["cache_present"] = CACHE_PATH.exists()
    diag["cache_readable"] = False
    diag["cache_entries"] = 0
    diag["cache_model"] = None
    if diag["cache_present"]:
        try:
            with CACHE_PATH.open(encoding="utf-8") as f:
                c = json.load(f)
            validate_cache_document(c)
            diag["cache_readable"] = True
            diag["cache_entries"] = len(c.get("embeddings", {}))
            diag["cache_model"] = c.get("model")
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            diag["cache_readable"] = False
            diag["cache_error"] = str(exc)
    diag["thresholds_present"] = THRESHOLDS_PATH.exists()
    diag["thresholds_readable"] = False
    if diag["thresholds_present"]:
        diag["thresholds_readable"] = True
        diag["thresholds_calibrated"] = bool(thresholds.get("calibrated", False))
        diag["thresholds_bands"] = thresholds["bands"]
    print(json.dumps(diag, indent=2))
    if diag["cache_present"] and not diag["cache_readable"]:
        return EXIT_CACHE_CORRUPT
    return EXIT_OK


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vault", help="Explicit vault root")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="deprecated and rejected; reports are stdout-only",
    )
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--peek", action="store_true")
    p.add_argument("--allow-remote-ollama", action="store_true",
                   help="allow OLLAMA_URL env override pointing outside localhost")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)
    if not configure_vault(args.vault):
        return EXIT_USAGE

    env_url = os.environ.get("OLLAMA_URL")
    ollama_url = env_url or DEFAULT_OLLAMA_URL
    parsed_url = _validate_ollama_base_url(ollama_url)
    if parsed_url is None:
        log(
            "ERR: OLLAMA_URL must be a credential-free http(s) base URL "
            "without a query or fragment."
        )
        return EXIT_USAGE
    if (parsed_url.hostname or "").lower() == "localhost":
        netloc = "127.0.0.1"
        if parsed_url.port is not None:
            netloc += f":{parsed_url.port}"
        ollama_url = parsed_url._replace(netloc=netloc).geturl()
    if env_url and not _is_local_url(ollama_url) and not args.allow_remote_ollama:
        log(f"ERR: OLLAMA_URL={ollama_url!r} is not localhost. "
            f"Vault content would be POSTed to a non-local host. "
            f"Pass --allow-remote-ollama to override.")
        return EXIT_USAGE

    if args.peek:
        return cmd_peek(ollama_url, args.model)
    return run_check(
        rebuild=args.rebuild_cache,
        report_path=args.report,
        ollama_url=ollama_url,
        model=args.model,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
