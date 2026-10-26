#!/usr/bin/env python3
"""rerank.py — query-time reranker for chunk candidates.

Takes a query string + a list of candidate chunks (from BM25, vector, or any
upstream stage) and reorders them using semantic similarity.

Strategy (in preference order, automatically chosen at runtime):
  1. If Ollama is reachable AND the selected embedding model is pulled
       → embed the query with Nomic's ``search_query:`` task prefix and
         each candidate's contextualized_text with ``search_document:``, then
         rank by cosine. Caches per-chunk embeddings in
         .vault-meta/embed-cache.json under a model + scheme + input-hash key.
  2. Otherwise
       → no-op rerank: return candidates in input order with a synthesized
         note. Caller (retrieve.py) still gets a useful result; downstream
         drill-into-page logic is unchanged.

The default is Ollama's multilingual nomic-embed-text-v2-moe. It is optional
and never downloaded automatically. The smaller English-oriented v1.5 model
remains available through explicit ``--model nomic-embed-text`` selection.

Mirrors the localhost-only OLLAMA_URL guard from scripts/tiling-check.py:
remote ollama endpoints require --allow-remote-ollama because page bodies
are POSTed as embedding input.

Usage:
  rerank.py "query string" --candidates candidates.json [--top 5]
  rerank.py "query string" --candidates - --top 5    # stdin
  rerank.py --peek "query string"                     # show strategy chosen
  rerank.py --model nomic-embed-text "query" --peek  # explicit v1.5 opt-in

Candidates JSON shape:
  [{"chunk_id": "c-000042:3", "path": ".vault-meta/chunks/.../chunk-003.json", "score": 7.1}, ...]

Output: ranked candidates with `rerank_score` added.

Exit codes:
  0 — success
  2 — usage error
  3 — candidate input malformed
  10 — ollama unreachable (no-op rerank performed, exit 0 with note)
  11 — model not pulled (no-op rerank performed, exit 0 with note)
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
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from claude_obsidian.paths import VaultSelectionError, assert_within, resolve_vault_root
from claude_obsidian.ollama import (
    NoRedirectHandler,
    OllamaLocalProofError,
    OllamaModelUnavailable,
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
META_DIR = VAULT_ROOT / ".vault-meta"
EMBED_CACHE_PATH = META_DIR / "embed-cache.json"
CACHE_LOCK = META_DIR / ".embed-cache.lock"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text-v2-moe"
OLLAMA_TIMEOUT_SEC = 3
EMBED_TIMEOUT_SEC = 30
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_MODEL_NAME_LENGTH = 255
MAX_TOP_RESULTS = 1_000
MAX_QUERY_CHARS = 8_000
MAX_CANDIDATES = 1_000
MODEL_NAME_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)

QUERY_TASK_PREFIX = "search_query: "
DOCUMENT_TASK_PREFIX = "search_document: "
NOMIC_EMBED_SCHEME = "nomic-search-v1"
RAW_EMBED_SCHEME = "raw-v1"


def configure_vault(explicit=None):
    global VAULT_ROOT, META_DIR, EMBED_CACHE_PATH, CACHE_LOCK
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
    META_DIR = VAULT_ROOT / ".vault-meta"
    EMBED_CACHE_PATH = META_DIR / "embed-cache.json"
    CACHE_LOCK = META_DIR / ".embed-cache.lock"
    try:
        for relative in (
            ".vault-meta",
            ".vault-meta/embed-cache.json",
            ".vault-meta/.embed-cache.lock",
            ".vault-meta/chunks",
        ):
            _safe_vault_path(VAULT_ROOT, relative)
    except (VaultSelectionError, TransactionError) as exc:
        log(f"ERR {exc.code}: {exc}")
        return False
    return True

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CANDIDATES = 3
EXIT_NO_OLLAMA = 10
EXIT_NO_MODEL = 11


def log(msg):
    print(msg, file=sys.stderr)


def validate_model_name(model):
    """Return a bounded Ollama model identifier or raise ValueError.

    Model selection is data sent to Ollama, not a URL, path, or shell fragment.
    Keeping the grammar narrow also makes cache and diagnostic keys stable.
    """
    repository = model.split(":", 1)[0] if isinstance(model, str) else ""
    if (
        not isinstance(model, str)
        or not model
        or len(model) > MAX_MODEL_NAME_LENGTH
        or MODEL_NAME_RE.fullmatch(model) is None
        or any(part in {"", ".", ".."} for part in repository.split("/"))
    ):
        raise ValueError(
            "model must be a bounded Ollama name such as "
            "nomic-embed-text-v2-moe or nomic-embed-text:v1.5"
        )
    return model


def parse_top_k(value):
    """Parse a positive, bounded result count for command-line callers."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= MAX_TOP_RESULTS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_TOP_RESULTS}"
        )
    return parsed


def validate_query(query):
    """Validate query text before any model, endpoint, cache, or embed work."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(unicodedata.normalize("NFKC", query)) > MAX_QUERY_CHARS
    ):
        raise ValueError(
            f"query must be non-empty text within {MAX_QUERY_CHARS} normalized characters"
        )
    return query


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_candidate_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        path.as_posix() == value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def validate_candidates(candidates):
    """Validate the complete candidate envelope before provider or cache use."""
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a JSON list")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"candidates must contain at most {MAX_CANDIDATES} objects")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} must be an object")
        if not _safe_candidate_path(candidate.get("path")):
            raise ValueError(f"candidate {index} path must be a safe relative path")
        score_fields = [
            field for field in ("bm25_score", "score") if field in candidate
        ]
        if not score_fields:
            raise ValueError(f"candidate {index} must contain a finite score")
        for field in score_fields:
            score = candidate[field]
            if not _finite_number(score):
                raise ValueError(f"candidate {index} {field} must be finite")
    return candidates


def validate_rerank_inputs(query, candidates):
    validate_query(query)
    validate_candidates(candidates)


def model_is_available(model, available_models):
    """Match the exact request, treating only ``:latest`` as its untagged alias."""
    selected = validate_model_name(model)
    names = {
        item
        for item in available_models
        if isinstance(item, str) and item
    }
    if selected in names:
        return True
    if ":" in selected:
        return False
    return f"{selected}:latest" in names


def is_nomic_model(model):
    """Return whether ``model`` uses Nomic's asymmetric retrieval contract."""
    leaf = str(model).lower().rsplit("/", 1)[-1].split(":", 1)[0]
    return leaf.startswith("nomic")


def embedding_scheme(model):
    """Version token for the exact text-to-vector input convention."""
    return NOMIC_EMBED_SCHEME if is_nomic_model(model) else RAW_EMBED_SCHEME


def with_task_prefix(text, role, model=None):
    """Apply the task prefix required by Nomic retrieval embeddings.

    Non-Nomic models keep their prior raw-text behavior. A role typo is rejected
    instead of silently putting a query and document in incompatible spaces.
    """
    model = model or DEFAULT_MODEL
    if role not in ("query", "document"):
        raise ValueError(f"role must be 'query' or 'document', got {role!r}")
    if not is_nomic_model(model):
        return text
    prefix = QUERY_TASK_PREFIX if role == "query" else DOCUMENT_TASK_PREFIX
    return prefix + text


def embedding_input_hash(text):
    """Hash the exact prefixed text sent to the embedding model."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_cache_key(model, input_hash, digest=None):
    """Key embeddings by verified model bytes and exact embedded content."""
    identity = f"{model}:{digest}:" if digest is not None else f"{model}:"
    return f"{identity}{embedding_scheme(model)}:{input_hash}"


def fallback_score(candidate):
    """Read the sparse score from either supported candidate contract."""
    value = candidate.get("bm25_score", candidate.get("score", 0.0))
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return score if _finite_number(score) else 0.0


def mark_noop(candidates, source):
    """Annotate a fallback without changing the upstream BM25 ordering."""
    for candidate in candidates:
        candidate["rerank_score"] = fallback_score(candidate)
        candidate["rerank_source"] = source


def usable_embedding(value, expected_len=None):
    """Reject empty, malformed, non-finite, zero, or wrong-sized vectors."""
    if not isinstance(value, list) or not value:
        return False
    if expected_len is not None and len(value) != expected_len:
        return False
    return all(_finite_number(item) for item in value) and any(item != 0 for item in value)


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    if not all(_finite_number(item) for item in (*a, *b)):
        return 0.0
    scale_a = max(abs(item) for item in a)
    scale_b = max(abs(item) for item in b)
    if scale_a == 0 or scale_b == 0:
        return 0.0
    scaled_a = [item / scale_a for item in a]
    scaled_b = [item / scale_b for item in b]
    dot = sum(x * y for x, y in zip(scaled_a, scaled_b))
    na = math.sqrt(sum(x * x for x in scaled_a))
    nb = math.sqrt(sum(y * y for y in scaled_b))
    result = dot / (na * nb)
    if not math.isfinite(result):
        return 0.0
    return max(-1.0, min(1.0, result))


def validate_ollama_base_url(url):
    """Accept only credential-free HTTP(S) base URLs."""

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

# Failures that select a no-op rerank. retrieve.py relies on rerank() never
# raising for a broken or hostile responder on the Ollama port.
OLLAMA_UNAVAILABLE_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    OSError,
    RecursionError,
    UnicodeError,
    ValueError,
)


def _open_ollama_request(request, timeout):
    """Open an Ollama request without proxies or redirects, within one deadline."""
    return open_direct(request, timeout)


def _ollama_json(request, timeout):
    deadline = time.monotonic() + timeout
    with _open_ollama_request(request, timeout) as response:
        return read_json_object(
            response,
            request,
            limit=MAX_RESPONSE_BYTES,
            deadline=deadline,
            label="Ollama",
        )


def ollama_url(allow_remote):
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    parsed = validate_ollama_base_url(url)
    if parsed is None:
        log(
            "ERR: OLLAMA_URL must be a credential-free http(s) base URL "
            "without a query or fragment."
        )
        sys.exit(EXIT_USAGE)
    host = (parsed.hostname or "").lower()
    if host == "localhost":
        netloc = "127.0.0.1"
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
        url = parsed.geturl()
    if not allow_remote:
        host = (parsed.hostname or "").lower()
        if host not in ("127.0.0.1", "::1"):
            log(f"ERR: OLLAMA_URL={url} points off-localhost (host={host!r}).")
            log("  Either: (a) run ollama locally — `systemctl --user start ollama` or `ollama serve`")
            log("  Or:     (b) pass --allow-remote-ollama through retrieve.py, which forwards it here.")
            log("  Or:     (c) unset OLLAMA_URL to fall back to the local default (127.0.0.1:11434).")
            sys.exit(EXIT_USAGE)
    return url


def ollama_alive(url):
    if validate_ollama_base_url(url) is None:
        return False, []
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        data = _ollama_json(req, OLLAMA_TIMEOUT_SEC)
        listed = data.get("models", [])
        models = [
            model["name"]
            for model in (listed if isinstance(listed, list) else [])
            if isinstance(model, dict)
            and isinstance(model.get("name"), str)
            and model["name"]
        ]
        return True, models
    except OLLAMA_UNAVAILABLE_ERRORS:
        return False, []


def local_model_proof(url, model):
    """Prove cloud-disabled local model state without sending query text."""
    def get_json(endpoint):
        request = urllib.request.Request(f"{url}/api/{endpoint}", method="GET")
        return _ollama_json(request, OLLAMA_TIMEOUT_SEC)

    def post_json(endpoint, payload):
        request = urllib.request.Request(
            f"{url}/api/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _ollama_json(request, OLLAMA_TIMEOUT_SEC)

    return prove_local_model(model, get_json, post_json)


def embed_one(url, model, text, *, proof=None):
    if validate_ollama_base_url(url) is None:
        raise ValueError("Ollama URL must be a credential-free HTTP(S) base URL")
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data = _ollama_json(req, EMBED_TIMEOUT_SEC)
    if proof is not None:
        validate_model_response(data, proof)
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        return []
    return embeddings[0]


def load_cache():
    if not EMBED_CACHE_PATH.is_file():
        return {}
    try:
        cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        return cache if isinstance(cache, dict) else {}
    except (ValueError, RecursionError, OSError, UnicodeError):
        return {}


def save_cache(cache):
    """Persist the optional cache through the canonical pinned mutation path.

    Cache persistence is intentionally brief and separate from embedding work.
    A bounded lock attempt preserves the historical non-blocking query
    behavior, while the pinned root and metadata descriptors ensure a replaced
    vault root or ``.vault-meta`` entry cannot redirect the write.
    """
    payload = json.dumps(cache, ensure_ascii=False).encode("utf-8")
    root_fd = -1
    meta_fd = -1
    try:
        with MutationLock(VAULT_ROOT, timeout=0.3) as mutation_lock:
            root_fd = mutation_lock.duplicate_root_fd()
            meta_fd = mutation_lock.duplicate_parent_fd()
            _atomic_vault_write(
                VAULT_ROOT,
                ".vault-meta/embed-cache.json",
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


def load_chunk(chunk_rel_path):
    if not isinstance(chunk_rel_path, str) or not chunk_rel_path:
        return None
    rel = Path(chunk_rel_path)
    # On Windows "/x" and "C:x" are not "absolute" under pathlib but still
    # escape a joined root.
    if (
        rel.is_absolute()
        or chunk_rel_path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", chunk_rel_path)
    ):
        return None
    p = (VAULT_ROOT / rel).resolve()
    if not p.is_relative_to((META_DIR / "chunks").resolve()):
        return None
    if not p.is_file():
        return None
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        text = value.get("contextualized_text") or value.get("raw_text", "")
        return value if isinstance(text, str) else None
    except (ValueError, RecursionError, OSError, UnicodeError):
        return None


def rerank(query, candidates, top_k=5, allow_remote=False, model=None):
    """Returns candidates list, possibly truncated to top_k, with rerank_score added.
    Falls back to input-order if ollama is unavailable (still adds rerank_source: 'noop').
    """
    validate_rerank_inputs(query, candidates)
    selected_model = validate_model_name(model or DEFAULT_MODEL)
    url = ollama_url(allow_remote)
    alive, models = ollama_alive(url)
    if not alive:
        log("ollama unreachable — no-op rerank")
        mark_noop(candidates, "noop-no-ollama")
        return candidates[:top_k]
    if not model_is_available(selected_model, models):
        log(
            f"model {selected_model} not pulled — no-op rerank; "
            f"install explicitly with: ollama pull {selected_model}"
        )
        mark_noop(candidates, "noop-no-model")
        return candidates[:top_k]

    try:
        proof = local_model_proof(url, selected_model)
    except OllamaModelUnavailable:
        log(
            f"model {selected_model} not pulled - no-op rerank; "
            f"install explicitly with: ollama pull {selected_model}"
        )
        mark_noop(candidates, "noop-no-model")
        return candidates[:top_k]
    except (OllamaLocalProofError, *OLLAMA_UNAVAILABLE_ERRORS) as exc:
        log(f"local model proof failed: {type(exc).__name__}: {exc}; no-op rerank")
        mark_noop(candidates, "noop-local-proof")
        return candidates[:top_k]

    resolved_model = proof["model"]

    cache = load_cache()
    cache_dirty = False
    try:
        q_emb = embed_one(
            url,
            resolved_model,
            with_task_prefix(query, "query", model=resolved_model),
            proof=proof,
        )
        if not usable_embedding(q_emb):
            raise ValueError("query embedder returned an unusable vector")
    except Exception as e:
        log(f"query embed failed: {e}")
        mark_noop(candidates, "noop-embed-error")
        return candidates[:top_k]

    for c in candidates:
        chunk = load_chunk(c.get("path", ""))
        if not chunk:
            log(f"chunk missing for {c.get('chunk_id')} — no-op rerank")
            mark_noop(candidates, "noop-missing-chunk")
            return candidates[:top_k]
        text = chunk.get("contextualized_text") or chunk.get("raw_text", "")
        document_input = with_task_prefix(text, "document", model=resolved_model)
        cache_key = embedding_cache_key(
            resolved_model,
            embedding_input_hash(document_input),
            digest=proof["digest"],
        )
        emb = cache.get(cache_key)
        if not usable_embedding(emb, expected_len=len(q_emb)):
            try:
                emb = embed_one(
                    url,
                    resolved_model,
                    document_input,
                    proof=proof,
                )
                if not usable_embedding(emb, expected_len=len(q_emb)):
                    raise ValueError("document embedder returned an unusable vector")
            except Exception as e:
                log(f"embed failed for {c.get('chunk_id')}: {e}")
                # Cosine scores and BM25 scores are not comparable scales. A
                # partial rerank would promote failures unpredictably, so one
                # candidate failure falls back the complete ordered set.
                mark_noop(candidates, "noop-embed-error")
                return candidates[:top_k]
            cache[cache_key] = emb
            cache_dirty = True
        c["rerank_score"] = cosine(q_emb, emb)
        c["rerank_source"] = f"cosine:{resolved_model}"

    if cache_dirty:
        try:
            save_cache(cache)
        except (OSError, TransactionError) as e:
            # Cache persistence is optional. A valid in-memory rerank result
            # must not be discarded just because the derived cache is read-only.
            log(f"WARN: could not persist embed cache: {e}")

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_k]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rerank chunk candidates by semantic similarity.")
    parser.add_argument("--vault", help="Explicit vault root")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("--candidates", help="Path to candidates JSON or `-` for stdin",
                        default=None)
    parser.add_argument("--top", type=parse_top_k, default=5, help="Top-K to return")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Installed Ollama embedding model; defaults to multilingual "
            f"{DEFAULT_MODEL} and is never downloaded automatically"
        ),
    )
    parser.add_argument("--peek", action="store_true",
                        help="Print rerank strategy chosen and exit")
    parser.add_argument("--allow-remote-ollama", action="store_true",
                        help="Accept non-localhost OLLAMA_URL (potential data exfil)")
    args = parser.parse_args(argv)
    if not configure_vault(args.vault):
        return EXIT_USAGE

    if args.peek:
        if not args.query:
            log("--peek needs a query string")
            sys.exit(EXIT_USAGE)
        try:
            validate_query(args.query)
        except ValueError as exc:
            log(f"ERR: invalid rerank input: {exc}")
            return EXIT_CANDIDATES
        try:
            args.model = validate_model_name(args.model)
        except ValueError as exc:
            parser.error(str(exc))
        url = ollama_url(args.allow_remote_ollama)
        alive, models = ollama_alive(url)
        strategy = "noop-no-ollama"
        present = model_is_available(args.model, models) if alive else False
        if alive:
            strategy = f"cosine:{args.model}" if present else "noop-no-model"
        if present:
            try:
                proof = local_model_proof(url, args.model)
                strategy = f"cosine:{proof['model']}"
            except (OllamaLocalProofError, *OLLAMA_UNAVAILABLE_ERRORS):
                strategy = "noop-local-proof"
        print(json.dumps({
            "query": args.query,
            "strategy": strategy,
            "model": args.model,
            "ollama_url": url,
            "ollama_alive": alive,
            "model_present": present,
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, indent=2))
        return EXIT_OK

    if not args.query or args.candidates is None:
        log("usage: rerank.py <query> --candidates <path|-> [--top N]")
        return EXIT_USAGE

    try:
        if args.candidates == "-":
            cand_text = sys.stdin.read()
        else:
            candidate_path = assert_within(VAULT_ROOT, args.candidates)
            cand_text = candidate_path.read_text(encoding="utf-8")
    except (VaultSelectionError, OSError, UnicodeError) as exc:
        log(f"ERR: candidates must be readable UTF-8 JSON from stdin or an in-vault file: {exc}")
        return EXIT_CANDIDATES
    try:
        candidates = json.loads(cand_text)
        validate_rerank_inputs(args.query, candidates)
    except (ValueError, RecursionError) as e:
        log(f"ERR: bad candidates JSON: {e}")
        return EXIT_CANDIDATES

    try:
        args.model = validate_model_name(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    result = rerank(
        args.query,
        candidates,
        top_k=args.top,
        allow_remote=args.allow_remote_ollama,
        model=args.model,
    )
    print(json.dumps(result, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
