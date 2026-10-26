"""Bounded local ledger capacity inspection and lossless compaction plans."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .json_utils import strict_json_loads
from .limits import CANONICAL_LEDGER_PATHS, MAX_CANONICAL_LEDGER_BYTES
from .transaction import BUNDLE_SCHEMA, TransactionValidationError, read_vault_regular

MAX_LEGACY_READ_BYTES = 64 * 1024 * 1024
NEAR_CAPACITY_RATIO = 0.85
JSON_WHITESPACE = b' \t\r\n'
# A complete JSON string (kept verbatim) or a run of insignificant JSON
# whitespace (dropped). UTF-8 continuation bytes are never ASCII, so a
# byte-level scan of valid JSON sees the same quotes and escapes as text.
_STRING_OR_WHITESPACE = re.compile(rb'("[^"\\]*(?:\\.[^"\\]*)*")|[ \t\r\n]+', re.DOTALL)


def _near_capacity(size: int) -> bool:
    return size >= MAX_CANONICAL_LEDGER_BYTES * NEAR_CAPACITY_RATIO


def _compact_json_bytes(raw: bytes) -> bytes:
    """Drop JSON whitespace outside strings from already validated JSON."""
    return _STRING_OR_WHITESPACE.sub(rb'\1', raw) + b'\n'


def inspect_capacity(vault: Path | str) -> dict[str, Any]:
    ledgers = []
    for relative in CANONICAL_LEDGER_PATHS:
        try:
            raw = read_vault_regular(Path(vault), relative, limit=MAX_LEGACY_READ_BYTES)
        except TransactionValidationError as exc:
            ledgers.append({'path': relative, 'status': 'unreadable', 'code': exc.code,
                            'message': str(exc), 'read_limit_bytes': MAX_LEGACY_READ_BYTES})
            continue
        size = len(raw) if raw is not None else 0
        status = ('absent' if raw is None else 'over_capacity' if size > MAX_CANONICAL_LEDGER_BYTES
                  else 'near_capacity' if _near_capacity(size) else 'within_capacity')
        ledgers.append({'path': relative, 'size_bytes': size if raw is not None else None,
                        'limit_bytes': MAX_CANONICAL_LEDGER_BYTES, 'status': status,
                        'sha256': hashlib.sha256(raw).hexdigest() if raw is not None else None})
    return {'schema': 'claude-obsidian.capacity.v1', 'ledgers': ledgers,
            'ok': all(x['status'] not in {'over_capacity', 'unreadable'} for x in ledgers),
            'recovery': 'Preview capacity compact; it only removes JSON formatting from near- or over-capacity ledgers. If records still exceed the limit, preserve the source and request an explicitly designed migration. Never truncate ledger records or raw sources.'}


def compaction_bundle(vault: Path | str, *, operation_id: str) -> dict[str, Any]:
    writes = []
    hashes = {}
    for relative in CANONICAL_LEDGER_PATHS:
        raw = read_vault_regular(Path(vault), relative, limit=MAX_LEGACY_READ_BYTES)
        # Only ledgers that inspect_capacity reports as near_capacity or
        # over_capacity are rewritten; others stay byte-for-byte unchanged.
        if raw is None or not _near_capacity(len(raw)):
            continue
        refusal = TransactionValidationError('LEDGER_MIGRATION_REQUIRED', f'{relative}: lossless formatting compaction cannot fit all records within {MAX_CANONICAL_LEDGER_BYTES} bytes; no records were removed and no files changed')
        # Compaction keeps every byte except JSON whitespace (plus one final
        # newline), so this cheap lower bound refuses before parse or scan.
        retained_at_least = len(raw) - sum(raw.count(byte) for byte in JSON_WHITESPACE) + 1
        if retained_at_least > MAX_CANONICAL_LEDGER_BYTES:
            raise refusal
        try:
            document = strict_json_loads(raw.decode('utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise TransactionValidationError('INVALID_PROVENANCE_LEDGER', f'{relative}: cannot losslessly compact invalid JSON') from exc
        if not isinstance(document, dict):
            raise TransactionValidationError('INVALID_PROVENANCE_LEDGER', f'{relative}: root must be an object')
        # Remove only JSON whitespace outside strings. Re-serialization would
        # round high-precision numeric tokens through Python binary floats.
        # The strict parse above checks syntax and duplicate keys; original
        # number and string spellings are retained byte for byte.
        compact = _compact_json_bytes(raw)
        if len(compact) > MAX_CANONICAL_LEDGER_BYTES:
            raise refusal
        if compact != raw:
            writes.append({'path': relative, 'mode': 'replace', 'content': compact.decode('utf-8')})
            hashes[relative] = hashlib.sha256(raw).hexdigest()
    return {'schema': BUNDLE_SCHEMA, 'operation_id': operation_id,
            'operation_type': 'migration', 'expected_hashes': hashes, 'writes': writes}
