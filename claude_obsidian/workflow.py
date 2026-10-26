"""Structural completeness checks for named knowledge workflows.

These checks supplement storage safety. They do not grade prose, prove a claim,
or replace the user's review of selected content and sources.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping
from .hook_adapter import MAX_CONTEXT_BYTES
from .json_utils import strict_json_loads
from .lint_engine import _parse_frontmatter, _yaml_scalar
from .markdown_links import iter_markdown_links, mask_protected_regions
from .visible_paths import is_canonical_visible_wiki_markdown

from .transaction import (
    MAX_TRANSACTION_FILE_BYTES, RESULT_SCHEMA, TransactionConflict, TransactionRecoveryError,
    TransactionValidationError, _load_bundle, _raw_write_bytes,
    apply_bundle, inspect_bundle, read_vault_regular, safe_operation_id,
)

SUPPORTED_WORKFLOWS = frozenset({'save', 'ingest'})
_HEADER = re.compile(r'\A---\r?\n.*?\r?\n---(?:\r?\n|$)', re.S)
_LOG = re.compile(r'^## \[(\d{4}-\d{2}-\d{2})\] (save|ingest)(?:\s|\||$)', re.M)
_WIKI_LINK = re.compile(r'\[\[([^\]\n]+)\]\]')


def _iter_wiki_links(text: str) -> Iterator[re.Match[str]]:
    """Yield ``_WIKI_LINK.finditer`` matches in linear time.

    A failed body ends at its first ``]`` or line break, and every opener
    before that terminator fails for the same reason, so scanning resumes there.
    """
    cursor = 0
    while (opening := text.find('[[', cursor)) >= 0:
        match = _WIKI_LINK.match(text, opening)
        if match is not None:
            yield match
            cursor = match.end()
            continue
        stops = [index for index in (text.find(']', opening + 2), text.find('\n', opening + 2)) if index >= 0]
        if not stops:
            return
        cursor = min(stops)
_DEFAULT_CATALOG = 'wiki/index.md'
_LEDGERS = ('wiki/meta/ledgers/source-ledger.json', 'wiki/meta/ledgers/claim-ledger.json')


def freeze_bundle(bundle_or_path: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    """Read each content file once; inspect and apply the same frozen bytes."""
    bundle, directory = _load_bundle(bundle_or_path)
    return _freeze_loaded_bundle(bundle, directory)


def _validate_workflow_write_shape(bundle: Mapping[str, Any]) -> None:
    """Validate fields used by workflow checks before indexing any write."""
    if not isinstance(bundle.get('operation_type'), str):
        raise TransactionValidationError('INVALID_OPERATION_TYPE', 'operation_type must be text')
    writes = bundle.get('writes')
    if not isinstance(writes, list) or not writes:
        raise TransactionValidationError('NO_WRITES', 'bundle must contain at least one write')
    for index, write in enumerate(writes):
        if not isinstance(write, dict):
            raise TransactionValidationError('INVALID_WRITE', f'write {index} must be an object')
        if not isinstance(write.get('path'), str):
            raise TransactionValidationError('INVALID_WRITE', f'write {index} must contain a string path')
        if not isinstance(write.get('mode'), str) or write['mode'] not in {'create', 'replace'}:
            raise TransactionValidationError('INVALID_WRITE_MODE', f'write {index} mode must be create or replace')
        has_inline = 'content' in write
        has_file = 'content_file' in write
        if has_inline == has_file:
            raise TransactionValidationError(
                'INVALID_WRITE_CONTENT',
                f'write {index} must contain exactly one of content or content_file',
            )
        if has_inline and not isinstance(write.get('content'), str):
            raise TransactionValidationError(
                'INVALID_WRITE_CONTENT', f'write {index} inline content must be text'
            )
        if has_file and not isinstance(write.get('content_file'), str):
            raise TransactionValidationError(
                'INVALID_WRITE_CONTENT', f'write {index} content_file must be a string path'
            )


def _undo_generated_address(data: bytes, write: Mapping[str, Any], bundle: Mapping[str, Any]) -> bytes:
    """Reverse only the transaction engine's exact second-line insertion."""
    requests = bundle.get('address_requests', [])
    if not isinstance(requests, list):
        return data
    prefixes = {request.get('prefix', 'c') for request in requests
                if isinstance(request, dict) and request.get('path') == write.get('path')
                and request.get('prefix', 'c') in ('c', 'l')}
    lines = data.splitlines(keepends=True)
    if len(lines) < 2 or lines[0].strip() != b'---':
        return data
    newline = b'\r\n' if lines[0].endswith(b'\r\n') else b'\n'
    if not any(re.fullmatch(b'address: ' + prefix.encode('ascii') + rb'-[0-9]{6}' + re.escape(newline), lines[1])
               for prefix in prefixes):
        return data
    original = b''.join(lines[:1] + lines[2:])
    return original if hashlib.sha256(original).hexdigest() == write['sha256'] else data


def _freeze_loaded_bundle(bundle: dict[str, Any], directory: Path, *, replay_vault: Path | None = None) -> dict[str, Any]:
    _validate_workflow_write_shape(bundle)
    try:
        # The bundle is already JSON data; a C-level round trip copies it
        # without Python recursion limits on deeply nested extra fields.
        frozen = json.loads(json.dumps(bundle))
    except (RecursionError, TypeError, ValueError) as exc:
        raise TransactionValidationError('INVALID_BUNDLE', 'bundle nesting exceeds the supported depth') from exc
    for write in frozen['writes']:
        if 'content_file' in write:
            if not isinstance(write.get('sha256'), str):
                raise TransactionValidationError('CONTENT_HASH_REQUIRED', 'content_file writes require a declared SHA-256 before materialization')
            try:
                data = _raw_write_bytes(write, directory)
            except TransactionValidationError as exc:
                if replay_vault is None or exc.code != 'CONTENT_FILE_MISSING':
                    raise
                # Only a completed operation may recover a disposable draft from
                # its target. The transaction engine still checks the complete
                # materialized bundle, original approval and every result path
                # under its pinned mutation lock before returning a receipt.
                data = read_vault_regular(replay_vault, write.get('path'), limit=MAX_TRANSACTION_FILE_BYTES)
                if data is not None and hashlib.sha256(data).hexdigest() != write['sha256']:
                    data = _undo_generated_address(data, write, frozen)
                if data is None or hashlib.sha256(data).hexdigest() != write['sha256']:
                    raise TransactionConflict('OPERATION_RESULT_DRIFT', 'completed workflow target differs from the declared content file') from exc
            if hashlib.sha256(data).hexdigest() != write['sha256']:
                raise TransactionValidationError('CONTENT_HASH_MISMATCH', 'content_file differs from the declared SHA-256')
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise TransactionValidationError('WORKFLOW_CONTENT_NOT_TEXT', 'named workflow content files must be UTF-8') from exc
            write.pop('content_file')
            write['content'] = text
    return frozen


def _body(text: str) -> str:
    return _HEADER.sub('', text, count=1).lstrip()


def _log_entry_split(body: str) -> tuple[list[str], int | None]:
    """Return body lines and the index of the first level-2 heading (an entry)."""
    lines = body.splitlines(keepends=True)
    return lines, next((i for i, line in enumerate(lines) if line.startswith('## ')), None)


def _log_preserves_history(previous: str, prospective: str, entry: int | None) -> bool:
    """Existing entries stay unchanged after the new one; other lines keep their order.

    The title and introduction above the first existing entry may surround the
    new entry, so a freshly initialized log accepts an entry directly below its
    seeded header. No existing line may be dropped or reordered.
    """
    old_lines, old_entry = _log_entry_split(previous)
    new_lines = prospective.splitlines(keepends=True)
    if old_entry is not None:
        history = ''.join(old_lines[old_entry:]).rstrip()
        if entry is None or history not in ''.join(new_lines[entry + 1:]):
            return False
    remaining = iter(line.rstrip('\r\n') for line in new_lines)
    preamble = old_lines if old_entry is None else old_lines[:old_entry]
    return all(any(line == candidate for candidate in remaining)
               for line in (item.rstrip('\r\n') for item in preamble) if line.strip())


def _ledger_has_sources(text: str) -> bool:
    try:
        document = strict_json_loads(text)
    except (ValueError, RecursionError):
        return False
    sources = document.get('sources') if isinstance(document, dict) else None
    return isinstance(sources, (dict, list)) and bool(sources)


def _catalog_path(value: Any) -> bool:
    return (is_canonical_visible_wiki_markdown(value)
            and value not in {'wiki/hot.md', 'wiki/log.md', 'wiki/overview.md'})


def _navigation_sources(text: str) -> tuple[str, tuple[str, ...]]:
    """Return protected body text and active flat property values."""
    normalized = text.replace('\r\n', '\n')
    frontmatter = _parse_frontmatter(normalized)
    if not frontmatter.present:
        return mask_protected_regions(normalized), ()
    properties: list[str] = []
    for start, end in frontmatter.property_ranges:
        value = _yaml_scalar(normalized[start:end])
        if isinstance(value, str):
            properties.append(mask_protected_regions(value))
    body = mask_protected_regions(normalized[frontmatter.end_offset:])
    return body, tuple(properties)


def _catalog_links(text: str, catalog: str) -> tuple[frozenset[str], frozenset[str]]:
    """Parse one catalog's navigation once: wikilink targets and resolved paths.

    Wikilink targets keep their written form; relative Markdown destinations
    are normalized against the catalog directory and never escape the vault.
    """
    # Exclude comments/examples. Active flat metadata links remain candidates.
    masked, property_values = _navigation_sources(text)
    wikilinks = {match.group(1).split('|', 1)[0].split('#', 1)[0].strip()
                 for source in (masked, *property_values) for match in _iter_wiki_links(source)}
    resolved: set[str] = set()
    for match in iter_markdown_links(masked):
        if match.embed or any(ord(c) < 32 for c in match.destination):
            continue
        try:
            destination = urllib.parse.urlsplit(match.destination)
        except ValueError:
            continue
        if destination.scheme or destination.netloc:
            continue
        target = urllib.parse.unquote(destination.path)
        if target.startswith('/') or '\\' in target or any(ord(c) < 32 for c in target):
            continue
        # Normalize a relative Markdown destination without reading any path.
        pieces = list(PurePosixPath(catalog).parent.parts)
        escaped = False
        for part in target.split('/'):
            if part == '..':
                if not pieces:
                    escaped = True
                    break
                pieces.pop()
            elif part not in {'', '.'}:
                pieces.append(part)
        if not escaped:
            resolved.add('/'.join(pieces))
    return frozenset(wikilinks), frozenset(resolved)


def _links_to(text: str, page: str, catalog: str,
              links: tuple[frozenset[str], frozenset[str]] | None = None) -> bool:
    """Require unambiguous path-qualified navigation for selected pages."""
    wikilinks, resolved = _catalog_links(text, catalog) if links is None else links
    candidates = {page, page.removesuffix('.md'), page.removeprefix('wiki/'), page.removeprefix('wiki/').removesuffix('.md')}
    if PurePosixPath(page).parent == PurePosixPath(catalog).parent:
        candidates.update({PurePosixPath(page).name, PurePosixPath(page).stem})
    return bool(candidates & wikilinks) or bool({page, page.removesuffix('.md')} & resolved)


def inspect_workflow(vault: Path | str, bundle_or_path: Mapping[str, Any] | Path | str, *, _completed_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    frozen = freeze_bundle(bundle_or_path)
    storage = inspect_bundle(vault, frozen) if _completed_plan is None else _completed_plan
    operation = frozen.get('operation_type')
    findings: list[dict[str, str]] = []
    def fail(code: str, path: str, message: str) -> None:
        findings.append({'code': code, 'path': path, 'message': message})
    if operation not in SUPPORTED_WORKFLOWS:
        fail('UNSUPPORTED_WORKFLOW', '', 'workflow inspection supports save and ingest; use transaction for other operation types')
    raw_writes = frozen.get('writes', [])
    writes = {w['path']: w for w in raw_writes}
    policy = frozen.get('workflow', {})
    if not isinstance(policy, dict) or set(policy) - {'catalogs'}:
        fail('INVALID_WORKFLOW_POLICY', '', 'workflow must contain only an optional catalogs array')
        policy = {}
    catalogs = policy.get('catalogs', ['wiki/index.md'])
    if (not isinstance(catalogs, list) or not 1 <= len(catalogs) <= 32
            or any(not _catalog_path(p) for p in catalogs) or len(set(catalogs)) != len(catalogs)):
        fail('INVALID_CATALOGS', '', 'catalogs must be 1 to 32 distinct canonical wiki Markdown paths')
        catalogs = []
    if raw_writes and operation in SUPPORTED_WORKFLOWS:
        # Each catalog is parsed once per inspection, however many pages it serves.
        catalog_links: dict[str, tuple[frozenset[str], frozenset[str]]] = {}

        def links_to(catalog: str, page: str) -> bool:
            if catalog not in catalog_links:
                catalog_links[catalog] = _catalog_links(writes[catalog].get('content', ''), catalog)
            return _links_to('', page, catalog, catalog_links[catalog])

        if not any(path in writes for path in catalogs):
            fail('MISSING_CATALOG_WRITE', '', 'named workflow requires a hash-bound declared active catalog write')
        # An active catalog is the root index or a page that already exists
        # (a hash-bound replace). A catalog created by this bundle must itself
        # be reachable from an active catalog, so a throwaway page cannot
        # stand in for vault navigation.
        active = [p for p in catalogs if p in writes
                  and (p == _DEFAULT_CATALOG or writes[p].get('mode') == 'replace')]
        for path in catalogs:
            if path in writes and path not in active and not any(
                    links_to(a, path) for a in active):
                fail('INACTIVE_CATALOG', path, 'a new catalog needs a path-qualified link from wiki/index.md or an existing declared catalog')
        for required in ('wiki/log.md', 'wiki/hot.md'):
            if required not in writes:
                fail('MISSING_COUPLED_WRITE', required, 'named workflow requires a batch log and bounded hot context')
        hot = writes.get('wiki/hot.md', {}).get('content', '')
        if not _body(hot).strip():
            fail('EMPTY_HOT_CONTEXT', 'wiki/hot.md', 'hot context must contain the reviewed recent context')
        if len(_body(hot).split()) > 500:
            fail('HOT_CONTEXT_TOO_LARGE', 'wiki/hot.md', 'hot context exceeds 500 words')
        elif len(hot.encode('utf-8')) > MAX_CONTEXT_BYTES:
            fail('HOT_CONTEXT_TOO_LARGE', 'wiki/hot.md', f'hot context exceeds the {MAX_CONTEXT_BYTES}-byte session context bound')
        log = writes.get('wiki/log.md', {}).get('content', '')
        if not _body(log).strip():
            fail('INVALID_BATCH_LOG', 'wiki/log.md', 'batch log must contain a newest operation entry')
        if log:
            lines, entry = _log_entry_split(_body(log))
            match = _LOG.match(lines[entry]) if entry is not None else None
            valid_date = False
            if match is not None:
                try:
                    date.fromisoformat(match.group(1))
                    valid_date = True
                except ValueError:
                    pass
            if not valid_date or match is None or match.group(2) != operation:
                fail('INVALID_BATCH_LOG', 'wiki/log.md', 'first log entry must name this save or ingest operation with an ISO date')
            # Completed replay is checked against its retained result; the
            # current log is not an input there and is never read.
            old = None if _completed_plan is not None else read_vault_regular(
                Path(vault), 'wiki/log.md', limit=16 * 1024 * 1024)
            if old:
                try:
                    previous = _body(old.decode('utf-8'))
                except UnicodeDecodeError as exc:
                    raise TransactionValidationError(
                        'WORKFLOW_LOG_NOT_TEXT',
                        'existing wiki/log.md must be valid UTF-8 text',
                    ) from exc
                if previous and not _log_preserves_history(previous, _body(log), entry):
                    fail('LOG_HISTORY_NOT_PRESERVED', 'wiki/log.md', 'add the new entry above every existing entry and keep the existing log lines unchanged')
        candidates = [p for p in writes if p.startswith('wiki/') and p.endswith('.md')
                   and p not in {'wiki/hot.md', 'wiki/log.md', 'wiki/overview.md'} and p not in catalogs
                   and not p.startswith('wiki/meta/')]
        selected = [p for p in candidates if is_canonical_visible_wiki_markdown(p)]
        for page in candidates:
            if page not in selected:
                fail('INVALID_SELECTED_NOTE', page, 'selected canonical notes must use visible wiki Markdown paths')
        for page in selected:
            if not any(p in writes and links_to(p, page) for p in catalogs):
                fail('MISSING_CATALOG_REFERENCE', page, 'each selected canonical page needs a path-qualified link in a catalog write from workflow.catalogs')
        if operation == 'save' and not selected:
            fail('MISSING_SELECTED_NOTE', '', 'save must include the selected canonical note')
        if operation == 'ingest':
            for p in _LEDGERS:
                if p not in writes:
                    fail('MISSING_PROVENANCE_WRITE', p, 'ingest must include the reviewed source and claim ledger pair')
            expected = frozen.get('expected_hashes')
            expected = expected if isinstance(expected, dict) else {}
            prospective = storage.get('hashes') if isinstance(storage.get('hashes'), dict) else {}
            present = [p for p in _LEDGERS if p in writes]
            if present and all(prospective.get(p) == expected.get(p) for p in present):
                fail('UNCHANGED_PROVENANCE', '', 'ingest must record its evidence in a changed source or claim ledger')
            has_raw = any(p.startswith('.raw/') and p != '.raw/.manifest.json' for p in writes)
            if not has_raw and not _ledger_has_sources(writes.get(_LEDGERS[0], {}).get('content', '')):
                fail('MISSING_INGEST_SOURCE', '', 'ingest needs a create-only .raw/ source or a recorded source in the source ledger')
    return {'schema': 'claude-obsidian.workflow-plan.v1', 'storage_valid': True,
            'workflow_complete': not findings, 'workflow_findings': findings,
            'semantic_review': 'required', 'plan': storage,
            'approval_sha256': storage['approval_sha256'], 'changed_paths': storage['changed_paths'],
            'operation_id': storage['operation_id']}


def apply_workflow(vault: Path | str, bundle_or_path: Mapping[str, Any] | Path | str, *, approved_plan_sha256: str, **kwargs: Any) -> dict[str, Any]:
    if not approved_plan_sha256:
        raise TransactionValidationError('PLAN_APPROVAL_REQUIRED', 'workflow apply requires the exact workflow inspect approval hash')
    bundle, directory = _load_bundle(bundle_or_path)
    operation_id = safe_operation_id(bundle.get('operation_id'))
    completed = read_vault_regular(Path(vault), f'.vault-meta/transactions/{operation_id}/changed-paths.json', limit=8 * 1024 * 1024)
    try:
        completed_plan = strict_json_loads(completed.decode('utf-8')) if completed is not None else None
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise TransactionRecoveryError(
            'CORRUPT_RESULT',
            'completed operation result is not valid strict UTF-8 JSON',
        ) from exc
    if completed is not None:
        if (
            not isinstance(completed_plan, dict)
            or completed_plan.get('schema') != RESULT_SCHEMA
            or completed_plan.get('status') != 'complete'
            or completed_plan.get('operation_id') != operation_id
            or not isinstance(completed_plan.get('operation_type'), str)
            or not isinstance(completed_plan.get('bundle_sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', completed_plan['bundle_sha256']) is None
            or not isinstance(completed_plan.get('expanded_bundle_sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', completed_plan['expanded_bundle_sha256']) is None
            or not isinstance(completed_plan.get('approval_sha256'), str)
            or re.fullmatch(r'[0-9a-f]{64}', completed_plan['approval_sha256']) is None
            or not isinstance(completed_plan.get('changed_paths'), list)
            or any(not isinstance(path, str) for path in completed_plan['changed_paths'])
        ):
            raise TransactionRecoveryError('CORRUPT_RESULT', 'completed workflow result metadata is invalid')
        if completed_plan['operation_type'] != bundle.get('operation_type'):
            # Same contract as transaction apply: another bundle under a
            # completed operation ID is a reuse conflict, not corruption.
            raise TransactionConflict('OPERATION_ID_REUSED', f'operation_id {operation_id!r} already completed with another bundle')
    frozen = _freeze_loaded_bundle(bundle, directory, replay_vault=Path(vault) if completed_plan is not None else None)
    report = inspect_workflow(vault, frozen, _completed_plan=completed_plan)
    if not report['workflow_complete']:
        raise TransactionValidationError('WORKFLOW_INCOMPLETE', '; '.join(f"{x['code']}: {x['path']} {x['message']}" for x in report['workflow_findings']))
    # The transaction engine validates completed journal/result correlation,
    # the identical bundle and original approval under its pinned lock.
    return apply_bundle(vault, frozen, approved_plan_sha256=approved_plan_sha256, **kwargs)
