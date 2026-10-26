"""Deterministic static checks for native artifacts, without rewriting bytes.

Canvas checks cover JSON Canvas 1.0 fields and product policy. Requiring both
top-level arrays, existing or prospectively created vault references, and HTTPS links is product policy;
the open standard permits the arrays to be absent. Bases checks cover UTF-8
text and control characters only. Neither validates native
Obsidian rendering, formula execution, or installed community extensions.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlsplit

from .json_utils import strict_json_loads
from .url_safety import url_credential_issue


class ArtifactValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def validate_base_bytes(data: bytes) -> None:
    """Check transport text only; no YAML or Bases semantic claim is made."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError("Base content must be UTF-8") from exc
    _require(
        not any(ord(character) < 32 and character not in "\n\r\t" for character in text),
        "Base content contains forbidden control characters",
    )


def _reference(value: Any, location: str, reference_exists: Callable[[str], bool]) -> None:
    _require(isinstance(value, str) and bool(value), f"{location} must be a non-empty vault-relative path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and value not in {".", "~"}
        and not value.startswith(("~/", "~\\"))
        and not re.match(r"^[A-Za-z]:", value)
        and ".." not in path.parts
        and path.as_posix() == value
        and not any(ord(character) < 32 or character == "\\" for character in value),
        f"{location} must be a canonical vault-relative path without traversal",
    )
    try:
        exists = reference_exists(value)
    except (ValueError, OSError) as exc:
        raise ArtifactValidationError(f"{location} is not a safe existing vault file: {exc}") from exc
    _require(exists, f"{location} does not name an existing regular vault file: {value}")


def _optional_text(value: dict[str, Any], field: str, location: str) -> None:
    if field in value:
        _require(isinstance(value[field], str), f"{location}.{field} must be text")


def _color(value: dict[str, Any], location: str) -> None:
    if "color" in value:
        color = value["color"]
        _require(
            isinstance(color, str) and bool(re.fullmatch(r"[1-6]|#[0-9a-fA-F]{6}", color)),
            f"{location}.color must be a preset 1 to 6 or a six-digit hex color",
        )


def validate_canvas_bytes(data: bytes, *, reference_exists: Callable[[str], bool]) -> None:
    """Validate known fields in place, preserving every original byte."""

    try:
        document = strict_json_loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ArtifactValidationError(f"Canvas content is not strict UTF-8 JSON: {exc}") from exc
    _require(isinstance(document, dict), "Canvas root must be an object")
    nodes, edges = document.get("nodes"), document.get("edges")
    _require(isinstance(nodes, list) and isinstance(edges, list), "Canvas must contain nodes and edges arrays")
    identifiers: set[str] = set()
    node_ids: set[str] = set()

    def identifier(value: dict[str, Any], location: str) -> str:
        identity = value.get("id")
        _require(isinstance(identity, str) and bool(identity), f"{location}.id must be non-empty text")
        _require(identity not in identifiers, f"{location}.id is duplicated: {identity}")
        identifiers.add(identity)
        return identity

    for index, node in enumerate(nodes):
        location = f"nodes[{index}]"
        _require(isinstance(node, dict), f"{location} must be an object")
        node_ids.add(identifier(node, location))
        kind = node.get("type")
        _require(isinstance(kind, str) and kind in {"text", "file", "link", "group"}, f"{location}.type is unsupported")
        for field in ("x", "y", "width", "height"):
            _require(type(node.get(field)) is int, f"{location}.{field} must be an integer")
        _require(node["width"] > 0 and node["height"] > 0, f"{location} dimensions must be positive")
        _color(node, location)
        if kind == "text":
            _require(isinstance(node.get("text"), str), f"{location}.text must be text")
        elif kind == "file":
            _reference(node.get("file"), f"{location}.file", reference_exists)
            _optional_text(node, "subpath", location)
            if "subpath" in node:
                _require(node["subpath"].startswith("#"), f"{location}.subpath must begin with #")
        elif kind == "link":
            url = node.get("url")
            _require(isinstance(url, str), f"{location}.url must be text")
            try:
                parsed = urlsplit(url)
                safe = parsed.scheme == "https" and bool(parsed.hostname) and parsed.port != 0 and url_credential_issue(parsed) is None
            except ValueError:
                safe = False
            _require(safe and not any(ord(character) < 33 for character in url), f"{location}.url must be HTTPS without credentials or control characters")
        elif kind == "group":
            _optional_text(node, "label", location)
            if "background" in node:
                _reference(node["background"], f"{location}.background", reference_exists)
            if "backgroundStyle" in node:
                _require(isinstance(node["backgroundStyle"], str) and node["backgroundStyle"] in {"cover", "ratio", "repeat"}, f"{location}.backgroundStyle is unsupported")
    for index, edge in enumerate(edges):
        location = f"edges[{index}]"
        _require(isinstance(edge, dict), f"{location} must be an object")
        identifier(edge, location)
        for field in ("fromNode", "toNode"):
            _require(isinstance(edge.get(field), str) and edge[field] in node_ids, f"{location}.{field} must reference an existing node ID")
        for field in ("fromSide", "toSide"):
            if field in edge:
                _require(isinstance(edge[field], str) and edge[field] in {"top", "right", "bottom", "left"}, f"{location}.{field} is unsupported")
        for field in ("fromEnd", "toEnd"):
            if field in edge:
                _require(isinstance(edge[field], str) and edge[field] in {"none", "arrow"}, f"{location}.{field} is unsupported")
        _optional_text(edge, "label", location)
        _color(edge, location)
