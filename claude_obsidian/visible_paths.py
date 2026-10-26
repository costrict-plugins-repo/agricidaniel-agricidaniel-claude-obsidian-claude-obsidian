"""Shared visible wiki path rules for workflow navigation and lint discovery."""
from pathlib import PurePosixPath
from typing import Any


def is_visible_path(value: str) -> bool:
    """Obsidian content paths cannot contain a dot-prefixed component."""
    return all(not part.startswith('.') for part in PurePosixPath(value).parts)


def is_canonical_visible_wiki_markdown(value: Any) -> bool:
    if not isinstance(value, str) or '\\' in value or any(ord(c) < 32 for c in value):
        return False
    path = PurePosixPath(value)
    return (path.as_posix() == value and len(path.parts) >= 2
            and path.parts[0] == 'wiki' and path.suffix == '.md'
            and is_visible_path(value))
