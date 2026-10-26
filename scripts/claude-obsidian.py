#!/usr/bin/env python3
"""Compatibility entry point for plugin hooks and legacy script callers."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.dont_write_bytecode = True

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

# Default-off SessionStart must return before loading any vault or CLI code.
# Only exact hook invocations use this path; argparse retains normal errors
# and global option handling for every other invocation.
# Both fast paths emit UTF-8 like the full CLI, so vault text cannot crash a
# hook on a legacy console code page.
if sys.argv[1:] == ["hook", "session-start"]:
    if os.environ.get("CLAUDE_OBSIDIAN_SESSION_CONTEXT") != "1":
        raise SystemExit(0)
    from claude_obsidian.hook_adapter import emit_session_start, force_utf8_stdio

    force_utf8_stdio()
    raise SystemExit(emit_session_start(plugin_root=PLUGIN_ROOT))
if sys.argv[1:] == ["hook", "stop"]:
    from claude_obsidian.hook_adapter import emit_stop_status, force_utf8_stdio

    force_utf8_stdio()
    raise SystemExit(emit_stop_status(plugin_root=PLUGIN_ROOT))

from claude_obsidian.cli import main


raise SystemExit(main())
