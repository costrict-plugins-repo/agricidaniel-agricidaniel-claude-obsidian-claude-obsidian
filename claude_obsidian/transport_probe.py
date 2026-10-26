"""Read-only CLI identity probes, separate from filesystem availability."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def _stop_probe(process: subprocess.Popen) -> None:
    """Kill a timed-out probe and its session; native Windows has no killpg."""
    killpg = getattr(os, "killpg", None)
    try:
        if killpg is not None and hasattr(signal, "SIGKILL"):
            killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def probe_selected_vault(binary: str, vault: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Prove the selected path and exact read command before preferring CLI.

    No note body is requested. A version string alone cannot establish this
    identity. All subprocesses share one bounded probe budget.
    """
    deadline = time.monotonic() + timeout
    def run(arguments: list[str]) -> tuple[int, str]:
        with tempfile.TemporaryFile() as output:
            process = None
            try:
                process = subprocess.Popen([binary, *arguments], cwd=vault,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if process is not None:
                    _stop_probe(process)
                return 124, 'selected-vault probe exceeded its time budget'
            except OSError as exc:
                return 126, f'cannot execute selected-vault probe: {exc.__class__.__name__}'
            output.seek(0)
            raw = output.read(8193)
            if len(raw) > 8192:
                return 2, 'selected-vault probe response exceeded 8192 bytes'
            return code, raw.decode('utf-8', errors='replace').strip()
    code, response = run(['vault', 'info=path'])
    if code or not response or '\n' in response or '\r' in response:
        return False, 'Cannot confirm the selected vault path. Open/register that vault or use filesystem fallback.'
    try:
        path = Path(response)
        if not path.is_absolute() or path.resolve(strict=True) != Path(vault).resolve(strict=True):
            return False, 'CLI resolved a different or unrecognized vault path; filesystem fallback selected.'
    except (OSError, RuntimeError, ValueError):
        return False, 'CLI returned an invalid vault path; filesystem fallback selected.'
    code, help_text = run(['help', 'read'])
    error_shaped = re.search(r'(?im)^\s*(?:error|err|fatal|exception|failed|unknown|unsupported)(?:\s*[:\-]|\s+)', help_text)
    if code or 'path=' not in help_text or error_shaped:
        return False, 'CLI exact path read capability could not be confirmed; filesystem fallback selected.'
    return True, ''
