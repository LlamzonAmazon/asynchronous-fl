"""
Tee stdout/stderr to a log file so terminal output is recorded for later review.
Used by centralized train.py and federated run_fl.py (and FL server/client when FL_LOG_FILE is set).
"""

import sys
import atexit
from pathlib import Path
from typing import TextIO


class _Tee:
    """Write to both the original stream and a log file."""

    def __init__(self, stream: TextIO, log_file: TextIO):
        self._stream = stream
        self._file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._stream.flush()
        if self._file is not None:
            self._file.write(data)
            self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        if self._file is not None:
            self._file.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()


_log_file_handle = None


def tee_to_file(log_path: str | Path, mode: str = "w") -> None:
    """
    Redirect stdout and stderr to both the console and a log file.
    Call at the start of main() to record all subsequent print output.

    Args:
        log_path: Path to the log file (e.g. results/centralized/last_run.log).
        mode: 'w' to overwrite (e.g. main driver script), 'a' to append (e.g. FL server/client).
    """
    global _log_file_handle
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handle = open(path, mode, encoding="utf-8", errors="replace")

    sys.stdout = _Tee(sys.__stdout__, _log_file_handle)
    sys.stderr = _Tee(sys.__stderr__, _log_file_handle)

    def _close():
        global _log_file_handle
        if _log_file_handle is not None:
            try:
                _log_file_handle.close()
            except Exception:
                pass
            _log_file_handle = None

    atexit.register(_close)
