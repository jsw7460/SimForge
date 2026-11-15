"""Refuse ``except`` handlers that swallow the error without saying why.

A handler that neither raises nor re-raises turns a failure into a default
(``pass``, ``return None``, a ``"?"``, a fallback path), and the run goes on
with a value nobody asked for. Let the error propagate, or catch the specific
exception the code expects and act on it. A handler that legitimately ends
the error is control flow, not a fallback: a retry on ``FileExistsError``, a
``KeyboardInterrupt`` that stops a server, a worker that sends the traceback
to its parent, a parse attempt that tries the next type. It says so on the
``except`` line::

    except FileExistsError:  # except-ok: the name was taken, try the next one

Usage: ``python JaxRLWorld/tools/check_silent_except.py FILE...`` (run by
pre-commit). Exits 1 listing every unmarked handler.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "# except-ok:"


def _raises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(node, ast.Raise) for stmt in handler.body for node in ast.walk(stmt))


def unmarked_handlers(path: Path) -> list[tuple[int, str]]:
    source = path.read_text()
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.ExceptHandler) or _raises(node):
            continue
        line = lines[node.lineno - 1]
        if MARKER in line and line.split(MARKER, 1)[1].strip():
            continue
        caught = "everything" if node.type is None else ast.unparse(node.type)
        found.append((node.lineno, f"except {caught}: {ast.unparse(node.body[0])[:60]}"))
    return found


def main(argv: list[str]) -> int:
    failures = 0
    for name in argv:
        for lineno, handler in unmarked_handlers(Path(name)):
            print(f"{name}:{lineno}: {handler}")
            failures += 1
    if failures:
        print(
            f"\n{failures} except handler(s) that do not raise, without '{MARKER} <reason>'. Let the error "
            "propagate, catch the specific exception and act on it, or mark legitimate control flow."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
