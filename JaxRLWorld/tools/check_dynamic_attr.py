"""Refuse ``getattr`` / ``hasattr`` calls that are not marked as by-name lookups.

Probing an attribute that the type is supposed to have hides the bug the
missing attribute is (a default fills in for a field that was renamed, a
type that lacks it takes another branch). Read the attribute directly; make
an optional one explicit (a ``None`` field, a protocol method, an
``isinstance`` branch). A lookup whose attribute NAME is data (a string
from a config, ``"module:func"`` resolution, a delegation ``__getattr__``)
is legitimate and says so on the call's line::

    fn = getattr(module, name)  # dynamic-attr: resolves the configured "module:func"

Usage: ``python JaxRLWorld/tools/check_dynamic_attr.py FILE...`` (run by
pre-commit on the package's Python files). Exits 1 listing every unmarked
call.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "# dynamic-attr:"


def unmarked_calls(path: Path) -> list[tuple[int, str]]:
    source = path.read_text()
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ("getattr", "hasattr"):
            continue
        span = lines[node.lineno - 1 : node.end_lineno]
        reasons = [line.split(MARKER, 1)[1].strip() for line in span if MARKER in line]
        if not any(reasons):
            found.append((node.lineno, ast.unparse(node)))
    return found


def main(argv: list[str]) -> int:
    failures = 0
    for name in argv:
        for lineno, call in unmarked_calls(Path(name)):
            print(f"{name}:{lineno}: {call}")
            failures += 1
    if failures:
        print(
            f"\n{failures} getattr/hasattr call(s) without '{MARKER} <reason>'. Access the attribute directly, "
            "make an optional one explicit, or mark a genuine by-name lookup with its reason."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
