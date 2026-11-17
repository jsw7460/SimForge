"""Refuse imports inside functions without a stated reason.

An import belongs at the top of its module: that is where a reader looks for
what the module depends on, where a missing dependency fails at once rather
than in the middle of a run, and where a cycle shows up as a cycle. An import
inside a function is legitimate only when the top would be wrong, and the
line says why::

    import genesis as gs  # local-import: backend, loaded only on the Genesis path
    from jaxrlworld.rl.envs import World  # local-import: circular (envs imports configs)

Accepted reasons are a circular import, a simulator backend that must not load
on the other backends (see CLAUDE.md), and an optional dependency the package
does not declare. Anything else moves to the top.

Usage: ``python JaxRLWorld/tools/check_local_imports.py FILE...`` (run by
pre-commit). Exits 1 listing every unmarked import.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "# local-import:"


def unmarked_imports(path: Path) -> list[tuple[int, str]]:
    source = path.read_text()
    lines = source.splitlines()
    found = []
    seen: set[int] = set()
    for func in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Import | ast.ImportFrom) or node.lineno in seen:
                continue
            seen.add(node.lineno)
            # The marker may sit on any line of a parenthesized import.
            span = lines[node.lineno - 1 : node.end_lineno]
            if any(MARKER in line and line.split(MARKER, 1)[1].strip() for line in span):
                continue
            found.append((node.lineno, ast.unparse(node)[:80]))
    return sorted(found)


def main(argv: list[str]) -> int:
    failures = 0
    for name in argv:
        for lineno, stmt in unmarked_imports(Path(name)):
            print(f"{name}:{lineno}: {stmt}")
            failures += 1
    if failures:
        print(
            f"\n{failures} import(s) inside a function without '{MARKER} <reason>'. Move it to the top of the "
            "module, or mark a circular import, a backend that must stay lazy, or an optional dependency."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
