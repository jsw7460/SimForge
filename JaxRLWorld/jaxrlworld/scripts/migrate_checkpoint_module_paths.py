"""Rewrite ``rlworld.`` module paths in saved checkpoints to ``jaxrlworld.``.

The package was called ``rlworld`` until 2026-09. A checkpoint records its
reward, termination, event and sensor callables in ``config.yaml`` as
``module.path:attr`` strings, so every checkpoint written before the rename
names modules that no longer exist and fails to load. Loading one raises a
``ModuleNotFoundError`` from ``jaxrlworld.rl.utils.resolve`` that points here.

Only whole-word ``rlworld`` is rewritten. Other packages whose names merely
start with those characters keep theirs.

Usage::

    # see what would change
    python -m jaxrlworld.scripts.migrate_checkpoint_module_paths outputs/
    # write it
    python -m jaxrlworld.scripts.migrate_checkpoint_module_paths outputs/ --apply
"""

import argparse
import re
import sys
from pathlib import Path

#: Whole-word ``rlworld`` only, so sibling packages that merely share the
#: prefix or suffix (``rlworld_extras``, ``my_rlworld``) are left alone.
_OLD_PACKAGE_RE = re.compile(r"(?<![A-Za-z0-9_])rlworld(?![A-Za-z0-9_])")

_NEW_PACKAGE = "jaxrlworld"


def find_checkpoint_configs(root: Path) -> list[Path]:
    """Every ``config.yaml`` under ``root``, or ``root`` itself if it is one."""
    if root.is_file():
        return [root]
    return sorted(root.rglob("config.yaml"))


def migrate_file(path: Path, apply: bool) -> int:
    """Rewrite ``path`` in place and return how many references changed."""
    text = path.read_text()
    migrated, count = _OLD_PACKAGE_RE.subn(_NEW_PACKAGE, text)
    if count and apply:
        path.write_text(migrated)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint directories to walk, or single config.yaml files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes. Without this the run only reports them.",
    )
    args = parser.parse_args()

    for path in args.paths:
        if not path.exists():
            raise FileNotFoundError(f"No such path: {path}")

    configs = [config for path in args.paths for config in find_checkpoint_configs(path)]
    if not configs:
        print(f"No config.yaml found under {', '.join(str(p) for p in args.paths)}.")
        return 0

    total_files = 0
    total_refs = 0
    for config in configs:
        count = migrate_file(config, apply=args.apply)
        if count:
            total_files += 1
            total_refs += count
            print(f"{'migrated' if args.apply else 'would migrate'} {config} ({count} refs)")

    print(f"\n{total_files}/{len(configs)} config.yaml carry pre-rename paths ({total_refs} refs).")
    if total_files and not args.apply:
        print("Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
