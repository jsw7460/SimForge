"""Dump the bytecode of `_func_narrowphase_contact0` and search for the
patched literals/instructions. If the bytecode does NOT contain ``LOAD_CONST 100.0``
(or ``LOAD_CONST False`` for whichever patch is currently in source), the
Python ``__code__`` object itself is stale — quadrants is compiling against
a frozen earlier version of the function.
"""

from __future__ import annotations


def main() -> None:
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    from genesis.engine.solvers.rigid.collider import narrowphase

    fn = narrowphase._func_narrowphase_contact0
    underlying = getattr(fn, "fn", fn)
    print("fn type:", type(fn).__name__)
    print("underlying type:", type(underlying).__name__)
    print("underlying.__code__.co_filename:", underlying.__code__.co_filename)
    print("underlying.__code__.co_firstlineno:", underlying.__code__.co_firstlineno)
    print("co_consts (first 40):")
    for i, c in enumerate(underlying.__code__.co_consts[:40]):
        print(f"  [{i}] {c!r}")
    has_100 = any(c == 100.0 for c in underlying.__code__.co_consts)
    has_false_const = False in underlying.__code__.co_consts
    has_42 = any(c == 0.42 for c in underlying.__code__.co_consts)
    print()
    print("co_consts contains 100.0:", has_100)
    print("co_consts contains False:", has_false_const)
    print("co_consts contains 0.42:", has_42)


if __name__ == "__main__":
    main()
