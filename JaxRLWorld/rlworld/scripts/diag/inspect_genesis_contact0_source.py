"""Print what source the qd kernel decorator captured for the patched
plane-capsule narrowphase function. If this prints ``source has 0.42: False``
the decorator is NOT seeing our patched source — the import-marker fires but
quadrants is binding to a stale function body somehow.
"""

from __future__ import annotations

import inspect


def main() -> None:
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    from genesis.engine.solvers.rigid.collider import narrowphase

    fn = narrowphase._func_narrowphase_contact0
    print("contact0 type:", type(fn).__name__)
    attrs = sorted(a for a in dir(fn) if not a.startswith("_"))
    print("contact0 attrs:", attrs[:30])

    for attr in (
        "func",
        "_func",
        "py_func",
        "_py_func",
        "fn",
        "_fn",
        "original_func",
        "_original_func",
        "src",
        "_src",
        "source",
        "_source",
        "wrapped",
        "__wrapped__",
    ):
        if hasattr(fn, attr):
            val = getattr(fn, attr)
            print(f"  {attr}: {val!r}")

    try:
        print("inspect.getsourcefile:", inspect.getsourcefile(fn))
    except Exception as e:
        print("inspect.getsourcefile err:", repr(e))

    try:
        src = inspect.getsource(fn)
        print("source first 80 chars:", repr(src[:80]))
        print("source has 0.42:", "0.42" in src)
        print("source has DEBUG marker:", "DEBUG (short-capsule" in src)
    except Exception as e:
        print("inspect.getsource err:", repr(e))


if __name__ == "__main__":
    main()
