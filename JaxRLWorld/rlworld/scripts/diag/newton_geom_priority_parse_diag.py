"""Does Newton's MJCF importer still lose ``<geom priority>``?

``NewtonSceneManager`` calls ``_force_collision_shape_priority`` after
every MJCF/URDF load, painting ``mujoco:geom_priority = 1`` onto every
collision shape. Its comment describes a Newton parser bug -- XML
``priority`` attributes lost for most shapes when visuals are also parsed
-- and says to remove the workaround once upstream is fixed.

That matters beyond tidiness. ``priority`` decides how a contact pair's
friction is combined: equal priorities take the element-wise MAX of the
two geoms, an unequal pair takes the HIGHER one's values outright. The
workaround paints every shape with the same value, which makes every pair
equal-priority, which forces max() everywhere. So a preset that gives its
feet a priority so their randomised friction wins over the ground's gets
that intent erased on this backend, silently -- which is the very thing
the workaround was written to protect.

Newton's own tests now assert the attribute parses, including through a
``<default>`` class and a per-geom override. They do not settle it: the
claim is about what happens WHEN VISUALS ARE PARSED, and the scene
manager loads with ``parse_visuals`` at its default of True.

So load a model that has both, the same way the manager does, with the
workaround NOT applied, and read the attribute back. Every collision geom
here states a priority the importer either kept or did not.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile

import newton

MODEL = """<mujoco model="priority_parse">
  <compiler angle="radian"/>
  <default>
    <default class="vis">
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="col">
      <geom type="box" size="0.04 0.04 0.04" contype="1" conaffinity="1" priority="3"/>
    </default>
  </default>
  <worldbody>
    <body name="link_a" pos="0 0 1">
      <freejoint name="free_a"/>
      <geom name="a_visual" class="vis"/>
      <geom name="a_class" class="col"/>
      <geom name="a_override" class="col" priority="7" pos="0.2 0 0"/>
      <geom name="a_bare" type="box" size="0.03 0.03 0.03" pos="0.4 0 0"
            contype="1" conaffinity="1" priority="5"/>
    </body>
    <body name="link_b" pos="0 0.5 1">
      <freejoint name="free_b"/>
      <geom name="b_visual" class="vis"/>
      <geom name="b_class" class="col"/>
      <geom name="b_override" class="col" priority="9" pos="0.2 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

EXPECTED = {
    "a_class": 3,
    "a_override": 7,
    "a_bare": 5,
    "b_class": 3,
    "b_override": 9,
}
"""What the XML says, per COLLISION geom. Visual geoms are excluded: the
workaround skips them too, and they take no part in contact pairs."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parse-visuals",
        default="both",
        choices=("both", "true", "false"),
        help="load with visuals parsed, not parsed, or both (default)",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("  DOES NEWTON'S MJCF IMPORTER KEEP <geom priority>")
    print("=" * 78)
    print(f"  newton {getattr(newton, '__version__', 'unknown')}")
    print("  loaded exactly as NewtonSceneManager does: parse_sites=True,")
    print("  collapse_fixed_joints=False, and WITHOUT the priority workaround")

    folder = tempfile.mkdtemp()
    path = pathlib.Path(folder) / "priority_parse.xml"
    path.write_text(MODEL)

    variants = (True, False) if args.parse_visuals == "both" else (args.parse_visuals == "true",)
    failures: list[str] = []

    for parse_visuals in variants:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(
            str(path),
            floating=True,
            collapse_fixed_joints=False,
            parse_sites=True,
            parse_visuals=parse_visuals,
        )

        attr = builder.custom_attributes.get("mujoco:geom_priority")
        print("\n" + "-" * 78)
        print(f"  parse_visuals={parse_visuals}")
        print("-" * 78)
        if attr is None:
            print("      the builder has no 'mujoco:geom_priority' attribute at all")
            failures.append(f"parse_visuals={parse_visuals}: attribute absent")
            continue

        values = attr.values or {}
        print(f"      {len(builder.shape_label)} shapes loaded, " f"{len(values)} carry a priority")
        print(f"      {'shape':<28}{'expected':>10}{'got':>10}")
        for index, label in enumerate(builder.shape_label):
            leaf = label.rsplit("/", 1)[-1]
            want = next((v for k, v in EXPECTED.items() if leaf.endswith(k)), None)
            if want is None:
                continue
            got = values.get(index, "<missing>")
            mark = "" if got == want else "   <-- LOST"
            print(f"      {leaf:<28}{want:>10}{str(got):>10}{mark}")
            if got != want:
                failures.append(f"parse_visuals={parse_visuals}: {leaf} authored {want}, got {got}")

    print("\n" + "=" * 78)
    if failures:
        print(f"  {len(failures)} MISMATCHES — the workaround is still needed")
        for line in failures:
            print(f"    {line}")
    else:
        print("  every authored priority survived, with and without visuals.")
        print("  The parser bug the workaround cites is fixed in this Newton,")
        print("  so _force_collision_shape_priority can go -- and should, since")
        print("  it erases whatever priority a preset authors.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
